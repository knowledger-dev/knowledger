# main.py

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Path
from pydantic import BaseModel, Field

from config import Config
from models import NoteInput, NoteOutput, QueryInput, RAGQueryInput, UserCreate, UserRead, UserInDB
from db import MongoDBConnection
from llm_client import get_llm_client
from utils import preprocess_for_embedding, hash_password, verify_password, convert_objectid_to_str
from sentence_transformers import SentenceTransformer

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import numpy as np
import pymongo
import networkx as nx
from sklearn.cluster import DBSCAN

# For user authentication
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

# Initialize Logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(title=Config.APP_TITLE)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB Connection
mongodb_conn = MongoDBConnection()

# Initialize SentenceTransformer Model
model_cache = {}

def get_sentence_transformer():
    if Config.SENTENCE_TRANSFORMER_MODEL not in model_cache:
        model_cache[Config.SENTENCE_TRANSFORMER_MODEL] = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
        logger.info(f"SentenceTransformer model '{Config.SENTENCE_TRANSFORMER_MODEL}' loaded.")
    return model_cache[Config.SENTENCE_TRANSFORMER_MODEL]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def generate_note_id() -> str:
    return os.urandom(8).hex()

def create_note_in_mongodb(note_id: str, content: str, processed_content: str,
                           embedding: List[float], timestamp: datetime, summary: str = "", owner_username: str = ""):
    try:
        mongodb_conn.create_note(
            note_id=note_id,
            content=content,
            processed_content=processed_content,
            embedding=embedding,
            timestamp=timestamp,
            summary=summary,
            owner_username=owner_username
        )
        logger.debug(f"Note {note_id} created in MongoDB.")
    except Exception as e:
        logger.error(f"Error creating note {note_id}: {e}")
        raise

async def generate_cluster_summary(note_ids: List[str]) -> str:
    combined_content = mongodb_conn.get_cluster_content(note_ids)
    if not combined_content:
        logger.warning("No combined content found for cluster summary.")
        return ""
    prompt = Config.CLUSTER_SUMMARY_PROMPT_TEMPLATE.format(content=combined_content)
    llm_client = get_llm_client()
    summary = await llm_client.generate_content(prompt)
    return summary

# Authentication setup
SECRET_KEY = Config.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=180))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = mongodb_conn.get_user_by_username(username)
        if user is None:
            raise credentials_exception
        return UserInDB(**user)
    except JWTError:
        raise credentials_exception

# -----------------------------------------------------------------------------
# Graph and Clustering Functions
# -----------------------------------------------------------------------------

def compute_pagerank():
    """
    Computes PageRank using NetworkX and updates the scores in MongoDB.
    """
    logger.info("Starting PageRank computation using NetworkX...")
    start_time = time.time()
    try:
        # Build the graph efficiently using edge lists
        notes_cursor = mongodb_conn.db.notes.find(
            {"similar_notes": {"$exists": True, "$ne": []}},
            {"_id": 1, "similar_notes": 1}
        )

        edge_list = []
        for note in notes_cursor:
            note_id = note["_id"]
            similar_notes = note["similar_notes"]
            edge_list.extend([(note_id, sim_note_id) for sim_note_id in similar_notes])

        if not edge_list:
            logger.warning("No edges found for PageRank computation.")
            return

        # Build the graph
        G = nx.DiGraph()
        G.add_edges_from(edge_list)

        # Compute PageRank
        pagerank_scores = nx.pagerank(G, alpha=Config.PAGERANK_ALPHA, max_iter=100, tol=1e-06)

        # Update notes with PageRank scores
        bulk_operations = [
            pymongo.UpdateOne(
                {"_id": note_id},
                {"$set": {"pagerank": score}}
            ) for note_id, score in pagerank_scores.items()
        ]

        if bulk_operations:
            mongodb_conn.db.notes.bulk_write(bulk_operations)
            logger.info("PageRank scores updated in MongoDB.")

        elapsed_time = time.time() - start_time
        logger.info(f"PageRank computation and update completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during PageRank computation: {e}")

def perform_clustering():
    """
    Performs clustering on note embeddings and updates MongoDB with cluster information.
    Uses DBSCAN algorithm.
    """
    logger.info("Starting clustering process...")
    start_time = time.time()
    try:
        # Fetch all notes with embeddings
        notes_cursor = mongodb_conn.db.notes.find(
            {"embedding": {"$exists": True, "$ne": []}},
            {"_id": 1, "embedding": 1, "pagerank": 1}
        )
        notes = list(notes_cursor)
        if not notes:
            logger.warning("No embeddings found for clustering.")
            return

        note_ids = [note["_id"] for note in notes]
        embeddings = np.array([note["embedding"] for note in notes])
        pageranks = np.array([note.get("pagerank", 0.0) for note in notes])

        # Perform DBSCAN clustering
        logger.info(f"Performing DBSCAN clustering with eps={Config.DBSCAN_EPS}, min_samples={Config.DBSCAN_MIN_SAMPLES}.")
        dbscan = DBSCAN(eps=Config.DBSCAN_EPS, min_samples=Config.DBSCAN_MIN_SAMPLES, metric='cosine', n_jobs=-1)
        cluster_labels = dbscan.fit_predict(embeddings)

        # Clear existing clusters
        mongodb_conn.db.clusters.delete_many({})

        # Create mapping from note_id to index for faster lookup
        note_id_to_index = {note_id: idx for idx, note_id in enumerate(note_ids)}

        # Create new clusters
        clusters = {}
        for note_id, label, pagerank in zip(note_ids, cluster_labels, pageranks):
            if label == -1:
                continue  # Ignore noise points
            clusters.setdefault(label, []).append((note_id, pagerank))

        cluster_documents = []
        update_operations = []

        for label, members in clusters.items():
            cluster_id = str(label)
            member_ids = [member[0] for member in members]
            member_pageranks = np.array([member[1] for member in members])

            # Get indices of member notes
            member_indices = [note_id_to_index[mid] for mid in member_ids]
            member_embeddings = embeddings[member_indices]

            # Compute Weighted Average Embedding
            total_pagerank = np.sum(member_pageranks)
            if total_pagerank > 0:
                weighted_embeddings = member_embeddings.T * member_pageranks
                cluster_embedding = (np.sum(weighted_embeddings, axis=1) / total_pagerank).tolist()
            else:
                cluster_embedding = member_embeddings.mean(axis=0).tolist()

            # Generate cluster summary using LLM
            summary = asyncio.run(generate_cluster_summary(member_ids))

            # Create cluster document
            cluster_document = {
                "_id": cluster_id,
                "label": f"Cluster_{cluster_id}",
                "summary": summary,
                "size": len(members),
                "embedding": cluster_embedding,
                "note_ids": member_ids
            }
            cluster_documents.append(cluster_document)

            # Prepare update operations for notes
            update_operations.append(
                pymongo.UpdateMany(
                    {"_id": {"$in": member_ids}},
                    {"$set": {"cluster_id": cluster_id}}
                )
            )

        # Insert cluster documents in bulk
        if cluster_documents:
            mongodb_conn.db.clusters.insert_many(cluster_documents)

        # Update notes with cluster_id in bulk
        if update_operations:
            for operation in update_operations:
                mongodb_conn.db.notes.bulk_write([operation])

        elapsed_time = time.time() - start_time
        logger.info(f"Clustering completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during clustering: {e}")

# -----------------------------------------------------------------------------
# Scheduler and Startup Events
# -----------------------------------------------------------------------------

scheduler = BackgroundScheduler()
scheduler.start()

# Schedule the compute_pagerank function to run every 10 minutes
scheduler.add_job(
    func=compute_pagerank,
    trigger=IntervalTrigger(minutes=10),
    id='compute_pagerank_job',
    name='Compute PageRank every 10 minutes',
    replace_existing=True
)

# Schedule the perform_clustering function to run every 10 minutes
scheduler.add_job(
    func=perform_clustering,
    trigger=IntervalTrigger(minutes=10),
    id='perform_clustering_job',
    name='Perform Clustering every 10 minutes',
    replace_existing=True
)

# Shut down the scheduler when exiting the app
import atexit
atexit.register(lambda: scheduler.shutdown())

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete. Scheduled background tasks.")

@app.on_event("shutdown")
async def shutdown_event():
    mongodb_conn.client.close()
    logger.info("Application shutdown complete.")

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.post("/")
async def default_post():
    return {"message": "Welcome to the KnowledgeRank API"}

# User registration endpoint
@app.post("/register", response_model=UserRead)
async def register_user(user_input: UserCreate):
    if mongodb_conn.get_user_by_username(user_input.username):
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = hash_password(user_input.password)
    user_in_db = UserInDB(
        username=user_input.username,
        email=user_input.email,
        hashed_password=hashed_password,
        is_active=True,
        created_at=datetime.utcnow()
    )

    # Convert to dict, excluding unset and None values
    user_data = user_in_db.dict(by_alias=True, exclude_unset=True, exclude_none=True)

    mongodb_conn.create_user(user_data)

    user = mongodb_conn.get_user_by_username(user_input.username)
    if not user:
        raise HTTPException(status_code=500, detail="User creation failed.")
    return UserRead(**user)

# User login endpoint
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = mongodb_conn.get_user_by_username(form_data.username)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if not verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/notes/{note_id}", response_model=NoteOutput)
async def get_note_endpoint(
    note_id: str,
    current_user: UserInDB = Depends(get_current_user)
):

    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    note = mongodb_conn.get_note_owned_by_user(note_id, current_user.username)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    return NoteOutput(**note)

@app.get("/notes/", response_model=List[NoteOutput])
async def get_all_notes_endpoint(
    current_user: UserInDB = Depends(get_current_user)
):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    notes = mongodb_conn.get_all_notes_owned_by_user(current_user.username)
    if not notes:
        raise HTTPException(status_code=404, detail="No notes found")

    return [NoteOutput(**note) for note in notes]

@app.put("/notes/{note_id}", response_model=NoteOutput)
async def update_note_endpoint(
    note_input: NoteInput,
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_user),
    note_id: str = Path(..., description="The ID of the note to update")
):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    # Verify that the note exists and is owned by the current user
    existing_note = mongodb_conn.get_note_owned_by_user(note_id, current_user.username)
    if not existing_note:
        raise HTTPException(status_code=404, detail="Note not found or access denied")

    try:
        # Process the updated content
        processed_content = preprocess_for_embedding(note_input.content)
        model = get_sentence_transformer()
        embedding = model.encode([processed_content], show_progress_bar=False)[0].tolist()

        # Generate new summary using LLM
        llm_client = get_llm_client()
        summary_prompt = Config.NOTE_SUMMARY_PROMPT_TEMPLATE.format(content=processed_content)
        summary = await llm_client.generate_content(summary_prompt)

        # Update the note in MongoDB
        mongodb_conn.update_note(
            note_id=note_id,
            content=note_input.content,
            processed_content=processed_content,
            embedding=embedding,
            timestamp=note_input.timestamp,
            summary=summary
        )

        # Update relationships and PageRank in the background
        background_tasks.add_task(
            mongodb_conn.update_relationships,
            note_id,
            Config.SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS
        )
        background_tasks.add_task(compute_pagerank)
        background_tasks.add_task(perform_clustering)

        # Retrieve the updated note to include all fields
        note = mongodb_conn.get_note(note_id)
        if not note:
            raise HTTPException(status_code=500, detail="Failed to retrieve the updated note.")
        note_output = NoteOutput(**note)
        return note_output

    except Exception as e:
        logger.error(f"Error updating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/notes", response_model=NoteOutput)
async def create_note_endpoint(
    note_input: NoteInput,
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_user)
):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        note_id = generate_note_id()
        processed_content = preprocess_for_embedding(note_input.content)
        model = get_sentence_transformer()
        embedding = model.encode([processed_content], show_progress_bar=False)[0].tolist()

        # Generate summary using LLM
        llm_client = get_llm_client()
        summary_prompt = Config.NOTE_SUMMARY_PROMPT_TEMPLATE.format(content=processed_content)
        summary = await llm_client.generate_content(summary_prompt)

        # Create note in MongoDB
        create_note_in_mongodb(
            note_id=note_id,
            content=note_input.content,
            processed_content=processed_content,
            embedding=embedding,
            timestamp=note_input.timestamp,
            summary=summary,
            owner_username=current_user.username
        )

        # Update relationships and PageRank in the background
        background_tasks.add_task(
            mongodb_conn.update_relationships,
            note_id,
            Config.SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS
        )
        background_tasks.add_task(compute_pagerank)
        background_tasks.add_task(perform_clustering)

        # Retrieve the created note to include all fields
        note = mongodb_conn.get_note(note_id)
        if not note:
            raise HTTPException(status_code=500, detail="Failed to retrieve the created note.")
        note_output = NoteOutput(**note)
        return note_output

    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/query", response_model=List[NoteOutput])
async def query_notes(
    query_input: QueryInput,
    current_user: UserInDB = Depends(get_current_user)
):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        processed_query = preprocess_for_embedding(query_input.query)
        model = get_sentence_transformer()
        query_embedding = model.encode([processed_query], show_progress_bar=False)[0].tolist()

        similar_notes = mongodb_conn.get_similar_notes(
            query_embedding=query_embedding,
            similarity_threshold=Config.SIMILARITY_THRESHOLD_NOTE,
            limit=query_input.limit,
            owner_username=current_user.username
        )
        logger.info(f"Retrieved {len(similar_notes)} notes for query.")

        # Since similar_notes now contain full note data, we can construct NoteOutput directly
        notes_output = [NoteOutput(**note) for note in similar_notes]

        return notes_output

    except Exception as e:
        logger.error(f"Error querying notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag_query")
async def rag_query(
    rag_query_input: RAGQueryInput,
    current_user: UserInDB = Depends(get_current_user)
):
    start_time = time.time()

    if not mongodb_conn.verify_connectivity():
        logger.error("Database connection error")
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        user_query = rag_query_input.query
        logger.info(f"Received Query: {user_query}")

        # Step 1: Refine Query
        refined_user_query = preprocess_for_embedding(user_query)
        logger.debug(f"Refined Query: {refined_user_query}")
        model = get_sentence_transformer()
        query_embedding = model.encode([refined_user_query], show_progress_bar=False)[0].tolist()

        # Step 2: Retrieve Similar Notes with PageRank Weighting
        similar_notes = mongodb_conn.get_similar_notes(
            query_embedding=query_embedding,
            similarity_threshold=Config.SIMILARITY_THRESHOLD_NOTE,
            limit=Config.RAG_MAX_NOTES_PER_CLUSTER,
            use_pagerank_weighting=True,
            owner_username=current_user.username
        )
        logger.info(f"Retrieved {len(similar_notes)} similar notes.")

        # Step 3: Include Cluster Summaries with Damping Factor
        cluster_damping_factor = Config.CLUSTER_DAMPING_FACTOR
        cluster_summaries = {}
        cluster_ids = set(note.get("cluster_id") for note in similar_notes if note.get("cluster_id"))

        if cluster_ids:
            clusters = mongodb_conn.get_clusters(list(cluster_ids))
            for cluster in clusters:
                cluster_id = cluster["_id"]
                cluster_summary = cluster.get("summary", "")
                adjusted_length = int(len(cluster_summary) * cluster_damping_factor)
                adjusted_cluster_summary = cluster_summary[:adjusted_length]
                cluster_summaries[cluster_id] = adjusted_cluster_summary

        # Step 4: Aggregate Context
        aggregated_context = ""
        referenced_note_ids = []
        max_context_length = Config.RAG_MAX_CONTEXT_LENGTH

        for note in similar_notes:
            note_id = note["id"]
            content = note["content"]
            formatted_content = f"Note ID: {note_id}\nContent: {content}\n\n"

            if len(aggregated_context) + len(formatted_content) > max_context_length:
                logger.info("Reached maximum context length while adding notes. Stopping aggregation.")
                break
            aggregated_context += formatted_content
            referenced_note_ids.append(note_id)
            logger.debug(f"Added Note ID: {note_id}")

        # Include cluster summaries
        for cluster_id, cluster_summary in cluster_summaries.items():
            formatted_cluster_summary = f"Cluster ID: {cluster_id}\nSummary: {cluster_summary}\n\n"
            if len(aggregated_context) + len(formatted_cluster_summary) > max_context_length:
                logger.info("Reached maximum context length while adding cluster summaries. Stopping aggregation.")
                break
            aggregated_context += formatted_cluster_summary
            logger.debug(f"Added Cluster Summary ID: {cluster_id}")

        logger.info("Context aggregation complete.")

        # Step 5: Generate Final Answer with Markdown Formatting and Hyperlinks
        llm_client = get_llm_client()
        final_prompt = f"""
You are provided with the following context:

{aggregated_context}

Based on this context, please provide a concise and informative answer to the following question:

"{user_query}"

Your answer should be formatted in Markdown. For each section of your answer that is primarily based on a specific note, include a hyperlink to the note's GET endpoint in the format `[Note Content](http://knowledger.onrender.com/notes/{{note_id}})`. Ensure that the answer is well-structured and easy to read.

Remember to:

- Use bullet points or headings where appropriate.
- Highlight key points.
- Embed hyperlinks to the notes for further reference.

Notes GET endpoint format: `http://knowledger.onrender.com/notes/{{note_id}}`
"""

        logger.debug(f"Final Prompt Length: {len(final_prompt)}")
        final_response = await llm_client.generate_content(final_prompt)
        final_answer = final_response.strip()
        logger.info("Final answer generated successfully.")

        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds.")

        if total_time > 3:
            logger.warning("Processing time exceeded 3 seconds.")

    except Exception as e:
        logger.error(f"Error generating final answer from LLM: {e}")
        final_answer = "Error: Unable to generate final answer from LLM."
        referenced_note_ids = []

    return {
        "answer": final_answer,
        "referenced_note_ids": referenced_note_ids,
        "error": "" if not final_answer.startswith("Error") else final_answer
    }

@app.post("/compute_pagerank")
async def trigger_pagerank(background_tasks: BackgroundTasks, current_user: UserInDB = Depends(get_current_user)):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    background_tasks.add_task(compute_pagerank)
    return {"message": "PageRank computation started in the background."}

@app.post("/recalculate_all")
async def trigger_full_recalculation(background_tasks: BackgroundTasks, current_user: UserInDB = Depends(get_current_user)):
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    background_tasks.add_task(compute_pagerank)
    background_tasks.add_task(perform_clustering)
    return {"message": "Full recalculation and clustering started in the background."}

# -----------------------------------------------------------------------------
# Other Endpoints and Tester Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/test/mongodb")
async def test_mongodb():
    if not mongodb_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="MongoDB connection failed.")
    return {"mongodb": "Connection successful."}

@app.get("/test/llm")
async def test_llm():
    try:
        llm_client = get_llm_client()
        response = await llm_client.generate_content("Write a test summary.")
        return {"llm": "Connection successful.", "response": response}
    except Exception as e:
        logger.error(f"LLM API test failed: {e}")
        raise HTTPException(status_code=500, detail="LLM API test failed.")

@app.get("/test/model")
async def test_model():
    try:
        model = get_sentence_transformer()
        sample_text = "Test encoding"
        embedding = model.encode([sample_text], show_progress_bar=False)[0]
        return {
            "model": "Loaded successfully.",
            "embedding_length": len(embedding),
            "sample_embedding": embedding[:5].tolist()  # Return first 5 elements as a sample
        }
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        raise HTTPException(status_code=500, detail="SentenceTransformer model test failed.")

@app.head("/")
async def root():
    return {"message": "Hello from FastAPI!"}

# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------

import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=10000)
