# main.py

import os
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any


from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from config import Config
from models import NoteInput, NoteOutput, QueryInput, RAGQueryInput, SubPromptResponse, ParameterUpdate
from db import Neo4jConnection
from llm_client import get_llm_client
from utils import preprocess_for_embedding, parse_tree_of_thoughts
from sentence_transformers import SentenceTransformer

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Initialize Logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(title=Config.APP_TITLE)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Neo4j Connection
neo4j_conn = Neo4jConnection()
neo4j_conn.create_constraints()

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

def create_note_in_neo4j(note_id: str, content: str, processed_content: str,
                         embedding: List[float], timestamp: datetime, summary: str = ""):
    try:
        neo4j_conn.create_note(
            note_id=note_id,
            content=content,
            processed_content=processed_content,
            embedding=embedding,
            timestamp=timestamp.isoformat(),
            summary=summary
        )
        logger.debug(f"Note {note_id} created in Neo4j.")
    except Exception as e:
        logger.error(f"Error creating note {note_id}: {e}")
        raise

def generate_cluster_summary(note_ids: List[str]) -> str:
    combined_content = neo4j_conn.get_cluster_content(note_ids)
    prompt = Config.CLUSTER_SUMMARY_PROMPT_TEMPLATE.format(content=combined_content)
    llm_client = get_llm_client()
    summary = asyncio.run(llm_client.generate_content(prompt))
    return summary

async def generate_sub_prompt_response(idx: int, sub_prompt: str, note_id: str, llm_client, max_retries: int = 3) -> Dict[str, Any]:
    detailed_prompt = (
        f"{sub_prompt}\n\n"
        "Guidelines:\n"
        "1. Be concise and clear.\n"
        "2. Reference relevant notes where applicable by their ID.\n"
        "3. Ensure the information is accurate and well-organized.\n"
        "4. Maintain a neutral and informative tone."
    )
    logger.info(f"Processing Sub-Prompt {idx}: {sub_prompt[:50]}...")

    attempt = 0
    while attempt < max_retries:
        try:
            response = await llm_client.generate_content(prompt=detailed_prompt)
            logger.info(f"Completed Sub-Prompt {idx}: {sub_prompt[:50]}...")
            return {"prompt": sub_prompt, "response": response, "note_id": note_id}
        except Exception as e:
            attempt += 1
            wait_time = 2 ** attempt  # Exponential backoff
            logger.error(f"Error generating response for Sub-Prompt {idx}: {e}. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    logger.error(f"Failed to generate response for Sub-Prompt {idx} after {max_retries} attempts.")
    return {"prompt": sub_prompt, "response": f"Error generating response after {max_retries} attempts.", "note_id": note_id}

# -----------------------------------------------------------------------------
# Graph and Clustering Functions
# -----------------------------------------------------------------------------

import networkx as nx

def compute_pagerank():
    """
    Computes PageRank using NetworkX and updates the scores in Neo4j.
    """
    logger.info("Starting PageRank computation using NetworkX...")
    start_time = time.time()
    try:
        with neo4j_conn.driver.session() as session:
            # Retrieve all SIMILAR relationships
            logger.info("Retrieving relationships from Neo4j...")
            result = session.run("""
            MATCH (n1:Note)-[:SIMILAR]->(n2:Note)
            RETURN n1.id AS source, n2.id AS target
            """)
        
            # Build a directed graph
            G = nx.DiGraph()
            for record in result:
                source = record["source"]
                target = record["target"]
                if source and target:
                    G.add_edge(source, target)
        
            # Ensure all nodes are present in the graph
            logger.info("Retrieving all nodes from Neo4j...")
            nodes_result = session.run("""
            MATCH (n:Note)
            RETURN n.id AS id
            """)
            all_node_ids = [record["id"] for record in nodes_result if record["id"]]
            G.add_nodes_from(all_node_ids)
        
            if len(G) == 0:
                logger.warning("No nodes found in the graph. PageRank computation skipped.")
                return
        
            # Compute PageRank using NetworkX with optimized parameters
            logger.info("Computing PageRank scores...")
            pagerank_scores = nx.pagerank(G, alpha=Config.PAGERANK_ALPHA, max_iter=100, tol=1.0e-6)
        
            # Update PageRank scores back to Neo4j
            logger.info("Updating PageRank scores in Neo4j...")
            for node_id, score in pagerank_scores.items():
                session.run("""
                MATCH (n:Note {id: $id})
                SET n.pagerank = $score
                """, id=node_id, score=score)
        
        elapsed_time = time.time() - start_time
        logger.info(f"PageRank computation and update completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during PageRank computation: {e}")

def perform_clustering():
    """
    Performs clustering on note embeddings and updates Neo4j with Cluster nodes and relationships.
    """
    logger.info("Starting clustering process...")
    start_time = time.time()
    try:
        from sklearn.cluster import KMeans  # Faster alternative to DBSCAN
        import numpy as np

        with neo4j_conn.driver.session() as session:
            # Retrieve all embeddings
            logger.info("Retrieving embeddings from Neo4j...")
            result = session.run("""
            MATCH (n:Note)
            WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
            RETURN n.id AS id, n.embedding AS embedding, n.pagerank AS pagerank
            """)
        
            note_ids = []
            embeddings = []
            pageranks = []
            for record in result:
                note_id = record["id"]
                embedding = record["embedding"]
                pagerank = record["pagerank"]
                if note_id and embedding and pagerank is not None:
                    note_ids.append(note_id)
                    embeddings.append(embedding)
                    pageranks.append(pagerank)
        
            if not embeddings:
                logger.warning("No embeddings found for clustering.")
                return
        
            embeddings_array = np.array(embeddings)
            pageranks_array = np.array(pageranks)
        
            # Perform K-Means clustering
            num_clusters = Config.RAG_MAX_CLUSTERS
            logger.info(f"Performing K-Means clustering with {num_clusters} clusters.")
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
        
            # Remove existing Cluster nodes and relationships to prevent duplicates
            logger.info("Removing existing Cluster nodes and relationships...")
            session.run("""
            MATCH (c:Cluster)
            DETACH DELETE c
            """)
        
            # Create new Cluster nodes and assign notes
            clusters = {}
            for note_id, label, pagerank in zip(note_ids, cluster_labels, pageranks_array):
                clusters.setdefault(label, []).append((note_id, pagerank))
        
            for label, members in clusters.items():
                cluster_label = f"Cluster_{label}"
                member_ids = [member[0] for member in members]
                member_pageranks = [member[1] for member in members]
        
                # Compute Weighted Average Embedding
                member_embeddings = embeddings_array[[note_ids.index(mid) for mid in member_ids]]
                member_pageranks_np = np.array(member_pageranks)
                weighted_sum = np.sum(member_embeddings.T * member_pageranks_np, axis=1)
                total_pagerank = np.sum(member_pageranks_np)
                if total_pagerank > 0:
                    cluster_embedding = (weighted_sum / total_pagerank).tolist()
                else:
                    cluster_embedding = member_embeddings.mean(axis=0).tolist()  # Fallback to simple average
        
                # Generate cluster summary using LLM
                summary = generate_cluster_summary(member_ids)
        
                # Create Cluster node
                session.run("""
                MERGE (c:Cluster {label: $label})
                SET c.summary = $summary,
                    c.size = $size,
                    c.embedding = $embedding
                """, label=cluster_label, summary=summary, size=len(members), embedding=cluster_embedding)
        
                # Create BELONGS_TO relationships
                session.run("""
                UNWIND $note_ids AS note_id
                MATCH (n:Note {id: note_id}), (c:Cluster {label: $label})
                MERGE (n)-[:BELONGS_TO]->(c)
                """, note_ids=member_ids, label=cluster_label)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Clustering completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during clustering: {e}")

# -----------------------------------------------------------------------------
# Scheduler and Startup Events
# -----------------------------------------------------------------------------

scheduler = BackgroundScheduler()
scheduler.start()

# Schedule the compute_pagerank function to run every 30 minutes
scheduler.add_job(
    func=compute_pagerank,
    trigger=IntervalTrigger(minutes=30),
    id='compute_pagerank_job',
    name='Compute PageRank every 30 minutes',
    replace_existing=True
)

# Schedule the perform_clustering function to run every 30 minutes
scheduler.add_job(
    func=perform_clustering,
    trigger=IntervalTrigger(minutes=30),
    id='perform_clustering_job',
    name='Perform Clustering every 30 minutes',
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
    neo4j_conn.close()
    logger.info("Application shutdown complete.")

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.post("/notes", response_model=NoteOutput)
async def create_note(note_input: NoteInput, background_tasks: BackgroundTasks):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        note_id = generate_note_id()
        processed_content = preprocess_for_embedding(note_input.content)
        model = get_sentence_transformer()
        embedding = model.encode(processed_content)
        embedding = embedding.tolist()

        # Generate summary using LLM
        llm_client = get_llm_client()
        summary_prompt = Config.NOTE_SUMMARY_PROMPT_TEMPLATE.format(content=processed_content)
        summary = await llm_client.generate_content(summary_prompt)

        # Create note in Neo4j
        create_note_in_neo4j(
            note_id=note_id,
            content=note_input.content,
            processed_content=processed_content,
            embedding=embedding,
            timestamp=note_input.timestamp,
            summary=summary
        )

        # Update relationships and PageRank in the background
        background_tasks.add_task(neo4j_conn.update_relationships, note_id, Config.SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS)
        background_tasks.add_task(compute_pagerank)
        background_tasks.add_task(perform_clustering)

        # Retrieve the created note
        with neo4j_conn.driver.session() as session:
            result = session.run("""
            MATCH (n:Note {id: $id})
            RETURN n.id as id, n.content as content, n.processed_content as processed_content,
                   n.timestamp as timestamp, n.commonness as commonness, n.pagerank as pagerank, n.summary as summary
            """, id=note_id).single()

            if not result:
                raise HTTPException(status_code=404, detail="Note not found after creation.")

            return NoteOutput(
                id=result["id"],
                content=result["content"],
                processed_content=result["processed_content"],
                timestamp=datetime.fromisoformat(result["timestamp"]),
                commonness=result["commonness"],
                pagerank=result["pagerank"],
                summary=result.get("summary", "")
            )

    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag_query")
async def rag_query(
    rag_query_input: RAGQueryInput
):
    start_time = time.time()

    if not neo4j_conn.verify_connectivity():
        logger.error("Database connection error")
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        user_query = rag_query_input.query
        logger.info(f"Received Query: {user_query}")

        # Step 1: Refine Query
        refined_user_query = preprocess_for_embedding(user_query)
        logger.debug(f"Refined Query: {refined_user_query}")
        model = get_sentence_transformer()
        query_embedding = model.encode(refined_user_query).tolist()

        # Step 2: Retrieve Similar Notes
        similar_notes = neo4j_conn.get_similar_notes(
            query_embedding=query_embedding,
            similarity_threshold=Config.SIMILARITY_THRESHOLD_NOTE,
            limit=Config.RAG_MAX_NOTES_PER_CLUSTER
        )
        logger.info(f"Retrieved {len(similar_notes)} similar notes.")

        # Step 3: Aggregate Context
        aggregated_context = ""
        referenced_note_ids = []

        for note in similar_notes:
            note_id = note["id"]
            content = note["content"]

            formatted_content = f"Note ID: {note_id}\nContent: {content}\n\n"
            if len(aggregated_context) + len(formatted_content) > Config.RAG_MAX_CONTEXT_LENGTH:
                logger.info("Reached maximum context length while adding notes. Stopping aggregation.")
                break
            aggregated_context += formatted_content
            referenced_note_ids.append(note_id)
            logger.info(f"Added Note ID: {note_id}")
        logger.info("Context aggregation complete.")

        import google.generativeai as genai

        # Step 4: Generate Tree of Thoughts (ToT)
        llm_client = get_llm_client()
        logger.info("LLM Client created.")
        tree_of_thoughts_prompt = (
                f"{refined_user_query}\n\n"
                "Please analyze the above context and generate a JSON object representing a tree of thoughts. "
                "The JSON should have a key 'thoughts' which is a list of sub-prompts or questions to explore. "
                "Ensure that the JSON is properly formatted with double quotes."
            )
        tree_response = await llm_client.generate_content(tree_of_thoughts_prompt,
                                                          generation_config=genai.GenerationConfig(
                        response_mime_type="application/json", response_schema=list[SubPromptResponse]
                    ))
        logger.debug(f"Tree of Thoughts Response: {tree_response[:500]}...")  # Log first 500 chars

        try:
            parsed_tree = parse_tree_of_thoughts(tree_response)
            sub_prompts = [x['prompt'] for x in parsed_tree]
            logger.info(f"Extracted {len(sub_prompts)} sub-prompts.")
        except ValueError as e:
            logger.error(f"Failed to parse Tree of Thoughts JSON: {e}")
            raise HTTPException(status_code=500, detail="Error parsing Tree of Thoughts JSON from LLM")

        # Step 5: Process Sub-Prompts Concurrently
        sub_prompt_tasks = [
            generate_sub_prompt_response(idx=i, sub_prompt=prompt, note_id=referenced_note_ids[i % len(referenced_note_ids)], llm_client=llm_client)
            for i, prompt in enumerate(sub_prompts)
        ]

        sub_prompt_responses = await asyncio.gather(*sub_prompt_tasks)
        logger.info(f"Generated responses for {len(sub_prompt_responses)} sub-prompts.")

        # Step 6: Aggregate Sub-Responses
        aggregated_sub_responses = "\n".join([
            f"Sub-Prompt {i+1}: {resp['response']}" for i, resp in enumerate(sub_prompt_responses)
            if resp['response'] and not resp['response'].startswith("Error")
        ])
        logger.debug(f"Aggregated Sub-Responses Length: {len(aggregated_sub_responses)}")

        # Step 7: Final LLM Generation
        final_prompt = f"""{aggregated_context}\n\n{aggregated_sub_responses}\n\n
#CONTEXT: You are to create a concise synthesis of the provided information, suitable for someone who needs to quickly understand the core content but may want to explore certain aspects in more detail.

#ROLE: You act as a concise guide, summarizing key points while offering a roadmap for further exploration through embedded links.

#RESPONSE GUIDELINES:

Use brief, clear sentences or bullet points to summarize.
Embed links to notes that you consulted in a way that doesn't clutter the main points.
The goal is to balance quick readability with opportunities for deeper dives.
#TASK CRITERIA:

Provide bullet points.
Ensure linked notes are relevant and enhance the reader's ability to explore the topic further.
Keep sentences short to support quick understanding, but prefer to keep more information rather than shorten sentences. The information content is the most important part here.

#OUTPUT:
A list of concise bullet points with relevant notes to expand on key ideas.
Use formatting and highlighting syntax in order to draw attention to important points in the overview.
The less bullet points the better, though making sure that the information is as full as possible would be best. 
        """
        logger.debug(f"Final Prompt Length: {len(final_prompt)}")
        final_response = await asyncio.wait_for(
            llm_client.generate_content(final_prompt),
            timeout=3  # Ensuring the LLM call doesn't exceed the overall timeout
        )
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

@app.post("/query", response_model=List[NoteOutput])
async def query_notes(query_input: QueryInput):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        processed_query = preprocess_for_embedding(query_input.query)
        model = get_sentence_transformer()
        query_embedding = model.encode(processed_query).tolist()

        similar_notes = neo4j_conn.get_similar_notes(
            query_embedding=query_embedding,
            similarity_threshold=Config.SIMILARITY_THRESHOLD_RAG,
            limit=query_input.limit
        )
        logger.info(f"Retrieved {len(similar_notes)} notes for query.")

        return [
            NoteOutput(
                id=note["id"],
                content=note["content"],
                processed_content=note["processed_content"],
                timestamp=datetime.fromisoformat(note["timestamp"]),
                commonness=note["commonness"],
                pagerank=note["pagerank"],
                summary=note.get("summary", "")
            ) for note in similar_notes
        ]
    except Exception as e:
        logger.error(f"Error querying notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notes/{note_id}", response_model=NoteOutput)
async def get_note(note_id: str):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        result = neo4j_conn.get_note(note_id)
        if not result:
            raise HTTPException(status_code=404, detail="Note not found")

        return NoteOutput(
            id=result["id"],
            content=result["content"],
            processed_content=result["processed_content"],
            timestamp=datetime.fromisoformat(result["timestamp"]),
            commonness=result["commonness"],
            pagerank=result["pagerank"],
            summary=result.get("summary", "")
        )

    except Exception as e:
        logger.error(f"Error retrieving note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compute_pagerank")
async def trigger_pagerank(background_tasks: BackgroundTasks):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    background_tasks.add_task(compute_pagerank)
    return {"message": "PageRank computation started in the background."}

@app.post("/recalculate_all")
async def trigger_full_recalculation(background_tasks: BackgroundTasks):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    background_tasks.add_task(compute_pagerank)
    background_tasks.add_task(perform_clustering)
    return {"message": "Full recalculation and clustering started in the background."}

# -----------------------------------------------------------------------------
# Tester Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.get("/test/neo4j")
async def test_neo4j():
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Neo4j connection failed.")

    try:
        with neo4j_conn.driver.session() as session:
            result = session.run("RETURN 1 AS result")
            record = result.single()
            if record and record["result"] == 1:
                return {"neo4j": "Connection successful.", "result": record["result"]}
            else:
                return {"neo4j": "Unexpected result.", "result": record["result"] if record else None}
    except Exception as e:
        logger.error(f"Neo4j test failed: {e}")
        raise HTTPException(status_code=500, detail="Neo4j test failed.")

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
        embedding = await asyncio.to_thread(model.encode, sample_text)
        return {
            "model": "Loaded successfully.",
            "embedding_length": len(embedding),
            "sample_embedding": embedding[:5]  # Return first 5 elements as a sample
        }
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        raise HTTPException(status_code=500, detail="SentenceTransformer model test failed.")
    

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}

import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
