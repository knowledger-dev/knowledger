# --- START OF FILE app.py ---

import os
import re
import logging
import asyncio
import atexit
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from functools import lru_cache

# -----------------------------------------------------------------------------
# Configuration Variables
# -----------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# FastAPI configuration
APP_TITLE = os.getenv("APP_TITLE", "Note-Taking API with LLM Features")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Gemini API Configuration
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")  # Ensure this is set in .env
GENIUS_MODEL = os.getenv("GENIUS_MODEL", "gemini-1.5-flash")

# Similarity thresholds
SIMILARITY_THRESHOLD_RECALCULATE_ALL = float(os.getenv("SIMILARITY_THRESHOLD_RECALCULATE_ALL", 0.3875))
SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS = float(os.getenv("SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS", 0.3875))
SIMILARITY_THRESHOLD_RAG = 0.1

# DBSCAN parameters
DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", 1.1725))
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 2))

# RAG configuration
RAG_MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", 128000))  # Adjusted to be within typical token limits
RAG_DEFAULT_MAX_TOKENS = 4096  # Adjusted for typical LLMs

# -----------------------------------------------------------------------------
# Initialize Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Initialize FastAPI App
# -----------------------------------------------------------------------------

app = FastAPI(title=APP_TITLE)

# -----------------------------------------------------------------------------
# Initialize Neo4j Connection
# -----------------------------------------------------------------------------

from neo4j import GraphDatabase  # Imported here since it's lightweight

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def verify_connectivity(self):
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            # Use IF NOT EXISTS to avoid redundant operations
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Note) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cluster) REQUIRE c.label IS UNIQUE")

neo4j_conn = Neo4jConnection()
neo4j_conn.create_constraints()

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class NoteInput(BaseModel):
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class NoteOutput(BaseModel):
    id: str
    content: str
    processed_content: str
    timestamp: datetime
    commonness: int
    pagerank: float
    summary: str = ""  # New field for AI-generated insights

class QueryInput(BaseModel):
    query: str
    limit: int = 5

class RAGQueryInput(BaseModel):
    query: str
    max_tokens: int = RAG_DEFAULT_MAX_TOKENS  # Default maximum number of tokens in the response

# -----------------------------------------------------------------------------
# Text Preprocessing
# -----------------------------------------------------------------------------

class LightweightTextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
            r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.punct_pattern = re.compile(r'[^\w\s]')

    def clean_text(self, text: str) -> str:
        text = self.url_pattern.sub('[URL]', text)
        text = ' '.join(text.split())
        return text.strip()

    def process_text(self, text: str) -> Dict[str, Any]:
        cleaned_text = self.clean_text(text)
        processed_text = self.punct_pattern.sub(' ', cleaned_text)
        processed_text = ' '.join(processed_text.split())
        return {
            'processed_text': processed_text,
            'original_text': cleaned_text,
            'token_count': len(processed_text.split())
        }

preprocessor = LightweightTextPreprocessor()

@lru_cache(maxsize=1000)
def preprocess_for_embedding(text: str) -> str:
    result = preprocessor.process_text(text)
    return result['processed_text']

# -----------------------------------------------------------------------------
# LLM Client and Helper Functions
# -----------------------------------------------------------------------------

import google.generativeai as genai  # Importing Gemini API client

class LLMClient:
    def __init__(self, model: str):
        self.model = genai.GenerativeModel(model)

    def generate_content(self, prompt: str, stream: bool = False, generation_config: Dict[str, Any] = None):
        """
        Generates content using Gemini API.
        Supports both streaming and non-streaming responses.
        """
        if generation_config:
            gen_config = genai.types.GenerationConfig(**generation_config)
        else:
            gen_config = None

        if stream:
            response = self.model.generate_content(prompt, stream=True, generation_config=gen_config)
            return response  # This should be a generator
        else:
            response = self.model.generate_content(prompt, stream=False, generation_config=gen_config)
            return response.text.strip()

# Initialize Gemini Client lazily
def get_llm_client():
    if not hasattr(get_llm_client, "client"):
        if not GENIUS_API_KEY:
            raise ValueError("GENIUS_API_KEY environment variable not set")
        genai.configure(api_key=GENIUS_API_KEY)
        get_llm_client.client = LLMClient(model=GENIUS_MODEL)
    return get_llm_client.client

# -----------------------------------------------------------------------------
# Dependency Injection for SentenceTransformer
# -----------------------------------------------------------------------------

async def get_sentence_transformer():
    if not hasattr(get_sentence_transformer, "model"):
        from sentence_transformers import SentenceTransformer  # Lazy import
        loop = asyncio.get_event_loop()
        get_sentence_transformer.model = await loop.run_in_executor(None, SentenceTransformer, 'all-MiniLM-L6-v2')
    return get_sentence_transformer.model

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def generate_note_id() -> str:
    return os.urandom(8).hex()

def create_note_in_neo4j(note_id: str, content: str, processed_content: str,
                         embedding: List[float], timestamp: datetime, summary: str = ""):
    with neo4j_conn.driver.session() as session:
        session.run("""
        CREATE (n:Note {
            id: $id,
            content: $content,
            processed_content: $processed_content,
            embedding: $embedding,
            timestamp: $timestamp,
            commonness: 0,
            pagerank: 0.0,
            summary: $summary
        })
        """,
        id=note_id,
        content=content,
        processed_content=processed_content,
        embedding=embedding,
        timestamp=timestamp.isoformat(),
        summary=summary)

def generate_cluster_label(cluster_id: int) -> str:
    return f"Cluster_{cluster_id}"

def generate_cluster_content(note_ids: List[str]) -> str:
    with neo4j_conn.driver.session() as session:
        contents = session.run("""
        MATCH (n:Note)
        WHERE n.id IN $ids
        RETURN n.content AS content
        """, ids=note_ids)
        combined_content = "\n".join([record["content"] for record in contents])
    return combined_content

def generate_cluster_title(note_ids: List[str]) -> str:
    combined_content = generate_cluster_content(note_ids)
    prompt = f"Generate a concise and descriptive title for the following content:\n{combined_content}"
    llm_client = get_llm_client()
    title = llm_client.generate_content(prompt)
    return title if title else "Untitled Cluster"

def generate_cluster_summary(note_ids: List[str]) -> str:
    combined_content = generate_cluster_content(note_ids)
    prompt = f"Summarize the following content in a few sentences:\n{combined_content}"
    llm_client = get_llm_client()
    summary = llm_client.generate_content(prompt)
    return summary

def refine_query(query: str) -> str:
    prompt = f"Please refine the following query for better search results:\n{query}"
    llm_client = get_llm_client()
    refined_query = llm_client.generate_content(prompt)
    return refined_query if refined_query else query

def get_dynamic_context(results) -> str:
    context = ""
    for record in results:
        note_id = record.get("id", "unknown_id")
        content = record.get("content", "")
        # Format each note with its ID for reference
        formatted_content = f"Note ID: {note_id}\nContent: {content}\n\n"
        if len(context) + len(formatted_content) > RAG_MAX_CONTEXT_LENGTH:
            break
        context += formatted_content
    return context

# -----------------------------------------------------------------------------
# Graph and Clustering Functions
# -----------------------------------------------------------------------------

def compute_pagerank():
    """
    Retrieves the graph data from Neo4j, computes PageRank using NetworkX,
    and updates the PageRank scores back to Neo4j.
    """
    import networkx as nx  # Lazy import
    logger.info("Starting PageRank computation using NetworkX...")

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
            else:
                logger.warning(f"Edge with missing source or target: {record}")

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

        # Compute PageRank using NetworkX
        logger.info("Computing PageRank scores...")
        pagerank_alpha = float(os.getenv("PAGERANK_ALPHA", 0.85))
        pagerank_scores = nx.pagerank(G, alpha=pagerank_alpha)

        # Update PageRank scores back to Neo4j
        logger.info("Updating PageRank scores in Neo4j...")
        pagerank_data = [{'id': node_id, 'score': score} for node_id, score in pagerank_scores.items()]
        session.run("""
        UNWIND $pagerank_data AS pr
        MATCH (n:Note {id: pr.id})
        SET n.pagerank = pr.score
        """, pagerank_data=pagerank_data)

    logger.info("PageRank computation and update completed.")

def update_note_relationships(note_id: str):
    with neo4j_conn.driver.session() as session:
        logger.info(f"Updating relationships for note ID: {note_id}")
        # Compute cosine similarity and create relationships
        session.run("""
        MATCH (n1:Note {id: $id})
        MATCH (n2:Note)
        WHERE n1 <> n2 AND n1.embedding IS NOT NULL AND n2.embedding IS NOT NULL
        WITH n1, n2,
             reduce(a = 0.0, x IN range(0, size(n1.embedding)-1) |
                   a + (n1.embedding[x] * n2.embedding[x])) AS dotProduct,
             sqrt(reduce(a = 0.0, x IN n1.embedding | a + x * x)) AS mag1,
             sqrt(reduce(a = 0.0, x IN n2.embedding | a + x * x)) AS mag2
        WITH n1, n2, dotProduct / (mag1 * mag2) AS similarity
        WHERE similarity > $similarity_threshold
        MERGE (n1)-[r:SIMILAR {score: similarity}]->(n2)
        """, id=note_id, similarity_threshold=SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS)

    # Recompute PageRank after updating relationships
    compute_pagerank()

def recalculate_all():
    logger.info("Starting full recalculation of similarity scores, relationships, and PageRank...")
    with neo4j_conn.driver.session() as session:
        # Step 1: Update embeddings for all notes (if necessary)
        logger.info("Ensuring all notes have embeddings...")
        missing_embeddings = session.run("""
        MATCH (n:Note)
        WHERE n.embedding IS NULL OR size(n.embedding) = 0
        RETURN n.id AS id, n.content AS content
        """)

        # Lazy import SentenceTransformer here
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        for record in missing_embeddings:
            note_id = record["id"]
            content = record["content"]
            processed_content = preprocess_for_embedding(content)
            embedding = model.encode(processed_content).tolist()

            session.run("""
            MATCH (n:Note {id: $id})
            SET n.embedding = $embedding
            """, id=note_id, embedding=embedding)

        # Step 2: Delete existing SIMILAR relationships
        logger.info("Deleting existing SIMILAR relationships...")
        session.run("""
        MATCH ()-[r:SIMILAR]->()
        DELETE r
        """)

        # Step 3: Recalculate similarity scores and create new relationships
        logger.info("Recalculating similarity scores and relationships...")
        session.run("""
        MATCH (n1:Note), (n2:Note)
        WHERE id(n1) < id(n2) AND n1.embedding IS NOT NULL AND n2.embedding IS NOT NULL
        WITH n1, n2,
             reduce(a = 0.0, x IN range(0, size(n1.embedding)-1) |
                   a + (n1.embedding[x] * n2.embedding[x])) AS dotProduct,
             sqrt(reduce(a = 0.0, x IN n1.embedding | a + x * x)) AS mag1,
             sqrt(reduce(a = 0.0, x IN n2.embedding | a + x * x)) AS mag2
        WITH n1, n2, dotProduct / (mag1 * mag2) AS similarity
        WHERE similarity > $similarity_threshold
        MERGE (n1)-[:SIMILAR {score: similarity}]->(n2)
        MERGE (n2)-[:SIMILAR {score: similarity}]->(n1)
        """, similarity_threshold=SIMILARITY_THRESHOLD_RECALCULATE_ALL)

    # Step 4: Recompute PageRank
    compute_pagerank()

    # Step 5: Recluster
    perform_clustering()
    logger.info("Full recalculation completed.")

def perform_clustering():
    """
    Performs DBSCAN clustering on all note embeddings and updates Neo4j with Cluster nodes and relationships.
    """
    logger.info("Starting DBSCAN clustering...")
    from sklearn.cluster import DBSCAN  # Lazy import
    import numpy as np  # Lazy import

    with neo4j_conn.driver.session() as session:
        # Retrieve all embeddings
        logger.info("Retrieving embeddings from Neo4j...")
        result = session.run("""
        MATCH (n:Note)
        WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
        RETURN n.id AS id, n.embedding AS embedding
        """)

        note_ids = []
        embeddings = []
        for record in result:
            note_id = record["id"]
            embedding = record["embedding"]
            if note_id and embedding:
                note_ids.append(note_id)
                embeddings.append(embedding)

        if not embeddings:
            logger.warning("No embeddings found for clustering.")
            return

        embeddings_array = np.array(embeddings)

        # Perform DBSCAN clustering with configurable parameters
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        cluster_labels = dbscan.fit_predict(embeddings_array)

        unique_labels = set(cluster_labels)
        logger.info(f"Number of clusters found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")

        # Remove existing Cluster nodes and relationships to prevent duplicates
        logger.info("Removing existing Cluster nodes and relationships...")
        session.run("""
        MATCH (c:Cluster)
        DETACH DELETE c
        """)

        # Create new Cluster nodes and assign notes
        clusters = {}
        for note_id, label in zip(note_ids, cluster_labels):
            if label == -1:
                continue  # Skip noise points
            clusters.setdefault(label, []).append(note_id)

        # Prepare data for batch operations
        cluster_data = []
        for label, members in clusters.items():
            cluster_label = generate_cluster_label(label)
            title = generate_cluster_title(members)       # Generate cluster title using Gemini
            summary = generate_cluster_summary(members)   # Generate cluster summary using Gemini
            cluster_data.append({
                'label': cluster_label,
                'title': title,
                'summary': summary,
                'size': len(members),
                'note_ids': members
            })

        # Create Cluster nodes and BELONGS_TO relationships in batch
        for cluster in cluster_data:
            session.run("""
            MERGE (c:Cluster {label: $label})
            SET c.title = $title,
                c.summary = $summary,
                c.size = $size
            """, label=cluster['label'], title=cluster['title'], summary=cluster['summary'], size=cluster['size'])

            # Create BELONGS_TO relationships
            session.run("""
            UNWIND $note_ids AS note_id
            MATCH (n:Note {id: note_id}), (c:Cluster {label: $label})
            MERGE (n)-[:BELONGS_TO]->(c)
            """, note_ids=cluster['note_ids'], label=cluster['label'])

    logger.info("DBSCAN clustering and Neo4j updates completed.")

# -----------------------------------------------------------------------------
# Scheduler and Startup Events
# -----------------------------------------------------------------------------

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

scheduler = BackgroundScheduler()
scheduler.start()

# Schedule the recalculate_all function to run every 5 minutes
scheduler.add_job(
    func=recalculate_all,
    trigger=IntervalTrigger(minutes=5),
    id='recalculate_all_job',
    name='Recalculate all similarities and PageRank every 5 minutes',
    replace_existing=True
)

# Schedule the clustering to run every 5 minutes as well
scheduler.add_job(
    func=perform_clustering,
    trigger=IntervalTrigger(minutes=5),
    id='dbscan_clustering_job',
    name='Perform DBSCAN clustering every 5 minutes',
    replace_existing=True
)

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

@app.on_event("startup")
async def startup_event():
    """
    This function runs when the application starts.
    It triggers the recalculate_all function in the background.
    """
    logger.info("Starting initial recalculation...")
    asyncio.create_task(asyncio.to_thread(recalculate_all))

@app.on_event("shutdown")
async def shutdown_event():
    neo4j_conn.close()

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.post("/notes", response_model=NoteOutput)
async def create_note(note_input: NoteInput, background_tasks: BackgroundTasks, model=Depends(get_sentence_transformer)):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        note_id = generate_note_id()
        processed_content = preprocess_for_embedding(note_input.content)
        embedding = await asyncio.to_thread(model.encode, processed_content)
        embedding = embedding.tolist()

        # Generate summary using Gemini API
        llm_client = get_llm_client()
        summary_prompt = f"Summarize the following content:\n{processed_content}"
        summary = llm_client.generate_content(prompt=summary_prompt)

        create_note_in_neo4j(
            note_id,
            note_input.content,
            processed_content,
            embedding,
            note_input.timestamp,
            summary=summary
        )

        # Update relationships and PageRank in the background
        background_tasks.add_task(update_note_relationships, note_id)

        # Retrieve the created note
        with neo4j_conn.driver.session() as session:
            result = session.run("""
            MATCH (n:Note {id: $id})
            RETURN n.id as id, n.content as content, n.processed_content as processed_content,
                   n.timestamp as timestamp, n.commonness as commonness, n.pagerank as pagerank, n.summary as summary
            """, id=note_id).single()

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

@app.post("/query", response_model=List[NoteOutput])
async def query_notes(query_input: QueryInput, model=Depends(get_sentence_transformer)):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        processed_query = preprocess_for_embedding(query_input.query)
        query_embedding = await asyncio.to_thread(model.encode, processed_query)
        query_embedding = query_embedding.tolist()

        with neo4j_conn.driver.session() as session:
            # Ensure all notes have embeddings
            missing_embeddings = session.run("""
            MATCH (n:Note)
            WHERE n.embedding IS NULL OR size(n.embedding) = 0
            RETURN n.id AS id, n.content AS content
            """)

            for record in missing_embeddings:
                note_id = record["id"]
                content = record["content"]
                processed_content = preprocess_for_embedding(content)
                embedding = await asyncio.to_thread(model.encode, processed_content)
                embedding = embedding.tolist()

                session.run("""
                MATCH (n:Note {id: $id})
                SET n.embedding = $embedding
                """, id=note_id, embedding=embedding)

            # Find similar notes using cosine similarity
            results = session.run("""
            MATCH (n:Note)
            WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
            WITH n,
                 reduce(a = 0.0, x IN range(0, size(n.embedding)-1) |
                       a + (n.embedding[x] * $query_embedding[x])) AS dotProduct,
                 sqrt(reduce(a = 0.0, x IN $query_embedding | a + x * x)) AS magQuery,
                 sqrt(reduce(a = 0.0, x IN n.embedding | a + x * x)) AS magNote
            WITH n, dotProduct / (magQuery * magNote) AS similarity
            WHERE similarity > $similarity_threshold
            ORDER BY similarity DESC
            LIMIT $limit
            RETURN n.id as id, n.content as content, n.processed_content as processed_content,
                   n.timestamp as timestamp, n.commonness as commonness, n.pagerank as pagerank, n.summary as summary
            """, query_embedding=query_embedding, similarity_threshold=SIMILARITY_THRESHOLD_RAG, limit=query_input.limit)

            return [NoteOutput(
                id=record["id"],
                content=record["content"],
                processed_content=record["processed_content"],
                timestamp=datetime.fromisoformat(record["timestamp"]),
                commonness=record["commonness"],
                pagerank=record["pagerank"],
                summary=record.get("summary", "")
            ) for record in results]

    except Exception as e:
        logger.error(f"Error querying notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notes/{note_id}", response_model=NoteOutput)
async def get_note(note_id: str):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        with neo4j_conn.driver.session() as session:
            result = session.run("""
            MATCH (n:Note {id: $id})
            SET n.commonness = n.commonness + 1
            RETURN n.id as id, n.content as content, n.processed_content as processed_content,
                   n.timestamp as timestamp, n.commonness as commonness, n.pagerank as pagerank, n.summary as summary
            """, id=note_id).single()

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
    """
    Endpoint to manually trigger PageRank computation.
    """
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    background_tasks.add_task(compute_pagerank)
    return {"message": "PageRank computation started in the background."}

@app.post("/recalculate_all")
async def trigger_full_recalculation(background_tasks: BackgroundTasks):
    """
    Endpoint to manually trigger full recalculation of similarities, clustering, and PageRank.
    """
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    background_tasks.add_task(recalculate_all)
    return {"message": "Full recalculation and clustering started in the background."}

# -----------------------------------------------------------------------------
# Tester Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """
    General health check endpoint.
    """
    return {"status": "OK"}

@app.get("/test/neo4j")
async def test_neo4j():
    """
    Test Neo4j connection by performing a simple query.
    """
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

@app.get("/test/genius")
async def test_genius():
    """
    Test Gemini API client by generating a simple content.
    """
    try:
        llm_client = get_llm_client()
        response = llm_client.generate_content("Write a test summary.")
        return {"gemini": "Connection successful.", "response": response}
    except Exception as e:
        logger.error(f"Gemini API test failed: {e}")
        raise HTTPException(status_code=500, detail="Gemini API test failed.")

@app.get("/test/model")
async def test_model():
    """
    Test SentenceTransformer model by encoding a sample text.
    """
    try:
        model = await get_sentence_transformer()
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

# -----------------------------------------------------------------------------
# RAG Endpoint
# -----------------------------------------------------------------------------

@app.post("/rag_query")
async def rag_query(rag_query_input: RAGQueryInput, model=Depends(get_sentence_transformer)):
    if not neo4j_conn.verify_connectivity():
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        # Refine the user's query
        refined_query = refine_query(rag_query_input.query)
        processed_query = preprocess_for_embedding(refined_query)
        query_embedding = await asyncio.to_thread(model.encode, processed_query)
        query_embedding = query_embedding.tolist()

        with neo4j_conn.driver.session() as session:
            # Retrieve relevant notes from Neo4j with higher pagerank priority
            results = session.run("""
            MATCH (n:Note)
            WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
            WITH n,
                 reduce(a = 0.0, x IN range(0, size(n.embedding)-1) |
                       a + (n.embedding[x] * $query_embedding[x])) AS dotProduct,
                 sqrt(reduce(a = 0.0, x IN $query_embedding | a + x * x)) AS magQuery,
                 sqrt(reduce(a = 0.0, x IN n.embedding | a + x * x)) AS magNote,
                 n.pagerank AS pagerank
            WITH n, dotProduct / (magQuery * magNote) AS similarity, pagerank
            WHERE similarity > $similarity_threshold
            RETURN n.id AS id, n.content AS content, similarity, pagerank
            ORDER BY similarity DESC, pagerank DESC
            LIMIT 5
            """, query_embedding=query_embedding, similarity_threshold=SIMILARITY_THRESHOLD_RAG)

            results = list(results)

        if not results:
            return {"answer": "No relevant information found to answer your query.", "referenced_note_ids": []}

        # Prepare the context from the retrieved notes
        context = get_dynamic_context(results)
        referenced_note_ids = [record["id"] for record in results]

        # Ensure context length is within limits
        if len(context) > RAG_MAX_CONTEXT_LENGTH:
            context = context[:RAG_MAX_CONTEXT_LENGTH]

        # Generate the response using Gemini API
        prompt = f"Context:\n{context}\nQuestion:\n{rag_query_input.query}"
        llm_client = get_llm_client()
        answer = llm_client.generate_content(
            prompt=prompt,
            stream=False,
            generation_config={
                "candidate_count": 1,
                "stop_sequences": ["x"],
                "max_output_tokens": rag_query_input.max_tokens,
                "temperature": 1.0,
            }
        )

        return {"answer": answer, "referenced_note_ids": referenced_note_ids}

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

# --- END OF FILE app.py ---
