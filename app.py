# --- START OF FILE app.py ---

import json
import os
import re
import time
import logging
import asyncio
import atexit
from datetime import datetime
import typing
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from functools import lru_cache

# -----------------------------------------------------------------------------
# CONFIGURATION VARIABLES
# -----------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# -------------------- Logging Configuration --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # e.g., DEBUG, INFO, WARNING, ERROR

# -------------------- FastAPI Configuration --------------------
APP_TITLE = os.getenv("APP_TITLE", "Note-Taking API with LLM Features")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))

# -------------------- Neo4j Configuration --------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# -------------------- Gemini API Configuration --------------------
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY_1")  # Ensure this is set in .env
GENIUS_API_KEY_2 = os.getenv("GENIUS_API_KEY_2")  # Ensure this is set in .env
GENIUS_API_KEY_3 = os.getenv("GENIUS_API_KEY_3")  # Ensure this is set in .env
GENIUS_MODEL = "gemini-1.5-flash-8b"

# -------------------- Similarity Thresholds --------------------
# Constants (Adjust these as per your requirements)
SIMILARITY_THRESHOLD_NOTE = 0.1
CLUSTER_FAVOR_WEIGHT = 1.2  # Slightly favor clusters
AGENTIC_PROMPT_TIMEOUT = 10  # Total timeout in seconds for prompting and exploration


SIMILARITY_THRESHOLD_RECALCULATE_ALL = float(os.getenv("SIMILARITY_THRESHOLD_RECALCULATE_ALL", 0.3875))
SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS = float(os.getenv("SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS", 0.3875))
SIMILARITY_THRESHOLD_RAG = float(os.getenv("SIMILARITY_THRESHOLD_RAG", 0.1))
# *** New Threshold for Cluster Retrieval ***
SIMILARITY_THRESHOLD_CLUSTER = float(os.getenv("SIMILARITY_THRESHOLD_CLUSTER", 0.2))

# -------------------- DBSCAN Parameters --------------------
DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", 1.1725))
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 2))

# -------------------- RAG Configuration --------------------
RAG_MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", 1000000))  # Adjusted to be within typical token limits
RAG_DEFAULT_MAX_TOKENS = int(os.getenv("RAG_DEFAULT_MAX_TOKENS", 32768))  # Adjusted for typical LLMs
# *** New RAG Hyperparameters ***
RAG_MAX_CLUSTERS = int(os.getenv("RAG_MAX_CLUSTERS", 5))
RAG_MAX_NOTES_PER_CLUSTER = int(os.getenv("RAG_MAX_NOTES_PER_CLUSTER", 10))

# -------------------- PageRank Configuration --------------------
PAGERANK_ALPHA = float(os.getenv("PAGERANK_ALPHA", 0.85))  # Damping factor for PageRank

# -------------------- SentenceTransformer Model --------------------
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# -------------------- Gemini Generation Configuration --------------------
GENIUS_GENERATION_CONFIG = {
    "candidate_count": int(os.getenv("GENIUS_CANDIDATE_COUNT", 1)),
    # "stop_sequences": os.getenv("GENIUS_STOP_SEQUENCES").split(","),
    "temperature": float(os.getenv("GENIUS_TEMPERATURE", 1.0)),
}

# *** Prompt Templates ***
REFINE_QUERY_PROMPT_TEMPLATE = os.getenv("REFINE_QUERY_PROMPT_TEMPLATE",
    "Please refine the following query for better search results:\n{query}")

CLUSTER_TITLE_PROMPT_TEMPLATE = os.getenv("CLUSTER_TITLE_PROMPT_TEMPLATE",
    "Generate a concise and descriptive title for the following content:\n{content}")

CLUSTER_SUMMARY_PROMPT_TEMPLATE = os.getenv("CLUSTER_SUMMARY_PROMPT_TEMPLATE",
    "Summarize the following content in a few sentences:\n{content}")

NOTE_SUMMARY_PROMPT_TEMPLATE = os.getenv("NOTE_SUMMARY_PROMPT_TEMPLATE",
    "Summarize the following content:\n{content}")


# Global configuration dictionary
CONFIG = {
    "SIMILARITY_THRESHOLD_RECALCULATE_ALL": 0.3875,
    "SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS": 0.3875,
    "SIMILARITY_THRESHOLD_RAG": 0.1,
    "SIMILARITY_THRESHOLD_CLUSTER": 0.2,
    "DBSCAN_EPS": 1.1725,
    "DBSCAN_MIN_SAMPLES": 2,
    "RAG_MAX_CLUSTERS": 5,
    "RAG_MAX_NOTES_PER_CLUSTER": 10,
    "PAGERANK_ALPHA": 0.85
}

# -----------------------------------------------------------------------------
# END OF CONFIGURATION VARIABLES
# -----------------------------------------------------------------------------

# Initialize Logging
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

class SubPromptResponse(TypedDict):
    prompt: str
    response: str

class ParameterUpdate(BaseModel):
    SIMILARITY_THRESHOLD_RECALCULATE_ALL: float
    SIMILARITY_THRESHOLD_UPDATE_RELATIONSHIPS: float
    SIMILARITY_THRESHOLD_RAG: float
    SIMILARITY_THRESHOLD_CLUSTER: float
    DBSCAN_EPS: float
    DBSCAN_MIN_SAMPLES: int
    RAG_MAX_CLUSTERS: int
    RAG_MAX_NOTES_PER_CLUSTER: int
    PAGERANK_ALPHA: float

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

    def generate_content(self, prompt: str, stream: bool = False, generation_config = None):
        """
        Generates content using Gemini API.
        Supports both streaming and non-streaming responses.
        """
        if generation_config:
            gen_config = generation_config
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
        get_sentence_transformer.model = await loop.run_in_executor(None, SentenceTransformer, SENTENCE_TRANSFORMER_MODEL)
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
    prompt = CLUSTER_TITLE_PROMPT_TEMPLATE.format(content=combined_content)
    llm_client = get_llm_client()
    title = llm_client.generate_content(prompt)
    return title if title else "Untitled Cluster"

def generate_cluster_summary(note_ids: List[str]) -> str:
    combined_content = generate_cluster_content(note_ids)
    prompt = CLUSTER_SUMMARY_PROMPT_TEMPLATE.format(content=combined_content)
    llm_client = get_llm_client()
    summary = llm_client.generate_content(prompt)
    return summary

def refine_query(query: str) -> str:
    prompt = REFINE_QUERY_PROMPT_TEMPLATE.format(query=query)
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

# *** Updated Helper Function for Prompt Refinement ***
def refine_user_query_for_rag(context: str, query: str) -> str:
    """
    Refines the user query into a structured prompt with Context and Message sections.
    """
    refined_prompt = f"""### Task
Assist in retrieving, organizing, and summarizing relevant knowledge from the Knowledger platform to effectively respond to user queries. Leverage the platform's graph-based note-taking system, dynamic clustering, and PageRank-weighted insights to provide comprehensive and contextually accurate answers.

### Instructions
1. **Understand the User Query:**
   - Analyze the user's input to determine the primary intent and the specific information they seek.
   - Identify any keywords or phrases that indicate the topic or area of interest.

2. **Incorporate Relevant Context:**
   - Utilize Knowledger's dynamic embeddings and clustering (NoteRank) to gather relevant notes and insights.
   - Prioritize information based on PageRank scores to ensure high-quality and influential knowledge is surfaced.
   - Aggregate context from multiple clusters if necessary to provide a comprehensive response.

3. **Generate the Response:**
   - Use the refined prompt and the aggregated context to generate a coherent and insightful answer.
   - Ensure the response is clear, concise, and directly addresses the user's query.
   - Maintain transparency by referencing relevant clusters and notes where applicable.

4. **Structure the Output:**
   - Present the response in a structured format, distinguishing between the "Context" used and the final "Message" generated.
   - Ensure readability and coherence in the formatting to enhance user understanding.

- ## Context:
{context}

-- ## Message
{query}
"""
    return refined_prompt


def clean_llm_output(output: str) -> str:
    """
    Cleans the LLM output by removing code block markers and unnecessary escape characters.
    """
    # Remove code block markers (```json and ```)
    output = re.sub(r'```json\s*', '', output)
    output = re.sub(r'```', '', output)
    
    # Remove leading/trailing whitespace
    output = output.strip()
    
    # Replace escaped newlines with actual newlines
    output = output.replace('\\n', '\n')
    
    return output

def fix_common_json_errors(json_str: str) -> str:
    """
    Attempts to fix common JSON errors in the string.
    """
    # Add missing commas between objects
    json_str = re.sub(r'}\s*{', '}, {', json_str)
    
    # Ensure all keys and string values use double quotes
    json_str = json_str.replace("'", '"')
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    return json_str

def parse_tree_of_thoughts(output: str) -> Dict[str, Any]:
    """
    Parses the cleaned JSON string into a Python dictionary.
    """
    try:
        cleaned_output = clean_llm_output(output)
        corrected_output = fix_common_json_errors(cleaned_output)
        tree_of_thoughts = json.loads(corrected_output)
        return tree_of_thoughts
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.debug(f"Failed JSON string: {corrected_output}")
        raise ValueError(f"Unable to parse JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {e}")
        raise ValueError(f"Unexpected error: {e}")

def extract_all_sub_prompts(thoughts: List[Dict[str, Any]]) -> List[str]:
    """
    Recursively extracts all sub-prompts from the tree of thoughts.
    """
    sub_prompts = []
    for thought in thoughts:
        prompt = thought.get("prompt")
        if prompt:
            sub_prompts.append(prompt)
        nested = thought.get("sub_prompts", [])
        if nested:
            sub_prompts.extend(extract_all_sub_prompts(nested))
    return sub_prompts

def validate_response(response: Dict[str, Any], schema: typing.Type[TypedDict]) -> bool:
    """
    Validates that the response adheres to the provided schema.
    """
    for key, value_type in schema.__annotations__.items():
        if key not in response:
            logger.error(f"Missing key in response: {key}")
            return False
        if not isinstance(response[key], value_type):
            logger.error(f"Incorrect type for key '{key}': Expected {value_type}, got {type(response[key])}")
            return False
    return True

def generate_sub_prompt_response(idx: int, sub_prompt: str, llm_client, max_retries: int = 3) -> Dict[str, Any]:
    """
    Generates a response for a given sub-prompt using the LLM client with retry logic.
    
    Args:
        idx (int): The index of the sub-prompt.
        sub_prompt (str): The sub-prompt string.
        llm_client: The LLM client instance.
        max_retries (int): Maximum number of retry attempts.
    
    Returns:
        Dict[str, Any]: A dictionary containing the prompt and its response.
    """
    detailed_prompt = (
        f"{sub_prompt}\n\n"
        "Guidelines:\n"
        "1. Be concise and clear.\n"
        "2. Reference relevant notes where applicable.\n"
        "3. Ensure the information is accurate and well-organized.\n"
        "4. Maintain a neutral and informative tone."
    )
    logger.info(f"Processing Sub-Prompt {idx}: {sub_prompt[:50]}...")

    attempt = 0
    while attempt < max_retries:
        try:
            response = llm_client.generate_content(
                prompt=detailed_prompt
            )
            logger.info(f"Completed Sub-Prompt {idx}: {sub_prompt[:50]}...")
            return {"prompt": sub_prompt, "response": response}
        except Exception as e:
            attempt += 1
            wait_time = 2 ** attempt  # Exponential backoff
            logger.error(f"Error generating response for Sub-Prompt {idx}: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    logger.error(f"Failed to generate response for Sub-Prompt {idx} after {max_retries} attempts.")
    return {"prompt": sub_prompt, "response": f"Error generating response after {max_retries} attempts."}


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
        pagerank_scores = nx.pagerank(G, alpha=PAGERANK_ALPHA)

        # Update PageRank scores back to Neo4j
        logger.info("Updating PageRank scores in Neo4j...")
        pagerank_data = [{'id': node_id, 'score': score} for node_id, score in pagerank_scores.items()]
        session.run("""
        UNWIND $pagerank_data AS pr
        MATCH (n:Note {id: pr.id})
        SET n.pagerank = pr.score
        """, pagerank_data=pagerank_data)

    logger.info("PageRank computation and update completed.")

def compute_cluster_pagerank():
    """
    Computes PageRank for clusters based on the PageRank of their constituent notes.
    """
    with neo4j_conn.driver.session() as session:
        logger.info("Computing PageRank weights for clusters...")
        # Sum the PageRank scores of all notes within each cluster
        result = session.run("""
        MATCH (n:Note)-[:BELONGS_TO]->(c:Cluster)
        WITH c, sum(n.pagerank) AS cluster_pagerank
        SET c.pagerank_weight = cluster_pagerank
        RETURN c.label AS label, c.pagerank_weight AS pagerank_weight
        """)

        for record in result:
            logger.info(f"Cluster {record['label']} has pagerank weight {record['pagerank_weight']}")

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
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

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

    # Step 6: Compute Cluster PageRank Weights
    compute_cluster_pagerank()

    logger.info("Full recalculation completed.")

def perform_clustering():
    """
    Performs DBSCAN clustering on all note embeddings and updates Neo4j with Cluster nodes and relationships.
    Additionally, computes and stores the weighted average embedding for each cluster.
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
        for note_id, label, pagerank in zip(note_ids, cluster_labels, pageranks_array):
            if label == -1:
                continue  # Skip noise points
            clusters.setdefault(label, []).append((note_id, pagerank))
    
        # Prepare data for batch operations
        cluster_data = []
        for label, members in clusters.items():
            cluster_label = generate_cluster_label(label)
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
    
            # Generate cluster title and summary using Gemini
            title = generate_cluster_title(member_ids)       # Generate cluster title using Gemini
            summary = generate_cluster_summary(member_ids)   # Generate cluster summary using Gemini
    
            cluster_data.append({
                'label': cluster_label,
                'title': title,
                'summary': summary,
                'size': len(members),
                'note_ids': member_ids,
                'embedding': cluster_embedding
            })
    
        # Create Cluster nodes and BELONGS_TO relationships in batch
        for cluster in cluster_data:
            session.run("""
            MERGE (c:Cluster {label: $label})
            SET c.title = $title,
                c.summary = $summary,
                c.size = $size,
                c.embedding = $embedding
            """, label=cluster['label'], title=cluster['title'], summary=cluster['summary'], size=cluster['size'], embedding=cluster['embedding'])
    
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
    trigger=IntervalTrigger(minutes=30),
    id='recalculate_all_job',
    name='Recalculate all similarities and PageRank every 5 minutes',
    replace_existing=True
)

# Schedule the clustering to run every 5 minutes as well
scheduler.add_job(
    func=perform_clustering,
    trigger=IntervalTrigger(minutes=30),
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
        summary_prompt = NOTE_SUMMARY_PROMPT_TEMPLATE.format(content=processed_content)
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
            WITH n, dotProduct / (magQuery * magNote) AS similarity, n.pagerank AS pagerank
            WHERE similarity > $similarity_threshold
            ORDER BY similarity DESC, pagerank DESC
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
async def apply_parameter_updates(params: ParameterUpdate):
    """
    Applies the parameter updates.
    This function should include any necessary logic to re-cluster,
    recompute embeddings, or other operations based on the new parameters.
    """
    # Update the global CONFIG
    CONFIG.update(params.dict())
    logging.info(f"Configuration updated: {CONFIG}")
    
    # Perform a recalculaton
    asyncio.create_task(recalculate_all())

@app.post("/update_parameters")
async def update_parameters(params: ParameterUpdate, background_tasks: BackgroundTasks):
    """
    Endpoint to update algorithm parameters.
    The update is handled in the background to prevent blocking.
    """
    try:
        background_tasks.add_task(apply_parameter_updates, params)
        logging.info(f"Received parameter update request: {params.dict()}")
        return {"message": "Parameter update initiated successfully."}
    except Exception as e:
        logging.error(f"Failed to initiate parameter update: {e}")
        raise HTTPException(status_code=500, detail="Failed to update parameters.")

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
async def rag_query(
    rag_query_input: RAGQueryInput,
    background_tasks: BackgroundTasks,
    model=Depends(get_sentence_transformer)
):
    if not neo4j_conn.verify_connectivity():
        logger.error("Database connection error")
        raise HTTPException(status_code=500, detail="Database connection error")

    start_time = time.time()  # Track total processing time

    try:
        user_query = rag_query_input.query
        logger.info(f"Received Query: {user_query}")

        # *** Step 1: Prompt Refining, Engineering, and Optimization ***
        refined_user_query = refine_query(user_query)
        logger.info(f"Refined Query: {refined_user_query}")
        processed_query = preprocess_for_embedding(refined_user_query)
        logger.info(f"Processed Query: {processed_query}")
        query_embedding = await asyncio.to_thread(model.encode, processed_query)
        query_embedding = query_embedding.tolist()
        logger.debug(f"Query Embedding (first 5 elements): {query_embedding[:5]}...")

        # *** Step 2: Query Notes First ***
        with neo4j_conn.driver.session() as session:
            note_results = session.run("""
            MATCH (n:Note)
            WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
            WITH n,
                 reduce(a = 0.0, x IN range(0, size(n.embedding)-1) |
                       a + (n.embedding[x] * $query_embedding[x])) AS dotProduct,
                 sqrt(reduce(a = 0.0, x IN $query_embedding | a + x * x)) AS magQuery,
                 sqrt(reduce(a = 0.0, x IN n.embedding | a + x * x)) AS magNote
            WITH n, (dotProduct / (magQuery * magNote)) AS similarity
            WHERE similarity > $similarity_threshold
            RETURN n.id AS id, n.content AS content, n.pagerank AS pagerank, similarity
            ORDER BY similarity DESC, pagerank DESC
            LIMIT $max_notes
            """, query_embedding=query_embedding, similarity_threshold=SIMILARITY_THRESHOLD_NOTE, max_notes=RAG_MAX_NOTES_PER_CLUSTER)

            notes = [record for record in note_results]
            logger.info(f"Retrieved {len(notes)} notes.")

        # *** Step 3: Aggregate Notes and Track Clusters ***
        aggregated_context = ""
        referenced_note_ids = []
        cluster_note_count = {}  # To track number of notes per cluster

        with neo4j_conn.driver.session() as session:
            for note in notes:
                note_id = note["id"]
                content = note["content"]
                pagerank = note["pagerank"]
                similarity = note["similarity"]

                # Fetch the cluster label for this note
                cluster_label_result = session.run("""
                MATCH (n:Note)-[:BELONGS_TO]->(c:Cluster)
                WHERE n.id = $note_id
                RETURN c.label AS label
                """, note_id=note_id)

                cluster_label_record = cluster_label_result.single()
                if cluster_label_record:
                    cluster_label = cluster_label_record["label"]
                    cluster_note_count[cluster_label] = cluster_note_count.get(cluster_label, 0) + 1
                else:
                    cluster_label = "Unknown"

                formatted_content = f"Note ID: {note_id}\nContent: {content}\n\n"
                if len(aggregated_context) + len(formatted_content) > RAG_MAX_CONTEXT_LENGTH:
                    logger.info("Reached maximum context length while adding notes. Stopping aggregation.")
                    break
                aggregated_context += formatted_content
                referenced_note_ids.append(note_id)
                logger.info(f"Added Note ID: {note_id} from Cluster: {cluster_label}")

        # *** Step 4: Query Clusters Separately ***
        with neo4j_conn.driver.session() as session:
            cluster_results = session.run("""
            MATCH (c:Cluster)
            WHERE c.embedding IS NOT NULL AND size(c.embedding) > 0
            WITH c,
                 reduce(a = 0.0, x IN range(0, size(c.embedding)-1) |
                       a + (c.embedding[x] * $query_embedding[x])) AS dotProduct,
                 sqrt(reduce(a = 0.0, x IN $query_embedding | a + x * x)) AS magQuery,
                 sqrt(reduce(a = 0.0, x IN c.embedding | a + x * x)) AS magCluster
            WITH c, (dotProduct / (magQuery * magCluster)) AS similarity
            WHERE similarity > $similarity_threshold
            RETURN c.label AS label, c.title AS title, c.summary AS summary, c.pagerank_weight AS pagerank_weight, similarity
            ORDER BY similarity DESC, pagerank_weight DESC
            LIMIT $max_clusters
            """, query_embedding=query_embedding, similarity_threshold=SIMILARITY_THRESHOLD_CLUSTER, max_clusters=RAG_MAX_CLUSTERS)

            clusters = [record for record in cluster_results]
            logger.info(f"Retrieved {len(clusters)} clusters.")

        # *** Step 5: Adjust Cluster Similarity Based on Number of Notes Selected ***
        clusters = [dict(record) for record in clusters]  # Convert records to dictionaries

        for cluster in clusters:
            cluster_label = cluster["label"]
            num_notes = cluster_note_count.get(cluster_label, 0)
            original_similarity = cluster["similarity"]
            # Adjust similarity based on number of notes: more notes -> higher similarity
            similarity_factor = 1 + (num_notes / RAG_MAX_NOTES_PER_CLUSTER)  # Example adjustment
            adjusted_similarity = original_similarity * similarity_factor

            # Apply cluster favor weight
            adjusted_similarity *= CLUSTER_FAVOR_WEIGHT

            # Update the similarity score in the cluster dictionary
            cluster["adjusted_similarity"] = adjusted_similarity
            logger.debug(f"Cluster: {cluster_label}, Original Similarity: {original_similarity}, "
                        f"Number of Notes: {num_notes}, Adjusted Similarity: {adjusted_similarity}")

        # *** Step 6: Sort Clusters Based on Adjusted Similarity ***
        clusters_sorted = sorted(clusters, key=lambda x: (x["adjusted_similarity"], x["pagerank_weight"]), reverse=True)
        logger.info("Clusters sorted based on adjusted similarity.")

        # *** Step 7: Aggregate Clusters into Context ***
        referenced_cluster_labels = []
        for cluster in clusters_sorted:
            cluster_label = cluster["label"]
            cluster_summary = cluster["summary"]
            formatted_cluster = f"Cluster: {cluster_label}\nSummary: {cluster_summary}\n\n"
            if len(aggregated_context) + len(formatted_cluster) > RAG_MAX_CONTEXT_LENGTH:
                logger.info("Reached maximum context length while adding cluster summaries. Stopping aggregation.")
                break
            aggregated_context += formatted_cluster
            referenced_cluster_labels.append(cluster_label)
            logger.info(f"Added Cluster Summary: {cluster_label}")

        logger.info(f"Total Aggregated Context Length: {len(aggregated_context)} characters.")

        # *** Step 8: Generate the Answer Using LLM ***
        refined_prompt = refine_user_query_for_rag(context=aggregated_context, query=user_query)
        logger.debug(f"Refined Prompt Length: {len(refined_prompt)}")
        llm_client = get_llm_client()

        # **New Steps: Tree of Thoughts and Multi-Step Reasoning**

        # **Sub-Step 1: Generate Tree of Thoughts **
        try:
            # Instruct the LLM to generate a tree of thoughts in JSON format
            tree_of_thoughts_prompt = (
                f"{refined_prompt}\n\n"
                "Please analyze the above context and generate a JSON object representing a tree of thoughts. "
                "The JSON should have a key 'thoughts' which is a list of sub-prompts or questions to explore. "
                "Ensure that the JSON is properly formatted with double quotes."
            )
            logger.info("Generating Tree of Thoughts.")
            tree_response = await asyncio.to_thread(
                lambda: llm_client.generate_content(
                    prompt=tree_of_thoughts_prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json", response_schema=list[SubPromptResponse]
                    )
                )
            )
            logger.info(f"Tree response: {tree_response}")
            logger.debug(f"Tree of Thoughts JSON: {tree_response[:500]}...")  # Log first 500 chars

            parsed_tree_response = json.loads(tree_response)
            logger.info("Tree of thoughts: ", parsed_tree_response)
            sub_prompts = [x['prompt'] for x in parsed_tree_response]
            logger.info("Sub prompts: ", sub_prompts)
        except ValueError as e:
            logger.error(f"Failed to parse Tree of Thoughts JSON: {e}")
            raise HTTPException(status_code=500, detail="Error parsing Tree of Thoughts JSON from LLM")
        except Exception as e:
            logger.error(f"Unexpected error during Tree of Thoughts generation: {e}")
            raise HTTPException(status_code=500, detail="Unexpected error during Tree of Thoughts generation")

        # **Sub-Step 2: Generate Responses for Each Sub-Prompt **
        responses = []

        for idx, prompt in enumerate(sub_prompts, start=0):
            try:
                subresponse = generate_sub_prompt_response(
                    sub_prompt=prompt,
                    idx=idx,
                    llm_client=llm_client
                )
                responses.append(subresponse)
            except Exception as e:
                logger.error(f"Error processing Sub-Prompt {idx}: {e}")
                responses.append({"prompt": prompt, "response": f"Error generating response: {e}"})

            # Check if total processing time is approaching the timeout
            # elapsed_time = time.time() - start_time
            # if elapsed_time >= AGENTIC_PROMPT_TIMEOUT:
            #     logger.error("Agentic prompting and exploration timed out.")
            #     break

        logger.info(f"Generated responses for {len(responses)} sub-prompts.")

        # **Sub-Step 3: Aggregate and Reformat Answers Through Another RAG Run**
        aggregated_sub_responses = "\n".join([
            f"Sub-Prompt {i}: {resp['response']}" for i, resp in enumerate(responses, 1)
            if resp['response'] and not resp['response'].startswith("Error")
        ])
        logger.debug(f"Aggregated Sub-Responses Length: {len(aggregated_sub_responses)}")

        # **Final Prompt for LLM**
        final_prompt = (
            f"{aggregated_context}\n\n"
            f"{aggregated_sub_responses}\n\n"
            "Please synthesize the above information into a clear and concise answer, "
            "linking relevant notes where appropriate to enhance the response."
        )
        logger.info("Generating final synthesized answer.")
        logger.debug(f"Final Prompt Length: {len(final_prompt)}")
        logger.debug(f"Final Prompt: {final_prompt[:500]}...")  # Log first 500 chars

        try:
            final_response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: llm_client.generate_content(
                        prompt=final_prompt
                    )
                ),
                timeout=3000000
            )
            final_answer = final_response
            logger.info("Final answer generated successfully.")
            logger.debug(f"Final LLM Answer: {final_response[:500]}...")  # Log first 500 chars
        except asyncio.TimeoutError:
            logger.error("Final answer generation timed out.")
            final_answer = "Error: Final answer generation timed out."
    except Exception as e:
        logger.error(f"Error generating final answer from LLM: {e}")
        final_answer = "Error: Unable to generate final answer from LLM."

    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds.")

    return {
        "answer": final_answer,
        "referenced_cluster_labels": referenced_cluster_labels,
        "referenced_note_ids": referenced_note_ids,
        "error": "" if not final_answer.startswith("Error") else final_answer
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

# --- END OF FILE app.py ---
