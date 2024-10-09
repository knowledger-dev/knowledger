# models.py

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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
    summary: str = ""  # AI-generated insights

class QueryInput(BaseModel):
    query: str
    limit: int = 5

class RAGQueryInput(BaseModel):
    query: str
    max_tokens: int = Field(default=2048)

class SubPromptResponse(TypedDict):
    prompt: str
    response: str
    note_id: str  # Link back to the note

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
