# models.py

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr
from typing_extensions import TypedDict

class NoteInput(BaseModel):
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# models.py

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Model for user registration (input)
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str  # Plain text password input

# Model for user data stored in the database (internal use)
class UserInDB(BaseModel):
    id: str = Field(..., alias='_id')  # MongoDB's _id field as string
    username: str
    email: EmailStr
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        from_attributes = True

# Model for user data returned in responses (output)
class UserRead(BaseModel):
    id: str = Field(..., alias='_id')
    username: str
    email: EmailStr
    is_active: bool
    created_at: datetime

    class Config:
        populate_by_name = True
        from_attributes = True

# Model for note input
class NoteInput(BaseModel):
    content: str
    timestamp: datetime

# Model for note output
class NoteOutput(BaseModel):
    id: str = Field(..., alias='_id')  # Map MongoDB's '_id' to 'id'
    content: str
    processed_content: str
    timestamp: datetime
    commonness: int = 0
    pagerank: float = 0.0
    summary: str = ""  # AI-generated insights
    similar_notes: List[str] = Field(default_factory=list)
    cluster_id: Optional[str] = None
    owner_username: str  # Owner's username

    class Config:
        populate_by_name = True
        from_attributes = True

# Model for query input
class QueryInput(BaseModel):
    query: str
    limit: int

# Model for RAG query input
class RAGQueryInput(BaseModel):
    query: str

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

# Optionally, define a model for Cluster if needed
class ClusterOutput(BaseModel):
    id: str = Field(..., alias='_id')
    label: str
    title: str
    summary: str
    size: int
    pagerank_weight: float = 0.0
    note_ids: List[str]