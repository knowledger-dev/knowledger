# db.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pymongo
from pymongo import MongoClient, UpdateOne
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from utils import convert_objectid_to_str  # Import the helper function

# Initialize Logging
logger = logging.getLogger(__name__)

class MongoDBConnection:
    def __init__(self):
        try:
            self.client = MongoClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
            self.db = self.client[Config.MONGODB_DB_NAME]
            self.notes = self.db.notes
            self.users = self.db.users  # Users collection
            self.clusters = self.db.clusters
            # Ensure indexes for performance
            self._ensure_indexes()
            logger.info("Connected to MongoDB successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def _ensure_indexes(self):
        """
        Ensure that necessary indexes are created for optimal query performance.
        """
        try:
            # Unique index on username and email to prevent duplicates
            self.users.create_index("username", unique=True)
            self.users.create_index("email", unique=True)
            # Index on 'similar_notes' for PageRank computations
            self.notes.create_index("similar_notes")
            # Index on 'cluster_id' for clustering
            self.notes.create_index("cluster_id")
            # Index on 'owner_username' to quickly retrieve user-specific notes
            self.notes.create_index("owner_username")
            logger.info("MongoDB indexes ensured.")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise

    def verify_connectivity(self) -> bool:
        """
        Verify if the MongoDB connection is alive.
        """
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB connectivity verification failed: {e}")
            return False

    # User Management Methods

    def create_user(self, user_data: Dict[str, Any]):
        """
        Insert a new user into the 'users' collection.
        """
        try:
            self.users.insert_one(user_data)
            logger.debug(f"User {user_data['username']} created.")
        except pymongo.errors.DuplicateKeyError as e:
            logger.error(f"Duplicate user detected: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a user by their username.
        """
        try:
            user = self.users.find_one({"username": username})
            if user:
                user = convert_objectid_to_str(user)  # Convert ObjectId to string
            return user
        except Exception as e:
            logger.error(f"Error retrieving user {username}: {e}")
            raise

    # Note Management Methods (as previously defined)
    
    def create_note(self, note_id: str, content: str, processed_content: str,
                   embedding: List[float], timestamp: datetime, summary: str,
                   owner_username: str):
        """
        Insert a new note into the 'notes' collection.
        """
        try:
            note_document = {
                "_id": note_id,  # Custom string ID
                "content": content,
                "processed_content": processed_content,
                "embedding": embedding,
                "timestamp": timestamp,  # Stored as datetime
                "summary": summary,
                "commonness": 0,
                "pagerank": 0.0,
                "similar_notes": [],
                "cluster_id": None,
                "owner_username": owner_username
            }
            self.notes.insert_one(note_document)
            logger.debug(f"Note {note_id} inserted into MongoDB.")
        except pymongo.errors.DuplicateKeyError:
            logger.error(f"Duplicate note ID detected: {note_id}")
            raise
        except Exception as e:
            logger.error(f"Error inserting note {note_id}: {e}")
            raise
    
    def get_note(self, note_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single note by its ID.
        """
        try:
            note = self.notes.find_one({"_id": note_id})
            return note
        except Exception as e:
            logger.error(f"Error retrieving note {note_id}: {e}")
            raise
    
    def get_notes(self, note_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple notes by their IDs.
        """
        try:
            notes_cursor = self.notes.find({"_id": {"$in": note_ids}})
            notes = list(notes_cursor)
            return notes
        except Exception as e:
            logger.error(f"Error retrieving notes {note_ids}: {e}")
            raise
    
    def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single cluster by its ID.
        """
        try:
            cluster = self.clusters.find_one({"_id": cluster_id})
            return cluster
        except Exception as e:
            logger.error(f"Error retrieving cluster {cluster_id}: {e}")
            raise
    
    def get_clusters(self, cluster_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple clusters by their IDs.
        """
        try:
            clusters_cursor = self.clusters.find({"_id": {"$in": cluster_ids}})
            clusters = list(clusters_cursor)
            return clusters
        except Exception as e:
            logger.error(f"Error retrieving clusters {cluster_ids}: {e}")
            raise
    
    def get_cluster_content(self, note_ids: List[str]) -> str:
        """
        Concatenate the content of all notes in a cluster.
        """
        try:
            notes_cursor = self.notes.find({"_id": {"$in": note_ids}}, {"content": 1})
            contents = [note["content"] for note in notes_cursor]
            combined_content = "\n".join(contents)
            return combined_content
        except Exception as e:
            logger.error(f"Error getting cluster content for notes {note_ids}: {e}")
            raise
    
    def update_relationships(self, note_id: str, similarity_threshold: float):
        """
        Update the 'similar_notes' field for a given note based on embedding similarity.
        Also updates the 'similar_notes' field of similar notes to include the current note.
        """
        try:
            # Fetch the embedding of the current note
            current_note = self.get_note(note_id)
            if not current_note:
                logger.error(f"Note {note_id} not found for updating relationships.")
                return

            current_embedding = np.array(current_note["embedding"]).reshape(1, -1)

            # Fetch embeddings of all other notes
            other_notes_cursor = self.notes.find(
                {"_id": {"$ne": note_id}, "embedding": {"$exists": True, "$ne": []}},
                {"_id": 1, "embedding": 1}
            )

            similar_note_ids = []
            for note in other_notes_cursor:
                other_note_id = note["_id"]
                other_embedding = np.array(note["embedding"]).reshape(1, -1)
                similarity = cosine_similarity(current_embedding, other_embedding)[0][0]
                if similarity >= similarity_threshold:
                    similar_note_ids.append(other_note_id)
                    # Update the 'similar_notes' field of the similar note to include the current note
                    self.notes.update_one(
                        {"_id": other_note_id},
                        {"$addToSet": {"similar_notes": note_id}}
                    )

            # Update the 'similar_notes' field of the current note
            self.notes.update_one(
                {"_id": note_id},
                {"$set": {"similar_notes": similar_note_ids}}
            )
            logger.debug(f"Updated similar_notes for note {note_id} with {len(similar_note_ids)} similar notes.")

        except Exception as e:
            logger.error(f"Error updating relationships for note {note_id}: {e}")
            raise

    
    def get_similar_notes(
        self,
        query_embedding: List[float],
        similarity_threshold: float,
        limit: int,
        use_pagerank_weighting: bool = False,
        owner_username: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Retrieve notes similar to the query_embedding based on cosine similarity.
        Optionally weight the similarity by PageRank scores.
        Filters notes by owner_username.
        """
        try:
            # Convert query_embedding to numpy array
            query_emb = np.array(query_embedding).reshape(1, -1)

            # Fetch notes belonging to the user with embeddings
            notes_cursor = self.notes.find(
                {
                    "embedding": {"$exists": True, "$ne": []},
                    "owner_username": owner_username
                }
            )
            similar_notes = []
            for note in notes_cursor:
                note_id = note["_id"]
                embedding = np.array(note["embedding"]).reshape(1, -1)
                similarity = cosine_similarity(query_emb, embedding)[0][0]
                if similarity >= similarity_threshold:
                    pagerank = note.get("pagerank", 0.0)
                    if use_pagerank_weighting:
                        weighted_similarity = similarity * pagerank
                    else:
                        weighted_similarity = similarity
                    # Include all note fields in the result
                    note_data = note.copy()
                    note_data["id"] = note_data.pop("_id")
                    note_data["similarity"] = similarity
                    note_data["weighted_similarity"] = weighted_similarity
                    similar_notes.append(note_data)

            # Sort notes based on similarity or weighted_similarity
            if use_pagerank_weighting:
                similar_notes.sort(key=lambda x: x["weighted_similarity"], reverse=True)
            else:
                similar_notes.sort(key=lambda x: x["similarity"], reverse=True)

            # Limit the number of results
            limited_notes = similar_notes[:limit]
            return limited_notes

        except Exception as e:
            logger.error(f"Error retrieving similar notes: {e}")
            raise

    def get_note_owned_by_user(self, note_id: str, owner_username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single note by its ID and owner.
        """
        try:
            note = self.notes.find_one({"_id": note_id, "owner_username": owner_username})
            return note
        except Exception as e:
            logger.error(f"Error retrieving note {note_id}: {e}")
            raise

    
    def get_all_embeddings(self) -> List[np.ndarray]:
        """
        Retrieve all embeddings from the notes collection.
        Useful for bulk operations like clustering.
        """
        try:
            embeddings_cursor = self.notes.find(
                {"embedding": {"$exists": True, "$ne": []}},
                {"embedding": 1}
            )
            embeddings = [np.array(note["embedding"]) for note in embeddings_cursor]
            return embeddings
        except Exception as e:
            logger.error(f"Error retrieving all embeddings: {e}")
            raise
    
    def get_notes_by_ids(self, note_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve notes by a list of note_ids.
        """
        return self.get_notes(note_ids)
    
    def close(self):
        self.client.close()
