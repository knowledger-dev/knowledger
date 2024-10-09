# db.py

from typing import List
from neo4j import GraphDatabase
import logging
from config import Config

logger = logging.getLogger(__name__)

class Neo4jConnection:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
                max_connection_pool_size=100  # Optimized pool size
            )
            logger.info("Neo4j connection established.")
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}")
            raise

    def verify_connectivity(self):
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed.")

    def create_constraints(self):
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Note) REQUIRE n.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cluster) REQUIRE c.label IS UNIQUE")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Note) ON (n.pagerank)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Note) ON (n.embedding)")
                session.run("CREATE INDEX IF NOT EXISTS FOR (c:Cluster) ON (c.pagerank_weight)")
                logger.info("Neo4j constraints and indexes ensured.")
            except Exception as e:
                logger.error(f"Error creating constraints/indexes: {e}")

    # Function to create a note
    def create_note(self, note_id: str, content: str, processed_content: str,
                   embedding: list, timestamp: str, summary: str):
        with self.driver.session() as session:
            try:
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
                timestamp=timestamp,
                summary=summary)
                logger.debug(f"Note {note_id} created in Neo4j.")
            except Exception as e:
                logger.error(f"Error creating note {note_id}: {e}")
                raise

    # Function to retrieve a note
    def get_note(self, note_id: str):
        with self.driver.session() as session:
            try:
                result = session.run("""
                MATCH (n:Note {id: $id})
                SET n.commonness = n.commonness + 1
                RETURN n.id as id, n.content as content, n.processed_content as processed_content,
                       n.timestamp as timestamp, n.commonness as commonness, n.pagerank as pagerank, n.summary as summary
                """, id=note_id).single()
                return result
            except Exception as e:
                logger.error(f"Error retrieving note {note_id}: {e}")
                raise

    # Modified update_relationships function without GDS
    def update_relationships(self, note_id: str, similarity_threshold: float):
        with self.driver.session() as session:
            try:
                session.run("""
                MATCH (n1:Note {id: $id}), (n2:Note)
                WHERE n1 <> n2 AND n1.embedding IS NOT NULL AND n2.embedding IS NOT NULL
                WITH n1, n2,
                     reduce(dot = 0.0, i IN range(0, size(n1.embedding)-1) | dot + n1.embedding[i] * n2.embedding[i]) AS dotProduct,
                     sqrt(reduce(acc = 0.0, x IN n1.embedding | acc + x * x)) AS magA,
                     sqrt(reduce(acc = 0.0, x IN n2.embedding | acc + x * x)) AS magB
                WITH n1, n2, dotProduct / (magA * magB) AS cosine_similarity
                WHERE cosine_similarity > $threshold
                MERGE (n1)-[r:SIMILAR]->(n2)
                SET r.score = cosine_similarity
                """, id=note_id, threshold=similarity_threshold)
                logger.debug(f"SIMILAR relationships updated for note {note_id}.")
            except Exception as e:
                logger.error(f"Error updating relationships for note {note_id}: {e}")
                raise

    # Modified get_similar_notes function without GDS
    def get_similar_notes(self, query_embedding: List[float], similarity_threshold: float, limit: int):
        with self.driver.session() as session:
            try:
                result = session.run("""
                MATCH (n:Note)
                WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0
                WITH n,
                     reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * $query_embedding[i]) AS dotProduct,
                     sqrt(reduce(acc = 0.0, x IN n.embedding | acc + x * x)) AS magN,
                     sqrt(reduce(acc = 0.0, x IN $query_embedding | acc + x * x)) AS magQ
                WITH n, dotProduct / (magN * magQ) AS cosine_similarity
                WHERE cosine_similarity > $threshold
                RETURN n.id AS id, n.content AS content, n.pagerank AS pagerank, cosine_similarity
                ORDER BY cosine_similarity DESC, n.pagerank DESC
                LIMIT $limit
                """, query_embedding=query_embedding, threshold=similarity_threshold, limit=limit)
                
                notes = [record for record in result]
                return notes
            except Exception as e:
                logger.error(f"Error retrieving similar notes: {e}")
                raise

    # Function to retrieve cluster content
    def get_cluster_content(self, note_ids: List[str]) -> str:
        with self.driver.session() as session:
            try:
                contents = session.run("""
                MATCH (n:Note)
                WHERE n.id IN $ids
                RETURN n.content AS content
                """, ids=note_ids)
                combined_content = "\n".join([record["content"] for record in contents])
                return combined_content
            except Exception as e:
                logger.error(f"Error retrieving cluster content: {e}")
                raise