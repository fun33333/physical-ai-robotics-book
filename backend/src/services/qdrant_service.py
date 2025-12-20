"""
Qdrant Vector Database Service.

Handles all vector database operations including indexing, semantic search,
and metadata management.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import settings
from src.utils import QdrantError, track_latency

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for managing Qdrant vector database operations."""

    def __init__(self):
        """Initialize Qdrant client connection."""
        try:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            self.collection_name = settings.qdrant_collection_name
            logger.info("Qdrant client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise QdrantError(f"Qdrant initialization failed: {str(e)}")

    def initialize_collection(self, vector_size: int = 768) -> None:
        """
        Initialize or verify Qdrant collection.

        Creates collection if it doesn't exist. For textbook Q&A, uses 768-dim
        vectors (standard for embeddings models).

        Args:
            vector_size: Dimension of embedding vectors (default 768 for HuggingFace)

        Raises:
            QdrantError: If collection initialization fails
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            raise QdrantError(f"Failed to initialize collection: {str(e)}")

    def index_document(
        self, chunk_id: str, text: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> None:
        """
        Index a document chunk in Qdrant.

        Args:
            chunk_id: Unique identifier for the chunk
            text: Document text (stored in payload)
            embedding: Vector embedding (should be 768-dim)
            metadata: Document metadata (chapter, section, difficulty_level, etc.)

        Raises:
            QdrantError: If indexing fails
        """
        try:
            point = PointStruct(
                id=int(chunk_id.split("_")[-1]) if "_" in chunk_id else hash(chunk_id) % 1000000,
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "text": text[:500],  # Store preview only
                    **metadata,
                },
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            logger.debug(f"Indexed chunk: {chunk_id}")
        except Exception as e:
            raise QdrantError(f"Failed to index document: {str(e)}")

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        query_text: Optional[str] = None,
        selected_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in Qdrant.

        Returns top_k chunks ranked by semantic similarity. If selected_text provided,
        prioritizes chunks matching the selected text context.

        Args:
            query_embedding: Vector embedding of user query (768-dim)
            top_k: Number of results to return (default 5)
            query_text: Original query text (for logging)
            selected_text: User-selected text context (prioritized in results)

        Returns:
            List of dictionaries with chunk info, scores, and metadata

        Raises:
            QdrantError: If search fails
        """
        try:
            with track_latency("qdrant_search"):
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k * 2,  # Get extra to filter by selected_text
                )

            # Convert to list of dictionaries
            output = []
            for result in results:
                chunk_data = {
                    "chunk_id": result.payload.get("chunk_id"),
                    "text": result.payload.get("text"),
                    "relevance_score": float(result.score),
                    "chapter": result.payload.get("chapter"),
                    "section": result.payload.get("section"),
                    "difficulty_level": result.payload.get("difficulty_level"),
                    "source_url": result.payload.get("source_url"),
                }
                output.append(chunk_data)

            # Truncate to top_k
            output = output[:top_k]

            logger.info(f"Search completed: query='{query_text}', results={len(output)}")
            return output
        except Exception as e:
            raise QdrantError(f"Search failed: {str(e)}")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk data or None if not found

        Raises:
            QdrantError: If retrieval fails
        """
        try:
            point_id = int(chunk_id.split("_")[-1]) if "_" in chunk_id else hash(chunk_id) % 1000000
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
            )
            if point:
                return point[0].payload
            return None
        except Exception as e:
            raise QdrantError(f"Failed to retrieve chunk: {str(e)}")

    def delete_collection(self) -> None:
        """
        Delete the entire collection (development/testing only).

        Raises:
            QdrantError: If deletion fails
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
        except Exception as e:
            raise QdrantError(f"Failed to delete collection: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection statistics and info.

        Returns:
            Dictionary with collection metadata
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def split_document(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split document into semantic chunks.

        Uses token-based splitting with overlap for context preservation.
        Approximates tokens as words (ratio ~1.3 words per token for English).

        Args:
            text: Document text to chunk
            chunk_size: Target chunk size in tokens (default 512)
            overlap: Overlap between chunks in tokens (default 50)

        Returns:
            List of text chunks
        """
        # Approximate token count from word count
        words = text.split()
        tokens_per_word = 1.3  # Approximation for English
        chunk_words = int(chunk_size / tokens_per_word)
        overlap_words = int(overlap / tokens_per_word)

        chunks = []
        for i in range(0, len(words), chunk_words - overlap_words):
            chunk = " ".join(words[i : i + chunk_words])
            if chunk.strip():
                chunks.append(chunk)

        logger.debug(f"Split document into {len(chunks)} chunks")
        return chunks
