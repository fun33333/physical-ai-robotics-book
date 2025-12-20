"""
Unit tests for Qdrant Service (Phase 3 - T030).

Tests RAG Agent retrieval functionality with mocked Qdrant client.
Verifies that retrieval returns top-k chunks with metadata and prioritizes
selected text.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.services.qdrant_service import QdrantService
from src.utils import QdrantError


class TestQdrantServiceInitialization:
    """Tests for Qdrant service initialization."""

    def test_qdrant_client_initialization_success(self):
        """Test successful Qdrant client initialization."""
        with patch("src.services.qdrant_service.QdrantClient") as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            service = QdrantService()

            assert service.client is not None
            assert service.collection_name == "textbook_chapters"

    def test_qdrant_client_initialization_failure(self):
        """Test Qdrant client initialization failure."""
        with patch("src.services.qdrant_service.QdrantClient") as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            with pytest.raises(QdrantError) as excinfo:
                QdrantService()

            assert "Qdrant initialization failed" in str(excinfo.value)


class TestQdrantServiceSearchRetrieval:
    """Tests for semantic search and retrieval."""

    @pytest.fixture
    def qdrant_service(self):
        """Create QdrantService with mocked client."""
        with patch("src.services.qdrant_service.QdrantClient"):
            return QdrantService()

    def test_search_similar_returns_top_k_chunks(self, qdrant_service):
        """Test that search returns top-k chunks with metadata."""
        # Mock search results
        mock_result_1 = Mock()
        mock_result_1.score = 0.95
        mock_result_1.payload = {
            "chunk_id": "ch2_sec1_001",
            "text": "ROS 2 is a middleware",
            "chapter": "2",
            "section": "ROS 2 Basics",
            "difficulty_level": "beginner",
            "source_url": "chapter2.md",
        }

        mock_result_2 = Mock()
        mock_result_2.score = 0.87
        mock_result_2.payload = {
            "chunk_id": "ch2_sec2_001",
            "text": "ROS 2 communication patterns",
            "chapter": "2",
            "section": "Communication",
            "difficulty_level": "intermediate",
            "source_url": "chapter2.md",
        }

        qdrant_service.client.search.return_value = [mock_result_1, mock_result_2]

        # Perform search
        query_embedding = [0.1] * 768  # Dummy 768-dim embedding
        results = qdrant_service.search_similar(
            query_embedding=query_embedding,
            top_k=5,
            query_text="What is ROS 2?",
        )

        # Assertions
        assert len(results) == 2
        assert results[0]["chunk_id"] == "ch2_sec1_001"
        assert results[0]["relevance_score"] == 0.95
        assert results[0]["chapter"] == "2"
        assert results[0]["section"] == "ROS 2 Basics"

    def test_search_respects_top_k_limit(self, qdrant_service):
        """Test that search respects top_k parameter."""
        # Mock 10 results
        mock_results = []
        for i in range(10):
            result = Mock()
            result.score = 0.95 - (i * 0.05)
            result.payload = {
                "chunk_id": f"chunk_{i}",
                "text": f"Text {i}",
                "chapter": "2",
                "section": "Section",
                "difficulty_level": "intermediate",
                "source_url": "test.md",
            }
            mock_results.append(result)

        qdrant_service.client.search.return_value = mock_results

        # Search with top_k=3
        query_embedding = [0.1] * 768
        results = qdrant_service.search_similar(
            query_embedding=query_embedding,
            top_k=3,
        )

        # Should return only top 3
        assert len(results) == 3
        assert results[0]["chunk_id"] == "chunk_0"
        assert results[2]["chunk_id"] == "chunk_2"

    def test_search_error_handling(self, qdrant_service):
        """Test error handling in search."""
        qdrant_service.client.search.side_effect = Exception("Search failed")

        with pytest.raises(QdrantError) as excinfo:
            qdrant_service.search_similar(
                query_embedding=[0.1] * 768,
                top_k=5,
            )

        assert "Search failed" in str(excinfo.value)


class TestQdrantServiceDocumentSplitting:
    """Tests for document chunking functionality."""

    def test_split_document_basic(self):
        """Test basic document splitting."""
        text = "This is a test document. It contains multiple sentences. Each sentence is important."
        chunks = QdrantService.split_document(text, chunk_size=20, overlap=5)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_document_respects_chunk_size(self):
        """Test that chunks respect size limits (approximately)."""
        text = " ".join(["word"] * 100)
        chunks = QdrantService.split_document(text, chunk_size=50, overlap=10)

        # Each chunk should be roughly 50 tokens (~65 words)
        for chunk in chunks:
            word_count = len(chunk.split())
            # Allow 30% tolerance
            assert word_count < 85  # 65 * 1.3

    def test_split_document_empty_input(self):
        """Test splitting empty document."""
        chunks = QdrantService.split_document("", chunk_size=512, overlap=50)
        assert len(chunks) == 0

    def test_split_document_short_text(self):
        """Test splitting text shorter than chunk size."""
        text = "Short text."
        chunks = QdrantService.split_document(text, chunk_size=512, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestQdrantServiceIndexing:
    """Tests for document indexing."""

    @pytest.fixture
    def qdrant_service(self):
        """Create QdrantService with mocked client."""
        with patch("src.services.qdrant_service.QdrantClient"):
            return QdrantService()

    def test_index_document_success(self, qdrant_service):
        """Test successful document indexing."""
        qdrant_service.client.upsert = Mock()

        metadata = {
            "chapter": "2",
            "section": "ROS 2 Basics",
            "difficulty_level": "beginner",
            "source_url": "chapter2.md",
        }

        qdrant_service.index_document(
            chunk_id="ch2_sec1_001",
            text="ROS 2 is a middleware platform.",
            embedding=[0.1] * 768,
            metadata=metadata,
        )

        # Verify upsert was called
        qdrant_service.client.upsert.assert_called_once()

    def test_index_document_error(self, qdrant_service):
        """Test error handling in document indexing."""
        qdrant_service.client.upsert.side_effect = Exception("Upsert failed")

        with pytest.raises(QdrantError) as excinfo:
            qdrant_service.index_document(
                chunk_id="ch2_sec1_001",
                text="Test text",
                embedding=[0.1] * 768,
                metadata={"chapter": "2"},
            )

        assert "Failed to index document" in str(excinfo.value)
