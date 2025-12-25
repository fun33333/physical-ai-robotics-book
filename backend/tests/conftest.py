"""
Pytest configuration and fixtures for RAG Chatbot tests.

Provides common test fixtures, database setup/teardown, and mocks
for testing the multi-agent pipeline.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add backend/src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_settings():
    """Mock settings with test configuration."""
    with patch("src.config.settings") as mock:
        mock.database_url = "sqlite:///:memory:"
        mock.gemini_api_keys = ["test_key_1", "test_key_2", "test_key_3"]
        mock.qdrant_url = "http://localhost:6333"
        mock.qdrant_api_key = ""
        mock.qdrant_collection_name = "test_collection"
        mock.environment = "test"
        mock.cors_origin = "http://localhost:3000"
        mock.rate_limit_rpm = 15
        yield mock


@pytest.fixture
def mock_db_session():
    """Mock SQLAlchemy database session."""
    session = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.close = MagicMock()
    session.add = MagicMock()
    session.query = MagicMock()
    return session


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "chunk_id": "chunk_001",
            "text": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides tools, libraries, and conventions to simplify building complex robots.",
            "chapter": "Module 1",
            "section": "ROS 2 Fundamentals",
            "difficulty_level": "beginner",
        },
        {
            "chunk_id": "chunk_002",
            "text": "ROS 2 uses DDS (Data Distribution Service) for reliable communication between nodes. DDS provides real-time, scalable, and reliable data transfer capabilities.",
            "chapter": "Module 1",
            "section": "ROS 2 Architecture",
            "difficulty_level": "intermediate",
        },
        {
            "chunk_id": "chunk_003",
            "text": "SLAM (Simultaneous Localization and Mapping) allows robots to build a map of their environment while simultaneously tracking their location within that map.",
            "chapter": "Module 2",
            "section": "Navigation Stack",
            "difficulty_level": "intermediate",
        },
    ]


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for multi-turn testing."""
    return [
        {
            "query": "What is ROS 2?",
            "response": "ROS 2 is a flexible framework for writing robot software.",
            "tone": "english",
            "created_at": "2025-12-20T10:00:00Z",
        },
        {
            "query": "How does it communicate?",
            "response": "ROS 2 uses DDS for communication between nodes.",
            "tone": "english",
            "created_at": "2025-12-20T10:01:00Z",
        },
    ]


@pytest.fixture
def mock_gemini_response():
    """Mock response from Gemini API."""
    response = MagicMock()
    response.text = "This is a test response from Gemini about ROS 2."
    return response


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for vector search."""
    with patch("qdrant_client.QdrantClient") as mock_client:
        instance = MagicMock()
        mock_client.return_value = instance

        # Mock search results
        instance.search.return_value = [
            MagicMock(
                id="chunk_001",
                score=0.95,
                payload={
                    "text": "ROS 2 is a framework...",
                    "chapter": "Module 1",
                    "section": "ROS 2 Fundamentals",
                },
            ),
        ]

        yield instance


@pytest.fixture
def mock_runner():
    """Mock OpenAI Agents SDK Runner."""
    with patch("agents.Runner") as mock:
        result = MagicMock()
        result.final_output = "Mocked agent response"
        mock.run = MagicMock(return_value=result)
        yield mock
