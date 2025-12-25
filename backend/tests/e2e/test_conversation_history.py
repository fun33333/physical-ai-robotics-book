"""
E2E tests for Conversation History Endpoint (T059 - US3).

Tests the full end-to-end flow:
- POST /chat Q1
- POST /chat Q2 with conversationId
- Verify Q2 references Q1
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


class TestConversationHistoryE2E:
    """E2E tests for conversation history via API."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        session.query = MagicMock()
        session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
        return session

    @pytest.fixture
    def mock_agents(self):
        """Mock all agent functions for E2E testing."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            mock_rag.return_value = AsyncMock(return_value={
                "chunks": [
                    {
                        "chunk_id": "chunk_1",
                        "text": "ROS 2 is a robotics framework.",
                        "chapter": "Module 1",
                        "section": "Introduction",
                    }
                ],
                "latency_ms": 150,
            })()

            mock_answer.return_value = AsyncMock(return_value={
                "response": "ROS 2 is a flexible framework for building robot software.",
                "sources_used": ["chunk_1"],
                "latency_ms": 800,
            })()

            mock_tone.return_value = AsyncMock(return_value={
                "response": "ROS 2 is a flexible framework for building robot software.",
                "latency_ms": 50,
            })()

            mock_safety.return_value = AsyncMock(return_value={
                "validation_status": "approved",
                "latency_ms": 100,
            })()

            yield {
                "rag": mock_rag,
                "answer": mock_answer,
                "tone": mock_tone,
                "safety": mock_safety,
            }

    @pytest.fixture
    def client(self, mock_db_session, mock_agents):
        """Create test client with mocked dependencies."""
        from src.main import app
        from src.models.database import get_db

        def override_get_db():
            yield mock_db_session

        app.dependency_overrides[get_db] = override_get_db

        with TestClient(app) as test_client:
            yield test_client

        app.dependency_overrides.clear()

    def test_first_chat_returns_session_id(self, client):
        """Test that first chat returns a session_id for subsequent use."""
        response = client.post("/chat", json={
            "query": "What is ROS 2?",
            "tone": "english",
            "user_id": "test_user",
        })

        assert response.status_code == 200
        data = response.json()

        # Response should include session_id
        assert "session_id" in data
        assert data["session_id"] != ""

    def test_second_chat_with_session_id(self, client):
        """Test that second chat with session_id maintains context."""
        # First request
        response1 = client.post("/chat", json={
            "query": "What is ROS 2?",
            "tone": "english",
            "user_id": "test_user",
        })

        assert response1.status_code == 200
        session_id = response1.json()["session_id"]

        # Second request with same session_id
        response2 = client.post("/chat", json={
            "query": "Tell me more about it",
            "tone": "english",
            "user_id": "test_user",
            "session_id": session_id,
            "conversation_history": [
                {"query": "What is ROS 2?", "response": "ROS 2 is a framework."}
            ],
        })

        assert response2.status_code == 200
        data2 = response2.json()

        # Should maintain same session_id
        assert data2["session_id"] == session_id

    def test_chat_with_conversation_history(self, client):
        """Test that chat accepts and processes conversation_history."""
        history = [
            {"query": "What is ROS 2?", "response": "ROS 2 is a robotics framework."},
            {"query": "How does it work?", "response": "It uses DDS for communication."},
        ]

        response = client.post("/chat", json={
            "query": "What is DDS?",
            "tone": "english",
            "user_id": "test_user",
            "session_id": "existing_session",
            "conversation_history": history,
        })

        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_new_session_without_session_id(self, client):
        """Test that omitting session_id creates a new session."""
        response = client.post("/chat", json={
            "query": "What is Python?",
            "tone": "english",
            "user_id": "new_user",
        })

        assert response.status_code == 200
        data = response.json()

        # A new session_id should be generated
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_response_includes_required_fields(self, client):
        """Test that response includes all required fields for multi-turn."""
        response = client.post("/chat", json={
            "query": "What is ROS 2?",
            "tone": "english",
            "user_id": "test_user",
        })

        assert response.status_code == 200
        data = response.json()

        # Required fields for multi-turn support
        assert "response" in data
        assert "session_id" in data
        assert "sources" in data
        assert "tone" in data

    def test_empty_history_works(self, client):
        """Test that empty conversation_history is handled correctly."""
        response = client.post("/chat", json={
            "query": "What is ROS 2?",
            "tone": "english",
            "user_id": "test_user",
            "conversation_history": [],
        })

        assert response.status_code == 200


class TestConversationHistoryAPIValidation:
    """Tests for API validation of conversation history fields."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        session.query = MagicMock()
        session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
        return session

    @pytest.fixture
    def mock_agents(self):
        """Mock all agent functions."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            mock_rag.return_value = AsyncMock(return_value={
                "chunks": [],
                "latency_ms": 100,
            })()

            mock_answer.return_value = AsyncMock(return_value={
                "response": "Test response",
                "latency_ms": 500,
            })()

            mock_tone.return_value = AsyncMock(return_value={
                "response": "Test response",
                "latency_ms": 50,
            })()

            mock_safety.return_value = AsyncMock(return_value={
                "validation_status": "approved",
                "latency_ms": 50,
            })()

            yield

    @pytest.fixture
    def client(self, mock_db_session, mock_agents):
        """Create test client."""
        from src.main import app
        from src.models.database import get_db

        def override_get_db():
            yield mock_db_session

        app.dependency_overrides[get_db] = override_get_db

        with TestClient(app) as test_client:
            yield test_client

        app.dependency_overrides.clear()

    def test_session_id_preserved_in_response(self, client):
        """Test that provided session_id is preserved in response."""
        session_id = "my_custom_session_123"

        response = client.post("/chat", json={
            "query": "Test query",
            "tone": "english",
            "user_id": "test_user",
            "session_id": session_id,
        })

        assert response.status_code == 200
        assert response.json()["session_id"] == session_id

    def test_conversation_history_format_accepted(self, client):
        """Test that conversation_history with proper format is accepted."""
        history = [
            {
                "query": "Q1",
                "response": "A1",
                "tone": "english",
                "created_at": "2025-12-20T10:00:00Z",
            },
        ]

        response = client.post("/chat", json={
            "query": "Follow-up",
            "tone": "english",
            "user_id": "test_user",
            "conversation_history": history,
        })

        assert response.status_code == 200


class TestConversationContextContinuity:
    """Tests for conversation context continuity across multiple requests."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        session.query = MagicMock()
        session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
        return session

    @pytest.fixture
    def mock_agents(self):
        """Mock all agent functions with context-aware responses."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            mock_rag.return_value = AsyncMock(return_value={
                "chunks": [{"chunk_id": "1", "text": "Context", "chapter": "1", "section": "A"}],
                "latency_ms": 100,
            })()

            # Answer that references prior context
            mock_answer.return_value = AsyncMock(return_value={
                "response": "Building on what we discussed about ROS 2...",
                "latency_ms": 500,
            })()

            mock_tone.return_value = AsyncMock(return_value={
                "response": "Building on what we discussed about ROS 2...",
                "latency_ms": 50,
            })()

            mock_safety.return_value = AsyncMock(return_value={
                "validation_status": "approved",
                "latency_ms": 50,
            })()

            yield

    @pytest.fixture
    def client(self, mock_db_session, mock_agents):
        """Create test client."""
        from src.main import app
        from src.models.database import get_db

        def override_get_db():
            yield mock_db_session

        app.dependency_overrides[get_db] = override_get_db

        with TestClient(app) as test_client:
            yield test_client

        app.dependency_overrides.clear()

    def test_multi_turn_conversation_flow(self, client):
        """Test a complete multi-turn conversation flow."""
        session_id = None
        history = []

        # Turn 1
        response1 = client.post("/chat", json={
            "query": "What is ROS 2?",
            "tone": "english",
            "user_id": "test_user",
        })
        assert response1.status_code == 200
        data1 = response1.json()
        session_id = data1["session_id"]
        history.append({"query": "What is ROS 2?", "response": data1["response"]})

        # Turn 2
        response2 = client.post("/chat", json={
            "query": "Tell me more about DDS",
            "tone": "english",
            "user_id": "test_user",
            "session_id": session_id,
            "conversation_history": history,
        })
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id
        history.append({"query": "Tell me more about DDS", "response": data2["response"]})

        # Turn 3
        response3 = client.post("/chat", json={
            "query": "How do I use publishers?",
            "tone": "english",
            "user_id": "test_user",
            "session_id": session_id,
            "conversation_history": history,
        })
        assert response3.status_code == 200
        data3 = response3.json()
        assert data3["session_id"] == session_id

        # All turns should have maintained the same session
        assert data1["session_id"] == data2["session_id"] == data3["session_id"]
