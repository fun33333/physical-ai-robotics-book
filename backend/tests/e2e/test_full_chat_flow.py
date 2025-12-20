"""
End-to-End tests for Full Chat Flow (Phase 3 - T036).

Tests complete chat flow through FastAPI endpoint,
verifying response structure, latency, and tone parameter handling.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from src.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestChatEndpoint:
    """Tests for /chat endpoint."""

    def test_chat_endpoint_basic_request(self, client):
        """Test basic chat request."""
        request_body = {
            "query": "What is ROS 2?",
            "tone": "english",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "response" in data
        assert "sources" in data
        assert "agent_used" in data
        assert "tone" in data
        assert "latency_breakdown" in data
        assert "total_latency_ms" in data

    def test_chat_endpoint_with_selected_text(self, client):
        """Test chat endpoint with selected text context."""
        request_body = {
            "query": "Explain this concept",
            "selected_text": "ROS 2 uses DDS for publish-subscribe messaging.",
            "tone": "english",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()
        assert len(data["response"]) > 0

    def test_chat_endpoint_with_roman_urdu_tone(self, client):
        """Test chat endpoint with roman_urdu tone."""
        request_body = {
            "query": "یہ کیا ہے؟",
            "tone": "roman_urdu",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()
        assert data["tone"] == "roman_urdu"

    def test_chat_endpoint_with_bro_guide_tone(self, client):
        """Test chat endpoint with bro_guide tone."""
        request_body = {
            "query": "Yo, what's ROS?",
            "tone": "bro_guide",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()
        assert data["tone"] == "bro_guide"

    def test_chat_endpoint_latency_breakdown(self, client):
        """Test that latency breakdown is provided."""
        request_body = {
            "query": "Test query",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()

        breakdown = data["latency_breakdown"]

        # Should have latency for main stages
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

    def test_chat_endpoint_respects_latency_slo(self, client):
        """Test that endpoint respects 2-second SLO."""
        request_body = {
            "query": "Quick test",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()

        # Total latency should be < 2 seconds (2000ms)
        assert data["total_latency_ms"] < 2000

    def test_chat_endpoint_includes_sources(self, client):
        """Test that sources are included in response."""
        request_body = {
            "query": "What chapter discusses this?",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()

        sources = data["sources"]
        assert isinstance(sources, list)

    def test_chat_endpoint_empty_query_validation(self, client):
        """Test that empty query is rejected."""
        request_body = {
            "query": "",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_chat_endpoint_invalid_tone_validation(self, client):
        """Test that invalid tone is rejected."""
        request_body = {
            "query": "Valid query",
            "tone": "invalid_tone",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 400

    def test_chat_endpoint_invalid_user_level_validation(self, client):
        """Test that invalid user_level is rejected."""
        request_body = {
            "query": "Valid query",
            "user_level": "super_advanced",
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 400

    def test_chat_endpoint_query_too_long(self, client):
        """Test that very long queries are rejected."""
        request_body = {
            "query": "a" * 10000,  # Way too long
            "user_id": "test_user",
            "session_id": "test_session",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 400

    def test_chat_endpoint_default_tone(self, client):
        """Test that default tone is applied."""
        request_body = {
            "query": "What is ROS?",
            "user_id": "test_user",
            "session_id": "test_session",
            # tone not specified
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()
        assert data["tone"] == "english"  # Default tone

    def test_chat_endpoint_default_user_level(self, client):
        """Test that default user_level is applied."""
        request_body = {
            "query": "What is ROS?",
            "user_id": "test_user",
            "session_id": "test_session",
            # user_level not specified
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()
        # Verify response was generated (user_level defaults to intermediate)
        assert len(data["response"]) > 0

    def test_chat_endpoint_with_conversation_history(self, client):
        """Test chat endpoint with conversation history."""
        request_body = {
            "query": "What else?",
            "user_id": "test_user",
            "session_id": "test_session",
            "conversation_history": [
                {
                    "query": "What is ROS?",
                    "response": "ROS is a robotics middleware.",
                }
            ],
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()
        assert len(data["response"]) > 0


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_endpoint_success(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "operational"
        assert "version" in data
        assert "service" in data
        assert "coordinator" in data

    def test_health_endpoint_structure(self, client):
        """Test health endpoint response structure."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "status" in data
        assert "version" in data
        assert "service" in data
        assert "checks" in data or "coordinator" in data


class TestIndexEndpoint:
    """Tests for /index endpoint."""

    def test_index_endpoint_not_implemented(self, client):
        """Test index endpoint (not implemented in Phase 1)."""
        response = client.post("/index", json={})

        # Should return "not_implemented" message
        assert response.status_code == 200
        data = response.json()
        assert "not_implemented" in data["status"]


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_endpoint_not_implemented(self, client):
        """Test search endpoint (not implemented in Phase 1)."""
        response = client.get("/search?query=test")

        # Should return "not_implemented" message
        assert response.status_code == 200
        data = response.json()
        assert "not_implemented" in data["status"]


class TestEndpointErrorHandling:
    """Tests for error handling in endpoints."""

    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON."""
        response = client.post("/chat", content="invalid json")

        # Should return error
        assert response.status_code >= 400

    def test_missing_required_field(self, client):
        """Test handling of missing required fields."""
        request_body = {
            "tone": "english",
            # Missing 'query'
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 400

    def test_chat_response_content_type(self, client):
        """Test that response is valid JSON."""
        request_body = {
            "query": "Test",
            "user_id": "user_123",
            "session_id": "session_456",
        }

        response = client.post("/chat", json=request_body)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        # Should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
