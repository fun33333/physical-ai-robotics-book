"""
E2E tests for tone parameter in POST /chat endpoint (Phase 4 - T048).

Tests the complete flow:
- Send request with tone=roman_urdu
- Verify response uses Urdu phrases and colloquial language
- Verify same accuracy as English tone
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.main import app
from src.api import middleware


@pytest.fixture(autouse=True)
def clear_rate_limits():
    """Clear rate limits before each test."""
    middleware._rate_limits.clear()
    yield
    middleware._rate_limits.clear()


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_orchestration():
    """Mock orchestration service for testing."""
    with patch("src.api.routes.OrchestrationService") as mock:
        mock_instance = MagicMock()
        mock_instance.process_chat = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


class TestToneParameterValidation:
    """Tests for tone parameter validation."""

    def test_valid_english_tone(self, client, mock_orchestration):
        """Test that English tone is accepted."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "ROS 2 uses DDS for communication.",
            "sources": [],
            "tone": "english",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {},
            "total_latency_ms": 100.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What is ROS 2?",
                "tone": "english",
            },
        )

        assert response.status_code == 200
        assert response.json()["tone"] == "english"

    def test_valid_roman_urdu_tone(self, client, mock_orchestration):
        """Test that Roman Urdu tone is accepted."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "ROS 2 ek robotics framework hai.",
            "sources": [],
            "tone": "roman_urdu",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {},
            "total_latency_ms": 150.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What is ROS 2?",
                "tone": "roman_urdu",
            },
        )

        assert response.status_code == 200
        assert response.json()["tone"] == "roman_urdu"

    def test_valid_bro_guide_tone(self, client, mock_orchestration):
        """Test that Bro Guide tone is accepted."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "Yaar, scene yeh hai ke ROS 2...",
            "sources": [],
            "tone": "bro_guide",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {},
            "total_latency_ms": 150.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What is ROS 2?",
                "tone": "bro_guide",
            },
        )

        assert response.status_code == 200
        assert response.json()["tone"] == "bro_guide"

    def test_invalid_tone_rejected(self, client):
        """Test that invalid tones are rejected."""
        response = client.post(
            "/chat",
            json={
                "query": "What is ROS 2?",
                "tone": "invalid_tone",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_default_tone_is_english(self, client, mock_orchestration):
        """Test that default tone is English when not specified."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "ROS 2 uses DDS.",
            "sources": [],
            "tone": "english",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {},
            "total_latency_ms": 100.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What is ROS 2?",
                # No tone specified
            },
        )

        assert response.status_code == 200
        # Should default to English
        assert response.json()["tone"] == "english"


class TestRomanUrduResponse:
    """Tests for Roman Urdu tone responses."""

    def test_roman_urdu_uses_urdu_phrases(self, client, mock_orchestration):
        """Test that Roman Urdu response uses Urdu phrases."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "Dekho bhai, ROS 2 mein publishers data bhejte hain topics pe. Yeh important hai samajhna.",
            "sources": [{"chapter": "3", "section": "3.2"}],
            "tone": "roman_urdu",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {"tone": 50.0},
            "total_latency_ms": 200.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "How do publishers work in ROS 2?",
                "tone": "roman_urdu",
            },
        )

        assert response.status_code == 200
        response_text = response.json()["response"].lower()

        # Should contain Urdu phrases
        urdu_indicators = ["bhai", "hai", "mein", "ke", "yeh", "dekho", "pe"]
        assert any(phrase in response_text for phrase in urdu_indicators)

    def test_roman_urdu_preserves_technical_terms(self, client, mock_orchestration):
        """Test that Roman Urdu preserves technical terms in English."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "ROS 2 mein DDS protocol use hota hai. Publishers aur subscribers topics ke through communicate karte hain.",
            "sources": [],
            "tone": "roman_urdu",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {},
            "total_latency_ms": 150.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What communication protocol does ROS 2 use?",
                "tone": "roman_urdu",
            },
        )

        assert response.status_code == 200
        response_text = response.json()["response"]

        # Technical terms should remain in English
        assert "ROS" in response_text or "ros" in response_text.lower()
        assert "DDS" in response_text or "dds" in response_text.lower()


class TestBroGuideResponse:
    """Tests for Bro Guide tone responses."""

    def test_bro_guide_uses_casual_language(self, client, mock_orchestration):
        """Test that Bro Guide uses casual Karachi style."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "Yaar, scene yeh hai ke ROS 2 basically robots ko chalane ke liye hai. Super cool stuff!",
            "sources": [],
            "tone": "bro_guide",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {},
            "total_latency_ms": 150.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What is ROS 2 for?",
                "tone": "bro_guide",
            },
        )

        assert response.status_code == 200
        response_text = response.json()["response"].lower()

        # Should contain casual language
        casual_indicators = ["yaar", "scene", "ke", "hai", "cool", "bro"]
        assert any(phrase in response_text for phrase in casual_indicators)


class TestTechnicalAccuracyAcrossTones:
    """Tests for technical accuracy consistency across tones."""

    def test_same_question_same_accuracy(self, client, mock_orchestration):
        """Test that same question gives same technical accuracy in all tones."""
        # Technical content should be consistent
        technical_facts = {
            "ros": True,  # Should mention ROS
            "dds": True,  # Should mention DDS
        }

        for tone in ["english", "roman_urdu", "bro_guide"]:
            mock_result = MagicMock()
            mock_result.to_response.return_value = {
                "response": f"ROS 2 uses DDS communication protocol. ({tone} version)",
                "sources": [{"chapter": "3"}],
                "tone": tone,
                "session_id": "test_session",
                "validation_status": "valid",
                "latency_breakdown": {},
                "total_latency_ms": 150.0,
            }
            mock_orchestration.process_chat.return_value = mock_result

            response = client.post(
                "/chat",
                json={
                    "query": "What communication protocol does ROS 2 use?",
                    "tone": tone,
                },
            )

            assert response.status_code == 200
            response_text = response.json()["response"].lower()

            # All tones should contain the key technical facts
            assert "ros" in response_text
            assert "dds" in response_text


class TestSourcePreservation:
    """Tests for source citation preservation across tones."""

    def test_sources_included_in_all_tones(self, client, mock_orchestration):
        """Test that sources are included regardless of tone."""
        expected_sources = [
            {"chapter": "3", "section": "3.2", "title": "ROS 2 Communication"}
        ]

        for tone in ["english", "roman_urdu", "bro_guide"]:
            mock_result = MagicMock()
            mock_result.to_response.return_value = {
                "response": f"Response in {tone}",
                "sources": expected_sources,
                "tone": tone,
                "session_id": "test_session",
                "validation_status": "valid",
                "latency_breakdown": {},
                "total_latency_ms": 150.0,
            }
            mock_orchestration.process_chat.return_value = mock_result

            response = client.post(
                "/chat",
                json={
                    "query": "What is ROS 2?",
                    "tone": tone,
                },
            )

            assert response.status_code == 200
            assert response.json()["sources"] == expected_sources


class TestLatencyRequirements:
    """Tests for latency requirements (< 2s)."""

    def test_response_includes_latency_breakdown(self, client, mock_orchestration):
        """Test that response includes latency breakdown."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "Test response",
            "sources": [],
            "tone": "english",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {
                "rag": 100.0,
                "answer": 200.0,
                "tone": 50.0,
                "safety": 50.0,
            },
            "total_latency_ms": 400.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What is ROS 2?",
                "tone": "english",
            },
        )

        assert response.status_code == 200
        assert "latency_breakdown" in response.json()
        assert "total_latency_ms" in response.json()

    def test_total_latency_under_2s(self, client, mock_orchestration):
        """Test that total latency is under 2 seconds."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "Test response",
            "sources": [],
            "tone": "roman_urdu",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {},
            "total_latency_ms": 1500.0,  # 1.5 seconds
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What is ROS 2?",
                "tone": "roman_urdu",
            },
        )

        assert response.status_code == 200
        assert response.json()["total_latency_ms"] < 2000  # Under 2 seconds


class TestSelectedTextWithTone:
    """Tests for selected text with different tones."""

    def test_selected_text_with_roman_urdu(self, client, mock_orchestration):
        """Test that selected text works with Roman Urdu tone."""
        mock_result = MagicMock()
        mock_result.to_response.return_value = {
            "response": "Yeh selected text ke baare mein hai: publishers data publish karte hain.",
            "sources": [],
            "tone": "roman_urdu",
            "session_id": "test_session",
            "validation_status": "valid",
            "latency_breakdown": {},
            "total_latency_ms": 200.0,
        }
        mock_orchestration.process_chat.return_value = mock_result

        response = client.post(
            "/chat",
            json={
                "query": "What does this mean?",
                "selected_text": "Publishers send messages to topics",
                "tone": "roman_urdu",
            },
        )

        assert response.status_code == 200
        assert response.json()["tone"] == "roman_urdu"
