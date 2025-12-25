"""
Integration tests for fallback message when all keys exhausted (Phase 8 - T086).

Tests that when all API keys are exhausted:
- The exact fallback message is returned
- The message is: "Your today's limit of AI Guide is exceeded. Please try again tomorrow."
"""

import pytest
from unittest.mock import MagicMock, patch

from src.services.gemini_service import GeminiService
from src.services.orchestration_service import OrchestrationService, PipelineContext
from src.utils import GeminiError


class TestFallbackMessageOnAllKeysExhausted:
    """Tests for fallback message when all API keys are exhausted."""

    def test_exact_fallback_message_from_gemini_service(self):
        """Test that GeminiService returns exact fallback message."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2", "key3"]

            service = GeminiService()

            # Exhaust all keys
            with pytest.raises(GeminiError) as excinfo:
                for _ in range(3):
                    service._rotate_to_next_key()

            expected_message = "Your today's limit of AI Guide is exceeded. Please try again tomorrow."
            assert str(excinfo.value) == expected_message

    @patch("src.services.gemini_service.genai")
    def test_fallback_message_on_generate_response(self, mock_genai):
        """Test fallback message when generate_response exhausts all keys."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2"]

            mock_model = MagicMock()
            # All calls fail
            mock_model.generate_content.side_effect = Exception("Quota exceeded")
            mock_genai.GenerativeModel.return_value = mock_model

            service = GeminiService()

            with pytest.raises(GeminiError) as excinfo:
                service.generate_response("Test query", "Test context")

            assert "today's limit of AI Guide is exceeded" in str(excinfo.value)
            assert "try again tomorrow" in str(excinfo.value)


class TestFallbackMessageFormat:
    """Tests for fallback message format consistency."""

    def test_message_is_user_friendly(self):
        """Test that message is user-friendly, not technical."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1"]

            service = GeminiService()

            with pytest.raises(GeminiError) as excinfo:
                service._rotate_to_next_key()

            message = str(excinfo.value)

            # Should not contain technical terms
            assert "api" not in message.lower() or "AI" in message  # AI Guide is ok
            assert "key" not in message.lower()
            assert "quota" not in message.lower()
            assert "rate limit" not in message.lower()

            # Should be encouraging
            assert "tomorrow" in message.lower()

    def test_message_mentions_ai_guide(self):
        """Test that message mentions 'AI Guide' branding."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1"]

            service = GeminiService()

            with pytest.raises(GeminiError) as excinfo:
                service._rotate_to_next_key()

            assert "AI Guide" in str(excinfo.value)


class TestFallbackThroughOrchestration:
    """Tests for fallback message through orchestration service."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        session.query = MagicMock(return_value=MagicMock(
            filter=MagicMock(return_value=MagicMock(
                all=MagicMock(return_value=[])
            ))
        ))
        return session

    @pytest.mark.asyncio
    async def test_orchestration_handles_fallback(self, mock_db_session):
        """Test that orchestration service handles fallback gracefully."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            mock_rag.return_value = {"chunks": [], "sources": []}
            # Simulate all keys exhausted
            mock_answer.side_effect = GeminiError(
                "Your today's limit of AI Guide is exceeded. Please try again tomorrow."
            )

            orchestrator = OrchestrationService(db_session=mock_db_session)
            context = PipelineContext(
                query="What is ROS 2?",
                user_id="test_user",
                session_id="test_session",
            )

            # Orchestration catches errors and returns graceful response
            result = await orchestrator.process_chat(context)

            # Should return error response instead of crashing
            assert result is not None
            assert "error" in result.response.lower() or "sorry" in result.response.lower()


class TestFallbackTimingScenarios:
    """Tests for fallback in various timing scenarios."""

    @patch("src.services.gemini_service.genai")
    def test_fallback_after_first_key_exhausted(self, mock_genai):
        """Test fallback after single key exhaustion (1 key configured)."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["single_key"]

            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("Quota exceeded")
            mock_genai.GenerativeModel.return_value = mock_model

            service = GeminiService()

            with pytest.raises(GeminiError) as excinfo:
                service.generate_response("Query", "Context")

            assert "limit of AI Guide is exceeded" in str(excinfo.value)

    @patch("src.services.gemini_service.genai")
    def test_fallback_after_all_three_keys_exhausted(self, mock_genai):
        """Test fallback after all three keys are exhausted."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2", "key3"]

            mock_model = MagicMock()
            # All calls fail
            mock_model.generate_content.side_effect = Exception("Quota exceeded")
            mock_genai.GenerativeModel.return_value = mock_model

            service = GeminiService()

            with pytest.raises(GeminiError) as excinfo:
                service.generate_response("Query", "Context")

            assert "limit of AI Guide is exceeded" in str(excinfo.value)


class TestFallbackRecovery:
    """Tests for recovery after fallback."""

    def test_can_make_requests_after_reset(self):
        """Test that requests can be made after rotation tracking reset."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2", "key3"]

            service = GeminiService()

            # Exhaust all keys
            try:
                for _ in range(3):
                    service._rotate_to_next_key()
            except GeminiError:
                pass

            # Reset rotation tracking (simulating new day/request)
            service._reset_rotation_tracking()

            # Should be able to rotate again
            next_key = service._rotate_to_next_key()
            assert next_key is not None
