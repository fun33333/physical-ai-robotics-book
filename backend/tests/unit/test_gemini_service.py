"""
Unit tests for Gemini Service (Phase 3 - T033).

Tests the Gemini Service API key rotation logic:
- Key rotation when quota exceeded
- Fallback message when all keys exhausted
- Caching behavior
- Quota tracking
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

from src.services.gemini_service import GeminiService
from src.utils import GeminiError


class TestGeminiServiceInitialization:
    """Tests for GeminiService initialization."""

    def test_init_with_api_keys(self):
        """Test initialization with API keys."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2", "key3"]

            service = GeminiService()

            assert service.api_keys == ["key1", "key2", "key3"]
            assert service.current_key_index == 0

    def test_init_without_api_keys(self):
        """Test initialization without API keys logs warning."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = []

            service = GeminiService()

            assert service.api_keys == []


class TestApiKeyRotation:
    """Tests for API key rotation functionality."""

    @pytest.fixture
    def service_with_keys(self):
        """Create service with multiple API keys."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2", "key3"]
            return GeminiService()

    def test_get_current_api_key(self, service_with_keys):
        """Test getting current API key."""
        key = service_with_keys._get_current_api_key()
        assert key == "key1"

    def test_get_current_api_key_no_keys_raises(self):
        """Test that getting key with no keys raises error."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = []
            service = GeminiService()

            with pytest.raises(GeminiError) as excinfo:
                service._get_current_api_key()

            assert "No API keys configured" in str(excinfo.value)

    def test_rotate_to_next_key(self, service_with_keys):
        """Test rotation to next key."""
        # Initial key is index 0 (key1)
        assert service_with_keys.current_key_index == 0

        # Rotate to next key
        next_key = service_with_keys._rotate_to_next_key()

        assert service_with_keys.current_key_index == 1
        assert next_key == "key2"

    def test_rotate_wraps_around(self, service_with_keys):
        """Test that rotation wraps around to first key."""
        service_with_keys.current_key_index = 2  # Last key
        service_with_keys._rotation_count = 0  # Reset rotation tracking

        # Rotate should go back to first key
        next_key = service_with_keys._rotate_to_next_key()

        assert service_with_keys.current_key_index == 0
        assert next_key == "key1"

    def test_all_keys_exhausted_raises_error(self):
        """Test that exhausting all keys raises error with fallback message."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2", "key3"]
            service = GeminiService()

            # Simulate exhaustion - rotation method goes through all keys
            with pytest.raises(GeminiError) as excinfo:
                # Force exhaustion by calling rotate multiple times
                for _ in range(4):  # More than number of keys
                    try:
                        service._rotate_to_next_key()
                    except GeminiError:
                        raise

            assert "limit of AI Guide is exceeded" in str(excinfo.value)
            assert "try again tomorrow" in str(excinfo.value)


class TestCaching:
    """Tests for response caching."""

    @pytest.fixture
    def service(self):
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1"]
            return GeminiService()

    def test_cache_key_generation(self, service):
        """Test cache key is generated consistently."""
        key1 = service._get_cache_key("What is ROS?", "ROS is a framework")
        key2 = service._get_cache_key("What is ROS?", "ROS is a framework")

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length

    def test_cache_key_differs_for_different_input(self, service):
        """Test cache key differs for different queries."""
        key1 = service._get_cache_key("What is ROS?", "context")
        key2 = service._get_cache_key("What is SLAM?", "context")

        assert key1 != key2

    def test_store_and_retrieve_cache(self, service):
        """Test storing and retrieving from cache."""
        query = "What is ROS?"
        context = "ROS is a robotics framework"
        response = "ROS is a framework for building robots"

        # Initially not cached
        assert service._check_cache(query, context) is None

        # Store in cache
        service._store_cache(query, context, response)

        # Now should be cached
        cached = service._check_cache(query, context)
        assert cached == response


class TestGenerateResponse:
    """Tests for response generation."""

    @pytest.fixture
    def service(self):
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2"]
            return GeminiService()

    def test_returns_cached_response(self, service):
        """Test that cached response is returned without API call."""
        query = "What is ROS?"
        context = "ROS is a framework"

        # Pre-populate cache
        service._store_cache(query, context, "Cached response about ROS")

        # Should return cached response
        response = service.generate_response(query, context)

        assert response == "Cached response about ROS"

    @patch("src.services.gemini_service.genai")
    def test_calls_gemini_api(self, mock_genai, service):
        """Test that Gemini API is called for uncached queries."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="Generated response")
        mock_genai.GenerativeModel.return_value = mock_model

        response = service.generate_response(
            "New question",
            "Some context",
        )

        assert mock_genai.configure.called
        assert mock_model.generate_content.called

    @patch("src.services.gemini_service.genai")
    def test_rotates_key_on_error(self, mock_genai, service):
        """Test that key is rotated on API error."""
        mock_model = MagicMock()
        # First call fails, second succeeds
        mock_model.generate_content.side_effect = [
            Exception("Quota exceeded"),
            MagicMock(text="Response from key2"),
        ]
        mock_genai.GenerativeModel.return_value = mock_model

        # Add a second key for rotation to work
        service.api_keys = ["key1", "key2"]

        response = service.generate_response("Question", "Context")

        # Should have rotated and retried
        assert service.current_key_index == 1
        assert response == "Response from key2"

    @patch("src.services.gemini_service.genai")
    def test_returns_fallback_when_all_keys_fail(self, mock_genai, service):
        """Test fallback message when all keys are exhausted."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("All failed")
        mock_genai.GenerativeModel.return_value = mock_model

        with pytest.raises(GeminiError) as excinfo:
            service.generate_response("Question", "Context")

        assert "limit of AI Guide is exceeded" in str(excinfo.value)


class TestPromptBuilding:
    """Tests for prompt construction."""

    @pytest.fixture
    def service(self):
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1"]
            return GeminiService()

    def test_prompt_includes_tone_instruction(self, service):
        """Test that prompt includes tone-specific instruction."""
        prompt = service._build_prompt(
            query="What is ROS?",
            context="ROS is a framework",
            tone="roman_urdu",
            user_level="beginner",
            conversation_history=None,
        )

        assert "Roman Urdu" in prompt or "Urdu" in prompt

    def test_prompt_includes_user_level(self, service):
        """Test that prompt includes user level instruction."""
        prompt = service._build_prompt(
            query="What is ROS?",
            context="ROS is a framework",
            tone="english",
            user_level="beginner",
            conversation_history=None,
        )

        assert "simple" in prompt.lower() or "ELI5" in prompt

    def test_prompt_includes_context(self, service):
        """Test that prompt includes retrieved context."""
        prompt = service._build_prompt(
            query="What is ROS?",
            context="ROS 2 uses DDS middleware",
            tone="english",
            user_level="intermediate",
            conversation_history=None,
        )

        assert "DDS middleware" in prompt

    def test_prompt_includes_conversation_history(self, service):
        """Test that prompt includes conversation history."""
        history = [
            {"query": "What is ROS?", "response": "ROS is a robotics framework."},
            {"query": "Tell me more", "response": "It provides tools and libraries."},
        ]

        prompt = service._build_prompt(
            query="How do I install it?",
            context="Installation requires Ubuntu",
            tone="english",
            user_level="intermediate",
            conversation_history=history,
        )

        assert "robotics framework" in prompt
        assert "tools and libraries" in prompt


class TestQuotaTracking:
    """Tests for API quota tracking."""

    def test_tracks_successful_request(self):
        """Test that successful requests are tracked."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1"]

            mock_session = MagicMock()
            mock_quota = MagicMock()
            mock_quota.last_reset = datetime.utcnow()
            mock_quota.requests_today = 0
            mock_quota.requests_per_minute_today = 0
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_quota

            service = GeminiService(db_session=mock_session)
            service._track_quota("key1", success=True)

            assert mock_quota.requests_today == 1

    def test_resets_quota_at_midnight(self):
        """Test that quota resets at midnight UTC."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1"]

            mock_session = MagicMock()
            mock_quota = MagicMock()
            # Last reset was yesterday
            mock_quota.last_reset = datetime.utcnow() - timedelta(days=1)
            mock_quota.requests_today = 100
            mock_session.query.return_value.filter_by.return_value.first.return_value = mock_quota

            service = GeminiService(db_session=mock_session)
            service._track_quota("key1", success=True)

            # Should have reset
            assert mock_quota.requests_today == 1


class TestKeyStatus:
    """Tests for key status reporting."""

    def test_get_key_status(self):
        """Test getting status of all keys."""
        with patch("src.services.gemini_service.settings") as mock_settings:
            mock_settings.gemini_api_keys = ["key1", "key2", "key3"]

            service = GeminiService()
            status = service.get_key_status()

            assert status["total_keys"] == 3
            assert status["current_key_index"] == 0
            assert "gemini_1" in status["keys"]
            assert "gemini_2" in status["keys"]
            assert "gemini_3" in status["keys"]
