"""
OpenAI Agents SDK Configuration with Gemini via LiteLLM.

Uses LitellmModel from OpenAI Agents SDK to connect to Gemini API.
Supports multiple API keys with automatic rotation on quota exceeded.

Reference: https://openai.github.io/openai-agents-python/models/
Reference: https://github.com/panaversity/learn-agentic-ai/tree/main/01_ai_agents_first/05_model_configuration
"""

from typing import Optional

from agents import set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)

# Disable tracing for non-OpenAI models (required per documentation)
set_tracing_disabled(disabled=True)

# Gemini model to use via LiteLLM
GEMINI_MODEL = "gemini/gemini-2.0-flash"

# Global state for key rotation
_current_key_index = 0


def get_gemini_model() -> LitellmModel:
    """
    Get LitellmModel configured for Gemini.

    Returns the currently active Gemini model using LiteLLM integration.
    Use rotate_api_key() when quota is exceeded.

    Returns:
        LitellmModel configured for Gemini API
    """
    global _current_key_index

    api_keys = settings.gemini_api_keys

    if not api_keys:
        logger.warning("No Gemini API keys configured. Using placeholder.")
        return LitellmModel(
            model=GEMINI_MODEL,
            api_key="placeholder_key",
        )

    current_key = api_keys[_current_key_index]

    logger.debug(f"Using Gemini API key {_current_key_index + 1} of {len(api_keys)}")

    return LitellmModel(
        model=GEMINI_MODEL,
        api_key=current_key,
    )


def rotate_api_key() -> Optional[LitellmModel]:
    """
    Rotate to next Gemini API key.

    Called when current key hits quota limit (15 req/min for free tier).
    Returns None if all keys exhausted.

    Returns:
        Next LitellmModel or None if all keys exhausted
    """
    global _current_key_index

    api_keys = settings.gemini_api_keys

    if not api_keys:
        logger.warning("No API keys available for rotation.")
        return None

    # Move to next key
    _current_key_index = (_current_key_index + 1) % len(api_keys)

    # Check if we've cycled through all keys
    if _current_key_index == 0 and len(api_keys) > 1:
        logger.warning("All Gemini API keys exhausted. Cycling back to first key.")

    logger.info(f"Rotated to Gemini API key {_current_key_index + 1} of {len(api_keys)}")

    return LitellmModel(
        model=GEMINI_MODEL,
        api_key=api_keys[_current_key_index],
    )


def get_current_key_index() -> int:
    """
    Get index of current API key.

    Returns:
        Current key index (0-based)
    """
    return _current_key_index


def get_key_count() -> int:
    """
    Get number of available API keys.

    Returns:
        Number of configured Gemini API keys
    """
    return len(settings.gemini_api_keys)


def reset_key_rotation() -> None:
    """
    Reset key rotation to first key.

    Called at daily quota reset (00:00 UTC).
    """
    global _current_key_index
    _current_key_index = 0
    logger.info("API key rotation reset to first key")
