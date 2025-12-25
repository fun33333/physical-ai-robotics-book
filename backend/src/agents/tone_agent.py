"""
Tone Agent using OpenAI Agents SDK.

Transforms responses into different communication styles with conciseness support.
"""

from typing import Any, Dict, Optional

from agents import Agent, Runner

from src.agents.agents_config import get_gemini_model
from src.agents.tone_prompts import (
    get_system_prompt,
    get_transform_prompt,
    TONE_DESCRIPTIONS,
)
from src.utils import get_logger, track_latency

logger = get_logger(__name__)

# Conciseness threshold in characters (T051)
CONCISENESS_THRESHOLD = 250

# Tone descriptions (exported for backward compatibility)
TONES = TONE_DESCRIPTIONS


def create_tone_agent(tone: str) -> Agent:
    """Create Tone Agent for specific style.

    Args:
        tone: The tone identifier (english, roman_urdu, bro_guide)

    Returns:
        Configured Agent instance for the specified tone
    """
    system_prompt = get_system_prompt(tone)
    return Agent(
        name=f"Tone Agent ({tone})",
        instructions=system_prompt,
        model=get_gemini_model(),
    )


def apply_conciseness(response: str) -> Dict[str, Any]:
    """Apply conciseness to a response if it exceeds threshold.

    Truncates long responses to 1-2 sentences and adds "Ask for longer?" prompt.
    Stores full response for retrieval on explicit request.

    Args:
        response: The response to potentially truncate

    Returns:
        Dictionary with:
        - response: The (possibly truncated) response
        - is_truncated: Whether truncation was applied
        - full_response: The original full response if truncated, else None
    """
    if not response or len(response.strip()) == 0:
        return {
            "response": response,
            "is_truncated": False,
            "full_response": None,
        }

    if len(response) <= CONCISENESS_THRESHOLD:
        return {
            "response": response,
            "is_truncated": False,
            "full_response": None,
        }

    # Find natural break points (sentences)
    sentences = []
    current = ""
    for char in response:
        current += char
        if char in ".!?":
            sentences.append(current.strip())
            current = ""

    # Keep first 1-2 complete sentences within threshold
    truncated = ""
    for sentence in sentences[:2]:
        if len(truncated) + len(sentence) + 1 <= CONCISENESS_THRESHOLD:
            truncated += sentence + " "
        else:
            break

    # If no complete sentences found, take first part
    if not truncated.strip():
        truncated = response[:CONCISENESS_THRESHOLD - 50].rsplit(" ", 1)[0] + "..."

    truncated = truncated.strip()

    # Add "Ask for longer?" prompt
    truncated += "\n\nWould you like a more detailed explanation?"

    return {
        "response": truncated,
        "is_truncated": True,
        "full_response": response,
    }


async def apply_tone(
    response: str,
    tone: str = "english",
    apply_concise: bool = False,
) -> Dict[str, Any]:
    """Apply tone transformation to response with optional conciseness.

    Args:
        response: The response to transform
        tone: The target tone (english, roman_urdu, bro_guide)
        apply_concise: Whether to apply conciseness logic

    Returns:
        Dictionary with transformed response and metadata
    """
    result: Dict[str, Any] = {"tone": tone}

    # For English, skip LLM transformation
    if tone == "english":
        result["response"] = response
    else:
        # Transform to target tone using LLM
        with track_latency("tone_transform") as latency:
            try:
                agent = create_tone_agent(tone)
                transform_prompt = get_transform_prompt(tone, response)
                llm_result = await Runner.run(agent, transform_prompt)
                result["response"] = llm_result.final_output or response
                result["latency_ms"] = latency["elapsed_ms"]
            except Exception as e:
                logger.error(f"Tone transform failed: {e}")
                result["response"] = response
                result["tone"] = "english"
                result["error"] = str(e)

    # Apply conciseness if requested
    if apply_concise:
        concise_result = apply_conciseness(result["response"])
        result["response"] = concise_result["response"]
        result["is_truncated"] = concise_result["is_truncated"]
        result["full_response"] = concise_result["full_response"]

    return result


async def apply_tone_with_conciseness(
    response: str,
    tone: str = "english",
) -> Dict[str, Any]:
    """Apply tone transformation with conciseness enabled.

    Convenience function that enables conciseness by default.

    Args:
        response: The response to transform
        tone: The target tone

    Returns:
        Dictionary with transformed, potentially truncated response
    """
    return await apply_tone(response, tone, apply_concise=True)


def get_full_response(truncated_result: Dict[str, Any]) -> Optional[str]:
    """Get the full response from a truncated result.

    Args:
        truncated_result: Result from apply_tone with apply_concise=True

    Returns:
        The full response if truncated, else None
    """
    return truncated_result.get("full_response")
