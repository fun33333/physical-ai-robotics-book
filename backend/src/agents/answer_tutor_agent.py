"""
Answer/Tutor Agent using OpenAI Agents SDK.

Generates educational responses based on retrieved context.
"""

from typing import Any, Dict, List, Optional

from agents import Agent, Runner

from src.agents.agents_config import get_gemini_model
from src.utils import get_logger, track_latency

logger = get_logger(__name__)

LEVEL_PROMPTS = {
    "beginner": "Explain simply, avoid jargon, use analogies.",
    "intermediate": "Use technical terms but explain complex ones.",
    "advanced": "Provide deep technical analysis with implementation details.",
}


def create_answer_agent(level: str = "intermediate") -> Agent:
    """Create Answer Agent with level-specific instructions."""
    return Agent(
        name="Answer Tutor Agent",
        instructions=f"""You are an AI tutor for Physical AI and Robotics.

RULES:
1. ONLY use information from the provided SOURCES
2. Cite sources (e.g., "According to SOURCE 1...")
3. If info not in sources, say "I don't find this in the textbook"
4. {LEVEL_PROMPTS.get(level, LEVEL_PROMPTS["intermediate"])}
5. Keep response concise (1-2 paragraphs)
""",
        model=get_gemini_model(),
    )


def _format_conversation_history(history: List[Dict], max_exchanges: int = 5) -> str:
    """
    Format conversation history for inclusion in prompt.

    Compresses history to last N exchanges to manage context window.

    Args:
        history: List of prior Q&A exchanges
        max_exchanges: Maximum number of exchanges to include

    Returns:
        Formatted string of prior conversation context
    """
    if not history:
        return ""

    # Take only the most recent exchanges
    recent_history = history[-max_exchanges:]

    formatted_parts = []
    for i, exchange in enumerate(recent_history):
        q = exchange.get("query", "")
        a = exchange.get("response", "")

        # Truncate long responses to save context
        if len(a) > 200:
            a = a[:200] + "..."

        formatted_parts.append(f"Q{i+1}: {q}\nA{i+1}: {a}")

    return "\n\n".join(formatted_parts)


async def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict]] = None,
    user_level: str = "intermediate",
    selected_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate educational response from retrieved context with conversation awareness.

    Includes prior Q&A in the prompt to enable contextual follow-up responses.

    Args:
        query: Current user question
        chunks: Retrieved document chunks with sources
        conversation_history: List of prior Q&A exchanges for context
        user_level: User expertise level (beginner, intermediate, advanced)
        selected_text: User-highlighted text from textbook

    Returns:
        Dict with response, sources_used, and latency_ms
    """
    with track_latency("answer_generation") as latency:
        try:
            agent = create_answer_agent(user_level)

            # Format sources from retrieved chunks
            sources = "\n".join(
                f"[SOURCE {i+1}] Ch.{c.get('chapter', '?')} - {c.get('section', '?')}:\n{c.get('text', '')}"
                for i, c in enumerate(chunks)
            )

            # Build prompt with conversation context (T061)
            prompt_parts = []

            # Add conversation history for multi-turn context
            if conversation_history:
                history_text = _format_conversation_history(conversation_history)
                if history_text:
                    prompt_parts.append(f"PRIOR CONVERSATION:\n{history_text}")
                    prompt_parts.append("---")

            # Add highlighted text if provided
            if selected_text:
                prompt_parts.append(f"HIGHLIGHTED TEXT: {selected_text[:300]}")

            # Add retrieved sources
            prompt_parts.append(f"SOURCES:\n{sources}")

            # Add the current question
            prompt_parts.append(f"\nCURRENT QUESTION: {query}")

            # Add instruction for contextual response
            if conversation_history:
                prompt_parts.append(
                    "\n(Reference prior conversation when relevant, but focus on the current question.)"
                )

            prompt = "\n\n".join(prompt_parts)

            result = await Runner.run(agent, prompt)
            response = result.final_output or "Unable to generate response."

            return {
                "response": response,
                "sources_used": [c.get("chunk_id") for c in chunks if c.get("chunk_id")],
                "latency_ms": latency["elapsed_ms"],
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {"response": "Error generating response.", "error": str(e)}
