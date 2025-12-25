"""
RAG Agent using OpenAI Agents SDK.

Retrieves relevant document chunks from Qdrant vector database.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from agents import Agent, Runner, function_tool

from src.agents.agents_config import get_gemini_model
from src.services.qdrant_service import QdrantService
from src.utils import get_logger, track_latency

logger = get_logger(__name__)

_qdrant: Optional[QdrantService] = None


def get_qdrant() -> QdrantService:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantService()
    return _qdrant


def generate_embedding(text: str) -> List[float]:
    """Generate 768-dim embedding (placeholder - replace with real model)."""
    hash_bytes = hashlib.sha256(text.encode()).digest()
    embedding = [float(b) / 255.0 for b in hash_bytes]
    embedding = (embedding + [0.0] * 768)[:768]
    magnitude = sum(x * x for x in embedding) ** 0.5
    return [x / magnitude for x in embedding] if magnitude > 0 else embedding


@function_tool
def search_textbook(query: str, selected_text: str = "") -> str:
    """Search textbook for relevant content."""
    try:
        combined = f"{query} {selected_text[:500]}" if selected_text else query
        embedding = generate_embedding(combined)
        results = get_qdrant().search_similar(embedding, top_k=5, query_text=query)
        return json.dumps({"chunks": results, "count": len(results)})
    except Exception as e:
        return json.dumps({"chunks": [], "error": str(e)})


rag_agent = Agent(
    name="RAG Agent",
    instructions="Search the textbook for relevant content using the search_textbook tool.",
    model=get_gemini_model(),
    tools=[search_textbook],
)


async def retrieve_context(
    query: str,
    selected_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve relevant chunks for a query."""
    with track_latency("rag_retrieval") as latency:
        try:
            # Direct tool call for efficiency
            result = search_textbook(query, selected_text or "")
            data = json.loads(result)
            return {
                "chunks": data.get("chunks", []),
                "count": len(data.get("chunks", [])),
                "latency_ms": latency["elapsed_ms"],
            }
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return {"chunks": [], "count": 0, "error": str(e)}
