"""
FastAPI routes for RAG Chatbot API.

Defines endpoints for chat, indexing, search, and health checks.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.agents.coordinator import MainCoordinatorAgent
from src.models.database import get_db
from src.services.conversation_service import ConversationService
from src.utils import get_logger, ValidationError

logger = get_logger(__name__)
router = APIRouter()

# Initialize coordinator agent (will be instantiated per request in Phase 3)
coordinator_agent = MainCoordinatorAgent()


# Request/Response Models (simplified for now, Pydantic models in Phase 2)
class ChatRequest:
    """Chat endpoint request."""

    def __init__(self, data: Dict[str, Any]):
        self.query = data.get("query", "")
        self.selected_text = data.get("selected_text")
        self.tone = data.get("tone", "english")
        self.user_id = data.get("user_id", "anonymous")
        self.session_id = data.get("session_id", "default")
        self.user_level = data.get("user_level", "intermediate")
        self.conversation_history = data.get("conversation_history", [])

    def validate(self) -> None:
        """Validate request fields."""
        if not self.query:
            raise ValidationError("Query cannot be empty")
        if len(self.query) > 5000:
            raise ValidationError("Query too long (max 5000 characters)")
        if self.tone not in ["english", "roman_urdu", "bro_guide"]:
            raise ValidationError(f"Invalid tone: {self.tone}")
        if self.user_level not in ["beginner", "intermediate", "advanced"]:
            raise ValidationError(f"Invalid user level: {self.user_level}")


@router.post("/chat", tags=["Chat"])
async def chat_endpoint(request: Dict[str, Any], db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Chat endpoint for question-answering.

    Accepts user query with optional context and returns AI response from orchestrator.

    Request body:
    ```json
    {
        "query": "What is ROS 2?",
        "selected_text": "Optional selected text from textbook",
        "tone": "english|roman_urdu|bro_guide",
        "user_id": "user_123",
        "session_id": "session_456",
        "user_level": "beginner|intermediate|advanced",
        "conversation_history": []
    }
    ```

    Response:
    ```json
    {
        "response": "AI-generated response",
        "sources": [{"chapter": "2", "section": "ROS 2 Basics", "relevance_score": 0.95}],
        "agent_used": "orchestrator",
        "tone": "english",
        "latency_breakdown": {"coordinator": 5.2, "rag": 150.3, ...},
        "total_latency_ms": 285.5
    }
    ```

    Args:
        request: Chat request with query and options
        db: Database session

    Returns:
        Chat response with answer and metadata

    Raises:
        HTTPException: If validation or processing fails
    """
    try:
        # Validate request
        chat_req = ChatRequest(request)
        chat_req.validate()

        # Route through coordinator
        response = await coordinator_agent.route_through_pipeline(
            query=chat_req.query,
            selected_text=chat_req.selected_text,
            tone=chat_req.tone,
            user_id=chat_req.user_id,
            session_id=chat_req.session_id,
            user_level=chat_req.user_level,
            conversation_history=chat_req.conversation_history,
        )

        # Save conversation to database
        conv_service = ConversationService(db)
        conv_service.save_conversation(
            user_id=chat_req.user_id,
            session_id=chat_req.session_id,
            query=chat_req.query,
            response=response["response"],
            agent_used=response["agent_used"],
            tone=chat_req.tone,
            user_level=chat_req.user_level,
            selected_text=chat_req.selected_text,
            sources=response.get("sources"),
        )

        logger.info(f"Chat request processed: user={chat_req.user_id}, latency={response['total_latency_ms']:.2f}ms")
        return response

    except ValidationError as e:
        logger.warning(f"Validation error in chat: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat request",
        )


@router.post("/index", tags=["Document Management"])
async def index_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Document indexing endpoint.

    Triggers re-indexing of textbook content into Qdrant vector database.

    Request body:
    ```json
    {
        "collection_name": "textbook_chapters",
        "force_reindex": false
    }
    ```

    Response:
    ```json
    {
        "status": "indexing_started",
        "indexed_chunks": 245,
        "storage_used_mb": 12.5
    }
    ```

    Args:
        request: Indexing request

    Returns:
        Indexing status

    Raises:
        HTTPException: If indexing fails
    """
    try:
        # TODO: In Phase 9, implement actual indexing pipeline
        return {
            "status": "not_implemented_yet",
            "message": "Indexing endpoint will be fully implemented in Phase 9",
        }
    except Exception as e:
        logger.error(f"Index endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to trigger indexing",
        )


@router.get("/search", tags=["Search"])
async def search_endpoint(
    query: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(5, ge=1, le=20),
    difficulty: Optional[str] = Query(None, regex="^(beginner|intermediate|advanced)$"),
) -> Dict[str, Any]:
    """
    Direct vector search endpoint for debugging.

    Performs semantic search in Qdrant without going through Answer Agent.

    Query parameters:
    - query: Search query string
    - top_k: Number of results (1-20, default 5)
    - difficulty: Filter by difficulty level (optional)

    Response:
    ```json
    {
        "query": "What is ROS?",
        "results": [
            {
                "chunk_id": "ch2_sec1_001",
                "text": "...",
                "chapter": "2",
                "section": "ROS Basics",
                "relevance_score": 0.95
            }
        ],
        "total_results": 1
    }
    ```

    Args:
        query: Search query
        top_k: Number of results
        difficulty: Optional difficulty filter

    Returns:
        Search results with scores

    Raises:
        HTTPException: If search fails
    """
    try:
        # TODO: In Phase 2, implement with QdrantService
        return {
            "status": "not_implemented_yet",
            "message": "Search endpoint will be implemented in Phase 2",
        }
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to search",
        )


@router.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns comprehensive health status of all system components.

    Response:
    ```json
    {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": "2025-12-20T12:34:56Z",
        "checks": {
            "api": "healthy",
            "database": "healthy",
            "qdrant": "healthy",
            "gemini": "healthy"
        }
    }
    ```

    Returns:
        Health status dictionary
    """
    try:
        health_status = {
            "status": "operational",
            "version": "0.1.0",
            "service": "rag-chatbot-api",
            "coordinator": coordinator_agent.get_agent_status(),
            "checks": {
                "api": "healthy",
                "database": "pending",  # Will check in Phase 2
                "qdrant": "pending",  # Will check in Phase 2
                "gemini": "pending",  # Will check in Phase 2
            },
        }
        return health_status
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }
