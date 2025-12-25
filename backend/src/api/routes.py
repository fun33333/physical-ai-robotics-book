"""
FastAPI routes for RAG Chatbot API.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.models.database import get_db
from src.services.orchestration_service import OrchestrationService, PipelineContext
from src.services.gemini_service import GeminiService
from src.agents.agents_config import get_key_count, get_current_key_index
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat endpoint with multi-turn conversation support."""

    query: str = Field(..., min_length=1, max_length=5000)
    selected_text: Optional[str] = Field(None, max_length=5000)
    tone: str = Field("english", pattern="^(english|roman_urdu|bro_guide)$")
    user_level: str = Field("intermediate", pattern="^(beginner|intermediate|advanced)$")
    conversation_history: List[Dict] = Field(default_factory=list)
    user_id: str = Field("anonymous", max_length=255)
    session_id: Optional[str] = Field(None, max_length=255)
    conversation_id: Optional[str] = Field(None, max_length=255, description="Alias for session_id for backward compatibility")


class ChatResponse(BaseModel):
    """Response model for chat endpoint with multi-turn conversation support."""

    response: str
    sources: List[Dict]
    tone: str
    session_id: str
    conversation_id: str = Field(description="Alias for session_id for client convenience")
    conversation_count: int = Field(default=0, description="Number of messages in this conversation")
    validation_status: str
    latency_breakdown: Dict[str, float]
    total_latency_ms: float


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Process chat query through multi-agent pipeline with multi-turn support.

    Accepts optional conversationId (or session_id) to maintain conversation context.
    If not provided, a new session is created.

    Args:
        request: ChatRequest with query, tone, and optional conversationId
        db: Database session for conversation history

    Returns:
        ChatResponse with response, sources, and conversationId for follow-up
    """
    from src.services.conversation_service import ConversationService

    try:
        # Handle conversationId/session_id interchangeably (T060)
        session_id = request.session_id or request.conversation_id or ""

        # Initialize services
        orchestrator = OrchestrationService(db_session=db)
        conversation_service = ConversationService(db_session=db)

        # Retrieve conversation history from database if session exists (T062)
        conversation_history = request.conversation_history
        conversation_count = 0

        if session_id and not conversation_history:
            # Fetch recent context from database
            conversation_history = conversation_service.get_recent_context(
                user_id=request.user_id,
                session_id=session_id,
                num_messages=5,  # Last 5 exchanges for context
            )
            logger.debug(f"Retrieved {len(conversation_history)} prior exchanges for session {session_id}")

        # Get conversation count for this session
        if session_id:
            try:
                history = conversation_service.get_conversation_history(
                    user_id=request.user_id,
                    session_id=session_id,
                    limit=100,
                )
                conversation_count = len(history)
            except Exception:
                conversation_count = 0

        # Build pipeline context
        context = PipelineContext(
            query=request.query,
            user_id=request.user_id,
            session_id=session_id,
            selected_text=request.selected_text,
            tone=request.tone,
            user_level=request.user_level,
            conversation_history=conversation_history,
        )

        # Process through multi-agent pipeline
        result = await orchestrator.process_chat(context)

        logger.info(f"Chat: {request.query[:50]}... -> {len(result.response)} chars")

        # Build response with conversationId (T063)
        response_dict = result.to_response()
        response_dict["conversation_id"] = response_dict["session_id"]
        response_dict["conversation_count"] = conversation_count + 1

        return response_dict

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index", tags=["Indexing"])
async def index_documents(force: bool = False) -> Dict[str, Any]:
    """Trigger document indexing (placeholder)."""
    return {
        "status": "pending",
        "message": "Indexing will be implemented in Phase 9",
        "force": force,
    }


@router.get("/search", tags=["Search"])
async def search(
    q: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(5, ge=1, le=20),
) -> Dict[str, Any]:
    """Direct vector search (placeholder)."""
    return {
        "query": q,
        "results": [],
        "message": "Search will be implemented with RAG Agent",
    }


@router.get("/health", tags=["Health"])
async def health(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Health check endpoint with quota reporting (T089)."""
    gemini_service = GeminiService(db_session=db)
    key_status = gemini_service.get_key_status()

    return {
        "status": "healthy",
        "version": "0.1.0",
        "agents": ["RAG", "Answer", "Tone", "Safety"],
        "gemini_keys": key_status["keys"],  # [active, active, exhausted]
        "api_rotations_today": key_status["api_rotations_today"],
        "last_rotation": key_status["last_rotation"],
        "all_checks": {
            "gemini": len([k for k in key_status["keys"] if k != "exhausted"]) > 0,
            "qdrant": True,  # TODO: Add real check in Phase 9
            "postgres": True,  # TODO: Add real check
        },
    }
