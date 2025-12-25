"""
Orchestration Service using OpenAI Agents SDK.

Manages the multi-agent pipeline: RAG -> Answer -> Tone -> Safety.
Includes conversation history logging after Safety Guardian approval.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.agents.rag_agent import retrieve_context
from src.agents.answer_tutor_agent import generate_answer
from src.agents.tone_agent import apply_tone
from src.agents.safety_guardian import validate_response
from src.services.conversation_service import ConversationService
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    """Context passed through the pipeline."""
    query: str
    user_id: str = "anonymous"
    session_id: str = ""
    selected_text: Optional[str] = None
    tone: str = "english"
    user_level: str = "intermediate"
    conversation_history: List[Dict] = field(default_factory=list)

    # Pipeline stage results
    retrieved_chunks: List[Dict] = field(default_factory=list)
    generated_response: str = ""
    toned_response: str = ""
    sources: List[Dict] = field(default_factory=list)
    latencies: Dict[str, float] = field(default_factory=dict)
    validation_status: str = "pending"
    is_hallucination: bool = False

    # Legacy compatibility
    chunks: List[Dict] = field(default_factory=list)
    response: str = ""

    def __post_init__(self):
        """Generate session_id if not provided."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())

    def add_latency(self, stage: str, latency_ms: float) -> None:
        """Add latency measurement for a pipeline stage."""
        self.latencies[stage] = latency_ms

    def total_latency_ms(self) -> float:
        """Calculate total latency across all stages."""
        return sum(self.latencies.values())

    def to_response_dict(self) -> Dict[str, Any]:
        """Convert context to API response dictionary."""
        return {
            "response": self.toned_response or self.generated_response or self.response,
            "sources": self.sources,
            "tone": self.tone,
            "session_id": self.session_id,
            "validation_status": self.validation_status,
            "is_hallucination": self.is_hallucination,
            "latency_breakdown": self.latencies,
            "total_latency_ms": self.total_latency_ms(),
        }

    def to_response(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.to_response_dict()


class OrchestrationService:
    """Orchestrates multi-agent pipeline with conversation logging."""

    # Maximum query length to prevent abuse
    MAX_QUERY_LENGTH = 5000

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize orchestration service.

        Args:
            db_session: SQLAlchemy session for conversation logging (optional)
        """
        self.db_session = db_session
        self.timeout_seconds = 2.0
        self.conversation_service = (
            ConversationService(db_session) if db_session else None
        )

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the status of the pipeline stages."""
        return {
            "status": "operational",
            "stages": {
                "coordinator": "ready",
                "rag": "ready",
                "answer": "ready",
                "tone": "ready",
                "safety_guardian": "ready",
            },
        }

    async def process_chat(self, context: PipelineContext) -> PipelineContext:
        """
        Process query through RAG -> Answer -> Tone -> Safety pipeline.

        After Safety Guardian approval, the conversation is logged to the database.

        Raises:
            ValueError: If the query is empty.
        """
        import time

        try:
            # Stage 0: Coordinator - Validate and prepare input
            coordinator_start = time.time()
            if not context.query or not context.query.strip():
                raise ValueError("Query cannot be empty")

            # Truncate long queries
            if len(context.query) > self.MAX_QUERY_LENGTH:
                context.query = context.query[:self.MAX_QUERY_LENGTH]

            context.add_latency("coordinator", (time.time() - coordinator_start) * 1000)

            # Stage 1: RAG - Retrieve chunks
            rag_result = await retrieve_context(context.query, context.selected_text)
            context.chunks = rag_result.get("chunks", [])
            context.retrieved_chunks = context.chunks  # Populate new attribute
            context.add_latency("rag", rag_result.get("latency_ms", 0))
            context.sources = [
                {"chapter": c.get("chapter"), "section": c.get("section")}
                for c in context.chunks
            ]
            logger.info(f"RAG: {len(context.chunks)} chunks")

            # Stage 2: Answer - Generate response
            answer_result = await generate_answer(
                context.query,
                context.chunks,
                context.conversation_history,
                context.user_level,
                context.selected_text,
            )
            context.response = answer_result.get("response", "")
            context.generated_response = context.response  # Populate new attribute
            context.add_latency("answer", answer_result.get("latency_ms", 0))
            logger.info(f"Answer: {len(context.response)} chars")

            # Stage 3: Tone - Transform style
            tone_result = await apply_tone(context.response, context.tone)
            context.response = tone_result.get("response", context.response)
            context.toned_response = context.response  # Populate new attribute
            context.add_latency("tone", tone_result.get("latency_ms", 0))
            logger.info(f"Tone: {context.tone}")

            # Stage 4: Safety - Validate
            safety_result = await validate_response(
                context.response, context.query, context.chunks
            )
            context.validation_status = safety_result.get("validation_status", "approved")
            context.is_hallucination = context.validation_status == "flagged"
            if context.is_hallucination:
                context.response += "\n\n*Note: Please verify this with the textbook.*"
                context.toned_response = context.response
            context.add_latency("safety", safety_result.get("latency_ms", 0))
            logger.info(f"Safety: {context.validation_status}")

            # Log conversation after Safety Guardian approval
            await self._log_conversation(context)

            return context

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            context.response = "Sorry, I encountered an error. Please try again."
            context.toned_response = context.response
            return context

    async def _log_conversation(self, context: PipelineContext) -> None:
        """
        Log conversation to database after Safety Guardian approval.

        Only logs if a database session is available and the response
        was generated successfully.

        Args:
            context: Pipeline context with query and response
        """
        if not self.conversation_service:
            logger.debug("No database session, skipping conversation logging")
            return

        if not context.response:
            logger.debug("Empty response, skipping conversation logging")
            return

        try:
            self.conversation_service.save_conversation(
                user_id=context.user_id,
                session_id=context.session_id,
                query=context.query,
                response=context.response,
                agent_used="orchestrator",
                tone=context.tone,
                user_level=context.user_level,
                selected_text=context.selected_text,
                sources=context.sources,
            )
            logger.info(
                f"Logged conversation: user={context.user_id}, "
                f"session={context.session_id}, validation={context.validation_status}"
            )
        except Exception as e:
            logger.warning(f"Failed to log conversation: {e}")
            # Don't fail the request if logging fails
