"""
Orchestration Service.

Manages the multi-agent pipeline orchestration, routing queries through
specialized agents in sequence: Coordinator → RAG → Answer → Tone → Safety.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    """Context object passed through the agent pipeline."""

    # Input
    query: str
    selected_text: Optional[str] = None
    user_id: str = "anonymous"
    session_id: str = ""
    tone: str = "english"
    user_level: str = "intermediate"
    conversation_history: List[Dict] = field(default_factory=list)

    # Pipeline stages (populated as we progress)
    retrieved_chunks: List[Dict] = field(default_factory=list)
    generated_response: str = ""
    toned_response: str = ""
    is_hallucination: bool = False
    hallucination_reason: Optional[str] = None

    # Metadata
    latencies: Dict[str, float] = field(default_factory=dict)
    agent_used: str = "orchestrator"
    sources: List[Dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def add_latency(self, stage_name: str, duration_ms: float) -> None:
        """Record latency for a pipeline stage."""
        self.latencies[stage_name] = duration_ms

    def total_latency_ms(self) -> float:
        """Get total pipeline latency in milliseconds."""
        return sum(self.latencies.values())

    def to_response_dict(self) -> Dict[str, Any]:
        """Convert context to API response dictionary."""
        return {
            "response": self.toned_response,
            "sources": self.sources,
            "agent_used": self.agent_used,
            "tone": self.tone,
            "latency_breakdown": self.latencies,
            "total_latency_ms": self.total_latency_ms(),
        }


class OrchestrationService:
    """Service for orchestrating multi-agent pipeline."""

    def __init__(self):
        """Initialize orchestration service."""
        logger.info("Orchestration service initialized")
        self.timeout_seconds = 2.0  # 2s total timeout per constitution NFR-001

    async def process_chat(self, context: PipelineContext) -> PipelineContext:
        """
        Process user query through the full orchestrator pipeline.

        Pipeline stages (in order):
        1. Coordinator: Route and validate input
        2. RAG Agent: Retrieve relevant context
        3. Answer/Tutor Agent: Generate response
        4. Tone Agent: Format to preferred tone
        5. Safety Guardian: Validate for hallucinations

        Args:
            context: Pipeline context with user input

        Returns:
            Updated context with generated response

        Raises:
            TimeoutError: If pipeline exceeds 2s timeout
        """
        start_time = time.time()

        try:
            # Stage 1: Coordinator
            context = await self._coordinator_stage(context)
            self._log_stage_complete("Coordinator", context)

            # Stage 2: RAG Agent
            context = await self._rag_stage(context)
            self._log_stage_complete("RAG", context)

            # Stage 3: Answer/Tutor Agent
            context = await self._answer_stage(context)
            self._log_stage_complete("Answer", context)

            # Stage 4: Tone Agent
            context = await self._tone_stage(context)
            self._log_stage_complete("Tone", context)

            # Stage 5: Safety Guardian
            context = await self._safety_stage(context)
            self._log_stage_complete("Safety", context)

            # Final check
            elapsed = (time.time() - start_time) * 1000
            if elapsed > self.timeout_seconds * 1000:
                logger.warning(f"Pipeline exceeded timeout: {elapsed:.0f}ms > {self.timeout_seconds * 1000}ms")

            return context

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            raise

    async def _coordinator_stage(self, context: PipelineContext) -> PipelineContext:
        """
        Coordinator stage: Route and validate input.

        Args:
            context: Pipeline context

        Returns:
            Updated context
        """
        stage_start = time.time()

        # Validate input
        if not context.query:
            raise ValueError("Query cannot be empty")

        if len(context.query) > 5000:
            context.query = context.query[:5000]  # Truncate long queries
            logger.warning("Query truncated to 5000 characters")

        # Set agent used
        context.agent_used = "orchestrator"

        # Record latency (minimal for coordinator)
        latency_ms = (time.time() - stage_start) * 1000
        context.add_latency("coordinator", latency_ms)

        logger.debug(f"Coordinator stage: query={context.query[:50]}...")
        return context

    async def _rag_stage(self, context: PipelineContext) -> PipelineContext:
        """
        RAG stage: Retrieve relevant document chunks.

        NOTE: In Phase 3, this will call RAGAgent with actual Qdrant search.
        For now, returns placeholder.

        Args:
            context: Pipeline context

        Returns:
            Updated context with retrieved_chunks
        """
        stage_start = time.time()

        # TODO: In Phase 3, call RAGAgent.retrieve(context.query, context.selected_text)
        context.retrieved_chunks = [
            {
                "chunk_id": "placeholder_1",
                "text": "This is a placeholder chunk about ROS 2.",
                "chapter": "Chapter 2",
                "section": "ROS 2 Basics",
                "relevance_score": 0.95,
            },
        ]

        # Extract sources from retrieved chunks
        context.sources = [
            {
                "chapter": chunk.get("chapter"),
                "section": chunk.get("section"),
                "relevance_score": chunk.get("relevance_score"),
            }
            for chunk in context.retrieved_chunks
        ]

        latency_ms = (time.time() - stage_start) * 1000
        context.add_latency("rag", latency_ms)

        logger.debug(f"RAG stage: retrieved {len(context.retrieved_chunks)} chunks")
        return context

    async def _answer_stage(self, context: PipelineContext) -> PipelineContext:
        """
        Answer/Tutor stage: Generate response from retrieved context.

        NOTE: In Phase 3, this will call AnswerTutorAgent with Gemini.
        For now, returns placeholder.

        Args:
            context: Pipeline context

        Returns:
            Updated context with generated_response
        """
        stage_start = time.time()

        # TODO: In Phase 3, call AnswerTutorAgent.generate(context)
        context.generated_response = "This is a placeholder response about the retrieved content."

        latency_ms = (time.time() - stage_start) * 1000
        context.add_latency("answer", latency_ms)

        logger.debug(f"Answer stage: generated {len(context.generated_response)} chars")
        return context

    async def _tone_stage(self, context: PipelineContext) -> PipelineContext:
        """
        Tone stage: Format response to preferred tone.

        NOTE: In Phase 4, this will call ToneAgent.apply_tone().
        For now, returns generated_response as-is.

        Args:
            context: Pipeline context

        Returns:
            Updated context with toned_response
        """
        stage_start = time.time()

        # TODO: In Phase 4, call ToneAgent.apply_tone(context.generated_response, context.tone)
        context.toned_response = context.generated_response

        latency_ms = (time.time() - stage_start) * 1000
        context.add_latency("tone", latency_ms)

        logger.debug(f"Tone stage: applied tone={context.tone}")
        return context

    async def _safety_stage(self, context: PipelineContext) -> PipelineContext:
        """
        Safety Guardian stage: Validate for hallucinations.

        NOTE: In Phase 7, this will call SafetyGuardianAgent.validate().
        For now, skips validation.

        Args:
            context: Pipeline context

        Returns:
            Updated context with hallucination detection
        """
        stage_start = time.time()

        # TODO: In Phase 7, call SafetyGuardianAgent.validate(context)
        context.is_hallucination = False

        latency_ms = (time.time() - stage_start) * 1000
        context.add_latency("safety", latency_ms)

        logger.debug(f"Safety stage: is_hallucination={context.is_hallucination}")
        return context

    def _log_stage_complete(self, stage_name: str, context: PipelineContext) -> None:
        """Log completion of a pipeline stage."""
        latency = context.latencies.get(stage_name, 0)
        logger.debug(f"Stage complete: {stage_name} ({latency:.2f}ms)")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get overall pipeline status for health check.

        Returns:
            Dictionary with pipeline status
        """
        return {
            "status": "operational",
            "timeout_seconds": self.timeout_seconds,
            "stages": [
                "coordinator",
                "rag",
                "answer",
                "tone",
                "safety_guardian",
            ],
        }
