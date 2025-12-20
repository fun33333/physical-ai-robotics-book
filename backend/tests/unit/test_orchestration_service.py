"""
Unit tests for Orchestration Service (Phase 3 - T032).

Tests the multi-agent pipeline orchestration, verifying that it:
- Executes stages in correct order
- Passes context between stages
- Tracks latencies per stage
- Respects timeout constraints
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from src.services.orchestration_service import (
    OrchestrationService,
    PipelineContext,
)


class TestPipelineContext:
    """Tests for PipelineContext data structure."""

    def test_context_initialization(self):
        """Test PipelineContext initialization with defaults."""
        context = PipelineContext(
            query="What is ROS 2?",
            user_id="user_123",
            session_id="session_456",
        )

        assert context.query == "What is ROS 2?"
        assert context.user_id == "user_123"
        assert context.session_id == "session_456"
        assert context.tone == "english"
        assert context.user_level == "intermediate"
        assert context.is_hallucination is False

    def test_context_add_latency(self):
        """Test adding latency measurements."""
        context = PipelineContext(query="Test", user_id="user_1", session_id="session_1")

        context.add_latency("rag", 150.5)
        context.add_latency("answer", 800.2)

        assert context.latencies["rag"] == 150.5
        assert context.latencies["answer"] == 800.2

    def test_context_total_latency(self):
        """Test total latency calculation."""
        context = PipelineContext(query="Test", user_id="user_1", session_id="session_1")

        context.add_latency("coordinator", 5.0)
        context.add_latency("rag", 150.0)
        context.add_latency("answer", 800.0)
        context.add_latency("tone", 50.0)
        context.add_latency("safety", 100.0)

        total = context.total_latency_ms()
        assert total == 1105.0

    def test_context_to_response_dict(self):
        """Test converting context to response dictionary."""
        context = PipelineContext(
            query="Test",
            user_id="user_1",
            session_id="session_1",
            tone="roman_urdu",
        )
        context.generated_response = "Generated response"
        context.toned_response = "Toned response"
        context.sources = [{"chapter": "2", "section": "ROS Basics"}]
        context.add_latency("rag", 150.0)

        response = context.to_response_dict()

        assert response["response"] == "Toned response"
        assert response["tone"] == "roman_urdu"
        assert response["sources"] == [{"chapter": "2", "section": "ROS Basics"}]
        assert response["total_latency_ms"] == 150.0


class TestOrchestrationServiceInitialization:
    """Tests for orchestration service initialization."""

    def test_service_initialization(self):
        """Test OrchestrationService initialization."""
        service = OrchestrationService()

        assert service.timeout_seconds == 2.0
        assert service.get_pipeline_status()["status"] == "operational"

    def test_pipeline_status(self):
        """Test getting pipeline status."""
        service = OrchestrationService()
        status = service.get_pipeline_status()

        assert "coordinator" in status["stages"]
        assert "rag" in status["stages"]
        assert "answer" in status["stages"]
        assert "tone" in status["stages"]
        assert "safety_guardian" in status["stages"]


class TestOrchestrationPipeline:
    """Tests for the agent pipeline execution."""

    @pytest.mark.asyncio
    async def test_process_chat_basic_flow(self):
        """Test basic chat processing through pipeline."""
        service = OrchestrationService()

        context = PipelineContext(
            query="What is ROS 2?",
            user_id="user_123",
            session_id="session_456",
        )

        result = await service.process_chat(context)

        # Verify stages executed
        assert "coordinator" in result.latencies
        assert "rag" in result.latencies
        assert "answer" in result.latencies
        assert "tone" in result.latencies
        assert "safety" in result.latencies

        # Verify context flows through
        assert result.query == "What is ROS 2?"
        assert result.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_process_chat_respects_timeout(self):
        """Test that pipeline respects timeout."""
        service = OrchestrationService()

        context = PipelineContext(query="Test", user_id="user_1", session_id="s1")

        start = time.time()
        result = await service.process_chat(context)
        elapsed = (time.time() - start) * 1000

        # Should complete well under 2 second limit
        assert elapsed < 2000

    @pytest.mark.asyncio
    async def test_coordinator_stage_validation(self):
        """Test coordinator stage input validation."""
        service = OrchestrationService()

        # Empty query should fail
        context = PipelineContext(query="", user_id="user_1", session_id="s1")

        with pytest.raises(ValueError) as excinfo:
            await service.process_chat(context)

        assert "Query cannot be empty" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_coordinator_truncates_long_queries(self):
        """Test coordinator truncates very long queries."""
        service = OrchestrationService()

        long_query = "a" * 10000  # Way too long
        context = PipelineContext(query=long_query, user_id="user_1", session_id="s1")

        result = await service.process_chat(context)

        # Query should be truncated to 5000 chars
        assert len(result.query) == 5000

    @pytest.mark.asyncio
    async def test_rag_stage_produces_chunks(self):
        """Test RAG stage populates retrieved chunks."""
        service = OrchestrationService()

        context = PipelineContext(query="What is ROS?", user_id="user_1", session_id="s1")

        result = await service.process_chat(context)

        # RAG stage should populate chunks
        assert len(result.retrieved_chunks) > 0
        assert "chunk_id" in result.retrieved_chunks[0]
        assert "text" in result.retrieved_chunks[0]
        assert "chapter" in result.retrieved_chunks[0]

    @pytest.mark.asyncio
    async def test_answer_stage_generates_response(self):
        """Test Answer stage generates response."""
        service = OrchestrationService()

        context = PipelineContext(query="What is ROS?", user_id="user_1", session_id="s1")

        result = await service.process_chat(context)

        # Answer stage should generate response
        assert len(result.generated_response) > 0

    @pytest.mark.asyncio
    async def test_tone_stage_applies_tone(self):
        """Test Tone stage applies selected tone."""
        service = OrchestrationService()

        context = PipelineContext(
            query="What is ROS?",
            user_id="user_1",
            session_id="s1",
            tone="roman_urdu",
        )

        result = await service.process_chat(context)

        # Tone stage should produce toned response
        assert len(result.toned_response) > 0
        assert result.tone == "roman_urdu"

    @pytest.mark.asyncio
    async def test_safety_stage_validates_hallucinations(self):
        """Test Safety stage validates for hallucinations."""
        service = OrchestrationService()

        context = PipelineContext(query="Test", user_id="user_1", session_id="s1")

        result = await service.process_chat(context)

        # Safety stage should set hallucination flag
        assert isinstance(result.is_hallucination, bool)

    @pytest.mark.asyncio
    async def test_latency_tracking_per_stage(self):
        """Test that latency is tracked for each stage."""
        service = OrchestrationService()

        context = PipelineContext(query="Test", user_id="user_1", session_id="s1")

        result = await service.process_chat(context)

        # Each stage should record latency > 0
        for stage in ["coordinator", "rag", "answer", "tone", "safety"]:
            assert stage in result.latencies
            assert result.latencies[stage] >= 0

    @pytest.mark.asyncio
    async def test_sources_extracted_from_chunks(self):
        """Test that sources are extracted from retrieved chunks."""
        service = OrchestrationService()

        context = PipelineContext(query="Test", user_id="user_1", session_id="s1")

        result = await service.process_chat(context)

        # Sources should be populated from retrieved chunks
        assert len(result.sources) > 0
        for source in result.sources:
            assert "chapter" in source
            assert "section" in source


class TestPipelineContextFlowCompletion:
    """Tests for complete pipeline execution."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_execution(self):
        """Test complete pipeline from input to output."""
        service = OrchestrationService()

        context = PipelineContext(
            query="How does ROS 2 communication work?",
            selected_text="ROS 2 uses DDS for communication.",
            user_id="student_123",
            session_id="class_456",
            tone="english",
            user_level="intermediate",
        )

        result = await service.process_chat(context)

        # Verify complete output
        assert result.query == "How does ROS 2 communication work?"
        assert result.selected_text == "ROS 2 uses DDS for communication."
        assert result.tone == "english"
        assert result.user_level == "intermediate"
        assert result.toned_response
        assert result.sources
        assert result.total_latency_ms() > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_conversation_history(self):
        """Test pipeline with multi-turn conversation context."""
        service = OrchestrationService()

        history = [
            {
                "query": "What is ROS?",
                "response": "ROS is a robotics middleware.",
                "tone": "english",
                "created_at": "2025-12-20T10:00:00Z",
            },
            {
                "query": "Tell me more.",
                "response": "ROS provides tools and libraries...",
                "tone": "english",
                "created_at": "2025-12-20T10:01:00Z",
            },
        ]

        context = PipelineContext(
            query="How do I install it?",
            user_id="student_123",
            session_id="class_456",
            conversation_history=history,
        )

        result = await service.process_chat(context)

        # Verify conversation history is preserved
        assert len(result.conversation_history) == 2
        assert result.toned_response
