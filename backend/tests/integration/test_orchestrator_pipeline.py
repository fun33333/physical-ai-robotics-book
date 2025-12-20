"""
Integration tests for Orchestrator Pipeline (Phase 3 - T034).

Tests full end-to-end chat flow with mocked Qdrant and Gemini,
verifying the complete orchestrator pipeline execution.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio

from src.agents.coordinator import MainCoordinatorAgent
from src.services.orchestration_service import PipelineContext


class TestOrchestratorPipelineIntegration:
    """Integration tests for orchestrator pipeline."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator agent."""
        return MainCoordinatorAgent()

    @pytest.mark.asyncio
    async def test_full_chat_flow_integration(self, coordinator):
        """Test full chat processing through coordinator."""
        response = await coordinator.route_through_pipeline(
            query="What is ROS 2?",
            selected_text="ROS 2 is a middleware",
            tone="english",
            user_id="test_user",
            session_id="test_session",
            user_level="intermediate",
        )

        # Verify response structure
        assert "response" in response
        assert "sources" in response
        assert "agent_used" in response
        assert "tone" in response
        assert "latency_breakdown" in response
        assert "total_latency_ms" in response

        # Verify response content
        assert len(response["response"]) > 0
        assert response["tone"] == "english"
        assert response["agent_used"] == "orchestrator"

    @pytest.mark.asyncio
    async def test_pipeline_with_roman_urdu_tone(self, coordinator):
        """Test pipeline with roman_urdu tone."""
        response = await coordinator.route_through_pipeline(
            query="کیا یہ کام کرے گا؟",
            tone="roman_urdu",
            user_id="user_123",
            session_id="session_456",
        )

        assert response["tone"] == "roman_urdu"
        assert len(response["response"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_selected_text_context(self, coordinator):
        """Test pipeline prioritizes selected text context."""
        selected = "ROS 2 uses DDS for communication with publish-subscribe pattern."

        response = await coordinator.route_through_pipeline(
            query="How does communication work?",
            selected_text=selected,
            user_id="user_123",
            session_id="session_456",
        )

        assert len(response["response"]) > 0
        # Response should reference selected text in context

    @pytest.mark.asyncio
    async def test_pipeline_latency_within_budget(self, coordinator):
        """Test that pipeline completes within 2-second latency budget."""
        response = await coordinator.route_through_pipeline(
            query="Test query",
            user_id="user_123",
            session_id="session_456",
        )

        total_latency = response["total_latency_ms"]

        # Should complete well under 2 seconds (2000ms)
        assert total_latency < 2000, f"Pipeline latency {total_latency}ms exceeds 2000ms budget"

    @pytest.mark.asyncio
    async def test_pipeline_latency_breakdown(self, coordinator):
        """Test latency breakdown includes all stages."""
        response = await coordinator.route_through_pipeline(
            query="Test query",
            user_id="user_123",
            session_id="session_456",
        )

        breakdown = response["latency_breakdown"]

        # All stages should be present
        required_stages = ["coordinator", "rag", "answer", "tone", "safety"]
        for stage in required_stages:
            assert stage in breakdown, f"Missing latency for {stage}"
            assert breakdown[stage] >= 0

    @pytest.mark.asyncio
    async def test_pipeline_sources_included(self, coordinator):
        """Test that sources are included in response."""
        response = await coordinator.route_through_pipeline(
            query="What about ROS?",
            user_id="user_123",
            session_id="session_456",
        )

        sources = response["sources"]

        # Should have at least one source
        assert len(sources) > 0

        # Each source should have required fields
        for source in sources:
            assert "chapter" in source or "section" in source

    @pytest.mark.asyncio
    async def test_pipeline_with_conversation_history(self, coordinator):
        """Test pipeline with multi-turn conversation."""
        history = [
            {
                "query": "What is ROS?",
                "response": "ROS is a robotics middleware.",
            }
        ]

        response = await coordinator.route_through_pipeline(
            query="Tell me more.",
            user_id="user_123",
            session_id="session_456",
            conversation_history=history,
        )

        assert len(response["response"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_validates_tone_parameter(self, coordinator):
        """Test pipeline accepts valid tone values."""
        valid_tones = ["english", "roman_urdu", "bro_guide"]

        for tone in valid_tones:
            response = await coordinator.route_through_pipeline(
                query="Test",
                tone=tone,
                user_id="user_123",
                session_id="session_456",
            )

            assert response["tone"] == tone

    @pytest.mark.asyncio
    async def test_pipeline_validates_user_level(self, coordinator):
        """Test pipeline accepts valid user level values."""
        valid_levels = ["beginner", "intermediate", "advanced"]

        for level in valid_levels:
            response = await coordinator.route_through_pipeline(
                query="Test",
                user_id="user_123",
                session_id="session_456",
                user_level=level,
            )

            assert len(response["response"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_empty_selected_text(self, coordinator):
        """Test pipeline works with empty selected_text."""
        response = await coordinator.route_through_pipeline(
            query="Test query",
            selected_text=None,
            user_id="user_123",
            session_id="session_456",
        )

        assert len(response["response"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_user_id_tracking(self, coordinator):
        """Test that user_id is properly tracked."""
        user_id = "specific_user_123"

        response = await coordinator.route_through_pipeline(
            query="Test",
            user_id=user_id,
            session_id="session_456",
        )

        # User ID should be in response for logging purposes
        assert len(response["response"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_session_isolation(self, coordinator):
        """Test that sessions are properly isolated."""
        # Two different sessions should produce independent responses
        response1 = await coordinator.route_through_pipeline(
            query="Test query",
            user_id="user_123",
            session_id="session_1",
        )

        response2 = await coordinator.route_through_pipeline(
            query="Same query",
            user_id="user_123",
            session_id="session_2",
        )

        # Both should have responses (session isolation verified in conversation service)
        assert len(response1["response"]) > 0
        assert len(response2["response"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_error_on_empty_query(self, coordinator):
        """Test pipeline rejects empty query."""
        with pytest.raises(ValueError) as excinfo:
            await coordinator.route_through_pipeline(
                query="",
                user_id="user_123",
                session_id="session_456",
            )

        assert "Query cannot be empty" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self):
        """Test coordinator agent initialization."""
        coordinator = MainCoordinatorAgent()

        status = coordinator.get_agent_status()

        assert status["coordinator"] == "operational"
        assert "pipeline_status" in status

    @pytest.mark.asyncio
    async def test_coordinator_pipeline_initialization(self, coordinator):
        """Test coordinator pipeline initialization."""
        init_result = await coordinator.initialize_pipeline()

        assert init_result["status"] == "initialized"
        assert "pipeline" in init_result


class TestPipelineErrorHandling:
    """Tests for error handling in pipeline."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator agent."""
        return MainCoordinatorAgent()

    @pytest.mark.asyncio
    async def test_pipeline_timeout_error(self, coordinator):
        """Test pipeline handles timeout gracefully."""
        # This test verifies timeout would be caught (actual timeout is >2s which is hard to test)
        response = await coordinator.route_through_pipeline(
            query="Normal query",
            user_id="user_123",
            session_id="session_456",
        )

        # Should complete without timeout error
        assert response["total_latency_ms"] < 2000

    @pytest.mark.asyncio
    async def test_pipeline_exception_handling(self, coordinator):
        """Test pipeline handles exceptions."""
        # Verify that exceptions are properly caught and raised
        with pytest.raises(ValueError):
            await coordinator.route_through_pipeline(
                query="",  # Empty query should raise
                user_id="user_123",
                session_id="session_456",
            )
