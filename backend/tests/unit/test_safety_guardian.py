"""
Unit tests for Safety Guardian Agent (Phase 3 - T044a).

Tests the Safety Guardian's hallucination detection capabilities:
- Detecting unsourced claims
- Approving valid responses
- Parsing validation results
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.safety_guardian import validate_response, safety_guardian


class TestSafetyGuardianAgent:
    """Tests for the Safety Guardian agent configuration."""

    def test_agent_exists(self):
        """Test that the safety_guardian agent is properly defined."""
        assert safety_guardian is not None
        assert safety_guardian.name == "Safety Guardian"

    def test_agent_has_instructions(self):
        """Test that agent has validation instructions."""
        assert safety_guardian.instructions is not None
        assert len(safety_guardian.instructions) > 0

    def test_agent_instructions_include_validation_rules(self):
        """Test that instructions mention key validation concepts."""
        instructions = safety_guardian.instructions.lower()

        # Should mention validation/checking
        assert "validate" in instructions or "check" in instructions

        # Should mention sources
        assert "source" in instructions

        # Should mention claims
        assert "claim" in instructions or "support" in instructions

    def test_agent_returns_json_format(self):
        """Test that agent is instructed to return JSON format."""
        instructions = safety_guardian.instructions

        assert "JSON" in instructions or "json" in instructions
        assert "status" in instructions


class TestValidateResponse:
    """Tests for the validate_response function."""

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks with factual content for testing."""
        return [
            {
                "chunk_id": "chunk_001",
                "text": "ROS 2 (Robot Operating System 2) was released in December 2017.",
                "chapter": "Module 1",
                "section": "History",
            },
            {
                "chunk_id": "chunk_002",
                "text": "ROS 2 uses DDS (Data Distribution Service) for communication.",
                "chapter": "Module 1",
                "section": "Architecture",
            },
        ]

    @pytest.mark.asyncio
    async def test_approves_valid_response(self, sample_chunks):
        """Test that a valid response using source info is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 was released in December 2017.",
                query="When was ROS 2 released?",
                chunks=sample_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_flags_hallucinated_response(self, sample_chunks):
        """Test that a hallucinated response is flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["ROS 3 not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 3 is coming soon.",
                query="What is the latest ROS version?",
                chunks=sample_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_includes_latency_tracking(self, sample_chunks):
        """Test that latency is tracked for validation."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved"}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Test response",
                query="Test query",
                chunks=sample_chunks,
            )

            assert "latency_ms" in result
            assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_returns_original_response(self, sample_chunks):
        """Test that the original response is included in result."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved"}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            original_response = "This is the original response about ROS."

            result = await validate_response(
                response=original_response,
                query="Test query",
                chunks=sample_chunks,
            )

            assert result["response"] == original_response

    @pytest.mark.asyncio
    async def test_handles_empty_chunks(self):
        """Test validation with empty chunks list."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["No sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Some response",
                query="Test query",
                chunks=[],
            )

            assert "validation_status" in result

    @pytest.mark.asyncio
    async def test_handles_runner_error(self, sample_chunks):
        """Test graceful handling of runner errors."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_runner.run = AsyncMock(side_effect=Exception("API Error"))

            result = await validate_response(
                response="Test response",
                query="Test query",
                chunks=sample_chunks,
            )

            # Should fail-open (approve on error)
            assert result["validation_status"] == "approved"
            assert "error" in result


class TestHallucinationDetection:
    """Tests for specific hallucination detection scenarios."""

    @pytest.fixture
    def source_chunks(self):
        """Chunks with specific facts for testing."""
        return [
            {
                "chunk_id": "c1",
                "text": "SLAM allows robots to build maps while tracking their position.",
                "chapter": "Module 2",
                "section": "Navigation",
            },
            {
                "chunk_id": "c2",
                "text": "Common SLAM algorithms include EKF-SLAM and FastSLAM.",
                "chapter": "Module 2",
                "section": "SLAM Algorithms",
            },
        ]

    @pytest.mark.asyncio
    async def test_detects_invented_algorithm(self, source_chunks):
        """Test that invented algorithm names are flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["SuperSLAM not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="The best algorithm is SuperSLAM 3000.",
                query="What is the best SLAM algorithm?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_detects_false_dates(self, source_chunks):
        """Test that false dates are flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Date not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="SLAM was invented in 1950.",
                query="When was SLAM invented?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_detects_invented_statistics(self, source_chunks):
        """Test that invented statistics are flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["99% statistic not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="SLAM is used by 99% of robots worldwide.",
                query="How popular is SLAM?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_approves_accurate_paraphrase(self, source_chunks):
        """Test that accurate paraphrasing is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="SLAM enables robots to simultaneously map their environment and track where they are.",
                query="What does SLAM do?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_approves_honest_uncertainty(self, source_chunks):
        """Test that honest admission of uncertainty is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="I don't find information about this in the textbook.",
                query="What is quantum SLAM?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "approved"


class TestResultParsing:
    """Tests for parsing validation results."""

    @pytest.fixture
    def basic_chunks(self):
        return [{"chunk_id": "c1", "text": "Sample text"}]

    @pytest.mark.asyncio
    async def test_parses_approved_status(self, basic_chunks):
        """Test parsing approved status from various formats."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "The response is approved and factually correct."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Test",
                query="Test",
                chunks=basic_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_parses_flagged_status(self, basic_chunks):
        """Test parsing flagged status from various formats."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "FLAGGED: Contains unsupported claims"
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Test",
                query="Test",
                chunks=basic_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_handles_empty_output(self, basic_chunks):
        """Test handling empty output from agent."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = ""
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Test",
                query="Test",
                chunks=basic_chunks,
            )

            # Should default to flagged when unsure
            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_handles_none_output(self, basic_chunks):
        """Test handling None output from agent."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = None
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Test",
                query="Test",
                chunks=basic_chunks,
            )

            # Should default to flagged when unsure
            assert result["validation_status"] == "flagged"