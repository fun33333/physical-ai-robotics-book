"""
Integration tests for Safety Guardian hallucination detection (Phase 3 - T035).

Tests the Safety Guardian's ability to:
- Detect hallucinations (claims not supported by source chunks)
- Approve valid responses
- Rewrite or flag responses with unsupported claims
- Add transparency notes when needed
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.safety_guardian import validate_response, safety_guardian


class TestHallucinationDetection:
    """Tests for hallucination detection functionality."""

    @pytest.fixture
    def source_chunks(self):
        """Source chunks with specific factual content."""
        return [
            {
                "chunk_id": "chunk_001",
                "text": "ROS 2 was released in December 2017. It replaced ROS 1 as the main robotics framework.",
                "chapter": "Module 1",
                "section": "History",
            },
            {
                "chunk_id": "chunk_002",
                "text": "ROS 2 uses DDS (Data Distribution Service) for communication between nodes. DDS provides real-time, reliable messaging.",
                "chapter": "Module 1",
                "section": "Architecture",
            },
        ]

    @pytest.mark.asyncio
    async def test_approves_valid_response(self, source_chunks):
        """Test that a response using only source information is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Response that only uses information from chunks
            valid_response = "ROS 2 was released in December 2017 and uses DDS for communication."

            result = await validate_response(
                response=valid_response,
                query="When was ROS 2 released?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_flags_false_claim(self, source_chunks):
        """Test that a response with false claims is flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["ROS 2 was released in 2015 is not supported by sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Response with incorrect date (sources say 2017, not 2015)
            hallucinated_response = "ROS 2 was released in 2015 by Google."

            result = await validate_response(
                response=hallucinated_response,
                query="When was ROS 2 released?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_flags_extrapolation(self, source_chunks):
        """Test that extrapolated information not in sources is flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["No source mentions Python support"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Response with information not in sources
            extrapolated_response = "ROS 2 was released in 2017 and supports Python 3.8+ exclusively."

            result = await validate_response(
                response=extrapolated_response,
                query="What languages does ROS 2 support?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_flags_external_knowledge(self, source_chunks):
        """Test that responses using external knowledge are flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Information about ROS 3 not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Response with external knowledge not in sources
            external_knowledge_response = "ROS 2 uses DDS, and ROS 3 is currently in development at Open Robotics."

            result = await validate_response(
                response=external_knowledge_response,
                query="What communication does ROS use?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_flags_invented_statistics(self, source_chunks):
        """Test that invented statistics are flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["95% statistic not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Response with invented statistics
            invented_stats_response = "ROS 2 is used by 95% of robotics companies worldwide."

            result = await validate_response(
                response=invented_stats_response,
                query="How popular is ROS 2?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_approves_partial_paraphrase(self, source_chunks):
        """Test that paraphrased source content is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Paraphrased response
            paraphrased_response = "The second version of ROS came out at the end of 2017, featuring DDS-based messaging."

            result = await validate_response(
                response=paraphrased_response,
                query="When was ROS 2 released?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "approved"


class TestSafetyGuardianAgent:
    """Tests for Safety Guardian agent configuration."""

    def test_agent_has_validation_instructions(self):
        """Test that agent instructions include validation rules."""
        instructions = safety_guardian.instructions.lower()

        assert "validate" in instructions or "check" in instructions
        assert "source" in instructions
        assert "claim" in instructions or "supported" in instructions

    def test_agent_returns_json_format(self):
        """Test that agent is instructed to return JSON format."""
        instructions = safety_guardian.instructions

        assert "JSON" in instructions or "json" in instructions
        assert "status" in instructions
        assert "approved" in instructions or "flagged" in instructions


class TestLatencyTracking:
    """Tests for latency tracking in validation."""

    @pytest.mark.asyncio
    async def test_tracks_validation_latency(self):
        """Test that validation latency is tracked."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved"}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            chunks = [{"text": "Sample text", "chunk_id": "c1"}]

            result = await validate_response(
                response="Test response",
                query="Test query",
                chunks=chunks,
            )

            assert "latency_ms" in result
            assert result["latency_ms"] >= 0


class TestErrorHandling:
    """Tests for error handling in validation."""

    @pytest.mark.asyncio
    async def test_handles_runner_error_gracefully(self):
        """Test that errors in runner are handled gracefully."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_runner.run = AsyncMock(side_effect=Exception("API Error"))

            chunks = [{"text": "Sample text", "chunk_id": "c1"}]

            result = await validate_response(
                response="Test response",
                query="Test query",
                chunks=chunks,
            )

            # Should default to approved on error (fail-open)
            assert result["validation_status"] == "approved"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_handles_empty_chunks(self):
        """Test behavior with empty chunks list."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["No sources to validate against"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Test response",
                query="Test query",
                chunks=[],
            )

            assert "validation_status" in result


class TestValidationScenarios:
    """Integration tests for various hallucination scenarios."""

    @pytest.fixture
    def robotics_chunks(self):
        """Chunks about robotics topics."""
        return [
            {
                "chunk_id": "nav_001",
                "text": "SLAM (Simultaneous Localization and Mapping) allows robots to build maps while tracking their position. Common algorithms include EKF-SLAM and FastSLAM.",
                "chapter": "Module 2",
                "section": "Navigation",
            },
            {
                "chunk_id": "sensor_001",
                "text": "LIDAR sensors emit laser beams to measure distances. Common LIDAR units include the Velodyne VLP-16 and SICK TIM571.",
                "chapter": "Module 3",
                "section": "Sensors",
            },
        ]

    @pytest.mark.asyncio
    async def test_scenario_correct_slam_info(self, robotics_chunks):
        """Test correct SLAM information is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            response = "SLAM helps robots build maps while tracking their position. EKF-SLAM is one common algorithm."

            result = await validate_response(
                response=response,
                query="What is SLAM?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_scenario_invented_algorithm(self, robotics_chunks):
        """Test invented algorithm name is flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["SuperSLAM not mentioned in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            response = "SLAM builds maps while tracking position. The best algorithm is SuperSLAM 3000."

            result = await validate_response(
                response=response,
                query="What is the best SLAM algorithm?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_mixed_valid_invalid(self, robotics_chunks):
        """Test response with both valid and invalid information is flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["GPS statement not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Mix of valid (LIDAR) and invalid (GPS) info
            response = "LIDAR sensors emit laser beams to measure distances. GPS provides 1cm accuracy indoors."

            result = await validate_response(
                response=response,
                query="How do robots sense their environment?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_correct_sensor_info(self, robotics_chunks):
        """Test correct sensor information is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            response = "LIDAR sensors like the Velodyne VLP-16 use laser beams to measure distances."

            result = await validate_response(
                response=response,
                query="What LIDAR sensors are used?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_scenario_honest_uncertainty(self, robotics_chunks):
        """Test that honest admission of not knowing is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            response = "I do not find information about camera sensors in the textbook. The sources only mention LIDAR."

            result = await validate_response(
                response=response,
                query="What camera sensors are used?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "approved"
