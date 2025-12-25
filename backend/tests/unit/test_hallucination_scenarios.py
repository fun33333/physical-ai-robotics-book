"""
Unit tests for hallucination scenario detection (Phase 3 - T044b).

Tests 5+ hallucination scenarios:
1. False claims (incorrect facts)
2. Extrapolation (going beyond source material)
3. External knowledge (using non-textbook information)
4. Invented statistics
5. Unsupported conclusions

All scenarios should be caught by the Safety Guardian.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.safety_guardian import validate_response


class TestHallucinationScenarios:
    """Comprehensive tests for various hallucination types."""

    @pytest.fixture
    def robotics_chunks(self):
        """Source chunks about robotics topics."""
        return [
            {
                "chunk_id": "ros_001",
                "text": "ROS 2 (Robot Operating System 2) was released in December 2017. It is the successor to ROS 1.",
                "chapter": "Module 1",
                "section": "ROS History",
            },
            {
                "chunk_id": "ros_002",
                "text": "ROS 2 uses DDS (Data Distribution Service) for communication between nodes.",
                "chapter": "Module 1",
                "section": "ROS Architecture",
            },
            {
                "chunk_id": "slam_001",
                "text": "SLAM (Simultaneous Localization and Mapping) allows robots to build maps while tracking their position.",
                "chapter": "Module 2",
                "section": "Navigation",
            },
            {
                "chunk_id": "sensor_001",
                "text": "LIDAR sensors emit laser beams to measure distances to objects. Common LIDAR units include the Velodyne VLP-16.",
                "chapter": "Module 3",
                "section": "Sensors",
            },
        ]

    # Scenario 1: False Claims (Incorrect Facts)
    @pytest.mark.asyncio
    async def test_scenario_false_date(self, robotics_chunks):
        """Test detection of incorrect date (sources say 2017, not 2015)."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["ROS 2 release date is 2017, not 2015"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 was released in 2015 by Google.",
                query="When was ROS 2 released?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_false_organization(self, robotics_chunks):
        """Test detection of incorrect attribution (not by Google)."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Google attribution not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 was developed by Google's robotics team.",
                query="Who developed ROS 2?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_false_version(self, robotics_chunks):
        """Test detection of non-existent version (ROS 3 doesn't exist in sources)."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["ROS 3 not mentioned in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 3 is the latest version and was released in 2024.",
                query="What is the latest ROS version?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    # Scenario 2: Extrapolation (Going Beyond Source Material)
    @pytest.mark.asyncio
    async def test_scenario_extrapolation_python_support(self, robotics_chunks):
        """Test detection of extrapolated programming language info."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Python support details not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 exclusively supports Python 3.8+ and dropped C++ support.",
                query="What languages does ROS 2 support?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_extrapolation_performance(self, robotics_chunks):
        """Test detection of extrapolated performance claims."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Performance comparison not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 is 10x faster than ROS 1 in message passing.",
                query="How does ROS 2 compare to ROS 1?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    # Scenario 3: External Knowledge (Non-Textbook Information)
    @pytest.mark.asyncio
    async def test_scenario_external_future_plans(self, robotics_chunks):
        """Test detection of external knowledge about future plans."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Future development plans not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 is planning to integrate with Kubernetes in the next release.",
                query="What is the future of ROS 2?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_external_company_info(self, robotics_chunks):
        """Test detection of external knowledge about companies."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Boston Dynamics info not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Boston Dynamics uses ROS 2 for all their Spot robots.",
                query="Who uses ROS 2?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    # Scenario 4: Invented Statistics
    @pytest.mark.asyncio
    async def test_scenario_invented_percentage(self, robotics_chunks):
        """Test detection of invented percentage statistics."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["95% statistic not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 is used by 95% of robotics companies worldwide.",
                query="How widely used is ROS 2?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_invented_counts(self, robotics_chunks):
        """Test detection of invented numerical counts."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["10 million downloads claim not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 has been downloaded over 10 million times.",
                query="How popular is ROS 2?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_invented_speed(self, robotics_chunks):
        """Test detection of invented speed metrics."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Latency figures not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="SLAM can process maps at 100Hz with sub-millisecond latency.",
                query="How fast is SLAM?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    # Scenario 5: Unsupported Conclusions
    @pytest.mark.asyncio
    async def test_scenario_unsupported_recommendation(self, robotics_chunks):
        """Test detection of unsupported recommendations."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Recommendation not supported by sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="You should always use ROS 2 instead of ROS 1 because it's objectively better.",
                query="Should I use ROS 2?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    @pytest.mark.asyncio
    async def test_scenario_unsupported_causation(self, robotics_chunks):
        """Test detection of unsupported cause-effect claims."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["Causation claim not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="Because ROS 2 uses DDS, it is immune to all security vulnerabilities.",
                query="Is ROS 2 secure?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "flagged"

    # Correct Responses (Should Be Approved)
    @pytest.mark.asyncio
    async def test_approves_correct_date(self, robotics_chunks):
        """Test that correct date from sources is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 was released in December 2017.",
                query="When was ROS 2 released?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_approves_correct_architecture(self, robotics_chunks):
        """Test that correct architecture info from sources is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="ROS 2 uses DDS (Data Distribution Service) for communication between nodes.",
                query="What does ROS 2 use for communication?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_approves_honest_uncertainty(self, robotics_chunks):
        """Test that honest admission of uncertainty is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="I don't find information about ROS 2 installation in the textbook.",
                query="How do I install ROS 2?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "approved"

    @pytest.mark.asyncio
    async def test_approves_paraphrased_content(self, robotics_chunks):
        """Test that paraphrased content from sources is approved."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "approved", "issues": []}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await validate_response(
                response="SLAM helps robots create maps of their surroundings while keeping track of where they are.",
                query="What is SLAM?",
                chunks=robotics_chunks,
            )

            assert result["validation_status"] == "approved"


class TestMixedValidInvalidContent:
    """Tests for responses containing both valid and invalid information."""

    @pytest.fixture
    def source_chunks(self):
        return [
            {
                "chunk_id": "c1",
                "text": "LIDAR sensors emit laser beams to measure distances.",
                "chapter": "Module 3",
                "section": "Sensors",
            },
        ]

    @pytest.mark.asyncio
    async def test_flags_mixed_content(self, source_chunks):
        """Test that response with both valid and invalid info is flagged."""
        with patch("src.agents.safety_guardian.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = '{"status": "flagged", "issues": ["GPS accuracy claim not in sources"]}'
            mock_runner.run = AsyncMock(return_value=mock_result)

            # Mix of valid (LIDAR) and invalid (GPS accuracy) information
            result = await validate_response(
                response="LIDAR sensors emit laser beams. GPS provides centimeter-level accuracy indoors.",
                query="How do robots sense their environment?",
                chunks=source_chunks,
            )

            assert result["validation_status"] == "flagged"
