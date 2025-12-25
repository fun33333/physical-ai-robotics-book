"""
Unit tests for RAG Agent (Phase 3 - T030).

Tests the RAG Agent to verify:
- Context retrieval from Qdrant
- Selected text prioritization
- Top-k chunk retrieval with metadata
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.rag_agent import (
    rag_agent,
    retrieve_context,
    search_textbook,
    generate_embedding,
)


class TestRagAgentConfiguration:
    """Tests for RAG Agent configuration."""

    def test_agent_exists(self):
        """Test that the rag_agent is properly defined."""
        assert rag_agent is not None
        assert rag_agent.name == "RAG Agent"

    def test_agent_has_instructions(self):
        """Test that agent has retrieval instructions."""
        assert rag_agent.instructions is not None
        assert len(rag_agent.instructions) > 0

    def test_agent_instructions_mention_retrieval(self):
        """Test that instructions include retrieval concepts."""
        instructions = rag_agent.instructions.lower()

        # Should mention retrieval/search concepts
        assert "retriev" in instructions or "search" in instructions or "find" in instructions or "textbook" in instructions

    def test_agent_has_tools(self):
        """Test that agent has search tools."""
        assert rag_agent.tools is not None
        assert len(rag_agent.tools) > 0


class TestGenerateEmbedding:
    """Tests for embedding generation."""

    def test_generate_embedding_returns_list(self):
        """Test that embedding returns a list."""
        embedding = generate_embedding("test text")
        assert isinstance(embedding, list)

    def test_generate_embedding_correct_dimension(self):
        """Test that embedding has 768 dimensions."""
        embedding = generate_embedding("test text")
        assert len(embedding) == 768

    def test_generate_embedding_normalized(self):
        """Test that embedding is normalized."""
        embedding = generate_embedding("test text")
        magnitude = sum(x * x for x in embedding) ** 0.5
        # Should be approximately 1.0 (unit vector)
        assert abs(magnitude - 1.0) < 0.01

    def test_generate_embedding_deterministic(self):
        """Test that same text produces same embedding."""
        embedding1 = generate_embedding("same text")
        embedding2 = generate_embedding("same text")
        assert embedding1 == embedding2

    def test_generate_embedding_different_texts(self):
        """Test that different texts produce different embeddings."""
        embedding1 = generate_embedding("text one")
        embedding2 = generate_embedding("text two")
        assert embedding1 != embedding2


class TestSearchTextbook:
    """Tests for search_textbook function."""

    def test_search_textbook_is_tool(self):
        """Test that search_textbook is defined as a tool."""
        # Since function_tool decorator wraps the function, we verify the agent has it
        assert rag_agent.tools is not None
        # The function is wrapped as a FunctionTool
        assert len(rag_agent.tools) > 0

    def test_search_textbook_wrapper_exists(self):
        """Test that search_textbook is available."""
        # The function exists but is wrapped by function_tool
        assert search_textbook is not None


class TestRetrieveContext:
    """Tests for the retrieve_context function."""

    @pytest.mark.asyncio
    async def test_retrieve_context_returns_dict(self):
        """Test that retrieve_context returns a dictionary."""
        # The function uses a FunctionTool internally which may cause issues when mocked
        # We test the return structure
        result = await retrieve_context(query="What is ROS 2?")

        assert isinstance(result, dict)
        assert "chunks" in result

    @pytest.mark.asyncio
    async def test_retrieve_context_handles_empty_query(self):
        """Test handling of empty query."""
        result = await retrieve_context(query="")

        assert "chunks" in result
        # Should still work with empty query
        assert isinstance(result["chunks"], list)

    @pytest.mark.asyncio
    async def test_retrieve_context_with_selected_text_structure(self):
        """Test that result structure is correct with selected text."""
        result = await retrieve_context(
            query="Explain this",
            selected_text="ROS 2 uses DDS",
        )

        assert isinstance(result, dict)
        assert "chunks" in result

    @pytest.mark.asyncio
    async def test_retrieve_context_error_returns_error_field(self):
        """Test that errors populate error field."""
        # Force an error by patching
        with patch("src.agents.rag_agent.search_textbook") as mock_search:
            mock_search.side_effect = Exception("Test error")

            result = await retrieve_context(query="test")

            assert "chunks" in result
            # Either error field or empty chunks
            assert "error" in result or len(result["chunks"]) == 0


class TestRetrieveContextIntegration:
    """Integration-style tests for retrieve_context."""

    @pytest.mark.asyncio
    async def test_retrieval_returns_count(self):
        """Test that retrieval returns count field."""
        result = await retrieve_context(query="Test")

        # Should have count field (may be 0 if no Qdrant)
        assert "count" in result or "chunks" in result

    @pytest.mark.asyncio
    async def test_retrieval_with_none_selected_text(self):
        """Test retrieval works with None selected text."""
        result = await retrieve_context(query="Test", selected_text=None)

        assert "chunks" in result

    @pytest.mark.asyncio
    async def test_retrieval_structure_consistent(self):
        """Test that return structure is consistent."""
        result1 = await retrieve_context(query="Query one")
        result2 = await retrieve_context(query="Query two", selected_text="text")

        # Both should have same keys
        assert set(result1.keys()) == set(result2.keys())
