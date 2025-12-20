"""
Main Coordinator Agent.

Central orchestrator that routes user queries through the specialized agent pipeline.
Responsible for initialization, delegation, and context management.
"""

import logging
from typing import Any, Dict, Optional

from src.services.orchestration_service import OrchestrationService, PipelineContext
from src.utils import get_logger

logger = get_logger(__name__)


class MainCoordinatorAgent:
    """Main coordinator agent managing the multi-agent pipeline."""

    def __init__(self):
        """Initialize coordinator agent."""
        self.orchestration_service = OrchestrationService()
        logger.info("MainCoordinatorAgent initialized")

    async def initialize_pipeline(self) -> Dict[str, Any]:
        """
        Initialize the full agent pipeline.

        Returns:
            Status dictionary with initialization results
        """
        try:
            status = self.orchestration_service.get_pipeline_status()
            logger.info("Pipeline initialized successfully")
            return {
                "status": "initialized",
                "pipeline": status,
            }
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
            }

    async def route_through_pipeline(
        self,
        query: str,
        selected_text: Optional[str] = None,
        tone: str = "english",
        user_id: str = "anonymous",
        session_id: str = "",
        user_level: str = "intermediate",
        conversation_history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Route a user query through the complete agent pipeline.

        Orchestrates the flow: Query → Coordinator → RAG → Answer → Tone → Safety → Response

        Args:
            query: User's question
            selected_text: Text selected by user from textbook (optional)
            tone: Preferred response tone
            user_id: User identifier
            session_id: Session identifier for conversation isolation
            user_level: User expertise level
            conversation_history: Previous messages for context

        Returns:
            Dictionary with response, sources, latencies, and metadata

        Raises:
            ValueError: If input validation fails
            TimeoutError: If pipeline exceeds timeout
        """
        try:
            # Create pipeline context
            context = PipelineContext(
                query=query,
                selected_text=selected_text,
                user_id=user_id,
                session_id=session_id,
                tone=tone,
                user_level=user_level,
                conversation_history=conversation_history or [],
            )

            # Process through pipeline
            result_context = await self.orchestration_service.process_chat(context)

            # Return response
            return result_context.to_response_dict()

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except TimeoutError as e:
            logger.error(f"Pipeline timeout: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            raise

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of coordinator and sub-agents.

        Returns:
            Dictionary with agent status information
        """
        return {
            "coordinator": "operational",
            "pipeline_status": self.orchestration_service.get_pipeline_status(),
        }
