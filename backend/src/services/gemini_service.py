"""
Google Gemini API Service.

Manages LLM API calls with automatic key rotation, quota tracking,
response caching, and fallback strategies for rate limit compliance.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from sqlalchemy.orm import Session

from src.config import settings
from src.models.api_key_quota import APIKeyQuota
from src.utils import GeminiError, get_logger

logger = get_logger(__name__)


class GeminiService:
    """Service for managing Google Gemini API calls with quota tracking."""

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize Gemini service.

        Args:
            db_session: SQLAlchemy session for quota tracking
        """
        self.api_keys = settings.gemini_api_keys
        self.current_key_index = 0
        self.db_session = db_session
        self.response_cache: Dict[str, str] = {}  # In-memory cache (Postgres in Phase 3)

        if not self.api_keys:
            logger.warning("No Gemini API keys configured")
        else:
            logger.info(f"Initialized Gemini service with {len(self.api_keys)} API keys")

    def _get_current_api_key(self) -> str:
        """Get current active API key."""
        if not self.api_keys:
            raise GeminiError("No API keys configured")
        return self.api_keys[self.current_key_index]

    def _rotate_to_next_key(self) -> str:
        """
        Rotate to next available API key.

        Cycles through configured keys when one is exhausted.

        Returns:
            Next API key

        Raises:
            GeminiError: If all keys are exhausted
        """
        exhausted_count = 0
        for i in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            exhausted_count += 1

            if exhausted_count >= len(self.api_keys):
                logger.error("All Gemini API keys exhausted")
                raise GeminiError(
                    "Your today's limit of AI Guide is exceeded. Please try again tomorrow."
                )

        key = self._get_current_api_key()
        logger.info(f"Rotated to API key index {self.current_key_index}")
        return key

    def _get_cache_key(self, query: str, context: str) -> str:
        """
        Generate cache key for query + context.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            SHA256 hash of query + context
        """
        combined = f"{query}:{context}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _check_cache(self, query: str, context: str) -> Optional[str]:
        """
        Check if response is cached.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Cached response or None
        """
        cache_key = self._get_cache_key(query, context)
        return self.response_cache.get(cache_key)

    def _store_cache(self, query: str, context: str, response: str) -> None:
        """
        Store response in cache.

        Args:
            query: User query
            context: Retrieved context
            response: Generated response
        """
        cache_key = self._get_cache_key(query, context)
        self.response_cache[cache_key] = response

        # TODO: Store in Postgres for persistence in Phase 3

    def generate_response(
        self,
        query: str,
        context: str,
        tone: str = "english",
        user_level: str = "intermediate",
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate response using Gemini API.

        Implements caching and automatic key rotation on quota exceeded.

        Args:
            query: User question
            context: Retrieved document chunks (RAG context)
            tone: Response tone (english, roman_urdu, bro_guide)
            user_level: User expertise level
            conversation_history: Previous messages in conversation

        Returns:
            Generated response from Gemini

        Raises:
            GeminiError: If generation fails or all keys exhausted
        """
        # Check cache first
        cached_response = self._check_cache(query, context)
        if cached_response:
            logger.info("Cache hit for query")
            return cached_response

        # Build prompt
        prompt = self._build_prompt(query, context, tone, user_level, conversation_history)

        try:
            # Configure API
            api_key = self._get_current_api_key()
            genai.configure(api_key=api_key)

            # Call Gemini
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)

            if not response.text:
                raise GeminiError("Empty response from Gemini")

            # Store in cache
            self._store_cache(query, context, response.text)

            # Track quota
            if self.db_session:
                self._track_quota(api_key, success=True)

            logger.info(f"Generated response for query: {query[:50]}...")
            return response.text

        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")

            # Try next key on error
            try:
                next_key = self._rotate_to_next_key()
                logger.info("Retrying with next API key")
                genai.configure(api_key=next_key)

                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(prompt)

                if response.text:
                    self._store_cache(query, context, response.text)
                    return response.text

            except GeminiError:
                raise
            except Exception as retry_error:
                logger.error(f"Retry also failed: {str(retry_error)}")

            raise GeminiError(
                "Your today's limit of AI Guide is exceeded. Please try again tomorrow."
            )

    def _build_prompt(
        self,
        query: str,
        context: str,
        tone: str,
        user_level: str,
        conversation_history: Optional[List[Dict]],
    ) -> str:
        """
        Build system prompt for Gemini.

        Args:
            query: User question
            context: Retrieved context
            tone: Response tone
            user_level: User expertise level
            conversation_history: Previous messages

        Returns:
            Complete prompt for Gemini
        """
        # Tone-specific instructions
        tone_instructions = {
            "english": "Provide a professional, educational response. Be clear and formal.",
            "roman_urdu": "Use friendly language with Roman Urdu phrases. Keep it conversational and engaging.",
            "bro_guide": "Use Karachi street slang and colloquial language. Be casual and friendly.",
        }

        # User level instructions
        level_instructions = {
            "beginner": "Explain concepts in simple terms (ELI5 style). Avoid jargon.",
            "intermediate": "Assume moderate knowledge. Use technical terms but explain where needed.",
            "advanced": "Provide deep technical analysis. Use advanced terminology.",
        }

        tone_inst = tone_instructions.get(tone, tone_instructions["english"])
        level_inst = level_instructions.get(user_level, level_instructions["intermediate"])

        # Build conversation context
        history = ""
        if conversation_history:
            for msg in conversation_history[-3:]:  # Last 3 messages
                history += f"\nUser: {msg.get('query', '')}\nAssistant: {msg.get('response', '')}"

        prompt = f"""You are an AI tutor helping students understand textbook content about Physical AI and Robotics.

**Tone**: {tone_inst}
**User Level**: {level_inst}

**Source Material**: {context}

**Conversation History**:
{history}

**User Question**: {query}

**Instructions**:
1. Answer ONLY using the provided source material. Do not use general knowledge.
2. Cite specific chapters or sections from the source.
3. Keep responses concise (1-2 sentences) unless asking for longer explanation.
4. If the question cannot be answered from the source, say "I don't find this in the textbook."
5. Be honest about limitations.

**Response**:"""

        return prompt

    def _track_quota(self, api_key: str, success: bool = True) -> None:
        """
        Track API key usage for quota management.

        Args:
            api_key: API key used
            success: Whether request succeeded
        """
        if not self.db_session:
            return

        try:
            key_id = f"gemini_{self.current_key_index + 1}"

            # Get or create quota record
            quota = self.db_session.query(APIKeyQuota).filter_by(api_key_id=key_id).first()

            if not quota:
                quota = APIKeyQuota(api_key_id=key_id)
                self.db_session.add(quota)

            # Reset daily at UTC 00:00
            now = datetime.utcnow()
            if quota.last_reset.date() < now.date():
                quota.requests_today = 0
                quota.requests_per_minute_today = 0
                quota.last_reset = now

            # Update counters
            if success:
                quota.requests_today += 1
                quota.requests_per_minute_today += 1

                # Check if exhausted (15 req/min * 60 min = 900 per hour max)
                if quota.requests_today >= 900:
                    quota.status = "exhausted"
                    quota.last_rotated_at = now

            else:
                quota.status = "error"
                quota.error_message = "API call failed"

            self.db_session.commit()

        except Exception as e:
            logger.warning(f"Failed to track quota: {str(e)}")

    def get_key_status(self) -> Dict[str, Any]:
        """
        Get status of all API keys.

        Returns:
            Dictionary with status of each key
        """
        status = {
            "current_key_index": self.current_key_index,
            "total_keys": len(self.api_keys),
            "keys": {},
        }

        for i, _ in enumerate(self.api_keys):
            key_id = f"gemini_{i + 1}"
            status["keys"][key_id] = {
                "status": "active" if i != self.current_key_index else "in_use",
            }

        return status
