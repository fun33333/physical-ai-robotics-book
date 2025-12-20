"""
Configuration management for RAG Chatbot backend.

Handles environment variable loading, validation, and provides configuration classes
for different application components.
"""

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Server Configuration
    environment: str = "development"
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # Database Configuration
    database_url: str = "postgresql://localhost/rag_chatbot_db"

    # Gemini API Keys (3 keys for rotation)
    gemini_api_key_1: str = ""
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""

    @property
    def gemini_api_keys(self) -> List[str]:
        """Return list of available Gemini API keys."""
        keys = []
        if self.gemini_api_key_1:
            keys.append(self.gemini_api_key_1)
        if self.gemini_api_key_2:
            keys.append(self.gemini_api_key_2)
        if self.gemini_api_key_3:
            keys.append(self.gemini_api_key_3)
        return keys

    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "textbook_chapters"

    # CORS Configuration
    cors_origin: str = "http://localhost:3000"

    # Logging
    log_level: str = "INFO"

    # Rate Limiting (15 requests/minute per constitution)
    rate_limit_rpm: int = 15

    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()  # type: ignore


settings = get_settings()
