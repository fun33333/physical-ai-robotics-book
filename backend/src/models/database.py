"""
Database configuration and session management.

Sets up SQLAlchemy ORM, database connection, and session factory for use
across the application.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import settings

# Database URL from environment
DATABASE_URL = settings.database_url

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=settings.environment == "development",
    pool_pre_ping=True,  # Test connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.

    Yields a database session and ensures it's closed after use.

    Yields:
        SQLAlchemy Session instance
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database with all tables."""
    Base.metadata.create_all(bind=engine)


def drop_db() -> None:
    """Drop all tables from database (development only)."""
    Base.metadata.drop_all(bind=engine)
