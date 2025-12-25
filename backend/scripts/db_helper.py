#!/usr/bin/env python3
"""
Database migration helper script.

Provides utilities for managing database migrations, resetting the database,
and seeding test data for local development.

Usage:
    python scripts/db_helper.py migrate      # Run all pending migrations
    python scripts/db_helper.py reset        # Drop all tables and re-run migrations
    python scripts/db_helper.py seed         # Seed test data for development
    python scripts/db_helper.py status       # Show current migration status
"""

import argparse
import os
import sys
import uuid
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alembic import command
from alembic.config import Config

from src.config import settings
from src.models.database import Base, engine, SessionLocal
from src.models.conversation import Conversation
from src.models.embeddings_metadata import EmbeddingsMetadata
from src.models.api_key_quota import APIKeyQuota


def get_alembic_config() -> Config:
    """Get Alembic configuration."""
    alembic_cfg = Config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url)
    return alembic_cfg


def run_migrations() -> None:
    """Run all pending database migrations."""
    print("Running database migrations...")
    alembic_cfg = get_alembic_config()
    command.upgrade(alembic_cfg, "head")
    print("Migrations complete!")


def reset_database() -> None:
    """Drop all tables and recreate from migrations."""
    print("WARNING: This will delete all data!")
    confirm = input("Are you sure? (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborted.")
        return

    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Tables dropped.")

    print("Running migrations...")
    run_migrations()
    print("Database reset complete!")


def show_status() -> None:
    """Show current migration status."""
    print("Current migration status:")
    alembic_cfg = get_alembic_config()
    command.current(alembic_cfg)
    print("\nPending migrations:")
    command.heads(alembic_cfg)


def seed_test_data() -> None:
    """Seed database with test data for development."""
    print("Seeding test data...")
    db = SessionLocal()

    try:
        # Create test API key quotas
        api_keys = [
            APIKeyQuota(
                api_key_id="gemini_1",
                requests_today=0,
                requests_per_minute_today=0,
                status="active",
            ),
            APIKeyQuota(
                api_key_id="gemini_2",
                requests_today=0,
                requests_per_minute_today=0,
                status="active",
            ),
            APIKeyQuota(
                api_key_id="gemini_3",
                requests_today=0,
                requests_per_minute_today=0,
                status="active",
            ),
        ]
        for key in api_keys:
            existing = db.query(APIKeyQuota).filter_by(api_key_id=key.api_key_id).first()
            if not existing:
                db.add(key)
        print(f"  Created {len(api_keys)} API key quota records")

        # Create test embeddings metadata
        test_chunks = [
            EmbeddingsMetadata(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                chapter="Module 1",
                section="ROS 2 Fundamentals",
                subsection="Introduction",
                difficulty_level="beginner",
                token_count=256,
                source_url="/docs/module-1/ros2-fundamentals",
                content_preview="ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software..."
            ),
            EmbeddingsMetadata(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                chapter="Module 1",
                section="ROS 2 Architecture",
                subsection="DDS Communication",
                difficulty_level="intermediate",
                token_count=312,
                source_url="/docs/module-1/ros2-architecture",
                content_preview="ROS 2 uses DDS (Data Distribution Service) as its middleware for communication..."
            ),
            EmbeddingsMetadata(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                chapter="Module 2",
                section="Navigation Stack",
                subsection="SLAM Basics",
                difficulty_level="intermediate",
                token_count=428,
                source_url="/docs/module-2/navigation-slam",
                content_preview="SLAM (Simultaneous Localization and Mapping) allows robots to build a map of their environment..."
            ),
        ]
        for chunk in test_chunks:
            db.add(chunk)
        print(f"  Created {len(test_chunks)} embeddings metadata records")

        # Create test conversations
        test_user = "test_user_001"
        test_session = "test_session_001"
        conversations = [
            Conversation(
                user_id=test_user,
                session_id=test_session,
                query="What is ROS 2?",
                response="ROS 2 is a flexible framework for writing robot software. It provides tools, libraries, and conventions to simplify building complex robots.",
                agent_used="orchestrator",
                tone="english",
                user_level="beginner",
                sources=[{"chapter": "Module 1", "section": "ROS 2 Fundamentals"}],
            ),
            Conversation(
                user_id=test_user,
                session_id=test_session,
                query="How does it communicate?",
                response="ROS 2 uses DDS (Data Distribution Service) for communication between nodes. This provides real-time, scalable, and reliable data transfer.",
                agent_used="orchestrator",
                tone="english",
                user_level="beginner",
                sources=[{"chapter": "Module 1", "section": "ROS 2 Architecture"}],
            ),
            Conversation(
                user_id=test_user,
                session_id=test_session,
                query="Tell me about SLAM",
                response="SLAM (Simultaneous Localization and Mapping) allows robots to build a map of their environment while simultaneously tracking their location within that map.",
                agent_used="orchestrator",
                tone="english",
                user_level="intermediate",
                sources=[{"chapter": "Module 2", "section": "Navigation Stack"}],
            ),
        ]
        for conv in conversations:
            db.add(conv)
        print(f"  Created {len(conversations)} conversation records")

        db.commit()
        print("Test data seeded successfully!")

    except Exception as e:
        db.rollback()
        print(f"Error seeding data: {e}")
        raise
    finally:
        db.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Database migration helper for RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "action",
        choices=["migrate", "reset", "seed", "status"],
        help="Action to perform",
    )

    args = parser.parse_args()

    if args.action == "migrate":
        run_migrations()
    elif args.action == "reset":
        reset_database()
    elif args.action == "seed":
        seed_test_data()
    elif args.action == "status":
        show_status()


if __name__ == "__main__":
    main()
