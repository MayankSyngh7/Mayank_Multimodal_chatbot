from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import Generator

from backend.app.core.config import settings

# Ensure data directory exists before creating SQLite file
data_dir = Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent
data_dir.mkdir(parents=True, exist_ok=True)

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
    echo=settings.DEBUG,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLAlchemy database session per request
    and ensures the session is closed cleanly when the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes all database tables registered with Base.metadata."""
    import backend.app.db.models  # noqa: F401
    Base.metadata.create_all(bind=engine)


# Auto-create tables on import if they do not exist
init_db()
