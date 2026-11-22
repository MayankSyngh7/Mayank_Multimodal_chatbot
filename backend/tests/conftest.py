from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.database import Base, get_db
import backend.app.db.models  # noqa: F401

# Use a test SQLite database file inside data/ directory
TEST_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "test_chatbot.db"
TEST_DATABASE_URL = f"sqlite:///{TEST_DB_PATH}"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Ensure tables are created once on test engine
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override database dependency globally for tests
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def clean_test_db():
    """Clean all tables before each test to maintain full test isolation."""
    db = TestingSessionLocal()
    try:
        for table in reversed(Base.metadata.sorted_tables):
            db.execute(table.delete())
        db.commit()
    finally:
        db.close()
    yield


@pytest.fixture
def client():
    """Yields a TestClient instance."""
    with TestClient(app) as c:
        yield c
