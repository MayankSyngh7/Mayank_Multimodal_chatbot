import pytest
from unittest.mock import patch
from backend.app.core.config import settings


def test_root_endpoint(client):
    """Test the root endpoint returns 200 and expected info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Welcome" in data["message"]
    assert data["docs_url"] == "/docs"
    assert data["health_check"] == "/health"


def test_health_check_endpoint(client):
    """Test the health check returns status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gemini_configured" in data
    assert "model" in data


def test_gemini_status_endpoint(client):
    """Test the chat status endpoint."""
    response = client.get("/api/chat/status")
    assert response.status_code == 200
    data = response.json()
    assert "gemini_configured" in data
    assert "model" in data


def test_chat_endpoint_unconfigured_gemini(client):
    """
    When Gemini API key is missing or not configured,
    the endpoint should return 503 with a helpful message.
    """
    with patch("backend.app.services.gemini_service.gemini_service.is_configured", return_value=False):
        response = client.post(
            "/api/chat",
            json={"message": "Hello!"}
        )
        assert response.status_code == 503
        data = response.json()
        assert "Gemini API is not configured" in data["detail"]


def test_chat_endpoint_validation_error(client):
    """Test validation errors on empty or missing message."""
    response = client.post("/api/chat", json={})
    assert response.status_code == 422  # Pydantic validation error


def test_chat_endpoint_successful_mock(client):
    """
    Test successful chat response when Gemini service generates text.
    """
    mock_reply = "Hello! I am Gemini, your multimodal AI assistant."

    with patch(
        "backend.app.services.gemini_service.gemini_service.is_configured",
        return_value=True
    ), patch(
        "backend.app.services.gemini_service.gemini_service.generate_chat_response",
        return_value=mock_reply
    ):
        response = client.post(
            "/api/chat",
            json={
                "message": "Hi there!",
                "session_id": "test-session-123",
                "system_instruction": "You are a friendly assistant.",
                "temperature": 0.5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == mock_reply
        assert data["session_id"] == "test-session-123"
        assert data["model"] == settings.GEMINI_MODEL
