import pytest
from unittest.mock import patch

from backend.tests.conftest import TestingSessionLocal
from backend.app.services.conversation_service import conversation_service


def test_create_conversation_endpoint(client):
    """Test POST /api/conversations creates a conversation."""
    response = client.post("/api/conversations", json={"title": "Test Chat"})
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Test Chat"
    assert "id" in data
    assert data["message_count"] == 0


def test_list_conversations_endpoint(client):
    """Test GET /api/conversations lists conversations."""
    # Create two conversations
    client.post("/api/conversations", json={"title": "Chat 1"})
    client.post("/api/conversations", json={"title": "Chat 2"})

    response = client.get("/api/conversations")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    titles = [c["title"] for c in data]
    assert "Chat 1" in titles
    assert "Chat 2" in titles


def test_get_conversation_detail_endpoint(client):
    """Test GET /api/conversations/{id} retrieves conversation with messages."""
    # Create conversation
    create_resp = client.post("/api/conversations", json={"title": "Detail Chat"})
    conv_id = create_resp.json()["id"]

    # Add messages directly via service
    db = TestingSessionLocal()
    conversation_service.add_message(db, conv_id, role="user", content="Hello!")
    conversation_service.add_message(db, conv_id, role="assistant", content="Hi there!")
    db.close()

    # Retrieve via API
    response = client.get(f"/api/conversations/{conv_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == conv_id
    assert data["title"] == "Detail Chat"
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Hello!"
    assert data["messages"][1]["role"] == "assistant"
    assert data["messages"][1]["content"] == "Hi there!"


def test_get_nonexistent_conversation_returns_404(client):
    """Test GET /api/conversations/{id} returns 404 for unknown ID."""
    response = client.get("/api/conversations/non-existent-id-999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_conversation_endpoint(client):
    """Test DELETE /api/conversations/{id} deletes conversation and cascades to messages."""
    create_resp = client.post("/api/conversations", json={"title": "To Delete"})
    conv_id = create_resp.json()["id"]

    # Add message
    db = TestingSessionLocal()
    conversation_service.add_message(db, conv_id, role="user", content="Temporary message")
    db.close()

    # Delete conversation
    del_resp = client.delete(f"/api/conversations/{conv_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # Confirm conversation is gone
    get_resp = client.get(f"/api/conversations/{conv_id}")
    assert get_resp.status_code == 404

    # Confirm messages were cascade deleted from database
    db = TestingSessionLocal()
    messages = conversation_service.get_messages(db, conv_id)
    assert len(messages) == 0
    db.close()


def test_delete_nonexistent_conversation_returns_404(client):
    """Test DELETE /api/conversations/{id} returns 404 when ID does not exist."""
    response = client.delete("/api/conversations/unknown-id-888")
    assert response.status_code == 404


def test_chat_endpoint_persists_messages(client):
    """
    Test POST /api/chat automatically creates a session (if omitted),
    calls Gemini service, and persists both user message and AI response to the DB.
    """
    mock_reply = "I am a helpful AI tutor powered by Gemini."

    with patch(
        "backend.app.services.gemini_service.gemini_service.is_configured",
        return_value=True
    ), patch(
        "backend.app.services.gemini_service.gemini_service.generate_chat_response",
        return_value=mock_reply
    ):
        response = client.post(
            "/api/chat",
            json={"message": "What is Python?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == mock_reply
        assert data["session_id"] is not None
        session_id = data["session_id"]

        # Check database: conversation and messages must exist
        get_resp = client.get(f"/api/conversations/{session_id}")
        assert get_resp.status_code == 200
        conv_data = get_resp.json()
        assert len(conv_data["messages"]) == 2
        assert conv_data["messages"][0]["role"] == "user"
        assert conv_data["messages"][0]["content"] == "What is Python?"
        assert conv_data["messages"][1]["role"] == "assistant"
        assert conv_data["messages"][1]["content"] == mock_reply
