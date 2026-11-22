from unittest.mock import patch, MagicMock
import requests

from frontend.utils.api_client import APIClient


def test_api_client_check_health_success():
    """Test health check parsing when backend is healthy."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "gemini_configured": True}

    with patch("requests.get", return_value=mock_resp):
        result = client.check_health()
        assert result["connected"] is True
        assert result["data"]["gemini_configured"] is True
        assert result["error"] is None


def test_api_client_check_health_connection_error():
    """Test health check handles connection failure gracefully."""
    client = APIClient(base_url="http://offlinebackend:8000")

    with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
        result = client.check_health()
        assert result["connected"] is False
        assert "Could not connect" in result["error"]


def test_api_client_list_conversations():
    """Test listing conversations."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"id": "conv1", "title": "Chat 1", "message_count": 2}
    ]

    with patch("requests.get", return_value=mock_resp):
        conversations = client.list_conversations()
        assert len(conversations) == 1
        assert conversations[0]["id"] == "conv1"


def test_api_client_send_chat_message_success():
    """Test sending chat message and receiving response."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "reply": "Hello, world!",
        "session_id": "session-abc",
        "model": "gemini-1.5-flash",
    }

    with patch("requests.post", return_value=mock_resp):
        res = client.send_chat_message(message="Hi", session_id="session-abc")
        assert res["success"] is True
        assert res["data"]["reply"] == "Hello, world!"
        assert res["data"]["session_id"] == "session-abc"


def test_api_client_send_chat_message_connection_error():
    """Test graceful handling when backend is down during message send."""
    client = APIClient(base_url="http://offlinebackend:8000")

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        res = client.send_chat_message(message="Hi")
        assert res["success"] is False
        assert "Connection Error" in res["error"]
