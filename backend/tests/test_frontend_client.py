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


def test_api_client_send_vision_message():
    """Test sending image and prompt to vision endpoint."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "reply": "The image contains a sunset.",
        "session_id": "vision-session-1",
        "model": "gemini-1.5-flash",
    }

    with patch("requests.post", return_value=mock_resp):
        res = client.send_vision_message(
            prompt="What is this?",
            image_bytes=b"fake-image-bytes",
            filename="sunset.jpg",
            mime_type="image/jpeg",
        )
        assert res["success"] is True
        assert res["data"]["reply"] == "The image contains a sunset."
        assert res["data"]["session_id"] == "vision-session-1"


def test_api_client_transcribe_audio():
    """Test audio transcription request."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "Hello Gemini", "status": "success"}

    with patch("requests.post", return_value=mock_resp):
        res = client.transcribe_audio(audio_bytes=b"dummy-audio", mime_type="audio/wav")
        assert res["success"] is True
        assert res["text"] == "Hello Gemini"


def test_api_client_text_to_speech():
    """Test Text-to-Speech audio bytes retrieval."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fake-mp3-bytes"

    with patch("requests.post", return_value=mock_resp):
        audio_data = client.text_to_speech("Hello")
        assert audio_data == b"fake-mp3-bytes"


def test_api_client_upload_pdf():
    """Test uploading a PDF document via client."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {
        "filename": "doc.pdf",
        "page_count": 5,
        "chunk_count": 12,
        "status": "indexed",
        "message": "Indexed",
    }

    with patch("requests.post", return_value=mock_resp):
        res = client.upload_pdf(pdf_bytes=b"%PDF-1.4...", filename="doc.pdf")
        assert res["success"] is True
        assert res["data"]["filename"] == "doc.pdf"
        assert res["data"]["chunk_count"] == 12


def test_api_client_query_rag():
    """Test querying RAG via client."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answer": "This is the answer from page 3.",
        "session_id": "rag-session-1",
        "sources": [{"doc_name": "doc.pdf", "page": 3, "snippet": "..."}],
        "model": "gemini-1.5-flash",
    }

    with patch("requests.post", return_value=mock_resp):
        res = client.query_rag(question="Where is Section 2?", doc_name="doc.pdf")
        assert res["success"] is True
        assert res["data"]["answer"] == "This is the answer from page 3."
        assert len(res["data"]["sources"]) == 1


def test_api_client_list_documents():
    """Test listing documents via client."""
    client = APIClient(base_url="http://testbackend:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"documents": ["doc1.pdf", "doc2.pdf"], "count": 2}

    with patch("requests.get", return_value=mock_resp):
        docs = client.list_documents()
        assert len(docs) == 2
        assert "doc1.pdf" in docs
