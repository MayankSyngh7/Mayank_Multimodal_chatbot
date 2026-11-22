import io
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from backend.app.core.config import settings


def create_dummy_image_bytes(format="JPEG") -> bytes:
    """Helper to generate a real, valid in-memory image bytes."""
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_vision_endpoint_success(client):
    """Test POST /api/chat/vision with a valid JPG image and prompt."""
    image_bytes = create_dummy_image_bytes("JPEG")
    mock_reply = "I see a solid blue square in the image."

    with patch(
        "backend.app.services.gemini_service.gemini_service.is_configured",
        return_value=True,
    ), patch(
        "backend.app.services.gemini_service.gemini_service.generate_vision_response",
        return_value=mock_reply,
    ):
        response = client.post(
            "/api/chat/vision",
            data={"prompt": "What color is in this image?"},
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == mock_reply
        assert data["session_id"] is not None

        # Verify message was persisted into database
        conv_resp = client.get(f"/api/conversations/{data['session_id']}")
        assert conv_resp.status_code == 200
        conv_data = conv_resp.json()
        assert len(conv_data["messages"]) == 2
        assert "📷 [Image: test.jpg]" in conv_data["messages"][0]["content"]
        assert conv_data["messages"][1]["content"] == mock_reply


def test_vision_endpoint_unsupported_file_type(client):
    """Test POST /api/chat/vision rejects text or executable files."""
    response = client.post(
        "/api/chat/vision",
        data={"prompt": "Describe this file"},
        files={"file": ("test.txt", b"Hello text file", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["detail"]


def test_audio_transcribe_endpoint_success(client):
    """Test POST /api/audio/transcribe transcribes audio successfully."""
    dummy_audio = b"RIFF....WAVEfmt ...."  # Dummy wav bytes
    mock_transcription = "What is the weather today?"

    with patch(
        "backend.app.services.gemini_service.gemini_service.is_configured",
        return_value=True,
    ), patch(
        "backend.app.services.gemini_service.gemini_service.transcribe_audio",
        return_value=mock_transcription,
    ):
        response = client.post(
            "/api/audio/transcribe",
            files={"file": ("audio.wav", dummy_audio, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == mock_transcription
        assert data["status"] == "success"


def test_audio_transcribe_unsupported_type(client):
    """Test POST /api/audio/transcribe rejects non-audio file types."""
    response = client.post(
        "/api/audio/transcribe",
        files={"file": ("doc.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Unsupported audio type" in response.json()["detail"]


def test_tts_endpoint_success(client):
    """Test POST /api/audio/tts returns audio/mpeg stream."""
    fake_mp3 = io.BytesIO(b"\xff\xfb\x90\x00...")  # Dummy MP3 header bytes

    with patch(
        "backend.app.services.audio_service.audio_service.text_to_speech",
        return_value=fake_mp3,
    ):
        response = client.post(
            "/api/audio/tts",
            json={"text": "Hello, how can I help you?", "lang": "en"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 0


def test_tts_endpoint_empty_text_validation(client):
    """Test POST /api/audio/tts rejects empty text input."""
    response = client.post(
        "/api/audio/tts",
        json={"text": ""},
    )
    assert response.status_code == 422  # Pydantic validation error
