import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Response
from backend.app.schemas.audio import TTSRequest, TranscriptionResponse
from backend.app.services.gemini_service import gemini_service
from backend.app.services.audio_service import audio_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["Audio"])

# Allowed audio MIME types for voice transcription
ALLOWED_AUDIO_TYPES = [
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp3",
    "audio/mpeg",
    "audio/ogg",
    "audio/webm",
    "audio/m4a",
    "audio/x-m4a",
]
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe Speech Audio into Text",
    description="Uploads an audio file and transcribes speech into text using Gemini audio understanding.",
)
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio recording file (WAV, MP3, OGG, WebM, M4A)"),
) -> TranscriptionResponse:
    """Transcribes an uploaded audio file into text."""
    # Validate audio MIME type first
    content_type = file.content_type or "audio/wav"
    if content_type.lower() not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported audio type '{content_type}'. "
                f"Allowed formats: WAV, MP3, OGG, WebM, M4A."
            ),
        )

    if not gemini_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API is not configured. Add your GEMINI_API_KEY to the .env file.",
        )

    # Read and validate audio size
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )
    if len(audio_bytes) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum size limit of {MAX_AUDIO_SIZE // (1024 * 1024)} MB.",
        )

    try:
        transcribed_text = gemini_service.transcribe_audio(
            audio_bytes=audio_bytes,
            mime_type=content_type,
        )
        return TranscriptionResponse(text=transcribed_text, status="success")
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech transcription failed: {str(e)}",
        )


@router.post(
    "/tts",
    summary="Convert Text to Speech (TTS)",
    description="Converts input text into MP3 audio stream for playback in the client.",
)
async def tts_endpoint(request: TTSRequest):
    """Generates MP3 audio for given text using gTTS."""
    try:
        audio_stream = audio_service.text_to_speech(text=request.text, lang=request.lang)
        return Response(
            content=audio_stream.getvalue(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"},
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text-to-Speech generation failed: {str(e)}",
        )
