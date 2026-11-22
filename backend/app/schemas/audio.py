from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Payload for Text-to-Speech conversion."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Text to convert into speech",
        json_schema_extra={"example": "Hello! How can I assist you today?"}
    )
    lang: str = Field(
        default="en",
        description="Language code for speech synthesis (e.g. 'en')",
        json_schema_extra={"example": "en"}
    )


class TranscriptionResponse(BaseModel):
    """Result of speech-to-text audio transcription."""
    text: str
    status: str = "success"
