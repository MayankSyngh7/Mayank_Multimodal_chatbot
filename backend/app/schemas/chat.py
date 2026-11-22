from typing import Optional, List
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single message in a conversation."""
    role: str = Field(..., description="Role of the sender ('user' or 'model'/'assistant')")
    content: str = Field(..., description="Text content of the message")


class ChatRequest(BaseModel):
    """Payload sent by the client to generate a chatbot response."""
    message: str = Field(
        ..., 
        min_length=1, 
        description="The prompt or question asked by the user",
        json_schema_extra={"example": "Hello! What can you help me with?"}
    )
    session_id: Optional[str] = Field(
        None, 
        description="Optional session ID for tracking conversational context"
    )
    system_instruction: Optional[str] = Field(
        None, 
        description="Optional persona or guiding instruction for the AI model",
        json_schema_extra={"example": "You are a helpful, friendly, and concise technical tutor."}
    )
    temperature: Optional[float] = Field(
        0.7, 
        ge=0.0, 
        le=2.0, 
        description="Controls response randomness (0 = deterministic, 1 = creative)"
    )


class ChatResponse(BaseModel):
    """Response returned to the client containing the generated reply."""
    reply: str = Field(..., description="AI generated text response")
    session_id: Optional[str] = Field(None, description="Active conversation session ID")
    model: str = Field(..., description="Name of the Gemini model used for generation")


class HealthCheckResponse(BaseModel):
    """Health check status information."""
    status: str
    gemini_configured: bool
    model: str
    version: str
