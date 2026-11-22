from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class MessageRead(BaseModel):
    """Schema for reading a chat message from the database."""
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationCreate(BaseModel):
    """Payload to create a new conversation."""
    title: Optional[str] = Field(
        default="New Conversation",
        description="Optional title for the conversation",
        json_schema_extra={"example": "Python Questions"}
    )


class ConversationSummary(BaseModel):
    """Summary of a conversation for list views (without full messages)."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class ConversationDetail(BaseModel):
    """Detailed conversation object including its message history."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageRead] = []

    model_config = ConfigDict(from_attributes=True)
