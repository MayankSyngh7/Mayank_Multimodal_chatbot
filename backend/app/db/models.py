import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from backend.app.db.database import Base


def generate_uuid() -> str:
    """Generate a random 32-character hexadecimal UUID string."""
    return uuid.uuid4().hex


class Conversation(Base):
    """
    Represents a chat conversation thread/session.
    Stores metadata like title, creation time, and last updated time.
    """
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # One-to-many relationship: A conversation has many messages.
    # cascade="all, delete-orphan" ensures all messages are deleted when conversation is deleted.
    messages = relationship(
        "ChatMessageModel",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessageModel.created_at.asc()",
    )

    def __repr__(self) -> str:
        return f"<Conversation(id='{self.id}', title='{self.title}')>"


class ChatMessageModel(Base):
    """
    Represents an individual message (user prompt or AI response) in a conversation.
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)  # "user" or "assistant" / "model"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Many-to-one relationship: Each message belongs to a conversation.
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessageModel(id={self.id}, role='{self.role}', conversation_id='{self.conversation_id}')>"
