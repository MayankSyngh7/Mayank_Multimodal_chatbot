from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.db.models import Conversation, ChatMessageModel


class ConversationService:
    """
    CRUD operations and business logic for conversations and messages.
    """

    @staticmethod
    def create_conversation(
        db: Session,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Conversation:
        """Create and persist a new conversation session."""
        conversation = Conversation(
            id=conversation_id or None,
            title=(title.strip() if title and title.strip() else "New Conversation")[:255],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    @staticmethod
    def get_conversation(db: Session, conversation_id: str) -> Optional[Conversation]:
        """Fetch a conversation by ID, including its messages."""
        return db.query(Conversation).filter(Conversation.id == conversation_id).first()

    @staticmethod
    def list_conversations(
        db: Session, skip: int = 0, limit: int = 50
    ) -> List[Conversation]:
        """Retrieve all conversations ordered by most recently updated."""
        return (
            db.query(Conversation)
            .order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def delete_conversation(db: Session, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages (cascade delete).
        Returns True if deleted, False if not found.
        """
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            return False
        db.delete(conv)
        db.commit()
        return True

    @staticmethod
    def add_message(
        db: Session,
        conversation_id: str,
        role: str,
        content: str,
    ) -> ChatMessageModel:
        """
        Save a message (user or assistant) to a conversation
        and update the conversation's updated_at timestamp.
        """
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            # Auto-create conversation if it doesn't exist
            conv = Conversation(
                id=conversation_id,
                title="New Conversation",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(conv)
            db.flush()

        # Update title if it's the first user message and default title
        if role == "user" and (conv.title == "New Conversation" or not conv.title):
            summary_title = content.strip().split("\n")[0][:45]
            if len(content.strip().split("\n")[0]) > 45:
                summary_title += "..."
            conv.title = summary_title

        conv.updated_at = datetime.utcnow()

        message = ChatMessageModel(
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=datetime.utcnow(),
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_messages(db: Session, conversation_id: str) -> List[ChatMessageModel]:
        """Fetch all messages for a given conversation in chronological order."""
        return (
            db.query(ChatMessageModel)
            .filter(ChatMessageModel.conversation_id == conversation_id)
            .order_by(ChatMessageModel.created_at.asc())
            .all()
        )


conversation_service = ConversationService()
