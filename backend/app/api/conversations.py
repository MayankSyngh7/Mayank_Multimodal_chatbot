from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.conversation import (
    ConversationCreate,
    ConversationSummary,
    ConversationDetail,
)
from backend.app.services.conversation_service import conversation_service

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


@router.post(
    "",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
)
def create_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
) -> ConversationSummary:
    """Create a new conversation session with an optional title."""
    conv = conversation_service.create_conversation(db=db, title=payload.title)
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=0,
    )


@router.get(
    "",
    response_model=List[ConversationSummary],
    summary="List all conversations",
)
def list_conversations(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> List[ConversationSummary]:
    """Retrieve all conversation summaries ordered by last updated."""
    conversations = conversation_service.list_conversations(db=db, skip=skip, limit=limit)
    return [
        ConversationSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=len(c.messages),
        )
        for c in conversations
    ]


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation details and messages",
)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> ConversationDetail:
    """Retrieve a conversation along with its full chronological message history."""
    conv = conversation_service.get_conversation(db=db, conversation_id=conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID '{conversation_id}' was not found.",
        )
    return conv


@router.delete(
    "/{conversation_id}",
    summary="Delete a conversation",
)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
):
    """Delete a conversation and all associated messages."""
    deleted = conversation_service.delete_conversation(db=db, conversation_id=conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID '{conversation_id}' was not found.",
        )
    return {
        "status": "success",
        "message": "Conversation deleted successfully.",
        "id": conversation_id,
    }
