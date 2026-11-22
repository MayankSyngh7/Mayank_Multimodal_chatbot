from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.gemini_service import gemini_service
from backend.app.services.conversation_service import conversation_service
from backend.app.db.database import get_db
from backend.app.core.config import settings

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Gemini Chat Response and Save History",
    description="Accepts a user message, queries Gemini API, and persists messages to SQLite.",
)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Chat endpoint for text conversation.
    1. Verifies Gemini API is configured.
    2. Retrieves or creates conversation in SQLite database.
    3. Saves user message.
    4. Generates AI response from Gemini.
    5. Saves AI response in conversation history.
    6. Returns ChatResponse with session_id.
    """
    if not gemini_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Gemini API is not configured. Please add your GEMINI_API_KEY to the "
                ".env file in the project root."
            ),
        )

    try:
        # Determine or create conversation session
        if request.session_id:
            conv = conversation_service.get_conversation(db, request.session_id)
            if not conv:
                conv = conversation_service.create_conversation(
                    db,
                    title="New Conversation",
                    conversation_id=request.session_id,
                )
        else:
            conv = conversation_service.create_conversation(db, title="New Conversation")

        session_id = conv.id

        # 1. Save user prompt to SQLite database
        conversation_service.add_message(
            db=db,
            conversation_id=session_id,
            role="user",
            content=request.message,
        )

        # 2. Call Google Gemini service to generate response
        reply_text = gemini_service.generate_chat_response(
            message=request.message,
            system_instruction=request.system_instruction,
            temperature=request.temperature or 0.7,
        )

        # 3. Save AI assistant response to SQLite database
        conversation_service.add_message(
            db=db,
            conversation_id=session_id,
            role="assistant",
            content=reply_text,
        )

        return ChatResponse(
            reply=reply_text,
            session_id=session_id,
            model=settings.GEMINI_MODEL,
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(re),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}",
        )


@router.get(
    "/status",
    summary="Check Gemini Service Status",
    description="Returns whether the Gemini API is configured and ready to process requests.",
)
async def gemini_status():
    """Returns the operational status of the Gemini service."""
    configured = gemini_service.is_configured()
    return {
        "gemini_configured": configured,
        "model": settings.GEMINI_MODEL,
        "message": (
            "Gemini API is ready."
            if configured
            else "Gemini API key is not configured. Check your .env file."
        ),
    }
