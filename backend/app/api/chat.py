from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.gemini_service import gemini_service
from backend.app.core.config import settings

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Gemini Chat Response",
    description="Accepts a user message, calls Google Gemini API, and returns the AI response.",
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for text conversation.
    Handles user prompts, applies optional system instructions, and queries Gemini.
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
        reply_text = gemini_service.generate_chat_response(
            message=request.message,
            system_instruction=request.system_instruction,
            temperature=request.temperature or 0.7,
        )

        return ChatResponse(
            reply=reply_text,
            session_id=request.session_id,
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
