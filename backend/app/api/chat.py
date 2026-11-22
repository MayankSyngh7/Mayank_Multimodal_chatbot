import uuid
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.gemini_service import gemini_service
from backend.app.services.conversation_service import conversation_service
from backend.app.db.database import get_db
from backend.app.core.config import settings
from backend.app.core.security import (
    detect_prompt_injection,
    validate_content_safety,
    MAX_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])

# Allowed image MIME types
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/jpg", "image/png"]
ALLOWED_IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


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
    # Guardrail: Check for prompt injection or system override attempts
    injection_warning = detect_prompt_injection(request.message)
    if injection_warning:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=injection_warning,
        )

    # Guardrail: Content safety policy check
    safety_warning = validate_content_safety(request.message)
    if safety_warning:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safety_warning,
        )

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


@router.post(
    "/vision",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Multimodal Image Understanding Chat",
    description="Upload an image (JPG, PNG) with a prompt and get Gemini Vision answers.",
)
async def vision_chat_endpoint(
    prompt: str = Form(..., min_length=1, description="Question or prompt about the image"),
    file: UploadFile = File(..., description="Image file (JPG, JPEG, PNG)"),
    session_id: Optional[str] = Form(None, description="Optional conversation session ID"),
    system_instruction: Optional[str] = Form(None),
    temperature: float = Form(0.7),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Multimodal vision endpoint:
    1. Validates image MIME type and file size.
    2. Securely saves the uploaded file to data/uploads/.
    3. Persists conversation and messages to SQLite database.
    4. Calls Gemini Vision API.
    5. Returns response.
    """
    # Validate image MIME type and extension first
    content_type = (file.content_type or "").lower()
    file_ext = Path(file.filename or "").suffix.lower()

    if content_type not in ALLOWED_IMAGE_TYPES or file_ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image format '{content_type}'. "
                f"Allowed formats are JPG, JPEG, and PNG."
            ),
        )

    # Guardrail: Check for prompt injection or system override attempts
    injection_warning = detect_prompt_injection(prompt)
    if injection_warning:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=injection_warning,
        )

    # Guardrail: Content safety policy check
    safety_warning = validate_content_safety(prompt)
    if safety_warning:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safety_warning,
        )

    # Read and validate image size
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file exceeds maximum limit of {MAX_IMAGE_SIZE // (1024 * 1024)} MB.",
        )

    # Validate image can be parsed and decoded by Pillow
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file. Could not decode image.",
        )

    if not gemini_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API is not configured. Add your GEMINI_API_KEY to the .env file.",
        )

    # Save image securely into uploads directory
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = f"{uuid.uuid4().hex[:12]}_{Path(file.filename).name}"
    saved_path = settings.UPLOAD_DIR / safe_filename
    with open(saved_path, "wb") as f:
        f.write(image_bytes)

    try:
        # Manage conversation session
        if session_id:
            conv = conversation_service.get_conversation(db, session_id)
            if not conv:
                conv = conversation_service.create_conversation(
                    db, title="Image Chat", conversation_id=session_id
                )
        else:
            conv = conversation_service.create_conversation(db, title="Image Chat")

        active_session_id = conv.id

        # Persist user prompt with image reference in conversation history
        user_message_content = f"📷 [Image: {file.filename}]\n\n{prompt}"
        conversation_service.add_message(
            db=db,
            conversation_id=active_session_id,
            role="user",
            content=user_message_content,
        )

        # Query Gemini Vision model
        reply_text = gemini_service.generate_vision_response(
            prompt=prompt,
            image_bytes=image_bytes,
            mime_type=content_type,
            system_instruction=system_instruction,
            temperature=temperature,
        )

        # Persist AI response
        conversation_service.add_message(
            db=db,
            conversation_id=active_session_id,
            role="assistant",
            content=reply_text,
        )

        return ChatResponse(
            reply=reply_text,
            session_id=active_session_id,
            model=settings.GEMINI_MODEL,
        )

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(re))
    except Exception as e:
        logger.error(f"Vision endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision processing failed: {str(e)}",
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
