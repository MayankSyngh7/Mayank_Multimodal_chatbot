import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.schemas.rag import (
    DocumentUploadResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SourceReference,
)
from backend.app.services.rag_service import rag_service
from backend.app.services.conversation_service import conversation_service
from backend.app.services.gemini_service import gemini_service
from backend.app.db.database import get_db
from backend.app.core.config import settings
from backend.app.core.security import detect_prompt_injection, validate_content_safety, MAX_PDF_SIZE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG Document QA"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload, Parse, and Vector-Index a PDF Document",
    description="Extracts text from a PDF, splits into chunks with page numbers, and stores vector embeddings.",
)
async def upload_pdf_document(
    file: UploadFile = File(..., description="PDF document to index for RAG QA"),
) -> DocumentUploadResponse:
    """Uploads, validates, extracts text from, and indexes a PDF document."""
    # 1. Validate file extension and MIME type
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents (.pdf) are supported.",
        )

    # 2. Read and validate file size
    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF file is empty.",
        )
    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_PDF_SIZE // (1024 * 1024)} MB.",
        )

    # 3. Extract text from PDF
    try:
        pages = rag_service.extract_text_from_pdf(pdf_bytes)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse PDF document: {str(e)}",
        )

    # 4. Chunk document pages preserving page numbers
    chunks = rag_service.chunk_pages(pages, doc_name=filename)

    # 5. Save copy to uploads directory
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = settings.UPLOAD_DIR / filename
    with open(saved_path, "wb") as f:
        f.write(pdf_bytes)

    # 6. Generate embeddings and save to vector store
    try:
        indexed_count = rag_service.index_document(doc_name=filename, chunks=chunks)
    except Exception as e:
        logger.error(f"Failed to index document chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings and index document: {str(e)}",
        )

    return DocumentUploadResponse(
        filename=filename,
        page_count=len(pages),
        chunk_count=indexed_count,
        status="indexed",
        message=f"Document '{filename}' successfully parsed and indexed ({len(pages)} pages, {indexed_count} chunks).",
    )


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Uploaded Document with RAG",
    description="Retrieves relevant document excerpts, generates a grounded answer, and returns citations.",
)
async def query_document(
    request: RAGQueryRequest,
    db: Session = Depends(get_db),
) -> RAGQueryResponse:
    """Answers user questions strictly using retrieved document context."""
    # Guardrails: Check prompt injection & content safety
    injection_warning = detect_prompt_injection(request.question)
    if injection_warning:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=injection_warning,
        )

    safety_warning = validate_content_safety(request.question)
    if safety_warning:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safety_warning,
        )

    if not gemini_service.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API is not configured. Add your GEMINI_API_KEY to the .env file.",
        )

    # 1. Manage conversation session
    if request.session_id:
        conv = conversation_service.get_conversation(db, request.session_id)
        if not conv:
            conv = conversation_service.create_conversation(
                db, title="PDF Q&A", conversation_id=request.session_id
            )
    else:
        conv = conversation_service.create_conversation(db, title="PDF Q&A")

    session_id = conv.id

    # 2. Persist user question into conversation history
    user_prompt_display = (
        f"📄 [Document Q&A: {request.doc_name}]\n\n{request.question}"
        if request.doc_name
        else f"📄 [Document Q&A]\n\n{request.question}"
    )
    conversation_service.add_message(
        db=db,
        conversation_id=session_id,
        role="user",
        content=user_prompt_display,
    )

    # 3. Retrieve context and generate grounded answer
    try:
        rag_result = rag_service.answer_question(
            question=request.question,
            doc_name=request.doc_name,
            top_k=request.top_k or 4,
            temperature=request.temperature or 0.3,
        )
    except Exception as e:
        logger.error(f"RAG query execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer from document context: {str(e)}",
        )

    answer_text = rag_result["answer"]
    raw_sources = rag_result.get("sources", [])

    sources = [
        SourceReference(
            doc_name=s["doc_name"],
            page=s["page"],
            snippet=s["snippet"],
        )
        for s in raw_sources
    ]

    # 4. Persist AI answer into conversation history
    conversation_service.add_message(
        db=db,
        conversation_id=session_id,
        role="assistant",
        content=answer_text,
    )

    return RAGQueryResponse(
        answer=answer_text,
        session_id=session_id,
        sources=sources,
        model=settings.GEMINI_MODEL,
    )


@router.get(
    "/documents",
    summary="List Uploaded Documents",
    description="Returns a list of all PDF documents stored in the system.",
)
def list_documents():
    """Lists all uploaded PDF documents available for querying."""
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = [f.name for f in settings.UPLOAD_DIR.glob("*.pdf")]
    return {"documents": pdf_files, "count": len(pdf_files)}
