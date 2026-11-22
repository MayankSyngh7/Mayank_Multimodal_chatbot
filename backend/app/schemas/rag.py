from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Metadata response returned after a PDF is uploaded, parsed, and indexed."""
    filename: str
    page_count: int
    chunk_count: int
    status: str = "indexed"
    message: str


class SourceReference(BaseModel):
    """Citation reference for a retrieved document chunk."""
    doc_name: str
    page: int
    snippet: str


class RAGQueryRequest(BaseModel):
    """Payload to query indexed PDF documents using RAG."""
    question: str = Field(
        ...,
        min_length=1,
        description="The question to ask about the uploaded document(s)",
        json_schema_extra={"example": "What are the main findings in Section 3?"}
    )
    session_id: Optional[str] = Field(
        None,
        description="Active conversation session ID for persistence"
    )
    doc_name: Optional[str] = Field(
        None,
        description="Optional filter to search only a specific document"
    )
    top_k: Optional[int] = Field(
        4,
        ge=1,
        le=10,
        description="Number of most relevant chunks to retrieve"
    )
    temperature: Optional[float] = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for answer generation (defaults to 0.3 for grounded precision)"
    )


class RAGQueryResponse(BaseModel):
    """Answer response with document citations and sources."""
    answer: str
    session_id: Optional[str] = None
    sources: List[SourceReference] = []
    model: str
