import io
import pytest
from unittest.mock import patch, MagicMock

from backend.app.services.rag_service import rag_service
from backend.app.schemas.rag import RAGQueryRequest


def test_chunk_pages_preserves_page_metadata():
    """Test chunking logic splits text and retains document name and page number."""
    pages_data = [
        {"page": 1, "text": "Page one text. " * 50},  # ~750 chars
        {"page": 2, "text": "Page two text. " * 30},  # ~450 chars
    ]
    chunks = rag_service.chunk_pages(pages_data, doc_name="manual.pdf", chunk_size=400, chunk_overlap=100)
    
    assert len(chunks) >= 3
    # Verify page tracking
    page_numbers = {c["page"] for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert all(c["doc_name"] == "manual.pdf" for c in chunks)
    assert all("text" in c for c in chunks)


def test_extract_text_empty_pdf_raises_error():
    """Test extract_text_from_pdf raises ValueError when PDF has no text."""
    import pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(ValueError) as excinfo:
        rag_service.extract_text_from_pdf(buf.getvalue())
    assert "No readable text found" in str(excinfo.value)


def test_rag_upload_rejects_non_pdf(client):
    """Test POST /api/rag/upload rejects non-PDF file formats."""
    response = client.post(
        "/api/rag/upload",
        files={"file": ("notes.docx", b"Word document bytes", "application/msword")},
    )
    assert response.status_code == 400
    assert "Only PDF documents (.pdf) are supported" in response.json()["detail"]


def test_rag_upload_successful_indexing(client):
    """Test POST /api/rag/upload parses and indexes chunks."""
    fake_pages = [
        {"page": 1, "text": "Chapter 1: Artificial intelligence and modern search algorithms."},
        {"page": 2, "text": "Chapter 2: Retrieval-Augmented Generation improves accuracy."},
    ]

    with patch.object(
        rag_service, "extract_text_from_pdf", return_value=fake_pages
    ), patch.object(
        rag_service, "index_document", return_value=2
    ):
        response = client.post(
            "/api/rag/upload",
            files={"file": ("guide.pdf", b"%PDF-1.4...", "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "guide.pdf"
        assert data["page_count"] == 2
        assert data["chunk_count"] == 2
        assert data["status"] == "indexed"


def test_rag_query_endpoint(client):
    """Test POST /api/rag/query generates grounded answers with sources."""
    mock_rag_result = {
        "answer": "RAG combines retrieval with generative models to reduce hallucinations [guide.pdf, p.2].",
        "sources": [
            {"doc_name": "guide.pdf", "page": 2, "snippet": "Retrieval-Augmented Generation improves accuracy..."}
        ],
    }

    with patch(
        "backend.app.services.gemini_service.gemini_service.is_configured",
        return_value=True,
    ), patch.object(
        rag_service, "answer_question", return_value=mock_rag_result
    ):
        response = client.post(
            "/api/rag/query",
            json={
                "question": "What is RAG?",
                "doc_name": "guide.pdf",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "RAG combines retrieval" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["doc_name"] == "guide.pdf"
        assert data["sources"][0]["page"] == 2
        assert data["session_id"] is not None

        # Verify Q&A was recorded in conversation history
        conv_resp = client.get(f"/api/conversations/{data['session_id']}")
        assert conv_resp.status_code == 200
        conv_data = conv_resp.json()
        assert len(conv_data["messages"]) == 2
        assert "What is RAG?" in conv_data["messages"][0]["content"]
        assert "RAG combines retrieval" in conv_data["messages"][1]["content"]


def test_rag_list_documents(client):
    """Test GET /api/rag/documents lists uploaded documents."""
    response = client.get("/api/rag/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "count" in data
