# Multimodal AI Chatbot

A modular, extensible Multimodal AI Chatbot built with **Python**, **Streamlit**, **FastAPI**, **Google Gemini**, **SQLite/SQLAlchemy**, **RAG (Retrieval-Augmented Generation)**, **Image Understanding**, and **Voice I/O**.

---

## 🌟 Key Features (Phased Roadmap)

- 💬 **Conversational AI**: Context-aware chat powered by the Google Gemini API.
- 🗄️ **Persistent Chat History**: SQLite database managed with SQLAlchemy ORM to save and restore chat sessions.
- 📄 **Document RAG (Retrieval-Augmented Generation)**: Upload PDF documents, chunk text, generate vector embeddings, and ask questions with cited sources.
- 🖼️ **Image Understanding**: Upload and analyze images, extract insights, and converse about visual inputs.
- 🎙️ **Voice Input & Text-to-Speech (TTS)**: Speak your prompts via audio input and listen to the AI's responses with natural text-to-speech.
- ⚡ **Decoupled Architecture**: Clean separation between the Streamlit user interface and the FastAPI backend service.
- 🍏 **Apple Silicon & macOS Optimized**: Lightweight, pure-Python compatible dependencies avoiding problematic C-library compilation issues.

---

## 🏗️ Project Architecture & Directory Structure

```
Mayank_Multimodal_chatbot/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py              # FastAPI application entrypoint & middleware
│       ├── api/                 # REST API route handlers
│       │   ├── __init__.py
│       │   ├── chat.py          # Chatbot text & multimodal endpoints
│       │   ├── rag.py           # PDF upload & RAG query endpoints
│       │   └── audio.py         # Voice processing & TTS endpoints
│       ├── core/                # Core settings & configuration
│       │   ├── __init__.py
│       │   └── config.py        # Environment variables & settings loader
│       ├── db/                  # Database setup & models
│       │   ├── __init__.py
│       │   ├── database.py      # SQLAlchemy connection & session manager
│       │   └── models.py        # Session & Message ORM models
│       ├── schemas/             # Pydantic validation schemas
│       │   ├── __init__.py
│       │   ├── chat.py          # Request/response schemas for chat
│       │   └── rag.py           # Request/response schemas for RAG
│       └── services/            # Business logic & external AI integrations
│           ├── __init__.py
│           ├── gemini_service.py# Google Gemini API (Text, Vision, Audio)
│           ├── rag_service.py   # PDF text extraction & vector retrieval
│           └── audio_service.py # Text-to-Speech (gTTS) & audio handling
├── frontend/
│   ├── app.py                   # Streamlit main dashboard application
│   ├── components/              # Reusable UI widgets
│   │   ├── __init__.py
│   │   ├── chat_view.py         # Chat conversation history & input
│   │   ├── file_upload.py       # PDF & Image upload widgets
│   │   └── voice_view.py        # Voice recorder & audio player widgets
│   └── utils/                   # Frontend utilities
│       ├── __init__.py
│       └── api_client.py        # Backend API HTTP client
├── data/
│   ├── uploads/                 # Local directory for uploaded files (.gitignored)
│   │   └── .gitkeep
│   └── vectorstore/             # Local directory for vector database (.gitignored)
│       └── .gitkeep
├── .env.example                 # Template for environment variables
├── .gitignore                   # Git ignore file protecting keys & databases
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Step-by-Step Implementation Plan

| Phase | Milestone | Description |
|---|---|---|
| **Phase 1** | **Foundation & Setup** | Project structure, environment configuration, and dependency management *(Completed)* |
| **Phase 2** | **FastAPI Backend & Gemini Integration** | FastAPI setup, configuration loading, basic Gemini text chat endpoint *(Completed)* |
| **Phase 3** | **Database & Chat History Persistence** | SQLite + SQLAlchemy ORM models, session management, and conversation storage *(Completed)* |
| **Phase 4** | **Streamlit Frontend** | Clean chat interface, sidebar for session history, and backend API integration *(Completed)* |
| **Phase 5** | **Multimodal: Vision & Voice** | Image analysis with Gemini Vision, voice recording input, and gTTS audio output |
| **Phase 6** | **RAG for PDF Question Answering** | PDF document parsing, vector indexing with ChromaDB, and context-retrieved answering |

---

## 🛠️ Getting Started (macOS M1 / Apple Silicon)

### 1. Prerequisites
- macOS (Apple Silicon M1 / M2 / M3)
- Python 3.9+ installed

### 2. Create and Activate Virtual Environment
```bash
# In the project root directory
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
```
Open `.env` and insert your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 5. Running the Application (Coming in subsequent phases)
- **Start FastAPI Backend**:
  ```bash
  uvicorn backend.app.main:app --reload --port 8000
  ```
- **Start Streamlit Frontend**:
  ```bash
  streamlit run frontend/app.py
  ```

---

## 🔒 Security & Best Practices
- **API Keys**: Never hardcoded. Loaded securely from `.env` using `pydantic-settings` and `python-dotenv`.
- **Git Safety**: `.env` and `*.db` are strictly excluded in `.gitignore` to prevent secret leaks to GitHub.
