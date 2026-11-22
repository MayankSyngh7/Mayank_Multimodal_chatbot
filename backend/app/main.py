import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.schemas.chat import HealthCheckResponse
from backend.app.api.chat import router as chat_router

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot.backend")

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FastAPI Backend for Multimodal AI Chatbot with Google Gemini integration.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS)
# Allows the Streamlit frontend (running on e.g. localhost:8501) to interact with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(chat_router)


@app.get(
    "/",
    summary="Root Endpoint",
    description="Welcome message and API service information.",
)
async def root():
    """Root endpoint welcoming visitors and linking to interactive docs."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Health Check",
    description="Checks the health of the backend API and Gemini configuration status.",
)
async def health_check() -> HealthCheckResponse:
    """Verifies server responsiveness and whether Gemini is configured."""
    return HealthCheckResponse(
        status="healthy",
        gemini_configured=settings.is_gemini_configured(),
        model=settings.GEMINI_MODEL,
        version=settings.APP_VERSION,
    )
