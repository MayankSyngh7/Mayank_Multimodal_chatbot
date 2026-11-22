import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.schemas.chat import HealthCheckResponse
from backend.app.api.chat import router as chat_router
from backend.app.api.conversations import router as conversations_router
from backend.app.db.database import init_db

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager:
    Initializes database tables on application startup.
    """
    logger.info("Initializing database tables...")
    init_db()
    logger.info("Database initialized successfully.")
    yield
    logger.info("Application shutting down.")


# Initialize FastAPI application with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FastAPI Backend for Multimodal AI Chatbot with Google Gemini & SQLite history.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(chat_router)
app.include_router(conversations_router)


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
