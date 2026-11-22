import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import settings
from backend.app.core.security import rate_limiter
from backend.app.schemas.chat import HealthCheckResponse
from backend.app.api.chat import router as chat_router
from backend.app.api.conversations import router as conversations_router
from backend.app.api.audio import router as audio_router
from backend.app.api.rag import router as rag_router
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
    Initializes database tables and storage folders on application startup.
    """
    logger.info("Initializing database tables...")
    init_db()
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Database and storage directories initialized successfully.")
    yield
    logger.info("Application shutting down.")


# Initialize FastAPI application with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="FastAPI Backend for Multimodal AI Chatbot with Vision, Voice, and PDF RAG.",
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


@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """
    Enforces in-memory sliding-window rate limiting on all /api endpoints.
    Returns HTTP 429 when client requests exceed quota.
    """
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "127.0.0.1"
        allowed, remaining = rate_limiter.check_rate_limit(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Maximum 120 requests per minute allowed. Please wait a moment."
                },
                headers={"Retry-After": "60"},
            )
    return await call_next(request)


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """
    Masks internal database errors to prevent leaking schema details or stack traces.
    """
    logger.error(f"Database error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "A database error occurred. Your request could not be processed."},
    )

# Serve uploaded files statically for image rendering
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Register API Routers
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(audio_router)
app.include_router(rag_router)


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
