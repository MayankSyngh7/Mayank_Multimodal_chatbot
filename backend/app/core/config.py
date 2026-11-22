import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory: absolute path to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """
    Application Settings
    Loads configuration from environment variables or a local .env file.
    Keeps API keys and sensitive settings out of source code.
    """
    APP_NAME: str = "Multimodal AI Chatbot API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Server Configuration
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    BACKEND_API_URL: str = "http://127.0.0.1:8000"

    # Google Gemini Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Database Configuration (SQLite)
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/chatbot.db"

    # Storage Directories
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    VECTOR_DB_DIR: Path = BASE_DIR / "data" / "vectorstore"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def is_gemini_configured(self) -> bool:
        """Returns True if a valid-looking Gemini API key is set."""
        return bool(
            self.GEMINI_API_KEY 
            and self.GEMINI_API_KEY.strip() 
            and self.GEMINI_API_KEY != "your_gemini_api_key_here"
        )


settings = Settings()
