import io
import logging
from typing import Optional
from PIL import Image
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Service layer handling all interactions with the Google Gemini API.
    Provides text generation, multimodal vision understanding, and audio transcription.
    """

    def __init__(self):
        self._initialized = False
        self._setup_client()

    def _setup_client(self):
        """Configures the Gemini SDK with the API key if available."""
        if settings.is_gemini_configured():
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._initialized = True
                logger.info("Google Gemini SDK initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to configure Gemini SDK: {e}")
                self._initialized = False
        else:
            logger.warning(
                "GEMINI_API_KEY not configured or set to placeholder. "
                "Set GEMINI_API_KEY in .env to enable live AI responses."
            )
            self._initialized = False

    def is_configured(self) -> bool:
        """Check if the Gemini client is configured and ready."""
        if not self._initialized and settings.is_gemini_configured():
            self._setup_client()
        return self._initialized

    def generate_chat_response(
        self,
        message: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        model_name: Optional[str] = None,
    ) -> str:
        """Generates a text response using Google Gemini."""
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY "
                "in your .env file to enable live AI responses."
            )

        active_model = model_name or settings.GEMINI_MODEL

        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
            )

            model = genai.GenerativeModel(
                model_name=active_model,
                system_instruction=system_instruction if system_instruction else None,
                generation_config=generation_config,
            )

            response = model.generate_content(message)

            if response.text:
                return response.text
            return "The model did not return any text. Please try asking again."

        except GoogleAPIError as e:
            logger.error(f"Google Gemini API error: {e}")
            raise RuntimeError(f"Gemini API communication error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in Gemini service: {e}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def generate_vision_response(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Generates a multimodal response for image understanding using Gemini Vision.

        Args:
            prompt: Question or instruction regarding the image.
            image_bytes: Raw bytes of the image file.
            mime_type: MIME type of the image ('image/jpeg', 'image/png').
            system_instruction: Optional system persona prompt.
            temperature: Creativity control.
            model_name: Model name.

        Returns:
            AI-generated response describing or answering questions about the image.
        """
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY "
                "in your .env file to enable vision understanding."
            )

        active_model = model_name or settings.GEMINI_MODEL

        try:
            # Load and validate image using Pillow
            image = Image.open(io.BytesIO(image_bytes))

            generation_config = genai.types.GenerationConfig(temperature=temperature)

            model = genai.GenerativeModel(
                model_name=active_model,
                system_instruction=system_instruction if system_instruction else None,
                generation_config=generation_config,
            )

            # Gemini accepts a list of [prompt, image]
            response = model.generate_content([prompt, image])

            if response.text:
                return response.text
            return "The model could not analyze the image. Please try another image."

        except GoogleAPIError as e:
            logger.error(f"Gemini Vision API error: {e}")
            raise RuntimeError(f"Gemini Vision API error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise RuntimeError(f"Failed to analyze image: {str(e)}")

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/wav",
        model_name: Optional[str] = None,
    ) -> str:
        """
        Transcribes spoken audio into text using Google Gemini's native audio capabilities.

        Args:
            audio_bytes: Raw bytes of the audio file.
            mime_type: MIME type of the audio ('audio/wav', 'audio/mp3', 'audio/ogg', etc.)
            model_name: Model name.

        Returns:
            Clean transcribed text string.
        """
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY "
                "in your .env file to enable voice transcription."
            )

        active_model = model_name or settings.GEMINI_MODEL

        try:
            model = genai.GenerativeModel(model_name=active_model)

            audio_part = {
                "mime_type": mime_type,
                "data": audio_bytes,
            }

            transcription_prompt = (
                "Listen to this audio recording carefully and transcribe all spoken words into plain text. "
                "Do NOT add introductory or concluding remarks, explanations, or quotes. "
                "Return ONLY the exact spoken transcription."
            )

            response = model.generate_content([audio_part, transcription_prompt])

            if response.text:
                return response.text.strip()
            return "No audible speech was recognized."

        except GoogleAPIError as e:
            logger.error(f"Gemini Audio API error: {e}")
            raise RuntimeError(f"Gemini Audio error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise RuntimeError(f"Audio transcription error: {str(e)}")


# Singleton instance for application-wide use
gemini_service = GeminiService()
