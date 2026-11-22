import io
import re
import logging
from gtts import gTTS

logger = logging.getLogger(__name__)


class AudioService:
    """
    Service handling Text-to-Speech (TTS) conversion using gTTS.
    Pure Python, highly compatible with Apple Silicon (M1/M2/M3) without C-dependencies.
    """

    @staticmethod
    def text_to_speech(text: str, lang: str = "en") -> io.BytesIO:
        """
        Convert text into speech MP3 bytes.
        Cleans markdown syntax and code blocks for natural listening.

        Args:
            text: Text to be converted into spoken audio.
            lang: Language code (default 'en').

        Returns:
            io.BytesIO containing MP3 audio data.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for Text-to-Speech conversion.")

        # Clean markdown code blocks and raw formatting for natural speech
        clean_text = re.sub(r'```[\s\S]*?```', ' [Code block omitted] ', text)
        clean_text = re.sub(r'`.*?`', '', clean_text)
        clean_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_text)  # Keep link text
        clean_text = re.sub(r'[#*_~>|]', '', clean_text)             # Remove markdown symbols
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if not clean_text:
            clean_text = "Here is the response from Gemini."

        # Truncate text to avoid excessively long audio synthesis
        if len(clean_text) > 3500:
            clean_text = clean_text[:3500] + "... and more."

        try:
            audio_buffer = io.BytesIO()
            tts = gTTS(text=clean_text, lang=lang, slow=False)
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer
        except Exception as e:
            logger.error(f"gTTS conversion failed: {e}")
            raise RuntimeError(f"Text-to-Speech conversion failed: {str(e)}")


audio_service = AudioService()
