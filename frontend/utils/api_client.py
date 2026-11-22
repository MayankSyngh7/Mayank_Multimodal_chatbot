import os
import requests
from typing import Optional, Dict, Any, List

# Default backend URL (configurable via BACKEND_API_URL environment variable)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")


class APIClient:
    """
    Client for interacting with the FastAPI backend service.
    Encapsulates all REST API calls and handles network errors gracefully.
    """

    def __init__(self, base_url: str = BACKEND_API_URL, timeout: int = 45):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Check backend connectivity and configuration status.
        Returns health status dict or error info.
        """
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return {"connected": True, "data": resp.json(), "error": None}
            return {
                "connected": True,
                "data": None,
                "error": f"Server returned HTTP {resp.status_code}: {resp.text}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "connected": False,
                "data": None,
                "error": "Could not connect to FastAPI backend. Ensure the backend server is running on port 8000.",
            }
        except Exception as e:
            return {"connected": False, "data": None, "error": str(e)}

    def list_conversations(self) -> List[Dict[str, Any]]:
        """Fetch list of all saved conversations."""
        try:
            resp = requests.get(f"{self.base_url}/api/conversations", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific conversation with all message history."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/conversations/{conversation_id}",
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def create_conversation(self, title: str = "New Conversation") -> Optional[Dict[str, Any]]:
        """Create a new conversation session."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/conversations",
                json={"title": title},
                timeout=self.timeout,
            )
            if resp.status_code == 201:
                return resp.json()
            return None
        except Exception:
            return None

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        try:
            resp = requests.delete(
                f"{self.base_url}/api/conversations/{conversation_id}",
                timeout=self.timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def send_chat_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Send a chat message to Gemini via the backend.
        Returns response dict with 'reply' and 'session_id', or an error dict.
        """
        payload = {
            "message": message,
            "session_id": session_id,
            "system_instruction": system_instruction,
            "temperature": temperature,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )

            if resp.status_code == 200:
                return {"success": True, "data": resp.json(), "error": None}

            # Handle structured error response
            try:
                error_detail = resp.json().get("detail", resp.text)
            except Exception:
                error_detail = resp.text

            return {
                "success": False,
                "data": None,
                "error": f"Error ({resp.status_code}): {error_detail}",
            }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "data": None,
                "error": "Connection Error: Cannot reach FastAPI backend. Please ensure the backend server is running.",
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "data": None,
                "error": "Timeout Error: Gemini API took too long to respond. Please try again.",
            }
        except Exception as e:
            return {"success": False, "data": None, "error": f"Unexpected error: {str(e)}"}


# Global API client instance
api_client = APIClient()
