import re
import time
import logging
from collections import defaultdict
from typing import Optional, Tuple

logger = logging.getLogger("chatbot.security")

# Maximum input limits
MAX_MESSAGE_LENGTH = 4000
MAX_IMAGE_SIZE = 10 * 1024 * 1024    # 10 MB
MAX_PDF_SIZE = 25 * 1024 * 1024      # 25 MB
MAX_AUDIO_SIZE = 25 * 1024 * 1024    # 25 MB

# Known Prompt Injection / System Override heuristic patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions?",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions?",
    r"override\s+(?:system|safety)\s+(?:rules|prompt|instruction)",
    r"system\s+override",
    r"you\s+are\s+now\s+(?:in\s+)?(?:developer|dan|unrestricted|god)\s+mode",
    r"(?:print|reveal|output|display|show)\s+(?:your\s+)?(?:initial\s+|original\s+)?(?:system\s+prompt|instructions?|secret|api\s+key)",
    r"jailbreak",
]
COMPILED_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]

# Content safety violation patterns (harmful/malicious queries)
UNSAFE_CONTENT_PATTERNS = [
    r"how\s+to\s+(?:make|build|synthesize)\s+(?:a\s+)?(?:bomb|explosive|biological\s+weapon)",
    r"write\s+(?:a\s+)?(?:ransomware|trojan|keylogger|exploit\s+code)",
]
COMPILED_UNSAFE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in UNSAFE_CONTENT_PATTERNS]


def detect_prompt_injection(text: str) -> Optional[str]:
    """
    Heuristically checks text for direct prompt injection or system override attempts.
    Returns warning message if flagged, or None if clean.
    """
    if not text:
        return None
    for pattern in COMPILED_INJECTION_PATTERNS:
        if pattern.search(text):
            return "Security Guardrail: Potential prompt injection or system override attempt detected."
    return None


def validate_content_safety(text: str) -> Optional[str]:
    """
    Checks for high-risk safety violations.
    Returns warning message if flagged, or None if clean.
    """
    if not text:
        return None
    for pattern in COMPILED_UNSAFE_PATTERNS:
        if pattern.search(text):
            return "Content Policy: Request involves potentially harmful or malicious material."
    return None


def sanitize_untrusted_document_context(chunk_text: str) -> str:
    """
    Sanitizes retrieved PDF document chunks to mitigate indirect prompt injection.
    Neutralizes role-hijacking tokens like 'System:', 'Assistant:', or 'User:'.
    """
    if not chunk_text:
        return ""
    # Neutralize artificial prompt delimiters
    sanitized = re.sub(
        r"(?i)\b(system\s*:|assistant\s*:|human\s*:|user\s*:|instructions?\s*:)",
        r"[\1]",
        chunk_text,
    )
    return sanitized


class InMemoryRateLimiter:
    """
    Sliding window in-memory rate limiter per IP address.
    Zero external dependencies, fast and thread-safe for local and small-scale deployments.
    """

    def __init__(self, requests_per_minute: int = 120):
        self.limit = requests_per_minute
        self.window = 60.0  # seconds
        self._requests = defaultdict(list)

    def check_rate_limit(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if client_ip has exceeded allowed requests.
        Returns (is_allowed, remaining_requests).
        """
        now = time.time()
        timestamps = self._requests[client_ip]

        # Clean expired timestamps older than window
        self._requests[client_ip] = [t for t in timestamps if now - t < self.window]

        if len(self._requests[client_ip]) >= self.limit:
            return False, 0

        self._requests[client_ip].append(now)
        remaining = self.limit - len(self._requests[client_ip])
        return True, remaining

    def reset(self):
        """Reset rate limiter state (useful for automated testing)."""
        self._requests.clear()


# Global rate limiter instance (120 requests/minute per client IP)
rate_limiter = InMemoryRateLimiter(requests_per_minute=120)
