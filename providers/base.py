"""Common interface every AI provider adapter implements.

To add a new provider, create providers/<name>_provider.py with a class that
implements AIProvider (and a ChatSession implementation for create_chat()),
then register it in providers/__init__.py's get_provider(). See
CONTRIBUTING.md for a worked example.
"""

from abc import ABC, abstractmethod


class ChatSession(ABC):
    """A stateful, multi-turn conversation with one model."""

    @abstractmethod
    async def send(self, text: str, image: tuple[bytes, str] | None = None) -> str:
        """Send a message (with an optional (bytes, mime_type) image) and
        return the model's reply text."""


class AIProvider(ABC):
    """A swappable backend for chat, one-off completions, and embeddings."""

    name: str = "base"
    supports_embeddings: bool = False

    @abstractmethod
    def create_chat(self, system_prompt: str) -> ChatSession:
        """Start a new stateful chat session with the given system prompt."""

    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        """One-off, non-chat text completion (e.g. reminder time parsing)."""

    @abstractmethod
    async def generate_json(self, prompt: str) -> dict:
        """One-off completion constrained to return a single JSON object
        (e.g. intent classification)."""

    async def embed(self, text: str) -> list[float] | None:
        """Return an embedding vector for `text`, or None if this provider
        doesn't support embeddings (callers must check supports_embeddings
        first and skip embedding-dependent features otherwise)."""
        return None
