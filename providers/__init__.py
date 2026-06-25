"""Provider registry — picks an AIProvider implementation based on AI_PROVIDER."""

import os

from .base import AIProvider, ChatSession

_instances: dict[str, AIProvider] = {}


def get_provider(name: str) -> AIProvider:
    """Return a cached AIProvider instance for the given name (gemini/openai/claude).

    Reads that provider's API key (and optional AI_MODEL override) from the
    environment. Raises ValueError if the name is unknown or its required
    API key isn't set.
    """
    name = (name or "gemini").lower()

    if name in _instances:
        return _instances[name]

    model_override = os.getenv("AI_MODEL") or None

    if name == "gemini":
        from .gemini_provider import GeminiProvider
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("AI_PROVIDER=gemini requires GEMINI_API_KEY to be set.")
        kwargs = {"model": model_override} if model_override else {}
        provider = GeminiProvider(api_key, **kwargs)

    elif name == "openai":
        from .openai_provider import OpenAIProvider
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AI_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        kwargs = {"model": model_override} if model_override else {}
        provider = OpenAIProvider(api_key, **kwargs)

    elif name == "claude":
        from .claude_provider import ClaudeProvider
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("AI_PROVIDER=claude requires ANTHROPIC_API_KEY to be set.")
        kwargs = {"model": model_override} if model_override else {}
        provider = ClaudeProvider(api_key, **kwargs)

    else:
        raise ValueError(f"Unknown AI_PROVIDER '{name}'. Supported: gemini, openai, claude.")

    _instances[name] = provider
    return provider


__all__ = ["AIProvider", "ChatSession", "get_provider"]
