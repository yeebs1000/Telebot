"""Anthropic Claude adapter — chat and prompted JSON.

Anthropic has no embeddings API, so supports_embeddings is False: the bot
automatically skips background message embedding and disables the semantic
memory-search feature while AI_PROVIDER=claude. Structured JSON output is
achieved by instructing the model directly, since Claude has no dedicated
JSON response-format flag.
"""

import asyncio
import base64
import json

from anthropic import Anthropic

from .base import AIProvider, ChatSession


class ClaudeChatSession(ChatSession):
    def __init__(self, client: Anthropic, model: str, system_prompt: str):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._history: list[dict] = []

    async def send(self, text: str, image: tuple[bytes, str] | None = None) -> str:
        content: list[dict] = []
        if image:
            data, mime_type = image
            b64 = base64.b64encode(data).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": b64},
            })
        content.append({"type": "text", "text": text})

        self._history.append({"role": "user", "content": content})

        response = await asyncio.to_thread(
            lambda: self._client.messages.create(
                model=self._model,
                system=self._system_prompt,
                messages=self._history,
                max_tokens=1024,
                temperature=0.7,
            )
        )
        reply = response.content[0].text.strip()
        self._history.append({"role": "assistant", "content": reply})
        return reply


class ClaudeProvider(AIProvider):
    name = "claude"
    supports_embeddings = False

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def create_chat(self, system_prompt: str) -> ChatSession:
        return ClaudeChatSession(self._client, self._model, system_prompt)

    async def generate_text(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            lambda: self._client.messages.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
        )
        return response.content[0].text.strip()

    async def generate_json(self, prompt: str) -> dict:
        json_prompt = (
            f"{prompt}\n\nRespond with ONLY a raw JSON object — no markdown code "
            f"fences, no explanation, no extra text."
        )
        raw = await self.generate_text(json_prompt)
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
