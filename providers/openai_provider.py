"""OpenAI adapter — chat, structured JSON, and embeddings.

text-embedding-3-small/large support a `dimensions` parameter that truncates
the output vector, so we request 768 dims to stay compatible with the
vector(768) column used by the Gemini provider's schema.
"""

import asyncio
import base64
import json

from openai import OpenAI

from .base import AIProvider, ChatSession

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 768  # matches the vector(768) column in supabase_schema.sql


class OpenAIChatSession(ChatSession):
    def __init__(self, client: OpenAI, model: str, system_prompt: str):
        self._client = client
        self._model = model
        self._history: list[dict] = [{"role": "system", "content": system_prompt}]

    async def send(self, text: str, image: tuple[bytes, str] | None = None) -> str:
        content: list[dict] = [{"type": "text", "text": text}]
        if image:
            data, mime_type = image
            b64 = base64.b64encode(data).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            })

        self._history.append({"role": "user", "content": content})

        response = await asyncio.to_thread(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=self._history,
                temperature=0.7,
            )
        )
        reply = response.choices[0].message.content.strip()
        self._history.append({"role": "assistant", "content": reply})
        return reply


class OpenAIProvider(AIProvider):
    name = "openai"
    supports_embeddings = True

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def create_chat(self, system_prompt: str) -> ChatSession:
        return OpenAIChatSession(self._client, self._model, system_prompt)

    async def generate_text(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        return response.choices[0].message.content.strip()

    async def generate_json(self, prompt: str) -> dict:
        response = await asyncio.to_thread(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        )
        return json.loads(response.choices[0].message.content.strip())

    async def embed(self, text: str) -> list[float] | None:
        response = await asyncio.to_thread(
            lambda: self._client.embeddings.create(
                model=EMBED_MODEL,
                input=text,
                dimensions=EMBED_DIMENSIONS,
            )
        )
        return response.data[0].embedding
