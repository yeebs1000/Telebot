"""Google Gemini adapter — chat, structured JSON, and native embeddings."""

import asyncio
import json

from google import genai
from google.genai import types

from .base import AIProvider, ChatSession

EMBED_MODEL = "text-embedding-004"  # native 768-dim, matches supabase_schema.sql


class GeminiChatSession(ChatSession):
    def __init__(self, chat):
        self._chat = chat

    async def send(self, text: str, image: tuple[bytes, str] | None = None) -> str:
        contents = []
        if image:
            data, mime_type = image
            contents.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        contents.append(text)
        response = await asyncio.to_thread(lambda: self._chat.send_message(contents))
        return response.text.strip()


class GeminiProvider(AIProvider):
    name = "gemini"
    supports_embeddings = True

    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def create_chat(self, system_prompt: str) -> ChatSession:
        chat = self._client.chats.create(
            model=self._model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            ),
        )
        return GeminiChatSession(chat)

    async def generate_text(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            lambda: self._client.models.generate_content(model=self._model, contents=prompt)
        )
        return response.text.strip()

    async def generate_json(self, prompt: str) -> dict:
        response = await asyncio.to_thread(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
        )
        return json.loads(response.text.strip())

    async def embed(self, text: str) -> list[float] | None:
        response = await asyncio.to_thread(
            lambda: self._client.models.embed_content(model=EMBED_MODEL, contents=text)
        )
        return response.embeddings[0].values
