"""OpenAI-compatible client for the skainet gateway.

skainet exposes an OpenAI-style /v1 endpoint. We use the `openai` SDK with a
custom base_url and api_key, which lets us switch between the OCR model and
the chat model by changing the `model` parameter per call.

Multimodal calls (OCR) follow the OpenAI vision schema:
    messages = [{"role": "user", "content": [
        {"type": "text",  "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]}]
"""

from __future__ import annotations

import base64
import json
from typing import Any, cast

from openai import AsyncOpenAI

from app.core.config import get_settings

_settings = get_settings()

# A single async client is safe to reuse across requests.
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=_settings.skainet_base_url,
            api_key=_settings.skainet_api_key or "missing",
        )
    return _client


def _image_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


async def chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """Send a text chat completion and parse the response as JSON.

    The prompt is expected to instruct the model to return strict JSON. We
    strip ``` fences if present before parsing.
    """
    client = _get_client()
    resp = await client.chat.completions.create(
        model=model or _settings.model_chat,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or "{}"
    return _parse_json(text)


async def chat_text(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 800,
) -> str:
    """Send a text chat completion and return raw text."""
    client = _get_client()
    resp = await client.chat.completions.create(
        model=model or _settings.model_chat,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def ocr_image(
    image_bytes: bytes,
    *,
    prompt: str,
    mime: str = "image/png",
    max_tokens: int = 2000,
) -> str:
    """Send an image to the OCR model and return extracted text."""
    client = _get_client()
    data_url = _image_to_data_url(image_bytes, mime)
    resp = await client.chat.completions.create(
        model=_settings.model_ocr,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def embed(text: str) -> list[float]:
    """Embed text for pgvector personalization.

    Falls back to a zero vector if the gateway has no embeddings endpoint, so
    the rest of the pipeline degrades gracefully (no personalization) rather
    than crashing.
    """
    client = _get_client()
    try:
        resp = await client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return resp.data[0].embedding
    except Exception:
        return [0.0] * 1536


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return cast(dict[str, Any], json.loads(cleaned))
    except json.JSONDecodeError:
        return {"_raw": text}
