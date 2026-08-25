"""Adapter: skainet LLM client (wraps `app/core/llm.py`).

`SkainetLLM` is a thin object that delegates to the existing module-level
functions in `app/core/llm.py`. It satisfies the `LLMClient` Protocol from
`app/ports/llm.py`. For PR1 the functions still read settings at module
import; PR2 will move settings into the constructor.

A service that wants to depend on the port instead of the legacy module can
ask for `SkainetLLM` via FastAPI's `Depends`. Existing services continue to
call `app.core.llm.*` directly — both paths work during the migration.
"""

from __future__ import annotations

from typing import Any

from app.core import llm as _llm
from app.core.config import get_settings


class SkainetLLM:
    """Adapter that exposes `app/core/llm.py` as an `LLMClient`-shaped object."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings or get_settings()

    async def ocr_image(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        mime: str = "image/png",
        max_tokens: int = 2000,
    ) -> str:
        return await _llm.ocr_image(
            image_bytes, prompt=prompt, mime=mime, max_tokens=max_tokens
        )

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        return await _llm.chat_json(
            system,
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 800,
    ) -> str:
        return await _llm.chat_text(
            system,
            user,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_llm_client() -> SkainetLLM:
    """FastAPI dependency: return the skainet LLM adapter."""
    return SkainetLLM()
