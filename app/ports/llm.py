"""Port: LLM client.

The `LLMClient` Protocol describes the surface the service layer may call.
Adapters (`app/adapters/llm_skainet.py`, `app/adapters/in_memory.py`) implement
it. For PR1 the existing `app/core/llm.py` module-level functions still carry
the real implementation; this Protocol exists so services *can* be typed
against it and so the in-memory fake (used in tests) is a structural subtype.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    async def ocr_image(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        mime: str = "image/png",
        max_tokens: int = 2000,
    ) -> str: ...

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 1200,
    ) -> dict[str, Any]: ...

    async def chat_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 800,
    ) -> str: ...
