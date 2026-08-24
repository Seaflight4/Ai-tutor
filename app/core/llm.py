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
import re
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
    return _parse_json(_strip_cot(text))


async def chat_text(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 800,
) -> str:
    """Send a text chat completion and return raw text.

    GLM models sometimes leak chain-of-thought before the actual answer,
    terminated by an explicit answer marker. We strip everything before
    that marker. If no marker is present, the text is returned unchanged.
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
    text = (resp.choices[0].message.content or "").strip()
    return _strip_cot(text)


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


# ---------------------------------------------------------------------------
# Chain-of-thought stripping
# ---------------------------------------------------------------------------
# GLM-5.2 emits reasoning before the answer in several shapes:
#   1. An explicit reasoning tag:  <think>...reasoning...</think>  (primary)
#   2. A fenced block:  ```\n<reasoning>\n```\n<answer>  (or ```markdown / ```json)
#   3. A quote-delimited block ending with "\u201d\n"
# We remove ALL <think>...</think> blocks, then fall back to fence/quote
# trimming. What remains is the clean answer.

# Matches a block (DOTALL, non-greedy). GLM uses the literal
# tag name "think" inside angle brackets.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Matches a closing fence ``` possibly followed by a language tag on the same
# line; we rfind the last ``` and take what follows.
_FENCE = re.compile(r"```")

# GLM also uses a right-double-quote + newline to end its reasoning block.
_ANSWER_MARKER = re.compile(re.escape("\u201d\n"))


def _strip_cot(text: str) -> str:
    """Remove leaked chain-of-thought from a chat completion.

    1. Drop all <think>...</think> blocks (GLM-5.2's primary format).
    2. If the result still starts with a fenced reasoning block, keep text
       after the last ``` fence.
    3. Else fall back to the GLM quote-marker.
    Returns the text stripped. If no reasoning is present, returns it as-is.
    """
    if not text:
        return text
    # 1. Remove <think>...</think> blocks (there may be more than one).
    text = _THINK_BLOCK.sub("", text)
    # 2. Fenced reasoning: keep text after the LAST ``` fence (only when there
    #    are >= 2 fences, i.e. an open+close pair).
    fence_matches = list(_FENCE.finditer(text))
    if len(fence_matches) >= 2:
        text = text[fence_matches[-1].end():]
    else:
        # 3. GLM quote-delimited reasoning: keep text after the last "\u201d\n".
        quote_matches = list(_ANSWER_MARKER.finditer(text))
        if quote_matches:
            text = text[quote_matches[-1].end():]
    return text.strip()


def _parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating reasoning + fences.

    GLM-5.2 may emit chain-of-thought before the JSON, optionally wrapping the
    JSON in a ```json fence. We locate the LAST ``` fence pair and parse what's
    inside; if that fails we try the trailing brace-balanced substring; if all
    else fails, return {"_raw": text} so callers degrade gracefully.
    """
    cleaned = text.strip()

    # 0. Strip any ilda reasoning blocks first (belt-and-suspenders; _strip_cot
    #    already did this for chat_json, but be safe for direct callers).
    cleaned = _THINK_BLOCK.sub("", cleaned).strip()

    # 1. Fenced JSON: ```...```  -> take content between the last two fences.
    fence_starts = [m.start() for m in _FENCE.finditer(cleaned)]
    if len(fence_starts) >= 2:
        # Last fence pair: open at fence_starts[-2], close at fence_starts[-1].
        open_idx = fence_starts[-2]
        close_idx = fence_starts[-1]
        inner = cleaned[open_idx:close_idx]
        # Strip the opening fence and an optional language tag like "json".
        inner = inner.split("```", 1)[1]
        if inner[:4].lower() in {"json", "```"}:
            inner = inner[4:]
        cleaned = inner.strip()
    # 2. No fence: keep as-is (may still be plain JSON or reasoning+JSON).

    try:
        return cast(dict[str, Any], json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # 3. Fallback: find the last top-level {...} in the text (reasoning before
    #    JSON case) and try parsing that.
    last_obj = _last_json_object(cleaned)
    if last_obj is not None:
        try:
            return cast(dict[str, Any], json.loads(last_obj))
        except json.JSONDecodeError:
            pass

    return {"_raw": text}


def _last_json_object(text: str) -> str | None:
    """Return the substring of the last balanced top-level {...} block, or None."""
    start = text.rfind("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
