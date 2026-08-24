"""OCR service: image -> structured problem via the skainet OCR model.

Two-step:
  1. olmocr-7B-faithful extracts text from the image. It returns a JSON
     envelope ({"natural_text": "...", ...}); we unwrap that.
  2. The chat model parses that text into a structured `OCRResult`.

Keeping the steps separate means OCR errors can be logged / re-prompted
without re-running the (more expensive) parsing step.
"""

from __future__ import annotations

import json

from app.core import llm
from app.models.schemas import OCRResult
from app.prompts import guided_discovery as p


def _unwrap_ocr(raw: str) -> str:
    """olmocr returns a JSON envelope with a `natural_text` field. Fall back
    to the raw string if it isn't JSON or lacks that field."""
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(envelope, dict) and "natural_text" in envelope:
        return str(envelope["natural_text"])
    return raw


def _clean_problem_text(text: str) -> str:
    """Strip markdown code fences and leading/trailing whitespace."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence (```markdown, ```, etc.)
        if lines:
            lines = lines[1:]
        # Drop a trailing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


async def extract_problem(image_bytes: bytes, mime: str = "image/png") -> OCRResult:
    raw_ocr = await llm.ocr_image(image_bytes, prompt=p.OCR_PROMPT, mime=mime)
    markdown = _unwrap_ocr(raw_ocr)
    parsed = await llm.chat_json(
        p.OCR_PARSE_SYSTEM,
        p.ocr_parse_user(markdown),
        temperature=0.0,
        max_tokens=1000,
    )
    problem_text = _clean_problem_text(parsed.get("problem_text", markdown))
    return OCRResult(
        problem_text=problem_text,
        formulas=parsed.get("formulas", []) or [],
        concepts=[c.lower() for c in (parsed.get("concepts") or []) if c],
        topic=parsed.get("topic"),
        diagram_description=parsed.get("diagram_description"),
        raw={"ocr_markdown": markdown, "ocr_raw": raw_ocr, "parsed": parsed},
    )
