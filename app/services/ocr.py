"""OCR service: image -> structured problem via the skainet OCR model.

Two-step:
  1. olmocr-7B-faithful extracts faithful markdown text from the image.
  2. The chat model parses that markdown into a structured `OCRResult`.

Keeping the steps separate means OCR errors can be logged / re-prompted
without re-running the (more expensive) parsing step.
"""

from __future__ import annotations

from app.core import llm
from app.models.schemas import OCRResult
from app.prompts import guided_discovery as p


async def extract_problem(image_bytes: bytes, mime: str = "image/png") -> OCRResult:
    markdown = await llm.ocr_image(image_bytes, prompt=p.OCR_PROMPT, mime=mime)
    parsed = await llm.chat_json(
        p.OCR_PARSE_SYSTEM,
        p.ocr_parse_user(markdown),
        temperature=0.0,
        max_tokens=1000,
    )
    return OCRResult(
        problem_text=parsed.get("problem_text", markdown),
        formulas=parsed.get("formulas", []) or [],
        concepts=[c.lower() for c in (parsed.get("concepts") or []) if c],
        topic=parsed.get("topic"),
        diagram_description=parsed.get("diagram_description"),
        raw={"ocr_markdown": markdown, "parsed": parsed},
    )
