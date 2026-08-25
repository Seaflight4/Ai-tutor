"""OCR service: image -> structured problem via the skainet OCR model.

Two-step:
  1. olmocr-7B-faithful extracts text from the image. It returns a JSON
     envelope ({"natural_text": "...", ...}); we unwrap that.
  2. The chat model parses that text into a structured `OCRResult`.

Keeping the steps separate means OCR errors can be logged / re-prompted
without re-running the (more expensive) parsing step.

If the OCR output looks invalid (empty, too short, or lacking physics
indicators), we retry the OCR call once with a neutral transcription prompt
before giving up and proceeding with whatever we have.
"""

from __future__ import annotations

import json
import logging
import re

from app.core import llm
from app.core.config import get_settings
from app.models.schemas import OCRResult
from app.prompts import guided_discovery as p

logger = logging.getLogger(__name__)

_PHYSICS_INDICATOR = re.compile(
    r"(?:m/s|\d\s*kg|\d\s*N\b|\d\s*J\b|\d\s*W\b|\d\s*V\b|\d\s*A\b|Hz|Pa|°|"
    r"find|calculate|determine|what is|how many|\?\s*$)",
    re.IGNORECASE,
)

_RETRY_PROMPT = (
    "Transcribe ALL text in this image exactly as printed. "
    "Return ONLY the transcribed text, nothing else."
)


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


def _validate_ocr_text(text: str) -> bool:
    """Check whether OCR output looks like a real physics problem.

    Returns True if the text is non-trivial and contains at least one
    physics indicator (units, a question, or a problem-solving verb).
    """
    if len(text.strip()) < 30:
        return False
    return bool(_PHYSICS_INDICATOR.search(text))


async def extract_problem(image_bytes: bytes, mime: str = "image/png") -> OCRResult:
    settings = get_settings()
    raw_ocr = await llm.ocr_image(
        image_bytes,
        prompt=p.OCR_PROMPT,
        mime=mime,
        max_tokens=settings.ocr_max_tokens,
    )
    markdown = _unwrap_ocr(raw_ocr)

    if not _validate_ocr_text(markdown):
        logger.warning(
            "OCR output failed validation (len=%d); retrying with neutral prompt.",
            len(markdown),
        )
        raw_retry = await llm.ocr_image(
            image_bytes,
            prompt=_RETRY_PROMPT,
            mime=mime,
            max_tokens=settings.ocr_max_tokens,
        )
        retry_markdown = _unwrap_ocr(raw_retry)
        if _validate_ocr_text(retry_markdown):
            markdown = retry_markdown
            raw_ocr = raw_retry
        else:
            logger.warning(
                "OCR retry also failed validation; proceeding with best output."
            )

    parsed = await llm.chat_json(
        p.OCR_PARSE_SYSTEM,
        p.ocr_parse_user(markdown),
        temperature=settings.ocr_parse_temp,
        max_tokens=settings.ocr_parse_max_tokens,
    )
    problem_text = _clean_problem_text(parsed.get("problem_text", markdown))
    return OCRResult(
        problem_text=problem_text,
        formulas=parsed.get("formulas", []) or [],
        concepts=[c.lower() for c in (parsed.get("concepts") or []) if c],
        topic=parsed.get("topic"),
        problem_type=parsed.get("problem_type"),
        diagram_description=parsed.get("diagram_description"),
        raw={"ocr_markdown": markdown, "ocr_raw": raw_ocr, "parsed": parsed},
    )
