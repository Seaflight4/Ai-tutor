"""Diagnosis service: classify the student's stuck-point."""

from __future__ import annotations

from app.core import llm
from app.models.schemas import Classification, Diagnosis
from app.prompts import guided_discovery as p


async def diagnose(
    problem_text: str,
    concepts: list[str],
    dialogue: str,
    student_reply: str,
    current_loop: int,
) -> Diagnosis:
    raw = await llm.chat_json(
        p.DIAGNOSIS_SYSTEM,
        p.diagnosis_user(problem_text, concepts, dialogue, student_reply, current_loop),
        temperature=0.0,
        max_tokens=400,
    )
    try:
        classification = Classification(raw.get("classification", "on_track"))
    except ValueError:
        classification = Classification.on_track
    level = raw.get("next_hint_level", 1)
    try:
        next_hint_level = max(1, min(3, int(level)))
    except (TypeError, ValueError):
        next_hint_level = 1
    return Diagnosis(
        classification=classification,
        reasoning=raw.get("reasoning", ""),
        target_concept=raw.get("target_concept"),
        next_hint_level=next_hint_level,
    )
