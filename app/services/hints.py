"""Hint generation service (progressive depth)."""

from __future__ import annotations

from app.core import llm
from app.models.schemas import Classification
from app.prompts import guided_discovery as p


async def generate_hint(
    problem_text: str,
    concepts: list[str],
    dialogue: str,
    classification: Classification,
    target_concept: str | None,
    hint_level: int,
) -> str:
    level = max(1, min(3, hint_level))
    return await llm.chat_text(
        p.HINT_SYSTEM,
        p.hint_user(problem_text, concepts, dialogue, classification, target_concept, level),
        temperature=0.5,
        max_tokens=400,
    )


async def generate_opening(
    problem_text: str,
    concepts: list[str],
    weak_concepts: list[str] | None = None,
) -> str:
    return await llm.chat_text(
        p.OPENING_SYSTEM,
        p.opening_user(problem_text, concepts, weak_concepts),
        temperature=0.5,
        max_tokens=300,
    )
