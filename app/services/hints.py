"""Hint rendering helpers.

The structured hint is produced by `app.services.tutor.assess_and_respond`
(a single merged LLM call). This module renders the `HintOutput` into a
one-line summary string for `TutorReply.content` and generates the opening
message. The frontend renders the structured `HintOutput` directly; `content`
is a plain-text fallback.
"""

from __future__ import annotations

from app.core import llm
from app.models.schemas import Classification, HintOutput
from app.prompts import guided_discovery as p


def summarize_hint(h: HintOutput, classification: Classification) -> str:
    """One-line plain-text summary used as `TutorReply.content` fallback."""
    if classification is Classification.knowledge_gap:
        return h.explanation or "Here's the concept you need."
    if classification is Classification.misapplication:
        return h.mistake or "There's a small mistake in your setup."
    # on_track
    return h.confirmation or h.next_step_hint or "You're on the right track!"


async def generate_opening(
    problem_text: str,
    concepts: list[str],
    weak_concepts: list[str] | None = None,
) -> str:
    return await llm.chat_text(
        p.OPENING_SYSTEM,
        p.opening_user(problem_text, concepts, weak_concepts),
        temperature=0.6,
        max_tokens=300,
    )
