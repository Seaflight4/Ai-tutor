"""Hint rendering helpers.

The structured hint is produced by `app.services.tutor.assess_and_respond`
(a single merged LLM call). This module renders the `HintOutput` into a
one-line summary string for `TutorReply.content` and generates the opening
message. The frontend renders the structured `HintOutput` directly; `content`
is a plain-text fallback.
"""

from __future__ import annotations

from app.core import llm
from app.core.config import get_settings
from app.models.schemas import Classification, HintOutput
from app.prompts import guided_discovery as p


def summarize_hint(h: HintOutput, classification: Classification) -> str:
    """One-line plain-text summary used as `TutorReply.content` fallback."""
    if classification is Classification.knowledge_gap:
        return h.explanation or "Here's the concept you need."
    if classification is Classification.misapplication:
        return h.mistake or "There's a small mistake in your setup."
    if classification is Classification.on_track:
        return h.confirmation or h.next_step_hint or "What do you think the next step is?"
    if classification is Classification.answer_check:
        if h.answer_status == "correct":
            return h.method_feedback or "That's correct!"
        if h.answer_status == "incorrect":
            return h.mistake or "That's not quite right — let's find the error."
        return h.method_feedback or "Your approach is right — keep going."
    if classification is Classification.incorrect_answer:
        return h.mistake or "That answer isn't right — let's find the error."
    if classification is Classification.solved:
        return h.confirmation or "Nice — you've solved it!"
    # meta
    return h.meta_response or "Could you tell me a bit more about what you're stuck on?"


async def generate_opening(
    problem_text: str,
    concepts: list[str],
    weak_concepts: list[str] | None = None,
) -> str:
    settings = get_settings()
    text = await llm.chat_text(
        p.OPENING_SYSTEM,
        p.opening_user(problem_text, concepts, weak_concepts),
        temperature=settings.opening_temp,
        max_tokens=settings.opening_max_tokens,
    )
    if not text or not text.strip():
        return (
            "Hey there! This looks like an interesting physics problem. "
            "Where are you stuck?"
        )
    return text
