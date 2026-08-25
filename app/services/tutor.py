"""Merged tutor service: diagnose + respond in a single LLM call.

Replaces the previous two-call design (diagnosis.diagnose then hints.generate_hint).
The model now reasons about the stuck-point and writes the structured hint in
one coherent pass, so the hint is grounded in the diagnosis rationale rather
than re-deriving intent from a one-word classification label.
"""

from __future__ import annotations

import logging

from app.core import llm
from app.core.config import get_settings
from app.models.schemas import Classification, Diagnosis, HintOutput, ReferenceChunk
from app.prompts import guided_discovery as p

logger = logging.getLogger(__name__)


def _source_strings(chunks: list[ReferenceChunk] | None) -> list[str]:
    """Render reference chunks into the numbered SOURCES block for the prompt."""
    if not chunks:
        return []
    out: list[str] = []
    for c in chunks:
        head = c.source_title
        if c.chapter:
            head += f" — {c.chapter}"
        if c.heading:
            head += f" — {c.heading}"
        out.append(f"{head} ({c.source_url})\n{c.chunk_text}")
    return out


async def assess_and_respond(
    problem_text: str,
    concepts: list[str],
    dialogue: str,
    student_reply: str,
    current_loop: int,
    sources: list[ReferenceChunk] | None = None,
    student_context: str | None = None,
) -> tuple[Diagnosis, HintOutput]:
    """Diagnose the stuck-point and produce a structured hint in one LLM call.

    `sources` are retrieved reference chunks injected into the prompt so the
    model grounds `explanation`/`formula` in them rather than parametric memory.
    `student_context` is a compressed block of the student's past session
    summaries, injected so the tutor can reference prior problems and mistakes.
    """
    settings = get_settings()
    raw = await llm.chat_json(
        p.TUTOR_SYSTEM,
        p.tutor_user(
            problem_text,
            concepts,
            dialogue,
            student_reply,
            current_loop,
            _source_strings(sources),
            student_context,
        ),
        temperature=settings.tutor_temp,
        max_tokens=settings.tutor_max_tokens,
    )

    # When the model returns no classification (empty JSON, parse failure,
    # or a malformed response), default to `meta` rather than `on_track`.
    # `on_track` emits a false-positive affirmation ("You're on the right
    # track!") which is the most harmful default in a tutoring context.
    # `meta` falls through to `meta_response or ""` — a neutral empty reply
    # that doesn't mislead the student.
    raw_class = raw.get("classification")
    if raw_class is None:
        logger.warning(
            "tutor LLM returned no classification; defaulting to meta. "
            "raw keys: %s", list(raw.keys()),
        )
        classification = Classification.meta
    else:
        try:
            classification = Classification(raw_class)
        except ValueError:
            logger.warning(
                "tutor LLM returned unknown classification %r; defaulting to meta.",
                raw_class,
            )
            classification = Classification.meta

    diagnosis = Diagnosis(
        classification=classification,
        reasoning=raw.get("reasoning", ""),
        target_concept=raw.get("target_concept"),
        wants_solution=raw.get("wants_solution") is True,
    )

    hint = HintOutput(
        formula=raw.get("formula"),
        explanation=raw.get("explanation"),
        example=raw.get("example"),
        mistake=raw.get("mistake"),
        reason=raw.get("reason"),
        application_hint=raw.get("application_hint"),
        confirmation=raw.get("confirmation"),
        next_step_hint=raw.get("next_step_hint"),
        source_title=raw.get("source_title"),
        source_url=raw.get("source_url"),
        answer_status=raw.get("answer_status"),
        answer_value=raw.get("answer_value"),
        method_feedback=raw.get("method_feedback"),
        meta_response=raw.get("meta_response"),
    )

    return diagnosis, hint
