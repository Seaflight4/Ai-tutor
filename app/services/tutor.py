"""Merged tutor service: diagnose + respond in a single LLM call.

Replaces the previous two-call design (diagnosis.diagnose then hints.generate_hint).
The model now reasons about the stuck-point and writes the structured hint in
one coherent pass, so the hint is grounded in the diagnosis rationale rather
than re-deriving intent from a one-word classification label.
"""

from __future__ import annotations

from app.core import llm
from app.models.schemas import Classification, Diagnosis, HintOutput
from app.prompts import guided_discovery as p


async def assess_and_respond(
    problem_text: str,
    concepts: list[str],
    dialogue: str,
    student_reply: str,
    current_loop: int,
) -> tuple[Diagnosis, HintOutput]:
    """Diagnose the stuck-point and produce a structured hint in one LLM call."""
    raw = await llm.chat_json(
        p.TUTOR_SYSTEM,
        p.tutor_user(problem_text, concepts, dialogue, student_reply, current_loop),
        temperature=0.3,
        max_tokens=1100,
    )

    try:
        classification = Classification(raw.get("classification", "on_track"))
    except ValueError:
        classification = Classification.on_track

    diagnosis = Diagnosis(
        classification=classification,
        reasoning=raw.get("reasoning", ""),
        target_concept=raw.get("target_concept"),
        wants_solution=bool(raw.get("wants_solution", False)),
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
    )

    return diagnosis, hint
