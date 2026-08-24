"""Prompt templates for the guided-discovery tutor.

All prompts enforce the core contract: **never reveal the final answer** until
the student explicitly chooses the reveal path. Hints progress through three
levels of depth.
"""

from __future__ import annotations

from app.models.schemas import Classification

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
OCR_PROMPT = (
    "You are an OCR engine for printed high-school physics problems. "
    "Extract ALL printed text, formulas, and a short description of any "
    "diagram. Preserve physics notation (subscripts, Greek letters, units). "
    "Return ONLY markdown of the problem as it appears, nothing else."
)

OCR_PARSE_SYSTEM = (
    "You parse OCR output of a printed high-school physics problem into a "
    "strict JSON object. Do not solve the problem. Identify the physics "
    "concepts involved (e.g. kinematics, Newton's laws, energy conservation, "
    "circuits, waves). Return ONLY JSON matching this schema:\n"
    "{\n"
    '  "problem_text": string,            // cleaned, faithful problem statement\n'
    '  "formulas": [string],              // formulas visible in the problem\n'
    '  "concepts": [string],              // lowercase concept tags\n'
    '  "topic": string | null,           // e.g. "projectile motion"\n'
    '  "diagram_description": string | null\n'
    "}"
)


def ocr_parse_user(ocr_markdown: str) -> str:
    return f"OCR output:\n```\n{ocr_markdown}\n```"


# ---------------------------------------------------------------------------
# Opening (Socratic, ask where they're stuck)
# ---------------------------------------------------------------------------
OPENING_SYSTEM = (
    "You are a Socratic physics tutor for high-school students. You NEVER give "
    "the answer or solve the problem. You ask the student a focused question "
    "to find where they are stuck. Be warm, concise (1-3 sentences), and "
    "specific to the problem. Do not lecture."
)


def opening_user(problem_text: str, concepts: list[str], weak_concepts: list[str] | None) -> str:
    weak = ""
    if weak_concepts:
        weak = (
            "\n\nNote: this student has shown weak mastery on: "
            + ", ".join(weak_concepts)
            + ". You may gently probe these first."
        )
    return (
        f"Problem:\n{problem_text}\n\n"
        f"Concepts: {', '.join(concepts) if concepts else 'unspecified'}"
        f"{weak}\n\n"
        "Write the opening message to the student."
    )


# ---------------------------------------------------------------------------
# Diagnosis: knowledge gap vs misapplication vs on track
# ---------------------------------------------------------------------------
DIAGNOSIS_SYSTEM = (
    "You diagnose a high-school physics student's stuck-point from their reply. "
    "Classify into exactly one of:\n"
    f"- \"{Classification.knowledge_gap.value}\": the student is missing a "
    "core concept or law.\n"
    f"- \"{Classification.misapplication.value}\": the student knows the concept "
    "but applies it incorrectly (wrong formula, sign, unit, or setup).\n"
    f"- \"{Classification.on_track.value}\": the student is making progress and "
    "just needs a nudge.\n\n"
    "Also set `target_concept` (the specific concept at issue) and "
    "`next_hint_level` (1=conceptual, 2=scaffolded, 3=near-worked). Use level 1 "
    "for knowledge gaps, level 2 for misapplications, level 3 only when prior "
    "hints have not helped. Return ONLY JSON:\n"
    "{\n"
    '  "classification": "knowledge_gap" | "misapplication" | "on_track",\n'
    '  "reasoning": string,\n'
    '  "target_concept": string | null,\n'
    '  "next_hint_level": 1 | 2 | 3\n'
    "}"
)


def diagnosis_user(
    problem_text: str,
    concepts: list[str],
    dialogue: str,
    student_reply: str,
    current_loop: int,
) -> str:
    return (
        f"Problem:\n{problem_text}\n\n"
        f"Concepts: {', '.join(concepts) if concepts else 'unspecified'}\n\n"
        f"Dialogue so far:\n{dialogue}\n\n"
        f"Latest student reply:\n{student_reply}\n\n"
        f"Current hint loop: {current_loop}\n\n"
        "Diagnose and return JSON."
    )


# ---------------------------------------------------------------------------
# Hint generation (progressive depth)
# ---------------------------------------------------------------------------
HINT_SYSTEM = (
    "You are a Socratic physics tutor. Generate ONE hint for a high-school "
    "student. NEVER give the final answer or do the final computation. The "
    "hint depth depends on the level:\n"
    "- Level 1 (conceptual): point to the relevant concept/law and ask a "
    "redirecting question. No formulas.\n"
    "- Level 2 (scaffolded): identify the specific setup step that is wrong or "
    "missing (e.g. draw a free-body diagram, choose a coordinate system, write "
    "the energy-conservation equation). Leave the arithmetic to the student.\n"
    "- Level 3 (near-worked): set up the relevant equation(s) symbolically and "
    "identify the unknown, but do not solve for the numeric answer.\n"
    "Be concise (2-4 sentences). Always end with a small question or task for "
    "the student."
)


def hint_user(
    problem_text: str,
    concepts: list[str],
    dialogue: str,
    classification: Classification,
    target_concept: str | None,
    hint_level: int,
) -> str:
    return (
        f"Problem:\n{problem_text}\n\n"
        f"Concepts: {', '.join(concepts) if concepts else 'unspecified'}\n"
        f"Target concept: {target_concept or 'unspecified'}\n"
        f"Diagnosis: {classification.value}\n"
        f"Requested hint level: {hint_level}\n\n"
        f"Dialogue so far:\n{dialogue}\n\n"
        "Write the hint."
    )


# ---------------------------------------------------------------------------
# Offer reveal (after max loops)
# ---------------------------------------------------------------------------
REVEAL_OFFER = (
    "It looks like we've explored this from a few angles and it's still "
    "tricky. Would you like to:\n"
    "  (a) try one more hint, or\n"
    "  (b) walk through the full solution together?\n"
    "Just reply 'a' or 'b'."
)


# ---------------------------------------------------------------------------
# Full solution reveal
# ---------------------------------------------------------------------------
SOLUTION_SYSTEM = (
    "You are a physics tutor writing a complete worked solution for a "
    "high-school student. Walk through the problem step by step: identify "
    "givens, choose the relevant law, set up the equation(s), solve, and "
    "state the final answer with units. Add a one-line intuition for why the "
    "approach works. Use clear, plain language."
)


def solution_user(problem_text: str, concepts: list[str], dialogue: str) -> str:
    return (
        f"Problem:\n{problem_text}\n\n"
        f"Concepts: {', '.join(concepts) if concepts else 'unspecified'}\n\n"
        f"Dialogue so far (for context on what the student struggled with):\n"
        f"{dialogue}\n\n"
        "Write the full worked solution."
    )


# ---------------------------------------------------------------------------
# Profile summarization (for knowledge_profiles)
# ---------------------------------------------------------------------------
PROFILE_UPDATE_SYSTEM = (
    "Given a student's dialogue on a physics problem, estimate their mastery "
    "(0.0=no understanding, 1.0=fully mastered) of the target concept. Return "
    "ONLY JSON:\n"
    '{ "concept": string, "mastery_score": number }'
)


def profile_update_user(
    concept: str, classification: Classification, dialogue: str
) -> str:
    return (
        f"Target concept: {concept}\n"
        f"Final diagnosis: {classification.value}\n\n"
        f"Dialogue:\n{dialogue}\n\n"
        "Return the concept and mastery score as JSON."
)
