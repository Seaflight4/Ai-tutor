"""Session orchestration: the guided-discovery loop.

Flow:
  start_session(image)
      -> OCR -> parse -> persist session -> opening question -> persist turn
  reply(session_id, student_message)
      -> persist student turn -> assess_and_respond (diagnose + hint in one call)
      -> if wants_solution: reveal full solution
      -> else persist tutor turn -> update loop_count
      -> if resolved: update profiles
  reveal(session_id)
      -> generate full solution -> persist -> mark resolved
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from app.core import config, supabase
from app.models.schemas import (
    Classification,
    ResolutionType,
    TutorReply,
)
from app.services import hints, ocr, profile, solution, tutor

logger = logging.getLogger(__name__)


async def start_session(
    image_bytes: bytes,
    mime: str,
    student_id: UUID | None,
    external_ref: str | None,
) -> dict[str, Any]:
    """Run OCR, create session, generate opening. Return session + opening text."""
    extracted = await ocr.extract_problem(image_bytes, mime=mime)
    student = supabase.upsert_student(student_id, external_ref)
    sid = UUID(student["id"])

    # Upload image to Supabase Storage (best-effort; not required for logic).
    image_url = await _maybe_upload_image(sid, image_bytes, mime)

    session = supabase.create_session(
        sid,
        subject="physics",
        problem_text=extracted.problem_text,
        problem_image_url=image_url,
        concepts=extracted.concepts,
        ocr_raw=extracted.raw,
    )
    session_id = UUID(session["id"])

    weak = await profile.weak_concepts_for(sid, extracted.concepts)
    opening = await hints.generate_opening(extracted.problem_text, extracted.concepts, weak)
    supabase.add_turn(
        session_id,
        role="tutor",
        content=opening,
        loop_index=0,
        hint_level=None,
        classification=None,
    )
    session["opening"] = opening
    return session


async def reply(session_id: UUID, student_message: str) -> TutorReply:
    """Process a student reply and produce the tutor's next hint or a reveal.

    The loop runs continuously — there is no forced reveal offer after N hints.
    The session reveals the full solution only when the student explicitly
    requests it, either via a conservative keyword heuristic (saves an LLM
    call on obvious requests) or via the diagnosis `wants_solution` flag.
    """
    session = supabase.get_session(session_id)
    problem_text = session["problem_text"]
    concepts = session["concepts"] or []
    loop_count = int(session["loop_count"])

    turns = supabase.list_turns(session_id)
    dialogue = _render_dialogue(turns)

    # Persist the student's reply.
    supabase.add_turn(
        session_id,
        role="student",
        content=student_message,
        loop_index=loop_count + 1,
    )

    # 1. Obvious explicit solution request -> reveal immediately (skip diagnosis).
    if _is_solution_request(student_message):
        out = await _do_reveal(session_id, problem_text, concepts, dialogue)
        return TutorReply(
            session_id=session_id,
            content=out["solution"],
            loop_index=out["loop_index"],
            hint_level=None,
            classification=None,
            offer_reveal=False,
            resolved=True,
            resolution_type=ResolutionType.revealed,
            solution=out["solution"],
        )

    # 2. Diagnose + respond in a single LLM call (coherent reasoning pass).
    diag, hint = await tutor.assess_and_respond(
        problem_text, concepts, dialogue, student_message, loop_count
    )

    if diag.wants_solution:
        out = await _do_reveal(session_id, problem_text, concepts, dialogue)
        return TutorReply(
            session_id=session_id,
            content=out["solution"],
            loop_index=out["loop_index"],
            hint_level=None,
            classification=None,
            offer_reveal=False,
            resolved=True,
            resolution_type=ResolutionType.revealed,
            solution=out["solution"],
        )

    # 3. Otherwise emit the structured hint; the loop continues indefinitely.
    new_loop = loop_count + 1
    content = hints.summarize_hint(hint, diag.classification)
    hint_level = None

    supabase.add_turn(
        session_id,
        role="tutor",
        content=content,
        loop_index=new_loop,
        hint_level=hint_level,
        classification=diag.classification.value,
        metadata={
            "reasoning": diag.reasoning,
            "target_concept": diag.target_concept,
            "hint": hint.model_dump(),
        },
    )
    supabase.update_session(session_id, loop_count=new_loop)

    return TutorReply(
        session_id=session_id,
        content=content,
        loop_index=new_loop,
        hint_level=hint_level,
        classification=diag.classification,
        offer_reveal=False,
        resolved=False,
        hint=hint,
    )


async def reveal_solution(session_id: UUID) -> dict[str, Any]:
    session = supabase.get_session(session_id)
    turns = supabase.list_turns(session_id)
    return await _do_reveal(
        session_id,
        session["problem_text"],
        session["concepts"] or [],
        _render_dialogue(turns),
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
def _render_dialogue(turns: list[dict[str, Any]]) -> str:
    lines = []
    for t in turns:
        role = t["role"].upper()
        lines.append(f"{role}: {t['content']}")
    return "\n".join(lines)


# Phrases that count as an explicit request to reveal the full solution.
# Conservative: a normal stuck reply or a sub-question ("what's the answer to
# part a?") must NOT match. Used as a fast path before diagnosis; the LLM's
# `wants_solution` flag catches the softer cases.
_SOLUTION_REQUESTS = {
    "show me the solution",
    "show me the full solution",
    "show solution",
    "show the solution",
    "reveal the solution",
    "reveal solution",
    "give me the solution",
    "give me the answer",
    "just tell me the answer",
    "tell me the answer",
    "i give up",
    "i want the solution",
    "i want the answer",
    "solve it",
    "solve it for me",
    "walk me through the solution",
    "show me how to solve it",
    "show me how to solve this",
    "can you solve it",
    "solve this for me",
}

# Substring needles for requests that tolerate filler words ("full", "please",
# "can you", "how to", "this/it"). Order: longer/odd phrases first so we match
# the most specific intent.
_SOLUTION_NEEDLES = (
    "show me the solution",
    "show me the full solution",
    "show me how to solve",
    "show the solution",
    "give me the solution",
    "give me the answer",
    "give me the full solution",
    "reveal the solution",
    "walk me through the solution",
    "solve it for me",
    "solve this for me",
    "can you solve",
    "i give up",
    "just tell me the answer",
)


def _is_solution_request(message: str) -> bool:
    msg = message.strip().lower()
    if msg in _SOLUTION_REQUESTS:
        return True
    return any(n in msg for n in _SOLUTION_NEEDLES)


async def _do_reveal(
    session_id: UUID,
    problem_text: str,
    concepts: list[str],
    dialogue: str,
) -> dict[str, Any]:
    sol = await solution.generate_solution(problem_text, concepts, dialogue)
    supabase.add_turn(
        session_id,
        role="tutor",
        content=sol,
        loop_index=999,  # sentinel for reveal turn
        hint_level=None,
        classification=None,
        metadata={"reveal": True},
    )
    supabase.update_session(
        session_id, resolved=True, resolution_type=ResolutionType.revealed.value
    )

    # Update knowledge profiles for personalization.
    session_row = supabase.get_session(session_id)
    try:
        await profile.update_profiles(
            UUID(session_row["student_id"]),
            session_id,
            concepts,
            None,
            Classification.on_track,  # neutral; mastery estimated from dialogue
            dialogue,
        )
    except Exception as exc:
        logger.warning("profile update failed for session %s: %s", session_id, exc)

    return {
        "session_id": str(session_id),
        "solution": sol,
        "loop_index": 999,
        "resolved": True,
        "resolution_type": ResolutionType.revealed.value,
    }


async def _maybe_upload_image(student_id: UUID, image_bytes: bytes, mime: str) -> str | None:
    """Best-effort upload to Supabase Storage. Returns public URL or None.

    Skipped entirely on the local backend (no storage configured).
    """
    if not config.get_settings().supabase_url:
        return None
    try:
        client = supabase.get_client()
        bucket = config.get_settings().supabase_bucket
        path = f"{student_id}/{student_id}.png"
        client.storage.from_(bucket).upload(path, image_bytes, {"content-type": mime})
        url: str = client.storage.from_(bucket).get_public_url(path)
        return url
    except Exception:
        return None
