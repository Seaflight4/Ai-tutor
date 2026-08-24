"""Session orchestration: the guided-discovery loop.

Flow:
  start_session(image)
      -> OCR -> parse -> persist session -> opening question -> persist turn
  reply(session_id, student_message)
      -> persist student turn -> diagnose -> generate hint (level 1..3)
      -> if loop_count >= MAX_HINT_LOOPS: offer reveal instead of a 4th hint
      -> persist tutor turn -> update loop_count
      -> if resolved: update profiles
  reveal(session_id)
      -> generate full solution -> persist -> mark resolved
"""

from __future__ import annotations

import contextlib
from typing import Any
from uuid import UUID

from app.core import config, supabase
from app.models.schemas import (
    Classification,
    ResolutionType,
    TutorReply,
)
from app.services import diagnosis, hints, ocr, profile, solution


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
    """Process a student reply and produce the tutor's next hint / offer."""
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

    # Check for reveal choice after an offer was made.
    if _is_reveal_choice(student_message) and loop_count >= config.get_settings().max_hint_loops:
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

    diag = await diagnosis.diagnose(
        problem_text, concepts, dialogue, student_message, loop_count
    )

    new_loop = loop_count + 1
    max_loops = config.get_settings().max_hint_loops

    if new_loop >= max_loops:
        # Offer the reveal choice instead of going deeper.
        content = _REVEAL_OFFER
        hint_level = None
        offer_reveal = True
    else:
        content = await hints.generate_hint(
            problem_text,
            concepts,
            dialogue,
            diag.classification,
            diag.target_concept,
            diag.next_hint_level,
        )
        hint_level = diag.next_hint_level
        offer_reveal = False

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
        },
    )
    supabase.update_session(session_id, loop_count=new_loop)

    return TutorReply(
        session_id=session_id,
        content=content,
        loop_index=new_loop,
        hint_level=hint_level,
        classification=diag.classification,
        offer_reveal=offer_reveal,
        resolved=False,
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
_REVEAL_OFFER = (
    "It looks like we've explored this from a few angles and it's still tricky. "
    "Would you like to:\n"
    "  (a) try one more hint, or\n"
    "  (b) walk through the full solution together?\n"
    "Just reply 'a' or 'b'."
)


def _render_dialogue(turns: list[dict[str, Any]]) -> str:
    lines = []
    for t in turns:
        role = t["role"].upper()
        lines.append(f"{role}: {t['content']}")
    return "\n".join(lines)


def _is_reveal_choice(message: str) -> bool:
    return message.strip().lower() in {"b", "(b)", "reveal", "solution", "yes"}


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
    with contextlib.suppress(Exception):
        await profile.update_profiles(
            UUID(session_row["student_id"]),
            session_id,
            concepts,
            None,
            Classification.on_track,  # neutral; mastery estimated from dialogue
            dialogue,
        )

    return {
        "session_id": str(session_id),
        "solution": sol,
        "loop_index": 999,
        "resolved": True,
        "resolution_type": ResolutionType.revealed.value,
    }


async def _maybe_upload_image(student_id: UUID, image_bytes: bytes, mime: str) -> str | None:
    """Best-effort upload to Supabase Storage. Returns public URL or None."""
    try:
        client = supabase.get_client()
        bucket = config.get_settings().supabase_bucket
        path = f"{student_id}/{student_id}.png"
        client.storage.from_(bucket).upload(path, image_bytes, {"content-type": mime})
        return client.storage.from_(bucket).get_public_url(path)
    except Exception:
        return None
