"""Session orchestration: the guided-discovery loop.

Flow:
  start_session(image)
      -> OCR -> parse -> persist session -> opening question -> persist turn
  reply(session_id, student_message)
      -> persist student turn -> assess_and_respond (diagnose + hint in one call)
      -> if wants_solution: reveal full solution
      -> else persist tutor turn -> update loop_count
      -> after max_hint_loops: set offer_reveal=True (offer, not forced)
      -> if resolved: update profiles + transition status
  reveal(session_id)
      -> generate full solution -> persist -> transition to 'revealed'

State machine (replaces the previous `loop_count = 999` sentinels):
  active -> active  (every hint turn)
  active -> revealed (explicit reveal request / wants_solution)
  active -> solved   (student demonstrated a full correct solution)
  revealed / solved are terminal; `reply()` on a terminal session raises
  `SessionTerminalError`, which the routes layer maps to 409.

All persistence calls (`supabase.*` / `local_store.*`) are synchronous (SQLite
I/O or a blocking HTTP round-trip to PostgREST). They are wrapped in
`asyncio.to_thread` so the uvicorn event loop is never blocked while a request
waits on the database.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from app.core import config, supabase
from app.domain.state import SessionStatus, SessionTerminalError, is_terminal, transition
from app.models.schemas import (
    Classification,
    Diagnosis,
    HintOutput,
    ReferenceChunk,
    ResolutionType,
    TutorReply,
)
from app.services import hints, ocr, profile, retrieval, solution, tutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async wrappers over the sync persistence layer.
# ---------------------------------------------------------------------------
# `app.core.supabase` (and its local_store fallback) perform synchronous I/O.
# Every call from an `async def` service goes through `to_thread` so the
# event loop stays free to handle other requests while we wait on the DB.
# When the services are migrated to the `Backend` Protocol (a later PR), the
# adapter will own this wrapping and these helpers go away.

async def _upsert_student(student_id: UUID | None, external_ref: str | None) -> dict[str, Any]:
    return await asyncio.to_thread(supabase.upsert_student, student_id, external_ref)


async def _create_session(
    student_id: UUID,
    *,
    subject: str,
    problem_text: str,
    problem_image_url: str | None,
    concepts: list[str],
    ocr_raw: dict[str, Any],
    problem_type: str | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        supabase.create_session,
        student_id,
        subject=subject,
        problem_text=problem_text,
        problem_image_url=problem_image_url,
        concepts=concepts,
        ocr_raw=ocr_raw,
        problem_type=problem_type,
    )


async def _get_session(session_id: UUID) -> dict[str, Any]:
    return await asyncio.to_thread(supabase.get_session, session_id)


async def _update_session(
    session_id: UUID,
    *,
    loop_count: int | None = None,
    status: str | None = None,
    resolved: bool | None = None,
    resolution_type: str | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        supabase.update_session,
        session_id,
        loop_count=loop_count,
        status=status,
        resolved=resolved,
        resolution_type=resolution_type,
    )


async def _add_turn(
    session_id: UUID,
    *,
    role: str,
    content: str,
    loop_index: int,
    hint_level: int | None = None,
    classification: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        supabase.add_turn,
        session_id,
        role=role,
        content=content,
        loop_index=loop_index,
        hint_level=hint_level,
        classification=classification,
        metadata=metadata,
    )


async def _list_turns(session_id: UUID) -> list[dict[str, Any]]:
    return await asyncio.to_thread(supabase.list_turns, session_id)


# ---------------------------------------------------------------------------
# Public service surface
# ---------------------------------------------------------------------------
async def start_session(
    image_bytes: bytes,
    mime: str,
    student_id: UUID | None,
    external_ref: str | None,
) -> dict[str, Any]:
    """Run OCR, create session, generate opening. Return session + opening text."""
    extracted = await ocr.extract_problem(image_bytes, mime=mime)
    student = await _upsert_student(student_id, external_ref)
    sid = UUID(student["id"])

    # Upload image to Supabase Storage (best-effort; not required for logic).
    image_url = await _maybe_upload_image(sid, image_bytes, mime)

    session = await _create_session(
        sid,
        subject="physics",
        problem_text=extracted.problem_text,
        problem_image_url=image_url,
        concepts=extracted.concepts,
        ocr_raw=extracted.raw,
        problem_type=extracted.problem_type,
    )
    session_id = UUID(session["id"])

    weak = await profile.weak_concepts_for(sid, extracted.concepts)
    opening = await hints.generate_opening(extracted.problem_text, extracted.concepts, weak)
    await _add_turn(
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

    The session reveals the full solution only when the student explicitly
    requests it, either via a conservative keyword heuristic (saves an LLM
    call on obvious requests) or via the diagnosis `wants_solution` flag.
    After `max_hint_loops` hint turns, the tutor sets `offer_reveal=True` so
    the student knows they can ask for the solution — but the loop never
    force-reveals.

    Raises `SessionTerminalError` if the session is already resolved (revealed
    or solved); the routes layer maps that to 409.
    """
    session = await _get_session(session_id)
    status = SessionStatus(session.get("status") or SessionStatus.active.value)
    if is_terminal(status):
        raise SessionTerminalError(f"session {session_id} is already {status.value}")

    problem_text = session["problem_text"]
    concepts = session["concepts"] or []
    loop_count = int(session["loop_count"])

    turns = await _list_turns(session_id)
    dialogue = _render_dialogue(turns)

    # Persist the student's reply.
    await _add_turn(
        session_id,
        role="student",
        content=student_message,
        loop_index=loop_count + 1,
    )

    # 1. Obvious explicit solution request -> reveal immediately (skip diagnosis).
    if _is_solution_request(student_message):
        out = await _do_reveal(session_id, problem_text, concepts, dialogue, status)
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
    #    Pre-retrieve reference chunks so the single call can ground
    #    explanation/formula in curated sources rather than parametric memory.
    sources = await retrieval.retrieve_for_concepts(concepts, student_message)
    sid_str = session.get("student_id")
    student_context = await profile.build_student_context(
        UUID(sid_str) if sid_str else None,
        concepts,
        session.get("problem_type"),
    )
    diag, hint = await tutor.assess_and_respond(
        problem_text, concepts, dialogue, student_message, loop_count, sources,
        student_context=student_context,
    )

    if diag.wants_solution:
        out = await _do_reveal(session_id, problem_text, concepts, dialogue, status)
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

    # 2b. Student has demonstrated a full correct solution -> mark solved and
    #     record mastery (mirrors the reveal path's profile update).
    if diag.classification is Classification.solved:
        return await _do_solved(
            session_id, concepts, dialogue, diag, hint, sources, status,
            student_context=student_context,
        )

    # 3. Otherwise emit the structured hint; the loop continues.
    new_loop = loop_count + 1
    content = hints.summarize_hint(hint, diag.classification)
    hint_level = None

    # After the configured cap, offer the student the option to reveal — but
    # never force it. The student may keep working indefinitely.
    max_loops = config.get_settings().max_hint_loops
    offer_reveal = new_loop >= max_loops
    if diag.classification is Classification.solved or (  # type: ignore[comparison-overlap]
        diag.classification is Classification.answer_check
        and hint.answer_status == "correct"
    ):
        offer_reveal = False

    await _add_turn(
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
            "student_context": student_context,
        },
    )
    await _update_session(session_id, loop_count=new_loop)

    return TutorReply(
        session_id=session_id,
        content=content,
        loop_index=new_loop,
        hint_level=hint_level,
        classification=diag.classification,
        offer_reveal=offer_reveal,
        resolved=False,
        hint=hint,
        sources=sources or None,
    )


async def reveal_solution(session_id: UUID) -> dict[str, Any]:
    session = await _get_session(session_id)
    status = SessionStatus(session.get("status") or SessionStatus.active.value)
    if is_terminal(status):
        # Reveal on an already-resolved session is idempotent: return the
        # recorded resolution rather than re-running the LLM.
        last = await _last_tutor_content(session_id)
        return {
            "session_id": str(session_id),
            "solution": last or "This session has already been resolved.",
            "loop_index": int(session.get("loop_count") or 0),
            "resolved": True,
            "resolution_type": session.get("resolution_type")
            or ResolutionType.revealed.value,
        }
    turns = await _list_turns(session_id)
    return await _do_reveal(
        session_id,
        session["problem_text"],
        session["concepts"] or [],
        _render_dialogue(turns),
        status,
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


async def _last_tutor_content(session_id: UUID) -> str | None:
    """Return the most recent tutor turn's content, or None."""
    turns = await _list_turns(session_id)
    for t in reversed(turns):
        if t["role"] == "tutor":
            return str(t["content"])
    return None


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
    from_status: SessionStatus,
) -> dict[str, Any]:
    sol = await solution.generate_solution(problem_text, concepts, dialogue)
    # Transition validates that the session is still active (not already
    # resolved) and raises SessionTerminalError otherwise.
    transition(from_status, SessionStatus.revealed)

    session = await _get_session(session_id)
    loop_index = int(session["loop_count"]) + 1
    await _add_turn(
        session_id,
        role="tutor",
        content=sol,
        loop_index=loop_index,
        hint_level=None,
        classification=None,
        metadata={"reveal": True},
    )
    await _update_session(
        session_id,
        status=SessionStatus.revealed.value,
        resolved=True,
        resolution_type=ResolutionType.revealed.value,
    )

    # Update knowledge profiles for personalization.
    try:
        await profile.update_profiles(
            UUID(session["student_id"]),
            session_id,
            concepts,
            None,
            Classification.on_track,  # neutral; mastery estimated from dialogue
            dialogue,
        )
    except Exception as exc:
        logger.warning("profile update failed for session %s: %s", session_id, exc)

    # Generate a session summary for the cross-session learning record.
    await _generate_session_summary(
        session_id,
        UUID(session["student_id"]),
        problem_text=problem_text,
        concepts=concepts,
        problem_type=session.get("problem_type"),
        outcome="revealed",
        target_concept=None,
        dialogue=dialogue,
    )

    return {
        "session_id": str(session_id),
        "solution": sol,
        "loop_index": loop_index,
        "resolved": True,
        "resolution_type": ResolutionType.revealed.value,
    }


async def _do_solved(
    session_id: UUID,
    concepts: list[str],
    dialogue: str,
    diag: Diagnosis,
    hint: HintOutput,
    sources: list[ReferenceChunk] | None,
    from_status: SessionStatus,
    student_context: str | None = None,
) -> TutorReply:
    """Mark the session solved (student solved it themselves) and record mastery.

    Mirrors `_do_reveal` for the profile update, but uses `solved_with_hints`
    as the resolution type and persists the tutor's terminal `confirmation`
    rather than a generated full solution.
    """
    transition(from_status, SessionStatus.solved)

    content = hints.summarize_hint(hint, diag.classification)
    session = await _get_session(session_id)
    new_loop = int(session["loop_count"]) + 1
    await _add_turn(
        session_id,
        role="tutor",
        content=content,
        loop_index=new_loop,
        hint_level=None,
        classification=diag.classification.value,
        metadata={
            "reasoning": diag.reasoning,
            "target_concept": diag.target_concept,
            "hint": hint.model_dump(),
            "student_context": student_context,
        },
    )
    await _update_session(
        session_id,
        status=SessionStatus.solved.value,
        loop_count=new_loop,
        resolved=True,
        resolution_type=ResolutionType.solved_with_hints.value,
    )

    # Record mastery — same call as on reveal, so self-solved sessions get
    # credit. Uses `Classification.solved` so the mastery estimator sees the
    # student succeeded.
    try:
        await profile.update_profiles(
            UUID(session["student_id"]),
            session_id,
            concepts,
            diag.target_concept,
            Classification.solved,
            dialogue,
        )
    except Exception as exc:
        logger.warning("profile update failed for session %s: %s", session_id, exc)

    # Generate a session summary for the cross-session learning record.
    await _generate_session_summary(
        session_id,
        UUID(session["student_id"]),
        problem_text=session.get("problem_text", ""),
        concepts=concepts,
        problem_type=session.get("problem_type"),
        outcome="solved",
        target_concept=diag.target_concept,
        dialogue=dialogue,
    )

    return TutorReply(
        session_id=session_id,
        content=content,
        loop_index=new_loop,
        hint_level=None,
        classification=diag.classification,
        offer_reveal=False,
        resolved=True,
        resolution_type=ResolutionType.solved_with_hints,
        hint=hint,
        sources=sources or None,
    )


async def _generate_session_summary(
    session_id: UUID,
    student_id: UUID,
    *,
    problem_text: str,
    concepts: list[str],
    problem_type: str | None,
    outcome: str,
    target_concept: str | None,
    dialogue: str,
    mastery_after: float | None = None,
) -> None:
    """Generate an LLM session summary and persist it to session_summaries.

    Best-effort: logs and swallows errors — summary generation must never
    block session termination.
    """
    try:
        from app.core import llm
        from app.prompts import guided_discovery as p

        settings = config.get_settings()
        raw = await llm.chat_json(
            p.SESSION_SUMMARY_SYSTEM,
            p.session_summary_user(
                problem_text, concepts, problem_type, outcome, target_concept, dialogue
            ),
            temperature=settings.profile_temp,
            max_tokens=settings.profile_max_tokens,
        )
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            summary = f"Student worked on a {problem_type or 'physics'} problem ({outcome})."
        # key_mistakes is now a JSON array from the LLM; join into a single
        # string for the TEXT column. Empty array or null → None.
        raw_mistakes = raw.get("key_mistakes")
        if isinstance(raw_mistakes, list):
            joined = "; ".join(m.strip() for m in raw_mistakes if isinstance(m, str) and m.strip())
            key_mistakes = joined or None
        elif isinstance(raw_mistakes, str) and raw_mistakes.strip():
            key_mistakes = raw_mistakes.strip()
        else:
            key_mistakes = None
        await asyncio.to_thread(
            supabase.add_session_summary,
            student_id,
            session_id,
            problem_text=problem_text,
            concepts=concepts,
            problem_type=problem_type,
            outcome=outcome,
            target_concept=target_concept,
            summary=summary,
            key_mistakes=key_mistakes,
            mastery_after=mastery_after,
        )
    except Exception as exc:
        logger.warning("session summary generation failed for %s: %s", session_id, exc)


async def _maybe_upload_image(student_id: UUID, image_bytes: bytes, mime: str) -> str | None:
    """Best-effort upload to Supabase Storage. Returns public URL or None.

    Skipped entirely on the local backend (no storage configured). The
    `supabase-py` storage client is synchronous, so the upload runs in a
    thread to avoid blocking the event loop.
    """
    if not config.get_settings().supabase_url:
        return None
    try:
        bucket = config.get_settings().supabase_bucket
        path = f"{student_id}/{student_id}.png"

        def _upload() -> str | None:
            client = supabase.get_client()
            client.storage.from_(bucket).upload(path, image_bytes, {"content-type": mime})
            url: str = client.storage.from_(bucket).get_public_url(path)
            return url

        return await asyncio.to_thread(_upload)
    except Exception:
        return None
