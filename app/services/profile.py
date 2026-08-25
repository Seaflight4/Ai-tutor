"""Knowledge profile service: per-concept mastery tracking.

After a session resolves: estimate mastery per concept and upsert into
knowledge_profiles. Mastery is read back on new sessions to surface weak
concepts (mastery_score < threshold) and adjust the opening hint depth.

All persistence calls (`supabase.*`) are synchronous and wrapped in
`asyncio.to_thread` so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from app.core import llm, supabase
from app.core.config import get_settings
from app.models.schemas import Classification
from app.prompts import guided_discovery as p


async def weak_concepts_for(student_id: UUID, concepts: list[str]) -> list[str]:
    """Return concepts the student has historically struggled with.

    Uses the explicit mastery_score stored in knowledge_profiles; no vector
    search needed for the MVP since the session already tags concepts.
    """
    if not concepts:
        return []
    profiles = await asyncio.to_thread(supabase.get_profiles, student_id)
    if not profiles:
        return []
    by_concept = {row["concept"]: row for row in profiles}
    return [
        c for c in concepts if by_concept.get(c, {}).get("mastery_score", 1.0) < 0.5
    ]


async def update_profiles(
    student_id: UUID,
    session_id: UUID,
    concepts: list[str],
    target_concept: str | None,
    classification: Classification,
    dialogue: str,
) -> None:
    """Estimate mastery per concept and upsert into knowledge_profiles."""
    # Choose the single concept most relevant to the final diagnosis.
    concept = target_concept or (concepts[0] if concepts else None)
    if not concept:
        return

    settings = get_settings()
    raw = await llm.chat_json(
        p.PROFILE_UPDATE_SYSTEM,
        p.profile_update_user(concept, classification, dialogue),
        temperature=settings.profile_temp,
        max_tokens=settings.profile_max_tokens,
    )
    try:
        score = float(raw.get("mastery_score", 0.5))
        score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        score = 0.5
    concept = raw.get("concept", concept) or concept

    await asyncio.to_thread(
        supabase.upsert_profile,
        student_id,
        concept=concept,
        mastery_score=score,
        last_session_id=session_id,
    )


async def build_student_context(
    student_id: UUID | None,
    concepts: list[str],
    problem_type: str | None,
) -> str | None:
    """Build a compressed student-context block for the tutor prompt.

    Returns None if the student has no history or no student_id (anonymous).
    """
    if student_id is None:
        return None
    summaries = await asyncio.to_thread(
        supabase.find_related_summaries, student_id, concepts, problem_type, 3
    )
    if not summaries:
        return None
    lines: list[str] = []
    for i, s in enumerate(summaries, 1):
        line = (
            f"[{i}] Type: {s.get('problem_type') or '?'} | "
            f"Concepts: {', '.join(s.get('concepts', []))} | "
            f"Outcome: {s.get('outcome', '?')}"
        )
        if s.get("key_mistakes"):
            line += f" | Mistakes: {s['key_mistakes']}"
        lines.append(line)
    return "\n".join(lines)
