"""Knowledge profile service: personalization via pgvector.

On new session: embed the problem text -> cosine similarity search against the
student's `knowledge_profiles` -> surface prior weak concepts -> adjust the
opening hint depth.

After a session resolves: upsert mastery scores per concept touched.
"""

from __future__ import annotations

from uuid import UUID

from app.core import llm, supabase
from app.models.schemas import Classification
from app.prompts import guided_discovery as p


async def weak_concepts_for(student_id: UUID, concepts: list[str]) -> list[str]:
    """Return concepts the student has historically struggled with.

    Uses the explicit mastery_score stored in knowledge_profiles; no vector
    search needed for the MVP since the session already tags concepts.
    """
    if not concepts:
        return []
    profiles = supabase.get_profiles(student_id)
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

    raw = await llm.chat_json(
        p.PROFILE_UPDATE_SYSTEM,
        p.profile_update_user(concept, classification, dialogue),
        temperature=0.0,
        max_tokens=200,
    )
    try:
        score = float(raw.get("mastery_score", 0.5))
        score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        score = 0.5
    concept = raw.get("concept", concept) or concept

    embedding = await llm.embed(f"{concept}: {dialogue[-1000:]}")
    supabase.upsert_profile(
        student_id,
        concept=concept,
        mastery_score=score,
        last_session_id=session_id,
        embedding=embedding,
    )
