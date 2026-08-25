"""Supabase client + thin persistence helpers.

When `SUPABASE_URL` is set, delegates to a real Supabase project. When it is
empty (local dev / E2E), delegates to an in-memory SQLite backend in
`app.core.local_store` so the full flow runs without external services.

Wraps the `supabase-py` client with a handful of typed helpers used across
services. Keeps SQL out of the service layer so the schema stays the single
source of truth in `db/schema.sql`.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from app.core import local_store
from app.core.config import get_settings
from app.ports.repositories import NotFoundError

_client: Any = None


def _is_local() -> bool:
    return not get_settings().supabase_url


def get_client() -> Any:
    """Return a module-level Supabase client (singleton). Local backend returns None."""
    if _is_local():
        return local_store.get_client()
    global _client
    if _client is None:
        from supabase import create_client

        settings = get_settings()
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


def _first(data: Any) -> dict[str, Any]:
    if not data:
        raise IndexError("expected at least one row, got none")
    return cast(dict[str, Any], data[0])


def _row(data: Any) -> dict[str, Any]:
    return cast(dict[str, Any], data)


def _maybe_single_data(res: Any) -> dict[str, Any] | None:
    """Extract data from a maybe_single() response, tolerating None."""
    if res is None:
        return None
    data = res.data
    return _row(data) if data else None


# ---------------------------------------------------------------------------
# students
# ---------------------------------------------------------------------------
def upsert_student(student_id: UUID | None, external_ref: str | None) -> dict[str, Any]:
    if _is_local():
        return local_store.upsert_student(student_id, external_ref)
    client = get_client()
    if student_id is not None:
        res = (
            client.table("students")
            .select("*")
            .eq("id", str(student_id))
            .maybe_single()
            .execute()
        )
        existing = _maybe_single_data(res)
        if existing is not None:
            return existing
    payload = {"external_ref": external_ref}
    insert_res = client.table("students").insert(payload).execute()
    return _first(insert_res.data)


def get_student(student_id: UUID) -> dict[str, Any]:
    if _is_local():
        return local_store.get_student(student_id)
    res = (
        get_client()
        .table("students")
        .select("*")
        .eq("id", str(student_id))
        .maybe_single()
        .execute()
    )
    row = _maybe_single_data(res)
    if row is None:
        raise NotFoundError(f"student {student_id} not found")
    return row


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
def create_session(
    student_id: UUID,
    *,
    subject: str,
    problem_text: str,
    problem_image_url: str | None,
    concepts: list[str],
    ocr_raw: dict[str, Any],
    problem_type: str | None = None,
) -> dict[str, Any]:
    if _is_local():
        return local_store.create_session(
            student_id,
            subject=subject,
            problem_text=problem_text,
            problem_image_url=problem_image_url,
            concepts=concepts,
            ocr_raw=ocr_raw,
            problem_type=problem_type,
        )
    payload = {
        "student_id": str(student_id),
        "subject": subject,
        "problem_text": problem_text,
        "problem_image_url": problem_image_url,
        "concepts": concepts,
        "problem_type": problem_type,
        "ocr_raw": ocr_raw,
    }
    res = get_client().table("sessions").insert(payload).execute()
    return _first(res.data)


def get_session(session_id: UUID) -> dict[str, Any]:
    if _is_local():
        return local_store.get_session(session_id)
    res = (
        get_client()
        .table("sessions")
        .select("*")
        .eq("id", str(session_id))
        .maybe_single()
        .execute()
    )
    row = _maybe_single_data(res)
    if row is None:
        raise NotFoundError(f"session {session_id} not found")
    return row


def update_session(
    session_id: UUID,
    *,
    loop_count: int | None = None,
    status: str | None = None,
    resolved: bool | None = None,
    resolution_type: str | None = None,
) -> dict[str, Any]:
    if _is_local():
        return local_store.update_session(
            session_id,
            loop_count=loop_count,
            status=status,
            resolved=resolved,
            resolution_type=resolution_type,
        )
    payload: dict[str, Any] = {}
    if loop_count is not None:
        payload["loop_count"] = loop_count
    if status is not None:
        payload["status"] = status
    if resolved is not None:
        payload["resolved"] = resolved
    if resolution_type is not None:
        payload["resolution_type"] = resolution_type
    if not payload:
        return get_session(session_id)
    res = (
        get_client()
        .table("sessions")
        .update(payload)
        .eq("id", str(session_id))
        .execute()
    )
    if not res.data:
        raise NotFoundError(f"session {session_id} not found")
    return _first(res.data)


# ---------------------------------------------------------------------------
# turns
# ---------------------------------------------------------------------------
def add_turn(
    session_id: UUID,
    *,
    role: str,
    content: str,
    loop_index: int,
    hint_level: int | None = None,
    classification: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _is_local():
        return local_store.add_turn(
            session_id,
            role=role,
            content=content,
            loop_index=loop_index,
            hint_level=hint_level,
            classification=classification,
            metadata=metadata,
        )
    payload = {
        "session_id": str(session_id),
        "role": role,
        "content": content,
        "loop_index": loop_index,
        "hint_level": hint_level,
        "classification": classification,
        "metadata": metadata or {},
    }
    res = get_client().table("turns").insert(payload).execute()
    return _first(res.data)


def list_turns(session_id: UUID) -> list[dict[str, Any]]:
    if _is_local():
        return local_store.list_turns(session_id)
    res = (
        get_client()
        .table("turns")
        .select("*")
        .eq("session_id", str(session_id))
        .order("created_at")
        .execute()
    )
    return cast(list[dict[str, Any]], res.data)


# ---------------------------------------------------------------------------
# knowledge_profiles
# ---------------------------------------------------------------------------
def upsert_profile(
    student_id: UUID,
    *,
    concept: str,
    mastery_score: float,
    last_session_id: UUID | None = None,
) -> dict[str, Any]:
    if _is_local():
        return local_store.upsert_profile(
            student_id,
            concept=concept,
            mastery_score=mastery_score,
            last_session_id=last_session_id,
        )
    payload: dict[str, Any] = {
        "student_id": str(student_id),
        "concept": concept,
        "mastery_score": mastery_score,
        "last_session_id": str(last_session_id) if last_session_id else None,
    }
    # Read existing attempts so the upsert increments rather than resetting to
    # the DB default of 0 (mirrors the local_store read-then-write). Without
    # this, every upsert on an existing (student_id, concept) would reset the
    # attempt counter.
    existing = (
        get_client()
        .table("knowledge_profiles")
        .select("attempts")
        .eq("student_id", str(student_id))
        .eq("concept", concept)
        .maybe_single()
        .execute()
    )
    existing_row = _maybe_single_data(existing)
    payload["attempts"] = (existing_row or {}).get("attempts", 0) + 1
    res = (
        get_client()
        .table("knowledge_profiles")
        .upsert(payload, on_conflict="student_id,concept")
        .execute()
    )
    return _first(res.data)


def get_profiles(student_id: UUID) -> list[dict[str, Any]]:
    if _is_local():
        return local_store.get_profiles(student_id)
    res = (
        get_client()
        .table("knowledge_profiles")
        .select("*")
        .eq("student_id", str(student_id))
        .execute()
    )
    return cast(list[dict[str, Any]], res.data)


# ---------------------------------------------------------------------------
# session_summaries  (compressed learning record for cross-session reference)
# ---------------------------------------------------------------------------
def add_session_summary(
    student_id: UUID,
    session_id: UUID,
    *,
    problem_text: str,
    concepts: list[str],
    problem_type: str | None,
    outcome: str,
    target_concept: str | None,
    summary: str,
    key_mistakes: str | None = None,
    mastery_after: float | None = None,
) -> dict[str, Any]:
    if _is_local():
        return local_store.add_session_summary(
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
    payload = {
        "student_id": str(student_id),
        "session_id": str(session_id),
        "problem_text": problem_text,
        "concepts": concepts,
        "problem_type": problem_type,
        "outcome": outcome,
        "target_concept": target_concept,
        "summary": summary,
        "key_mistakes": key_mistakes,
        "mastery_after": mastery_after,
    }
    res = (
        get_client()
        .table("session_summaries")
        .upsert(payload, on_conflict="session_id")
        .execute()
    )
    return _first(res.data)


def list_session_summaries(student_id: UUID, limit: int = 5) -> list[dict[str, Any]]:
    if _is_local():
        return local_store.list_session_summaries(student_id, limit)
    res = (
        get_client()
        .table("session_summaries")
        .select("*")
        .eq("student_id", str(student_id))
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return cast(list[dict[str, Any]], res.data)


def find_related_summaries(
    student_id: UUID,
    concepts: list[str],
    problem_type: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if _is_local():
        return local_store.find_related_summaries(student_id, concepts, problem_type, limit)
    # Supabase: fetch recent summaries and filter in Python (concept overlap
    # with array containment is possible via PostgREST but doing it in Python
    # keeps the local and remote paths identical).
    all_summaries = list_session_summaries(student_id, limit=50)
    if not all_summaries or not concepts:
        return []
    concept_set = set(concepts)
    related: list[dict[str, Any]] = []
    for s in all_summaries:
        s_concepts = set(s.get("concepts", []))
        if not (concept_set & s_concepts):
            continue
        s["_type_match"] = bool(
            problem_type
            and s.get("problem_type")
            and problem_type == s["problem_type"]
        )
        related.append(s)
    related.sort(key=lambda x: (x["_type_match"], x.get("created_at", "")), reverse=True)
    for x in related:
        x.pop("_type_match", None)
    return related[:limit]


# ---------------------------------------------------------------------------
# reference_chunks  (curated physics reference corpus for source grounding)
# ---------------------------------------------------------------------------
def add_reference_chunk(
    *,
    source_id: str,
    source_title: str,
    source_url: str,
    chapter: str | None,
    heading: str | None,
    chunk_text: str,
    concepts: list[str],
) -> dict[str, Any]:
    """Insert a reference chunk. Used by the ingest script, not at request time."""
    if _is_local():
        return local_store.add_reference_chunk(
            source_id=source_id,
            source_title=source_title,
            source_url=source_url,
            chapter=chapter,
            heading=heading,
            chunk_text=chunk_text,
            concepts=concepts,
        )
    payload: dict[str, Any] = {
        "source_id": source_id,
        "source_title": source_title,
        "source_url": source_url,
        "chapter": chapter,
        "heading": heading,
        "chunk_text": chunk_text,
        "concepts": concepts,
    }
    res = get_client().table("reference_chunks").insert(payload).execute()
    return _first(res.data)


def list_reference_chunks_by_concepts(concepts: list[str]) -> list[dict[str, Any]]:
    """Return reference chunks tagged with at least one of the given concepts.

    Fetches metadata + chunk_text; the retrieval service re-ranks by keyword
    overlap (no vector search).
    """
    if _is_local():
        return local_store.list_reference_chunks_by_concepts(concepts)
    query = get_client().table("reference_chunks").select(
        "id,source_id,source_title,source_url,chapter,heading,chunk_text,concepts"
    )
    if concepts:
        # Reject concepts containing PostgREST filter metacharacters that
        # would break the `or_`/`cs` syntax or allow filter injection. Spaces
        # and word characters are safe inside the `{...}` array literal.
        # Concepts come from the LLM (ocr.py) so this is a real attack surface.
        unsafe = set(",.()")
        safe = [c for c in concepts if c and not any(ch in c for ch in unsafe)]
        if safe:
            query = query.or_(",".join(f"concepts.cs.{{{c}}}" for c in safe))
    res = query.execute()
    return cast(list[dict[str, Any]], res.data)


def reset_reference_chunks() -> None:
    """Clear all reference chunks. Used by tests and corpus re-ingestion.

    On the local backend this deletes from the SQLite table; on Supabase it
    issues a delete against the `reference_chunks` table so both backends
    behave identically (previously this was a silent no-op on Supabase, which
    let stale chunks leak between test runs and re-ingestions).
    """
    if _is_local():
        local_store.reset_reference_chunks()
        return
    get_client().table("reference_chunks").delete().neq("id", "00000000").execute()


def reset_client() -> None:
    """Drop the cached client (used by tests)."""
    global _client
    _client = None
    if _is_local():
        local_store.reset_client()
