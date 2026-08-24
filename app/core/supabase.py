"""Supabase client + thin persistence helpers.

When `SUPABASE_URL` is set, delegates to a real Supabase project. When it is
empty (local dev / E2E), delegates to an in-memory SQLite backend in
`app.core.local_store` so the full flow runs without external services.

Wraps the `supabase-py` client with a handful of typed helpers used across
services. Keeps SQL out of the service layer so the schema stays the single
source of truth in `db/schema.sql`.
"""

from __future__ import annotations

import json
from typing import Any, cast
from uuid import UUID

from app.core import local_store
from app.core.config import get_settings

_settings = get_settings()
_USE_LOCAL = not _settings.supabase_url

_client: Any = None


def _is_local() -> bool:
    return _USE_LOCAL


def get_client() -> Any:
    """Return a module-level Supabase client (singleton). Local backend returns None."""
    if _is_local():
        return local_store.get_client()
    global _client
    if _client is None:
        from supabase import create_client

        _client = create_client(_settings.supabase_url, _settings.supabase_key)
    return _client


def _first(data: Any) -> dict[str, Any]:
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
        raise KeyError(f"student {student_id} not found")
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
) -> dict[str, Any]:
    if _is_local():
        return local_store.create_session(
            student_id,
            subject=subject,
            problem_text=problem_text,
            problem_image_url=problem_image_url,
            concepts=concepts,
            ocr_raw=ocr_raw,
        )
    payload = {
        "student_id": str(student_id),
        "subject": subject,
        "problem_text": problem_text,
        "problem_image_url": problem_image_url,
        "concepts": concepts,
        "ocr_raw": json.loads(json.dumps(ocr_raw)),  # jsonb-safe
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
        raise KeyError(f"session {session_id} not found")
    return row


def update_session(
    session_id: UUID,
    *,
    loop_count: int | None = None,
    resolved: bool | None = None,
    resolution_type: str | None = None,
) -> dict[str, Any]:
    if _is_local():
        return local_store.update_session(
            session_id,
            loop_count=loop_count,
            resolved=resolved,
            resolution_type=resolution_type,
        )
    payload: dict[str, Any] = {}
    if loop_count is not None:
        payload["loop_count"] = loop_count
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
        "metadata": json.loads(json.dumps(metadata or {})),
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
    embedding: list[float] | None = None,
) -> dict[str, Any]:
    if _is_local():
        return local_store.upsert_profile(
            student_id,
            concept=concept,
            mastery_score=mastery_score,
            last_session_id=last_session_id,
            embedding=embedding,
        )
    payload: dict[str, Any] = {
        "student_id": str(student_id),
        "concept": concept,
        "mastery_score": mastery_score,
        "last_session_id": str(last_session_id) if last_session_id else None,
    }
    if embedding is not None:
        payload["embedding"] = embedding
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


def reset_client() -> None:
    """Drop the cached client (used by tests)."""
    global _client
    _client = None
    if _is_local():
        local_store.reset_client()
