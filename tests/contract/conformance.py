"""Contract test suite for the persistence `Backend` Protocol.

Runs the SAME assertions against every concrete backend so behavioral
divergences (attempts counting, missing-row exceptions, status transitions,
JSON round-trips, timestamp formats) surface as test failures rather than as
silent production bugs.

Backends covered:
- `InMemoryBackend` (the test double in `app/adapters/in_memory.py`)
- `SQLiteBackend` pointed at a temp file (exercises `app/core/local_store.py`)
- `SupabaseBackend` backed by a fluent fake client (exercises the
  `app/core/supabase.py` branching logic without a network)

Each backend gets its own fixture in `conftest.py`; the shared assertions
live here as plain functions that accept a `backend` implementing the
`Backend` Protocol. The fixtures are wired via `pytest.mark.parametrize` in
`test_backend_contract.py`.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from app.ports.repositories import NotFoundError


# ---------------------------------------------------------------------------
# Helpers: the contract tests create rows through the backend's own surface
# so they don't depend on a specific DB schema's column ordering.
# ---------------------------------------------------------------------------
def _make_student(backend: Any, external_ref: str = "stu") -> UUID:
    row = backend.upsert_student(None, external_ref)
    return UUID(row["id"])


def _make_session(backend: Any, student_id: UUID) -> UUID:
    row = backend.create_session(
        student_id,
        subject="physics",
        problem_text="A block slides down a ramp.",
        problem_image_url=None,
        concepts=["energy conservation"],
        ocr_raw={"ocr_markdown": "A block slides down a ramp."},
    )
    return UUID(row["id"])


# ---------------------------------------------------------------------------
# students
# ---------------------------------------------------------------------------
def test_upsert_student_returns_existing_for_known_id(backend: Any) -> None:
    sid = _make_student(backend, "stu-1")
    again = backend.upsert_student(sid, "stu-1")
    assert UUID(again["id"]) == sid
    assert again["external_ref"] == "stu-1"


def test_upsert_student_creates_new_for_none_id(backend: Any) -> None:
    row = backend.upsert_student(None, "stu-new")
    assert row["id"]
    assert row["external_ref"] == "stu-new"


def test_get_student_missing_raises_not_found_error(backend: Any) -> None:
    with pytest.raises(NotFoundError):
        backend.get_student(uuid4())


def test_get_student_returns_row(backend: Any) -> None:
    sid = _make_student(backend, "stu-2")
    row = backend.get_student(sid)
    assert UUID(row["id"]) == sid
    assert row["external_ref"] == "stu-2"


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
def test_create_session_defaults(backend: Any) -> None:
    sid = _make_student(backend)
    session_id = _make_session(backend, sid)
    row = backend.get_session(session_id)
    assert UUID(row["id"]) == session_id
    assert UUID(row["student_id"]) == sid
    assert row["subject"] == "physics"
    assert row["problem_text"] == "A block slides down a ramp."
    assert row["concepts"] == ["energy conservation"]
    assert row["loop_count"] == 0
    assert row["resolved"] is False
    assert row.get("status") == "active"
    assert row["resolution_type"] is None


def test_get_session_missing_raises_not_found_error(backend: Any) -> None:
    with pytest.raises(NotFoundError):
        backend.get_session(uuid4())


def test_update_session_status_and_loop(backend: Any) -> None:
    sid = _make_student(backend)
    session_id = _make_session(backend, sid)
    backend.update_session(
        session_id, loop_count=2, status="active"
    )
    row = backend.get_session(session_id)
    assert row["loop_count"] == 2
    assert row["status"] == "active"


def test_update_session_to_revealed(backend: Any) -> None:
    sid = _make_student(backend)
    session_id = _make_session(backend, sid)
    backend.update_session(
        session_id, status="revealed", resolved=True, resolution_type="revealed"
    )
    row = backend.get_session(session_id)
    assert row["status"] == "revealed"
    assert row["resolved"] is True
    assert row["resolution_type"] == "revealed"


# ---------------------------------------------------------------------------
# turns
# ---------------------------------------------------------------------------
def test_add_and_list_turns_preserves_order(backend: Any) -> None:
    sid = _make_student(backend)
    session_id = _make_session(backend, sid)
    backend.add_turn(session_id, role="tutor", content="opening", loop_index=0)
    backend.add_turn(session_id, role="student", content="help", loop_index=1)
    backend.add_turn(session_id, role="tutor", content="hint", loop_index=1)
    turns = backend.list_turns(session_id)
    assert len(turns) == 3
    assert [t["role"] for t in turns] == ["tutor", "student", "tutor"]
    assert turns[0]["content"] == "opening"
    assert turns[2]["loop_index"] == 1


def test_list_turns_empty_for_unknown_session(backend: Any) -> None:
    # An unknown session returns an empty list (turns are scoped to a session;
    # missing session is not an error at the list level).
    assert backend.list_turns(uuid4()) == []


# ---------------------------------------------------------------------------
# knowledge_profiles
# ---------------------------------------------------------------------------
def test_upsert_profile_increments_attempts(backend: Any) -> None:
    """The headline contract: a second upsert for the same (student, concept)
    MUST increment attempts, not reset it to 1. This was a real divergence
    before PR4 (Supabase reset to 0, local returned a hardcoded 1)."""
    sid = _make_student(backend, "stu-prof")
    session_id = _make_session(backend, sid)

    first = backend.upsert_profile(
        sid, concept="energy conservation", mastery_score=0.4,
        last_session_id=session_id,
    )
    assert first["attempts"] == 1

    second = backend.upsert_profile(
        sid, concept="energy conservation", mastery_score=0.6,
        last_session_id=session_id,
    )
    assert second["attempts"] == 2, "second upsert must increment attempts"
    assert second["mastery_score"] == 0.6


def test_upsert_profile_different_concepts_separate(backend: Any) -> None:
    sid = _make_student(backend)
    session_id = _make_session(backend, sid)
    backend.upsert_profile(sid, concept="kinematics", mastery_score=0.3,
                           last_session_id=session_id)
    backend.upsert_profile(sid, concept="energy conservation", mastery_score=0.5,
                           last_session_id=session_id)
    profiles = backend.get_profiles(sid)
    concepts = {p["concept"] for p in profiles}
    assert concepts == {"kinematics", "energy conservation"}


def test_get_profiles_empty_for_unknown_student(backend: Any) -> None:
    assert backend.get_profiles(uuid4()) == []


# ---------------------------------------------------------------------------
# reference_chunks
# ---------------------------------------------------------------------------
def test_reference_chunks_concept_filter(backend: Any) -> None:
    backend.reset_reference_chunks()
    backend.add_reference_chunk(
        source_id="openstax", source_title="OpenStax", source_url="https://openstax.org",
        chapter="Ch. 5", heading="Energy", chunk_text="Energy is conserved.",
        concepts=["energy conservation"],
    )
    backend.add_reference_chunk(
        source_id="openstax", source_title="OpenStax", source_url="https://openstax.org",
        chapter="Ch. 3", heading="Kinematics", chunk_text="v = u + at.",
        concepts=["kinematics"],
    )
    hits = backend.list_reference_chunks_by_concepts(["energy conservation"])
    assert len(hits) == 1
    assert hits[0]["concepts"] == ["energy conservation"]
    assert "Energy" in hits[0]["heading"]


def test_reference_chunks_empty_filter_returns_all(backend: Any) -> None:
    backend.reset_reference_chunks()
    backend.add_reference_chunk(
        source_id="a", source_title="A", source_url="u",
        chapter=None, heading=None, chunk_text="x",
        concepts=["kinematics"],
    )
    all_hits = backend.list_reference_chunks_by_concepts([])
    assert len(all_hits) == 1


def test_reset_reference_chunks_clears(backend: Any) -> None:
    backend.add_reference_chunk(
        source_id="a", source_title="A", source_url="u",
        chapter=None, heading=None, chunk_text="x",
        concepts=["x"],
    )
    backend.reset_reference_chunks()
    assert backend.list_reference_chunks_by_concepts([]) == []
