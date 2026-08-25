"""Ports: repository + storage interfaces.

The service layer talks to these Protocols, never to `supabase.*` or
`local_store.*` directly. Adapters (`app/adapters/supabase_repo.py`,
`app/adapters/sqlite_repo.py`, `app/adapters/in_memory.py`) implement them.

For PR1 (scaffolding) the signatures mirror the current `app/core/supabase.py`
helpers verbatim — including returning `dict[str, Any]` rows — so services can
keep calling the same surface. PR4 will tighten the contract tests around
the divergences (attempts, _now(), error types) and a later PR will replace
`dict[str, Any]` with the dataclasses in `app/domain/entities.py`.

`NotFoundError` is the canonical missing-row exception. Today the local
backend raises `KeyError` and the Supabase backend raises `IndexError`/
`TypeError`; PR4 will standardize both on `NotFoundError`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID


class NotFoundError(Exception):
    """Raised by a repository when a row lookup misses."""


@runtime_checkable
class StudentRepository(Protocol):
    def upsert_student(
        self, student_id: UUID | None, external_ref: str | None
    ) -> dict[str, Any]: ...

    def get_student(self, student_id: UUID) -> dict[str, Any]: ...


@runtime_checkable
class SessionRepository(Protocol):
    def create_session(
        self,
        student_id: UUID,
        *,
        subject: str,
        problem_text: str,
        problem_image_url: str | None,
        concepts: list[str],
        ocr_raw: dict[str, Any],
        problem_type: str | None = None,
    ) -> dict[str, Any]: ...

    def get_session(self, session_id: UUID) -> dict[str, Any]: ...

    def update_session(
        self,
        session_id: UUID,
        *,
        loop_count: int | None = None,
        status: str | None = None,
        resolved: bool | None = None,
        resolution_type: str | None = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class TurnRepository(Protocol):
    def add_turn(
        self,
        session_id: UUID,
        *,
        role: str,
        content: str,
        loop_index: int,
        hint_level: int | None = None,
        classification: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def list_turns(self, session_id: UUID) -> list[dict[str, Any]]: ...


@runtime_checkable
class ProfileRepository(Protocol):
    def upsert_profile(
        self,
        student_id: UUID,
        *,
        concept: str,
        mastery_score: float,
        last_session_id: UUID | None = None,
    ) -> dict[str, Any]: ...

    def get_profiles(self, student_id: UUID) -> list[dict[str, Any]]: ...


@runtime_checkable
class SessionSummaryRepository(Protocol):
    def add_session_summary(
        self,
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
    ) -> dict[str, Any]: ...

    def list_session_summaries(self, student_id: UUID, limit: int = 5) -> list[dict[str, Any]]: ...

    def find_related_summaries(
        self,
        student_id: UUID,
        concepts: list[str],
        problem_type: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class ReferenceRepository(Protocol):
    def add_reference_chunk(
        self,
        *,
        source_id: str,
        source_title: str,
        source_url: str,
        chapter: str | None,
        heading: str | None,
        chunk_text: str,
        concepts: list[str],
    ) -> dict[str, Any]: ...

    def list_reference_chunks_by_concepts(
        self, concepts: list[str]
    ) -> list[dict[str, Any]]: ...

    def reset_reference_chunks(self) -> None: ...


@runtime_checkable
class StoragePort(Protocol):
    """Object storage (problem images). The local backend is a no-op."""

    def upload(self, key: str, data: bytes, *, content_type: str) -> str | None:
        """Upload and return a public URL, or None if storage is unavailable."""


@runtime_checkable
class Backend(
    StudentRepository,
    SessionRepository,
    TurnRepository,
    ProfileRepository,
    SessionSummaryRepository,
    ReferenceRepository,
    Protocol,
):
    """Aggregate repository interface — the full persistence surface.

    Implemented by `SupabaseBackend` and `SQLiteBackend`. Today both backends
    expose all of these as free module-level functions with identical
    signatures; this Protocol just names that fact. A later PR will bundle
    them into a single object per backend.
    """

    def get_client(self) -> Any: ...
    def reset_client(self) -> None: ...
