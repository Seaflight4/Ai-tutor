"""Adapter: local SQLite persistence (wraps `app/core/local_store.py`).

`SQLiteBackend` bundles the existing module-level functions in
`app/core/local_store.py` into a single object that satisfies the `Backend`
Protocol. The Supabase adapter delegates to this one when `SUPABASE_URL` is
empty (see `app/core/supabase.py`), so this module is the true fallback path.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.core import local_store as _ls


class SQLiteBackend:
    """Object-shaped adapter over the legacy `app.core.local_store` module."""

    def upsert_student(
        self, student_id: UUID | None, external_ref: str | None
    ) -> dict[str, Any]:
        return _ls.upsert_student(student_id, external_ref)

    def get_student(self, student_id: UUID) -> dict[str, Any]:
        return _ls.get_student(student_id)

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
    ) -> dict[str, Any]:
        return _ls.create_session(
            student_id,
            subject=subject,
            problem_text=problem_text,
            problem_image_url=problem_image_url,
            concepts=concepts,
            ocr_raw=ocr_raw,
            problem_type=problem_type,
        )

    def get_session(self, session_id: UUID) -> dict[str, Any]:
        return _ls.get_session(session_id)

    def update_session(
        self,
        session_id: UUID,
        *,
        loop_count: int | None = None,
        status: str | None = None,
        resolved: bool | None = None,
        resolution_type: str | None = None,
    ) -> dict[str, Any]:
        return _ls.update_session(
            session_id,
            loop_count=loop_count,
            status=status,
            resolved=resolved,
            resolution_type=resolution_type,
        )

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
    ) -> dict[str, Any]:
        return _ls.add_turn(
            session_id,
            role=role,
            content=content,
            loop_index=loop_index,
            hint_level=hint_level,
            classification=classification,
            metadata=metadata,
        )

    def list_turns(self, session_id: UUID) -> list[dict[str, Any]]:
        return _ls.list_turns(session_id)

    def upsert_profile(
        self,
        student_id: UUID,
        *,
        concept: str,
        mastery_score: float,
        last_session_id: UUID | None = None,
    ) -> dict[str, Any]:
        return _ls.upsert_profile(
            student_id,
            concept=concept,
            mastery_score=mastery_score,
            last_session_id=last_session_id,
        )

    def get_profiles(self, student_id: UUID) -> list[dict[str, Any]]:
        return _ls.get_profiles(student_id)

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
    ) -> dict[str, Any]:
        return _ls.add_session_summary(
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

    def list_session_summaries(self, student_id: UUID, limit: int = 5) -> list[dict[str, Any]]:
        return _ls.list_session_summaries(student_id, limit)

    def find_related_summaries(
        self,
        student_id: UUID,
        concepts: list[str],
        problem_type: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return _ls.find_related_summaries(student_id, concepts, problem_type, limit)

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
    ) -> dict[str, Any]:
        return _ls.add_reference_chunk(
            source_id=source_id,
            source_title=source_title,
            source_url=source_url,
            chapter=chapter,
            heading=heading,
            chunk_text=chunk_text,
            concepts=concepts,
        )

    def list_reference_chunks_by_concepts(
        self, concepts: list[str]
    ) -> list[dict[str, Any]]:
        return _ls.list_reference_chunks_by_concepts(concepts)

    def reset_reference_chunks(self) -> None:
        _ls.reset_reference_chunks()

    def get_client(self) -> Any:
        return _ls.get_client()

    def reset_client(self) -> None:
        _ls.reset_client()
