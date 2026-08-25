"""Persistent record dataclasses — the shape of rows in any backend.

These describe the *storage* representation of the core entities, independent
of whether they came from Supabase or the local SQLite fallback. A later PR
will have the repository adapters return these instead of `dict[str, Any]`;
for PR1 they exist so the repository `Protocol`s (in `app/ports/`) can be
typed against something concrete and the contract tests have a target shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class StudentRecord:
    id: UUID
    external_ref: str | None = None
    profile_summary: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class SessionRecord:
    id: UUID
    student_id: UUID
    subject: str
    problem_text: str
    problem_image_url: str | None
    concepts: list[str]
    ocr_raw: dict[str, Any] = field(default_factory=dict)
    loop_count: int = 0
    resolved: bool = False
    resolution_type: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class TurnRecord:
    id: UUID
    session_id: UUID
    role: str
    content: str
    loop_index: int
    hint_level: int | None = None
    classification: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass
class ProfileRecord:
    student_id: UUID
    concept: str
    mastery_score: float
    attempts: int = 0
    last_session_id: UUID | None = None
    updated_at: str | None = None


@dataclass
class ReferenceChunkRecord:
    source_id: str
    source_title: str
    source_url: str
    chunk_text: str
    concepts: list[str] = field(default_factory=list)
    id: UUID | None = None
    chapter: str | None = None
    heading: str | None = None
