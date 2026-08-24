"""Pydantic schemas (API + internal DTOs)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------
class ResolutionType(StrEnum):
    solved_with_hints = "solved_with_hints"
    revealed = "revealed"
    abandoned = "abandoned"


class Classification(StrEnum):
    knowledge_gap = "knowledge_gap"
    misapplication = "misapplication"
    on_track = "on_track"


class Role(StrEnum):
    tutor = "tutor"
    student = "student"
    system = "system"


# ---------------------------------------------------------------------------
# OCR / problem extraction
# ---------------------------------------------------------------------------
class OCRResult(BaseModel):
    problem_text: str
    formulas: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    topic: str | None = None
    diagram_description: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session + turns
# ---------------------------------------------------------------------------
class SessionCreate(BaseModel):
    external_ref: str | None = None
    student_id: UUID | None = None


class SessionOut(BaseModel):
    id: UUID
    student_id: UUID
    subject: str
    problem_text: str
    problem_image_url: str | None
    concepts: list[str]
    loop_count: int
    resolved: bool
    resolution_type: str | None
    created_at: str


class TurnOut(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    loop_index: int
    hint_level: int | None
    classification: str | None
    created_at: str


class ReplyIn(BaseModel):
    """Student reply to the tutor's opening / hint."""
    message: str


class TutorReply(BaseModel):
    """What the API returns after a student replies."""
    session_id: UUID
    content: str
    loop_index: int
    hint_level: int | None
    classification: Classification | None
    offer_reveal: bool = False
    resolved: bool = False
    resolution_type: ResolutionType | None = None
    solution: str | None = None


class RevealOut(BaseModel):
    session_id: UUID
    solution: str
    resolved: bool
    resolution_type: ResolutionType


# ---------------------------------------------------------------------------
# Diagnosis (LLM JSON output)
# ---------------------------------------------------------------------------
class Diagnosis(BaseModel):
    classification: Classification
    reasoning: str = ""
    target_concept: str | None = None
    next_hint_level: int = 1  # 1..3


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
class ProfileEntry(BaseModel):
    concept: str
    mastery_score: float
    attempts: int
    last_session_id: UUID | None
    updated_at: str | None


class ProfileOut(BaseModel):
    student_id: UUID
    profile_summary: str | None
    entries: list[ProfileEntry]
