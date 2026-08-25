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
    answer_check = "answer_check"          # student asks to verify a proposed answer
    incorrect_answer = "incorrect_answer"  # student states a wrong final answer
    solved = "solved"                       # student has demonstrated a full correct solution
    meta = "meta"                           # procedural / clarification / off-topic


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
    problem_type: str | None = None
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
    problem_type: str | None = None
    loop_count: int
    status: str = "active"
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

    message: str = Field(min_length=1, max_length=4000)


class ReferenceChunk(BaseModel):
    """A retrieved passage from the curated physics reference corpus.

    Carried through to the prompt as grounding context and surfaced to the
    student as a citation.
    """

    id: UUID | None = None
    source_id: str
    source_title: str
    source_url: str
    chapter: str | None = None
    heading: str | None = None
    chunk_text: str
    concepts: list[str] = Field(default_factory=list)


class HintOutput(BaseModel):
    """Structured hint returned to the student, scoped to the diagnosis.

    For a knowledge_gap: fill `explanation` (definition), `formula`, `example`.
    For a misapplication: fill `mistake`, `reason`, `application_hint`.
    For on_track: fill `confirmation` (affirmation) + `next_step_hint` (small nudge).
    Unused fields stay null.

    `source_title`/`source_url` cite the reference chunk a knowledge_gap or
    formula-bearing misapplication hint was grounded in. Null when no source
    covered the concept (the model is told to say so rather than invent).
    """

    formula: str | None = None
    explanation: str | None = None
    example: str | None = None
    mistake: str | None = None
    reason: str | None = None
    application_hint: str | None = None
    confirmation: str | None = None
    next_step_hint: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    # answer_check branch
    answer_status: str | None = None     # "correct" | "incorrect" | "partial"
    answer_value: str | None = None      # the value the student proposed, echoed back
    method_feedback: str | None = None   # brief feedback on the method (1 sentence)
    # meta branch
    meta_response: str | None = None     # free-text answer to a procedural/clarification question


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
    hint: HintOutput | None = None
    sources: list[ReferenceChunk] | None = None


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
    wants_solution: bool = False


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
