"""Domain entities — pure dataclasses, no I/O, no Pydantic.

These describe the core business objects (Session, Turn, Profile, ...) in
transport-agnostic terms. The persistence adapters translate between these
and their respective row formats (dict[str, Any] today; will move to entities
fully in a later PR). The API layer translates entities -> Pydantic response
models.
"""

from __future__ import annotations

from app.domain.entities import (
    ProfileRecord,
    ReferenceChunkRecord,
    SessionRecord,
    StudentRecord,
    TurnRecord,
)
from app.domain.state import SessionStatus, SessionTerminalError, can_reply, is_terminal, transition
from app.domain.types import (
    Classification,
    Diagnosis,
    HintOutput,
    OCRResult,
    ReferenceChunk,
    ResolutionType,
    Role,
)

__all__ = [
    "Classification",
    "Diagnosis",
    "HintOutput",
    "OCRResult",
    "ProfileRecord",
    "ReferenceChunk",
    "ReferenceChunkRecord",
    "ResolutionType",
    "Role",
    "SessionRecord",
    "SessionStatus",
    "SessionTerminalError",
    "StudentRecord",
    "TurnRecord",
    "can_reply",
    "is_terminal",
    "transition",
]
