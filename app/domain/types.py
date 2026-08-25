"""Domain value types — enums and value objects shared across layers.

These are the canonical home for the enums (`Classification`, `ResolutionType`,
`Role`) and the structured LLM-output value objects (`HintOutput`,
`Diagnosis`, `OCRResult`, `ReferenceChunk`).

For PR1 (scaffolding) these re-export the Pydantic models that already live in
`app/models/schemas.py`. A later PR will migrate them to plain dataclasses
under `app/domain/` so the domain layer no longer depends on Pydantic; the
Pydantic API schemas in `app/models/schemas.py` will then subclass / wrap
these domain types. This file is the seam that makes that migration local.
"""

from __future__ import annotations

from app.models.schemas import (
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
    "ReferenceChunk",
    "ResolutionType",
    "Role",
]
