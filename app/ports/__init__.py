"""Ports — interfaces the service layer depends on.

A "port" is a seam defined in the domain/service layer; an "adapter" is an
implementation of that port in the outside world (a specific LLM gateway, a
specific database). Services import from `app.ports`, never from
`app.adapters` or `app.core`. Wiring (which adapter implements which port)
happens in `app/main.py`.
"""

from __future__ import annotations

from app.ports.llm import LLMClient
from app.ports.repositories import (
    Backend,
    NotFoundError,
    ProfileRepository,
    ReferenceRepository,
    SessionRepository,
    StoragePort,
    StudentRepository,
    TurnRepository,
)

__all__ = [
    "Backend",
    "LLMClient",
    "NotFoundError",
    "ProfileRepository",
    "ReferenceRepository",
    "SessionRepository",
    "StoragePort",
    "StudentRepository",
    "TurnRepository",
]
