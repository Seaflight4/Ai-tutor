"""HTTP routes for the AI tutor."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.core import supabase
from app.models.schemas import (
    ProfileEntry,
    ProfileOut,
    ReplyIn,
    RevealOut,
    SessionOut,
    TurnOut,
    TutorReply,
)
from app.services import session as session_service

router = APIRouter()

_ACCEPTED_IMAGE_TYPES = {"image/png", "image/jpeg"}


def _normalize_image(upload: UploadFile) -> tuple[bytes, str]:
    content_type = upload.content_type or "image/png"
    if content_type not in _ACCEPTED_IMAGE_TYPES:
        raise HTTPException(415, f"unsupported image type: {content_type}")
    raw = upload.file.read()
    # Convert JPEG -> PNG bytes so the OCR model receives a stable format.
    if content_type == "image/jpeg":
        import io

        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
        content_type = "image/png"
    return raw, content_type


@router.post("/sessions", response_model=SessionOut, status_code=201)
async def create_session(
    file: UploadFile = File(...),  # noqa: B008
    student_id: UUID | None = Form(None),  # noqa: B008
    external_ref: str | None = Form(None),  # noqa: B008
) -> SessionOut:
    """Upload a problem image and start a guided-discovery session."""
    raw, mime = _normalize_image(file)
    session = await session_service.start_session(raw, mime, student_id, external_ref)
    # `opening` is attached by the service but not part of SessionOut.
    session.pop("opening", None)
    return _to_session_out(session)


@router.post("/sessions/{session_id}/reply", response_model=TutorReply)
async def reply(session_id: UUID, body: ReplyIn) -> TutorReply:
    """Student replies; tutor responds with the next hint or a reveal offer."""
    try:
        return await session_service.reply(session_id, body.message)
    except KeyError as exc:
        raise HTTPException(404, "session not found") from exc


@router.post("/sessions/{session_id}/reveal", response_model=RevealOut)
async def reveal(session_id: UUID) -> RevealOut:
    """Force-reveal the full solution."""
    try:
        out = await session_service.reveal_solution(session_id)
    except KeyError as exc:
        raise HTTPException(404, "session not found") from exc
    return RevealOut(**out)


@router.get("/sessions/{session_id}")
async def get_session(session_id: UUID) -> dict[str, Any]:
    """Return the session metadata and full transcript."""
    try:
        session = supabase.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(404, "session not found") from exc
    turns = supabase.list_turns(session_id)
    return {
        "session": _to_session_out(session),
        "turns": [_to_turn_out(t) for t in turns],
    }


@router.get("/students/{student_id}/profile", response_model=ProfileOut)
async def get_profile(student_id: UUID) -> ProfileOut:
    try:
        student = supabase.get_student(student_id)
    except KeyError as exc:
        raise HTTPException(404, "student not found") from exc
    profiles = supabase.get_profiles(student_id)
    return ProfileOut(
        student_id=student_id,
        profile_summary=student.get("profile_summary"),
        entries=[
            ProfileEntry(
                concept=p["concept"],
                mastery_score=p["mastery_score"],
                attempts=p["attempts"],
                last_session_id=p.get("last_session_id"),
                updated_at=p.get("updated_at"),
            )
            for p in profiles
        ],
    )


# ---------------------------------------------------------------------------
# mappers
# ---------------------------------------------------------------------------
def _to_session_out(row: dict[str, Any]) -> SessionOut:
    return SessionOut(
        id=row["id"],
        student_id=row["student_id"],
        subject=row["subject"],
        problem_text=row["problem_text"],
        problem_image_url=row.get("problem_image_url"),
        concepts=row.get("concepts") or [],
        loop_count=row["loop_count"],
        resolved=row["resolved"],
        resolution_type=row.get("resolution_type"),
        created_at=row["created_at"],
    )


def _to_turn_out(row: dict[str, Any]) -> TurnOut:
    return TurnOut(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        loop_index=row["loop_index"],
        hint_level=row.get("hint_level"),
        classification=row.get("classification"),
        created_at=row["created_at"],
    )
