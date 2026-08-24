"""In-memory storage backend for local development / E2E tests.

Auto-activates when `SUPABASE_URL` is empty. Implements the same surface as
`app.core.supabase` so services don't know which backend they're using.
Backed by SQLite (file: ai_tutor_local.db) so data survives restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any
from uuid import UUID, uuid4

_DB_PATH = "ai_tutor_local.db"
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _init(_conn)
    return _conn


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            external_ref TEXT,
            profile_summary TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            problem_text TEXT NOT NULL,
            problem_image_url TEXT,
            concepts TEXT NOT NULL DEFAULT '[]',
            ocr_raw TEXT NOT NULL DEFAULT '{}',
            loop_count INTEGER NOT NULL DEFAULT 0,
            resolved INTEGER NOT NULL DEFAULT 0,
            resolution_type TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS turns (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            loop_index INTEGER NOT NULL DEFAULT 0,
            hint_level INTEGER,
            classification TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS knowledge_profiles (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            concept TEXT NOT NULL,
            mastery_score REAL NOT NULL DEFAULT 0.5,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_session_id TEXT,
            embedding TEXT,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE (student_id, concept)
        );
        """
    )
    conn.commit()


def _now() -> str:
    return "2026-01-01T00:00:00Z"


def _row(d: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    if isinstance(d, sqlite3.Row):
        return dict(d)
    return d


# ---------------------------------------------------------------------------
# students
# ---------------------------------------------------------------------------
def upsert_student(student_id: UUID | None, external_ref: str | None) -> dict[str, Any]:
    with _lock, _db() as conn:
        if student_id is not None:
            row = conn.execute(
                "SELECT * FROM students WHERE id = ?", (str(student_id),)
            ).fetchone()
            if row:
                return _row(row)
        sid = str(student_id or uuid4())
        conn.execute(
            "INSERT INTO students (id, external_ref) VALUES (?, ?)",
            (sid, external_ref),
        )
        conn.commit()
        return {"id": sid, "external_ref": external_ref, "profile_summary": None,
                "created_at": _now(), "updated_at": _now()}


def get_student(student_id: UUID) -> dict[str, Any]:
    with _lock, _db() as conn:
        row = conn.execute("SELECT * FROM students WHERE id = ?", (str(student_id),)).fetchone()
        if not row:
            raise KeyError(f"student {student_id} not found")
        return _row(row)


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
def create_session(
    student_id: UUID,
    *,
    subject: str,
    problem_text: str,
    problem_image_url: str | None,
    concepts: list[str],
    ocr_raw: dict[str, Any],
) -> dict[str, Any]:
    sid = str(uuid4())
    with _lock, _db() as conn:
        conn.execute(
            """INSERT INTO sessions
               (id, student_id, subject, problem_text, problem_image_url,
                concepts, ocr_raw)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sid, str(student_id), subject, problem_text, problem_image_url,
             json.dumps(concepts), json.dumps(ocr_raw, default=str)),
        )
        conn.commit()
    return get_session(UUID(sid))


def get_session(session_id: UUID) -> dict[str, Any]:
    with _lock, _db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (str(session_id),)).fetchone()
        if not row:
            raise KeyError(f"session {session_id} not found")
        d = _row(row)
        d["concepts"] = json.loads(d["concepts"])
        d["ocr_raw"] = json.loads(d["ocr_raw"])
        d["resolved"] = bool(d["resolved"])
        return d


def update_session(
    session_id: UUID,
    *,
    loop_count: int | None = None,
    resolved: bool | None = None,
    resolution_type: str | None = None,
) -> dict[str, Any]:
    with _lock, _db() as conn:
        if loop_count is not None:
            conn.execute("UPDATE sessions SET loop_count = ?, updated_at = ? WHERE id = ?",
                         (loop_count, _now(), str(session_id)))
        if resolved is not None:
            conn.execute("UPDATE sessions SET resolved = ?, updated_at = ? WHERE id = ?",
                         (int(resolved), _now(), str(session_id)))
        if resolution_type is not None:
            conn.execute("UPDATE sessions SET resolution_type = ?, updated_at = ? WHERE id = ?",
                         (resolution_type, _now(), str(session_id)))
        conn.commit()
    return get_session(session_id)


# ---------------------------------------------------------------------------
# turns
# ---------------------------------------------------------------------------
def add_turn(
    session_id: UUID,
    *,
    role: str,
    content: str,
    loop_index: int,
    hint_level: int | None = None,
    classification: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tid = str(uuid4())
    with _lock, _db() as conn:
        conn.execute(
            """INSERT INTO turns
               (id, session_id, role, content, loop_index, hint_level,
                classification, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (tid, str(session_id), role, content, loop_index, hint_level,
             classification, json.dumps(metadata or {})),
        )
        conn.commit()
    return {
        "id": tid, "session_id": str(session_id), "role": role, "content": content,
        "loop_index": loop_index, "hint_level": hint_level,
        "classification": classification, "metadata": metadata or {},
        "created_at": _now(),
    }


def list_turns(session_id: UUID) -> list[dict[str, Any]]:
    with _lock, _db() as conn:
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at, rowid",
            (str(session_id),),
        ).fetchall()
        return [_row(r) for r in rows]


# ---------------------------------------------------------------------------
# knowledge_profiles
# ---------------------------------------------------------------------------
def upsert_profile(
    student_id: UUID,
    *,
    concept: str,
    mastery_score: float,
    last_session_id: UUID | None = None,
    embedding: list[float] | None = None,
) -> dict[str, Any]:
    sid = str(student_id)
    with _lock, _db() as conn:
        existing = conn.execute(
            "SELECT id, attempts FROM knowledge_profiles WHERE student_id = ? AND concept = ?",
            (sid, concept),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE knowledge_profiles
                   SET mastery_score = ?, attempts = ?, last_session_id = ?,
                       embedding = ?, updated_at = ?
                   WHERE id = ?""",
                (mastery_score, existing["attempts"] + 1,
                 str(last_session_id) if last_session_id else None,
                 json.dumps(embedding) if embedding else None, _now(), existing["id"]),
            )
        else:
            conn.execute(
                """INSERT INTO knowledge_profiles
                   (id, student_id, concept, mastery_score, attempts,
                    last_session_id, embedding)
                   VALUES (?, ?, ?, ?, 1, ?, ?)""",
                (str(uuid4()), sid, concept, mastery_score,
                 str(last_session_id) if last_session_id else None,
                 json.dumps(embedding) if embedding else None),
            )
        conn.commit()
    return {"student_id": sid, "concept": concept, "mastery_score": mastery_score,
            "attempts": 1, "last_session_id": str(last_session_id) if last_session_id else None,
            "updated_at": _now()}


def get_profiles(student_id: UUID) -> list[dict[str, Any]]:
    with _lock, _db() as conn:
        rows = conn.execute(
            "SELECT * FROM knowledge_profiles WHERE student_id = ?",
            (str(student_id),),
        ).fetchall()
        return [_row(r) for r in rows]


def get_client() -> Any:
    """No-op for local backend; image upload is skipped."""
    return None


def reset_client() -> None:
    """Reset the connection (used by tests)."""
    global _conn
    _conn = None
