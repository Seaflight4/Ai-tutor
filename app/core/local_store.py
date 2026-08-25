"""In-memory storage backend for local development / E2E tests.

Auto-activates when `SUPABASE_URL` is empty. Implements the same surface as
`app.core.supabase` so services don't know which backend they're using.
Backed by SQLite (file: ai_tutor_local.db) so data survives restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.ports.repositories import NotFoundError

# Resolve the DB path relative to the project root so the store works regardless
# of the current working directory (important inside the Docker image).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = str(_PROJECT_ROOT / "ai_tutor_local.db")
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
            problem_type TEXT,
            ocr_raw TEXT NOT NULL DEFAULT '{}',
            loop_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
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
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE (student_id, concept)
        );
        CREATE TABLE IF NOT EXISTS reference_chunks (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_title TEXT NOT NULL,
            source_url TEXT NOT NULL,
            chapter TEXT,
            heading TEXT,
            chunk_text TEXT NOT NULL,
            concepts TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS reference_chunks_concepts_idx
            ON reference_chunks(concepts);
        CREATE TABLE IF NOT EXISTS session_summaries (
            id TEXT PRIMARY KEY,
            student_id TEXT NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            problem_text TEXT NOT NULL,
            concepts TEXT NOT NULL DEFAULT '[]',
            problem_type TEXT,
            outcome TEXT NOT NULL,
            target_concept TEXT,
            summary TEXT NOT NULL,
            key_mistakes TEXT,
            mastery_after REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS session_summaries_student_idx
            ON session_summaries(student_id, created_at DESC);
        """
    )
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that were introduced after the initial schema.

    CREATE TABLE IF NOT EXISTS won't alter an existing table, so old DBs
    that predate a schema change need an explicit ALTER TABLE.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    if "status" not in cols:
        conn.execute(
            "ALTER TABLE sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
        )
    if "problem_type" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN problem_type TEXT")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
            raise NotFoundError(f"student {student_id} not found")
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
    problem_type: str | None = None,
) -> dict[str, Any]:
    sid = str(uuid4())
    with _lock, _db() as conn:
        conn.execute(
            """INSERT INTO sessions
               (id, student_id, subject, problem_text, problem_image_url,
                concepts, problem_type, ocr_raw)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (sid, str(student_id), subject, problem_text, problem_image_url,
             json.dumps(concepts), problem_type, json.dumps(ocr_raw, default=str)),
        )
        conn.commit()
    return get_session(UUID(sid))


def get_session(session_id: UUID) -> dict[str, Any]:
    with _lock, _db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (str(session_id),)).fetchone()
        if not row:
            raise NotFoundError(f"session {session_id} not found")
        d = _row(row)
        d["concepts"] = json.loads(d["concepts"])
        d["ocr_raw"] = json.loads(d["ocr_raw"])
        d["resolved"] = bool(d["resolved"])
        return d


def update_session(
    session_id: UUID,
    *,
    loop_count: int | None = None,
    status: str | None = None,
    resolved: bool | None = None,
    resolution_type: str | None = None,
) -> dict[str, Any]:
    with _lock, _db() as conn:
        if loop_count is not None:
            conn.execute("UPDATE sessions SET loop_count = ?, updated_at = ? WHERE id = ?",
                         (loop_count, _now(), str(session_id)))
        if status is not None:
            conn.execute("UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                         (status, _now(), str(session_id)))
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
) -> dict[str, Any]:
    sid = str(student_id)
    with _lock, _db() as conn:
        existing = conn.execute(
            "SELECT id, attempts FROM knowledge_profiles WHERE student_id = ? AND concept = ?",
            (sid, concept),
        ).fetchone()
        if existing:
            new_attempts = existing["attempts"] + 1
            conn.execute(
                """UPDATE knowledge_profiles
                   SET mastery_score = ?, attempts = ?, last_session_id = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (mastery_score, new_attempts,
                 str(last_session_id) if last_session_id else None, _now(), existing["id"]),
            )
            result_attempts = new_attempts
        else:
            conn.execute(
                """INSERT INTO knowledge_profiles
                   (id, student_id, concept, mastery_score, attempts,
                    last_session_id)
                   VALUES (?, ?, ?, ?, 1, ?)""",
                (str(uuid4()), sid, concept, mastery_score,
                 str(last_session_id) if last_session_id else None),
            )
            result_attempts = 1
        conn.commit()
    return {"student_id": sid, "concept": concept, "mastery_score": mastery_score,
            "attempts": result_attempts,
            "last_session_id": str(last_session_id) if last_session_id else None,
            "updated_at": _now()}


def get_profiles(student_id: UUID) -> list[dict[str, Any]]:
    with _lock, _db() as conn:
        rows = conn.execute(
            "SELECT * FROM knowledge_profiles WHERE student_id = ?",
            (str(student_id),),
        ).fetchall()
        return [_row(r) for r in rows]


# ---------------------------------------------------------------------------
# session_summaries  (compressed learning record for cross-session reference)
# ---------------------------------------------------------------------------
def add_session_summary(
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
    sid = str(uuid4())
    with _lock, _db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO session_summaries
               (id, student_id, session_id, problem_text, concepts,
                problem_type, outcome, target_concept, summary,
                key_mistakes, mastery_after)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sid, str(student_id), str(session_id), problem_text,
             json.dumps(concepts), problem_type, outcome, target_concept,
             summary, key_mistakes, mastery_after),
        )
        conn.commit()
    return {
        "id": sid, "student_id": str(student_id), "session_id": str(session_id),
        "problem_text": problem_text, "concepts": concepts,
        "problem_type": problem_type, "outcome": outcome,
        "target_concept": target_concept, "summary": summary,
        "key_mistakes": key_mistakes, "mastery_after": mastery_after,
        "created_at": _now(),
    }


def list_session_summaries(student_id: UUID, limit: int = 5) -> list[dict[str, Any]]:
    with _lock, _db() as conn:
        rows = conn.execute(
            """SELECT * FROM session_summaries
               WHERE student_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (str(student_id), limit),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row(r)
            d["concepts"] = json.loads(d["concepts"])
            out.append(d)
        return out


def find_related_summaries(
    student_id: UUID,
    concepts: list[str],
    problem_type: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return past sessions sharing at least one concept, prioritized by
    problem_type match. Excludes sessions with no concept overlap."""
    all_summaries = list_session_summaries(student_id, limit=50)
    if not all_summaries or not concepts:
        return []
    concept_set = set(concepts)
    related: list[dict[str, Any]] = []
    for s in all_summaries:
        s_concepts = set(s.get("concepts", []))
        if not (concept_set & s_concepts):
            continue
        s["_type_match"] = bool(
            problem_type
            and s.get("problem_type")
            and problem_type == s["problem_type"]
        )
        related.append(s)
    related.sort(key=lambda x: (x["_type_match"], x["created_at"]), reverse=True)
    for x in related:
        x.pop("_type_match", None)
    return related[:limit]


def get_client() -> Any:
    """No-op for local backend; image upload is skipped."""
    return None


# ---------------------------------------------------------------------------
# reference_chunks  (curated physics reference corpus)
# ---------------------------------------------------------------------------
def add_reference_chunk(
    *,
    source_id: str,
    source_title: str,
    source_url: str,
    chapter: str | None,
    heading: str | None,
    chunk_text: str,
    concepts: list[str],
) -> dict[str, Any]:
    cid = str(uuid4())
    with _lock, _db() as conn:
        conn.execute(
            """INSERT INTO reference_chunks
               (id, source_id, source_title, source_url, chapter, heading,
                chunk_text, concepts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, source_id, source_title, source_url, chapter, heading,
             chunk_text, json.dumps(concepts)),
        )
        conn.commit()
    return {"id": cid, "source_id": source_id, "source_title": source_title,
            "source_url": source_url, "chapter": chapter, "heading": heading,
            "chunk_text": chunk_text, "concepts": concepts}


def list_reference_chunks_by_concepts(concepts: list[str]) -> list[dict[str, Any]]:
    """Return reference chunks tagged with at least one of the given concepts.

    If `concepts` is empty, returns all chunks.
    """
    with _lock, _db() as conn:
        if concepts:
            rows = conn.execute("SELECT * FROM reference_chunks").fetchall()
            rows = [
                r for r in rows
                if set(concepts) & set(json.loads(r["concepts"] or "[]"))
            ]
        else:
            rows = conn.execute("SELECT * FROM reference_chunks").fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = _row(r)
            d["concepts"] = json.loads(d["concepts"] or "[]")
            out.append(d)
        return out


def reset_reference_chunks() -> None:
    """Clear all reference chunks (used by tests)."""
    with _lock, _db() as conn:
        conn.execute("DELETE FROM reference_chunks")
        conn.commit()


def reset_client() -> None:
    """Reset the connection (used by tests)."""
    global _conn
    _conn = None
