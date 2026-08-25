"""In-memory adapters — the test/double implementations of the ports.

`InMemoryLLM` and `InMemoryBackend` are moved here from `tests/conftest.py`
so they live next to the other adapters and can be reused by contract tests
(`tests/contract/`) and by any caller that wants a zero-dependency run. The
test fixtures in `tests/conftest.py` re-import these and monkeypatch the
legacy module-level functions in `app.core.llm` / `app.core.supabase`, which
keeps the existing tests working unchanged in PR1.

These structs intentionally mirror the *current* surface (free functions
returning `dict[str, Any]`) rather than the eventual dataclass return types.
A later PR will tighten them against the `app.ports` Protocols.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from app.ports.repositories import NotFoundError


# ---------------------------------------------------------------------------
# LLM fake
# ---------------------------------------------------------------------------
class InMemoryLLM:
    """Queue-based fake for the LLM client. Drop-in for `app.core.llm`."""

    def __init__(self) -> None:
        self.ocr_responses: list[str] = ["A block slides down a frictionless ramp..."]
        self.json_responses: list[dict[str, Any]] = []
        self.text_responses: list[str] = []
        self.calls: list[dict[str, Any]] = []

    async def ocr_image(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        mime: str = "image/png",
        max_tokens: int = 2000,
    ) -> str:
        self.calls.append({"kind": "ocr", "prompt": prompt})
        return self.ocr_responses.pop(0) if self.ocr_responses else "OCR"

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        self.calls.append({"kind": "json", "system": system, "user": user})
        return self.json_responses.pop(0) if self.json_responses else {}

    async def chat_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 800,
    ) -> str:
        self.calls.append({"kind": "text", "system": system, "user": user})
        return self.text_responses.pop(0) if self.text_responses else "OK"


# ---------------------------------------------------------------------------
# In-memory backend (fake supabase)
# ---------------------------------------------------------------------------
class InMemoryBackend:
    """In-memory persistence fake. Implements the same surface as
    `app.core.supabase` / `app.core.local_store`."""

    def __init__(self) -> None:
        self.students: dict[UUID, dict[str, Any]] = {}
        self.sessions: dict[UUID, dict[str, Any]] = {}
        self.turns: list[dict[str, Any]] = []
        self.profiles: dict[UUID, list[dict[str, Any]]] = {}
        self.session_summaries: list[dict[str, Any]] = []
        self.reference_chunks: list[dict[str, Any]] = []

    def upsert_student(
        self, student_id: UUID | None, external_ref: str | None
    ) -> dict[str, Any]:
        if student_id and student_id in self.students:
            return self.students[student_id]
        sid = student_id or uuid4()
        row: dict[str, Any] = {
            "id": str(sid),
            "external_ref": external_ref,
            "profile_summary": None,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        self.students[sid] = row
        return row

    def get_student(self, student_id: UUID) -> dict[str, Any]:
        if student_id not in self.students:
            raise NotFoundError(student_id)
        return self.students[student_id]

    def create_session(self, student_id: UUID, **kw: Any) -> dict[str, Any]:
        sid = uuid4()
        row: dict[str, Any] = {
            "id": str(sid),
            "student_id": str(student_id),
            "subject": kw["subject"],
            "problem_text": kw["problem_text"],
            "problem_image_url": kw.get("problem_image_url"),
            "concepts": kw["concepts"],
            "problem_type": kw.get("problem_type"),
            "ocr_raw": kw["ocr_raw"],
            "loop_count": 0,
            "status": "active",
            "resolved": False,
            "resolution_type": None,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        self.sessions[sid] = row
        return row

    def get_session(self, session_id: UUID) -> dict[str, Any]:
        if session_id not in self.sessions:
            raise NotFoundError(session_id)
        return self.sessions[session_id]

    def update_session(self, session_id: UUID, **kw: Any) -> dict[str, Any]:
        row = self.sessions[session_id]
        row.update(kw)
        return row

    def add_turn(self, session_id: UUID, **kw: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": str(uuid4()),
            "session_id": str(session_id),
            "role": kw["role"],
            "content": kw["content"],
            "loop_index": kw.get("loop_index", 0),
            "hint_level": kw.get("hint_level"),
            "classification": kw.get("classification"),
            "metadata": kw.get("metadata", {}),
            "created_at": "2026-01-01T00:00:00Z",
        }
        self.turns.append(row)
        return row

    def list_turns(self, session_id: UUID) -> list[dict[str, Any]]:
        return [t for t in self.turns if t["session_id"] == str(session_id)]

    def upsert_profile(self, student_id: UUID, **kw: Any) -> dict[str, Any]:
        existing = [
            p for p in self.profiles.get(student_id, [])
            if p["concept"] == kw["concept"]
        ]
        new_attempts = (existing[0]["attempts"] + 1) if existing else 1
        # Replace any existing entry for the same concept (upsert semantics).
        self.profiles[student_id] = [
            p for p in self.profiles.get(student_id, [])
            if p["concept"] != kw["concept"]
        ]
        row: dict[str, Any] = {
            "concept": kw["concept"],
            "mastery_score": kw["mastery_score"],
            "attempts": new_attempts,
            "last_session_id": kw.get("last_session_id"),
            "updated_at": "2026-01-01T00:00:00Z",
        }
        self.profiles.setdefault(student_id, []).append(row)
        return row

    def get_profiles(self, student_id: UUID) -> list[dict[str, Any]]:
        return self.profiles.get(student_id, [])

    def add_session_summary(self, student_id: UUID, session_id: UUID, **kw: Any) -> dict[str, Any]:
        # Upsert: remove existing summary for this session_id.
        self.session_summaries = [
            s for s in self.session_summaries if s["session_id"] != str(session_id)
        ]
        row: dict[str, Any] = {
            "id": str(uuid4()),
            "student_id": str(student_id),
            "session_id": str(session_id),
            "problem_text": kw["problem_text"],
            "concepts": kw["concepts"],
            "problem_type": kw.get("problem_type"),
            "outcome": kw["outcome"],
            "target_concept": kw.get("target_concept"),
            "summary": kw["summary"],
            "key_mistakes": kw.get("key_mistakes"),
            "mastery_after": kw.get("mastery_after"),
            "created_at": "2026-01-01T00:00:00Z",
        }
        self.session_summaries.append(row)
        return row

    def list_session_summaries(self, student_id: UUID, limit: int = 5) -> list[dict[str, Any]]:
        rows = [s for s in self.session_summaries if s["student_id"] == str(student_id)]
        return rows[:limit]

    def find_related_summaries(
        self,
        student_id: UUID,
        concepts: list[str],
        problem_type: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        all_summaries = self.list_session_summaries(student_id, limit=50)
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

    def add_reference_chunk(self, **kw: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": str(uuid4()),
            "source_id": kw["source_id"],
            "source_title": kw["source_title"],
            "source_url": kw["source_url"],
            "chapter": kw.get("chapter"),
            "heading": kw.get("heading"),
            "chunk_text": kw["chunk_text"],
            "concepts": kw["concepts"],
        }
        self.reference_chunks.append(row)
        return row

    def list_reference_chunks_by_concepts(
        self, concepts: list[str]
    ) -> list[dict[str, Any]]:
        if not concepts:
            return list(self.reference_chunks)
        return [
            r for r in self.reference_chunks
            if set(concepts) & set(r.get("concepts") or [])
        ]

    def reset_reference_chunks(self) -> None:
        self.reference_chunks = []

    def get_client(self) -> Any:
        class _Storage:
            def from_(self, bucket: str) -> _Storage:
                return self

            def upload(self, *a: Any, **k: Any) -> Any:
                raise RuntimeError("storage disabled in tests")

            def get_public_url(self, path: str) -> str:
                return f"http://fake/{path}"

        class _Client:
            storage = _Storage()

            def table(self, name: str) -> Any:
                raise RuntimeError("raw table access disabled in tests")

        return _Client()

    def reset_client(self) -> None:
        """No-op for the in-memory backend (no cached connection)."""
        return None
