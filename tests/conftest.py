"""Shared test fixtures and mocks.

Tests never hit the real skainet gateway or Supabase. We monkeypatch the LLM
and supabase modules with lightweight in-memory fakes.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(fake_llm, fake_supabase):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# LLM fake
# ---------------------------------------------------------------------------
class FakeLLM:
    def __init__(self) -> None:
        self.ocr_responses: list[str] = ["A block slides down a frictionless ramp..."]
        self.json_responses: list[dict[str, Any]] = []
        self.text_responses: list[str] = []
        self.calls: list[dict[str, Any]] = []

    async def ocr_image(self, image_bytes, *, prompt, mime="image/png", max_tokens=2000):
        self.calls.append({"kind": "ocr", "prompt": prompt})
        return self.ocr_responses.pop(0) if self.ocr_responses else "OCR"

    async def chat_json(self, system, user, *, model=None, temperature=0.4, max_tokens=1200):
        self.calls.append({"kind": "json", "system": system, "user": user})
        return self.json_responses.pop(0) if self.json_responses else {}

    async def chat_text(self, system, user, *, model=None, temperature=0.5, max_tokens=800):
        self.calls.append({"kind": "text", "system": system, "user": user})
        return self.text_responses.pop(0) if self.text_responses else "OK"

    async def embed(self, text):
        return [0.0] * 1536


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    import app.core.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_get_client", lambda: fake)
    monkeypatch.setattr(llm_mod, "ocr_image", fake.ocr_image)
    monkeypatch.setattr(llm_mod, "chat_json", fake.chat_json)
    monkeypatch.setattr(llm_mod, "chat_text", fake.chat_text)
    monkeypatch.setattr(llm_mod, "embed", fake.embed)
    return fake


# ---------------------------------------------------------------------------
# In-memory supabase fake
# ---------------------------------------------------------------------------
class FakeSupabase:
    def __init__(self) -> None:
        self.students: dict[UUID, dict] = {}
        self.sessions: dict[UUID, dict] = {}
        self.turns: list[dict] = []
        self.profiles: dict[UUID, list[dict]] = {}

    def upsert_student(self, student_id, external_ref):
        if student_id and student_id in self.students:
            return self.students[student_id]
        sid = student_id or uuid4()
        row = {
            "id": str(sid),
            "external_ref": external_ref,
            "profile_summary": None,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        self.students[sid] = row
        return row

    def get_student(self, student_id):
        if student_id not in self.students:
            raise KeyError(student_id)
        return self.students[student_id]

    def create_session(self, student_id, **kw):
        sid = uuid4()
        row = {
            "id": str(sid),
            "student_id": str(student_id),
            "subject": kw["subject"],
            "problem_text": kw["problem_text"],
            "problem_image_url": kw.get("problem_image_url"),
            "concepts": kw["concepts"],
            "ocr_raw": kw["ocr_raw"],
            "loop_count": 0,
            "resolved": False,
            "resolution_type": None,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        self.sessions[sid] = row
        return row

    def get_session(self, session_id):
        if session_id not in self.sessions:
            raise KeyError(session_id)
        return self.sessions[session_id]

    def update_session(self, session_id, **kw):
        row = self.sessions[session_id]
        row.update(kw)
        return row

    def add_turn(self, session_id, **kw):
        row = {
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

    def list_turns(self, session_id):
        return [t for t in self.turns if t["session_id"] == str(session_id)]

    def upsert_profile(self, student_id, **kw):
        self.profiles.setdefault(student_id, []).append(
            {
                "concept": kw["concept"],
                "mastery_score": kw["mastery_score"],
                "attempts": 1,
                "last_session_id": kw.get("last_session_id"),
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
        return self.profiles[student_id][-1]

    def get_profiles(self, student_id):
        return self.profiles.get(student_id, [])

    def get_client(self):
        class _Storage:
            def from_(self, bucket):
                return self

            def upload(self, *a, **k):
                raise RuntimeError("storage disabled in tests")

            def get_public_url(self, path):
                return f"http://fake/{path}"

        class _Client:
            storage = _Storage()

            def table(self, name):
                raise RuntimeError("raw table access disabled in tests")

        return _Client()


@pytest.fixture
def fake_supabase(monkeypatch):
    fake = FakeSupabase()
    import app.core.supabase as sb_mod

    for name in [
        "upsert_student",
        "get_student",
        "create_session",
        "get_session",
        "update_session",
        "add_turn",
        "list_turns",
        "upsert_profile",
        "get_profiles",
        "get_client",
    ]:
        monkeypatch.setattr(sb_mod, name, getattr(fake, name))
    return fake
