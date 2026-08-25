"""Shared test fixtures and mocks.

Tests never hit the real skainet gateway or Supabase. We monkeypatch the LLM
and supabase modules with lightweight in-memory fakes.

The fake implementations live in `app/adapters/in_memory.py` (so they can be
reused by contract tests and live next to the other adapters). This file
just wires them into the legacy module-level functions that the services
still call.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.adapters.in_memory import InMemoryBackend, InMemoryLLM
from app.main import app

# Back-compat aliases for tests that import the old names from conftest.
FakeLLM = InMemoryLLM
FakeSupabase = InMemoryBackend


@pytest.fixture
def client(fake_llm, fake_supabase):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# LLM fake
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_llm(monkeypatch):
    fake = InMemoryLLM()
    import app.core.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_get_client", lambda: fake)
    monkeypatch.setattr(llm_mod, "ocr_image", fake.ocr_image)
    monkeypatch.setattr(llm_mod, "chat_json", fake.chat_json)
    monkeypatch.setattr(llm_mod, "chat_text", fake.chat_text)
    return fake


# ---------------------------------------------------------------------------
# In-memory supabase fake
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_supabase(monkeypatch):
    fake = InMemoryBackend()
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
        "add_session_summary",
        "list_session_summaries",
        "find_related_summaries",
        "get_client",
        "add_reference_chunk",
        "list_reference_chunks_by_concepts",
        "reset_reference_chunks",
    ]:
        monkeypatch.setattr(sb_mod, name, getattr(fake, name))
    return fake
