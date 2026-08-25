"""Adapters — implementations of the `app.ports` Protocols.

- `llm_skainet`: real OpenAI-compatible client (wraps `app/core/llm.py`).
- `supabase_repo`: Supabase persistence (wraps `app/core/supabase.py`).
- `sqlite_repo`: local SQLite fallback (wraps `app/core/local_store.py`).
- `in_memory`: test doubles (`InMemoryLLM`, `InMemoryBackend`).
"""

from __future__ import annotations

from app.adapters.in_memory import InMemoryBackend, InMemoryLLM

__all__ = ["InMemoryBackend", "InMemoryLLM"]
