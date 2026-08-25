"""Fixtures for the contract test suite.

Three backends implement the same `Backend` Protocol and must behave
identically:

1. `in_memory` — `InMemoryBackend` (pure Python; the test double).
2. `sqlite` — `SQLiteBackend` pointed at a fresh temp DB (exercises
   `app/core/local_store.py` end-to-end, including the JSON round-trips and
   the `_now()` timestamp that PR2 fixed).
3. `supabase_mock` — `SupabaseBackend` backed by a fluent fake of the
   `supabase-py` client. This exercises `app/core/supabase.py`'s branching
   logic (`_first`, `_maybe_single_data`, the attempts-increment read-then-
   write, `NotFoundError` mapping) without a network.

Each fixture yields an object implementing the `Backend` Protocol surface
(the free-function-style methods: `upsert_student`, `get_student`, ...).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.adapters.in_memory import InMemoryBackend
from app.adapters.sqlite_repo import SQLiteBackend
from app.adapters.supabase_repo import SupabaseBackend
from app.core import local_store
from app.core import supabase as sb_mod


# ---------------------------------------------------------------------------
# 1. In-memory backend
# ---------------------------------------------------------------------------
@pytest.fixture
def in_memory() -> InMemoryBackend:
    return InMemoryBackend()


# ---------------------------------------------------------------------------
# 2. SQLite backend (fresh temp DB per test)
# ---------------------------------------------------------------------------
@pytest.fixture
def sqlite(monkeypatch, tmp_path) -> SQLiteBackend:
    # Point local_store at a fresh temp DB so tests are isolated.
    db_path = tmp_path / "contract.db"
    monkeypatch.setattr(local_store, "_DB_PATH", str(db_path))
    monkeypatch.setattr(local_store, "_conn", None)
    return SQLiteBackend()


# ---------------------------------------------------------------------------
# 3. Supabase-backed, with a fluent fake client
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, data: Any) -> None:
        self.data = data


class _FakeQuery:
    """Records the chain (.select/.insert/.update/.upsert/.eq/.or_/.order/
    .maybe_single/.delete/.neq) and resolves on `.execute()` against the
    fake store. DB column defaults are applied on insert so inserted rows
    match the shape a real Postgres backend would return."""

    # Column defaults applied to inserted rows (mirrors db/schema.sql).
    _SESSION_DEFAULTS: dict[str, Any] = {
        "loop_count": 0,
        "status": "active",
        "resolved": False,
        "resolution_type": None,
    }
    _PROFILE_DEFAULTS: dict[str, Any] = {"attempts": 0}

    def __init__(self, store: _FakeSupabaseStore, table: str) -> None:
        self._store = store
        self._table = table
        self._op: str | None = None
        self._payload: dict[str, Any] | None = None
        self._filters: list[tuple[str, str]] = []
        self._neq_filters: list[tuple[str, str]] = []
        self._or_filter: str | None = None
        self._order_col: str | None = None
        self._maybe_single = False
        self._select_cols: str = "*"
        self._on_conflict: str = ""

    def select(self, cols: str = "*") -> _FakeQuery:
        self._op = "select"
        self._select_cols = cols
        return self

    def insert(self, payload: dict[str, Any]) -> _FakeQuery:
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload: dict[str, Any]) -> _FakeQuery:
        self._op = "update"
        self._payload = payload
        return self

    def upsert(self, payload: dict[str, Any], on_conflict: str = "") -> _FakeQuery:
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def delete(self) -> _FakeQuery:
        self._op = "delete"
        return self

    def eq(self, col: str, val: str) -> _FakeQuery:
        self._filters.append((col, val))
        return self

    def neq(self, col: str, val: str) -> _FakeQuery:
        self._neq_filters.append((col, val))
        return self

    def or_(self, expr: str) -> _FakeQuery:
        self._or_filter = expr
        return self

    def order(self, col: str) -> _FakeQuery:
        self._order_col = col
        return self

    def maybe_single(self) -> _FakeQuery:
        self._maybe_single = True
        return self

    def _apply_filters(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for col, val in self._filters:
            rows = [r for r in rows if str(r.get(col)) == str(val)]
        for col, val in self._neq_filters:
            rows = [r for r in rows if str(r.get(col)) != str(val)]
        if self._or_filter:
            # PostgREST: "concepts.cs.{tag},concepts.cs.{tag}" — a row matches
            # if its `concepts` array contains any of the tags.
            tags: list[str] = []
            for part in self._or_filter.split(","):
                if ".cs.{" in part and part.endswith("}"):
                    tags.append(part[part.index("{") + 1 : -1])
            if tags:
                rows = [r for r in rows if set(tags) & set(r.get("concepts") or [])]
        return rows

    def execute(self) -> _FakeResponse:
        rows = self._store.tables.get(self._table, [])
        rows = self._apply_filters(list(rows))

        if self._op == "select":
            if self._maybe_single:
                return _FakeResponse(rows[0] if rows else None)
            if self._order_col == "created_at":
                rows = sorted(rows, key=lambda r: r.get("created_at") or "")
            return _FakeResponse(list(rows))

        if self._op == "insert":
            row = {**self._payload}
            # Apply DB column defaults so the returned row matches a real
            # Postgres backend (which fills NOT NULL DEFAULT columns).
            if self._table == "sessions":
                for k, v in self._SESSION_DEFAULTS.items():
                    row.setdefault(k, v)
            elif self._table == "knowledge_profiles":
                for k, v in self._PROFILE_DEFAULTS.items():
                    row.setdefault(k, v)
            row.setdefault("id", str(uuid4()))
            self._store.tables.setdefault(self._table, []).append(row)
            return _FakeResponse([row])

        if self._op == "update":
            updated: list[dict[str, Any]] = []
            for r in self._store.tables.get(self._table, []):
                if all(str(r.get(c)) == str(v) for c, v in self._filters):
                    r.update(self._payload or {})
                    updated.append(r)
            return _FakeResponse(updated)

        if self._op == "upsert":
            conflict_cols = self._on_conflict.split(",") if self._on_conflict else []
            existing = None
            if conflict_cols:
                for r in self._store.tables.get(self._table, []):
                    if all(
                        str(r.get(c)) == str(self._payload.get(c))
                        for c in conflict_cols
                    ):
                        existing = r
                        break
            if existing is not None:
                existing.update(self._payload or {})
                return _FakeResponse([existing])
            row = {**self._payload}
            if self._table == "knowledge_profiles":
                for k, v in self._PROFILE_DEFAULTS.items():
                    row.setdefault(k, v)
            row.setdefault("id", str(uuid4()))
            self._store.tables.setdefault(self._table, []).append(row)
            return _FakeResponse([row])

        if self._op == "delete":
            table_rows = self._store.tables.get(self._table, [])
            keep = [r for r in table_rows if not all(
                str(r.get(c)) == str(v) for c, v in self._filters
            ) and all(str(r.get(c)) != str(v) for c, v in self._neq_filters)]
            self._store.tables[self._table] = keep
            return _FakeResponse([])

        return _FakeResponse(None)


class _FakeStorage:
    def from_(self, bucket: str) -> _FakeStorage:
        return self

    def upload(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("storage disabled in contract tests")

    def get_public_url(self, path: str) -> str:
        return f"http://fake/{path}"


class _FakeClient:
    def __init__(self) -> None:
        self.storage = _FakeStorage()

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self._store, name)

    @property
    def _store(self) -> _FakeSupabaseStore:
        return self._store_ref  # type: ignore[attr-defined]

    def _set_store(self, store: _FakeSupabaseStore) -> None:
        self._store_ref = store  # type: ignore[attr-defined]


class _FakeSupabaseStore:
    """Holds the in-memory tables shared by one `_FakeClient` instance."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.client = _FakeClient()
        self.client._set_store(self)


@pytest.fixture
def supabase_mock(monkeypatch) -> SupabaseBackend:
    """A `SupabaseBackend` wired to a fluent fake client.

    Forces `_is_local()` to return False so `app.core.supabase` exercises its
    Supabase (not local_store) branch, and swaps `get_client()` for the fake.
    """
    store = _FakeSupabaseStore()
    monkeypatch.setattr(sb_mod, "_is_local", lambda: False)
    monkeypatch.setattr(sb_mod, "get_client", lambda: store.client)
    return SupabaseBackend()
