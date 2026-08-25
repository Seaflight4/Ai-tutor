"""Clear sessions and turns from the local SQLite DB for test isolation.

Keeps reference_chunks and knowledge_profiles (they survive across iterations).
Run before each student-agent session batch so the reviewer only sees current data.

    python -m scripts.clear_sessions
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("ai_tutor_local.db")

TABLES_TO_CLEAR = ["turns", "sessions", "students"]


def main() -> None:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return
    conn = sqlite3.connect(DB_PATH)
    for table in TABLES_TO_CLEAR:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    print("cleared turns, sessions, students", file=sys.stderr)


if __name__ == "__main__":
    main()
