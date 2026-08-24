"""Live end-to-end test: full HTTP flow against the real skainet gateway.

Boots a real uvicorn server on an ephemeral port, generates a sample physics
problem image, and walks the entire guided-discovery loop:

    POST /api/sessions            (image upload -> OCR -> parse -> opening)
    GET  /api/sessions/{id}       (transcript check)
    POST /api/sessions/{id}/reply (loop 1: hint)
    POST /api/sessions/{id}/reply (loop 2: hint)
    POST /api/sessions/{id}/reply (loop 3: reveal offer)
    POST /api/sessions/{id}/reply (choose 'b' -> reveal solution)
    GET  /api/sessions/{id}       (resolved + full transcript)
    GET  /api/students/{id}/profile (knowledge profile updated)

The test makes REAL, billed LLM calls (olmocr-7B + GLM-5.2) and takes ~1-2 min.
It is skipped unless RUN_LIVE_E2E=1 is set, so the normal `pytest` run stays
fast/offline and CI-safe.
"""

from __future__ import annotations

import os
import socket
import threading
from contextlib import closing, suppress
from typing import Any

import httpx
import pytest

# ---------------------------------------------------------------------------
# Gating: only run when explicitly requested AND a real key is configured.
# ---------------------------------------------------------------------------
_LIVE = os.getenv("RUN_LIVE_E2E") == "1"

# Pull the key from the app's own settings (which loads .env via pydantic-settings)
# so the gating works whenever the app itself can run, without requiring the key
# to be exported into the shell environment.
try:
    from app.core.config import get_settings as _get_settings

    _settings = _get_settings()
    _KEY = _settings.skainet_api_key
except Exception:  # pragma: no cover
    _KEY = ""
_KEY_PRESENT = bool(_KEY) and not _KEY.startswith("tngai_replace_me")

pytestmark = pytest.mark.skipif(
    not (_LIVE and _KEY_PRESENT),
    reason="set RUN_LIVE_E2E=1 with a real SKAINET_API_KEY to run live E2E tests",
)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def live_base_url() -> str:
    """Start a real uvicorn server on an ephemeral port and return its base URL.

    Module-scoped so the server is shared across tests (one server, one session
    flow). The DB file is deleted before/after to start clean.
    """
    from uvicorn import Config, Server

    # Clean any stale local DB so the session starts fresh.
    _db_path = "ai_tutor_local.db"
    if os.path.exists(_db_path):
        os.remove(_db_path)

    port = _free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    config = Config(
        "app.main:app",
        host=host,
        port=port,
        log_level="warning",
        reload=False,
        access_log=False,
    )
    server = Server(config)

    # Run uvicorn in a daemon thread. Server.run() manages its own event loop.
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until the server is accepting connections.
    with httpx.Client(timeout=30.0) as probe:
        for _ in range(100):
            try:
                if probe.get(f"{base_url}/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            import time

            time.sleep(0.1)
        else:  # pragma: no cover
            server.should_exit = True
            raise RuntimeError("uvicorn did not start within 10s")

    yield base_url

    # Shutdown: signal the server to exit and join the thread.
    server.should_exit = True
    thread.join(timeout=10.0)
    with suppress(OSError):
        os.remove(_db_path)


@pytest.fixture
def sample_image_png() -> bytes:
    """Reuse the project's sample-problem generator."""
    from scripts.make_sample import make_sample_problem

    return make_sample_problem()


def test_full_session_reveal(live_base_url: str, sample_image_png: bytes) -> None:
    """Drive the complete happy-path: upload -> 3 hints -> reveal -> verify."""
    with httpx.Client(base_url=live_base_url, timeout=120.0) as client:
        # 1. Create session via image upload.
        r = client.post(
            "/api/sessions",
            files={"file": ("problem.png", sample_image_png, "image/png")},
            data={"external_ref": "e2e-1"},
        )
        assert r.status_code == 201, r.text
        session = r.json()
        assert session["subject"] == "physics"
        assert session["problem_text"], "OCR must return non-empty problem text"
        # `concepts` is best-effort: the OCR-parse LLM may occasionally return
        # an empty list, so we don't hard-assert it. The flow still works.
        session_concepts = set(session["concepts"])
        assert session["loop_count"] == 0
        assert session["resolved"] is False

        session_id = session["id"]
        student_id = session["student_id"]

        # 2. Verify the opening turn was persisted.
        r = client.get(f"/api/sessions/{session_id}")
        assert r.status_code == 200
        body: dict[str, Any] = r.json()
        turns = body["turns"]
        assert any(t["role"] == "tutor" for t in turns), "opening tutor turn should exist"

        # 3-5. Drive three replies through the hint loop. The loop now runs
        # continuously — no forced reveal offer at loop 3.
        reply_msgs = [
            "I don't know how to start this problem.",
            "I tried using F=ma but I'm confused about the ramp.",
            "I still don't see how to find the speed.",
        ]
        for i, msg in enumerate(reply_msgs, start=1):
            r = client.post(
                f"/api/sessions/{session_id}/reply",
                json={"message": msg},
            )
            assert r.status_code == 200, r.text
            reply = r.json()
            assert reply["loop_index"] == i, f"loop_index should be {i}"
            assert reply["content"], "tutor content must be non-empty"
            assert reply["offer_reveal"] is False, (
                f"continuous loop must not force a reveal offer (loop {i})"
            )
            assert reply["resolved"] is False

        # 6. Student explicitly requests the solution -> reveal.
        r = client.post(
            f"/api/sessions/{session_id}/reply",
            json={"message": "show me the full solution please"},
        )
        assert r.status_code == 200, r.text
        revealed = r.json()
        assert revealed["resolved"] is True
        assert revealed["resolution_type"] == "revealed"
        assert revealed["solution"], "solution text must be non-empty"

        # 7. Verify resolved state + full transcript.
        r = client.get(f"/api/sessions/{session_id}")
        assert r.status_code == 200
        body = r.json()
        session_final = body["session"]
        assert session_final["resolved"] is True
        assert session_final["resolution_type"] == "revealed"

        final_turns = body["turns"]
        # Opening + 3 tutor hints + 3 student replies + 1 student reveal-request
        # + 1 reveal = >= 9 turns.
        assert len(final_turns) >= 9, (
            f"expected >= 9 turns in transcript, got {len(final_turns)}"
        )
        # A reveal turn uses the sentinel loop_index 999.
        assert any(t["loop_index"] == 999 for t in final_turns), (
            "reveal turn (loop_index=999) must be present"
        )

        # 8. Knowledge profile. When the session had concepts, the profile
        # should have at least one matching entry. When the OCR-parse returned
        # no concepts, `profile.update_profiles` short-circuits (no concept to
        # upsert), so the profile legitimately stays empty — we only assert
        # the endpoint responds 200 in that case.
        r = client.get(f"/api/students/{student_id}/profile")
        assert r.status_code == 200
        profile = r.json()
        if session_concepts:
            assert profile["entries"], (
                "knowledge profile should have at least one entry after reveal"
            )
            profile_concepts = {e["concept"] for e in profile["entries"]}
            assert profile_concepts & session_concepts, (
                f"profile concepts {profile_concepts} should overlap "
                f"with session concepts {session_concepts}"
            )
