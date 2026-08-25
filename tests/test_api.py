"""API route tests using FastAPI TestClient (LLM + supabase mocked)."""

from __future__ import annotations

import io
from uuid import uuid4

from PIL import Image


def _png_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_serves_index_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "AI Tutor" in r.text


def test_static_app_js_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers.get("content-type", "")
    assert "startSession" in r.text


def test_static_styles_css_served(client):
    r = client.get("/static/styles.css")
    assert r.status_code == 200
    assert "text/css" in r.headers.get("content-type", "")


def test_create_session_requires_image(client):
    r = client.post("/api/sessions", files={})
    assert r.status_code == 422


def test_create_session_success(client, fake_llm, fake_supabase):
    fake_llm.ocr_responses = ["problem text"]
    fake_llm.json_responses = [
        {
            "problem_text": "problem text",
            "formulas": [],
            "concepts": ["energy"],
            "topic": None,
            "diagram_description": None,
        }
    ]
    fake_llm.text_responses = ["Opening question"]

    r = client.post(
        "/api/sessions",
        files={"file": ("p.png", _png_bytes(), "image/png")},
        data={"external_ref": "stu-1"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["subject"] == "physics"
    assert body["concepts"] == ["energy"]
    assert body["loop_count"] == 0
    assert body["resolved"] is False


def test_reply_404_for_unknown_session(client):
    r = client.post(f"/api/sessions/{uuid4()}/reply", json={"message": "hi"})
    assert r.status_code == 404


def test_reveal_404_for_unknown_session(client):
    r = client.post(f"/api/sessions/{uuid4()}/reveal")
    assert r.status_code == 404


def test_get_session_404_for_unknown(client):
    r = client.get(f"/api/sessions/{uuid4()}")
    assert r.status_code == 404


def test_get_profile_404_for_unknown_student(client, fake_supabase):
    r = client.get(f"/api/students/{uuid4()}/profile")
    assert r.status_code == 404


def test_reply_409_for_resolved_session(client, fake_llm, fake_supabase):
    """A reply to an already-resolved session returns 409, not 500."""
    from app.services import session as session_service

    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        {
            "problem_text": "A block slides down a frictionless ramp. Find v.",
            "formulas": [], "concepts": ["energy conservation"],
            "topic": None, "diagram_description": None,
        },
        {"concept": "energy conservation", "mastery_score": 0.2},
    ]
    fake_llm.text_responses = [
        "Opening: where are you stuck?",
        "Full solution: $mgh = \\tfrac12 mv^2$.",
    ]

    import asyncio

    session = asyncio.run(
        session_service.start_session(_png_bytes(), "image/png", None, "stu-409")
    )
    sid = session["id"]
    asyncio.run(session_service.reveal_solution(__import__("uuid").UUID(sid)))

    r = client.post(f"/api/sessions/{sid}/reply", json={"message": "more help"})
    assert r.status_code == 409
