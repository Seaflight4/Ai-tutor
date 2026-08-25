"""Tests for PR7 hardening: auth, validation, rate limiting, error shape."""

from __future__ import annotations

import io
from uuid import uuid4

import pytest
from PIL import Image

from app.api.middleware import _window
from app.core.config import get_settings


def _png_bytes(size: int = 32) -> bytes:
    img = Image.new("RGB", (size, size), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def test_auth_disabled_by_default(client):
    """When API_SECRET is empty, no key is required."""
    r = client.get("/api/sessions", params={"session_id": str(uuid4())})
    # 404 (not found) not 401 (unauthorized)
    assert r.status_code != 401


def test_auth_required_when_secret_set(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_secret", "s3cret", raising=False)
    r = client.post("/api/sessions", files={})
    assert r.status_code == 401
    assert r.json()["code"] == "unauthorized"
    assert r.headers["WWW-Authenticate"] == "ApiKey"


def test_auth_passes_with_correct_key(client, monkeypatch, fake_llm, fake_supabase):
    monkeypatch.setattr(get_settings(), "api_secret", "s3cret", raising=False)
    fake_llm.ocr_responses = ["problem text"]
    fake_llm.json_responses = [
        {
            "problem_text": "problem text", "formulas": [], "concepts": ["energy"],
            "topic": None, "diagram_description": None,
        }
    ]
    fake_llm.text_responses = ["Opening question"]

    r = client.post(
        "/api/sessions",
        files={"file": ("p.png", _png_bytes(), "image/png")},
        headers={"X-API-Key": "s3cret"},
    )
    assert r.status_code == 201


def test_auth_rejects_wrong_key(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_secret", "s3cret", raising=False)
    r = client.post("/api/sessions", files={}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_health_not_authenticated(client, monkeypatch):
    """Health endpoint is exempt from auth even when secret is set."""
    monkeypatch.setattr(get_settings(), "api_secret", "s3cret", raising=False)
    r = client.get("/health")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Error-response shape
# ---------------------------------------------------------------------------
def test_error_shape_has_code_field(client):
    """All error responses carry a `code` field."""
    r = client.get(f"/api/sessions/{uuid4()}")
    assert r.status_code == 404
    body = r.json()
    assert "detail" in body
    assert body["code"] == "not_found"


def test_validation_error_shape(client):
    """422 from pydantic validation returns the standard shape."""
    # ReplyIn requires non-empty message; empty body triggers 422.
    r = client.post(f"/api/sessions/{uuid4()}/reply", json={"message": ""})
    assert r.status_code == 422


def test_unsupported_image_type_error_shape(client):
    r = client.post(
        "/api/sessions",
        files={"file": ("p.gif", b"GIF89a", "image/gif")},
    )
    assert r.status_code == 415
    assert r.json()["code"] == "unsupported_media_type"


# ---------------------------------------------------------------------------
# Input validation / limits
# ---------------------------------------------------------------------------
def test_reply_message_too_long_rejected(client, monkeypatch, fake_supabase):
    monkeypatch.setattr(get_settings(), "max_reply_chars", 10, raising=False)
    r = client.post(
        f"/api/sessions/{uuid4()}/reply",
        json={"message": "x" * 11},
    )
    # The schema max_length=4000 won't catch a 11-char message, but the
    # runtime check (max_reply_chars=10) will, returning 413.
    assert r.status_code == 413


def test_image_too_large_rejected(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_image_bytes", 100, raising=False)
    # A valid PNG but larger than 100 bytes.
    big_png = _png_bytes(size=200)
    r = client.post(
        "/api/sessions",
        files={"file": ("p.png", big_png, "image/png")},
    )
    assert r.status_code == 413
    assert r.json()["code"] == "payload_too_large"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def test_rate_limit_blocks_after_threshold(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 3, raising=False)
    _window._hits.clear()  # reset the global limiter state

    # First 3 requests to /api/* succeed (404, not 429).
    for _ in range(3):
        r = client.get(f"/api/sessions/{uuid4()}")
        assert r.status_code != 429

    # 4th request is rate-limited.
    r = client.get(f"/api/sessions/{uuid4()}")
    assert r.status_code == 429
    assert r.json()["code"] == "rate_limited"
    assert "Retry-After" in r.headers


def test_rate_limit_not_applied_to_health(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 1, raising=False)
    _window._hits.clear()

    # Many health requests should all pass — not rate-limited.
    for _ in range(5):
        r = client.get("/health")
        assert r.status_code == 200


def test_rate_limit_disabled_when_zero(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 0, raising=False)
    _window._hits.clear()

    for _ in range(10):
        r = client.get(f"/api/sessions/{uuid4()}")
        assert r.status_code != 429


# ---------------------------------------------------------------------------
# Cleanup fixture: reset limiter state between tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    _window._hits.clear()
    yield
    _window._hits.clear()
