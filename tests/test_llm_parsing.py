"""Tests for the _strip_think_only / classification-default fixes.

Covers two bugs found in session history where the tutor replied
"You're on the right track!" to wrong scenarios:

1. `chat_json` called `_strip_cot`, which discards fenced JSON (the fence
   stripping assumes the answer follows the closing fence, but for JSON
   the payload IS the fenced content). Fix: `chat_json` now uses
   `_strip_think_only` (think-block removal only), leaving `_parse_json`
   to handle fences.

2. `tutor.assess_and_respond` defaulted to `Classification.on_track` when
   the LLM returned no classification, causing a false-positive
   affirmation. Fix: default to `Classification.meta` (neutral).
"""

from __future__ import annotations

import json

from app.core.llm import _parse_json, _strip_cot, _strip_think_only
from app.models.schemas import Classification
from app.services import hints, tutor


# ---------------------------------------------------------------------------
# Fix 1: _strip_think_only preserves fenced JSON
# ---------------------------------------------------------------------------
def test_strip_think_only_preserves_fenced_json():
    """Fenced JSON survives _strip_think_only so _parse_json can extract it."""
    text = "ildaanalysis here</think>\n```json\n{\"classification\": \"answer_check\"}\n```"
    result = _strip_think_only(text)
    assert "```" in result, "fences must survive _strip_think_only"
    assert "answer_check" in result


def test_strip_cot_destroys_fenced_json_regression_guard():
    """Documents the original bug: _strip_cot eats fenced JSON.

    This test asserts the OLD (buggy) behavior of _strip_cot so the contrast
    with _strip_think_only is explicit and the regression is understood.
    """
    text = "ildareasoning here\n```json\n{\"classification\": \"answer_check\"}\n```"
    stripped = _strip_cot(text)
    # _strip_cot takes text after the last fence -> empty (the bug).
    assert stripped == ""


def test_parse_json_handles_fenced_json_after_strip_think_only():
    """End-to-end: think block + fenced JSON -> correct parse."""
    text = "ildareasoning here\n```json\n{\"classification\": \"answer_check\"}\n```"
    parsed = _parse_json(_strip_think_only(text))
    assert parsed.get("classification") == "answer_check"


def test_parse_json_handles_raw_json_after_strip_think_only():
    """Think block + raw JSON (no fences) still works."""
    text = "ildareasoning here\n{\"classification\": \"meta\"}"
    parsed = _parse_json(_strip_think_only(text))
    assert parsed.get("classification") == "meta"


def test_strip_think_only_handles_no_think_block():
    """Plain fenced JSON without a think block is passed through."""
    text = "```json\n{\"x\": 1}\n```"
    result = _strip_think_only(text)
    assert "```" in result
    assert json.loads(_parse_json(result).get("_raw", "{}") or "{}") or _parse_json(result)


def test_strip_think_only_empty():
    assert _strip_think_only("") == ""


# ---------------------------------------------------------------------------
# Fix 2: default classification is meta, not on_track
# ---------------------------------------------------------------------------
async def test_tutor_defaults_to_meta_on_missing_classification(monkeypatch):
    """When the LLM returns no classification, we default to meta (neutral),
    not on_track (false-positive affirmation)."""
    async def fake_chat_json(*args, **kwargs):
        return {"_raw": ""}  # parse failure — no classification

    monkeypatch.setattr(tutor.llm, "chat_json", fake_chat_json)

    diag, hint = await tutor.assess_and_respond(
        problem_text="A ball bounces off a wall.",
        concepts=["elastic collision"],
        dialogue="student: is the force 14N?",
        student_reply="is the force 14N?",
        current_loop=4,
    )
    assert diag.classification is Classification.meta


async def test_tutor_defaults_to_meta_on_unknown_classification(monkeypatch):
    """An unrecognized classification string also defaults to meta."""
    async def fake_chat_json(*args, **kwargs):
        return {"classification": "not_a_real_label"}

    monkeypatch.setattr(tutor.llm, "chat_json", fake_chat_json)

    diag, hint = await tutor.assess_and_respond(
        problem_text="A ball bounces off a wall.",
        concepts=["elastic collision"],
        dialogue="student: hi",
        student_reply="hi",
        current_loop=0,
    )
    assert diag.classification is Classification.meta


async def test_tutor_preserves_valid_classification(monkeypatch):
    """A valid classification is passed through unchanged."""
    async def fake_chat_json(*args, **kwargs):
        return {"classification": "answer_check", "answer_status": "correct"}

    monkeypatch.setattr(tutor.llm, "chat_json", fake_chat_json)

    diag, hint = await tutor.assess_and_respond(
        problem_text="A ball bounces off a wall.",
        concepts=["elastic collision"],
        dialogue="student: is it 14N?",
        student_reply="is it 14N?",
        current_loop=3,
    )
    assert diag.classification is Classification.answer_check


def test_meta_default_does_not_emit_false_positive():
    """The meta fallback in hints.py returns a neutral clarification prompt,
    not an empty string and not an affirmation."""
    from app.models.schemas import HintOutput

    result = hints.summarize_hint(HintOutput(), Classification.meta)
    assert result != ""
    assert "right track" not in result.lower()


def test_on_track_with_null_fields_still_falls_back():
    """The on_track fallback still exists but is a neutral nudge, not a
    false-positive affirmation of progress the student didn't make."""
    from app.models.schemas import HintOutput

    result = hints.summarize_hint(HintOutput(), Classification.on_track)
    assert "right track" not in result.lower()
