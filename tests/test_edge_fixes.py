"""Regression tests for diagnosed edge-case fixes."""

from __future__ import annotations

from app.models.schemas import Classification, HintOutput
from app.prompts import guided_discovery as p
from app.services.hints import summarize_hint


def test_summarize_hint_meta_with_null_meta_response_is_nonempty():
    h = HintOutput(meta_response=None)
    out = summarize_hint(h, Classification.meta)
    assert out
    assert out.strip()
    assert out != ""


def test_summarize_hint_on_track_null_fields_is_not_false_praise():
    h = HintOutput(confirmation=None, next_step_hint=None)
    out = summarize_hint(h, Classification.on_track)
    assert out != "You're on the right track!"
    assert out.strip()


def test_tutor_system_requires_work_before_confirming_answer():
    assert "verify the student has shown their reasoning" in p.TUTOR_SYSTEM
    assert "based on the number alone" in p.TUTOR_SYSTEM


def test_tutor_system_classifies_non_substantive_as_meta():
    assert "non-substantive messages" in p.TUTOR_SYSTEM
    assert "not demonstrated any progress" in p.TUTOR_SYSTEM.lower()
