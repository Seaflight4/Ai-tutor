"""Unit tests for prompt construction and small pure helpers."""

from __future__ import annotations

from app.models.schemas import Classification
from app.prompts import guided_discovery as p


def test_ocr_prompt_is_faithful_directive():
    assert "OCR" in p.OCR_PROMPT
    assert "physics" in p.OCR_PROMPT.lower()


def test_opening_user_includes_weak_concepts():
    msg = p.opening_user("block on ramp", ["energy"], ["energy"])
    assert "weak mastery" in msg
    assert "energy" in msg


def test_opening_user_without_weak_concepts():
    msg = p.opening_user("block on ramp", ["energy"], None)
    assert "weak mastery" not in msg


def test_diagnosis_user_includes_loop():
    msg = p.diagnosis_user("prob", ["c"], "TUTOR: hi\nSTUDENT: hi", "I'm stuck", 2)
    assert "Current hint loop: 2" in msg


def test_hint_user_level_appears():
    msg = p.hint_user(
        "prob", ["c"], "dialogue", Classification.misapplication, "energy", 3
    )
    assert "Requested hint level: 3" in msg
    assert "misapplication" in msg


def test_reveal_offer_is_choice():
    assert "(a)" in p.REVEAL_OFFER
    assert "(b)" in p.REVEAL_OFFER


def test_solution_user_includes_problem_and_dialogue():
    msg = p.solution_user("prob", ["c"], "dialogue")
    assert "Problem:" in msg
    assert "Dialogue so far" in msg
