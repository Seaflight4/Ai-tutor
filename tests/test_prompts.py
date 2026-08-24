"""Unit tests for prompt construction and small pure helpers."""

from __future__ import annotations

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


def test_tutor_user_includes_loop_and_reply():
    msg = p.tutor_user("prob", ["c"], "TUTOR: hi\nSTUDENT: hi", "I'm stuck", 2)
    assert "Current hint loop: 2" in msg
    assert "I'm stuck" in msg


def test_tutor_system_mentions_all_diagnosis_fields():
    for field in ("classification", "reasoning", "target_concept", "wants_solution"):
        assert field in p.TUTOR_SYSTEM, f"TUTOR_SYSTEM must mention {field}"


def test_tutor_system_mentions_all_hint_fields():
    for field in (
        "formula",
        "explanation",
        "example",
        "mistake",
        "reason",
        "application_hint",
        "confirmation",
        "next_step_hint",
    ):
        assert field in p.TUTOR_SYSTEM, f"TUTOR_SYSTEM must mention {field}"


def test_tutor_system_branches_on_diagnosis():
    assert "knowledge_gap" in p.TUTOR_SYSTEM
    assert "misapplication" in p.TUTOR_SYSTEM
    assert "on_track" in p.TUTOR_SYSTEM


def test_tutor_system_forbids_solving():
    assert "NEVER give the final answer" in p.TUTOR_SYSTEM
    assert "ONLY what the student asked" in p.TUTOR_SYSTEM


def test_tutor_system_asks_for_latex():
    assert "LaTeX" in p.TUTOR_SYSTEM
    assert "$" in p.TUTOR_SYSTEM


def test_opening_system_three_parts():
    # Opening must mandate greeting + problem-type summary + "where are you stuck".
    assert "greeting" in p.OPENING_SYSTEM.lower()
    assert "problem type" in p.OPENING_SYSTEM.lower()
    assert "where" in p.OPENING_SYSTEM.lower() and "stuck" in p.OPENING_SYSTEM.lower()


def test_solution_system_asks_for_latex():
    assert "LaTeX" in p.SOLUTION_SYSTEM


def test_solution_user_includes_problem_and_dialogue():
    msg = p.solution_user("prob", ["c"], "dialogue")
    assert "Problem:" in msg
    assert "Dialogue so far" in msg
