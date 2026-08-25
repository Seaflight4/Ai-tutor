"""Tests for the 5 new review checks in scripts/review_sessions.py.

These tests exercise the check functions directly (not via the CLI) so
they run fast and don't need a real DB.
"""

from __future__ import annotations

import json

from scripts.review_sessions import (
    _answer_leak_check,
    _classification_missing_check,
    _context_not_injected_check,
    _empty_response_check,
    _false_positive_affirmation_check,
    _hallucinated_reference_check,
    _json_parse_failure_check,
    _missed_connection_check,
    _problem_type_missing_check,
    _summary_quality_check,
)


def _turn(role="tutor", content="", classification=None, metadata=None,
          loop_index=1, turn_id="turn-001"):
    return {
        "id": turn_id,
        "role": role,
        "content": content,
        "classification": classification,
        "metadata": json.dumps(metadata or {}),
        "loop_index": loop_index,
    }


# ---------------------------------------------------------------------------
# hallucinated_reference_check
# ---------------------------------------------------------------------------
def test_hallucinated_reference_no_context_but_references_past():
    """Tutor references a past session but no context was injected → hallucination."""
    turn = _turn(
        content="This is similar to the circuit problem you solved before.",
        metadata={"student_context": None},
    )
    issues = _hallucinated_reference_check(turn)
    assert len(issues) == 1
    assert issues[0]["check"] == "hallucinated_reference"
    assert issues[0]["severity"] == "high"


def test_hallucinated_reference_context_present_no_past_cue():
    """Tutor doesn't reference past session → no hallucination check."""
    turn = _turn(
        content="Think about energy conservation on the ramp.",
        metadata={"student_context": "[1] Type: ramp | Outcome: solved"},
    )
    assert _hallucinated_reference_check(turn) == []


def test_hallucinated_reference_problem_type_not_in_context():
    """Tutor mentions a problem type not in the context block → hallucination."""
    turn = _turn(
        content="You're making the same mistake as in the circuit problem.",
        metadata={"student_context": "[1] Type: 1D collision | Outcome: solved"},
    )
    issues = _hallucinated_reference_check(turn)
    assert len(issues) == 1
    assert "circuit" in issues[0]["description"].lower()


def test_hallucinated_reference_problem_type_in_context_ok():
    """Tutor mentions a problem type that IS in the context → no issue."""
    turn = _turn(
        content="This is similar to the 1D collision problem you worked on before.",
        metadata={"student_context": "[1] Type: 1D collision | Outcome: solved"},
    )
    assert _hallucinated_reference_check(turn) == []


# ---------------------------------------------------------------------------
# missed_connection_check
# ---------------------------------------------------------------------------
def test_missed_connection_overlap_but_no_reference():
    """Current mistake overlaps past key_mistakes but tutor didn't reference → miss."""
    turn = _turn(
        content="Think about the impulse-momentum theorem again.",
        metadata={
            "student_context": "[1] Type: 1D collision | Mistakes: confused impulse with force",
            "hint": {"mistake": "confused impulse with the net force"},
        },
    )
    issues = _missed_connection_check(turn)
    assert len(issues) == 1
    assert issues[0]["check"] == "missed_connection"
    assert issues[0]["severity"] == "medium"


def test_missed_connection_referenced_so_no_issue():
    """Tutor DID reference the past session → no miss."""
    turn = _turn(
        content="You're making the same mistake as last time — confused impulse with force.",
        metadata={
            "student_context": "[1] Type: 1D collision | Mistakes: confused impulse with force",
            "hint": {"mistake": "confused impulse with the net force"},
        },
    )
    assert _missed_connection_check(turn) == []


def test_missed_connection_no_context_injected():
    """No student_context → no missed connection check."""
    turn = _turn(
        content="Think about momentum.",
        metadata={"student_context": None, "hint": {"mistake": "confused impulse"}},
    )
    assert _missed_connection_check(turn) == []


def test_missed_connection_no_mistake_in_hint():
    """No mistake field in the hint → can't assess overlap."""
    turn = _turn(
        content="Think about momentum.",
        metadata={
            "student_context": "[1] Type: 1D collision | Mistakes: confused impulse with force",
            "hint": {"mistake": None},
        },
    )
    assert _missed_connection_check(turn) == []


def test_missed_connection_no_mistakes_in_context():
    """Context has no key_mistakes → no missed connection."""
    turn = _turn(
        content="Think about momentum.",
        metadata={
            "student_context": "[1] Type: 1D collision | Outcome: solved",
            "hint": {"mistake": "confused impulse with force"},
        },
    )
    assert _missed_connection_check(turn) == []


# ---------------------------------------------------------------------------
# problem_type_missing_check
# ---------------------------------------------------------------------------
def test_problem_type_missing():
    session = {"id": "sess-001", "problem_type": None}
    issues = _problem_type_missing_check(session)
    assert len(issues) == 1
    assert issues[0]["check"] == "problem_type_missing"
    assert issues[0]["severity"] == "low"


def test_problem_type_present():
    session = {"id": "sess-001", "problem_type": "1D collision"}
    assert _problem_type_missing_check(session) == []


# ---------------------------------------------------------------------------
# summary_quality_check
# ---------------------------------------------------------------------------
def test_summary_quality_empty():
    summary = {"session_id": "sess-001", "summary": "", "outcome": "solved"}
    issues = _summary_quality_check(summary)
    assert len(issues) == 1
    assert issues[0]["severity"] == "high"


def test_summary_quality_too_long():
    summary = {"session_id": "sess-001", "summary": "x" * 350, "outcome": "solved"}
    issues = _summary_quality_check(summary)
    assert len(issues) == 1
    assert issues[0]["check"] == "summary_quality"


def test_summary_quality_invalid_outcome():
    summary = {"session_id": "sess-001", "summary": "Student solved momentum problem.", "outcome": "unknown"}
    issues = _summary_quality_check(summary)
    assert len(issues) == 1
    assert "outcome" in issues[0]["description"].lower()


def test_summary_quality_no_physics_vocabulary():
    summary = {"session_id": "sess-001", "summary": "The work was done.", "outcome": "solved"}
    issues = _summary_quality_check(summary)
    assert len(issues) == 1
    assert "vocabulary" in issues[0]["description"].lower()


def test_summary_quality_good():
    summary = {
        "session_id": "sess-001",
        "summary": "Student solved a momentum conservation problem with guidance.",
        "outcome": "solved",
    }
    assert _summary_quality_check(summary) == []


# ---------------------------------------------------------------------------
# context_not_injected_check
# ---------------------------------------------------------------------------
def test_context_not_injected_with_overlap():
    """Student has a prior summary with concept overlap but context is null."""
    session = {
        "id": "sess-002",
        "student_id": "stu-001",
        "concepts": json.dumps(["momentum", "impulse"]),
    }
    turns = [_turn(metadata={"student_context": None}, loop_index=1)]
    summaries = [{
        "student_id": "stu-001",
        "session_id": "sess-001",
        "concepts": json.dumps(["momentum"]),
        "summary": "Past session.",
    }]
    issues = _context_not_injected_check(session, turns, summaries)
    assert len(issues) == 1
    assert issues[0]["check"] == "context_not_injected"
    assert issues[0]["severity"] == "medium"


def test_context_not_injected_no_prior_summaries():
    """No prior summaries → no issue."""
    session = {
        "id": "sess-002",
        "student_id": "stu-001",
        "concepts": json.dumps(["momentum"]),
    }
    turns = [_turn(metadata={"student_context": None}, loop_index=1)]
    summaries = []
    assert _context_not_injected_check(session, turns, summaries) == []


def test_context_not_injected_no_concept_overlap():
    """Prior summary exists but no concept overlap → no issue."""
    session = {
        "id": "sess-002",
        "student_id": "stu-001",
        "concepts": json.dumps(["circuits"]),
    }
    turns = [_turn(metadata={"student_context": None}, loop_index=1)]
    summaries = [{
        "student_id": "stu-001",
        "session_id": "sess-001",
        "concepts": json.dumps(["momentum"]),
        "summary": "Past session.",
    }]
    assert _context_not_injected_check(session, turns, summaries) == []


def test_context_not_injected_context_was_present():
    """Context was injected → no issue."""
    session = {
        "id": "sess-002",
        "student_id": "stu-001",
        "concepts": json.dumps(["momentum"]),
    }
    turns = [_turn(metadata={"student_context": "[1] Type: 1D collision"}, loop_index=1)]
    summaries = [{
        "student_id": "stu-001",
        "session_id": "sess-001",
        "concepts": json.dumps(["momentum"]),
        "summary": "Past session.",
    }]
    assert _context_not_injected_check(session, turns, summaries) == []


# ---------------------------------------------------------------------------
# Existing checks still work
# ---------------------------------------------------------------------------
def test_existing_empty_response_still_works():
    turn = _turn(content="")
    issues = _empty_response_check(turn)
    assert len(issues) == 1
    assert issues[0]["check"] == "empty_response"


def test_existing_answer_leak_still_works():
    turn = _turn(
        content="$$F = ma$$ $$v = \\sqrt{2gh}$$ therefore the answer is 14 N.",
        metadata={"reveal": False},
    )
    issues = _answer_leak_check(turn)
    assert len(issues) == 1
    assert issues[0]["check"] == "answer_leak"


def test_existing_false_positive_affirmation_still_works():
    turn = _turn(content="You're on the right track!", classification="knowledge_gap")
    issues = _false_positive_affirmation_check(turn)
    assert len(issues) == 1


def test_existing_classification_missing_still_works():
    turn = _turn(content="Some hint.", classification=None, metadata={}, loop_index=2)
    issues = _classification_missing_check(turn)
    assert len(issues) == 1


def test_existing_json_parse_failure_still_works():
    turn = _turn(
        content="Some hint.",
        classification="on_track",
        metadata={"hint": {}, "reasoning": None},
    )
    issues = _json_parse_failure_check(turn)
    assert len(issues) == 1
