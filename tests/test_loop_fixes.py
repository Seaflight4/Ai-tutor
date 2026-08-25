"""Regression tests for three diagnosed loop issues.

1. `generate_opening` falls back to a non-empty message when the LLM returns
   an empty string (after chain-of-thought stripping).
2. `offer_reveal` is suppressed when the student demonstrates a full solution
   (`Classification.solved`).
3. `offer_reveal` is suppressed when the student proposes a correct answer
   (`Classification.answer_check` with `answer_status == "correct"`), even at
   the `max_hint_loops` cap.
"""

from __future__ import annotations

import io
from uuid import UUID

from PIL import Image

from app.core import config
from app.prompts import guided_discovery as p
from app.services import hints
from app.services import session as session_service


def _png_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ocr_json() -> dict:
    return {
        "problem_text": "A block slides down a frictionless ramp. Find v.",
        "formulas": [],
        "concepts": ["energy conservation"],
        "topic": None,
        "diagram_description": None,
    }


async def test_generate_opening_falls_back_when_llm_returns_empty(fake_llm):
    """An empty LLM response must not produce an empty opening turn."""
    fake_llm.text_responses = [""]

    opening = await hints.generate_opening("A block slides down a ramp.", ["energy"])
    assert opening and opening.strip(), "opening must be non-empty even if the LLM returns empty"
    assert "stuck" in opening.lower()


async def test_offer_reveal_false_when_classification_solved(fake_llm, fake_supabase):
    """A `solved` classification must not offer a reveal."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        {
            "classification": "solved",
            "reasoning": "Student demonstrated the full correct solution.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "confirmation": "Nice — you've solved it!",
        },
        {"concept": "energy conservation", "mastery_score": 0.9},
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-solved-fix")
    sid = UUID(session["id"])
    r = await session_service.reply(sid, "so v = sqrt(2gh), got it!")
    assert r.classification.value == "solved"
    assert r.offer_reveal is False


async def test_offer_reveal_false_when_answer_check_correct_at_cap(
    monkeypatch, fake_llm, fake_supabase
):
    """A correct answer-check at the max_hint_loops cap must not offer a reveal.

    Without the fix, `offer_reveal = new_loop >= max_loops` would be True at the
    cap. We force the cap to 1 so the very first reply hits it.
    """
    monkeypatch.setattr(
        config, "get_settings", lambda: config.Settings(max_hint_loops=1)
    )

    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        {
            "classification": "answer_check",
            "reasoning": "Student proposes the correct final answer.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "answer_status": "correct",
            "answer_value": "14 N",
            "method_feedback": "Yes, 14 N is correct!",
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-ac-fix")
    sid = UUID(session["id"])
    r = await session_service.reply(sid, "is the answer 14 N?")
    assert r.classification.value == "answer_check"
    assert r.hint is not None
    assert r.hint.answer_status == "correct"
    assert r.offer_reveal is False


def test_tutor_system_distinguishes_how_from_solution_request():
    """The wants_solution guidance must call out that 'how to solve' is NOT a
    solution request, and list explicit solution-request phrases."""
    prompt = p.TUTOR_SYSTEM
    assert "just tell me how to solve this" in prompt
    assert "how do I solve this?" in prompt
    assert "what's the next step?" in prompt
    assert "give me the answer" in prompt
    assert "show me the full solution" in prompt
    assert "I give up" in prompt
    assert "reveal the solution" in prompt
    assert "NOT a solution request" in prompt
