"""End-to-end flow test for the guided-discovery loop (mocked LLM + DB).

The merged tutor service consumes ONE chat_json call per reply (diagnosis +
hint in a single JSON). The canned `json_responses` queue reflects that: each
reply pops one merged object containing both diagnosis fields and hint fields.
"""

from __future__ import annotations

import io
from uuid import UUID

import pytest
from PIL import Image

from app.domain.state import SessionTerminalError
from app.services import session as session_service


def _png_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def test_full_loop_then_reveal(fake_llm, fake_supabase):
    fake_llm.ocr_responses = ["A 2 kg block slides down a 30° frictionless ramp. Find its speed at the bottom."]
    fake_llm.json_responses = [
        # 1. OCR parse
        {
            "problem_text": "A 2 kg block slides down a 30° frictionless ramp. Find its speed at the bottom.",
            "formulas": ["v = sqrt(2gh)"],
            "concepts": ["energy conservation", "kinematics"],
            "topic": "energy conservation",
            "diagram_description": None,
        },
        # 2. merged reply 1 (misapplication -> mistake/reason/application_hint)
        {
            "classification": "misapplication",
            "reasoning": "Student tried kinematics without considering energy.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "formula": None,
            "explanation": None,
            "example": None,
            "mistake": "You used a kinematics formula on a ramp with no friction info.",
            "reason": "You don't know the angle or acceleration, so kinematics is the wrong tool.",
            "application_hint": "Try energy conservation: $E_i = E_f$.",
            "confirmation": None,
            "next_step_hint": None,
        },
        # 3. merged reply 2 (knowledge_gap -> explanation/formula/example)
        {
            "classification": "knowledge_gap",
            "reasoning": "Student does not know how to set up energy equation.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "formula": "$mgh = \\tfrac12 mv^2$",
            "explanation": "Energy conservation: the potential energy at the top equals the kinetic energy at the bottom.",
            "example": "A ball dropped from height $h$ hits the ground at $v = \\sqrt{2gh}$.",
            "mistake": None,
            "reason": None,
            "application_hint": None,
            "confirmation": None,
            "next_step_hint": None,
            "source_title": "OpenStax College Physics",
            "source_url": "https://openstax.org/books/college-physics",
        },
        # 4. merged reply 3 (on_track -> confirmation + next_step_hint)
        {
            "classification": "on_track",
            "reasoning": "Student is making progress.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "formula": None,
            "explanation": None,
            "example": None,
            "mistake": None,
            "reason": None,
            "application_hint": None,
            "confirmation": "Good — you've got the energy-conservation idea.",
            "next_step_hint": "Now solve $mgh = \\tfrac12 mv^2$ for $v$.",
        },
        # 5. merged reply 4 — student asks for the solution (wants_solution true)
        {
            "classification": "on_track",
            "reasoning": "Student explicitly wants the solution.",
            "target_concept": "energy conservation",
            "wants_solution": True,
        },
        # 6. profile update
        {"concept": "energy conservation", "mastery_score": 0.3},
    ]
    fake_llm.text_responses = [
        "Hey there! This is a ramp energy problem. Where are you stuck?",  # opening
        "Full solution: $mgh = \\tfrac12 mv^2 \\Rightarrow v = \\sqrt{2gh}$.",  # reveal
    ]

    # start
    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-1")
    sid = UUID(session["id"])
    assert "stuck" in session["opening"].lower()
    turns = fake_supabase.list_turns(sid)
    assert len(turns) == 1 and turns[0]["role"] == "tutor"

    # reply 1 -> merged misapplication hint
    r1 = await session_service.reply(sid, "I tried v = u + at but got stuck.")
    assert r1.loop_index == 1
    assert r1.hint_level is None
    assert not r1.offer_reveal
    assert not r1.resolved
    assert r1.classification.value == "misapplication"
    assert r1.hint is not None
    assert r1.hint.mistake and "kinematics" in r1.hint.mistake.lower()
    assert r1.hint.reason is not None
    assert r1.hint.application_hint is not None

    # reply 2 -> merged knowledge_gap hint
    r2 = await session_service.reply(sid, "I don't know energy conservation.")
    assert r2.loop_index == 2
    assert not r2.offer_reveal
    assert not r2.resolved
    assert r2.hint is not None
    assert r2.hint.explanation and "energy" in r2.hint.explanation.lower()
    assert r2.hint.formula is not None
    assert r2.hint.example is not None
    assert r2.hint.source_title == "OpenStax College Physics"
    assert r2.hint.source_url == "https://openstax.org/books/college-physics"

    # reply 3 -> merged on_track hint (confirmation + next_step_hint). At loop 3
    # (the default max_hint_loops cap) the tutor offers the reveal — but does
    # NOT force it; the student may keep working.
    r3 = await session_service.reply(sid, "Ok so set mgh equal to half mv squared?")
    assert r3.loop_index == 3
    assert r3.offer_reveal, "at the max_hint_loops cap the tutor must offer (not force) the reveal"
    assert not r3.resolved
    assert r3.hint is not None
    assert r3.hint.confirmation is not None
    assert r3.hint.next_step_hint is not None

    # reply 4 -> student asks for the solution; merged call flags wants_solution
    r4 = await session_service.reply(sid, "can you show me how to solve it?")
    assert r4.resolved is True
    assert r4.resolution_type.value == "revealed"
    assert r4.solution is not None
    assert "Full solution" in r4.solution

    # Profile was upserted
    profiles = fake_supabase.get_profiles(UUID(session["student_id"]))
    assert any(p["concept"] == "energy conservation" for p in profiles)


async def test_explicit_solution_request_skips_tutor_call(fake_llm, fake_supabase):
    """An obvious 'show me the solution' request short-circuits the tutor call."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp."]
    fake_llm.json_responses = [
        {
            "problem_text": "A block slides down a frictionless ramp.",
            "formulas": [],
            "concepts": ["energy conservation"],
            "topic": None,
            "diagram_description": None,
        },
        # profile update (no tutor json consumed because of the fast path)
        {"concept": "energy conservation", "mastery_score": 0.2},
    ]
    fake_llm.text_responses = [
        "Opening: where are you stuck?",
        "Full solution: $mgh = \\tfrac12 mv^2$.",
    ]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-2")
    sid = UUID(session["id"])

    # An explicit request should reveal without invoking the merged tutor call.
    r = await session_service.reply(sid, "show me the solution")
    assert r.resolved is True
    assert r.resolution_type.value == "revealed"
    assert r.solution is not None

    # The fast path must skip the tutor (diagnose+respond) call. Other
    # chat_json calls (OCR parse, profile mastery estimate) are expected.
    from app.prompts.guided_discovery import TUTOR_SYSTEM

    tutor_calls = [
        c for c in fake_llm.calls
        if c["kind"] == "json" and c["system"] == TUTOR_SYSTEM
    ]
    assert len(tutor_calls) == 0, "explicit request must skip the tutor call (fast path)"


async def test_answer_check_correct_confirms_and_does_not_resolve(fake_llm, fake_supabase):
    """A correct answer-check confirms the number but does not auto-resolve —
    the student must say 'got it' (-> solved) or ask for the reveal."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        {  # OCR parse
            "problem_text": "A block slides down a frictionless ramp. Find v.",
            "formulas": [], "concepts": ["energy conservation"],
            "topic": None, "diagram_description": None,
        },
        {  # merged reply: answer_check, correct
            "classification": "answer_check",
            "reasoning": "Student proposes the correct final speed.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "answer_status": "correct",
            "answer_value": "14 N",
            "method_feedback": "Yes, 14 N is correct! Your use of energy conservation is spot on.",
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-ac")
    sid = UUID(session["id"])
    r = await session_service.reply(sid, "is the answer 14 N?")
    assert r.classification.value == "answer_check"
    assert r.hint is not None
    assert r.hint.answer_status == "correct"
    assert r.hint.answer_value == "14 N"
    assert r.hint.method_feedback is not None
    # Correct answer alone does NOT resolve — student may still want to wrap up.
    assert r.resolved is False
    assert "14 N" in (r.hint.method_feedback or "") or "correct" in (r.hint.method_feedback or "").lower()


async def test_solved_resolves_and_records_mastery(fake_llm, fake_supabase):
    """A `solved` classification marks the session solved_with_hints and runs
    the profile mastery update (the bug this fixes: self-solved sessions used
    to never get mastery credit)."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        {  # OCR parse
            "problem_text": "A block slides down a frictionless ramp. Find v.",
            "formulas": [], "concepts": ["energy conservation"],
            "topic": None, "diagram_description": None,
        },
        {  # merged reply: solved
            "classification": "solved",
            "reasoning": "Student demonstrated the full correct solution.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "confirmation": "Nice — you've solved it!",
        },
        {  # profile mastery estimate (from _do_solved)
            "concept": "energy conservation", "mastery_score": 0.9,
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-solved")
    sid = UUID(session["id"])
    r = await session_service.reply(sid, "so v = sqrt(2gh), got it!")
    assert r.resolved is True
    assert r.resolution_type.value == "solved_with_hints"
    assert r.classification.value == "solved"
    assert r.hint is not None
    assert r.hint.confirmation is not None

    # Mastery was recorded — the personalization bug is fixed.
    profiles = fake_supabase.get_profiles(UUID(session["student_id"]))
    assert any(p["concept"] == "energy conservation" for p in profiles), (
        "self-solved session must record mastery"
    )


async def test_incorrect_answer_returns_mistake(fake_llm, fake_supabase):
    """A wrong final answer stated without asking for confirmation -> incorrect_answer."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        {  # OCR parse
            "problem_text": "A block slides down a frictionless ramp. Find v.",
            "formulas": [], "concepts": ["energy conservation"],
            "topic": None, "diagram_description": None,
        },
        {  # merged reply: incorrect_answer
            "classification": "incorrect_answer",
            "reasoning": "Student used the full momentum, not the vertical component.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "mistake": "You used the horizontal momentum component, which doesn't change.",
            "reason": "Only the vertical component contributes to the net impulse here.",
            "application_hint": "Recompute using Δp_y only.",
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-ia")
    sid = UUID(session["id"])
    r = await session_service.reply(sid, "the answer is 20 N")
    assert r.classification.value == "incorrect_answer"
    assert r.hint is not None
    assert r.hint.mistake is not None
    assert r.hint.reason is not None
    assert r.resolved is False


async def test_meta_returns_meta_response(fake_llm, fake_supabase):
    """A procedural/clarification question -> meta, with a direct answer."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        {  # OCR parse
            "problem_text": "A block slides down a frictionless ramp. Find v.",
            "formulas": [], "concepts": ["energy conservation"],
            "topic": None, "diagram_description": None,
        },
        {  # merged reply: meta
            "classification": "meta",
            "reasoning": "Student asks to re-explain energy conservation in this context.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "meta_response": "Sure — energy conservation here means the PE at the top turns into KE at the bottom.",
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-meta")
    sid = UUID(session["id"])
    r = await session_service.reply(sid, "can you explain that again?")
    assert r.classification.value == "meta"
    assert r.hint is not None
    assert r.hint.meta_response is not None
    assert r.resolved is False


async def test_reply_after_reveal_raises_session_terminal(fake_llm, fake_supabase):
    """A session that has been revealed is terminal; further replies raise
    `SessionTerminalError` (the routes layer maps this to 409)."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        {  # OCR parse
            "problem_text": "A block slides down a frictionless ramp. Find v.",
            "formulas": [], "concepts": ["energy conservation"],
            "topic": None, "diagram_description": None,
        },
        {  # profile mastery estimate (from the reveal)
            "concept": "energy conservation", "mastery_score": 0.2,
        },
    ]
    fake_llm.text_responses = [
        "Opening: where are you stuck?",
        "Full solution: $mgh = \\tfrac12 mv^2$.",
    ]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-term")
    sid = UUID(session["id"])

    # Reveal -> session becomes terminal.
    await session_service.reveal_solution(sid)

    # Any further reply must raise SessionTerminalError (no extra LLM calls consumed).
    with pytest.raises(SessionTerminalError):
        await session_service.reply(sid, "one more question")


async def test_reveal_idempotent_on_terminal_session(fake_llm, fake_supabase):
    """Reveal on an already-resolved session is idempotent: it returns the
    recorded resolution without re-running the LLM."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        {  # OCR parse
            "problem_text": "A block slides down a frictionless ramp. Find v.",
            "formulas": [], "concepts": ["energy conservation"],
            "topic": None, "diagram_description": None,
        },
        {  # profile mastery estimate (from the first reveal)
            "concept": "energy conservation", "mastery_score": 0.2,
        },
    ]
    fake_llm.text_responses = [
        "Opening: where are you stuck?",
        "Full solution: $mgh = \\tfrac12 mv^2$.",
    ]

    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-idem")
    sid = UUID(session["id"])

    first = await session_service.reveal_solution(sid)
    assert first["resolved"] is True
    assert first["resolution_type"] == "revealed"

    # Second reveal must NOT consume another text response (idempotent).
    second = await session_service.reveal_solution(sid)
    assert second["resolved"] is True
    assert second["resolution_type"] == "revealed"
    assert "Full solution" in second["solution"]
    # Only one text response was queued for the reveal; the idempotent path
    # must not have popped beyond it. Verify by checking no extra calls were
    # made beyond the first opening + first reveal.
    text_calls = [c for c in fake_llm.calls if c["kind"] == "text"]
    assert len(text_calls) == 2, "second reveal must not consume another LLM call"


async def test_parallel_replies_do_not_block(fake_llm, fake_supabase):
    """Concurrency smoke test: multiple reply() calls in flight concurrently
    should all complete. A blocking sync DB call (not wrapped in to_thread)
    would serialize on the event loop; with to_thread the LLM awaits can
    interleave. We assert all N parallel replies return and each gets the
    correct classification from its own queued LLM response.
    """
    import asyncio

    n = 4
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    # Each session gets one OCR parse + one reply (on_track). Queue enough
    # responses for n sessions.
    fake_llm.json_responses = [
        {  # OCR parse (repeated for n sessions)
            "problem_text": "A block slides down a frictionless ramp. Find v.",
            "formulas": [], "concepts": ["energy conservation"],
            "topic": None, "diagram_description": None,
        }
        for _ in range(n)
    ] + [
        {  # reply (on_track) for each session
            "classification": "on_track",
            "reasoning": "Student is making progress.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "confirmation": "Good progress!",
            "next_step_hint": "Keep going.",
        }
        for _ in range(n)
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"] * n

    # Start n sessions in parallel, then reply to each in parallel.
    sessions = await asyncio.gather(*[
        session_service.start_session(_png_bytes(), "image/png", None, f"stu-par-{i}")
        for i in range(n)
    ])
    sids = [UUID(s["id"]) for s in sessions]

    replies = await asyncio.gather(*[
        session_service.reply(sid, f"reply {i}") for i, sid in enumerate(sids)
    ])

    assert len(replies) == n
    for r in replies:
        assert r.classification.value == "on_track"
        assert not r.resolved
