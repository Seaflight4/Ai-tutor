"""End-to-end flow test for the guided-discovery loop (mocked LLM + DB).

The merged tutor service consumes ONE chat_json call per reply (diagnosis +
hint in a single JSON). The canned `json_responses` queue reflects that: each
reply pops one merged object containing both diagnosis fields and hint fields.
"""

from __future__ import annotations

import io
from uuid import UUID

from PIL import Image

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

    # reply 3 -> merged on_track hint (confirmation + next_step_hint)
    r3 = await session_service.reply(sid, "Ok so set mgh equal to half mv squared?")
    assert r3.loop_index == 3
    assert not r3.offer_reveal, "continuous loop must not force a reveal offer"
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
