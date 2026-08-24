"""End-to-end flow test for the guided-discovery loop (mocked LLM + DB)."""

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
    # OCR parse returns a structured problem
    fake_llm.ocr_responses = ["A 2 kg block slides down a 30° frictionless ramp. Find its speed at the bottom."]
    fake_llm.json_responses = [
        {
            "problem_text": "A 2 kg block slides down a 30° frictionless ramp. Find its speed at the bottom.",
            "formulas": ["v = sqrt(2gh)"],
            "concepts": ["energy conservation", "kinematics"],
            "topic": "energy conservation",
            "diagram_description": None,
        },
        # diagnosis after reply 1
        {
            "classification": "misapplication",
            "reasoning": "Student tried kinematics without considering energy.",
            "target_concept": "energy conservation",
            "next_hint_level": 2,
        },
        # diagnosis after reply 2
        {
            "classification": "knowledge_gap",
            "reasoning": "Student does not know how to set up energy equation.",
            "target_concept": "energy conservation",
            "next_hint_level": 3,
        },
        # profile update
        {"concept": "energy conservation", "mastery_score": 0.3},
    ]
    fake_llm.text_responses = [
        "Opening: Where are you stuck on this ramp problem?",  # opening
        "Hint 2: Try writing the energy conservation equation.",  # loop 1
        "Hint 3: Set PE_top = KE_bottom and solve for v.",       # loop 2 (becomes offer)
        "Full solution: PE = KE -> mgh = 0.5mv^2 -> v = sqrt(2gh)...",  # reveal
    ]

    # start
    session = await session_service.start_session(_png_bytes(), "image/png", None, "stu-1")
    sid = UUID(session["id"])
    assert "Opening" in session["opening"]
    turns = fake_supabase.list_turns(sid)
    assert len(turns) == 1 and turns[0]["role"] == "tutor"

    # reply 1 -> loop 1 hint
    r1 = await session_service.reply(sid, "I tried v = u + at but got stuck.")
    assert r1.loop_index == 1
    assert r1.hint_level == 2
    assert not r1.offer_reveal
    assert r1.classification.value == "misapplication"

    # reply 2 -> with max_hint_loops=3, loop 2 is < 3, so still a hint
    r2 = await session_service.reply(sid, "Still not sure how to start.")
    assert r2.loop_index == 2
    assert not r2.offer_reveal

    # reply 3 -> loop 3 == max -> offer reveal
    r3 = await session_service.reply(sid, "I don't know.")
    assert r3.loop_index == 3
    assert r3.offer_reveal is True

    # reply 4 -> student picks reveal ('b')
    r4 = await session_service.reply(sid, "b")
    assert r4.resolved is True
    assert r4.resolution_type.value == "revealed"
    assert r4.solution is not None
    assert "Full solution" in r4.solution

    # Profile was upserted
    profiles = fake_supabase.get_profiles(UUID(session["student_id"]))
    assert any(p["concept"] == "energy conservation" for p in profiles)
