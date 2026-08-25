"""Tests for the cross-session learning record (Phase 1).

Covers:
- `add_session_summary` / `list_session_summaries` / `find_related_summaries`
  on the in-memory backend.
- `build_student_context` formatting + None for anonymous students.
- `_generate_session_summary` making an LLM call and storing the result.
- `assess_and_respond` accepting and passing `student_context` through to the
  prompt.
- `reply` injecting student context when the student has history.
- `reply` skipping student context when no student_id is available.
- `problem_type` flowing from OCR parse through to session creation.
- `TUTOR_SYSTEM` prompt containing the STUDENT CONTEXT instructions.
"""

from __future__ import annotations

import io
from uuid import UUID

from PIL import Image

from app.prompts import guided_discovery as p
from app.services import profile
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
        "problem_type": "energy conservation on incline",
        "diagram_description": None,
    }


# ---------------------------------------------------------------------------
# Store-level tests (in-memory backend)
# ---------------------------------------------------------------------------
def test_add_and_list_session_summaries(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000001")
    sess = UUID("00000000-0000-0000-0000-000000000010")
    fake_supabase.add_session_summary(
        sid, sess,
        problem_text="Block on ramp",
        concepts=["energy conservation"],
        problem_type="energy conservation on incline",
        outcome="solved",
        target_concept="energy conservation",
        summary="Student solved the ramp problem.",
        key_mistakes="forgot friction term initially",
    )
    out = fake_supabase.list_session_summaries(sid)
    assert len(out) == 1
    row = out[0]
    assert row["summary"] == "Student solved the ramp problem."
    assert row["outcome"] == "solved"
    assert row["problem_type"] == "energy conservation on incline"
    assert row["key_mistakes"] == "forgot friction term initially"
    assert row["concepts"] == ["energy conservation"]


def test_list_session_summaries_respects_limit(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000001")
    for i in range(7):
        fake_supabase.add_session_summary(
            sid, UUID(f"00000000-0000-0000-0000-{i:012d}"),
            problem_text=f"problem {i}",
            concepts=["momentum"],
            problem_type="1D collision",
            outcome="solved",
            target_concept=None,
            summary=f"summary {i}",
        )
    out = fake_supabase.list_session_summaries(sid, limit=3)
    assert len(out) == 3


def test_list_session_summaries_empty_for_new_student(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000999")
    assert fake_supabase.list_session_summaries(sid) == []


def test_find_related_summaries_matches_by_concept_overlap(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000001")
    # Related — shares "momentum"
    fake_supabase.add_session_summary(
        sid, UUID("00000000-0000-0000-0000-000000000010"),
        problem_text="1D collision",
        concepts=["momentum", "impulse"],
        problem_type="1D collision",
        outcome="solved",
        target_concept=None,
        summary="Student solved a 1D collision.",
    )
    # Unrelated — no concept overlap
    fake_supabase.add_session_summary(
        sid, UUID("00000000-0000-0000-0000-000000000020"),
        problem_text="Circuit analysis",
        concepts=["ohms law"],
        problem_type="circuit analysis",
        outcome="revealed",
        target_concept=None,
        summary="Student needed a reveal for the circuit problem.",
    )
    related = fake_supabase.find_related_summaries(
        sid, ["momentum", "elastic collision"], problem_type="elastic collision with angle"
    )
    assert len(related) == 1
    assert related[0]["problem_type"] == "1D collision"


def test_find_related_summaries_prioritizes_problem_type_match(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000001")
    # Same concepts, different problem_type
    fake_supabase.add_session_summary(
        sid, UUID("00000000-0000-0000-0000-000000000010"),
        problem_text="Energy on incline",
        concepts=["energy conservation"],
        problem_type="projectile motion",
        outcome="solved",
        target_concept=None,
        summary="Projectile problem.",
    )
    fake_supabase.add_session_summary(
        sid, UUID("00000000-0000-0000-0000-000000000020"),
        problem_text="Energy on incline 2",
        concepts=["energy conservation"],
        problem_type="energy conservation on incline",
        outcome="solved",
        target_concept=None,
        summary="Incline energy problem.",
    )
    related = fake_supabase.find_related_summaries(
        sid, ["energy conservation"], problem_type="energy conservation on incline"
    )
    assert len(related) == 2
    # The type-matching one should come first
    assert related[0]["problem_type"] == "energy conservation on incline"


def test_find_related_summaries_empty_when_no_concept_overlap(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000001")
    fake_supabase.add_session_summary(
        sid, UUID("00000000-0000-0000-0000-000000000010"),
        problem_text="Circuit",
        concepts=["ohms law"],
        problem_type="circuit analysis",
        outcome="solved",
        target_concept=None,
        summary="Circuit problem.",
    )
    assert fake_supabase.find_related_summaries(sid, ["momentum"]) == []


def test_find_related_summaries_empty_when_no_history(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000999")
    assert fake_supabase.find_related_summaries(sid, ["momentum"]) == []


# ---------------------------------------------------------------------------
# build_student_context
# ---------------------------------------------------------------------------
async def test_build_student_context_returns_none_for_anonymous(fake_supabase):
    ctx = await profile.build_student_context(None, ["momentum"], "1D collision")
    assert ctx is None


async def test_build_student_context_returns_none_for_new_student(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000999")
    ctx = await profile.build_student_context(sid, ["momentum"], "1D collision")
    assert ctx is None


async def test_build_student_context_formats_correctly(fake_supabase):
    sid = UUID("00000000-0000-0000-0000-000000000001")
    fake_supabase.add_session_summary(
        sid, UUID("00000000-0000-0000-0000-000000000010"),
        problem_text="1D collision",
        concepts=["momentum", "impulse"],
        problem_type="1D collision",
        outcome="solved",
        target_concept=None,
        summary="Student solved a 1D collision.",
        key_mistakes="confused impulse with force",
    )
    ctx = await profile.build_student_context(sid, ["momentum"], "1D collision")
    assert ctx is not None
    assert "[1]" in ctx
    assert "1D collision" in ctx
    assert "momentum" in ctx
    assert "solved" in ctx
    assert "confused impulse with force" in ctx
    # The summary sentence is intentionally dropped — only structured fields.
    assert "Student solved a 1D collision." not in ctx


# ---------------------------------------------------------------------------
# _generate_session_summary (via reply termination)
# ---------------------------------------------------------------------------
async def test_session_summary_generated_on_reveal(fake_llm, fake_supabase):
    """When a session is revealed, a summary should be stored."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        # Profile update on reveal
        {"concept": "energy conservation", "mastery_score": 0.4},
        # Session summary on reveal
        {"summary": "Student struggled with energy conservation.", "key_mistakes": "sign error"},
    ]
    fake_llm.text_responses = [
        "Opening: where are you stuck?",
        # Full solution (generated by solution.generate_solution -> chat_text)
        "The solution is v = sqrt(2gh).",
    ]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-summary-reveal"
    )
    sid = UUID(session["id"])

    # Force a reveal by sending an explicit solution request.
    await session_service.reply(sid, "just give me the solution please")

    summaries = fake_supabase.list_session_summaries(UUID(session["student_id"]))
    assert len(summaries) == 1
    assert summaries[0]["outcome"] == "revealed"
    assert "energy conservation" in summaries[0]["summary"].lower()
    assert summaries[0]["key_mistakes"] == "sign error"
    assert summaries[0]["problem_type"] == "energy conservation on incline"


async def test_session_summary_generated_on_solved(fake_llm, fake_supabase):
    """When a student solves the problem, a summary should be stored."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        # The reply diagnosis -> solved
        {
            "classification": "solved",
            "reasoning": "Student demonstrated the full correct solution.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "confirmation": "Nice — you've solved it!",
        },
        # Profile update on solved
        {"concept": "energy conservation", "mastery_score": 0.95},
        # Session summary on solved
        {"summary": "Student solved the ramp problem independently.", "key_mistakes": None},
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-summary-solved"
    )
    sid = UUID(session["id"])
    await session_service.reply(sid, "so v = sqrt(2gh), got it!")

    summaries = fake_supabase.list_session_summaries(UUID(session["student_id"]))
    assert len(summaries) == 1
    assert summaries[0]["outcome"] == "solved"
    assert "independently" in summaries[0]["summary"].lower()
    assert summaries[0]["key_mistakes"] is None


async def test_session_summary_falls_back_when_llm_returns_empty(fake_llm, fake_supabase):
    """If the LLM returns an empty summary, a fallback string is stored."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        {"concept": "energy conservation", "mastery_score": 0.4},
        # Empty summary
        {"summary": "", "key_mistakes": None},
    ]
    fake_llm.text_responses = [
        "Opening: where are you stuck?",
        "The solution is v = sqrt(2gh).",
    ]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-summary-empty"
    )
    sid = UUID(session["id"])
    await session_service.reply(sid, "just give me the solution please")

    summaries = fake_supabase.list_session_summaries(UUID(session["student_id"]))
    assert len(summaries) == 1
    assert summaries[0]["summary"]  # non-empty fallback


# ---------------------------------------------------------------------------
# student_context injection into reply
# ---------------------------------------------------------------------------
async def test_reply_injects_student_context_when_history_exists(fake_llm, fake_supabase):
    """When the student has past summaries, the tutor prompt receives a
    STUDENT CONTEXT block."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        # The reply diagnosis
        {
            "classification": "on_track",
            "reasoning": "Student is on the right track.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "next_step_hint": "Think about energy conservation.",
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-ctx-inject"
    )
    sid = UUID(session["id"])
    student_id = UUID(session["student_id"])

    # Seed a past session summary so find_related_summaries returns something.
    fake_supabase.add_session_summary(
        student_id, UUID("00000000-0000-0000-0000-000000000099"),
        problem_text="Past ramp problem",
        concepts=["energy conservation"],
        problem_type="energy conservation on incline",
        outcome="solved",
        target_concept=None,
        summary="Student solved a similar ramp problem before.",
        key_mistakes="forgot the friction term",
    )

    await session_service.reply(sid, "I think I should use energy conservation")

    # The tutor call should have a STUDENT CONTEXT block in its user prompt.
    tutor_calls = [c for c in fake_llm.calls if c["kind"] == "json"]
    # Second json call is the reply diagnosis (first is OCR parse)
    reply_call = tutor_calls[-1]
    assert "STUDENT CONTEXT" in reply_call["user"]
    assert "energy conservation on incline" in reply_call["user"]
    assert "forgot the friction term" in reply_call["user"]


async def test_reply_skips_student_context_for_anonymous_student(fake_llm, fake_supabase):
    """Anonymous sessions (no student_id) should not get a STUDENT CONTEXT block."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        {
            "classification": "on_track",
            "reasoning": "Student is on the right track.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "next_step_hint": "Think about energy conservation.",
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-ctx-anon"
    )
    sid = UUID(session["id"])

    await session_service.reply(sid, "I think I should use energy conservation")

    tutor_calls = [c for c in fake_llm.calls if c["kind"] == "json"]
    reply_call = tutor_calls[-1]
    assert "STUDENT CONTEXT" not in reply_call["user"]


# ---------------------------------------------------------------------------
# problem_type flow
# ---------------------------------------------------------------------------
async def test_problem_type_stored_on_session(fake_llm, fake_supabase):
    """problem_type from OCR parse should flow through to the session row."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [_ocr_json()]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-ptype"
    )
    assert session["problem_type"] == "energy conservation on incline"


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------
def test_tutor_system_contains_student_context_instructions():
    prompt = p.TUTOR_SYSTEM
    assert "STUDENT CONTEXT" in prompt
    assert "CONCEPT CONNECTION" in prompt
    assert "RECURRING MISTAKE" in prompt
    assert "Do not invent details" in prompt


def test_ocr_parse_system_requests_problem_type():
    prompt = p.OCR_PARSE_SYSTEM
    assert "problem_type" in prompt


def test_tutor_user_includes_student_context_when_provided():
    msg = p.tutor_user(
        "Problem text",
        ["momentum"],
        "tutor: hi",
        "student: hello",
        1,
        student_context="[1] Type: 1D collision | Outcome: solved | Past summary.",
    )
    assert "STUDENT CONTEXT" in msg
    assert "1D collision" in msg


def test_tutor_user_omits_student_context_when_none():
    msg = p.tutor_user(
        "Problem text",
        ["momentum"],
        "tutor: hi",
        "student: hello",
        1,
    )
    assert "STUDENT CONTEXT" not in msg


def test_session_summary_system_prompt_exists():
    assert "summarize" in p.SESSION_SUMMARY_SYSTEM.lower()
    assert "summary" in p.SESSION_SUMMARY_SYSTEM
    assert "key_mistakes" in p.SESSION_SUMMARY_SYSTEM


def test_session_summary_user_prompt_includes_fields():
    msg = p.session_summary_user(
        "Block on ramp",
        ["energy conservation"],
        "energy conservation on incline",
        "solved",
        "energy conservation",
        "tutor: hi\nstudent: I got it",
    )
    assert "Block on ramp" in msg
    assert "energy conservation on incline" in msg
    assert "solved" in msg


# ---------------------------------------------------------------------------
# Step 1: student_context stored in turn metadata
# ---------------------------------------------------------------------------
async def test_student_context_stored_in_turn_metadata(fake_llm, fake_supabase):
    """The injected student_context should be persisted in the tutor turn's
    metadata for reviewer auditability."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        {
            "classification": "on_track",
            "reasoning": "Student is on the right track.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "next_step_hint": "Think about energy conservation.",
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-meta-ctx"
    )
    sid = UUID(session["id"])
    student_id = UUID(session["student_id"])

    # Seed a past summary so context is injected.
    fake_supabase.add_session_summary(
        student_id, UUID("00000000-0000-0000-0000-000000000099"),
        problem_text="Past ramp problem",
        concepts=["energy conservation"],
        problem_type="energy conservation on incline",
        outcome="solved",
        target_concept=None,
        summary="Student solved a similar ramp problem before.",
        key_mistakes="forgot the friction term",
    )

    await session_service.reply(sid, "I think I should use energy conservation")

    # Find the tutor turn (loop_index >= 1) and check its metadata.
    turns = fake_supabase.list_turns(sid)
    tutor_turns = [t for t in turns if t["role"] == "tutor" and t["loop_index"] >= 1]
    assert len(tutor_turns) >= 1
    meta = tutor_turns[-1]["metadata"]
    assert meta.get("student_context") is not None
    assert "energy conservation on incline" in meta["student_context"]


async def test_student_context_null_in_metadata_when_anonymous(fake_llm, fake_supabase):
    """Anonymous sessions should store null student_context in metadata."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        {
            "classification": "on_track",
            "reasoning": "Student is on the right track.",
            "target_concept": "energy conservation",
            "wants_solution": False,
            "next_step_hint": "Think about energy conservation.",
        },
    ]
    fake_llm.text_responses = ["Opening: where are you stuck?"]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-meta-anon"
    )
    sid = UUID(session["id"])
    await session_service.reply(sid, "I think I should use energy conservation")

    turns = fake_supabase.list_turns(sid)
    tutor_turns = [t for t in turns if t["role"] == "tutor" and t["loop_index"] >= 1]
    meta = tutor_turns[-1]["metadata"]
    assert meta.get("student_context") is None


# ---------------------------------------------------------------------------
# Step 4c: key_mistakes as JSON array, joined with "; "
# ---------------------------------------------------------------------------
async def test_key_mistakes_array_joined_on_reveal(fake_llm, fake_supabase):
    """key_mistakes returned as a JSON array should be joined with '; '
    before storing."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        {"concept": "energy conservation", "mastery_score": 0.4},
        {
            "summary": "Student struggled with energy conservation.",
            "key_mistakes": ["confused impulse with force", "sign error on velocity"],
        },
    ]
    fake_llm.text_responses = [
        "Opening: where are you stuck?",
        "The solution is v = sqrt(2gh).",
    ]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-mistakes-array"
    )
    sid = UUID(session["id"])
    await session_service.reply(sid, "just give me the solution please")

    summaries = fake_supabase.list_session_summaries(UUID(session["student_id"]))
    assert len(summaries) == 1
    # The array should be joined with "; "
    assert summaries[0]["key_mistakes"] == "confused impulse with force; sign error on velocity"


async def test_key_mistakes_empty_array_becomes_none(fake_llm, fake_supabase):
    """An empty key_mistakes array should be stored as None."""
    fake_llm.ocr_responses = ["A block slides down a frictionless ramp. Find v."]
    fake_llm.json_responses = [
        _ocr_json(),
        {"concept": "energy conservation", "mastery_score": 0.4},
        {
            "summary": "Student solved the problem with no notable mistakes.",
            "key_mistakes": [],
        },
    ]
    fake_llm.text_responses = [
        "Opening: where are you stuck?",
        "The solution is v = sqrt(2gh).",
    ]

    session = await session_service.start_session(
        _png_bytes(), "image/png", None, "stu-mistakes-empty"
    )
    sid = UUID(session["id"])
    await session_service.reply(sid, "just give me the solution please")

    summaries = fake_supabase.list_session_summaries(UUID(session["student_id"]))
    assert len(summaries) == 1
    assert summaries[0]["key_mistakes"] is None


# ---------------------------------------------------------------------------
# Step 4a: MUST-reference prompt directive
# ---------------------------------------------------------------------------
def test_tutor_system_has_must_reference_directive():
    """The RECURRING MISTAKE section should use MUST, not may."""
    prompt = p.TUTOR_SYSTEM
    assert "MUST point out the connection" in prompt
    assert "Do not mention problem types or mistakes not listed" in prompt


def test_session_summary_system_requests_array_mistakes():
    """The summary prompt should ask for key_mistakes as an array."""
    prompt = p.SESSION_SUMMARY_SYSTEM
    assert "[string]" in prompt
    assert "key_mistakes" in prompt
