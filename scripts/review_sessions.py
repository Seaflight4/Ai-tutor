"""Rule-based review of tutor responses in the local SQLite DB.

Takes session IDs as arguments (or "all" for every session), queries
ai_tutor_local.db, and runs deterministic checks on every tutor turn.
Outputs a JSON report to stdout:

    {
      "sessions_reviewed": N,
      "issues": [
        {"severity": "high", "check": "empty_response", "turn_id": "...",
         "description": "...", "root_cause": "...", "suggested_fix": "..."}
      ]
    }

Exit code 0 = no issues, 1 = issues found.

    python -m scripts.review_sessions <session_id> [<session_id> ...]
    python -m scripts.review_sessions all
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path("ai_tutor_local.db")

# Past-session cue phrases — if these appear in tutor content, the tutor is
# referencing a past session and the hallucination / missed-connection checks
# apply.
_PAST_CUES = (
    "last time", "before", "previously", "similar to", "same mistake",
    "last session", "you worked on", "you solved", "you struggled",
    "like the", "as in the",
)

# Physics-relevant stopwords to skip when checking summary vocabulary.
_STOPWORDS = frozenset([
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "for",
    "is", "was", "are", "were", "be", "been", "being", "this", "that",
    "those", "these", "with", "from", "by", "as", "it", "its", "their",
    "his", "her", "your", "you", "we", "they", "he", "she", "not", "no",
    "but", "if", "then", "so", "than", "when", "where", "what", "which",
    "who", "whom", "whose", "how", "why", "did", "do", "does", "done",
    "has", "have", "had", "can", "could", "should", "would", "may",
    "might", "must", "shall", "will", "about", "into", "over", "under",
    "up", "down", "out", "off", "again", "more", "most", "other", "some",
    "such", "only", "own", "same", "too", "very", "just", "now",
    "problem", "student", "tutor", "session", "work", "worked",
])

# Severity-ordered issue dict helper.
def _issue(severity: str, check: str, turn_id: str, desc: str,
           cause: str, fix: str) -> dict:
    return {
        "severity": severity,
        "check": check,
        "turn_id": turn_id,
        "description": desc,
        "root_cause": cause,
        "suggested_fix": fix,
    }


# ---------------------------------------------------------------------------
# Existing checks (turn-level, single-session)
# ---------------------------------------------------------------------------
def _empty_response_check(turn: dict) -> list[dict]:
    if turn["role"] != "tutor":
        return []
    content = (turn["content"] or "").strip()
    if not content:
        return [_issue(
            "high", "empty_response", turn["id"],
            f"Tutor turn {turn['id'][:8]} has empty content",
            "LLM returned empty response or JSON parse failure "
            "left all hint fields null",
            "Check _strip_think_only + _parse_json; ensure "
            "fallback content is never empty",
        )]
    return []


def _json_parse_failure_check(turn: dict) -> list[dict]:
    if turn["role"] != "tutor" or turn["classification"] is None:
        return []
    meta = json.loads(turn["metadata"]) if turn["metadata"] else {}
    hint = meta.get("hint", {})
    all_null = all(
        hint.get(k) is None
        for k in ("formula", "explanation", "example", "mistake", "reason",
                   "application_hint", "confirmation", "next_step_hint",
                   "answer_status", "answer_value", "method_feedback",
                   "meta_response")
    )
    if all_null and not meta.get("reasoning"):
        return [_issue(
            "high", "json_parse_failure", turn["id"],
            f"Tutor turn {turn['id'][:8]} has classification "
            f"'{turn['classification']}' but all hint fields and "
            "reasoning are null — JSON parse likely failed",
            "LLM JSON output not parsed correctly",
            "Check _strip_think_only and _parse_json; verify "
            "the LLM response format matches parser expectations",
        )]
    return []


def _false_positive_affirmation_check(turn: dict) -> list[dict]:
    if turn["role"] != "tutor":
        return []
    content = (turn["content"] or "").lower()
    classification = turn["classification"]
    if "right track" in content and classification != "on_track":
        return [_issue(
            "high", "false_positive_affirmation", turn["id"],
            f"Tutor says 'right track' but classification is "
            f"'{classification}', not 'on_track'",
            "Classification default or hint rendering mismatch",
            "Check tutor.assess_and_respond default "
            "classification and hints.summarize_hint fallback",
        )]
    return []


def _classification_missing_check(turn: dict) -> list[dict]:
    if turn["role"] != "tutor":
        return []
    meta = json.loads(turn["metadata"]) if turn["metadata"] else {}
    if meta.get("reveal"):
        return []
    if turn["classification"] is None and turn["loop_index"] > 0:
        return [_issue(
            "medium", "classification_missing", turn["id"],
            f"Tutor turn {turn['id'][:8]} has no classification "
            "(non-opening, non-reveal)",
            "LLM did not return a classification field",
            "Check tutor.assess_and_respond default handling",
        )]
    return []


def _answer_leak_check(turn: dict) -> list[dict]:
    if turn["role"] != "tutor":
        return []
    meta = json.loads(turn["metadata"]) if turn["metadata"] else {}
    if meta.get("reveal"):
        return []
    content = turn["content"] or ""
    has_multiple_eqs = content.count("$") >= 4
    has_answer_phrase = any(
        p in content.lower()
        for p in ["therefore", "so the answer", "the answer is", "final answer"]
    )
    if has_multiple_eqs and has_answer_phrase:
        return [_issue(
            "medium", "answer_leak", turn["id"],
            f"Tutor turn {turn['id'][:8]} may be leaking the full "
            "solution (multiple equations + answer phrase) without "
            "a reveal request",
            "Tutor prompt not strict enough about not solving",
            "Strengthen the TUTOR_SYSTEM prompt's no-solve rule",
        )]
    return []


# ---------------------------------------------------------------------------
# New checks (learning-record / cross-session)
# ---------------------------------------------------------------------------
def _student_context(meta: dict) -> str | None:
    """Extract the student_context string from turn metadata, or None."""
    ctx = meta.get("student_context")
    if isinstance(ctx, str) and ctx.strip():
        return ctx
    return None


def _has_past_cue(content: str) -> bool:
    lower = content.lower()
    return any(cue in lower for cue in _PAST_CUES)


def _hallucinated_reference_check(turn: dict) -> list[dict]:
    """Flag tutor turns that mention past-session details not in the
    injected student_context block."""
    if turn["role"] != "tutor":
        return []
    content = turn["content"] or ""
    if not _has_past_cue(content):
        return []
    meta = json.loads(turn["metadata"]) if turn["metadata"] else {}
    ctx = _student_context(meta)
    # If no context was injected but the tutor references a past session,
    # that's a hallucination (it invented the reference).
    if ctx is None:
        return [_issue(
            "high", "hallucinated_reference", turn["id"],
            f"Tutor turn {turn['id'][:8]} references a past session but "
            "no student_context was injected — the reference is invented",
            "TUTOR_SYSTEM prompt allows references without grounding",
            "Add a guard: 'Only reference past sessions if a STUDENT "
            "CONTEXT block is present; never invent past-session details'",
        )]
    # If context was injected, check that mentioned problem types / mistakes
    # appear in the context block. We extract capitalized noun phrases and
    # known problem-type patterns from the tutor content and verify they
    # appear in the context. This is a coarse heuristic.
    ctx_lower = ctx.lower()
    # Check for quoted problem types — "the X problem" pattern.
    import re
    refs = re.findall(r"(?:the|like the|as in the)\s+([a-z][a-z\s]{3,40}?)\s+problem",
                      content.lower())
    hallucinated = [r for r in refs if r.strip() not in ctx_lower]
    if hallucinated:
        return [_issue(
            "high", "hallucinated_reference", turn["id"],
            f"Tutor turn {turn['id'][:8]} references problem type(s) "
            f"{hallucinated} not present in the injected student_context",
            "TUTOR_SYSTEM prompt not strict enough about staying "
            "within STUDENT CONTEXT facts",
            "Add: 'Do not mention problem types or mistakes not listed "
            "in the STUDENT CONTEXT block'",
        )]
    return []


def _missed_connection_check(turn: dict) -> list[dict]:
    """Flag when student_context was injected AND the current hint has a
    mistake field that overlaps a past key_mistakes entry, but the tutor
    didn't reference the past session."""
    if turn["role"] != "tutor":
        return []
    meta = json.loads(turn["metadata"]) if turn["metadata"] else {}
    ctx = _student_context(meta)
    if ctx is None:
        return []
    content = turn["content"] or ""
    if _has_past_cue(content):
        return []  # The tutor did reference it — no miss.
    hint = meta.get("hint", {}) or {}
    current_mistake = (hint.get("mistake") or "").lower()
    if not current_mistake:
        return []
    # Extract key_mistakes entries from the context block. They appear
    # after "Mistakes:" and before the next "|" or end of line.
    import re
    mistakes_in_ctx = re.findall(r"Mistakes:\s*([^|]+)", ctx)
    if not mistakes_in_ctx:
        return []
    # Tokenize the current mistake and check overlap with past mistakes.
    cur_tokens = {w for w in current_mistake.split() if len(w) > 3}
    if not cur_tokens:
        return []
    for past in mistakes_in_ctx:
        past_tokens = {w for w in past.lower().split() if len(w) > 3}
        overlap = cur_tokens & past_tokens
        # If meaningful overlap (>1 shared content word) and no reference → miss.
        if len(overlap) >= 2:
            return [_issue(
                "medium", "missed_connection", turn["id"],
                f"Tutor turn {turn['id'][:8]}: current mistake "
                f"'{current_mistake[:50]}' overlaps past key_mistakes "
                f"'{past.strip()[:50]}' but tutor did not reference the "
                "past session",
                "TUTOR_SYSTEM RECURRING MISTAKE directive not strong enough",
                "Strengthen TUTOR_SYSTEM: 'You MUST point out the "
                "connection when the current error matches a past "
                "key_mistakes entry'",
            )]
    return []


def _problem_type_missing_check(session: dict) -> list[dict]:
    """Flag sessions where problem_type is null."""
    if session.get("problem_type"):
        return []
    return [_issue(
        "low", "problem_type_missing", session["id"],
        f"Session {session['id'][:8]} has no problem_type — OCR parse "
        "did not extract it",
        "OCR_PARSE_SYSTEM prompt did not yield a problem_type field, "
        "or the LLM returned null",
        "Check OCR_PARSE_SYSTEM prompt examples and the OCR parse "
        "fallback in app/services/ocr.py",
    )]


def _summary_quality_check(summary: dict) -> list[dict]:
    """Flag session_summaries that are empty, too long, or lack physics
    vocabulary."""
    text = (summary.get("summary") or "").strip()
    if not text:
        return [_issue(
            "high", "summary_quality", summary.get("session_id", "?"),
            "Session summary is empty",
            "_generate_session_summary fallback failed or LLM returned null",
            "Check _generate_session_summary fallback string and the "
            "SESSION_SUMMARY_SYSTEM prompt",
        )]
    if len(text) > 300:
        return [_issue(
            "medium", "summary_quality", summary.get("session_id", "?"),
            f"Session summary is {len(text)} chars (max 300) — too verbose",
            "SESSION_SUMMARY_SYSTEM prompt allows overly long summaries",
            "Add 'keep the summary to 1-2 sentences, under 200 chars' "
            "to SESSION_SUMMARY_SYSTEM",
        )]
    outcome = summary.get("outcome")
    if outcome not in ("solved", "revealed", "abandoned"):
        return [_issue(
            "medium", "summary_quality", summary.get("session_id", "?"),
            f"Session summary has invalid outcome: '{outcome}'",
            "outcome field not constrained to solved/revealed/abandoned",
            "Validate outcome in _generate_session_summary before storing",
        )]
    # Check for at least one physics-relevant word (>4 chars, not stopword).
    words = [w for w in text.lower().split()
             if len(w) > 4 and w.strip(".,;:!?") not in _STOPWORDS]
    if not words:
        return [_issue(
            "medium", "summary_quality", summary.get("session_id", "?"),
            f"Session summary lacks physics vocabulary: '{text[:60]}'",
            "SESSION_SUMMARY_SYSTEM prompt not guiding content",
            "Add 'include the physics concept and problem type in the "
            "summary' to SESSION_SUMMARY_SYSTEM",
        )]
    return []


def _context_not_injected_check(
    session: dict, turns: list[dict], summaries: list[dict]
) -> list[dict]:
    """Flag sessions where the student has a prior summary with concept
    overlap but student_context was not injected on the first reply."""
    student_id = session.get("student_id")
    if not student_id:
        return []
    concepts = session.get("concepts")
    if not concepts:
        return []
    # Parse concepts if stored as JSON string.
    if isinstance(concepts, str):
        try:
            concepts = json.loads(concepts)
        except (json.JSONDecodeError, TypeError):
            concepts = []
    concept_set = set(concepts)
    # Find prior summaries for this student (exclude this session).
    prior = [
        s for s in summaries
        if s.get("student_id") == student_id
        and s.get("session_id") != session["id"]
    ]
    if not prior:
        return []
    # Check if any prior summary shares at least one concept.
    has_overlap = False
    for s in prior:
        s_concepts = s.get("concepts")
        if isinstance(s_concepts, str):
            try:
                s_concepts = json.loads(s_concepts)
            except (json.JSONDecodeError, TypeError):
                s_concepts = []
        if concept_set & set(s_concepts or []):
            has_overlap = True
            break
    if not has_overlap:
        return []
    # Check the first non-opening tutor turn (loop_index >= 1) for
    # student_context in metadata.
    for t in turns:
        if t["role"] != "tutor" or t["loop_index"] == 0:
            continue
        meta = json.loads(t["metadata"]) if t["metadata"] else {}
        if _student_context(meta) is not None:
            return []  # Context was injected — no issue.
        # Found a reply turn with no context despite overlap.
        return [_issue(
            "medium", "context_not_injected", t["id"],
            f"Session {session['id'][:8]}: student has prior summaries "
            "with concept overlap but student_context is null on "
            f"turn {t['id'][:8]}",
            "find_related_summaries returned empty or "
            "build_student_context failed silently",
            "Check find_related_summaries concept-overlap logic and "
            "build_student_context error handling",
        )]
    return []


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
TURN_CHECKS = [
    _empty_response_check,
    _json_parse_failure_check,
    _false_positive_affirmation_check,
    _classification_missing_check,
    _answer_leak_check,
    _hallucinated_reference_check,
    _missed_connection_check,
]


def main() -> None:
    if not DB_PATH.exists():
        print(json.dumps({"error": f"DB not found: {DB_PATH}"}))
        sys.exit(2)

    args = sys.argv[1:]
    session_filter = None if not args or args == ["all"] else args

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    sessions = conn.execute("SELECT * FROM sessions ORDER BY created_at").fetchall()
    all_session_ids = [r["id"] for r in sessions]
    if session_filter is not None:
        all_session_ids = [s for s in all_session_ids if s in session_filter]

    # Load session_summaries once (for cross-session checks).
    try:
        summaries = [dict(r) for r in conn.execute(
            "SELECT * FROM session_summaries"
        ).fetchall()]
    except sqlite3.OperationalError:
        summaries = []  # Table doesn't exist yet (old DB).

    all_issues: list[dict] = []

    for srow in sessions:
        sid = srow["id"]
        if sid not in all_session_ids:
            continue
        session = dict(srow)

        # Session-level check: problem_type missing.
        all_issues.extend(_problem_type_missing_check(session))

        turns = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at",
            (sid,),
        ).fetchall()
        turn_list = [dict(t) for t in turns]

        for turn in turn_list:
            for check in TURN_CHECKS:
                all_issues.extend(check(turn))

        # Cross-session check: context not injected.
        all_issues.extend(_context_not_injected_check(session, turn_list, summaries))

    # Summary-level checks.
    for s in summaries:
        if session_filter is not None and s.get("session_id") not in all_session_ids:
            # Still check summary quality even for non-filtered sessions —
            # summaries are global. But if a filter is active, only check
            # summaries for the filtered sessions.
            continue
        all_issues.extend(_summary_quality_check(s))

    conn.close()

    report = {
        "sessions_reviewed": len(all_session_ids),
        "issues": all_issues,
    }
    print(json.dumps(report, indent=2))
    sys.exit(1 if all_issues else 0)


if __name__ == "__main__":
    main()
