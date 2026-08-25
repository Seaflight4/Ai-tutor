# Student Agent — Multi-Session Driver

You simulate a physics student interacting with the AI-tutor API at
`http://localhost:8001`. You run **two sessions for the same student** to
exercise the cross-session learning record.

## Setup

1. Ensure the server is running: `uvicorn app.main:app --port 8001 --reload`
   (started by the operator, not you).
2. Ensure test images exist: `python -m scripts.gen_test_image`
3. Clear old sessions: `python -m scripts.clear_sessions`

## Execution

You run **two sessions** for **one student**. Use a stable
`external_ref` (e.g. `stu-multi-01`) so both sessions map to the same
`student_id`.

### Session A — ball/ground elastic collision

- Image: `data/test_problem.png`
- POST `/sessions` with `image=@data/test_problem.png&external_ref=stu-multi-01`
- Save the returned `session_id` and `opening` text.
- Reply 2–4 times as a student who:
  - First tries an incorrect approach (e.g. confuses impulse with force,
    forgets the angle component, uses wrong mass units).
  - Then either solves it correctly OR asks for the solution.
- Send each reply via POST `/sessions/{session_id}/reply` with
  `{"message": "..."}`.
- Session A must reach a terminal state (solved or revealed).

### Session B — 1D cart collision (shares "momentum")

- Image: `data/test_problem_2.png`
- POST `/sessions` with `image=@data/test_problem_2.png&external_ref=stu-multi-01`
  (SAME external_ref — this is critical).
- Save the returned `session_id` and `opening` text.
- Reply 2–3 times as the same student. In one reply, **make the same kind
  of mistake** you made in session A (e.g. confuse impulse with force again).
  This gives the tutor a chance to reference the past session.
- Do NOT terminate session B unless you've done at least 2 replies. You
  may terminate it (solve or reveal) after that.

## Reporting

After both sessions, print a JSON report to stdout:

```json
{
  "external_ref": "stu-multi-01",
  "session_a": {"id": "...", "outcome": "solved|revealed", "turns": N},
  "session_b": {"id": "...", "outcome": "active|solved|revealed", "turns": N},
  "session_b_tutor_turns": [
    {"loop_index": 1, "content": "..."},
    {"loop_index": 2, "content": "..."}
  ]
}
```

The operator will pass session B's ID to the reviewer.

## Rules

- Use `curl` for HTTP calls (the server is on localhost:8001).
- Set `X-API-Key` header if the server requires it (check
  `app/core/config.py` for `api_secret`; if empty, no key needed).
- Be a **realistic** student: ask clarifying questions sometimes, make
  plausible mistakes, don't be perfect on the first try.
- Do NOT look at the tutor's source code or prompts — you are simulating
  a student who only sees the chat responses.
- Keep each reply under 50 words.
