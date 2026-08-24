# AI Tutor — Guided Discovery for High-School Physics

An AI tutor that helps high-school students solve **printed physics problems** through
**guided discovery**: instead of giving answers, it asks where the student is stuck,
diagnoses whether the issue is a *knowledge gap* or a *misapplication*, and provides
progressively deeper hints. After up to 3 hint loops, the tutor offers the choice to
continue or reveal the full solution — preventing frustration while preserving learning
integrity.

## Scope (MVP)

- **Subject:** physics only
- **Input:** printed text problems (no handwriting)
- **Storage:** Supabase (PostgreSQL + pgvector)
- **LLM provider:** skainet (OpenAI-compatible gateway)

## Architecture

```
Image upload
  └─► [olmocr-7B-faithful]  →  problem text + formulas + concepts
        └─► [GLM-5.2]  stuck-point diagnosis (gap | misapplication | on_track)
              └─► [GLM-5.2]  hint generator  (loop ≤ 3, progressive depth)
                    └─► after loop 3: continue OR reveal full solution
                          └─► Supabase  (sessions, turns, profiles, embeddings)
```

### Models (served by skainet)

| Role | Model ID |
|---|---|
| OCR (image → text) | `tngtech/olmocr-7B-faithful` |
| Dialogue engine (diagnosis, hints, solution) | `zai-org/GLM-5.2` |
| Fast/cheap fallback for simple turns | `deepseek-ai/DeepSeek-V4-Flash` |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in SKAINET_API_KEY + Supabase creds
uvicorn app.main:app --reload
```

## API (summary)

- `POST /sessions` — upload problem image, start a session
- `POST /sessions/{id}/reply` — student replies; tutor responds with next hint
- `GET /sessions/{id}` — full session transcript + metrics
- `POST /sessions/{id}/reveal` — reveal full solution
- `GET /students/{id}/profile` — knowledge profile

See `app/api/routes.py` for details.

## Configuration

All runtime config lives in `.env` (see `.env.example`). Loaded via
`pydantic-settings` in `app/core/config.py`.
