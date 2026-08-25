# AI Tutor — Workflow

```mermaid
flowchart TD
    classDef user fill:#e1f5ff,stroke:#0288d1,color:#01579b
    classDef api fill:#fff4e1,stroke:#f57c00,color:#e65100
    classDef llm fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c
    classDef persist fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
    classDef ret fill:#fce4ec,stroke:#c2185b,color:#880e4f
    classDef prof fill:#e0f7fa,stroke:#00838f,color:#006064
    classDef term fill:#ffebee,stroke:#c62828,color:#b71c1c

    U([Student])

    %% ---- Session start ----
    U -->|"POST /api/sessions (image upload)"| R1[/"routes.py<br/>validate + JPEG to PNG"/]
    R1 -->|"no image / too large"| E1[/"415 / 422"/]
    R1 -->|"image OK"| O1["session.start_session"]

    subgraph START [Session Start]
        O1 --> IMG["_maybe_upload_image<br/>Supabase Storage if configured<br/>else local None"]
        O1 --> OCR["ocr.run_ocr"]
        OCR -->|"vision call"| OCRM["llm.ocr_image<br/>olmocr-7B-faithful<br/>to markdown"]
        OCR -->|"chat call"| PARSE["llm.chat_json<br/>GLM-5.2 parses<br/>problem_text / concepts / formula / diagrams"]
        OCR --> CLEAN["_clean_problem_text<br/>strip fences and blanks"]
        OCR -->|"OCRResult"| SESS["supabase.create_session<br/>(problem_text, concepts, image_url)"]
        SESS --> OPEN["hints.generate_opening<br/>GLM-5.2 chat_text temp 0.6"]
        OPEN --> ADD1["supabase.add_turn<br/>opening turn"]
    end
    ADD1 --> REPLY_OUT["TutorReply / SessionOut"]
    REPLY_OUT --> U

    %% ---- Reply loop ----
    U -->|"POST /api/sessions/:id/reply"| R2[/"routes.py<br/>reply_text"/]
    R2 -->|"unknown id"| E2[/"404"/]
    R2 --> O2["session.reply"]

    subgraph LOOP [Reply Loop]
        O2 --> GETSESS["supabase.get_session"]
        GETSESS --> GUARD{"solved or<br/>revealed?"}
        GUARD -->|"yes"| E3[/"409 conflict"/]
        GUARD -->|"no"| SOLCHK{"_is_solution_request?"}

        SOLCHK -->|"yes (fast path)"| DO_REVEAL["_do_reveal<br/>loop_index=999"]
        SOLCHK -->|"no"| DIAG["tutor.diagnose_and_respond"]

        DIAG --> GETPROF["profile.weak_concepts_for<br/>read mastery_score < threshold"]
        DIAG --> RET["retrieval.retrieve_for_concepts<br/>concept filter + keyword overlap"]
        RET --> LISTC["supabase.list_reference_chunks_by_concepts"]
        DIAG -->|"merged chat_json<br/>GLM-5.2 temp 0.3"| LLM1["single call returns<br/>classification gap/misapplication/on_track<br/>+ HintOutput question/hint/method_feedback"]

        LLM1 --> RESOLVE{"classification /<br/>answer_status"}
        RESOLVE -->|"answer_check + correct"| SOLVED["_do_solved<br/>loop_count=999<br/>profile.update_profiles"]
        RESOLVE -->|"meta (wants_solution or<br/>loop >= 3 hinted)"| ASK{"wants_solution<br/>or offered?"}
        ASK -->|"yes to reveal"| DO_REVEAL
        ASK -->|"no to continue"| HINTED["_do_hinted<br/>loop_count += 1<br/><= max_hint_loops"]
        RESOLVE -->|"gap / misapplication / on_track"| HINTED

        HINTED --> ADD2["supabase.add_turn<br/>tutor turn"]
        DO_REVEAL --> ADD3["supabase.add_turn<br/>reveal turn"]
        SOLVED --> ADD4["supabase.add_turn<br/>solved turn"]
    end

    DO_REVEAL --> SOL["solution.generate_solution<br/>GLM-5.2 chat_text temp 0.3 max 1500"]
    SOLVED --> PROF["profile.update_profiles<br/>parse mastery_score<br/>llm.embed computed, unused"]
    PROF --> UPS["supabase.upsert_profile<br/>(student_id, concept, mastery, attempts)"]

    ADD2 --> OUT1["TutorReply"]
    ADD3 --> OUT2["RevealOut"]
    ADD4 --> OUT3["TutorReply"]
    OUT1 --> U
    OUT2 --> U
    OUT3 --> U

    %% ---- Other endpoints ----
    U -->|"GET /api/sessions/:id"| R3[/"routes.py<br/>get_session"/]
    R3 --> GT["supabase.get_session + list_turns"]
    GT --> TR[/"SessionOut + TurnOut[]"/]
    TR --> U

    U -->|"POST /api/sessions/:id/reveal"| R4[/"routes.py<br/>reveal"/]
    R4 --> REVF["session.reveal<br/>(force _do_reveal)"]
    REVF --> DO_REVEAL

    U -->|"GET /api/students/:id/profile"| R5[/"routes.py<br/>get_profile"/]
    R5 --> GP["supabase.get_profiles"]
    GP --> PE[/"ProfileEntry[]"/]
    PE --> U

    %% ---- Persistence layer ----
    subgraph STORE [Persistence - app/core]
        direction LR
        SUPB["supabase.py<br/>(Postgres + pgvector<br/>+ Storage)"]
        LOC["local_store.py<br/>(SQLite fallback<br/>when SUPABASE_URL empty)"]
        SUPB -.->|"delegates"| LOC
    end
    STORE -.->|"used by all supabase calls"| START
    STORE -.-> LOOP
    STORE -.-> R3
    STORE -.-> R5

    %% ---- Config / LLM client ----
    subgraph CFG [Config and LLM client]
        direction LR
        SETTINGS["config.get_settings<br/>(lru_cache, .env)"]
        LLMC["llm.py<br/>OpenAI-compatible client<br/>chat_json / chat_text / ocr_image / embed"]
        SETTINGS --> LLMC
    end
    CFG -.-> OCR
    CFG -.-> OPEN
    CFG -.-> LLM1
    CFG -.-> SOL
    CFG -.-> PROF

    %% ---- Reference ingestion (offline) ----
    subgraph INGEST [Offline ingestion - scripts]
        direction LR
        MD["data/reference/*.md<br/>YAML frontmatter + sections"]
        SCR["ingest_references.py"]
        MD --> SCR --> CHUNKS["reference_chunks table"]
    end
    INGEST -.-> LISTC

    %% ---- Class assignments ----
    class U user
    class R1,O1,R2,O2,R3,R4,R5,REVF,DO_REVEAL,SOLCHK,HINTED,SCR api
    class OCR,OCRM,PARSE,CLEAN,OPEN,DIAG,LLM1,SOL,LLMC llm
    class IMG,SESS,ADD1,GETSESS,LISTC,ADD2,ADD3,ADD4,SUPB,LOC,SETTINGS,GT,GP,MD,CHUNKS,UPS persist
    class RET ret
    class GETPROF,SOLVED,PROF prof
    class E1,E2,E3,GUARD,RESOLVE,ASK term
```

## Key design components

| Layer | Component | Role |
|---|---|---|
| API | `app/api/routes.py` | FastAPI router: image upload, reply, reveal, get-session, get-profile |
| Services | `session.py` | Orchestration state machine: start → reply → reveal/solved |
| Services | `ocr.py` | Two-step OCR (olmocr markdown → GLM-5.2 parse) |
| Services | `tutor.py` | Single merged LLM call: diagnose classification + produce `HintOutput` |
| Services | `hints.py` | Opening message + human-readable hint summarization |
| Services | `solution.py` | Full-solution generation (post-loop or explicit request) |
| Services | `retrieval.py` | Concept-filter + keyword-overlap retrieval over reference chunks |
| Services | `profile.py` | Weak-concept detection + post-resolution mastery upsert |
| LLM | `app/core/llm.py` | Async OpenAI-compatible client; CoT stripping + JSON extraction |
| Prompts | `app/prompts/guided_discovery.py` | All prompt templates (OCR, opening, tutor, solution, profile) |
| Config | `app/core/config.py` | pydantic-settings `Settings` (cached) |
| Persistence | `app/core/supabase.py` | Postgres + pgvector + Storage helpers |
| Persistence | `app/core/local_store.py` | SQLite fallback when `SUPABASE_URL` is empty |
| Offline | `scripts/ingest_references.py` | Parse `data/reference/*.md` into `reference_chunks` |
| Frontend | `app/static/index.html` + `app.js` | Minimal SPA; KaTeX rendering |

## Loop / resolution semantics

- **Normal flow:** up to `max_hint_loops` (default 3) progressive hints, each via `tutor.diagnose_and_respond`.
- **Fast paths:**
  - Student types a solution-request phrase → `_do_reveal` immediately (skips LLM diagnose).
  - `classification == answer_check` + student's answer is correct → `_do_solved` (records mastery).
  - LLM returns `wants_solution=True` or loop ≥ 3 hinted → offer reveal.
- **Terminal markers:** `loop_count = 999` and `loop_index = 999` on the reveal/solved turn (sentinel, not a real count).
