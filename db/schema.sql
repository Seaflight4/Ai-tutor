-- AI Tutor schema: PostgreSQL
-- Run on Supabase via the SQL editor.

-- ---------------------------------------------------------------------------
-- students
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_ref    TEXT UNIQUE,                       -- e.g. auth user id / cookie
    profile_summary TEXT,                              -- LLM-generated summary
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- sessions  (one problem per session, MVP)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id        UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    subject           TEXT NOT NULL DEFAULT 'physics',
    problem_text      TEXT NOT NULL,
    problem_image_url TEXT,
    concepts          TEXT[] NOT NULL DEFAULT '{}',    -- physics concepts tagged
    problem_type      TEXT,                            -- e.g. "elastic collision with angle"
    ocr_raw           JSONB,                           -- raw OCR payload
    loop_count        INT  NOT NULL DEFAULT 0,         -- hint loops used
    status            TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active','revealed','solved')),
    resolved          BOOLEAN NOT NULL DEFAULT FALSE,  -- denormalized: status != 'active'
    resolution_type   TEXT,                            -- solved_with_hints | revealed | abandoned
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sessions_student_idx ON sessions(student_id, created_at DESC);

-- Migration for existing deployments: add the status column if missing and
-- backfill it from the legacy resolved/resolution_type fields.
--   ALTER TABLE sessions ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'
--     CHECK (status IN ('active','revealed','solved'));
--   UPDATE sessions SET status = 'revealed'
--     WHERE resolved AND resolution_type = 'revealed';
--   UPDATE sessions SET status = 'solved'
--     WHERE resolved AND resolution_type = 'solved_with_hints';

-- ---------------------------------------------------------------------------
-- turns  (the dialogue: tutor <-> student)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS turns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('tutor','student','system')),
    content         TEXT NOT NULL,
    loop_index      INT  NOT NULL DEFAULT 0,            -- which hint loop (0 = opening)
    hint_level      SMALLINT,                          -- 1=conceptual, 2=scaffolded, 3=near-worked
    classification   TEXT,                             -- knowledge_gap | misapplication | on_track
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS turns_session_idx ON turns(session_id, created_at);

-- ---------------------------------------------------------------------------
-- knowledge_profiles  (per-student, per-concept mastery + embedding)
--   embedding is built from concept + recent turn context for similarity
--   search during personalization.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_profiles (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id     UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    concept        TEXT NOT NULL,
    mastery_score  REAL NOT NULL DEFAULT 0.5 CHECK (mastery_score BETWEEN 0 AND 1),
    attempts       INT  NOT NULL DEFAULT 0,
    last_session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (student_id, concept)
);

CREATE INDEX IF NOT EXISTS knowledge_profiles_student_idx
    ON knowledge_profiles(student_id);

-- ---------------------------------------------------------------------------
-- session_summaries  (compressed learning record for cross-session reference)
--   One row per terminated session. Written at reveal/solved/abandon time.
--   Queried by find_related_summaries() to inject past-session context into
--   the tutor prompt during active sessions.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_summaries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id      UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    problem_text    TEXT NOT NULL,
    concepts        TEXT[] NOT NULL DEFAULT '{}',
    problem_type    TEXT,                           -- e.g. "elastic collision with angle"
    outcome         TEXT NOT NULL,                  -- "solved" | "revealed" | "abandoned"
    target_concept  TEXT,                           -- the concept the student struggled with
    summary         TEXT NOT NULL,                  -- LLM-generated 1-2 sentence summary
    key_mistakes    TEXT,                           -- compressed: "confused impulse with force; sign error"
    mastery_after   REAL,                           -- mastery_score after this session (or null)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id)
);

CREATE INDEX IF NOT EXISTS session_summaries_student_idx
    ON session_summaries(student_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- reference_chunks  (curated physics reference corpus for source grounding)
--   embedding is built from chunk_text for cosine retrieval at hint time.
--   concepts pre-filters candidates using the problem's tagged concepts.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reference_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id    TEXT NOT NULL,                      -- e.g. "openstax-college-physics"
    source_title TEXT NOT NULL,                      -- "OpenStax College Physics"
    source_url   TEXT NOT NULL,                      -- canonical URL for citation
    chapter      TEXT,                               -- "Ch. 5: Work and Energy"
    heading      TEXT,                               -- section heading
    chunk_text   TEXT NOT NULL,                      -- the actual passage
    concepts     TEXT[] NOT NULL DEFAULT '{}'        -- concept tags from the taxonomy
);

CREATE INDEX IF NOT EXISTS reference_chunks_concepts_idx
    ON reference_chunks USING gin (concepts);

-- ---------------------------------------------------------------------------
-- updated_at triggers
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS students_touch ON students;
CREATE TRIGGER students_touch BEFORE UPDATE ON students
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS sessions_touch ON sessions;
CREATE TRIGGER sessions_touch BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS knowledge_profiles_touch ON knowledge_profiles;
CREATE TRIGGER knowledge_profiles_touch BEFORE UPDATE ON knowledge_profiles
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
