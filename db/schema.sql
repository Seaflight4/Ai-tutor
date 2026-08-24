-- AI Tutor schema: PostgreSQL + pgvector
-- Run on Supabase via the SQL editor. Enable the vector extension first:
--   CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS vector;

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
    ocr_raw           JSONB,                           -- raw OCR payload
    loop_count        INT  NOT NULL DEFAULT 0,         -- hint loops used
    resolved          BOOLEAN NOT NULL DEFAULT FALSE,
    resolution_type   TEXT,                            -- solved_with_hints | revealed | abandoned
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sessions_student_idx ON sessions(student_id, created_at DESC);

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
    embedding      vector(1536),                       -- text-embedding-3-small dim
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (student_id, concept)
);

CREATE INDEX IF NOT EXISTS knowledge_profiles_student_idx
    ON knowledge_profiles(student_id);

-- vector index for personalization (IVFFlat; fine for MVP scale)
CREATE INDEX IF NOT EXISTS knowledge_profiles_embedding_idx
    ON knowledge_profiles USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

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
