-- AegisRAG schema. Idempotent: safe to re-run.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file       TEXT NOT NULL,
    document_version  TEXT NOT NULL DEFAULT '1.0',
    effective_date    DATE,
    status             TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded')),
    checksum           TEXT NOT NULL UNIQUE,
    page_count         INTEGER,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- nomic-embed-text produces 768-dim vectors.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id    UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index    INTEGER NOT NULL,
    section_path   TEXT,
    page_no        INTEGER,
    content         TEXT NOT NULL,
    embedding       vector(768),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- Lexical (BM25-ish) search support for hybrid retrieval.
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
    ON chunks USING gin (to_tsvector('english', content));

-- Hash-chain audit log (§19 of the design doc).
CREATE TABLE IF NOT EXISTS audit_log (
    event_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id             TEXT,
    agent                 TEXT NOT NULL,
    action                 TEXT NOT NULL,
    document_ids          UUID[],
    model                  TEXT,
    prompt_version        TEXT,
    policy_version        TEXT,
    input_hash             TEXT,
    output_hash            TEXT,
    previous_event_hash   TEXT,
    event_hash             TEXT NOT NULL,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent add for databases initialized before policy_version existed.
ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS policy_version TEXT;

CREATE INDEX IF NOT EXISTS audit_log_created_at_idx ON audit_log (created_at);

-- Human-in-the-loop review queue (§17 of the design doc). A HIGH-risk answer is withheld from
-- the caller and lands here instead; a MEDIUM-risk answer is returned but still logged here as
-- "flagged" for after-the-fact review. Nothing here auto-decides anything — it's the workflow a
-- human reviewer would use, via the /review-queue API.
CREATE TABLE IF NOT EXISTS review_queue (
    review_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id         TEXT NOT NULL,
    query             TEXT NOT NULL,
    draft_answer      TEXT NOT NULL,
    confidence         DOUBLE PRECISION,
    risk_level         TEXT NOT NULL CHECK (risk_level IN ('medium', 'high')),
    risk_reasons       TEXT[],
    status              TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewer_note       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS review_queue_status_idx ON review_queue (status);

-- User feedback (§20 of the design doc — RAG error taxonomy classification happens in Python).
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id         TEXT,
    query             TEXT NOT NULL,
    answer            TEXT NOT NULL,
    rating            TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    error_category   TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
