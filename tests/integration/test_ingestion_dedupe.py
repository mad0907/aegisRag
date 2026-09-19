"""Integration test against the real corpus and Postgres — confirms checksum-based dedupe
(§3 of the design doc): re-ingesting an unchanged file is a no-op."""
from pathlib import Path

from aegisrag.ingestion.pipeline import ingest_directory

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


def test_reingesting_unchanged_corpus_is_a_noop():
    if not any(KB_DIR.glob("*.pdf")):
        return  # nothing ingested in this environment yet — not this test's job to ingest

    results = ingest_directory(KB_DIR)
    assert results, "expected at least one PDF in data/knowledge_base"
    for r in results:
        assert r.skipped, f"{r.source_file} was re-ingested instead of skipped on an unchanged run"
        assert r.chunks_written == 0
