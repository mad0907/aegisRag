"""Ingestion: Docling parse -> structural+semantic chunk -> Ollama embed -> PGVector.

Checksum-based dedupe: an unchanged file is a no-op on re-run (§3 of the design doc).
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Docling checks HuggingFace for the "latest revision" of its models before using its local
# cache, even when the models are already cached — that metadata call has been observed to
# hang for 30+ minutes on a flaky connection. Models are pinned by the pyproject/docling version
# anyway, so skip the network check and use the local cache directly.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from llama_index.embeddings.ollama import OllamaEmbedding
from pgvector.psycopg import register_vector

from aegisrag.config.settings import get_settings
from aegisrag.database.db import get_connection

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    source_file: str
    document_id: str | None
    chunks_written: int
    skipped: bool
    reason: str = ""


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _section_path(headings: list[str] | None) -> str | None:
    if not headings:
        return None
    return " > ".join(headings)


def _page_no(meta) -> int | None:
    for item in getattr(meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []) or []:
            return prov.page_no
    return None


def ingest_file(path: Path, embedder: OllamaEmbedding | None = None) -> IngestResult:
    settings = get_settings()
    embedder = embedder or OllamaEmbedding(
        model_name=settings.ollama_embed_model, base_url=settings.ollama_base_url
    )
    checksum = _checksum(path)

    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT document_id FROM documents WHERE checksum = %s", (checksum,))
            existing = cur.fetchone()
            if existing:
                return IngestResult(
                    source_file=path.name,
                    document_id=str(existing[0]),
                    chunks_written=0,
                    skipped=True,
                    reason="unchanged (checksum match)",
                )

        logger.info("Converting %s with Docling...", path.name)
        result = DocumentConverter().convert(str(path))
        doc = result.document
        chunker = HybridChunker()
        chunks = list(chunker.chunk(doc))

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (source_file, checksum, page_count, status)
                VALUES (%s, %s, %s, 'active')
                RETURNING document_id
                """,
                (path.name, checksum, doc.num_pages()),
            )
            document_id = cur.fetchone()[0]

            written = 0
            for idx, chunk in enumerate(chunks):
                text = chunker.contextualize(chunk)
                vector = embedder.get_text_embedding(text)
                cur.execute(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, section_path, page_no, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        document_id,
                        idx,
                        _section_path(chunk.meta.headings),
                        _page_no(chunk.meta),
                        chunk.text,
                        vector,
                    ),
                )
                written += 1
        conn.commit()

    logger.info("Ingested %s: %d chunks", path.name, written)
    return IngestResult(
        source_file=path.name, document_id=str(document_id), chunks_written=written, skipped=False
    )


def ingest_directory(directory: Path) -> list[IngestResult]:
    settings = get_settings()
    embedder = OllamaEmbedding(
        model_name=settings.ollama_embed_model, base_url=settings.ollama_base_url
    )
    results = []
    for path in sorted(directory.glob("*.pdf")):
        results.append(ingest_file(path, embedder=embedder))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kb_dir = Path(__file__).resolve().parents[3] / "data" / "knowledge_base"
    for r in ingest_directory(kb_dir):
        status = "SKIPPED" if r.skipped else "OK"
        print(f"[{status}] {r.source_file}: {r.chunks_written} chunks ({r.reason})")
