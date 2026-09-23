"""Semantic answer cache. A new query's embedding is compared against every past query's
embedding (pgvector cosine similarity); a close-enough match returns that cached answer directly
— skipping retrieval and every agent LLM call entirely. This is the biggest possible latency win
for a repeat or paraphrased question, precisely because it does the least work: one embedding
call and one indexed vector lookup, nothing else.

Only ever populated with genuinely-answered results (`answered` / `answered_with_caveat`) — an
abstain, a pending-human-review, or a degraded response is never cached, so a cache hit can never
serve a non-answer or an answer that was withheld for review.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from llama_index.embeddings.ollama import OllamaEmbedding
from pgvector import Vector
from pgvector.psycopg import register_vector

from aegisrag.config.settings import get_settings
from aegisrag.database.db import get_connection

_CACHEABLE_STATUSES = ("answered", "answered_with_caveat")


@dataclass
class CachedAnswer:
    answer: str
    confidence: float
    status: str
    citations: list[dict]
    source_trace_id: str
    similarity: float
    cache_id: str


@dataclass
class LookupResult:
    """`hit` is the cached answer only when the best match cleared `threshold`. `similarity` is
    the best candidate's score either way (None only if the cache is empty) -- callers that want
    to show *why* a lookup missed (e.g. "0.71 found, needed 0.80") need this even on a miss,
    which a plain Optional[CachedAnswer] return can't carry. `query_embedding` is this query's own
    embedding, computed once here -- `store()` accepts it back so a cache-miss turn never embeds
    the same question text twice (once to look it up, once again to save it)."""
    hit: CachedAnswer | None
    similarity: float | None
    query_embedding: Vector


class AnswerCache:
    def __init__(self):
        settings = get_settings()
        self._embedder = OllamaEmbedding(
            model_name=settings.ollama_embed_model, base_url=settings.ollama_base_url
        )

    def lookup(self, query: str, threshold: float) -> LookupResult:
        query_vector = Vector(self._embedder.get_text_embedding(query))
        with get_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT cache_id, answer, confidence, status, citations, source_trace_id,
                           1 - (query_embedding <=> %s) AS similarity
                    FROM answer_cache
                    ORDER BY query_embedding <=> %s
                    LIMIT 1
                    """,
                    (query_vector, query_vector),
                )
                row = cur.fetchone()
        if not row:
            return LookupResult(hit=None, similarity=None, query_embedding=query_vector)

        cache_id, answer, confidence, status, citations, source_trace_id, similarity = row
        similarity = float(similarity)
        if similarity < threshold:
            return LookupResult(hit=None, similarity=similarity, query_embedding=query_vector)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE answer_cache SET hit_count = hit_count + 1, last_hit_at = now() WHERE cache_id = %s",
                    (cache_id,),
                )
            conn.commit()

        return LookupResult(
            hit=CachedAnswer(
                answer=answer, confidence=confidence, status=status, citations=citations,
                source_trace_id=source_trace_id, similarity=similarity, cache_id=str(cache_id),
            ),
            similarity=similarity,
            query_embedding=query_vector,
        )

    def store(
        self, query: str, answer: str, confidence: float, status: str,
        citations: list[dict], source_trace_id: str, query_embedding: Vector | None = None,
    ) -> None:
        """`query_embedding`: pass the `LookupResult.query_embedding` from this same turn's
        `lookup()` call to skip re-embedding identical text. Only recomputed when None, so
        `store()` still works standalone (e.g. a future backfill/import path with no prior
        lookup)."""
        if status not in _CACHEABLE_STATUSES:
            return
        query_vector = query_embedding if query_embedding is not None else Vector(
            self._embedder.get_text_embedding(query)
        )
        with get_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO answer_cache
                        (query, query_embedding, answer, confidence, status, citations, source_trace_id)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (query, query_vector, answer, confidence, status,
                     json.dumps(citations), source_trace_id),
                )
            conn.commit()
