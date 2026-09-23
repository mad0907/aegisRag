"""Hybrid retrieval: PGVector cosine similarity + Postgres full-text search, fused by
reciprocal-rank fusion. Implemented as a LlamaIndex BaseRetriever so it composes with
LlamaIndex query engines and CrewAI tools alike.

Also exposes vector-only and lexical-only retrieval directly — used by the agent-level
recovery chain (§18 of the design doc) as alternate strategies when the hybrid pass comes back
with insufficient evidence, rather than just repeating the same search.
"""
from __future__ import annotations

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from pgvector import Vector
from pgvector.psycopg import register_vector

from aegisrag.config.settings import get_settings
from aegisrag.database.db import get_connection


def _reciprocal_rank_fusion(
    vector_ids: list[str], lexical_ids: list[str], k: int = 60
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(vector_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(lexical_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _row_to_node(row, score: float, vector_similarity: float | None = None) -> NodeWithScore:
    chunk_id, document_id, source_file, section_path, page_no, content, status = row[:7]
    metadata = {
        "document_id": str(document_id),
        "source_file": source_file,
        "section_path": section_path,
        "page_no": page_no,
        "status": status,
    }
    if vector_similarity is not None:
        # Raw cosine similarity, distinct from `score` (which may be an RRF-fused rank score) —
        # this is the number the smart-bypass evidence check (§ agent_policy.yaml) reads to
        # decide whether an LLM call is even needed. Only present when a vector search ran.
        metadata["vector_similarity"] = vector_similarity
    node = TextNode(text=content, id_=str(chunk_id), metadata=metadata)
    return NodeWithScore(node=node, score=score)


class HybridPGRetriever(BaseRetriever):
    def __init__(self, top_k: int = 6, status: str = "active"):
        self._settings = get_settings()
        self._embedder = OllamaEmbedding(
            model_name=self._settings.ollama_embed_model, base_url=self._settings.ollama_base_url
        )
        self._top_k = top_k
        self._status = status
        super().__init__()

    def _vector_rows(self, cur, query_vector, limit: int):
        cur.execute(
            """
            SELECT c.chunk_id, c.document_id, d.source_file, c.section_path, c.page_no,
                   c.content, d.status,
                   1 - (c.embedding <=> %s) AS similarity
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE d.status = %s
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            (query_vector, self._status, query_vector, limit),
        )
        return cur.fetchall()

    def _lexical_rows(self, cur, query_str: str, limit: int):
        cur.execute(
            """
            SELECT c.chunk_id, c.document_id, d.source_file, c.section_path, c.page_no,
                   c.content, d.status,
                   ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', %s)) AS rank
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id
            WHERE d.status = %s
              AND to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query_str, self._status, query_str, limit),
        )
        return cur.fetchall()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_str = query_bundle.query_str
        query_vector = Vector(self._embedder.get_text_embedding(query_str))

        with get_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                vector_rows = self._vector_rows(cur, query_vector, self._top_k * 3)
                lexical_rows = self._lexical_rows(cur, query_str, self._top_k * 3)

        by_id = {str(r[0]): r for r in vector_rows}
        for r in lexical_rows:
            by_id.setdefault(str(r[0]), r)
        vector_similarity_by_id = {str(r[0]): float(r[7]) for r in vector_rows}

        vector_ids = [str(r[0]) for r in vector_rows]
        lexical_ids = [str(r[0]) for r in lexical_rows]
        fused = _reciprocal_rank_fusion(vector_ids, lexical_ids)
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[: self._top_k]

        return [
            _row_to_node(by_id[cid], fused[cid], vector_similarity_by_id.get(cid))
            for cid in ranked_ids
        ]

    def retrieve_lexical_only(self, query_str: str) -> list[NodeWithScore]:
        """Recovery strategy: drop the embedding step entirely — useful when the vector pass
        is dominated by semantically-similar-but-wrong passages and a plain keyword match
        would actually find the right section."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                rows = self._lexical_rows(cur, query_str, self._top_k)
        return [_row_to_node(r, float(r[7])) for r in rows]

    def retrieve_relaxed(self, query_str: str) -> list[NodeWithScore]:
        """Recovery strategy: hybrid search with a larger candidate pool and no score floor —
        useful when the right passage exists but ranked just outside the normal top_k."""
        query_vector = Vector(self._embedder.get_text_embedding(query_str))
        with get_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                vector_rows = self._vector_rows(cur, query_vector, self._top_k * 6)
                lexical_rows = self._lexical_rows(cur, query_str, self._top_k * 6)

        by_id = {str(r[0]): r for r in vector_rows}
        for r in lexical_rows:
            by_id.setdefault(str(r[0]), r)
        vector_similarity_by_id = {str(r[0]): float(r[7]) for r in vector_rows}
        vector_ids = [str(r[0]) for r in vector_rows]
        lexical_ids = [str(r[0]) for r in lexical_rows]
        fused = _reciprocal_rank_fusion(vector_ids, lexical_ids)
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[: self._top_k * 2]
        return [
            _row_to_node(by_id[cid], fused[cid], vector_similarity_by_id.get(cid))
            for cid in ranked_ids
        ]
