"""Hybrid retrieval: PGVector cosine similarity + Postgres full-text search, fused by
reciprocal-rank fusion. Implemented as a LlamaIndex BaseRetriever so it composes with
LlamaIndex query engines and CrewAI tools alike.
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


class HybridPGRetriever(BaseRetriever):
    def __init__(self, top_k: int = 6, status: str = "active"):
        self._settings = get_settings()
        self._embedder = OllamaEmbedding(
            model_name=self._settings.ollama_embed_model, base_url=self._settings.ollama_base_url
        )
        self._top_k = top_k
        self._status = status
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_str = query_bundle.query_str
        query_vector = Vector(self._embedder.get_text_embedding(query_str))

        with get_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # Vector similarity (cosine distance, lower = closer)
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
                    (query_vector, self._status, query_vector, self._top_k * 3),
                )
                vector_rows = cur.fetchall()

                # Lexical full-text search
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
                    (query_str, self._status, query_str, self._top_k * 3),
                )
                lexical_rows = cur.fetchall()

        by_id = {str(r[0]): r for r in vector_rows}
        for r in lexical_rows:
            by_id.setdefault(str(r[0]), r)

        vector_ids = [str(r[0]) for r in vector_rows]
        lexical_ids = [str(r[0]) for r in lexical_rows]
        fused = _reciprocal_rank_fusion(vector_ids, lexical_ids)
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[: self._top_k]

        results = []
        for cid in ranked_ids:
            chunk_id, document_id, source_file, section_path, page_no, content, status, _score = by_id[cid]
            node = TextNode(
                text=content,
                id_=cid,
                metadata={
                    "document_id": str(document_id),
                    "source_file": source_file,
                    "section_path": section_path,
                    "page_no": page_no,
                    "status": status,
                },
            )
            results.append(NodeWithScore(node=node, score=fused[cid]))
        return results
