# ADR-001: PGVector inside PostgreSQL, not a dedicated vector database

## Status
Accepted — implemented (`src/aegisrag/database/schema.sql`, `retrieval/hybrid_retriever.py`).

## Context
The mandatory stack requires "PostgreSQL with PGVector." Separately, a dedicated vector database
(Pinecone, Weaviate, Qdrant, Milvus) would give higher-throughput ANN search at very large scale
and native hybrid-search features.

## Decision
Use PGVector inside the same Postgres instance that holds document/chunk metadata, and implement
hybrid search ourselves (vector cosine + Postgres full-text `tsvector`, fused by reciprocal-rank
fusion) rather than reaching for a second datastore.

## Alternatives considered
- **Qdrant/Weaviate + Postgres for metadata**: two systems to run, back up and keep consistent
  for a corpus that is, at assessment scale, a handful of PDFs. The operational cost isn't
  justified until the corpus and query volume are large enough that PGVector's HNSW index stops
  being fast enough — which is a measurable, later decision, not a day-one one.
- **A managed vector service (e.g. a cloud vendor's)**: rejected for the same reason as any cloud
  dependency in this assessment — no cloud environment was provisioned; see
  `doc/12-cloud-reference-architecture.md` for where this would sit if the platform moved to GCP.

## Consequences
- One database, one connection pool, one backup story, transactional consistency between a
  chunk's metadata and its embedding (impossible to have a chunk row without its vector, or
  vice versa — they're the same INSERT).
- Joining vector search results with relational metadata (`document.status = 'active'`, page
  numbers, section headings) is a plain SQL `JOIN`, not a cross-system round trip.
- Ceiling: PGVector's HNSW index is very good but not the fastest possible ANN implementation at
  tens-of-millions-of-vectors scale. Revisit if the corpus grows past what a single Postgres
  instance comfortably serves.
