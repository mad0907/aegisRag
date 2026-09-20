# Solution Architecture

See the [README](../README.md) for the six-plane diagram and quick-reference mermaid charts. This
document is the prose walkthrough of the same system, with the actual module each piece lives in.

## The six planes, and what's really in each one

| Plane | Components | Code |
|---|---|---|
| Experience | OpenWebUI (docker-compose, unmodified) | — |
| API / Security | FastAPI, OpenAI-compatible `/v1/chat/completions`, guardrail checkpoints | `api/main.py`, `guardrails/guardrails.py` |
| AI Runtime | 5 CrewAI agents, orchestrated by an explicit Python loop | `agents/definitions.py`, `agents/orchestrator.py` |
| Knowledge | Docling ingestion, PostgreSQL/PGVector | `ingestion/pipeline.py`, `database/schema.sql` |
| Trust & Governance | Guardrails, human review queue, hash-chain audit log | `guardrails/`, `audit/hashchain.py` |
| Observability & Learning | Phoenix tracing, RAGAs eval, feedback capture | `observability/tracing.py`, `evaluation/run.py` |

## Request lifecycle (a real `/v1/chat/completions` call)

1. **Guardrail — input.** `guardrails.check_input()`: empty/oversized query rejected outright;
   a prompt-injection pattern match is flagged (not blocked — see ADR-007) and travels with the
   request for later inspection.
2. **Plan.** The Query Planner agent classifies intent and decides whether retrieval is even
   needed (a "what's 2+2" question short-circuits before touching the database).
3. **Retrieve + validate, bounded.** `HybridPGRetriever.retrieve()` (PGVector cosine + Postgres
   full-text, reciprocal-rank fused) feeds the Evidence Validator agent. Insufficient evidence
   reformulates the query and retries, up to `agent_policy.yaml`'s `max_iterations`.
4. **Recovery, if still insufficient.** Two different retrieval strategies — lexical-only,
   relaxed hybrid — are tried before giving up (ADR-008). If still nothing usable: the pipeline
   returns a retrieval-only degrade (real passages, no synthesis) or abstains outright if truly
   nothing was retrieved.
5. **Synthesize.** The Synthesis agent writes from approved evidence only, citing inline.
6. **Verify & score.** The Citation & Quality agent checks every claim against the evidence and
   produces a confidence score.
7. **Human-in-the-loop.** `guardrails/human_review.assess_risk()` scores risk from the confidence,
   the evidence verdict, and a configurable keyword list. HIGH risk withholds the answer and
   queues it; MEDIUM logs it to the queue but still returns it; LOW passes straight through.
8. **Guardrail — output**, then the confidence-tiered response: `>= auto_threshold` → answered,
   `qualify_threshold..auto_threshold` → answered with a caveat appended, `< qualify_threshold` →
   abstain.

Every step in 2, 3, 5, 6 writes to the hash-chain audit log (agent, model, prompt version,
policy version, input/output hashes) and is independently visible as a span in Phoenix.

## Data model

`documents` → `chunks` (1:N, `ON DELETE CASCADE`), plus `audit_log`, `review_queue` and
`feedback` as independent tables. Full DDL: `src/aegisrag/database/schema.sql`. Rationale for
each design choice (why PGVector, why this chunking, why this hash scheme) is in
`doc/architecture/ADR-*.md`.

## What this document does not cover

Cloud deployment topology is deliberately a separate document —
[`12-cloud-reference-architecture.md`](./12-cloud-reference-architecture.md) — because nothing in
it has been deployed or tested; keeping it separate from this (implemented, verified) architecture
keeps the two from being confused with each other.
