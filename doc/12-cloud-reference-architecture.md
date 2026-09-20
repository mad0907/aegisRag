# Cloud Reference Architecture (GCP) — Recommendation, Not a Deployment

**Nothing in this document has been deployed or tested.** No cloud environment was provisioned
for this assessment; this is the target architecture this codebase is designed to move into,
written so the migration is a known, scoped piece of work rather than a guess made later. GCP is
used as the primary worked example because the governance design (hash-chain audit signing,
future eval/fine-tuning tooling) leans on Cloud KMS and Vertex AI specifically; the same shapes
translate to AWS or Azure with their equivalent services.

## Why this is a document and not a deployment

Building this locally-first and writing the cloud target as a recommendation, rather than
guessing at a cloud deployment without being able to test it, is the more defensible engineering
choice: an untested Terraform module and a set of IAM bindings nobody has actually applied would
be worse than useless — they'd look done while hiding exactly the kind of integration bugs a real
deployment surfaces (wrong security group, a forgotten firewall rule, a service account missing
one IAM role). Everything below is scoped precisely enough to build from, and no more certain
than that.

## Target topology

```
                            Internet
                               |
                               v
                        Cloud Load Balancer
                          (+ Cloud Armor)
                               |
                 +-------------+--------------+
                 v                             v
          Cloud Run: OpenWebUI          Cloud Run: AegisRAG API
          (or GKE, if session          (FastAPI — same container
           affinity is needed)          image as local, env-driven)
                                               |
                        +----------------------+----------------------+
                        v                      v                      v
                Cloud SQL for            Vertex AI (LLM +       Cloud KMS
                PostgreSQL              embeddings, OR a       (audit-log
                + pgvector extension     GKE node pool          signing key)
                (documents, chunks,      running Ollama on
                 audit_log,              GPU, per ADR-004)
                 review_queue)
                        |
                        v
                Cloud Storage (raw PDF corpus,
                versioned, this is the source
                of truth data/knowledge_base/
                moves to)

        Cloud Logging + Cloud Monitoring  <-- structured logs, uptime checks
        Arize Phoenix: self-hosted on a small Cloud Run service or GKE pod
                          (Phoenix itself has no first-party GCP managed offering)
```

## Component-by-component mapping from the local build

| Local (this repo, running) | GCP target | Why |
|---|---|---|
| `docker-compose` Postgres+PGVector | Cloud SQL for PostgreSQL, `pgvector` extension enabled | Managed backups, HA, IAM-based auth instead of a password in `.env` |
| Ollama, native on host | Vertex AI (managed models) **or** GKE GPU node pool running Ollama | Vertex AI removes GPU ops entirely but changes the LLM-provider abstraction target (ADR-004 already isolates this — it's a config change via LiteLLM's provider support, not a code rewrite); self-hosted Ollama on GKE keeps the exact local behavior but re-introduces GPU capacity planning |
| FastAPI, `uvicorn --reload` | Cloud Run (stateless, scales to zero) or GKE if WebSocket/session affinity is ever needed | The API is already stateless (all state is in Postgres) — Cloud Run is the lower-ops choice |
| OpenWebUI container | Cloud Run or GKE, same image | No change to OpenWebUI itself — only where it runs and how it reaches the API (Cloud Run service URL instead of `host.docker.internal`) |
| `.env` file | Secret Manager, injected as env vars at deploy time | Never a file on disk in the image |
| Hash-chain audit log (ADR-006) | Same Postgres table, **plus** Cloud KMS-backed signing of periodic chain checkpoints, **plus** the underlying Cloud SQL storage under an immutable backup/retention policy | Closes the gap ADR-006 states explicitly: hashing alone detects tampering, it doesn't prevent a privileged actor from editing a row — KMS-signed checkpoints + storage-level immutability do |
| `data/knowledge_base/*.pdf`, committed to git | Cloud Storage bucket, versioned | PDFs shouldn't live in git at real scale; the ingestion pipeline's `ingest_directory()` would point at a GCS-mounted or GCS-synced path instead |
| Phoenix, docker-compose | Self-hosted on Cloud Run or a small GKE deployment (no managed Phoenix offering exists) | Same container image; needs a persistent volume or GCS-backed storage for trace history |
| No auth on the API (§09-security-design.md gap) | API Gateway or Cloud Run's built-in IAM auth, tokens minted per caller | Closes the "no API authentication" gap stated plainly in the security doc |
| RAGAs eval, run manually (`make eval`) | Cloud Build / Cloud Scheduler triggered job, results to BigQuery or Cloud Storage, regression check gates a Cloud Deploy promotion | Turns the eval harness (already regression-baseline-aware, see `evaluation/run.py`) into an actual release gate |

## What Vertex AI is for here, specifically

Not used as the LLM provider by default (Ollama stays local-first even in the cloud target, to
keep cost predictable and avoid a hard dependency on a single vendor's model availability) —
Vertex AI's role in this architecture is:

1. **A drop-in alternative LLM/embedding provider** when GPU-hosted Ollama isn't worth the ops
   cost at a given scale — swappable via config (ADR-004), not a rewrite.
2. **Managed evaluation and experiment tracking** for the RAGAs harness's results over time, once
   there's enough eval history to be worth a dashboard instead of local JSON files
   (`evaluation/results/*.json` today).
3. **Where actual model fine-tuning would live**, if the feedback loop's RAG error taxonomy
   (§20 of the design doc) ever produces enough classified examples to justify fine-tuning
   instead of prompt/retrieval tuning. Not needed yet — the corpus and query volume in this
   assessment are far too small to justify it, and reaching for it now would be exactly the kind
   of "AutoML because it sounds advanced" the design doc explicitly argues against.

## Cost shape (directional, not a quote)

The two cost drivers that dominate: GPU time (if self-hosting Ollama on GKE) and Cloud SQL
instance size (driven by corpus size, not query volume, since PGVector's index lives in the
database). Cloud Run's per-request billing means the API and OpenWebUI cost roughly nothing at
this assessment's traffic level. No actual cost estimate is given here because it would depend on
real corpus size and query volume neither of which exist yet at cloud scale — a specific number
without that data would be a guess dressed up as an estimate.

## What would need to change in this codebase to deploy this

1. `config/settings.py` reads from Secret Manager-injected env vars — no code change, same
   `pydantic-settings` mechanism already in place.
2. `ingestion/pipeline.py`'s `ingest_directory()` needs a GCS-backed path option alongside the
   local filesystem glob it uses today.
3. `agents/llm.py`'s `get_llm()` / `get_fallback_llm()` need a Vertex AI branch alongside the
   Ollama one — LiteLLM (already a dependency, per ADR-004) supports Vertex AI natively, so this
   is a provider-string change (`vertex_ai/gemini-...` instead of `ollama/...`), not new
   integration code.
4. `audit/hashchain.py` needs a periodic checkpoint-signing job calling Cloud KMS — new, small,
   additive; doesn't change the existing hash-chain logic, just adds a second layer on top.
5. A Terraform module per the table above — not written, since nothing here has been applied and
   an untested Terraform module is not meaningfully more trustworthy than this document.
