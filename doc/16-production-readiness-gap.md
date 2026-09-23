# Production Readiness Gap

The canonical, current version of the built-vs-designed distinction — the README's status table
is a summary of this; this is the detail.

## Assessment environment (what exists, running, today)

```
Laptop (Apple M3, 16GB RAM)
  Docker Desktop
    Postgres 16 + pgvector 0.8.6   (5433)
    Arize Phoenix                  (6006 UI, 4317 OTLP)
    OpenWebUI                       (3000)
  Ollama (native, brew service)     (11434)
    llama3.2:3b (primary, ADR-004)
    qwen2.5:7b-instruct (fallback)
    nomic-embed-text (embeddings)
  Python 3.11 venv
    FastAPI (uvicorn, :8080)
```

One PDF ingested (`EthicsofAI.pdf`, 21 pages → 69 chunks). 22 tests passing (unit + integration
against the real Postgres). A RAGAs eval harness exists and runs against a 4-question golden set.

## Production environment (target, per `12-cloud-reference-architecture.md`)

```
GCP
  Cloud Load Balancer + Cloud Armor
  Cloud Run: OpenWebUI, AegisRAG API
  Cloud SQL for PostgreSQL (pgvector)
  Vertex AI (optional LLM/embedding provider, eval tooling)
  Cloud KMS (audit-chain checkpoint signing)
  Cloud Storage (corpus, versioned)
  Secret Manager (all credentials)
  Cloud Logging / Monitoring
  Cloud Build + Cloud Deploy (CI/CD, eval-gated)
```

## Gap table

| Capability | Assessment env | Production target | Gap |
|---|---|---|---|
| Ingestion, retrieval, 5-agent crew | ✅ Running, verified | Same code, different LLM/storage backends | Config changes only (ADR-004, cloud doc §4) |
| Guardrails, human review queue, audit hash chain | ✅ Running, tested | Same, + KMS-signed checkpoints, + authenticated reviewers | Additive — new signing job, new auth middleware |
| Phoenix tracing | ✅ Running, verified | Self-hosted on Cloud Run/GKE | Same container, needs persistent storage config |
| RAGAs eval | ✅ Harness works, 4-question golden set | Scheduled, eval-gated CI/CD | Needs a much larger golden+adversarial set, and the CI wiring itself |
| API authentication | ⬜ None | API Gateway / Cloud Run IAM | Not built — no auth need at single-local-user scale, but a real gap for any networked deployment (§09-security-design.md) |
| Rate limiting | ⬜ None | At the load balancer / gateway | Not built |
| PII redaction at ingestion | ⬜ None | A redaction pass before chunking, if the corpus needs it | Not built — current corpus (a public paper) doesn't need it |
| CI/CD | ⬜ No GitHub Actions workflow yet | Cloud Build, eval-gated | Not built |
| Multi-document corpus | 🟡 1 of 2 assessment PDFs ingested | N/A | Second PDF (`EthicalLeadership1.pdf`) dropped this pass — Docling's OCR/layout pipeline took 30+ min on it; see the deployment guide's known issues |
| Document versioning (`active`/`superseded`) | 🟡 Schema exists, unexercised | Same | No second version of any document in the corpus yet to test against |
| Presentation deck, README, ADRs | ✅ Done | — | — |

## Reading this table honestly

Most of the "gap" is genuinely just infrastructure substitution (swap Postgres for Cloud SQL,
swap a `.env` file for Secret Manager) that the codebase is already shaped to make cheap — because
`config/settings.py` and `config/control_plane.py` (ADR-008) already centralize everything that
would need to change. The real, non-trivial gaps are the ones explicitly marked ⬜: authentication,
rate limiting, and CI/CD are genuinely not built, not because they're hard, but because building
them against a threat model and traffic pattern this local assessment doesn't have would have been
speculative engineering — the design doc argues against exactly that (§"don't over-engineer").
They're scoped here so building them later is a known task, not a surprise.
