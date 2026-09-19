# AegisRAG — Agentic Enterprise Document Intelligence Platform

An observable, evaluable, agentic RAG platform that lets users search and reason over enterprise
documents through OpenWebUI, backed by Docling, PostgreSQL/PGVector, LlamaIndex, CrewAI, Ollama,
Arize Phoenix and RAGAs. Built for a technical assessment, designed the way a production knowledge
platform would be: ingestion → agentic retrieval → grounded, cited answers → traced → evaluated →
governed.

> The full design write-up (fresher vs. senior approach per component, guardrails, human-in-the-loop,
> fallback agents, hash-chain audit logging, the feedback/RAG-error-taxonomy loop, the AI control
> plane, ADRs, and the cloud reference architecture) lives in [`/doc`](./doc) — this README is the
> entry point and the accurate, current status; `/doc` is the deep dive.

## Status — what's actually built vs. designed

This is stated explicitly so nothing here is overclaimed.

| Layer | Status |
|---|---|
| Local tooling (Docker, Ollama, Python venv) | ✅ Installed & verified |
| `docker-compose` services (Postgres+PGVector, Phoenix, OpenWebUI) | ✅ Running locally |
| Repository scaffold + dependencies | ✅ Done |
| Document corpus | ✅ 1 PDF ingested (`EthicsofAI.pdf`, 21 pages → 69 chunks) — see note below |
| Ingestion pipeline (Docling → chunk → embed → PGVector) | ✅ Working — checksum dedupe verified |
| Retrieval (hybrid: PGVector cosine + Postgres full-text, RRF-fused) | ✅ Working |
| CrewAI 5-agent crew (planner → retrieval → evidence validation → synthesis → citation/quality) | ✅ Working end-to-end, bounded retry loop |
| FastAPI OpenAI-compatible endpoint | ✅ Working |
| OpenWebUI wired to the backend | ✅ Verified in-browser — real cited answers rendering |
| Phoenix tracing (every LLM/embedding/retrieval call) | ✅ Verified — spans visible per agent hop |
| Prompts externalized (`/prompts/*.yaml`, versioned) | ✅ Done |
| Guardrails (input/retrieval/output) | ✅ Implemented, unit-tested |
| Hash-chain audit log + `verify` CLI | ✅ Implemented — tamper-detection integration-tested |
| RAGAs golden set + evaluation harness | ✅ Harness built and runs against the real corpus (small golden set — see `/evaluation`) |
| Human-in-the-loop escalation, fallback/recovery agent chain | ⬜ Not yet implemented (guardrail/abstention path exists; the dedicated recovery agent from §18 doesn't yet) |
| AI control plane (single config surface for all policy) | 🟡 Partial — `agent_policy.yaml` + `.env` exist; not yet unified into one loader |
| ADRs, presentation deck, `/doc` write-ups | ⬜ Not yet written |
| Cloud reference architecture | 📄 Documented only — no cloud environment for this assessment |

**On the corpus**: two PDFs were provided; the second (`EthicalLeadership1.pdf`, 8 pages but image/slide-heavy)
made Docling's layout+OCR model take 30+ minutes per the user's own test and was dropped for this pass
rather than block the rest of the build — noted here rather than silently ignored. It can be re-added
later with OCR disabled (it's born-digital, not scanned, so OCR is unnecessary and is most of the cost).

**Known rough edge**: OpenWebUI issues a second, separate call to generate the chat title for every new
conversation, and that call goes through the same 5-agent pipeline as a real question — so the first
message in a new chat currently costs roughly 2x the pipeline run. Documented here rather than fixed
silently; the fix is to detect OpenWebUI's title-generation system prompt and short-circuit it.

Everything below the fold describes the **target architecture** this repo is being built to. Sections
are marked `[planned]` where the code doesn't exist yet, so this file stays trustworthy as the build
progresses instead of describing code that isn't there.

## Architecture

### Six planes

```mermaid
flowchart TB
    subgraph Experience["Experience Plane"]
        OWU["OpenWebUI"]
    end
    subgraph API["API / Security Plane"]
        FA["FastAPI Gateway<br/>REST + OpenAI-compatible"]
    end
    subgraph Runtime["AI Runtime Plane"]
        CR["CrewAI Crew"]
        LI["LlamaIndex Query Engines"]
    end
    subgraph Knowledge["Knowledge Plane"]
        DL["Docling Ingestion"]
        PG[("PostgreSQL + PGVector")]
    end
    subgraph Governance["Trust & Governance Plane"]
        GR["Guardrails"]
        HITL["Human-in-the-loop"]
        AU["Hash-chain Audit Log"]
    end
    subgraph Obs["Observability & Learning Plane"]
        PH["Arize Phoenix"]
        RG["RAGAs Eval"]
        FB["Feedback Loop"]
    end

    OWU -->|REST| FA
    FA --> CR
    CR --> LI
    LI --> PG
    DL --> PG
    CR --> GR
    GR --> HITL
    CR -->|traces| PH
    PH --> RG
    RG --> FB
    FB -.->|tunes| CR
```

### Ingestion flow — working, verified against the real corpus

```mermaid
flowchart LR
    PDF["Google Drive PDFs"] --> DL["Docling parser<br/>layout, tables, headings"]
    DL --> CH["Structural + semantic chunker"]
    CH --> EM["Ollama embeddings<br/>nomic-embed-text"]
    EM --> PG[("PGVector")]
```

`data/knowledge_base/EthicsofAI.pdf` (21 pages) ingests in ~36s into 69 chunks with page numbers and
section headings attached; re-running is a no-op via checksum dedupe (integration-tested).

### Agentic retrieval — the decision loop — working end-to-end

```mermaid
flowchart TD
    Q["User query"] --> P["Query Planner agent<br/>intent, decomposition"]
    P --> R["Retrieval agent<br/>hybrid: PGVector + lexical"]
    R --> E{"Evidence Validation agent"}
    E -->|"insufficient (≤2 retries)"| R
    E -->|supported| S["Synthesis agent<br/>evidence-only, cited"]
    S --> C["Citation & Quality agent"]
    C --> CF{"Confidence score"}
    CF -->|">= 0.85"| A1["Answer"]
    CF -->|"0.60 – 0.85"| A2["Answer + caveat"]
    CF -->|"< 0.60"| A3["Abstain → human review"]
```

Five agents, not fifteen — see `/doc/04-agentic-rag-design.md` for why that's the deliberate choice.
Verified live against the real corpus via OpenWebUI, with full Phoenix tracing and audit logging.

## Database design — schema applied, populated, verified

```mermaid
erDiagram
    DOCUMENTS ||--o{ CHUNKS : contains
    DOCUMENTS {
        uuid document_id PK
        text source_file
        text document_version
        date effective_date
        text status "active | superseded"
        text checksum
        int page_count
        timestamptz ingested_at
    }
    CHUNKS {
        uuid chunk_id PK
        uuid document_id FK
        text section_path
        int page_no
        text content
        vector embedding "pgvector, dim = embedding model's"
    }
    AUDIT_LOG {
        uuid event_id PK
        text agent
        text action
        text input_hash
        text output_hash
        text previous_event_hash
        text event_hash "sha256 hash-chain, §19 of the design doc"
        timestamptz created_at
    }
```

`DOCUMENTS.status` is how version lineage (§3 of the design doc) keeps retrieval from silently mixing
a superseded policy with the current one. `AUDIT_LOG` is the tamper-evident hash chain (§19) — each
row's `event_hash` covers its own content plus the previous row's hash.

## Repository structure

```
aegisrag/
├── README.md                # you are here
├── docker-compose.yml        # Postgres+PGVector, Phoenix, OpenWebUI
├── pyproject.toml            # Python deps
├── Makefile                  # venv / install / up / ingest / api / test / eval
├── .env.example
│
├── src/aegisrag/
│   ├── api/            FastAPI app — OpenAI-compatible /v1/chat/completions, /documents, /ingestion, /feedback, /health
│   ├── config/         settings.py (pydantic-settings from .env), agent_policy.yaml, prompts.py (prompt registry)
│   ├── ingestion/      pipeline.py — Docling parse + HybridChunker + Ollama embed + PGVector upsert, in one place
│   ├── retrieval/      hybrid_retriever.py — LlamaIndex BaseRetriever: PGVector cosine + Postgres full-text, RRF-fused
│   ├── agents/         definitions.py (5 CrewAI Agents), orchestrator.py (the decision loop), llm.py, json_utils.py
│   ├── guardrails/     input / retrieval-content / output checks (§16) — unit-tested
│   ├── audit/           hash-chain writer + `verify` CLI (§19) — tamper-detection integration-tested
│   ├── evaluation/     run.py — RAGAs harness against the golden set, baseline-regression check
│   ├── observability/  tracing.py — Phoenix/OTel wiring (LlamaIndex + LiteLLM instrumentation)
│   └── database/       schema.sql (raw DDL, not an ORM) + db.py (psycopg3 connection helper)
│   # chunking/ and embeddings/ from the original plan were folded into ingestion/ and retrieval/ —
│   # splitting them into their own modules would've been indirection with no real seam, given each
│   # is one library call (Docling's HybridChunker, llama-index's OllamaEmbedding); no feedback/
│   # module either yet — see the Status table for what §18/§20 (fallback agents, error-taxonomy
│   # feedback loop) still need.
│
├── prompts/              # externalized prompt templates (§5), versioned YAML — planner/retrieval/validation/synthesis
├── evaluation/
│   ├── datasets/golden/       # ethics_of_ai.json — 4 real questions against the ingested PDF
│   └── datasets/adversarial/  # prompt-injection / contradiction / stale-version test cases — not yet populated
├── tests/
│   ├── unit/             guardrails, JSON-repair parsing
│   └── integration/      audit hash-chain (tamper detection), ingestion dedupe — against real Postgres
├── doc/                  # architecture, ADRs, deployment, the full write-up — not yet authored
├── diagrams/
├── infra/                # docker (used) + terraform/kubernetes (reference only — not applied)
└── .github/workflows/    # CI file — not yet added
```

## Prerequisites

- Git, Python 3.11 (already on this machine via Homebrew: `/opt/homebrew/bin/python3.11` — the system
  default `python3` is 3.13, which is too new for `crewai`'s published releases, so the venv is
  pinned to 3.11)
- [Docker Desktop](https://www.docker.com/products/docker-desktop) (installed, running)
- [Ollama](https://ollama.com) (installed via Homebrew, running as a background service)

## Setup

```bash
git clone https://github.com/mad0907/aegisRag.git
cd aegisRag

# Python environment
make install          # creates .venv (Python 3.11) and installs all dependencies
                       # also applies scripts/patch_ragas.py — see "Known issues" below

# Infra services
cp .env.example .env
make up                # docker compose up -d : Postgres+PGVector, Phoenix, OpenWebUI
make init-db            # applies src/aegisrag/database/schema.sql

# Local LLM
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

# Put PDFs in data/knowledge_base/, then:
make ingest
make api                # serves on :8080; OpenWebUI (docker-compose) is already pointed at it
```

Open http://localhost:3000, pick the `aegisrag` model (auto-detected), and ask a question about the
ingested documents.

## Running the project

| Command | What it does |
|---|---|
| `make install` | Create `.venv` (Python 3.11) and install all dependencies |
| `make up` / `make down` | Start / stop Postgres+PGVector, Phoenix, OpenWebUI |
| `make init-db` | Apply the database schema (idempotent) |
| `make ingest` | Run the Docling → PGVector ingestion pipeline over `data/knowledge_base/` |
| `make api` | Run the FastAPI backend on `:8080` |
| `make eval` | Run the RAGAs evaluation harness against `evaluation/datasets/golden/` |
| `make verify-audit` | Walk the hash-chain audit log and report PASS/FAIL |
| `make test` | Run the test suite |

## Local service URLs

| Service | URL |
|---|---|
| OpenWebUI | http://localhost:3000 |
| Arize Phoenix | http://localhost:6006 |
| Postgres (PGVector) | `localhost:5433` (db `aegisrag`, user `aegisrag`) |
| Ollama | http://localhost:11434 |
| AegisRAG API | http://localhost:8080 (Swagger UI at `/docs`) |

## Known issues (disclosed, not hidden)

- **`ragas==0.4.3` has a real upstream bug**: it unconditionally imports
  `langchain_community.chat_models.vertexai.ChatVertexAI`, a submodule `langchain-community>=0.4`
  removed. `scripts/patch_ragas.py` patches the installed package to make that import optional (it's
  only used for an `isinstance()` check against a provider we don't use). Runs automatically as part
  of `make install`; safe to re-run.
- **OpenWebUI's title-generation call doubles the cost of the first message in a chat** — see the
  Status table above.
- **The second PDF from the assessment corpus (`EthicalLeadership1.pdf`) isn't ingested** — see the
  Status table for why and the fix.

## Configuration

All runtime config is externalized — see `.env.example`. Nothing is hard-coded: Ollama model names,
Postgres connection, Phoenix collector endpoint, and the agent policy thresholds (`max_iterations`,
`retrieval.min_score`, confidence thresholds for auto-answer / qualify / abstain) all come from
environment/config, not from Python source — see §21 (AI control plane) in the design doc.

## Further reading

- [`/doc`](./doc) — architecture, ADRs, deployment guide, security/threat model, production-readiness
  gap (once authored)
- The full design doc (six-plane architecture, agent design, guardrails, human-in-the-loop, fallback
  chains, hash-chain audit logging, the feedback/error-taxonomy loop, cloud reference architecture,
  fresher-vs-senior notes per component) — ask for the link if you don't have it bookmarked.

## Roadmap

Done: ingestion, hybrid retrieval, the 5-agent crew, FastAPI + OpenWebUI wiring, Phoenix tracing,
guardrails, hash-chain audit log, a first RAGAs pass. What's left:

1. Expand the golden + adversarial eval sets (currently 4 questions on 1 document)
2. Fallback/recovery agent chain (§18) — right now a failed evidence check ends in abstention, not
   a query-reformulation retry via a dedicated recovery agent
3. Human-in-the-loop risk-based escalation (§17)
4. Unify config into one AI-control-plane loader (§21) — `agent_policy.yaml` and `.env` both exist
   but aren't merged into a single source of truth yet
5. Fix the OpenWebUI title-generation double-cost issue
6. `/doc` write-up (architecture, ADRs, security/threat model, production-readiness gap), diagrams,
   presentation deck, OpenAPI export
7. GitHub Actions CI (lint/test/eval-gate — not yet added)
