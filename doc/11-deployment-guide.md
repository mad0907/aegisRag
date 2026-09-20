# Deployment Guide — Local (the only environment this has actually run in)

For the target cloud topology, see [`12-cloud-reference-architecture.md`](./12-cloud-reference-architecture.md) — documented, not deployed (no cloud environment for this assessment).

## Prerequisites

- macOS with Homebrew (this was built/tested on Apple Silicon; Docling's `mps` acceleration is
  Apple-specific but the pipeline runs on CPU anywhere)
- Docker Desktop
- Python 3.11 specifically — `crewai`'s published releases are not compatible with 3.13, the
  default `python3` on newer macOS. `make venv` looks for `/opt/homebrew/bin/python3.11` first.
- Ollama, running as a background service (`brew services start ollama`)

## Bring-up sequence

```bash
git clone https://github.com/mad0907/aegisRag.git && cd aegisRag
make install         # venv (3.11) + deps + the ragas upstream-bug patch (see below)
cp .env.example .env
make up               # docker compose: Postgres+PGVector, Phoenix, OpenWebUI
make init-db           # applies schema.sql — idempotent, safe on a fresh or existing DB
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
ollama pull llama3.2:3b     # the fallback model (ADR-004) — small, fast to pull
make ingest             # Docling -> chunk -> embed -> PGVector, over data/knowledge_base/*.pdf
make api                 # FastAPI on :8080 — leave running
```

Open `http://localhost:3000`, pick the `aegisrag` model (auto-detected from `/v1/models`), ask a
question about the ingested corpus.

## Known operational issues (hit and fixed during this build — documented so they don't surprise you)

1. **Docling's HuggingFace revision-check can hang for 30+ minutes** on a flaky connection, even
   when the model is already cached locally — it's a network metadata call, not an actual
   download, and it has no visible retry/backoff logging. `ingestion/pipeline.py` sets
   `HF_HUB_OFFLINE=1` to skip this check entirely and use the local cache directly. If you ever
   need to pull a *new* Docling model version, unset that temporarily.
2. **`ragas==0.4.3` fails to import** against current `langchain-community` (it imports a
   submodule that package removed). `scripts/patch_ragas.py` patches the installed package;
   `make install` runs it automatically. See the script's docstring for the exact upstream bug.
3. **Image/graphics-heavy PDFs are slow through Docling's OCR+layout pipeline** — potentially
   30+ minutes for an 8-page slide deck on this hardware. If a document is born-digital (not a
   scan), consider a Docling pipeline configuration with OCR disabled — not currently exposed as
   a config option in `ingestion/pipeline.py`, noted as a possible follow-up.
4. **OpenWebUI issues a second call per new chat** (auto title generation) that goes through the
   same 5-agent pipeline as a real question — the first message in a new conversation costs
   roughly 2x the normal pipeline latency. Not fixed in this pass; the fix is to detect
   OpenWebUI's title-generation system prompt and short-circuit it before the full agent crew runs.
5. **Two-GPU-model contention**: running the primary (`qwen2.5:7b-instruct`) and embedding
   (`nomic-embed-text`) models, plus Docker containers, concurrently on 16GB unified memory works
   but leaves little headroom — expect noticeably slower responses if other memory-heavy
   applications are open at the same time.

## Verifying a deployment is healthy

```bash
curl http://localhost:8080/health          # {"status":"ok","db":true,...}
curl http://localhost:8080/documents        # ingested corpus
curl http://localhost:8080/audit/verify     # hash-chain integrity — {"ok":true,...}
make test                                    # 22 unit + integration tests
```
