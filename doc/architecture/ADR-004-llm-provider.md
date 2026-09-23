# ADR-004: Ollama, with an explicit primary/fallback model pair

## Status
Accepted — implemented (`src/aegisrag/agents/llm.py`, `config/settings.py`).

**Updated 2026-09-23**: primary/fallback swapped. `llama3.2:3b` is now primary and
`qwen2.5:7b-instruct` is now fallback — measured ~2x faster generation on this CPU-only hardware,
and `llama3.2:3b` had already been proven against these exact prompts in its former fallback role,
so the swap carried low risk. The Decision section below is left as originally written for
historical context; the config defaults in `settings.py`/`.env.example` reflect the swap.

## Context
The assessment mandates Ollama as "local or remote LLM provider." A single hard-coded model name
is the simplest implementation.

## Decision
Two models, both via Ollama: `qwen2.5:7b-instruct` (primary) and `llama3.2:3b` (fallback),
selected via `OLLAMA_PRIMARY_MODEL` / `OLLAMA_FALLBACK_MODEL` in `.env` — never hard-coded in
Python. `agents/llm.py` builds a CrewAI `LLM` for either on demand; `orchestrator.py`'s
`_run_task` tries the primary model, and only on an actual failure (not just a low-confidence
answer) retries once with the fallback model before giving up.

## Alternatives considered
- **One fixed model, no fallback**: simpler, but a model that's unloaded, crashed, or simply
  absent from `ollama list` takes the whole system down with no degraded path — contradicts §18
  of the design (a fallback chain should exist below the primary path).
- **A larger primary model (e.g. 13B+)**: this machine's hardware (Apple M3, 16GB unified memory)
  was profiled before picking a model (see the conversation this build came from) — a 7B
  Q4-quantized model was the practical ceiling that leaves headroom for Docling, Postgres and
  Docker containers running concurrently.

## Consequences
- Model choice is entirely environment-config, so swapping to a larger model on better hardware,
  or to a hosted provider, is a config change, not a code change (see ADR-004's cousin decision:
  the LLM is always accessed through CrewAI's `LLM` wrapper / LiteLLM, which already speaks
  OpenAI, Anthropic, Bedrock, Vertex AI, etc. — Ollama is not hard-wired into the agent code).
- The fallback path is real code, exercised in `orchestrator._run_task`, but has not been
  chaos-tested against an actually-down Ollama instance in this pass (that would have required
  taking down the shared local Ollama service mid-build) — stated honestly here rather than
  claimed as fully verified.
