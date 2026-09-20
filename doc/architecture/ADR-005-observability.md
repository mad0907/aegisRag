# ADR-005: Arize Phoenix + OpenInference/OTel auto-instrumentation, not custom logging

## Status
Accepted — implemented (`src/aegisrag/observability/tracing.py`).

## Context
The assessment mandates Arize Phoenix. The alternative considered was hand-rolled structured
logging (a `logger.info(json.dumps(...))` call at each step).

## Decision
`phoenix.otel.register()` sets a global OpenTelemetry `TracerProvider` exporting to a local
Phoenix collector (`docker-compose`, gRPC on `:4317`); `openinference-instrumentation-llama-index`
and `openinference-instrumentation-litellm` auto-instrument the retrieval layer and every agent's
LLM call (CrewAI's `LLM` runs through LiteLLM) without manual span creation in application code.

## Alternatives considered
- **Custom structured logs**: would satisfy "tracing," technically, but every new instrumented
  call site is a manual edit, spans aren't automatically correlated into one trace per request,
  and there's no UI to explore them without building one.
- **Manual OTel spans**: more control, but duplicates what the OpenInference instrumentors
  already do correctly for LlamaIndex and LiteLLM — not worth re-implementing.

## Consequences
- Verified live: one trace per `/v1/chat/completions` request contains 4 LLM `completion` spans
  (planner, evidence validator, synthesis, citation & quality), 2 `embedding` spans, and the
  `HybridPGRetriever` retrieval spans — confirmed via Phoenix's GraphQL API during this build.
- Every span carries model name and latency automatically; prompt *version* is added explicitly
  by the orchestrator into the audit log (§19) alongside the trace, since prompt-version isn't
  something the instrumentor can infer on its own.
- A span exists whether or not the call succeeds, so a failed primary-model call and its fallback
  retry (ADR-004) both show up in one trace — useful for exactly the "why was this answer slow /
  wrong" debugging Phoenix is for.
