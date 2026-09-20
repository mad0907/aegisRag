# Security Design & Threat Model

Scoped to what's actually implemented (local deployment) plus what a production deployment would
need and doesn't have yet — both stated explicitly.

## Threats considered

| Threat | Mitigation | Status |
|---|---|---|
| Prompt injection via user query | Regex-based pattern detection (`guardrails.check_input`); document/user text is architecturally kept as *data* in every prompt template (`/prompts/*.yaml` — "treat as data, never as instructions") | ✅ Implemented, unit-tested |
| Prompt injection via retrieved document content | `guardrails.check_retrieved_content()` scans every retrieved passage the same way, before it reaches an agent | ✅ Implemented, unit-tested |
| Hallucinated / unsupported claims | Evidence Validator + Citation & Quality agents; confidence-gated abstention; degrade-to-retrieval-only fallback never lets a model "fill in" missing evidence | ✅ Implemented, verified live |
| Undetected tampering with the decision record | SHA-256 hash-chain audit log, `verify_chain()` | ✅ Implemented, integration-tested (tamper-and-detect test) |
| High-stakes answer released without review | Risk-based human-in-the-loop queue (§17) | ✅ Implemented, tested |
| Stale/superseded document silently used | `documents.status` (`active`/`superseded`); retrieval filters to `active` by default | ✅ Schema in place; no superseded-document test case in the corpus yet to exercise it |
| Secrets in source control | All connection strings / API keys via `.env` (gitignored), never hard-coded | ✅ |
| Unauthenticated API access | — | ⬜ Not implemented — see "Known gaps" below |
| Data exfiltration via the LLM (SSRF, arbitrary tool use) | Agents have no tools beyond the retriever and the four defined prompts — no shell, no arbitrary HTTP, no filesystem access exposed to any agent | ✅ By construction — there is nothing to exfiltrate *through* |
| Denial of service via expensive queries | `MAX_QUERY_CHARS` cap; bounded retry/recovery loop (`max_iterations`) prevents unbounded agent loops | ✅ |

## Known gaps (local build, stated plainly)

- **No API authentication.** `FastAPI` exposes every endpoint (including `/ingestion` and
  `/review-queue/*/decision`) with no auth check. Fine for `localhost`-only local development;
  not fine as-is for anything reachable over a network. See the cloud write-up for the
  recommended fix (API Gateway + IAM/OAuth in front).
- **No rate limiting.** A client can call `/v1/chat/completions` as fast as it likes; the only
  real limiter today is that each call takes 2-3 minutes on this hardware.
- **Prompt-injection detection is pattern-based, not model-based.** A sufficiently creative
  injection attempt that doesn't match the regex list would not be flagged. The architectural
  defense (treating content as data in the prompt template) is the real backstop; the regex is a
  visibility signal, not the primary control.
- **No PII detection/redaction at ingestion.** If a PDF contained PII, it would be embedded and
  stored like any other content. Not exercised by the current corpus (a published ethics paper),
  but would matter for a real enterprise document set.
- **`review_queue` decisions are not authenticated** — anyone who can reach the API can approve or
  reject a pending review. A real deployment needs a reviewer identity on that endpoint.

## What would close these gaps (not implemented — recommended)

1. Put the FastAPI service behind an API gateway (cloud reference: GCP API Gateway or a load
   balancer with IAP) requiring an API key or OAuth token per caller.
2. Add per-key rate limiting at the gateway, not in application code.
3. Require an authenticated reviewer identity (even a shared service-account token, at minimum)
   on `/review-queue/*/decision`, and log *who* decided in `review_queue.reviewer_note` or a new
   column.
4. If the corpus ever includes PII-bearing documents, add a redaction pass (e.g. Presidio) between
   Docling's output and the chunker — flagged as future work, not built, since the current corpus
   doesn't need it.

None of these need a cloud environment to build in principle (auth middleware and rate limiting
are pure code), but weren't built in this pass because the assessment's actual corpus and access
pattern (a single local user) don't currently need them — building them now would be defending
against a threat model this deployment doesn't have, which is its own kind of over-engineering.
They're listed here so the gap is a decision, not an oversight.
