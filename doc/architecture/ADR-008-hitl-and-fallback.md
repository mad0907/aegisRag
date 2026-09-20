# ADR-008: Risk-based human review queue + two-tier fallback, both entirely local

## Status
Accepted — implemented (`guardrails/human_review.py`, `agents/orchestrator.py`'s recovery block),
tested (`tests/unit/test_human_review.py`, `tests/integration/test_review_queue.py`).

## Context
The design doc calls for two things this build initially left unimplemented: human-in-the-loop
escalation (§17) and a fallback/recovery chain below the primary retrieval-and-answer path (§18).
Both could plausibly need external infrastructure (a real reviewer UI, a paging/notification
system, a secondary model-serving cluster) — but neither actually does, at this scale.

## Decision

**Human-in-the-loop (§17)** — a `review_queue` Postgres table plus two endpoints
(`GET /review-queue`, `POST /review-queue/{id}/decision`). Risk is assessed *after* an answer is
computed (so the assessment has a real confidence score and evidence verdict to work with), from
three signals: confidence below the human-review threshold, a `contradictory` evidence verdict,
and a configurable high-risk keyword list (`compliance`, `legal`, `approve`, `terminate`, etc., in
`agent_policy.yaml`). HIGH risk withholds the answer from the caller entirely, replacing it with a
review reference; MEDIUM risk returns the answer but still logs it to the queue for after-the-fact
review; LOW risk is untouched.

**Fallback/recovery (§18)** — two independent tiers:
- *Agent-level*: when the primary retrieve→validate loop exhausts its retries without
  `supported`/`partially_supported` evidence, two different retrieval strategies are tried before
  giving up — lexical-only search (drops the embedding step, catches cases where semantic search
  is confidently wrong) and a relaxed hybrid search (larger candidate pool, no implicit score
  floor). If evidence is found this way, the pipeline continues normally to synthesis.
- *Model-level*: every LLM call in every agent step (`orchestrator._run_task`) is wrapped —
  primary model failure retries once against the fallback model (ADR-004); if both fail, the
  request degrades to a `degraded_retrieval_only` response (the raw retrieved passages, cited,
  with no synthesis) rather than a 500 error or an invented answer.

## Alternatives considered
- **A real ticketing/paging integration for HITL** (Slack, PagerDuty, email): the right answer for
  an actual production deployment, and exactly the kind of thing that belongs in the cloud
  write-up, not this local build — there's no user directory or notification channel to page here.
  The Postgres queue + REST endpoints are the complete, honest local equivalent: a human polls
  `GET /review-queue` (or a future scheduled job could push it somewhere) and decides.
- **Retrying the identical query on evidence failure**: what the code did before this pass — not
  real recovery, just hoping a re-run behaves differently. Replaced with genuinely different
  retrieval strategies.

## Consequences
- Both features are fully local, fully tested, and add zero new infrastructure dependencies —
  consistent with this build's overall local-first stance.
- The review queue has no push notification; a pending review is only visible by polling the API.
  Stated as a known limitation rather than glossed over — see
  `doc/16-production-readiness-gap.md`.
- Every audit_log row now also stamps `policy_version` (from `agent_policy.yaml`), so a change to
  risk thresholds or fallback behavior is traceable to the exact run that started using it — this
  is what makes `agent_policy.yaml` + `config/control_plane.py` a genuine single control-plane
  surface rather than config scattered across modules.
