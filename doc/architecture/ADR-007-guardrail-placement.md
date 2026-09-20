# ADR-007: Guardrails as an in-process pipeline stage, not a separate gateway service

## Status
Accepted — implemented (`src/aegisrag/guardrails/guardrails.py`), unit-tested.

## Context
Guardrails (input / retrieval-content / output checks) could be a separate service in front of
the API (a "guardrail gateway" pattern common in larger platforms), or functions called directly
inside the request path.

## Decision
Plain Python functions (`check_input`, `check_retrieved_content`, `check_output`) called directly
from `orchestrator.answer()` at the three points the design doc specifies (§16), reading their
thresholds from the same control plane (`config/control_plane.py`) as everything else.

## Alternatives considered
- **A separate guardrail microservice / sidecar**: the right shape at a scale with multiple
  backend services sharing one guardrail policy. At this assessment's scale (one FastAPI
  service), a network hop to a sidecar for a regex check adds latency and an operational
  dependency for no benefit.
- **Guardrails only in the system prompt ("please don't...")**: rejected outright — this is
  exactly the fresher-vs-senior distinction the design doc draws. A prompt instruction is a
  request the model can ignore; a guardrail function is a check the code enforces regardless of
  what the model does.

## Consequences
- Document content and user input are explicitly treated as **data**, never as instructions, at
  the code level (the regex-based injection check), not merely by asking the model nicely in its
  system prompt — the prompts in `/prompts/*.yaml` also say this, but the guardrail function is
  the actual enforcement layer, independent of whether the model listens.
- Because it's in-process, the guardrail check has zero added latency and no extra failure mode
  (no "guardrail service is down" case to handle).
- Trade-off: if this ever becomes multiple backend services, the guardrail logic would need
  extracting into a shared library or an actual gateway to avoid drift between copies — a real
  cost if the platform grows, noted rather than hidden.
