# ADR-003: Five CrewAI agents with a Python-controlled loop, not a larger autonomous crew

## Status
Accepted — implemented (`src/aegisrag/agents/`).

## Context
CrewAI supports sequential and hierarchical `Process` types that let a crew run itself. It would
be possible to build many narrowly-scoped agents (10-15) and let CrewAI's own orchestration
manage them.

## Decision
Five agents — Query Planner, Retrieval Specialist, Evidence Validator, Synthesis Writer,
Citation & Quality Reviewer — each a genuine `crewai.Agent`, but orchestrated by an explicit
Python controller (`agents/orchestrator.py`) rather than CrewAI's own `Crew`/`Process` loop.

## Alternatives considered
- **A larger crew (10-15 agents)**: more agents does not automatically mean better answers — each
  additional agent is another LLM round-trip (latency, cost, another chance to fail on a small
  local model) and another prompt to maintain. Five map cleanly onto five *distinct
  responsibilities* the design doc calls for; a sixth agent would need its own distinct job, not
  just a relabeling of an existing step.
- **CrewAI's own sequential/hierarchical `Process`**: these are excellent for a linear pipeline
  or a manager-delegates-to-workers pattern, but neither natively expresses "retry the retrieval
  step with a reformulated query, bounded at N attempts, and branch on a confidence score
  computed after the fact." That control flow needs an explicit loop and explicit branching in
  code — CrewAI's `Process` abstractions don't have a first-class "retry until a condition, then
  branch" primitive. Forcing it into `Process.hierarchical` would mean writing the same control
  flow anyway, just hidden inside a manager-agent's prompt instead of visible in Python.

## Consequences
- Each of the five steps genuinely is a `crewai.Agent` + `crewai.Task.execute_sync()` — CrewAI is
  actually doing the LLM orchestration work for each step, satisfying "CrewAI: Multi-agent
  implementation" in the assessment's mandatory-tools list.
- The loop bound (`max_iterations` in `agent_policy.yaml`), the retry/recovery strategy (§18),
  and the confidence-based branching (§4) are explicit, readable Python in `orchestrator.py` —
  testable and auditable, not implicit in a manager-agent's system prompt.
- Trade-off: this is not "pure" CrewAI multi-agent autonomy — the crew doesn't decide its own
  workflow. That's a deliberate choice for a system whose whole premise is *controlled* autonomy
  (see the AegisRAG name itself), not maximal autonomy.
