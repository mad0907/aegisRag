"""The agent decision loop (§4 of the design doc), extended with agent- and model-level
fallback/recovery (§18) and risk-based human-in-the-loop escalation (§17).

QUESTION -> PLAN -> RETRIEVE -> ASSESS EVIDENCE
                                     |
                     insufficient? --+-- (<= max retries) --> REFINE QUERY --> RETRIEVE
                                     | still insufficient after all retries
                                     v
                       AGENT-LEVEL RECOVERY (lexical-only, then relaxed hybrid)
                                     |
                     still insufficient?  --> found SOME evidence --> DEGRADE: retrieval-only
                                     |                                  response, no synthesis
                                     |  found NOTHING
                                     v
                                 ABSTAIN
                     (evidence found) |
                                     v
                               SYNTHESIZE -> VERIFY & SCORE CONFIDENCE
                                     |
                          RISK ASSESSMENT (§17)
                     high risk  -> withhold, queue for human review
                     medium risk -> answer + logged for after-the-fact review
                     low risk    -> straight through
                                     |
                  confidence >= auto_threshold      -> ANSWER
                  qualify_threshold <= conf < auto    -> ANSWER + caveat
                  confidence < qualify_threshold       -> ABSTAIN / human review

Every LLM call in every step is itself wrapped with model-level fallback: primary model fails
or errors -> retry once with the secondary model -> if that also fails, the whole run degrades
to a retrieval-only response rather than crashing or inventing an answer.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from crewai import Task
from opentelemetry import trace as otel_trace

from aegisrag.agents.definitions import build_agents
from aegisrag.agents.json_utils import extract_json
from aegisrag.agents.llm import get_fallback_llm, get_light_llm, get_llm
from aegisrag.audit.hashchain import AuditEvent, write_event
from aegisrag.config.control_plane import get_policy
from aegisrag.config.prompts import get_registry
from aegisrag.config.settings import get_settings
from aegisrag.guardrails import guardrails
from aegisrag.guardrails.human_review import assess_risk, queue_for_review
from aegisrag.observability.tracing import get_tracer_provider
from aegisrag.retrieval.answer_cache import AnswerCache
from aegisrag.retrieval.hybrid_retriever import HybridPGRetriever


class ModelUnavailableError(Exception):
    """Raised when both the primary and fallback models fail for one call."""


# One span per named agent step, so Phoenix shows "agent.planner", "agent.synthesizer" etc. as
# distinct, timed spans instead of every LLM call collapsing into an identical unlabeled
# "completion" span from the generic LiteLLM instrumentation.
#
# Must come from Phoenix's own tracer_provider (init_tracing()), not the global
# opentelemetry.trace API: `import crewai` sets the global TracerProvider as a side effect of its
# own bundled telemetry, before Phoenix's register() runs, and OTel only honors the first
# provider set per process -- see observability/tracing.get_tracer_provider().
def _tracer():
    provider = get_tracer_provider()
    if provider is not None:
        return provider.get_tracer("aegisrag.orchestrator")
    return otel_trace.get_tracer("aegisrag.orchestrator")  # pre-startup / test fallback (no-op)


_GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|thx|good (morning|afternoon|evening)|ok|okay|cool|bye|goodbye)[\s!.,]*$",
    re.IGNORECASE,
)


@dataclass
class Citation:
    source_file: str
    page_no: int | None
    section_path: str | None


@dataclass
class AnswerResult:
    answer: str
    confidence: float
    status: str
    # "answered" | "answered_with_caveat" | "abstained" | "declined" |
    # "degraded_retrieval_only" | "pending_human_review"
    citations: list[Citation] = field(default_factory=list)
    trace_id: str = ""
    iterations: int = 0
    review_id: str | None = None
    # One dict per pipeline decision, in execution order — for the "show pipeline steps" panel in
    # OpenWebUI (api/main.py builds a <details> block from this). Shape:
    #   {"agent": str, "status": "cache_hit"|"cache_miss"|"cache_stored"|"bypassed"|"ran",
    #    "method": "regex"|"cosine_similarity"|"llm"|None, "detail": str}
    steps: list[dict] = field(default_factory=list)


def _passages_block(nodes) -> str:
    return "\n\n".join(
        f"[{i}] ({n.node.metadata.get('source_file')}, "
        f"p.{n.node.metadata.get('page_no')}): {n.node.get_content()}"
        for i, n in enumerate(nodes)
    ) or "(no passages retrieved)"


def _citations_from_nodes(nodes) -> list[Citation]:
    return [
        Citation(
            source_file=n.node.metadata.get("source_file"),
            page_no=n.node.metadata.get("page_no"),
            section_path=n.node.metadata.get("section_path"),
        )
        for n in nodes
    ]


class AegisRAGOrchestrator:
    # Must match the assignment in agents/definitions.py's build_agents(). Only Planner is
    # tiered down to OLLAMA_LIGHT_MODEL by default — not because it's the only short-JSON step
    # (Evidence Validator and Citation & Quality are too), but because it's the only one of the
    # three where being wrong is cheap. Measured on this corpus with llama3.2:1b:
    #   - Evidence Validator: started misjudging sufficient evidence as "insufficient", which
    #     triggered extra retrieval retries and recovery attempts -- net SLOWER than not tiering
    #     at all, and it degraded to retrieval-only instead of answering. Excluded.
    #   - Citation & Quality: returned an unreliable confidence score (0.00 on a genuinely
    #     answerable question) instead of a defensible one -- exactly the number the human-review
    #     routing and auto/qualified-answer thresholds depend on. Excluded.
    #   - Planner: its decision is coarse (needs_retrieval, almost always true on this corpus --
    #     see smart_bypass.planner_heuristic's own docstring) and a wrong call doesn't corrupt
    #     anything downstream. Tiering held up in testing. Kept.
    # This is a real, if narrower, finding: don't tier a step by output size alone -- tier it by
    # how expensive being wrong is. See doc/13-latency-optimization.md for the measured numbers.
    _LIGHT_TIER_AGENTS = {"planner"}

    def __init__(self):
        self.settings = get_settings()
        policy = get_policy()
        # model_tiering (agent_policy.yaml): planner/evidence_validator/citation_quality get a
        # smaller, faster model when enabled; synthesizer always stays on the primary model
        # (build_agents wires this per-agent — see agents/definitions.py). Read once at startup,
        # same restart-to-change contract as every other policy value (get_policy() is cached).
        tiering_on = policy.model_tiering.enabled
        light_factory = get_light_llm if tiering_on else None
        self.agents = build_agents(get_llm, light_factory)
        # Which real model name each agent's PRIMARY attempt actually uses — _run_task needs
        # this for accurate audit-log / Phoenix-span records instead of assuming everything
        # runs on ollama_primary_model, which is only true when tiering is off.
        self._primary_model_by_agent = {
            key: (self.settings.ollama_light_model if tiering_on and key in self._LIGHT_TIER_AGENTS
                  else self.settings.ollama_primary_model)
            for key in self.agents
        }
        self._fallback_agents: dict | None = None  # built lazily, only if ever needed
        self.prompts = get_registry()
        self.retriever = HybridPGRetriever(top_k=policy.retrieval.top_k)
        self.cache = AnswerCache()

    def _get_fallback_agent(self, agent_key: str):
        if self._fallback_agents is None:
            self._fallback_agents = build_agents(get_fallback_llm)
        return self._fallback_agents[agent_key]

    def _run_task(self, agent_key: str, prompt_name: str, trace_id: str, steps: list[dict], **prompt_vars) -> str:
        with _tracer().start_as_current_span(
            f"agent.{agent_key}",
            attributes={"agent.key": agent_key, "agent.task": prompt_name, "agent.trace_id": trace_id},
        ) as span:
            policy = get_policy()
            prompt = self.prompts.get(prompt_name)
            rendered = prompt.render(**prompt_vars)
            input_data = {"vars": {k: str(v)[:500] for k, v in prompt_vars.items()}}

            model_used = self._primary_model_by_agent.get(agent_key, self.settings.ollama_primary_model)
            raw: str | None = None
            primary_error: Exception | None = None
            used_fallback = False

            try:
                task = Task(
                    description=rendered,
                    expected_output="Follow the format instructions in the prompt exactly.",
                    agent=self.agents[agent_key],
                )
                raw = task.execute_sync().raw
            except Exception as exc:  # noqa: BLE001 — any LLM/transport failure triggers fallback
                primary_error = exc

            if raw is None and policy.fallback.enabled and policy.fallback.secondary_model_enabled:
                span.set_attribute("agent.used_fallback_model", True)
                used_fallback = True
                try:
                    fallback_agent = self._get_fallback_agent(agent_key)
                    task = Task(
                        description=rendered,
                        expected_output="Follow the format instructions in the prompt exactly.",
                        agent=fallback_agent,
                    )
                    raw = task.execute_sync().raw
                    model_used = self.settings.ollama_fallback_model
                except Exception as fallback_exc:  # noqa: BLE001
                    write_event(
                        AuditEvent(
                            agent=agent_key,
                            action=f"{prompt_name}_failed",
                            trace_id=trace_id,
                            model="none",
                            prompt_version=f"{prompt.id}@{prompt.version}",
                            input_data=input_data,
                            output_data={
                                "primary_error": str(primary_error),
                                "fallback_error": str(fallback_exc),
                            },
                        )
                    )
                    steps.append({
                        "agent": agent_key, "status": "failed", "method": "llm",
                        "detail": f"{prompt_name}: both primary and fallback models failed",
                    })
                    raise ModelUnavailableError(
                        f"Both {model_used} and fallback ({self.settings.ollama_fallback_model}) "
                        f"models failed for {agent_key}/{prompt_name}"
                    ) from fallback_exc

            if raw is None:
                # fallback disabled and primary failed
                steps.append({
                    "agent": agent_key, "status": "failed", "method": "llm",
                    "detail": f"{prompt_name}: primary model failed, fallback disabled",
                })
                raise ModelUnavailableError(str(primary_error))

            span.set_attribute("agent.model_used", model_used)
            write_event(
                AuditEvent(
                    agent=agent_key,
                    action=prompt_name,
                    trace_id=trace_id,
                    model=model_used,
                    prompt_version=f"{prompt.id}@{prompt.version}",
                    input_data=input_data,
                    output_data={"raw": raw[:2000]},
                )
            )
            steps.append({
                "agent": agent_key, "status": "ran", "method": "llm",
                "detail": f"{prompt_name} → model={model_used}" + (" (fallback)" if used_fallback else ""),
            })
            return raw

    def _plan(self, query: str, trace_id: str, steps: list[dict]) -> dict:
        """Smart bypass (agent_policy.yaml): a document-QA system's queries almost always need
        retrieval, and in every case observed during testing the LLM planner's sub_queries was
        just [query] anyway — the only real decision is "is this a greeting, not a question".
        A regex answers that in microseconds; skip the LLM call unless smart_bypass is off."""
        policy = get_policy()
        if policy.smart_bypass.enabled and policy.smart_bypass.planner_heuristic:
            with _tracer().start_as_current_span(
                "agent.planner", attributes={"agent.key": "planner", "agent.bypassed": True}
            ):
                if _GREETING_RE.match(query.strip()):
                    write_event(AuditEvent(agent="planner", action="bypassed_heuristic_greeting", trace_id=trace_id))
                    steps.append({
                        "agent": "planner", "status": "bypassed", "method": "regex",
                        "detail": "greeting pattern matched — no retrieval needed",
                    })
                    return {"needs_retrieval": False, "sub_queries": []}
                write_event(
                    AuditEvent(agent="planner", action="bypassed_heuristic_default_retrieval", trace_id=trace_id,
                               output_data={"query": query[:200]})
                )
                steps.append({
                    "agent": "planner", "status": "bypassed", "method": "regex",
                    "detail": "not a greeting → defaulted to retrieval on the full query",
                })
                return {"needs_retrieval": True, "sub_queries": [query]}

        plan_raw = self._run_task("planner", "query_planner", trace_id, steps, query=query)
        return extract_json(plan_raw, fallback={"needs_retrieval": True, "sub_queries": [query]})

    def _evaluate_evidence(self, query: str, nodes, trace_id: str, steps: list[dict]) -> dict:
        policy = get_policy()
        if policy.smart_bypass.enabled and nodes:
            top_similarity = nodes[0].node.metadata.get("vector_similarity")
            if top_similarity is not None and top_similarity >= policy.smart_bypass.evidence_similarity_threshold:
                with _tracer().start_as_current_span(
                    "agent.evidence_validator",
                    attributes={"agent.key": "evidence_validator", "agent.bypassed": True,
                                "agent.top_similarity": round(top_similarity, 4)},
                ):
                    write_event(
                        AuditEvent(agent="evidence_validator", action="bypassed_heuristic_high_similarity",
                                   trace_id=trace_id,
                                   output_data={"top_similarity": round(top_similarity, 4),
                                                "threshold": policy.smart_bypass.evidence_similarity_threshold})
                    )
                    steps.append({
                        "agent": "evidence_validator", "status": "bypassed", "method": "cosine_similarity",
                        "detail": f"top passage similarity {top_similarity:.3f} ≥ threshold "
                                  f"{policy.smart_bypass.evidence_similarity_threshold} — evidence accepted without an LLM call",
                    })
                    return {
                        "verdict": "supported",
                        "reasoning": f"heuristic bypass: top cosine similarity {top_similarity:.3f} >= "
                                     f"threshold {policy.smart_bypass.evidence_similarity_threshold}",
                        "usable_passage_ids": list(range(len(nodes))),
                    }

        passages_text = _passages_block(nodes)
        guardrails.check_retrieved_content([n.node.get_content() for n in nodes])
        eval_raw = self._run_task(
            "evidence_validator", "evidence_validator", trace_id, steps, query=query, passages=passages_text
        )
        return extract_json(eval_raw, fallback={"verdict": "insufficient"})

    def _degrade_to_retrieval_only(self, nodes, trace_id: str, iterations: int, reason: str, steps: list[dict]) -> AnswerResult:
        write_event(
            AuditEvent(agent="orchestrator", action="degrade_retrieval_only", trace_id=trace_id,
                       output_data={"reason": reason, "n_nodes": len(nodes)})
        )
        lines = [
            f"[{n.node.metadata.get('source_file')}, p.{n.node.metadata.get('page_no')}] "
            f"{n.node.get_content()[:400]}"
            for n in nodes
        ]
        answer = (
            "I couldn't produce a fully synthesized, validated answer, so here is the raw "
            "evidence retrieved instead of a possibly-unreliable summary:\n\n" + "\n\n".join(lines)
        )
        return AnswerResult(
            answer=answer,
            confidence=0.0,
            status="degraded_retrieval_only",
            citations=_citations_from_nodes(nodes),
            trace_id=trace_id,
            iterations=iterations,
            steps=steps,
        )

    def answer(self, query: str) -> AnswerResult:
        trace_id = str(uuid.uuid4())
        policy = get_policy()
        steps: list[dict] = []

        input_check = guardrails.check_input(query)
        if not input_check.allowed:
            return AnswerResult(
                answer=f"Request declined: {input_check.reason}", confidence=0.0,
                status="declined", trace_id=trace_id, steps=steps,
            )

        # SEMANTIC CACHE: a close-enough repeat/paraphrase of a past question skips retrieval
        # and every agent LLM call entirely — the fastest possible answer is the one you don't
        # have to compute. Only ever a hit against a genuinely-answered past result (see
        # AnswerCache.store — abstains/pending-reviews/degraded answers are never cached).
        query_embedding = None  # reused by store() below on a miss, so the question is embedded once, not twice
        if policy.answer_cache.enabled:
            lookup = self.cache.lookup(query, policy.answer_cache.similarity_threshold)
            query_embedding = lookup.query_embedding
            if lookup.hit is not None:
                cached = lookup.hit
                write_event(
                    AuditEvent(agent="orchestrator", action="cache_hit", trace_id=trace_id,
                               output_data={"similarity": round(cached.similarity, 4),
                                            "source_trace_id": cached.source_trace_id,
                                            "cache_id": cached.cache_id})
                )
                steps.append({
                    "agent": "semantic_cache", "status": "cache_hit", "method": "cosine_similarity",
                    "detail": f"similarity {cached.similarity:.3f} ≥ threshold "
                              f"{policy.answer_cache.similarity_threshold} — served from cache, "
                              f"pipeline skipped entirely",
                })
                return AnswerResult(
                    answer=cached.answer, confidence=cached.confidence, status=cached.status,
                    citations=[Citation(**c) for c in cached.citations], trace_id=trace_id,
                    steps=steps,
                )
            if lookup.similarity is not None:
                steps.append({
                    "agent": "semantic_cache", "status": "cache_miss", "method": "cosine_similarity",
                    "detail": f"closest cached question was {lookup.similarity:.3f} similar — below "
                              f"threshold {policy.answer_cache.similarity_threshold}, running full pipeline",
                })
            else:
                steps.append({
                    "agent": "semantic_cache", "status": "cache_miss", "method": "cosine_similarity",
                    "detail": "cache is empty — running full pipeline",
                })

        try:
            result = self._answer_inner(query, trace_id, policy, steps)
        except ModelUnavailableError as exc:
            write_event(
                AuditEvent(agent="orchestrator", action="model_unavailable", trace_id=trace_id,
                           output_data={"error": str(exc)})
            )
            return AnswerResult(
                answer=(
                    "Both the primary and fallback local models are unavailable right now "
                    "(check `ollama list` / `brew services list`). No answer was generated — "
                    "this is a degraded/unavailable response, not a guess."
                ),
                confidence=0.0, status="model_unavailable", trace_id=trace_id, steps=steps,
            )

        # STORE is independent of LOOKUP (policy.answer_cache.enabled above): a demo can disable
        # lookups to force the full pipeline every time while the cache keeps accumulating data
        # in the background, so switching lookups back on later has something to hit against.
        if policy.answer_cache.store_enabled:
            self.cache.store(
                query=query, answer=result.answer, confidence=result.confidence,
                status=result.status,
                citations=[c.__dict__ for c in result.citations],
                source_trace_id=result.trace_id,
                query_embedding=query_embedding,
            )
            if result.status in ("answered", "answered_with_caveat"):
                embed_note = "embedding reused from the earlier lookup" if query_embedding is not None \
                    else "lookup was off, so this needed its own embedding call"
                result.steps.append({
                    "agent": "semantic_cache", "status": "cache_stored", "method": None,
                    "detail": f"this answer cached ({embed_note}) for future matches ≥ "
                              f"{policy.answer_cache.similarity_threshold} similarity",
                })
        return result

    def _answer_inner(self, query: str, trace_id: str, policy, steps: list[dict]) -> AnswerResult:
        # PLAN
        plan = self._plan(query, trace_id, steps)

        if not plan.get("needs_retrieval", True):
            return AnswerResult(
                answer=(
                    "That doesn't look like a question about the indexed documents. "
                    "Ask about something in the knowledge base and I'll look it up."
                ),
                confidence=1.0, status="answered", trace_id=trace_id, steps=steps,
            )

        sub_queries = plan.get("sub_queries") or [query]
        search_query = sub_queries[0]

        # PRIMARY LOOP: hybrid retrieval + evidence validation, bounded retries
        nodes, evaluation = [], {"verdict": "insufficient"}
        attempt, max_attempts = 0, 1 + policy.max_iterations
        while attempt < max_attempts:
            attempt += 1
            with _tracer().start_as_current_span(
                "agent.retrieval", attributes={"agent.key": "retrieval", "agent.attempt": attempt}
            ):
                nodes = self.retriever.retrieve(search_query)
            steps.append({
                "agent": "retrieval", "status": "ran", "method": None,
                "detail": f"hybrid retrieval (attempt {attempt}): {len(nodes)} chunks for \"{search_query[:80]}\"",
            })
            evaluation = self._evaluate_evidence(query, nodes, trace_id, steps)
            if evaluation.get("verdict") in ("supported", "partially_supported"):
                break
            if attempt < max_attempts:
                search_query = f"{query} {search_query}"

        # AGENT-LEVEL RECOVERY (§18): the primary loop exhausted its retries without success —
        # try genuinely different retrieval strategies, not just repeating the same search.
        if evaluation.get("verdict") not in ("supported", "partially_supported") and policy.fallback.enabled:
            for strategy_name, strategy_nodes in (
                ("lexical_only", self.retriever.retrieve_lexical_only(query)),
                ("relaxed_hybrid", self.retriever.retrieve_relaxed(query)),
            ):
                write_event(
                    AuditEvent(agent="recovery_agent", action=strategy_name, trace_id=trace_id,
                               output_data={"n_nodes": len(strategy_nodes)})
                )
                steps.append({
                    "agent": "recovery_agent", "status": "ran", "method": strategy_name,
                    "detail": f"recovery strategy '{strategy_name}': {len(strategy_nodes)} chunks",
                })
                if not strategy_nodes:
                    continue
                evaluation = self._evaluate_evidence(query, strategy_nodes, trace_id, steps)
                nodes = strategy_nodes
                if evaluation.get("verdict") in ("supported", "partially_supported"):
                    break

        if evaluation.get("verdict") not in ("supported", "partially_supported") or not nodes:
            if nodes:
                return self._degrade_to_retrieval_only(
                    nodes, trace_id, attempt, evaluation.get("verdict", "insufficient"), steps
                )
            return AnswerResult(
                answer=(
                    "I couldn't find sufficient evidence in the indexed documents to answer "
                    "this reliably."
                ),
                confidence=0.0, status="abstained", trace_id=trace_id, iterations=attempt, steps=steps,
            )

        # CONTEXT TRIM: the Evidence Validator already judged which retrieved passages actually
        # support an answer (usable_passage_ids) -- previously computed and then ignored, so
        # Synthesis and Citation & Quality always saw every retrieved passage regardless of
        # what was actually validated. Narrowing to just the validated subset (when the
        # validator ran and returned a real, non-empty subset) means a shorter prompt into the
        # two most expensive remaining calls, and citations that only ever point at passages
        # that were actually judged usable. The smart_bypass path already marks every passage
        # usable (see _evaluate_evidence), so this is a no-op there, not a second bypass.
        usable_ids = evaluation.get("usable_passage_ids")
        if isinstance(usable_ids, list) and usable_ids:
            filtered = [nodes[i] for i in usable_ids if isinstance(i, int) and 0 <= i < len(nodes)]
            if filtered and len(filtered) < len(nodes):
                steps.append({
                    "agent": "evidence_validator", "status": "ran", "method": "context_trim",
                    "detail": f"narrowed {len(nodes)} retrieved passages to {len(filtered)} validated "
                              f"ones before synthesis",
                })
            if filtered:
                nodes = filtered

        evidence_text = _passages_block(nodes)

        # SYNTHESIZE
        draft = self._run_task("synthesizer", "synthesis", trace_id, steps, query=query, evidence=evidence_text)

        # VERIFY & SCORE CONFIDENCE
        quality_raw = self._run_task(
            "citation_quality", "citation_quality", trace_id, steps,
            query=query, evidence=evidence_text, draft_answer=draft,
        )
        quality = extract_json(quality_raw, fallback={"confidence": 0.5, "needs_correction": False})
        try:
            confidence = float(quality.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        # v1.1 contract (see prompts/validation/citation_quality.yaml): the model only writes
        # corrected_answer when it actually changed something — saves regenerating the full
        # answer text on every call, which was the single largest latency cost in the pipeline.
        final_answer = draft
        if quality.get("needs_correction") and isinstance(quality.get("corrected_answer"), str):
            final_answer = quality["corrected_answer"]
        elif isinstance(quality.get("final_answer"), str):
            # backward-compat: tolerate a model that still returns the old key shape
            final_answer = quality["final_answer"]

        citations = _citations_from_nodes(nodes)

        # HUMAN-IN-THE-LOOP (§17)
        if policy.human_review.enabled:
            risk = assess_risk(query, evaluation.get("verdict", "insufficient"), confidence)
            if risk.level == "high":
                review_id = queue_for_review(trace_id, query, final_answer, confidence, risk)
                write_event(
                    AuditEvent(agent="human_review", action="withheld_high_risk", trace_id=trace_id,
                               output_data={"review_id": review_id, "reasons": risk.reasons})
                )
                steps.append({
                    "agent": "human_review", "status": "ran", "method": None,
                    "detail": f"high risk — withheld pending human review ({'; '.join(risk.reasons)})",
                })
                return AnswerResult(
                    answer=(
                        "This question was routed for human review before an answer is released "
                        f"({'; '.join(risk.reasons)}). Reference: review_id={review_id}. "
                        "Check `GET /review-queue` or ask a reviewer to approve it."
                    ),
                    confidence=confidence, status="pending_human_review", trace_id=trace_id,
                    review_id=review_id, steps=steps,
                )
            if risk.level == "medium":
                review_id = queue_for_review(trace_id, query, final_answer, confidence, risk)
                write_event(
                    AuditEvent(agent="human_review", action="flagged_medium_risk", trace_id=trace_id,
                               output_data={"review_id": review_id, "reasons": risk.reasons})
                )
                steps.append({
                    "agent": "human_review", "status": "ran", "method": None,
                    "detail": f"medium risk — answered, flagged for after-the-fact review ({'; '.join(risk.reasons)})",
                })

        output_check = guardrails.check_output(final_answer, confidence, policy.answer.qualify_threshold)
        if not output_check.allowed:
            return AnswerResult(
                answer=(
                    "I couldn't find sufficient evidence in the indexed documents to answer "
                    "this reliably."
                ),
                confidence=confidence, status="abstained", trace_id=trace_id, iterations=attempt, steps=steps,
            )

        if confidence >= policy.answer.auto_threshold:
            status = "answered"
        else:
            status = "answered_with_caveat"
            final_answer += (
                "\n\n_Note: this answer is based on partial or lower-confidence evidence — "
                "verify against the source documents for anything decision-critical._"
            )

        return AnswerResult(
            answer=final_answer, confidence=confidence, status=status,
            citations=citations, trace_id=trace_id, iterations=attempt, steps=steps,
        )
