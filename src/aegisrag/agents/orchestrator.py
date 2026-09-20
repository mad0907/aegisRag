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

import uuid
from dataclasses import dataclass, field

from crewai import Task

from aegisrag.agents.definitions import build_agents
from aegisrag.agents.json_utils import extract_json
from aegisrag.agents.llm import get_fallback_llm, get_llm
from aegisrag.audit.hashchain import AuditEvent, write_event
from aegisrag.config.control_plane import get_policy
from aegisrag.config.prompts import get_registry
from aegisrag.config.settings import get_settings
from aegisrag.guardrails import guardrails
from aegisrag.guardrails.human_review import assess_risk, queue_for_review
from aegisrag.retrieval.hybrid_retriever import HybridPGRetriever


class ModelUnavailableError(Exception):
    """Raised when both the primary and fallback models fail for one call."""


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
    def __init__(self):
        self.settings = get_settings()
        self.agents = build_agents(get_llm())
        self._fallback_agents: dict | None = None  # built lazily, only if ever needed
        self.prompts = get_registry()
        self.retriever = HybridPGRetriever(top_k=get_policy().retrieval.top_k)

    def _get_fallback_agent(self, agent_key: str):
        if self._fallback_agents is None:
            self._fallback_agents = build_agents(get_fallback_llm())
        return self._fallback_agents[agent_key]

    def _run_task(self, agent_key: str, prompt_name: str, trace_id: str, **prompt_vars) -> str:
        policy = get_policy()
        prompt = self.prompts.get(prompt_name)
        rendered = prompt.render(**prompt_vars)
        input_data = {"vars": {k: str(v)[:500] for k, v in prompt_vars.items()}}

        model_used = self.settings.ollama_primary_model
        raw: str | None = None
        primary_error: Exception | None = None

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
                raise ModelUnavailableError(
                    f"Both primary ({self.settings.ollama_primary_model}) and fallback "
                    f"({self.settings.ollama_fallback_model}) models failed for {agent_key}/{prompt_name}"
                ) from fallback_exc

        if raw is None:
            # fallback disabled and primary failed
            raise ModelUnavailableError(str(primary_error))

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
        return raw

    def _evaluate_evidence(self, query: str, nodes, trace_id: str) -> dict:
        passages_text = _passages_block(nodes)
        guardrails.check_retrieved_content([n.node.get_content() for n in nodes])
        eval_raw = self._run_task(
            "evidence_validator", "evidence_validator", trace_id, query=query, passages=passages_text
        )
        return extract_json(eval_raw, fallback={"verdict": "insufficient"})

    def _degrade_to_retrieval_only(self, nodes, trace_id: str, iterations: int, reason: str) -> AnswerResult:
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
        )

    def answer(self, query: str) -> AnswerResult:
        trace_id = str(uuid.uuid4())
        policy = get_policy()

        input_check = guardrails.check_input(query)
        if not input_check.allowed:
            return AnswerResult(
                answer=f"Request declined: {input_check.reason}", confidence=0.0,
                status="declined", trace_id=trace_id,
            )

        try:
            return self._answer_inner(query, trace_id, policy)
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
                confidence=0.0, status="model_unavailable", trace_id=trace_id,
            )

    def _answer_inner(self, query: str, trace_id: str, policy) -> AnswerResult:
        # PLAN
        plan_raw = self._run_task("planner", "query_planner", trace_id, query=query)
        plan = extract_json(plan_raw, fallback={"needs_retrieval": True, "sub_queries": [query]})

        if not plan.get("needs_retrieval", True):
            return AnswerResult(
                answer=(
                    "That doesn't look like a question about the indexed documents. "
                    "Ask about something in the knowledge base and I'll look it up."
                ),
                confidence=1.0, status="answered", trace_id=trace_id,
            )

        sub_queries = plan.get("sub_queries") or [query]
        search_query = sub_queries[0]

        # PRIMARY LOOP: hybrid retrieval + evidence validation, bounded retries
        nodes, evaluation = [], {"verdict": "insufficient"}
        attempt, max_attempts = 0, 1 + policy.max_iterations
        while attempt < max_attempts:
            attempt += 1
            nodes = self.retriever.retrieve(search_query)
            evaluation = self._evaluate_evidence(query, nodes, trace_id)
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
                if not strategy_nodes:
                    continue
                evaluation = self._evaluate_evidence(query, strategy_nodes, trace_id)
                nodes = strategy_nodes
                if evaluation.get("verdict") in ("supported", "partially_supported"):
                    break

        if evaluation.get("verdict") not in ("supported", "partially_supported") or not nodes:
            if nodes:
                return self._degrade_to_retrieval_only(nodes, trace_id, attempt, evaluation.get("verdict", "insufficient"))
            return AnswerResult(
                answer=(
                    "I couldn't find sufficient evidence in the indexed documents to answer "
                    "this reliably."
                ),
                confidence=0.0, status="abstained", trace_id=trace_id, iterations=attempt,
            )

        evidence_text = _passages_block(nodes)

        # SYNTHESIZE
        draft = self._run_task("synthesizer", "synthesis", trace_id, query=query, evidence=evidence_text)

        # VERIFY & SCORE CONFIDENCE
        quality_raw = self._run_task(
            "citation_quality", "citation_quality", trace_id,
            query=query, evidence=evidence_text, draft_answer=draft,
        )
        quality = extract_json(quality_raw, fallback={"confidence": 0.5, "final_answer": draft})
        try:
            confidence = float(quality.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        final_answer = quality.get("final_answer", draft)
        if not isinstance(final_answer, str):
            final_answer = draft

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
                return AnswerResult(
                    answer=(
                        "This question was routed for human review before an answer is released "
                        f"({'; '.join(risk.reasons)}). Reference: review_id={review_id}. "
                        "Check `GET /review-queue` or ask a reviewer to approve it."
                    ),
                    confidence=confidence, status="pending_human_review", trace_id=trace_id,
                    review_id=review_id,
                )
            if risk.level == "medium":
                review_id = queue_for_review(trace_id, query, final_answer, confidence, risk)
                write_event(
                    AuditEvent(agent="human_review", action="flagged_medium_risk", trace_id=trace_id,
                               output_data={"review_id": review_id, "reasons": risk.reasons})
                )

        output_check = guardrails.check_output(final_answer, confidence, policy.answer.qualify_threshold)
        if not output_check.allowed:
            return AnswerResult(
                answer=(
                    "I couldn't find sufficient evidence in the indexed documents to answer "
                    "this reliably."
                ),
                confidence=confidence, status="abstained", trace_id=trace_id, iterations=attempt,
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
            citations=citations, trace_id=trace_id, iterations=attempt,
        )
