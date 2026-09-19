"""The agent decision loop (§4 of the design doc):

QUESTION -> PLAN -> RETRIEVE -> ASSESS EVIDENCE
                                     |
                     insufficient? --+-- (<= max retries) --> REFINE QUERY --> RETRIEVE
                                     | supported
                                     v
                               SYNTHESIZE -> VERIFY & SCORE CONFIDENCE
                                     |
                     confidence >= auto_threshold      -> ANSWER
                     qualify_threshold <= conf < auto   -> ANSWER + caveat
                     confidence < qualify_threshold      -> ABSTAIN / human review
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from crewai import Task

from aegisrag.agents.definitions import build_agents
from aegisrag.agents.json_utils import extract_json
from aegisrag.audit.hashchain import AuditEvent, write_event
from aegisrag.config.prompts import get_registry
from aegisrag.config.settings import get_settings
from aegisrag.guardrails import guardrails
from aegisrag.retrieval.hybrid_retriever import HybridPGRetriever


@dataclass
class Citation:
    source_file: str
    page_no: int | None
    section_path: str | None


@dataclass
class AnswerResult:
    answer: str
    confidence: float
    status: str  # "answered" | "answered_with_caveat" | "abstained" | "declined"
    citations: list[Citation] = field(default_factory=list)
    trace_id: str = ""
    iterations: int = 0


class AegisRAGOrchestrator:
    def __init__(self):
        self.settings = get_settings()
        self.agents = build_agents()
        self.prompts = get_registry()
        self.retriever = HybridPGRetriever(top_k=6)

    def _run_task(self, agent_key: str, prompt_name: str, trace_id: str, **prompt_vars) -> str:
        prompt = self.prompts.get(prompt_name)
        rendered = prompt.render(**prompt_vars)
        task = Task(
            description=rendered,
            expected_output="Follow the format instructions in the prompt exactly.",
            agent=self.agents[agent_key],
        )
        output = task.execute_sync()
        write_event(
            AuditEvent(
                agent=agent_key,
                action=prompt_name,
                trace_id=trace_id,
                model=self.settings.ollama_primary_model,
                prompt_version=f"{prompt.id}@{prompt.version}",
                input_data={"vars": {k: str(v)[:500] for k, v in prompt_vars.items()}},
                output_data={"raw": output.raw[:2000]},
            )
        )
        return output.raw

    def answer(self, query: str) -> AnswerResult:
        trace_id = str(uuid.uuid4())

        input_check = guardrails.check_input(query)
        if not input_check.allowed:
            return AnswerResult(
                answer=f"Request declined: {input_check.reason}",
                confidence=0.0,
                status="declined",
                trace_id=trace_id,
            )

        # PLAN
        plan_raw = self._run_task("planner", "query_planner", trace_id, query=query)
        plan = extract_json(plan_raw, fallback={"needs_retrieval": True, "sub_queries": [query]})

        if not plan.get("needs_retrieval", True):
            return AnswerResult(
                answer=(
                    "That doesn't look like a question about the indexed documents. "
                    "Ask about something in the knowledge base and I'll look it up."
                ),
                confidence=1.0,
                status="answered",
                trace_id=trace_id,
            )

        sub_queries = plan.get("sub_queries") or [query]
        search_query = sub_queries[0]

        nodes = []
        verdict = "insufficient"
        max_attempts = 1 + self.settings.agent_max_iterations
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            nodes = self.retriever.retrieve(search_query)
            passages_text = "\n\n".join(
                f"[{i}] ({n.node.metadata.get('source_file')}, "
                f"p.{n.node.metadata.get('page_no')}): {n.node.get_content()}"
                for i, n in enumerate(nodes)
            ) or "(no passages retrieved)"

            guardrails.check_retrieved_content([n.node.get_content() for n in nodes])

            eval_raw = self._run_task(
                "evidence_validator",
                "evidence_validator",
                trace_id,
                query=query,
                passages=passages_text,
            )
            evaluation = extract_json(eval_raw, fallback={"verdict": "insufficient"})
            verdict = evaluation.get("verdict", "insufficient")

            if verdict in ("supported", "partially_supported"):
                break
            if attempt < max_attempts:
                search_query = f"{query} {search_query}"  # broaden, don't just repeat

        if verdict not in ("supported", "partially_supported") or not nodes:
            return AnswerResult(
                answer=(
                    "I couldn't find sufficient evidence in the indexed documents to answer "
                    "this reliably."
                ),
                confidence=0.0,
                status="abstained",
                trace_id=trace_id,
                iterations=attempt,
            )

        evidence_text = "\n\n".join(
            f"[{i}] ({n.node.metadata.get('source_file')}, "
            f"p.{n.node.metadata.get('page_no')}): {n.node.get_content()}"
            for i, n in enumerate(nodes)
        )

        # SYNTHESIZE
        draft = self._run_task(
            "synthesizer", "synthesis", trace_id, query=query, evidence=evidence_text
        )

        # VERIFY & SCORE CONFIDENCE
        quality_raw = self._run_task(
            "citation_quality",
            "citation_quality",
            trace_id,
            query=query,
            evidence=evidence_text,
            draft_answer=draft,
        )
        quality = extract_json(
            quality_raw, fallback={"confidence": 0.5, "final_answer": draft}
        )
        try:
            confidence = float(quality.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        final_answer = quality.get("final_answer", draft)
        if not isinstance(final_answer, str):
            # Small models occasionally nest the answer (e.g. {"text": "..."}) instead of
            # returning a plain string — fall back to the synthesis draft rather than crash.
            final_answer = draft

        citations = [
            Citation(
                source_file=n.node.metadata.get("source_file"),
                page_no=n.node.metadata.get("page_no"),
                section_path=n.node.metadata.get("section_path"),
            )
            for n in nodes
        ]

        output_check = guardrails.check_output(
            final_answer, confidence, self.settings.answer_qualify_threshold
        )
        if not output_check.allowed:
            return AnswerResult(
                answer=(
                    "I couldn't find sufficient evidence in the indexed documents to answer "
                    "this reliably."
                ),
                confidence=confidence,
                status="abstained",
                trace_id=trace_id,
                iterations=attempt,
            )

        if confidence >= self.settings.answer_auto_threshold:
            status = "answered"
        else:
            status = "answered_with_caveat"
            final_answer += (
                "\n\n_Note: this answer is based on partial or lower-confidence evidence — "
                "verify against the source documents for anything decision-critical._"
            )

        return AnswerResult(
            answer=final_answer,
            confidence=confidence,
            status=status,
            citations=citations,
            trace_id=trace_id,
            iterations=attempt,
        )
