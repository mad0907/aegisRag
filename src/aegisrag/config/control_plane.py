"""The AI control plane (§21 of the design doc): ONE loader for ALL agent behavior policy.

Orchestrator, guardrails and the human-review queue all read `get_policy()` — nothing about how
the agents behave (loop bounds, confidence thresholds, what forces human review, whether a
fallback model is used) is hard-coded in Python or scattered across modules. Change behavior by
editing agent_policy.yaml; every audit_log row stamps `policy_version` so a behavior change is
traceable to the exact run that started using it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

POLICY_PATH = Path(__file__).parent / "agent_policy.yaml"


class RetrievalPolicy(BaseModel):
    min_score: float = 0.70
    top_k: int = 6


class AnswerPolicy(BaseModel):
    auto_threshold: float = 0.85
    qualify_threshold: float = 0.60


class HumanReviewPolicy(BaseModel):
    enabled: bool = True
    confidence_threshold: float = 0.60
    high_risk_keywords: list[str] = []


class FallbackPolicy(BaseModel):
    enabled: bool = True
    max_attempts: int = 2
    secondary_model_enabled: bool = True


class SmartBypassPolicy(BaseModel):
    """Not every step needs an LLM call — where a cheap, deterministic signal already answers
    the question with high confidence, skip the LLM and use the signal directly. Toggle this
    off (e.g. on GPU/cloud hardware where LLM calls are cheap and fast) to run every step
    through the model for maximum reasoning quality; leave it on for latency-constrained local
    CPU inference. Every bypass decision is still written to the audit log."""
    enabled: bool = True
    # Skip the Query Planner LLM call: a regex greeting/thanks check handles the "no retrieval
    # needed" case, and every other query defaults to needs_retrieval=True with sub_queries=[query]
    # — which is what the LLM planner already returned in every case observed during testing.
    planner_heuristic: bool = True
    # Skip the Evidence Validator LLM call when the top retrieved passage's cosine similarity is
    # already at or above this threshold — treat it as "supported" using the number retrieval
    # already computed, instead of paying an LLM call to re-confirm what the number already shows.
    # Measured on the real corpus: on-topic queries scored 0.73-0.77, off-topic queries scored
    # 0.37-0.45 — 0.60 sits in the gap with margin on both sides.
    evidence_similarity_threshold: float = 0.60


class AnswerCachePolicy(BaseModel):
    """Semantic cache: a new query's embedding is compared against past queries; a close-enough
    match (cosine similarity >= threshold) returns the cached answer directly — no retrieval, no
    agent LLM calls at all. Only genuinely-answered results are ever cached (never an abstain,
    a pending-review, or a degraded response)."""
    enabled: bool = True
    similarity_threshold: float = 0.92


class AgentPolicy(BaseModel):
    version: str = "1.0"
    max_iterations: int = 3
    retrieval: RetrievalPolicy = RetrievalPolicy()
    answer: AnswerPolicy = AnswerPolicy()
    human_review: HumanReviewPolicy = HumanReviewPolicy()
    fallback: FallbackPolicy = FallbackPolicy()
    smart_bypass: SmartBypassPolicy = SmartBypassPolicy()
    answer_cache: AnswerCachePolicy = AnswerCachePolicy()


@lru_cache
def get_policy() -> AgentPolicy:
    data = yaml.safe_load(POLICY_PATH.read_text())
    return AgentPolicy.model_validate(data)
