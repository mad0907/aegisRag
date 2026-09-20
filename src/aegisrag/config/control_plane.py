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


class AgentPolicy(BaseModel):
    version: str = "1.0"
    max_iterations: int = 3
    retrieval: RetrievalPolicy = RetrievalPolicy()
    answer: AnswerPolicy = AnswerPolicy()
    human_review: HumanReviewPolicy = HumanReviewPolicy()
    fallback: FallbackPolicy = FallbackPolicy()


@lru_cache
def get_policy() -> AgentPolicy:
    data = yaml.safe_load(POLICY_PATH.read_text())
    return AgentPolicy.model_validate(data)
