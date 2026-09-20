"""Human-in-the-loop (§17 of the design doc): risk-based escalation, not "review everything"
or "review nothing".

    LOW    -> auto-answer
    MEDIUM -> answer returned, but logged to review_queue as flagged (after-the-fact review)
    HIGH   -> answer withheld from the caller, held in review_queue until a human decides
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aegisrag.config.control_plane import get_policy
from aegisrag.database.db import get_connection


@dataclass
class RiskAssessment:
    level: str  # "low" | "medium" | "high"
    reasons: list[str] = field(default_factory=list)


def assess_risk(query: str, verdict: str, confidence: float) -> RiskAssessment:
    policy = get_policy().human_review
    reasons = []

    query_lower = query.lower()
    matched_keywords = [kw for kw in policy.high_risk_keywords if kw in query_lower]
    if matched_keywords:
        reasons.append(f"query touches high-risk topic: {', '.join(matched_keywords)}")

    if verdict == "contradictory":
        reasons.append("evidence agents found contradictory passages")

    if matched_keywords or verdict == "contradictory":
        return RiskAssessment(level="high", reasons=reasons)

    if confidence < policy.confidence_threshold:
        reasons.append(f"confidence {confidence:.2f} below human-review threshold {policy.confidence_threshold}")
        return RiskAssessment(level="medium", reasons=reasons)

    if verdict == "partially_supported":
        reasons.append("evidence only partially supported the answer")
        return RiskAssessment(level="medium", reasons=reasons)

    return RiskAssessment(level="low", reasons=[])


def queue_for_review(
    trace_id: str, query: str, draft_answer: str, confidence: float, risk: RiskAssessment
) -> str:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO review_queue (trace_id, query, draft_answer, confidence, risk_level, risk_reasons)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING review_id
                """,
                (trace_id, query, draft_answer, confidence, risk.level, risk.reasons),
            )
            review_id = cur.fetchone()[0]
        conn.commit()
    return str(review_id)


def decide(review_id: str, decision: str, note: str | None = None) -> bool:
    """decision: 'approved' | 'rejected'. Returns False if no pending row matched."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE review_queue
                SET status = %s, reviewer_note = %s, reviewed_at = now()
                WHERE review_id = %s AND status = 'pending'
                """,
                (decision, note, review_id),
            )
            updated = cur.rowcount > 0
        conn.commit()
    return updated


def list_queue(status: str = "pending") -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT review_id, trace_id, query, draft_answer, confidence, risk_level,
                       risk_reasons, status, reviewer_note, created_at, reviewed_at
                FROM review_queue
                WHERE status = %s
                ORDER BY created_at ASC
                """,
                (status,),
            )
            rows = cur.fetchall()
    return [
        {
            "review_id": str(r[0]),
            "trace_id": r[1],
            "query": r[2],
            "draft_answer": r[3],
            "confidence": r[4],
            "risk_level": r[5],
            "risk_reasons": r[6],
            "status": r[7],
            "reviewer_note": r[8],
            "created_at": r[9].isoformat(),
            "reviewed_at": r[10].isoformat() if r[10] else None,
        }
        for r in rows
    ]
