"""Integration test against the real Postgres — the human-review queue's write/list/decide cycle."""
from aegisrag.guardrails.human_review import RiskAssessment, decide, list_queue, queue_for_review


def test_queue_list_and_decide_roundtrip():
    risk = RiskAssessment(level="high", reasons=["query touches high-risk topic: compliance"])
    review_id = queue_for_review(
        trace_id="test-trace-review", query="test query", draft_answer="test draft",
        confidence=0.9, risk=risk,
    )

    pending = list_queue("pending")
    assert any(r["review_id"] == review_id for r in pending)
    row = next(r for r in pending if r["review_id"] == review_id)
    assert row["risk_level"] == "high"
    assert row["status"] == "pending"

    ok = decide(review_id, "approved", note="looks fine")
    assert ok

    approved = list_queue("approved")
    assert any(r["review_id"] == review_id for r in approved)

    # A second decision on an already-decided row should be a no-op (not pending anymore).
    ok_again = decide(review_id, "rejected")
    assert not ok_again
