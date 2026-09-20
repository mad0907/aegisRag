from aegisrag.guardrails.human_review import assess_risk


def test_high_risk_keyword_forces_high_regardless_of_confidence():
    risk = assess_risk("Does this compliance policy require termination?", "supported", 0.99)
    assert risk.level == "high"
    assert any("compliance" in r for r in risk.reasons)


def test_contradictory_evidence_is_high_risk():
    risk = assess_risk("What does the document say?", "contradictory", 0.9)
    assert risk.level == "high"


def test_low_confidence_is_medium_risk():
    risk = assess_risk("What does the document say about AI safety?", "supported", 0.5)
    assert risk.level == "medium"


def test_partially_supported_is_medium_risk():
    risk = assess_risk("What does the document say about AI safety?", "partially_supported", 0.95)
    assert risk.level == "medium"


def test_clean_high_confidence_query_is_low_risk():
    risk = assess_risk("What does the document say about AI safety?", "supported", 0.95)
    assert risk.level == "low"
    assert risk.reasons == []
