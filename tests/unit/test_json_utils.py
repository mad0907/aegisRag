from aegisrag.agents.json_utils import extract_json


def test_clean_json_parses():
    raw = '{"verdict": "supported", "confidence": 0.9}'
    result = extract_json(raw, fallback={})
    assert result == {"verdict": "supported", "confidence": 0.9}


def test_json_with_surrounding_prose():
    raw = 'Here is my answer:\n{"verdict": "supported"}\nHope that helps!'
    result = extract_json(raw, fallback={})
    assert result == {"verdict": "supported"}


def test_json_with_literal_newlines_in_string_is_repaired():
    raw = '{\n  "final_answer": "Line one.\nLine two.\nLine three.",\n  "confidence": 0.8\n}'
    result = extract_json(raw, fallback={"confidence": 0.5})
    assert result["confidence"] == 0.8
    assert result["final_answer"] == "Line one.\nLine two.\nLine three."


def test_unparseable_returns_fallback():
    raw = "not json at all, sorry"
    fallback = {"verdict": "insufficient"}
    assert extract_json(raw, fallback) == fallback


def test_escaped_quotes_still_parse():
    raw = '{"final_answer": "The report says \\"AI is risky\\" on page 3."}'
    result = extract_json(raw, fallback={})
    assert result["final_answer"] == 'The report says "AI is risky" on page 3.'
