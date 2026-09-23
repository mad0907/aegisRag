from aegisrag.agents.orchestrator import _GREETING_RE


def test_greeting_regex_matches_common_greetings():
    for text in ["hi", "Hello!", "hey", "thanks", "Thank you.", "ok", "bye"]:
        assert _GREETING_RE.match(text), f"expected {text!r} to match"


def test_greeting_regex_does_not_match_real_questions():
    for text in [
        "What does the document say about AI safety?",
        "hi, what is the moral status of AI according to the paper?",
        "hello there, can you summarize section 3",
    ]:
        assert not _GREETING_RE.match(text), f"expected {text!r} to NOT match"
