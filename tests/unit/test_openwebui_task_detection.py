from aegisrag.api.main import ChatMessage, _cheap_task_response, _detect_openwebui_task, _extract_real_question


def test_real_question_is_not_detected_as_a_task():
    messages = [ChatMessage(role="user", content="What does the document say about AI safety?")]
    assert _detect_openwebui_task(messages) is None


def test_openwebui_title_task_is_detected():
    messages = [ChatMessage(
        role="user",
        content="### Task:\nGenerate a concise, 3-5 word title with an emoji for this chat.\n\n### Chat History:\nWhat does the document say about AI safety?",
    )]
    assert _detect_openwebui_task(messages) == "title"


def test_openwebui_tags_task_is_detected():
    messages = [ChatMessage(
        role="user",
        content="### Task:\nGenerate 1-3 broad tags categorizing the main themes of the chat history.",
    )]
    assert _detect_openwebui_task(messages) == "tags"


def test_cheap_title_response_is_short_json():
    messages = [ChatMessage(role="user", content="What does the document say about AI safety and moral status?")]
    resp = _cheap_task_response("title", messages)
    assert resp.startswith('{"title"')
    assert "moral status" not in resp or len(resp) < 80  # truncated, not the full question


def test_title_derived_from_embedded_chat_history_not_the_task_instruction():
    # The real shape OpenWebUI sends: the task instruction AND the actual conversation are both
    # inside one message's content. The derived title must come from the question, not the
    # instruction text itself (a real bug found while testing this before the fix).
    messages = [ChatMessage(
        role="user",
        content=(
            "### Task:\nGenerate a concise, 3-5 word title with an emoji for the chat history.\n\n"
            "### Chat History:\nUSER: What does the document say about AI safety?"
        ),
    )]
    extracted = _extract_real_question(messages)
    assert extracted == "What does the document say about AI safety?"

    resp = _cheap_task_response("title", messages)
    assert "Task" not in resp
    assert "Generate" not in resp
    assert resp.startswith('{"title": "What does the document say about')
