"""Local 7B models don't always return clean JSON. Two failure modes handled here:
1. Extra prose before/after the JSON block -> extract the first {...} block.
2. Literal (unescaped) newlines inside string values -> repair by escaping them,
   since that's the single most common way small models break otherwise-valid JSON
   when asked for a multi-line "final_answer"-style field.
"""
from __future__ import annotations

import json
import re

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _escape_literal_newlines_in_strings(text: str) -> str:
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def extract_json(raw: str, fallback: dict) -> dict:
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return fallback

    block = match.group(0)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_escape_literal_newlines_in_strings(block))
    except json.JSONDecodeError:
        return fallback
