"""Workaround for a real upstream bug: ragas==0.4.3's ragas/llms/base.py unconditionally
imports `langchain_community.chat_models.vertexai.ChatVertexAI`, a submodule that
langchain-community>=0.4 removed (moved to the separate `langchain-google-vertexai` package,
which does NOT re-export it under the old path). Since we don't use VertexAI at all (Ollama is
our only provider), the fix is to make those two imports optional — they're only used to extend
an isinstance() check list in `is_multiple_completion_supported`.

Idempotent: safe to run multiple times / after a fresh `pip install`.
Run via: .venv/bin/python scripts/patch_ragas.py
"""
from __future__ import annotations

import re
import sysconfig
from pathlib import Path

OLD_IMPORTS = (
    "from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "from langchain_community.llms import VertexAI\n"
)
NEW_IMPORTS = (
    "try:\n"
    "    from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "    from langchain_community.llms import VertexAI\n"
    "except ImportError:  # patched by scripts/patch_ragas.py — see module docstring\n"
    "    ChatVertexAI = None\n"
    "    VertexAI = None\n"
)

OLD_LIST_ENTRY = "    ChatVertexAI,\n    VertexAI,\n"
NEW_LIST_ENTRY = (
    "    *([ChatVertexAI] if ChatVertexAI is not None else []),\n"
    "    *([VertexAI] if VertexAI is not None else []),\n"
)


def find_ragas_base() -> Path:
    purelib = Path(sysconfig.get_paths()["purelib"])
    path = purelib / "ragas" / "llms" / "base.py"
    if not path.exists():
        raise FileNotFoundError(f"ragas/llms/base.py not found at {path} — is ragas installed?")
    return path


def patch() -> None:
    path = find_ragas_base()
    text = path.read_text()

    if "patched by scripts/patch_ragas.py" in text:
        print(f"Already patched: {path}")
        return

    if OLD_IMPORTS not in text or OLD_LIST_ENTRY not in text:
        raise RuntimeError(
            "ragas/llms/base.py doesn't match the expected shape for this patch — "
            "the ragas version may have changed; check manually."
        )

    text = text.replace(OLD_IMPORTS, NEW_IMPORTS)
    text = text.replace(OLD_LIST_ENTRY, NEW_LIST_ENTRY)
    path.write_text(text)
    print(f"Patched: {path}")


if __name__ == "__main__":
    patch()
