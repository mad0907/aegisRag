"""Prompt registry — externalized, versioned prompts loaded from /prompts (§5 of the design doc)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Template

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@dataclass
class Prompt:
    id: str
    version: str
    description: str
    template: str
    input_variables: list[str]

    def render(self, **kwargs) -> str:
        missing = set(self.input_variables) - set(kwargs)
        if missing:
            raise ValueError(f"Missing prompt variables for {self.id}: {missing}")
        return Template(self.template).render(**kwargs)


class PromptRegistry:
    def __init__(self, root: Path = PROMPTS_DIR):
        self._root = root
        self._cache: dict[str, Prompt] = {}

    def get(self, name: str) -> Prompt:
        if name not in self._cache:
            matches = list(self._root.rglob(f"{name}.yaml"))
            if not matches:
                raise FileNotFoundError(f"No prompt file found for '{name}' under {self._root}")
            data = yaml.safe_load(matches[0].read_text())
            self._cache[name] = Prompt(
                id=data["id"],
                version=str(data["version"]),
                description=data.get("description", ""),
                template=data["template"],
                input_variables=data.get("input_variables", []),
            )
        return self._cache[name]


_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
