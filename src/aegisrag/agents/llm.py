from crewai import LLM

from aegisrag.config.settings import get_settings


def get_llm(temperature: float = 0.1, max_tokens: int | None = None) -> LLM:
    settings = get_settings()
    return LLM(
        model=f"ollama/{settings.ollama_primary_model}",
        base_url=settings.ollama_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_fallback_llm(temperature: float = 0.1, max_tokens: int | None = None) -> LLM:
    """Model-level fallback (§18): a secondary, smaller model tried when the primary model
    errors or times out. Never used unless the primary call actually fails."""
    settings = get_settings()
    return LLM(
        model=f"ollama/{settings.ollama_fallback_model}",
        base_url=settings.ollama_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_light_llm(temperature: float = 0.1, max_tokens: int | None = None) -> LLM:
    """model_tiering (agent_policy.yaml): a third, independent tier from primary/fallback --
    used for the short structured-JSON agents (Planner, Evidence Validator, Citation & Quality)
    when tiering is enabled, never as an error-recovery path."""
    settings = get_settings()
    return LLM(
        model=f"ollama/{settings.ollama_light_model}",
        base_url=settings.ollama_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
