from crewai import LLM

from aegisrag.config.settings import get_settings


def get_llm(temperature: float = 0.1) -> LLM:
    settings = get_settings()
    return LLM(
        model=f"ollama/{settings.ollama_primary_model}",
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


def get_fallback_llm(temperature: float = 0.1) -> LLM:
    """Model-level fallback (§18): a secondary, smaller model tried when the primary model
    errors or times out. Never used unless the primary call actually fails."""
    settings = get_settings()
    return LLM(
        model=f"ollama/{settings.ollama_fallback_model}",
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )
