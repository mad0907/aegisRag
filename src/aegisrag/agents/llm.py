from crewai import LLM

from aegisrag.config.settings import get_settings


def get_llm(temperature: float = 0.1) -> LLM:
    settings = get_settings()
    return LLM(
        model=f"ollama/{settings.ollama_primary_model}",
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )
