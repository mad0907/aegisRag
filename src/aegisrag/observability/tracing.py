"""Wires OpenTelemetry tracing to Arize Phoenix (§19 §21 — every inference call traced).

Instruments LlamaIndex directly and LiteLLM (which CrewAI's LLM calls run through), so both the
retrieval layer and every agent's LLM call show up as spans under one trace per request.
"""
from __future__ import annotations

import logging

from phoenix.otel import register

from aegisrag.config.settings import get_settings

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing(project_name: str = "aegisrag") -> None:
    global _initialized
    if _initialized:
        return

    settings = get_settings()
    tracer_provider = register(
        project_name=project_name,
        endpoint=settings.phoenix_collector_endpoint,
        protocol="grpc",
        auto_instrument=False,
    )

    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:
        logger.exception("Failed to instrument LlamaIndex for tracing")

    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor

        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:
        logger.exception("Failed to instrument LiteLLM for tracing")

    _initialized = True
    logger.info(
        "Tracing initialized -> %s (Phoenix UI: %s)",
        settings.phoenix_collector_endpoint,
        settings.phoenix_ui_url,
    )
