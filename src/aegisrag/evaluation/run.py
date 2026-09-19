"""RAGAs evaluation harness (§7/§20 of the design doc).

Runs the golden Q&A set through the real orchestrator, scores faithfulness / answer relevancy /
context precision / context recall, and compares against a baseline so regressions are visible
(the CI-gate story from the design doc — wiring this into GitHub Actions is a follow-up, this is
the script that gate would call).
"""
from __future__ import annotations

import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from aegisrag.agents.orchestrator import AegisRAGOrchestrator
from aegisrag.config.settings import get_settings

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "evaluation" / "datasets" / "golden"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "evaluation" / "results"
BASELINE_PATH = RESULTS_DIR / "baseline.json"


def _load_golden_set() -> list[dict]:
    cases = []
    for f in sorted(GOLDEN_DIR.glob("*.json")):
        cases.extend(json.loads(f.read_text()))
    return cases


async def _run_case(orch: AegisRAGOrchestrator, case: dict) -> dict:
    result = orch.answer(case["question"])
    contexts = [
        f"{c.source_file} p.{c.page_no}: (retrieved passage)" for c in result.citations
    ] or ["(no context retrieved)"]
    return {
        "question": case["question"],
        "ground_truth": case.get("ground_truth", ""),
        "expected_unanswerable": case.get("expected_unanswerable", False),
        "answer": result.answer,
        "contexts": contexts,
        "status": result.status,
        "confidence": result.confidence,
    }


async def main() -> None:
    settings = get_settings()
    orch = AegisRAGOrchestrator()
    cases = _load_golden_set()
    if not cases:
        print(f"No golden set found under {GOLDEN_DIR}")
        return

    runs = [await _run_case(orch, c) for c in cases]

    client = AsyncOpenAI(base_url=f"{settings.ollama_base_url}/v1", api_key="ollama")
    ragas_llm = llm_factory(model=settings.ollama_primary_model, provider="openai", client=client)
    ragas_embeddings = embedding_factory(
        provider="openai", model=settings.ollama_embed_model, client=client
    )

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    context_precision = ContextPrecisionWithoutReference(llm=ragas_llm)
    context_recall = ContextRecall(llm=ragas_llm)

    scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": [], "context_recall": []}
    per_case = []

    for run in runs:
        if run["expected_unanswerable"]:
            # Abstention correctness, not a faithfulness/relevancy question.
            correct = run["status"] == "abstained"
            per_case.append({**run, "abstention_correct": correct})
            continue

        case_scores = {}
        try:
            case_scores["faithfulness"] = (
                await faithfulness.ascore(
                    user_input=run["question"], response=run["answer"], retrieved_contexts=run["contexts"]
                )
            ).value
            case_scores["answer_relevancy"] = (
                await answer_relevancy.ascore(
                    user_input=run["question"], response=run["answer"], retrieved_contexts=run["contexts"]
                )
            ).value
            case_scores["context_precision"] = (
                await context_precision.ascore(
                    user_input=run["question"], response=run["answer"], retrieved_contexts=run["contexts"]
                )
            ).value
            if run["ground_truth"]:
                case_scores["context_recall"] = (
                    await context_recall.ascore(
                        user_input=run["question"],
                        retrieved_contexts=run["contexts"],
                        reference=run["ground_truth"],
                    )
                ).value
        except Exception as exc:  # noqa: BLE001 — one bad case shouldn't kill the whole eval run
            case_scores["error"] = str(exc)

        for k, v in case_scores.items():
            if k in scores and isinstance(v, (int, float)):
                scores[k].append(v)
        per_case.append({**run, **case_scores})

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.ollama_primary_model,
        "n_cases": len(cases),
        "metrics": {k: (statistics.mean(v) if v else None) for k, v in scores.items()},
        "cases": per_case,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Full results: {out_path}")

    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())["metrics"]
        regressions = []
        for k, v in summary["metrics"].items():
            b = baseline.get(k)
            if v is not None and b is not None and v < b - 0.05:
                regressions.append(f"{k}: {v:.3f} < baseline {b:.3f}")
        if regressions:
            print("REGRESSION DETECTED:\n  " + "\n  ".join(regressions))
        else:
            print("No regression vs. baseline.")
    else:
        BASELINE_PATH.write_text(json.dumps(summary, indent=2, default=str))
        print(f"No baseline existed — this run is now the baseline ({BASELINE_PATH}).")


if __name__ == "__main__":
    asyncio.run(main())
