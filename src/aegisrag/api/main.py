from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aegisrag.agents.orchestrator import AegisRAGOrchestrator
from aegisrag.config.settings import get_settings
from aegisrag.database.db import get_connection
from aegisrag.guardrails import human_review
from aegisrag.ingestion.pipeline import ingest_directory
from aegisrag.observability.tracing import init_tracing

app = FastAPI(title="AegisRAG", version="0.1.0")

_orchestrator: AegisRAGOrchestrator | None = None


@app.on_event("startup")
def _startup():
    init_tracing()
    global _orchestrator
    _orchestrator = AegisRAGOrchestrator()


def _get_orchestrator() -> AegisRAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AegisRAGOrchestrator()
    return _orchestrator


# ---- OpenAI-compatible chat endpoint (what OpenWebUI talks to) ----------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "aegisrag"
    messages: list[ChatMessage]
    stream: bool = False


# OpenWebUI issues internal "task" calls (title generation, tag generation, follow-up
# suggestions) through this same OpenAI-compatible endpoint — indistinguishable from a real
# question except for a distinctive instruction OpenWebUI itself writes into the prompt. Left
# unhandled, every one of these silently ran the full 5-agent pipeline, roughly doubling the
# cost of the first message in every new chat for zero user-facing benefit (a chat title). This
# check is deliberately narrow — it only matches OpenWebUI's own well-known task markers, so an
# ambiguous real question always falls through to the real pipeline rather than risk a
# misclassified answer.
_TASK_MARKERS = ("### task:", "generate a concise", "generate 1-3 broad tags", "generate a title")


def _detect_openwebui_task(messages: list[ChatMessage]) -> str | None:
    if not messages:
        return None
    text = messages[-1].content.lower()
    if not any(marker in text for marker in _TASK_MARKERS):
        return None
    if "tag" in text:
        return "tags"
    if "title" in text:
        return "title"
    return None


def _extract_real_question(messages: list[ChatMessage]) -> str:
    """OpenWebUI's task call often embeds the actual conversation *inside* a single message's
    content, after a '### Chat History:' marker, rather than as separate messages — so "the
    first user message" can BE the task instruction itself. Prefer any message that isn't the
    task instruction; within the task-instruction message, prefer text after a known marker."""
    for m in messages:
        content = m.content.strip()
        lowered = content.lower()
        if any(marker in lowered for marker in _TASK_MARKERS):
            # This message IS the task instruction — look inside it for the embedded history.
            for split_marker in ("### chat history:", "chat history:"):
                idx = lowered.find(split_marker)
                if idx != -1:
                    after = content[idx + len(split_marker):].strip()
                    for line in after.splitlines():
                        line = line.strip().lstrip("-*").strip()
                        if line and ":" in line[:12]:  # "USER: ..." / "User: ..."
                            line = line.split(":", 1)[1].strip()
                        if line:
                            return line
            continue  # nothing usable found inside the instruction message, try the next one
        if content:
            return content
    return "New chat"


def _cheap_task_response(task: str, messages: list[ChatMessage]) -> str:
    source = _extract_real_question(messages)
    words = source.strip().split()
    short = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
    if task == "tags":
        return '{"tags": ["general"]}'
    return f'{{"title": "{short}"}}'


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    task = _detect_openwebui_task(req.messages)
    if task:
        content = _cheap_task_response(task, req.messages)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "aegisrag": {"status": "openwebui_task_shortcircuit", "task": task},
        }

    user_messages = [m.content for m in req.messages if m.role == "user"]
    query = user_messages[-1] if user_messages else ""

    result = _get_orchestrator().answer(query)

    citations_text = ""
    if result.citations:
        seen = set()
        lines = []
        for c in result.citations:
            key = (c.source_file, c.page_no)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {c.source_file}, p.{c.page_no}" + (f" ({c.section_path})" if c.section_path else ""))
        citations_text = "\n\n**Sources**\n" + "\n".join(lines)

    content = result.answer + citations_text

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "aegisrag": {
            "trace_id": result.trace_id,
            "status": result.status,
            "confidence": result.confidence,
            "iterations": result.iterations,
            "review_id": result.review_id,
        },
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "aegisrag", "object": "model", "owned_by": "aegisrag"}],
    }


# ---- Management endpoints -------------------------------------------------------------------

@app.get("/health")
def health():
    settings = get_settings()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_ok = True
    except Exception as exc:  # noqa: BLE001
        db_ok = False
        return JSONResponse(status_code=503, content={"status": "unhealthy", "db_error": str(exc)})
    return {"status": "ok", "db": db_ok, "primary_model": settings.ollama_primary_model}


@app.get("/documents")
def list_documents():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.document_id, d.source_file, d.document_version, d.status,
                       d.page_count, d.ingested_at, count(c.chunk_id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                GROUP BY d.document_id
                ORDER BY d.ingested_at DESC
                """
            )
            rows = cur.fetchall()
    return [
        {
            "document_id": str(r[0]),
            "source_file": r[1],
            "document_version": r[2],
            "status": r[3],
            "page_count": r[4],
            "ingested_at": r[5].isoformat(),
            "chunk_count": r[6],
        }
        for r in rows
    ]


@app.post("/ingestion")
def trigger_ingestion():
    kb_dir = Path(__file__).resolve().parents[3] / "data" / "knowledge_base"
    results = ingest_directory(kb_dir)
    return [
        {
            "source_file": r.source_file,
            "document_id": r.document_id,
            "chunks_written": r.chunks_written,
            "skipped": r.skipped,
            "reason": r.reason,
        }
        for r in results
    ]


class Feedback(BaseModel):
    trace_id: str
    query: str
    answer: str
    rating: str  # "up" | "down"
    error_category: str | None = None


@app.post("/feedback")
def submit_feedback(fb: Feedback):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (trace_id, query, answer, rating, error_category)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (fb.trace_id, fb.query, fb.answer, fb.rating, fb.error_category),
            )
        conn.commit()
    return {"status": "recorded"}


# ---- Human-in-the-loop review queue (§17) ----------------------------------------------------

@app.get("/review-queue")
def get_review_queue(status: str = "pending"):
    if status not in ("pending", "approved", "rejected"):
        return JSONResponse(status_code=400, content={"error": "status must be pending|approved|rejected"})
    return human_review.list_queue(status)


class ReviewDecision(BaseModel):
    decision: str  # "approved" | "rejected"
    note: str | None = None


@app.post("/review-queue/{review_id}/decision")
def decide_review(review_id: str, body: ReviewDecision):
    if body.decision not in ("approved", "rejected"):
        return JSONResponse(status_code=400, content={"error": "decision must be approved|rejected"})
    updated = human_review.decide(review_id, body.decision, body.note)
    if not updated:
        return JSONResponse(status_code=404, content={"error": "no pending review with that id"})
    return {"review_id": review_id, "status": body.decision}


# ---- Audit chain integrity (§19) -------------------------------------------------------------

@app.get("/audit/verify")
def verify_audit():
    from aegisrag.audit.hashchain import verify_chain

    ok, message = verify_chain()
    return {"ok": ok, "message": message}
