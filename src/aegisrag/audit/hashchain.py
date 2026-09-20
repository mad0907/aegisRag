"""Tamper-evident hash-chain audit log (§19 of the design doc).

Each event's hash covers its own content plus the previous event's hash. This makes tampering
DETECTABLE — it does not by itself make the log immutable, which also needs append-only storage
and access controls (stated plainly, not oversold).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from aegisrag.config.control_plane import get_policy
from aegisrag.database.db import get_connection


def _hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class AuditEvent:
    agent: str
    action: str
    trace_id: str | None = None
    document_ids: list[str] = field(default_factory=list)
    model: str | None = None
    prompt_version: str | None = None
    # Defaults to the currently-loaded control-plane version (config/agent_policy.yaml) if not
    # given — every row is stamped with which policy was active, so a behavior change traces
    # back to the run that started using it.
    policy_version: str | None = None
    input_data: dict | None = None
    output_data: dict | None = None


def _get_last_event_hash(cur) -> str:
    cur.execute("SELECT event_hash FROM audit_log ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else "GENESIS"


def write_event(event: AuditEvent) -> str:
    input_hash = _hash(event.input_data or {})
    output_hash = _hash(event.output_data or {})
    policy_version = event.policy_version or get_policy().version
    # Computed once here and stored verbatim in created_at, so verify_chain() can recompute
    # the exact same hash later from what's persisted — not an approximation.
    timestamp = datetime.now(timezone.utc)

    with get_connection() as conn:
        with conn.cursor() as cur:
            previous_hash = _get_last_event_hash(cur)
            event_payload = {
                "trace_id": event.trace_id,
                "agent": event.agent,
                "action": event.action,
                "document_ids": sorted(event.document_ids),
                "model": event.model,
                "prompt_version": event.prompt_version,
                "policy_version": policy_version,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "previous_event_hash": previous_hash,
                "timestamp": timestamp.isoformat(),
            }
            event_hash = _hash(event_payload)

            cur.execute(
                """
                INSERT INTO audit_log
                    (trace_id, agent, action, document_ids, model, prompt_version, policy_version,
                     input_hash, output_hash, previous_event_hash, event_hash, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    event.trace_id,
                    event.agent,
                    event.action,
                    event.document_ids or None,
                    event.model,
                    event.prompt_version,
                    policy_version,
                    input_hash,
                    output_hash,
                    previous_hash,
                    event_hash,
                    timestamp,
                ),
            )
        conn.commit()
    return event_hash


def verify_chain() -> tuple[bool, str]:
    """Walks the chain in order and confirms each event's previous_event_hash matches
    the actual hash of the prior row. Returns (ok, message)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, trace_id, agent, action, document_ids, model, prompt_version,
                       policy_version, input_hash, output_hash, previous_event_hash, event_hash,
                       created_at
                FROM audit_log ORDER BY created_at ASC
                """
            )
            rows = cur.fetchall()

    prev = "GENESIS"
    for row in rows:
        (event_id, trace_id, agent, action, document_ids, model, prompt_version, policy_version,
         input_hash, output_hash, previous_event_hash, event_hash, created_at) = row
        if previous_event_hash != prev:
            return False, f"Broken link at event {event_id}: expected previous={prev}, found={previous_event_hash}"

        doc_ids_str = sorted(str(d) for d in document_ids) if document_ids else []
        recomputed = _hash({
            "trace_id": trace_id,
            "agent": agent,
            "action": action,
            "document_ids": doc_ids_str,
            "model": model,
            "prompt_version": prompt_version,
            "policy_version": policy_version,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "previous_event_hash": previous_event_hash,
            "timestamp": created_at.isoformat(),
        })
        if recomputed != event_hash:
            return False, f"Tampered content at event {event_id}: stored hash does not match recomputed hash"
        prev = event_hash

    return True, f"{len(rows)} events verified"


if __name__ == "__main__":
    ok, message = verify_chain()
    print(("PASS: " if ok else "FAIL: ") + message)
