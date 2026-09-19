"""Integration tests against the real dockerized Postgres — run with `make up` active."""
from aegisrag.audit.hashchain import AuditEvent, verify_chain, write_event
from aegisrag.database.db import get_connection


def test_hash_chain_links_and_verifies():
    write_event(AuditEvent(agent="test_agent", action="test_action_1", trace_id="test-trace-1"))
    write_event(AuditEvent(agent="test_agent", action="test_action_2", trace_id="test-trace-1"))

    ok, message = verify_chain()
    assert ok, message


def test_tampering_is_detected():
    event_hash = write_event(AuditEvent(agent="test_agent", action="tamper_target", trace_id="tamper-test"))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE audit_log SET action = 'tampered_action' WHERE event_hash = %s",
                (event_hash,),
            )
        conn.commit()

    ok, message = verify_chain()
    assert not ok
    assert "Tampered" in message or "Broken link" in message

    # Restore so the chain (and subsequent test runs) stay valid.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE audit_log SET action = 'tamper_target' WHERE event_hash = %s",
                (event_hash,),
            )
        conn.commit()
    ok, _ = verify_chain()
    assert ok
