# ADR-006: SHA-256 hash-chain audit log in Postgres, not an external immutable log service

## Status
Accepted — implemented (`src/aegisrag/audit/hashchain.py`), integration-tested (a test writes an
event, tampers with its stored row directly via SQL, and confirms `verify_chain()` reports the
tamper — `tests/integration/test_audit_hashchain.py`).

## Context
The design goal is a *tamper-evident* record of every agent decision. A managed immutable-log
service (e.g. a cloud provider's, or a dedicated ledger database) would give append-only storage
guarantees at the infrastructure level.

## Decision
Every `audit_log` row's `event_hash` is `SHA256(row content + previous row's event_hash)` —
computed and verified in `audit/hashchain.py`, stored in an ordinary (if append-oriented)
Postgres table. `verify_chain()` walks the table in `created_at` order and recomputes every hash.

## Alternatives considered
- **A cloud KMS-backed immutable log** (e.g. GCP Cloud Logging + Cloud KMS signing): the correct
  production answer, but requires a cloud environment this assessment doesn't have. Documented
  as the target in `doc/12-cloud-reference-architecture.md` rather than implemented here.
- **No hashing, just an append-only table with restricted UPDATE/DELETE grants**: relies entirely
  on database-level access control (a DBA with sufficient privilege can still edit a row
  undetected); the hash chain adds a second, independent integrity signal that doesn't depend on
  privilege escalation *not* happening.

## Consequences
- **Stated explicitly, not oversold**: SHA-256 chaining makes tampering *detectable* after the
  fact. It does **not** by itself make the log *immutable* — an attacker with UPDATE privilege on
  `audit_log` can still edit a row; they just can't do it without `verify_chain()` catching it on
  the next run. True immutability additionally needs append-only storage permissions, WAL-level
  audit, or a write-once backing store — none of which this local Postgres container provides out
  of the box. The cloud write-up covers where that would come from (Cloud KMS + immutable Cloud
  Storage retention policies).
- A real, fixed bug found during this build: the timestamp used inside the hash must be the exact
  same value persisted to `created_at`, or `verify_chain()` can never recompute a matching hash
  from what's stored. The implementation stamps `datetime.now(timezone.utc)` once and passes it
  explicitly into the `INSERT`, rather than relying on Postgres's own `DEFAULT now()`.
