# Ticket Failure Handling — Canonical Reference

This document is the single source of truth for how callers should handle a
`failed` status returned by the two-phase `submit_task` / `resolve_ticket`
pattern.  Four skills reference it:

- [`skills/review/references/phase3-triage.md`](../review/references/phase3-triage.md) — review cycle task creation
- [`skills/unblock/SKILL.md`](../unblock/SKILL.md) — unblock non-blocker queuing
- [`skills/escalation-watcher/SKILL.md`](../escalation-watcher/SKILL.md) — escalation suggestion queuing
- [`skills/orchestrate/SKILL.md`](../orchestrate/SKILL.md) — PRD-decomposition task writing

---

## Two-phase submit pattern (recap)

```
ticket = submit_task(title=..., description=..., metadata=...)
resolve = resolve_ticket(ticket=ticket, project_root=...)
```

`resolve["status"]` is one of `"created"`, `"combined"`, or `"failed"`.
When `"failed"`, `resolve["reason"]` names the failure class.

---

## Retryable reasons

### `server_restart`

The task server restarted between the `submit_task` and `resolve_ticket` calls.
The original ticket is **dead** — it exists only in memory and was lost.

**Policy:** retry the full `submit_task + resolve_ticket` pair exactly **once**
with the **same metadata**.  Do not retry more than once; if the second attempt
also returns `failed`, treat it as terminal and apply caller-specific handling.

**Dedup caveat:** if the caller does not supply `(escalation_id,
suggestion_hash)` in the `submit_task` metadata, the R4 idempotency gate
(see §R4 below) does not fire on retry.  The curator may still merge the
duplicate into an existing task via `"combined"`, but this is not guaranteed.

---

### `timeout`

The `resolve_ticket` call hit its timeout limit (default 115 s); the worker
may still be processing the original ticket.

**Policy:**
1. First, retry `resolve_ticket(ticket=same_ticket, ...)` with the **original
   ticket id** — the worker may finish during the retry window.
2. If that retry returns `"created"` or `"combined"`, the submission succeeded.
3. If that retry returns `"failed"` with reason `unknown_ticket` or `expired`
   (the ticket has since been swept), fall through to the `server_restart`
   policy: re-submit the full pair once with the same metadata.
4. Apply the same dedup caveat as `server_restart` if re-submit becomes
   necessary.

---

## Terminal reasons

The following reasons are **not retryable**.  Apply caller-specific handling
(skip / surface / record) and move on.

| Reason | Meaning |
|--------|---------|
| `unknown_ticket` | Ticket id not found — already swept or never existed |
| `server_closed` | Task server shut down cleanly; no new work accepted |
| `expired` | Ticket TTL elapsed before the curator processed it |

Each caller file documents what "caller-specific handling" means in its context
(skip the finding, surface to user, record in report, etc.).

---

## R4 idempotency gate

`task_interceptor._check_escalation_idempotency` (implementation:
`fused-memory/src/fused_memory/middleware/task_interceptor.py:1154`) short-circuits
a `submit_task` call and returns the existing task id when the same
`(escalation_id, suggestion_hash)` pair has already been processed.  This is
the **R4 gate**.

**Conditions for R4 to fire:**
- Both `escalation_id` and `suggestion_hash` must be **non-empty strings** in
  the `submit_task` metadata.
- The pair must match a previously recorded entry in the interceptor's
  idempotency log (see `task_interceptor.py:1221-1231`).

Callers that naturally carry an escalation id (e.g. escalation-watcher when
processing suggestions) get R4 for free.  Callers that do not (PRD
decomposition, unblock triage) need to synthesize a stable pair if they want
R4 protection on retry.

---

## Synthesising `(escalation_id, suggestion_hash)` for callers without a native escalation

When a caller doesn't have a natural `escalation_id`, it can opt into the R4
gate by deriving a **stable, content-addressed pair** from its submission
payload.  The recipe below is taken from
`skills/escalation-watcher/SKILL.md` (the `cleanup_needed` section), which
already demonstrates this pattern:

```python
import hashlib

# escalation_id — a non-empty string that identifies the logical source of
# this submission, stable across retries.  Use a human-readable prefix plus
# a stable identity token (e.g. task id, PRD slug, or finding fingerprint).
escalation_id = f"<source>-<stable-identity>"   # e.g. "prd-decomp-task-42"

# suggestion_hash — sha256 of a stable payload, truncated to 16 hex chars.
# The payload should be the same on every retry for the same logical work item.
suggestion_hash = hashlib.sha256("<stable-payload>".encode()).hexdigest()[:16]

# Then pass both in submit_task metadata:
submit_task(
    title=...,
    description=...,
    metadata={
        ...,
        "escalation_id": escalation_id,
        "suggestion_hash": suggestion_hash,
    },
)
```

**Which callers benefit:**
- **PRD-decomposition** (`skills/orchestrate/SKILL.md`) — use the task title +
  PRD slug as the stable payload; prefix with `"prd-decomp-"`.
- **Unblock triage** (`skills/unblock/SKILL.md`) — use the non-blocker
  description + parent task id as the stable payload; prefix with
  `"unblock-triage-"`.

Adding this pair changes retry behaviour from "the curator may or may not
de-duplicate" to "the R4 gate guarantees exactly-once task creation".
