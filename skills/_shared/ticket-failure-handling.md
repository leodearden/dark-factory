# Ticket Failure Handling — Canonical Reference

This document is the single source of truth for how callers should handle the
status returned by the two-phase `submit_task` / `resolve_ticket` pattern.  Six
skills reference it:

- [`skills/review/references/phase3-triage.md`](../review/references/phase3-triage.md) — review cycle task creation
- [`skills/unblock/SKILL.md`](../unblock/SKILL.md) — unblock non-blocker queuing
- [`skills/escalation-watcher/SKILL.md`](../escalation-watcher/SKILL.md) — escalation suggestion queuing
- [`skills/orchestrate/SKILL.md`](../orchestrate/SKILL.md) — PRD-decomposition task writing
- [`skills/recon-escalation-watcher/SKILL.md`](../recon-escalation-watcher/SKILL.md) — recon-queue finding triage
- [`skills/escalation-watcher-auto/SKILL.md`](../escalation-watcher-auto/SKILL.md) — auto watcher follow-up filing

---

## Two-phase submit pattern (recap)

```
ticket = submit_task(title=..., description=..., metadata=...)
resolve = resolve_ticket(ticket=ticket, project_root=..., timeout_seconds=60)
```

`timeout_seconds=60` is set below the server default (115 s) so that an expired
wait answers before MCP's ~120 s transport window closes — producing an explicit
open answer instead of a silently dropped result.  60 s leaves ~55 s headroom
below the 120 s MCP transport window to absorb network jitter and
response-serialization overhead; intermediate values like 90 s or 110 s would
leave too little margin.  See [Open statuses](#open-statuses-pending-awaiting_curation).

`resolve["status"]` is one of seven values.  OPEN (not decided yet):
`"pending"` and `"awaiting_curation"`.  TERMINAL: `"created"`, `"combined"`,
`"refused"`, `"cancelled"` and `"failed"`.  Only `created` and `combined` carry a
`task_id`.  When `"failed"`, `resolve["reason"]` names the failure class.

A `refused` ticket is a **success**, not a failure: a deterministic guard
(the cancelled-premise blocklist or the recon premise registry) rejected the
candidate because its premise is dead.  **No task was created and there is no
`task_id` key at all** — not a null one.  Do NOT retry it (a retry re-hits the
same guard), do NOT record a task id for it, and do NOT route it through any of
the failure policies below.  `resolve["reason"]` carries the justification;
record that verbatim wherever you would have recorded the created task id.

---

## Open statuses (`pending`, `awaiting_curation`)

An open answer is **not a decision** — neither a failure nor a creation.  The
ticket is still being curated and nothing on it changed.  It carries `ticket`
(the same ticket id) and a `reason`, and **never a `task_id`**:

| `reason` | Meaning |
|----------|---------|
| `wait_timeout` | The wait expired; the row was re-read and is still open |
| `wait_interrupted` | The waiter was woken while the row was still open (e.g. a server shutdown) |
| `store_unavailable` | The ticket store was closing; after a restart the row reads `failed` / `server_restart` |
| the parked row's own reason | `awaiting_curation`: the curator could not judge it yet (e.g. `curation-deferred: all-accounts-capped`); the ticket janitor retries it until it is created, combined or refused |

**Rule:** call `resolve_ticket` again with the **SAME ticket** and never
re-submit (a second `submit_task` files a duplicate ticket), inside the cap
below.  If the second resolve is still open, record the ticket id where a task
id would go and move on: the verdict lands without further action.  Do not
count an open ticket as created and do not route it through the failure
policies below.

---

## Retryable reasons

### `server_restart`

The task server restarted while the ticket was still `pending` and never
deferred, so the restart failed it.  Tickets are persisted, but this one is
**dead**: no curator will decide it.

**Policy:** retry the full `submit_task + resolve_ticket` pair exactly **once**
with the **same metadata**.  The "once" budget applies to the `submit_task`
re-issue specifically: if the re-submitted ticket answers open, follow the open
rule above (resolve the same ticket again) before escalating further.  Only
declare the submission terminal once the appropriate retry path has been
exhausted.

**Total-submission cap:** at most **two** `submit_task` calls per logical work
item, regardless of which outcome triggers the retry.  The full allowed budget
is: 1 initial submit + optional 1 `resolve_ticket` retry (on an open answer) +
optional 1 re-submit (on `failed` / `server_restart`, or when the re-resolve of
an open ticket returns `failed` / `unknown_ticket` or `failed` /
`server_restart`).  After the second submit, any further failure is terminal —
do not submit a third time.

| Action | Max per logical work item |
|--------|--------------------------|
| `submit_task` | 2 |
| `resolve_ticket` | 2 |

**Dedup caveat:** if the caller does not supply `(escalation_id,
suggestion_hash)` in the `submit_task` metadata, the R4 idempotency gate
(see §R4 below) does not fire on retry.  The curator may still merge the
duplicate into an existing task via `"combined"`, but this is not guaranteed.

---

## Terminal reasons

The following `failed` reasons are **not retryable**.  Apply caller-specific
handling (skip / surface / record) and move on.

| Reason | Meaning |
|--------|---------|
| `unknown_ticket` | Ticket id not found — never existed, or mistyped |
| `ticket_store not configured` | The server has no ticket store; nothing can be filed |
| `worker_dead` | The project's curator worker died with the ticket still pending |
| `curator_failed` | The curator worker crashed while handling the ticket's batch |
| `bad_candidate_json` | The persisted submission could not be parsed |

A `cancelled` ticket (reason `user_cancelled`) is also terminal: it was
withdrawn by `cancel_ticket`, created nothing and carries no `task_id`.

Each caller file documents what "caller-specific handling" means in its context
(skip the finding, surface to user, record in report, etc.).

---

## R4 idempotency gate

`task_interceptor._check_escalation_idempotency` (in
`fused-memory/src/fused_memory/middleware/task_interceptor.py`) short-circuits
a `submit_task` call and returns the existing task id when the same
`(escalation_id, suggestion_hash)` pair has already been processed.  This is
the **R4 gate**.

**Conditions for R4 to fire:**
- Both `escalation_id` and `suggestion_hash` must be **non-empty strings** in
  the `submit_task` metadata.
- The pair must match a previously recorded entry in the interceptor's match
  loop inside `_check_escalation_idempotency`.

Callers that naturally carry an escalation id (e.g. escalation-watcher when
processing suggestions) get R4 for free.  Callers that do not (PRD
decomposition, unblock triage) need to synthesize a stable pair if they want
R4 protection on retry.

---

## Synthesising idempotency metadata for callers that need it

Callers that don't natively supply both `escalation_id` and `suggestion_hash`
can opt into the R4 gate by deriving a **stable, content-addressed** value from
their submission payload.  The right approach depends on what the caller already
has at hand.

**Type requirement:** both `escalation_id` and `suggestion_hash` must be
non-empty `str` values — R4 silently no-ops if either is non-string (the
interceptor's `_check_escalation_idempotency` uses an `isinstance(..., str)`
check).  Coerce numeric ids via `str()` before placing them in metadata.

### Case A — caller has a native `escalation_id`, needs to synthesize `suggestion_hash` only

Use this when the caller already holds a real `escalation_id` from context (for
example, escalation-watcher processing a `cleanup_needed` info item already has
the escalation id; only the suggestion hash must be derived).

**Empty-payload collision hazard:** `hashlib.sha256(b'').hexdigest()[:16]`
always returns the fixed prefix `e3b0c44298fc1c14`.  If the field you hash
(e.g. `escalation['detail']`) may be blank, every blank-detail escalation
produces the same `suggestion_hash` and they all collapse into the same R4
follow-up task.  Always use a fallback chain that terminates at a field
guaranteed to be non-empty.  For `Escalation` objects, `id` is always
`"esc-{task_id}-{seq}"` by the dataclass contract, so it is the safe terminal
fallback:

```python
import hashlib

# stable_payload — first non-empty field; 'id' is the guaranteed-non-empty
# fallback (always "esc-{task_id}-{seq}" by the Escalation dataclass contract).
stable_payload = escalation['detail'] or escalation['summary'] or escalation['id']
suggestion_hash = hashlib.sha256(stable_payload.encode()).hexdigest()[:16]

submit_task(
    title=...,
    description=...,
    metadata={
        ...,
        "escalation_id": escalation_id,   # real id already in scope
        "suggestion_hash": suggestion_hash,
    },
)
```

### Case B — caller has neither `escalation_id` nor `suggestion_hash`

Use this when the caller has no existing escalation context (PRD-decomposition,
unblock-triage).  Both values must be synthesized from stable identifiers.

```python
import hashlib

# escalation_id — a non-empty string that identifies the logical source of
# this submission, stable across retries.  Use a human-readable prefix plus
# a stable identity token (e.g. task id, PRD slug, or finding fingerprint).
escalation_id = f"<source>-<stable-identity>"   # e.g. "prd-decomp-task-42"

# suggestion_hash — sha256 of a stable payload, truncated to 16 hex chars.
# The payload should be the same on every retry for the same logical work item.
suggestion_hash = hashlib.sha256("<stable-payload>".encode()).hexdigest()[:16]

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

**Callers in Case B:**
- **PRD-decomposition** (`skills/orchestrate/SKILL.md`) — use the task title +
  PRD slug as the stable payload; prefix with `"prd-decomp-"`.
- **Unblock triage** (`skills/unblock/SKILL.md`) — use the non-blocker
  description + parent task id as the stable payload; prefix with
  `"unblock-triage-"`.
- **Review-cycle / phase3-triage** (`skills/review/references/phase3-triage.md`) —
  prefix `escalation_id` with `"review-cycle-"` and append the `review_id` (e.g.
  `f"review-cycle-{review_id}"`).  For `suggestion_hash`, sha256 the finding's
  stable key (file path + line number + title concatenated), truncated to 16 hex
  chars.
- **Invariants-gauge findings** (`skills/_shared/reading-the-invariants-gauge.md`) —
  prefix `escalation_id` with `"invariants-"`; the payload, the filing store and the
  handling of an R4 hit on a `done` task are defined there.

Adding idempotency metadata changes retry behaviour from "the curator may or may
not de-duplicate" to "the R4 gate guarantees exactly-once task creation".
