# AFK A1: Make cap-hit a patient wait, not a blocking escalation

## Problem

When all six accounts (G,F,E,C,B,D) are simultaneously capped, `invoke_with_cap_retry` raises `AllAccountsCappedException` after 20 retries OR a 3600s deadline (whichever fires first). The exception escalates to L1 (human). Currently three reify tasks (1237/1238/1239) sit blocked at "All accounts capped: 2 retries in 119000s" (~33 hours and counting). Anthropic's weekly cap can leave us capped for >48h, well past the existing deadline.

## Solution

Remove the deadline and retry-count ceiling for the cap-hit branch. Just keep awaiting `gate.wait_for_open()` indefinitely (with a 14-day sanity bound). The asyncio coroutine costs essentially nothing while waiting — no subprocess running, no MCP held, no agent process alive. When the gate opens, the natural retry inside `invoke_with_cap_retry` resumes the API call and the task continues.

Releasing the worker slot ("park" machinery) was considered and rejected: with all accounts capped, there is no spare capacity for a freed slot to serve. 12% of zero is zero.

## Concrete changes

In `shared/src/shared/cli_invoke.py`:

- Make `_DEFAULT_MAX_CAP_RETRIES = 20` apply only to the auth-failed and unexpected-error branches; cap-hit retries no longer count against this ceiling.
- Make `_DEFAULT_CAP_RETRY_DEADLINE_SECS` 14 days for cap-hit (sanity-only). Other failure modes keep the existing 3600s.
- Add periodic structured log line every ~10 minutes: `{"event": "cap_wait", "task_id": <>, "elapsed_s": <>, "soonest_open_at": <>, "next_probe_in_s": <>}` so journalctl shows liveness during long waits.

In `orchestrator/src/orchestrator/workflow.py:950-957`:

- The `AllAccountsCappedException → blocking escalation` path triggers only if the 14-day sanity bound is exceeded.
- When it does fire, escalate to L0 (steward-retryable, NOT human) with `suggested_action="cap_wait_exceeded_sanity_bound"`.

In `fused-memory/src/fused_memory/middleware/ticket_janitor.py`:

- Verify the worker-liveness reaper treats "waiting on cap" as alive rather than stuck `in_progress`. If the heuristic trips, add a heartbeat from the cap-wait loop (e.g., touch a per-worker liveness file every 5min).

## Acceptance criteria

- A task experiencing all-accounts-capped does NOT escalate to human; the await-loop continues
- A 60h synthetic cap-window test (mock UsageGate): task waits the full 60h then resumes naturally
- 14-day sanity-bound test: task escalates as L0 (not L1) after the bound is exceeded
- journalctl shows periodic `cap_wait` log lines during a long wait
- Currently-blocked tasks (1237/1238/1239 or their successors): unblock automatically once accounts uncap

## Risks

- **UsageGate misreports gate state (false-open)** — task wakes, immediately re-caps, returns to waiting. Harmless loop.
- **Anthropic API extended outage** — 14-day bound triggers L0; steward retries.
- **Worker-liveness reaper kills cap-waiting workers as "stuck"** — fix or work around per above.

## Out of scope

Cap-park / wake-hook / worker-release machinery — explicitly rejected.

---

## Per-caller cap-wait policy (post-1365 audit, task 1401)

> **Implementation note:** The "Concrete changes" bullets above refer to
> `_DEFAULT_MAX_CAP_RETRIES` and `_DEFAULT_CAP_RETRY_DEADLINE_SECS`, which
> were removed in task 1401 (no production caller passed them; they silently
> no-op'd after 1365).  The cap-wait mechanism today is controlled solely by
> `cap_wait_sanity_secs=` on each `invoke_with_cap_retry` call site.

The authoritative per-caller policy table lives as a comment block adjacent to `_DEFAULT_CAP_WAIT_SANITY_SECS` in [`shared/src/shared/cli_invoke.py`](../shared/src/shared/cli_invoke.py).
