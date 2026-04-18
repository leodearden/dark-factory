# WP-B: Fire-and-forget reconciliation journal

## Mission
Decouple `event_buffer.push` from the MCP write hot path. A SQLite failure must never leave canonical state (`tasks.json`, Graphiti, Mem0) mutated but unjournalled, and must never bubble up as an MCP-tool error to the caller.

## Defect context
In `fused-memory/src/fused_memory/middleware/task_interceptor.py`, every mutating method runs the canonical write first (`tm.add_task` / `tm.update_task` / …) then awaits `self.buffer.push(event)`. `buffer.push` hits SQLite (`reconciliation.db`). When SQLite is locked (a documented recurring state — see project memory "SQLite lock requires process restart"), `push` raises `OperationalError: database is locked`, the exception unwinds through the MCP boundary in `server/tools.py`, and the caller sees `{"error": "database is locked"}` — even though the canonical write already succeeded. Agents retry → duplicate tasks (e.g. 2026-04-17's reify 1953/1954/1955 triplet).

This is one of the two root causes of that incident. The other is the SQLite lock itself (WP-C).

## Design decision (agreed)
**Fire-and-forget journal.** The MCP tool call returns success as soon as the canonical mutation succeeds. Event journalling becomes eventually consistent via a background drainer.

Semantics:
- Write path: `await tm.<mutation>(...)` → success → `queue.put_nowait(event)` (non-blocking) → return to caller.
- Background drainer: owns the SQLite connection, pulls events from the in-memory queue, calls `buffer.push_sync(event)`. Retries on `OperationalError` with exponential backoff (cap ~30s per attempt).
- Queue is bounded (e.g. 10 000 events). Overflow → emit WARN and invoke the WP-D escalation policy (not in scope here — emit a TODO / hook that WP-D will wire up).
- Graceful shutdown: on SIGTERM, drainer gets a bounded flush window (~10s), then dumps remainder to a persistent dead-letter file for later replay.

## Scope

### 1. In-memory queue + drainer
Create `fused-memory/src/fused_memory/reconciliation/event_queue.py` (or similar; check if something like this already exists first). Responsibilities:
- An `asyncio.Queue` with a capacity.
- A drainer coroutine that:
  - `await queue.get()`
  - Attempts `event_buffer.push` (reuse existing EventBuffer.push; do not reimplement).
  - On success: mark done, update `last_successful_commit_ts` (watchdog hook for WP-C).
  - On `aiosqlite.OperationalError` (or similar retriable): exponential backoff, retry same event. Do NOT drop.
  - On non-retriable / schema errors: log ERROR, drop event to dead-letter file.
- A `close()` / shutdown protocol: bounded flush window, then dump remainder to dead-letter.
- A `stats()` method: queue depth, last_commit_ts, retries-in-flight — drives WP-D policy and dashboards.

### 2. Call-site migration
Every `await self.buffer.push(event)` in `task_interceptor.py` becomes `self.event_queue.enqueue(event)` (non-blocking).

Grep `self.buffer.push` in `fused-memory/src/fused_memory/middleware/task_interceptor.py` — expect ~5-8 call sites (add_task, add_subtask, set_task_status, update_task, remove_task, add_dependency, remove_dependency). Confirm full list.

Also check for push call sites elsewhere — there may be direct pushes from elsewhere (e.g., bulk ops). Grep the whole `fused-memory/src/`.

### 3. Lifecycle wiring
Start the drainer with the fused-memory server lifespan (`fused-memory/src/fused_memory/server/main.py`). Stop it gracefully on shutdown.

The `TaskInterceptor` constructor probably needs to accept the event queue (or a fuse-point on the EventBuffer). Design the interface to be testable — inject-friendly.

### 4. Dead-letter file
Path: `data/reconciliation/event_dead_letter.jsonl` (or follow existing dead-letter conventions — grep `dead_letter` in the tree; there's already a `replay_dead_letters` MCP tool, so a pattern exists — conform to it).

On startup, optionally: drainer can try to replay the dead-letter file first (append-mode). Don't make this automatic in this WP — too much scope. Just leave a clean file and a replay helper.

### 5. EventBuffer changes
`EventBuffer.push` currently commits inline. After this WP, it's called only from the drainer — so keep its current behavior (one INSERT + one commit per event). Don't batch; that's a later optimization.

Consider: does `EventBuffer.push` need any change? Probably minor:
- Accept a retry count for logging purposes.
- Differentiate retriable (OperationalError) vs non-retriable.

## Files
- `fused-memory/src/fused_memory/middleware/task_interceptor.py` — all push call sites
- `fused-memory/src/fused_memory/reconciliation/event_buffer.py` — may need minor changes
- `fused-memory/src/fused_memory/server/main.py` — lifespan hooks
- New: `fused-memory/src/fused_memory/reconciliation/event_queue.py`
- Tests under `fused-memory/tests/`

## Tests to add
1. **Hot path immunity**: inject a failure into `EventBuffer.push` (raise OperationalError). `add_task` via interceptor still returns a valid result in <500ms. Event remains in queue for retry.
2. **Drainer recovery**: failure for N seconds, then recovery. Assert: queued events are eventually persisted to SQLite.
3. **Queue overflow**: fill the queue past capacity. Assert: WARN logged, overflow hook fired, oldest events preferentially dropped (or the policy you chose — document in plan).
4. **Graceful shutdown**: enqueue 100 events, kick off shutdown. Assert: flushed within bounded time or dumped to dead-letter. No events lost.
5. **Order preservation**: a single agent's events for the same task arrive in enqueue order at the DB. (Not strict global ordering — that's too strong — just per-project or per-agent preservation if cheap. Document your actual guarantee in the plan.)

## Out of scope
- Fixing the underlying SQLite lock leak (that's WP-C — but this WP **enables** the system to survive while WP-C investigates).
- Changing event schema.
- Changing reconciliation harness behavior (drain cadence, etc.).
- Backlog threshold / escalation (WP-D consumes the `stats()` this WP exposes).

## Acceptance
- [ ] Every mutating MCP tool returns success within 500ms even under continuous SQLite write failure.
- [ ] No event silently dropped except by bounded-queue-overflow (logged) or shutdown-timeout (dead-lettered).
- [ ] Existing tests pass.
- [ ] New tests cover injection, recovery, overflow, shutdown.
- [ ] Drainer exposes `stats()` surface useful to WP-C (watchdog) and WP-D (backlog policy).

## Dependencies
- **WP-A complete** (healthy fused-memory to test against).
- **WP-E should be merged first** if possible — avoids rebase pain since both edit `task_interceptor.py`. If WP-E isn't merged, coordinate with the parent session.

## Workflow for this session

1. Read this briefing.
2. Search fused-memory for prior context: "event_buffer push", "reconciliation drainer", "fire and forget journal", "dead letter" — especially any prior decisions about batching or ordering guarantees.
3. Read:
   - `fused-memory/src/fused_memory/middleware/task_interceptor.py` (full, or at minimum every method with `buffer.push`)
   - `fused-memory/src/fused_memory/reconciliation/event_buffer.py` (full)
   - `fused-memory/src/fused_memory/server/main.py` (lifespan patterns)
   - Any existing queue / drainer pattern elsewhere in the codebase (grep for asyncio.Queue, asyncio.create_task in fused-memory/src)
4. Write `plans/wp-b-plan.md`: queue capacity, backoff schedule, shutdown flush window, dead-letter file format, ordering guarantee, interface between TaskInterceptor and the queue, migration order for call sites.
5. Create branch `wp-b/fire-and-forget-journal` off main (rebase on top of WP-E's branch if it's not merged yet — coordinate via the merge plan).
6. Implement. Suggested chunking:
   - Commit 1: event_queue.py with drainer + tests
   - Commit 2: migrate task_interceptor call sites
   - Commit 3: server lifespan wiring
   - Commit 4: dead-letter + graceful shutdown tests
7. Run `cd fused-memory && uv run pytest tests/ -x`. Fix.
8. Run a manual end-to-end: start the server locally (or against a test instance), induce a SQLite failure (mock or by holding a transaction externally), call add_task, verify it returns success fast and the event drains after recovery.
9. `/reflect` with notes on: backoff decisions, ordering tradeoffs, gotchas found in asyncio queue usage, implications for WP-C/WP-D.
10. Emit JSON summary:
    ```json
    {
      "wp": "B",
      "branch": "wp-b/fire-and-forget-journal",
      "commits": ["<sha>", ...],
      "queue_capacity": N,
      "backoff_schedule": "...",
      "call_sites_migrated": N,
      "tests_added": [...],
      "tests_passed": N,
      "reflection_saved": true,
      "notes_for_wp_c": "...",
      "notes_for_wp_d": "..."
    }
    ```

## Permission mode
`--dangerously-skip-permissions`. Do NOT push, modify git config, merge to main, restart fused-memory systemd, or touch reify tasks.json.
