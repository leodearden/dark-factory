# WP-B: Fire-and-forget reconciliation journal — Plan

## Goal
Sever the synchronous dependency between MCP-tool writes and the SQLite event buffer. The tool call returns success as soon as the canonical mutation (tasks.json / Graphiti / Mem0) succeeds; journalling becomes an eventually-consistent in-memory hand-off drained by a background task.

## Context (pre-implementation)

`task_interceptor.py` has **8 `await self.buffer.push(event)` call sites** (set_task_status, expand_task, parse_prd, `_add_task_locked`, update_task, `_add_subtask_locked`, remove_task, add_dependency, remove_dependency — so actually 9). One more call site exists in `services/memory_service.py::_emit_event` (memory ops, not task ops). That site is OUT OF SCOPE for WP-B because the MCP memory-write path is not the documented cause of the 2026-04-17 duplicate-task incident. Flagged in the reflection for a future WP.

The existing `EventBuffer.push` is async and hits aiosqlite directly. After this WP, `push` is only invoked by the drainer.

Per the coordinator's note: **WP-E is merged**. Every mutating method in task_interceptor already holds a per-project `asyncio.Lock` around only `tm.<method>` and the no-op guard. `buffer.push` currently sits AFTER the `async with` block in all cases except `set_task_status` (where it's also after) — so the migration is trivially lock-safe: enqueue stays outside the lock.

## Design decisions

### Ordering guarantee
**Per-process FIFO.** A single drainer coroutine consumes from the queue; events reach SQLite in enqueue order. No global-order guarantee across processes (fused-memory runs as a singleton via abstract socket, so this is effectively global today). No per-project ordering guarantee documented — the queue is flat. This is acceptable because:
- The reconciliation consumer (`EventBuffer.drain`) reads by `ORDER BY timestamp`.
- Event timestamps are set at `_make_event` time (before enqueue), not at DB insert time. So even if the drainer reorders slightly under retry, the SQL query restores logical order.

### Queue capacity
**10,000 events.** At the reify project's worst observed burst (≈30 tasks/min triaged), this covers ~5.5 hours of total backlog before overflow. Configurable via a new `reconciliation.event_queue_capacity` config knob (default 10_000).

### Backoff schedule
Retriable errors (OperationalError from aiosqlite — lock or transient IO):
- Initial: 0.1 s
- Multiplier: 2.0 with jitter ±20 %
- Cap: 30 s per attempt
- Retry budget: **unlimited** for transient lock errors (the whole point of the WP)

Non-retriable errors (schema, constraint, anything else):
- Log ERROR with event id + type, send event straight to dead-letter file. Do not retry.

Because WP-C's real root cause (the stuck `git show` subprocess) is NOT a transient lock but a cgroup-pinned hold, retries here won't dislodge it. But the queue buffering means the MCP hot path is never blocked, regardless of lock-root-cause.

### Shutdown flush window
**10 s bounded drain** on close. After 10 s, any remaining events are written to the dead-letter file in one append + fsync.

### Dead-letter file
Path: `{reconciliation.data_dir}/event_dead_letter.jsonl`. Format:
```jsonl
{"event": {...event.model_dump()...}, "reason": "non_retriable|shutdown_timeout|overflow_drop", "attempts": N, "failed_at": "ISO-8601"}
```
Append-only. One event per line. No automatic replay on startup (noted as TODO for WP-D); a manual replay tool / MCP endpoint can be added later.

### Overflow policy
Queue is `asyncio.Queue(maxsize=capacity)`. `enqueue(event)` uses `put_nowait`. If the queue is full:
1. Log WARNING once per 60 s (rate-limited to avoid log spam).
2. Write event to dead-letter (reason="overflow_drop").
3. Increment `overflow_drops` counter in `stats()`.
4. Do NOT block the caller.

This is "oldest preferentially kept, newest dropped." The inverse (drop oldest) is more work for no operational benefit — an operator running `replay_dead_letters` will resurrect them either way.

### Interface

New module: `fused-memory/src/fused_memory/reconciliation/event_queue.py`

```python
class EventQueue:
    def __init__(
        self,
        event_buffer: EventBuffer,
        *,
        dead_letter_path: Path,
        maxsize: int = 10_000,
        retry_initial_seconds: float = 0.1,
        retry_max_seconds: float = 30.0,
        shutdown_flush_seconds: float = 10.0,
    ): ...

    async def start(self) -> None: ...          # spawn drainer task
    async def close(self) -> None: ...          # flush + dead-letter remainder + cancel drainer
    def enqueue(self, event: ReconciliationEvent) -> bool: ...
    def stats(self) -> dict: ...                # queue_depth, last_commit_ts, overflow_drops, dead_letters, retry_in_flight
```

`TaskInterceptor` constructor gains an optional `event_queue: EventQueue | None = None` kwarg.
- If provided: `buffer.push(event)` calls become `self.event_queue.enqueue(event)` (sync, non-blocking).
- If None (legacy / tests that don't wire a queue): fall through to `await self.buffer.push(event)`.

A private helper `_journal(event)` encapsulates the branch so migration is mechanical.

### Migration order
1. Commit 1 — add `event_queue.py` with drainer + unit tests (no integration yet).
2. Commit 2 — wire optional `event_queue` into `TaskInterceptor`, migrate all 9 call sites via `_journal` helper. Existing tests still pass because ``event_queue=None`` falls back to inline push.
3. Commit 3 — wire `EventQueue` into `server/main.py` lifespan (start after `event_buffer.initialize()`, close during `_graceful_shutdown` BEFORE `event_buffer.close()` so the last flush can still reach SQLite).
4. Commit 4 — shutdown + dead-letter integration tests, plus overflow test.

## Tests to add

In `tests/test_event_queue.py`:
1. `test_enqueue_returns_fast_under_push_failure` — mock `EventBuffer.push` to always raise `aiosqlite.OperationalError('database is locked')`; fire 10 enqueues; each returns True in <5 ms; after short sleep, queue depth > 0 (drainer retrying).
2. `test_drainer_recovers_after_transient_failure` — mock push to fail 3 times then succeed; enqueue 5 events; after ~1 s all 5 are in the real SQLite buffer.
3. `test_non_retriable_error_goes_to_dead_letter` — mock push to raise `ValueError`; enqueue 3 events; assert dead-letter file has 3 lines with `reason="non_retriable"`.
4. `test_overflow_writes_to_dead_letter` — tiny queue (maxsize=2); mock push to block forever; enqueue 5; assert 3 dead-lettered with `reason="overflow_drop"` and `stats().overflow_drops == 3`.
5. `test_graceful_shutdown_flushes_within_window` — enqueue 50 events; mock push fast (no failure); call close(); assert all 50 persisted, dead-letter empty.
6. `test_shutdown_timeout_dumps_remainder` — enqueue 20 events; mock push slow (100 ms each); call close(flush_timeout=0.2); assert some events in SQLite AND some in dead-letter with `reason="shutdown_timeout"`.
7. `test_stats_surface` — after drain, `stats()` has all keys with numeric values.

In `tests/test_task_interceptor.py` (extend):
- `test_add_task_hot_path_immunity_with_queue` — fixture builds a queue over a buffer whose push is patched to raise OperationalError; call `add_task`; assert the MCP call returns the task dict (not an exception) within 500 ms.

In `tests/test_server_shutdown.py` (extend):
- `test_event_queue_closed_before_buffer` — assert EventQueue.close is awaited before EventBuffer.close during `_graceful_shutdown`.

## Out-of-scope notes
- SQLite lock root cause is WP-C's job. Our drainer tolerates transient locks but would not recover from a permanently-held lock without operator action (kill the stuck holder, then drained events keep retrying indefinitely — fine).
- Memory-service `_emit_event` call site is not migrated (different hot path, different incident scope).
- Dead-letter auto-replay on startup deferred to WP-D.

## Files touched
- NEW: `fused-memory/src/fused_memory/reconciliation/event_queue.py`
- NEW: `fused-memory/tests/test_event_queue.py`
- `fused-memory/src/fused_memory/middleware/task_interceptor.py` — 9 call sites via new `_journal` helper; constructor param; no lock logic changes.
- `fused-memory/src/fused_memory/server/main.py` — construct + start + pass + close in both branches of the reconciliation-enabled/disabled flow.
- `fused-memory/src/fused_memory/config/schema.py` — add `event_queue_capacity` knob.
- `fused-memory/tests/test_task_interceptor.py` — one extension test.
- `fused-memory/tests/test_server_shutdown.py` — one extension test.

## Notes for WP-C
The drainer's retry logic is designed for transient aiosqlite locks. It will NOT help the "stuck git subprocess pinning the WAL" class that WP-A identified on 2026-04-17 — those are unbounded in duration and must be killed. But the drainer DOES make the problem no longer user-facing: the MCP caller never sees the lock error and so never retries, so no duplicate tasks. WP-C's job shifts from "urgent incident" to "chronic operational hygiene."

## Notes for WP-D
`EventQueue.stats()` exposes the signals WP-D needs:
- `queue_depth` — input to backlog-threshold policy.
- `last_commit_ts` — watchdog signal (if not advancing, drainer wedged).
- `overflow_drops`, `dead_letters` — escalation triggers.
- `retry_in_flight` — distinguishes "backlogged" (queue growing, draining) from "wedged" (queue growing, no drainage).

Suggested WP-D hook: a callback on the `EventQueue` invoked on overflow + on `last_commit_ts` staleness. Not wired in this WP; WP-D owns the policy.
