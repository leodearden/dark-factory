# WP-C: SQLite lock health & watchdog — Plan

## Audit findings

### aiosqlite callers (fused-memory/src/)

| File | DB file | Per-conn PRAGMAs | Cursor discipline | Tx discipline |
|------|---------|------------------|-------------------|---------------|
| `reconciliation/event_buffer.py` | `reconciliation.db` | WAL, busy_timeout=5000 | `async with` ✓ | implicit auto-commit; one rollback in `mark_run_active` |
| `reconciliation/journal.py` | **same** `reconciliation.db` | **NONE** (no busy_timeout, no WAL) | `async with` ✓ | none |
| `services/write_journal.py` | `write_journal.db` | WAL only (no busy_timeout) | `async with` ✓ | logged-only try/except |
| `services/durable_queue.py` | `write_queue.db` | (verified separately) | mixed | n/a here |
| `services/planned_episode_registry.py` | `planned_episodes.db` | WAL, busy_timeout=5000 | `async with` ✓ | none |

### Critical finding 1 — Two connections to `reconciliation.db`
`EventBuffer` and `ReconciliationJournal` both open separate `aiosqlite.Connection`
instances pointing at the same physical `reconciliation.db` file.

- `EventBuffer.initialize()` sets `journal_mode=WAL` and `busy_timeout=5000`.
- `ReconciliationJournal.initialize()` sets **neither**. WAL mode persists at the
  file level (so it sticks once EventBuffer has set it), but `busy_timeout` is
  per-connection. Default is 0 ms — so any transient lock from the EventBuffer
  side surfaces as immediate `SQLITE_BUSY` ("database is locked") on the
  journal side. The harness heartbeat (every 60s) and run-start writes
  (`start_run`, `update_run_stage_reports`, `complete_run`) all flow through
  this lock-impatient connection.

This is the first-class intra-process contention vector. WP-A's stuck-`git show`
finding was a different mechanism (cgroup-pinned fd inheritance) — both can
hit the same observable symptom.

### Critical finding 2 — Git subprocess timeout discipline
`task_file_committer.py` spawns four git subprocesses (`git show`, `git add`,
`git diff --cached`, `git commit`) inside `_do_commit`. Each uses
`asyncio.wait_for(proc.communicate(), timeout=10)`.

On `TimeoutError`:
- The exception unwinds.
- `proc.communicate()` task is cancelled, but **the OS subprocess keeps running**
  (per project memory `feedback_subprocess_cancel_pattern.md`).
- Without `proc.terminate()` / `proc.kill()` + `proc.wait()` reaping, the orphan
  is owned by the fused-memory cgroup, inherits any non-CLOEXEC fds (potentially
  including SQLite handles), and python's child-watcher thread keeps a reference.

This matches WP-A's report exactly: a `git show HEAD:.taskmaster/tasks/tasks.json`
process running for 4d6h as a child of fused-memory.

The same gap exists on `CancelledError` — graceful shutdown cancels the
committer task and orphans whatever subprocess was in flight.

### Other subprocess sites (noted, NOT in scope)
- `reconciliation/judge.py` and `reconciliation/agent_loop.py`: bare `proc.kill()`
  on TimeoutError (no `await proc.wait()`). Same orphan-thread risk but for
  Claude CLI, not the SQLite hot path. Out of scope per briefing — flagged for
  future WP.
- `reconciliation/verify.py`: grep / git_log / git_show with no terminate at all.
  Same shape, same out-of-scope reasoning (verifier agent, not hot path).

## Changes by file

### 1. `task_file_committer.py` — subprocess hygiene (commit 1)
Add a private `_run_subprocess(cmd, *, timeout, **kwargs)` helper that:
- Spawns via `asyncio.create_subprocess_exec`.
- Awaits `proc.communicate()` under `wait_for(timeout)`.
- On `TimeoutError`: `proc.terminate()` → `await asyncio.shield(asyncio.wait_for(proc.wait(), timeout=2))` → `proc.kill()` if still alive → `await asyncio.shield(proc.wait())` → re-raise.
- On `CancelledError`: same cleanup, then re-raise.
- Returns `(returncode, stdout_bytes, stderr_bytes)`.

Replace the four inline spawn blocks with helper calls. The existing `try/except`
around the stale-snapshot check stays — but the helper guarantees the subprocess
is always reaped before that exception fires.

### 2. `journal.py` — PRAGMA parity (commit 2)
Add `await self._db.execute('PRAGMA busy_timeout=5000')` and
`await self._db.execute('PRAGMA synchronous=NORMAL')` immediately after
`aiosqlite.connect`. Rationale:
- `busy_timeout=5000`: matches EventBuffer; lets the connection wait 5s for a
  lock instead of failing immediately. With WP-B's drainer-as-sole-writer in the
  buffer path, the only contention is journal vs drainer; 5s is plenty for
  single-row inserts.
- `synchronous=NORMAL`: safe under WAL, ~3× faster commits, reduces the window
  where the journal connection holds the WAL writer lock. SQLite documentation
  explicitly recommends NORMAL with WAL; only durability against OS crash
  (not application crash) is reduced. Acceptable for an audit log.

No `wal_autocheckpoint` change — default (1000 pages ≈ 4 MB) is fine; the
observed 4.2 MB stall was a symptom of the lock leak, not WAL growth policy.

Apply the same two PRAGMAs to `event_buffer.py` (`synchronous=NORMAL` only;
busy_timeout already set) and to `write_journal.py` (`busy_timeout=5000`,
`synchronous=NORMAL`).

### 3. `journal.py` + `write_journal.py` + `event_buffer.py` — explicit transaction discipline (commit 2)
Audit shows aiosqlite's default isolation behavior implicitly opens a transaction
on the first write and commits on `await db.commit()`. The current code relies
on this. Risk: an exception between the first write and the commit leaves the
transaction open, which holds the writer lock until the connection is closed.

For each public write method, wrap the body in:
```python
try:
    await db.execute(...)
    await db.commit()
except BaseException:
    with contextlib.suppress(Exception):
        await db.rollback()
    raise
```

This is mechanical and small. `BaseException` so `CancelledError` also rolls back
(otherwise the next commit on this connection inherits the open transaction).

### 4. New: `reconciliation/sqlite_watchdog.py` — drainer health watchdog (commit 3)
A coroutine that wakes every `check_interval_secs` (default 30) and inspects
`EventQueue.stats()`:

- Wedged condition: `now - last_commit_ts > stall_threshold_secs (default 120)`
  AND `queue_depth > 0`.
- On wedge: emit ERROR log with structured payload:
  - `queue_depth`, `retry_in_flight`, `events_committed`, `last_commit_ts`,
    `dead_letters`, `overflow_drops`.
  - Last 20 SQL operations attempted by the drainer (ring buffer added to
    EventQueue — see commit-3 note below).
  - `asyncio.all_tasks()` summary (count + names) as a coarse stack signal.
  - PID of the holder via `lsof reconciliation.db` if available — best effort
    (subprocess discipline applies; 5 s timeout, terminate-on-fail).
- On wedge: invoke an optional `wedge_callback` (asyncio.Event-backed) so WP-D
  can subscribe.

Add a tiny ring buffer in `EventQueue` to track the last N committed/attempted
event types — `(monotonic_ts, event_id, event_type, status)` — exposed via a
new `recent_ops()` accessor. ~20 entries, no allocation hot-path concern.

Watchdog is launched from `server/main.py` when `event_queue` is constructed
(reconciliation enabled branch only) and cancelled in `_graceful_shutdown`
between `task_interceptor.close()` and `event_queue.close()`.

Configurable knobs (added to `config.reconciliation`):
- `event_queue_watchdog_check_interval_seconds: float = 30.0`
- `event_queue_watchdog_stall_threshold_seconds: float = 120.0`
- `event_queue_watchdog_enabled: bool = True`

### 5. Connection reset — NOT IMPLEMENTED this WP
With finding 1 fixed (PRAGMA parity) and finding 2 fixed (subprocess
hygiene), the structural causes of stuck writers in fused-memory are removed.
The watchdog gives operators visible signal if a new vector emerges.

A `Connection.close()`/reopen path inside the drainer would be a sledgehammer
that buys little: aiosqlite's connection thread holds the SQLite handle and
is hard to interrupt mid-commit. If a wedge recurs in the field, we revisit
with concrete evidence — for now the watchdog ERROR log + WP-D escalation
is the remediation path.

## Tests

In `fused-memory/tests/test_task_file_committer.py` (new):
1. `test_subprocess_killed_on_timeout` — patch `git show` to sleep > timeout;
   assert `_do_commit` raises (or logs) AND no orphan child remains
   (verify by checking returncode is set on the proc via spy).
2. `test_subprocess_killed_on_cancel` — start `_do_commit`, cancel the task,
   assert subprocess is reaped (returncode set).
3. `test_concurrent_commits_serialized_per_root` — fire 5 commits at the same
   project_root concurrently, assert lock serializes them.

In `fused-memory/tests/test_event_buffer.py` (extend):
4. `test_journal_and_buffer_no_busy_under_load` — create both EventBuffer and
   ReconciliationJournal pointing at the same DB; fire 200 concurrent writes
   from each; assert no `OperationalError("database is locked")` raised.

In `fused-memory/tests/test_sqlite_watchdog.py` (new):
5. `test_watchdog_fires_when_drainer_stalled` — build a queue whose buffer
   `push` blocks forever; enqueue an event; start the watchdog with a 1 s
   stall threshold; assert ERROR log emitted within 2 s containing the
   diagnostic payload keys.
6. `test_watchdog_silent_when_drainer_healthy` — happy path; no ERROR log
   for 5 s.
7. `test_watchdog_callback_invoked` — callback is called when wedge detected.

## Out of scope / deferred
- `judge.py` / `agent_loop.py` / `verify.py` subprocess-on-cancel hardening
  (separate WP — not on the SQLite hot path).
- Connection-reset capability (revisit if watchdog fires in production).
- Schema changes / Postgres / Litestream.
- WP-D escalation policy / dashboards.

## Files touched
- NEW: `fused-memory/src/fused_memory/reconciliation/sqlite_watchdog.py`
- NEW: `fused-memory/tests/test_task_file_committer.py`
- NEW: `fused-memory/tests/test_sqlite_watchdog.py`
- `fused-memory/src/fused_memory/middleware/task_file_committer.py` — subprocess helper
- `fused-memory/src/fused_memory/reconciliation/journal.py` — PRAGMAs + tx discipline
- `fused-memory/src/fused_memory/reconciliation/event_buffer.py` — synchronous PRAGMA + tx discipline
- `fused-memory/src/fused_memory/services/write_journal.py` — PRAGMAs + tx discipline
- `fused-memory/src/fused_memory/reconciliation/event_queue.py` — recent_ops ring buffer
- `fused-memory/src/fused_memory/server/main.py` — wire watchdog + cancel order
- `fused-memory/src/fused_memory/config/schema.py` — three new knobs
- `fused-memory/tests/test_event_buffer.py` — concurrent-load test

## Acceptance trace
- [ ] Audit findings documented above.
- [ ] All write paths have explicit tx + cursor discipline.
- [ ] Watchdog fires on artificial stall with diagnostic payload.
- [ ] Existing tests pass.
- [ ] Stress test: 10-minute concurrent-write run with no `database is locked`.
