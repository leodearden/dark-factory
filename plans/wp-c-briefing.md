# WP-C: SQLite lock health & watchdog

## Mission
Fix the in-process SQLite writer-lock leak in fused-memory so a process restart is no longer required to recover, and add a watchdog that raises the stuck-writer condition instead of letting it rot.

## Defect context
The fused-memory process (systemd user unit `fused-memory.service`) holds the `reconciliation.db` SQLite writer lock for 16+ hours at a time. An external `sqlite3` client can read the DB fine — the contention is **inside** the fused-memory process. Project memory documents this: *"SQLite lock requires process restart — Row updates don't release connection-level locks; restart fused-memory directly."*

Symptoms:
- `journalctl --user -u fused-memory.service` shows continuous `database is locked` errors every ~5s from `fused_memory.reconciliation.harness`.
- `reconciliation.db-wal` stops growing (no writer can commit).
- Every `add_task` / `set_task_status` / `update_task` returns `{"error": "database is locked"}` to the client.
- Only remedy today is `scripts/restart-fused-memory.sh`.

## Design decision (agreed: both sidestep and diagnose)
WP-B decoupled the hot path from `buffer.push`, so clients no longer see lock errors. That's the **sidestep**. This WP is the **diagnose + fix**: find what inside the process holds the writer lock, and make it not hold the lock indefinitely.

## Scope

### 1. Audit every aiosqlite caller
Grep `aiosqlite` in `fused-memory/src/`. Known callers (as of 2026-04-18):
- `fused-memory/src/fused_memory/reconciliation/event_buffer.py`
- `fused-memory/src/fused_memory/services/write_journal.py`
- Possibly: `fused-memory/src/fused_memory/services/runs.py` (CostStore)
- Possibly: `fused-memory/src/fused_memory/reconciliation/journal.py`

For each caller:
- Inventory every write path (execute + commit).
- Verify every write is wrapped in explicit transaction discipline: `try: … await db.commit() except: await db.rollback() finally: …`. Don't rely on implicit auto-commit behavior.
- Verify every `db.execute(...)` used as an async context manager (`async with db.execute(...) as cursor`) so cursors are always closed.
- Check that only ONE `aiosqlite.Connection` object exists per physical DB file per process. If you find multiple, consolidate (each extra connection is a potential lock contender).

### 2. Single-writer pattern where appropriate
If multiple subsystems write to the same DB file:
- Consolidate writes behind a single owner task (e.g. the drainer from WP-B owns reconciliation.db writes).
- Other subsystems enqueue commands rather than writing directly.

This synergizes with WP-B. If WP-B's drainer becomes the sole reconciliation.db writer, we've largely eliminated the intra-process contention already.

### 3. Watchdog
Add a lightweight watchdog coroutine (can live in the same module as WP-B's drainer or its own). Responsibilities:
- Track the drainer's `last_successful_commit_ts`.
- If `now - last_successful_commit_ts > N seconds` AND queue depth > 0 → emit a structured ERROR log with:
  - Current queue depth
  - Retries in flight
  - Last N SQL operations attempted (drainer should expose this; keep ring buffer of ~20)
  - Stack sample or task list (any diagnostic signal that helps narrow which writer/transaction is stuck)
- Raise a condition WP-D can consume (shared flag, callback, or event) — so WP-D escalates at L1 instead of silently rotting.

`N` should be configurable. Default: 120 seconds (twice the typical busy_timeout × some slack).

### 4. PRAGMA review
Current init in `event_buffer.py`:
- `PRAGMA journal_mode=WAL`
- `PRAGMA busy_timeout=5000`

Consider adding:
- `PRAGMA synchronous=NORMAL` (safer under WAL, but verify if project already sets something else)
- Explicit `PRAGMA wal_autocheckpoint=<N>` if WAL growth is a concern (it was at 4.2 MB stuck; with draining it should recover naturally, but explicit is defensive)
- `PRAGMA cache_size=…` if load-testing shows benefit

Do NOT add PRAGMAs speculatively — justify each in the plan.

### 5. Connection reset capability
If the audit shows the leak is real and hard to eliminate structurally, add a **reset** path: after the watchdog fires, the drainer can close and reopen its Connection. This is a blunt instrument — only fire it from the watchdog, not casually.

Only implement reset if you can't structurally eliminate the leak within this WP.

## Files
- `fused-memory/src/fused_memory/reconciliation/event_buffer.py`
- `fused-memory/src/fused_memory/services/write_journal.py`
- Any other aiosqlite callers surfaced by audit
- `fused-memory/src/fused_memory/reconciliation/harness.py` — cleanup on cancel/shutdown
- New or extended: watchdog utility (in event_queue.py from WP-B if natural, else its own file)
- Tests under `fused-memory/tests/`

## Tests to add
1. **Cursor-close discipline**: injection test or static audit confirming every write path closes its cursor.
2. **Transaction rollback on error**: induce an error mid-write; verify the connection is not left in a pending transaction.
3. **Multi-coroutine write stress**: spawn N coroutines, each doing M writes. Assert no lock leak at end, completion time within budget.
4. **Watchdog fires on stall**: artificially hold a transaction, confirm watchdog emits the ERROR log with diagnostics within N+tolerance seconds.
5. **Connection reset (if implemented)**: stall → watchdog → reset → writes succeed.
6. **24h-ish soak** (optional, only if time allows — mark skipped otherwise): sustained synthetic load for ≥1h, no lock leaks.

## Out of scope
- Switching storage engine (no Postgres, no Litestream).
- Schema changes.
- Changing the reconciliation harness loop semantics.
- Backlog policy (WP-D).
- Dashboards.

## Acceptance
- [ ] Audit findings documented in `plans/wp-c-plan.md` (what was leaking, what you changed).
- [ ] All write paths have explicit transaction + cursor discipline.
- [ ] Watchdog fires on artificial stall, log contains actionable diagnostic.
- [ ] Existing tests pass.
- [ ] New tests cover discipline + stress + watchdog (+ reset if implemented).
- [ ] No `database is locked` occurrences during a 10-minute stress run.

## Dependencies
- **WP-B merged** — you rely on the drainer + queue as the sole writer for reconciliation.db.
- **WP-A complete** — healthy service to stress-test.

## Workflow for this session

1. Read this briefing.
2. Search fused-memory memory: "SQLite lock requires process restart", "aiosqlite connection", "reconciliation harness cleanup", "write_journal" — anything relevant to prior lock incidents.
3. Grep `aiosqlite` across `fused-memory/src/`. Enumerate callers.
4. For each caller, read its write paths. Note where transaction discipline is implicit/missing.
5. Read WP-B's drainer (it should be landed on main by now) — understand how/where writes flow.
6. Write `plans/wp-c-plan.md`: audit findings, changes by file, watchdog design (thresholds, diagnostic payload), PRAGMA decisions with rationale, reset strategy (or not), test list.
7. Create branch `wp-c/sqlite-lock-health` off main.
8. Implement in logical commits:
   - Commit 1: transaction/cursor discipline in event_buffer.py
   - Commit 2: same for write_journal.py and others
   - Commit 3: watchdog
   - Commit 4: connection reset (if needed)
   - Commit 5: tests
9. Run `cd fused-memory && uv run pytest tests/ -x`.
10. Run a manual stress test — write a small script that fires concurrent writes for a few minutes; confirm no lock leaks (query reconciliation.db-wal size, confirm it stays bounded).
11. `/reflect` with notes on: root cause you found for the leak (connection-per-subsystem? missing rollback? cursor leak? WAL checkpoint starvation?), what the watchdog looks like, whether further work is needed.
12. Emit JSON summary:
    ```json
    {
      "wp": "C",
      "branch": "wp-c/sqlite-lock-health",
      "commits": ["<sha>", ...],
      "root_cause_found": "...",
      "files_changed": [...],
      "watchdog_threshold_secs": N,
      "connection_reset_implemented": true|false,
      "tests_added": [...],
      "tests_passed": N,
      "reflection_saved": true,
      "notes_for_wp_d": "..."
    }
    ```

## Permission mode
`--dangerously-skip-permissions`. Do NOT push, modify git config, merge to main, or restart the fused-memory systemd unit. (Your stress tests should either run in a test harness or against a dedicated test instance — not the production fused-memory service.)
