# WP-C-continuation: finish the SqliteWatchdog wiring

## Context
WP-C was launched against `plans/wp-c-briefing.md`. The agent made excellent progress on the **root-cause-informed scope**:

- Committed `fix(task_file_committer): reap git subprocesses on timeout/cancel` (commit 3257e4f9bd) — addresses the actual root cause of the 2026-04-17 16h lock (a stuck `git show` child subprocess held under the fused-memory cgroup).
- Committed `fix(sqlite): explicit transaction discipline + PRAGMA parity for shared reconciliation.db` (commit 9937e036cd) — aiosqlite cleanliness across event_buffer, journal, write_journal.
- Started implementing the drainer watchdog but hit "Prompt is too long" before finishing.

The committed work is verified: 146 tests pass across event_buffer, write_journal, event_queue, and task_interceptor.

## Current state — already on branch `wp-c/sqlite-lock-health`
You are (or should be) checked out on `wp-c/sqlite-lock-health`. Two commits are on top of main. There is **uncommitted, partially-finished work** on disk:

**New file (untracked):**
- `fused-memory/src/fused_memory/reconciliation/sqlite_watchdog.py` — a `SqliteWatchdog` class with:
  - `__init__` accepting `event_queue`, `check_interval_seconds=30`, `stall_threshold_seconds=120`, `rearm_after_seconds=600`, and an optional `wedge_callback`.
  - Wedge condition: `retry_in_flight > 0` AND `now - last_commit_ts > stall_threshold`.
  - Intended to emit a structured ERROR log with diagnostics and optionally call `wedge_callback(payload)` for WP-D to escalate.
  - Re-arms after `rearm_after_seconds` so a persistent wedge logs at most once per window.

**Modified, uncommitted:**
- `fused-memory/src/fused_memory/config/schema.py` — adds `event_queue_watchdog_*` config fields.
- `fused-memory/src/fused_memory/reconciliation/event_queue.py` — adds a ring buffer `_recent_ops` (last 20 drainer attempts) and a `recent_ops()` accessor feeding watchdog diagnostics.
- `fused-memory/src/fused_memory/server/main.py` — instantiates and `start()`s the watchdog in the server's lifespan.

**Almost certainly NOT yet done (verify first, do only what's missing):**
1. `sqlite_watchdog.py` may be missing `close()` / shutdown; `server/main.py` likely doesn't call it on shutdown.
2. **No tests exist** for `sqlite_watchdog.py`. The briefing asks for: watchdog fires on artificial stall, re-arm behavior, graceful close, integration with EventQueue stats.
3. Commits haven't been created for the uncommitted work.

Ignore these pre-existing uncommitted modifications (they were there before WP-C; they are not your work and must not be touched):
- `config/usage-accounts.yaml`
- `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py`
- `taskmaster-ai` (submodule)

## Your mission
1. Read the original `plans/wp-c-briefing.md` to understand full scope.
2. Read the partial work on disk — `sqlite_watchdog.py` + the three modified files.
3. Complete whatever's missing from the watchdog implementation: graceful shutdown, server-lifespan shutdown call, any remaining diagnostic payload fields the briefing asked for.
4. Add tests under `fused-memory/tests/test_sqlite_watchdog.py` (or similar) covering:
   - Wedge detected on artificial stall (mock EventQueue with controlled stats).
   - Re-arm behavior — doesn't spam.
   - Wedge NOT fired when drainer is committing normally.
   - Graceful close cancels the check loop.
5. Run targeted tests: `cd fused-memory && uv run pytest tests/test_sqlite_watchdog.py tests/test_event_queue.py tests/test_task_interceptor.py tests/test_event_buffer.py tests/test_write_journal.py tests/test_server_shutdown.py -x -q`. All must pass.
6. Commit the new work in logical chunks. Reasonable chunking:
   - Commit A: event_queue.py recent_ops ring buffer + config schema additions
   - Commit B: sqlite_watchdog.py module + tests
   - Commit C: server/main.py lifespan wiring (start + shutdown)
   (Or a single squashed commit if that feels cleaner — your call.)
7. `/reflect` at the end. If fused-memory is wedged again for any reason, fall back to writing to `plans/wp-c-continuation-reflection.md`.

## Constraints
- Do NOT re-do the committed work (task_file_committer subprocess timeouts, aiosqlite discipline) — it's already on this branch.
- Do NOT touch files outside what the watchdog needs; this is a focused continuation, not a rescope.
- Do NOT merge to main. Parent coordinator handles.
- Do NOT push, do NOT modify git config, do NOT restart fused-memory systemd.
- Do NOT touch the pre-existing uncommitted modifications listed above.

## Acceptance
- [ ] `SqliteWatchdog` has a `close()` method; server lifespan calls it during shutdown.
- [ ] Watchdog tests pass.
- [ ] Targeted test set passes (command in step 5).
- [ ] Branch `wp-c/sqlite-lock-health` has additional commits covering the new watchdog work.

## Emit final JSON summary
```json
{
  "wp": "C-continuation",
  "branch": "wp-c/sqlite-lock-health",
  "new_commits": ["<sha>", ...],
  "watchdog_tests_added": [...],
  "targeted_tests_passed": N,
  "reflection_saved": true,
  "notes": "..."
}
```

## Permission mode
`--dangerously-skip-permissions`. Free to edit/test/commit. Do NOT push, merge, or restart services.
