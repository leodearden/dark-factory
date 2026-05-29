# WP-E: Serialize all Taskmaster mutating calls per-project

## Mission
Eliminate the lost-update race on `tasks.json` by serializing every mutating Taskmaster call per-project in our middleware — without forking taskmaster-ai.

## Defect context
In `taskmaster-ai/scripts/modules/task-manager/add-task.js` (and peers), `readJSON` reads tasks.json **outside** the file lock, while `writeJSON` takes a lock for the write. Concurrent mutating operations can each read at state S1, both compute "append new task", and the second write silently overwrites the first's new task — or produces near-duplicate IDs with different content.

This is contributory (not the sole cause) to the 2026-04-17 reify corruption where task 1959 went missing and tasks 1953/1954/1955 ended up as three identical duplicates.

Full prior analysis is in the session that produced the briefings — see especially the finding: *"readJSON is lock-free; writeJSON takes a file lock but only re-reads+merges when the caller passed `_rawTaggedData` — and add-task.js:208-210 deliberately strips `_rawTaggedData`, so the merge branch never fires."*

## Design decision (agreed)
**Do not patch taskmaster-ai.** The user values low-friction upstream updates. Serialize on our side instead.

Today, `fused-memory/src/fused_memory/middleware/task_interceptor.py` holds an `asyncio.Lock` per `project_id` (`self._project_locks`) but **only** wraps `add_task` and `add_subtask`. Extend this lock to wrap every mutating taskmaster tool call, so the taskmaster subprocess never has to handle concurrent mutations on the same project.

Fused-memory is the ONLY writer of `tasks.json` (per project memory "Never write tasks.json directly — use fused-memory MCP"). So if every write path goes through this lock, the race cannot be exercised.

## Scope

**Methods that currently need the lock but don't have it** (confirm by grepping `async def` + checking whether `async with self._project_lock(project_id)` is present):

- `set_task_status`
- `update_task`
- `remove_task`
- `add_dependency`
- `remove_dependency`
- `expand_task`
- `parse_prd` (if exposed here)
- `move_task` (if present)
- Any other method that ultimately calls `tm.<mutating>` on the TaskmasterBackend

**Methods already guarded** (verify and keep as-is):
- `add_task`
- `add_subtask`

**Methods that don't need the lock** (reads):
- `get_tasks`, `get_task`, `next_task`, `complexity_report`, and similar read-only ops. Do not add locks to these (would unnecessarily cap read throughput).

## Lock scope
Hold the lock tightly: only around the taskmaster subprocess call + anything that must be atomic with the tasks.json mutation (e.g., the curator's `note_created`, which already runs under the lock for add_task).

Do NOT hold the lock across LLM calls (the curator's `curate()` call currently runs under the lock in add_task — that's pre-existing, leave it alone for now; extending it everywhere is not this WP's job). For the newly-locked methods, just wrap the `await tm.<method>(...)` + any event/curator bookkeeping that follows.

Do NOT hold the lock across `buffer.push(...)` if you can avoid it — that couples to an unreliable DB (see WP-B).

## Files
- `fused-memory/src/fused_memory/middleware/task_interceptor.py` — primary. Touch every mutating method.
- `fused-memory/src/fused_memory/backends/taskmaster_client.py` — only if you find a mutating method bypasses the interceptor. Audit but prefer to keep this file untouched.
- Tests: `fused-memory/tests/test_task_interceptor.py` (may not exist — may need creation) or wherever the project's task-interceptor tests live. Grep the tests/ tree.

## Tests to add
Add concurrency tests under `fused-memory/tests/`:

1. **Concurrent add_task bursts**: fire 20 concurrent `add_task` calls to the same project. Assert: all 20 tasks appear with distinct IDs, all with distinct titles (use a counter in the prompt/title to force distinctness), no task lost. This should pass today for add_task but add it for regression.

2. **Mixed-op concurrency**: fire a mix of N adds + M set_task_status + K update_task concurrently on the same project. Assert: final tasks.json is consistent with serial application (all adds succeeded, all status changes applied, update_task changes persisted).

3. **Two projects in parallel**: Fire concurrent ops on two different projects. Assert: no serialization between them (throughput shouldn't drop by half).

4. **No-contention baseline**: single sequential call p95 latency doesn't regress beyond a reasonable tolerance (add a perf guardrail).

Use the existing test harness patterns in `fused-memory/tests/` — don't invent a new one. Look at how existing tests mock TaskmasterBackend or use real subprocess — follow precedent.

## Out of scope
- Changes to taskmaster-ai.
- Read-lock discipline (pure reads stay unlocked).
- Cross-project lock (per-project is correct).
- Fixing `buffer.push` race (WP-B).
- Project-root normalization (WP-G).

## Acceptance
- [ ] Every mutating method in `task_interceptor.py` acquires `self._project_lock(project_id)` before its taskmaster call.
- [ ] All existing tests pass.
- [ ] New concurrency tests pass (including mixed-op).
- [ ] No regression in single-call latency.

## Dependencies
- **WP-A must be complete** (recovery) — you need a healthy fused-memory to run integration tests.

## Workflow for this session

1. Read this briefing.
2. Search fused-memory for relevant context: `mcp__fused-memory__search` for keys like "task_interceptor lock", "_project_lock", "add_task serialize". Read the latest project memories about the 2026-04-17 corruption.
3. Read `/home/leo/src/dark-factory/fused-memory/src/fused_memory/middleware/task_interceptor.py` fully. Enumerate every mutating method.
4. Write `plans/wp-e-plan.md`: which methods get locks, in what order you'll touch them, test plan.
5. Create branch `wp-e/serialize-taskmaster-writes` off main.
6. Implement. Commit in logical chunks (e.g., "add lock to set_task_status", "add lock to update_task", etc., or one squashed commit if the change is small — your call).
7. Run the fused-memory test suite (`cd fused-memory && uv run pytest tests/ -x` or the project's standard). Fix until green.
8. Add the new concurrency tests. Verify they would have caught a regression (run once with a lock deliberately removed, confirm failure; then restore lock, confirm pass — note this in your summary).
9. Final: `/reflect` with notes on: what surprised you, any patterns you noticed that might affect WP-B/C/D/G, what conventions you established.
10. Emit JSON summary:
    ```json
    {
      "wp": "E",
      "branch": "wp-e/serialize-taskmaster-writes",
      "commits": ["<sha>", ...],
      "methods_locked": ["set_task_status", "update_task", ...],
      "tests_added": ["test_...", ...],
      "tests_passed": N,
      "tests_failed": 0,
      "reflection_saved": true,
      "notes": "..."
    }
    ```

## Permission mode
`--dangerously-skip-permissions`. Freely edit, test, commit. Do NOT push, do NOT modify git config, do NOT restart fused-memory systemd unit (the parent session handles restart between WPs), do NOT run reset/clean destructive git ops. Do NOT merge to main — that's the parent session's job.
