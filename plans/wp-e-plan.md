# WP-E Plan: Serialize Taskmaster Mutating Calls Per-Project

## Goal
Eliminate the lost-update race on `tasks.json` by ensuring every mutating
taskmaster call goes through `self._project_lock(project_id)` in the
`TaskInterceptor`. No taskmaster-ai fork.

## Enumeration of mutating methods in `task_interceptor.py`

| Method               | Has lock today? | Plan |
|----------------------|-----------------|------|
| `add_task`           | yes (`_add_task_locked`)   | keep as-is |
| `add_subtask`        | yes (`_add_subtask_locked`) | keep as-is |
| `set_task_status`    | **no**  | add lock around `tm.get_task` + no-op check + `tm.set_task_status` |
| `update_task`        | **no**  | add lock around `tm.update_task` only; re-embed path stays lock-free (reads `tm.get_task`) |
| `remove_task`        | **no**  | add lock around `tm.remove_task` |
| `add_dependency`     | **no**  | add lock around `tm.add_dependency` |
| `remove_dependency`  | **no**  | add lock around `tm.remove_dependency` |
| `expand_task`        | **no**  | add lock around `pre_snapshot` + `tm.expand_task` + `_await_commit`; dedup stays outside lock but its internal mutations acquire lock per-op |
| `parse_prd`          | **no**  | same as `expand_task` |

Internal helpers:
- `_execute_combine` — calls `tm.update_task` directly; callers responsible for lock.
  - Called from `_add_task_locked` (lock held).
  - Called from `_dedupe_bulk_created` (lock NOT held — wrap the call).
- `_dedupe_bulk_created` — the bulk dedup path. LLM calls (`curator.curate`)
  must stay outside any lock. Each mutation within the loop gets its own
  `async with self._project_lock(...)` block:
  - `await tm.remove_task(tid, ...)` (drop path) → lock.
  - `await self._execute_combine(...)` + `await tm.remove_task(tid, ...)` (combine path) → one lock block, tight.

Pure reads (`get_tasks`, `get_task`) stay unlocked — confirmed scope rule.

## Lock scope discipline

- Hold the lock **only** around the taskmaster subprocess call (+ anything
  that must be atomic with the tasks.json mutation, e.g. a pre-read
  needed for a no-op check in `set_task_status`).
- **Do not** hold the lock across `buffer.push(...)`, `_schedule_commit(...)`,
  reconciler dispatch, LLM calls, or curator re-embed.
- Exception: `add_task` / `add_subtask` already hold the lock across
  `curate()`. Pre-existing by R3 design; leave alone (briefing explicitly
  says so).

## Edits

1. `set_task_status`: wrap `before = await tm.get_task(...)` + the
   same-status early-return + `result = await tm.set_task_status(...)` in
   a single `async with self._project_lock(project_id):` block. Resolve
   `project_id` once at the top.
2. `update_task`: wrap only `await tm.update_task(...)` in a lock block.
3. `remove_task`: wrap `await tm.remove_task(...)`.
4. `add_dependency`: wrap `await tm.add_dependency(...)`.
5. `remove_dependency`: wrap `await tm.remove_dependency(...)`.
6. `expand_task`: wrap `pre_snapshot = await tm.get_tasks(...)` through
   `await self._await_commit(...)` in one block (the whole atomic
   "expand + commit" is one unit; dedup runs outside).
7. `parse_prd`: same shape as `expand_task`.
8. `_dedupe_bulk_created`: grab `project_id` once, wrap each mutation
   path inside a tight lock block. LLM `curate()` calls stay outside.

## Tests (to add in `tests/test_task_interceptor.py`)

Under a new `# ── Tests for per-project serialisation (WP-E) ──` section:

1. `test_concurrent_add_task_burst_all_distinct` — fire 20 concurrent
   `add_task` calls with distinct titles on the same project. Assert all
   20 survive with distinct ids; backend sees 20 calls serialised (no
   interleaved state). Use a real per-call `tm.add_task` that increments
   an id counter under the same asyncio lock-aware pattern already
   present in `test_concurrent_add_task_produces_single_task`.

2. `test_mixed_op_concurrency_serialises` — fire mixed
   `add_task` + `set_task_status` + `update_task` calls on one project
   concurrently. Use a shared AsyncMock-driven tasks-dict proxy that
   asserts *no two calls overlap* (enter/exit counter with peak==1).
   Assert the post-condition is consistent with serial application.

3. `test_two_projects_do_not_serialise` — fire N concurrent `add_task`
   calls on `/projA` and `/projB`. Assert enter/exit counter's peak can
   reach 2, proving the lock is per-project (not global).

4. `test_set_task_status_holds_lock_across_read_and_write` — use a
   blocking `tm.get_task` to prove the lock is held across the read, so
   two concurrent `set_task_status` calls do not both see the stale
   before-state.

5. `test_dedupe_bulk_created_mutations_are_locked` — fire a concurrent
   `add_task` while `expand_task` is running with dedup decisions that
   require `tm.remove_task`. Assert the taskmaster-side mutations
   observed are serialised (overlap counter ≤ 1 for mutating calls).

6. `test_single_call_latency_not_regressed` (soft perf guardrail) —
   time N=100 sequential `set_task_status` calls. Assert p95 is under a
   generous bound (e.g. 200ms on a mock).

## Regression verification

Before committing tests: temporarily remove the lock from
`set_task_status`, confirm the mixed-op test fails. Restore lock,
confirm pass. Note in reflect + JSON summary.

## Out of scope (confirm per briefing)

- No taskmaster-ai edits.
- No lock on read-only methods.
- Do not fix `buffer.push` (WP-B).
- Do not widen `curate()` lock scope beyond existing.

## Branch + commits

- Branch: `wp-e/serialize-taskmaster-writes`
- Commit plan (one squashed or small sequence — commit as one cohesive
  change since the edits are small):
  1. "feat(middleware): serialize all taskmaster mutating calls per-project"
- Tests added in the same commit or a follow-up ("test(middleware): concurrency coverage for per-project serialisation").
