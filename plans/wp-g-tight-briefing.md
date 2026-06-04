# WP-G (tight): Canonical `tasks.json` in main checkout only

Last of seven WPs. Tight briefing to avoid context overflow — start implementing immediately.

## Goal
Make the project's main git checkout the sole source of truth for `tasks.json`. Worktrees never have or modify their own copy. All fused-memory task-tool writes normalize `project_root` to the main checkout regardless of the caller's worktree.

## Policy (already decided — do not redesign)
- Add `resolve_main_checkout(path: str) -> str` helper.
- Enforce normalization at the MCP boundary in `server/tools.py` (not in TaskInterceptor) — one choke point is easier to audit.
- `task_file_committer` commits against the normalized main-checkout path.
- Reads (`get_tasks`, `get_task`) also normalize — agents must see one consistent view.
- **Do NOT** touch orchestrator code; server-side normalization is sufficient. If orchestrator sends a worktree path, server silently redirects. That's the design.
- **Do NOT** gitignore `tasks.json`, delete existing stale worktree copies, or touch worktree creation.

## Files to touch (exact list)

Write:
1. `fused-memory/src/fused_memory/models/scope.py` — add `resolve_main_checkout`. Existing `resolve_project_id` lives here; extend that module. Cache results in a module-level dict keyed by input path.
2. `fused-memory/tests/test_main_checkout_resolver.py` — unit tests.
3. `fused-memory/tests/test_canonical_tasks_json.py` — end-to-end redirection tests (one real temp-repo fixture with a worktree, then exercise add_task with worktree path, assert write lands in main).

Edit:
4. `fused-memory/src/fused_memory/server/tools.py` — at the top of every task tool function (`add_task`, `get_tasks`, `get_task`, `set_task_status`, `update_task`, `remove_task`, `add_dependency`, `remove_dependency`, `expand_task`, `parse_prd`, `move_task`, and any others you find in the file — grep `project_root` inside this one file to be thorough): after the existing `validate_project_root(...)` check, `project_root = resolve_main_checkout(project_root)`. (add_subtask removed — DF-D) If resolution fails, return the validation error.
5. `fused-memory/src/fused_memory/middleware/task_file_committer.py` — if the method signature receives `project_root`, normalize it via `resolve_main_checkout` before running git commands. Quick one-liner near the top of `_do_commit`.

## `resolve_main_checkout` spec

Implementation approach:
- Use `git -C <path> worktree list --porcelain` to enumerate all worktrees. The first `worktree <path>` entry in the porcelain output is the main working tree (that's a documented git invariant).
- If the input path is inside that main working tree (or inside any of the listed worktrees whose main is the same), return the main path.
- Cache results; key the cache by the resolved absolute input path.
- Raise `ValueError` with a clear message if the input isn't inside any git working tree, or if `git` isn't available.

Signature:
```python
def resolve_main_checkout(path: str | Path) -> str:
    """Return absolute string path of the main working tree that contains `path`."""
```

Don't be fancy — subprocess `git -C <absolute-input-path> worktree list --porcelain`, parse the first `worktree X` line, verify the input path is a descendant of some listed worktree (to sanity-check).

## Tests

`test_main_checkout_resolver.py`:
1. `test_resolves_main_path_to_itself` (tmp git repo, no worktrees, input is the repo root → returns root).
2. `test_resolves_subdir_inside_main` (input is a subdir → returns repo root).
3. `test_resolves_worktree_path_to_main` (create a worktree, input is worktree path → returns main path).
4. `test_resolves_subdir_inside_worktree` (input is a subdir within the worktree → returns main path).
5. `test_raises_on_non_git_path` (`/tmp` that's not a git repo → raises).
6. `test_caches_results` (patch subprocess, call twice, assert one invocation).

`test_canonical_tasks_json.py`:
7. `test_add_task_from_worktree_path_writes_to_main` — create tmp repo + worktree, call `add_task` with worktree path, assert task appears only in main's tasks.json.
8. `test_committer_commits_to_main_from_worktree_path` — after a tool call, assert `git log` on main shows the auto-commit, worktree branch is untouched.

Use existing test-harness patterns in `fused-memory/tests/` (look at `conftest.py` and how other tests set up a tmp repo). Keep these focused — don't build a full orchestrator-style fixture.

## Workflow
1. `git checkout -b wp-g/canonical-tasks-json main`.
2. Write `resolve_main_checkout` + `test_main_checkout_resolver.py` first. Get them green.
3. Wire normalization into `server/tools.py` task tools + `task_file_committer.py`.
4. Add `test_canonical_tasks_json.py` end-to-end tests. Get them green.
5. Run full targeted suite: `cd fused-memory && uv run pytest tests/test_main_checkout_resolver.py tests/test_canonical_tasks_json.py tests/test_task_interceptor.py tests/test_backlog_policy.py tests/test_event_queue.py tests/test_sqlite_watchdog.py tests/test_server_shutdown.py -x -q`. All green.
6. Commit in logical chunks: (A) resolver + unit tests, (B) server/tools normalization + committer, (C) end-to-end tests. Or squash if small.
7. `/reflect` at the end.
8. Emit JSON summary:
```json
{"wp": "G", "branch": "wp-g/canonical-tasks-json", "commits": [...], "tests_added": [...], "tests_passed": N, "reflection_saved": true, "notes": "..."}
```

## Constraints
- Don't explore beyond the listed files. Grep `project_root` ONLY in `server/tools.py` and `task_file_committer.py`.
- Do NOT modify orchestrator code.
- Do NOT touch taskmaster-ai, memory_consolidator.py, or usage-accounts.yaml.
- Do NOT push, merge, or restart fused-memory.
- Do NOT delete any existing worktree's tasks.json.
- `--dangerously-skip-permissions` enabled.

## Quick context for confidence
- WP-A through WP-F merged. Main's task writes currently flow: MCP → server/tools → TaskInterceptor (per-project lock) → Taskmaster subprocess.
- WP-E added per-project serialization; WP-B the event queue; WP-C the drainer watchdog; WP-D backlog policy; WP-F curator combine safety. Your work is orthogonal to all of those — just adds path normalization at the MCP entry.
- One concrete example the corruption analysis produced: worktree `/home/leo/src/reify/.worktrees/1959/.taskmaster/tasks/tasks.json` has a stale string-ID schema, max id 1952, while main has int-or-string IDs through id 1956. That divergence is exactly what this WP prevents going forward.
