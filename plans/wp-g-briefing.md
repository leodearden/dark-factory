# WP-G: Canonical `tasks.json` lives only in main checkout

## Mission
Make the project's main git checkout the single source of truth for `tasks.json`. Worktrees never have (or never modify) their own copy. All fused-memory writes normalize to the main checkout regardless of which worktree an agent runs in. Auto-commits of tasks.json land only in main.

## Defect context
Each reify worktree has its own `.taskmaster/tasks/tasks.json` with divergent git history and sometimes a different schema. As of 2026-04-18:
- Main reify tasks.json: int task IDs, max id 1955.
- `/home/leo/src/reify/.worktrees/1959/.taskmaster/tasks/tasks.json`: string task IDs, max numeric id 1952.

Agents pass `project_root` to fused-memory. Depending on which worktree they're running in, writes land in different files. This produces:
- Reviewer "data_regression" false positives on PR diffs (e.g. esc-872-216 on 2026-04-17).
- Cross-agent debugging confusion (e.g. the 1959 steward seeing a "task 1959" that later didn't exist — possibly partly because the steward and other agents were reading different tasks.json files).
- Per-worktree `chore(tasks): auto-commit` noise on every branch.

## Design decision (agreed: Q5=a)
Canonical `tasks.json` lives only in the main checkout. Worktrees do NOT have one (we don't gitignore it — per user, if it gets corrupted we're hosed; we keep it tracked in main only).

## Scope

### 1. Project-root normalizer
Introduce a helper: `resolve_main_checkout(path: str) -> str`.

Implementation sketch:
- `git rev-parse --git-common-dir` — returns `.git` for main, `<main>/.git/worktrees/<N>` for worktrees. From this derive the main working tree path.
- `git worktree list` — canonical source for the main working tree path.
- Fall back / sanity-check: the main working tree is the one with a `.git` directory (not a `.git` file).
- Cache results aggressively (keyed by input path, since the mapping is stable per process lifetime).
- Raise a clear error if input isn't inside any git working tree at all.

Place this helper somewhere cross-cutting. Candidates:
- `fused-memory/src/fused_memory/models/scope.py` (extends the existing `resolve_project_id`).
- `shared/` (if there's a shared helper package usable by both fused-memory and orchestrator).

### 2. Normalize at the MCP boundary
In `fused-memory/src/fused_memory/server/tools.py`, every task-related tool normalizes `project_root` via the resolver before passing downstream.

Alternative: normalize inside `TaskInterceptor` — either works. Pick the layer that's easier to keep comprehensive (i.e., fewer places to forget).

Normalize **both reads and writes** so agents get a consistent view.

### 3. Stop auto-committing in worktrees
`fused-memory/src/fused_memory/middleware/task_file_committer.py`:
- Its `commit(project_root, operation)` should commit against the normalized main checkout, not whatever `project_root` it was handed.
- This ends the `chore(tasks): auto-commit after …` churn in worktree branches.

### 4. Stale-worktree warning
When the normalizer receives a worktree path (not main), emit a DEBUG or INFO log once per process per distinct worktree path. Useful for detecting agents that haven't been updated to pass main directly.

### 5. Orchestrator side
Audit orchestrator code for places that explicitly pass a worktree `project_root` to fused-memory MCP calls. Options:
- **Trust server-side normalization** — change nothing in orchestrator. Simpler but hides the pattern.
- **Normalize in orchestrator too** — callers pass main paths explicitly. Cleaner but more code churn.

Pick **server-side normalization only** (simpler) unless you find strong reason to push changes into orchestrator. Document your choice.

### 6. Do NOT:
- Gitignore `tasks.json` in worktrees.
- Delete existing worktree `tasks.json` files (may still be referenced by in-flight branches; just stop writing to them from our code).
- Rewrite worktree branch git history.

## Files
- Add/extend: `fused-memory/src/fused_memory/models/scope.py` (or `shared/` equivalent)
- `fused-memory/src/fused_memory/server/tools.py` — normalization at entry
- `fused-memory/src/fused_memory/middleware/task_file_committer.py` — target main checkout
- Orchestrator code: audit for explicit worktree paths; likely no changes needed but document any found
- Tests under `fused-memory/tests/` and/or `orchestrator/tests/`

## Tests to add
1. **Normalizer correctness**:
   - Input: main path → returns main path (identity).
   - Input: worktree path → returns corresponding main.
   - Input: non-git path → raises clear error.
   - Input: path in a subdirectory → returns main of the enclosing worktree/tree.
2. **End-to-end write redirection**: call `add_task(project_root=/reify/.worktrees/X)` → assert the task appears in `/reify/.taskmaster/tasks/tasks.json`, NOT in the worktree's tasks.json.
3. **No worktree auto-commit**: perform a mutating op with a worktree project_root; confirm the auto-commit lands on the main checkout's branch (HEAD), not in any worktree.
4. **Cache correctness**: call normalizer repeatedly; assert only one `git worktree list` invocation (or however you implement caching).

## Out of scope
- Migration / cleanup of existing stale worktree tasks.json files (leave them; they're ignored going forward).
- Gitignoring tasks.json.
- Changing orchestrator worktree-creation behavior.
- Dashboards.

## Acceptance
- [ ] `resolve_main_checkout` helper exists, tested, cached.
- [ ] Every fused-memory MCP task tool normalizes `project_root` on entry.
- [ ] Auto-commits only land on main checkout.
- [ ] An agent inside any reify worktree calling `add_task` lands the task in main's `tasks.json`.
- [ ] Existing tests pass.
- [ ] New tests cover normalization + redirection + no-worktree-commit.

## Dependencies
- **WP-A complete**.
- **WP-E merged** — serialization landed first reduces the chance of reintroducing a race while moving writes.
- WP-B / WP-C / WP-D / WP-F can be in any state relative to this WP.

## Workflow for this session

1. Read this briefing.
2. Search memory: "worktree tasks.json", "project_root normalize", "task_file_committer", "main checkout" — anything prior.
3. Read:
   - `fused-memory/src/fused_memory/models/scope.py` (existing `resolve_project_id`)
   - `fused-memory/src/fused_memory/middleware/task_file_committer.py`
   - `fused-memory/src/fused_memory/server/tools.py` (task tools)
   - `fused-memory/src/fused_memory/middleware/task_interceptor.py` (where project_root is used)
   - One or two orchestrator call sites for fused-memory task tools — confirm whether they pass worktree or main paths today.
4. Write `plans/wp-g-plan.md`: normalizer implementation + location, cache policy, where normalization is enforced (server/tools.py vs TaskInterceptor), orchestrator-side decision, test list.
5. Create branch `wp-g/canonical-tasks-json` off main.
6. Implement:
   - Commit 1: normalizer + tests
   - Commit 2: enforce at MCP boundary
   - Commit 3: task_file_committer targets main
   - Commit 4: integration / end-to-end tests
7. Run `cd fused-memory && uv run pytest tests/ -x` and `cd orchestrator && uv run pytest tests/ -x` (or whatever the project convention is; probe with `ls orchestrator/tests/`).
8. Manual check: in a reify worktree (don't create one — use an existing one like `/home/leo/src/reify/.worktrees/1959/` if safe), call `mcp__fused-memory__add_task` via your session — confirm it lands in main's tasks.json. Remember: you're just observing; don't commit to reify.
9. `/reflect`: notes on what you found about orchestrator passing of `project_root`, any surprises about `git rev-parse` behavior, whether server-side-only normalization felt sufficient.
10. Emit JSON summary:
    ```json
    {
      "wp": "G",
      "branch": "wp-g/canonical-tasks-json",
      "commits": ["<sha>", ...],
      "normalizer_location": "fused_memory.models.scope",
      "orchestrator_side_changes": "none|list",
      "tests_added": [...],
      "tests_passed": N,
      "reflection_saved": true,
      "notes": "..."
    }
    ```

## Permission mode
`--dangerously-skip-permissions`. Do NOT push, modify git config, merge to main, restart fused-memory systemd, create new worktrees, or delete any existing worktree's `tasks.json`. Do NOT commit to any branch under `/home/leo/src/reify` — this WP's commits go on dark-factory (`/home/leo/src/dark-factory`) only.
