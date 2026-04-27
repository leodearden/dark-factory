---
name: merge-queue
description: "Merge a task branch to main via the orchestrator's merge queue. Use this skill whenever you need to merge a completed task branch into main and the orchestrator might be running — it routes through the escalation MCP's merge_request tool, which serializes merges and prevents races. Trigger this when an agent says 'merge to main', 'submit merge', 'merge task branch', finishes fixing a blocked task and needs to merge, or any time code on a task branch is ready to land on main. If the escalation MCP isn't reachable, the skill falls back to direct merge. Prefer this over raw git merge --no-ff whenever working in the dark-factory repo."
---

# Merge Queue

When the orchestrator is running, all merges to main go through the **merge queue** — a serial worker that rebases, verifies, and atomically advances main using compare-and-swap. This prevents races between concurrent tasks, the steward, and interactive sessions.

The escalation MCP exposes a `merge_request` tool that lets you submit to this queue from outside the orchestrator workflow. This skill tells you how to use it.

## Why this matters

Direct `git merge --no-ff` into main bypasses the merge queue. If the orchestrator is also running, you'll race with its merge worker — two actors trying to advance the same ref simultaneously. The queue serializes this safely. It also runs post-merge verification and prevents `.task/` directory contamination from reaching main.

## Workflow

### 1. Prepare your branch

Before submitting, make sure your branch is in mergeable shape:

```bash
# In the task worktree
git rebase main
# Resolve any conflicts
# Run verification (tests, lint, type-check)
```

Pre-rebasing reduces the chance of conflicts inside the merge queue. If you skip this, the queue will attempt the merge anyway but is more likely to return `conflict`.

### 2. Check if the escalation MCP is reachable

Call any lightweight escalation MCP tool to confirm the server is up:

```
mcp__escalation__get_pending_escalations()
```

- **If it responds:** proceed to step 3 (use the merge queue).
- **If it errors or times out:** the orchestrator isn't running. Fall back to direct merge (step 5).

### 3. Submit the merge request

```
mcp__escalation__merge_request(
  task_id="<TASK_ID>",
  branch="<TASK_ID>",
  worktree="<path to worktree>",
  description="<brief description of what's being merged>"
)
```

Parameters:
- `task_id` — the task number (string)
- `branch` — the task ID only (e.g., `"466"`), **not** the full branch name. The merge worker prepends the `task/` prefix automatically.
- `worktree` — absolute path to the task's worktree (e.g., `/home/leo/src/dark-factory/.worktrees/42/`)
- `description` — optional context for logs

This call blocks until the merge worker processes your request. It may take a few seconds if other merges are queued ahead of yours.

### 4. Handle the outcome

The tool returns `{ status, reason, conflict_details }`. Handle each status:

**`done`** — Merge succeeded. Main has been advanced atomically.
- Update the task: `set_task_status(id="<TASK_ID>", status="done", project_root="<PROJECT_ROOT>", done_provenance={"kind": "merged", "commit": "<merge-commit-sha>"})`
  - Use `{"kind": "merged", "commit": "<sha>"}` when this branch's merge commit landed on main (the normal case — the merge tool's return value has the merge SHA). The server backstops with `git merge-base --is-ancestor <sha> main`.
  - Use `{"kind": "found_on_main", "note": "<one-sentence explanation>", "commit": "<optional landing sha>"}` when the implementation is already on main from a sibling task / prior orchestrator run and no merge applied to this branch.
- Clean up worktree and branch:
  ```bash
  git worktree remove .worktrees/<TASK_ID>
  git branch -d task/<TASK_ID>
  ```

**`already_merged`** — The branch was already an ancestor of main (another merge or a manual push landed it).
- Same as `done` — update task status and clean up.

**`conflict`** — Merge conflicts detected. The `conflict_details` field has the specifics.
- Resolve conflicts in your worktree.
- Rebase onto current main again (main may have moved).
- Resubmit to the merge queue (go back to step 3).

**`blocked`** — Post-merge verification failed, or CAS retries exhausted. The `reason` field explains why.
- Read the reason carefully. Common causes:
  - Verification failure (tests/lint broke after merge) — fix in your worktree, resubmit.
  - CAS retry limit — main was moving too fast (rare at normal concurrency). Wait a moment and retry.
  - `.task/` contamination detected — check that `.task/` isn't committed on your branch.

### 5. Fallback: direct merge

If the escalation MCP is down (orchestrator not running), merge directly. There's no queue to race with.

```bash
cd <worktree>
git rebase main
# resolve conflicts if any, run verification
git checkout main
git merge --no-ff task/<TASK_ID> -m "Merge task/<TASK_ID>: <description>"
# run verification on main
git checkout task/<TASK_ID>  # return to worktree branch
```

After a successful direct merge:
- Update task status via fused-memory MCP (if available)
- Clean up worktree and branch

## Quick reference

| Situation | Action |
|-----------|--------|
| Orchestrator running | Use `merge_request` via escalation MCP |
| Orchestrator not running | Direct `git merge --no-ff` |
| Merge returns `conflict` | Fix in worktree, resubmit |
| Merge returns `blocked` | Read reason, fix, resubmit |
| Merge returns `done` or `already_merged` | Update task status, clean up |
| Unsure if orchestrator is running | Probe `get_pending_escalations()` — if it responds, use the queue |
