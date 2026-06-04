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
  description="<brief description of what's being merged>",
  wait_secs=100
)
```

Parameters:
- `task_id` — the task number (string)
- `branch` — the task ID only (e.g., `"466"`), **not** the full branch name. The merge worker prepends the `task/` prefix automatically.
- `worktree` — absolute path to the task's worktree (e.g., `/home/leo/src/dark-factory/.worktrees/42/`)
- `description` — optional context for logs
- `wait_secs=100` — **always pass this explicitly.** The value 100 equals the server's maximum bounded wait (`_MAX_WAIT_SECS`), so fast/idle-queue merges return their terminal outcome in the same call. The call always returns within ≤100 s.

The call returns with **either** a **terminal** status **or** a **non-terminal** status:

- **Terminal at submit time** (`done`, `already_merged`, `conflict`, `blocked`, `unknown_branch`, `failed`): the merge resolved within the bounded wait. Jump straight to step 4.
  - `already_merged` means the branch tip was already an ancestor of main — treat it the same as `done`.

- **Non-terminal** (`queued`, `attached`): the submission succeeded as **durable intent** — the merge worker has accepted the request and will process it. This is **not a failure**. The `request_id` in the response identifies your submission. Proceed to "Poll for completion" below.
  - `attached` means your submission was coalesced with an already-in-flight request for the same branch; you share that request's `request_id`.

### Poll for completion

Call `merge_status(request_id)` on a backoff schedule until the state is terminal:

```
mcp__escalation__merge_status(request_id="<request_id from submit>")
```

**Backoff:** start at **15 s**, cap at **60 s**. When the response contains an `eta_seconds` field, use that value as the sleep duration instead (capped at 60 s).

**Live states** (`queued`, `verifying`, `gate`, `finalizing`) — keep polling.

**Terminal states** (`done`, `conflict`, `blocked`, `abandoned`) — proceed to step 4.

**`state: "unknown"`** — the orchestrator restarted and the retention ring no longer holds this request. The response includes `hint: "check git log main"`. Run:
```bash
git log main --oneline -20  # confirm whether the merge already landed
```
If the commit is on main: treat as `done`. If not: resubmit (go back to step 3).

### 4. Handle the outcome

The outcome arrives from either the submit call (terminal at submit time) or the poll loop. Handle each status:

**`done`** — Merge succeeded. Main has been advanced atomically.
- Update the task: `set_task_status(id="<TASK_ID>", status="done", project_root="<PROJECT_ROOT>", done_provenance={"kind": "merged", "commit": "<merge-commit-sha>"})`
  - Use `{"kind": "merged", "commit": "<sha>"}` when this branch's merge commit landed on main (the normal case — the merge tool's return value has the merge SHA). The server backstops with `git merge-base --is-ancestor <sha> main`.
  - Use `{"kind": "found_on_main", "commit": "<landing sha>", "note": "<one-sentence explanation>"}` when the implementation is already on main from a sibling task / prior orchestrator run; the server runs the same git merge-base --is-ancestor backstop as for kind="merged".
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
