---
name: merge-queue
description: "Merge a task branch to main via the orchestrator's merge queue using the submit→poll protocol. Use this skill whenever you need to merge a completed task branch into main and the orchestrator might be running — it submits via merge_request(wait_secs=100) then polls merge_status until the outcome is terminal, preventing races without blocking indefinitely. Trigger this when an agent says 'merge to main', 'submit merge', 'merge task branch', finishes fixing a blocked task and needs to merge, or any time code on a task branch is ready to land on main. If the escalation MCP isn't reachable, the skill falls back to direct merge. Prefer this over raw git merge --no-ff whenever working in the dark-factory repo."
---

# Merge Queue

When the orchestrator is running, all merges to main go through the **merge queue** — a serial worker that rebases, verifies, and atomically advances main using compare-and-swap. This prevents races between concurrent tasks, the steward, and interactive sessions.

The escalation MCP exposes a `merge_request` tool that lets you submit to this queue from outside the orchestrator workflow. This skill tells you how to use it.

> **Core rule:** every `merge_request` call passes an explicit bounded `wait_secs`; completion is awaited only via `merge_status` polling.
>
> The server default is now `0` (immediate, non-blocking). Passing an explicit bounded `wait_secs` (use `100`) lets a fast or idle-queue merge resolve terminally within the submit call, avoiding a poll round-trip. Never omit `wait_secs`.

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
- **If it errors or times out:** the orchestrator isn't running. Fall back to direct merge (step 6).

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

- **Terminal at submit time** (`done`, `already_merged`, `conflict`, `blocked`, `unknown_branch`, `failed`, `superseded`): the merge resolved within the bounded wait. Jump straight to step 4.
  - `already_merged` means the branch tip was already an ancestor of main — treat it the same as `done`.
  - `superseded` means your request was absorbed into a coalesced train before it could be individually processed. The response includes `superseded_by: "<train_request_id>"`. Re-poll the train request immediately (step 4 / follow-the-train protocol below).

- **Non-terminal** (`queued`, `attached`): the submission succeeded as **durable intent** — the merge worker has accepted the request and will process it. This is **not a failure**. The `request_id` in the response identifies your submission. Proceed to "Poll for completion" below.
  - `attached` means your submission was coalesced with an already-in-flight request for the same branch; you share that request's `request_id`.

### Poll for completion

Call `merge_status(request_id)` on a backoff schedule until the state is terminal:

```
mcp__escalation__merge_status(request_id="<request_id from submit>")
```

**Backoff:** start at **15 s**, cap at **60 s**. When the response contains an `eta_seconds` field whose value is a **positive number**, use that value as the sleep duration (capped at 60 s); if `eta_seconds` is absent or `null`, fall back to the 15 s→60 s backoff schedule.

**Live states** (`queued`, `verifying`, `gate`, `finalizing`) — keep polling.

**Terminal states** (`done`, `conflict`, `blocked`, `abandoned`, `superseded`) — proceed to step 4.

**`state: "superseded"`** — Your request was absorbed into a coalesced train. The response includes `superseded_by: "<train_request_id>"`. **Do NOT fall back to direct merge** — the train is already in flight and a direct merge would race it. Follow the train:

```
mcp__escalation__merge_status(request_id="<superseded_by value>")
```

Poll the train request with the same 15 s→60 s backoff until it reaches a terminal state (`done`, `conflict`, `blocked`, `abandoned`). Your absorbed branch lands when the train lands. Handle the train's terminal state per step 4.

**`state: "unknown"`** — the orchestrator restarted and the retention ring no longer holds this request. `merge_status` now self-resolves a landed merge via its git-authority tier: if the branch is provably on `main` it returns `state: "done"` with `kind: "found_on_main"` and `merge_sha` directly. If `merge_status` still returns `unknown`, confirm deterministically:
```bash
git merge-base --is-ancestor task/<TASK_ID> main && echo "on main" || echo "not on main"
# If exit 0 (on main): treat as done/found_on_main — use done_provenance kind='found_on_main',
#   commit=<landing sha: git log --format=%H -1 main>
#   (git log gives the merge commit; git merge-base gives the common ancestor, NOT the merge commit)
# If exit 1 (not on main) AND queue is healthy: resubmit (go back to step 3).
```
**Never fall back to a direct merge in response to `unknown`** — `unknown` means the server lost its record, NOT that the merge failed. The direct-merge fallback (step 6) is ONLY for orchestrator down/congested.

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

**`failed`** — The merge worker encountered an unexpected error (surfaces from the submit call only; `merge_status` collapses this into `blocked`). Treat it the same as `blocked`: read any `failure_diagnostic` or `reason` field, fix the underlying problem, and resubmit.

**`unknown_branch`** — The branch ref does not exist in the target repository (surfaces from the submit call only). Likely causes: the branch name is wrong, the branch was deleted before submission, or the request was routed to the wrong repo's escalation MCP. Verify the branch exists locally (`git branch -a`) and that you're submitting to the correct escalation server.

**`abandoned`** — The submission was cancelled via `merge_cancel` before it finished (surfaces from the poll loop). If the merge is still wanted, resubmit (go back to step 3); otherwise, no further action is needed.

**`needs_rebase`** — Your branch was **bounced at conflict-graph time** (disk-free, before any verify slot was consumed) because it has a textual conflict with another branch already in the queue. The merge worker attempted a mechanical speculative rebase:
- **Clean auto-rebase:** your branch was re-queued automatically with work preserved; no agent was dispatched, and `merge_first_enqueued_at` (your aging priority) was unchanged. No action required — the queue will re-process it.
- **Real conflict:** the rebase failed; the merge was bounced with a conflict that requires resolution. Fix the conflict in your worktree and resubmit.
- **Bounce cap reached (`MERGE_BOUNCE_CAP=3`):** the 1688 thrash-backstop triggered — the task is blocked without further rebase. Read the reason, resolve the underlying conflict, and unblock manually.

**`superseded`** — Your request was absorbed into a coalesced train (surfaces from either the submit call or the poll loop). The response includes `superseded_by: "<train_request_id>"`.

**Critical: do NOT fall back to a direct merge or resubmit on `superseded`** — doing so would race the in-flight train that already carries your branch's work. Follow the train instead:

1. Take the `superseded_by` value from the response.
2. Poll `merge_status(request_id=<superseded_by>)` with the same 15 s→60 s backoff.
3. When the train reaches a terminal state, handle it exactly as you would handle that status for your own request (e.g., `done` → update task status; `conflict` → resolve and resubmit your branch).

Your absorbed branch lands when the train lands.

### 5. Abandoning a submission (merge_cancel)

To abandon a submitted merge — for example, the work was superseded, the wrong branch was submitted, or the queued entry is redundant — call:

```
mcp__escalation__merge_cancel(request_id="<request_id from submit>")
```

The response is `{ cancelled, state, reason }`:

- **`cancelled: true`** with `state: "abandoned"` — a pending waiter was dropped. The merge will not proceed.
- **`cancelled: false`** — the request was already finalized (terminal), unknown, or already cancelled. The `state` and `reason` fields explain why.

**Important:** `merge_cancel` is the **only** explicit-cancellation path. An MCP client disconnect no longer cancels the merge (durable intent), so an abandoned submission must be cancelled deliberately.

If your submit returned `status: "attached"`, your submission was coalesced with an in-flight entry. Cancel using the `request_id` returned with that `attached` response — it points to the shared in-flight entry.

### 6. Fallback: direct merge

**This fallback is ONLY for orchestrator down/congested — NEVER a response to `state: "unknown"`.**

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
| Orchestrator running | Submit via `merge_request(wait_secs=100)`, then poll `merge_status` |
| Orchestrator not running | Direct `git merge --no-ff` |
| Submit returns `queued` or `attached` | Submission succeeded (durable intent); poll `merge_status(request_id)` with 15 s→60 s backoff |
| `merge_status` returns `state: "unknown"` | Check `git merge-base --is-ancestor task/<TASK_ID> main` (exit 0 → done/found_on_main; exit 1 + healthy queue → resubmit). Never direct-merge in response to `unknown`. |
| Outcome `conflict` | Fix in worktree, resubmit |
| Outcome `blocked` | Read reason, fix, resubmit |
| Outcome `done` or `already_merged` | Update task status, clean up |
| Outcome `superseded` | Re-poll `merge_status(superseded_by)` until terminal; do NOT direct-merge or resubmit |
| Outcome `needs_rebase` — auto-rebase succeeded | Queue re-processed automatically; no action needed |
| Outcome `needs_rebase` — real conflict or cap | Fix conflict in worktree, resubmit (or unblock if cap reached) |
| Abandon a queued submission | `merge_cancel(request_id)` — the only explicit-cancellation path |
| Unsure if orchestrator is running | Probe `get_pending_escalations()` — if it responds, use the queue |

## The two-layer merge queue

The orchestrator's merge queue uses a **two-layer pipeline** to separate fast conflict detection (disk-free, reorderable) from in-flight verification (immutable, frozen).

### Layer 1: Speculative merge graph (suffix)

The unfrozen suffix (`_lane_buffers`) holds items that are queued but not yet verifying. The **conflict graph** (`suffix_conflict_graph`) tracks footprint overlaps between suffix items. At each recompute cycle (`recompute_suffix_conflict_graph()`), the queue detects textual conflicts and:

- **Bounces the younger conflicting suffix item** (`_bounce_conflicting_suffix_items`) — graph-time, disk-free, before any verify slot is consumed or `_merge-*` worktree is created.
- Attempts a **mechanical speculative rebase** first: if clean, the item is re-queued with work preserved and `merge_first_enqueued_at` unchanged; if a real conflict is found, the bounce escalates; if the cap (`MERGE_BOUNCE_CAP=3`) is hit, the 1688 thrash-backstop triggers → blocked without further rebase.

Reordering within the suffix is always disk-free (only recomputes the merge-tree for the affected suffix items, never touches in-flight verify state).

### Layer 2: Frozen verify frontier (prefix)

The **frozen prefix** = {verifying} ∪ {landed}. Items in the frozen prefix are **immutable**:
- A verify is always dispatched against the tip of the frozen prefix (`frozen_prefix_tip`).
- No reorder or re-base ever touches an in-flight verify item.
- The suffix recompute only touches `_lane_buffers`; in-flight `_inflight` order and `base_sha` are unchanged.

Health check: `two_layer_invariants(main_sha)` → `[]` when all §5.3 invariants hold.

### Conflict-clique aging order and disjoint throughput bypass

Within a footprint conflict clique, items are ordered by **age of first submission** (`_aging_key = (merge_first_enqueued_at, request_id)`). The oldest first-submission wins — preserving the most expensive work. `merge_first_enqueued_at` is persisted write-once in task metadata and survives orchestrator restarts (falls back to `enqueued_at` for legacy entries).

Items whose footprint is **disjoint from everything ahead** bypass out-of-order for throughput — a disjoint item never waits behind a blocked clique.

### No-landings circuit-breaker

When the landing rate ≈ 0 over a window **and** warm-lane free bytes are falling, `NoLandingsCircuitBreaker` automatically:
1. Calls `force_halt_scheduler` to stop dispatch.
2. Files an L2-INFO escalation (role `orchestrator-no-landings-breaker`).
3. Auto-resumes when a clean landing occurs (`landings_total` rises) or disk recovers.

### Operator-observable heartbeat keys

The `snapshot()` call exposes these additive, backward-compatible keys:

| Key | What it shows |
|-----|---------------|
| `suffix_conflict_graph` | Current conflict-graph edges for suffix items |
| `frozen_prefix` | `{request_ids, tip_merge_commit, verify_depth}` |
| `metrics` | `{retries_per_landing, drift_at_detection, landings_total}` |
| `two_layer_invariants` | `[]` when healthy; list of violation strings otherwise |

For the full architectural companion, see [references/two-layer-model.md](references/two-layer-model.md).

### Related work

- **Warm-lane Δp space-safety batch (1859–1861 / reify 4716–4719):** attacks Δp on the *task-dispatch* path (warm-lane disk-space gates). Complementary to the merge-queue path targeted here; no shared seam.
- **Merge-verify ENOSPC fail-soft (workflow.py transient-infra block → re-queue):** handles ENOSPC at the individual verify step. This is a separate symptom task and is **out of scope** for the two-layer merge queue (§10 of the PRD). Referenced here for orientation only.
