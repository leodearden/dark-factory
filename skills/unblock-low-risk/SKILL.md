---
name: unblock-low-risk
description: "Autonomous, NON-INTERACTIVE unblock of a single blocked task — but ONLY when the orchestrator's at-block-time dry-run investigation labelled the fix `risk_label == 'low'`. This is the unattended counterpart to /unblock, built for AFK windows: the L2 escalation-watcher launches it as a sub-agent (no terminal) for `task_failure`/`review_issues` escalations whose latest `metadata.dry_run_proposals[-1]` is low-risk and fresh. It re-derives the fix from the proposal prose, applies it scoped to `files_referenced`, runs the project verify suite, and merges via the orchestrator merge queue — aborting cleanly (leaving the escalation pending for a human) on ANY doubt: non-low risk, stale proposal, scope creep, rebase conflict, verify failure, or a merge-queue result other than done. It NEVER touches main directly, never --no-verify, never retries. NOT for: medium / human-review-required proposals, interactive unblocking (use /unblock), or any task without a low-risk dry-run proposal."
---

# Unblock — Low-Risk (autonomous, non-interactive)

You are an **unattended** sub-agent resolving ONE blocked task whose at-block-time dry-run
investigation produced a **`low`-risk** fix proposal. No human is watching. Your job is to apply the
fix, verify it, and merge it through the orchestrator merge queue — **or abort cleanly and leave the
work for a human**. There is no middle ground and no retry: you get one careful attempt.

**Caution is the whole point.** You exist because a human is AFK and the alternative is the task
sitting blocked for days. But a wrong unattended merge is far worse than a delayed one. When *any*
precondition or step below is not unambiguously satisfied, **ABORT** (see "Aborting"). Aborting is a
success, not a failure — it routes the item back to the human exactly as if you had never run.

## Inputs (provided by the caller)

The escalation-watcher passes you: `task_id`, `escalation_id`, `project_root`, the `worktree` path,
and the latest dry-run `proposal` object (`proposal_text`, `files_referenced`, `risk_label`,
`timestamp`). If any is missing, ABORT immediately.

## Hard preconditions — re-check ALL of them yourself

The watcher pre-gated, but you re-assert defensively (state can change between the gate and now):

1. `get_task(task_id)` → `metadata.dry_run_proposals` is non-empty; take `latest = proposals[-1]`.
2. `latest['risk_label'] == 'low'`. Anything else (`medium`, `human-review-required`) → **ABORT**.
3. `latest` has **no** `status` key (`investigation_failed` / `budget_exhausted` entries force
   `human-review-required` anyway, but check explicitly) → else **ABORT**.
4. The escalation `category` is `task_failure` or `review_issues` → else **ABORT**.
5. The `worktree` directory exists and `git -C <worktree> status` is clean-or-sane (a task branch,
   not detached, not mid-rebase/merge) → else **ABORT**.
6. **Freshness:** `latest` is genuinely the last entry, its `timestamp` is the most recent, and the
   branch has no commits you can't account for since it was investigated. The proposal carries no
   HEAD sha, so this is heuristic — if the branch state looks materially different from what the
   proposal describes (referenced files gone, unexpected commits), **ABORT**.

## Procedure

Run these strictly in order. Stop and ABORT at the first step that is not cleanly satisfied.

1. **Release the orchestrator's grip.** `mcp__escalation__release_workflow(task_id, timeout_secs=30)`.
   This soft-cancels any active workflow and parks the task as `blocked` (the reaper-immune holding
   state). Inspect the result: if `was_active` is true but `slot_cleared` is false, the orchestrator
   is still working the task — **ABORT** (do not race it).

2. **Understand the issue.** In the worktree, read `latest['proposal_text']`, the review findings
   (`.task/reviews/*.json`), and the failing iteration (`.task/iterations.jsonl`). Treat the proposal
   as a *starting hypothesis*, not gospel — the investigating agent was read-only and may be wrong.

3. **Apply the fix — scoped.** Edit/Write **only** files in `latest['files_referenced']` (and their
   direct test files). The moment the correct fix demands touching a file outside that set — or any
   of `main`-only paths, CI config, infra, or `orchestrator/config.yaml`/`.mcp.json`/systemd units —
   the change is no longer low-risk: **ABORT**. Do not commit `.task/`.

4. **Rebase onto main.** `git -C <worktree> rebase main` (main may have moved). On **any conflict** →
   **ABORT** (a conflict means the change is no longer self-contained).

5. **Verify.** Run the project's full verify suite from `orchestrator/config.yaml` in the worktree —
   `test_command`, then `lint_command`, then `type_check_command` (read them from the file; do not
   hardcode). Any non-zero exit → **ABORT**. (Pipe to a file + check exit status; never trust a
   tail.)

6. **Commit.** `git -C <worktree> add <only the files you changed>` then commit on `task/<task_id>`
   with a clear message. Never `git add -A` (avoids `.task/` contamination); never `--no-verify`.

7. **Merge via the queue — never directly.** `mcp__escalation__merge_request(task_id=task_id,
   branch=task_id, worktree=<abs path>, description="unblock-low-risk: <one-line summary>")`. `branch`
   is the **bare task id** — the worker prepends `task/`. This blocks until the merge worker (which
   re-rebases and runs authoritative post-merge verification) finishes. **If it returns `{'error':
   ...}` (orchestrator not running) → ABORT.** Do NOT fall back to a direct `git merge` — an
   unattended direct merge to main is exactly the risk this skill refuses to take.

8. **Handle the outcome:**
   - **`done` / `already_merged`:** success.
     a. `set_task_status(id=task_id, status="done", project_root=project_root,
        done_provenance={"kind": "merged", "commit": "<merge sha>"})`.
     b. **Restore metadata** — `set_task_status` overwrites the metadata blob, nuking
        `dry_run_proposals` / `memory_hints` / `files`. Immediately
        `update_task(..., append=true)` to restore them.
     c. Clean up: `git worktree remove .worktrees/<task_id>` and `git branch -d task/<task_id>`.
     d. `mcp__escalation__resolve_issue(escalation_id, resolution="Auto-merged low-risk fix
        (unblock-low-risk): <what changed + merge sha>", resolved_by="unblock-low-risk")`.
   - **Anything else** (`conflict`, `blocked`, `failed`, `unknown_branch`, `in_flight`): **ABORT.**
     Do not resolve, do not retry, do not direct-merge.

## Aborting

On any ABORT:
- **Do NOT change the task status** beyond the `blocked` park that `release_workflow` already applied
  (that is the safe, sweep-immune state — leave it there).
- **Leave the escalation `pending`.** Do not resolve or dismiss it — the human will handle it on
  return, exactly as the normal `task_failure`/`review_issues` path does.
- Leave the worktree intact (do not remove it) so the human can inspect your partial work.
- Return a structured result so the watcher can log it to the digest.

## Return value (your final message IS the data)

Return ONLY this JSON object (no prose):

```json
{
  "outcome": "merged" | "aborted",
  "task_id": "<id>",
  "escalation_id": "<id>",
  "reason": "<merged: what changed + merge sha | aborted: which precondition/step failed and why>",
  "commit": "<merge sha if merged, else null>"
}
```

## Non-negotiable safety rails (summary)

- Only `risk_label == 'low'`; only `task_failure` / `review_issues`.
- Edits scoped to `files_referenced`; never main-only / CI / infra / config.
- Merge ONLY through `merge_request`; never a direct `git merge`; never `--no-verify`.
- One attempt. No retry loops. ABORT on the first sign of doubt.
- After `set_task_status`, restore metadata via `update_task(append=true)`.
- Future hardening (not yet available): the proposal has no recorded HEAD sha, so the freshness gate
  is heuristic. If a `head_sha` field is later added to dry-run proposals, treat a mismatch against
  the current worktree HEAD as a hard ABORT.
