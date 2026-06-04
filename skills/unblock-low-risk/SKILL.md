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
6. **Freshness gate (mechanical).** From the **primary dark-factory checkout** (where `.venv/` and
   `orchestrator/` live — NOT the worktree, which has no `.venv`), run:

   ```
   .venv/bin/python -m orchestrator.b3_gate check \
     --task-id <task_id> \
     --worktree <worktree> \
     --project-root <project_root> \
     --category <category> \
     --config <project_root>/orchestrator/config.yaml
   ```

   `<category>` is the escalation category already asserted in precondition 4. The gate reads
   `metadata.dry_run_proposals[-1]` itself — do **not** hardcode any sha.

   Parse JSON stdout; `verdict` is one of `fresh | drift | abort`. **ONLY `verdict == "fresh"`
   proceeds.** Any other verdict (both `drift` and `abort`) → **ABORT**, copying the gate's `reason`
   field verbatim into the return value's `reason`:
   `"freshness gate (b3_gate check): <gate reason>"`.

   This mechanically enforces what was previously heuristic: the recorded `head_sha` hard-stop (P1:
   HEAD moved → `abort`) and file-scoped main drift (P2 → `drift`).

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

7. **Charge the merge slot.** From the **primary dark-factory checkout**, run:

   ```
   .venv/bin/python -m orchestrator.b3_gate charge \
     --task-id <task_id> \
     --project-root <project_root> \
     --config <project_root>/orchestrator/config.yaml
   ```

   Note: `charge` takes **no `--worktree`** argument — cap state is keyed on `project_root` only.

   Parse JSON stdout; if `charged` is `false` (rolling-24h cap exceeded), **ABORT**, carrying the
   gate's `reason` verbatim into the return value's `reason`. Only `charged: true` proceeds.

   Rationale (PRD §4.2): the charge sits at the merge-submit choke point — the actual unattended-merge
   risk axis — so every ABORT before this step (preconditions, scope check, rebase conflict, verify
   failure) spends no slot and is free by design.

   Note: once `charged: true` is returned the slot is consumed even if the merge does not succeed.
   Completion is observed via `merge_status` polling (not a blocking return); outcomes that cost a
   slot: `{'error': ...}` (orchestrator down), or a polled/resolved state of `conflict`, `blocked`,
   `abandoned`, `superseded`, `failed`, `unknown_branch`, or `unknown` (unconfirmed on main). These
   post-charge aborts cost a slot. This is the accepted §4.2 tradeoff.

8. **Merge via the queue — never directly.** Submit:

   ```
   mcp__escalation__merge_request(task_id=task_id, branch=task_id,
       worktree=<abs path>, description="unblock-low-risk: <one-line summary>",
       wait_secs=100)
   ```

   `branch` is the **bare task id** — the worker prepends `task/`. The server clamps the wait to
   ≤100 s (its internal `_MAX_WAIT_SECS`); `wait_secs=100` is the PRD-prescribed bounded value.

   **If it returns `{'error': ...}` (orchestrator not running) → ABORT.** Never fall back to a
   direct `git merge` — an unattended direct merge to main is exactly the risk this skill refuses
   to take.

   The response shape determines the next action:

   - **TERMINAL (resolved within the bounded window):** `status` ∈ `done` | `conflict` | `blocked`
     | `already_merged` | `unknown_branch` | `failed`. The call also returns a `request_id` in
     all cases except `already_merged`, which short-circuits before entry construction and returns
     no `request_id`. Proceed to step 9.

   - **NON-TERMINAL — `status` ∈ `queued` | `attached`:** This is a **successful submission**, not
     a failure. `queued` means the entry is waiting in the merge queue; `attached` means it was
     coalesced with an existing in-flight request. Poll `merge_status` until the entry reaches a
     terminal state:

     ```
     while state ∈ {queued, verifying, gate, finalizing}:
         wait = clamp(eta_seconds if eta_seconds else 30, min=15, max=60)
         sleep(wait)
         result = mcp__escalation__merge_status(request_id)
     ```

     Use `eta_seconds` from each `merge_status` response as the cadence hint; clamp to [15 s,
     60 s]. Proceed to step 9 with the final `result.state`.

9. **Handle the outcome** (use the polled `merge_status` `state` from step 8's polling loop, or the
   `status` returned directly by the bounded `merge_request` call if it resolved in-window — treat
   them uniformly):

   - **`done` (polled state) or `already_merged` (submit-time fast-path):** success.
     `already_merged` carries no `request_id` — nothing to poll or cancel.
     a. `set_task_status(id=task_id, status="done", project_root=project_root,
        done_provenance={"kind": "merged", "commit": "<merge sha>"})`.
     b. **Restore metadata** — `set_task_status` overwrites the metadata blob, nuking
        `dry_run_proposals` / `memory_hints` / `files`. Immediately
        `update_task(..., append=true)` to restore them.
     c. Clean up: `git worktree remove .worktrees/<task_id>` and `git branch -d task/<task_id>`.
     d. `mcp__escalation__resolve_issue(escalation_id, resolution="Auto-merged low-risk fix
        (unblock-low-risk): <what changed + merge sha>", resolved_by="unblock-low-risk")`.

   - **`conflict` | `blocked` | `abandoned` | `superseded` | `failed` | `unknown_branch`:**
     call `mcp__escalation__merge_cancel(request_id)` then **ABORT**. Do not resolve, do not
     retry, do not direct-merge.

   - **`unknown`** (e.g., after an orchestrator restart; `merge_status` carries
     `hint="check git log main"`): fall back to `git log main --oneline -20` and check whether
     the task's commit landed on main.
     - Confirmed on main → success; use `done_provenance={"kind": "found_on_main"}` and proceed
       with sub-steps a–d above.
     - Not found → `mcp__escalation__merge_cancel(request_id)` then **ABORT**.

## Aborting

On any ABORT:
- **Cancel any live durable-intent entry.** If the ABORT occurs AFTER a successful
  `merge_request` submission (i.e., a `request_id` was returned — which excludes the
  `already_merged` fast-path, which returns no `request_id`), call
  `mcp__escalation__merge_cancel(request_id)` FIRST, before anything else. This prevents
  an orphaned durable-intent entry (PRD D2 — submitted merges now outlive the MCP call and
  session) from accumulating. `merge_cancel` never raises and is a safe no-op if the entry
  already finalized (it returns `cancelled: false` with the terminal state). A coalesced or
  `attached` `request_id` may resolve to `state: "unknown"` — treat the cancel as best-effort.
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
- **Freshness enforced by `b3_gate check`** (precondition 6): proceed only on `verdict == "fresh"`;
  any `drift` or `abort` verdict → ABORT with the gate's `reason` verbatim.
- **Rolling-24h merge cap enforced by `b3_gate charge`** (step 7, immediately before `merge_request`):
  a refused charge (`charged: false`) → ABORT. All aborts before this step cost no slot.
- Merge ONLY through `merge_request` with explicit `wait_secs=100`; completion is awaited ONLY via
  `merge_status` polling (15 s→60 s backoff using `eta_seconds`); never a direct `git merge`;
  never `--no-verify`.
- **Cancel on abort:** any ABORT after a successful submission (a `request_id` was returned) MUST
  call `merge_cancel(request_id)` first — so no durable-intent entry outlives the aborted run.
  Skip only when `already_merged` (no `request_id` exists). The call is a safe no-op on terminal
  entries.
- One attempt. No retry loops. ABORT on the first sign of doubt.
- After `set_task_status`, restore metadata via `update_task(append=true)`.
