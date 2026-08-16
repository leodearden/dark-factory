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
  wait_secs=100,
  verified_green=<True|False>  # decide per the verified_green rule below — do not default to True
)
```

Parameters:
- `task_id` — the task number (string)
- `branch` — the task ID only (e.g., `"466"`), **not** the full branch name. The merge worker prepends the `task/` prefix automatically.
- `worktree` — absolute path to the task's worktree (e.g., `/home/leo/src/dark-factory/.worktrees/42/`)
- `description` — optional context for logs
- `wait_secs=100` — **always pass this explicitly.** The value 100 equals the server's maximum bounded wait (`_MAX_WAIT_SECS`), so fast/idle-queue merges return their terminal outcome in the same call. The call always returns within ≤100 s.
- `verified_green` — a **per-submission decision, not a default.** Pass `True` **only** when the verification suite (tests, lint, type-check from step 1 "Prepare your branch") actually ran and passed on this branch, on top of its own base; omit it (or pass `False`) if you skipped verification, or it failed and you're submitting anyway for an unrelated reason (e.g. resubmitting after a `conflict`/`blocked` fix-up you haven't re-verified). This vouches to the merge queue that the branch was seen green pre-merge — it emits a `workflow_verify` event so a later merge failure caused by an unrelated main landing can be attributed as `INTEGRATION_SKEW` instead of degrading to `INDETERMINATE`. Since `/do` lands its branches through this skill, honoring this rule here also covers `/do` submissions. `/unblock`'s step 6 D2 is this same rule, with an added D1/D2 split for its rebase/re-verify loop — an agent holding both skills is applying one rule, not two.
  - **Caution — not retractable:** the classifier's green fact is *any-prior-green, keyed by task ID* — once `verified_green=True` has been emitted for a task ID, omitting it (or passing `False`) on a **later** resubmission for that same task ID does not undo the earlier green. If you resubmit after an unverified fix-up (e.g. a quick conflict resolution you didn't re-run the suite on), a genuine `BRANCH_BUG` in that fix-up can still be misattributed to `INTEGRATION_SKEW` on a later merge failure, because the classifier only sees "this task ID was green once," not which commit. When in doubt, re-run verification before resubmitting rather than relying on an earlier `True`.

The call returns with **either** a **terminal** status **or** a **non-terminal** status:

- **Terminal at submit time** (`done`, `already_merged`, `conflict`, `blocked`, `unknown_branch`, `failed`, `superseded`): the merge resolved within the bounded wait. Jump straight to step 4.
  - `already_merged` means the branch tip was already an ancestor of main — treat it the same as `done`.
  - `superseded` means your request was replaced by a successor before it could be individually processed. The response includes `superseded_by: "<mr-* request id | coalesce-* train id>"`. See [Follow the superseded successor](#follow-the-superseded-successor) below to determine how to resolve it.

- **Non-terminal** (`queued`, `attached`): the submission succeeded as **durable intent** — the merge worker has accepted the request and will process it. This is **not a failure**. The `request_id` in the response identifies your submission. Proceed to "Poll for completion" below.
  - `attached` means your submission was coalesced with an already-in-flight request for the same branch. Whether you share that in-flight entry's `request_id` depends on the response's own disclosure — see "Poll for completion" below, which branches on the `poll_by` field the response carries alongside `source`, `inflight_request_id`, `inflight_task_id`, and `pollable`.

### Poll for completion

**Pick the poll handle from the submit response's `poll_by` field** (present on `attached` responses since task 3148; a `queued` response has no in-flight entry to disclose, so it's always polled by its own `request_id`). Read it defensively — `result.get("poll_by", "request_id")` — so a response from a server predating this field degrades to today's behaviour instead of raising:

- `poll_by == "request_id"` (or absent) — the returned `request_id` IS the handle:
  ```
  mcp__escalation__merge_status(request_id="<request_id from submit>")
  ```
- `poll_by == "task_id"` — no in-flight `request_id` is known; the returned `request_id` is your *own* submitting call's id, not a handle. Poll the in-flight task instead:
  ```
  mcp__escalation__merge_status(task_id="<inflight_task_id from submit>")
  ```
  **Terminal states on this arm are UNCONFIRMED too, for the same reason as the `branch` arm below.** A task-keyed poll is submission-scoped only while the live-snapshot tier is still serving it; once that stops (e.g. a mid-flight orchestrator restart) the durable tiers resolve it to the most-recent *finalized* record for the task (retention ring `get_by_task` → event store `latest_merge_finalized(task_id=...)`, `escalation/server.py:2244-2287`), and task_ids are reused verbatim across resubmissions exactly like branches. Apply the identical rule: a `done` carrying `kind: "found_on_main"` is already git-authoritative — **accept it directly, do not re-check ancestry**; the ancestry re-check applies only to a `done` served from the durable retention-ring/event-store tiers (no `kind`, no `merge_sha`), which you confirm with the [canonical ancestry check](#canonical-ancestry-check) below. Re-check `get_merge_queue()` before acting on a `conflict`/`blocked`/`abandoned`. Because a rejected `done` would otherwise re-serve forever, this arm shares the `branch` arm's 20-minute ceiling and the same stop-and-report exit.
- `poll_by == "branch"` (equivalently `pollable == false`) — neither handle is known: a foreign or pre-restart merger owns the worktree, so there's no in-process entry, no retention alias, and no waiter. The returned `request_id` was never enqueued — polling it directly resolves `state: "unknown"`, which here is expected, NOT a lost record. Poll by branch instead, which benefits from `merge_status`'s git-authority tier (self-resolves a landed merge to `state: "done"` / `kind: "found_on_main"`):
  ```
  mcp__escalation__merge_status(branch="task/<TASK_ID>")
  ```
  **Terminal states on this arm are UNCONFIRMED.** A branch-keyed poll resolves the most-recent *finalized* record for that branch (retention ring `get_by_branch` → event store `latest_merge_finalized(branch=...)`, `escalation/server.py:2251-2286`) — nothing ties that record to *your* submission, and the event store survives restarts. Task branches are reused verbatim across resubmissions, so a prior round's `done`/`conflict`/`blocked`/`abandoned` can satisfy the terminal test on the very first tick. **Exception — a `done` carrying `kind: "found_on_main"` is NOT subject to this:** that shape comes from the git-authority tier, a live probe of main reached only because the durable tiers missed, so it structurally cannot be a stale prior-round record. Accept it directly. For a `done` served from the durable tiers (no `kind`, no `merge_sha`), confirm with the [canonical ancestry check](#canonical-ancestry-check) below — not-landed means stale, so keep polling. On a `conflict`/`blocked`/`abandoned` from this arm, treat it as possibly predating this submission: re-check `get_merge_queue()` and who owns the worktree before resolving or resubmitting.

  Cross-check `get_merge_queue()` for queue-wide context if useful. On this arm `unknown` is the **live** state, not a first-tick fluke — it persists for as long as the foreign merger is still working, since the git-authority tier can only resolve once the merge is provably on main — by *either* of its two paths: `is_ancestor(tip, main)` while the branch ref still exists, **or** `find_merge_marker` once the ref is gone (the `elif tip is None` deleted-branch path, `escalation/server.py:2448-2459`, which makes no `is_ancestor` call at all). Do NOT read it as a terminal failure or as licence to direct-merge, and do NOT resubmit on it (the `state: "unknown"` resubmit rule below is scoped to the other two arms). **Exception to "same for all three arms" below: this arm and the `task_id` arm are bounded by a 20-minute wall-clock ceiling, starting from your first poll on that arm** (not from the original submit) — matching `skills/unblock/SKILL.md` and `skills/unblock-low-risk/SKILL.md:177-189`. At expiry, in order: (1) run the [canonical ancestry check](#canonical-ancestry-check) one final time; (2) landed (rc=0, or rc=128 with a merge-marker hit) → treat as `done`/`found_on_main` per the `state: "unknown"` handling below; (3) only if it comes back genuinely not-landed → stop and report to the human — never resubmit, never fall back to a direct merge.

Whichever handle you poll, the cadence and state handling below are the same for all three arms:

**Backoff:** start at **15 s**, cap at **60 s**. When the response contains an `eta_seconds` field whose value is a **positive number**, use that value as the sleep duration (capped at 60 s); if `eta_seconds` is absent or `null`, fall back to the 15 s→60 s backoff schedule.

**Live states** (`queued`, `verifying`, `gate`, `finalizing`) — keep polling.

**Terminal states** (`done`, `conflict`, `blocked`, `abandoned`, `superseded`) — proceed to step 4. (Unlike the other four, `superseded` doesn't tell you the actual outcome by itself — it names a successor you still need to resolve; see [Follow the superseded successor](#follow-the-superseded-successor) below.)

**`state: "superseded"`** — Your request was superseded by a successor. The response includes `superseded_by: "<mr-* request id | coalesce-* train id>"`. **Never fall back to direct merge, and never resubmit, while that successor is unresolved** — it may already be in flight, and either would race it. The successor isn't always pollable the same way your own request was; see below for the shape branch.

<a id="follow-the-superseded-successor"></a>**Follow the superseded successor.** `superseded_by` names one of two shapes, and only one of them is a request id you can poll by `request_id`:

- **`mr-*` id** (generation-advance path) — a real request id. `gen_next` was enqueued the normal way through `enqueue_merge_request`, and this outcome resolves as `MergeOutcome("superseded", superseded_by=gen_next.request_id, ...)` (`orchestrator/src/orchestrator/merge_queue.py:4387`). Poll it:
  ```
  mcp__escalation__merge_status(request_id="<superseded_by value>")
  ```
  with the same 15 s→60 s backoff, bounded by its own 20-minute wall-clock ceiling (same as the `task_id`/`branch` arms above). Handle its eventual terminal state per step 4. If this poll returns `state: "unknown"` on **any** tick — not only at the ceiling — do not fall through to the **`state: "unknown"`** handling below (it ends in a resubmit for the `request_id`/`task_id` arms, which risks racing a successor that may still be in flight); go straight to the escape below instead.

- **`coalesce-*` id** (coalesce-train path) — this names the *train*, not a request. `_COALESCE_TRAIN_ID_PREFIX = 'coalesce-'` (`merge_queue.py:5380`); `train_id = f'{_COALESCE_TRAIN_ID_PREFIX}{tip_id}-{uuid.uuid4().hex[:8]}'` (`merge_queue.py:13739`) is what every absorbed single receives as its `superseded_by` (`merge_queue.py:13787`). **Do not poll it by `request_id`.** It resolves through none of `merge_status`'s tiers: no retention-ring alias is ever recorded for a train id (the ring is keyed on `req.request_id`, `merge_queue.py:4448-4449`), no event-store `merge_finalized` row is keyed on one, and the git-authority Tier 3.5 probe is skipped whenever only `request_id` is passed (`key = branch if branch is not None else task_id; if key is not None:`, `escalation/server.py:2619-2622`). Polling it by `request_id` returns an honest `state: "unknown"` that will never resolve to anything else. (The `GroupMergeRequest` carrying the train does get its own `mr-*` `request_id`, auto-generated at construction — `merge_types.py:804` — but the caller never sees it; the train id is not a stand-in for it.)

**Escape** — for the `coalesce-*` case, or an `mr-*` poll that came back `unknown` or is still unresolved at its ceiling: stop polling by `request_id`. Fall back to the branch handle plus the [canonical ancestry check](#canonical-ancestry-check) on every tick:
```
mcp__escalation__merge_status(branch="task/<TASK_ID>")
```
Four rules make this fallback loop terminate correctly — for the full recovery procedure see the `poll["state"] == "superseded"` bullet under *Polled terminal failures* in `skills/unblock/SKILL.md` (including its `#resumed-poll` subsection):
1. **Drop `superseded` from this resumed loop's terminal set.** The branch handle re-serves the identical frozen `superseded` record on tick 1, and for a coalesce-absorbed member that record never changes — polling to a plain terminal set would just bounce you back into this same bullet. Use `("done", "conflict", "blocked", "abandoned")` plus a `superseded` whose `superseded_by` *differs* from the one you just disregarded (a genuinely new absorption — re-enter this subsection against it).
2. **rc=128-with-empty-marker does not mean "not landed" here.** A train merges only the tip branch (`tip_branch=tip_req.branch`, `merge_queue.py:13754`), so a non-tip absorbed member's commits land on main with no `Merge task/<TASK_ID> into main` marker of its own. Under rc=128-with-empty-marker, check two signals instead: (a) the **tip's** merge marker on main — `git log main --fixed-strings --grep="Merge task/<TIP_ID> into main" --max-count=1 --format=%H`, where `<TIP_ID>` is parsed off the `coalesce-<TIP_ID>-<hex>` id by stripping the `coalesce-` prefix and the trailing `-` plus 8 hex chars (`uuid.uuid4().hex[:8]`, `merge_queue.py:13739`) — not a naive split on `-`; and (b) whether this task's own scheduler status has already flipped to `done` — read with `get_task(id="<TASK_ID>", project_root="<PROJECT_ROOT>")`. The orchestrator flips it automatically once the train lands (`mark_member_done`, `orchestrator/src/orchestrator/harness.py:1192`, which calls `scheduler.mark_done(..., kind="merged", ...)`); if it already reads `done`, that flip already happened and no further write is needed. Otherwise, either signal saying landed → record it yourself: `set_task_status(id="<TASK_ID>", status="done", project_root="<PROJECT_ROOT>", done_provenance={"kind": "found_on_main", "commit": "<tip merge sha>", "note": "absorbed into train <train_id>"})`.
3. **rc=1 does not mean resubmit here.** Right after absorption the successor is typically still in flight, so ancestry rc=1 (branch ref **exists**, its commits genuinely not on main) means only "not landed *yet*" — it is not evidence that the `superseded` record is a stale prior round, and not a reason to give up. **The [canonical ancestry check](#canonical-ancestry-check)'s rc=1 resubmit line does not apply in this loop** — whether you arrived here via the `coalesce-*` arm or via an `mr-*` poll that came back `unknown`/unresolved, a still-in-flight successor makes rc=1 equally expected either way. Keep polling the branch handle under rule 1's resumed terminal set, re-running the ancestry check each tick — it re-derives the real answer from git rather than from the frozen `superseded` record. Stop-and-report only once rc=1 still holds at this resumed loop's own 20-minute wall-clock ceiling (same ceiling and stop-and-report exit already used by the `task_id`/`branch` arms above and the `mr-*` arm above) — never resubmit, never direct-merge, even at expiry. Keep this distinct from rule 2: the tip's merge marker and this task's own scheduler status are consultable **only** under rc=128-with-empty-marker, not under rc=1.
4. **`get_merge_queue()` no longer showing the train is not a landing signal.** It means only "stop waiting on the train" — and is equally consistent with a **derail**: on any non-`done` train outcome the orchestrator re-pends the still-unlanded members for solo re-merge (`_redrive_coalesce_members`, `merge_queue.py:13345`), which also removes the train from the queue with nothing of yours on main. This skill elsewhere directs you to consult `get_merge_queue()` for context (§"Poll for completion" above) — do not read its absence as evidence the train landed. On queue-absence with neither of rule 2's two signals saying landed, keep polling the branch handle under rule 1's resumed terminal set to the ceiling; never flip the task on queue-absence alone.

**Whichever shape `superseded_by` takes, never direct-merge and never resubmit while a successor is unresolved.**

**`state: "unknown"`** — for the `request_id`/`task_id` poll arms, this means the orchestrator restarted and the retention ring no longer holds this request (a record that *was* enqueued). `merge_status` now self-resolves a landed merge via its git-authority tier: if the branch is provably on `main` it returns `state: "done"` with `kind: "found_on_main"` and `merge_sha` directly. If `merge_status` still returns `unknown`, confirm deterministically:
<a id="canonical-ancestry-check"></a>**The canonical ancestry check — three outcomes, not two.** Every "is it on main?" confirmation in this skill means this check. **Never use the two-way idiom `git merge-base --is-ancestor ... && echo "on main" || echo "not on main"`**: a deleted branch ref exits **128**, which that idiom silently reports as "not on main" — inverting the truth for the normal post-merge state, since the merge lane deletes task branches on cleanup (`_delete_branch_if_on_main`, `orchestrator/src/orchestrator/git_ops.py:7538-7574`), and on the `branch` arm a *foreign* merger's cleanup deletes it out from under you.
```bash
git merge-base --is-ancestor task/<TASK_ID> main; rc=$?; echo "ancestry rc=$rc"
# The trailing `echo` is REQUIRED, not decoration. `--is-ancestor` prints
# nothing on rc=0 OR rc=1, and the `rc=$?` assignment itself exits 0, so
# without it the tool reports exit 0 and identical empty output for "on
# main" and "NOT on main" -- silence you would have to guess at. Echoing the
# numeric rc is NOT the two-outcome `&& echo` idiom banned above: it prints
# on every path and keeps all three outcomes distinguishable. Do not "tidy"
# it away.
# rc=0   → on main. Treat as done/found_on_main — done_provenance kind='found_on_main',
#          commit=<landing sha: git log --format=%H -1 main>
#          (git log gives the merge commit; git merge-base gives the common ancestor, NOT it)
# rc=128 → branch ref is GONE ("fatal: Not a valid object name"). This is the normal state
#          AFTER a successful merge + cleanup — NOT "not on main". Search main for THIS
#          branch's merge commit, by exact subject:
git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" --max-count=1 --format=%H
#          Non-empty output → that SHA IS the true merge commit; treat as done/found_on_main.
#          Empty output    → not landed (branch never existed, or never merged).
# rc=1   → genuinely not on main. AND queue is healthy: resubmit (go back to step 3).
```
**The rc=128 search must be the exact-subject one above — never an unfiltered `git log main --merges | head -5`.** That listing takes no task argument, so on any repo with merge history it always prints something: "a hit" becomes unconditionally true and "no hit" unreachable, and every rc=128 — *including a typo'd branch name, the wrong worktree, or a branch never pushed*, all of which also exit 128 — would be recorded as landed with some unrelated task's merge SHA. The server's `done_provenance` backstop is only `git merge-base --is-ancestor <sha> main`, which any recent merge on main passes, so nothing downstream catches it.

This mirrors the in-repo authority, `GitOps.find_merge_marker` (`orchestrator/src/orchestrator/git_ops.py:7862-7905`) — the same function the git-authority tier calls on the deleted-branch path. `--fixed-strings` against the exact subject from `_merge_subject(branch, main_branch)` (`git_ops.py:1874`, canonical form `Merge <full-branch> into <main-branch>`) is what makes it substring-safe: `Merge task/1 into main` cannot match inside `Merge task/10 into main`, because the `0` falls where the pattern has a space. Do **not** substitute a bare `--grep="task/<TASK_ID>"` — BRE, not restricted to merges, matches any commit merely *mentioning* the task, and re-opens that collision. If a project overrides `git.branch_prefix` (default `task/`) or `git.main_branch`, build the subject from `_merge_subject` rather than hardcoding.
**Never fall back to a direct merge in response to `unknown`** — `unknown` means the server lost its record, NOT that the merge failed. The direct-merge fallback (step 6) is ONLY for orchestrator down/congested. **This block's `resubmit` line does not apply to the `poll_by == "branch"` arm** — there nothing was ever enqueued, so `unknown` is that arm's expected live state, not a lost record; see its 20-minute ceiling above for the bounded exit instead. **Nor does it apply when this check is reached from [Follow the superseded successor](#follow-the-superseded-successor)'s Escape** — that subsection's rule 3 governs rc=1 there instead; do not resubmit.

### 4. Handle the outcome

The outcome arrives from either the submit call (terminal at submit time) or the poll loop. Handle each status:

**`done`** — Merge succeeded. Main has been advanced atomically.
- Update the task: `set_task_status(id="<TASK_ID>", status="done", project_root="<PROJECT_ROOT>", done_provenance={"kind": "merged", "commit": "<merge-commit-sha>"})`
  - Use `{"kind": "merged", "commit": "<sha>"}` when this branch's merge commit landed on main (the normal case). The server backstops with `git merge-base --is-ancestor <sha> main`.
  - **Where the SHA comes from depends on which call produced the `done`, and not every source is safe to record as `kind: "merged"`.** A terminal-at-submit-time `merge_request` response carries the true merge SHA in `commit` — safe to use directly. A *polled* `done` does not, unless the response carries `merge_sha` (the git-authority tier's `kind: "found_on_main"` shape) — the durable retention-ring/event-store tiers return only `state`/`request_id`/`generation`/`outcome`/`finished_at` (`escalation/server.py:2404-2420`), and `outcome` is the raw state string `"done"`, not a commit hash. **A `merge_sha` is not necessarily the merge commit**: on the live-branch resolution path (branch still exists, `is_ancestor` hit) it is the *branch tip* SHA — for a `--no-ff` merge, a distinct commit from the actual merge commit — while only the deleted-branch (`find_merge_marker`) path's `merge_sha` is the true merge-commit SHA (`_found_on_main_response`'s docstring, `escalation/server.py:2290-2305`). So: record a bare `merge_sha` as `{"kind": "found_on_main", "commit": "<merge_sha>", "note": "<explanation>"}`, not `kind: "merged"` — only re-derive and use the actual merge commit if `kind: "merged"` provenance is specifically wanted — and re-derive it with the exact-subject search, `git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" --max-count=1 --format=%H`, **not** by eyeballing `git log main --oneline | head -5` (unscoped to this task, so any SHA picked from it is likely an unrelated task's merge, and the server's only backstop — `git merge-base --is-ancestor <sha> main` — passes for every recent commit on main and would not catch it). If the response has neither `commit` nor `merge_sha` — including on the `poll_by == "branch"` arm, where a durable record can resolve `done` with no `merge_sha` — re-derive the merge commit the same way. Never fabricate a SHA; if the search comes back empty, fall back to `{"kind": "found_on_main", "commit": "<landing sha>", "note": "<explanation>"}` with a SHA you've actually verified is on main.
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

**`superseded`** — Your request was superseded (surfaces from either the submit call or the poll loop). The response includes `superseded_by: "<mr-* request id | coalesce-* train id>"`.

**Critical: do NOT fall back to a direct merge or resubmit on `superseded`** — a successor may already be in flight, and either would race it. Follow the successor instead:

1. Take the `superseded_by` value from the response.
2. Determine how — and whether — to poll it per [Follow the superseded successor](#follow-the-superseded-successor) above; the handle depends on whether the value is an `mr-*` request id or a `coalesce-*` train id.
3. When the successor reaches a terminal state, handle it exactly as you would handle that status for your own request (e.g., `done` → update task status; `conflict` → resolve and resubmit your branch).

Your branch lands when the successor lands — but for a `coalesce-*` absorption of a non-tip member, that is confirmed via the tip's merge marker and this task's own scheduler status, not via `merge_status` (see the linked subsection).

### 5. Abandoning a submission (merge_cancel)

To abandon a submitted merge — for example, the work was superseded, the wrong branch was submitted, or the queued entry is redundant — call:

```
mcp__escalation__merge_cancel(request_id="<request_id from submit>")
```

The response is `{ cancelled, state, reason }`:

- **`cancelled: true`** with `state: "abandoned"` — a pending waiter was dropped. The merge will not proceed.
- **`cancelled: false`** — the request was already finalized (terminal), unknown, or already cancelled. The `state` and `reason` fields explain why.

**Important:** `merge_cancel` is the **only** explicit-cancellation path. An MCP client disconnect no longer cancels the merge (durable intent), so an abandoned submission must be cancelled deliberately.

If your submit returned `status: "attached"`, whether the returned `request_id` cancels the shared in-flight entry depends on that response's `poll_by`:

- `poll_by == "request_id"` — the returned `request_id` IS the in-flight entry's id; cancel it as shown above.
- `poll_by == "task_id"` or `poll_by == "branch"` — the returned `request_id` names your own coalesced submission, not the in-flight entry. Treat the cancel as best-effort: `cancelled: false` / `state: "unknown"` is the expected outcome here, not evidence of a problem. On `"branch"` specifically, a foreign or pre-restart merger owns the worktree — reaching the real entry isn't this caller's cancel to make. Re-check the actual state first (`merge_status(task_id=...)` / `merge_status(branch="task/<TASK_ID>")`, which self-resolves a landed merge via the git-authority tier), then the `git merge-base --is-ancestor` confirmation described under `state: "unknown"` above, before deciding anything further.

Separately — if you're holding a `coalesce-*` train id from a `superseded` response's `superseded_by` (not from an `attached` submit's `poll_by`), it is **not** cancellable via `merge_cancel` either: the train was never registered as a waiter under that id, so the call resolves to `cancelled: false` / `state: "unknown"` for the same reason it can't be polled by `request_id` (`escalation/server.py:2679-2681`: "callers holding a coalesced id will resolve to 'unknown' here"). There is no cancel path for an in-flight train — see [Follow the superseded successor](#follow-the-superseded-successor) instead.

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
| Orchestrator running | Submit via `merge_request(wait_secs=100, verified_green=...)` — `True` only when verification actually passed, `False` otherwise — then poll `merge_status` |
| Orchestrator not running | Direct `git merge --no-ff` |
| Submit returns `queued` or `attached` | Durable intent confirmed; poll the handle `poll_by` names (`request_id` for `queued` and the `poll_by=="request_id"` arm; `task_id`/`branch` when `attached` discloses them) with 15 s→60 s backoff |
| `merge_status` returns `state: "unknown"` | `request_id`/`task_id` arms: run the canonical ancestry check (rc=0 → done/found_on_main; rc=128 → branch ref gone; confirm with the exact-subject search `git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" --max-count=1 --format=%H` — non-empty = landed, empty = not landed; never an unfiltered `--merges` listing; rc=1 + healthy queue → resubmit). `poll_by == "branch"` arm: `unknown` is that arm's live state — keep polling to the 20-min ceiling, then stop and report, never resubmit. Never direct-merge in response to `unknown`. |
| Terminal state arrived on an unscoped arm (`poll_by` `"branch"` or `"task_id"`) | Unconfirmed — a branch- or task-keyed poll can match a *prior* round's record on the same reused branch/task_id. A `done` with `kind: "found_on_main"` is git-authoritative — accept directly. Otherwise accept `done` only when the canonical ancestry check says landed (rc=0, or rc=128 with a merge-marker hit); re-check `get_merge_queue()` before acting on a `conflict`/`blocked`/`abandoned`. |
| Outcome `conflict` | Fix in worktree, resubmit |
| Outcome `blocked` | Read reason, fix, resubmit |
| Outcome `done` or `already_merged` | Update task status, clean up |
| Outcome `superseded` | `superseded_by` is `mr-*` → poll it by `request_id` (own 20-min ceiling); `coalesce-*` → names the train, not a request — `merge_status(request_id=...)` returns `unknown` forever, so fall back to `merge_status(branch=...)` + the canonical ancestry check (rc=128-with-empty-marker is expected for a non-tip member; rc=1 means not-landed-*yet*, not resubmit — rule 3). See [Follow the superseded successor](#follow-the-superseded-successor). Never direct-merge, never resubmit. |
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
| `hosts` | Per-verify-host `{name, is_local, slot_state, quarantined, quarantine_class, unavailable_since, unavailable_secs, streak, reason}` — tells RU-quarantine (`ru`) from divergence-quarantine (`divergence`) from a leaked slot (busy/parked, no occupant) from free-and-never-asked-for. An RU-tracked host with no allocator slot is appended with `slot_state: null` |

When any host is quarantined the heartbeat line also carries an inline
` | DEGRADED <n>/<m> hosts quarantined: <name>=<class>` segment, and the
`merge_heartbeat` event carries the same `hosts` block structurally.

For the full architectural companion, see [references/two-layer-model.md](references/two-layer-model.md).

### Related work

- **Warm-lane Δp space-safety batch (1859–1861 / reify 4716–4719):** attacks Δp on the *task-dispatch* path (warm-lane disk-space gates). Complementary to the merge-queue path targeted here; no shared seam.
- **Merge-verify ENOSPC fail-soft (workflow.py transient-infra block → re-queue):** handles ENOSPC at the individual verify step. This is a separate symptom task and is **out of scope** for the two-layer merge queue (§10 of the PRD). Referenced here for orientation only.
