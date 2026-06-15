# PRD: `merge_status` false-`unknown` → redundant land.sh

**Status:** authored 2026-06-15. Decompose target: `dark_factory` (`/home/leo/src/dark-factory`).

## Problem / consumer

A merge that **actually landed on `main`** can be reported by `merge_status` as
`state="unknown"`. During the reify 4352 incident (2026-06-14) the orchestrator
queue merge succeeded (`mr-5c4fd75a`, full gate passed, landed `5257df0662`), but
a follow-up `merge_status` returned `unknown`; reading that as "didn't land," the
session fell back to the manual `scripts/land.sh task/4352` path — a redundant
duplicate that aborted with no effect on main, but wasted a cycle and produced the
confusing "4352 is being merged by land.sh" state, while the merge queue was
halted.

`merge_status` (`escalation/src/escalation/server.py:1221`) walks four tiers —
live snapshot → retention ring (in-memory, **request_id-keyed only**) → event
store (**scoped to current `run_id`**) → `{state:"unknown", hint:"check git log
main"}`. `unknown` therefore means *"the record isn't in any cache still valid for
this run"* — **not** "the merge didn't happen." `main` is always authoritative;
the probe just can't always find the record. A successful merge reads as `unknown`
whenever the caller's key misses every still-valid tier:
- **id/key mismatch** — polling a coalesced/superseded `request_id` (which never
  gets its own terminal record), or polling by `branch`/`task_id` which **bypasses
  the request_id-keyed ring** (`server.py:1186`) and then misses the event-store
  row.
- **ring eviction** past `maxlen=200` (`merge_queue.py:2369`).
- **orchestrator restart** → in-memory ring lost; event store still has the row
  but is `run_id`-scoped, so a prior-run landing is dropped (`event_store.py:301`).

This was the reify 4352 trigger: the orchestrator had **not** restarted (3d10h
uptime) and the serial queue could not have pushed >200 merges in the window, so
the cause was an id/key mismatch, not the restart the skill text blames.

**Consumers (G1).** The `/merge-queue` and `/unblock` skills (the submit→poll
callers) and any operator/agent polling `merge_status`. User-observable surface:
`merge_status` returns a definitive `done` (instead of `unknown`) when the work is
provably on main, so the caller stops reaching for `land.sh`.

## Relationship to prior work (G4 — no contested ownership)

- Extends the `merge_status` contract built by **async-merge-request-prd.md**
  (task 1630, the four-tier lookup; task 1629, the `is_ancestor` already-merged
  fast-path on the *submit* side). Owner of the `merge_status` contract = this
  PRD's escalation-server work; no reciprocal ambiguity.
- **Task 1741 (done)** fixed a *different* sub-case: the **mid-flight** `unknown`
  under Lever-C K=2 (a finalizing local-lane head missing from `snapshot()`), by
  adding `_finalizing_head` to the worker snapshot. This PRD is the **post-finalize
  / lost-record** case and the key-mismatch case — disjoint from 1741.

## Approach (G5 — B, not B+H)

Incremental hardening of the **existing** `merge_status` tier contract; no new
cross-module seam. The authoritative oracle is `git main`, and the code already
trusts `git merge-base --is-ancestor` (used for `done_provenance`, and on the
`merge_request` submit-side coalesce gate at `server.py:769`). Per-task blast
radius ≤ 2 packages; the seam is pre-existing → approach **B**.

## Pre-conditions for activating

None — all substrate exists today (G3 verified, see capability manifest):
- `GitOps.is_ancestor` (`git_ops.py:1240`), `GitOps.find_merge_marker`
  (`git_ops.py`, the deleted-branch companion — exactly the 4352 shape, where the
  branch/worktree was already gone), `resolve_branch_sha`, `get_main_sha`.
- A `git_ops` handle is reachable from the escalation server (already used at
  `server.py:769`).
- `done_provenance` already defines `kind="found_on_main"` with the
  `is-ancestor`-against-main semantics this PRD reuses.
- `merge_finalized` events already carry `request_id` + `branch` + `task_id`
  (`merge_queue.py:2851`), so re-keying the ring needs no new data at record time.

## Out of scope

- Changing the merge gate, verify scope, or CAS-advance path.
- The mid-flight K=2 snapshot case (already done — task 1741).
- Removing `land.sh` (it stays the blessed manual path for "orchestrator
  down/congested"; this PRD only stops `unknown` from *routing* to it).

## Decomposition plan

Recommended order: **α** is the core fix; **β** is complementary tier hardening
(serialized after α to avoid `server.py` `merge_status`-region lock contention);
**γ** is a deferred completeness backstop, pursued only if α+β prove insufficient.

- **α — server self-resolution of `unknown` via git, + skill alignment** (leaf).
  In `merge_status`'s Tier-4 branch (`server.py:1298`), before returning
  `unknown`: if a branch is resolvable (passed `branch`, or `task_id`→`task/<id>`),
  consult `git main` via the existing primitives — `is_ancestor(branch_tip, main)`
  for a live branch, **and** `find_merge_marker(branch)` for the deleted-branch
  case — and return `state="done"`, `kind="found_on_main"`, `merge_sha=<resolved>`
  when the work is on main. Stay fire-safe (git failure degrades to honest
  `unknown`, never raises). In the **same task**, update `skills/merge-queue/SKILL.md`
  and `skills/unblock/SKILL.md`: replace the heuristic `git log main -20`
  resolution of `unknown` with the deterministic `git merge-base --is-ancestor
  task/<id> main` rule (done → found_on_main; resubmit only if NOT on main **and**
  queue healthy), and state explicitly that **`land.sh` is only for "orchestrator
  down/congested", never a response to `unknown`**. (Item 1 is folded here: the
  doc must describe the behavior α makes true, and a standalone doc-only task is
  not TDD-plannable and would churn the orchestrator.)
  - **Signal:** an integration test lands a branch, then removes the branch ref
    and clears/evicts the ring + event-store record (the 4352 lost-record shape);
    `merge_status(task_id=T)` returns `state="done"`/`kind="found_on_main"`, NOT
    `unknown`. Skills no longer route `unknown`→`land.sh`.

- **β — widen retention-ring keys + alias coalesced ids** (leaf; depends on α).
  Index `TerminalOutcomeRetention` (`merge_queue.py:2358`) by `branch` and
  `task_id` in addition to `request_id`, and record coalesced/superseded
  `request_id`s as aliases pointing at the primary terminal record. Update
  `merge_status` Tier-2 (`server.py:1186`) so `branch=`/`task_id=` polls consult
  the ring instead of skipping straight to the event store.
  - **Signal:** `merge_status(branch=B)` and `merge_status(task_id=T)` return the
    terminal outcome from the ring (not fall-through), and polling a
    coalesced/superseded `request_id` resolves to the primary outcome instead of
    `unknown` — proven by unit + tool-level tests.

- **γ — run-spanning "did it land" resolution** (leaf; depends on α, β) — **filed
  DEFERRED**, gated: pursue only if α+β leave a residual `unknown` after an
  orchestrator restart. Answer "is this branch on main now?" across runs — either
  a small durable `branch → merge_sha` map not scoped to `run_id`, or a
  git-reachability fallback — while **preserving** `run_id` scoping for *live*
  queries (the scoping exists deliberately to avoid surfacing stale prior-run
  outcomes; `event_store.py:301`).
  - **Signal:** after an orchestrator restart (new `run_id`),
    `merge_status(task_id=T)` for a branch landed in a *prior* run returns `done`,
    not `unknown`.

## Open questions (tactical)

- α: prefer `is_ancestor(branch_tip, main)` first and fall back to
  `find_merge_marker` only when the branch ref is gone (cheaper common path) —
  architect's call.
- β: alias storage (second dict vs. composite index) — local, recoverable.
- γ: durable map vs. pure git-reachability — decide at activation, informed by
  whether α+β actually leave a gap.
