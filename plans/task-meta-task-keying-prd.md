# PRD: `.task-meta` task-keying — task-lifetime artifacts stop dying on lane reassignment

**Status:** active · 2026-07-28 · extension of `plans/worktree-lane-lifecycle-prd.md` (W11)
**Priority:** high — the standing cost is a full architect re-plan per occurrence.

## Goal

A task that is re-dispatched into a **different** warm lane keeps its plan,
its iteration history, and its task-lifetime review counters. Today it keeps
only its `task/<id>` branch: the commits survive, everything else is left
behind in the previous lane's `.task-meta/<lane>/` and is destroyed when that
lane is recycled.

User-observable outcome: on a cross-lane re-dispatch the orchestrator logs
`Task N: adopted task-scoped artifacts from <lane>` and then
`Task N: revalidating existing plan` — instead of dispatching the architect to
re-derive a plan it already had, and instead of silently handing the task a
fresh `max_amendment_rounds` allowance.

## Background / evidence

### Defect 1 — the plan does not survive a lane change (root cause, confirmed)

`TaskArtifacts.meta_root_for(worktree_base, worktree_name)`
(`artifacts.py:263`) keys the durable artifacts root by **worktree name** —
for a pooled lane, that is the lane. `WarmLanePool.acquire_for`
(`warm_lane_pool.py:279-296`) allocates **the first FREE lane**, with no task
affinity of any kind; the `_assignments` map exists only while the lane is
held. `task/<id>` branches are repo-global. Those are two different lifetimes,
so a task that releases its lane and is later re-dispatched generally lands
elsewhere and arrives with its branch but without its artifacts.

Traced end-to-end for reify task 5069 (per-worktree HEAD reflogs +
`journalctl -u orchestrator-reify`):

- 5069 ran in `_lane-20` on 07-19, 07-23 (×3) and 07-26; its plan was written
  to `.task-meta/_lane-20/plan.json`.
- 2026-07-28 09:59:16Z it was re-dispatched into `_lane-36`
  (`route=reset_in_place_reattach`). `_lane-36` correctly got
  `_clear_foreign_meta_root` (it held 5587's metadata), then `init()` +
  `ensure_lane_plan_symlink()` produced a **dangling** `.task/plan.json`.
- The old plan was still on disk at that moment — `_lane-20` was not recycled
  until 10:56:30Z, ~57 minutes later. **Nothing looked for it.**
- The architect then spent **$2.79 / 48 turns / 17 minutes** reconstructing an
  8-step plan from `git log`, filed `esc-5069-4`, was auto-promoted to
  `esc-5069-5`, and consumed human triage. The reconstructed plan's 8 steps
  mapped 1:1 onto the branch's 10 commits — the work had been complete the
  whole time.

**Prevalence** (reify, 2026-07-28 alone, from acquire-route log lines):
`recycle` 54 · `reuse` 13 · **`reset_in_place_reattach` 12** ·
`disk_backstop_reuse` 1. Every `reset_in_place_reattach` is by construction
"the branch carries commits and this lane does not hold the plan" — ~15% of
acquisitions, each paying a re-plan.

**This is not a regression introduced by task 2763.** Task artifacts were
lane-keyed before it — first inside `<worktree>/.task/`, then at
`.task-meta/<lane>/` after W11. 2763's symlink only changed the presentation
from "file silently absent" to "dangling symlink", which is strictly better
and is retained here. W11's own resolved decision 1 already *describes*
`.task-meta` as "task-keyed execution artifacts"; this PRD closes the gap
between that stated intent and the realized keying.

### Defect 2 — task-lifetime review budgets silently reset (same root cause)

`review_state.json` is documented in `artifacts.py:527-537` as holding "the
task-lifetime amendment/review counters, so `max_amendment_rounds` /
`max_review_cycles` bound the WHOLE task lifetime, not each dispatch", and
`workflow.py:5455-5459` (task 2749) restates the intent: "a re-dispatch
(restart churn, requeue, resume) no longer grants a fresh allowance." Because
the file lives in the lane-keyed root, **a cross-lane re-dispatch grants
exactly the fresh allowance 2749 set out to prevent**, and drops the
tree-hash-keyed verdict cache so an unchanged committed tree is re-reviewed
and re-mints reviewer nits. Same fix, no extra mechanism.

## Substrate reality (G3 — code-verified 2026-07-28, HEAD `7658f909fc`)

| Assumed capability | Evidence | Verdict |
|---|---|---|
| `TaskArtifacts` is the single path-derivation owner | `artifacts.py:262-287` is the only site joining `TASK_META_DIRNAME` to a per-worktree name; the other two references (`git_ops.py:511` PROTECTED owner tag, `harness.py:2840` reaper skip) never build a per-task path | exists |
| Lane→task identity is recoverable without `plan.json` | `.task-meta/<lane>/metadata.json` carries `task_id`/`title` (written by `init()`, `artifacts.py:305-313`); `.lane-state/<lane>.json` carries `task_id`/`branch` (`warm_lane_pool.py:_note_assigned_durable`) | exists |
| Sandbox can be granted a second writable root | `agents/write_set.py:152` — one call site, `task_meta=` field | exists |
| Plan-tools MCP targets an arbitrary root | `mcp_lifecycle.py:142-167` — `--meta-root` passthrough | exists |
| Per-round review staleness has precedent | `workflow.py:5449-5453` clears a stale `reviews/merge.json` each loop | exists |
| Terminal-task status is queryable for a sweep | `scheduler.get_statuses()` (compact `{id: status}`) | exists |

No novel substrate. Every mechanism below is a re-keying or re-wiring of a
capability that exists on main today.

## Consumers (G1)

| Mechanism | Consumer |
|---|---|
| Task-keyed artifact root | `TaskWorkflow._plan()` / `_execute_iterations()` / `_execute_verify_review_loop()` (`workflow.py`) — reads the plan, iteration history and review counters it previously lost |
| Adoption at dispatch | `TaskWorkflow._setup_worktree_and_artifacts()` (`workflow.py:2229-2246`) |
| Plan-carried base stamp | `TaskWorkflow._plan()`'s revalidation branch (`workflow.py:3733-3772`) and Layer-B iteration hygiene (`workflow.py:2248-2259`) |
| Lane↔task resolution off `metadata.json` | `GitOps.acquire_warm_lane` disk-backstop (`git_ops.py:5440-5482`) and `release_lane_for_terminal_task` (`git_ops.py:6307`) |
| Symlinked lane surface | the `implementer` / `fixer` / reviewer role prompts (`agents/roles.py`, `agents/briefing.py`) |
| GC sweep | operator, via the digest line + the streak escalation |

No producer here lacks a named in-repo consumer. Not a cross-project PRD; no
reify-side change is required or permitted by it.

## Sketch of approach

### M1 — two key spaces under one store

`.task-meta/` gains a second key space; the existing one is untouched.

- **lane-keyed** `.task-meta/<worktree-name>/` — the *dispatch/occupancy*
  record: `metadata.json`, `agent_session.json`, `interactive.json`, and the
  architect structured exits (`blocking_dependency.json`, `already_done.json`,
  `false_premise.json`, `unactionable_task.json`).
- **task-keyed** `.task-meta/task-<id>/` — the *task-lifetime* record:
  `plan.json`, `plan.lock`, `iterations.jsonl`, `reviews/`, `verdicts/`,
  `review_state.json`, `reconcile_state.json`.

`TaskArtifacts` owns both derivations (`meta_root_for` unchanged;
`task_root_for(worktree_base, task_id)` added) and routes each artifact to its
root through one table, so no caller picks a root by hand.

Cold (non-pool) worktrees are named `<task_id>`, so they get
`.task-meta/<id>/` **and** `.task-meta/task-<id>/`. Slightly redundant, but the
same code path serves both modes with no special-casing — and dark-factory
itself runs cold, so the split is exercised by this repo's own suite.

The architect structured exits deliberately stay lane-keyed: they are
per-dispatch signals, and `_clear_foreign_meta_root`'s contract
(`git_ops.py:5840-5858`) exists precisely to stop an incoming task inheriting a
stale one. `agent_session.json` stays lane-keyed for the same reason plus a
stronger one — it carries an `owner_pid` and a session bound to a specific
worktree, so travelling it would invite a resume against a lane the task no
longer occupies (regressing tasks 2771-2777).

### M2 — the plan carries its own base

Moving `plan.json` alone is **not sufficient**. `_plan()` only takes the cheap
revalidation branch when `self._old_plan_base` is truthy
(`workflow.py:3733-3737`), and that comes from `metadata.json`'s `base_commit`
— lane-keyed, cleared on a lane change. A travelled plan would therefore still
fall through to `build_architect_prompt` and burn a full architect run, which
is the entire cost this PRD exists to remove.

Fix: stamp `_base_commit` into `plan.json` beside the existing `_session_id` /
`_finalized_at` / `_revalidated_at` stamps, and have `_plan()` prefer
`existing_plan['_base_commit']`, falling back to `_old_plan_base`. The base
then travels with the artifact it describes rather than living in a sibling
file that does not travel (INV-5: one site, not two that must agree).

`_apply_revalidation_skip` and `update_base_commit` must keep the two coherent,
and Layer-B iteration hygiene (`workflow.py:2248-2259`), whose discriminator is
`_old_plan_base != base_commit`, must retain its current semantics: wipe
`iterations.jsonl` on a definitive re-dispatch onto a **new fork point**, and
only then.

### M3 — self-healing adoption at dispatch

At `_setup_worktree_and_artifacts`, if `.task-meta/task-<id>/` is absent, find
the lane-keyed root whose `plan.json` carries this `task_id` and **move** its
task-scoped members into the task root, logging
`Task N: adopted task-scoped artifacts from <lane>`.

Corroborated before acting (INV-3): the source lane is only harvested when
`.lane-state/<source>.json` does **not** show it currently ASSIGNED to a
different task. A no-match is a silent no-op — first dispatch is the common
case.

This is migration and recovery in one mechanism: it needs no flag day, and it
would have saved task 5069 (its plan was still on disk in `_lane-20`, 57
minutes from destruction).

### M4 — per-round review staleness

With `reviews/` travelling, a reviewer that errors out in dispatch N could
leave a file that dispatch N+1 aggregates as current. Generalize the existing
single-file precedent (`workflow.py:5449-5453` clears a stale
`reviews/merge.json`) to clear the whole `reviews/` directory at the start of
each review round. `verdicts` need no new rule — `review_state.json`'s cache is
already keyed by committed-tree hash (`artifacts.py:599`), so a verdict can
only be reused against the exact tree that minted it.

### M5 — bounded lifetime

A periodic sweep removes `.task-meta/task-<id>/` for tasks that are terminal
(`done`/`cancelled`) **and** older than 14 days, logging a reclaimed count each
pass. Per INV-4 the sweep carries a consecutive-failure streak counter that
escalates rather than degrading silently. 14 days is deliberately longer than
any observed requeue gap (5069's was 2 days), so the sweep can never race a
live task's return.

### M6 — one path for agents

`ensure_lane_plan_symlink` retargets to the task root and gains a sibling
symlink for `iterations.jsonl`, so the role prompts can name `.task/plan.json`
and `.task/iterations.jsonl` and stop instructing agents to compute
`<worktree_base>/.task-meta/<worktree-name>/…` from their cwd. This directly
retires the confusion cost task 2763 was filed against (1–8.5 min per session
re-discovering the meta root, worst case a 163s `find` sweep).

## Resolved design decisions

1. **Split by lifetime, not wholesale re-key.** Every "which task owns this
   lane?" reader (`worktree_identity.read_worktree_title`, the disk-backstop,
   `_clear_foreign_meta_root`) needs a *lane*-keyed answer. Re-keying the whole
   store by task — the literal wording of task 3107 — would force a redesign of
   the lane identity model for no gain, and `_iact-*` worktrees have no task id
   at all so they would need a second scheme regardless.
2. **Lane↔task resolution moves off `plan.json` and onto `metadata.json`.**
   Once `plan.json` leaves the lane root, `_find_lane_by_plan_task_id`
   (`git_ops.py:6386`) and the disk-backstop reuse check (`git_ops.py:5442`)
   have nothing to read. `metadata.json` already carries `task_id` in the same
   directory and is written every dispatch. This is independently valid today
   and lands first, so no window exists in which the pool cannot resolve a lane.
3. **The plan carries its own base commit** (M2) — without it the fix moves
   bytes and saves no tokens.
4. **The architect structured exits and `agent_session.json` do not travel.**
   They are per-dispatch signals; travelling them creates the staleness class
   `_clear_foreign_meta_root` was written to prevent.
5. **`review_state.json` travels**, restoring task 2749's stated intent rather
   than extending it (§Background defect 2).
6. **Adoption is a move, not a copy**, and is corroborated against
   `.lane-state` before it fires. A copy would leave a second plan on disk for
   the same task and re-open the divergence trap 2763 closed.
7. **No compat window / new-then-old reader fallback.** W11 needed one because
   it moved artifacts between two locations both of which existing deployments
   held. Here M3's adoption *is* the migration and it is self-healing on first
   dispatch, so a reader fallback would be dead code from the first green cycle.
8. **2763's symlink property is preserved, not reverted.** The lane path
   remains a symlink into the durable copy, so the lane copy can still never
   diverge from the durable one.

## Pre-conditions for activating

None outstanding. Every substrate capability in the G3 table is on main at
`7658f909fc`. Dark-factory task **3107** ("Re-key .task-meta durable store by
task id…") is the recon-filed placeholder for this work and is currently
`deferred`; it is superseded by this PRD's batch and should be cancelled with a
pointer to it at decompose time rather than queued alongside.

## Cross-PRD relationship (G4)

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/worktree-lane-lifecycle-prd.md` (W11) | extends | `TaskArtifacts.meta_root_for` / `.task-meta` path-derivation contract | **this PRD** | this PRD amends W11's §`.task-meta` path-derivation contract; W11 is landed, so the amendment is recorded here and W11 gains a pointer |
| `plans/warm-lane-session-resume-prd.md` | preserves | `agent_session.json` sidecar | that PRD | no change — decision 4 keeps the sidecar lane-keyed precisely so its recovery path is untouched |
| `plans/warm-lane-hardening-prd.md` | adjacent | acquire routes / `_clear_foreign_meta_root` | that PRD | this PRD narrows what `_clear_foreign_meta_root` deletes; it does not change when it fires |

No reciprocal-ownership ambiguity: all three counterparties are landed PRDs,
and this one holds every integration task.

## Contract (B+H)

### Path-derivation contract (amends W11 §`.task-meta` path-derivation contract)

- `meta_root_for(worktree_base, worktree_name)` = `<base>/.task-meta/<name>` —
  **unchanged**, now explicitly the *lane/dispatch* root.
- `task_root_for(worktree_base, task_id)` = `<base>/.task-meta/task-<id>` —
  new, the *task-lifetime* root. `task_id` is validated against
  `^[A-Za-z0-9._-]+$` before use (same guard shape as `_validate_verdict_role`,
  `artifacts.py:191`) so it can never escape `.task-meta/`.
- `TaskArtifacts(worktree, meta_root=None, task_root=None)`. Both `None` =
  legacy single-root mode, byte-identical to today. A per-artifact routing
  table is the only thing that decides which root a filename resolves against;
  no caller joins a root by hand.
- Writes: the artifact's own root only. Reads: same. No new-then-old fallback
  (decision 7).

### Artifact routing table (normative)

| Artifact | Root | Rationale |
|---|---|---|
| `metadata.json` | lane | dispatch/occupancy identity; read by the lane↔task resolvers |
| `agent_session.json` | lane | `owner_pid` + worktree-bound session (tasks 2771-2777) |
| `interactive.json` | lane | `_iact-*` worktrees have no task id |
| `blocking_dependency.json` | lane | per-dispatch architect exit |
| `already_done.json` | lane | per-dispatch architect exit |
| `false_premise.json` | lane | per-dispatch architect exit |
| `unactionable_task.json` | lane | per-dispatch architect exit |
| `plan.json`, `plan.lock` | task | the expensive artifact; the whole point |
| `iterations.jsonl` | task | implementer's prior-iteration context |
| `reviews/`, `verdicts/` | task | review context across dispatches (M4 rule applies) |
| `review_state.json` | task | documented task-lifetime counters (defect 2) |
| `reconcile_state.json` | task | cross-restart dedup keys for this task's steps (2764) |

### Adoption invariants

1. Adoption fires only when `task_root_for(id)` is absent.
2. A source lane is eligible only when its `metadata.json` *or* `plan.json`
   names this `task_id` **and** `.lane-state/<source>.json` is not ASSIGNED to
   a different task.
3. Adoption is a move; on success the source's task-scoped members no longer
   exist. Partial failure leaves the source intact and logs at WARNING — never
   raises into the dispatch path.
4. Adoption emits exactly one INFO line naming the source lane.

## Boundary-test sketch (the ω integration gate's signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Cross-lane re-dispatch preserves the plan | task planned in lane A, lane A released, lane B is the first FREE lane | `read_plan()` in lane B returns the lane-A plan byte-identically; `<B>/.task/plan.json` resolves (not dangling) |
| B2 | Cross-lane re-dispatch does not re-plan | as B1, main unchanged w.r.t. the plan's `files` | `_plan()` takes the revalidation (or revalidation-skip) branch; **zero** `architect` invocations |
| B3 | Review budget is not refreshed | task consumed 2 of 3 amendment rounds in lane A | the loop in lane B seeds from 2, not 0 |
| B4 | Foreign lane meta is still cleared | lane B previously held task X with a `blocking_dependency.json` | the incoming task sees no `blocking_dependency.json`; X's plan in `task-<X>` is untouched |
| B5 | Adoption refuses a live source | source lane is ASSIGNED to a different live task in `.lane-state` | no move; incoming task plans fresh; source lane's artifacts intact |
| B6 | Session sidecar does not travel | task had an in-flight `agent_session.json` in lane A | lane B sees no sidecar; no resume is attempted against lane A's session |
| B7 | Stale reviewer file cannot cross a dispatch | reviewer R wrote `reviews/R.json` in dispatch N and errors in N+1 | N+1's aggregation contains no entry for R |
| B8 | Sweep never reaps a live task | task terminal 20d but subsequently re-opened to `pending` | `task-<id>/` survives the sweep |
| B9 | Cold mode unaffected | non-pool worktree named `<task_id>` | plan/iterations resolve; existing cold-path tests green |

## Decomposition plan

Labels are intra-batch; real ids are assigned at decompose time.

- **α — lane↔task resolution off `plan.json`, onto `metadata.json`.**
  Modules: `orchestrator/git_ops.py`. Rewrites the disk-backstop reuse check
  (`:5440-5482`) and `_find_lane_by_plan_task_id` (`:6386`) to read
  `metadata.json`'s `task_id`. Independently valid before anything moves.
  *Signal:* acquire-route log shows `route=disk_backstop_reuse` for a same-lane
  re-dispatch of a task whose architect never wrote a plan — a case that logs
  `reset_in_place_reattach` today.
  *Intermediate* — unlocks β.

- **β — `TaskArtifacts` dual-root + the routing table + sandbox grant.**
  Modules: `orchestrator/artifacts.py`, `orchestrator/workflow.py`,
  `orchestrator/agents/write_set.py`, plus the four small readers
  (`steward.py:727`, `task_runtime.py:223/255`,
  `stranded_verified_green.py:160`, `worktree_identity.py:83`). Adds
  `task_root_for` + validation, the per-artifact routing table, retargets
  `ensure_lane_plan_symlink` and adds the `iterations.jsonl` symlink, and
  grants the task root in `write_set.py:152`.
  *Signal:* after one dispatch, `.task-meta/task-<id>/plan.json` exists and
  `<lane>/.task/plan.json` resolves to it; `.task-meta/<lane>/` holds only the
  lane-keyed set. *Intermediate* — unlocks γ/δ/ε/ζ/η.

- **γ — self-healing adoption at dispatch (M3).** Modules:
  `orchestrator/workflow.py`, `orchestrator/artifacts.py`. Implements the
  §Adoption invariants including the `.lane-state` corroboration.
  *Signal:* operator log line `Task N: adopted task-scoped artifacts from
  <lane>` on the first post-deploy cross-lane re-dispatch. Depends β.

- **δ — plan-carried `_base_commit` (M2).** Modules:
  `orchestrator/artifacts.py`, `orchestrator/workflow.py`. Stamps
  `_base_commit` on write, prefers it in `_plan()`, keeps
  `_apply_revalidation_skip` / `update_base_commit` coherent, and preserves
  Layer-B's new-fork-point semantics.
  *Signal:* `Task N: revalidating existing plan (…)` on a cross-lane
  re-dispatch where today the log shows a fresh architect dispatch. Depends β.

- **ε — per-round review staleness (M4).** Modules:
  `orchestrator/workflow.py`. Generalizes the `reviews/merge.json` clear at
  `:5449-5453` to the whole directory.
  *Signal:* a reviewer that errors in round N+1 contributes no entry to that
  round's aggregation (previously it contributed round N's file). Depends β.

- **ζ — terminal-task GC sweep (M5).** Modules: `orchestrator/harness.py`,
  `orchestrator/artifacts.py`. Terminal + 14-day sweep, reclaimed-count log
  line, INV-4 streak counter escalating on repeated failure.
  *Signal:* operator-visible `task-meta gc: reclaimed N task roots` line and,
  on repeated failure, a born-at-L1 escalation. Depends β.

- **η — one path for agents (M6).** Modules:
  `orchestrator/agents/roles.py`, `orchestrator/agents/briefing.py`. Prompts
  name `.task/plan.json` and `.task/iterations.jsonl` only; the
  `<worktree_base>/.task-meta/<worktree-name>/…` arithmetic is deleted from
  every role and briefing site.
  *Signal:* `grep -c 'task-meta/<worktree-name>' agents/` is 0; an implementer
  session reads both artifacts from `.task/` without a `find` sweep. Depends β.

- **ω — integration gate.** Modules: `orchestrator/tests/`. The B1–B9 boundary
  table above, driven against a simulated two-lane pool (the fixture shape used
  by `tests/test_warm_lane_pool.py` / `tests/test_lane_lifecycle_gitops.py`).
  *Signal (leaf):* the B1–B9 suite green; in particular B2 asserting **zero**
  architect invocations on a cross-lane re-dispatch. Depends γ, δ, ε, η.

G7 walk (advisory at author time; re-walked at decompose): ζ is the only task
introducing a fail-soft path and it ships the INV-4 streak counter by
construction. γ acts on `.lane-state` snapshot state and carries the INV-3
corroboration explicitly. β and η both *remove* duplication (one routing table
replacing hand-joined roots; one prompt path replacing two spellings), so
INV-5 is satisfied rather than waived. No waivers anticipated.

## Out of scope

- **Re-keying `.lane-state`.** It is lane state and correctly lane-keyed.
- **Lane affinity in the pool.** Giving a returning task its previous lane
  would make the loss rarer without closing it, and would fight the pool's
  free-lane allocation. Explicitly rejected.
- **Committing the plan to the task branch.** Re-contaminates `.task/`; reify
  forbids it on main.
- **Storing the plan in the fused-memory task record.** ~28KB mutated per step
  from inside the sandbox; puts the network in the workflow hot path.
- **Migrating the Claude config dir / session JSONL** (`TaskConfigDir`,
  `workflow.py:2222`). It lives inside the worktree deliberately so the session
  travels with the worktree; a lane change *should* start a fresh session.
- **The `_iact-*` interactive stamp.** Not a task; stays lane-keyed.

## Open questions (tactical — surfaced, not blocking)

1. **β's file-lock breadth.** β touches `artifacts.py` + `workflow.py` +
   `write_set.py` + four small readers in one atomic change; under the narrow
   file-lock scheduler that is a wide lock and may wait. Splitting it risks a
   half-migrated main, so atomic is the right call — but if it starves, the
   implementer may land the four small readers first as a no-op prep commit.
   Decide during β.
2. **Sweep cadence and host.** Whether ζ piggybacks the existing warm-lane GC
   pass or takes its own timer. Either is coherent; decide during ζ.
3. **Whether δ should backfill `_base_commit` onto plans already on disk** at
   first read, or leave pre-deploy plans to fall back to `_old_plan_base`. The
   fallback is already correct, so backfill is an optimization. Decide during δ.
