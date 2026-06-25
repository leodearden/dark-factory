# Capability Manifest — warm-lane-branch-lifecycle-decouple-prd

Mechanizes G3 (substrate exists/wired) + G6 (premise valid) per leaf. One block per leaf signal; each asserted capability bound to evidence. Any **FAIL** binding blocks queueing.

**Domain flags:** no grammar/DSL → grammar-fixture checks **N/A**. The numeric assertions here are *count invariants* (commits-beyond-main `> 0` retained; `rev-list` count preserved across acquire) — covered by the **count-preservation** check, not an empirical numeric floor. Live checks: **capability→producer (wired)**, **guard-mechanism (rejection-direction)**, **DAG-direction**, **count-preservation**.

All `file:line` evidence verified by direct read of dark-factory `orchestrator/src/orchestrator/{git_ops.py,harness.py,workflow.py,merge_queue.py}` during the study that produced this PRD (2026-06-25).

---

## α — branch-retention guard in `release_warm_lane` + cold `cleanup_worktree`  *(leaf — the load-bearing retention guarantee)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `_branch_has_commits_beyond_main(full_branch)` exists, fail-safe `True` | capability→producer (exists) | `git_ops.py:2272` — `rev-list --count main..<branch>`; returns `True` on rc≠0 / `ValueError` | PASS |
| `release_warm_lane` currently runs UNGUARDED `branch -D` | guard-mechanism (the corner α fixes) | `git_ops.py:2048` inside `release_warm_lane` (def `2009`) — bare `['git','branch','-D',full_branch]` | PASS (corner exists to guard) |
| Cold `cleanup_worktree` shares the UNGUARDED `branch -D` | guard-mechanism | `git_ops.py:4484` inside `cleanup_worktree` (def `4446`) | PASS |
| α's RED test observes RETENTION (branch with commits-beyond-main survives release; pool still flips FREE) | count-preservation | **producer = α** — RED test authors a lane on `task/<id>` with `n>0` commits beyond main, calls `release_warm_lane`, asserts `rev-parse --verify task/<id>` succeeds AND `warm_lane_pool.release` ran (slot FREE). Retention is α's deliverable, not assumed substrate | PASS (built+bound by α) |
| α's RED test observes DELETION still happens when on-main (no commits beyond main) | guard-mechanism (negative side) | producer = α — same test, branch with `0` commits beyond main → ref deleted. Two-sided: retains-when-unmerged AND deletes-when-merged | PASS |

## β — tag synthetic CANCELLED so hard-cancel never deletes branches  *(leaf — defense-in-depth on Fault B)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Synthetic `TaskReport(outcome=CANCELLED)` is constructed at a single hard-cancel site | capability→producer (exists) | `harness.py:3896` — `report = TaskReport(... outcome=WorkflowOutcome.CANCELLED)` in the `except asyncio.CancelledError` of `_run_slot` | PASS |
| B1 release path reads terminal outcome to fire release | capability→producer (wired) | `workflow.py:2002` — `if self.state in (DONE, CANCELLED) and not _worktree_external: release_lane_for_terminal_task` | PASS (the path β gates) |
| B2 release path reads `report.outcome` to fire release | capability→producer (wired) | `harness.py:3929` — `if report is not None and report.outcome in (DONE, CANCELLED): release_lane_for_terminal_task` | PASS (the path β gates) |
| A `synthetic_cancel`-style tag can be added to `TaskReport` and read at B1/B2 | field-population | **producer = β** — adds the tag at the `3896` construction and the read at `2002`/`3929`; no schema migration (TaskReport is an in-memory dataclass) | PASS (built by β) |
| β's RED test observes a hard-cancelled mid-merge task RETAINS its branch | count-preservation | producer = β — RED test simulates the synthetic-cancel exit, asserts `task/<id>` still resolves; lane cache reclaimable | PASS |
| Lane cache is not permanently leaked when B1/B2 skip release | capability→producer (wired) | **exists** — the periodic terminal-lane reconciler (A-path) + next `acquire` both route through the now-α-guarded `release_warm_lane` (`release_lane_for_terminal_task`, `git_ops.py:2063`) | PASS |

## γ — branch-aware acquire / re-entry: detect-and-reattach  *(leaf — permanently un-wedges the 3984 state)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `_reset_warm_lane`'s `checkout -f -B` RESETS the branch (the corner γ guards) | guard-mechanism | `git_ops.py:1988` — `['git','checkout','-f','-B',full_branch,target_commit]` resets `task/<id>` to `target_commit` | PASS (corner exists) |
| Create-once `git worktree add -b` COLLIDES on an existing branch (the other corner) | guard-mechanism | `git_ops.py:1785` — `['git','worktree','add','-b',full_branch,...]`; collision = esc-3984-112 "preparing worktree (new branch)" | PASS (corner exists) |
| `_reuse_warm_lane` (reattach target) exists: commit-WIP → rebase-onto-main → recompute base → reprovision port | capability→producer (wired) | `git_ops.py:1920` (def) → `commit`, `rebase_onto_main` (`git_ops.py:2584`), `_provision_reify_debug_port` | PASS |
| Reattach preserves commits (`rev-list main..task/<id>` count unchanged; lane HEAD == branch tip) | count-preservation | **producer = γ** — RED test orphans a `task/<id>` with `n>0` commits, acquires a FREE lane, asserts count preserved + HEAD on `task/<id>` (NOT `start_ref`) | PASS (built+bound by γ) |
| Retention guard upstream of γ's reattach decision | DAG-direction | producer = α (`_branch_has_commits_beyond_main` guard), **upstream** (γ depends α) | PASS |
| Create-once gains the cold path's leftover-branch guard | capability→producer (wired) | model exists — `_cleanup_leftover_branch` (`git_ops.py:2290`) raises rather than destroys; γ mirrors its predicate into the create-once path | PASS |

## ω — integration gate: restart-mid-merge → branch survives → merge completes  *(leaf — G2 user-observable, two-way boundary)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Every retention/reattach/skip capability the e2e exercises | DAG-direction | all produced by α/β/γ, **upstream** of ω (ω depends α, β, γ) | PASS |
| Merge worker reports `unknown_branch` when a `task/<id>` ref is missing (the failure ω asserts is GONE) | capability→producer (exists) | `merge_queue.py:2372` — emits `unknown_branch`, `reason=f'branch {full_branch!r} not found in repo'` | PASS (the negative the test pins) |
| Restart-mid-pre-merge-rebase is simulable | capability→producer (wired) | **exists** — pre-merge-rebase retry loop `workflow.py:2080` (`max_pre_merge_retries`, `config.py:1243`); existing restart-simulation harness in `test_crash_recovery.py` / `test_warm_lane_integration_gate.py` | PASS |
| e2e asserts branch survival AND successful merge (both sides) | end-to-end premise | **producer = ω** — `rev-parse --verify task/<id>` succeeds after the simulated release+restart AND the merge resolves the ref + completes (no `unknown_branch`) | PASS (built by ω) |

---

## Scope-out bindings (verified immune / out of scope — no task)

| Asserted-immune surface | Check | Evidence | Verdict |
|---|---|---|---|
| `_spec-*` merge-speculation pool cannot delete a `task/<id>` ref | guard-mechanism (structural) | `release_spec_lane` (`git_ops.py:1584`) runs **no `branch -D`** — warm path flips pool FREE, cold path removes a throwaway `_merge-` wt; `_spec-*` lanes are `worktree add --detach <merge_commit>`, never own a `task/<id>` | PASS (immune — scoped out, G5) |
| Ephemeral solo/train merge-artifact deletes are not `task/<id>` | guard-mechanism | `git_ops.py:2822/2862/2912` delete `solo_name`/`solo_branch` (short-lived merge-train artifacts), never a task's own work ref | PASS (out of scope) |
| `_cleanup_leftover_branch` already guards | capability→producer (exists) | `git_ops.py:2290` — raises on commits-beyond-main / dirty / unverifiable (fail-safe) | PASS (already correct) |
