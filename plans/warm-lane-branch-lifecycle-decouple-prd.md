# PRD: Full-decouple of the warm-lane branch-ref lifecycle from the lane-cache lifecycle

**Status:** active — ready to decompose
**Date:** 2026-06-25
**Type:** bug-fix / infrastructure hardening (warm-lane CoW pool lifecycle, dark-factory orchestrator)
**Approach:** B + H (contract + two-way boundary tests) — blast radius ≥ 3 (lane acquire/release in `git_ops.py` + cancel/recovery wiring in `harness.py`/`workflow.py` + the path to `main` via the merge worker); load-bearing seams on cancel/recovery/merge. **G5 = YES.**

---

## 1. Goal

Make the warm lane's reusable build **cache** lifecycle and the task's **`task/<id>` branch-ref** lifecycle fully independent, so that returning a lane to the pool (or cancelling/recovering a task, or re-acquiring a lane) can **never** delete or reset a branch that still carries unmerged work.

**User-observable surface (what changes for the orchestrator operator / every mid-merge task):**

- A task whose work is committed but **not yet on main** (e.g. parked in the pre-merge-rebase retry loop) keeps its `task/<id>` branch **across orchestrator restarts, hard-cancels, and lane churn**. The merge worker still finds the ref → the merge completes. No more `unknown_branch` / "branch `task/<id>` not found in repo" for live unmerged work.
- A lane left in DETACHED-HEAD by a prior release, or whose branch was orphaned (lane churned away), is **re-attached** to its existing `task/<id>` on the next dispatch (commits preserved) instead of colliding on `git worktree add` ("failed to create worktree: preparing worktree (new branch)") or being reset by `checkout -f -B`.
- The **only** condition under which a `task/<id>` ref is deleted becomes "the branch is on main" (no commits beyond main). Cache release, hard-cancel, crash-recovery, and lane re-acquire never delete an unmerged branch.

This eliminates the loss-WINDOW behind three L2 escalations (esc-3984-112, esc-3984-115, esc-4760-104). No work was ultimately lost in those incidents (each survived as the lane's detached-HEAD commit and was steward-recovered or re-landed), but the window is real and recurs under restart churn.

## 2. Background

The warm-lane CoW pool gives every concurrent build a warm `target/` by CoW-cloning a rolling base into fixed-path lanes (reify ships the seed/gc/refresh **primitives**; dark-factory ζ wires the **consumers** — the seam documented in reify `CLAUDE.md` "Warm-lane CoW pool (Phase 6, task ε #4663)"). The lifecycle code lives entirely in **dark-factory** orchestrator: `git_ops.py`, `harness.py`, `workflow.py`, `merge_queue.py`, `warm_lane_pool.py`. **reify's primitives are correct and unchanged by this PRD.**

**Single root cause (journal-confirmed 2026-06-25):** the warm lane's reusable `target/` cache and the task's `task/<id>` ref are bound to the **same release/acquire cycle**. `release_warm_lane` (`git_ops.py:2009`), whose legitimate job is to return the warm `target/` cache to the pool, also unconditionally runs `git checkout --detach` (HEAD → work tip, lane on no branch) then `git branch -D task/<id>` (`git_ops.py:2048`, **UNGUARDED**).

Two faults make this destructive:

- **Fault A — no commits-beyond-main guard on the delete.** The COLD path already guards: `_cleanup_leftover_branch` (`git_ops.py:2290`) *refuses* to delete a branch carrying commits beyond main and raises instead; reify's `warm-lane-gc.sh` honors the same `inv.preserve`. `release_warm_lane` honors NEITHER. The cold `cleanup_worktree` (`git_ops.py:4484`) shares the unguarded `branch -D`. The helper to reuse already exists: `_branch_has_commits_beyond_main` (`git_ops.py:2272`, fail-safe `True`).
- **Fault B — release fires for NON-terminal tasks.** A hard-cancel on orchestrator restart returns a **synthetic `TaskReport(outcome=CANCELLED)`** (`harness.py:3896`). Both B1 (`workflow.py:2002`, `run()` finally) and B2 (`harness.py:3929`, `_run_slot` finally) treat CANCELLED as terminal → `release_lane_for_terminal_task` → `cleanup_worktree` → `release_warm_lane` → `branch -D`. A hard-cancel is **process teardown**, not "work finished and discardable." Crash-recovery sweep cleanup (`harness.py:1556–1896`, many `cleanup_worktree` call sites) and the terminal-lane reconciler do the same when status momentarily reads done/cancelled.

**Amplifier — the MERGE phase.** A task in the pre-merge-rebase retry loop (`workflow.py:2080`, `max_pre_merge_retries`) stays *in-progress for tens of minutes to hours* under a hot merge train, spanning one or more restarts. Every restart/cancel in that window deletes its branch; the merge worker then can't find the ref → `unknown_branch` (`merge_queue.py:2372`). Both incidents were mid-merge tasks (direct proof for 4760: `release_warm_lane: released _lane-2 (branch task/4760)` immediately precedes the merge worker reporting the branch missing).

**The detached-HEAD symptom (esc-3984-115)** is the post-release lane state: `checkout --detach` left HEAD at the work tip; the `branch -D` either succeeded (4760 shape) or failed because the branch was momentarily checked out / interrupted (3984 shape — branch survives, lane orphaned-detached). Either way the lane is no longer ON the task branch, so re-dispatch via create-once `git worktree add -b task/<id>` collides.

PSI is clean (operator-confirmed CPU full=0%) — this is a lifecycle/logic bug, not resource pressure. Restart churn during the active warm-lane rollout merely *exposes* it.

## 3. Sketch of approach — FULL DECOUPLE

Split the two lifecycles. The lane **cache** (the `target/` dir + the pool ASSIGNED/FREE slot) is released, re-seeded, and reset freely on every acquire/release cycle — unchanged. The **`task/<id>` ref** is treated as load-bearing WIP and is touched only under a proven-safe condition. Concretely, four changes (one contract):

1. **Guard every branch-delete in `git_ops.py`'s release/cleanup paths** with `_branch_has_commits_beyond_main`. `release_warm_lane` keeps `checkout --detach` + `pool.release` (cache returns to pool) but deletes `task/<id>` **only when `not await self._branch_has_commits_beyond_main(full_branch)`** (fail-safe `True` → retain). Same guard on cold `cleanup_worktree`'s `branch -D`. **This alone makes "branch is on main (no commits beyond main)" the SOLE deletion condition on every release path** — realizing the "single authoritative deletion site" intent: the existing done-AND-on-main gate (`_maybe_cleanup_done_worktree`, `workflow.py:7438`, already gated on `is_ancestor(branch_head, main)`) is the canonical site that *produces* the deletable condition; the guard ensures no other path deletes before it.
2. **Hard-cancel never deletes branches.** Tag the synthetic `TaskReport(outcome=CANCELLED)` (`harness.py:3896`) so B1 (`workflow.py:2002`) and B2 (`harness.py:3929`) do **not** trigger the branch-deleting release for it. Defense-in-depth with (1): even if a release fires, the guard retains the branch. The lane **cache** is still reclaimed (by the periodic terminal-lane reconciler / next acquire), so no lane leaks.
3. **Branch-aware acquire / re-entry.** Before `_reset_warm_lane`'s `git checkout -f -B task/<id> <start_ref>` (which RESETS) or the create-once `git worktree add -b task/<id>` (which COLLIDES), detect an existing `task/<id>` carrying commits beyond main and **RE-ATTACH** (`git checkout -f task/<id>` then rebase onto main, routing through `_reuse_warm_lane` — commits preserved). Give the create-once path the same leftover-branch guard the cold path has (`_cleanup_leftover_branch`). This permanently un-wedges the 3984-class orphaned-branch state.
4. **Integration gate (two-way boundary leaf).** An e2e test: a task with committed unmerged work, lane released + orchestrator restarted mid-pre-merge-rebase → `task/<id>` SURVIVES → merge completes (no `unknown_branch`, no detached-orphan wedge).

## 4. Resolved design decisions (the new release/acquire/cleanup contract)

**The decoupling invariant (new — call it inv.10):** the `task/<id>` ref lifecycle is INDEPENDENT of the lane-cache lifecycle. A branch carrying commits beyond main is NEVER deleted or reset by release / hard-cancel / crash-recovery / re-acquire. The **sole** deletion condition anywhere is "the branch is reachable from main" (`_branch_has_commits_beyond_main == False`, fail-safe retain). The sole authoritative deletion site that produces that condition is the done-AND-on-main gate.

1. **`release_warm_lane` (`git_ops.py:2009`) = cache-release-only by default.** Keeps `git checkout --detach` + `await self.warm_lane_pool.release(lane)`. Deletes `task/<id>` IFF `not await self._branch_has_commits_beyond_main(full_branch)`. Stays best-effort / never-raise. *(Decision: cache release and branch delete are orthogonal operations.)*
2. **Cold `cleanup_worktree` (`git_ops.py:4484`) gets the same guard** on its `branch -D` — the harness reconcile / crash-recovery sweep route through it, so guarding here covers every sweep call site (`harness.py:1556–1896`) without touching each individually.
3. **Synthetic CANCELLED is tagged at construction** (`harness.py:3896`) — e.g. a `synthetic_cancel` flag on `TaskReport` (or reuse the existing hard-cancel signal). B1/B2 read the tag and skip `release_lane_for_terminal_task` for it. The lane cache is reclaimed by the existing periodic terminal-lane reconciler (A-path) / next acquire, both of which route through the now-guarded `release_warm_lane` — so the branch is doubly safe and the cache never leaks permanently.
4. **Branch-aware acquire detect-and-reattach.** In both the `_reset_warm_lane` reset-in-place path and the create-once `git worktree add` path, BEFORE the reset/add: if `task/<id>` exists AND `await self._branch_has_commits_beyond_main(task/<id>)` is `True`, re-attach (`git checkout -f task/<id>`, no `-B`; or `git worktree add <lane> task/<id>`, no `-b`) and route to `_reuse_warm_lane` (commit-WIP → rebase-onto-main → recompute base → re-provision debug port). Otherwise the existing fresh-reset / create-once path is byte-identical. The create-once path additionally gains the cold path's `_cleanup_leftover_branch`-style guard (raise rather than destroy if the leftover carries work and cannot be safely re-attached).
5. **`_reset_warm_lane`'s `checkout -f -B` is NEVER reached for a branch with commits beyond main.** The reattach check in (4) routes such a branch to `_reuse_warm_lane` first; only a branch that is on main (or does not exist) reaches `checkout -f -B`, which is then non-destructive.
6. **Fail-safe direction is RETAIN.** Every guard uses `_branch_has_commits_beyond_main` (returns `True` on any git error / unparseable output). On uncertainty the branch is RETAINED, never deleted/reset.

## 5. Out of scope (explicit, with rationale)

- **The merge-speculation `_spec-*` pool is correct as-is and is explicitly scoped OUT.** `_spec-*` lanes are checked out **detached at a merge commit** (`git worktree add --detach <lane> <merge_commit>`) and **never own a `task/<id>` ref**; `release_spec_lane` (`git_ops.py:1584`) only flips the pool ASSIGNED→FREE (warm) or removes the throwaway worktree (cold) — it runs **no `branch -D` at all**. So the spec pool cannot delete a task branch and needs no change. *(This is the G5 "spec pool — same treatment or scoped-out" answer: scoped out because it is structurally immune.)*
- **The ephemeral solo/train merge-artifact branch deletes** (`git_ops.py:2822/2862/2912`, `solo_name`/`solo_branch`) are out of scope — they delete short-lived `_solo-`/merge-train artifact refs, never a task's own `task/<id>` work ref.
- **No reify code change.** reify's seed/gc/refresh primitives are correct (verified). Do NOT file reify tasks editing dark-factory paths (known cross-repo trap). *(Follow-up, non-blocking: reflect inv.10 in reify `CLAUDE.md` "Warm-lane CoW pool" + reify warm-lane PRD §9.5 as a separate reify docs task — not part of this dark-factory batch.)*
- **No new pool, no base-refresh change, no D8/D9/D10 lifecycle redesign** — this is a targeted lifecycle-decouple of the branch ref only.

## 6. Cross-PRD / cross-repo seam ownership (G4)

| Seam | Owner | Notes |
|---|---|---|
| Warm-lane lifecycle code (`release/acquire/reset/reuse_warm_lane`, `cleanup_worktree`, B1/B2, crash-recovery sweep) | **dark-factory** (this PRD) | All four tasks land here. |
| Seed/gc/refresh primitives (`seed-warm-lane.sh`, `warm-lane-gc.sh`, `refresh-warm-base.sh`) | reify (unchanged) | Correct as-is; honors `inv.preserve` already. |
| Deploy (orchestrator restart to pick up the code change) | this session, post-land | reify `scripts/orchestrator-redeploy-restart.sh`; commit/land FIRST. |
| inv.10 doc reflection in reify `CLAUDE.md`/PRD | reify (separate follow-up task) | Non-blocking; not in this batch. |

## 7. Decomposition plan (one bullet per task → observable signal)

All tasks filed against `project_root=/home/leo/src/dark-factory`, `task_kind=normal` (TDD: RED test → GREEN impl within the task).

- **α — branch-retention guard in `release_warm_lane` + cold `cleanup_worktree` (Fix 1 + realizes Fix 2).** Files: `git_ops.py`, `test_git_ops.py`. **Signal:** RED-first test in `test_git_ops.py` — `release_warm_lane` on a lane whose `task/<id>` has commits beyond main RETAINS the ref (`git rev-parse --verify task/<id>` succeeds; pool still flips FREE; cache returns); on a branch with NO commits beyond main the ref is still deleted. Same for cold `cleanup_worktree`. Deps: none.
- **β — tag synthetic CANCELLED so hard-cancel never deletes branches (Fix 3).** Files: `harness.py`, `workflow.py`, `test_crash_recovery.py`. **Signal:** RED-first test — a mid-merge task hard-cancelled (synthetic `TaskReport(outcome=CANCELLED)`) RETAINS its `task/<id>` ref (B1/B2 skip the branch-deleting release for the tagged report); the lane cache is still reclaimable. Deps: none (independent files; parallel with α).
- **γ — branch-aware acquire / re-entry: detect-and-reattach (Fix 4).** Files: `git_ops.py`, `test_git_ops.py`, `test_warm_lane_pool.py`. **Signal:** RED-first test — acquiring a FREE lane when `task/<id>` already exists orphaned with commits beyond main RE-ATTACHES (`rev-list main..task/<id>` count preserved, lane HEAD == `task/<id>` tip), never resets to `start_ref`; create-once with an existing leftover branch re-attaches instead of colliding ("preparing worktree (new branch)"). Deps: α (same file; builds on the retention contract).
- **ω — integration gate (G2 leaf, two-way boundary).** Files: `test_warm_lane_integration_gate.py` (extend `test_crash_recovery.py` if needed). **Signal:** e2e test — task with committed unmerged work + lane released + simulated orchestrator restart mid-pre-merge-rebase → `git rev-parse --verify task/<id>` succeeds AND the merge worker resolves the branch (NO `unknown_branch`, NO detached-orphan wedge) AND the merge completes. Deps: α, β, γ.

## 8. Test plan (RED-first, dark-factory `orchestrator/tests/`)

- (a) `release_warm_lane` / cold `cleanup_worktree` with commits-beyond-main RETAINS the branch (α). On main → still deletes.
- (b) hard-cancel of a mid-merge task RETAINS the branch (β).
- (c) acquire onto an orphaned `task/<id>` RE-ATTACHES (commits preserved), never resets; create-once re-attaches instead of colliding (γ).
- (d) e2e: simulated restart mid-pre-merge-rebase → branch survives → merge worker finds it → merge completes (ω).

## 9. Open questions (tactical, implementation-time)

- Exact tag mechanism for the synthetic CANCELLED (new `TaskReport.synthetic_cancel` field vs reuse of the existing hard-cancel registry signal) — implementer's call; either satisfies the contract (β).
- Whether the reattach in γ uses `git checkout -f task/<id>` + `_reuse_warm_lane` or a thinner direct re-point — implementer's call provided commits are preserved and the lane ends ON `task/<id>`.
- Whether ω lives in `test_warm_lane_integration_gate.py` or extends `test_crash_recovery.py` — wherever the existing restart-simulation harness is cleanest.

## 10. Gate answers (folded from the design brief)

- **G1 (named consumer):** the dark-factory orchestrator task-dispatch + merge train; every mid-merge task during a restart; direct consumers of `release_warm_lane`/`acquire_warm_lane`/the crash-recovery sweep.
- **G2 (user-observable leaf signal):** the ω e2e — unmerged `task/<id>` survives lane-release + restart-mid-merge and the merge completes; no `unknown_branch`, no detached-orphan wedge.
- **G3 (assumed substrate):** warm-lane pool, XFS-reflink CoW base, reify seed/gc primitives all exist + correct (verified). `_branch_has_commits_beyond_main` (`git_ops.py:2272`) and `_cleanup_leftover_branch` (`git_ops.py:2290`) exist and are the guard model to mirror — confirmed by read.
- **G4 (cross-PRD/cross-repo seam):** fix lands 100% in dark-factory; reify primitives unchanged. See §6.
- **G5 (design-first, high stakes):** YES — cancel/recovery/merge wiring + path to main. New contract specified up front (§4). Spec pool explicitly scoped out with rationale (§5).
- **G6 (premise valid):** YES — root cause journal-confirmed + live repro (orchestrator thrashing `_lane-2` 11:12–11:26 on 2026-06-25). No code ultimately lost, but the loss WINDOW is real under churn; the decouple removes it.
