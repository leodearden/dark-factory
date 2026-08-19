# Rebase-before-verify gap — status as of 2026-06-07

Investigation of how far the dark-factory merge-queue "rebase-before-verify gap" has been
closed by the merge-queue/worker batch landed since 2026-06-04. Read-only; code-grounded.

## (a) Verdict — **substantially addressed** (the gap itself is closed; one narrow residual)

The specific behavior flagged unowned on 2026-06-04 — *"rebase the merge worktree onto CURRENT
main BEFORE running verify, so already-landed fixes are present"* — is now **implemented and
owned** on the production merge path. Task **1646** added **Mechanism 2 (freshness re-base at
verify-pickup)** to `SpeculativeMergeWorker`: when the verifier dequeues a real merge item whose
base is no longer current main, it discards the stale merge worktree and **re-merges the branch
against actual main before `_run_post_merge_verify` runs**
(`merge_queue.py:4440-4489`). This is the production worker (`harness.py:3238` builds
`SpeculativeMergeWorker`, never the deprecated serial `MergeWorker`). Combined with the
pre-existing #1595 disjoint-delta re-verify gate (which still fires when main moves *during*
verify) and the #1602 pre-advance unscoped type-check gate, the merge tree is now freshened onto
current main both at pickup and during the advance window. The StatusBar-class re-break (an
already-on-main fix re-surfacing as a verify failure in a stale merge tree) **can no longer
occur** on this path. The **one residual** is an interaction with the `skip_verify`
optimization: for `pre_rebased` requests, Mechanism 2 re-merges onto fresh main but then
**preserves `skip_verify=True`**, so the freshly re-merged tree is advanced *without* re-verifying
it — and because the pickup re-merge removes the rebase that `advance_main` would otherwise have
performed, the #1595 re-verify gate no longer fires for these items either. That window is narrow
and is the *inverse* of the StatusBar failure (it risks a genuine new break landing unverified,
not a false failure), but it is a real stale-base verify-integrity gap.

## (b) Tasks that addressed it

| Task | Title (abridged) | Merge commit | Landed | What it changed |
|---|---|---|---|---|
| **1646** | Bound stale-merge accumulation: cap merger-ahead + freshness re-base at verify-pickup | `9351e6e348` | 2026-06-04 20:46 | **The headline fix.** *Mechanism 1*: `_MERGE_AHEAD_BOUND=1` semaphore caps non-speculative, non-train build-ahead so the merger naturally re-reads fresh main when it resumes (impl `4f5fe6021f`). *Mechanism 2*: re-base (re-merge) at verify-pickup when `item.base_sha != current_main` (impl `92666feaa8`). `merge_queue.py` only. |
| **1658** | Post-merge verify can race a stale branch tip — pin verify to the merge-result SHA | `6fdf991a43` | 2026-06-06 05:53 | In `advance_main`, derive `verified_branch_tip = merge_sha^2` once before the CAS loop and pin the re-merge fallback to it (not the moving `task/<x>` ref), so post-verify commits pushed to the branch during the verify/advance window can't ride onto main unverified. Adds a divergence-canary WARNING. `git_ops.py:1602-1613, 1670-1705`. |
| **1645** | Verify-phase broken-main contagion guard | `bbc92fc715` | ~2026-06-04 | Workflow-side complement: `_verify_debugfix_loop` now detects a verify failure that already exists on main (inherited from merge-base) and escalates *once* instead of letting N tasks each self-patch the same file. This addresses the **4293 conflict-storm** root cause (the StatusBar.tsx conflict cluster), which Mechanism 2 alone does not. `workflow.py`, `verify.py`. |
| **1644** | Deliver immediate merge outcomes out-of-band of the verifier FIFO | `45dae9c12b` | ~2026-06-04 | Latency fix (conflict/blocked delivered without waiting behind a 12–90 min verify). Not freshness, but part of the same batch and removes the head-of-line stall that *amplified* staleness. |

**Pre-existing, confirmed still present (not the focus, verified intact):**
- **#1595** disjoint-delta re-verify gate (`67f572601e`): `reverify_on_rebase=True` is still passed at `merge_queue.py:4837`; the gate is at `git_ops.py:1737-1756`; `_reverify_rebased_tree` + `_rebase_delta_touched_overlap` at `merge_queue.py:1273-1475`.
- **#1602** fail-closed pre-advance unscoped type-check gate: `_run_unscoped_typechecks` inside `_run_post_merge_verify` at `merge_queue.py:447-477`.
- **#1603** raised cold merge-verify budget: `merge_verify_cold_command_timeout_secs` honored at `merge_queue.py:383`.

## (c) Current code path (SpeculativeMergeWorker — the live path)

```
_merger_loop (merge_queue.py:3927)
  └─ merge_to_main(worktree, branch, base_sha=main-at-dequeue)   git_ops.py:1248
       └─ _create_merge_worktree: `git worktree add --detach` at base; `git merge --no-ff`; scrub .task/
  └─ skip_verify = pre_rebased AND pre_merge_sha == base_for_merge   merge_queue.py:4232
  └─ counts_against_cap = not speculative;  _merge_ahead_cap.acquire()  [Mechanism 1]  :4241-4243
  └─ _verifier_queue.put(SpeculativeItem(base_sha=base_for_merge, ...))            :4245

_verifier_loop (merge_queue.py:4369)
  └─ item = _verifier_queue.get();  if counts_against_cap: cap.release()           :4389-4399
  └─ ★ Mechanism 2 (task 1646): for a real item (immediate_outcome is None,
       merge_result is not None, not GroupMergeRequest):
         current_main = get_main_sha()                                            :4461
         if item.base_sha != current_main:  remerge_reason = 'main_advanced'      :4462-4463
         → cleanup stale merge_wt; emit speculative_discard(main_advanced);
           item = await _remerge(req)   # RE-MERGE onto actual main BEFORE verify :4478-4489
  └─ _verify_and_advance(item)                                                     :4504
       ├─ Step 4 VERIFY (if not item.skip_verify):
       │     _run_post_merge_verify(merge_wt)   # incl. #1602 typecheck gate       :4768-4782 / 345-479
       └─ Step 5 ADVANCE:
             advance_main(current_sha, merge_wt, expected_main=item.base_sha,
                          reverify_on_rebase=True)                                 :4832-4838
               • 'advanced' → _finalize_advanced_merge (equivalence + chain)       :4840-4890
               • 'rebased_pending_reverify' (main moved DURING verify) →
                   #1595 _reverify_rebased_tree gate: disjoint → advance;
                   overlap → re-verify rebased tree, advance only if green         :4892-4990
```

`_remerge` (`merge_queue.py:4580`) re-merges `branch` against a freshly-read main
(`merge_to_main(base_sha=None)` → worktree built at current main HEAD, `git_ops.py:1340-1380`),
with a one-shot speculation-race retry.

**Train path** (`_do_train_merge`, `merge_queue.py:2742`) is *exempt* from both mechanisms but is
**rebase-before-verify by construction**: it rebases the tip onto current main (`rebase_onto_main`,
`:2922`), reads main HEAD *after* the rebase (`:2939-2940`), merges (`:2943`), then verifies
(`:2980`). Exemption is enforced by two guards in the Mechanism-2 `elif` (`immediate_outcome is
None` and `not isinstance(req, GroupMergeRequest)`, `:4441-4443`).

**Test coverage** (`orchestrator/tests/test_merge_queue.py`): Mechanism 1 cap bound
(`test_merger_ahead_cap_bounds_blocking_path`, ~:3754); Mechanism 2 pickup re-base
(`test_verify_pickup_rebases_when_main_advanced`, :3873); Mechanism 2 × chain-invalidation
(:3987); train double-exemption (:4133). The residual in (e) is **not** covered.

## (d) Original failure mode — can the StatusBar stale-base re-break still happen?

**No, not as a false verify failure.** Walking task 4178 (math-linalg Rust; the StatusBar.tsx
TS2769 was a stale-base artifact, not a 4178 defect) through the current code:
- The merger builds 4178's merge tree against main-at-dequeue. By the time the verifier picks it
  up, the canonical StatusBar fix (`fc93ce0ac6`) has landed on main, so `item.base_sha !=
  current_main`. **Mechanism 2 fires**: the stale tree is discarded and 4178 is re-merged onto
  current main — which *contains the fix*. `_run_post_merge_verify` then runs the gui typecheck on
  a tree that has the fix → **passes**. No re-surfaced TS2769.
- Even in the race where 4178 was built fresh and main advanced *during* its verify, the #1595
  `reverify_on_rebase` gate handles it: 4178 touches only Rust, the intervening main delta
  (StatusBar.tsx) is **disjoint** from 4178's branch-touched set, so `advance_main` rebases and
  the gate fast-paths to advance the rebased SHA (now containing the fix) without re-running the
  failed verify.

Task 4293 (`merge_request` returned `conflict` in StatusBar.tsx) is a *genuine overlapping-edit*
conflict, which a re-merge cannot dissolve. Its driver — every in-flight task inheriting a broken
StatusBar.tsx from main and self-patching it in-branch, producing a 7-conflict storm — is
addressed by **task 1645** (detect inherited-from-main failures, escalate once, don't self-patch).
So the conflict *storm* is cut off at the source; a lone true conflict is still correctly reported
as a conflict, not as a stale-base artifact.

## (e) Residual gap / recommendation

**Residual:** Mechanism 2 re-merges onto current main but the resulting `SpeculativeItem` keeps
`skip_verify=True` for `pre_rebased` requests, so the freshly re-merged tree is **not re-verified**.

Trace: a `pre_rebased=True` request was rebased onto main `M0` and verified green in-branch with
main *unchanged* (`workflow.py:1330-1331`). It is built against `M0` (`item.base_sha=M0`,
`skip_verify=True`) and queued. While it waits, main advances to `M1`. At pickup, Mechanism 2
sees `M0 != M1` and calls `_remerge`, whose **normal success return** recomputes
`skip_verify = req.pre_rebased AND pre_merge_sha == actual_main` (`merge_queue.py:4741-4751`) →
`True` (the re-merge was built against `M1`, so `pre_merge_sha == actual_main == M1`).
`_verify_and_advance` then takes `if not item.skip_verify` = False (`:4768`) and **skips both
`_run_post_merge_verify` and the #1602 type-check gate**. Because the pickup re-merge already put
the tree on `M1`, `advance_main(expected_main=M1)` fast-forwards with **no rebase**, so the #1595
`reverify_on_rebase` gate **does not fire** either. Net: for `pre_rebased` items, an interacting
change in the `M0→M1` delta can land **unverified** — the exact overlap #1595 was built to catch
at advance time, now bypassed because Mechanism 2 moved the freshening to pickup time.

This is the same hazard the **speculation-race-retry path already guards**: that branch forces
`skip_verify=False` unconditionally with an explicit comment — *"this branch is reached ONLY after
the gate confirmed main advanced … skipping verification would let semantically-unverified main
commits land … Always verify"* (`merge_queue.py:4629-4647`). The Mechanism-2 `main_advanced`
re-merge is reached under the *same* precondition (main advanced since build) but does not apply
the same reasoning.

Severity is bounded: it requires `pre_rebased=True` (low contention at submit, since main must not
move during the pre-merge rebase+verify) *and* main advancing only while that item waits in the
verifier queue *and* the `M0→M1` delta semantically interacting with the branch. It does **not**
reintroduce the StatusBar false-failure (opposite direction). But it is a genuine narrowing of the
verify-integrity guarantee the batch was meant to strengthen.

**Recommendation:** in `_remerge`, force `skip_verify=False` on the success-return path when the
re-merge was triggered by main advancing (mirror the race-retry path at `:4642-4647`). Simplest:
have `_verify_and_advance` / the Mechanism-2 site treat a `main_advanced` re-merge as
verify-required, or pass a flag into `_remerge` so the normal return sets `skip_verify=False`. Add
a boundary test: `pre_rebased` item, main advances while queued, assert `_run_post_merge_verify`
*is* called on the re-merged tree. **Filed as task 1672** (follow-up to 1646).
