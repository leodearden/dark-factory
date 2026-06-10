# PRD: Merge-worktree liveness heartbeat — reconcile the enforced liveness-margin guard with multi-host verify (K>1)

**Date:** 2026-06-10 · **Status:** approved for decomposition · **Origin brief:**
`~/.claude/spawn-briefs/prd-liveness-guard-reconciliation-2026-06-10.md`

Cite by symbol; line refs are as-of `main` 8b703a550a and drift.

## 1. Consumer + user-observable surface

- **Consumer:** the operator enabling Lever C (multi-host verify) for reify — the
  `verify_runners` enable checklist in `plans/lever-c-enable-path-gap-2026-06-10.md`
  (steps 2–5), currently blocked at step 0 by this bug. Secondary consumers: the merge
  worker's startup path (`Harness._start_merge_worker`) and the stale-worktree reaper's
  threat model (`coalesce_or_enqueue_merge_request`).
- **User-observable surface:** a reify `orchestrator.yaml` with one enabled
  `verify_runners` entry (K=2) that today crash-loops the orchestrator at startup with
  `MergeLivenessConfigError` **starts cleanly** ("Speculative merge worker started", stable
  NRestarts) — while a config that genuinely risks mid-flight reaping is **still refused**,
  and a legitimately-queued local `_merge-*` worktree is **never** reaped mid-verify.

## 2. G6 premise validation — VERDICT (resolved this session, 2026-06-10, by code reading)

The brief required validating, before choosing a fix: *with K=2, how many LOCAL `_merge-*`
worktrees can sit frozen at once?* Answer: **up to K queued + 1 in-verify — and the frozen
window is even longer than the guard's formula models.** Evidence:

1. **No early-free for remote verifies.** `_run_post_merge_verify` (merge_queue.py:648)
   holds `merge_wt` for the whole verify: cleanup happens only after `pool.dispatch`
   returns (fail path :760; pass path via the verifier's finalize). `RemoteRunner.run_merge_verify`
   (verify_runner.py:662) never executes in the local worktree — it `git push`es the merge
   SHA (using `merge_wt` only as push `cwd`) and runs `ssh <host> orchestrator verify-merge`;
   the remote host builds its own worktree. The local worktree lingers, mtime-frozen,
   until the remote verify completes.
2. **The brief's per-host hypothesis is FALSE.** The reverted reify yaml comment ("the
   LOCAL reaper holds ≤1 `_merge-*` worktree, so the liveness call likely should use the
   per-host bound") confuses verify-*execution* locality with worktree locality. Every
   speculative merge commit is created **locally** by the merger (`SpeculativeItem.merge_wt`,
   merge_queue.py:3223); remote hosts only receive pushed SHAs. `_merge_ahead_cap`
   (K permits, released on-drain — :5660) bounds handed-off-not-yet-drained items at K, so
   up to **K local worktrees sit mtime-frozen in the verifier queue** while one more is
   in-verify. Worst queue wait ≈ K × cold_timeout — the guard's formula is *grounded*.
3. **Prefer-remote makes it worse, not better.** `VerifyRunnerPool._select_runner`
   (verify_runner.py:855) prefers the remote runner for every merge; the verifier is a
   single serial coroutine (`run()` spawns exactly one `_verifier_loop`, :4936). A
   remotely-verified item gets **no local mtime refresh during its own verify** (the
   LocalRunner's build activity is what incidentally refreshed it before; there is no
   explicit touch anywhere — verified by grep for `utime`/`touch` over merge_queue.py,
   git_ops.py, verify.py). Worst frozen age ≈ (K+1) × cold_timeout = 21600s at K=2 —
   above even the un-factored liveness window (10800s).
4. **Latent K=1 hazard.** Even at K=1 with a remote runner, a queued item can freeze for
   queue-wait (7200s) + its own remote verify (7200s) = 14400s > 10800s. The guard passes
   that config (7200 < 8100) — the static formula already under-models prefer-remote.

**Conclusion: this is not a guard bug. The K=2 config is genuinely over-budget under the
current worktree lifecycle. Passing a per-host bound (`ceil(K/num_hosts)=1`) would silence
the guard while leaving the physical hazard — a hard-constraint violation. The fix must
change the physics: make the worktree mtime a true liveness signal.**

## 3. Approach — owner heartbeat on live merge worktrees

The reaper's real question is "is this worktree's owner alive and still intending to
verify it?" — mtime-age is its proxy. Today the proxy is refreshed only by incidental
local build activity. Fix: refresh it **deliberately**.

- **Mechanism (α):** `SpeculativeMergeWorker` keeps a ledger (`set[Path]`) of every live
  owned `_merge-*` worktree — registered when the merger creates it (every path that sets
  `SpeculativeItem.merge_wt`, including `_remerge`'s replacement worktrees), deregistered
  at every cleanup site (dereg-before-cleanup so a failed remove can't immortalize an
  orphan). The existing independent `_heartbeat_loop` (merge_queue.py:4914, poll
  `_HEARTBEAT_POLL_S=30.0` :178 — already designed to keep running while merger/verifier
  block) additionally `os.utime`-touches each ledger path every tick, unconditionally
  (NOT subject to `_maybe_log_queue_heartbeat`'s log rate-limit). ENOENT → drop from
  ledger (INFO); persistent other failure → one WARNING per path (loud, never silent —
  standing directive).
- **Consequence:** a live worker's worktrees never age past ~1 poll period (30s) — 360×
  inside the 10800s liveness window — independent of K, cold timeout, prefer-remote
  routing, queue depth, or any future concurrent-verify refinement. A dead owner stops
  touching; its worktrees age and are reaped exactly as today. The reaper's protection is
  **strengthened** (mtime now means "owner alive recently", which is what it always
  pretended to mean), and the latent K=1-remote hazard (§2.4) is fixed as a side effect.
- **Guard reformulation (β):** `check_merge_liveness_margin` /
  `enforce_merge_liveness_margin` (merge_queue.py:6622/:6750) re-derive the worst case
  from the new physical model: worst frozen age of an owned worktree =
  `_HEARTBEAT_POLL_S × TOUCH_MISS_TOLERANCE` (tolerance covers event-loop stalls; e.g. 20
  → 600s floor), enforced `< safety_factor × liveness_secs`. `MergeLivenessConfigError`
  and the fail-closed shape at `Harness._start_merge_worker` (harness.py:3355-3360) are
  retained; the raw-`_k` coupling at :3356 is removed (resolving the asymmetry vs the
  serial-lane call at :3367). A genuinely over-budget config — one whose injected
  `liveness_secs` falls at/below the heartbeat floor threshold — is **still refused**.
  K drops out of the formula because the model changed, not because the guard weakened.
- **Boundary gate (γ):** two-way integration tests: the exact reproduced production crash
  config now starts; the reaper still reaps dead-owner worktrees and never reaps
  live-owner ones.

### Rejected alternatives (recorded for the next session that re-derives this)

| Alternative | Why rejected |
|---|---|
| Per-host bound `ceil(K/num_hosts)` at harness.py:3356 | Premise false (§2.2) — silences the guard, leaves the hazard. Violates the hard constraint. |
| Free local `merge_wt` early on remote dispatch | Only frees the 1 in-verify worktree; the K *queued* frozen worktrees (the formula's actual subject) remain. Also disturbs the coalesce in-flight on-disk marker and the merge SHA's gc anchor (worktree HEAD). Insufficient AND riskier. |
| Raise `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` (≥(K+1)×7200/0.75 = 28800s under the corrected model) | Scales the abandoned-detection window with topology (~8h reap latency; post-crash same-branch merges coalesce-wedge that long), must be re-derived on every K/timeout change, and dies again when concurrent verify lands. |
| Lower cold timeout | Not viable — reify cold merge verify is ~90 min; needs 7200s. |

## 4. Pre-conditions (G3 — all verified on main this session)

`enforce_merge_liveness_margin`/`check_merge_liveness_margin`/`MergeLivenessConfigError`/
`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` (merge_queue.py:6750/:6622/:6740/:2184);
`_start_merge_worker` K wiring `_k = 1 + len(enabled_verify_runners)` (harness.py:3347)
and `enforce_persistent_worktree_serial_lane(num_hosts=_k)` (:3367); reaper branch in
`coalesce_or_enqueue_merge_request` (merge_queue.py:3004-3041, mtime-stat at :3009);
`SpeculativeItem.merge_wt` (:3223); independent `_heartbeat_loop` (:4914) +
`_HEARTBEAT_POLL_S` (:178); `RemoteRunner` (verify_runner.py:592) +
`VerifyRunnerPool._select_runner` (:855); `OrchestratorConfig.verify_runners`/
`enabled_verify_runners` (config.py:1328/:1411); guard tests exist in
`orchestrator/tests/test_merge_queue.py`, `test_harness_k_from_config.py`,
`test_multihost_verify_integration.py`. The **owned-worktree ledger is the one new
mechanism** (produced by α, consumed by α's heartbeat + β's model + γ's boundary tests).

## 5. Resolved design decisions

1. **Heartbeat over static re-derivation** — §3, converged on evidence (§2).
2. **Touch every tick of the existing `_heartbeat_loop`** — no new task/thread; the loop
   already survives merger/verifier blocking and swallows exceptions.
3. **Ledger lives on `SpeculativeMergeWorker`** — it owns every worktree whose frozen age
   scales with K. Other `_merge-*` creators (cold-shadow verify :7681, drift check) are
   short-lived local executions whose build activity refreshes mtime; out of ledger scope
   (documented in code).
4. **Reaper and `prune_stale_merge_worktrees` semantics unchanged** — zero changes to
   coalesce/reap/prune code paths. Note: ENOSPC prune (`keep=merge_wt`, :744) already
   force-removes *other* queued worktrees regardless of mtime; chain re-merge recovers.
   Heartbeat does not alter that (pre-existing, separate concern).
5. **Guard stays fail-closed with the same exception type and call-site shape** — only the
   formula's physical model changes. `check_…` stays the warn-only twin, same new model.
6. **K=1 shipped config: behaviour-identical outcome** — guard passes (as today), worktree
   lifecycle unchanged except added protective touches.

## 6. Out of scope

- Flipping reify's `verify_runners` on (operator action; enable checklist steps 2–5 in the
  gap report — this PRD is its step-0 unblock). The reify yaml comment block and gap
  report are updated by the operator at flip time, not by these tasks.
- Concurrent multi-host verify dispatch (the pool's "K-permit free/busy refinement");
  heartbeat is deliberately robust to it landing later.
- Early-free of local worktrees for remote verifies (worthwhile only as a disk-footprint
  optimisation; file separately if ever needed).
- The ENOSPC-prune-vs-queued-worktrees interplay (pre-existing, recovered by chain
  re-merge).

## 7. Cross-PRD seams (G4)

| Seam | Owner | Status |
|---|---|---|
| Lever C enable path (df 1716, merged 26894cceec) — `verify_runners`→K wiring | landed; this PRD reconciles its startup guard interaction | this PRD |
| Liveness guard (df 1714/1715) — the enforcing wrapper | formula re-derived here; fail-closed contract preserved | this PRD |
| Warm-builds persistent worktree (df 1692) — `enforce_persistent_worktree_serial_lane`, prune-exemption | untouched (num_hosts call unchanged; persistent worktree never in ledger — it is reset-in-place, not frozen-queued) | df 1692 invariants, regression-held by γ |
| Lever C PRD mechanism ζ (`plans/merge-throughput-multihost-verify-prd.md`) "liveness recompute per host" | superseded by this PRD's §2 verdict (per-host premise false) | this PRD |

## 8. Decomposition (G5: B+H — contract above, two-way boundary tests in γ)

- **α — Owned-worktree liveness heartbeat** (`orchestrator/src/orchestrator/merge_queue.py`
  + tests). Ledger register/deregister at all `SpeculativeItem.merge_wt`
  creation/cleanup sites; `_heartbeat_loop` touches ledger paths every tick; ENOENT
  self-heal; loud WARNING on persistent touch failure. **Signal:** with the verifier
  blocked on a slow fake verify and ≥1 item queued, the queued item's worktree-root mtime
  advances within 2×`_HEARTBEAT_POLL_S` (test-observable; DEBUG "touched N owned merge
  worktree(s)" in the journal); after `stop()`, touches cease and mtime freezes.
  **Consumer:** β (formula premise), γ, the reaper's threat model.
- **β — Re-derive the liveness-margin guard to the heartbeat model**
  (`merge_queue.py` + `harness.py` + tests). Depends on α. New worst-case =
  heartbeat-floor; drop the `_k` coupling at harness.py:3356; keep
  `MergeLivenessConfigError`, fail-closed shape, warn-only twin. **Signal:** the exact
  reproduced crash parameters (cold_timeout=7200s, one enabled runner ⇒ K=2) pass the
  guard with default constants; an injected over-budget config (`liveness_secs` at/below
  the heartbeat-floor threshold) still raises `MergeLivenessConfigError`; shipped K=1
  defaults pass (regression); the refusal message describes the heartbeat model.
  **Consumer:** γ, `_start_merge_worker`.
- **γ — K=2 startup + reaper-protection boundary gate** (tests; integration). Depends on
  α, β. (a) Startup repro: `_start_merge_worker` with a config bearing one enabled
  `verify_runners` entry + `merge_verify_cold_command_timeout_secs=7200` starts the merge
  worker cleanly — this test is RED on pre-α/β main (it is the production crash);
  (b) two-way reaper boundary: `coalesce_or_enqueue_merge_request` coalesces (does NOT
  reap) a worktree with live-owner mtime, and still reaps + re-dispatches one aged past
  `liveness_secs` (dead owner). **Signal:** both tests green in CI; (a) demonstrably RED
  against pre-batch main. **Consumer:** the operator enable checklist (gap report steps
  2–5) — the user-facing "Lever C can now be switched on".

**DAG:** α → β → γ; α → γ.

## 9. Open questions (tactical)

- Exact `TOUCH_MISS_TOLERANCE` constant (anything in 10–60 keeps ≥3× margin; pick and
  document).
- Whether `enforce_merge_liveness_margin` keeps accepting `merge_ahead_bound` as an
  informational/deprecated parameter or drops it (callers/tests updated in β either way).
- Ledger structure (`set[Path]` vs dict with registration timestamps for diagnostics).
- Whether the touch DEBUG log is per-tick or rate-limited alongside the queue heartbeat.
