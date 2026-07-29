# Deep merge-ahead: multi-item speculative chains for the reify merge queue

**Status:** active — 2026-07-29. Milestone: reify merge-queue throughput under the
post-path-concurrency arrival regime.
**Evidence base:** `plans/deep-speculative-verify-ahead-analysis-2026-07-22.md`
(+ its 2026-07-23 adversarial review), and the offline replay study
(`scripts/replay_deep_stack_study.py`, results in
`/home/leo/src/reify-replay-study/results.jsonl`, `script_version>=3`).

## Goal

When reify's merge queue holds N≥2 mergeable items, land multiple items per
verify run instead of one: build the full queued chain at dispatch time,
verify its tip, and CAS-land the whole verified prefix in order. Operator
observable: during a backlog episode a single passing verify lands ≥2 items
(`merge_finalized` events carry `landed_via_chain`), the depth histogram in
the merge report is populated, and landings/day tracks the new ~40–50/day
arrival rate instead of the current 15–30.

## Background and premise validity (G6)

The 07-22 analysis returned throughput-NO-GO on the then-true premise that the
queue was demand-limited (~85% idle). That premise is retired: completed
path-concurrency PRDs structurally raised arrivals (33/44/52 per day since
07-25 vs 15–30 landings/day; queue verify-bound since 07-25, mean arrival
depth 4.1 on 07-29). Under a persistently non-empty queue the saturated
capacity model applies: multi-item chains are a direct throughput lever.

The safety premise is empirically validated by the replay study, which
exercised **exactly this mechanism** (sequential merges in one worktree,
verify the tip, merge-faithful `--scope all` + `DF_VERIFY_ROLE=merge`):

- **57/59 historical 3–6-item chains passed** the full verify; both failures
  are item-attributable (task 5120's *single* fails the same tests).
  Stacking individually-good items introduced zero detectable interference.
- **~190/190 chain builds merged rebase-clean** (truncate-at-conflict never
  fired on real co-queued episodes; as-landed tips in land order).
- **Marginal cost per added item is small:** wall +~8%/item, CPU ~flat
  (fixed test-execution floor dominates). At the observed max queue (16),
  projected wall ≈ 2.1 ks against the 7200 s timeout.
- Landing safety is structural, not statistical: a chain lands **only** when
  the same merge-verify suite passes on the exact cumulative tree being
  landed. There is no new false-green path.

Pending (non-blocking, confirmatory): the study's S/C control re-runs
(singles baseline, same-tree flake at depth) — paused during the backlog,
~36 h of compute once resumed.

## Resolved design decisions

1. **Build-on-dispatch, not incremental build-ahead** (Leo, this session).
   When the deep slot dispatches, build the chain *right then*: sequentially
   merge queued items in submission order onto the head's merge commit in
   **one scratch worktree**, truncating at the first textual conflict, then
   verify the tip. No `_merge_ahead_cap` interaction, no per-item chain
   worktrees (intermediate links are commit SHAs only), no persistent chain
   state — invalidation = rebuild-per-round. Merges cost ~1–2 s per 6-chain.
2. **Anchor + deep tip allocation (v1).** Slot 1 is unchanged: head I0
   verified against real main (trust anchor). Slot 2 redirects to the chain
   tip. Nested/deeper allocations are explicitly deferred to the week-after
   assessment (task θ below).
3. **Tip pass is authoritative for the whole prefix.** On tip pass, land
   I0..Ik in order and cancel the head's in-flight verify; the tip tree is a
   verified superset and one-flake-roll-per-chain is the point of the
   mechanism. Head results matter only when the tip fails.
4. **Chain conflict ≠ genuine conflict.** A textual conflict at chain
   position j truncates the chain to j−1 and **must not** fire item j's
   conflict path — j may conflict with an *unlanded* predecessor; it takes
   its normal sequential path later.
5. **Failure policy: halve on fail, reset on pass.** Tip fail at depth d →
   next round builds at max(1, ⌊d/2⌋); any pass resets to min(queue, cap).
   Consecutive failures log-bisect toward the bad item; the floor (d=1) is
   byte-identical to today's adjacent verify. No separate rolling-flake
   suppressor in v1 (halving is the degradation mechanism; revisit in θ).
6. **Cap staging: 6 → 32 ("uncapped-in-practice").** Ship enabled at
   `chain_cap=6` (the study-validated depth), promote to 32 (≥2× max
   observed queue depth; still bounds wall vs the 7200 s timeout) via the
   milestone-predicate gate (η1/η2). `chain_cap=0` (the default) is the kill
   switch — byte-identical current behavior, `probe_fraction=0` precedent.
7. **Knob is green-tier hot-reloadable** (`merge_deep.chain_cap` in
   RELOADABLE_FIELDS), so enable/adjust/kill never needs a restart.
8. **Telemetry gets a fresh, truthful field.** `merge_verify.chain_items`
   (1-indexed count of items in the verified tree) replaces reliance on the
   broken `depth` label (probe-era off-by-one / attribution semantics; see
   the 07-23 investigation). `merge_finalized` gains `landed_via_chain: k`.
   Historical `depth>=2` events remain excluded from calibration.
9. **CAS staleness aborts the walk.** The chain anchors on I0's merge commit
   (parent = main-at-build). If main moved externally (direct-to-main
   commits happen), the first failing CAS aborts the remaining walk; unlanded
   items stay queued and the next round rebuilds on the new main.

## Pre-conditions (G3 — all verified against code this session)

- Frozen-prefix / in-order CAS machinery: `merge_queue.py` frozen_prefix
  (task 1890), per-item CAS advance — exists, exercised at K=2 daily.
- Scratch chain building: plain sequential `git merge` in a worktree — the
  replay study's `build_stack` is the working reference implementation.
- Hot-reload green-tier plumbing: `speculation_probe.*` precedent.
- Verify dispatch onto an arbitrary built commit: **new code** (the probe's
  Phase-1 limitation never redirected dispatch) — this PRD builds it (γ).
- Runtime conservation audits (`speculation_accounting_violations`,
  PermitLedger identities): exist; chain path must keep them green (ι).

## Cross-PRD seams (G4)

| Seam | Direction | Mechanism | Owner |
|---|---|---|---|
| spec warm-lane pool (`git.merge_spec_warm_lane_pool`, live since reify 4941) | consumes | chain build + tip verify claim a pooled `_spec-N` lane via `acquire_spec_lane` (`merge_liveness.py:697-708`) — no bespoke scratch dir, pool owns worktree lifecycle/locking | pool machinery owns lanes; **this PRD** owns chain usage of one |
| **DF 3003** (in flight 07-29): typed contended-DEFER on `reset_persistent_merge_worktree` | consumes + depends | the prefix-landing walk (δ) runs the terminal finalize path (incl. `warm_swap_worktree`, `merge_liveness.py:715`) k× per round and must inherit 3003's contended→DEFER classification, never the bare-RuntimeError→blocked path; γ/δ carry a task dep on 3003 | **3003** owns classification; **this PRD** consumes |
| **DF 3071** (pending, ext-dep reify:5608): serial-head lane-lock admission gate | coordinates | chain dispatch is spec-lane-side and structurally exempt from the serial-head gate; but both edit `merge_liveness.py`/`merge_queue.py`/`config.py`/`defaults.yaml` (module-lock serialization), and δ's head-verify cancellation must release the verify lease cleanly or 3071's guard reads `_merge-verify` BUSY and defers the fleet | **3071** owns the gate; **this PRD** owns lease-clean cancellation (δ) |
| merge-worktree-lifecycle (tasks 2924–2930) | consumes | lane/worktree lifecycle primitives (via the spec pool) | lifecycle PRD |
| config-hot-reload PRD | extends | adds `merge_deep.chain_cap` to RELOADABLE_FIELDS | **this PRD** |
| deep-speculative-verify-ahead analysis + replay study | consumes | evidence only; study's `build_stack` is the chain-builder reference | — |
| speculation probe (task 2359, deactivated 07-23) | supersedes | probe stays inert (`probe_fraction=0`); `chain_items` replaces its broken depth labels | **this PRD** |

Deep chains also *reduce* serial-head `_merge-verify` contention (fewer, batched
verifies per landing), so this PRD, 3003, and 3071 attack the same incident
class (reify esc-5354-4 / esc-5363-5) from complementary sides.

## Decomposition plan

Phase 1 — foundation:

- **α — config: `merge_deep.chain_cap` knob + kill-switch default.**
  Modules: orchestrator config. Signal: `reload_config` applies the knob
  live (`applied` disposition); at default 0 the full existing merge suite
  passes unchanged (byte-identity). Prereqs: none.
- **β — chain builder (build-on-dispatch).** Modules: orchestrator merge
  queue. Build ordered chain in one scratch worktree from a queue snapshot;
  truncate at first conflict; never emit conflict outcomes; return links +
  tip. Signal: integration test — queue fixture incl. a conflicting item
  yields the clean prefix, conflicted item untouched, exactly one worktree
  used. Prereqs: α.

Phase 2 — vertical slice (minimum end-to-end):

- **γ — deep-tip verify dispatch + halving state.** Modules: merge queue,
  merge_liveness (spec-lane dispatch seam), verify runner. Slot 2 claims a
  `_spec-N` lane and dispatches onto the chain tip when cap>0 and queue≥2;
  halving/reset state machine; emits `chain_items`. Signal: integration —
  dispatch depths across scripted pass/fail sequences match the policy; d=1
  floor byte-identical to today's adjacent verify. Prereqs: β, **DF 3003**
  (out-of-batch: typed contended-DEFER must precede new dispatch logic on
  the shared seam).
- **δ — prefix landing on tip pass.** Modules: merge queue (CAS/finalize),
  merge_liveness. Cancel head verify **with clean verify-lease release**
  (3071's guard must read the lane IDLE afterward), CAS-land links in order
  via the terminal finalize path (inheriting 3003's contended→DEFER
  classification), per-item `merge_finalized` done + `landed_via_chain`;
  stale-CAS abort semantics (decision 9). Signal: integration — one passing
  tip lands k+1 items in order, main history linear, conservation audits
  green, lane lock released post-cancel; tip fail lands nothing via the
  chain and head path proceeds untouched. Prereqs: γ, **DF 3003**.
- **ι — two-way boundary/integration gate (B+H).** Modules: orchestrator
  tests. The boundary-test sketch below, implemented against both the worker
  and the CAS/ledger sides. Signal: suite green in CI; this is the
  integration gate for the slice. Prereqs: δ.

Phase 3 — observability + rollout:

- **ε — telemetry + report.** Modules: event store, report/digest,
  `analyze_speculation_depth.py`. Depth histogram + items-per-verify +
  deep-fail rate in the merge report; reader keys on `chain_items`. Signal:
  report renders from live events; reader output matches git-derived truth
  on fixtures. Prereqs: δ.
- **ζ — reify canary enable at cap=6.** Deterministic deploy task
  (`before_done` script: set `merge_deep.chain_cap: 6` in reify's
  `dark-factory-orchestrator.yaml`, commit, reload, assert `applied`).
  Signal: reify orchestrator reload disposition shows the knob live; first
  `chain_items>=2` verify event observed. Prereqs: ι, ε.
- **η1 — 7-day canary predicate.** Deterministic task, milestone
  {delayed, 604800 s after ζ}, `before_done` {kind: predicate} script
  reading reify runs.db: deep-fail rate, items-landed-per-verify, timeout
  proximity, drain-time vs the pre-canary window. Exit 0 → done; non-zero →
  born-at-L2 `milestone_check_failed`. Signal: task reaches done with the
  predicate's stdout tail as `note`, or escalates. Prereqs: ζ.
- **η2 — promote cap 6 → 32.** Deterministic deploy task depending on η1
  (runs only if the predicate passed). Signal: reload disposition shows
  cap=32 on reify. Prereqs: η1.
- **θ — week-after aggressiveness assessment** (Leo, decision 1). Normal
  task, milestone {delayed, ~604800 s after η2}: assess nested allocation,
  halving-policy tuning (retry-once vs halve), cap ceiling, and the S/C
  study confirmations against canary telemetry; file follow-ups if
  warranted. Signal: written assessment committed to `plans/`, follow-up
  tasks filed or explicitly declined. Prereqs: η2.
- **κ — docs correction pass.** Update `skills/orchestrate/SKILL.md` merge
  notes + OPERATIONS.md for the new knob/telemetry; mark the analysis doc's
  §11.5 as implemented. Signal: docs name the knob and the halving policy.
  Prereqs: δ.

## Contract (B+H)

- `build_chain(queue_snapshot, head_merge_commit, cap, target_depth) ->
  ChainResult { links: [(task_id, merge_commit)], tip: sha,
  truncated_at: task_id | None }` — sequential submission-order merges in
  one scratch worktree; pure w.r.t. queue state; never emits per-item
  outcomes; ~O(seconds).
- Dispatch invariant: chain built iff slot 2 free ∧ cap>0 ∧ queue≥2;
  `target_depth = min(len(queue), cap, halving_state)`;
  halving_state: fail(d)→max(1,⌊d/2⌋), pass→min(queue, cap).
- Landing invariant: tip pass ⇒ walk links in order through the existing
  CAS advance; each landed item gets its normal finalize path + events;
  first CAS failure aborts the walk (remaining items stay queued). Tip fail
  ⇒ chain discarded, zero queue mutation, head path unaffected.
- Kill switch: cap=0 ⇒ no chain code executes on any dispatch path
  (byte-identity, asserted by test).
- Telemetry: `merge_verify.chain_items >= 1` on every merge verify;
  `merge_finalized.landed_via_chain = k` iff landed by a chain walk.
- Conservation: existing PermitLedger / speculation identities hold under
  chain landing (chain consumes no per-item speculation permits).

## Boundary-test sketch (B+H)

| Scenario | Preconditions | Postconditions |
|---|---|---|
| Tip pass lands full prefix | 4-item clean chain, tip verify green | 4 in-order landings, one verify, `landed_via_chain=4`, audits green |
| Tip fail leaves queue intact | chain built, tip verify red | zero landings via chain, items still queued, halving state = 2 |
| Halving walk isolates bad item | item 3 of 6 genuinely red | depths 6→3→1 over rounds; items 1–2 land sequentially at floor; deep resumes after bad item blocks |
| Chain conflict truncates silently | item 2 conflicts with item 1 textually | chain = [item 1], no conflict outcome for item 2, item 2 handled sequentially later |
| Head-fail + tip-pass | head verify red (flake), tip green | full prefix lands (tip authoritative), head verify cancelled |
| Stale CAS aborts walk | main advanced externally mid-verify | walk aborts at first CAS failure; unlanded items requeue; next round rebuilds |
| Kill switch byte-identity | cap=0 | dispatch/behavior identical to pre-PRD golden transcript |
| Deep fails never feed thrash guard | 2 consecutive tip fails | zero blocked MergeOutcomes, `consecutive_merge_thrash` untouched (3003's signature class cannot recur via chains) |
| Lease released on head-cancel | tip pass cancels in-flight head verify | `warm-lane-lock-guard.sh check` (3071's oracle) reads IDLE within one round |
| Hot-reload | cap 0→6 via reload_config | next dispatch round builds a chain; no restart |
| Timeout margin | 16-item chain (cap 32) | verify completes ≪ 7200 s or times out cleanly via existing path |

## Out of scope

- Nested / both-slots-deep allocation, adaptive depth controller, K>2 or
  extra verify hosts, retry-once-before-halve — all live in θ's remit.
- Dashboard depth-rollup panel (report/digest only in this PRD).
- Any change to task-side verification or the task pipeline.

## Open questions (tactical)

1. ~~Scratch worktree provisioning~~ **RESOLVED 07-29** (task-3071 scope
   correction surfaced `merge_spec_warm_lane_pool` live since reify 4941):
   chain build + verify claim a pooled `_spec-N` lane via
   `acquire_spec_lane`; no bespoke scratch dir.
2. **Head-verify cancellation mechanics** — cooperative cancel point vs
   letting it finish and discarding the result. Hard requirement either
   way: the verify lease / lane lock must be released promptly (3071's
   admission guard and 3003's contended path both key on that inode).
   Decide in δ using the existing `verify_cancel.py` machinery.
3. **η1 predicate thresholds** — exact numeric gates derived from the first
   soak week's baseline; the script owns the comparison per the predicate
   contract. Decide when authoring η1's script.
4. **`depth` field back-compat** — keep emitting legacy `depth` alongside
   `chain_items` or retire it. Suggested: keep one release, then retire.
   Decide in ε.
