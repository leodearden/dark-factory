# Two-Layer Merge Queue — Architectural Reference

**Status:** built and landed (2026-06-25, λ=1895 pipeline landed via `bbaec52696`)
**PRD:** [plans/two-layer-merge-queue-prd.md](../../../plans/two-layer-merge-queue-prd.md)
**Operator guide:** [skills/merge-queue/SKILL.md](../SKILL.md) (§ "The two-layer merge queue")

This document is the developer-facing architectural companion for the two-layer merge queue.  It covers the layer model, §5.3 invariants, the as-built symbol map, Greek→task-ID provenance, and the gain model.  The operator guide (SKILL.md) is the right starting point for day-to-day use; this doc is for contributors modifying the merge-queue implementation.

---

## 1. Layer model

The merge queue is divided into two structurally distinct layers:

| Layer | Role | Contents | Mutable? | Disk usage |
|-------|------|----------|----------|------------|
| **Layer 1: Speculative merge graph (suffix)** | Deep conflict-graph over all queued items not yet verifying | `_lane_buffers` (unfrozen suffix) | Fully reorderable | Disk-free — conflict graph is in-memory git-objects only; no `_merge-*` worktrees created |
| **Layer 2: Frozen verify frontier (prefix)** | Shallow, immutable set of items currently in verify or already landed | `_inflight` (frozen prefix) | Immutable — no reorder, no re-base | Each item holds a real `_merge-*` worktree for the in-flight verify |

The **frozen prefix** = {verifying} ∪ {landed}.  The **unfrozen suffix** = everything else in the lane.

---

## 2. Key design invariants (§5.3)

All four invariants must hold simultaneously.  `two_layer_invariants(main_sha)` → `[]` when healthy; a non-empty list describes specific violations.

### I1. Frozen prefix immutable

An item in the frozen prefix (verifying or landed) is **never reordered or re-based** out from under an in-flight verify.  Its `_inflight` position and `base_sha` are set at dispatch time and do not change.

### I2. Verify base equals frozen-prefix tip

Every real-verify dispatch uses **exactly the tip of the frozen prefix** as its base SHA (`frozen_prefix_tip(main_sha)`).  Dispatching against any other base is a violation (logged at WARNING by `_warn_if_base_not_frozen_tip`).

### I3. Reorder touches only the unfrozen suffix

`recompute_suffix_conflict_graph()` updates only `_lane_buffers`; `_inflight` order and `base_sha` are unchanged.  Reordering is always disk-free (no `_merge-*` worktree side effects).

### I4. Liveness / no-starvation

An item in a conflict clique eventually becomes head of its clique and is picked (age-of-first-submission ordering ensures no item is permanently blocked by newer arrivals).  Items disjoint from all ahead bypass a blocked clique.

---

## 3. Graph-time bounce (`needs_rebase`, η=1892)

When `recompute_suffix_conflict_graph()` detects a textual conflict between two suffix items, it **bounces the younger item** (`_bounce_conflicting_suffix_items`) — at conflict-graph time, **disk-free**, before any verify slot is consumed or `_merge-*` worktree is created.

**Mechanical-rebase-first protocol:**

1. Attempt a speculative rebase of the younger item onto the frozen-prefix tip.
2. **Clean rebase** → item is re-queued; `merge_first_enqueued_at` is unchanged (aging priority preserved); no agent dispatched.
3. **Real conflict** → item is escalated with `NEEDS_REBASE_REASON_PREFIX` in the reason string.
4. **Bounce cap exceeded** (`MERGE_BOUNCE_CAP = 3`) → the 1688 thrash-backstop triggers: item is blocked without further rebase attempts.

Constant: `NEEDS_REBASE_REASON_PREFIX = 'Suffix item needs rebase onto frozen-prefix tip'`

---

## 4. Conflict-clique aging order and disjoint throughput bypass (ζ=1891)

Within a footprint conflict clique, items are ordered by **age of first submission**:

```python
_aging_key(req) = (merge_first_enqueued_at or enqueued_at, request_id)
```

The item with the smallest `_aging_key` (oldest first submission) has priority.  This preserves the most expensive work — the branch that has been waiting longest gets the cleanest landing shot.

`merge_first_enqueued_at` (α=1886) is persisted **write-once** in task metadata at the per-task merge-submit chokepoint (`workflow.py`).  It survives orchestrator restarts.  Legacy entries without this field fall back to `enqueued_at`.

**Disjoint throughput bypass:** an item whose footprint overlaps no item ahead of it in the lane bypasses out-of-order — it never waits behind a blocked clique.  This preserves throughput for independent branches even when a conflict clique is stalled.

---

## 5. No-landings circuit-breaker (θ=1893)

`NoLandingsCircuitBreaker.observe()` fires when both conditions hold over a sliding window:
- **Landing rate ≈ 0** (no increase in `landings_total`)
- **Warm-lane free bytes falling** (disk pressure building)

On trigger, `NoLandingsCircuitBreaker.observe()` emits a `BreakerTrip` decision object.  The Harness pass (`_run_no_landings_breaker_pass`, harness.py) acts on that decision:
1. Calls `force_halt_scheduler` to stop dispatch.
2. Files an L2-INFO escalation (role `orchestrator-no-landings-breaker`).

The breaker itself is a pure read/decide component; the Harness pass owns all side effects.

**Auto-resume:** when a clean landing occurs (`landings_total` rises) or disk recovers, the breaker transitions to RECOVERING and emits a resume signal.

---

## 6. Operator-observable heartbeat keys

`SpeculativeMergeWorker.snapshot()` exposes these additive, backward-compatible keys:

| Key | Type | Description |
|-----|------|-------------|
| `suffix_conflict_graph` | dict | In-memory conflict-graph edges for unfrozen suffix items (δ=1889) |
| `frozen_prefix` | dict | `{request_ids, tip_merge_commit, verify_depth}` — current frozen-prefix state (ε=1890) |
| `metrics` | dict | `{retries_per_landing, drift_at_detection: {count, last, mean, max}, landings_total}` (ι=1894) |
| `two_layer_invariants` | list\[str\] | `[]` when all §5.3 invariants hold; violation strings otherwise (λ=1895) |
| `hosts` | list\[dict\] | Per-verify-host state: `{name, is_local, slot_state: free\|busy\|parked\|null, quarantined, quarantine_class: ru\|divergence\|null, unavailable_since, unavailable_secs, streak, reason}` (task 3275) |

`hosts` is what makes an under-full `verifying N/M hosts` line diagnosable — it
has four possible causes, each with its own discriminator: **RU-quarantine**
(`quarantine_class == 'ru'` — auto-recovers via `_reprobe_quarantined_hosts`);
**divergence-quarantine** (`quarantine_class == 'divergence'` — held for verdict
parity, operator-cleared only); **leaked slot** (`slot_state` `busy`/`parked`
with no matching `occupancy.inflight_by_host` occupant); and **free and never
asked for** (`slot_state == 'free'`, `quarantined == false`). `unavailable_since`
/ `unavailable_secs` / `streak` / `reason` are populated whenever the host is
RU-tracked, independent of quarantine, so a host accumulating failures below the
quarantine threshold is visible too — `unavailable_since` is an absolute epoch
(for log correlation), `unavailable_secs` the derived downtime relative to the
snapshot's own clock (the "how long has this host been down" form, matching every
other age field). `hosts == []` means the allocator has not been built yet (no
verify dispatched) — deliberately not a fabricated `local` entry.

An RU-tracked host with **no allocator slot** (an orphan — e.g. a remote dropped
from the pool while its failure streak was live; the streak map is pruned only on
recovery, and the allocator is built once per worker) is appended after the
managed hosts with `slot_state: null`, so an orphaned streak is never invisible.
Hence `len(hosts) == occupancy.hosts_total + <orphan count>`, the two being equal
in the steady state.

The `occupancy` block: `hosts_busy` is the count of **distinct busy hosts**, NOT
the number of verifies in flight — read `inflight_total` for that.
`inflight_by_host` is the authoritative lossless `{host: [task_id, ...]}` view
(finalize head first, then `_inflight` order); `by_host` is the
retained-for-compatibility lossy `{host: task_id}` map, which keeps only the last
occupant when two entries share a host.

---

## 7. As-built symbol map

> **Where the code lives (task 5036, PRD `plans/merge-lane-quality-prd.md` ζ2).** The lane is the
> package `orchestrator/src/orchestrator/merge_lane/`; Location cells below are relative to it
> unless they name another file. The old `orchestrator.merge_*` module paths
> (`orchestrator/src/orchestrator/merge_queue.py`, `merge_types.py`, `merge_gates.py`, …) are now
> thin module aliases that resolve to the new modules, and task 5037 deletes them — cite the
> new paths. The façade `orchestrator.merge_lane` (`merge_lane/__init__.py`) lazily exports the
> public surface (`MergeLane`, the request entry points, the value types, the reason constants,
> the ports). `merge_queue.py` became `worker.py`; the other `merge_`-prefixed modules dropped the
> prefix. `git_ops.py`, `lane_lifecycle.py`, `warm_lane_pool.py`, `offline_lane.py`,
> `recover_main.py` and `suffix_graph.py` stayed outside the package.

| Symbol | Location | Description |
|--------|----------|-------------|
| `NEEDS_REBASE_REASON_PREFIX` | merge_lane/worker.py | Prefix of the `needs_rebase` bounce reason string |
| `MERGE_BOUNCE_CAP` | merge_lane/worker.py | Max bounce count before thrash-backstop triggers (= 3) |
| `_aging_key(req)` | merge_lane/worker.py | Sort key: `(merge_first_enqueued_at or enqueued_at, request_id)` |
| `merge_first_enqueued_at` | merge_lane/types.py | Write-once field: epoch of first submission to the merge queue |
| `SuffixConflictGraph` | suffix_graph.py | In-memory conflict graph over the unfrozen suffix |
| `NoLandingsCircuitBreaker` | merge_lane/worker.py | No-landings circuit-breaker decision object (θ=1893) |
| `_pop_next_pickable()` | merge_lane/worker.py | Select next item using clique-scoped aging (ζ=1891) |
| `frozen_prefix()` | merge_lane/worker.py | Return ordered `request_id`s in the frozen prefix |
| `frozen_prefix_tip()` | merge_lane/worker.py | Return the base SHA for next verify dispatch |
| `check_frozen_prefix_invariant()` | merge_lane/worker.py | §5.3 I1+I2 violations (base-chain integrity) |
| `two_layer_invariants()` | merge_lane/worker.py | All §5.3 violations (I1–I4 + graph consistency) |
| `recompute_suffix_conflict_graph()` | merge_lane/worker.py | Worker delegator → `SuffixConflictTracker.recompute()` (suffix_graph.py); recomputes the conflict graph, triggers bounce |
| `_bounce_conflicting_suffix_items()` | merge_lane/worker.py | Worker delegator → `SuffixConflictTracker.bounce_conflicting_suffix_items()` (suffix_graph.py); graph-time disk-free bounce of the younger conflicting item |
| `_run_no_landings_breaker_pass()` | harness.py | Harness pass that acts on `BreakerTrip`: calls `force_halt_scheduler` + files L2-INFO escalation |
| `classify_and_merge()` | merge_lane/worker.py | Shared pre-merge guard + merge + drop-guard pipeline (branch-presence → already-merged → merge → conflict/non-conflict-failure → drop-guard), returning `MergedOk \| Decided`; `SpeculativeMergeWorker._merger_loop` and `SpeculativeMergeWorker._remerge` both delegate to it instead of each running its own duplicated inline copy (MQ-refactor task κ, task 1995) |
| `patch_content_contained()` | merge_lane/landing_evidence.py | Patch-id containment check: True iff every commit in `head` is already present in `upstream` (moved out of the worker, task 5036) |

### 7.1 merge_lane/types.py — request/outcome/item/entry types + registries (MQ-refactor task α)

The merge-queue data types and the registries that own them live in
`orchestrator/src/orchestrator/merge_lane/types.py` (task α of
`plans/merge-queue-modularization-invariants-prd.md`; moved from `merge_types.py` by task 5036).
Importers use `orchestrator.merge_lane.types`; the façade (`orchestrator/merge_lane/__init__.py`)
exports the public subset.

| Symbol | Location | Description |
|--------|----------|-------------|
| `MergeRequest` | merge_lane/types.py | A request to merge a task branch into main |
| `GroupMergeRequest` | merge_lane/types.py | `MergeRequest` subclass for an atomic linear-stacked train merge |
| `MergeOutcome` | merge_lane/types.py | Result delivered to the caller via the request's Future |
| `RealMergeItem` | merge_lane/types.py | REAL arm of the Merger→Verifier item union: a merge actually happened (`merge_result` + `merge_wt` required, no `immediate_outcome`) (MQ-refactor task ο, task 2000) |
| `DecidedItem` | merge_lane/types.py | DECIDED arm of the Merger→Verifier item union: a terminal `MergeOutcome` was already decided, delivered as a passthrough (`immediate_outcome` required, no `merge_result`/`merge_wt`) (MQ-refactor task ο, task 2000) |
| `SpeculativeItem` | merge_lane/types.py | `TypeAlias` for `RealMergeItem \| DecidedItem` — retained name for existing annotations/`isinstance` checks; no longer a constructible dataclass itself (MQ-refactor task ο, task 2000) |
| `item_merge_wt` | merge_lane/types.py | Helper returning the owned merge worktree for a `RealMergeItem` or `None` for a `DecidedItem`, via an `assert_never`-exhaustive match (MQ-refactor task ο, task 2000) |
| `MergedOk` | merge_lane/types.py | `classify_and_merge`'s REAL-arm return value (mirrors `SpeculativeItem`'s REAL/DECIDED split): `merge_result` + `merge_wt` + `branch_tip` for a merge that actually happened (MQ-refactor task κ, task 1995) |
| `Decided` | merge_lane/types.py | `classify_and_merge`'s DECIDED-arm return value: a terminal `MergeOutcome` (+ the failed `MergeResult`, when one was attempted) (MQ-refactor task κ, task 1995) |
| `InflightEntry` | merge_lane/types.py | An in-flight verify entry held in `SpeculativeMergeWorker._inflight` |
| `InflightVerifyResult` | merge_lane/types.py | Result returned by `SpeculativeMergeWorker._run_inflight_verify` |
| `SoloVerifyResult` | merge_lane/types.py | Result of verifying a single train member's delta in isolation |
| `WaiterRecord` | merge_lane/types.py | Server-side durable-intent waiter record keyed by `request_id` |
| `MergeDispatchResult` | merge_lane/types.py | Structured return value from `coalesce_or_enqueue_merge_request` |
| `InFlightMergeRegistry` (+ `_InFlightEntry`) | merge_lane/types.py | Per-branch in-flight de-dup registry and its slot record |
| `TerminalOutcomeRetention` (+ `TerminalOutcomeRecord`) | merge_lane/types.py | Bounded ring of recent terminal merge outcomes and its record type |
| `MergeBounceRegistry` | merge_lane/types.py | Monotonic per-branch bounce counter (η=1892 needs-rebase bounce cap) |
| `MainHealthAutoHealRegistry` | merge_lane/types.py | Monotonic per-signature attempt counter for main-health auto-heal |
| `TrainCallbacks` / `TrainCallbackFactory` | merge_lane/types.py | Scheduler-backed per-train callbacks and their factory type alias |
| `MergeReadyPredicate` | merge_lane/types.py | Type alias for the injectable merge-ready confidence-gate predicate (δ/1720) |
| `_HostUnavailability` | merge_lane/types.py | Per-host `RunnerUnavailable` streak tracker entry (task 1795) |
| `_INFLIGHT_MERGE_ETA_ESTIMATE_SECS` | merge_lane/types.py | Coarse ETA estimate (seconds) used by `InFlightMergeRegistry.eta_seconds` |

### 7.2 merge_lane/gates.py — post-merge gates + finalize + reason prefixes (MQ-refactor task β)

The pre-/post-merge gate functions, the advance-finalize and advance-failure-mapping
functions, the `merge_attempt` event emitter they share with the worker, and their supporting
types and gate-owned reason-prefix constants live in
`orchestrator/src/orchestrator/merge_lane/gates.py` (task β of
`plans/merge-queue-modularization-invariants-prd.md`; moved from `merge_gates.py` by task 5036).
Also here since task 5036 (formerly worker-resident): `AUTO_CHAIN_GENERATIONS_ENABLED`,
`_run_unscoped_typechecks` + `_POST_MERGE_PYRIGHT_MAX_DETAIL`, `_emit_merge_attempt`,
`_elapsed_ms` and `_MAX_EVENT_EVIDENCE_ITEMS`.

**Open Q1 (resolved NO):** verify *execution* — `_run_post_merge_verify`, `_ensure_verify_disk_space`,
`_classify_main_health_red`, `_verify_hit_enospc` — stays in `merge_lane/worker.py`. This module
owns gate *policy* only, and never imports the worker: where a gate needs a worker function the
worker injects it — the re-verify through `_reverify_rebased_tree`'s `run_post_merge_verify`
parameter, the γ2 auto-chain through `_GenerationChainContext`.

| Symbol | Location | Description |
|--------|----------|-------------|
| `DROPPED_PLAN_TARGETS_REASON_PREFIX` | merge_lane/gates.py | Reason prefix: drop-guard found branch work missing from the merge commit |
| `PLAN_FILES_NOT_TOUCHED_REASON_PREFIX` | merge_lane/gates.py | Reason prefix: pre-merge Decision-1 check found a declared plan file untouched by the branch |
| `POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX` | merge_lane/gates.py | Reason prefix: post-merge Decision-2 content-equivalence gate failed |
| `POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX` | merge_lane/gates.py | Reason prefix: post-merge unscoped type-check found a cross-PR union break |
| `DropGuardResult` | merge_lane/gates.py | Structured return value from `_check_plan_targets_in_tree` |
| `PlanFilesTouchedResult` | merge_lane/gates.py | Structured return value from `_check_plan_files_touched_in_branch` |
| `PostMergePyrightResult` | merge_lane/gates.py | Structured return value from `_check_post_merge_pyright` / `_run_unscoped_typechecks` |
| `_GenerationChainContext` | merge_lane/gates.py | Bundle passed into `_finalize_advanced_merge` for γ2 auto-chaining (queue + counters + retention) |
| `_OVERLAP_GIT_ERROR_SENTINEL` | merge_lane/gates.py | Fail-CLOSED sentinel returned by `_rebase_delta_touched_overlap` on a git error |
| `_check_plan_targets_in_tree()` | merge_lane/gates.py | Drop-guard: files on task HEAD but dropped from the merge commit |
| `_normalize_plan_path()` | merge_lane/gates.py | Git-canonical form of a declared plan path (helper of the plan-files-touched gate) |
| `_check_plan_files_touched_in_branch()` | merge_lane/gates.py | Pre-merge Decision-1: every declared plan file must be touched on the branch |
| `_check_post_merge_equivalence()` | merge_lane/gates.py | Post-merge Decision-2: branch-touched paths must match the advanced main tree |
| `_rebase_delta_touched_overlap()` | merge_lane/gates.py | Intersection of branch-touched and intervening-rebase-delta files (fail-closed) |
| `_reverify_rebased_tree()` | merge_lane/gates.py | Disjoint-delta re-verify gate; delegates to `_run_post_merge_verify` when overlapping |
| `_check_post_merge_pyright()` | merge_lane/gates.py | Post-merge Decision-3: unscoped package-wide type-check against the advanced main SHA |
| `_resolve_second_parent()` | merge_lane/gates.py | Second parent (`sha^2`) of a `--no-ff` merge commit, for equivalence-gate tip resolution |
| `_commit_is_linear()` | merge_lane/gates.py | True iff a commit has ≤1 parent (task-1928 worktree-HEAD-fallback fail-safe gate) |
| `_finalize_advanced_merge()` | merge_lane/gates.py | Post-advance success block: runs the equivalence + pyright gates, returns `MergeOutcome` |
| `_map_advance_failure()` | merge_lane/gates.py | `advance_main` failure-result → `MergeOutcome` mapping shared by both workers |
| `AUTO_CHAIN_GENERATIONS_ENABLED` | merge_lane/gates.py | γ2 auto-chain kill switch (module-level bool, default `False`); moved out of the worker, task 5036 |
| `_run_unscoped_typechecks()` | merge_lane/gates.py | Unscoped package-wide type-check runner behind `_check_post_merge_pyright`; moved out of the worker, task 5036 |
| `_POST_MERGE_PYRIGHT_MAX_DETAIL` | merge_lane/gates.py | Character cap on `PostMergePyrightResult.detail` |
| `_emit_merge_attempt()` | merge_lane/gates.py | `merge_attempt` event emitter shared by the gates and the worker; moved out of the worker, task 5036 |
| `_elapsed_ms()` | merge_lane/gates.py | Milliseconds since a `time.monotonic()` start value, or `None` |
| `_MAX_EVENT_EVIDENCE_ITEMS` | merge_lane/gates.py | Cap on evidence items carried on a `merge_attempt` event |

### 7.3 merge_lane/shadow.py — warm-vs-cold shadow-compare detective (MQ-refactor task γ)

The per-test result parsers, the persisted shadow-compare cadence state, and the warm-vs-cold
shadow-compare functions (PRD §10 invariant 6(b)) live in
`orchestrator/src/orchestrator/merge_lane/shadow.py` (task γ of
`plans/merge-queue-modularization-invariants-prd.md`; moved from `merge_shadow.py` by task 5036).
The module imports the verify-pool cluster (`build_merge_verify_spec` / `VerifyRunnerPool` /
`LocalRunner`) directly from `orchestrator/verify_runner.py` and the worker only under
`TYPE_CHECKING`, so there is no reach-back into the worker.

| Symbol | Location | Description |
|--------|----------|--------------|
| `ShadowCompareState` | merge_lane/shadow.py | Persisted cadence state (`merges_since_last_shadow`, `last_shadow_run_at`) |
| `ShadowCompareDiff` | merge_lane/shadow.py | Per-test divergence buckets between a warm and a cold verify run |
| `_NEXTEST_TEST_LINE_RE` | merge_lane/shadow.py | Regex matching cargo-nextest human-output per-test result lines |
| `_LIBTEST_TEST_LINE_RE` | merge_lane/shadow.py | Regex matching plain `cargo test` (libtest) per-test result lines |
| `_NEXTEST_SUMMARY_LINE_RE` | merge_lane/shadow.py | Regex matching the cargo-nextest `Summary [..] N tests run:` footer |
| `_classify_test_status()` | merge_lane/shadow.py | Map a raw nextest/libtest status token to `'pass'`/`'fail'`/`'inconclusive'` |
| `parse_per_test_results()` | merge_lane/shadow.py | Parse verify output into a per-test verdict map (nextest or libtest format) |
| `_nextest_reported_test_count()` | merge_lane/shadow.py | Sum of `N tests run:` counts across all Summary footer lines, or `None` |
| `diff_per_test_results()` | merge_lane/shadow.py | Compute the `ShadowCompareDiff` between a warm and a cold per-test result map |
| `_persistent_alarm_tests()` | merge_lane/shadow.py | Intersection of alarm-worthy test ids across two `ShadowCompareDiff`s (Option-B re-confirmation) |
| `_load_shadow_compare_state()` | merge_lane/shadow.py | Fail-safe JSON load of the persisted cadence state |
| `_save_shadow_compare_state()` | merge_lane/shadow.py | Persist the cadence state to JSON |
| `_shadow_compare_due()` | merge_lane/shadow.py | OR-cadence gate: every-N-merges leg OR nightly-timer leg |
| `_WARM_COLD_SHADOW_SENTINEL` | merge_lane/shadow.py | Dedup sentinel task_id for the divergence escalation |
| `_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL` | merge_lane/shadow.py | Dedup sentinel task_id for the fail-closed unparseable-format escalation |
| `_submit_shadow_divergence_escalation()` | merge_lane/shadow.py | Born-at-L2 critical escalation for a warm/cold divergence |
| `_alarm_warm_shadow_unparseable()` | merge_lane/shadow.py | Fail-closed born-at-L2 alarm when the warm verify output is unparseable despite tests having run |
| `_run_cold_shadow_verify()` | merge_lane/shadow.py | From-scratch cold verify of a landed merge commit in a throwaway worktree |
| `_run_shadow_compare()` | merge_lane/shadow.py | Detective control: cold-vs-warm compare with Option-B re-confirmation and alarm/parity-ok emission |
| `_maybe_schedule_shadow_compare()` | merge_lane/shadow.py | Non-blocking cadence-gated scheduler; spawns `_run_shadow_compare` off the serial lane |

### 7.4 merge_lane/drift.py — drift-check detective (MQ-refactor task γ)

The Lever-C drift-check runner and its land-hook cadence gate live in
`orchestrator/src/orchestrator/merge_lane/drift.py` (task γ; moved from `merge_drift.py` by task
5036), together with the module-level `_build_remote_runners` legacy-pool builder (formerly in the
worker).

**Correction to the step-3 plan prose:** despite both being off-serial-lane detective controls
spawned from the same `'done'`-land hook, `_run_drift_check` does **not** call
`_run_cold_shadow_verify` / `_run_shadow_compare` — drift-check and shadow-compare are
independent sibling detectives, not caller/callee. Both modules import the verify-pool cluster
(`build_merge_verify_spec` / `VerifyRunnerPool` / `LocalRunner`, from `orchestrator/verify_runner.py`)
directly; the worker is imported only under `TYPE_CHECKING`.

| Symbol | Location | Description |
|--------|----------|--------------|
| `_run_drift_check()` | merge_lane/drift.py | Drift detective: `DriftDetector.check` in a throwaway worktree against a 2-host (local + remote) pool |
| `_maybe_run_drift_check()` | merge_lane/drift.py | Cadence gate + off-serial-lane spawn, called immediately after `_maybe_schedule_shadow_compare` |
| `_build_remote_runners()` | merge_lane/drift.py | Builds the remote-only `RemoteRunner` list from operator config (Lever C); moved out of the worker, task 5036 |

### 7.5 merge_lane/liveness.py — startup liveness margin, verify-host alarms, persistent-worktree guards (MQ-refactor task γ)

**Open Q5 resolution:** three subsystems — the startup liveness-margin guard (heartbeat-floor
vs. reaper-window safety check), the verify-host-unreachable alarm/recovery helpers, and the
persistent warm-merge-verify-worktree serial-lane guards — are folded into one module as
"operational guards" rather than split into their own modules (plan.json design_decisions #1):
none is individually large, and all three gate/monitor worker-level operational health rather
than verify-parity detection (the shadow/drift detective family in `shadow.py` / `drift.py`).
The module is `orchestrator/src/orchestrator/merge_lane/liveness.py` (task 5036 moved it from
`merge_liveness.py`); every target in it is a SYNC function.

**Constants:** `TOUCH_MISS_TOLERANCE`, `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`, `_HEARTBEAT_POLL_S`
and `_MERGE_AHEAD_BOUND` are all defined in `liveness.py` itself; the worker imports them from
here. The module imports nothing from the worker.

**Engine-constant default-argument hazard:** `liveness_secs` (on `check_merge_liveness_margin` /
`enforce_merge_liveness_margin`) and `merge_ahead_bound` (on
`enforce_persistent_worktree_serial_lane`) default to a `None` sentinel, resolved in-body to
`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` / `_MERGE_AHEAD_BOUND` (effective defaults 10800 / 1) —
default values are evaluated at *def time*, so a bare-constant default would freeze the value a
test or config patches later.

| Symbol | Location | Description |
|--------|----------|--------------|
| `MergeLivenessAssessment` | merge_lane/liveness.py | Return value from `check_merge_liveness_margin` (heartbeat-floor vs. threshold, `safe` verdict) |
| `check_merge_liveness_margin()` | merge_lane/liveness.py | WARNING-only heartbeat-floor-vs-reaper-window assessment |
| `MergeLivenessConfigError` | merge_lane/liveness.py | Raised by `enforce_merge_liveness_margin` when the margin is unsafe |
| `enforce_merge_liveness_margin()` | merge_lane/liveness.py | Fail-closed wrapper: raises `MergeLivenessConfigError` when not safe |
| `PersistentWorktreeConfigError` | merge_lane/liveness.py | Raised by `enforce_persistent_worktree_serial_lane` when per-host in-flight count would exceed 1 |
| `_safety_valve_due()` | merge_lane/liveness.py | Periodic cold-verify safety-valve gate (every Nth verifying attempt, PRD §10 invariant 6) |
| `_VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX` | merge_lane/liveness.py | Per-host dedup sentinel prefix for unreachability alarms (task 1795) |
| `_VERIFY_HOST_RECOVERED_SENTINEL_PREFIX` | merge_lane/liveness.py | Per-host sentinel prefix for recovery info escalations, distinct from the unreachable prefix |
| `_MERGE_WORKER_LOOP_DIED_SENTINEL` | merge_lane/liveness.py | Sentinel task_id base for the merge-worker supervisor loop-death escalation |
| `_verify_host_unreachable_sentinel()` | merge_lane/liveness.py | Per-host dedup sentinel task_id for unreachability alarms |
| `_alarm_verify_host_unreachable()` | merge_lane/liveness.py | Dedup'd L1 escalation when a remote verify host is persistently unreachable |
| `_clear_verify_host_unreachable()` | merge_lane/liveness.py | Resolve any open unreachability alarm and emit a recovery event on reprobe success |
| `_acquire_warm_verify_worktree()` | merge_lane/liveness.py | Swap the ephemeral merge worktree for the persistent warm worktree (or the `_spec-` warm lane) |
| `enforce_persistent_worktree_serial_lane()` | merge_lane/liveness.py | Fail-closed startup guard: per-host in-flight verify count must not exceed 1 |
| `TOUCH_MISS_TOLERANCE` | merge_lane/liveness.py | Consecutive heartbeat ticks a live worker's `_merge-*` worktrees may miss before mtime ages into the reaper window (= 20); moved out of the worker, task 5036 |
| `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` | merge_lane/liveness.py | The reaper's liveness window in seconds (= 10800); moved out of the worker, task 5036 |
| `_HEARTBEAT_POLL_S` | merge_lane/liveness.py | How often the worker heartbeat loop wakes (30.0 s); moved out of the worker, task 5036 |
| `_MERGE_AHEAD_BOUND` | merge_lane/liveness.py | Max counted (non-speculative, non-train) items in the verifier queue at once (= 1; Mechanism 1, task 1646); moved out of the worker, task 5036 |

### 7.6 suffix_graph.py — SuffixConflictTracker (conflict graph + bounce state) (MQ-refactor task δ)

The two-layer suffix-conflict machinery — the `SuffixConflictGraph` immutable conflict-graph
dataclass and its `EMPTY_SUFFIX_CONFLICT_GRAPH` sentinel — were extracted verbatim, and a NEW
`SuffixConflictTracker` class that owns the state (`graph` / `signature` / `last_known_main_sha` /
`bounce_registry`) and logic (`recompute()` / `bounce_conflicting_suffix_items()`) was added, into
`orchestrator/src/orchestrator/suffix_graph.py` (task δ of `plans/merge-queue-modularization-invariants-prd.md`).
This module stayed OUTSIDE the `orchestrator/merge_lane/` package (task 5036).

Unlike α–γ's pure function/type extractions, this module also introduces a NEW owning class.
`SuffixConflictTracker` takes a live `GitOps` reference plus three narrow accessor callables —
`lane_buffers`, `frozen_prefix`, `frozen_prefix_tip` — instead of a worker reference, so it is
fully unit-testable without a `SpeculativeMergeWorker`. `SpeculativeMergeWorker` owns exactly one
instance (`self._suffix_tracker`, constructed immediately after `self._lane_buffers` in
`__init__`) and delegates to it via 4 get/set `@property` descriptors that preserve the worker's
original attribute names (`_suffix_conflict_graph`, `_suffix_conflict_signature`,
`_last_known_main_sha`, `_bounce_registry`) plus two thin async methods
(`recompute_suffix_conflict_graph()`, `_bounce_conflicting_suffix_items()`) that just `await` the
tracker — so `_acquire_next_request()`, `snapshot()`, `_pop_next_pickable()`,
`two_layer_invariants()`, and the existing conflict-graph/bounce test suites all keep working with
zero churn.

**Reach-back convention:** the two tracker methods resolve the three worker-resident constants
they read (`MERGE_LANES`, `MERGE_BOUNCE_CAP`, `NEEDS_REBASE_REASON_PREFIX`, all in
`merge_lane/worker.py`) through a function-local deferred `from orchestrator.merge_queue import
<name>` import (the `orchestrator.merge_queue` alias, until task 5037 re-points it at
`orchestrator.merge_lane.worker`) rather than a top-level import, keeping `suffix_graph.py` free of
any top-level import of the worker (which would be a cycle, since the worker imports this module).

| Symbol | Location | Description |
|--------|----------|--------------|
| `SuffixConflictGraph` | suffix_graph.py | Immutable conflict graph over the unfrozen suffix (moved verbatim from the monolith, now `merge_lane/worker.py`) |
| `EMPTY_SUFFIX_CONFLICT_GRAPH` | suffix_graph.py | Sentinel empty `SuffixConflictGraph` for the default/zero-suffix case (moved verbatim) |
| `SuffixConflictTracker` | suffix_graph.py | Owns `graph` / `signature` / `last_known_main_sha` / `bounce_registry`; constructed with `git_ops` + `lane_buffers`/`frozen_prefix`/`frozen_prefix_tip` callables |
| `SuffixConflictTracker.recompute()` | suffix_graph.py | Recompute and store the conflict graph over the unfrozen suffix (debounced, fail-open); `SpeculativeMergeWorker.recompute_suffix_conflict_graph()` delegates here |
| `SuffixConflictTracker.bounce_conflicting_suffix_items()` | suffix_graph.py | Graph-time disk-free bounce of the younger conflicting item (cap/escalation/TOCTOU); `SpeculativeMergeWorker._bounce_conflicting_suffix_items()` delegates here |

### 7.7 MergeWorker retirement — one merge worker, in production and in the tests (MQ-refactor task ν, merge-lane-quality δ)

The legacy serial `MergeWorker` — the single-coroutine worker with no lane-priority ordering
that predated the two-layer model — was retired from `orchestrator/src/orchestrator/merge_lane/worker.py` (then `merge_queue.py`)
by task ν of `plans/merge-queue-modularization-invariants-prd.md` (R7b). `SpeculativeMergeWorker`,
exported as `orchestrator.merge_lane.MergeLane`, is the sole merge worker.

R7b kept a frozen test-local copy of the serial class so the ~89 tests built on it kept running.
Task 5034 (`plans/merge-lane-quality-prd.md` task δ, decision 4: a fallback that is never
exercised is not a fallback) discarded that copy and re-homed the behaviours those tests checked
onto the production lane, driven through `orchestrator/tests/_merge_lane_fakes.py`
(`make_lane`, `merge_through_lane`, `FakeVerifier`). Behaviour that only the serial copy had —
the `_urgent` front-of-queue CAS re-enqueue, `_dequeue`/`_process` — went with it.
`orchestrator/tests/test_merge_worker_retired.py` pins that no `MergeWorker` class is defined in
production or anywhere under `orchestrator/tests`.

The shared pipeline pieces the serial worker used were never worker-specific, and remain:
`classify_and_merge()` (§7, task κ/1995), `_do_train_merge()` with its `_TrainMergeHost`
Protocol (the narrow statement of what the train pipeline touches), `_run_post_merge_verify()`,
`_finalize_advanced_merge()`, `_map_advance_failure()`, `_emit_merge_queued()`,
`_HALT_ADVANCE_RESULTS` and `_WipHaltMixin`.

---

## 8. Greek→task-ID provenance table

| Greek | Task | Mechanism delivered |
|-------|------|---------------------|
| α | 1886 | `merge_first_enqueued_at` — write-once first-submission timestamp (aging priority, survives restart) |
| δ | 1889 | `SuffixConflictGraph` — in-memory conflict graph over the unfrozen suffix |
| ε | 1890 | Frozen-prefix / verify-frontier partition (`frozen_prefix()`, `frozen_prefix_tip()`, `check_frozen_prefix_invariant()`) |
| ζ | 1891 | Clique-scoped aging comparator (`_aging_key`, `_pop_next_pickable`) |
| η | 1892 | `needs_rebase` graph-time bounce (`_bounce_conflicting_suffix_items`, `NEEDS_REBASE_REASON_PREFIX`, `MERGE_BOUNCE_CAP`) |
| θ | 1893 | `NoLandingsCircuitBreaker` — no-landings auto-halt + L2-info escalation + auto-resume |
| ι | 1894 | Operator metrics (`retries_per_landing`, `drift_at_detection`, `landings_total`) |
| λ | 1895 | Integration gate + `two_layer_invariants()` — §5.3 invariant health surface |
| μ | 1896 | This documentation task (SKILL.md + design-doc companion) |
| ν | 1897 | Follow-on task (post-integration cleanup / companion) |

---

## 9. Gain model summary

The two-layer pipeline reduces the **loop gain G** of the merge-churn feedback spiral:

- **`needs_rebase` disk-free bounce** (η): conflicts caught at graph time before consuming a verify slot, reducing Δp (the per-failure coupling term).
- **Age-of-first-submission ordering** (ζ): the oldest, most-expensive-to-redo task gets priority within a conflict clique, improving p′ (retry success probability).
- **Frozen-prefix immutability** (ε): an in-flight verify is never disrupted, preventing wasted verify-slot churn.
- **No-landings circuit-breaker** (θ): stops the spiral before ENOSPC by halting dispatch when landing-rate ≈ 0 AND disk is falling.

Together: G < 1 (self-damping) under normal operating conditions.

---

## 10. Related work

- **Warm-lane Δp space-safety batch (1859–1861 / reify 4716–4719):** attacks Δp on the *task-dispatch* path (warm-lane disk-space gates before a task is dispatched).  Complementary to the merge-queue-path Δp attacked here; no shared seam between the two mitigations.
- **Merge-verify ENOSPC fail-soft (workflow.py `TRANSIENT_INFRA_REASON_PREFIX` branch → re-queue):** handles ENOSPC at the individual verify step by re-queuing as a transient infra failure.  This is a **separate symptom task** and is explicitly **out of scope** for the two-layer merge queue (PRD §10).  Referenced here for orientation; see the `TRANSIENT_INFRA_REASON_PREFIX` short-circuit branch in `workflow.py` for the implementation.
