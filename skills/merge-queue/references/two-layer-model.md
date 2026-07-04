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

---

## 7. As-built symbol map

| Symbol | Location | Description |
|--------|----------|-------------|
| `NEEDS_REBASE_REASON_PREFIX` | merge_queue.py | Prefix of the `needs_rebase` bounce reason string |
| `MERGE_BOUNCE_CAP` | merge_queue.py | Max bounce count before thrash-backstop triggers (= 3) |
| `_aging_key(req)` | merge_queue.py | Sort key: `(merge_first_enqueued_at or enqueued_at, request_id)` |
| `merge_first_enqueued_at` | merge_queue.py | Write-once field: epoch of first submission to the merge queue |
| `SuffixConflictGraph` | suffix_graph.py | In-memory conflict graph over the unfrozen suffix |
| `NoLandingsCircuitBreaker` | merge_queue.py | No-landings circuit-breaker decision object (θ=1893) |
| `_pop_next_pickable()` | merge_queue.py | Select next item using clique-scoped aging (ζ=1891) |
| `frozen_prefix()` | merge_queue.py | Return ordered `request_id`s in the frozen prefix |
| `frozen_prefix_tip()` | merge_queue.py | Return the base SHA for next verify dispatch |
| `check_frozen_prefix_invariant()` | merge_queue.py | §5.3 I1+I2 violations (base-chain integrity) |
| `two_layer_invariants()` | merge_queue.py | All §5.3 violations (I1–I4 + graph consistency) |
| `recompute_suffix_conflict_graph()` | merge_queue.py | Worker delegator → `SuffixConflictTracker.recompute()` (suffix_graph.py); recomputes the conflict graph, triggers bounce |
| `_bounce_conflicting_suffix_items()` | merge_queue.py | Worker delegator → `SuffixConflictTracker.bounce_conflicting_suffix_items()` (suffix_graph.py); graph-time disk-free bounce of the younger conflicting item |
| `_run_no_landings_breaker_pass()` | harness.py | Harness pass that acts on `BreakerTrip`: calls `force_halt_scheduler` + files L2-INFO escalation |
| `classify_and_merge()` | merge_queue.py | Shared pre-merge guard + merge + drop-guard pipeline (branch-presence → already-merged → merge → conflict/non-conflict-failure → drop-guard), returning `MergedOk \| Decided`; `MergeWorker._do_merge`, `SpeculativeMergeWorker._merger_loop`, and `SpeculativeMergeWorker._remerge` all delegate to it instead of each running its own duplicated inline copy (MQ-refactor task κ, task 1995) |

### 7.1 merge_types.py — request/outcome/item/entry types + registries (MQ-refactor task α)

The merge-queue data types and the registries that own them were extracted verbatim into
`orchestrator/merge_types.py` (task α of `plans/merge-queue-modularization-invariants-prd.md`).
`merge_queue.py` re-exports every one of these names through a single top-level shim import
(`from orchestrator.merge_types import (...)  # noqa: F401  re-export shim`), so existing call
sites — `from orchestrator.merge_queue import MergeRequest`, etc. — keep working unchanged.

| Symbol | Location | Description |
|--------|----------|-------------|
| `MergeRequest` | merge_types.py | A request to merge a task branch into main |
| `GroupMergeRequest` | merge_types.py | `MergeRequest` subclass for an atomic linear-stacked train merge |
| `MergeOutcome` | merge_types.py | Result delivered to the caller via the request's Future |
| `SpeculativeItem` | merge_types.py | Internal message from the Merger coroutine to the Verifier coroutine |
| `MergedOk` | merge_types.py | `classify_and_merge`'s REAL-arm return value (mirrors `SpeculativeItem`'s REAL/DECIDED split): `merge_result` + `merge_wt` + `branch_tip` for a merge that actually happened (MQ-refactor task κ, task 1995) |
| `Decided` | merge_types.py | `classify_and_merge`'s DECIDED-arm return value: a terminal `MergeOutcome` (+ the failed `MergeResult`, when one was attempted) (MQ-refactor task κ, task 1995) |
| `InflightEntry` | merge_types.py | An in-flight verify entry held in `SpeculativeMergeWorker._inflight` |
| `InflightVerifyResult` | merge_types.py | Result returned by `SpeculativeMergeWorker._run_inflight_verify` |
| `SoloVerifyResult` | merge_types.py | Result of verifying a single train member's delta in isolation |
| `WaiterRecord` | merge_types.py | Server-side durable-intent waiter record keyed by `request_id` |
| `MergeDispatchResult` | merge_types.py | Structured return value from `coalesce_or_enqueue_merge_request` |
| `InFlightMergeRegistry` (+ `_InFlightEntry`) | merge_types.py | Per-branch in-flight de-dup registry and its slot record |
| `TerminalOutcomeRetention` (+ `TerminalOutcomeRecord`) | merge_types.py | Bounded ring of recent terminal merge outcomes and its record type |
| `MergeBounceRegistry` | merge_types.py | Monotonic per-branch bounce counter (η=1892 needs-rebase bounce cap) |
| `MainHealthAutoHealRegistry` | merge_types.py | Monotonic per-signature attempt counter for main-health auto-heal |
| `TrainCallbacks` / `TrainCallbackFactory` | merge_types.py | Scheduler-backed per-train callbacks and their factory type alias |
| `MergeReadyPredicate` | merge_types.py | Type alias for the injectable merge-ready confidence-gate predicate (δ/1720) |
| `_HostUnavailability` | merge_types.py | Per-host `RunnerUnavailable` streak tracker entry (task 1795) |
| `_INFLIGHT_MERGE_ETA_ESTIMATE_SECS` | merge_types.py | Coarse ETA estimate (seconds) used by `InFlightMergeRegistry.eta_seconds` |

### 7.2 merge_gates.py — post-merge gates + finalize + reason prefixes (MQ-refactor task β)

The pre-/post-merge gate functions, the advance-finalize and advance-failure-mapping
functions, and their supporting types and gate-owned reason-prefix constants were
extracted verbatim into `orchestrator/merge_gates.py` (task β of
`plans/merge-queue-modularization-invariants-prd.md`). `merge_queue.py` re-exports every
one of these names through a single top-level shim import (`from orchestrator.merge_gates
import (...)  # noqa: F401  re-export shim`), so existing call sites —
`from orchestrator.merge_queue import _finalize_advanced_merge`, etc. — keep working
unchanged.

**Open Q1 (resolved NO):** verify *execution* — `_run_post_merge_verify` and its cluster
(`_run_unscoped_typechecks` + `_POST_MERGE_PYRIGHT_MAX_DETAIL`, `_ensure_verify_disk_space`,
`_classify_main_health_red`, `_verify_hit_enospc`) — stays in `merge_queue.py`. This module
owns gate *policy* only. `_run_unscoped_typechecks` staying is decisive for behavior
preservation: the equivalence/pyright tests patch `orchestrator.merge_queue.run_verification`
(the dependency beneath it), and because it stays in `merge_queue.py` it resolves
`run_verification` in `merge_queue`'s own namespace — no reach-back needed for that
specific patch chain.

**Reach-back convention (for future γ/δ extractors):** a moved function that calls a
merge_queue-resident sibling — whether that sibling stays permanently or was co-moved here
but is monkeypatched by the existing test suite via the string path
`orchestrator.merge_queue.<name>` — resolves it through a function-local (deferred)
`from orchestrator.merge_queue import <name>` import rather than a direct intra-module
reference. This mirrors the pre-existing `_main_health_fingerprint` convention
(`merge_queue.py`) and keeps `merge_gates.py` free of any top-level import of
`merge_queue` (which would deadlock module load, since merge_queue's shim needs this
module fully defined first). Counterintuitively, this applies even to calls between two
functions that *both* moved here (e.g. `_finalize_advanced_merge` →
`_check_post_merge_pyright`), because the patch target the test suite uses is the
`merge_queue` shim binding, not the `merge_gates` definition. Watch also for **bare
module-level constants** referenced inside a moved function body (not just callables) —
`_finalize_advanced_merge`'s `AUTO_CHAIN_GENERATIONS_ENABLED` kill-switch check needed the
same deferred-import treatment even though it is a plain `bool`, not a function.

| Symbol | Location | Description |
|--------|----------|-------------|
| `DROPPED_PLAN_TARGETS_REASON_PREFIX` | merge_gates.py | Reason prefix: drop-guard found branch work missing from the merge commit |
| `PLAN_FILES_NOT_TOUCHED_REASON_PREFIX` | merge_gates.py | Reason prefix: pre-merge Decision-1 check found a declared plan file untouched by the branch |
| `POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX` | merge_gates.py | Reason prefix: post-merge Decision-2 content-equivalence gate failed |
| `POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX` | merge_gates.py | Reason prefix: post-merge unscoped type-check found a cross-PR union break |
| `DropGuardResult` | merge_gates.py | Structured return value from `_check_plan_targets_in_tree` |
| `PlanFilesTouchedResult` | merge_gates.py | Structured return value from `_check_plan_files_touched_in_branch` |
| `PostMergePyrightResult` | merge_gates.py | Structured return value from `_check_post_merge_pyright` / `_run_unscoped_typechecks` |
| `_GenerationChainContext` | merge_gates.py | Bundle passed into `_finalize_advanced_merge` for γ2 auto-chaining (queue + counters + retention) |
| `_OVERLAP_GIT_ERROR_SENTINEL` | merge_gates.py | Fail-CLOSED sentinel returned by `_rebase_delta_touched_overlap` on a git error |
| `_check_plan_targets_in_tree()` | merge_gates.py | Drop-guard: files on task HEAD but dropped from the merge commit |
| `_normalize_plan_path()` | merge_gates.py | Git-canonical form of a declared plan path (helper of the plan-files-touched gate) |
| `_check_plan_files_touched_in_branch()` | merge_gates.py | Pre-merge Decision-1: every declared plan file must be touched on the branch |
| `_check_post_merge_equivalence()` | merge_gates.py | Post-merge Decision-2: branch-touched paths must match the advanced main tree |
| `_rebase_delta_touched_overlap()` | merge_gates.py | Intersection of branch-touched and intervening-rebase-delta files (fail-closed) |
| `_reverify_rebased_tree()` | merge_gates.py | Disjoint-delta re-verify gate; delegates to `_run_post_merge_verify` when overlapping |
| `_check_post_merge_pyright()` | merge_gates.py | Post-merge Decision-3: unscoped package-wide type-check against the advanced main SHA |
| `_resolve_second_parent()` | merge_gates.py | Second parent (`sha^2`) of a `--no-ff` merge commit, for equivalence-gate tip resolution |
| `_commit_is_linear()` | merge_gates.py | True iff a commit has ≤1 parent (task-1928 worktree-HEAD-fallback fail-safe gate) |
| `_finalize_advanced_merge()` | merge_gates.py | Post-advance success block: runs the equivalence + pyright gates, returns `MergeOutcome` |
| `_map_advance_failure()` | merge_gates.py | `advance_main` failure-result → `MergeOutcome` mapping shared by both workers |

### 7.3 merge_shadow.py — warm-vs-cold shadow-compare detective (MQ-refactor task γ)

The per-test result parsers, the persisted shadow-compare cadence state, and the warm-vs-cold
shadow-compare functions (PRD §10 invariant 6(b)) were extracted verbatim into
`orchestrator/merge_shadow.py` (task γ of `plans/merge-queue-modularization-invariants-prd.md`).
`merge_queue.py` re-exports every one of these names through a single top-level shim import
(`from orchestrator.merge_shadow import (...)  # noqa: F401  re-export shim`), so existing call
sites — `from orchestrator.merge_queue import _run_shadow_compare`, etc. — keep working
unchanged.

**Reach-back convention:** identical in spirit to β's (§7.2) — a moved function that calls a
merge_queue-resident sibling, whether that sibling stays permanently
(`_run_unscoped_typechecks`) or is monkeypatched by the existing test suite via the string path
`orchestrator.merge_queue.<name>`, resolves it through a function-local deferred import from
`orchestrator.merge_queue`. Two distinct import styles are used, depending on whether the
target also carries a module-level "naive" import: `run_scoped_verification`,
`build_merge_verify_spec`, `VerifyRunnerPool`, and `LocalRunner` each have one (kept solely as a
`TestReachBackRouting` patch target), so `_run_cold_shadow_verify` reaches back to them — and to
the permanently-staying `_run_unscoped_typechecks`, for consistency — via
`import orchestrator.merge_queue as _mq` + `_mq.<name>` attribute access; a `from ... import`
reach-back would instead shadow the naive import and trip ruff's F811. `_run_cold_shadow_verify`
and `_run_shadow_compare` themselves carry no naive top-level import (they are local
definitions, not re-imported leaf symbols), so their respective callers use a plain
`from orchestrator.merge_queue import <name>` instead: `_run_shadow_compare` reaches back to
`_run_cold_shadow_verify` this way for both the initial and the Option-B re-confirmation cold
leg; `_maybe_schedule_shadow_compare` reaches back to `_run_shadow_compare` the same way when
spawning the off-serial-lane task.

| Symbol | Location | Description |
|--------|----------|--------------|
| `ShadowCompareState` | merge_shadow.py | Persisted cadence state (`merges_since_last_shadow`, `last_shadow_run_at`) |
| `ShadowCompareDiff` | merge_shadow.py | Per-test divergence buckets between a warm and a cold verify run |
| `_NEXTEST_TEST_LINE_RE` | merge_shadow.py | Regex matching cargo-nextest human-output per-test result lines |
| `_LIBTEST_TEST_LINE_RE` | merge_shadow.py | Regex matching plain `cargo test` (libtest) per-test result lines |
| `_NEXTEST_SUMMARY_LINE_RE` | merge_shadow.py | Regex matching the cargo-nextest `Summary [..] N tests run:` footer |
| `_classify_test_status()` | merge_shadow.py | Map a raw nextest/libtest status token to `'pass'`/`'fail'`/`'inconclusive'` |
| `parse_per_test_results()` | merge_shadow.py | Parse verify output into a per-test verdict map (nextest or libtest format) |
| `_nextest_reported_test_count()` | merge_shadow.py | Sum of `N tests run:` counts across all Summary footer lines, or `None` |
| `diff_per_test_results()` | merge_shadow.py | Compute the `ShadowCompareDiff` between a warm and a cold per-test result map |
| `_persistent_alarm_tests()` | merge_shadow.py | Intersection of alarm-worthy test ids across two `ShadowCompareDiff`s (Option-B re-confirmation) |
| `_load_shadow_compare_state()` | merge_shadow.py | Fail-safe JSON load of the persisted cadence state |
| `_save_shadow_compare_state()` | merge_shadow.py | Persist the cadence state to JSON |
| `_shadow_compare_due()` | merge_shadow.py | OR-cadence gate: every-N-merges leg OR nightly-timer leg |
| `_WARM_COLD_SHADOW_SENTINEL` | merge_shadow.py | Dedup sentinel task_id for the divergence escalation |
| `_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL` | merge_shadow.py | Dedup sentinel task_id for the fail-closed unparseable-format escalation |
| `_submit_shadow_divergence_escalation()` | merge_shadow.py | Born-at-L2 critical escalation for a warm/cold divergence |
| `_alarm_warm_shadow_unparseable()` | merge_shadow.py | Fail-closed born-at-L2 alarm when the warm verify output is unparseable despite tests having run |
| `_run_cold_shadow_verify()` | merge_shadow.py | From-scratch cold verify of a landed merge commit in a throwaway worktree |
| `_run_shadow_compare()` | merge_shadow.py | Detective control: cold-vs-warm compare with Option-B re-confirmation and alarm/parity-ok emission |
| `_maybe_schedule_shadow_compare()` | merge_shadow.py | Non-blocking cadence-gated scheduler; spawns `_run_shadow_compare` off the serial lane |

### 7.4 merge_drift.py — drift-check detective (MQ-refactor task γ)

The Lever-C drift-check runner and its land-hook cadence gate were extracted verbatim into
`orchestrator/merge_drift.py` (task γ). `merge_queue.py` re-exports both names through a
top-level shim import (`from orchestrator.merge_drift import (...)  # noqa: F401  re-export
shim`), so existing call sites — `from orchestrator.merge_queue import _run_drift_check`, etc. —
keep working unchanged.

**Correction to the step-3 plan prose:** despite both being off-serial-lane detective controls
spawned from the same `'done'`-land hook, `_run_drift_check` does **not** call
`_run_cold_shadow_verify` / `_run_shadow_compare` — drift-check and shadow-compare are
independent sibling detectives, not caller/callee. The reach-back cluster this module actually
needs is the verify-pool-construction cluster it shares with merge_shadow.py:
`build_merge_verify_spec` / `VerifyRunnerPool` / `LocalRunner` / `run_scoped_verification` (plus
the staying `_run_unscoped_typechecks` and the module-level `_build_remote_runners` legacy-pool
builder), all resolved via the same `import orchestrator.merge_queue as _mq` attribute-access
reach-back as merge_shadow.py, for the same F811-avoidance reason. `_maybe_run_drift_check`
separately reaches back to `_run_drift_check` via `from orchestrator.merge_queue import
_run_drift_check` when spawning the off-serial-lane task.

| Symbol | Location | Description |
|--------|----------|--------------|
| `_run_drift_check()` | merge_drift.py | Drift detective: `DriftDetector.check` in a throwaway worktree against a 2-host (local + remote) pool |
| `_maybe_run_drift_check()` | merge_drift.py | Cadence gate + off-serial-lane spawn, called immediately after `_maybe_schedule_shadow_compare` |

### 7.5 merge_liveness.py — startup liveness margin, verify-host alarms, persistent-worktree guards (MQ-refactor task γ)

**Open Q5 resolution:** three subsystems — the startup liveness-margin guard (heartbeat-floor
vs. reaper-window safety check), the verify-host-unreachable alarm/recovery helpers, and the
persistent warm-merge-verify-worktree serial-lane guards — are folded into one module as
"operational guards" rather than split into their own modules (plan.json design_decisions #1):
none is individually large, and all three gate/monitor worker-level operational health rather
than verify-parity detection (the shadow/drift detective family in merge_shadow.py /
merge_drift.py). `merge_queue.py` re-exports every one of these names through a single
top-level shim import (`from orchestrator.merge_liveness import (...)  # noqa: F401
re-export shim`), so existing call sites — `from orchestrator.merge_queue import
enforce_merge_liveness_margin`, etc. — keep working unchanged.

**Reach-back convention:** a moved function that reads or calls a merge_queue-resident
sibling — whether a function (`check_merge_liveness_margin`) or a module-level CONSTANT
(`_HEARTBEAT_POLL_S`, `TOUCH_MISS_TOLERANCE`, `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`,
`_MERGE_AHEAD_BOUND`) that stays in `merge_queue.py` and is monkeypatched by the existing test
suite via the string path `orchestrator.merge_queue.<name>` — resolves it through a
function-local deferred `from orchestrator.merge_queue import <name>` rather than a direct
intra-module reference. Every target in this module is a SYNC function.

**Engine-constant default-argument hazard:** `liveness_secs` (on `check_merge_liveness_margin` /
`enforce_merge_liveness_margin`) and `merge_ahead_bound` (on
`enforce_persistent_worktree_serial_lane`) used to default to a bare merge_queue-resident
constant (`INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` / `_MERGE_AHEAD_BOUND`). Default values are
evaluated at *def time* (module import), so a moved function cannot default to a
merge_queue-resident constant without a top-level `import orchestrator.merge_queue` — which
would deadlock module load, since merge_queue's shim needs this module fully defined first.
Each default is therefore a `None` sentinel, resolved in-body via the same deferred-import
reach-back — behavior-preserving (identical effective defaults: 10800 / 1); only the signature
default literal changed (to `None`).

| Symbol | Location | Description |
|--------|----------|--------------|
| `MergeLivenessAssessment` | merge_liveness.py | Return value from `check_merge_liveness_margin` (heartbeat-floor vs. threshold, `safe` verdict) |
| `check_merge_liveness_margin()` | merge_liveness.py | WARNING-only heartbeat-floor-vs-reaper-window assessment |
| `MergeLivenessConfigError` | merge_liveness.py | Raised by `enforce_merge_liveness_margin` when the margin is unsafe |
| `enforce_merge_liveness_margin()` | merge_liveness.py | Fail-closed wrapper: raises `MergeLivenessConfigError` when not safe |
| `PersistentWorktreeConfigError` | merge_liveness.py | Raised by `enforce_persistent_worktree_serial_lane` when per-host in-flight count would exceed 1 |
| `_safety_valve_due()` | merge_liveness.py | Periodic cold-verify safety-valve gate (every Nth verifying attempt, PRD §10 invariant 6) |
| `_VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX` | merge_liveness.py | Per-host dedup sentinel prefix for unreachability alarms (task 1795) |
| `_VERIFY_HOST_RECOVERED_SENTINEL_PREFIX` | merge_liveness.py | Per-host sentinel prefix for recovery info escalations, distinct from the unreachable prefix |
| `_MERGE_WORKER_LOOP_DIED_SENTINEL` | merge_liveness.py | Sentinel task_id base for the merge-worker supervisor loop-death escalation |
| `_verify_host_unreachable_sentinel()` | merge_liveness.py | Per-host dedup sentinel task_id for unreachability alarms |
| `_alarm_verify_host_unreachable()` | merge_liveness.py | Dedup'd L1 escalation when a remote verify host is persistently unreachable |
| `_clear_verify_host_unreachable()` | merge_liveness.py | Resolve any open unreachability alarm and emit a recovery event on reprobe success |
| `_acquire_warm_verify_worktree()` | merge_liveness.py | Swap the ephemeral merge worktree for the persistent warm worktree (or the `_spec-` warm lane) |
| `enforce_persistent_worktree_serial_lane()` | merge_liveness.py | Fail-closed startup guard: per-host in-flight verify count must not exceed 1 |

### 7.6 suffix_graph.py — SuffixConflictTracker (conflict graph + bounce state) (MQ-refactor task δ)

The two-layer suffix-conflict machinery — the `SuffixConflictGraph` immutable conflict-graph
dataclass and its `EMPTY_SUFFIX_CONFLICT_GRAPH` sentinel — were extracted verbatim, and a NEW
`SuffixConflictTracker` class that owns the state (`graph` / `signature` / `last_known_main_sha` /
`bounce_registry`) and logic (`recompute()` / `bounce_conflicting_suffix_items()`) was added, into
`orchestrator/suffix_graph.py` (task δ of `plans/merge-queue-modularization-invariants-prd.md`).
`merge_queue.py` re-exports all three names through a single top-level shim import (`from
orchestrator.suffix_graph import (...)  # noqa: F401  re-export shim`), so existing call sites —
`from orchestrator.merge_queue import SuffixConflictGraph`, etc. — keep working unchanged.

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

**Reach-back convention:** identical in spirit to β/γ/λ (§7.2/7.3/7.5) — the two tracker methods
resolve the three merge_queue-resident constants they read (`MERGE_LANES`, `MERGE_BOUNCE_CAP`,
`NEEDS_REBASE_REASON_PREFIX`) through function-local deferred `from orchestrator.merge_queue
import <name>` imports rather than a top-level import, keeping `suffix_graph.py` free of any
top-level import of `merge_queue` (which would deadlock module load, since merge_queue's shim
needs this module fully defined first). None of the three constants were moved — they stay in
`merge_queue.py`.

| Symbol | Location | Description |
|--------|----------|--------------|
| `SuffixConflictGraph` | suffix_graph.py | Immutable conflict graph over the unfrozen suffix (moved verbatim from merge_queue.py) |
| `EMPTY_SUFFIX_CONFLICT_GRAPH` | suffix_graph.py | Sentinel empty `SuffixConflictGraph` for the default/zero-suffix case (moved verbatim) |
| `SuffixConflictTracker` | suffix_graph.py | Owns `graph` / `signature` / `last_known_main_sha` / `bounce_registry`; constructed with `git_ops` + `lane_buffers`/`frozen_prefix`/`frozen_prefix_tip` callables |
| `SuffixConflictTracker.recompute()` | suffix_graph.py | Recompute and store the conflict graph over the unfrozen suffix (debounced, fail-open); `SpeculativeMergeWorker.recompute_suffix_conflict_graph()` delegates here |
| `SuffixConflictTracker.bounce_conflicting_suffix_items()` | suffix_graph.py | Graph-time disk-free bounce of the younger conflicting item (cap/escalation/TOCTOU); `SpeculativeMergeWorker._bounce_conflicting_suffix_items()` delegates here |

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
