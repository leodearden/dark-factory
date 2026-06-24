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
| `SuffixConflictGraph` | merge_queue.py | In-memory conflict graph over the unfrozen suffix |
| `NoLandingsCircuitBreaker` | merge_queue.py | No-landings circuit-breaker decision object (θ=1893) |
| `_pop_next_pickable()` | merge_queue.py | Select next item using clique-scoped aging (ζ=1891) |
| `frozen_prefix()` | merge_queue.py | Return ordered `request_id`s in the frozen prefix |
| `frozen_prefix_tip()` | merge_queue.py | Return the base SHA for next verify dispatch |
| `check_frozen_prefix_invariant()` | merge_queue.py | §5.3 I1+I2 violations (base-chain integrity) |
| `two_layer_invariants()` | merge_queue.py | All §5.3 violations (I1–I4 + graph consistency) |
| `recompute_suffix_conflict_graph()` | merge_queue.py | Recompute the conflict graph; triggers bounce |
| `_bounce_conflicting_suffix_items()` | merge_queue.py | Graph-time disk-free bounce of the younger conflicting item |
| `_run_no_landings_breaker_pass()` | harness.py | Harness pass that acts on `BreakerTrip`: calls `force_halt_scheduler` + files L2-INFO escalation |

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
