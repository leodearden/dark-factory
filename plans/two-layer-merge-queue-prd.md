# PRD — Two-Layer Merge Queue: conflict-graph + committed verify frontier (merge-churn gain reduction)

**Status:** built/landed — decomposed α–λ; full pipeline landed 2026-06-25 via λ=1895 (`bbaec52696`)
**As-built:** [skills/merge-queue/SKILL.md](../skills/merge-queue/SKILL.md) (§ "The two-layer merge queue") · [skills/merge-queue/references/two-layer-model.md](../skills/merge-queue/references/two-layer-model.md)
**Milestone:** orchestrator merge-queue hardening (follows 1595 / 1646 / 1862 speculation family; complements the warm-lane Δp space-safety batch and the verifier-loop-supervision batch 1856–1858)
**Type:** extension of the shipped merge-queue subsystem (dark-factory orchestrator), cross-project (small reify piece)
**Approach:** B + H (contract + two-way boundary tests) — the merge queue is the load-bearing seam; cross-project; ≥8 mechanisms.

---

## 1. Goal

Reduce the **loop gain** of the merge-churn feedback spiral so it stays self-damping (G < 1), making the conditions that trigger emergency containment (warm-lanes ENOSPC, 0 clean landings) rare rather than recurrent.

**User-observable (operator) signals when this lands:**
- The merge-queue heartbeat / dashboard shows resubmitted tasks ordered by **age of first submission** within a conflict clique (the oldest, most-expensive-to-redo task gets the cleanest landing shot), while disjoint tasks still land out-of-order for throughput.
- A textually-conflicting branch is **bounced fast** (a new `needs_rebase` merge outcome / log line) — at submission-graph time, before it consumes a verify slot — and is auto-resolved by a mechanical rebase where possible (no agent spend).
- New operator metrics: **retries-per-landing** and **drift-at-detection** (how far main advanced before a conflict was caught), visible on the dashboard.
- When the spiral does start (landing-rate ≈ 0 under disk pressure), the orchestrator **auto-halts dispatch and files an L2-info escalation** ("stop digging"), then auto-resumes on recovery — instead of a human noticing hours later.

## 2. Background

### 2.1 The gain model
Let **p** = probability an in-flight task fails to land cleanly (textual conflict, semantic verify-RED, or infra) and must retry. One failure feeds the next through three terms:
- **baseline p** — set by queue latency × main-advance rate × file-overlap. Longer wait at the head → more staleness → higher p.
- **Δp (coupling)** — how much one failure raises *everyone else's* p by consuming a shared resource (verify slot, warm-lane disk, build cache). This is the term that turns a local failure into a systemic spiral.
- **p′ (retry success)** — probability the retried task lands. If a retry re-enters with the same stale base (and rebuilds cold), p′ ≈ p and the loop barely damps.

Loop gain **G ≈ (failures caused per failure)**; G < 1 ⇒ self-damping. The deployed work (CoW warm builds, coalescence, two verify hosts) attacked **baseline p** via latency; the warm-lane space-safety batch (1859–1861 + reify 4716–4719) attacks **Δp** for the *task-dispatch* path. The largest remaining headroom is in the *merge-queue* path's **Δp** and **p′** — and those are the cheap levers this PRD targets.

### 2.2 The incident this generalizes
2026-06-22 warm-lanes `/dev/loop25` hit 100% repeatedly despite a 1 TB→4 TB grow; "stale-branch backlog churns (build→conflict-merge→retry, no landings) … refilling the disk faster than GC frees, regardless of volume size … 0 clean landings in 3.5h." Containment was **manual** (L2 watcher halt "per Leo's delegated recovery"). reify tasks 4368/3465 (among others) were repeatedly false-blocked. Diagnosis: the churn is a positive feedback loop with gain ≥ 1; disk-grow raises the ceiling but does not lower the gain.

### 2.3 Prior work this builds on
- **1595** — disjoint-delta re-verify gate (file-overlap relation; reused for ordering).
- **1646** — bound merger-ahead + freshness re-base at verify-pickup (the verify-frontier idea, bounded at K=1).
- **1862** — speculation attaches late requests to the in-flight predecessor tip (depth-1 stacked speculation; this PRD generalizes it to depth-K via `merge-tree`).
- **1688** — merge-outcome thrash signature (reused as the bounce-cap → escalate backstop).
- Scheduler `_starvation_first_seen` aging (scheduler.py:1099) — precedent pattern for an aging key (but in-memory, warns-only; this PRD persists + reorders).

## 3. Sketch of approach — a two-layer pipeline

| Layer | Depth | Cost | Mutable? | Job |
|---|---|---|---|---|
| **Speculative merge graph** (`git merge-tree --write-tree`) | whole unfrozen queue suffix | git objects only — no worktree, no build, no warm-lane disk | reorderable | fast textual-conflict bounce + the conflict relation that drives ordering |
| **Verify frontier** | shallow K (unchanged) | wall-clock + warm-lane disk | **frozen** (an item being verified is immutable: it fails or lands) | the authoritative gate |

Reordering only ever touches the **unfrozen suffix above the verify frontier**; recomputing the affected suffix's `merge-tree` results is cheap and disk-free. This decouples *merge depth* (deep, cheap) from *verify depth* (shallow, expensive) — the core idea.

**Levers, mapped to gain terms:**
- *baseline p ↓*: conflict-detection latency collapses (bounce at graph time, not at the head) → minimal drift-at-detection; disjoint throughput bypass keeps the queue draining.
- *Δp ≈ 0*: `merge-tree` conflict gating touches no worktree/disk, so a bounced item never consumes a verify slot or warm-lane bytes — a conflict failure stops coupling to others.
- *p′ ↑*: items bounce early (less drift to re-resolve) and the older/at-risk item in a conflict clique is prioritized to land before it re-drifts; **mechanical-rebase-first** resolution preserves the (often >$50) agent work, escalating to the agent only on a true semantic conflict.
- *backstop*: circuit-breaker A catches the residual spiral and escalates L2-info.

## 4. Resolved design decisions

1. **Conflict-probe via `git merge-tree --write-tree`** (verified present: git 2.43.0 on host). Stateless / idempotently recomputable — chosen over worktree-based speculation specifically because it adds *computation, not persistent state*, to a subsystem with a history of stateful wedges (1856/1857).
2. **Two thresholds, not one.** *Textual* conflict (merge-tree) drives the **bounce** decision (high precision — never bounce a task that would have merged). *Footprint overlap* (a superset) drives the **ordering** decision only — a clean textual merge can still cause a semantic verify-RED when two tasks touch the same file. Never reject on footprint overlap; only sequence on it.
3. **Stacked-on-frozen-prefix, not bare main.** An item is merge-probed onto the predicted tip of the frozen prefix, so we don't false-bounce an item that conflicts only with something ahead that is going to land anyway. Rule: **bounce the *younger* of two conflicting items; let the older proceed.**
4. **Aging key = persisted age-of-first-submission.** Substrate is absent today (no such timestamp exists anywhere — see §6), so it is a hard prerequisite (task α): stamp `metadata.merge_first_enqueued_at` **write-once** at the workflow submit chokepoint, surviving process restart (the orchestrator restarts often). Wall-clock. Aging gives a *stable* total order (converges; no reorder thrash), and protects exactly the large-slow tasks that cost the most to redo.
5. **Ordering is conflict-clique-scoped, within lane.** High lane still beats normal lane. Within a lane: among items that mutually conflict (footprint relation), order by first-submission age; items disjoint from everything ahead may be picked out-of-order (throughput bypass). This unifies footprint-aware land-ordering with age-based anti-starvation.
6. **Bounce resolution = mechanical-rebase-first** (decision: in scope). On bounce, auto-rebase the branch onto the frozen tip; clean → re-queue (work preserved, no agent); real conflict → escalate to agent/steward. Bounce-count cap → escalate, reusing the 1688 thrash signature, so a flapping A↔B conflict cannot become an agent-$ fire.
7. **Circuit-breaker A in scope** (decision). Detect landing-rate ≈ 0 over a window AND warm-lane disk-pressure rising → auto-halt new dispatch + L2-info escalation; auto-resume on a clean landing or disk recovery. Thresholds are **provisional**, set from the instrumentation leaf (§7 boundary tests cover fire/escalate/resume; calibration is a tactical follow-up, not a blocker).
8. **Per-project pluggable footprint detector.** dark-factory owns the interface + a default path/changed-file-set detector; reify supplies a crate-graph-aware detector. One-directional cross-project dependency (reify → dark-factory), matching the external-dep gate model.

## 5. Contract section (B+H)

### 5.1 Overlap-footprint detector (the cross-project seam)
```python
class OverlapFootprintDetector(Protocol):
    def footprint(self, changed_paths: Sequence[str]) -> Footprint: ...
    # Footprint is an opaque, comparable set-like value.
    def overlaps(self, a: Footprint, b: Footprint) -> bool: ...
```
- **Invariants:** `overlaps` is symmetric and reflexive; textual conflict ⇒ `overlaps` is True (footprint is a *superset* of textual conflict — never narrower). A detector that returns `overlaps=False` for two textually-conflicting changesets is a contract violation (the boundary test asserts this).
- **Default (dark-factory):** footprint = set of changed file paths; `overlaps` = non-empty path intersection (mirrors the 1595 disjoint-delta relation).
- **reify override:** footprint = the transitive crate set reachable from changed paths via the Cargo dep graph; `overlaps` = non-empty crate-set intersection (catches "different files, same crate's test target" semantic interactions the path default misses).
- **Registration:** the orchestrator selects the detector by `project_id`; absence → default. Fail-open: a detector raising is treated as `overlaps=True` (conservative — never skip a re-verify on detector error).

### 5.2 merge-tree conflict probe
- `git_ops.merge_tree_conflicts(base_tip: str, branch_head: str) -> ConflictProbe` where `ConflictProbe = (clean: bool, conflicted_paths: list[str])`.
- No worktree, no index mutation, no checkout; writes only loose tree/blob objects. MUST run against the in-repo object store, never touch `worktree_base` / the warm-lane volume.
- Deterministic and side-effect-free w.r.t. refs; safe to call many times per tick.

### 5.3 Conflict-graph + verify-frontier invariants
- **Frozen prefix** = {items currently verifying} ∪ {landed}. An item, once it enters verify, is immutable — it fails or lands; it is never reordered or re-based out from under an in-flight verify.
- **The frozen-prefix INVARIANT is INV-3 chain-intact adoption** (`plans/merge-verdict-integrity-prd.md` §1, which generalizes this section from dispatch time to adoption time: "the rule is chain-intact, NOT `base == main`"). It is enforced at adoption by `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker::_void_and_remerge`, which `_finalize_inflight` calls: a PASS *or* FAIL verdict whose `base_sha` is a recorded dead commit is "neither adopted nor advanced". `snapshot()['two_layer_invariants']` reports it over the frozen prefix through `check_frozen_prefix_invariant`, as two kinds of break: an **INV-3 dead link**, a frozen real merge built on a recorded dead commit, judged by `_chain_dead_link`, the same predicate behind INV-3's three enforcement points, so report and enforcement are one mechanism; and a **successor chain link**, a successor whose base is not its predecessor's merge commit. The head is **not** compared with current main (task 5302). Main legitimately moves under an in-flight verify (a direct-to-main commit, a foreign landing, a predecessor's rebased landing), and a head on an older LANDED main is intact: the CAS rebase and the gate re-verify own it.
- **A verify should start against a base that is the tip of the frozen prefix** (no verify against a speculative-only base). **This rule is a DETECTOR, not an invariant (task 5302).** It is evaluated once, at dispatch, by `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker::_verify_base_divergence_at_dispatch` against that dispatch's freshly fetched main; the policy is the pure `orchestrator/src/orchestrator/merge_lane/verify_base_detector.py::detect_verify_base_divergence`. A hit is stamped on the entry as a `VerifyBaseDivergence` (`InflightEntry.verify_base_divergence`), surfaced in `snapshot()['verify_base_divergences']`, logged at WARNING, and filed once as a dedup'd L1 `merge_verify_base_divergence` (sentinel `__merge_verify_base_divergence__`). One computation feeds every surface, so they cannot disagree; none reads the cached `_last_known_main_sha`, and a later move of main cannot create a record. There is **no dispatch-time refusal** (esc-3206-3). Dispatch time cannot know whether the item frozen ahead will land or fail; adoption time can, and the soundness property belongs to INV-3 above. Refusing at dispatch would trade a bounded, already-backstopped efficiency cost (an occasionally-wasted verify) for an unbounded liveness risk (refused dispatch → head-of-line wedge). This retires the earlier "advisory by decision" framing of tasks 1999 and 3206: under the esc-3206-3 ruling the rule is not a weakly-enforced invariant but a detector over a dispatch-time property.
  - **RE-MERGE CARVE-OUT — REVERSED (task 8524).** Leo's ruling of 2026-11-05 reverses task 3206's carve-out; `plans/merge-lane-policy-prd.md` D1 is the canonical record. A recovery re-merge from any of the five consumers of `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker::_remerge` (head-failure cascade, adoption-time void, RUNNER_UNAVAILABLE re-dispatch, dispatch-time dead-base straggler, Mechanism-2 discard) now stacks on the live frontier tip chosen by `orchestrator/src/orchestrator/merge_lane/remerge_base.py::choose_remerge_base`. It falls to main, with the reason recorded, only when no live tip exists or no permit can be held. Today only the dispatch-time dead-base straggler can find one: the cascade re-merges after the frontier is torn down, the adoption-time void and RUNNER_UNAVAILABLE sites count every in-flight commit as doomed (`orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker::_cascade_doomed_commits`), and the Mechanism-2 discard runs only with no verify in flight, so those four structurally fall to main. The item carries that decision as `remerge_base` (`orchestrator/src/orchestrator/merge_lane/types.py::RemergeBase`), stamped at `_remerge`'s single exit. The detector exemption narrows to the fell-to-main case (`orchestrator/src/orchestrator/merge_lane/types.py::fell_to_main`). A live-tip re-anchor is judged like any stacked dispatch, and a hit on one is the detector class `reanchor_onto_dead_tip` (`orchestrator/src/orchestrator/merge_lane/types.py::VerifyBaseDivergenceKind`). `check_frozen_prefix_invariant` still walks a fell-to-main entry as a chain link and annotates its broken successor link with the anchor reason. The three historical grounds, each answered by D1:
    1. **Circularity** (the frozen tip can be the dead commit being escaped): a re-anchor base must be live (not dead, its chain intact and unvacated, its owner's verify alive), else the re-merge falls to main.
    2. **Permit accounting** (`speculative=False` was hardcoded): a re-anchored replacement is `speculative=True` and holds exactly one permit, moved from the item or taken without waiting, enforced by `orchestrator/src/orchestrator/merge_lane/remerge_base.py::RemergeClaim`.
    3. **Recovery semantics** (main is the known-good base): main stays the fallback, and INV-3 still voids a verdict built on a dead base at adoption.
  - **BLAST-RADIUS COUPLING WITH THE 3082 STRANDED-FINALIZE-HEAD CLASS (task 3206; definition corrected by task 5302).** The 3082 class was survivable **only because this rule never refused a dispatch.** Before task 5302, `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker::_frozen_inflight_entries` counted the finalize head as frozen while its phase was `verifying`/`gate_reverify`/`finalizing`, with no liveness check, so a *phantom* (stranded) head made `frozen_prefix_tip()` return a **dead merge commit** — and every subsequent real-verify dispatch then mismatched it, even when the dispatching item's base was perfectly correct. Measured proof (orchestrator-reify): 2026-08-08 00:09:07 (task 5687, base `9d08d3d3`) and 01:35:00 (task 5830, base `2b903602`) both dispatched against the **live main tip** yet mismatched a phantom expected-tip `95908e44` that persisted 86 minutes; 5687 then passed and landed — "verify end (merge=2b903602, passed=True)" → "Advanced main to 2b903602" at 00:59:55. **Under hard enforcement both dispatches would have been REFUSED.** Corroborated by the 3082 signature ("extra InflightEntry object(s) in `_live_items` … beyond the expected single finalize head") firing 7× at 00:59:58. This identified the frozen-prefix *definition* as the weak point. **Task 5302 added a second guard, not a closure:** the finalize head counts as frozen only if it is in a qualifying phase AND not `vacated` (the vacated filter in `_frozen_inflight_entries`), the same liveness rule `_occupying_inflight_entries` already applies to the deque. It is redundant behind task 3082's disposal: `_requeue_from_verify` and `_drop_from_verify` swap or pop the head's `_live_items` entry synchronously right after stamping the status, so `_finalizing_head_entry()` returns a vacated head only when that registry transition is rejected. Two phantom-tip sources remain that the rule cannot see. (i) A finalize head stranded WITHOUT being vacated. (ii) A head parked at `gate_reverify` by `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker::_land_rebased_or_park_gate`, which records the head's own merge commit dead and then parks it in a verify phase, so for the whole gate window `frozen_prefix_tip()` can be that dead commit; every rebased landing takes this path (ticket tkt_0RWKN8Q02REA5YEFNQX4WDJRJT). So a detector hit whose `expected_tip` is a requeued or dropped head means the liveness rule regressed; one whose `expected_tip` belongs to a non-vacated head stuck in `finalizing` or `gate_reverify` is one of these residual sources, and the dispatched item is fine.
  - **MEASURED SWEEP BEHIND THE VERDICT (task 3206) — canonical record.** `ε=1890 §5.3 guard` swept across all SEVEN `orchestrator-*` units over the full journald retention window: **4 hits, all on orchestrator-reify, all `verify_depth=1`; zero on the other six units**. Every hit classified, **zero unexplained** (the task-1999 gate), and measured precision as an enforcement predicate was therefore **0/4**:
    - **class (b) — `_remerge`-by-construction (2 hits):** 2026-08-07 02:36:52 task 6063 (base `4485bf77` / expected `d8081962`); 12:11:29 task 6067 (base `e2aeeb7e` / expected `42847642`) — each preceded seconds earlier by "dead-base straggler at dispatch — re-merging". **Eliminated by construction** from task 3206 onward by the re-merge carve-out above; task 8524 keeps it eliminated, since a re-merge that fell to main stays exempt and a live-tip re-anchor sits on the frozen tip.
    - **class (a) — task-3082 stranded/phantom finalize head (2 hits):** 2026-08-08 00:09:07 task 5687 and 01:35:00 task 5830 (the pair detailed in the blast-radius bullet above).
    - **class (c) — stale `_last_known_main_sha` false positives: ZERO**, and *structurally* excluded: the dispatch guard uses its own freshly-fetched `get_main_sha()` and writes it into the cache, so task 2357's defect only ever reached the snapshot surface, never that call site.
    - **class (d) — genuine §5.3 violations: ZERO.**
    - Zero hits fleet-wide in the ~31h after task 3082 landed (`e85124dd31`, 2026-08-08 06:55).
    - **Consequence for the next sweep:** with class (b) gone by construction and class (c) structurally excluded, ANY future hit is either class (a) (a 3082-style phantom head) or a genuine class (d) — a materially sharper signal than the journal archaeology this classification required. Since task 5302 the detector escalates instead: each hit is filed as an L1 `merge_verify_base_divergence` carrying its record, which also sits in `snapshot()['verify_base_divergences']`, so the classification no longer needs the journal. This record is the durable home for the dated detail above; the code carries a pointer here, not a copy (it would silently rot once the journal retention window rolls).
  - **RECURRENCE ROOT CAUSE (task 5302) — canonical record.** Until task 5302, `two_layer_invariants` also re-judged the verify-base rule at snapshot time, and that report recurred three times:
    - **2026-07-08:** task 2357 DEFECT 2 — the cached `_last_known_main_sha` lagged BEHIND main. 2357 refreshed the cache at land, dequeue and dispatch, which removed that one contributing cause only.
    - **2026-08-12, mr-763d254b / task 3956 — MEASURED via git.** The reported actual base `0e06325856` is "Merge task/3223 into main" (18:33:51 +0100), the lane's speculative merge commit for 3223, which is NOT an ancestor of main. The expected `991b5b468e` is 3223's rebased linear landing (19:13:44 +0100), which IS on main. So the cache was correct and AHEAD of the head's dispatch base: the head had been stacked on its predecessor's speculative commit, and the predecessor then landed rebased.
    - **2026-08-28, mr-4ee12a75 / task 3122, then mr-627b607f.** Hypothesis: the same mechanism. 3122 landed as a rebased linear series at 15:50:00, with direct-to-main docs commits through the morning; the SHAs were not recorded.
    - **Mechanism — a class distinct from 2357's.** The snapshot surfaces re-evaluated a DISPATCH-TIME property against POST-DISPATCH main: the head arm of the base-chain check expected `head.base == current main`, and the snapshot verify-base sub-check repeated the same chain math. Both fired for the whole remaining verify whenever main moved under an in-flight head — a direct-to-main commit, a foreign landing, a predecessor's rebased landing — so in a repo with frequent direct commits the field could never read clean. `orchestrator/tests/test_merge_queue_invariant_integration_gate.py::TestScenario10VerifyBaseMismatch` pinned exactly that false positive as a violation until task 5302 flipped it; 2357's fresher cache only exposed the drift sooner. The 2026-08-12 shape now reads as one INV-3 dead link (the predecessor's superseded speculative commit is recorded dead), and pure main drift is silent.
    - **Why the new key is actionable rather than pull-only.** The old strings sat in `two_layer_invariants`, which no consumer alerts on, mixed with three internal-consistency checks, so one benign hit made the whole health surface read unhealthy. A `verify_base_divergences` record is judged once, against the main its dispatch actually saw, so each names a real dispatch decision; and each files a dedup'd L1, so a hit reaches the escalation ladder instead of waiting for someone to read the snapshot.
- The **unfrozen suffix** may be reordered/inserted freely; any reorder triggers recompute of `merge-tree` for the affected suffix only.
- Liveness: the frontier advances by exactly one when its head item reaches a terminal verify outcome; aging guarantees no item in a conflict clique starves (the oldest is always pickable once its clique-predecessors clear).

### 5.4 Aging key
- `metadata.merge_first_enqueued_at`: wall-clock epoch float, stamped **once** at the first merge submission of a branch's lineage, **never overwritten** on resubmit, persisted in task metadata (survives restart). Comparator reads it at `_pop_next_pickable`; missing value (legacy in-flight at upgrade) falls back to `enqueued_at`.

### 5.5 Circuit-breaker A
- Inputs: rolling landing-rate (lands / window) and warm-lane free-bytes trend (from the instrumentation leaf).
- Trip: landing-rate == 0 over `window` AND free-bytes monotonically falling over the same window. Action: `force_halt_scheduler(reason=…)` + L2-**info** escalation naming the window, landing count, and disk trend.
- Reset: a clean landing OR free-bytes recovered above a margin → `force_resume_scheduler`. Hysteresis to prevent flap.

## 6. Pre-conditions for activating

- **Substrate gap (hard prerequisite, task α):** no persisted first-submission timestamp exists anywhere today. Agent-confirmed: `MergeRequest.enqueued_at` (merge_queue.py:3421, wall-clock, in-memory) is **re-stamped on every resubmission** and lost on restart for any task not in-flight-at-crash; resubmissions are brand-new `MergeRequest`s dedupe-keyed only on `branch` (merge_queue.py:3298/3376/2511) with no lineage link; the durable journal drops the entry on every terminal `blocked`/`conflict`/`error` outcome. The aging key therefore has **zero existing substrate** and α must land before ζ.
- **Substrate present (verified):** `git merge-tree --write-tree` (git 2.43.0); comparator hook point `_lane_buffers` (merge_queue.py:5156–5158) + `_pop_next_pickable` (5597–5609), today pure `(lane high→normal, FIFO)`.

## 7. Cross-PRD / cross-project relationship (G4)

| Other PRD / work | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| reify repo (this PRD's κ) | this produces interface, reify consumes | `OverlapFootprintDetector` (§5.1) | **this PRD owns interface + default**; reify owns the crate-graph impl + its registration | queued (α/γ upstream of reify κ via external dep `dark_factory:<γ>`) |
| Warm-lane space-safety batch (1859–1861 / reify 4716–4719) | complementary (Δp) | warm-lane GC + requeue-exit-75 | that batch | landed; this PRD references, no seam |
| Merge-verify ENOSPC fail-soft (symptom task) | complementary | `workflow.py:5089` transient-infra `_mark_blocked` → re-queue | **separate task (not filed here)** | referenced — out of scope §9 |
| Verifier-loop supervision (1856–1858) | complementary | merge-worker loop liveness | that batch | landed; orthogonal |

## 8. Decomposition plan

Greek labels; actual IDs assigned at decompose. DF = dark-factory repo, RF = reify repo.

**Phase 0 — foundation / substrate**
- **α (DF)** — Persist first-submission timestamp. Stamp `metadata.merge_first_enqueued_at` write-once at the workflow submit chokepoint (workflow.py:~4888), guarded against overwrite on resubmit; survives restart. *Signal:* a resubmitted task's metadata retains its original timestamp across a block→re-dispatch cycle (observable via the task metadata read path). *Prereq:* none.
- **β (DF)** — `merge-tree` conflict-probe primitive in git_ops (§5.2). *Signal:* a CLI/log probe reports `clean`/`conflicted_paths` for two branches with zero worktree creation (assert no `_merge-*`/lane dir touched). *Prereq:* none.
- **γ (DF)** — `OverlapFootprintDetector` interface + default path-overlap detector + registration (§5.1). *Signal:* the default detector reports overlap for two changesets sharing a path and disjoint otherwise; textual-conflict⇒overlap property holds. *Prereq:* none.

**Phase 1 — vertical slice (the two-layer pipeline)**
- **δ (DF)** — Conflict-graph over the unfrozen suffix, built from β (textual) + γ (footprint); incremental recompute on submit/resubmit/reorder/landing. *Signal:* heartbeat/snapshot exposes the per-suffix conflict relation; a new submission that conflicts with main is marked in the graph. *Prereq:* β, γ.
- **ε (DF)** — Verify-frontier / frozen-prefix invariant; decouple speculative-merge depth (deep) from verify depth (shallow K). Extends 1646/1862. *Signal:* a property test shows an in-flight verify is never reordered/re-based; reordering touches only the unfrozen suffix. *Prereq:* δ.
- **ζ (DF)** — Aging-priority comparator at `_pop_next_pickable`, conflict-clique-scoped with disjoint throughput bypass, keyed on α. *Signal:* given two conflicting queued items, the older-first-submission one is picked first; a disjoint younger item bypasses. *Prereq:* α, δ.
- **η (DF)** — Fast textual-conflict bounce outcome + mechanical-rebase-first resolution; bounce-cap → escalate (reuse 1688). *Signal:* a textually-conflicting submission yields a `needs_rebase` outcome at graph time (no verify slot consumed); a clean auto-rebase re-queues it; a real conflict escalates; the Nth bounce escalates. *Prereq:* β, ε.

**Phase 2 — backstop**
- **θ (DF)** — No-landings circuit-breaker (§5.5): detect landing-rate≈0 + disk-pressure-rising → `force_halt_scheduler` + L2-info → auto-resume. *Signal:* in a forced no-landings+falling-disk scenario, dispatch halts and an L2-info escalation appears naming window/landings/disk; a clean landing resumes it. *Prereq:* ι.

**Phase 3 — reify piece**
- **κ (RF)** — Crate-graph-aware `OverlapFootprintDetector` implementing γ, registered for reify, with a reify-side two-way boundary test proving it satisfies the contract (incl. textual-conflict⇒overlap). *Signal:* two reify changesets touching different files in the same crate's test target are reported as overlapping (path default would miss it). *Prereq:* external `dark_factory:<γ>`.

**Phase 4 — instrumentation + integration gate**
- **ι (DF)** — Instrumentation: retries-per-landing + drift-at-detection metrics on the heartbeat/dashboard (the measure-to-tune leaf; θ and ζ-tuning consume it). *Signal:* the dashboard shows both metrics; drift-at-detection drops measurably once β/δ bounce early. *Prereq:* β.
- **λ (DF)** — B+H integration gate: drive the full pipeline (conflict-graph + frontier + aging + bounce + breaker) with the **default** detector and assert all §5.3 invariants + §7 boundary scenarios in one run. *Signal:* the integration test (the G2 leaf) is green. *Prereq:* δ, ε, ζ, η, θ, ι.
- **μ (DF)** — Companion correction tasks: update `skills/merge-queue/SKILL.md` + merge-queue docs for the two-layer model; cross-ref the warm-lane Δp batch + the merge-verify fail-soft task. *Signal:* docs describe the bounce outcome + aging order. *Prereq:* λ.
- **ν (DF)** — DEPLOY: restart orchestrator-reify to load the pipeline (merged ≠ deployed; long-lived process). *Signal:* post-restart heartbeat shows aging order + bounce outcomes live. *Prereq:* λ on main (+ κ on reify main).

## 9. Boundary-test sketch (B+H)

| Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|
| Textual conflict bounces disk-free | two branches edit the same line; one queued behind the other | younger item → `needs_rebase` at graph time; **no** `_merge-*`/lane worktree created; no verify slot consumed |
| Clean speculative rebase re-queues, no agent | bounced item rebases cleanly onto frozen tip | item re-queued with work preserved; no agent invocation; `merge_first_enqueued_at` unchanged |
| Real conflict escalates, capped | bounced item conflicts on rebase; repeats N times | escalates to steward/agent; Nth bounce trips the 1688 thrash signature → L1 |
| Aging beats FIFO within a clique | older-first-submission item B and younger A both conflict, A enqueued later | B picked before A despite A's later FIFO position |
| Disjoint throughput bypass | small disjoint item C, contested clique ahead | C lands out-of-order; frozen-prefix invariant holds |
| Verify frontier immutable under reorder | item D verifying; a resubmission reorders the suffix | D's verify base unchanged; only unfrozen suffix recomputed |
| reify detector catches same-crate, different-file | two reify changesets, different files, same crate test target | `overlaps=True` (default path detector returns False — proves the override earns its place) |
| Contract: textual⇒overlap | any detector, two textually-conflicting changesets | `overlaps=True` (violation = contract failure) |
| Circuit-breaker fires + escalates + resumes | forced landing-rate 0 + falling free-bytes | dispatch halts; L2-info filed; clean landing resumes dispatch |

## 10. Out of scope

- **Merge-verify ENOSPC fail-soft** (turning `workflow.py:5089` transient-infra block+escalate into a lane-release/re-queue) — separate symptom task; referenced, not filed here.
- **Warm-retry / re-seed-thin on re-dispatch** — belongs to the warm-lane space-safety family (Δp), already partly addressed by the just-queued freeing batch.
- **Speculation depth K > 1 for the *verify* layer** — explicitly NOT widened; verify depth stays at the current shallow value (widening verify-ahead raises Δp under disk pressure — 1791 was cancelled for this reason). This PRD deepens only the *merge* layer.
- **Retuning circuit-breaker thresholds from production data** — tactical calibration follow-up once ι ships.
- **Hard control-flow enforcement of the §5.3 verify-base rule** — the "a verify may only start against the frozen-prefix tip" rule is a **detector** (record, log, L1; task 5302); it is never promoted to a dispatch-time refusal (esc-3206-3). Task 1999 recorded this fence; task 3206 re-affirmed it against production evidence (measured precision of the guard as an enforcement predicate was 0/4 over the full journal retention window) and wrote the reasoning into §5.3 above. A future attempt to flip it must first re-derive that evidence and retire the INV-3 adoption-time backstop that makes dispatch-time enforcement unnecessary.

## 11. Open questions (tactical — deferred, not design-blocking)

1. **Circuit-breaker window + disk-margin constants.** Provisional defaults in θ; calibrate from ι's metrics. Decide during θ/ι impl.
2. **merge-tree probe caching granularity.** Recompute per reorder vs memoize per (base_tip, branch_head) pair. Pick the simpler (recompute) first; memoize only if a hot-loop profile shows it matters. Decide during δ.
3. **Aging tie-break for equal first-submission timestamps** (e.g. two members of a coalesced train). Suggested: fall back to `request_id` lexical order for determinism. Decide during ζ.
4. **Legacy `MergeWorker` parity.** The aging/bounce wiring targets `SpeculativeMergeWorker` (what reify runs). Whether to backport to the legacy worker or let it keep FIFO. Suggested: speculative-only; legacy stays FIFO. Decide during ζ.
