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
- A verify may only start against a base that is the tip of the frozen prefix (no verify against a speculative-only base).
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

## 11. Open questions (tactical — deferred, not design-blocking)

1. **Circuit-breaker window + disk-margin constants.** Provisional defaults in θ; calibrate from ι's metrics. Decide during θ/ι impl.
2. **merge-tree probe caching granularity.** Recompute per reorder vs memoize per (base_tip, branch_head) pair. Pick the simpler (recompute) first; memoize only if a hot-loop profile shows it matters. Decide during δ.
3. **Aging tie-break for equal first-submission timestamps** (e.g. two members of a coalesced train). Suggested: fall back to `request_id` lexical order for determinism. Decide during ζ.
4. **Legacy `MergeWorker` parity.** The aging/bounce wiring targets `SpeculativeMergeWorker` (what reify runs). Whether to backport to the legacy worker or let it keep FIFO. Suggested: speculative-only; legacy stays FIFO. Decide during ζ.
