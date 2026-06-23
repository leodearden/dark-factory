# Capability Manifest — Two-Layer Merge Queue (conflict-graph + committed verify frontier)

Mechanizes G3 + G6 for `plans/two-layer-merge-queue-prd.md` (committed 8e15b84fab).
Built at decompose time so the substrate check is paid **once, here** — a dispatch-time
architect or downstream verifier diffs intent against substrate instead of re-deriving it.

**Evidence vocabulary:** `grep:<file>:<line> wired` (on main today) · `producer:<label> upstream`
(delivered by a task in the transitive dependency closure, upstream of this leaf) ·
`absent-today→producer:<label>` (capability does not exist on main; the named upstream task
produces it) · `floor:<bound> > <method-floor>` · `rejection-check:<X> fires`.

**Verdict:** every binding resolves to PASS. No `declared-only` / `test-only` /
`producer-downstream` / `producer-absent` / `producer-extent-short` / `fixture-ERROR` /
`bound≤floor` / `rejection-absent`. **Batch clears the manifest gate.**

Substrate base (git 2.43.0 host, dark-factory main @ 8e15b84fab):
`git merge-tree --write-tree` present · `_lane_buffers` merge_queue.py:5156 · `_pop_next_pickable`
merge_queue.py:5597 · `MergeRequest(...)` submit chokepoint workflow.py:4888 ·
`register_and_enqueue_merge_request` workflow.py:943/4964 · `MergeRequest.enqueued_at`
(re-stamped per resubmit) merge_queue.py:3421 · `MergeOutcome.reason` merge_queue.py:130-321 ·
`rebase_onto_main` git_ops.py:2417 · 1688 thrash signature `_merge_outcome_signature`
merge_queue.py:442 · `force_halt_scheduler`/`force_resume_scheduler` harness.py:5973/5941 ·
`shutil.disk_usage().free` merge_queue.py:576 · `escalate_info` (escalation MCP, L2-info path) ·
scheduler aging precedent `_starvation_first_seen` scheduler.py:1099.
**Confirmed absent today:** `metadata.merge_first_enqueued_at` (grep empty) → producer α.

---

## α (DF, intermediate → ζ) — persist first-submission timestamp

*Signal:* a resubmitted task's metadata retains its original timestamp across a block→re-dispatch cycle.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| submit chokepoint to stamp at | `grep:workflow.py:4888 wired` (`MergeRequest(...)` + `register_and_enqueue_merge_request` :4964) | PASS |
| `metadata.merge_first_enqueued_at` field | `absent-today→producer:α` (this task creates it; grep on main empty) | PASS (self-produced) |
| write-once / survives restart | persisted in task metadata read path (observable via task metadata read) — produced by α | PASS |
| (G6 branch-3) timestamp readable across block→re-dispatch | end-to-end capability delivered **by α itself**; no downstream owner | PASS |

## β (DF, intermediate → δ, η, ι) — merge-tree conflict-probe primitive

*Signal:* CLI/log probe reports `clean`/`conflicted_paths` for two branches with **zero** worktree creation.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `git merge-tree --write-tree` | `grep`: git 2.43.0 host, `--write-tree` in help | PASS |
| `git_ops.merge_tree_conflicts(base,head)->ConflictProbe` | `producer:β` (this task adds it to git_ops.py beside `rebase_onto_main` git_ops.py:2417) | PASS (self-produced) |
| (G6 branch-4, negative) "no `_merge-*`/lane dir touched" | `rejection-check`: probe runs against in-repo object store only; boundary test asserts no worktree path created. Mechanism (object-store-only probe) built **by β**; firing verified in β's own RED + λ | PASS |

## γ (DF, intermediate → δ, κ) — OverlapFootprintDetector interface + default + registration

*Signal:* default detector reports overlap for two changesets sharing a path, disjoint otherwise; textual-conflict⇒overlap holds.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `OverlapFootprintDetector` Protocol + `Footprint` | `producer:γ` (this task defines §5.1 contract) | PASS (self-produced) |
| default path-intersection detector | `producer:γ` (mirrors 1595 disjoint-delta path relation) | PASS |
| `project_id`-keyed registration, default fallback | `producer:γ` | PASS |
| (G6 branch-4) textual⇒overlap invariant holds; fail-open on detector raise | `rejection-check`: contract invariant; γ's boundary test authors a textually-conflicting pair and asserts `overlaps=True` — mechanism + assertion both in γ | PASS |

## δ (DF, intermediate → ε, ζ, λ) — conflict-graph over the unfrozen suffix

*Signal:* heartbeat/snapshot exposes the per-suffix conflict relation; a new main-conflicting submission is marked.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| textual probe | `producer:β upstream` (δ depends_on β) | PASS |
| footprint relation | `producer:γ upstream` (δ depends_on γ) | PASS |
| unfrozen-suffix buffer to graph over | `grep:merge_queue.py:5156 wired` (`_lane_buffers`) | PASS |
| heartbeat/snapshot emit path | `grep:merge_queue.py:5684-5705 wired` (heartbeat snapshot fields) | PASS |

## ε (DF, intermediate → η, λ) — verify-frontier / frozen-prefix invariant

*Signal:* property test — an in-flight verify is never reordered/re-based; reordering touches only the unfrozen suffix.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| conflict-graph over suffix | `producer:δ upstream` (ε depends_on δ) | PASS |
| verify-frontier / `_verifier_queue` | extends 1646/1862 (both done on main); frontier substrate present | PASS |
| (G6 branch-3) frozen-prefix immutability | delivered by ε itself + δ upstream; no downstream owner | PASS |

## ζ (DF, intermediate → λ) — aging-priority comparator at `_pop_next_pickable`

*Signal:* given two conflicting queued items, older-first-submission picked first; a disjoint younger item bypasses.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| comparator hook point | `grep:merge_queue.py:5597 wired` (`_pop_next_pickable`) | PASS |
| `merge_first_enqueued_at` to sort on | `producer:α upstream` (ζ depends_on α) — **DAG-direction OK** (α is upstream, not downstream) | PASS |
| `enqueued_at` legacy fallback | `grep:merge_queue.py:3421 wired` | PASS |
| conflict-clique scoping + disjoint bypass | `producer:δ upstream` (ζ depends_on δ) | PASS |

## η (DF, intermediate → λ) — fast textual-conflict bounce + mechanical-rebase-first

*Signal:* textually-conflicting submission → `needs_rebase` at graph time (no verify slot); clean auto-rebase re-queues; real conflict escalates; Nth bounce escalates.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| textual probe at graph time | `producer:β upstream` (η depends_on β) | PASS |
| frozen-prefix to stack the probe onto | `producer:ε upstream` (η depends_on ε) | PASS |
| `needs_rebase` merge outcome | `grep:merge_queue.py:130-321` (`MergeOutcome.reason` family present) + `producer:η` adds the value | PASS |
| mechanical rebase | `grep:git_ops.py:2417 wired` (`rebase_onto_main`) | PASS |
| bounce-cap → escalate | `grep:merge_queue.py:442 wired` (1688 `_merge_outcome_signature` thrash) | PASS |
| (G6 branch-4) "no verify slot consumed" on bounce | `rejection-check`: bounce diverts before `_verifier_queue`; mechanism built by η; firing asserted in η RED + λ §9 scenario 1 | PASS |

## θ (DF, intermediate → λ) — no-landings circuit-breaker

*Signal:* forced no-landings + falling-disk → dispatch halts + L2-info naming window/landings/disk; a clean landing resumes.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| retries-per-landing / drift / landing-rate inputs | `producer:ι upstream` (θ depends_on ι) | PASS |
| free-bytes trend input | `grep:merge_queue.py:576 wired` (`shutil.disk_usage().free`) | PASS |
| `force_halt_scheduler` / `force_resume_scheduler` | `grep:harness.py:5973/5941 wired` | PASS |
| L2-info escalation | `grep`: `escalate_info` escalation MCP tool present | PASS |
| (G6 branch-4) halt fires + auto-resume on recovery | `rejection-check`: halt mechanism (`force_halt_scheduler`) present + built into θ; λ §9 scenario 9 asserts fire/escalate/resume | PASS |

## ι (DF, intermediate → θ, λ) — instrumentation: retries-per-landing + drift-at-detection

*Signal:* dashboard shows both metrics; drift-at-detection drops once β/δ bounce early.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| early-bounce signal to measure drift against | `producer:β upstream` (ι depends_on β) | PASS |
| heartbeat/dashboard metric emit path | `grep:merge_queue.py:5684-5705 wired` (heartbeat snapshot) | PASS |

## κ (RF, **leaf**) — crate-graph-aware OverlapFootprintDetector

*Signal:* two reify changesets touching different files in the **same crate's test target** report `overlaps=True` (path default returns False — proves the override earns its place).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `OverlapFootprintDetector` interface to implement | `producer:dark_factory:γ upstream` (external dep `dark_factory:<γ>`; resolves `done` before κ dispatches) — **DAG-direction OK, one-directional** | PASS |
| changed-path → crate mapping (Cargo dep graph) | end-to-end capability delivered **by κ itself** (reify-side crate-graph traversal; `cargo metadata` substrate) | PASS |
| reify-side two-way boundary test (incl. textual⇒overlap) | `rejection-check`: κ authors the same-crate/different-file pair, asserts override `overlaps=True` ∧ default `overlaps=False`; both mechanism + assertion in κ | PASS |

## λ (DF, integration-gate leaf-in-spirit → μ, ν) — B+H integration gate

*Signal:* the integration test (the G2 leaf) is green — drives conflict-graph + frontier + aging + bounce + breaker with the **default** detector, asserting all §5.3 invariants + §9 boundary scenarios in one run.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| conflict-graph | `producer:δ upstream` | PASS |
| frozen-prefix invariant | `producer:ε upstream` | PASS |
| aging comparator | `producer:ζ upstream` | PASS |
| bounce + rebase-first | `producer:η upstream` | PASS |
| circuit-breaker | `producer:θ upstream` | PASS |
| instrumentation metrics | `producer:ι upstream` | PASS |
| default footprint detector | `producer:γ upstream` (transitive via δ) | PASS |

All six §8λ prereqs (δ, ε, ζ, η, θ, ι) are upstream — no producer is downstream of λ. **DAG-direction OK.**

## μ (DF, **leaf**) — companion doc correction

*Signal:* `skills/merge-queue/SKILL.md` + merge-queue docs describe the bounce outcome + aging order.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| the two-layer behaviour to document | `producer:λ upstream` (μ depends_on λ — load-bearing wiring proven by λ; μ is a legitimate doc leaf, not a docs-only close of unbuilt wiring) | PASS |

## ν (DF, **leaf**) — DEPLOY: restart orchestrator-reify

*Signal:* post-restart heartbeat shows aging order + bounce outcomes live.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| pipeline merged to main | `producer:λ upstream` (ν depends_on λ) | PASS |
| reify detector on reify main (soft, deploy-time) | κ on reify main — **NOT a wired DF→κ edge** (one-directional cross-project policy; ν falls back to the §5.1 default detector if κ has not yet landed). Documented soft prerequisite only. | PASS (fail-open default) |
| (G6 branch-3) live heartbeat shows new order/outcomes | delivered by λ upstream + the restart action ν performs | PASS |

---

### Cross-project edge ledger (the one part not mechanized in the PRD)

| Edge | Form | Wired? |
|---|---|---|
| κ (reify) → γ (dark-factory) | qualified external dep `dark_factory:<γ_id>` on κ, filed with `project_root=/home/leo/src/reify` | **YES** |
| ν (dark-factory) → κ (reify) | §8 prose "(+ κ on reify main)" | **NO** — one-directional policy: no dark-factory task depends on κ. ν falls back to the default detector; κ landing is a documented deploy-time soft prerequisite, not a DAG edge |

No dark-factory task depends on κ. The only cross-project edge points reify → dark-factory.
