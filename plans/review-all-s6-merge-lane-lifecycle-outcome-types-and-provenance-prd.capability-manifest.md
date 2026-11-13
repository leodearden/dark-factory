# Capability manifest — S6 merge-lane lifecycle, outcome types and provenance

PRD: `plans/review-all-s6-merge-lane-lifecycle-outcome-types-and-provenance-prd.md`. YAML twin:
`plans/review-all-s6-merge-lane-lifecycle-outcome-types-and-provenance-prd.capability-manifest.yaml`.

**Code anchors** verified against main `38a200d213` (2026-11-10; fold-verify, re-linted). Bindings cite
symbols, not lines. A substrate row is verified at authoring and carries a `manual` check
("substrate verified at authoring: `<path::symbol>`") — a grep that passes today would be refused
as `vacuous_present`. A producer row fails at this tree and passes once its leaf lands (the polarity
gate at `commit_planning`). Every grep pattern is an ERE with `(` escaped; the sidecar loads with
`shared.capability_manifest.CapabilityManifestDoc` and `shared.delivered_check_polarity.lint_delivered_checks`
returns no finding for any of its 32 mechanical checks against each leaf's declared files at
`38a200d213`. Upstream producers in other streams (S3 α, S3 κ, S4 β, S5 α, S5 λ0, S5 θ, S7 β) and the live tasks
5040 and 4833 carry no grep here: the dependency edge is the gate (a grep on 5040's module would turn vacuous
if 5040, which finishes first under S6-2 = B, lands before S6 is decomposed). The former leaf ν is
folded into α (INV-15 lands first, lead order revised 2026-11-10); φ, ξ and υ are new (S6-5). The lead's
sequencing rulings of 2026-11-10 add edges only (β1 → S3 κ, ζ → 4833, μ → λ); the alignment pass of
2026-11-10 (the lead's rulings after the fold-verifiers) leaves α with no cross-stream edge (S5 α, S5 λ0,
S7 δ, S7 β and S7 ε depend on α where they share a file) and adds β1 → S5 λ0, β1 → S7 β and κ → 4833; no
check changes. α's resolver keyword is `merge_cold`, no default.

## α — Lane timing budgets derived from the merge-verify budget; the load-time validator; INV-15

| Capability | Binding | Verdict |
|---|---|---|
| The authorised per-command merge-verify budget is computable from config + module registry | `orchestrator/src/orchestrator/verify.py::merge_verify_command_budget_secs`, `::_resolve_verify_timeout` (task 5371) | PASS (substrate; manual) |
| A start-time config refusal has a house pattern | `orchestrator/src/orchestrator/merge_lane/liveness.py::enforce_merge_liveness_margin` raising `MergeLivenessConfigError`, which the harness's merge-worker start re-raises | PASS (substrate) |
| `load_config` knows every module before the check | `orchestrator/src/orchestrator/config.py::load_config` sets `_module_configs` from `_discover_module_configs` | PASS (substrate) |
| The fleet-redeploy window is a config value | `OrchestratorConfig.orchestrator_restart_min_interval_secs` (default 259200; 0 disables) | PASS (substrate) |
| Four-site INV registration guard exists, numbers contiguous from 1; INV-15 is free | `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`; `docs/legibility/design-invariants.md` ends at INV-14 | PASS (substrate) |
| The four timing constants `LaneTimingBudgets` replaces are gone from the package | produced by α; `INFLIGHT_VERIFY_PROGRESS_BUDGET_SECS\|INFLIGHT_VERIFY_PROGRESS_PROBE_SECS\|RESOURCE_AUDIT_WORKTREE_GRACE_SECS\|PERIODIC_REAP_MIN_AGE_SECS` absent under `orchestrator/src/orchestrator/` | PASS (α delivers) |
| One definition of "authorised" | produced by α; `def resolve_verify_timeout\(` in `orchestrator/src/orchestrator/verify_budget.py` | PASS (α delivers) |
| `LaneTimingBudgets` exists | produced by α; `class LaneTimingBudgets` in `merge_lane/types.py` | PASS (α delivers) |
| Rejection: a pin below the governing merge-verify timeout fails `load_config` | produced by α; fixture `orchestrator/tests/fixtures/lane_budget_below_timeout.yaml` (row 1b), with row 1a loading the same pin | PASS (α delivers) |
| The invariant registered in the normative doc | produced by α; slug `lane-budgets-exceed-verify-timeouts` in `docs/legibility/design-invariants.md` | PASS (α delivers) |
| The leak reaper's thresholds are the grace and the reap floor, not the liveness window | `merge_lane/worker.py::SpeculativeMergeWorker.reap_orphaned_merge_worktrees` (startup sweep at the grace, periodic sweep at the reap floor); `liveness.py::INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS` read only by the intake dedupe gate, the heartbeat-floor guard and the stuck-request report | PASS (substrate) |

## β1 — `MergeOutcome.kind` and `GroupMergeRequest.train_kind`

| Capability | Binding | Verdict |
|---|---|---|
| Producers already know the kind | `orchestrator/src/orchestrator/merge_lane/gates.py::GateVerdict.block(emit_subtype: OutcomeKind)`; `merge_lane/types.py::OutcomeKind` (23 members, 4 non-terminal) | PASS (substrate; manual) |
| The reason strings are a frozen contract | `plans/merge-lane-quality-prd.md` §Contract "Frozen public surface" | PASS (substrate) |
| A producer-stated fact already rides `MergeOutcome` in this shape | `merge_lane/outcome_facts.py::MergeBlockCategory` carried as `MergeOutcome.block_category` (task 7302) | PASS (substrate) |
| `TrainKind` exists and the formers set it | produced by β1; `class TrainKind` in types.py | PASS (β1 delivers) |
| Workflow does not route on a reason prefix | produced by β1; `result\.reason\.startswith` absent from workflow.py | PASS (β1 delivers) |
| No code routes on the coalesce id prefix | produced by β1; `startswith\(_COALESCE_TRAIN_ID_PREFIX\)` absent from worker.py | PASS (β1 delivers) |
| Rejection: an unmapped prefix fails the equivalence table test | produced by β1; the table lives in `orchestrator/tests/test_outcome_kind.py` (boundary row 3) | PASS (β1 delivers) |

## β2 — `kind` mandatory for blocked/error

| Capability | Binding | Verdict |
|---|---|---|
| One test builder for failure outcomes | produced by β2; `def blocked_outcome\(` in `orchestrator/tests/_merge_lane_fakes.py` | PASS (β2 delivers) |
| Rejection: an unclassified failure outcome is refused at construction | produced by β2; boundary row 3b | PASS (β2 delivers; manual) |

## γ1 — `LandedOutbox` injected by the harness

| Capability | Binding | Verdict |
|---|---|---|
| The journal store exists with `lookup`/`consume`/`record` | `orchestrator/src/orchestrator/merge_lane/landed_outbox.py::LandedOutbox` | PASS (substrate; manual) |
| Production builds `TaskWorkflow` through one factory | `workflow.py::build_workflow` (callers `harness.py`, `evals/runner.py`) | PASS (substrate) |
| The harness is the one construction site of the worker | `harness.py` `self._merge_worker = SpeculativeMergeWorker(` | PASS (substrate) |
| The class global is gone | produced by γ1; `MergeProvenance` absent from `orchestrator/src/` | PASS (γ1 delivers) |
| The autouse rebinding is gone | produced by γ1; `_isolated_merge_provenance` absent from `orchestrator/tests/conftest.py` | PASS (γ1 delivers) |
| Producer of landed rows under the real trigger (INV-13) | `worker.py::_journal_landed_then_advance` writes the row before the CAS advance; unchanged; observed after restart | PASS (manual) |

## γ2 — `landed_outbox` required on `TaskWorkflow`

| Capability | Binding | Verdict |
|---|---|---|
| Rejection: an absent journal is a construction-time `TypeError` | produced by γ2; boundary row 5 second half | PASS (γ2 delivers; manual) |

## δ — One `MainHealthAutoHeal`

| Capability | Binding | Verdict |
|---|---|---|
| Both arms exist today and follow the same six steps | `workflow.py::TaskWorkflow._auto_heal_main_health` (109 lines) and `worker.py::SpeculativeMergeWorker._auto_heal_main_health_deferred` (89), which 5040's part 2 moves into `main_health_probe.py::MainHealthResponder` | PASS (substrate; no check — δ deletes one copy and builds on wherever 5040 left the other) |
| Escalation filing primitive with dedup | `escalation/src/escalation/dedupe.py::submit_or_dedupe`, `DedupeConfig` (called directly; `merge_lane/ports.py::EscalationPort` not widened) | PASS (substrate) |
| 5040 lands first | producer = task 5040 (pending, half-built branch); dependency edge (S6-2 = B) | PASS (upstream task, edge only) |
| The collaborator exists in the lane | produced by δ; `class MainHealthAutoHeal` in `merge_lane/main_health.py` | PASS (δ delivers) |
| The workflow's copy is deleted | produced by δ; `async def _auto_heal_main_health\(` absent from workflow.py | PASS (δ delivers) |
| No lane module imports workflow | produced by δ; `from orchestrator\.workflow import` absent under `merge_lane/` (today worker.py; after 5040 possibly `main_health_probe.py`) | PASS (δ delivers) |

## θ — A lane's halt is its holder set

| Capability | Binding | Verdict |
|---|---|---|
| The halt state is one object | `merge_lane/halt.py::HaltState` (task 5305) | PASS (substrate; manual) |
| Halt kind and rehydration hardening land first | producer = tasks 4838 and 3876 (pending); dependencies at decompose | PASS (upstream tasks filed) |
| Holders are added and released per lane | produced by θ; `def release\(self, lane` in halt.py | PASS (θ delivers) |
| The `lifecycle_refused` holder kind exists for φ | produced by θ; `lifecycle_refused` in halt.py | PASS (θ delivers) |
| Rejection: a train advance failure holds before filing; a failed filing releases that holder only and returns `FilingFailed` (never raises into the lane; the train keeps `_map_advance_failure`'s outcome) | produced by θ; boundary rows 13b, 13c (sync `hold` with a pre-minted id, then `file_holder_escalation` through plain `queue.submit`, called synchronously on the loop; re-key on a different returned id) | PASS (θ delivers; judged by the halt boundary tests) |
| No assert-based owner check | produced by θ; `^\s+assert ` absent from halt.py | PASS (θ delivers) |

## ζ — `ItemLifecycle` operative; one `retire`; refusals typed

| Capability | Binding | Verdict |
|---|---|---|
| The registry and legal-transition table exist | `worker.py::ItemLifecycle`, `_LEGAL_TRANSITIONS`, `types.py::ItemLifecycleState` | PASS (substrate; manual) |
| The intake already declares a lifecycle hop seam | `merge_lane/intake_buffers.py::IntakeLifecycle` (task 5305) | PASS (substrate) |
| The refusal alarm's event shape is the census's counting rule | `worker.py::_alarm_illegal_lifecycle_transition` emits `escalation_created` with `request_id`, `from_state`, `to_state` (2 on dark-factory, 23 on reify, all time) | PASS (substrate) |
| 5040 lands first | producer = task 5040; dependency edge (S6-2 = B) | PASS (upstream task, edge only) |
| The registry has its own module and a `retire` chokepoint | produced by ζ; `def retire\(` in `merge_lane/item_lifecycle.py` | PASS (ζ delivers) |
| Sync terminal-edge sites keep calling the registry | produced by ζ; `retire` is synchronous, hooks drained by `drain_hooks()` (row 8c) | PASS (ζ delivers; judged by the lifecycle boundary tests) |
| A re-entrant duplicate is resolved by identity | produced by ζ; `def resolve_duplicate\(` in `merge_lane/item_lifecycle.py` | PASS (ζ delivers) |
| No raw future resolution outside the registry | produced by ζ; code-line `\.result\.set_result\(` absent from worker.py, intake_buffers.py, intake.py, suffix_graph.py and 5040's verify_dispatch.py (the AST guard covers the package) | PASS (ζ delivers) |
| Rejection: an illegal transition is refused and typed | produced by ζ; boundary row 9 | PASS (ζ delivers; manual) |

## φ — The lifecycle-refusal policy, its hold arm, the census and the decision-4 amendment

| Capability | Binding | Verdict |
|---|---|---|
| Red-tier config values are read once at worker start | `config.py::RELOADABLE_FIELDS` (a union of green-tier leaves; a key outside it reports `restart_required` on reload) | PASS (substrate) |
| A predicate script over `runs.db` has a house shape | `scripts/merge_deep_canary_predicate.py` (`--runs-db`, `--window-days`, `--now`, read-only `mode=ro`) | PASS (substrate) |
| Decision 4 already carries a dated amendment to sit beside | `plans/merge-queue-modularization-invariants-prd.md` decision 4, "Amendment (task 3622, 2026-08-17)" | PASS (substrate) |
| The policy is a config value | produced by φ; `lifecycle_refusal_policy` in config.py | PASS (φ delivers) |
| The census exists for ξ | produced by φ; path `scripts/lifecycle_refusal_census.py` (row 18) | PASS (φ delivers) |
| Decision 4 names the exception | produced by φ; `lifecycle_refused` in `plans/merge-queue-modularization-invariants-prd.md` | PASS (φ delivers) |
| Rejection: under `hold_lane` a refusal holds its lane before filing, freezes the request, adds one holder per lane per refused edge, survives restart | produced by φ; boundary rows 9b–9e | PASS (φ delivers; manual) |

## κ — `LaneLimits` and `RetryBudget`; narrow host

| Capability | Binding | Verdict |
|---|---|---|
| S4 C1 `VerifyContext` and S5 η/θ land first | producer = S4 leaf β, S5 leaf θ; dependency edges | PASS (upstream leaves, edge only) |
| `VerifyDispatcher` exists to receive the limits | producer = task 5040 (κ of `plans/merge-lane-quality-prd.md`, pending, half-built branch); the edge is the gate | PASS (upstream task, edge only; manual) |
| Test seams reach 0 patch targets before the constants move | producer = tasks 5446 and 5777 (pending); dependencies at decompose | PASS (upstream tasks filed) |
| `LaneLimits` exists | produced by κ; `class LaneLimits` in types.py | PASS (κ delivers) |
| The 19 constants α leaves (18 after S7 δ) move onto `LaneLimits` | produced by κ | PASS (κ delivers) |
| No upper-case int class constant remains in the package | produced by κ; `MAX_CAS_RETRIES\s*:\s*int` absent under `merge_lane/` | PASS (κ delivers) |
| One retry ledger | produced by κ; `class RetryBudget` in types.py | PASS (κ delivers) |

## λ — Drift and shadow own their state

| Capability | Binding | Verdict |
|---|---|---|
| Both satellites take the worker today | `merge_lane/drift.py::_maybe_run_drift_check`, `merge_lane/shadow.py::_maybe_schedule_shadow_compare` (12 + 10 `worker._` lines at `38a200d213`) | PASS (substrate) |
| Neither reads the worker | produced by λ; `worker\._drift_land_count` absent from drift.py; `worker\._shadow_state_path` absent from shadow.py | PASS (λ delivers) |

## μ — S3's `MergeLaneQueries` built natively; `LIVE_STATE_OF`

| Capability | Binding | Verdict |
|---|---|---|
| The port definition exists | producer = S3's port task α (C1, `merge_lane/queries.py`: Protocol, records, `QueryUnavailable`; no adapter, S3-1 = A); the edge is the gate | PASS (upstream leaf, edge only) |
| The lane implements it | produced by μ; `class LaneQueries` in `merge_lane/queries.py` | PASS (μ delivers) |
| One total internal→public state table | produced by μ; `^LIVE_STATE_OF` in `merge_lane/queries.py`, exhaustiveness-tested (row 17) | PASS (μ delivers) |

## ι — Lane integration gate

| Capability | Binding | Verdict |
|---|---|---|
| The ratchet instrument exists | `scripts/merge_lane_metrics.py`, `orchestrator/tests/test_merge_lane_ratchet.py`, `orchestrator/tests/merge_lane_ratchet_baseline.json` | PASS (substrate; manual) |
| The residual worker and the merge-lane-quality gate land first | producer = tasks 5045, 5047, 5048 (pending); dependencies at decompose | PASS (upstream tasks filed) |
| S7's placement deletions land first | producer = S7 leaf β; dependency wired at decompose | PASS (upstream leaf, edge only) |
| The gate module exists | produced by ι; `orchestrator/tests/test_merge_lane_s6_integration_gate.py` present | PASS (ι delivers) |
| S6's boundary rows hold through the façade | produced by ι; ceilings cited from 5048, not restated | PASS (ι delivers; manual) |

## ξ — Lifecycle-refusal quiet period (deterministic predicate gate; filed by the lead once φ is on main)

| Capability | Binding | Verdict |
|---|---|---|
| A delayed milestone with a predicate verdict is a supported task shape | `docs/task-authoring.md` §6 (milestone `delayed`, `before_done.kind='predicate'`, exit 0 → done, other → born-at-L2 `milestone_check_failed`) | PASS (substrate) |
| The census is on main before the gate is filed | producer = φ; the lead's filing order and the `add_dependency(ξ, φ)` edge | PASS (S6 leaf, edge; manual) |

## υ — Arm `hold_lane` on dark-factory (filed by the lead with ξ)

| Capability | Binding | Verdict |
|---|---|---|
| The gate passed | producer = ξ; dependency edge | PASS (S6 leaf, edge only) |
| dark-factory runs `hold_lane` | produced by υ; `lifecycle_refusal_policy: hold_lane` in `dark-factory-orchestrator.yaml` | PASS (υ delivers) |
