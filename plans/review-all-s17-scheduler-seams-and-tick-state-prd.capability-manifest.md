# Capability manifest — S17 scheduler seams and tick state

PRD: `plans/review-all-s17-scheduler-seams-and-tick-state-prd.md`. YAML twin:
`plans/review-all-s17-scheduler-seams-and-tick-state-prd.capability-manifest.yaml`.

**Code anchors** verified against main `a85cb1ad78` (2026-11-13; ruling fold of the sitting of 2026-11-12) and
re-run at `fceb951e7e` (2026-11-13, fold-verify).
Bindings cite symbols, not lines. Every grep and path check in the sidecar was evaluated at `a85cb1ad78` with
the runtime's own
`shared.delivered_check_polarity.build_grep_argv`/`interpret_grep_rc`/`build_path_argv`/`interpret_path_listing`:
all 64 producer checks FAIL today (re-run at `fceb951e7e`) and pass once their leaf lands; the 7 substrate checks (named
`*-substrate`, "substrate" in the binding) PASS today by design and are never copied to a task (the polarity
gate refuses them as `vacuous_present`, the wave-2 precedent). `shared.delivered_check_polarity.lint_delivered_checks`
at `a85cb1ad78` (and again at `fceb951e7e` for ζ, δ2 and λ), linted against each changed leaf's declared files from the PRD (α0, β, ζ, δ2, λ, ν) and
without files for the rest: no finding. The sidecar loads through
`shared.capability_manifest.CapabilityManifestDoc` (22 task blocks, `signal_leaves: ["ω", "ζ", "δ2", "ν"]`,
every `task_id: null`). No row is OPEN: S17-OQ-9 (how the gate records transient windows 1 and 2) was ruled A
by the lead on 2026-11-13, so δ2's behavioural row is PASS with its manual check. Every question is ruled. ω's zero check greps
`PRIVATE_REACH_PATTERN`, the same pattern the D2 guard counts (1,479 lines at `a85cb1ad78`), so the trigger
gate's view and the guard's agree. Task 4082 (re-scoped by the sitting) is not a leaf of this plan and has no
block here; ζ's behavioural row asserts its rows.

## Substrate (passes today)

| Capability | Binding | Verdict |
|---|---|---|
| Typed snapshot the test seam extends | `orchestrator/src/orchestrator/scheduler.py::Scheduler.get_state_snapshot` returning `shared/src/shared/scheduler_state.py::SchedulerStateSnapshot` (S14 η 6717, done) | PASS |
| Forced snapshot write for tests | `scheduler.py::Scheduler.flush_state_snapshot`; constructor `state_snapshot_path=` | PASS |
| Public seeding calls | `scheduler.py::Scheduler.acquire_next`, `::release`, `::record_requeue`, `::note_workflow_cancelled`, `::note_merge_phase_entered`; `overrides.py::OverrideStore.set_override` (moves to `shared/scheduler_overrides.py` with S14 θ 6718) | PASS |
| Session seam for the in-memory store | `scheduler.py::McpSessionLike.call_tool`; constructor `mcp_session=`; `Scheduler.dispatch_tool` | PASS |
| Tri-state dependency verdict to type | `orchestrator/src/orchestrator/dependency_verdict.py::deps_verdict`, `::DepsVerdict`, `::DepState`, `::DepKind` (task 6518) | PASS |
| Recovery classifier the scheduler defers to | `orchestrator/src/orchestrator/task_ground_truth.py::TaskGroundTruth.recovery_for` (S8 δ 9219 reshapes it into `RECOVERY_RULES`) | PASS |
| Single-check evaluator `run` wraps | `orchestrator/src/orchestrator/delivered_checks.py::run_delivered_check` (never raises; ERRORED on malformed/runner error) | PASS |
| Escalate-once streak primitive | `shared/src/shared/streak.py::KeyedStreaks` (S14 I7) | PASS |
| Typed task metadata | `shared/src/shared/task_metadata.py::parse_metadata`, `::TaskMetadata`, `::BeforeDone` | PASS |
| Deploy phase machine | `shared/src/shared/deploy_state.py::DeployState`, `::DeployPhase`, re-exported by `orchestrator/src/orchestrator/deploy_state.py` (tasks 2239/2240) | PASS |
| Markup sink factories | `orchestrator/src/orchestrator/mcp/markup_sink.py::make_escalation_sink`, `::resolve_project_root`, `::open_escalation_channel`; `mcp/markup_journal.py::make_fact_journal` | PASS |
| Fused-memory's dependency-id reader α0 deletes | `fused-memory/src/fused_memory/utils/task_dependency_ids.py::task_dependency_ids` (task 3615; 4 importers) | PASS |
| Recovery table 4082 extends; its applier | `orchestrator/src/orchestrator/task_ground_truth.py::TaskGroundTruth.recovery_for` (S8 δ 9219 lands `RECOVERY_RULES`); `StrandedReconciler` (S8 ε2a 9223) | PASS (`recovery_for`; the rest is upstream, filed) |
| Phase timing λ reads | `orchestrator/src/orchestrator/scheduler.py::Scheduler._emit_tick_telemetry` (the `scheduler_tick` event's `phases` map) | PASS |
| Invariant family guard ν extends | `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES` | PASS |

## Per leaf

### α0 — One shared reader for a task's dependency ids, train id and deterministic kind; fused-memory's own copy is deleted

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| task-fields-module-exists | capability→producer (C11) | PASS | path `shared/src/shared/task_fields.py` present |
| dependency-ids-reader | capability→producer (C11) | PASS | grep `def dependency_ids` in `shared/src/shared/task_fields.py` present |
| fused-memory-copy-deleted | anti-inversion (C11; lead S17/OQ-4, no re-export; INV-9) | PASS | path `fused-memory/src/fused_memory/utils/task_dependency_ids.py` absent |
| fused-memory-importers-use-the-shared-reader | capability→consumer (C11) | PASS | grep `from shared\.task_fields import dependency_ids` in the four importers (`recon_write_policy.py`, `task_curator.py`, `targeted.py`, `live_workflow_detector.py`) present |

### β — Scheduler test builder, in-memory task store, per-task snapshot section, private-reach guard

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| scenario-builder-exists | capability→producer (C2) | PASS | grep `class SchedulerScenario` in `orchestrator/tests/_scheduler_scenario.py` present |
| in-memory-task-store-at-the-session-boundary | capability→producer (C2, D3) | PASS | grep `class InMemoryTaskStore` in `orchestrator/tests/_in_memory_task_store.py` present |
| snapshot-carries-task-lifecycle | capability→producer (C1; lead S17/OQ-7) | PASS | grep `class TaskLifecycle` in `shared/src/shared/scheduler_state.py` present |
| psi-reader-constructor-seam | capability→producer (task 6143 item) | PASS | grep `psi_reader: Callable` in `orchestrator/src/orchestrator/scheduler.py` present |
| private-reach-guard-exists | rejection-mechanism (D2) | PASS | path `orchestrator/tests/test_scheduler_private_reach.py` present |
| snapshot-seam-substrate | substrate | PASS | grep `def get_state_snapshot` in `scheduler.py` present (passes today) |

### ι1 — test_scheduler.py's lifecycle, constructor and streak reaches onto the builder

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| test-scheduler-off-dispatched-set | anti-inversion | PASS | grep `scheduler\._dispatched` in `orchestrator/tests/test_scheduler.py` absent |
| test-scheduler-off-private-callbacks | anti-inversion | PASS | grep `scheduler\._callbacks` in `orchestrator/tests/test_scheduler.py` absent |

### ι2 — The other lifecycle-heavy scheduler test files onto the builder

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| retry-cap-tests-off-privates | anti-inversion | PASS | grep `scheduler\._` in `orchestrator/tests/test_retry_cap.py` absent |
| dispatch-admission-tests-off-privates | anti-inversion | PASS | grep `scheduler\._` in `orchestrator/tests/test_scheduler_dispatch_admission.py` absent |
| deterministic-tests-off-privates | anti-inversion | PASS | grep `real_scheduler\._` in `orchestrator/tests/test_scheduler_deterministic.py` absent |

### ι3 — Remaining reaches outside the scheduler's own test files onto the builder

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| outside-files-off-lifecycle-containers | anti-inversion | PASS | grep `(sched[A-Za-z_]*\|Scheduler)\._(dispatched\|requeue_until\|module_cache\|skip_count\|external_unresolved_counts)([^A-Za-z0-9_]\|$)` in `_orch_helpers.py`, `test_cross_project_dispatch_integration.py`, `test_repend_state_machine.py`, `test_warm_lane_structural_exhaustion.py` absent |
| deterministic-dispatch-off-deps-satisfied | anti-inversion | PASS | grep `scheduler\._deps_satisfied` in `orchestrator/tests/test_harness_deterministic_dispatch.py` absent |

### α — Scheduler-side readers use the shared readers

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| scheduler-dep-id-copies-gone | anti-inversion (5 sites at `a85cb1ad78`, all dependency-id normalisations) | PASS | grep `isinstance\(_?d, dict\)` in `scheduler.py` absent |
| inline-deterministic-kind-tests-gone | anti-inversion (the method stays and delegates; corrected at the ruling fold: the harness calls it and 27 test files mock it) | PASS | grep `get\('task_kind'\) == 'deterministic'` in `scheduler.py`, `harness.py` absent (4 matching lines at `a85cb1ad78`) |
| scheduler-imports-the-readers | capability→consumer (C11) | PASS | grep `from shared.task_fields import` in `scheduler.py` present |

### ζ — The scheduler stops re-pending stranded blocked tasks; the recovery table's rows decide them

Ruled by Leo (S17-1 option D; S17-4 A plus a third row; R-2 = A). Task 4082's rows land first (ζ → 4082).

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| stranded-phase-deleted | anti-inversion | PASS | grep `def _phase_redispatch_stranded_blocked` in `scheduler.py` absent |
| redispatch-kill-switch-deleted | anti-inversion | PASS | grep `blocked_redispatch` in `orchestrator/src/orchestrator/config.py` absent |
| clean-strand-re-pended-by-a-code-row | behavioural (boundary rows 14, 14b–14e, 14g, 15; the cap cannot loop) | PASS | manual: `test_stranded_blocked_one_decider.py` |
| flip-log-helper-deleted | anti-inversion (task 6145's helper goes with the phase) | PASS | grep `def _describe_records_flipped_past` in `scheduler.py` absent |
| recovery-classifier-substrate | substrate | PASS | grep `async def recovery_for` in `task_ground_truth.py` present (passes today) |

### ζ2 — Documents stop saying the scheduler re-pends stranded blocked tasks, and name `deferred` as the park

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| architecture-doc-corrected | anti-inversion (covers the tick-order line, which has no `_phase_` prefix) | PASS | grep `redispatch_stranded_blocked` in `ARCHITECTURE.md` absent |
| transition-actor-note-corrected | anti-inversion | PASS | grep `_phase_redispatch_stranded_blocked` in `shared/src/shared/task_transitions.py` absent |

### γ — One dependency predicate over a required `DepGates`

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| dep-gates-value | capability→producer (C3) | PASS | grep `class DepGates` in `dependency_verdict.py` present |
| boolean-wrapper-gone | anti-inversion | PASS | grep `def _deps_satisfied` in `scheduler.py` absent |
| optional-cache-parameters-gone | anti-inversion | PASS | grep `external_status_cache: dict` in `dependency_verdict.py` absent |
| verdict-equivalence | behavioural (boundary row 5) | PASS | manual: `test_dep_gates.py` golden table |

### δ1 — `delivered_checks.run` is the one evaluator

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| one-evaluator-function | capability→producer (C4) | PASS | grep `^async def run\(` in `delivered_checks.py` present |
| mirror-comment-gone | anti-inversion | PASS | grep `mirror scheduler.py` in `delivered_checks.py` absent |
| second-aggregator-gone | anti-inversion | PASS | grep `def verify_delivered_checks_on_main` in `delivered_checks.py` absent |
| scheduler-calls-no-single-check-runner | anti-inversion | PASS | grep `run_delivered_check` in `scheduler.py` absent |
| landing-differential-through-run | anti-inversion | PASS | grep `run_delivered_check` in `merge_lane/landing_evidence.py` absent |

### δ2 — A check that cannot be evaluated is reported and escalated

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| gate-collaborator-module | capability→producer (C5) | PASS | grep `class DeliveredCheckGate` in `orchestrator/src/orchestrator/delivered_check_gate.py` present |
| errored-event-type | capability→producer (C5) | PASS | grep `delivered_check_errored` in `event_store.py` present |
| errored-grace-config-field | capability→producer (C5; S17-2 ruled) | PASS | grep `errored_grace_sweeps` in `orchestrator/src/orchestrator/config.py` present |
| log-only-threshold-gone | anti-inversion | PASS | grep `_DELIVERED_CHECK_ERROR_LOG_THRESHOLD` in `scheduler.py` absent |
| persistent-errored-escalates | behavioural (INV-11, INV-7; rows 8b, 10, 10b, 10c, 10d, 11; transient windows counted and cleared by the gate, no queue record for windows 1–2, born-at-L2 at the third; no watcher step; S17-OQ-9 = A) | PASS | manual: `test_delivered_check_errored.py` |
| keyed-streaks-substrate | substrate | PASS | grep `class KeyedStreaks` in `shared/src/shared/streak.py` present (passes today) |

### λ — Escalate to a human when a tick's delivered-check phase takes more than 30 seconds

Ruled by Leo 2026-11-13 (S17-2). Low. Born-at-L2 at severity `critical`, the lowest severity that reaches Leo's
queue at main, the floor of `BORN_AT_L2_SEVERITIES` (lead 2026-11-13, confirmed; PRD D15); one open record per project under a registered sentinel.

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| phase-alarm-threshold-setting | capability→producer (C13) | PASS | grep `phase_alarm_secs` in `orchestrator/src/orchestrator/config.py` present |
| gate-observes-the-phase-time | capability→producer (C13; INV-9, one measurement) | PASS | grep `def observe_phase_time` in `orchestrator/src/orchestrator/delivered_check_gate.py` present |
| phase-alarm-sentinel-registered | capability→producer (C13; S1's sentinel registry) | PASS | grep `__delivered_check_phase_slow__` in `escalation/src/escalation/models.py` present |
| slow-phase-reaches-a-human-once | behavioural (INV-4; row 11b; severity `critical`) | PASS | manual: `test_delivered_check_phase_alarm.py` |
| tick-telemetry-substrate | substrate | PASS | grep `def _emit_tick_telemetry` in `scheduler.py` present (passes today) |

### ε — Tick phases as a validated table of references

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| phase-table-type | capability→producer (C6) | PASS | grep `class TickPhase` in `orchestrator/src/orchestrator/tick_phases.py` present |
| string-phase-order-gone | anti-inversion | PASS | grep `_TICK_PHASE_ORDER` in `scheduler.py` absent |
| phase-order-validated | rejection-mechanism (rows 12, 12b, 13) | PASS | manual: `test_tick_phase_order.py` |

### η — One frozen record per task in a dispatch ledger

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| task-record-type | capability→producer (C7) | PASS | grep `class TaskRecord` in `orchestrator/src/orchestrator/dispatch_ledger.py` present |
| requeue-deadline-dict-gone | anti-inversion | PASS | grep `self._requeue_until` in `scheduler.py` absent |
| streak-aliases-gone | anti-inversion | PASS | grep `_external_unresolved_counts` in `scheduler.py` absent |

### θ — Backfill admission and the warm-base watchdog as collaborators

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| backfill-admission-module | capability→producer (C8) | PASS | grep `class BackfillAdmission` in `orchestrator/src/orchestrator/backfill_admission.py` present |
| warm-base-watchdog-module | capability→producer (C8) | PASS | grep `class WarmBaseHardDownWatchdog` in `orchestrator/src/orchestrator/warm_base_watchdog.py` present |
| inline-admission-gone | anti-inversion | PASS | grep `def _backfill_admission` in `scheduler.py` absent |

### κ1 — The shadow protocol goes

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| shadow-protocol-deleted | anti-inversion (R16) | PASS | grep `class SchedulerFacade` in `scheduler.py` absent |
| workflow-types-the-scheduler | anti-inversion (R16) | PASS | grep `SchedulerFacade` in `workflow.py` absent |

### κ2 — The hand-written FakeScheduler and TaskStoreDouble go

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| hand-fake-deleted | anti-inversion (R16) | PASS | grep `class FakeScheduler` in `orchestrator/tests/` absent |
| task-store-double-deleted | anti-inversion (R16) | PASS | grep `class TaskStoreDouble` in `orchestrator/tests/` absent |

### ω — Zero private Scheduler reaches (integration gate)

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| zero-private-scheduler-reaches | anti-inversion (D2; the guard's own source included, no file exempt) | PASS | grep `PRIVATE_REACH_PATTERN` (`(^\|[^A-Za-z0-9_])((real\|fresh)_)?(scheduler\|sched)\._[a-z]\|(^\|[^A-Za-z0-9_])Scheduler\._[A-Za-z]`) in `orchestrator/tests/` absent (1,479 matching lines at `a85cb1ad78`) |
| boundary-suite-green | integration gate (rows 1–22, with 10b–10d, 11b, 14b–14g) | PASS | manual |

### ξ1 — RunnerState and the gate and predicate steps

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| runner-state-type | capability→producer (C10) | PASS | grep `class RunnerState` in `orchestrator/src/orchestrator/deterministic_state.py` present |
| gate-step-module | capability→producer (D11) | PASS | path `orchestrator/src/orchestrator/deterministic_gate_step.py` present |
| raw-gate-stamp-read-gone | anti-inversion (one read, inside `run`, at `a85cb1ad78`) | PASS | grep `metadata\.get\('gate_escalated_at'\)` in `deterministic_runner.py`, `deterministic_state.py` and the gate, predicate and deploy step modules absent |
| parse-metadata-substrate | substrate | PASS | grep `def parse_metadata` in `shared/src/shared/task_metadata.py` present (passes today) |

### ξ2 — Deploy resume from the recorded phase

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| deploy-step-module | capability→producer (D11) | PASS | path `orchestrator/src/orchestrator/deterministic_deploy_step.py` present |
| legacy-stamp-mapping | capability→producer (D10; module-level function) | PASS | grep `def from_legacy_stamps` in `deploy_state.py` present |
| stamp-presence-resume-gone | anti-inversion (two reads, both inside `run`, at `a85cb1ad78`) | PASS | grep `metadata\.get\('before_done_ran_at'\)` in `deterministic_runner.py`, `deterministic_state.py` and the gate, predicate and deploy step modules absent |
| resume-equivalence | behavioural (row 19) | PASS | manual: `test_deterministic_deploy_resume.py` |

### μ — Markup channels as one bundle

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| markup-channels-type | capability→producer (C9) | PASS | grep `class MarkupChannels` in `mcp/markup_sink.py` present |
| global-root-resolver-gone | anti-inversion | PASS | grep `def _markup_project_root` in `mcp/plan_tools.py` absent |
| global-channel-opener-gone | anti-inversion | PASS | grep `def _escalation_channel` in `mcp/plan_tools.py` absent |
| escalation-sink-substrate | substrate | PASS | grep `def make_escalation_sink` in `mcp/markup_sink.py` present (passes today) |

### ν — Record Leo's principle as the design invariant one-decision-partition-per-area

Ruled by Leo 2026-11-13. The four-site lockstep edit; depends on task 4082 (house pattern); INV-23, minted last
of the six wave-3 invariant tasks (lead 2026-11-13); checks anchor on the slug. `INV-23-CODE` carries a row with
no decider, and the addendum walk shows the checkable question failing on it.

| Capability | Binding | Verdict | Check |
|---|---|---|---|
| invariant-in-the-normative-list | capability→producer | PASS | grep `one-decision-partition-per-area` in `docs/legibility/design-invariants.md` present |
| invariant-fixtures-prove-the-negative | capability→producer (violating PRD and code fixtures, addendum walk) | PASS | grep `one-decision-partition-per-area` in `docs/legibility/design-invariants-fixtures.md` present |
| invariant-in-the-g7-gate | capability→consumer (/prd G7 walks it) | PASS | grep `one-decision-partition-per-area` in `skills/prd/references/gates.md` present |
| family-consistency | behavioural (row 23) | PASS | manual: `scripts/tests/test_design_invariants_consistency.py` passes |
| consistency-guard-substrate | substrate | PASS | grep `^PINNED_SITES = ` in `scripts/tests/test_design_invariants_consistency.py` present (passes today) |
