# Capability manifest — S8 harness composition root and slot registry

PRD: `plans/review-all-s8-harness-composition-root-and-slot-registry-prd.md`. YAML twin:
`plans/review-all-s8-harness-composition-root-and-slot-registry-prd.capability-manifest.yaml`.

**Code anchors** verified against main `e634cfe658` (2026-11-10), after the rulings of the wave-2
sitting of 2026-11-10 were applied (the PRD's §Rulings). Bindings cite symbols, not lines. A
substrate row passes today and says so; in the sidecar a substrate row carries
`delivered_check: {kind: manual, reason: "substrate — green by design (deliberation.md §Launching 4c)"}`,
never a presence grep the polarity gate rejects. A producer row's `delivered_check` fails at
`e634cfe658` and passes once its leaf lands (every grep and path check evaluated at `main` on
2026-11-10, re-run at `bb1c659562` in the fold-verify pass with the three checks it added: all 83 fail
today; polarity lint `shared.delivered_check_polarity.lint_delivered_checks`
at `main`: no `vacuous_present` / `vacuous_absent` finding; the 42 `absent_overbroad` notes are
artefacts of linting without `metadata.files`; linted against each leaf's declared files from the
PRD, two remain, both on ω and both covered by the earlier leaves that declare those test files). The
sidecar loads through
`shared.capability_manifest.CapabilityManifestDoc` (22 task blocks, `signal_leaves: ["ω"]`).

## Substrate (passes today)

| Capability | Binding | Verdict |
|---|---|---|
| Frozen hook bundle on the Scheduler to extend | `orchestrator/src/orchestrator/scheduler.py::SchedulerCallbacks` (12 hooks; `Scheduler.__init__(callbacks=)`) | PASS (manual) |
| Ordered lifecycle with reverse-order stop | `orchestrator/src/orchestrator/background_service.py::LifecycleRegistry`, `::ManagedService` (task 2241); `stop_all` stops every registered service in reverse order whether or not it started | PASS (manual) |
| Startup-step table the composition root keeps | `orchestrator/src/orchestrator/startup_steps.py::StartupStep`, `::run_startup_step` (task 8267; `Harness.run` drives steps `1b`…`2g` through it) | PASS |
| One recovery classifier to grow rows onto | `orchestrator/src/orchestrator/task_ground_truth.py::TaskGroundTruth.recovery_for`, `::_RECOVERY`, `::classify_recovery` | PASS (manual) |
| Named pin predicates for condition columns | `orchestrator/src/orchestrator/recovery_pins.py::records_pin_recovery`, `::records_pin_blocked_recovery`, `::records_pin_blocked_done_flip`; `task_ground_truth.py::report_blocked_arm_would_move` | PASS |
| Identity-guarded slot release to generalise | `orchestrator/src/orchestrator/slot_lease.py::SlotLease`; `harness.py::Harness._unregister_slot`, `::_owns_slot` (task 6660) | PASS |
| Cancel-recency stamp the registry's cancel keeps | `scheduler.py::Scheduler.note_workflow_cancelled`, `::workflow_cancel_recent` | PASS |
| Already-landed gate protocol | `orchestrator/src/orchestrator/already_landed_tick_view.py::AlreadyLandedGate` (`LandedOutboxGate` does not exist; β defines it) | PASS |
| Pause class already typed and persisted | `orchestrator/src/orchestrator/pause_policy.py::PauseClass`; `run_store.py` `scheduler_state.pause_class` (task 4686, `730794ca3b`) | PASS |
| Transition legality table | `shared/src/shared/task_transitions.py::TRANSITIONS`, `::ActorClass` (`cb99e2c70e`, `8b3fbf5bc0`, 2026-07-07); `fused-memory/config/config.yaml` `enforce_transitions: true` | PASS |
| Durable per-task guard store for θ | `orchestrator/src/orchestrator/guard_state.py::PersistentMap` (TTL tracks the last bump), `::guard_path`; `steward.py::_STEWARD_GUARD_TTL` = 7 days (task 5352) | PASS (manual) |
| Unknown-key escalation for a deleted config key | `harness.py::Harness._file_config_unknown_keys_escalation` (startup step `1c0-config-keys`; born-at-L2, self-closing; `OrchestratorConfig` ignores unknown keys) | PASS |
| Born-at-L2 dedup substrate for an orchestrator alarm | `orchestrator/src/orchestrator/born_at_l2.py::file_deduped_born_at_l2`, `::BornAtL2Alarm` (task 2987) | PASS |
| Typed sqlite error codes for the lock-only retry | `sqlite3.Error.sqlite_errorcode`, `sqlite3.SQLITE_BUSY`, `sqlite3.SQLITE_LOCKED` (CPython ≥ 3.11; the venv is 3.13) | PASS |
| Shared claude invoker accepts a confinement closure | `shared/src/shared/cli_invoke.py::invoke_claude_agent(sandbox_wrap=)`, `::_invoke_claude(sandbox_wrap=)` | PASS (manual) |
| Scheduler already accepts an injected session | `scheduler.py::Scheduler.__init__(mcp_session=)`; `::dispatch_tool` | PASS (manual) |
| UsageGate already accepts a cost sink | `shared/src/shared/usage_gate.py::UsageGate.__init__(cost_store=)` | PASS |
| Agent child env rule for the non-claude backends | `shared/src/shared/secret_env.py::agent_subprocess_env` (task 6820) | PASS |
| Thread-off-loop emit and read callers the writer must serve | `escalation/src/escalation/server.py` `event_store.emit(` and 5 `EventStore` read calls inside FastMCP sync tool bodies | PASS |
| Cheap queue construction | `escalation/src/escalation/queue.py::EscalationQueue.__init__` (mkdir plus caches, re-read at `e634cfe658`) | PASS |

Consumed from sibling drafts (producer-absent at `e634cfe658`; the consuming leaf depends on the
producer if it is filed first): S9 C1 `PlanState` (ε1a), C5 `classify_merge_future`/`file_stranded_merge_failed`
(ε2b), C6 `corroborated_landing` (δ), C7 `RestoreOutcome`/`ResumeFailure`/`ResumeOutcomeSink` (γ; γ and
ε1a are written for both orders); S6 C4 `LandedOutbox` (γ1, γ2), C6 `LaneTimingBudgets`, `LaneLimits`
(β, δ); S5 C2 `GitOpsCallbacks`, C6 `LaneLifecycle` (β; OQ-10 ruled (a)); S3 C2 `WorkflowSlots` (α
implements it; S3 OQ-7 ruled (a)), C2 `HarnessOps`, C4 `set_resolve_callback(deliver_on=)` (β), C3
`LadderLookups` (S3 ζ binds it; ω deletes the forwarder after it); task 8926's file-or-refresh helper
(β); task 8563's `landings()` digest count (ι2).

## η — `convert_to_blocked_enforce` defaults to True (S8-7, step 1)

| Capability | Binding | Verdict |
|---|---|---|
| Log mode is no longer the shipped default | `is LOG MODE: the reconcile sweep logs` absent from `config.py` (stays satisfied after δ deletes the field) | PASS (η delivers) |
| Enforce is the default | boundary row 12a (manual: a grep would go vacuous once δ deletes the field) | PASS (η delivers) |

## α — SlotRegistry

| Capability | Binding | Verdict |
|---|---|---|
| Registry module exists | path `orchestrator/src/orchestrator/slot_registry.py` | PASS (α delivers) |
| Registry implements S3 C2 `WorkflowSlots` | `async def wait_cleared` in `slot_registry.py` | PASS (α delivers) |
| Terminal scan never cancels a live deterministic slot (N2) | boundary row 1c (manual) | PASS (α delivers) |
| Duplicate registration is typed (D1a) | `class SlotAlreadyLive` in `slot_registry.py` | PASS (α delivers) |
| One liveness definition | `_workflow_cancel_events` and `_escalation_events` absent from `harness.py` | PASS (α delivers) |
| Force-released late report dropped by `release()` (D2a) | `_force_released_slot_tasks` absent from `harness.py` | PASS (α delivers) |
| Lease folded into the record | `slot_lease.py` deleted; `SlotLease` defined in `slot_registry.py` | PASS (α delivers) |
| Deterministic slots are live to `release_workflow`; cancel refused (S8-4) | boundary row 1 (manual) | PASS (α delivers) |

## β0 — Scheduler test factory

| Capability | Binding | Verdict |
|---|---|---|
| Factory exists | path `orchestrator/tests/_scheduler_factory.py` | PASS (β0 delivers) |
| Named passthrough gates | `def passthrough_gates` in `scheduler.py` | PASS (β0 delivers) |
| Shim gone | `def _workflow_cancel_at` absent from `harness.py` | PASS (β0 delivers) |

## β — composition root

| Capability | Binding | Verdict |
|---|---|---|
| Required gates on the bundle | `^\s+already_landed_gate: AlreadyLandedGate\s*(#.*)?$` and `^\s+landed_outbox_gate: LandedOutboxGate\s*(#.*)?$` in `scheduler.py` | PASS (β delivers) |
| No post-construction gate install | `self.scheduler._landed_outbox_gate =` absent from `harness.py` | PASS (β delivers) |
| No run()-time collaborator writes | `self.usage_gate._cost_store` absent from `harness.py` | PASS (β delivers) |
| Teardown by registration order | `await self.usage_gate.shutdown()` absent; `name='usage-gate'` present in `harness.py` | PASS (β delivers) |
| No module-global hook write from `__init__` | `set_archival_failure_hook` absent from `harness.py` | PASS (β delivers) |
| No optional escalation import (D6, 6625) | `HAS_ESCALATION` absent from `harness.py` | PASS (β delivers) |
| Stores fail closed (D4a; S8-8) | `Failed to create (event\|cost) store\|persistence is DISABLED` absent from `harness.py` | PASS (β delivers) |
| A store failure files one critical born-at-L2 and raises; a locked db is retried (C3b) | boundary rows 8b, 8c (manual) | PASS (β delivers) |

## δ — RECOVERY_RULES

| Capability | Binding | Verdict |
|---|---|---|
| Rule table exists | `class RecoveryRule` in `task_ground_truth.py` | PASS (δ delivers) |
| Handlers return a typed outcome (D16) | `class HandlerOutcome` in `task_ground_truth.py` | PASS (δ delivers) |
| Equivalence with today's classifier (D7) | path `orchestrator/tests/test_recovery_rules_equivalence.py` | PASS (δ delivers) |
| No sweep-side rewrite | `action = RecoveryAction.MARK_DONE_WITH_PROVENANCE` absent from `harness.py` | PASS (δ delivers) |
| Sweep reads no config | `convert_to_blocked_enforce` absent from `harness.py` | PASS (δ delivers) |
| The field is deleted (S8-7 option B; D18 step 3) | `convert_to_blocked_enforce` absent from `config.py`; `^convert_to_blocked_enforce:` absent from `dark-factory-orchestrator.yaml` | PASS (δ delivers) |
| 7277's guard is a truth fact | `merge_in_flight: bool` in `task_ground_truth.py`; boundary row 13 | PASS (δ delivers; depends on 7277) |

## γ — SessionResumeStorm

| Capability | Binding | Verdict |
|---|---|---|
| Module exists | path `orchestrator/src/orchestrator/session_resume_storm.py` | PASS (γ delivers) |
| Reach-back gone | `from orchestrator\.harness import` absent from `storm_window_bound.py` | PASS (γ delivers) |
| Vocabulary has one home | `_BY_DESIGN_SESSION_RESUME_REASONS` absent from `harness.py` | PASS (γ delivers) |
| Log-once latch has a home | `_archive_available_fault_logged` absent from `harness.py` | PASS (γ delivers) |

## ε1a — CrashRecovery

| Capability | Binding | Verdict |
|---|---|---|
| Module exists | path `crash_recovery.py` | PASS (ε1a delivers) |
| One lane decision (C7) | `def decide_lane\(` in `crash_recovery.py` | PASS (ε1a delivers) |
| Method left harness | `async def _recover_crashed_tasks` absent from `harness.py` | PASS (ε1a delivers) |
| Public result value (7436) | `class CrashRecoveryResult` in `crash_recovery.py` | PASS (ε1a delivers) |
| One recovered-work ledger (D8, C7) | `class RecoveredWork[:(]` in `crash_recovery.py` | PASS (ε1a delivers) |

## ε1b — LaneReconciler

| Capability | Binding | Verdict |
|---|---|---|
| Module exists | path `lane_reconcile.py` | PASS (ε1b delivers) |
| Lane reconciliation left harness | `def _reconcile_lane_checkouts` absent from `harness.py` | PASS (ε1b delivers) |

## ε2a — StrandedReconciler

| Capability | Binding | Verdict |
|---|---|---|
| Module exists | path `stranded_reconcile.py` | PASS (ε2a delivers) |
| Applier left harness | `async def _reconcile_one_stranded` absent from `harness.py` | PASS (ε2a delivers) |
| Tests construct the class | `StrandedReconciler\(` in `test_reconcile_stranded.py` | PASS (ε2a delivers) |
| One gated done-write shared with the already-landed gate (D8, C7) | `class GatedDoneWriter` in `stranded_reconcile.py`; `def _mark_in_progress_done` absent from `harness.py` | PASS (ε2a delivers) |

## ε2b — RecoveryDispositions and the verified-green trio

| Capability | Binding | Verdict |
|---|---|---|
| One shared emitter | `class RecoveryDispositions` in `recovery_emission.py` | PASS (ε2b delivers) |
| Trio left harness | `def _maybe_submit_stranded_verified_green` absent from `harness.py` | PASS (ε2b delivers) |

## ε3 — DeterministicReconSweep

| Capability | Binding | Verdict |
|---|---|---|
| Module exists | path `deterministic_recon.py` | PASS (ε3 delivers) |
| Sweep left harness | `async def _run_deterministic_recon_sweep` absent from `harness.py` | PASS (ε3 delivers) |

## ζ1a — DispatchGateFilers

| Capability | Binding | Verdict |
|---|---|---|
| Module exists | path `dispatch_gate_filers.py` | PASS (ζ1a delivers) |
| Filers left harness | `def _block_and_escalate_external_dep` absent from `harness.py` | PASS (ζ1a delivers) |
| Tripwire widened | `dispatch_gate_filers` in `test_harness_escalation_filing_identity.py` | PASS (ζ1a delivers) |

## ζ1b — InfraFilers

| Capability | Binding | Verdict |
|---|---|---|
| Module exists | path `infra_filers.py` | PASS (ζ1b delivers) |
| Filers left harness | `def _file_structural_exhaustion_l2` absent from `harness.py` | PASS (ζ1b delivers) |
| Tripwire widened | `infra_filers` in `test_harness_escalation_filing_identity.py` | PASS (ζ1b delivers) |

## ζ2 — EscalationRecordSweeps

| Capability | Binding | Verdict |
|---|---|---|
| Module exists | path `escalation_record_sweeps.py` | PASS (ζ2 delivers) |
| Reaper left harness | `async def _reap_orphan_l0_escalations` absent from `harness.py` | PASS (ζ2 delivers) |
| Reaper reads C1 | `SlotRegistry` referenced in `escalation_record_sweeps.py` | PASS (ζ2 delivers) |
| Tripwire widened | `escalation_record_sweeps` in `test_harness_escalation_filing_identity.py` | PASS (ζ2 delivers) |

## θ — durable steward spend, $25 default (S8-1)

| Capability | Binding | Verdict |
|---|---|---|
| Durable spend home (forgotten 7 days after the last spend) | `steward_lifetime_spend` in `steward.py` | PASS (θ delivers) |
| Default cap $25 | `steward_lifetime_budget: float = Field\(default=25\.0` in `config.py`; `^steward_lifetime_budget: 25\.0` in `defaults.yaml` | PASS (θ delivers) |
| No lower project cap | `^steward_lifetime_budget:` absent from `dark-factory-orchestrator.yaml` | PASS (θ delivers) |
| Guard store exists | substrate row above (`PersistentMap`) | PASS (manual) |

## ι — one event writer (S8-3)

| Capability | Binding | Verdict |
|---|---|---|
| Sync flush surface | `^    def flush\(self` in `event_store.py` | PASS (ι delivers) |
| Async flush surface | `async def aflush` in `event_store.py` | PASS (ι delivers) |
| Writer registered first | `name='event-writer'` in `harness.py` | PASS (ι delivers) |
| Read-your-writes inside `EventStore` | boundary row 17b (manual) | PASS (ι delivers) |
| Per-emit cost measured | task record: loop occupancy per emit before/after on the verify host | PASS (ι delivers; manual) |

## ι2 — every in-process events read is an `EventStore` method (S8-3)

| Capability | Binding | Verdict |
|---|---|---|
| The raw-SQL guard exists | path `orchestrator/tests/test_event_store_encapsulation.py` (row 17c) | PASS (ι2 delivers) |
| Digest reads through `EventStore` | `FROM events` absent from `digest.py` | PASS (ι2 delivers) |
| Zero-progress gate reads through `EventStore` | `FROM events` absent from `strict_event_reads.py` | PASS (ι2 delivers) |

## κ1 — the invocation carries its sandbox backend

| Capability | Binding | Verdict |
|---|---|---|
| Process global gone | `def set_backend` absent from `sandbox_dispatch.py` | PASS (κ1 delivers) |
| Dry-run investigation reads no global | `get_backend` absent from `dry_run_unblock.py` | PASS (κ1 delivers) |
| Fixtures gone | `_restore_sandbox_backend` absent from `conftest.py` and `orchestrator/tests/conftest.py` | PASS (κ1 delivers) |

## κ2 — non-claude backends under `.task/`; one claude recipe

| Capability | Binding | Verdict |
|---|---|---|
| Claude recipe deduplicated | `_parse_claude_output` absent from `agents/invoke.py` | PASS (κ2 delivers) |
| pi never renames a tracked file | `pi-invoke-backup` absent from `agents/invoke.py`; boundary row 20 | PASS (κ2 delivers) |
| gemini/pi config levers (path or env) | neither CLI installed on the verify host at `e634cfe658`; the leaf measures the installed CLI, floor is the fail-loud refusal | OPEN (κ2 decides) |
| Non-claude backends use the env rule | `agent_subprocess_env` in `_invoke_codex`, `_invoke_gemini`, `_invoke_pi` (manual) | PASS (κ2 delivers) |

## μ1 — harness path carries its MCP session

| Capability | Binding | Verdict |
|---|---|---|
| Session required on the Scheduler | `mcp_session: McpSessionLike \| None` absent from `scheduler.py` | PASS (μ1 delivers) |
| Module global gone | `^_session: McpSession` absent from `mcp_lifecycle.py` | PASS (μ1 delivers) |
| Misparse gone | `rstrip('/mcp')` absent from `mcp_lifecycle.py` | PASS (μ1 delivers) |

## μ2 — briefing and recall carry the session

| Capability | Binding | Verdict |
|---|---|---|
| Session injected downstream | `MemoryRecall(config.fused_memory.url` absent from `agents/briefing.py` | PASS (μ2 delivers) |
| Recall takes an optional session (D13a) | `session: McpSession \| None` in `agents/memory_recall.py` | PASS (μ2 delivers) |

## ω — integration gate

| Capability | Binding | Verdict |
|---|---|---|
| No constructor bypass in tests | `Harness.__new__(Harness)` absent from `orchestrator/tests/` | PASS (ω delivers) |
| No tolerance reads | `getattr(self, '_` absent from `harness.py` | PASS (ω delivers) |
| Fixture initialiser gone | `_init_harness_state_for_test` absent from `orchestrator/tests/` | PASS (ω delivers) |
| Liveness forwarder gone after S3 ζ (S3 OQ-7 (a); the S3-1 / S8-2 cycle keeps it out of β) | `def is_workflow_active` absent from `harness.py` | PASS (ω delivers) |
| Boundary matrix green against fixtures | `orchestrator/tests/test_harness_integration_gate.py` runs rows 1–22c | PASS (ω delivers) |
| h14 re-measurement recorded | task record carries lines / prose share / method count beside the `e634cfe658` baseline (19,562 / 46.9% / 248) | PASS (ω delivers; manual) |

OQ-10 is ruled (a) (lead, 2026-11-02): no held λ block exists. No row is `FAIL`; the one `OPEN` row
(κ2's gemini/pi levers) is a measurement the leaf takes, not an undecided design question.
