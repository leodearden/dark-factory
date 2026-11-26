# Capability manifest — `plans/followup-riders-prd.md`

Per-leaf capability → evidence bindings (G3 + G6 mechanized). Machine-readable twin:
`plans/followup-riders-prd.capability-manifest.yaml` (task ids stamped by `commit_planning`).
Verified against main `c393a67c54` (2026-11-04): every `expect: present` pattern below is absent in its
`paths` today, and the polarity lint (`shared/delivered_check_polarity.py::lint_delivered_checks`) run
locally per leaf against `main` returned 0 findings. Every mechanical check is `kind: grep` on a symbol;
nothing is `kind: script` or `kind: path`. Anchors are symbols; nothing here is a line pin.

## Substrate (G3 re-check, symbol by symbol on `c393a67c54`)

Present and wired: `workflow.py::TaskWorkflow.{_plan, _run_architect_for_plan, _stamp_architect_plan,
_reconcile_plan_scope, _reconcile_scope_locks, _requeue_on_lock_conflict, _replan, _run_simple_task,
_try_narrow_plan}`, `artifacts.py::TaskArtifacts.lock_plan`, `scheduler.py::Scheduler.{handle_blast_radius_expansion,
_persist_files_metadata, mark_done, _module_cache}`, `module_charter.py::derive_modules`,
`agents/briefing.py::BriefingAssembler.{build_architect_prompt, build_plan_tightening_prompt,
build_context_complete_reviewer_prompt, build_completion_judge_prompt}`, `agents/roles.py::{_PLAN_CREATOR_TOOLS,
_NO_TASK_STATUS_WRITE, _SCOPE_BOUNDARY_RECOURSE}`, `mcp/plan_tools.py::{_create_plan, add_plan_step,
drop_plan_file, update_plan_metadata, mark_step_done, record_descoped_step, _PlanField, _COLLECTION_SCHEMA_TOOL}`,
`MarkupGuardMiddleware`, `mcp_lifecycle.py::plan_tools_mcp_server`, `plan_descope.py::{DescopedStep,
DESCOPED_STEPS_KEY, DESCOPE_PLAN_KEYS}`, `config.py::RELOADABLE_FIELDS`, `event_store.py::EventType`
(`invocation_end`), `shared/cli_invoke.py::AgentResult.turns`, `shared/task_metadata.py::{register_metadata_submodel,
DoneProvenance}`, `TaskInterceptor.{_apply_status_transition, _write_lock, _validate_done_provenance,
_execute_combine}`, `middleware/done_provenance_authz.py`, `middleware/lock_charter_guard.py`,
`middleware/scope_violation_escalator.py`, `reconciliation/targeted.py::TargetedReconciler._sweep_cancelled_descendants`,
`backends/sqlite_task_backend.py::stamp_pending_since`, `merge_lane/intake.py` (`_on_finalized`),
`escalation/git_authority.py::found_on_main_response`, `escalation/categories.py::SCOPE_VIOLATION`,
`harness.py::Harness._block_and_escalate_cross_repo`.

Absent, produced upstream by PRD 1 (all pending, wired as out-of-batch edges): `followup_hold` and the
hold rule (ε 8417 → α), `followup_items` / `merge_items` / `_format_followup_items` (β 8414 → β),
`promote(by='rider')` and, through its companion correction, `register_promotion_guard` (ζ 8418 → γ).
Produced by PRD 2 P1 (task 8452, filed 2026-11-04): `followup_packing.RIDER_RESERVE_DAYS`, `exclusion()`
(the γ binding below is PASS by producer, edge 8440 → 8452 wired). Absent as the PRD says (this batch creates them): `riders.py`,
`plan_riders.py`, `shared/rider_ledger.py`, `middleware/rider_carry.py`, every `rider_*` symbol.

Name corrections (no contract change; carried into the leaf descriptions):

1. The merge-phase narrowing pass is `workflow.py::TaskWorkflow._try_narrow_plan`; the PRD's
   `_run_plan_tightening` never existed (absent at `94d620f988` too). Its prompt builder,
   `build_plan_tightening_prompt`, is named correctly.
2. `RiderWriteAuthorityError(TaskBackendError)`: there is no `TaskBackendError`. The idiom is
   `backends/task_backend_errors.py::DeferralWriteAuthorityError(TaskmasterError)`, so β adds the
   error there with the same base.
3. In `_plan`, `_stamp_architect_plan` and `lock_plan` run before `_reconcile_plan_scope`, so decision
   1's "before `_stamp_architect_plan`'s lock" cannot hold literally. The offer call sits after the
   reconcile succeeds, which is after the lock. §2 already records that plan-tools never consult
   `lock_plan`, so this is harmless. β re-runs `_stamp_architect_plan` after an offer-mode pass.

## Bindings

| leaf | capability | binding | verdict |
|---|---|---|---|
| α | `HostView`, `select_rider_offers`, `offerable` | new `orchestrator/src/orchestrator/riders.py`; grep the three definitions there | PASS |
| α | selector wired at the plan boundary | grep `select_rider_offers` in `workflow.py` (`_plan`, after `_reconcile_plan_scope` succeeds) | PASS |
| α | `RidersConfig`, green-tier reloadable | grep `class RidersConfig` in `config.py` | PASS |
| α | `rider_offered`, `rider_offers_skipped` events | grep each in `event_store.py` | PASS |
| α | `RiderLedger` submodel registered | grep `class RiderLedger` in new `shared/src/shared/rider_ledger.py`; grep the quoted `rider_ledger` key literal in `shared/src/shared/` (0 today) | PASS |
| α | `apply_rider_verdicts` observe arm and MCP verb | grep `def apply_rider_verdicts\(` in new `middleware/rider_carry.py`, and the name in `server/tools.py` | PASS |
| α | observe mode actually on for df | **resolved at decompose**: grep `^riders:` in `dark-factory-orchestrator.yaml`. `defaults.yaml` keeps `enabled: false` (decision 14), and no leaf owned turning observe on, so δ's observe series would read "producer has not written" forever. α commits `riders: {enabled: true, mode: observe}` to df's own config. Observe mode changes no prompt and writes only the ledger | PASS |
| α | rows 1, 20, 21 on a live dispatch; `reload_config` reports `applied` | manual | PASS |
| β | `RiderVerdict`, `DECLINE_REASONS` | new `plan_riders.py`; grep both | PASS |
| β | `accept_rider` / `decline_rider` with typed refusals (rows 4, 5, rejection-style) | grep `def accept_rider\(`, `def decline_rider\(` in `mcp/plan_tools.py`. β creates the refusals, so they fire on β's tree | PASS |
| β | `_riders` / `_rider_offers` carried forward by `create_plan` | grep `RIDERS_KEY` in `mcp/plan_tools.py` | PASS |
| β | architect holds the tools | grep `accept_rider` in `agents/roles.py` | PASS |
| β | offer prompt | grep `def build_rider_offer_prompt\(` in `agents/briefing.py`; it renders items through `_format_followup_items` (8414, upstream) | PASS |
| β | rider pass wired | grep `_run_rider_pass` in `workflow.py` | PASS |
| β | typed `modules_changed` path (row 7, rejection-style), carry event | grep `rider_pass_failed`, `rider_carried` in `event_store.py` | PASS |
| β | carry merges items | grep `merge_items` in `rider_carry.py` (8414, upstream) | PASS |
| β | `RiderWriteAuthorityError` (row 18, rejection-style) | grep the class in `fused-memory/src/` + `shared/src/`, and its use in `task_interceptor.py`. β creates it | PASS |
| β | shadow and offer behaviour in production | manual | PASS |
| γ | `stamp_rider_deliverability`, `release_rider`, `close_or_release_riders_for`, `reconcile_carries`, `rider_promotion_guard` | grep each definition in `rider_carry.py` | PASS |
| γ | close wired in the status transition | grep `close_or_release_riders_for` in `task_interceptor.py` | PASS |
| γ | sweep and guard registered at startup | grep `reconcile_carries` and `rider_promotion_guard` in `server/main.py`. The hook `register_promotion_guard` is 8418's companion correction, applied at this decompose and upstream of γ | PASS |
| γ | sweep interval | grep `carry_sweep_interval_secs` in `fused-memory/src/fused_memory/config/schema.py` | PASS |
| γ | workflow carry reconcile | grep `_reconcile_rider_carry` in `workflow.py` | PASS |
| γ | total close event, human release verb | grep `rider_close_failed` in `fused-memory/src/` + `orchestrator/src/`; `release_rider` in `server/tools.py` | PASS |
| γ | carry clock = PRD 2's `RIDER_RESERVE_DAYS` | grep `RIDER_RESERVE_DAYS` in `rider_carry.py`. Producer: task-8452 (PRD 2 P1), upstream via the edge 8440 → 8452 wired 2026-11-04. γ imports the constant and never redefines it | PASS |
| γ | boundary suite | grep `def test_` in new `fused-memory/tests/integration/test_followup_riders.py` | PASS |
| γ | every §4.6 row green; `OPERATIONS.md` riders section | manual (row 19 is shared with PRD 2's gate; the section is prose, INV-10) | PASS |
| δ | report script with no-producer states and the `KILL` line | grep `producer has not written` and `KILL` in new `scripts/followup_riders_report.py` | PASS |
| δ | dated report | manual | PASS |

## Resolved at decompose

1. The observe-mode enablement had no owner. α now owns df's `riders:` stanza (above). Reify's own
   enablement is an operator decision, recorded in the PRD's decompose record. Without it, δ's reify
   series reads "producer has not written".
2. The three name corrections above.

Every binding is PASS; the γ carry-clock row became PASS when PRD 2 P1 was filed as 8452 and the edge 8440 → 8452 wired.
No binding resolves to declared-only, test-only, producer-downstream or producer-absent.
DAG: α→8417; β→α, 8414; γ→β, 8418, 8452; δ→γ. `signal_leaves: [γ]`.
