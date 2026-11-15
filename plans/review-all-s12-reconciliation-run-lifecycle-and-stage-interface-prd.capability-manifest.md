# Capability manifest — S12 reconciliation run lifecycle and stage interface

PRD: `plans/review-all-s12-reconciliation-run-lifecycle-and-stage-interface-prd.md`. YAML twin:
`plans/review-all-s12-reconciliation-run-lifecycle-and-stage-interface-prd.capability-manifest.yaml`.

**Code anchors** verified against main `1b106418da` (2026-11-12; the ruling fold re-anchored the author's
`1f240e6da2`). Main moves fast — cite-by-symbol; re-locate lines at implementation time. Bindings cite
symbols, not lines. A row marked *built here* or *anti-inversion* is a producer binding: its check FAILS at
today's tree and PASSES once its leaf lands. A row whose name ends `-substrate` passes today, says
"substrate" in its binding, and is never copied to the task as a producer check. Every check in the twin was
evaluated at `1b106418da` through
`shared/src/shared/delivered_check_polarity.py::evaluate_grep_at_tree` / `evaluate_path_at_tree` after loading
the twin with `shared.capability_manifest.CapabilityManifestDoc`: all 49 producer checks fail and all 4
substrate checks pass; fold-verify re-ran them at `0f21f5e457` with one row added (50 producer checks fail, 4 substrate pass). `lint_delivered_checks` with each task's declared files reports the 4 substrate rows
as `vacuous_present` (refused at `commit_planning`, by design: a substrate row is never a producer check) and
one non-blocking `absent_overbroad` warning on δ's `no-echoed-run-id-instruction`: today its second match is
in the self-model at its pre-move path, `reconciliation/recon_self_model.py`, which S19 β1 moves to the
`recon_state/` path δ declares before δ runs (δ → S19 β1). No row is `OPEN`. Every open question was ruled at the 2026-11-12 sitting; δ's nine rows added since the author pass (eight by the ruling fold, one by fold-verify) carry the OQ-3
ruling (refusal, the run-id header in task filing, and the instruction sweep). Checks on modules a sibling plan moves (S13's `task_knowledge_sync.py` split, S15's
`tool_context.py`, S19's `recon_state/`) read a directory, so they hold before and after each move. `fm/` abbreviates
`fused-memory/src/fused_memory/`. Signal leaf: ω.

## α — Typed reconciliation run record: StageId-keyed reports, a RunError field, the journal as sole decoder

Closes fk-377a7036ac7f. Deps: 7876, 6721 (stats_verifier.py), S10 λ 9254 (harness.py), S19 α (recon_state/journal.py, lead ruling OQ-12). Subsumes task 5912 (cancel).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| typed-stage-map | built here — fm/models/reconciliation.py::ReconciliationRun.stage_reports keyed by StageId, values StageReport only | grep `stage_reports: dict\[StageId, StageReport\]` in `fused-memory/src/fused_memory/models/reconciliation.py`, expect present | PASS |
| run-error-type | built here — fm/models/reconciliation.py::RunError, the typed run failure record | grep `class RunError\(` in `fused-memory/src/fused_memory/models/reconciliation.py`, expect present | PASS |
| no-shape-branch-on-stage-3-report | anti-inversion — the four isinstance(s3_report, dict) branches in harness.py go | grep `isinstance\(s3_report, dict\)` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |

## β1 — One reconciliation run driver for full and backlog-chunk runs; the run row before any drain

Closes fk-0485729893fe; builds C1 for fk-8621784d79d3 (closed by β2). Deps: α, 6695 (lands the narrow fix first).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| run-kind-value | built here — fm/reconciliation/run_context.py::RunKind (new module) | grep `class RunKind\(StrEnum\)` in `fused-memory/src/fused_memory/reconciliation/`, expect present | PASS |
| one-driver-entry | built here — fm/reconciliation/harness.py::ReconciliationHarness.drive | grep `async def drive\(` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect present | PASS |
| run-row-recovery-substrate | substrate — `grep:fused-memory/src/fused_memory/reconciliation/drained_event_restore.py::restore_unless_completed` and ReconciliationHarness._recover_one_run already restore a running row's own drained events; β1 only orders start_run first | grep `async def restore_unless_completed\(` in `fused-memory/src/fused_memory/reconciliation/drained_event_restore.py`, expect present | PASS |

## β2 — Remediation is RunKind.remediation on the one driver; the second run lifecycle is deleted

Closes fk-8621784d79d3 (high), fk-2571f7726184. High: dependency of gate 6698. Deps: β1. _maybe_remediate calls drive outside its own try, so a remediation failure is filed once (boundary row 17).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| no-second-run-construction | anti-inversion — _run_remediation_pass no longer builds its own ReconciliationRun; the driver maps RunKind to RunType | grep `run_type=RunType\.remediation` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |
| no-second-cancel-arm | anti-inversion — the remediation pass's own CancelledError arm goes; the driver's ladder serves every kind | grep `Remediation pass cancelled \(shutdown` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |

## γ0 — Reconciliation stage tests build and run stages through one test helper

Contributes to fk-63b65973bd33 and fk-0cce86f5dbfe (test side); closes none. Deps: S13 ι (stage_builders.py imports IntegrityCheck, lead ruling OQ-8) only; the test files it shares with β2 and S10 μ1 9278 / μ2 9280 are rebase-only (lead's test-file rule). Test-only, uniform substitution over 34 modules.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| stage-builders-module | built here — fused-memory/tests/reconciliation/stage_builders.py::build_stage / run_stage, the one way a test constructs and runs a stage | path `fused-memory/tests/reconciliation/stage_builders.py`, expect present | PASS |

## γ1 — Stage payload interface: assemble_payload(..., run) returns PayloadResult; the per-run stashes go

Closes fk-0cce86f5dbfe. Deps: β2 (harness.py), γ0 (stage_builders.py, which γ0 creates), S10 α 9180 (task_knowledge_sync.py), S26 μ 9203 (memory_consolidator.py), S13 ι (verbatim task_knowledge_sync.py split; stages/integrity_check.py, lead ruling OQ-8).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| payload-result-type | built here — fm/reconciliation/run_context.py::PayloadResult | grep `class PayloadResult` in `fused-memory/src/fused_memory/reconciliation/`, expect present | PASS |
| no-run-id-stash | anti-inversion — TaskKnowledgeSync._current_run_id and its 9 read sites go; the run id arrives in RunContext | grep `self\._current_run_id` in `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`, expect absent | PASS |

## γ2 — Reconciliation stages constructed with their collaborators and per-cycle inputs

Closes fk-63b65973bd33; contributes to fk-16baf39b1db6. Subsumes tasks 7343, 8835 and 7604 (cancel; θ finishes 7604's tests). Deps: γ1.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| no-harness-set-attributes | anti-inversion — the 'set by harness' class attributes on MemoryConsolidator, TaskKnowledgeSync and IntegrityCheck go (IntegrityCheck in stages/integrity_check.py after S13 ι; the check reads all of stages/) | grep `set by harness` in `fused-memory/src/fused_memory/reconciliation/stages/`, expect absent | PASS |
| no-configure-helpers | anti-inversion — ReconciliationHarness._configure_consolidator goes (with _configure_task_sync and _propagate_escalation_queue) | grep `def _configure_consolidator` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |

## δ — Recon stage identity and run id bound on the MCP transport; a differing value refused; the stage instructions swept; fused-memory URL from server config

Closes fk-a3598aec7dca (high), fk-d2c7f1b902ac. High: dependency of gate 6698. Deps: γ2, 6721 (S14 μ), S10 μ2 9280 (server/tools.py), S10 ν 9282 (middleware/task_interceptor.py, added by the OQ-3 ruling), S15 α (server/tool_context.py, lead ruling OQ-11), S19 β1 (recon_state/recon_self_model.py, lead ruling OQ-12). The recon-report MCP entry carries the same headers (lead ruling OQ-10). Sitting ruling OQ-3 (Leo, 2026-11-12, with the lead's run-id clause): a differing agent_id or _causation_id is refused with a structured error naming both values, one log line per refusal; task filing reads the run id from the header; the stage prompt says "do not pass agent_id or _causation_id"; CLAUDE.md, .claude/commands/memory.md, docs/task-authoring.md and the server's MCP instructions string carry the stage exception; all in one commit. search's agent_id is an authoring-agent filter and is outside the refusal. Supersedes task 6895 (cancel).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| no-hard-coded-client-config | anti-inversion — _find_fused_memory_server and its user-path .mcp.json candidate go | grep `/home/leo/src/dark-factory/\.mcp\.json` in `fused-memory/src/fused_memory/reconciliation/stages/base.py`, expect absent | PASS |
| no-prompt-echoed-agent-id | anti-inversion — the recon_context instruction asking the LLM to echo agent_id goes | grep `agent_id.: "recon-stage-` in `fused-memory/src/fused_memory/reconciliation/stages/base.py`, expect absent | PASS |
| server-reads-stage-header | built here — _resolve_identity prefers STAGE_AGENT_ID_HEADER (C3), in fm/server/tool_context.py once S15 α has moved it there (fm/server/tools.py before); the check reads all of fm/server/ | grep `STAGE_AGENT_ID_HEADER` in `fused-memory/src/fused_memory/server/`, expect present | PASS |
| mcp-headers-substrate | substrate — `grep:orchestrator/src/orchestrator/mcp_lifecycle.py` already writes a headers block into a Claude CLI http MCP entry (escalation identity); δ uses the same CLI capability | grep `escalation_block\['headers'\]` in `orchestrator/src/orchestrator/mcp_lifecycle.py`, expect present | PASS |
| stage-identity-refusal-predicate | built here — fm/server/tool_context.py::stage_identity_refusal, the one predicate that refuses an agent_id or _causation_id differing from its header (C3); the check reads all of fm/server/ | grep `def stage_identity_refusal\(` in `fused-memory/src/fused_memory/server/`, expect present | PASS |
| creation-journal-reads-run-header | built here — fm/middleware/task_creation_journal.py::CreationAttribution.from_submission takes the run-id header as bound_causation_id | grep `bound_causation_id` in `fused-memory/src/fused_memory/middleware/task_creation_journal.py`, expect present | PASS |
| stage-prompt-says-do-not-pass | built here — stages/base.py's recon_context carries "The connection already identifies you and this run. Do not pass `agent_id` or `_causation_id`." | grep `Do not pass .agent_id. or ._causation_id.` in `fused-memory/src/fused_memory/reconciliation/stages/base.py`, expect present | PASS |
| no-echoed-run-id-instruction | anti-inversion — prompts/stage2.py's submit_task example and recon_self_model.py's accounting text stop asking for `"_causation_id": <run_id>` (the check reads all of fm/, so it holds across S19 β1's move of the self-model) | grep `"_causation_id": <run_id>` in `fused-memory/src/fused_memory/`, expect absent | PASS |
| no-stage1-agent-id-example | anti-inversion — prompts/stage1.py's propose_consolidation example drops `agent_id='recon-stage-memory_consolidator'` | grep `agent_id='recon-stage-` in `fused-memory/src/fused_memory/reconciliation/prompts/`, expect absent | PASS |
| claude-md-stage-exception | built here — CLAUDE.md §Write-Tagging Convention gains the exception that reconciliation stages pass neither agent_id nor _causation_id (CLAUDE.md names no _causation_id today) | grep `_causation_id` in `CLAUDE.md`, expect present | PASS |
| memory-command-stage-exception | built here — .claude/commands/memory.md gains the same exception beside "Always include `project_id` and `agent_id`" (it names no _causation_id today) | grep `_causation_id` in `.claude/commands/memory.md`, expect present | PASS |
| server-instructions-stage-exception | built here — fm/server/tools.py::FUSED_MEMORY_INSTRUCTIONS, the MCP instructions every client receives (stages included), amends its "Include agent_id for attribution" convention with the stage exception (added at fold-verify; the check reads all of fm/server/) | grep `a reconciliation stage omits agent_id and _causation_id` in `fused-memory/src/fused_memory/server/`, expect present | PASS |
| no-misleading-stage-id-example | anti-inversion — the `"reconciliation-stage-1"` example leaves CLAUDE.md and docs/task-authoring.md | grep `reconciliation-stage-1` in `CLAUDE.md`, `docs/task-authoring.md`, expect absent | PASS |

## ι1 — Stage prompts render capabilities and the dedup rule from code; prompts and the harness import nothing from server

Closes fk-10c8c43d9e89 (high), fk-9850339b610c; the upward-import half of fk-20cb93d17ff3. High: dependency of gate 6698. Deps: δ, S10 μ2 9280 (server/main.py); S19 β1 (recon_state/recon_self_model.py, lead ruling OQ-12) is reached through δ.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| capabilities-derived | built here — fm/reconciliation/stage_capabilities.py::capabilities_for | grep `def capabilities_for\(` in `fused-memory/src/fused_memory/reconciliation/`, expect present | PASS |
| no-capability-literals | anti-inversion — the can_file_tasks= / can_escalate= literals in the stage prompts go | grep `can_(file_tasks\|escalate)=(True\|False)` in `fused-memory/src/fused_memory/reconciliation/prompts/`, expect absent | PASS |
| no-phantom-denial-name | anti-inversion — the sentence naming DISALLOW_SUBTASK_CREATE, defined nowhere, goes | grep `DISALLOW_SUBTASK_CREATE` in `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py`, expect absent | PASS |
| no-internal-narration | anti-inversion — the prompt passages narrating _derive_affected_ids go; the rule renders from recon_self_model | grep `_derive_affected_ids` in `fused-memory/src/fused_memory/reconciliation/prompts/`, expect absent | PASS |
| no-upward-import | anti-inversion — neither the prompt modules nor harness.py import fused_memory.server (prompts/__init__.py's function-local server.recon_report import and prompts/stage1.py's server.grouped_read import are removed, not moved into the harness) | grep `(from\|import) fused_memory\.server` in `fused-memory/src/fused_memory/reconciliation/prompts/`, `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |

## ι2 — Prompt fragment library moves out of the package __init__ into prompts/fragments.py

Closes fk-20cb93d17ff3 (library half). Deps: ι1, κ2. Also repoints fused-memory/scripts/freeze_write_triage_population.py.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| fragments-module | built here — fm/reconciliation/prompts/fragments.py | path `fused-memory/src/fused_memory/reconciliation/prompts/fragments.py`, expect present | PASS |
| no-reach-back-into-package | anti-inversion — stage1/2/3.py stop importing names from their parent package | grep `from fused_memory\.reconciliation\.prompts import` in `fused-memory/src/fused_memory/reconciliation/prompts/stage1.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py`, `fused-memory/src/fused_memory/reconciliation/prompts/stage3.py`, expect absent | PASS |

## ε — Consecutive-cycle recon alarms as RunStreak declarations over shared.streak.KeyedStreaks

Closes fk-90e488a174b7. Deps: ι1 (harness.py), γ2 (task_knowledge_sync.py), S1 ι1 (recon escalation registry C8 + FilingPolicy C1, lead ruling OQ-7), S13 ι (standing_decision_growth_sweep.py, lead ruling OQ-8). S14 λ 6712 is done. Fires at count >= threshold, repeats fold (sitting ruling OQ-1).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| streaks-on-shared-primitive | built here — fm/reconciliation/run_streaks.py evaluates every RunStreak through KeyedStreaks | grep `KeyedStreaks` in `fused-memory/src/fused_memory/reconciliation/run_streaks.py`, expect present | PASS |
| streaks-file-through-registry | built here — each RunStreak names an S1/C8 ReconEscalation member and files through policy_for (recon chooses a policy only) | grep `ReconEscalation` in `fused-memory/src/fused_memory/reconciliation/run_streaks.py`, expect present | PASS |
| no-snapshot-quartet | anti-inversion — task_count_snapshot_cadence.py's streak compute goes | grep `def compute_snapshot_miss_streak` in `fused-memory/src/fused_memory/reconciliation/task_count_snapshot_cadence.py`, expect absent | PASS |
| no-growth-quartet | anti-inversion — the growth-sweep streak compute goes, wherever the task_knowledge_sync.py split put it and under either name (stages/task_knowledge_sync.py today, reconciliation/standing_decision_growth_sweep.py after S13 ι, possibly made public there); the check reads all of fm/reconciliation/ | grep `def _?compute_growth_sweep_failure_streak\(` in `fused-memory/src/fused_memory/reconciliation/`, expect absent | PASS |
| keyed-streaks-substrate | substrate — `grep:shared/src/shared/streak.py::KeyedStreaks` (S14 λ, task 6712, done) | grep `class KeyedStreaks` in `shared/src/shared/streak.py`, expect present | PASS |

## ζ — Cycle-summary write state as a typed StageReport field

Closes fk-f23c6e137ce5. Deps: ε, 7877.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| typed-cycle-summary-state | built here — fm/models/reconciliation.py::CycleSummaryWrite | grep `class CycleSummaryWrite` in `fused-memory/src/fused_memory/models/reconciliation.py`, expect present | PASS |
| no-string-flag-parse | anti-inversion — the write-recovered stats flag is no longer parsed in harness.py | grep `cycle_summary_write_recovered_backstop` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |

## κ1 — Targeted reconciler: one gated write path, classified outcomes, an event per fail-open arm; deferred writes replayed with causation

Closes fk-05443fb78da6, fk-be8ed2cd9ce9, fk-2bb6c913d747, fk-147a9951edc7. Deps: ζ, S10 μ2 9280, S10 ν 9282, 8144. Not S10 ο 9285 (it waits on 8528; different symbols of targeted.py, file locks order them). The reconciler factory is a provider member of S10 μ2's InterceptorCollaborators (lead ruling OQ-2); S15 D12 consumes it.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| no-optional-interceptor | anti-inversion — TargetedReconciler no longer starts with task_interceptor = None | grep `self\.task_interceptor: TaskInterceptor \| None = None` in `fused-memory/src/fused_memory/reconciliation/targeted.py`, expect absent | PASS |
| no-post-construction-wiring | anti-inversion — run_server no longer assigns targeted.task_interceptor | grep `targeted\.task_interceptor = task_interceptor` in `fused-memory/src/fused_memory/server/main.py`, expect absent | PASS |
| failed-replay-released | built here — EventBuffer.release_deferred_write increments attempt_count and unclaims a failed replay, so the next cycle retries it (event_buffer.py is at its recon_state path after S19 α; the check reads all of fm/) | grep `def release_deferred_write\(` in `fused-memory/src/fused_memory/`, expect present | PASS |
| write-failure-reason-substrate | substrate — `grep:shared/src/shared/mcp_envelope.py::write_failure_reason` is the classifier κ1 calls | grep `def write_failure_reason` in `shared/src/shared/mcp_envelope.py`, expect present | PASS |

## κ2 — Reconciliation stages and harness write tasks only through the TaskInterceptor

Closes fk-875cca636438. Deps: κ1, S13 ι (policies/briefing_refresh.py, lead ruling OQ-8). The guard's one exemption, consolidation_auto_executor.py's hazard-gate filer, carries an INV-12 Policy row (lead ruling OQ-9).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| no-raw-add-task | anti-inversion — queue_briefing_refresh_tasks submits through the interceptor, not taskmaster.add_task, wherever the split put it (stages/task_knowledge_sync.py today, reconciliation/policies/briefing_refresh.py after S13 ι) | grep `taskmaster\.add_task\(` in `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`, `fused-memory/src/fused_memory/reconciliation/policies/`, expect absent | PASS |
| hazard-gate-exemption-ratified | built here — docs/legibility/exception-ratifications.yaml row inv12-recon-hazard-gate-raw-backend, the Policy the guard's one exemption names (plans/memory-auto-consolidation-prd.md D8) | grep `^- id: inv12-recon-hazard-gate-raw-backend` in `docs/legibility/exception-ratifications.yaml`, expect present | PASS |

## η1 — harness.py partition I: the cycle-summary backstop and run recovery in their own modules

Contributes to fk-a88506a1e327 and fk-16baf39b1db6. Deps: κ2.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| run-recovery-module | built here — fm/reconciliation/run_recovery.py | path `fused-memory/src/fused_memory/reconciliation/run_recovery.py`, expect present | PASS |
| backstop-module | built here — fm/reconciliation/cycle_summary_backstop.py | path `fused-memory/src/fused_memory/reconciliation/cycle_summary_backstop.py`, expect present | PASS |

## η2 — harness.py partition II: cycle diagnostics, the finding gate and BacklogIterator in their own modules; prose rehomed

Closes fk-a88506a1e327; contributes to fk-16baf39b1db6. Subsumes task 6783 (cancel). Deps: η1.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| backlog-iterator-module | built here — fm/reconciliation/backlog_iterator.py | path `fused-memory/src/fused_memory/reconciliation/backlog_iterator.py`, expect present | PASS |
| iterator-left-harness | anti-inversion — harness.py no longer defines BacklogIterator | grep `^class BacklogIterator` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |

## θ — Reconciliation harness tests drive public seams; the remaining private test seams are deleted

Closes fk-16baf39b1db6. Finishes task 7604's tests (7604 is cancelled onto γ2). Deps: η2 (not ι2: test files only, rebase-only).

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| no-test-only-stages-attribute | anti-inversion — ReconciliationHarness.stages, kept for tests, goes | grep `stages: list\[Any\]` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |
| no-module-sleep-seam | anti-inversion — the module-level _sleep patch seam goes; the harness takes an injected sleep | grep `^_sleep = asyncio\.sleep` in `fused-memory/src/fused_memory/reconciliation/harness.py`, expect absent | PASS |

## ω — Integration gate: the run-lifecycle boundary suite and the live journal check

Signal leaf. No key. Deps: θ, ι2.

| Capability | Binding | Check | Verdict |
|---|---|---|---|
| boundary-suite | built here — fused-memory/tests/reconciliation/test_run_lifecycle_boundary.py (boundary rows 1-17, 10a-10d) | path `fused-memory/tests/reconciliation/test_run_lifecycle_boundary.py`, expect present | PASS |
| live-check-script | built here — fused-memory/scripts/recon_run_lifecycle_check.py | path `fused-memory/scripts/recon_run_lifecycle_check.py`, expect present | PASS |
