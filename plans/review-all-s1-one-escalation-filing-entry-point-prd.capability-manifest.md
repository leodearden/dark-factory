# Capability manifest — S1 one escalation filing entry point

PRD: `plans/review-all-s1-one-escalation-filing-entry-point-prd.md`.
YAML twin: `plans/review-all-s1-one-escalation-filing-entry-point-prd.capability-manifest.yaml`.
The twin names the signal leaves ω (the door's integration gate), λ2 (eval hermeticity) and μ (the
corpus-cell sandbox). Every `task_id` stays `null` until decompose.

**Code anchors** verified against main `a85cb1ad78` (2026-11-13), after the sitting of 2026-11-12 and
Leo's residual answers of 2026-11-13 (PRD §Rulings), and re-checked by the fold-verifier at main
`4a87282ee7` (all 62 producer checks still fail there). Bindings cite symbols, not lines.

Polarity was measured at `ba17934861` (critic fold), `7baa5b997c` (fold 2), `c9ab19766a` (fold 3) and,
for this ruling fold, at `a85cb1ad78` with `shared/src/shared/delivered_check_polarity.py::evaluate_grep_at_tree`
and `evaluate_path_at_tree`; the sidecar loads through `shared/src/shared/capability_manifest.py::CapabilityManifestDoc`
(26 tasks). All 62 mechanical checks fail at today's tree, as a producer check must: each `expect:
present` check has no match (or names a file that does not exist) today, and each `expect: absent` check
matches (or names a file that exists) today. `lint_delivered_checks`, given each leaf's declared files,
reports nothing at `a85cb1ad78`. Absent-checks are anchored on code shapes (a `def`, a declaration, an
assignment, a call) so prose mentions in comments cannot hold them red, and are scoped to the leaf's own
files where a later leaf converts a sibling module (ε2b, D15). Rows named `*-substrate` pass today or once
an upstream task lands, and carry no mechanical check (a check that already passes is vacuous by the
polarity rule; it is never copied to the task). The rulings turned four of the five `OPEN` manual rows
into producer checks (α's offload shape, promotion policy and eval-lane scope) or removed them (κ4's
awaited give-up, covered by `steward-give-up-awaits-filing` and boundary rows 23–24); one row stays `OPEN`
with a manual check because it is a lead action, not code (β's lead edges and S3 amendments). No check is a
`kind: script`. Counts that moved since `ba17934861` (task 8926's `file_or_refresh`, task 6234's injected
queue factory) are relabelled in the bindings.

## α — The filing entry point

| Capability | Binding | Verdict |
|---|---|---|
| filing-module | built here — `escalation/src/escalation/filing.py` (C1); absent at `a85cb1ad78` | PASS (path present) |
| filer-class | built here — `EscalationFiler` with `file`/`file_soon`/`file_blocking`/`drain`/`resolve_filed` | PASS (grep present, `escalation/src/`) |
| refresh-open-dedup-member | built here — `RefreshOpen` in the `Dedup` sum type (D13) | PASS (grep present) |
| chokepoint-closure-gone | anti-inversion — the `async def _chokepoint_or_submit(` definition in `server.py`; scanned over all of `escalation/src` so a move into S3's `filing_tools.py` cannot pass it | PASS (grep absent) |
| attach-or-submit-split | anti-inversion — `dedupe.py::attach_or_submit` splits into `attach_to_parent` and the filer's write (D6) | PASS (grep absent) |
| has-open-level-parameterised | built here — `EscalationQueue.has_open(task_id, *, level, …)` (C3) | PASS (grep present) |
| sentinel-registry | built here — `escalation/models.py::Sentinel` (C3), values byte-identical to today's literals, dunder and prefix forms | PASS (grep present) |
| role-predicate-in-models | built here — `models.is_harness_sentinel_role` (task 6609's residue) | PASS (grep present) |
| offload-entry-shape | built here — D1 (OQ-1, Leo): offload by entry point; `FilingOnLoopThread` | PASS (grep present) |
| promotion-policy-shape | built here — D3 (OQ-7, lead): `FilingPolicy.promoted_l2(root_cause)` | PASS (grep present) |
| eval-lane-scope | built here — D14 (OQ-8, Leo): `EvalLaneScope`, containment at levels 1–2 in process, every level at the MCP door | PASS (grep present) |
| notify-delivery-substrate | substrate — producer:task-9157 (S3 β, C4 `deliver_on` and `_mutate_pending`) upstream through 9181 | PASS |
| observed-response-substrate | `grep:escalation/src/escalation/queue.py::observed_submit_response` substrate | PASS |
| category-registry-substrate | `grep:escalation/src/escalation/categories.py::EscalationCategory` `grep:escalation/src/escalation/categories.py::require_registered_category` substrate | PASS |
| eval-lane-predicate-substrate | `grep:shared/src/shared/eval_lane.py::eval_lane_provenance` substrate | PASS |
| fold-mechanics-substrate | `grep:escalation/src/escalation/dedupe.py::resolve_dedupe_parent` `grep:escalation/src/escalation/dedupe.py::DedupeConfig` substrate | PASS |
| born-at-l2-vocabulary-substrate | `grep:escalation/src/escalation/models.py::BORN_AT_L2_SEVERITIES` `grep:escalation/src/escalation/models.py::HARNESS_SENTINEL_ROLE_PREFIXES` substrate | PASS |

## β — Every other escalation-package filer, including the watcher's promotion

| Capability | Binding | Verdict |
|---|---|---|
| server-direct-writes-gone | anti-inversion — 5 code lines writing through `queue.submit`/`submit_resolved` in `server.py` at `a85cb1ad78` (unchanged); α removes three, β the rest | PASS (grep absent) |
| promotion-through-the-door | built here — `promote_to_l2`'s create step files with `FilingPolicy.promoted_l2(…)` | PASS (grep present) |
| note-to-file-filer-moved | anti-inversion — `note_to_file.py::file_note_to_file` becomes C2 step 6 | PASS (grep absent) |
| submit-cli-through-filer | built here — `submit.py` files with `file_blocking` | PASS (grep present) |
| storm-filer-substrate | `grep:escalation/src/escalation/guard_storm_alarm.py::file_storm_once` substrate | PASS |
| s3-merge-tools-bind-the-filer | lead edges 9188 → β and 9187 → β; 9188/9200 and the S3 PRD's C7a amended to bind `EscalationFiler` | OPEN (manual: a lead action; the lead confirms the edges and the amendments, OQ-3 and OQ-6 ruled by the lead) |

## γ — Typed status-write outcome and `file_and_block`

| Capability | Binding | Verdict |
|---|---|---|
| status-write-module | built here — `orchestrator/src/orchestrator/status_write.py` (C6) | PASS (path present) |
| runner-tails-through-file-and-block | built here — the five `_file_*_and_block` tails call `file_and_block` | PASS (grep present) |
| rejection-type-substrate | `grep:orchestrator/src/orchestrator/scheduler.py::SetTaskStatusRejected` `grep:orchestrator/src/orchestrator/scheduler.py::TerminalExitRejection` substrate | PASS |

## δ — `TaskWorkflow._file_escalation`

| Capability | Binding | Verdict |
|---|---|---|
| workflow-filing-helper | built here — `TaskWorkflow._file_escalation` (C7), with `agent_role` | PASS (grep present) |
| event-type-substrate | `grep:orchestrator/src/orchestrator/event_store.py::EventType` substrate | PASS |

## ε1 — The widened lane port; the worker; S7's sink absorbed

| Capability | Binding | Verdict |
|---|---|---|
| port-file-soon | built here — `EscalationPort.file_soon` (C4) | PASS (grep present) |
| worker-raw-submits-gone | anti-inversion — 5 `escalation_queue.submit(` lines in `worker.py` (S6 moves two out first) | PASS (grep absent) |
| storm-sink-substrate | substrate — producer:task-9136 (S7 α1 creates `StormEscalationSink`, which ε1 deletes) | PASS |
| port-substrate | `grep:orchestrator/src/orchestrator/merge_lane/ports.py::EscalationPort` `grep:orchestrator/src/orchestrator/merge_lane/ports.py::escalation_port` substrate | PASS |

## ε2a — Liveness, request-ledger, verify-base and skew-tripwire alarms

| Capability | Binding | Verdict |
|---|---|---|
| satellite-raw-submits-gone | anti-inversion — 5 `escalation_queue.submit(` lines in the four modules | PASS (grep absent) |

## ε2b — Shadow, landing-evidence, member re-drive and intake-buffer alarms

| Capability | Binding | Verdict |
|---|---|---|
| lane-raw-submits-gone | anti-inversion — 6 `escalation_queue.submit(` lines in `shadow.py`, `landing_evidence.py`, `intake_buffers.py`; scoped so S6's new modules (ε3's) cannot hold it red | PASS (grep absent) |
| shadow-level-one-dedup-gone | anti-inversion — 3 `has_open_l1(` dedups guarding level-2 filings in `shadow.py` | PASS (grep absent) |

## ε3 — Halt-owner, auto-heal and lifecycle-refusal escalations through the port

| Capability | Binding | Verdict |
|---|---|---|
| halt-state-takes-port | built here — `HaltState` files through the widened port with `FilingPolicy.none(level=1)` (C10); anchored on `FilingPolicy\.`, which no S6 task can write before S1 α | PASS (grep present) |
| auto-heal-takes-port | built here — `MainHealthAutoHeal` files with `FilingPolicy.folded(…)` through the port (C10); anchored on `FilingPolicy\.`; `main_health.py` is S6 δ's new module | PASS (grep present) |
| lane-record-type-gone | anti-inversion — `merge_lane/types.py::EscalationRecord` | PASS (grep absent) |
| s6-lane-filers-substrate | substrate — producer:task-9176, task-9171, task-9186 (S6 θ, δ, φ) | PASS |

## ζ1 — Scheduler sentinel alarms; `sentinel_l1.py` deleted

| Capability | Binding | Verdict |
|---|---|---|
| sentinel-l1-helper-gone | anti-inversion — `orchestrator/src/orchestrator/sentinel_l1.py` exists today | PASS (path absent) |
| scheduler-inline-sentinels-gone | anti-inversion — `Scheduler._FM_READ_STORM_SENTINEL` (5 lines) | PASS (grep absent) |

## ζ2 — One scheduler escalation port

| Capability | Binding | Verdict |
|---|---|---|
| scheduler-port-module | built here — `orchestrator/src/orchestrator/scheduler_escalations.py` (C5) | PASS (path present) |
| seven-hooks-gone | anti-inversion — the `on_warm_base_promote_l2` field declaration and wiring keyword | PASS (grep absent) |
| callbacks-owner-substrate | `grep:orchestrator/src/orchestrator/scheduler.py::SchedulerCallbacks` substrate (S8 owns it) | PASS |
| s8-filer-modules-substrate | substrate — producer:task-9226, task-9227 (S8 ζ1a/ζ1b); upstream through 9236 | PASS |

## η1 — Dispatch-gate, infrastructure and sweep filers; `born_at_l2.py` deleted

| Capability | Binding | Verdict |
|---|---|---|
| born-at-l2-helper-gone | anti-inversion — `orchestrator/src/orchestrator/born_at_l2.py` exists today | PASS (path absent) |
| born-at-l2-callers-gone | anti-inversion — `file_deduped_born_at_l2` called from `lane_record_divergence.py` | PASS (grep absent) |
| s8-sweeps-module-substrate | substrate — producer:task-9228 (S8 ζ2); upstream through 9236 | PASS |

## η2 — `harness.py`'s remaining filings, the drift detectors, the remaining swallow arms

Deps: η1 and κ1 (η2 deletes task 8926's helper `escalation/src/escalation/file_or_refresh.py`, on main
since 2026-11-12, after they convert its two landed callers), ε3 (`harness.py`; η2 is S1's last leaf on
it, and ε3 depends on ζ2); 9218 (S8 β, the helper's third caller) through ε3.

| Capability | Binding | Verdict |
|---|---|---|
| harness-raw-submits-gone | anti-inversion — 22 `_escalation_queue.submit(` lines in `harness.py` at `a85cb1ad78` (23 at `ba17934861`) | PASS (grep absent) |
| abandoned-member-sweep-substrate | `grep:orchestrator/src/orchestrator/merge_deferred_recovery.py::AbandonedMemberSweep` substrate (why fk-1b573c2428c2 is `fixed:2ef04c8241`) | PASS |

## η3 — Filings in the modules S8 and S9 extract from the harness

| Capability | Binding | Verdict |
|---|---|---|
| stranded-reconcile-builds-drafts | built here — `stranded_reconcile.py` (S8 ε2a's module) builds `EscalationDraft`s | PASS (grep present) |
| session-resume-storm-builds-drafts | built here — `session_resume_storm.py` (S8 γ's module) builds an `EscalationDraft` | PASS (grep present) |
| extracted-modules-substrate | substrate — producer:task-9220, 9223, 9224, 9225, 9213 (S8 γ, ε2a, ε2b, ε3; S9 θ) | PASS |

## θ1a — Folded filer async; the on-loop emitters

| Capability | Binding | Verdict |
|---|---|---|
| async-folded-filer | built here — `afile_folded_escalation` (C9), anchored over `fused-memory/src/*folded_escalation.py` (both layouts; post-move `escalators/folded_escalation.py` (S19/OQ-4 (b))) | PASS (grep present) |
| call-site-offload-prose-gone | anti-inversion — the "ASYNC ON PURPOSE" docstring in `server/tools.py` | PASS (grep absent) |
| folded-filer-substrate | `grep:fused-memory/src/fused_memory/middleware/_folded_escalation.py::file_folded_escalation` substrate | PASS |

## θ1b — The other eight folded-escalation callers

| Capability | Binding | Verdict |
|---|---|---|
| string-anchors-gone | anti-inversion — `\banchor_task_id\b` at the 8 call sites in the eight caller modules (word-bounded so `writer_anchor_task_id` cannot hold it red; globs match both layouts). θ1b declares 14 files, the two spies of converted callers included; θ2 removes the transitional keyword | PASS (grep absent) |

## θ2 — Escalators that build their own queue; the ticket janitor's outage alarm

| Capability | Binding | Verdict |
|---|---|---|
| escalator-queue-constructions-gone | anti-inversion — 7 `EscalationQueue(` constructions in the six escalators (globs match both layouts) | PASS (grep absent) |
| transitional-anchor-keyword-gone | anti-inversion — `\banchor_task_id\b` in `*folded_escalation.py` and `*middleware/ticket_janitor.py` (11 lines today: the keyword's definition and uses, and the janitor's `asyncio.to_thread(file_folded_escalation, …)` call, the twelfth C9 caller); θ2 depends on θ1b and removes the keyword (13 files) | PASS (grep absent) |

## ι1 — Recon escalation registry

| Capability | Binding | Verdict |
|---|---|---|
| recon-registry-module | built here — `fused-memory/src/fused_memory/reconciliation/recon_escalations.py` (C8) | PASS (path present) |
| hand-kept-fold-list-gone | anti-inversion — `_RECON_DEDUP_CONFIG` (definition, uses and prose pointers in three modules) | PASS (grep absent) |

## ι2 — The other recon filers and the backfill script; `file_or_fold_l1` deleted

| Capability | Binding | Verdict |
|---|---|---|
| fingerprint-l1-helper-gone | anti-inversion — `dedupe.py::file_or_fold_l1` (5 callers; β and ι2 migrate them) | PASS (grep absent) |
| backfill-resubmit-gone | anti-inversion — `backfill_recon_escalations.py::apply_plan`'s `queue.submit(canonical)` | PASS (grep absent) |

## κ1 / κ2 / κ3 — The remaining orchestrator filers

| Leaf | Capability | Binding | Verdict |
|---|---|---|---|
| κ1 | ladder-module-id-minting-gone | anti-inversion — 5 `make_id(` calls in the five modules at `a85cb1ad78` (6 at `ba17934861`); the filer mints ids | PASS (grep absent) |
| κ2 | four-module-id-minting-gone | anti-inversion — 6 `make_id(` calls in the four modules | PASS (grep absent) |
| κ2 | sink-eval-lane-copy-gone | anti-inversion — `make_escalation_sink`'s own `eval_lane_provenance` call | PASS (grep absent) |
| κ3 | seven-module-id-minting-gone | anti-inversion — 6 `make_id(` calls in six of the seven modules | PASS (grep absent) |
| κ3 | slot-lease-builder-gone | anti-inversion — `slot_lease.py` returns a hand-built `Escalation` | PASS (grep absent) |

## κ4 — The steward's level-1 re-escalation

| Capability | Binding | Verdict |
|---|---|---|
| steward-constructions-gone | anti-inversion — 2 `EscModel(` constructions in `steward.py` (`EscModel.from_dict` stays) | PASS (grep absent) |
| steward-give-up-awaits-filing | built here — `async def _auto_escalate_to_human(` | PASS (grep present) |

## λ1 — Eval cells write nothing to production stores (Leo, S1-2)

| Capability | Binding | Verdict |
|---|---|---|
| eval-profile-declares-transcript-root | built here — `EVAL_PROFILE` gains `transcript_archive.root` under `eval_worktree_root` | PASS (grep present) |
| cell-factory-refuses-production-stores | built here — `EvalHermeticityError` from `_cell_workflow` (C11, row 26) | PASS (grep present) |
| zero-write-boundary-row | built here — B11 in `orchestrator/tests/test_eval_boundary_suite.py` (row 27) | PASS (grep present) |
| eval-construction-point-substrate | `grep:orchestrator/src/orchestrator/evals/profile.py::EVAL_PROFILE` `grep:orchestrator/src/orchestrator/evals/runner.py::_cell_workflow` `grep:orchestrator/src/orchestrator/evals/snapshots.py::eval_worktree_root` substrate | PASS |

## λ2 — Eval verify logs under the eval root; invariant `eval-writes-no-production-state`

Deps: λ1; μ (`evals/runner.py`: this plan's three tasks on that file are chained λ1 → μ → λ2, lead
2026-11-13); 9232 (end of the wave-2 `workflow.py` chain). S1 δ and ω depend on λ2.

| Capability | Binding | Verdict |
|---|---|---|
| workflow-verify-log-root-seam | built here — `build_workflow(…, verify_log_root=None)`; hard-coded `project_root / 'data' / 'verify-logs'` today | PASS (grep present) |
| eval-invariant-minted | built here — the slug in `docs/legibility/design-invariants.md` | PASS (grep present) |

## μ — Corpus eval cells under the production sandbox (Leo, R-7 = A)

Deps: λ1 (`evals/runner.py`, same-file rule, lead 2026-11-13). Grants: production's one derivation,
`agents/write_set.py::compute_write_set`; a needed write gets an entry there, never an eval-only switch-off.

| Capability | Binding | Verdict |
|---|---|---|
| corpus-sandbox-override-gone | anti-inversion — `build_eval_orch_config`'s `SandboxConfig(enabled=False)` override | PASS (grep absent) |
| cell-records-sandbox-backend | built here — `EvalMetrics.sandbox_backend` | PASS (grep present) |
| sandbox-config-substrate | `grep:orchestrator/src/orchestrator/config.py::SandboxConfig` `grep:orchestrator/src/orchestrator/evals/shadow_invocation.py::build_shadow_orch_config` substrate | PASS |

## ω — Integration gate and invariant `escalation-filing-one-door`

| Capability | Binding | Verdict |
|---|---|---|
| one-door-guard | built here — `escalation/tests/test_filing_one_door.py` (rows 20–22) | PASS (path present) |
| invariant-minted | built here — the slug in `docs/legibility/design-invariants.md` | PASS (grep present) |
| level-one-forwarder-gone | anti-inversion — `EscalationQueue.has_open_l1` (41 non-test calls in 19 modules by AST at `a85cb1ad78`, converted by the leaves) | PASS (grep absent) |
| public-dedupe-filers-gone | anti-inversion — `dedupe.py::submit_or_dedupe` and `submit_or_dedupe_off_loop` | PASS (grep absent) |
| consistency-test-substrate | `grep:scripts/tests/test_design_invariants_consistency.py` substrate (`PINNED_SITES`) | PASS |
