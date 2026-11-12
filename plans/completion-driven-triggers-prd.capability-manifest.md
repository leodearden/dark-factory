# Capability manifest — completion-driven-triggers

PRD: `plans/completion-driven-triggers-prd.md` (with Leo's 2026-10-05 rulings, branch
`wip/quality-rulings-1005` at `2bdfa20ce2`).
Built at decompose on 2026-10-05 against main tip `1e73930085`. Anchors are cited as
`path::symbol`. No `file:line` anchors are used, and none are needed.
Machine-readable twin: `plans/completion-driven-triggers-prd.capability-manifest.yaml`.
Every `task_id` is `null` there until `commit_planning` stamps it.

The batch has eight labels: α, β, γ, δ, ζ, η, θ, λ. Two labels in PRD §10 are not filed:

- **ι** (`skills/_shared/filing-the-trigger-chain.md`) landed with the contract. It is on
  main already.
- **ε** (the `--pending` report) is coalesced into δ, which PRD §10 permits while δ has
  not dispatched. At decompose, neither task exists. ε's 150–300 LOC would sit at the
  overlay's ~100-LOC floor, and both labels edit the same file, so ε's capabilities are
  bound under δ. λ's dependency on ε becomes a dependency on δ.

Out-of-batch producers: task **5257** (pending, low; category vocabulary), which β and ζ
depend on, and the `task-metadata-lookup` PRD's **L1** (`find_tasks_by_metadata`), which
γ and δ depend on. L1 is in the same project and is filed by that PRD's decompose, so
this sidecar references it through the dependency and carries no `external_task_id`.

## Substrate re-verification (G3)

Every row of PRD §6 was re-checked by grep on 2026-10-05. All of them hold.

| Capability | Evidence | Verdict |
|---|---|---|
| Predicate exit-code dispatch | `orchestrator/src/orchestrator/deterministic_runner.py::DeterministicRunner._run_predicate`. Its `if rc != 0:` arm returns `_file_milestone_check_failed_and_block`, and `rc == 0` returns DONE with `deterministic-milestone` | PASS wired |
| Trailing-JSON payload extraction, 400-char note cap | `::_extract_predicate_payload`, `::_summarize_predicate_output`, `::_PREDICATE_NOTE_MAX_PAYLOAD_CHARS = 400` | PASS |
| Predicate resume re-runs the check | `DeterministicRunner.run` §1: when `gate_escalated_at` is set and nothing is pending, `before_done.get('kind') == 'predicate'` leads to `_run_predicate` | PASS wired |
| Pure-gate filing seam shared with strand recovery | `::build_milestone_gate_escalation_fields`, consumed by `DeterministicRunner._file_milestone_gate_and_block` and by `harness.py::Harness._recover_stranded_deterministic_gate` | PASS wired |
| Retry cap never sees a deterministic outcome | In `harness.py::Harness._run_slot`, `_run_deterministic_slot` returns early. The finally block notes "report is None, e.g. the deterministic-gate early return", so `_apply_retry_cap` is not reached | PASS |
| `REQUEUED` outcome | `orchestrator/src/orchestrator/workflow_types.py::WorkflowOutcome.REQUEUED` | PASS |
| `at` is read live on every tick | `orchestrator/src/orchestrator/scheduler.py::Scheduler._milestone_time_gated` | PASS wired |
| Cancelled deps satisfy | The `Scheduler._deps_satisfied` docstring says a dependency is satisfied when its status is in `TERMINAL_STATUSES` (`done` or `cancelled`) | PASS wired |
| Event store and typed events | `orchestrator/src/orchestrator/event_store.py::EventType`, `::EventStore.emit`. The scheduler holds `self.event_store` and the code tolerates None | PASS |
| Green-tier config registry | `orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS`, built from `_submodel_leaf_paths` groups. **No `deterministic` sub-config exists today**, so β creates it. That is in β's scope, so it is not a G3 gap | PASS (β produces) |
| Tooling-root resolver | `orchestrator/src/orchestrator/repo_paths.py::resolve_dark_factory_root` has only stdlib imports. Its importers are `harness.py`, `test_harness_watcher_supervisor.py` and `test_repo_paths.py` | PASS |
| Metadata models | `shared/src/shared/task_metadata.py::BeforeDone` (`extra='allow'`, seven declared fields), `::Milestone` (dated/delayed iff), `::TaskMetadata._deterministic_invariants`, `::register_metadata_submodel`, `::_BLESSED_METADATA_KEYS` | PASS |
| Write-boundary enforcement covers `update_task` | `fused-memory/src/fused_memory/backends/sqlite_task_backend.py` calls `parse_metadata(metadata, direction='write', enforce=True)` | PASS wired |
| Submit-guard order | `fused-memory/src/fused_memory/server/tools.py::submit_task` runs the markup guard, then `deterministic_task_error`, then `execution_class_error`/inject, then `recurring_gate_guard_error`, then the later guards, and only then the interceptor's `planning_mode` branch | PASS wired |
| Guard delegation pattern | `fused-memory/src/fused_memory/middleware/deterministic_task_guard.py::deterministic_task_error`, `::_validate_before_done` (containment, the "does not exist under project_root" refusal, `X_OK`), `::_validate_milestone`, `::_validate_recurrence` | PASS |
| Corpus-reading guard precedent | `fused-memory/src/fused_memory/middleware/recurring_gate_guard.py::recurring_gate_guard_error` | PASS wired |
| Priority vocabulary and NULL default | `tools.py::_PRIORITY_TIERS = ('critical','high','medium','low','polish')`; `sqlite_task_backend.py::_row_to_task` returns `row['priority'] or 'medium'` | PASS |
| Deny list and shadow derivation | `escalation/src/escalation/authority.py::L2_AUTO_CLOSE_DENY_CATEGORIES` = {design_concern, milestone_gate, milestone_check_failed, curator_adjudication_missing}. `escalation/src/escalation/shadow_ruling.py::GATED_CATEGORIES` is derived from it, and `::_MILESTONE_CATEGORIES` = {milestone_gate, milestone_check_failed} | PASS |
| Typed per-kind payloads | `escalation/src/escalation/models.py::TrainState`, `::IndexHealthState`, and the optional fields on `::Escalation`. The task-3709 REFACTOR TRIGGER is present | PASS |
| Escalation id minted before the record is built | `escalation/src/escalation/queue.py::EscalationQueue.make_id`, which the runner calls inside the `Escalation(...)` construction | PASS |
| `resume` then `pending` | `escalation/src/escalation/action_effects.py::ACTION_EFFECTS`, `harness.py::Harness._on_escalation_resolved` | PASS |
| Status surface for the capability report | `tools.py::get_status` already emits the top-level `reconciliation_halt` | PASS |
| Script transport | `scripts/legibility/census_trigger.py::post_mcp_tool_call(url, tool_name, arguments)`. A tool `error` envelope rides `isError: false`, so δ must test for the `error` key | PASS |
| Project-root discovery | `scripts/_task_db_scan.py::discover_project_roots` | PASS |
| Active status set | `shared/src/shared/task_statuses.py::ACTIVE` (= every status minus TERMINAL, so it includes `deferred`) | PASS |

Two premises were re-measured with read-only sqlite on 2026-10-05:

- Every stored `before_done` uses only the seven declared keys: 66 rows in the dark_factory
  store and 5 in reify. γ's closed key set therefore refuses nothing that exists.
- `x_finding_run` is carried by 0 tasks and `trigger_chain` by 0 tasks, so no producer has
  written either key yet (INV-13; see δ).

## G6 premise re-validation

Every leaf signal passes. Three notes follow, each resolved at filing without a PRD edit.

1. **The human gate's resume completes by itself (branch 3).** Task 3638 (pending)
   reported on 2026-08-04 that a resolved pure gate strands in `pending`. The live store
   contradicts that today: the runner's own pure-gate resume path wrote
   `done_provenance.note='pure gate resolved'` on 95 tasks, the latest on 2026-10-04. So the
   resolve-to-done path that C4 and C5 rely on (rows 2 and 3) is live. 3638's premise looks
   stale. That is an observation for 3638's owner, not for this batch.
2. **The worst-case verdict fits the note cap (branch 1).** The bound is 400 > 343. The
   worst case uses the run-id grammar `<instrument>-<project_id>-<YYYYMMDD>[-<n>]` from
   contract §5, with `hotspot-survey-dark_factory-20261004-99`, all five tiers at 999,999
   per `[done, open, cancelled]` bucket, weighted sums of 8 digits and compact separators.
   That comes to 343 chars. The cap is `_PREDICATE_NOTE_MAX_PAYLOAD_CHARS = 400`. The
   headroom holds for a `project_id` under about 70 characters.
3. **δ reads keys no producer has written (G2 reader rule, INV-13).** δ's signal therefore
   gains a live clause. Once L1 is live, the CLI run against the real dark_factory store
   exits 2 with `empty_run`, and `--pending` prints "no trigger chain has been filed in
   dark_factory". Both states are distinct from a healthy answer. The fixture rows
   (`SqliteTaskBackend` plus the real tool function) prove the arithmetic and the paging.
   The live clause proves that the reader does not render a never-written store as
   healthy.

Two more premises hold. **The scripts tests can import fused_memory:** the scripts verify
leg runs under `uv run --project shared`, and
`scripts/tests/test_scan_task_toolcall_leaks.py` already imports `fused_memory` there.
**λ can drive all three packages:** `orchestrator/tests/test_workflow_e2e.py` already
imports `fused_memory`, and `orchestrator/tests/test_milestone_integration_gate.py` is the
shape precedent.

## α — Factory-rooted predicate scripts

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `shared.repo_paths.resolve_dark_factory_root` | producer: α moves the stdlib-only module. Check: path `shared/src/shared/repo_paths.py` present | PASS producer α |
| No shim left in orchestrator | Check: path `orchestrator/src/orchestrator/repo_paths.py` absent | PASS |
| `BeforeDone.script_root` | Check: grep `script_root[[:space:]]*:[[:space:]]*Literal` in `task_metadata.py` | PASS producer α |
| Guard resolves the factory root | Check: grep `resolve_dark_factory_root` in `deterministic_task_guard.py` (the consuming entry path) | PASS wired by α |
| Runner resolves the factory root | Check: grep `script_root` in `deterministic_runner.py` | PASS wired by α |
| The signal's script exists in the factory root | `scripts/check_write_triage_enabled.py` exists with mode 0775 | PASS |
| Rejection without `script_root`; deploy+factory refused | The existing "does not exist under project_root" refusal in `_validate_before_done`, plus α's fixtures | PASS rejection-check (manual) |

## β — Re-arm verdict (exit 75)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `BeforeDone.recheck_secs` plus the milestone-dated rule on `TaskMetadata` | Check: grep `recheck_secs` in `task_metadata.py`. The write boundary applies `enforce=True` | PASS producer β |
| `os.EX_TEMPFAIL` arm in `_run_predicate` | Check: grep `EX_TEMPFAIL` in `deterministic_runner.py` | PASS producer β |
| `predicate_verdict` stamp on every verdict-bearing exit | Check: grep `predicate_verdict` in `deterministic_runner.py` | PASS producer β |
| `EventType.predicate_rearmed` / `predicate_recheck_stalled` | Check: grep `predicate_rearmed` in `event_store.py` | PASS producer β |
| `deterministic.predicate_recheck_stall_cap` is green-tier | Check: grep `_submodel_leaf_paths\(.deterministic.\|deterministic\.predicate_recheck_stall_cap` in `config.py` | PASS producer β |
| The cap is read when the runner is built per slot | Check: grep `predicate_recheck_stall_cap` in `harness.py` | PASS wired by β |
| `milestone_check_stalled` is deny-listed and is a milestone category | Check: grep `milestone_check_stalled` in `authority.py`. Vocabulary registration: task 5257, upstream | PASS producer β + 5257 |
| `REQUEUED` bypasses the retry cap | `Harness._run_slot`'s deterministic early return (G3 row) | PASS |
| Row 4: the L2 refuses auto-close | Observed through the `authority.py` check in β's row-4 test | PASS rejection-check (manual) |

## γ — `trigger_chain`, guards, capability report

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `TriggerChain` model registered and blessed | Check: grep `class TriggerChain\(` in `task_metadata.py` | PASS producer γ |
| `TASK_METADATA_CAPABILITIES` (one home) | Check: grep in `task_metadata.py` | PASS producer γ |
| `get_status.task_metadata_capabilities` | Check: grep in `tools.py`. Precedent: `reconciliation_halt` | PASS wired by γ |
| Open-chain guard wired after `recurring_gate_guard_error` | Check: grep `trigger_chain_open_error` in `tools.py` | PASS wired by γ |
| `TriggerChainOpen` refusal (row 7, guard half) | Check: grep in `trigger_chain_guard.py` | PASS rejection, producer γ |
| `UnsupportedTaskMetadata` on unknown `before_done` keys (row 16) | Check: grep in `deterministic_task_guard.py`. Corpus census: 0 rows refused | PASS rejection, producer γ |
| Carrier rules | Check: grep `_validate_trigger_chain` in `deterministic_task_guard.py` | PASS producer γ |
| `TaskInterceptor.find_tasks_by_metadata` | producer: task-metadata-lookup L1, upstream in the same project. The DAG direction was checked: no lookup task depends on this batch. γ passes `include_value=True` because `matched_value` became opt-in in the lookup PRD's 2026-10-05 amendment (decision 4) | PASS producer upstream |
| `'/review'` refused with the bare-name hint (row 15) | γ's fixtures | PASS rejection-check (manual) |

## δ — `scripts/check_run_completion.py` (ε coalesced)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Script exists at the path the census L8b and snippet §2 name | Check: path `scripts/check_run_completion.py` present | PASS producer δ |
| `weighted_completion` importable | Check: grep `def weighted_completion\(` | PASS producer δ |
| Selection over `find_tasks_by_metadata` | producer: L1, upstream. The `x_finding_run` read needs only `status`/`priority` (default `include_value=False`), while `--pending` passes `include_value=True` and pages on `offset += returned` under the lookup's `RESULT_BYTE_BUDGET` (lookup PRD amended 2026-10-05). Transport: `post_mcp_tool_call` (main). Check: grep `find_tasks_by_metadata` in the script | PASS producer upstream |
| Exit 75 from `os.EX_TEMPFAIL` | Check: grep `EX_TEMPFAIL` in the script | PASS producer δ |
| `--pending` mode (ε) | Check: grep `['"]--pending['"]` in the script | PASS producer δ |
| Verdict JSON ≤ 400 chars | floor: 400 > 343 (G6 note 2) | PASS floor |
| 7/10 against 0.7 is exact | `Fraction` identity (row 11) | PASS (manual) |
| The never-written key renders distinctly | 0 `x_finding_run` and 0 `trigger_chain` rows, live; the live signal clause (G6 note 3) | PASS (manual) |

## ζ — `review_due` escalation

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `ReviewDueState` and `Escalation.review_due` | Check: grep `class ReviewDueState\(` in `models.py`. Precedent: `TrainState`/`IndexHealthState` | PASS producer ζ |
| `review_due` deny-listed; shadow derivation automatic | Check: grep `review_due` in `authority.py`. Registration: 5257, upstream | PASS producer ζ + 5257 |
| One builder: `build_gate_escalation_fields` | Check: grep `def build_gate_escalation_fields\(` in the runner, plus a grep in `harness.py` | PASS wired by ζ |
| Old builder name retired | Check: grep `build_milestone_gate_escalation_fields` absent under `orchestrator/src/orchestrator/` | PASS |
| `review_due.verdict.share` populated (row 2) | Field population: β (upstream through γ) writes `predicate_verdict` as a parsed dict | PASS field-population (manual) |
| `resume` drives the human gate to `done` (row 3) | 95 runner-completed pure gates (G6 note 1) | PASS |

## η — Watcher `review_due` handler

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| A `### review_due` section routes on the category and payload | producer: ζ, upstream. The skill text is verified by review and by first live use. A prose grep would be the guard that INV-10 forbids | PASS (manual) |
| Carve-outs | "Always ask — keyed on record content" lists deterministic-runner filings, which is where `review_due` lands through `shadow_ruling`, and `spend_or_eval_launch`. η names `review_due` as the exception (D10) and states that the spawn is not a spend launch | PASS (manual) |

## θ — `docs/task-authoring.md`

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| §5/§6/§6.1 document `recheck_secs` and `script_root`, plus the 75 rows | Check: grep `recheck_secs` in `docs/task-authoring.md` (λ row 14 executes the blocks) | PASS producer θ |
| §8 Tier-A lists `trigger_chain`, `predicate_recheck` and `predicate_verdict` | Check: grep `trigger_chain` in `docs/task-authoring.md` | PASS producer θ |

## λ — Integration gate (rows 1–16)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| All 16 rows green in merge verify | DAG direction: α, β, γ, δ (with ε), ζ and θ are upstream, and L1 is upstream transitively. Precedent: `orchestrator/tests/test_milestone_integration_gate.py` | PASS (manual) |

All 31 mechanical checks pass the authoring-time polarity lint
(`shared/src/shared/delivered_check_polarity.py::lint_delivered_checks`, ref `main`,
declared `metadata.files` per leaf), with zero findings. No `declared-only`, `test-only`,
`producer-absent`, `producer-extent-short`, `producer-downstream`, `fixture-ERROR`,
`bound≤floor` or `rejection-absent` binding was found, so the batch is clear to queue.
