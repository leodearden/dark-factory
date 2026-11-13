# S10 — Task-write contract: metadata modes, success signal and registry (PRD)

**Status:** draft, authored 2026-11-01, folded twice 2026-11-02 (critic `critique/S10.critic.md`, delta
`critique/S10.critic2.md`), and folded a third time on 2026-11-10 under the wave-2 sitting's rulings
(`## Rulings (sitting of 2026-11-10)` below; log `critique/S10.rulingfold.md`), by the S10 seats of the
`/review-all` remediation program (`plans/review-all-program-dark_factory-2026-10-10.md`, stream S10, wave 2,
size L, risk high), then fold-verified the same day (`critique/S10.foldverify.md`). Every design question
is ruled; no open-question marker remains. B+H: contracts C1–C7, a
two-way boundary-test sketch and an integration-gate leaf ω. Twenty-one leaves in the decompose batch plus
one deterministic check (σ) that the lead files once its script is on main.

**Code anchors** verified against main `e634cfe658` (2026-11-10). Main moves fast — cite-by-symbol;
re-locate lines at implementation time. Main moved 2,532 commits since the previous stamp (`d40ba048e9`,
2026-11-02); 87 of the paths this plan touches changed in that window, and every symbol below was re-read
at `e634cfe658`. Every count is `measured at e634cfe658` unless the row says otherwise. Store counts are
read-only reads on 2026-11-10 of the dark-factory store (`.taskmaster/tasks/tasks.db`, 9,072 rows) and,
where a row says so, of all 11 stores on this host. The run's anchors were verified at `6551b75aba`
(2026-10-08); `git diff 6551b75aba..main` is non-empty for every file named here. Re-checked at main
`bb1c659562` (2026-11-10, 77 commits later): that window changes nothing under `shared/`,
`fused-memory/src/`, `skills/` or `docs/`, and its `workflow.py` and `escalation/server.py` edits touch no
`update_task` call, status comparison or declared-files read; the `update_task` AST (44 orchestrator and
escalation calls; 12 readers, 14 outside `try`, 30 inside), the 9 `files_to_modify` read positions and the
task-status literal counts reproduce there, and all 63 mechanical sidecar checks still FAIL there.

## Goal

A task-metadata write has one contract from the MCP tool to the sqlite row, and the orchestrator hears
about a failed write in exactly one way.

- Two orthogonal typed axes on the write — `metadata_mode: Literal['merge','additive','replace']`,
  required whenever metadata is sent, and `text_mode: Literal['replace','append']`, applied only to
  `details`/`prompt` — resolved once, in the backend. The `append` flag, which drove both axes and was
  resolved twice (orchestrator client and backend) under rejection carve-outs, leaves the wire.
- `Scheduler.update_task` raises a typed `MetadataWriteFailed` carrying the rejection envelope and
  returns `None`. The fourteen `try/except Exception` arms around `update_task` in `workflow.py` become
  live as written, so the three money-burning-loop guards (`_handle_no_plan_failure`,
  `_check_infra_resume_thrash`, `_check_merge_outcome_thrash`) escalate on a lost counter as their
  docstrings promise. `SchedulerFacade`, `FakeScheduler` and `FakeMetadataBackend` declare the same
  signature (R3), and a guard test fails when any code reads the call's result.
- `parse_metadata`'s slice set is one frozen table in `shared.task_metadata`, identical in a fresh
  interpreter whatever first-party module was imported first — a new design invariant,
  `metadata-slices-one-table` (R22 item 4).
- One `declared_files(metadata)` reader, over `metadata.files` only, feeds `candidate_key`,
  `CandidateTask` and lock derivation; `metadata.files_to_modify` is refused at write and read nowhere;
  one parse of the incoming metadata per request; every backend rejection reaches the interceptor's
  callers as a dict; the status FSM is enforced in the backend only, from one transition verdict at one
  call site; a `PrivilegedAuditWriter` protocol replaces the three `type: ignore[attr-defined]` calls;
  one candidate-key scope constant and one collision helper; one rule says which task is a human gate;
  the curator combine writes the union of the target's, the candidate's and the curator's files into
  `metadata.files`, and a collision is a counted refusal that a 30-day check reads; the 333-line
  blessed-key comment block becomes a table in `docs/task-authoring.md` §8.

What a user observes when this lands:

- `mcp__fused-memory__update_task(id, metadata={…})` without `metadata_mode` is refused with
  `error_type='WriteModeError', error='metadata_mode_required'` and a `repaired_call`; `append` is
  refused with `error='append_retired'` naming `metadata_mode='additive'` / `text_mode='append'`; a
  details append is `details=…, text_mode='append'` and no longer changes how metadata merges. Recon
  Stage 2's hint attach keeps working because its system and user prompts are rewritten with the wire.
- A refused metadata stamp from the orchestrator (a `done_provenance_via_update_task` rejection, or
  fused-memory unreachable) surfaces as the arm's documented escalation or block, not as a `Failed to
  update task` ERROR line followed by the phase loop proceeding on a counter it did not persist.
- `python -I` importing any one of `shared.task_metadata`, `fused_memory.backends.sqlite_task_backend`,
  `orchestrator.scheduler`, `orchestrator.cli`, `dashboard.app` and printing
  `sorted(registered_slices())` prints one identical set — every slice registered anywhere today
  (fourteen at `e634cfe658`) plus any that tasks 5281 and 8453 add. The same probe over
  `_SUBMODEL_REGISTRY` gives 9, 12, 11, 9 and 10 keys today.
- A curator combine leaves the target's `metadata.files` a superset of what it held before; its
  `details` no longer start with `FILES_TO_MODIFY:`. A combine refused because its union collides with
  another live task's key writes `refused_reason='candidate_key_collision'` to `data/combine_audit.jsonl`,
  and 30 days after the combine change lands a deterministic check counts those refusals across every
  project and escalates to Leo when they are frequent.
- A `submit_task` or `update_task` whose metadata carries `files_to_modify` is refused with
  `error='files_to_modify_retired'` and a corrected call that moves the paths into `files`.
- A status write that breaks Table A is refused by the backend — through the MCP tool, the interceptor,
  a maintenance script or a shell snippet alike — with the tokens callers already branch on
  (`terminal_exit_rejected`, `illegal_transition`).
- Every fused-memory question "is this task a human gate?" gets the answer dispatch acts on: deterministic
  and `always_escalates`. The recon duplicate-gate guard therefore sees the ~98 live always-escalating
  gates it ignores today; its refusals are counted for a week after λ deploys.
- The next agent adding a metadata slice edits one table in one file; widening the lock-charter
  extension list edits `shared.locking` only; blessing a key edits the frozenset and one docs table row
  that a drift test pins.

Heuristics this PRD acts on (`docs/code-quality.md`, cited by name): **3 carefully factored orthogonal
dimensions of variability** (the two write axes), **12 structured data instead of meaningful strings**
(typed modes, typed outcomes, declared error codes, `TaskStatus` members, no `str(exc)` sniff), **10
clear invariants informatively, redundantly, uniformly enforced** (one transition verdict enforced at the
lowest layer every writer passes; deterministic invariants at submit and update), **11 SPOT** (one slice
table, one declared-files key and reader, one scope constant, one gate rule, one lock-charter copy, one
error renderer), **7 prefer stateless interactions between modules** (no import-order registry), **9 deep
modules with coherent narrow interfaces** (`PrivilegedAuditWriter`, `TaskStore`), **13 files make
internal sense in isolation** (slice models in leaf modules), **14 no file too large** (the
interceptor partition follows the heuristic's own measurement), the **Comments** stance (the blessed-key
essays leave the code) and the **Tests** stance (tests stop patching `_SUBMODEL_REGISTRY`,
`_is_gate_metadata`, `_extract_meta_files_from_meta`; interceptor tests build through one public
factory; the doubles model the raising contract).

## Background

Synthesis stream S10 of `plans/review-all-dark_factory-20261008.md` (`## Candidate streams`); remedy,
`deletable` and `seam_candidates` are the record's `synthesis.streams[S10]`. Deliberation ruled option 1
for the only open choice (R3: `Scheduler.update_task` raises) and promoted the registry invariant (R22
item 4). Nothing here relitigates either. The wave-2 sitting of 2026-11-10 amended one half-sentence of
R3 (the status FSM is enforced in the backend only; see `## Rulings`). S10 has no R23 weakened keys.

The G4 table makes S10 the owner of the `shared.task_metadata` wire contract (`metadata_mode`,
`text_mode`, static slice registry, `declared_files`), `MetadataWriteFailed`, `PrivilegedAuditWriter`
and the candidate-key scope SQL; the fused-memory backend and interceptor, the orchestrator scheduler
(S17) and workflow (S9) import them; `SchedulerFacade`/`FakeScheduler` parity is S10's. The S9 draft of
this wave consumes C2 in S9 λ1 (after β2), the β1 signature parity in S9 μ1 (after β1) and C3 in S9 ξ
(after η), under those names.

Since the first draft (`ce0f71a384`), task 6387 (merge `4e19890832`, 2026-11-02) added two readers of
`update_task`'s result to `deterministic_runner.py` (`_rearm_predicate`, `_run_predicate`). Between the
2026-11-02 fold and `e634cfe658` the paths this plan depends on moved as follows (each re-read; counts
below include them):

- task 6388 (`TriggerChain`, done), 8142 (`cancel_dispositions`, done) and 8413 (`declared_scope`, done)
  registered three more slices inside `shared.task_metadata`; task 8415 (done) added
  `shared/src/shared/followup_items.py`, a fifth slice registered by import side effect.
- task 7276 (2026-11-10) added `orchestrator/src/orchestrator/scheduled_deploy_settle.py`, whose own
  `TaskStore` protocol declares `update_task(...) -> bool` and whose `_write` reads the result.
- `workflow._write_branch_base_sha` is a new `update_task` reader and passes `append=not overwrite`.
- tasks 8330 and 6550 reshaped `services/live_workflow_detector.py`; its pure-gate rule is
  `is_pure_gate_metadata`, read by three other modules.
- task 8413 made `_execute_combine` re-stamp `declared_scope` from the curator's file list; task 8453
  (pending, high) will extract that write half as `_write_combined_rewrite`.
- `fused-memory/src/fused_memory/reconciliation/enumeration_guard.py` now instructs recon Stage 2 to
  send `append=True`.
- tasks 6392, 6708 and 6558 closed (6558 as `found_on_main`, its premise gone); task 4867 is being
  hand-carried by Leo's ruling of 2026-11-10.

## Rulings (sitting of 2026-11-10)

Every open question of this stream was ruled at the wave-2 sitting (`digests/wave2-rulings-applied.md`
§S10; Leo's answers `digests/wave2-leo-answers-2026-11-10.md`; for a changed ruling the measured "Effect
on the plan if adopted" in `digests/discussion/S10.md` is part of the ruling). The text below is authored
under them.

| Id (sitting key) | Ruling | Ruled by |
|---|---|---|
| OQ-3 (S10-1) | As recommended: μ2 builds a frozen `InterceptorCollaborators` bundle at construction; the lifecycle-reset filer is the one provider member; μ2 waits on S26 γ. | Leo |
| OQ-4 (S10-2) | As recommended: the task-status literal guard covers task-status operands only, syntactically, with no exemption list (D13). | Leo |
| OQ-6 (S10-3) | New option: the status-transition rule is enforced in the **backend only** — one verdict function in `shared.task_transitions`, one call site in the backend's status writers inside the write transaction, `actor` required, the reopen reason passed. The interceptor's terminal-exit gate and legality gate are deleted; it renders the backend's refusal through the one error renderer and keeps the tokens. **Amends R3 (2026-10-10)**: "the status FSM enforced in the backend as well as the interceptor" becomes "in the backend only". | Leo |
| OQ-8 (S10-4) | As recommended: task 3260 is amended ((b) fixed by 3446; the combine is a sanctioned `files` writer through `shared.locking.strip_directory_locks`) and waits on ο and ξ. | Leo |
| OQ-9 (S10-5) | New: `metadata.files_to_modify` is destroyed. Fused-memory refuses an incoming metadata payload carrying it (submit and update, incoming keys only) with an error naming `files`; `declared_files` reads `files` only; every other reader of the key is deleted; a test fails if any module under `*/src` reads it; the lead backfills the 3 rows that differ after the refusal deploys. The precedence question is withdrawn. The details-text prefix stays as planned (stop writing, age out). Renaming the curator's internal field is an optional low follow-up outside this plan. | Leo |
| OQ-10 (S10-6) | New: one rule — a task is a human gate when it is deterministic and `always_escalates` is true (stored shape; for a submission, after the submit-time coercion). Refinements are named properties; the blocked-gate audit and the curator sweep keep their own honestly named selectors; the four other consumers move onto the rule; a one-week count of extra recon duplicate-gate refusals is the live signal. | Leo |
| OQ-14 (S10-7) | Option A plus Leo's check: a collision is an audited refusal with the distinct token `candidate_key_collision`; `scripts/combine_collision_census.py` must be on main before the check task is filed; a deterministic delayed check 30 days after the combine task (frequent = ≥ 5 collisions or ≥ 1% of combine writes). | Leo |
| OQ-5 | Revised by the lead on 2026-11-10 (cross-stream): S6 α `lane-budgets-exceed-verify-timeouts` is INV-15, S10 θ `metadata-slices-one-table` INV-16, S7 ζ `merge-gate-unmeasured-typed` INV-17 (slugs as the lead's sequencing rulings of 2026-11-10 spell them); edges S6 α → θ → S7 ζ; checks anchor on the slug. | lead |
| OQ-7 | Superseded by Leo on 2026-11-10: task 4867 is hand-carried now, so α depends on 4867 as originally drafted; the earlier inversion is void. | Leo |
| OQ-1, OQ-2, OQ-11, OQ-12, OQ-13, OQ-15 | Unchanged lead rulings of 2026-11-02 (C1 in `shared/task_metadata_wire.py`; C2 in `orchestrator/task_write.py`; `shared/delivered_check.py`; `error_envelope` in `shared.mcp_envelope` with 6719 amended; `-> None` plus the AST guard; legacy prefixed rows age out). | lead |

## Findings disposition

Every key of the stream appears exactly once, plus one carried mechanical key. No key is `fixed` or
`stale` at `e634cfe658` (every one of the 50 earlier delivered checks still fails there); two are partly
improved since the run and one is worse.

| Key | Sev | Disposition | Re-verification (measured at `e634cfe658`) |
|---|---|---|---|
| `fk-ff5234ac1013` write semantics in two overlapping parameters | medium | leaf α (client half β1) | `sqlite_task_backend.py::_resolve_metadata_mode(metadata_mode, append, *, metadata_present)` unchanged in shape, three rejections (invalid mode; `merge`+`append=True`; bare `append=False`); `Scheduler.update_task` still resolves `'additive' if append else 'merge'` and never forwards `append`; `SchedulerFacade.update_task` declares `append` only; 5 `# type: ignore[reportCallIssue]` on `metadata_mode=` calls in `workflow.py`. AST of non-test `update_task` calls (attribute and bare-name calls): 55 (orchestrator 43, fused-memory 11, escalation 1) — `'merge'` 27, neither keyword 18, `append=True` 3, `append=not overwrite` 1, `'additive'` 2, `'replace'` 3, pass-through 1. `git grep -cE "append=(True\|true)" -- fused-memory/src`: 66 hits in 9 files (`reconciliation/enumeration_guard.py` new). `server/tools.py::update_task`'s docstring: 123 lines, 22 naming `append`/`metadata_mode`. |
| `fk-116481623cfb` per-call mode, two carve-outs, `append` drives text | high | leaf α | Both carve-outs present; the "``append`` IS NOT SCOPED TO METADATA (task 4216)" block still drives the `details`/`prompt` concatenation; `AppendUnsupportedFieldError` still guards title/description/priority and its docstring still speaks of `append=True`. |
| `fk-de964b86bdd8` client returns False, workflow written for a raise | high | leaves β1, β2 | `Scheduler.update_task -> bool`: every tool error, structured rejection and transport exception returns `False`. `workflow.py`: 16 calls, 14 inside `try/except Exception`, 4 read the return (`_write_branch_base_sha` new). Orchestrator and escalation call sites: `workflow.py` 16, `harness.py` 9, `deterministic_runner.py` 8, `dry_run_unblock.py` 3, `scheduler.py` 2, `routing_dispatch.py` 1, `strand_conversion.py` 1, `merge_lane/skew_tripwire.py` 1, `stranded_verified_green.py` 1, `scheduled_deploy_settle.py` 1 (new); plus `escalation/src/escalation/server.py` 1. Twelve sites read the result; 14 sit outside any `try`; 30 sit inside one. Test files touching the contract: 52 at `d40ba048e9` plus `orchestrator/tests/test_scheduled_deploy_settle.py`; decompose re-measures. `.venv/bin/pyright` reports no error for a bool read of a `-> None` coroutine (critic, pyright 1.1.408, default and strict). Task 6129 is a symptom of this key. |
| `fk-5c98cc0bb7e0` (carried, mechanical; workflow-recovery.8) `SchedulerFacade` declares only `append`, so `metadata_mode=` calls carry suppressions | — | leaf β1 | Filed as task 3150, which carries no `x_finding_key` and is cancelled into β1 (lead action); β1 stamps the key (contract §8). |
| `fk-f6224c565a9b` no typed write outcome; two success predicates | medium | leaf γ | Improved (S14 ζ, task 6711 done): `interceptor_write_succeeded` is `shared.mcp_envelope.write_failure_reason(resp) is None`. Still present: `TaskInterceptor.update_task` converts five write-authority exceptions via `to_error_dict()`; `AppendUnsupportedFieldError`, mode `TaskmasterError`, `LeakedEnvelopeMarkupError`, `DuplicateCandidateKeyError` and pydantic `ValidationError` propagate; the CSV branch of `set_task_status` computes `all_ok` with its own `r['result'].get('error') is None`. |
| `fk-1f8a4c69798f` 8+ divergent metadata parsers | high | leaf γ | Improved: `fused_memory.middleware.metadata_dict.raw_metadata_dict` is the reader for 11 modules. Still present: `task_interceptor.py` keeps `_parse_metadata_value`, `_warn_metadata_discard`, `_METADATA_DISCARD_CODES`, `_parse_metadata`, `_extract_metadata_dict`, `_fresh_metadata_copy` and a `json.loads` in `_submit_task_planning_mode`; `sqlite_task_backend.py` has 14 `json.loads(` sites and `_validate_metadata_write` parses the merged blob twice. |
| `fk-fa3e3d765f50` seven declared-files readers, four rules | medium | leaf δ (orchestrator half ε) | All seven present (`sqlite_task_backend.py::_files_for_key`, `TaskInterceptor._extract_meta_files_from_meta`, `task_curator.py::_task_files`, `task_interceptor.py::_extract_metadata_files` — imported privately by `reconciliation/targeted.py`, `lock_charter_guard.py::extract_files`, `TaskInterceptor._extract_deliverable_signals_from_meta`, `shared/briefing_queries.py::_string_tuple`); `_DELIVERABLE_SIGNAL_KEYS` and the inline `('files', 'files_to_modify')` literal both remain; `path_scope_guard.py` documents and acts on the `files` ∪ `files_to_modify` union. Under OQ-9 (ruled) the second key is retired outright. The literal `'files_to_modify'` sits in a read position (`.get` argument, subscript, key tuple) at 9 `*/src` sites: 4 read task metadata (backend 1, curator 1, interceptor 2) and 5 read the curator's own field (`task_curator.py` payload/LLM-output/wire-shape reads). All 11 stores: 4 rows carry `metadata.files_to_modify`, all `done` (dark-factory 2350, 2507; reify 4344, 4865); 3 hold paths absent from `files`; 0 non-terminal rows carry it. Task 6653 (pending) carries the curator/interceptor half of this key and is cancelled into δ. |
| `fk-f528087f44c7` registry depends on import order | high | leaf η (invariant θ) | **Worse than at the run, and worse than on 2026-11-02.** `task_metadata.py` registers 9 slices (`cancel_dispositions`, `declared_scope`, `trigger_chain` new); import side effects register 5 more (`deploy_state.py`, `capability_manifest.py`, `defects.py` ×2, and `followup_items.py`, new). Fresh-interpreter probe (root `.venv` python `-I`, member `src/` dirs bound in the child): `shared.task_metadata`, `orchestrator.cli` → 9; `dashboard.app` → 10; `orchestrator.scheduler` → 11; `fused_memory.backends.sqlite_task_backend` → 12; `fused_memory.middleware.task_interceptor` → 13; `orchestrator.workflow`, `orchestrator.deterministic_runner`, `fused_memory.server.tools` → 14. Six distinct registries over nine entry modules. Task 6558 closed `found_on_main` (its cycle premise gone). Task 8453 (pending) registers a `pack_record` slice. |
| `fk-941b5369a3d7` 308 comment lines around 51 keys | medium | leaf ι | Header comment (`# Tier-A: the load-bearing …`) through the closing paren of `_BLESSED_METADATA_KEYS`: 398 lines, 333 `#`-led, 58 keys. File: 1,958 lines. The §8 mirror's markers are read by three test modules (`test_task_authoring_blessed_keys_drift.py`, `test_task_authoring_tier_b_canonical_keys.py`, `task_authoring_marker_span.py`). |
| `fk-d80c1c44bca1` FSM table consulted once, after gate 2a | medium | leaf κ1 | `is_legal_transition(` is called at one fused-memory site, gate 2e of `TaskInterceptor._apply_status_transition`, behind `task_status.enforce_transitions` (schema default `False`, deployed `true`, restart-only); gate 2a's inline terminal check (`_terminal_exit_error`) runs first, unconditionally, and duplicates the table's terminal clause. `SqliteTaskBackend.set_task_status` enforces vocabulary only. Direct backend status writers outside the interceptor: `fused-memory/scripts/audit_duplicate_tasks.py` (→ `cancelled`, pending/in-progress losers only), `fused-memory/scripts/correct_found_on_main_backlog.py` (done → pending with a reopen reason) and the v3→v4 self-heal migration; none has written an illegal transition (11-10 discussion: 0 `auto_cancelled_by_self_heal` stamps in dark-factory; 1 live `done_provenance` without `reopen_reason` across all stores, reify 2562, origin unknown). |
| `fk-d4d61005457b` candidate-key scope re-spelled, `str(exc)` sniff | low | leaf κ2 | 10 copies of `status != 'cancelled'`: 5 in migrations (one of them the `_migrate_v2_to_v3` backfill scan, not the index scope), 5 in live survivor SELECTs (`tag = ? AND candidate_key = ? AND status != 'cancelled'`); 5 `DuplicateCandidateKeyError(` sites; 3 `'candidate_key' not in str(exc)` classifications; index `ux_tasks_candidate_key`. Unchanged since 2026-11-02. |
| `fk-260352fa900a` privileged writers off the protocol | low | leaf κ2 | `TaskBackendProtocol`: 23 members under a "12-method + lifecycle surface" docstring; three privileged writers undeclared; 3 `type: ignore[attr-defined]`; the interceptor fixture stubs `set_status_and_stamp_audit` with a bare `AsyncMock`. |
| `fk-cc04e61d23ec` deterministic invariants only at submit | low | leaf γ | `deterministic_task_error` callers: the `server/tools.py` submit path and `recurrence_mint.py` only. Task 3093 carries this work. |
| `fk-06ab2d3e9148` four gate predicates, no home | medium | leaf λ | Six rules (11-10 discussion, re-read at `e634cfe658`): `TaskInterceptor._is_gate_metadata` (combine guard; three clauses); `recurring_gate_guard.is_gate_submission` (`execution_class == 'operational'`, `operational_mode` absent or `'gate'`, read before coercion); `gate_owned_finding_phrasing`'s selector (`operational_mode == 'gate'` or `always_escalates is True`); `blocked_gate_audit_section` (deterministic and `operational_mode` absent or `'gate'`, or a `gate_escalated_at` stamp); `curator_gate_resolution_sweep`, the interceptor's closure gate and `consolidation_gate.handrolled_member_enumeration` / `::declared_gate_topic` (`operational_mode == 'gate'`, the closure readers also needing the block under `consolidation_gate.GATE_METADATA_KEY`); `services/live_workflow_detector.py::is_pure_gate_metadata` (truthy `always_escalates`, no `before_done`), read by `recon_write_policy.py`, `reconciliation/harness.py` and `live_workflow_section.py`. Nine code reads of `operational_mode` across seven modules. A seventh copy appeared on 2026-11-05: `dashboard/src/dashboard/data/prd_view.py::is_human_gate`, which already spells the ruled rule. Dispatch hands a task to a human exactly when it is deterministic and `always_escalates` is true. All 11 stores: 14 rows the interceptor's rule calls gates dispatch to an agent (1 live: reify 5376); ~98 live tasks dispatch to a human but are invisible to the recon duplicate-gate guard. |
| `fk-c41aa02e0779` `get_statuses_fresh` returns `{}` on failure | medium | leaf κ2 | Returns `{}` when the db file is absent and on any `Exception`; consumers: `reconciliation/harness.py`, `orphaned_recon_escalation_sweep.py` (two calls), `fused-memory/scripts/derive_orphaned_recon_escalations.py`. |
| `fk-85e1293e0aa4` 13 optional collaborators + 4 setters | medium | leaves μ1, μ2 | `TaskInterceptor.__init__`: 16 parameters, 12 defaulting to `None`. `server/main.py::run_server` builds `MemoryService(config)` before both `TaskInterceptor(` constructions and wires journal, scroll and archive immediately after each; the lifecycle-reset filer is wired only after `_build_recon_report_components(…, task_interceptor=…)` returns the `ReconReportState` it needs; `targeted.task_interceptor = task_interceptor` back-wires the reconciler; the reconciliation-disabled arm has no filer; `server/tools.py` builds a bare fallback interceptor. Tests: 247 lines constructing a `TaskInterceptor(` in 49 files, two of them shared stacks (`_hermetic_task_stack.py`, `_submit_stack.py`, new); 9 test files call the setters on an interceptor. |
| `fk-1eb04ceac9e8` three objects sharing a lock | medium | leaf ν | 102 methods, 33 instance attributes, 5,776 class lines; status-transition and path-scope families read disjoint attribute sets; task 6332 owns the ticket-worker extraction. |
| `fk-e3e99447e11e` 7,417 lines, 43% prose | medium | leaf ν | 7,697 lines (7,576 on 2026-11-02); 1,872 docstring + 1,154 comment lines (39.3%). |
| `fk-de3df95984a5` combine writes files into a details prefix | medium | leaf ζ | `_execute_combine` still prefixes `details` with `FILES_TO_MODIFY:` (its comment still says "readers parse it back"; none does) and merges `curator_*` keys, `declared_scope` (derived from the curator's list) and follow-up items; `RewrittenTask.files_to_modify` is the LLM's list. Store, details starting exactly `FILES_TO_MODIFY:`: 502 rows (218 live), 428 with non-empty `metadata.files`, 200 of those holding files absent from the curator's list (upper bound; post-combine plan reconcile also grows `files`). `_append_combine_audit` records no files and no survivor; its four refusal tokens are `COMBINE_REFUSED_*` constants in `task_interceptor.py`. |
| `fk-fa355c2c0498` lock-charter copies | low | leaf δ | `lock_charter_guard.py` defines `FILE_EXTENSIONS`, `is_file_path`, `directory_locks` "duplicated here verbatim" under a false rationale (it imports `metadata_dict` → `shared.task_metadata`). |
| `fk-1695d6b02400` status literals | medium | leaves ξ, ο | `git grep -E "status (==\|!=) '(merge-deferred\|pending\|blocked\|done\|in-progress\|cancelled\|deferred\|review)'"` over `*.py` under `*/src`: orchestrator 68, fused-memory 58, escalation 29, dashboard 5. By operand (2026-11-02 split, totals moved by ±1): about a quarter are comments/docstrings; about 26 compare a typed record's `.status` of another vocabulary (escalation records — S2; merge outcomes — S6; eval and intake results); about 66 compare a task's status. `shared/task_statuses.py`'s docstring still says it "does NOT rewire the call sites". |

### Deletable accounting (synthesis `deletable`)

| Item | Leaf | Note |
|---|---|---|
| the client-side mode resolver | β1 | with `_reject_contradictory_metadata_mode` |
| 13 unreachable except arms | — | **kept, made live** (R3 option 1); 14 at main |
| 5 `type: ignore`s | β1 | the `reportCallIssue` five; the three `attr-defined` go in κ2 |
| 5 metadata parsers with 5 failure policies | γ | |
| 7 declared-files readers | δ, ε | and the `files_to_modify` metadata key itself (OQ-9 ruled) |
| 10 predicate copies and 3 `str(exc)` sniffs | κ2 | live copies onto one constant; migrations keep frozen literal DDL (D9a) |
| the `FILES_TO_MODIFY` details prefix and its false comment | ζ | |
| `lock_charter_guard`'s verbatim copies | δ | |
| 308 comment lines | ι | now 333 |
| (not in the synthesis) the interceptor's two transition gates | κ1 | OQ-6 ruled: the backend is the one enforcement point |

## Consumers and user-observable surface (G1)

| Mechanism (home) | In-batch consumers | Other-stream consumers (G4) | User-observable surface |
|---|---|---|---|
| C1 `MetadataMode`, `TextMode`, `write_modes`, `WriteModeError` — `shared/src/shared/task_metadata_wire.py` (OQ-1 ruled) | α β1 γ π | S9, S17 (client), S11 (combine) | the MCP tool refuses a mode-less metadata write and a retired `append` with named errors |
| C2 `MetadataWriteFailed`, `Scheduler.update_task -> None` — `orchestrator/src/orchestrator/task_write.py` (OQ-2 ruled) | β2 ε ξ ω | S9 λ1 (`BlockReason` row; after β2), S9 μ1 (`DryRunScheduler`; edge on β1, after β2 through λ1), S17 | a refused stamp escalates or blocks per the arm's documented policy |
| C3 `SLICES`, `registered_slices()`, `parse_metadata(…, slices=)` — `shared.task_metadata` | η θ ι ω | every `parse_metadata` caller; S9 ξ | identical slice set in every entry module |
| C4 `declared_files`, `RETIRED_FILES_KEY`, `FilesToModifyRetiredError`, `refuse_retired_files_key` — `shared/src/shared/declared_files.py`; sanitizer `shared.locking.strip_directory_locks` | δ ε ζ | S11 (curator), S9/S17 (lock derivation) | store `candidate_key` and curator key agree; a write carrying `files_to_modify` is refused naming `files`; the combine never shrinks `metadata.files` |
| C5 `transition_verdict`, `IllegalTransitionError`, the backend status writers' `actor`/`reopen_reason`, `PrivilegedAuditWriter`, `TaskStore`, `StatusCensusUnavailable`, `CANDIDATE_KEY_SCOPE_SQL` | κ1 κ2 λ μ2 ν ω | S12/S13 (recon status census) | any status write that breaks Table A is refused by the backend with the values named; a failed census raises |
| C6 `GateDeclaration` (`is_human_gate` and its named refinements and selectors), `gate_declaration`, `CONSOLIDATION_GATE_METADATA_KEY` — `shared/src/shared/gate_declaration.py` | λ | S12/S13 (recon gate selectors) | one rule for "is a human gate", the one dispatch acts on |
| `combine_audit` (`CombineRefusal`, `combine_audit_path`, `append_combine_audit`) — `fused-memory/src/fused_memory/middleware/combine_audit.py`; `scripts/combine_collision_census.py` | ζ ρ σ | the lead (σ's verdict) | a collision is a counted refusal; σ escalates to Leo when frequent |
| `InterceptorCollaborators`, `build_interceptor` (tests) | μ1 μ2 ν | S15 (server composition), S26 γ | an un-armed gate is a construction-time error |
| `StatusTransitionGate`, `PathScopeRouting` | ν | S11 (6332) | `task_interceptor.py` partitioned by the heuristic-14 measurement |
| `TaskStatus` members; task-status literal guard | ξ ο | S2 (escalation vocabulary), S6 (merge outcomes) | a new task-status spelling in a comparison fails the suite naming file and literal |
| update-task result guard (`orchestrator/tests/test_update_task_result_never_read.py`) | β2 | S9 (workflow edits) | a new `if not await …update_task(…)` fails the suite naming file and function |

## Pre-conditions and substrate (G3, verified at `e634cfe658`)

- `shared/src/shared/mcp_envelope.py::parse_tool_result` and `::write_failure_reason` exist (S14 ζ,
  task 6711 done); `write_failure_reason` takes the inner reply, never the envelope. C2 composes them.
- **Pending upstream, each a prerequisite of α (the chain head):** S14 ξ (task 6719, pending) adds
  `shared.mcp_envelope.McpErrorType`, `error_type: ClassVar[McpErrorType]` on the fused-memory
  exceptions and one `error_envelope(exc)` renderer, which 6719 is amended to home in
  `shared.mcp_envelope` (OQ-12 ruled; its plan named `server/tool_errors.py`, but R17's declared DAG puts
  `middleware` below `server` and `git grep "from fused_memory.server" -- fused-memory/src/fused_memory/middleware/`
  is empty at `e634cfe658`); S14 μ (task 6721, pending) replaces `shared.task_transitions.derive_actor_class`
  with `shared.agent_identity.parse_agent_identity(x).actor_class`. γ, κ1 and C1/C5 code against both.
  Each new exception class of this plan (`WriteModeError`, `FilesToModifyRetiredError`,
  `IllegalTransitionError`, `StatusCensusUnavailable`) adds its own `McpErrorType` member in the leaf that
  introduces it; 6719's values are today's wire strings, and the transition refusal has none today.
- Task 4867 (make a write-nothing `update_task` fail; make an undeclared MCP argument loud) is being
  hand-carried under Leo's ruling of 2026-11-10 (`deferred`, `held_by_session`, branch `task/4867`). It is
  a prerequisite of α as originally drafted; its argument guard is therefore live when α lands.
- `shared/src/shared/task_metadata_wire.py::coerce_task_metadata` and
  `fused_memory/middleware/metadata_dict.py::raw_metadata_dict` exist.
- `shared/src/shared/task_transitions.py::is_legal_transition(frm, to, actor, *, reopen)`, `ActorClass`,
  `TRANSITIONS` exist; `workflow_types.py` calls `is_legal_transition` (kept).
- `fused_memory/config/schema.py::TaskStatusConfig.enforce_transitions` (default `False`, deployed
  `true`, restart-only) is the one switch for edge enforcement; `server/main.py::_build_task_backend`
  builds the server's backend; 11 other non-test `SqliteTaskBackend(` constructions live in nine
  `fused-memory/scripts` and twice in `fused_memory/maintenance/backfill_curator_corpus.py`; tests construct
  one 131 times in 46 files. Every status write the server makes reaches the backend through
  `task_interceptor.py::_status_write_body`, the interceptor's one backend status write; the only other
  non-test callers of the backend's status writers are `audit_duplicate_tasks.py` and
  `correct_found_on_main_backlog.py`. Tests call the two writers directly 67 times in 8 fused-memory test
  files (`test_sqlite_task_backend.py` 30, `test_pending_since_anchor.py` 17,
  `test_status_authority_gate.py` 11, `test_recurrence_mint.py` 3, `test_task_interceptor.py` 2,
  `test_followup_r4.py` 2, `test_task_text_markup_chokepoint.py` 1, `test_candidate_key_boundary.py` 1).
- The interceptor's context gates that need the caller (identity from the agent-id header, the reopen
  reason it resolves, the recon write policy, the bulk-reset breaker, done provenance, the phantom-done
  file check, the pre-done hook, the consolidation closure check, the deferral gate) live in
  `TaskInterceptor._apply_status_transition`; the legality check itself needs only (from, to, actor,
  reopen?, enforce flag) (11-10 discussion, re-read).
- `sqlite3.IntegrityError.sqlite_errorname` is `'SQLITE_CONSTRAINT_UNIQUE'` for a partial-unique-index
  violation; the message names columns, not the index.
- `update_task` callers that build a dict payload rather than an `update_task(` call (scripts naming the
  `'update_task'` tool): all but `scripts/sitting/payloads.py::append_details_payload` (which sends
  `'append': True`) already send `metadata_mode`: `migrate_task_metadata_to_x_namespace`,
  `normalize_topic_slugs`, `strip_leaked_control_keys`, `exception_owner_sweep`,
  `migrate_metadata_modules_to_files`, `migrate_open_consolidation_gates`, `repair_wiped_metadata_files`,
  `restamp_degenerate_branch_bases` (new), `sweep_uuid_prefixes`, `scripts/sitting/payloads.py::update_task_payload`.
  LLM-facing instructions to call `update_task` live in `reconciliation/prompts/stage2.py` (system
  prompt), `reconciliation/stages/task_knowledge_sync.py::TaskKnowledgeSync.assemble_payload` (user
  prompt), `reconciliation/enumeration_guard.py` (its rejection text and rendered Stage-2 guidance) and
  the skills π lists.
- The orchestrator client always forwards an explicit `metadata_mode` and never `append`, so α can make
  the mode required at the backend before β1 lands.
- **Not a capability:** pyright does not flag a bool read of a `-> None` coroutine, so β2 ships an AST
  guard rather than relying on the type checker.
- Leaf-module registration precedent: `task_metadata.py` imports `CancelDispositionEntry`,
  `DeclaredScope`, `StrandConverted` and `DeferralRecord` downward and registers them. `deploy_state.py`,
  `defects.py` and `followup_items.py` import only `register_metadata_submodel`;
  `capability_manifest.py` imports it and `ExternalDep`, and imports `yaml`. `CHECK_SUBJECT_FIELD` is a
  diagnostic-renderer map, not slice validation; it stays in `capability_manifest.py`.
- `scripts/_checkout.py` binds every member `src/` on import and exposes `MEMBER_SRC` (S25 C1, task 6758
  done); θ's child binds through that import, not the private `_member_srcs`.
- `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES` enforces the lockstep edit over
  five registered sites (`CONTRIBUTING.md` as an absence, and the partial INV-to-heuristic mapping in
  `orchestrator/src/orchestrator/agents/code_quality.md`, pinned pair-wise, never for completeness);
  INV-14 is the last invariant at `e634cfe658`.
- `fused_memory/middleware/operational_routing_guard.py::inject_operational_routing` is pure (a fresh
  coerced dict on the coercion branch, the caller's object otherwise); it is the submit-time coercion λ
  applies to a submission before asking the gate rule. `TaskInterceptor.submit_task` and
  `server/tools.py`'s submit path both run it before the combine path sees the candidate, so a combine
  candidate is already in stored shape.
- All 11 stores, 2026-11-10: every row carrying a consolidation gate block (61) is deterministic with
  `always_escalates` true; 4 rows carry `operational_mode: 'gate'` without being human gates, all done
  coding tasks in dark-factory.
- `data/combine_audit.jsonl` is one file under dark-factory's `data/` written by the one fused-memory
  server for every project (`task_interceptor.py::_combine_audit_path`, `DARK_FACTORY_DATA_DIR` override).
- `orchestrator/tests/test_reopen_sticks_e2e.py::_Rig` builds a real `SqliteTaskBackend` + interceptor and
  bridges `Scheduler.dispatch_tool` straight to `TaskInterceptor.set_task_status` (`_bridge_dispatch_tool`
  accepts only `set_task_status`); ω swaps that bridge for the registered fused-memory `update_task` tool,
  built in process with `server/tools.py::create_mcp_server` as `fused-memory/tests/test_task_tools.py`
  does, so the reply is the wire-shaped result.
- `orchestrator/src/orchestrator/scheduler.py` carries priority inheritance (P1); it lifts a ready
  prerequisite but not one stalled for another reason, so no leaf's priority is hand-raised here.
- A deterministic task's `before_done.script` must exist on main when the task is submitted
  (`submit_task` validates it); a `delayed` milestone's timer starts the first tick all its dependencies
  are done (`docs/task-authoring.md` §6, frozen-once anchor). Hence σ is filed by the lead after ρ is on
  main, never in the same batch.

## Sketch of approach

**α — the wire (shared + fused-memory).** `shared.task_metadata_wire` gains C1.
`SqliteTaskBackend.update_task` takes `metadata_mode: MetadataMode | None` and `text_mode: TextMode =
'replace'` and no `append`; it calls `write_modes` once and applies `text_mode` to `details`/`prompt`
only. `_resolve_metadata_mode`, `_METADATA_MODES`, both carve-outs and the "append IS NOT SCOPED TO
METADATA" block go. `AppendUnsupportedFieldError` keeps its name, fires on `text_mode='append'` with
`title`/`description`/`priority`, and its message names `text_mode`. `TaskBackendProtocol.update_task`
and `TaskInterceptor.update_task` carry the two parameters. The MCP tool keeps `append` declared and
refuses any non-`None` value with `append_retired` and a `repaired_call` (D1a); the refusal text names
the parameter, never the spelling `append=True`. Fused-memory callers state their mode
(`targeted.py::_sweep_block_orphan` → `'additive'`; `TaskInterceptor._dispatch_ticket_decision`,
`manifest_stamping.py::_stamp_capability_manifests_impl` → `'merge'`), and so does
`scripts/sitting/payloads.py::append_details_payload` (`text_mode='append'`). Recon Stage 2's system
prompt (`prompts/stage2.py`), user prompt (`TaskKnowledgeSync.assemble_payload` items 4–5 and
`_needs_hint_attention`'s docstring) and `reconciliation/enumeration_guard.py`'s rejection text and
rendered guidance are rewritten together to `metadata_mode='additive'`;
`recurring_gate_guard.recurring_gate_error`'s hint likewise. Every fused-memory test that writes
metadata without a mode or passes `append` states the new axes.

**β1 — the client signature (orchestrator).** `Scheduler.update_task(task_id, metadata, *,
metadata_mode: MetadataMode) -> bool` — `append` gone, mode required; the client resolver and
`_reject_contradictory_metadata_mode` go; `SchedulerFacade`, `FakeScheduler`, `FakeMetadataBackend`
(which validates through `write_modes`), `scheduled_deploy_settle.TaskStore` and every hand-rolled
`async def update_task` fake declare it; every orchestrator and escalation call site states
`metadata_mode` (measured at `e634cfe658`: 15 mode-less → `'merge'` unless the site's intent says
otherwise; 2 `append=True` → `'additive'`; `workflow._write_branch_base_sha`'s `append=not overwrite` →
`'merge' if overwrite else 'additive'`); the five `reportCallIssue` suppressions go. Mechanical,
pyright-checked; behaviour unchanged.

**β2 — the raise.** `Scheduler.update_task -> None` raises `MetadataWriteFailed` (C2). The injected
writer types (`strand_conversion.TaskMetadataWriter`, `skew_tripwire.emit_pipeline_landing_tripwire`'s
and `stranded_verified_green.submit_verified_green_merge_request`'s `update_task` parameters, and
`scheduled_deploy_settle.TaskStore.update_task`) declare `-> None`. `orchestrator/tests/test_update_task_result_never_read.py` (OQ-13 ruled (a)) scans
`orchestrator/src` and `escalation/src` and fails, naming file and function, unless every `update_task(…)`
Call is the direct operand of an `Await` whose parent is an `ast.Expr` statement — so a result consumed
through `asyncio.gather`/`create_task` is caught too; its seeded fixture carries a bool read and a gather
case. Dispositions,
written here so decompose carries them (M10):

| Site | Today | β2 |
|---|---|---|
| `workflow._write_metadata_files` | `if not …: return None` | `except MetadataWriteFailed: return None` |
| `workflow._clear_merge_retry_pending`, `_mirror_plan_text_fp` | branch on `False` | `except MetadataWriteFailed:` same branch |
| `workflow._write_branch_base_sha` | `written = await …` inside `try/except Exception: return`, then `if not written:` warn "refused", `elif overwrite:` record | the `if not written` arm folds into an `except MetadataWriteFailed:` that logs "refused" and returns; the `except Exception` arm and the `overwrite` record stay |
| `workflow._clear_merge_phase_entered` | memory mutated before the write | memory assigned only after the write returns (task 6129's half) |
| `workflow._maybe_form_train` (no `try`) | per-member write in a loop | catch per member, log the half-formed train naming the members written |
| `harness._clear_merge_retry_pending_for_restart` | branch on `False` | `except MetadataWriteFailed:` same branch |
| `harness._check_reblock_guard` (no `try`) | persist, then allow the flip | catch → withhold the flip (return `False`), warn; the docstring's "persist before proceeding" holds |
| `harness._tag_prd_metadata`, `_tag_task_modules` (startup, no `try`) | logs and continues | catch, warn, continue — a failed tag never aborts boot |
| `deterministic_runner._advance_deploy_phase` (no `try`) | in-memory phase advanced even on `False` | propagate; the in-memory phase is assigned only after the write |
| `deterministic_runner._file_milestone_gate_and_block`, `_file_predicate_escalation_and_block` (no `try`) | stamp, then block | propagate; their comments' rule ("a stamp failure re-files on the next dispatch") holds |
| `deterministic_runner._rearm_predicate` (no `try`) | `if not …` → `_file_infra_issue_and_block` | `except MetadataWriteFailed:` → `_file_infra_issue_and_block` |
| `deterministic_runner._run_predicate` (no `try`) | warn, drive to done | `except MetadataWriteFailed:` → warn, drive to done |
| `deterministic_runner._writeback_deploy_success` (no `try`) | `stamped = bool(…)` in a retry loop | `except MetadataWriteFailed: stamped = False`; the loop retries |
| `scheduler._persist_files_metadata` (no `try`) | logs "non-critical", returns `False` | `except MetadataWriteFailed:` same log, return `False` |
| `scheduled_deploy_settle.ScheduledDeploySettler._write` (no `try`) | `if await …: return True`, else warn "retried next pass", return `False` | `except MetadataWriteFailed:` same warning, return `False`; the next pass retries |
| `strand_conversion._write_stamp` | `except Exception` and `if not accepted` | drop the bool branch; `except MetadataWriteFailed` keeps its warning |
| `escalation/server.py::merge_relane`'s `_write` (or its S3 ε1 home) | `written = … and await …` | `except MetadataWriteFailed: written = False` |
| the remaining 24 sites inside an existing `try` (`workflow.py` 10, `harness.py` 5, `dry_run_unblock.py` 3, `deterministic_runner.py` 2, `routing_dispatch.py`, `scheduler.py`, `skew_tripwire.py`, `stranded_verified_green.py` 1 each; R3's 13/14 counts `workflow.py`'s arms alone) | inside `except Exception` | unchanged; they now fire (S9 λ narrows them) |

The decompose seat re-runs the AST that produced this table and records the counts `measured at
<decompose sha>`; a new reader found then gets a row before filing. Test doubles: the `False` doubles
and bool-reader tests assert the raise; raising doubles already model a raise.

**γ — one parse per request, typed outcome, invariants on update (fused-memory).**
`TaskInterceptor.update_task` resolves the incoming metadata once, loudly, into a dict and passes that
dict to every gate and to the backend. `_validate_metadata_write` parses the merged blob once. Every
backend exception reaching `update_task` becomes a dict through one rule: `exc.to_error_dict()` when
the class defines one (the structured refusals), else S14's `error_envelope(exc)` (B4) — no second
converter; `DuplicateCandidateKeyError`, `LeakedEnvelopeMarkupError`, `ValidationError` and an
`IntegrityError` without a survivor (D9) all take the second path. The CSV branch's `all_ok` uses
`write_failure_reason` and `_consolidation_not_closed_error`'s docstring is rewritten to match. When
the delta touches `task_kind` or `before_done`, the merged result runs `deterministic_task_error` in
full; a delta touching only `always_escalates` or `milestone` runs its milestone and gate clauses but
not the script-existence clause, so a 6387 re-arm write never stats the script (task 3093's work). The
interceptor's `_parse_metadata_value`, `_warn_metadata_discard`, `_METADATA_DISCARD_CODES`,
`_parse_metadata`, `_fresh_metadata_copy` and the `json.loads` in `_submit_task_planning_mode` go;
`_extract_metadata_dict` (wherever task 6621 has moved it) becomes `metadata_dict`'s loud tri-state
reader (task 7085's work).

**π — prose callers of `update_task`.** The skills and the task-authoring reference that tell an agent
to call `update_task(…, append=True)` or a mode-less metadata write state the two axes
(`skills/escalation-watcher-auto/SKILL.md` now among them). `docs/task-authoring.md` also loses its three
mentions of `metadata.files_to_modify` as an accepted alternative (OQ-9 ruled): `files` is the one
declared-files key. π lands after γ and may land before δ's refusal; for that window the reference is
stricter than the server, which costs a caller nothing. π heads the `docs/task-authoring.md` chain
(task 6392, its former prerequisite, is done).

**δ — one declared-files key and reader (shared + fused-memory; OQ-9 ruled).** `shared.declared_files`
(C4): `declared_files(md)` reads `metadata.files` only. The backend computes `candidate_key` over it;
`_files_for_key` goes; the interceptor's readers, `_DELIVERABLE_SIGNAL_KEYS` and the inline
`('files', 'files_to_modify')` loop go; `task_curator._task_files` keeps its top-level wire-shape arm
(the curator's own field) and reads metadata through `declared_files`, which gives it the malformed-value
guard it lacks (task 6653's work, cancelled into δ); `path_scope_guard.py`'s union rationale collapses
(both of its inputs are now the same `files` list) and its docstrings say so; `lock_charter_guard.extract_files`,
`briefing_queries` and `reconciliation/targeted.py` read through it. The interceptor calls
`refuse_retired_files_key(incoming)` on the incoming metadata of `submit_task` (both arms) and
`update_task`, at γ's one parse — incoming keys only, never the merged blob, so the four stored legacy
rows stay writable. The curator's internal field keeps its name (LLM schema, `CandidateTask` /
`RewrittenTask`, vector-store payload, `search_tasks` output); its five reads move onto a named constant,
`curator_types.CURATOR_FILES_FIELD`, so the literal `'files_to_modify'` leaves every read position. A guard
test (`shared/tests/test_retired_files_key_unread.py`, seeded fixture) scans `shared/src`,
`fused-memory/src`, `orchestrator/src`, `escalation/src` and `dashboard/src` and fails, naming file and
line, on the literal `'files_to_modify'` as a `.get(…)` argument, a subscript key or an element of a
tuple/list/set literal; no exemption entries. `lock_charter_guard` imports `FILE_EXTENSIONS`,
`is_file_path`, `directory_locks` from `shared.locking`; its copies, the false rationale and the tier-1
drift test go; the tier-2 guard against reify's script stays in `shared/tests/test_locking.py`. A
boundary test computes the store key and the curator key for the same metadata and asserts equality.
After δ deploys (a fused-memory restart; one server writes every store), the lead re-measures all stores
and backfills each row whose `files_to_modify` adds paths (`lead_actions`; 3 rows on 2026-11-10).

**ε — orchestrator reads through `shared.declared_files`.** `module_charter.sanitize_files_for_persist`
is deleted and its callers call `shared.locking.strip_directory_locks` (D5); every read of a task's
declared files from metadata in `workflow.py`, `harness.py`, `scheduler.py` reads `declared_files(md)`.
Plan JSON `files` reads are untouched.

**ζ — the combine writes a union it computes.** The combine's write half (`_write_combined_rewrite`,
which task 8453 extracts from `_execute_combine` and which pack survivors share; ζ waits on 8453) writes
`{'files': strip_directory_locks(dedupe([*declared_files(target_md), *declared_files(candidate_md),
*rt.files_to_modify])), 'curator_action': …, 'curator_justification': …, 'combined_at': …}` under
`'merge'` (order-preserving, de-duplicated), derives the `declared_scope` re-stamp from the same list so
the stamp's file count equals the files it describes, sends `rt.details` unprefixed, re-embeds with the
same list, and deletes the false comment. A `DuplicateCandidateKeyError` from the write is an audited
refusal with `refused_reason='candidate_key_collision'` (D6, OQ-14 ruled). The combine audit's
vocabulary moves out of `task_interceptor.py` into `fused-memory/src/fused_memory/middleware/combine_audit.py`:
`CombineRefusal` (the four existing tokens plus `CANDIDATE_KEY_COLLISION`, values byte-identical),
`combine_audit_path()` and `append_combine_audit`, so ρ reads the token and the path through a public
seam. The record gains `old.files`, `new.files` and, on a collision, `survivor_id`; the historical keys
(`old.{title, description_truncated, status}`, `new.{title, description_truncated}`,
`justification_truncated`, `refused_reason`, and 8453's `pack_proposal_id`/`members` if it has landed)
stay byte-identical, so the live signal has fields to read. The bare `except Exception` around the write
narrows to the backend's error family. Legacy prefixed rows are left to age out (OQ-15 ruled (a)); ζ's
commit message records the counts.

**ρ — the collision census script (OQ-14 ruled).** `scripts/combine_collision_census.py --days N` reads
`combine_audit_path()`, counts the rows of the last N days whose `refused_reason` is
`CombineRefusal.CANDIDATE_KEY_COLLISION`, by `project_id`, against every combine row (refusals included)
in the same window, exits 0 below the threshold and 1 at or above it (frequent = at least 5 collisions,
or at least 1% of the window's combine rows), and always prints a trailing JSON verdict (window, counts
per project, total combine rows, survivor ids, up to three example rows). Its test drives fixture audit
files across both thresholds and an empty window.

**σ — the 30-day check (OQ-14 ruled; filed by the lead once ρ is on main).** A
`task_kind='deterministic'` task depending on ζ and ρ, with `before_done = {script:
'scripts/combine_collision_census.py', args: ['--days', '30'], kind: 'predicate', timeout_secs: 120}`,
`milestone = {mode: 'delayed', after_secs: 2592000}` and `always_escalates = false`. Its timer starts the
first tick both dependencies are done — in practice when the lead files it, shortly after ζ lands — so
the 30-day window it reads holds only post-ζ combines. Exit 0 closes it done with the counts in
`done_provenance`; exit 1 files a born-at-L2 `milestone_check_failed` escalation to Leo with the counts,
and Leo or the L2 watcher files the "more elegant" fix (a deterministic task cannot file it). Measured
for scale: 1,361 combine rewrites passed every guard from 2026-04-20 to 2026-11-10; recomputing the key
for the 105 that did not land found no collision with a live row; about 600 combine audit rows are
expected in 30 days.

**η — the slice table.** `shared.task_metadata` builds `SLICES` at import from models it imports
downward (C3) and exposes `registered_slices()`; `parse_metadata` takes `slices=`;
`register_metadata_submodel`, `_SUBMODEL_REGISTRY`, `_SUBMODEL_CARDINALITY` go. `ExternalDep` moves to
`shared/external_dep.py`; the delivered-check slice model, its check-field base, its conditional-field
validator and the mechanical-kind vocabulary move to `shared/delivered_check.py` (OQ-11 ruled), which
`capability_manifest.py` imports; `deploy_state.py`, `capability_manifest.py`, `defects.py` and
`followup_items.py` stop importing `task_metadata` (`followup_items.py`'s comment pointing at
`deploy_state.py`'s "why" goes with its registration). Fourteen slices at `e634cfe658`; any that tasks
5281 or 8453 (`pack_record`) register first become `SLICES` rows here, and a slice registered after η
is a row. Importers of the moved names, tests included, import them from their new homes. The
registration-chain tests are replaced by one table test in `shared/tests`; the autouse
registry-snapshot fixture and the `_stub` convention go.

**θ — the invariant.** The lockstep edit for `metadata-slices-one-table` (INV-16 under the lead's
revised chain) plus its fixture.

**ι — the blessed-key table.** `docs/task-authoring.md` §8 Tier-A becomes a table — key · writer ·
reader · ground — between `tier-a-blessed-keys-table` markers; `_BLESSED_METADATA_KEYS` keeps the keys
with at most one pointer line each; census figures, rejected-alternative essays and inventories leave
the code. The drift test and the two other marker readers follow the new markers.

**κ1 — the transition verdict, enforced in the backend only (OQ-6 ruled).** `shared.task_transitions`
gains `transition_verdict` (C5); `is_legal_transition` becomes `transition_verdict(…) is LEGAL` (its
orchestrator caller, `workflow_types.py`, is unchanged). One policy (D7): a terminal exit without a
reopen reason is refused unconditionally; an illegal edge is refused when `enforce_transitions` is on and
otherwise logs the existing `illegal_transition would-reject` WARNING, now from the backend; an
out-of-vocabulary status is legal at this check (vocabulary is the backend's separate floor), exactly as
gate 2e treats it today. `SqliteTaskBackend.set_task_status` and `set_status_and_stamp_audit` (and
`TaskBackendProtocol.set_task_status`) take `actor: ActorClass` — keyword-only, no default — and
`reopen_reason: str | None = None`; `_write_status_and_verify`, their shared tail, reads `frm` inside the
same `_write_unit` transaction as the UPDATE and applies the verdict there: the one call site. A refusal
raises `IllegalTransitionError` (its `to_error_dict()` is today's `terminal_exit_rejected` /
`illegal_transition` shape; κ1 adds its `McpErrorType` member). `SqliteTaskBackend.__init__` takes
`task_status: TaskStatusConfig = TaskStatusConfig()` — the schema's own Field default, no second home;
`_build_task_backend` and the 11 script/maintenance constructions pass their loaded `config.task_status`
explicitly, and a κ1 test asserts `_build_task_backend` passes the deployed config. A test that omits it
gets log-only edges; terminal exits are refused regardless (D7), so the 131 test constructions in 46
files need no edit for the config, while the 67 direct status-writer calls in 8 test files state
`actor=`.
The interceptor's gate 2a refusal (`_terminal_exit_error`) and gate 2e (`is_legal_transition`,
`_illegal_transition_error`, `_enforce_transitions`) are deleted. It still resolves and strips the reopen
reason (the bulk-reset breaker and the audit fields need it), derives the actor with
`parse_agent_identity(agent_id).actor_class`, passes both to `_status_write_body`, and renders an
`IllegalTransitionError` through γ's one rule (`exc.to_error_dict()`), so callers see the same tokens.
The deferral gate's comment that it "runs last, so a refusal's self-correction hint never sends the
caller into a gate above that refuses the retry" is rewritten: a corrected retry can now meet the
backend's transition refusal. The two direct-backend scripts pass `actor=ActorClass.HUMAN`.

**κ2 — the backend's typed surface.** `PrivilegedAuditWriter` and the `TaskStore` intersection type the
interceptor's dependency; the three `type: ignore[attr-defined]` go; the interceptor fixture uses an
autospec of `SqliteTaskBackend`. `get_statuses_fresh` raises `StatusCensusUnavailable` (D8); its
consumers handle it. `CANDIDATE_KEY_SCOPE_SQL` is the scope of the live survivor SELECTs and
`_raise_candidate_key_collision`; migrations keep their literal DDL; a check compares the live index's
`sqlite_master.sql` `WHERE` clause with the constant (D9a). `sqlite_errorname` plus the survivor lookup
replaces the three `str(exc)` sniffs.

**λ — one gate rule (OQ-10 ruled).** `shared.gate_declaration` (C6) owns
`CONSOLIDATION_GATE_METADATA_KEY` (`consolidation_gate.GATE_METADATA_KEY` re-binds it by import) and the
one rule dispatch acts on: `is_human_gate` — deterministic and `always_escalates is True` — read on the
stored shape. Its refinements are named properties: `is_pure_human_gate` (no `before_done`; an
act-then-ask gate is a human gate that is not pure) and `is_curator_gate` (also `operational_mode ==
'gate'`). Two consumers ask a different question and keep honestly named selectors on the same value:
the blocked-gate audit (`awaits_human_after_deterministic_step`: deterministic with `operational_mode`
absent or `'gate'`, or a `gate_escalated_at` stamp — so a deterministic task blocked by an
infrastructure failure stays in the report) and the curator-gate resolution sweep
(`carries_curator_resolution_mode`: `operational_mode == 'gate'`, the population whose Mem0 resolution
key exists). The consumers move:
- combine guard (`TaskInterceptor._is_gate_metadata`, deleted) → `is_human_gate` on candidate and target;
  an act-then-ask candidate is still refused into an ungated target;
- recon duplicate-gate guard → a submission is classified after `inject_operational_routing` (the
  submit-time coercion), a stored carrier on its stored shape, both by `is_human_gate`;
  `recurring_gate_guard.is_gate_submission` keeps its name and becomes that one line;
- Stage-2 finding phrasing selector → `is_human_gate`;
- live-workflow detector → `is_pure_human_gate`; `live_workflow_detector.py::is_pure_gate_metadata` is
  deleted and its three other readers (`recon_write_policy.py`, `reconciliation/harness.py`,
  `live_workflow_section.py`) read the property;
- the interceptor's consolidation closure gate and `consolidation_gate.handrolled_member_enumeration` /
  `::declared_gate_topic` → `is_curator_gate` plus the gate block (all 61 block-carrying rows in the 11
  stores are human gates, so no row moves);
- `dashboard/src/dashboard/data/prd_view.py::is_human_gate` (added 2026-11-05; already this exact rule, a
  second copy) is deleted and its two readers read the shared property.
Measured classification change (all 11 stores, 2026-11-10): 14 rows the combine guard called gates
dispatch to an agent (1 live: reify 5376, deferred); the recon duplicate-gate guard starts seeing about
98 live open gates; 0 live change for the phrasing selector and the detector. The data needs no
migration. λ's live signal: for the week after it deploys, the count of `RecurringGateViolation`
refusals to recon-stage callers (the guard's `rejecting duplicate human gate` WARNING names the open
carrier), against the week before.

**μ1 — one interceptor test factory.** `fused-memory/tests/_interceptor_helpers.py::build_interceptor`
constructs a `TaskInterceptor` with explicit collaborators; the 49 test files that construct one move
onto it, the two shared stacks (`_hermetic_task_stack.py`, `_submit_stack.py`) included. No production
change.

**μ2 — collaborators at construction.** `TaskInterceptor.__init__` takes a frozen
`InterceptorCollaborators` bundle: journal, audit-trail archive and consolidation scroll are required
members; the lifecycle-reset filer is the one provider-callable member, resolved after
`ReconReportState` exists; the reconciliation-disabled arm and `server/tools.py`'s bare fallback pass a
named null filer / null bundle. The four setters and `_idempotency_replay`'s no-journal arm go.
`targeted.task_interceptor` stays a back-wire owned by `TargetedReconciler` (out of scope). μ2 is
ordered after S26 γ and builds its bundle on S26's constructed `MemoryService` (OQ-3 ruled).

**ν — the partition.** `middleware/status_transition.py` (`StatusTransitionGate`, which after κ1
holds only the context gates; Table A is the backend's) and `middleware/path_scope_routing.py` are
extracted as composed objects; the interceptor delegates. Each new file imports nothing back from
`task_interceptor.py` (heuristics 13 and 14).

**ξ — orchestrator `TaskStatus` members and the guard.** Comparisons of a task's status become
`TaskStatus` members; `shared/tests/test_task_status_literals.py` lands here with its seeded fixture,
scanning the roots in a module-level `SCAN_ROOTS = ('orchestrator/src',)` kept on one line (D13), so ο's check can anchor on the tuple.

**ο — fused-memory and dashboard members.** The same conversion; ο appends `fused-memory/src` and
`dashboard/src` to `SCAN_ROOTS` and cuts `shared/task_statuses.py`'s docstring to what the module is.

**ω — integration gate.** `orchestrator/tests/test_task_write_contract_e2e.py` replays boundary rows 1,
2, 5, 7, 12, 14, 19, 21 and 27 through the real seams. Its bridge (`orchestrator/tests/_fused_memory_tool_bridge.py`)
replaces the `test_reopen_sticks_e2e.py` rig's `dispatch_tool` → interceptor shortcut with the registered
fused-memory tools: it builds the server with `server/tools.py::create_mcp_server` over the rig's
interceptor and backend and returns the tool's wire-shaped result, so `Scheduler.update_task`'s
`parse_tool_result`/`write_failure_reason` path, D1a's `append_retired` and `metadata_mode_required` (which
live in `server/tools.py::update_task`) are all exercised.

## Resolved design decisions

- **D1 — two axes, one resolver, no `append` on the wire (R3).** `metadata_mode` required whenever
  metadata is sent and refused when it is not; `text_mode` defaults to `'replace'` and `'append'` is
  refused when neither `details` nor `prompt` is sent. The symbols live (OQ-1 ruled) in
  `shared/task_metadata_wire.py`.
- **D1a — the MCP tool keeps `append` declared and refuses it with `append_retired`.** Task 4867 lands
  before α (hand-carried, ruled 2026-11-10), so an undeclared name is already loud when α lands; the
  declaration is kept for the targeted `repaired_call` (`metadata_mode='additive'` / `text_mode='append'`),
  which a generic unknown-argument refusal cannot give while skills prose (π) and agents' habits still
  spell `append`.
- **D2 — `Scheduler.update_task -> None` (R3 option 1).** A `-> bool` that can only be `True` would be a
  name that lies (heuristic 1). Because pyright does not flag a bool read of `None`, β2's AST guard is
  the enforcement (OQ-13 ruled (a)).
- **D3 — the slice registry is a frozen table built from downward imports (R3, R22 item 4).** Tests
  inject a table through `parse_metadata(…, slices=)` (Tests stance). Task 6558's lower registry keeps
  side-effect registration, is contrary to R3, and its cycle premise is gone.
- **D4 — `files` is the one declared-files key; `files_to_modify` is refused and read nowhere** (OQ-9
  ruled by Leo; the precedence question is withdrawn). `declared_files(md)` reads `metadata.files`: a
  `str` is one path, non-`str` entries and a `dict` value are dropped, absent → `()`. The refusal checks
  the caller's incoming metadata only (submit and update), never the stored or merged blob. All 11 stores
  on 2026-11-10: 4 rows carry the key, all `done`; 0 non-terminal rows, so no live candidate key or lock
  changes; δ re-measures before landing with the `_migrate_v3_to_v4` audit query. Ordering: refusal
  first (δ deploys with a fused-memory restart; one server writes every store, and no orchestrator writes
  the key), backfill second (lead action: `update_task(metadata={'files': union}, metadata_mode='merge')`
  for each row whose `files_to_modify` adds paths — dark-factory 2507, reify 4344 and 4865 today). The dead
  key stays on the 4 done rows (merge cannot delete a key; task 5310's delete-keys primitive can) and is
  inert. The curator's internal field of the same name is a different thing and keeps its name, read
  through `curator_types.CURATOR_FILES_FIELD`; renaming it is an optional low follow-up outside this plan.
  The details-text prefix is not merged into `files` (that would widen 222 live tasks' pre-plan locks
  across four projects); ζ stops writing it and old rows age out (OQ-15).
- **D5 — the sanctioned files sanitizer is `shared.locking.strip_directory_locks`.** C4 adds no alias;
  `module_charter.sanitize_files_for_persist` goes and its callers call the shared function.
- **D6 — the combine's union, and a collision is a counted refusal** (OQ-14, OQ-8 ruled). ζ computes
  the union in code (target ∪ candidate ∪ curator list), so the combine can only grow `files`. A
  `DuplicateCandidateKeyError` from that write leaves the target unchanged, appends a combine-audit row
  with `refused_reason='candidate_key_collision'`, `survivor_id`, `old.files` and `new.files`, and the
  ticket takes today's degrade-to-create path; a combine writes only its chosen target. Task 3260 is
  amended so its owned-field design counts the combine as a sanctioned `files` writer through
  `strip_directory_locks` (OQ-8 ruled; a lead action). Whether collisions are frequent enough to want a
  better answer is measured, not argued: D16.
- **D7 — one transition verdict, enforced in the backend only** (OQ-6 ruled by Leo; amends R3's "as well
  as the interceptor"). Policy: a terminal exit without a reopen reason is refused unconditionally
  (folding it into the flag-gated edge check would stop refusing terminal exits wherever
  `enforce_transitions` is `False`, the schema default); an illegal edge is refused iff
  `enforce_transitions`, else the `illegal_transition would-reject` WARNING is logged from the backend;
  out-of-vocabulary is legal at the verdict. One call site: `SqliteTaskBackend._write_status_and_verify`,
  reading `frm` in the UPDATE's `_write_unit` transaction, with `actor` required and the reopen reason
  passed. The interceptor holds no legality gate; it keeps only the gates that need the caller's context
  and renders the backend's refusal with the tokens `terminal_exit_rejected` and `illegal_transition`
  unchanged. Every writer — the MCP tool, the two maintenance scripts, an operator's shell snippet — gets
  the same refusal, with no ban list to maintain. Consequences accepted with the ruling: the refusal now
  comes after the context gates, so a refused write may already have run the pre-done hook and the
  phantom-done check (wasted work; nothing is written, because status and audit fields are one atomic
  write); the deferral gate's hint no longer runs after the legality check; the `actor` keyword has no
  default, so a caller passing the wrong actor gets the wrong table but none gets one silently.
  `enforce_transitions` is restart-only; the backend's `task_status` keyword defaults to
  `TaskStatusConfig()` and every production constructor passes the loaded config. Measured (11-10
  discussion): every routine status write already passes the interceptor, and no illegal transition has
  been found to reach a store by a bypass; the floor protects against future direct writers.
- **D8 — `get_statuses_fresh` raises.** `StatusCensusUnavailable(project_root, db_path, cause)` with
  `cause ∈ {'store_absent', 'read_failed'}`. A not-yet-created store for a newly registered project is
  `store_absent`: the recon harness records one WARNING per cycle for that project and skips its census
  — never an empty census.
- **D9 — collision classification by `sqlite_errorname`.** `SQLITE_CONSTRAINT_UNIQUE` plus a survivor
  lookup under `CANDIDATE_KEY_SCOPE_SQL`; no survivor → the backend lets the `IntegrityError` propagate
  unchanged, and the interceptor renders it like any other backend exception (γ); nothing raises past
  `TaskInterceptor.update_task`.
- **D9a — migrations are frozen history.** `_migrate_v3_to_v4` and the other migrations keep their
  literal DDL and audit predicates; the constant is the live scope; a check fails when the live index's
  `WHERE` clause differs from the constant, so narrowing uniqueness needs a new migration.
- **D10 — one gate rule: dispatch truth** (OQ-10 ruled by Leo). A task is a human gate when it is
  deterministic and `always_escalates is True` — the only definition the orchestrator acts on — read on
  the stored shape, and for a submission after the submit-time coercion. Refinements are named properties
  of it (pure; curator gate). The blocked-gate audit and the curator sweep ask other questions and keep
  honestly named selectors on the same `GateDeclaration`, never variants of "is a gate". The data needs
  no migration: `always_escalates` is always a real bool and `operational_mode` is only ever `'gate'` in
  all 11 stores. The one behaviour change worth watching is the recon duplicate-gate guard, measured for
  a week (λ's signal).
- **D11 — collaborators at construction** (OQ-3 ruled). Journal, archive and scroll are available before
  construction (measured); the filer is the one provider member; never a setter.
- **D12 — the partition follows the measurement; the ticket worker is 6332's.**
- **D13 — task-status literal scope and guard** (OQ-4 ruled). Conversion scope (ξ, ο): comparisons whose
  operand holds a task's status — `X.get('status')`, `X['status']`, or a local bound from one of those or
  from a status map (e.g. harness's `current`) — in `orchestrator/src`, `fused-memory/src`,
  `dashboard/src`. Out of scope: a typed record's `.status` attribute (S2, S6, eval and intake results),
  SQL, prompt text, comments, `escalation/src`. The guard is syntactic: a `Compare` against a
  `TaskStatus` value whose other operand is `X.get('status')`, `X['status']`, or a `Name` spelled
  `status` or ending `_status`; attribute operands are never flagged. No exemption entries (INV-12): a
  flagged local holding another vocabulary is renamed to say which (heuristic 1).
- **D14 — prose stance.** Rewritten docstrings follow the Comments stance; Greek labels never appear in
  code.
- **D15 — sequencing with live tasks** (OQ-7 superseded by Leo, 2026-11-10). A live task that goes
  first is a prerequisite of the chain head: α for `task_interceptor.py`/backend/`tools.py` — 4867
  (hand-carried now; as originally drafted, never inverted), 6621, 5281, 6719, 6720, 6721, 8143 (in
  progress, cancel-with-children on `_apply_status_transition`) and S5 μ, each one still open at
  decompose; β1 for the orchestrator files (6766, 6724). 6708 and 6392 are done and drop out. One live
  task with overlapping work is a mid-chain prerequisite instead: ζ waits on 8453, which extracts the
  combine's write half that ζ edits (pending high; its own prerequisites 8418, 8440, 8452 pending and
  8143 in progress). A live task that goes after waits on the tail of every chain sharing a file: 5310 on
  ο and ξ; 7173 on κ2 and ι; 6678 on ο; 3260 on ο and ξ; 7748 on δ; 6332 on ζ, and μ1 on 6332. 6332 is
dispatchable today (its one prerequisite, 6708, is done and its plan was authored 2026-11-07), so the
decompose seat reads its status first: if it is in progress or done, the 6332 → ζ edge is not added, ζ and
μ1 depend on 6332 instead, and α's and ζ's edits to `_dispatch_ticket_decision` and the combine call path
follow them to 6332's `ticket_dispatch.py` / `ticket_worker.py` (declared in those leaves' files at
decompose). Task 6653
  (one files extractor for the curator and the interceptor) is δ's work and is cancelled into it. The
  other live tasks declaring this plan's hot files (2026-11-10: 72 on `task_interceptor.py`, 18 on
  `sqlite_task_backend.py`, 29 on `shared/src/shared/task_metadata.py`) are ordered at decompose by the
  same rule only where the work overlaps; overlaps found: 8264 and 8148 (cancel-with-children on
  `_apply_status_transition`, against κ1/ν), 8589 (the schema ladder leaves `sqlite_task_backend.py`;
  κ2's D9a check reads the live index, not the migration text, so either order holds) and 8453's
  `pack_record` registration (a `SLICES` row if η lands first). Cross-stream: κ1 and μ2 wait on S26 γ; S26
  λ, μ and ρ (`server/tools.py` — ρ through its `metadata_row_text` import repoint — and μ's
  `reconciliation/stages/task_knowledge_sync.py`) and S26 γ0, θ2 and ν
  (four scripts: `invalidate_fabricated_shipping_edges`, `memory_eval_staleness_sweep` and
  `sweep_orphan_flag_markers`, which κ1 edits, and `normalize_topic_slugs`, which λ edits) are ordered
  against S10's `tools.py`, α, κ1 and λ chains at decompose; ε/ξ and S9's `workflow.py` chain likewise; ι and S9 ξ, and
  η/ι and S3 ε1, likewise — in each pair the chain filed second puts its head after the other's tail, or
  the lead wires the edge. The invariant chain is S6 α → θ → S7 ζ (lead, revised 2026-11-10).
- **D16 — the combine-collision check** (OQ-14 ruled by Leo). The distinct token (ζ) makes collisions
  countable; ρ counts them across every project from the one shared audit file; σ, a deterministic
  delayed predicate filed by the lead after ρ is on main (a `before_done.script` must exist at submit),
  runs ρ over the 30 days after ζ and escalates to Leo when collisions reach 5 or 1% of combine rows. The
  historical estimate says collisions are rare (no collision among the 105 non-landed rewrites since
  2026-04-20), and the union makes keys more specific, not less.

## Contracts

### C1 — write modes (`shared/src/shared/task_metadata_wire.py`, α)

```python
MetadataMode = Literal['merge', 'additive', 'replace']
TextMode = Literal['replace', 'append']
METADATA_MODES: frozenset[str]; TEXT_MODES: frozenset[str]   # derived from the Literals

class WriteModeError(ValueError):
    error_type: ClassVar[McpErrorType]      # S14 ξ vocabulary; value 'WriteModeError'
    code: Literal['metadata_mode_required', 'invalid_metadata_mode', 'metadata_mode_without_metadata',
                  'invalid_text_mode', 'text_mode_without_text', 'append_retired']
    field: str; value: object; allowed: tuple[str, ...]
    def to_error_dict(self) -> dict   # error_envelope(self)'s keys plus 'success': False, 'error': code,
                                      # 'field', 'value', 'allowed', 'hint', 'repaired_call'?

def write_modes(*, metadata_present: bool, metadata_mode: object,
                text_present: bool, text_mode: object = 'replace') -> tuple[MetadataMode | None, TextMode]
```

Invariants: `metadata_present ⇒ metadata_mode ∈ METADATA_MODES`; `¬metadata_present ⇒ metadata_mode is
None`; `text_mode == 'append' ⇒ text_present`; neither axis influences the other. Production caller:
`SqliteTaskBackend.update_task`; test caller: `FakeMetadataBackend`. `to_error_dict()['error_type']`
equals the declared `McpErrorType` value — the code `error_envelope` would render.

### C2 — the client's failure (`orchestrator/src/orchestrator/task_write.py`, β2; OQ-2 ruled)

```python
class MetadataWriteFailed(Exception):
    task_id: str
    reason: str                            # EnvelopeParseError / write_failure_reason text, or the transport error
    error_type: str | None                 # the reply's error_type when structured
    rejection: Mapping[str, Any] | None    # extract_structured_rejection(result), verbatim
    transport: BaseException | None        # set when the MCP call itself raised

class Scheduler:
    async def update_task(self, task_id: str, metadata: Mapping[str, Any] | str, *,
                          metadata_mode: MetadataMode) -> None: ...   # raises MetadataWriteFailed
# SchedulerFacade, FakeScheduler, FakeMetadataBackend: identical. (β1 lands the parameters with -> bool.)
```

Invariants: no `return False` path; `None` means the backend acknowledged the write. The client takes
`reply, err = parse_tool_result(result, None, dict)` and raises when `err is not None`, when
`write_failure_reason(reply) is not None`, or when the transport raises. No code reads the result (β2's
guard). Consumers: S9 λ1 (after β2) maps a propagated `MetadataWriteFailed` to a `BlockReason` row; S9 μ1
declares `DryRunScheduler` against this signature — its own edge is on β1, and it lands after β2 through
S9 λ2 → λ1 → β2.

### C3 — the slice table (`shared/src/shared/task_metadata.py`, η)

```python
@dataclass(frozen=True)
class SliceSpec:
    key: str; model: type[BaseModel]; cardinality: Literal['dict', 'list']
SLICES: Mapping[str, SliceSpec]          # MappingProxyType; built at import from downward imports only
def registered_slices() -> Mapping[str, SliceSpec]: ...
def parse_metadata(blob, *, direction, enforce=False,
                   slices: Mapping[str, SliceSpec] = SLICES) -> tuple[TaskMetadata, list[SchemaWarning]]: ...
# deleted: register_metadata_submodel, _SUBMODEL_REGISTRY, _SUBMODEL_CARDINALITY
# leaf modules (import nothing from shared.task_metadata): shared/external_dep.py, shared/delivered_check.py,
# shared/deploy_state.py, shared/defects.py, shared/task_deferral.py, shared/strand_conversion.py
```

Invariant (`metadata-slices-one-table`): `registered_slices()` is identical in every fresh interpreter.
Error semantics unchanged.

### C4 — declared files (`shared/src/shared/declared_files.py`, δ)

```python
RETIRED_FILES_KEY: Final = 'files_to_modify'     # the retired metadata key; its only use is the refusal
def declared_files(metadata: Mapping[str, Any] | None) -> tuple[str, ...]: ...
    # metadata['files'] only (D4): a str is one path; non-str entries and a dict value are dropped; None/absent → ()

class FilesToModifyRetiredError(ValueError):
    error_type: ClassVar[McpErrorType]          # member added by δ; value 'FilesToModifyRetiredError'
    code: Literal['files_to_modify_retired']
    def to_error_dict(self) -> dict             # error_envelope(self)'s keys plus 'success': False, 'error': code,
                                                # 'field': 'files_to_modify', 'hint' naming metadata.files,
                                                # 'repaired_call' with the paths moved into 'files'
def refuse_retired_files_key(incoming: Mapping[str, Any] | None) -> None: ...
    # raises FilesToModifyRetiredError when the CALLER's metadata carries the key; never reads a stored blob
```

Invariants: `compute_candidate_key(title, declared_files(md))` is both the store key and the curator
key; `shared.locking.strip_directory_locks` is the only function that sanitizes a value written to
`metadata.files`; no module under `*/src` reads a `files_to_modify` key from a mapping (δ's guard test);
the interceptor calls `refuse_retired_files_key` on `submit_task`'s and `update_task`'s incoming
metadata. Pure; `None` → `()`. The curator's own field keeps its name, spelled
`curator_types.CURATOR_FILES_FIELD` where it is read.

### C5 — transition verdict and the backend's typed surface (κ1, κ2)

```python
# shared/src/shared/task_transitions.py (κ1)
class TransitionVerdict(StrEnum): LEGAL; TERMINAL_EXIT_WITHOUT_REOPEN; ILLEGAL_EDGE
def transition_verdict(frm, to, actor, *, reopen: bool) -> TransitionVerdict
    # an out-of-vocabulary frm/to → LEGAL here (vocabulary is the backend's separate floor)
def is_legal_transition(frm, to, actor, *, reopen=False) -> bool   # == (transition_verdict(...) is LEGAL)

# fused-memory backends — the ONE enforcement point (κ1; OQ-6 ruled)
class IllegalTransitionError(TaskmasterError):     # κ1; error_type: ClassVar[McpErrorType] (member added by κ1)
    task_id; from_status; to_status; actor; verdict
    def to_error_dict(self) -> dict    # error='terminal_exit_rejected' | 'illegal_transition', today's interceptor shapes
SqliteTaskBackend.__init__(config, *, task_status: TaskStatusConfig = TaskStatusConfig(), ...)
SqliteTaskBackend.set_task_status(..., *, actor: ActorClass, reopen_reason: str | None = None, ...)          # actor: no default
SqliteTaskBackend.set_status_and_stamp_audit(..., *, actor: ActorClass, reopen_reason: str | None = None, ...)
TaskBackendProtocol.set_task_status(..., *, actor: ActorClass, reopen_reason: str | None = None, ...)
# deleted from the interceptor: _terminal_exit_error, _illegal_transition_error, _enforce_transitions, its is_legal_transition call

class PrivilegedAuditWriter(Protocol): ...        # κ2; the three writers as implemented at main
class TaskStore(TaskBackendProtocol, PrivilegedAuditWriter, Protocol): ...
class StatusCensusUnavailable(TaskmasterError):   # κ2; error_type: ClassVar[McpErrorType]
    project_root; db_path; cause: Literal['store_absent', 'read_failed']
SqliteTaskBackend.get_statuses_fresh(project_root, *, ids=None, tag=None) -> dict[str, str]   # raises StatusCensusUnavailable
CANDIDATE_KEY_SCOPE_SQL: Final[str] = "status != 'cancelled'"   # the live survivor scope
async def _raise_candidate_key_collision(conn, *, tag: str, candidate_key: str, exclude_id: int | None = None) -> None
```

Invariants: one verdict, one call site — `SqliteTaskBackend._write_status_and_verify`, the shared tail of
both status writers — under one policy (terminal exit refused unconditionally; illegal edge refused iff
`enforce_transitions`, else the one `illegal_transition would-reject` WARNING); the backend reads `frm`
inside the UPDATE's `_write_unit` transaction; no module under `fused-memory/src/fused_memory/middleware/`
calls `is_legal_transition` or `transition_verdict`. The live survivor SELECTs and the
collision helper use the constant; migrations keep literal DDL; a check asserts the live
`ux_tasks_candidate_key` `WHERE` clause from `sqlite_master.sql` equals `candidate_key IS NOT NULL AND
` + the constant. A `DuplicateCandidateKeyError` is raised only after a survivor was read; any other
`IntegrityError` propagates. In the interceptor the actor comes from
`parse_agent_identity(agent_id).actor_class` (S14 μ); the two maintenance scripts pass
`ActorClass.HUMAN`.

### C6 — gate declaration (`shared/src/shared/gate_declaration.py`, λ)

```python
CONSOLIDATION_GATE_METADATA_KEY: Final = 'x_recon_consolidation_gate'   # consolidation_gate.GATE_METADATA_KEY re-binds it
# The x_ prefix is re-spelled here deliberately: shared cannot import fused_memory.memory_metadata.EXPERIMENTAL_KEY_PREFIX;
# λ's test asserts the key equals f'{EXPERIMENTAL_KEY_PREFIX}recon_consolidation_gate', so the two cannot drift.

@dataclass(frozen=True)
class GateDeclaration:
    task_kind: str | None
    always_escalates: object          # raw (value-sensitive: only True counts)
    has_before_done: bool
    operational_mode: object          # raw
    operational_mode_present: bool    # key present (an explicit None is present)
    has_gate_escalated_at: bool
    has_gate_block: bool              # a mapping under CONSOLIDATION_GATE_METADATA_KEY
    # THE rule (OQ-10 ruled) — what dispatch hands to a human:
    is_human_gate: bool               # property: task_kind == 'deterministic' and always_escalates is True
    # refinements of the rule:
    is_pure_human_gate: bool          # property: is_human_gate and not has_before_done (else act-then-ask)
    is_curator_gate: bool             # property: is_human_gate and operational_mode == 'gate'
    # honestly named selectors for the two consumers that ask another question:
    awaits_human_after_deterministic_step: bool   # property: (task_kind == 'deterministic' and (not operational_mode_present
                                                  #   or operational_mode == 'gate')) or has_gate_escalated_at — blocked-gate audit
    carries_curator_resolution_mode: bool         # property: operational_mode == 'gate' — curator-gate resolution sweep
def gate_declaration(metadata: Mapping[str, Any] | None) -> GateDeclaration: ...   # stored shape; None → all-false
```

Invariants: no fused-memory or dashboard module outside `gate_declaration.py` reads
`always_escalates`/`operational_mode` to decide whether a task is a gate (the orchestrator's
`DeterministicRunner`, whose dispatch the rule describes, keeps reading its typed `TaskMetadata`
fields); a submission is classified after `inject_operational_routing`; the two selectors are never
used as "is a gate" (λ's test table maps each consumer to its property).

### C7 — combine audit vocabulary (`fused-memory/src/fused_memory/middleware/combine_audit.py`, ζ; ρ consumes)

```python
class CombineRefusal(StrEnum):            # values byte-identical to today's COMBINE_REFUSED_* constants
    TARGET_NOT_PENDING = 'target_not_pending'; TARGET_CLAIMED = 'target_claimed'
    TARGET_HAS_BRANCH = 'target_has_branch'; TARGET_IS_HOLDER = 'target_is_holder'
    CANDIDATE_KEY_COLLISION = 'candidate_key_collision'      # new (OQ-14 ruled)
def combine_audit_path() -> Path: ...     # DARK_FACTORY_DATA_DIR or data/, then combine_audit.jsonl
def append_combine_audit(*, ..., old_files: Sequence[str], new_files: Sequence[str],
                         refused_reason: CombineRefusal | None = None, survivor_id: str | None = None) -> None
```

Invariants: append-only, best-effort (a failed append logs and never blocks the combine); historical keys
byte-identical; `survivor_id` present iff `refused_reason is CANDIDATE_KEY_COLLISION`. The census script
reads the file and the token through this module only.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | tool→backend | `update_task(metadata={…})` with no `metadata_mode` | any task | `error='metadata_mode_required'`, `error_type='WriteModeError'`; row unchanged |
| 2 | tool | `update_task(metadata={…}, append=True)` | any task | `error='append_retired'`, `repaired_call` carries `metadata_mode='additive'`; nothing written |
| 3 | backend | `details='x', text_mode='append'` plus `metadata={…}, metadata_mode='merge'` | row with details and nested `memory_hints` | details concatenated once; `memory_hints` shallow-merged |
| 4 | backend | `title='t', text_mode='append'` | any | `AppendUnsupportedFieldError` naming `text_mode` → dict |
| 5 | client | fused-memory refuses with `done_provenance_via_update_task` | stub transport (ω: real server) | `MetadataWriteFailed(error_type='DoneProvenanceWriteAuthorityError', rejection=<payload>)` |
| 6 | client | transport raises / reply is `isError` | stub transport | `MetadataWriteFailed(transport=exc)` / naming the envelope shape |
| 7 | workflow | `_handle_no_plan_failure`'s persist refused | `FakeMetadataBackend` set to refuse (ω: real backend floor) | `_mark_blocked(..., escalate_to_human=True)` is called |
| 8 | workflow | `_clear_merge_retry_pending` / `_clear_merge_phase_entered` refused | same | memory not overwritten; a warning names task and reason |
| 9 | interceptor | `update_task(metadata='not json', metadata_mode='merge')` | any | one `task_metadata.schema_warning … source=update_task` line; dict; backend never called |
| 10 | interceptor | backend raises `DuplicateCandidateKeyError` / `ValidationError` / `LeakedEnvelopeMarkupError` / `WriteModeError` / a survivor-less `IntegrityError` | any | a dict whose `error_type` is the declared code (or the class name for an undeclared class); nothing propagates |
| 11 | interceptor | `set_task_status('1,2,3', …)` with one write refused | three rows | `success=False`, `all_ok` from `write_failure_reason` |
| 12 | interceptor | `update_task(metadata={'task_kind': 'deterministic'}, metadata_mode='merge')` on a normal task | normal task | refused with `deterministic_task_error`'s dict; a milestone-only delta does not stat the script |
| 13 | store↔curator | `{files: 'a.py'}`, `{files: ['a.py', 'b.py']}`, and a legacy `{files: ['a.py'], files_to_modify: ['b.py']}` | — | the backend's `candidate_key` equals `TaskCurator._normalize_key`'s for each blob; the legacy key contributes nothing |
| 14 | backend | `set_task_status(done → pending, actor=ORCHESTRATOR)` directly, no reopen reason, `enforce_transitions=False` | done row | `IllegalTransitionError`, `error='terminal_exit_rejected'`; row unchanged; a call without `actor` is a `TypeError` |
| 15 | backend | an illegal edge for `RECONCILIATION` (`in-progress → x`), flag off then on | row | off: the `illegal_transition would-reject` WARNING from the backend, write lands; on: `error='illegal_transition'` |
| 16 | interceptor | row 14 through `TaskInterceptor.set_task_status` | done row | the same dict, rendered from the backend's refusal; the interceptor computes no verdict |
| 17 | backend | `get_statuses_fresh` on a missing (new project) / corrupt db file | — | `StatusCensusUnavailable(cause='store_absent' / 'read_failed')`; the recon harness logs once per cycle and skips that census |
| 18 | backend | two inserts colliding on `(tag, candidate_key)`; and a store whose index `WHERE` differs from the constant | index present | `DuplicateCandidateKeyError(existing_id)`; a NOT NULL `IntegrityError` propagates; the index check fails naming both clauses |
| 19 | registry | `python -I` per entry module (R3's five plus `fused_memory.server.tools`), the child binding members by `import _checkout` | — | identical `sorted(registered_slices())`, equal to the in-process table |
| 20 | registry | `parse_metadata(…, slices=SLICES \| {'x_stub': …})` | test table | the injected slice validates; `SLICES` unchanged afterwards |
| 21 | combine | candidate `['b.py']` combined into target `{files: ['a.py']}`; the stub curator's list omits `a.py` | — | target `metadata.files == ['a.py', 'b.py']`, details without the prefix, re-embed uses the same list |
| 22 | combine | the union collides with a live row's key | two live rows | target unchanged; one audit row with `refused_reason='candidate_key_collision'`, `survivor_id`, `old.files`, `new.files`; the ticket degrades to create |
| 23 | gate | the classification table: a normal task with `execution_class: 'operational'` (reify 5376's shape); a normal coding task with a stray `operational_mode: 'gate'` (dark-factory 2963's shape); deterministic + `always_escalates: True`, no `before_done`; the same with `before_done`; a submission `{execution_class: 'decision'}` before coercion; deterministic with `before_done`, no `always_escalates`, `operational_mode` absent; `{task_kind: 'deterministic', always_escalates: 'false'}`; `{task_kind: 'deterministic', operational_mode: None}` (key present, explicit `None`) | — | human gate: no / no / yes, pure / yes, not pure / yes after `inject_operational_routing` / no / no / no; `awaits_human_after_deterministic_step` is true for rows 3, 4, 6, 7 and false for row 8 (a present key that is not `'gate'`, which `operational_mode_present` exists to tell from absence); `carries_curator_resolution_mode` is true only for row 2; each consumer reads the property λ's table names |
| 24 | interceptor | construct without a journal | — | `TypeError` at construction |
| 25 | guard | a seeded module with `task.get('status') == 'blocked'`, and one with `esc.status == 'pending'` | fixture dir | flags the first, ignores the second; the post-ο tree passes |
| 26 | guard | seeded modules with `if not await scheduler.update_task(…)` and with `asyncio.gather(scheduler.update_task(…))` | fixture | β2's guard fails on both, naming file and function; the post-β2 tree passes |
| 27 | interceptor | `submit_task(metadata={'files_to_modify': ['a.py']})` and `update_task(metadata={'files_to_modify': […]}, metadata_mode='merge')`; then `update_task(metadata={'x_note': 1}, metadata_mode='merge')` on a legacy row already carrying the key | any task; a legacy row | both refused `error='files_to_modify_retired'`, `error_type='FilesToModifyRetiredError'`, hint naming `metadata.files`, `repaired_call` with the paths in `files`; nothing written; the unrelated write to the legacy row succeeds (incoming keys only) |
| 28 | guard | seeded modules reading `md.get('files_to_modify')`, `meta['files_to_modify']` and looping over `('files', 'files_to_modify')` | fixture | δ's guard flags all three naming file and line; the post-δ tree passes |
| 29 | script | `scripts/combine_collision_census.py --days 30` over fixture audit files: 4 collisions in 600 rows; 5 in 1,000; 3 in 200; an empty window | fixture jsonl | exit 0 / 1 (count) / 1 (rate) / 0; the trailing JSON verdict carries per-project counts, the window's combine rows, survivor ids and up to three examples |
| 30 | interceptor | `set_task_status(cancelled → done)` with no reopen reason through the interceptor, the pre-done hook stubbed to record its call | cancelled row | `error='terminal_exit_rejected'` from the backend; the hook ran (accepted cost, D7); status and metadata unchanged |

## Design invariant (θ performs the lockstep edit)

Minted here: **INV-16 `metadata-slices-one-table`** (OQ-5, revised by the lead on 2026-11-10). The lead
chains this wave's three invariant leaves at decompose — S6 α `lane-budgets-exceed-verify-timeouts`
(INV-15, first: it fixes a live misconfiguration) → S10 θ (INV-16) → S7 ζ `merge-gate-unmeasured-typed`
(INV-17); θ depends on S6 α and S7 ζ on θ. The slug is the stable id and every delivered check anchors on
it, so a re-ordered chain changes the number only. INV-14 is the last invariant at `e634cfe658`.

Sites (`scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`):
`docs/legibility/design-invariants.md` (section, header dating paragraph, `## Fixtures` list);
`docs/legibility/design-invariants-fixtures.md` (`INV-16-PRD`, `INV-16-CODE`, a dated addendum walk with
its 2-row rehearsal table and the cumulative count); `skills/prd/references/gates.md` (family-inventory
row, the `inv-trigger-shapes` span and the calibration sentence); `CONTRIBUTING.md` gets nothing (pinned
as an absence); `orchestrator/src/orchestrator/agents/code_quality.md`'s partial INV-to-heuristic mapping
needs no row (pinned pair-wise, never for completeness). Draft normative text:

> **Rule**: `parse_metadata`'s set of typed metadata slices is one table, `shared.task_metadata.SLICES`,
> built from models the module imports downward. No module registers a slice as an import side effect,
> and the set a process validates against is identical in a fresh interpreter whatever first-party
> module was imported first.
>
> **Checkable design question(s)**: Does this feature add a typed metadata slice, or a model a slice
> depends on? Is it a row of `SLICES` with its model in a module that does not import
> `shared.task_metadata`, or a registration somewhere a process may or may not import?
>
> **Evidence**: at `6551b75aba` one blob received four verdicts by entry module; at `e634cfe658` six
> distinct registries (9 to 14 slices) over nine entry modules, and fused-memory's write-enforce gate
> accepted a malformed `delivered_checks` entry unless a module importing `shared.capability_manifest`
> happened to be imported (`fk-f528087f44c7`, run `review-all-dark_factory-20261008`).
>
> **House pattern**: `SLICES`; the fixture `shared/tests/test_metadata_slices_one_table.py`.
>
> **Family boundary**: INV-5 `no-lockstep-duplication` governs duplicated logic; INV-9
> `one-fact-one-home` governs world-facts in records; this invariant governs one code table whose
> completeness must not depend on import order (heuristic 7).

Trigger shape for gates.md: *a typed metadata slice registered, or a slice model imported for its
registration side effect, anywhere but `shared.task_metadata.SLICES` (`metadata-slices-one-table`)?*

The fixture has two halves (INV-10): the fresh-interpreter parity test (row 19; the child prepends
`scripts/` and does `import _checkout`, the module's public binding contract) and an AST scan of
`shared/src`, `fused-memory/src`, `orchestrator/src`, `dashboard/src` for a module that mutates a
module-level mapping of `shared.task_metadata` — flagged on a seeded fixture module, clean on the post-η
tree. No exemption entries.

## Cross-PRD relationship (G4)

| Other stream / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S9 workflow phase spine | produces (to) | C2 (S9 λ `BlockReason` row; S9 μ `DryRunScheduler`) from β2; C3 slice home (S9 ξ) from η | S10 defines, S9 consumes | S9 λ1 depends on β2, S9 μ1 on β1 (and on β2 through S9 λ2 → λ1, so `DryRunScheduler` is born against `-> None`), S9 ξ on η (S9's own Deps lines); ι ↔ S9 ξ and ε/ξ ↔ S9's `workflow.py` chain ordered at decompose |
| S17 scheduler seams | produces (to) | `Scheduler.update_task` signature, `SchedulerFacade` parity | S10 | S17 not authored |
| S14 shared leaves — ζ | consumes (landed) | `parse_tool_result` + `write_failure_reason` (6711 done) | S14 | no edge |
| S14 shared leaves — ξ (6719) | consumes (pending) | `McpErrorType`, declared `error_type` codes, `error_envelope` — γ's renderer and C1/C5's codes | S14 | α depends on 6719; OQ-12 ruled: 6719 homes `error_envelope` in `shared.mcp_envelope` (R17) |
| S14 shared leaves — μ (6721) | consumes (pending) | `parse_agent_identity(…).actor_class` replaces `derive_actor_class`; edits `task_transitions.py`, `task_interceptor.py`, `tools.py`, `targeted.py` | S14 | α depends on 6721 (and 6720, which 6721 waits on) |
| S25 scripts instruments | consumes (landed) | C1 `scripts/_checkout.py` (import contract, `MEMBER_SRC`) | S25 | no edge |
| S5 git runner | adjacent | S5 μ rebuilds the interceptor's done-provenance probes in `task_interceptor.py` | S5 | α depends on S5 μ (same file; id at decompose). No S10 key covers git probes |
| S26 memory service | adjacent | S26 C3 (γ) reshapes `server/main.py::run_server` (constructed `MemoryService`, setters gone); S26 λ, μ and ρ edit `server/tools.py` (ρ: the `metadata_row_text` import); S26 μ edits `reconciliation/stages/task_knowledge_sync.py` (an α file); S26 γ0, θ2 and ν edit `invalidate_fabricated_shipping_edges`, `memory_eval_staleness_sweep`, `sweep_orphan_flag_markers` (κ1's) and `normalize_topic_slugs` (λ's) | S26 | κ1 and μ2 depend on S26 γ; the other pairs are ordered at decompose (D15) |
| S3 escalation server | adjacent | S3 C7a (created by ε1, completed by ε2): ε1 moves `merge_relane`/`_write_merge_lane_metadata` to `orchestrator/mcp/merge_tools.py` (and its test to `orchestrator/tests/`), and edits `task_metadata.py` docstrings | S3 | S3 ε1 depends on β1 and β2 (S3's own edge: ε1 carries their edit into the moved module); η/ι ↔ S3 ε1 ordered at decompose |
| S8 harness composition | adjacent | β1, β2 and ξ edit `harness.py` | S8 | same-file edges at decompose |
| S11 curator pipeline | adjacent | ζ edits the combine's write; 6708 (S11 ζ) is done; 6332 waits on ζ; μ1 waits on 6332; δ adds `CURATOR_FILES_FIELD` to S11's `curator_types.py` | S11 / S10 | edges as listed |
| Follow-up packing sweep (`plans/followup-packing-sweep-prd.md`, task 8453 P2) | adjacent | 8453 extracts `_write_combined_rewrite` (the combine/pack write half, the `FILES_TO_MODIFY` block, `_append_combine_audit`) and registers a `pack_record` slice | packing-sweep PRD | ζ depends on 8453; if η lands first, 8453 adds `pack_record` as a `SLICES` row (lead amend) |
| Cancel-with-children (`plans/cancel-with-children-prd.md`; 8143 in progress, 8148, 8264 pending) | adjacent | each edits `TaskInterceptor._apply_status_transition`, whose legality gates κ1 deletes and whose context gates ν moves | cancel-with-children PRD | α depends on 8143; 8148 and 8264 ordered against κ1/ν at decompose (D15) |
| S15 MCP server boundary | adjacent | μ2 reorders construction in `server/main.py` and `server/tools.py` | S15 | not authored |
| S12/S13 reconciliation | produces (to) | κ2: `get_statuses_fresh` raises; λ: recon gate selectors read C6, and the recon duplicate-gate guard sees every human gate | S10 | in-leaf |
| S2 escalation records | boundary | ξ/ο's guard ignores attribute operands; escalation statuses are S2's R9 invariant | S2 | xref |
| S6 / S7 merge lane | boundary | merge outcome statuses (S6); invariant leaves chained S6 α (INV-15) → S10 θ (INV-16) → S7 ζ (INV-17) (lead, revised 2026-11-10) | S6, S7 | θ depends on S6 α; S7 ζ depends on θ; the lead wires both at decompose |
| S25 λ (6766), S14 ο (6724) | adjacent | `routing_dispatch.py`, `workflow.py`, `scheduler.py`, `harness.py` | S25, S14 | β1 depends on both |

## Decomposition plan

Overlay bands: each leaf ~300–1,500 changed LOC. α, β1, β2, κ1, λ and μ1 cross the 15-declared-file
review trigger because their declared files are the measured test blast radius of one signature or rule
change (keyword additions, fake signatures, one predicate's readers), not several changes; δ (20 files)
and η (20) cross it for the same reason — δ's are the readers of one retired key plus its refusal, reader
guard and parity tests, η's the importers of the names it moves to leaf modules; β2 is atomic
because a partial landing would leave a reader treating `None` as a refusal. Every leaf in the batch is
`task_kind=normal`; π is `complexity='simple'`; ρ is a small script leaf (~200 LOC, above the floor).
σ is `task_kind='deterministic'` and is not in the batch: the lead files it once ρ is on main (D16).
Priority = highest severity of the keys a leaf closes; **high** leaves (α, β1, β2, γ, η, θ, ω) become
dependencies of trigger-chain gate 6698 at decompose.

Same-file serialization: `task_interceptor.py` α → γ → δ → κ1 → κ2 → λ → ζ → (6332) → μ1 → μ2 → ν → ο;
`sqlite_task_backend.py` α → γ → δ → κ1 → κ2; `tools.py` α → γ → … → μ2 (S26 λ, μ and ρ also edit
`server/tools.py` and are ordered against this chain at decompose, D15); `server/main.py` κ1 → … → μ2;
`fused-memory/tests/test_task_curator.py` δ → ζ; `reconciliation/harness.py` κ2 → λ; `docs/task-authoring.md` γ → π → ι;
`task_metadata.py` η → ι; orchestrator hot files β1 → β2 → ε → ξ.

Dependency graph: α → β1, γ; β1 → β2; β2 → ε; γ → δ, π, η; δ → κ1, ε; κ1 → κ2; κ2 → λ; λ → ζ; ζ → ρ, μ1;
μ1 → μ2; μ2 → ν; ν → ο; ε → ξ; ξ → ο; π → ι; η → ι, θ; ω after β2, γ, ζ, κ2, θ, ο. σ (lead-filed) after
ζ and ρ. Out of batch, before: 4867 (hand-carried; ruled 2026-11-10), 6621, 5281, 6719, 6720, 6721, 8143,
S5 μ → α; 6766, 6724 → β1; S26 γ → κ1, μ2; S6 α → θ; 8453 → ζ; ζ → 6332 → μ1. Out of batch, after (lead
edges): 5310 on ο, ξ; 7173 on κ2, ι; 6678 on ο; 3260 on ο, ξ; 7748 on δ; S7 ζ on θ; S9 λ1 on β2, S9 μ1
on β1, S9 ξ on η; S3 ε1 on β1, β2. Cancelled into
leaves: 3150 → β1, 6129 → β2, 3093 and 7085 → γ, 6653 → δ.

**α — Two typed write axes on the wire; the backend resolves once** (priority **high**, est. ~1,100 LOC)
- Files: `shared/src/shared/task_metadata_wire.py`, `shared/tests/test_task_metadata_wire.py`,
  `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`,
  `fused-memory/src/fused_memory/backends/task_backend_protocol.py`,
  `fused-memory/src/fused_memory/backends/task_backend_errors.py`,
  `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/middleware/recurring_gate_guard.py`, `fused-memory/src/fused_memory/server/tools.py`,
  `fused-memory/src/fused_memory/server/manifest_stamping.py`,
  `fused-memory/src/fused_memory/reconciliation/targeted.py`,
  `fused-memory/src/fused_memory/reconciliation/prompts/stage2.py`,
  `fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py`,
  `fused-memory/src/fused_memory/reconciliation/enumeration_guard.py`, `scripts/sitting/payloads.py`,
  `scripts/tests/test_sitting_payloads.py`, `fused-memory/tests/test_sqlite_task_backend.py`,
  `fused-memory/tests/test_recon_write_policy.py`, `fused-memory/tests/test_audit_trail_rotation_boundary.py`,
  `fused-memory/tests/test_task_interceptor.py`, `fused-memory/tests/test_task_metadata_boundary.py`,
  `fused-memory/tests/test_defect_records_update_floor.py`, `fused-memory/tests/test_pending_since_anchor.py`,
  `fused-memory/tests/test_recon_reliability_integration.py`, `fused-memory/tests/test_stale_gate_citation_guard.py`,
  `fused-memory/tests/test_task_tools.py`, `fused-memory/tests/test_append_rejection_wire_shape.py`,
  `fused-memory/tests/test_task_text_markup_chokepoint.py`,
  `fused-memory/tests/reconciliation/test_citation_verifier.py`,
  `fused-memory/tests/test_recon_gate_subject_guidance.py`, `fused-memory/tests/test_recurring_gate_guard.py`,
  `fused-memory/tests/reconciliation/test_enumeration_guard.py`,
  `fused-memory/tests/test_targeted.py` (re-measure: AST of test `update_task(` calls with `metadata` and
  no `metadata_mode`, plus `git grep -lE "append=" -- fused-memory/tests`).
- Keys: `fk-ff5234ac1013`, `fk-116481623cfb`.
- Deps: 4867 (hand-carried, ruled by Leo 2026-11-10), 6621, 5281, 6719, 6720, 6721, 8143, S5 μ (each
  still open at decompose; 6708 is done). Seam: produces C1.
- Deletes: `_resolve_metadata_mode`, `_METADATA_MODES`, both carve-outs and their essays, the "append IS
  NOT SCOPED TO METADATA" block, the tool docstring's knob essay, `append` from protocol, interceptor
  and backend signatures, every `append=True` instruction in fused-memory prose.
- Signal: boundary rows 1–4 in CI; on the live server a mode-less metadata write returns
  `metadata_mode_required` and `append=True` returns `append_retired`; recon Stage 2 runs a cycle that
  attaches hints with no `WriteModeError` in its journal.
- G7: INV-2, INV-11, INV-5.

**β1 — Client signature: `metadata_mode` required, `append` gone, facade and fakes in parity** (priority
**high**, est. ~700 LOC, mechanical)
- Files: `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/deterministic_runner.py`,
  `orchestrator/src/orchestrator/dry_run_unblock.py`, `orchestrator/src/orchestrator/routing_dispatch.py`,
  `orchestrator/src/orchestrator/strand_conversion.py`, `orchestrator/src/orchestrator/merge_lane/skew_tripwire.py`,
  `orchestrator/src/orchestrator/stranded_verified_green.py`,
  `orchestrator/src/orchestrator/scheduled_deploy_settle.py`, `escalation/src/escalation/server.py` (or
  `orchestrator/src/orchestrator/mcp/merge_tools.py` after S3 ε1), `orchestrator/tests/_workflow_helpers.py`,
  and the test files with fakes, `append=` or call-arg assertions: `escalation/tests/test_merge_relane.py`,
  `orchestrator/tests/test_scheduled_deploy_settle.py`,
  `orchestrator/tests/test_auto_eval.py`, `orchestrator/tests/test_block_disposition.py`,
  `orchestrator/tests/test_convert_to_blocked.py`, `orchestrator/tests/test_deterministic_runner.py`,
  `orchestrator/tests/test_deterministic_runner_predicate_recheck.py`, `orchestrator/tests/test_deterministic_task.py`,
  `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_harness_action_dispatch.py`,
  `orchestrator/tests/test_harness_deterministic_dispatch.py`, `orchestrator/tests/test_harness_infra_hold_repend.py`,
  `orchestrator/tests/test_harness_module_tagging.py`, `orchestrator/tests/test_harness_pipeline_tripwire.py`,
  `orchestrator/tests/test_harness_prd_tagging.py`, `orchestrator/tests/test_harness_retry_tier.py`,
  `orchestrator/tests/test_merge_queue_dry_run_unblock.py`, `orchestrator/tests/test_merge_skew_tripwire.py`,
  `orchestrator/tests/test_milestone_integration_gate.py`, `orchestrator/tests/test_out_of_band_routing.py`,
  `orchestrator/tests/test_plan_revalidation_skip.py`, `orchestrator/tests/test_plan_text_fp_preserving_stamps.py`,
  `orchestrator/tests/test_recut_ruling_dispatch.py`, `orchestrator/tests/test_resume_claimant_liveness.py`,
  `orchestrator/tests/test_routing_dispatch.py`, `orchestrator/tests/test_routing_integration_gate.py`,
  `orchestrator/tests/test_scheduler.py`, `orchestrator/tests/test_scheduler_tick_phases.py`,
  `orchestrator/tests/test_simple_task_workflow.py`, `orchestrator/tests/test_strand_conversion.py`,
  `orchestrator/tests/test_stranded_verified_green.py`, `orchestrator/tests/test_verify_plan_integration.py`,
  `orchestrator/tests/test_workflow.py`, `orchestrator/tests/test_workflow_already_done.py`,
  `orchestrator/tests/test_workflow_dry_run_hook.py`, `orchestrator/tests/test_workflow_first_enqueue_stamp.py`,
  `orchestrator/tests/test_workflow_infra_thrash.py`, `orchestrator/tests/test_workflow_merge_phase_liveness.py`,
  `orchestrator/tests/test_workflow_merge_provenance.py`, `orchestrator/tests/test_workflow_merge_retry_pending.py`,
  `orchestrator/tests/test_workflow_merge_thrash.py`, `orchestrator/tests/test_workflow_metadata_files_reconcile.py`,
  `orchestrator/tests/test_workflow_no_plan_cycle.py`, `orchestrator/tests/test_workflow_ready_to_merge.py`,
  `orchestrator/tests/test_workflow_scope_wipe_regression.py`, `orchestrator/tests/test_workflow_state_machine_boundary.py`,
  `orchestrator/tests/test_workflow_status_on_resume.py`, `orchestrator/tests/test_workflow_train_former.py`,
  `orchestrator/tests/test_workflow_train_thrash_guard.py`, `orchestrator/tests/test_workflow_verify_infra_resume.py`
  (re-measure with the critic's union rule at decompose).
- Keys: closes the client half of `fk-ff5234ac1013`; `fk-5c98cc0bb7e0` (carried from 3150).
- Deps: α; 6766, 6724. Seam: consumes C1.
- Deletes: the client resolver, `_reject_contradictory_metadata_mode`, the five `reportCallIssue`
  suppressions, `append` on the orchestrator side.
- Signal: `pyright` clean on the touched modules with the suppressions removed; the orchestrator suite
  green; `Scheduler.update_task`'s signature has no `append`.
- G7: INV-5 (one resolver; the fake calls `write_modes`).

**β2 — `Scheduler.update_task` raises `MetadataWriteFailed`; nothing reads its result** (priority
**high**, est. ~800 LOC)
- Files: `orchestrator/src/orchestrator/task_write.py` (new), `orchestrator/src/orchestrator/scheduler.py`,
  `orchestrator/src/orchestrator/workflow.py`, `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/src/orchestrator/deterministic_runner.py`, `orchestrator/src/orchestrator/strand_conversion.py`,
  `orchestrator/src/orchestrator/merge_lane/skew_tripwire.py`,
  `orchestrator/src/orchestrator/stranded_verified_green.py`,
  `orchestrator/src/orchestrator/scheduled_deploy_settle.py`, `escalation/src/escalation/server.py` (or its
  S3 ε1 home), `orchestrator/tests/_workflow_helpers.py`, `orchestrator/tests/test_task_write.py` (new),
  `orchestrator/tests/test_scheduled_deploy_settle.py`,
  `orchestrator/tests/test_update_task_result_never_read.py` (new, with
  `orchestrator/tests/fixtures/update_task_result_read_seeded.py`), `orchestrator/tests/test_scheduler.py`,
  `orchestrator/tests/test_harness_action_dispatch.py`, `orchestrator/tests/test_workflow_merge_retry_pending.py`,
  `orchestrator/tests/test_recut_ruling_dispatch.py`, `orchestrator/tests/test_workflow_no_plan_cycle.py`,
  `orchestrator/tests/test_strand_conversion.py`, `orchestrator/tests/test_merge_skew_tripwire.py`,
  `orchestrator/tests/test_deterministic_runner.py`, `orchestrator/tests/test_deterministic_runner_predicate_recheck.py`,
  `escalation/tests/test_merge_relane.py`.
- Keys: `fk-de964b86bdd8`.
- Deps: β1. Seam: produces C2.
- Deletes: the `-> bool` contract and every read of it; the 14 `try/except` arms are **kept and become
  live** (R3).
- Signal: boundary rows 5–8 and 26 in CI; in production a refused stamp appears as the arm's documented
  block or escalation in the task's escalation history.
- G7: INV-11, INV-2, INV-10 (the guard runs against a seeded fixture).

**γ — One parse per request, every rejection a dict through one renderer, invariants on update**
(priority **high**, est. ~900 LOC)
- Files: `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/middleware/metadata_dict.py`,
  `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`,
  `fused-memory/src/fused_memory/backends/task_backend_errors.py`,
  `fused-memory/src/fused_memory/middleware/deterministic_task_guard.py`,
  `fused-memory/src/fused_memory/server/tools.py`, `fused-memory/tests/test_task_interceptor.py`,
  `fused-memory/tests/test_metadata_dict.py`, `fused-memory/tests/test_sqlite_task_backend.py`,
  `fused-memory/tests/test_task_metadata_boundary.py`, `fused-memory/tests/test_task_tools.py`,
  `fused-memory/tests/test_deterministic_task_guard.py`.
- Keys: `fk-f6224c565a9b`, `fk-1f8a4c69798f`, `fk-cc04e61d23ec`.
- Deps: α (and through α 6621, 5281, 6719). Seam: consumes C1, S14 ξ's `error_envelope`.
- Deletes: the interceptor helpers named in §Sketch γ, the second `parse_metadata` pass, the per-class
  enumeration in `update_task`'s `except`, the CSV `all_ok` predicate and the docstring of
  `_consolidation_not_closed_error` that restates it, `deterministic_task_guard`'s discard-on-malformed
  parse.
- Signal: boundary rows 9–12 in CI; on the live server a malformed metadata string returns a dict with
  exactly one census line, and a `task_kind='deterministic'` delta on a normal task is refused.
- G7: INV-11, INV-10, INV-5.

**π — Prose callers of `update_task` state the two axes** (priority medium, `complexity='simple'`, est.
~200 LOC of prose)
- Files: `skills/escalation-watcher/SKILL.md`, `skills/escalation-watcher-auto/SKILL.md`,
  `skills/unblock-low-risk/SKILL.md`, `skills/_shared/filing-the-trigger-chain.md`,
  `skills/review/references/phase3-triage.md`, `skills/_shared/reading-the-invariants-gauge.md`,
  `skills/review-all/scripts/program/clerk_batch.py` (one prompt string), `docs/task-authoring.md` (§5
  update-time deterministic validation; §9/§10 the two axes; its three `files_to_modify` mentions).
- Keys: none of its own. Deps: γ (task 6392, its former prerequisite, is done).
- Signal: `git grep -nE "append=(True|true)" -- skills docs/task-authoring.md` is empty, and
  `git grep -n files_to_modify -- docs/task-authoring.md` is empty.
- G7: INV-9.

**δ — `shared.declared_files`: one key, one reader; `files_to_modify` refused and unread;
`lock_charter_guard` imports `shared.locking`** (priority medium, est. ~850 LOC; OQ-9 ruled)
- Files: `shared/src/shared/declared_files.py` (new), `shared/tests/test_declared_files.py` (new),
  `shared/tests/test_retired_files_key_unread.py` (new), `shared/tests/fixtures/retired_files_key_read_seeded.py`
  (new), `shared/src/shared/briefing_queries.py`, `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`,
  `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/middleware/task_curator.py`,
  `fused-memory/src/fused_memory/middleware/curator_types.py`,
  `fused-memory/src/fused_memory/middleware/path_scope_guard.py`,
  `fused-memory/src/fused_memory/middleware/lock_charter_guard.py`,
  `fused-memory/src/fused_memory/middleware/gitignored_deliverable_guard.py`,
  `fused-memory/src/fused_memory/middleware/operational_suggestion_guard.py`,
  `fused-memory/src/fused_memory/middleware/simple_task_files_guard.py`,
  `fused-memory/src/fused_memory/reconciliation/targeted.py`, `fused-memory/tests/test_lock_charter_guard.py`,
  `fused-memory/tests/test_candidate_key_parity.py` (new; row 13), `fused-memory/tests/test_task_curator.py`,
  `fused-memory/tests/test_retired_files_key_refusal.py` (new; row 27), `shared/tests/test_locking.py`.
- Keys: `fk-fa3e3d765f50`, `fk-fa355c2c0498`. Deps: γ. Seam: produces C4. Absorbs task 6653 (cancelled
  into δ; its malformed-`files` test for `backfill_corpus` lands here).
- Deletes: `_files_for_key`, `_extract_meta_files_from_meta`, `_extract_metadata_files`,
  `_DELIVERABLE_SIGNAL_KEYS` and the inline `('files', 'files_to_modify')` loop, `task_curator._task_files`'s
  metadata arms, `path_scope_guard`'s union rationale, every read of a metadata `files_to_modify` key,
  `lock_charter_guard`'s own parsing and its three copies, the tier-1 drift test and the false rationale.
- Signal: boundary rows 13, 27 and 28 in CI; on the live server after the fused-memory restart a
  `submit_task` carrying `metadata.files_to_modify` returns `files_to_modify_retired`; the D4 audit query
  over the live store reports 0 re-keyed rows; `lock_charter_guard.FILE_EXTENSIONS is
  shared.locking.FILE_EXTENSIONS`. Lead action after the deploy: re-measure all stores and backfill the
  rows whose `files_to_modify` adds paths.
- G7: INV-5, INV-3, INV-10 (the guard runs against a seeded fixture), INV-12 (no exemption entries).

**ε — Orchestrator reads declared files through `shared.declared_files`; one sanitizer** (priority
medium, est. ~350 LOC)
- Files: `orchestrator/src/orchestrator/module_charter.py`, `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/scheduler.py`,
  `scripts/trial_module_tagger_haiku.py`, `scripts/migrate_metadata_modules_to_files.py`,
  `orchestrator/tests/test_module_charter.py`, `orchestrator/tests/test_workflow_metadata_files_reconcile.py`,
  `orchestrator/tests/test_harness_module_tagging.py`, `tests/scripts/test_trial_module_tagger_haiku.py`,
  `orchestrator/tests/test_lock_derivation_declared_files.py` (new; the boundary row below).
- Keys: contributes to `fk-fa3e3d765f50`. Deps: δ, β2. Seam: consumes C4.
- Deletes: `module_charter.sanitize_files_for_persist`; every direct `metadata.get('files')`
  lock-derivation read in the named modules.
- Signal: `git grep -n sanitize_files_for_persist -- orchestrator/src scripts` is empty; a CI boundary
  test derives locks for a task whose `metadata.files` is a plain string and for a legacy task that also
  carries `files_to_modify`, and asserts each equals `derive_modules(declared_files(md))` — the string is
  one path and the legacy key adds nothing (the live store has 0 non-terminal rows carrying the key, so a
  live signal would be vacuous).
- G7: INV-5.

**ζ — The combine writes a union it computes into `metadata.files`; a collision is a counted refusal**
(priority medium, est. ~450 LOC)
- Files: `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/middleware/combine_audit.py` (new; C7),
  `fused-memory/tests/test_combine_audit.py` (new), `fused-memory/tests/test_ticket_worker.py`,
  `fused-memory/tests/test_task_interceptor.py`, `fused-memory/tests/test_task_curator.py`.
- Keys: `fk-de3df95984a5`. Deps: λ (same file; δ and κ2 upstream); task 8453 (extracts the write half ζ
  edits). Seam: consumes C4, C5; produces C7.
- Deletes: the `FILES_TO_MODIFY:` prefix, the false comment, the bare `except Exception` around the write,
  the `COMBINE_REFUSED_*` constants and `_combine_audit_path` in `task_interceptor.py` (moved to C7).
  Adds `old.files`, `new.files`, the `candidate_key_collision` token and `survivor_id` to the audit record
  (historical keys byte-identical).
- Signal: boundary rows 21–22 in CI (row 22 asserts `refused_reason='candidate_key_collision'` and
  `survivor_id`); on the live store after the next fleet combine, rows with `combined_at` after the merge
  time carry no prefix and their `metadata.files` is a superset of the `old.files` that
  `data/combine_audit.jsonl` records for them.
- G7: INV-11, INV-13.

**ρ — Combine-collision census script** (priority medium, est. ~200 LOC; OQ-14 ruled)
- Files: `scripts/combine_collision_census.py` (new), `scripts/tests/test_combine_collision_census.py` (new).
- Keys: contributes to `fk-de3df95984a5` (the measurement behind D6). Deps: ζ (imports C7). Seam: consumes C7.
- Signal: boundary row 29 in CI; run by hand on main against `data/combine_audit.jsonl` it prints the
  verdict JSON with the window's per-project counts. Its landing is the lead's trigger to file σ.
- G7: INV-13 (reads the token its producer writes, through the producer's module), INV-10.

**σ — Deterministic 30-day collision check** (lead-filed after ρ is on main; not in the decompose batch;
`task_kind='deterministic'`, priority medium)
- Metadata: `before_done = {script: 'scripts/combine_collision_census.py', args: ['--days', '30'], kind:
  'predicate', timeout_secs: 120}`, `milestone = {mode: 'delayed', after_secs: 2592000}`,
  `always_escalates: false`, `prd_path` this PRD, `prd_task_label: 'σ'`.
- Deps: ζ, ρ. Keys: none of its own.
- Outcome: exit 0 → done, counts in `done_provenance`; exit 1 → born-at-L2 `milestone_check_failed` to
  Leo with the counts; the follow-up fix is filed by Leo or the L2 watcher.

**η — The slice table: static, downward, one home** (priority **high**, est. ~950 LOC)
- Files: `shared/src/shared/task_metadata.py`, `shared/src/shared/external_dep.py` (new),
  `shared/src/shared/delivered_check.py` (new), `shared/src/shared/capability_manifest.py`,
  `shared/src/shared/deploy_state.py`, `shared/src/shared/defects.py`, `shared/src/shared/followup_items.py`,
  `fused-memory/src/fused_memory/reconciliation/stale_gate_citation_guard.py`,
  `fused-memory/src/fused_memory/server/manifest_stamping.py`,
  `orchestrator/src/orchestrator/delivered_checks.py`, `scripts/audit_manifest_descriptor_drift.py`,
  `scripts/audit_combine_gate_marker_loss.py`, `shared/tests/test_task_metadata.py`,
  `shared/tests/test_capability_manifest.py`, `shared/tests/test_task_metadata_merge_retry_pending.py`,
  `shared/tests/test_delivered_check_polarity.py`, `orchestrator/tests/test_briefing.py`,
  `fused-memory/tests/server/test_write_triage_flip_gate_invariants.py`,
  `fused-memory/tests/test_deploy_state_registration.py` (deleted), `orchestrator/tests/test_deploy_state.py`
  (re-measure: `git grep -n "ExternalDep\|DeliveredCheckMeta\|MECHANICAL_CHECK_KINDS"`).
- Keys: `fk-f528087f44c7`. Deps: γ (shared test file; α's `manifest_stamping.py` edit upstream; 5281
  through α). Seam: produces C3.
- Deletes: `register_metadata_submodel`, `_SUBMODEL_REGISTRY`, `_SUBMODEL_CARDINALITY`, the five
  side-effect registrations (`deploy_state`, `capability_manifest`, `defects` ×2, `followup_items`) and
  their comments, the autouse registry-snapshot fixture, the `_stub` convention,
  `test_deploy_state_registration.py`.
- Signal: boundary row 20 in CI; the R3 probe prints one set (14 slices at `e634cfe658`, plus any that
  5281 or 8453 add first) for each of R3's five entry modules; `parse_metadata` with a malformed
  `delivered_checks` entry raises under `direction='write', enforce=True` with only
  `fused_memory.backends.sqlite_task_backend` imported.
- G7: INV-5, INV-9, INV-10.

**θ — INV-16 `metadata-slices-one-table`: lockstep edit and fixture** (priority **high**, est. ~350 LOC)
- Files: `docs/legibility/design-invariants.md`, `docs/legibility/design-invariants-fixtures.md`,
  `skills/prd/references/gates.md`, `shared/tests/test_metadata_slices_one_table.py` (new),
  `shared/tests/fixtures/metadata_slice_side_effect_registration.py` (new).
- Keys: contributes to `fk-f528087f44c7`; lands R22 item 4.
- Deps: η; S6 α (INV-15, lane-budgets-exceed-verify-timeouts) on the three invariant files; S7 ζ
  (INV-17) depends on θ (OQ-5, revised by the lead 2026-11-10; the lead wires both edges at decompose).
- Signal: `scripts/tests/test_design_invariants_consistency.py` passes with the slug in every pinned
  site; the fixture flags its seeded module and passes the post-η tree; the `python -I` probes agree.
- G7: INV-10, INV-12.

**ι — The blessed-key table in docs §8; the code keeps the frozenset** (priority medium, est. ~450 LOC)
- Files: `shared/src/shared/task_metadata.py`, `docs/task-authoring.md`,
  `tests/scripts/test_task_authoring_blessed_keys_drift.py`,
  `tests/scripts/test_task_authoring_tier_b_canonical_keys.py`, `tests/scripts/task_authoring_marker_span.py`.
- Keys: `fk-941b5369a3d7`. Deps: η, π.
- Deletes: the 333 comment lines' census, essays and inventories; the mirror block becomes the table.
- Signal: the drift test pins the table's key column against `_BLESSED_METADATA_KEYS`; §8 "Promoting a
  convention" asks for one row, one frozenset entry, and the census in the commit message.
- G7: INV-9, INV-5.

**κ1 — One transition verdict, enforced in the backend only** (priority medium, est. ~800 LOC; OQ-6 ruled)
- Files: `shared/src/shared/task_transitions.py`, `shared/tests/test_task_transitions.py`,
  `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`,
  `fused-memory/src/fused_memory/backends/task_backend_errors.py`,
  `fused-memory/src/fused_memory/backends/task_backend_protocol.py`,
  `fused-memory/src/fused_memory/middleware/task_interceptor.py`, `fused-memory/src/fused_memory/server/main.py`,
  `fused-memory/src/fused_memory/maintenance/backfill_curator_corpus.py`,
  `fused-memory/scripts/audit_duplicate_tasks.py`, `fused-memory/scripts/audit_found_on_main_provenance.py`,
  `fused-memory/scripts/check_found_on_main_spurious_rate.py`,
  `fused-memory/scripts/correct_found_on_main_backlog.py`,
  `fused-memory/scripts/derive_orphaned_recon_escalations.py`,
  `fused-memory/scripts/invalidate_fabricated_shipping_edges.py`,
  `fused-memory/scripts/measure_stale_status_snapshot_sweep.py`,
  `fused-memory/scripts/memory_eval_staleness_sweep.py`, `fused-memory/scripts/sweep_orphan_flag_markers.py`,
  `fused-memory/tests/test_sqlite_task_backend.py`, `fused-memory/tests/test_task_interceptor.py`,
  `fused-memory/tests/test_pending_since_anchor.py`, `fused-memory/tests/test_status_authority_gate.py`,
  `fused-memory/tests/test_recurrence_mint.py`, `fused-memory/tests/test_candidate_key_boundary.py`,
  `fused-memory/tests/test_task_text_markup_chokepoint.py`, `fused-memory/tests/test_followup_r4.py`
  (the 67 direct backend status-writer calls that gain `actor=`), `fused-memory/tests/test_deferral_gate.py`,
  `fused-memory/tests/test_consolidation_closure_seam.py` (interceptor-level tests asserting
  `terminal_exit_rejected`/`illegal_transition` move onto a real backend's refusal),
  `fused-memory/tests/test_build_task_backend_task_status.py` (new; `_build_task_backend` passes the
  deployed `TaskStatusConfig`).
- Keys: `fk-d80c1c44bca1`. Deps: δ; S26 γ (`server/main.py`). Seam: produces C5 (verdict half).
- Deletes: the interceptor's gate 2a refusal and gate 2e — `_terminal_exit_error`,
  `_illegal_transition_error`, `_enforce_transitions` and its `is_legal_transition` call — and the
  `derive_actor_class` read there (S14 μ's `parse_agent_identity` replaces it).
- Signal: boundary rows 14–16 and 30 in CI; against a copy of the live db a direct backend
  `set_task_status` from `done` to `pending` with no reopen reason raises `IllegalTransitionError`, and the
  same call through the MCP tool returns the unchanged `terminal_exit_rejected` dict.
- G7: INV-10, INV-2.

**κ2 — Privileged-writer protocol, census failure, candidate-key scope** (priority medium, est. ~600 LOC)
- Files: `fused-memory/src/fused_memory/backends/task_backend_protocol.py`,
  `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`,
  `fused-memory/src/fused_memory/backends/task_backend_errors.py`,
  `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/reconciliation/harness.py`,
  `fused-memory/src/fused_memory/reconciliation/orphaned_recon_escalation_sweep.py`,
  `fused-memory/scripts/derive_orphaned_recon_escalations.py`, `fused-memory/tests/test_sqlite_task_backend.py`,
  `fused-memory/tests/test_task_interceptor.py`,
  `fused-memory/tests/reconciliation/test_orphaned_recon_escalation_sweep.py`, `fused-memory/tests/test_harness.py`,
  `fused-memory/tests/test_derive_orphaned_recon_escalations.py`,
  `fused-memory/tests/test_sqlite_task_backend_atomicity.py`.
- Keys: `fk-260352fa900a`, `fk-c41aa02e0779`, `fk-d4d61005457b`. Deps: κ1. Seam: produces C5 (surface half).
- Deletes: the 5 live scope-predicate copies (migrations keep theirs), the 5 SELECT+raise constructions,
  the 3 `str(exc)` sniffs, the 3 `type: ignore[attr-defined]`, the `{}`-on-failure arms, the consumers'
  re-invented detection, the "12-method" claim, the fixture's bare `AsyncMock`.
- Signal: boundary rows 17–18 in CI; `pyright` clean on `task_interceptor.py` with the suppressions
  removed.
- G7: INV-11, INV-2, INV-5.

**λ — One gate rule: deterministic and `always_escalates`** (priority medium, est. ~600 LOC; OQ-10 ruled)
- Files: `shared/src/shared/gate_declaration.py` (new), `shared/tests/test_gate_declaration.py` (new),
  `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/middleware/recurring_gate_guard.py`,
  `fused-memory/src/fused_memory/middleware/recon_write_policy.py`,
  `fused-memory/src/fused_memory/reconciliation/gate_owned_finding_phrasing.py`,
  `fused-memory/src/fused_memory/reconciliation/consolidation_gate.py`,
  `fused-memory/src/fused_memory/reconciliation/curator_gate_resolution_sweep.py`,
  `fused-memory/src/fused_memory/reconciliation/blocked_gate_audit_section.py`,
  `fused-memory/src/fused_memory/reconciliation/harness.py`,
  `fused-memory/src/fused_memory/reconciliation/live_workflow_section.py`,
  `fused-memory/src/fused_memory/services/live_workflow_detector.py`, `fused-memory/scripts/normalize_topic_slugs.py`,
  `fused-memory/tests/reconciliation/test_gate_owned_finding_phrasing.py`,
  `fused-memory/tests/reconciliation/test_curator_gate_resolution_sweep.py`,
  `fused-memory/tests/test_recurring_gate_guard.py`, `fused-memory/tests/test_task_interceptor.py`,
  `fused-memory/tests/test_live_workflow_detector.py`, `fused-memory/tests/test_recon_write_policy.py`,
  `fused-memory/tests/test_harness.py`, `dashboard/src/dashboard/data/prd_view.py`,
  `dashboard/tests/test_prd_view.py`.
- Keys: `fk-06ab2d3e9148`. Deps: κ2. Seam: produces C6.
- Deletes: `_is_gate_metadata`, `is_gate_submission`'s predicate body, `is_pure_gate_metadata`,
  `prd_view.is_human_gate`, the phrasing selector's inline rule and essay, each module's own gate read.
- Signal: boundary row 23 in CI (with reify 5376's and dark-factory 2963's shapes as fixtures);
  `git grep -n "get('operational_mode'" -- fused-memory/src` finds only `operational_routing_guard.py` (a
  routing rule). Live, for the week after the fused-memory restart: the count of `RecurringGateViolation`
  refusals to recon-stage callers against the week before; the commit message records the before-count
  and the leaf's verifier records the after-count.
- G7: INV-5, INV-9.

**μ1 — One interceptor test factory** (priority medium, est. ~950 LOC, mechanical)
- Files: `fused-memory/tests/_interceptor_helpers.py` (new) and the 49 test files that construct a
  `TaskInterceptor` (measured at `e634cfe658`): `fused-memory/tests/_hermetic_task_stack.py`,
  `fused-memory/tests/_submit_stack.py`, `fused-memory/tests/test_curation_pipeline.py`,
  `fused-memory/tests/test_curator_corpus_streams.py`, `fused-memory/tests/test_data_dir_hermeticity.py`,
  `fused-memory/tests/test_declared_scope_stamp.py`, `fused-memory/tests/test_get_tasks_projection.py`,
  `fused-memory/tests/test_r4_idempotency.py`,
  `fused-memory/tests/_curator_helpers.py`, `fused-memory/tests/server/test_recon_report_citations.py`,
  `fused-memory/tests/server/test_recon_report_store.py`, `fused-memory/tests/test_append_rejection_wire_shape.py`,
  `fused-memory/tests/test_audit_trail_rotation_boundary.py`, `fused-memory/tests/test_auto_consolidation_e2e.py`,
  `fused-memory/tests/test_backfill_curator_corpus.py`, `fused-memory/tests/test_backlog_policy.py`,
  `fused-memory/tests/test_candidate_key_boundary.py`, `fused-memory/tests/test_check_consolidation_closure_cli.py`,
  `fused-memory/tests/test_consolidation_closure_seam.py`, `fused-memory/tests/test_curation_deferral_acceptance.py`,
  `fused-memory/tests/test_defect_creation_floor.py`, `fused-memory/tests/test_defect_records_update_floor.py`,
  `fused-memory/tests/test_defect_write_tools.py`, `fused-memory/tests/test_deferral_gate.py`,
  `fused-memory/tests/test_delivered_checks_e2e.py`, `fused-memory/tests/test_done_provenance_chokepoint.py`,
  `fused-memory/tests/test_filer_first_attribution.py`, `fused-memory/tests/test_find_tasks_by_metadata_tool.py`,
  `fused-memory/tests/test_interceptor_amendment_advisory.py`, `fused-memory/tests/test_interceptor_terminal_claimant.py`,
  `fused-memory/tests/test_live_task_write_guard.py`, `fused-memory/tests/test_main_ticket_janitor_wiring.py`,
  `fused-memory/tests/test_operational_routing_boundary_matrix.py`, `fused-memory/tests/test_pending_since_anchor.py`,
  `fused-memory/tests/test_recon_reliability_integration.py`, `fused-memory/tests/test_recon_write_policy.py`,
  `fused-memory/tests/test_recurrence_mint.py`, `fused-memory/tests/test_recurring_gate_boundary.py`,
  `fused-memory/tests/test_resolve_ticket_vocabulary.py`, `fused-memory/tests/test_review_suggestion_curator_intake_e2e.py`,
  `fused-memory/tests/test_stale_gate_citation_guard.py`, `fused-memory/tests/test_status_authority_gate.py`,
  `fused-memory/tests/test_task_creation_journal.py`, `fused-memory/tests/test_task_interceptor.py`,
  `fused-memory/tests/test_task_text_markup_internal_writers.py`, `fused-memory/tests/test_task_tools.py`,
  `fused-memory/tests/test_task_write_agent_id.py`, `fused-memory/tests/test_ticket_parking.py`,
  `fused-memory/tests/test_ticket_worker.py`.
- Keys: contributes to `fk-85e1293e0aa4` (μ2 closes). Deps: ζ (same test files); task 6332 (which waits on ζ).
- Deletes: 247 direct `TaskInterceptor(` construction lines in tests; the two shared stacks build through
  the factory.
- Signal: `git grep -c "TaskInterceptor(" -- fused-memory/tests` counts only `_interceptor_helpers.py`;
  the fused-memory suite is green.
- G7: Tests stance (one public construction seam).

**μ2 — Collaborators at construction; the setters go** (priority medium, est. ~600 LOC; OQ-3 ruled)
- Files: `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/server/main.py`, `fused-memory/src/fused_memory/server/tools.py`,
  `fused-memory/tests/_interceptor_helpers.py`, and the 9 test files that call the setters on an
  interceptor: `fused-memory/tests/test_audit_trail_rotation_boundary.py`,
  `fused-memory/tests/test_auto_consolidation_e2e.py`, `fused-memory/tests/test_check_consolidation_closure_cli.py`,
  `fused-memory/tests/test_consolidation_closure_seam.py`, `fused-memory/tests/test_deferral_gate.py`,
  `fused-memory/tests/test_interceptor_amendment_advisory.py`, `fused-memory/tests/test_live_task_write_guard.py`,
  `fused-memory/tests/test_task_creation_journal.py`, `fused-memory/tests/test_task_interceptor.py`.
- Keys: `fk-85e1293e0aa4`. Deps: μ1; S26 γ.
- Deletes: the four setters, `_idempotency_replay`'s no-journal arm, each "None means byte-identical
  legacy behaviour" branch replaced by a required member or a named null object.
- Signal: boundary row 24 in CI; the fused-memory unit is active after deploy with one construction per
  arm; `git grep -nE "def set_(write_journal|lifecycle_reset_filer|audit_trail_archive|consolidation_scroll)"
  -- fused-memory/src/fused_memory/middleware/task_interceptor.py` is empty.
- G7: INV-10, INV-11.

**ν — `task_interceptor.py` partitioned** (priority medium, est. ~1,400 LOC moved, ~250 changed)
- Files: `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/middleware/status_transition.py` (new),
  `fused-memory/src/fused_memory/middleware/path_scope_routing.py` (new),
  `fused-memory/tests/test_task_interceptor.py`, `fused-memory/tests/test_status_transition.py` (new),
  `fused-memory/tests/test_path_scope_routing.py` (new).
- Keys: `fk-1eb04ceac9e8`, `fk-e3e99447e11e`. Deps: μ2 (6332, S5 μ upstream).
- Signal: both new modules import without importing `task_interceptor`; the commit message carries the
  partition table; the suite is green with moved tests driving the new objects' public methods.
- G7: INV-5; heuristics 13 and 14.

**ξ — Orchestrator task-status comparisons use `TaskStatus` members; the literal guard lands** (priority
medium, est. ~500 LOC)
- Files: the orchestrator modules under D13 (re-measure at decompose; at `e634cfe658` chiefly
  `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/workflow.py`,
  `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/chronic_flake.py`,
  `orchestrator/src/orchestrator/dependency_verdict.py`, `orchestrator/src/orchestrator/flake_ledger.py`,
  `orchestrator/src/orchestrator/provenance_conflict.py`, `orchestrator/src/orchestrator/task_status.py`
  (shim, deleted if no importer remains)), `shared/tests/test_task_status_literals.py` (new),
  `shared/tests/fixtures/task_status_literal_seeded.py` (new).
- Keys: contributes to `fk-1695d6b02400`. Deps: ε.
- Signal: boundary row 25 in CI over `SCAN_ROOTS = ('orchestrator/src',)`; the guard flags its seeded
  module.
- G7: INV-12, INV-10.

**ο — fused-memory and dashboard task-status comparisons use `TaskStatus` members** (priority medium,
est. ~450 LOC)
- Files: the fused-memory and dashboard modules under D13 (re-measure at decompose; at `e634cfe658` they include
  `fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py`,
  `fused-memory/src/fused_memory/reconciliation/blocked_gate_audit_section.py`,
  `fused-memory/src/fused_memory/reconciliation/task_filter.py`, `fused-memory/src/fused_memory/reconciliation/targeted.py`,
  `fused-memory/src/fused_memory/services/live_workflow_detector.py`, `dashboard/src/dashboard/data/task_snapshot.py`),
  `shared/src/shared/task_statuses.py` (docstring), `shared/tests/test_task_status_literals.py`. Above 12
  files the leaf splits by package at plan time.
- Keys: `fk-1695d6b02400`. Deps: ν, ξ.
- Signal: the guard's `SCAN_ROOTS` includes `fused-memory/src` and `dashboard/src` and passes the tree.
- G7: INV-12, INV-10.

**ω — Integration gate: the task-write contract end to end** (priority **high**, est. ~400 LOC)
- Files: `orchestrator/tests/test_task_write_contract_e2e.py` (new), `orchestrator/tests/_fused_memory_tool_bridge.py`
  (new; registered tools via `create_mcp_server` over the rig's interceptor and backend).
- Keys: none of its own (the B+H integration gate of every key above).
- Deps: β2, γ, ζ, κ2, θ, ο.
- Signal: boundary rows 1, 2, 5, 7, 12, 14, 19, 21 and 27 replayed through the real seams (orchestrator
  `Scheduler.update_task` → registered fused-memory `update_task` tool → interceptor → `SqliteTaskBackend`),
  on the `test_reopen_sticks_e2e.py` rig with its `dispatch_tool` bridge swapped for the tool bridge.
- G7: INV-13 (the readers prove their producer end to end), INV-10.

## Mechanical and live tasks (contract §8)

Step 1, re-run on 2026-11-10 (`find_tasks_by_metadata(key='x_finding_key', value=<key>)` for each of the
20 keys and the carried `fk-5c98cc0bb7e0`, every status) returns no task. Step 2 (`search_tasks` per
leaf and per new leaf, threshold 0.55, hits confirmed with `get_task`) found no new task at or above 0.6
carrying this work; a `files`-membership scan of the live tasks declaring this plan's hot files found
6653 (δ's work) and 8453 (ζ's write half). Dispositions are `lead_actions`, never written by this seat.

| Task | State (2026-11-10) | Relation | Disposition |
|---|---|---|---|
| 3150 | deferred low | Facade widening (`x_coalesced_into` 5310); carries `fk-5c98cc0bb7e0` by the report's filing table | cancel → β1 (β1 stamps the key) |
| 6129 | pending low | `_clear_merge_phase_entered` ignores the bool, mutates memory first | cancel → β2 |
| 6558 | **done** (2026-11-10, `found_on_main`: its cycle was already gone) | lower registry module with side-effect registration | none; no cancel needed |
| 3093 | pending low | deterministic invariants on `update_task` | cancel → γ |
| 7085 | pending low | interceptor discard helpers onto `metadata_dict` | cancel → γ |
| 6653 | pending low (new to this table) | one files extractor for `task_curator._task_files` and `TaskInterceptor._extract_meta_files_from_meta`; `backfill_corpus` survives malformed `metadata.files` | cancel → δ |
| 5310 | pending low | A: `delete_keys`; B: widen `SchedulerFacade` | amend to A; waits on ο, ξ |
| 3260 | pending high | (a) owned `files` field; (b) combine whole-blob replace — fixed by 3446 | amend (OQ-8 ruled); waits on ο, ξ |
| 6678 | pending low | `targeted.py` imports `_extract_metadata_files`/`_missing_files` privately | amend to `_missing_files`; waits on ο |
| 7748 | pending low | `lock_charter_guard._has_unrecognised_extension` beside `is_file_path` | waits on δ |
| 7173 | pending low | extract `update_task`'s write floors | waits on κ2, ι |
| 4867 | **deferred, held by Leo's hand carry** (2026-11-10; branch `task/4867`) | fieldless `update_task`; argument guard | prerequisite of α, as originally drafted (the inversion is void) |
| 6621 | pending medium | moves `_extract_metadata_dict` and siblings down | prerequisite of α |
| 5281 | pending medium | two merge-lane slices; write-gate delta rule; docs §8 | prerequisite of α |
| 6708 | **done** (2026-11-06) | S11 ζ, `task_interceptor.py` | none; drops from α's prerequisites |
| 8143 | **in progress** | cancel-with-children β, `_apply_status_transition` | prerequisite of α |
| 8453 | pending high (→ 8143 in progress; 8418, 8440, 8452 pending) | packing sweep P2: extracts the combine write half `_write_combined_rewrite`; registers `pack_record` | prerequisite of ζ; amend at decompose if η lands first (`pack_record` as a `SLICES` row) |
| 6719, 6720, 6721 | pending (→ 6718; → 6719; → 6720) | S14 ξ, ν, μ | prerequisites of α; amend 6719 (OQ-12 ruled) |
| 6766, 6724 | pending medium | S25 λ, S14 ο on orchestrator hot files | prerequisites of β1 |
| 6392 | **done** | three Tier-A keys in §8 | none; drops from π's prerequisites |
| 6332 | pending low (prerequisite 6708 done; plan authored 2026-11-07, so dispatchable now) | ticket-worker extraction | waits on ζ; prerequisite of μ1 — unless it is in progress or done at decompose, when ζ and μ1 depend on it instead (D15) |
| 8148, 8264 | pending | cancel-with-children η and cancel reasons, `_apply_status_transition` | ordered against κ1/ν at decompose (D15) |
| 8589 | pending low | schema ladder out of `sqlite_task_backend.py` | ordered against κ2 at decompose (D15) |

## Out of scope

- `BlockReason`, and narrowing the `except Exception` arms (S9 λ).
- The `delete_keys` primitive (5310 A) and `files` as an owned field (3260 (a)).
- The MCP argument guard (4867).
- The ticket-worker extraction (6332), the packing sweep (8453) and curator pipeline semantics (S11).
- Escalation-record and merge-outcome status literals (S2, S6).
- Narrowing any actor's transition set; retyping the 10 legacy operational normal tasks (the rule, not the
  data, decides).
- Backfilling or stripping legacy `FILES_TO_MODIFY:` rows (OQ-15 ruled (a)); merging that prefix into
  `files`.
- Renaming the curator's internal `files_to_modify` field (LLM schema, `CandidateTask`/`RewrittenTask`,
  vector-store payload, `search_tasks` output; ~207 test references): an optional low follow-up (OQ-9
  ruling), filed by the lead if wanted.
- Deleting the dead `files_to_modify` key from the 4 legacy rows (waits on 5310's delete-keys primitive).
- Filing the fix that σ's escalation may call for.
- Fused-memory's seven remaining `create_subprocess_exec` git sites (no S10 key).
- `targeted.task_interceptor`'s back-wire (TargetedReconciler's).

## Open questions (tactical)

1. **Interpreter and lane for θ's parity fixture.** **Suggested:** keep it in `shared/tests`; if the
   lane's interpreter lacks a member's third-party dependency, move the file to `fused-memory/tests` in
   the same leaf and record why. Decide in θ.
2. **`WriteModeError` base class.** **Suggested:** `ValueError` with `to_error_dict()` and a declared
   `McpErrorType`. Decide in α.
3. **Where moved measurements and rejected alternatives land** — the commit message by default. Decide
   per leaf.
4. **`text_mode` on `prompt`.** **Suggested:** applies exactly as to `details`. Decide in α.
