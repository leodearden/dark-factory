# Capability manifest — S10 task-write contract: metadata modes, success signal and registry

PRD: `plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.md`.
YAML twin: `plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.capability-manifest.yaml`
(every `task_id: null` until `commit_planning` stamps it; σ's block is stamped when the lead files σ;
`signal_leaves: [ω, ι, ρ]` — the integration gate and the two leaves no in-batch leaf depends on).

**Code anchors** verified against main `e634cfe658` (2026-11-10). Main moves fast — cite-by-symbol;
re-locate lines at implementation time. Bindings cite symbols, not lines. Rulings of the 2026-11-10
sitting applied (PRD `## Rulings`). Re-checked at main `bb1c659562` (fold-verify, 2026-11-10): the window
since `e634cfe658` changes none of the bound symbols; the sidecar loads through `CapabilityManifestDoc`
(22 blocks), all 63 mechanical checks still FAIL, and `lint_delivered_checks` with each leaf's declared
files reports nothing.
Polarity: every mechanical `delivered_check` in the sidecar (63) was evaluated at `e634cfe658` with
`shared/src/shared/delivered_check_polarity.py::evaluate_grep_at_tree` / `::evaluate_path_at_tree` and
FAILS there (producer checks); `lint_delivered_checks` with each leaf's declared files reports nothing.
Substrate rows are `manual` (a present-check on existing substrate is vacuous and the gate rejects it).

## α — Two typed write axes on the wire; the backend resolves once

| Capability | Binding | Verdict |
|---|---|---|
| The orchestrator wire already sends an explicit mode and never `append` | `scheduler.py::Scheduler.update_task` forwards `metadata_mode` always, never `append` | PASS |
| An undeclared MCP argument is loud before α lands | task 4867 (hand-carried, ruled 2026-11-10) is α's prerequisite; `append` stays declared only for its targeted `repaired_call` (D1a) | PASS |
| Every caller that must state a mode or stop sending `append` | AST of `update_task(` calls plus the scripts naming the `'update_task'` tool: `targeted.py::_sweep_block_orphan`, `TaskInterceptor._dispatch_ticket_decision`, `manifest_stamping.py::_stamp_capability_manifests_impl`, `scripts/sitting/payloads.py::append_details_payload`; the ten other payload builders already send `metadata_mode` | PASS |
| LLM-facing instructions | Stage 2 system prompt `prompts/stage2.py`, user prompt `TaskKnowledgeSync.assemble_payload`, `reconciliation/enumeration_guard.py`'s rejection text and guidance (new at `e634cfe658`); `recurring_gate_guard.recurring_gate_error`'s hint; `AppendUnsupportedFieldError`'s message — all in α | PASS (α delivers) |
| Test blast radius declared | 17 fused-memory test files (AST mode-less metadata writes + `append=`) and `scripts/tests/test_sitting_payloads.py` | PASS |
| Upstream prerequisites | 4867, 6621, 5281, 6719, 6720, 6721, 8143, S5 μ — open on 2026-11-10 (6708 done) | PASS |
| `write_modes` exists — `write-modes-resolver-exists` | producer: `def write_modes` in `shared/src/shared/task_metadata_wire.py` | PASS (α delivers) |
| Backend resolver gone — `backend-mode-resolver-removed` | anti-inversion over `fused-memory/src/` | PASS (α delivers) |
| No `append=True` left — `fused-memory-src-off-append` | anti-inversion: `append['"]?(=\|: *)(True\|true)` absent from `fused-memory/src/`, `scripts/sitting/` (66 + 1 hits today in 9 + 1 files, all in α's files) | PASS (α delivers) |
| Rejection on the live tool | produced by α; rows 1–2 and a live probe (`manual`) | PASS (α delivers) |

## β1 — Client signature: `metadata_mode` required, `append` gone, facade and fakes in parity

| Capability | Binding | Verdict |
|---|---|---|
| Call sites and fakes enumerated | AST at `e634cfe658`: 43 orchestrator + 1 escalation calls (15 mode-less, 2 `append=True`, 1 `append=not overwrite`); `scheduled_deploy_settle.TaskStore` declares its own `update_task`; test union per the PRD list | PASS |
| Carried key | `fk-5c98cc0bb7e0` (task 3150, cancelled into β1) | PASS |
| Resolver gone — `client-side-resolver-removed` | anti-inversion over `orchestrator/src/`, `orchestrator/tests/` | PASS (β1 delivers) |
| Suppressions gone — `facade-suppressions-removed` | anti-inversion: 5 `reportCallIssue` in `workflow.py` today | PASS (β1 delivers) |

## β2 — `Scheduler.update_task` raises `MetadataWriteFailed`; nothing reads its result

| Capability | Binding | Verdict |
|---|---|---|
| Reply predicate to raise on | `shared.mcp_envelope.write_failure_reason` over `::parse_tool_result(result, None, dict)` (task 6711 done) | PASS |
| Typed-raise precedent | `scheduler.py::SetTaskStatusRejected` / `TerminalExitRejection` | PASS |
| The type checker does not catch a surviving reader | pyright 1.1.408 reports no error for a bool read of a `-> None` coroutine (default and strict) — hence the guard | PASS (measured absence) |
| Every reader has a disposition | 12 readers, 14 sites outside `try`, 30 inside at `e634cfe658` (incl. `_write_branch_base_sha` and `scheduled_deploy_settle._write`, new); table in §Sketch β2; re-measured at decompose | PASS |
| `MetadataWriteFailed` exists — `metadata-write-failed-exists` | producer in `orchestrator/src/orchestrator/task_write.py` (OQ-2 ruled) | PASS (β2 delivers) |
| No `False` double — `false-return-doubles-removed` | anti-inversion: `update_task(\.return_value *= *False\| *= *AsyncMock\(return_value=False)` | PASS (β2 delivers) |
| Result never read — `update-task-result-never-read` | producer: path `orchestrator/tests/test_update_task_result_never_read.py`; rule: every `update_task(…)` Call is the direct operand of an `Await` whose parent is an `ast.Expr`; fixture covers a bool read and `asyncio.gather` (OQ-13 ruled (a)) | PASS (β2 delivers) |
| The documented arm fires | produced by β2; row 7 (`manual`) | PASS (β2 delivers) |

## γ — One parse per request, every rejection a dict through one renderer, invariants on update

| Capability | Binding | Verdict |
|---|---|---|
| One loud reader to build on | `metadata_dict.py::raw_metadata_dict` | PASS |
| One error renderer | S14 ξ (task 6719): `McpErrorType` and `error_envelope(exc)`; γ uses `exc.to_error_dict()` else `error_envelope` | PASS (6719 upstream of α) |
| Renderer homed below `middleware` (R17 DAG) | OQ-12 ruled: `error_envelope` in `shared.mcp_envelope`, 6719 amended; `git grep "from fused_memory.server" -- fused-memory/src/fused_memory/middleware/` empty at `e634cfe658` | PASS (6719 delivers) |
| Deterministic gate to run on update | `deterministic_task_guard.py::deterministic_task_error`; script clause only on `task_kind`/`before_done` deltas | PASS |
| Census emitter copy gone — `discard-helper-copy-removed` | anti-inversion | PASS (γ delivers) |
| Interceptor parser gone — `interceptor-parse-helper-removed` | anti-inversion | PASS (γ delivers) |
| One success predicate — `csv-success-predicate-unified` | anti-inversion; the `_consolidation_not_closed_error` docstring is rewritten too | PASS (γ delivers) |
| Deterministic invariants on update — `deterministic-invariants-on-update` | producer: `deterministic_task_error(` called in `task_interceptor.py` | PASS (γ delivers) |

## π — Prose callers of `update_task` state the two axes

| Capability | Binding | Verdict |
|---|---|---|
| Prose callers located | `git grep -nE "append=(True\|true)" -- skills docs/task-authoring.md`: all in π's files (`skills/escalation-watcher-auto/SKILL.md` new) | PASS |
| Same-file prerequisite | task 6392 is done; π heads the `docs/task-authoring.md` chain | PASS |
| Prose off `append` — `skills-off-append` | anti-inversion | PASS (π delivers) |
| Reference off the retired key — `reference-off-files-to-modify` | anti-inversion: `files_to_modify` absent from `docs/task-authoring.md` (3 mentions today; OQ-9 ruled) | PASS (π delivers) |

## δ — `shared.declared_files`: one key, one reader; `files_to_modify` refused and unread

| Capability | Binding | Verdict |
|---|---|---|
| Canonical lock-charter copy | `shared/src/shared/locking.py::FILE_EXTENSIONS`, `::is_file_path`, `::directory_locks`, `::strip_directory_locks` | PASS |
| The retired key is nearly extinct | all 11 stores, 2026-11-10: 4 rows carry `metadata.files_to_modify`, all done; 3 hold paths absent from `files` (dark-factory 2507, reify 4344, 4865); 0 non-terminal | PASS |
| Read positions located | 9 `*/src` sites hold the literal in a read position: 4 metadata reads (deleted), 5 curator-field reads (onto `curator_types.CURATOR_FILES_FIELD`) | PASS |
| Reader exists — `declared-files-reader-exists` | producer | PASS (δ delivers) |
| Refusal exists — `retired-files-key-refusal-exists` | producer: `class FilesToModifyRetiredError` in `shared/src/shared/declared_files.py` | PASS (δ delivers) |
| Refusal wired — `retired-files-key-refusal-wired` | producer: `refuse_retired_files_key(` called under `fused-memory/src/fused_memory/middleware/` | PASS (δ delivers) |
| No reader — `retired-files-key-reader-guard-exists` | producer: path `shared/tests/test_retired_files_key_unread.py` (seeded fixture; no exemption entries) | PASS (δ delivers) |
| Store reader gone — `store-key-reader-removed` | anti-inversion | PASS (δ delivers) |
| Key tuple gone — `interceptor-key-tuple-removed` | anti-inversion | PASS (δ delivers) |
| Curator metadata arm gone — `curator-metadata-arm-removed` | anti-inversion (task 6653's half) | PASS (δ delivers) |
| Lock-charter copy gone — `lock-charter-copy-removed` | anti-inversion | PASS (δ delivers) |
| Backfill after deploy | lead action: re-measure all stores; `update_task(metadata={'files': union}, metadata_mode='merge')` per differing row | PASS (lead) |

## ε — Orchestrator reads declared files through `shared.declared_files`; one sanitizer

| Capability | Binding | Verdict |
|---|---|---|
| One sanitizer, no alias (D5) | `shared.locking.strip_directory_locks`; `module_charter.sanitize_files_for_persist` is a one-line alias of it | PASS |
| A live legacy-key dispatch never occurs | 0 non-terminal rows carry `files_to_modify` — so the signal is a CI boundary test | PASS |
| Alias gone — `orchestrator-sanitizer-removed` | anti-inversion over `orchestrator/src/`, `scripts/` | PASS (ε delivers) |
| Boundary test — `lock-derivation-boundary-test` | producer: path `orchestrator/tests/test_lock_derivation_declared_files.py` | PASS (ε delivers) |

## ζ — The combine writes a union it computes into `metadata.files`; a collision is a counted refusal

| Capability | Binding | Verdict |
|---|---|---|
| The curator's list is not a union | `RewrittenTask.files_to_modify` is the LLM output; store: 200 of 428 prefixed rows with non-empty files hold files absent from the curator's list | PASS (measured; ζ computes the union in code) |
| The write half to edit | task 8453 extracts `_write_combined_rewrite` from `_execute_combine` (pending high; ζ depends on it) | PASS (8453 upstream) |
| Collision handling | audited refusal with a distinct token (OQ-14 ruled); the audit records no files and no survivor today; its four tokens are `COMBINE_REFUSED_*` constants in `task_interceptor.py` | PASS (ζ adds them, in C7) |
| Prefix gone — `details-prefix-removed` | anti-inversion | PASS (ζ delivers) |
| Token exists — `collision-refusal-token-exists` | producer: `CANDIDATE_KEY_COLLISION` in `fused-memory/src/fused_memory/middleware/combine_audit.py` | PASS (ζ delivers) |
| A live combine never shrinks `files` (INV-13) | live read after merge against the audit's new `old.files` (`manual`) | PASS (ζ delivers) |

## ρ — Combine-collision census script

| Capability | Binding | Verdict |
|---|---|---|
| One audit file for every project | `task_interceptor.py::_combine_audit_path` (moved to C7 by ζ); 214 combine rows in the first 10 days of November across all projects | PASS |
| Script exists — `census-script-exists` | producer: path `scripts/combine_collision_census.py` | PASS (ρ delivers) |
| Reads the producer's token — `census-reads-the-producer-token` | producer: `CombineRefusal` in the script | PASS (ρ delivers) |

## σ — Deterministic 30-day combine-collision check (lead-filed after ρ is on main)

| Capability | Binding | Verdict |
|---|---|---|
| A deterministic task's script exists at submit | `submit_task` validates `before_done.script`; hence lead-filed after ρ lands | PASS |
| Delayed timer anchors on dependencies done | `docs/task-authoring.md` §6 frozen-once anchor | PASS |
| Collision-rate verdict | its own predicate at the milestone (`manual`) | PASS (σ delivers) |

## η — The slice table: static, downward, one home

| Capability | Binding | Verdict |
|---|---|---|
| Downward-import precedent | `CancelDispositionEntry`, `DeclaredScope`, `StrandConverted`, `DeferralRecord` imported and registered by `task_metadata.py` | PASS |
| Leaf modules need not import `task_metadata` | `deploy_state.py`, `defects.py`, `followup_items.py` import only the register call; `capability_manifest.py` also `ExternalDep` (moves to `shared/external_dep.py`) | PASS |
| Importers of moved names located | `capability_manifest`'s `DeliveredCheckMeta`/`MECHANICAL_CHECK_KINDS`: `orchestrator/delivered_checks.py`, `manifest_stamping.py`, two scripts, `orchestrator/tests/test_briefing.py`, `fused-memory/tests/server/test_write_triage_flip_gate_invariants.py`, `shared/tests/test_delivered_check_polarity.py`; `CHECK_SUBJECT_FIELD` stays | PASS |
| Mutator gone — `registry-mutator-removed` | anti-inversion | PASS (η delivers) |
| Read view exists — `slice-table-read-view-exists` | producer | PASS (η delivers) |
| ExternalDep leaf — `external-dep-leaf-exists` | producer: path | PASS (η delivers) |
| Chain pin replaced — `registration-chain-test-removed` | anti-inversion: path absent | PASS (η delivers) |

## θ — INV-16 `metadata-slices-one-table` (number per the lead's revised chain)

| Capability | Binding | Verdict |
|---|---|---|
| Lockstep guard | `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES` (five registered sites; INV-14 last at `e634cfe658`) | PASS |
| Member binding for the child | `scripts/_checkout.py`'s import-time bind and `MEMBER_SRC` (S25 C1; `manual` substrate) | PASS |
| Probe works for R3's five | registries today 9/12/11/9/10 | PASS |
| Number fixed without a race | OQ-5 revised by the lead 2026-11-10: S6 α `lane-budgets-exceed-verify-timeouts` (INV-15) → S10 θ (INV-16) → S7 ζ `merge-gate-unmeasured-typed` (INV-17); slug-anchored checks | PASS (lead wires at decompose) |
| Registered — `invariant-registered` | producer: slug in `design-invariants.md` | PASS (θ delivers) |
| Fixture — `parity-fixture-exists` | producer: path | PASS (θ delivers) |
| Fixture flags a registrant | produced by θ (`manual`) | PASS (θ delivers) |

## ι — The blessed-key table in docs §8

| Capability | Binding | Verdict |
|---|---|---|
| Marker readers located | `tests/scripts/test_task_authoring_blessed_keys_drift.py`, `test_task_authoring_tier_b_canonical_keys.py`, `task_authoring_marker_span.py` | PASS |
| Same-file prerequisites | η (`task_metadata.py`), π (`docs/task-authoring.md`) | PASS |
| Table exists — `docs-table-markers-exist` | producer | PASS (ι delivers) |
| Mirror replaced — `mirror-block-replaced` | anti-inversion over the doc and `tests/scripts/` | PASS (ι delivers) |
| Comment density ≤ 1 line per key | `manual` | PASS (ι delivers) |

## κ1 — One transition verdict, enforced in the backend only

| Capability | Binding | Verdict |
|---|---|---|
| One enforce switch | `TaskStatusConfig.enforce_transitions` (default `False`, deployed `true`, restart-only) | PASS |
| One backend tail for both status writers | `SqliteTaskBackend._write_status_and_verify` | PASS |
| Every server status write reaches the backend through one interceptor site | `task_interceptor.py::_status_write_body`; other non-test callers: `audit_duplicate_tasks.py`, `correct_found_on_main_backlog.py` | PASS |
| Construction sites to thread the config through | `server/main.py::_build_task_backend` + 11 script/maintenance constructions pass the loaded config; the keyword defaults to `TaskStatusConfig()`, so the 131 test constructions in 46 files are unchanged; a κ1 test pins `_build_task_backend` | PASS |
| Callers that state `actor` | 67 direct backend status-writer calls in 8 test files; the two scripts (`ActorClass.HUMAN`); the interceptor (`parse_agent_identity(…).actor_class`, S14 μ upstream of α) | PASS |
| Verdict exists — `transition-verdict-exists` | producer | PASS (κ1 delivers) |
| Refusal type — `illegal-transition-error-exists` | producer: `class IllegalTransitionError` | PASS (κ1 delivers) |
| Backend applies the verdict — `backend-applies-transition-verdict` | producer: `transition_verdict(` in `sqlite_task_backend.py` | PASS (κ1 delivers) |
| `actor` required — `backend-status-writers-require-actor` | producer: `actor: ActorClass[,)]` in `sqlite_task_backend.py` | PASS (κ1 delivers) |
| No interceptor legality gate — `interceptor-holds-no-transition-gate` | anti-inversion over `fused-memory/src/fused_memory/middleware/` | PASS (κ1 delivers) |

## κ2 — Privileged-writer protocol, census failure, candidate-key scope

| Capability | Binding | Verdict |
|---|---|---|
| Collision classification without message text | `sqlite3.IntegrityError.sqlite_errorname == 'SQLITE_CONSTRAINT_UNIQUE'` | PASS |
| Live index WHERE is readable | `sqlite_master.sql` for `ux_tasks_candidate_key` | PASS |
| Protocol — `privileged-writer-protocol-exists` | producer | PASS (κ2 delivers) |
| Suppressions gone — `attr-defined-suppressions-removed` | anti-inversion | PASS (κ2 delivers) |
| Sniff gone — `integrity-error-substring-sniff-removed` | anti-inversion | PASS (κ2 delivers) |
| Census failure typed — `census-failure-type-exists` | producer | PASS (κ2 delivers) |
| Scope constant — `scope-constant-exists` | producer | PASS (κ2 delivers) |

## λ — One gate rule: deterministic and `always_escalates`

| Capability | Binding | Verdict |
|---|---|---|
| The rule matches dispatch | `deterministic_runner.py` hands a task to a human iff deterministic and `always_escalates` (`docs/task-authoring.md` §5) | PASS |
| The submission coercion is pure | `operational_routing_guard.py::inject_operational_routing` returns a fresh dict on the coercion branch | PASS |
| The consolidation key can live in `shared` | `consolidation_gate.GATE_METADATA_KEY == 'x_recon_consolidation_gate'` (`EXPERIMENTAL_KEY_PREFIX` + suffix); λ moves the string down (re-spelling `x_` deliberately), fused-memory re-binds it, and λ's test asserts equality with the derived spelling | PASS |
| No row moves for the closure readers | all 61 block-carrying rows in the 11 stores are human gates | PASS |
| Declaration — `gate-declaration-exists` | producer | PASS (λ delivers) |
| The rule — `one-gate-rule-exists` | producer: `def is_human_gate` in `gate_declaration.py` | PASS (λ delivers) |
| Private predicate gone — `private-gate-predicate-removed` | anti-inversion | PASS (λ delivers) |
| Detector copy gone — `detector-pure-gate-copy-removed` | anti-inversion | PASS (λ delivers) |
| Dashboard copy gone — `dashboard-gate-copy-removed` | anti-inversion | PASS (λ delivers) |
| Recon duplicate-gate refusals counted for a week | `manual` (OQ-10 ruling's live signal) | PASS (λ delivers) |

## μ1 — One interceptor test factory

| Capability | Binding | Verdict |
|---|---|---|
| Construction sites | 247 `TaskInterceptor(` lines in 49 test files, two shared stacks included | PASS |
| Factory — `interceptor-test-factory-exists` | producer: path | PASS (μ1 delivers) |

## μ2 — Collaborators at construction

| Capability | Binding | Verdict |
|---|---|---|
| Journal, archive, scroll exist before construction | `run_server` builds `MemoryService(config)` first and wires them right after construction | PASS |
| The one late member | lifecycle-reset filer needs `ReconReportState` built from the interceptor — provider member (OQ-3 ruled); after S26 γ | PASS |
| Setter callers in tests | 9 test files call the setters on an interceptor | PASS |
| Setters gone — `post-construction-setters-removed` | anti-inversion: `def set_(…)\b` absent from `task_interceptor.py` | PASS (μ2 delivers) |
| Bundle — `collaborator-bundle-exists` | producer | PASS (μ2 delivers) |

## ν — The partition

| Capability | Binding | Verdict |
|---|---|---|
| Disjoint attribute families | AST: 102 methods, 33 attributes, 5,776 class lines | PASS |
| Modules exist | producer: two paths | PASS (ν delivers) |
| Gate chain left the file — `gate-chain-left-the-interceptor` | anti-inversion | PASS (ν delivers) |

## ξ — Orchestrator `TaskStatus` members; the guard lands

| Capability | Binding | Verdict |
|---|---|---|
| Vocabulary | `shared/task_statuses.py::TaskStatus` | PASS |
| Guard is ξ's own (no downstream producer) | `literal-guard-exists`: path `shared/tests/test_task_status_literals.py` | PASS (ξ delivers) |
| Cited example converted — `harness-merge-deferred-literal-removed` | anti-inversion over `orchestrator/src/` | PASS (ξ delivers) |

## ο — fused-memory and dashboard members

| Capability | Binding | Verdict |
|---|---|---|
| Guard widened — `guard-scans-fused-memory` | producer: `SCAN_ROOTS *=.*'fused-memory/src'` (one-line tuple; a docstring mention cannot satisfy it) | PASS (ο delivers) |
| Representative literal — `stall-detector-literal-removed` | anti-inversion over `fused-memory/src/` | PASS (ο delivers) |
| Guard flags a seeded literal | `manual` | PASS (ο delivers) |

## ω — Integration gate

| Capability | Binding | Verdict |
|---|---|---|
| Real rig to reuse | `orchestrator/tests/test_reopen_sticks_e2e.py::_Rig` (`manual` substrate); its `_bridge_dispatch_tool` accepts only `set_task_status` and skips the tool layer | PASS |
| Tool bridge — `tool-bridge-exists` | producer: path `orchestrator/tests/_fused_memory_tool_bridge.py` (registered tools via `server/tools.py::create_mcp_server`, as `fused-memory/tests/test_task_tools.py` builds them) | PASS (ω delivers) |
| Suite — `contract-e2e-suite-exists` | producer: path `orchestrator/tests/test_task_write_contract_e2e.py` | PASS (ω delivers) |
| Contract holds end to end | rows 1, 2, 5, 7, 12, 14, 19, 21, 27 through real seams (`manual`) | PASS (ω delivers) |
