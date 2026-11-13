# Capability manifest — S19 package layering and private seams

PRD: `plans/review-all-s19-package-layering-and-private-seams-prd.md`.
YAML twin: `plans/review-all-s19-package-layering-and-private-seams-prd.capability-manifest.yaml`.
The twin names signal leaves ε (fused-memory cluster), θ2 (session cluster), ι (CLI runner) and κ2
(dashboard). Every `task_id` stays `null` until decompose. Gate λ (Leo's addition to S19-2, ruled
2026-11-12) has a block with no mechanical check.

**Code anchors** re-verified against main `1b106418da` (2026-11-12) by the ruling fold. Bindings cite
symbols, not lines. The run pinned `6551b75aba` (2026-10-08); the author anchored at `1f240e6da2` and the
critique folds at `97be1d71fa` and `7baa5b997c` (2026-11-11).

Polarity was re-measured at `1b106418da` on 2026-11-12 with
`shared/src/shared/delivered_check_polarity.py::evaluate_grep_at_tree` and `::evaluate_path_at_tree`
against `main`, and again at `7848173a7d` by the fold-verifier (same result). The sidecar loads through
`shared/src/shared/capability_manifest.py::CapabilityManifestDoc`. All 40 mechanical checks (38, plus ν's two added by the second fold-verify) fail at
today's tree, as a producer check must (δ's `storm-counter-shim-deleted` path check was dropped: task
6547 deleted the shim, so it passed vacuously):

- each `expect: present` check finds no match, or no path;
- each `expect: absent` check finds a match, or a path.

`lint_delivered_checks`, given each leaf's folded declared files, reports `absent_overbroad` on four
checks. Each is benign: a dependency moves the matched file or symbol into the leaf's declared paths
before the leaf lands.

- β1 `policy-dotted-paths-gone`: `reconciliation/recon_ledger.py`, which α moves to
  `recon_state/recon_ledger.py` (declared by β1);
- β2 `contract-dotted-paths-gone`: `middleware/dependency_direction_check.py`, which γ moves to
  `recon_state/` (declared by β2);
- β2 `middleware-imports-no-reconciliation`: α, β1 and γ remove the other matches first;
- θ2 `archive-layout-mirror-gone`: `session_registry.py::ESCALATION_ARCHIVE_SUBDIR`, which 6179 and θ1
  bring into `shared/src/shared/session/decisions.py` (declared by θ2, and on the check's paths).

Five rows are substrate: they pass today and carry no check, so none is copied to a task (α's write
journal, δ's deleted shim and child vocabulary, η1's run-by-path contract, λ's capstone gate). One row
names a producer edge with no check (λ on θ2). Two rows are `manual`: β2's ordering confirmation and λ's
session ruling. One row is `OPEN`, and it carries a check: β2's ordering row (manual). θ2's
archive-layout row is PASS now that S19/OQ-3 is ruled (a).

## α — Create `fused_memory.recon_state`; move the event and journal stores

| Capability | Binding | Verdict |
|---|---|---|
| The nine stores import only lower units once moved | their outgoing edges at `1b106418da` (unchanged since `1f240e6da2`): `models`, `utils`, `memory_metadata`, each other; plus `journal → services.write_journal` (TYPE_CHECKING, cut by a Protocol) and `audit_trail_rotation → context_assembler.HINT_QUERIES_EXECUTED` (the constant moves down). One exception until β1: `recon_ledger` imports `reconciliation.standing_decision_constants` at module level, so B3 holds for every α module but `recon_ledger` until β1 moves that module | PASS |
| Every importer is enumerable | 148 tracked files named them at `1f240e6da2` (src 34, tests 107, live docs/config 3), excluding dated snapshots and data fixtures; 146 at `1b106418da` and `7848173a7d`, 147 at `565d15a06f`, all declared but prose and fixtures (D6) | PASS |
| `throughput`'s `-m` entry has two readers | `fused-memory/config/config.yaml` (comment) and `tests/test_config_schema.py` | PASS |
| Producer checks | path `recon_state/journal.py` present; the nine dotted reconciliation paths absent from `fused-memory/src` | PASS |
| `WriteJournal` exists for the Protocol to describe | `services/write_journal.py::WriteJournal` (substrate) | PASS |

## β1 — Move the self-model, pool map and backlog/bulk-reset policies; `orchestrator_detector` to utils

| Capability | Binding | Verdict |
|---|---|---|
| The six modules plus `orchestrator_detector` import only lower units | measured: `graphiti_degradation_probe` and `orchestrator_detector` import nothing first-party; `backlog_policy`'s only upward edge is `services.orchestrator_detector`, which moves with it | PASS |
| `orchestrator_detector`'s body is owned by task 6615 | 6615 is done (`f750373949`): `orchestrator_detector` imports `shared.proc_liveness.pid_alive`; β1's edge onto 6615 is satisfied | PASS |
| β1 needs no hot orchestrator file | 6615 deleted `harness._pid_alive` (whose docstring was `harness.py`'s only hit) and `orchestrator/tests/test_pid_alive_contract.py`; neither matches at `1b106418da` | PASS |
| Producer checks | path `recon_state/recon_self_model.py` present; path `services/orchestrator_detector.py` absent; six dotted paths absent | PASS |

## β2 — Move `task_filter`, `consolidation_gate` and the flag/standing-decision contracts

| Capability | Binding | Verdict |
|---|---|---|
| The five modules import only lower units | measured at `1b106418da`: no outgoing edge above `utils`/`models`/`memory_metadata` except `standing_decision_constants`, which β1 moves into recon_state first | PASS |
| Program tasks editing these bodies land first | 6599, 6721, 9203, 9240, 9254 (none waits on 5040 or 8528; 6621 done on 2026-11-12); S13 δ2, which splits out `recon_prose_grammar.py` (held on S13/δ2; S19/Q-C5 (a)); 9285 waits on 8528 and depends on β2 instead (D8, S19/OQ-5 (a)) | OPEN (manual check) |
| `recon_prose_grammar` belongs in recon_state | S13's draft: middleware (`dependency_direction_check`) and services (`completion_claim_gate`) import it, so C2's membership rule applies; it does not exist at `1b106418da` | PASS (edge) |
| Producer checks | path `recon_state/task_filter.py` present; five dotted paths absent; no `from fused_memory.reconciliation` in `middleware/` (the last of the 16 middleware→reconciliation statements goes here; γ lands before β2, so the check lives on β2) | PASS |

## γ — `fused_memory.escalators` (S19/OQ-4 (b)); the remaining middleware and backends upward edges

| Capability | Binding | Verdict |
|---|---|---|
| The five escalators-bound filers import no other fused_memory unit | measured at `1b106418da`: `_folded_escalation`, `entity_mint_storm_escalator` and `mem0_update_storm_escalator` import none; `referent_repair_storm_escalator` and `candidate_key_escalation` import only `_folded_escalation` | PASS |
| `dead_letter_escalator` belongs in services | since task 9143 (`204de188c4`) it imports `services.durable_queue.POST_EXECUTE_DEAD_PREFIX` and `services.queued_writes.QueueOperation` at module level, and only `services.memory_service` imports it (measured at `1b106418da`); γ moves it to `services/` (D5) | PASS |
| `candidate_key` is a leaf | 53 lines, no first-party import | PASS |
| `recon_archive_root` and `estimate_tokens` are separable | `cli_stage_runner.py::recon_archive_root` reads only `ReconciliationConfig`; `context_assembler.py::estimate_tokens` is pure | PASS |
| Their importers are enumerable | `recon_archive_root`: `task_curator`, `reconciliation/harness.py`, `stages/base.py` and 4 test files; `estimate_tokens`: `task_curator`, `context_assembler`'s test, the bake-off script and its test (function-local), measured at `1b106418da`; all declared by γ (other mentions are prose, D6) | PASS |
| Producer checks | path `escalators/folded_escalation.py` present; no `from fused_memory.middleware` in `services/` or `backends/` | PASS |

## δ — Demote server/

| Capability | Binding | Verdict |
|---|---|---|
| `consolidation.py` and `mem0_update_authz.py` need nothing from server | measured: `consolidation` imports `memory_metadata`, `models.enums`, `topic_slug`, `utils.validation` and (TYPE_CHECKING) `config.schema`, all below services; `mem0_update_authz` imports nothing first-party | PASS |
| `grouped_read` needs only lower units after S14 ν | its imports are `backends`, `config`, `memory_metadata`, `models`, `services.memory_service`, `utils`; the private `_MEM0_CONTENT_KEYS` becomes public at its owner | PASS |
| The storm_counter shim is gone already (substrate) | task 6547 deleted `server/storm_counter.py` and its test (`0f86a2cdcc`); absent at `1b106418da`, so δ does not touch it | PASS |
| The deferred `prompts → server.recon_report` import is not δ's | S12 ι1 owns the cut (S19/Q-C4 (a)); none of `server/main.py`, `server/recon_report.py` and `reconciliation/prompts/__init__.py` references a module δ moves (measured at `1b106418da`), so δ does not declare them | PASS |
| The child vocabulary lands first | S14 ν, task 6720 (substrate via dependency) | PASS (edge) |
| Producer checks | path `services/grouped_read.py` present; no import of `fused_memory.server.{grouped_read,consolidation,mem0_update_authz}` anywhere in `fused-memory/src` (13 statements in 9 files at `1b106418da`) | PASS |

## ε — Invariant `fused-memory-subpackage-dag` (integration gate)

| Capability | Binding | Verdict |
|---|---|---|
| The import classifier exists | `scripts/source_measures.py::import_statements_in_tree` classifies `runtime`/`deferred`; `orchestrator/tests/test_merge_lane_facade_names.py` shows the `sys.path` binding for a test outside `scripts/` | PASS |
| The snapshot's edge builder and SCC can be promoted | `scripts/quality_metrics_snapshot.py::_targets` (with `_from_targets`, `_from_module`, `_from_base`) and `::_cycles` (Kosaraju) are module-private; no test or script outside the snapshot calls them by name (`quality_snapshot_cadence.py` and the two snapshot tests use other names), measured at `97be1d71fa`; neither file has changed since (`1b106418da`) | PASS |
| The order is consistent with the tree after α–δ and S12 ι1 | every one of the 48 upward statements (the same 48 at `1b106418da`) has a cut in α, β1, β2, γ, δ, task 6720 or S12 ι1 (PRD §Module placement; ε waits on S12 ι1, S19/Q-C4 (a)) | PASS (edge) |
| The four-site lockstep sites | `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES` (five entries: the fifth, `orchestrator/src/orchestrator/agents/code_quality.md`, is a partial mapping that needs no change) | PASS |
| Producer checks | `LAYERS` declaration present; path `fused-memory/tests/test_package_layering.py` present; `def import_targets(` or `def strongly_connected_components(` present in `scripts/source_measures.py`; `## INV-<n> \`fused-memory-subpackage-dag\`` present | PASS |

## ζ — `shared.sandbox`

| Capability | Binding | Verdict |
|---|---|---|
| The moved modules are stdlib-only | `landlock.py`, `landlock_exec.py`, `sandbox.py` import only the stdlib (measured) | PASS |
| `landlock_exec` runs as a script next to its builder | `landlock.py::_LANDLOCK_EXEC = Path(__file__).with_name('landlock_exec.py')`; the pair moves together | PASS |
| Fail-closed semantics are preserved on both sides | `sandbox_dispatch.py::resolve_backend_or_refuse` and `sandbox_guard.py::RemediationSandboxUnavailable` stay with their owners (D11) | PASS |
| Producer checks | no `from orchestrator.` in `fused-memory/src`; `def resolve_available_backend(` present in `shared/src`; path `orchestrator/src/orchestrator/agents/landlock.py` absent | PASS |

## η1 / η2 — Split `session_registry.py` in place

| Capability | Binding | Verdict |
|---|---|---|
| The run-by-path contract can be kept | `session_registry.py` appends this checkout's `shared/src` and is run by absolute path (substrate); 6179's dual-mode sibling import satisfies `TestStdlibOnlySelfContainment`'s tier-1 rule | PASS |
| The partition is a DAG | the prior h6 finding's AST count: decisions → records/io; leases → records/io/reaper; wm → records/reaper/io; CLI → all; no reach-back | PASS |
| The window probe can be injected | `session_hooks` entry points already take `process_tree`; `windows` joins it (D10) | PASS |
| The window sweep keeps `windows` a leaf | `session_registry.py::mark_windowless_wm_sessions_exited` calls `_mark_exited_if_still_non_terminal`, the reaper's; with `_wm_window_id_of` and `_names_window` it moves to `session_reaper` (reaper → windows) | PASS |
| `service_restart.py` needs no session edit | 6615 (done, `f750373949`) replaced its `session_registry._pid_alive` import with `shared.proc_liveness.pid_alive`; 0 `session_registry` mentions at `1b106418da` | PASS |
| Producer checks | `class SessionRecord` absent from `session_registry.py` (η1); `def claim_lease(` absent from it (η2); no `_wmctrl_list` / `_normalize_wm_window_id` / `_canonical_window_id` definitions in `orchestrator/src`; no `_resolve_wm_window_id` under `orchestrator/` | PASS |

## θ0 — `shared.escalation_layout`

| Capability | Binding | Verdict |
|---|---|---|
| The layout and the status names are stdlib data | `escalation/src/escalation/archive.py::ARCHIVE_SUBDIR`, `DATE_FORMAT`, `archive_dir_for_date` import only the stdlib; `'resolved'` and `'dismissed'` are the terminal values `session_registry.py::DECISION_CLOSE_MAP` keys on | PASS |
| One owner | the G4 table gives S19 the shared modules it names; θ0 creates the whole module, θ2 adds `locate_record`, and S2 α consumes it (`S19/C9`; S19/Q-C8 (a), lead) | PASS |
| Producer check | path `shared/src/shared/escalation_layout.py` present (absent at `1b106418da`) | PASS |

## ν — `process_identity` asks the portable liveness check

| Capability | Binding | Verdict |
|---|---|---|
| One liveness check throughout | Leo, 2026-11-13: zombie detection negligible until evidence to the contrary, the portable check used throughout. Census at `565d15a06f`: 5 liveness call sites outside `shared.proc_liveness.pid_alive`; ν converts `process_identity.py`'s 2 (`capture`, `liveness` through `_running_stat`), S24 γ the watchdog's 2, and the warm-lane sweep's `kill -0` is the same portable test (PRD C1) | PASS |
| Identity stays | the start-ticks and boot-id compare (pid reuse, reboot) is process identity, which `pid_alive` cannot answer, so the `/proc/<pid>/stat` read stays for it | PASS |
| `pid_alive` is importable from a stdlib leaf | `shared/src/shared/proc_liveness.py` imports only `os`; both modules are in `shared/tests/test_pure_stdlib_leaves.py::PURE_STDLIB_LEAVES` | PASS (substrate) |
| Producer checks | `_EXITED_STATES` absent from `process_identity.py`; `from shared.proc_liveness import pid_alive` present there | PASS |

## θ1 / θ2 — `shared.session`; cockpit on shared only (integration gate)

| Capability | Binding | Verdict |
|---|---|---|
| Scripts can import `shared` | S25 ε (task 6760) binds `scripts/sitting` and `scripts/legibility` through `scripts/_checkout.py` | PASS (edge) |
| cockpit's only orchestrator import is the session registry | 19 import statements in 17 files, all `orchestrator.session_registry` (measured at `1b106418da`) | PASS |
| `session_liveness` can join the package | `shared/src/shared/session_liveness.py` is stdlib-only and has 7 importers at `1b106418da`, all declared by θ1 | PASS |
| The archive layout is stdlib data | `escalation/archive.py` imports only the stdlib; `ARCHIVE_SUBDIR`, `DATE_FORMAT`, `archive_dir_for_date` move unchanged; `scripts/escalation_ladder.py` already imports `shared` through `scripts/_checkout.py`, and `scripts/check_l0_settle_outcomes.py` can do the same (it sits beside `_checkout.py`) | PASS |
| The queue keeps its own locator | θ2 changes no locator logic in `escalation/src/escalation/queue.py::EscalationQueue._locate_path` (memoised listing, negative cache); it repoints its layout reads; S3 θ (9177) extends it; `locate_record` serves processes that hold no queue (C9) | PASS |
| `escalation.archive` does not become a re-export | 11 files read the layout through it at `1b106418da` (`queue.py`, `sweep.py` and 9 tests, `scripts/tests/test_check_l0_settle_outcomes.py` new since the draft: `git grep -lE 'archive\.(ARCHIVE_SUBDIR\|DATE_FORMAT\|archive_dir_for_date)\|from escalation\.archive import'`, `session_registry.py`'s docstring excluded); θ2 declares and repoints all 11 | PASS |
| Decision close keys from shared status names | S19/Q-C8 (a): θ0 creates `RESOLVED`, `DISMISSED` in `shared/escalation_layout.py`; θ2 → θ0 | PASS (edge) |
| Archive-layout mirror removed | S19/OQ-3 (a), ruled by Leo; `ESCALATION_ARCHIVE_SUBDIR` (orchestrator, read by scripts), `ARCHIVE_SUBDIR` (escalation/archive.py, scripts/escalation_ladder.py, scripts/check_l0_settle_outcomes.py) and `ARCHIVE_DAY_FORMAT` (the settle check) match today | PASS |
| Producer checks | path `shared/src/shared/session/records.py` present; no `from`/`import orchestrator` statement in `cockpit/` (θ1); no `"orchestrator"` in `cockpit/pyproject.toml`; no `encode_cwd` definition in `scripts/legibility/`; no `_window_id_int` in `cockpit/src` | PASS |

## λ — Operator gate on decision closing, after the review-all capstone

| Capability | Binding | Verdict |
|---|---|---|
| The capstone gate exists (substrate) | gate task 6698 ("Run /review-all on dark_factory") is pending on 2026-11-12, a deterministic human gate with a delayed milestone; λ depends on it (ruled_by: lead) | PASS |
| The layout has one home first | producer θ2 (`shared.escalation_layout.locate_record`, the decision registry repointed); the edge is the gate, and is needed because 6698 waits only on high and critical tasks | PASS (edge) |
| The gate shape is native | `task_kind='deterministic'`, `always_escalates=true`, no `before_done`, `metadata.milestone = {"mode": "delayed", "after_secs": 604800}` (`docs/task-authoring.md` §5, §6; precedents 5394, 3676) | PASS |
| The session's ruling is recorded | the escalation is resolved with a ruling (and a `/prd` path if B or D); the resume marks λ done | PASS (manual check) |

## ι — Public CLI runner

| Capability | Binding | Verdict |
|---|---|---|
| The two runners can share one spawn path | `invoke.py::_run_subprocess_local` already imports `_materialize_stdin` and `terminate_process_group` and returns `_SubprocessResult`; the shared runner's transcript watch is optional per call | PASS |
| Every code importer of the renamed names is declared | 18 code importers at `565d15a06f` (16 at `1b106418da`, 14 at `97be1d71fa`; task 5399 added `agents/zcode_backend.py` and `orchestrator/tests/test_zcode_backend.py`, task 7993 `agents/mimo_backend.py` and `orchestrator/tests/test_mimo_backend.py`), all among ι's 20 declared files with `orchestrator/tests/test_liveness_boundary_gate.py` and four shared tests; the other mentions are docstrings and comments (D6) | PASS |
| ι does not wait on the CLI-invocation plan | S19/Q-C2 (a), lead: S20 β and θ depend on ι; ι has no prerequisite (it shares only test files with ζ, and test-file edges are not wired), so 9232 → ι and 9233 → ι close no cycle | PASS (edge) |
| Producer checks | `_run_subprocess_local` absent from `orchestrator/src`; `async def run_cli_subprocess(` present in `shared/src`; none of the four private names appears in `invoke.py`, `zcode_backend.py` or `mimo_backend.py` (a word-bounded pattern, not a layout-dependent one) | PASS |

## κ1 / κ2 — Dashboard egress and public names

| Capability | Binding | Verdict |
|---|---|---|
| The shared MCP session lands first | S14 κ (task 6722) puts `data/memory.py` on `shared.mcp_session.McpSession` | PASS (edge) |
| The egress keeps today's signature and eviction rules | `dashboard/src/dashboard/data/memory.py::mcp_tool_call(client, base_url, tool_name, arguments, timeout=10)` evicts on hang-shaped exits only; `first_success`, `get_queue_stats`, `get_wal_status` evict a failed url on every `mcp_fanout.py::FANOUT_FAILURE_EXCEPTIONS` member themselves; C8 keeps the signature exactly and D13 keeps both rules, the fan-out calling `mcp_client.invalidate_session` at module level | PASS |
| κ2 does not rename the fan-out lock constant | task 6879 (pending) makes `_LOCK_ACQUIRE_TIMEOUT_SECONDS` and `_TASKS_PER_PROJECT_BUDGET` public; κ2 → 6879 | PASS (edge) |
| Producer checks | path `dashboard/src/dashboard/data/mcp_client.py` present; no `from dashboard.data.memory import invalidate_session`; `async def mcp_tool_call(` absent from `data/memory.py` (sees `metrics.py`'s multi-line import and `app.py`'s attribute calls); "vestigial" absent from `dashboard/src`; no `async def _project_scoped_dbs…` family in `project_dbs.py` | PASS |
