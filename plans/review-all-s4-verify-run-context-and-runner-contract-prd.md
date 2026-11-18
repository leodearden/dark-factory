# S4 — Verify run context and runner contract (PRD)

**Status:** draft, authored 2026-11-01 by the S4 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S4, wave 2, size L, risk high); critic
`critique/S4.critic.md` folded 2026-11-02; the rulings of the 2026-11-10 sitting folded 2026-11-10
(§Rulings), and the lead's sequencing rulings of 2026-11-10 plus a fold-verify pass applied 2026-11-10 at
`bb1c659562`. Ruling R4 (2026-10-10) is implemented here, not relitigated. B+H: seven contracts (§Contracts)
and a two-way boundary-test sketch whose cross-module rows are the integration gate π. Twenty leaves.

**Code anchors** verified against main `e634cfe658` (2026-11-10). The draft was verified at `ce0f71a384`
(2026-11-01) and `d40ba048e9` (2026-11-02); 2,532 commits have landed since. Of the paths this plan
touches, `workflow.py`, `harness.py`, `git_ops.py`, `merge_lane/worker.py`, `verify.py` (12,486 raw lines,
+330), `verify_runner.py` (4,693), `verify_hosts.py`, `task_verify_dispatch.py`, `verify_plan.py`,
`verify_evidence.py`, `config.py`, `cli.py` and `evals/runner.py` changed; every symbol below was re-read at
`e634cfe658` and every count a done-criterion or a delivered check rests on is labelled `measured at
e634cfe658` (counts carried over unchanged from the draft keep their older sha and say why they still
hold). Re-checked at `bb1c659562` (2026-11-10, 77 commits later, task 8524's merge-lane re-anchor work among
them): no symbol or count this plan relies on changed, and every sidecar check still fails as it should. The
run pinned `6551b75aba`. Main moves fast: cite by symbol and re-locate at implementation time.

## Goal

A verify run's kind is one frozen value, its outcome is one typed value, and the 12,486-line
`orchestrator/src/orchestrator/verify.py` (measured at `e634cfe658`) becomes a stack of modules each
statable in one sentence. What the next agent observes when this lands:

- A new verify caller (a probe, a sweep, a remote leg, a CLI) picks one named constructor —
  `VerifyContext.merge_gate(registry)`, `.main_probe(registry, resource_class=…)`, `.task(registry)`,
  `.background(registry, retry_owner=…)` — and never has to learn which of `role`, `is_merge_verify`,
  `allow_cold_cache` and `max_retries` mean what together. An impossible pairing raises at construction,
  naming the fields. Every existing call site resolves to today's cold rule, timeout track, attempt count
  and warm mark (C1 site table).
- No verify entry point raises for infrastructure trouble. An infra errno on the warm marker comes back as
  a `VerifyResult` whose `infra_faults` names the errno and archived logs, and every one of the seven entry
  points reads `result.infra_transient` the same way (R4) and branches explicitly: the merge lane, the task
  workflow's infra loop and post-rebase re-verify, the main probe and both baseline probes, the sweep, the
  review checkpoint and the eval runner. `VerifyInfraError` and the eight "merge_queue.py has no
  VerifyInfraError handler" restatements are gone (`merge_queue.py` itself no longer exists).
- A bare xdist worker crash is not infrastructure. It comes back as a failing result whose test leg is
  `XDIST_WORKER_CRASH`, at the merge gate as at task verify. Both gates re-run it once; a second crash is a
  real failure that goes to the fix loop (the debugger at task verify, the steward at the merge gate) with a
  report naming both crashes and the test running at each, never to a human hold (S4-5).
- Every verify function that runs a command takes its `VerifyHost` as a required argument, so a verify
  helper that forgets to pass its host on is a pyright error, not a test that silently runs real commands.
  The process host is resolved once, at the boundary objects the harness builds and at the two remote
  verify commands; production code never builds a second host (S4-1).
- An operator whom a cross-check divergence or a drift escalation tells to re-prove a remote host runs
  `orchestrator verdict-parity --sha <sha>`, then readmits it with `clear_verify_host_quarantine` (S4-8).
- A debugger's block report (`VerifyResult.failure_report`) is built from per-leg facts: a green test leg
  is never headlined `## Test Failures`, a `0 errors` pyright run never `## Type Errors`, and which legs
  timed out is read from the leg, not from `'timed out' in output.lower()`.
- Which commands a verify runs is decided once, by `verify_plan.py::derive_verify_plan`, for both the
  module-config branch and the zero-module fallback branch; the persisted `attempt-N.plan.json` is what
  ran. A config-only merge resolves through one no-source-diff resolver on dark-factory (module configs)
  and on reify (none) alike.
- Every `attempt-*` artefact name is rendered by one helper that `scripts/verify_budget_census.py` also
  parses with; the summary record carries `task_id` and `module_prefix`.
- Tests drive verify through a public `VerifyHost(runner=…)` seam; the 215 dotted-path patches of
  `orchestrator.verify._run_cmd` and the 12 module-level mutable caches are gone (both measured at
  `e634cfe658`).
- The remote merge leg and the local merge leg run from one lossless `MergeVerifySpec`; host selection and
  quarantine live only in `HostAllocator`.

## Background

The run `review-all-dark_factory-20261008` found that a verify run's context is threaded as separate
keyword arguments encoding at least five independent facts (`fk-0bb03714372a`), that whether
`VerifyInfraError` may be raised depends on caller context the raiser cannot see (`fk-5037fafc755c`),
that the scope decision has two engines (`fk-fac2cb6af858`, `fk-d99f146ff901`), and that `verify.py` is a
god module with a measured partition (`fk-73b03674a8e4`, `fk-a635d54da4ab`). R4 ruled option 1 —
infra-transient is data on a typed `VerifyResult`, callers branch — and accepted the rest of the
synthesis remedy. One key is already fixed on main (the remote wire identity, task 6565); a second has
its structural half fixed and a residual to file; a third (`fk-3756ad2f3357`) had its triplicated
re-run command builder merged into one helper by task 7918 and keeps its policy-table half (§Findings
disposition).

Rulings applied: R4 (this stream), R22 (S4 owns no invariant promotion), R23 (weakened keys
`fk-5037fafc755c`, `fk-cb29fac47dba`, `fk-591206484ea5` re-verified below), and every design question
of this plan as ruled at the sitting of 2026-11-10 (§Rulings).

## Rulings (sitting of 2026-11-10)

| Id (sitting key) | Ruling | By |
|---|---|---|
| OQ-1 (S4-1) | Option A, a `VerifyHost` value, with Leo's change: `host` is REQUIRED on every verify function (entry points and public probes), with no default anywhere inside the verify modules. The process host is resolved once at the long-lived boundary objects the harness builds (task workflow, merge-lane worker, review checkpoint, eval runner) and at the two remote verify commands. Two guards: no `host` default in any verify signature, and production code outside the composition roots never calls `VerifyHost.fresh()`. The ContextVar stays only as the process slot and the per-test reset (D8, C5, ι1). | Leo |
| OQ-2 (S4-2) | As recommended: `infra_transient` is derived from the failing legs, and `infra_faults` is the recorded data (D2). | Leo |
| OQ-3 (S4-3) | As recommended: the per-leg rule everywhere except the cross-check fleet-halt gate, which keeps the aggregate pending task 6652 item 3 (D3). | Leo |
| OQ-4 (S4-4) | As recommended: a lossless per-module profile on the spec, and the local leg applies the same spec (D10, η). | Leo |
| OQ-5 | The module names and places of the §Sketch table stand; the runner protocol is `VerifyCommandRunner`; `ResourceClass` lives in `shared/verify_admission.py`. | the lead |
| OQ-6 (S4-5) | New option: at BOTH gates a bare worker crash is re-run once; a second crash is a real failure whose report names the crash and the test running at the time, and it goes to the fix loop (the debugger at task verify, the steward at the merge gate), never to a human hold. The bare-crash category is not infra-transient. The cross-check fleet halt behaves as today (D1, D17, C2, α). Recorded as a clarification of R4: R4 retired the raise; this rules where the crash goes once it is data. | Leo |
| OQ-7 (S4-6) | As recommended: the result types move to `verify_result.py` in their own leaf μ0. | Leo |
| OQ-8 (S4-7) | As recommended, option (b): ε, which deletes the fallback engine, runs last on `verify.py`. | Leo |
| OQ-9 (S4-8) | Option A (the recommendation was to defer): expose the tool as `orchestrator verdict-parity`, defaulting to the single divergent SHA; rewrite both escalation action texts to name it and `clear_verify_host_quarantine`; one new leaf ρ (D15). | Leo |

No 2026-10-10 ruling is amended. OQ-6 is recorded as a clarification of R4, as above.

## Findings disposition

Contract §8 step 1 (`find_tasks_by_metadata x_finding_key`, re-run 2026-11-10 over all 136 tasks that
carry the key) found no task for any of the 24 keys; step 2 hits are under §Existing tasks. Each key is
carried by the leaf that closes it (its `x_finding_key`); `fk-3e806baf4c6e` is also carried by ι2, which
retires its patch half, and `fk-e906c4e914c2` by ο, which deletes the dead code before ρ gives the kept
parity code its caller. Leaves that only prepare a key's closing name it in their text and carry no key.

| Key | Sev | Disposition | Re-verification |
|---|---|---|---|
| `fk-0bb03714372a` run context as kwargs | medium | leaf β | Measured at `e634cfe658` (AST): `role` threads 24 defs (verify.py 17, verify_plan.py 3, `merge_lane/ports.py::VerifyPort.run_scoped` 1, `task_verify_dispatch.py` 3: `TaskVerifyPort.__call__`, `LocalVerifyPort.__call__` and `_VerifyCall.run`, each `role: Literal['task']`); `is_merge_verify` 5 defs (verify.py 4, ports.py 1) plus `verify_runner.py::MergeVerifySpec.is_merge_verify`; `max_retries` 5 defs; 3 `role == 'merge' and is_merge_verify` conjunctions in `verify.py::_run_scoped_verification_body`; 2 `not is_merge_verify` gates (bare-xdist reclassification, warm mark); 26 role comparisons in verify.py, 5 in verify_plan.py. `('merge', no flag)` is still the shape of `_whole_tree_main_baseline` and `_probe_modules_on_main`, and now of `verify.py::run_node_ids` too (task 7918, `role='merge', max_retries=0`); the main probe's own verify in `verify_failure_is_preexisting_on_main` runs `role='task'`. A third spelling of the vocabulary landed with task 5652: `verify_cache/modes.py::VerifyRole = Literal['task', 'merge', 'background']`, read in 5 source files (13 references). `allow_cold_cache=False` has no production caller. `max_retries` is only ever `0` or the default. |
| `fk-5037fafc755c` VerifyInfraError gated by caller context | medium, weakened | leaf α | Re-verified under G3/G6: raise sites `verify.py::_mark_verify_warm` and the bare-xdist block of `verify.py::run_verification`; handlers `workflow.py::TaskWorkflow._run_scoped_verification_with_infra_retry` (`except`) and `TaskWorkflow.run` (`isinstance` arm) plus the `workflow_types.py` `BlockDisposition` row. The 8 restatements (verify.py 5, verify_runner.py 2, flake_ledger.py 1, measured at `e634cfe658`) name `merge_queue.py`, which is deleted. The main probe still turns any raise into `(False, '')` at DEBUG; the baseline probes degrade a raise to `None`. `TaskVerifyDispatcher` (tasks 6670/7977, since the draft) is a new task-verify caller; it reads the aggregate category for its remote leg. The weakening (a second designed handler exists) stands; the cost — two outcome channels, gating held only by prose, per-caller divergence across seven entry points — is present. The S4-5 ruling decides where the bare crash goes once it is data (D17). |
| `fk-0f9cace0cd00` no-source-diff policy written twice | medium | leaf δ | Two blocks in `_run_scoped_verification_body` (647 lines at `e634cfe658`), same three outcomes; `verify.py::_trivial_pass_reason` still names "Site 1 / Site 2". |
| `fk-fac2cb6af858` two scope engines | high | leaf ε | The fallback branch still executes `verify.py::_build_fallback_config` (314 lines at `e634cfe658`) and derives the plan only for the record (`_safe_derive_executed_fallback_plan`, `_executed_fallback_plan`); the `verify_plan.py` module docstring still claims classification "EXACTLY ONCE". |
| `fk-d99f146ff901` scope policy computed twice | high | leaf ε | `_build_fallback_config` recomputes `has_conftest`/`collectable_tests`/`has_test_data`; `_select_subproject_pytest_targets` computes them a third time. |
| `fk-0c5e73f13485` module registry side channel | medium | leaf β | `verify_plan.py::effective_merge_module_configs` still lets the registry win. The private write moved from `run_merge_verify_on_worktree` into `verify_spec_shared.py::apply_spec_to_host_config`, now shared by the merge and task specs; `verify.py::run_full_verification` reads `config._module_configs` twice; 76 test assignments to `._module_configs` in 33 files (measured at `e634cfe658`). The 139-line in-function defence is gone; ~40 lines of it live in `apply_spec_to_host_config`'s docstring. |
| `fk-8dbae7d39adc` local and remote legs, two sources of truth | high | leaf η | `LocalRunner.run_merge_verify` still drives from live config, "not from the spec (by design)". `verify_spec_shared.py::module_config_from_command` drops per-module warm and cold timeouts, `concurrent_verify`, `sequential_lint_first`, `verify_env` and `scope_cargo` for both specs. Config-level fields are classified by task 5295's `orchestrator/tests/test_merge_verify_config_fidelity.py::MERGE_PATH_CONFIG_READS` (done); per-module fields are not. All unchanged at `e634cfe658`; `MergeVerifySpec` still lacks task 7465's two fields. |
| `fk-e906c4e914c2` verify_runner reach-backs and dead code | low | leaves ο and ρ (closes at ρ) | Module-level import of `verify._archive_merge_verify_logs` and `verify._target_subprocess_env`; `verify_categories._assert_sentinels_disjoint`; `verify.apply_merge_flake_suppression` through the module attribute; two function-local imports in `run_merge_verify_on_worktree`, one with a stale `# type: ignore[attr-defined]`, one the upward `merge_lane.worker._run_unscoped_typechecks` (defined in `merge_lane/gates.py`, re-exported by worker), used only as the default of the injectable `run_unscoped`. No caller: `SccacheStats`/`parse_sccache_stats`/`capture_sccache_stats`, `ColdWarmVerifyDelta` and codec. `run_verdict_parity`/`render_parity_report` have no code caller but two live escalation actions tell operators to run them; under S4-8 ο keeps them and ρ gives them their caller, `orchestrator verdict-parity` (D15). `capture_env_fingerprint`/`compare_env_fingerprints` are live through `DriftDetector.check_env`. `verify_runner.py` is 4,693 lines (measured at `e634cfe658`). |
| `fk-5fa21a4b85cb` remote wire has no schema version | high | `fixed:31d2464f04` | Task 6565: every remote verify runs in a gate checkout of the dispatcher's deployed sha (`deployed_sha.py::DeployedSha`, recorded around the harness import), and `verify_runner.py::RemoteRunner.ensure_gate` refuses, kind `wire_shape`, a gate whose `remote_gate.py::wire_fingerprint(wire_classes())` differs, before any verify. `sync_if_stale` is gone. That is the ruled schema identity in a stronger form; `wire_classes()` is a hand-kept tuple, so C3 and C7 oblige every new nested wire class to join it. |
| `fk-baa454b81f6f` host selection and quarantine have two homes | medium | leaf θ | Re-read at `e634cfe658`: `VerifyRunnerPool` keeps `_quarantined` (5 lines), `_select_runner`, `eligible_remote`, `clear_quarantine`; `HostAllocator` keeps the `verify_host_policy` acquire (now stamping `HostLease.reason`, task 8523), the worker-shared quarantine set and PARKED. The merge path builds single-runner pools (`merge_lane/worker.py`), drift takes its leases from the allocator but still builds a two-runner pool and quarantines through it (`merge_lane/drift.py::_run_drift_check`, `DriftDetector.check`), the worker writes the set directly too (`SpeculativeMergeWorker.bench_unavailable_runner`), and the drift action names `pool.clear_quarantine`. The set itself is built once by `Harness.__init__` (task 6670, measured at `bb1c659562`) and shared by reference with the worker, `verify_hosts.py::HostPool` (task 7977's task-verify pool, a reader with its own per-host admission gates) and the drain participant; `build_remote_runners` keeps a dead quarantine filter (task 8734). Task 5040 (pending, finishes first by S6-2 = B) moves the worker's sites into `merge_lane/verify_dispatch.py` and siblings. |
| `fk-3756ad2f3357` isolated rerun implemented three times | medium | leaf ζ (builder half `fixed:fca89bd55b`) | Task 7918 merged the three builders into one helper, `verify.py::_node_id_test_config` (`fca89bd55b`, 2026-11-08): the triplicated command composition is gone. The residual is the policy half: the sweep pre-filter (`_sweep_failure_reproduces_in_isolation`) and the sweep confirm (`confirm_main_tip_failure_is_real`) still run their own re-run paths outside `confirm_isolated_rerun_verdict`, `_CALL_SITE_POLICY` has two rows, and the sweep confirm still lacks the `session_truncated` precondition (task 6249). Leaf ζ keeps that half. |
| `fk-f994995001e5` sweep-confirm worktree hand-rolled | medium | leaf ζ | `confirm_main_tip_failure_is_real` still mints `_mainsweepconfirm-<hex>` with a 3-retry `git worktree add`, a scoped remove plus `remove_tree`, and a comment-only DD5. `git_ops.py::WorktreeKind` has `MAIN_PROBE` and `MAIN_SWEEP` only, and `_mainsweepconfirm-` does not start with the protected `_mainsweep-` (unchanged at `e634cfe658`). |
| `fk-caa8f680e1c2` leg outcome flattened into text | high | leaf γ | `VerifyResult` has 24 fields (measured at `e634cfe658`); outputs are strings; `verify.py::_aggregate_results` substring-scans child summaries for `'tests failed'`/`'lint issues'`/`'type errors'` and recovers notes with `removeprefix('Failures: ').split(...)`; `verify_runner.py::RemoteRunner` decides no-verdict with `verify.py::is_no_verdict_summary`, a substring scan; parallel fields `failing_test_ids_by_module`, `failing_leg_categories`, `test_leg_failed`. Task 5651's `units` carry per-leg rc as `module:<prefix>:<leg>` strings. |
| `fk-3b67e8761674` failure_report re-derives facts | medium | leaf γ | `VerifyResult.failure_report` still tests `'timed out' in self.*_output.lower()`, gates the type section on `'error' in self.type_output.lower()`, and tails `lint_output[-2000:]`. |
| `fk-09ad1a19f587` category taxonomy as strings | medium | leaf α | `VerifyResult.category: str`. Sentinels outside `FailureCategory`: `'merge_flake_suppressed'` (`verify.py::_merge_flake_suppressed_pass`, matched as a literal in `DriftDetector.check`), `verify.py::_MERGE_NO_EVIDENCE_CATEGORY`, `verify_runner.py` `UNSCOPED_TYPECHECK_FAILED_CATEGORY`/`UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY`/`FLOCK_CONTENTION_CATEGORY`, kept apart by `_assert_sentinels_disjoint`. 9 aggregate `in INFRA_TRANSIENT_CATEGORIES` reads (worker.py 4, verify.py 3, workflow.py 1, task_verify_dispatch.py 1 — now in `TaskVerifyDispatcher._remote_verdict`). 18 test constructions in 12 files pass non-member category literals (AST, measured at `e634cfe658`; the two new files are `test_flake_recorder.py` and `test_merge_queue_flake_recorder.py`). |
| `fk-a2fb58789c14` cause hint tool-blind | high | leaf γ | `verify.py::_extract_cause_hint(output)` runs one ladder over every leg, called from `_summarize_checks` without the tool; `verify_classify.py::classify_failure` already dispatches on `ToolKind`. `_classify_environmental` stays tool-blind by design (host infrastructure, not tool output). |
| `fk-c88f7d3ae54a` plan slot routed on reason prefix | low | leaf δ | Moved, not fixed. Task 5652 (`5fa05675a6`) moved the prefix table into `verify_plan.py::_LEG_BY_REASON_PREFIX` behind `planned_run_leg`, which still decodes a run's leg from its reason prefix; it now has two readers, `verify.py::_mc_attr_for_run_or_none` (through `_MC_ATTR_BY_LEG`) and `verify_cache/keys.py`. `_FALLBACK_TOOLKIND_BY_ATTR` is unchanged and `verify_plan.py::PlannedRun` still has no leg field. The same commit added a second leg enum, `verify_evidence.py::Leg`, with the same three values as `verify.py::LegSlot` (44 uses in verify.py, 28 in two test files, measured at `e634cfe658`). |
| `fk-37d63bf83608` attempt-* names unowned | medium | leaf λ | Composers: verify.py 9, `verify_runner.py::RemoteRunner._archive_failure_streams` (runner origin in the prefix slot), `task_verify_dispatch.py` (`attempt-N.dispatch-<stamp>.json`, new). Prefix sanitiser spelled 6 times (`verify.py::_make_infix`, `_warm_marker_name`, `_stream_log_path`, `verify_runner.py::_sanitize_runner_name`, `scripts/verify_budget_census.py::sanitise_prefix`, `scripts/suite_census_evidence.py`); stamp format 3 times (`verify.py::_archive_stamp`, `verify_runner.py::_STDERR_ARCHIVE_TS_FMT`, `task_verify_dispatch.py::_SIDECAR_STAMP_FMT`). The census imports no first-party module today and its docstring still says two facts come only from the path. Measured at `d40ba048e9`; still exact at `e634cfe658` (the 17 verify.py commits since touch no archive writer, and the census and `suite_census_evidence.py` are unchanged). |
| `fk-cb29fac47dba` second `&&` grammar | low, weakened | leaf ε | Re-verified under G3/G6: `_AND_CLAUSE_SPLIT_RE` and `_cd_clause_target` remain beside `verify_cmd.py`'s quote-aware `split_top_level_and`/`split_and_chain_segments`. Their only production caller is `_build_fallback_config` (through `_scope_fallback_tool_to_subproject`), which ε deletes; tests/scripts pin them (`test_fallback_verify_config.py`, `test_pyright_version_pin.py`, `verify_command_invariants.py`, two more). The weakening (no current config shape triggers the quote hazard) stands; closing it costs nothing beyond ε's required re-expression of fallback rendering over `verify_cmd`. |
| `fk-126a75a57ca5` module-level mutable state | low | leaf κ | 12 structures in verify.py (the run's 11 plus `_BASELINE_SKIPPED_IDS_CACHE`), by the mutation rule of κ's guard; a thirteenth, `verify_admission_gate.py::_GATES`, outside verify.py (the same 13 by that rule at `e634cfe658`; `verify_hosts.py::HostPool`'s per-host gates, task 7977, are instance state). `_suppressed_flake_records` has 3 appends and 0 readers; `verify.py::clear_main_probe_caches` resets 3 of 12. |
| `fk-73b03674a8e4` verify.py god module | medium | leaf ξ | Same five responsibilities. Partition symptoms re-derived by AST at `e634cfe658` (unchanged since `ce0f71a384`): `_serial_pytest_str → _worktree_reader`; `_run_scoped_verification_body → _emit_trivial_pass_escalated`; `_whole_tree_main_baseline → run_scoped_verification`; `_probe_modules_on_main → run_verification`. μ, μ0, ν prepare the close. |
| `fk-a635d54da4ab` verify.py h14 measurement | medium | leaf ε (S4-7) | Measured at `e634cfe658`: 12,486 raw lines; 3,881 docstring + 2,426 comment lines = 50.5% prose (53.6% at the pin). Region sizes in §Sketch (measured at `ce0f71a384`; the 330 lines since landed mostly in the node-id runner, task 7918, and the scope memory limits, task 6106). `run_verification` alone is 1,034 lines. The final measurement is taken once the fallback region goes, in ε. |
| `fk-3e806baf4c6e` tests reach verify internals | high | leaves ι2 and ξ (closes at ξ) | Closes at ξ: ι2 retires the `_run_cmd` patches; μ, μ0, ν and ξ publish the imported private names. Both leaves carry the key; ι2 keeps priority high as an explicit choice. Measured at `e634cfe658`: 215 dotted `orchestrator.verify._run_cmd` patches in 21 files (118 in `test_verify.py`), plus `patch.object(…, '_run_cmd')` in `test_merge_gate_rerun_end_to_end.py`, `test_verify_merge_flake_suppression.py`, `tests/scripts/test_verify_budget_census.py`, an import in `test_verify_run_cmd_reap.py`, and a child-script string import in `test_shutdown.py`; 11 distinct dotted private targets; 31 test files import 67 private names (254 sites); 8 production modules import 11 verify privates (`evals/snapshots.py`, `merge_lane/worker.py`, `offline_lane_confirm.py`, `offline_lane_runs.py`, `review_pre_test.py`, `verify_runner.py`, `workflow.py`, `workflow_types.py`). ι1 and ι3 prepare. |
| `fk-591206484ea5` offline lane: one axis, three mechanisms | medium, weakened | `open` (lead note: file one mechanical task; becomes `filed:<id>` when the lead files it) | Structural half fixed at `3ae75182ed` (tasks 6857/6858/6861): one streaming launcher `offline_lane_runs.py::launch` (`9bd64f1dc1`); legs are `offline_lane_pass.py::LaneLeg` values run by `run_pass`, the `_default_*` spawn copies retired (`3ae75182ed`); confirm seams delegate to `offline_lane_confirm` (`016f01478d`); the misleading β2 docstring rewritten (`a3ed13ef9e`). The residual the skeptic left standing is still present at `e634cfe658` (`offline_lane_runs.py` is unchanged since the draft): `launch` awaits `proc.wait()` with no wall-clock bound; `search_tasks` (2026-11-10) finds no live task for it at 0.6 or above. The lead files one mechanical task carrying the key (§Existing tasks). |

## Pre-conditions for activating

Task statuses below were read with `get_task`/`get_statuses` on 2026-11-10.

- **Task 7041** (pending, high, unclaimed: the fallback test floor; its `NO_TESTS_RAN` result is the second
  C3 `passed` exception). ε depends on it: ε deletes the fallback engine 7041 edits and captures its
  characterization corpus after it. Under ruling S4-7 ε runs last on verify.py, so 7041 is off the stream's
  critical path.
- **Task 6765** (S25 κ, pending, low: census policy module). λ depends on it, as S25's PRD records
  ("κ lands first; S4's leaf depends on κ"); ι2 has a same-file edge with it.
- **Task 5292** (pending, medium, unclaimed and dispatchable: probe and baseline hardening). κ depends on
  it; 5292 depends on α (§Existing tasks).
- **Task 7465** (pending, low, unclaimed: `scope_cargo` and the config-only globs on `MergeVerifySpec`). η
  depends on it (§Existing tasks).
- **Task 5040** (pending, high: extract `merge_lane/verify_dispatch.py`; plan authored 2026-11-04, a
  half-built branch). By the sitting's S6-2 = B it finishes first and nothing parks it; S6's lifecycle task
  (S6 ζ) waits for it. Its plan moves `_run_post_merge_verify`, `_run_inflight_verify`, `_dispatch_item`,
  the host-quarantine machinery and the main-health probe out of `merge_lane/worker.py` into
  `merge_lane/verify_dispatch.py` and sibling modules (its `metadata.files` names `post_merge_verify.py`,
  `verify_cross_check.py`, `verify_failure_outcome.py`, `host_quarantine.py`, `main_health_probe.py`,
  `inflight_verify.py`), including the worker's writes and reads of the harness-built `_runner_quarantine`
  set (task 6670). So every `worker.py` site this
  plan names — α's infra-transient and crash branches, β's and ι1's call sites, θ's pool sites, μ0's and
  ξ's imports, ρ's cross-check action — is edited wherever 5040 put it, read at the leaf's base. No S4 leaf
  needs a direct edge on 5040: each one that edits those sites already depends on S6 ζ, which depends on
  5040.
- **S6 ζ** (S6 C2, `ItemLifecycle` operative): α, β, θ and μ0 depend on it by lead ruling (same-file order on
  `merge_lane/worker.py` and `merge_lane/gates.py`; S6's typed-facts leaves land first), and ι1 and ρ, which
  now also edit those files, follow it through α and β. **S6 λ** precedes θ and **S6 μ** precedes μ0 (lead
  ruling: the last S6 leaves on those files; labels confirmed by the lead's sequencing rulings of
  2026-11-10). S6's request that μ0 also follow S6 λ already holds through the chain (μ0 → μ → λ → κ → ι3
  → θ → S6 λ), so no extra edge is needed. S4 consumes no S6 contract; G4 lists no upstream stream.
  Other seam-adjacent sequencing with S5, S8 and S9 is in §Cross-PRD relationship.

## Sketch of approach

**Target layering (downward imports only; each row imports only rows above it).** Sizes are raw lines of
the source region at `ce0f71a384`. Module names and placement as ruled by the lead (OQ-5).

| Module | Holds | From region (lines) | Leaf |
|---|---|---|---|
| `verify_categories.py` (kept) | `FailureCategory` incl. sentinels, `CATEGORY_POLICY`, `INFRA_ERRNOS`, `is_infra_oserror`, `BARE_CRASH_RERUNS` | — | α |
| `verify_artifacts.py` (new, stdlib only) | `ArtifactName` grammar, `sanitize_prefix`, `ARCHIVE_STAMP_FORMAT` | parts of archive (848) | λ |
| `shared/verify_admission.py` (kept) | `ResourceClass` (also the verify-result cache's role vocabulary) | — | β |
| `verify_budget.py` (S6's, landed by S6 α before any S4 leaf) | `resolve_verify_timeout`, the leaf budget function | moved by S6 α | — |
| `verify_context.py` (new) | `VerifyContext`, `ModuleRegistry`, `WorktreeProvenance`, `VerdictAuthority`, `RetryOwner` | — | β |
| `verify_evidence.py` (kept) | `Leg`, the one leg vocabulary (task 5652's enum; `verify.py::LegSlot` goes) | — | γ |
| `verify_cmd.py`, `verify_plan.py` (kept) | the one scope engine; `PlannedRun.leg` | — | δ, ε |
| `verify_pytest_output.py` (new) | pytest/xdist output parsing, failing ids, cause hints, excerpts, `WORKER_DEATH_SUMMARY_MARKER` | 1,088 | μ |
| `verify_commands.py` (new) | command-string façade, `worktree_reader` | 372 + 244 (minus the fallback-only rescoper, which ε deletes) | μ |
| `verify_result.py` (new) | `CheckRun`, `VerifyAttempt`, `VerifyResult`, `LegOutcome`, `InfraFault`, wire codec | 540 | α/γ shape it, μ0 moves it |
| `verify_exec.py` (new) | `VerifyCommandRunner`, `SubprocessRunner`, cgroup scope, clock stop, ruff settings, segmented legs, `reap_leftover_verify_scopes` | 1,536 | ι1 (in place), ι3 (move) |
| `verify_stores.py` (new) | the owned caches with stated bounds; `MainShaBaseline` | scattered globals | κ |
| `verify_host.py` (new) | `VerifyHost`, the process slot (`ContextVar`) that `VerifyHost.process()` fills once and tests reset | — | ι1 (in place), ι3 (move), κ |
| `verify_archive.py` (new) | log/junit/summary/plan writers, prune | 848 | λ |
| `verify_leg.py` (new) | `run_verification` and its leg helpers (env, admission, preprovision, load, xdist workers, summary); the timeout resolver is imported from `verify_budget.py` | 1,646 + 370 | ν |
| `verify.py` (kept) | `run_scoped_verification`, `run_full_verification`, plan bridge, no-source-diff resolver, aggregation, `_emit_trivial_pass_escalated`, `merge_verify_command_budget_secs`; the fallback region until ε | 1,492 + 774 | δ, ε, ν |
| `verify_discriminator.py` (new) | isolated-rerun policy table, `confirm_isolated_rerun_verdict`, merge flake suppression | part of 2,558 | ζ, ξ |
| `verify_main_probe.py` (new) | main baseline, preexisting-on-main probe, main-tip sweep | 521 + part of 2,558 | ξ |
| `verify_runner.py` (kept) | `MergeVerifySpec`, `LocalRunner`, `RemoteRunner`, `dispatch_merge_verify`, `run_verdict_parity` (D15) | — | η, θ, ο, ρ |
| `verify_hosts.py` (kept; holds `HostPool` and `build_remote_runners` today), `verify_drift.py` (new) | `HostAllocator`/`HostLease`; `DriftDetector` | from verify_runner | ο |

The moves follow the seat's measured partition (`plans/review-all-dark_factory-20261008.json`
`h14_measurements`, seat `orchestrator/verify`). The two moves the ruling names come first inside their
leaves: `_emit_trivial_pass_escalated` moves from the discriminator region into `verify.py` beside the
no-source-diff resolver (δ), which breaks the discriminator cycle, and `_worktree_reader` moves into
`verify_commands.py` (μ), which breaks the command-string cycle. `_load_sample`/`_xdist_workers` move down
from the discriminator region into `verify_leg.py` (ν). `VerifyResult` must sit below `run_verification`,
which constructs it, and above the pytest-output group its `failure_report` reads; that forces
`verify_result.py` after `verify_pytest_output.py` (ruling S4-6). `merge_verify_command_budget_secs`
stays in `verify.py` as the worker-side entry (S6 C6); S6 α moves `_resolve_verify_timeout` into its leaf
`orchestrator/verify_budget.py` before any S4 leaf lands, so ν imports `resolve_verify_timeout` from there
and moves nothing of it.

**Order (ruling S4-7).** The contract leaves (α, β, γ) land first so the contracts other streams consume (C1,
C2, C3) exist early. δ and ζ follow; ε, which deletes the fallback engine, runs last on `verify.py` so the
splits (ι1–ξ) do not wait on task 7041. The fallback region and its fallback-only helpers
(`_scope_fallback_tool_to_subproject`, `_rescope_clause_to_subproject`, `_cd_clause_target`,
`_AND_CLAUSE_SPLIT_RE`) stay in `verify.py` through the splits. Every leaf that moves or rewrites a symbol
rehomes its reviewer-answering prose under the Comments stance (`docs/code-quality.md`): task
chronologies, measurements and rejected alternatives go to the commit message, at most a one-line pointer
stays, and no comment or docstring may name a symbol the leaf retires (D14). Each new module's public
surface is what its importers use; a name a test or another module imports is public there (heuristic 9,
Tests stance), never re-exported from `verify.py` (heuristic 13).

## Resolved design decisions

- **D1 — infra-transient is data (R4).** `run_verification` never raises for infrastructure trouble. The
  warm-marker errno becomes an `InfraFault` on the result, category `VERIFY_IO_FAULT`, for every context.
  A bare xdist crash is not infrastructure (S4-5, D17): when the bare-crash precondition holds, the test
  leg's category becomes `XDIST_WORKER_CRASH` in every context, the merge gate included (the
  `not is_merge_verify` gate on the reclassification goes), and nothing is raised. `VerifyInfraError`, its
  `BlockDisposition` row and the `isinstance` arm in `TaskWorkflow.run` are deleted; the infra retry loop
  keeps one channel. Every entry point branches explicitly (D16).
- **D2 — `infra_transient` and `bare_crash` are derived; `infra_faults` is recorded data (ruling
  S4-2).** With `failing_legs = [leg for leg in legs or () if leg.rc != 0]`, one helper answers "does
  every failing leg's category satisfy P", falling back to the aggregate row when no leg is recorded:

  ```python
  def _every_failing_leg(self, holds: Callable[[FailureCategory], bool]) -> bool:
      return not self.passed and (
          holds(self.category) if not failing_legs else all(holds(leg.category) for leg in failing_legs))
  infra_transient = _every_failing_leg(lambda c: CATEGORY_POLICY[c].is_infra_transient)
  bare_crash      = _every_failing_leg(lambda c: c is FailureCategory.XDIST_WORKER_CRASH)
  ```

  An `InfraFault` is recorded data, not a disjunct, so a type-leg `ENV_TRANSIENT` beside a genuine
  `TEST_FAILURE` in another leg or module is not infra-transient (the masking task 5506 exists to remove),
  and a no-legs `MERGE_NO_EVIDENCE` result reads its own row (not infra-transient). A warm-marker fault
  after green legs has no failing leg and reads `VERIFY_IO_FAULT`'s row (infra-transient). A crash in one
  module beside a `TEST_FAILURE` in another is neither infra-transient nor a bare crash: it is a plain
  failure. The per-leg rule is ruling S4-3. Until γ creates `legs`, α reads `failing_legs` from
  `failing_leg_categories`; None means not recorded and falls back to the aggregate row. When the
  bare-crash precondition holds, `run_verification` replaces the test leg's entry with
  `XDIST_WORKER_CRASH` (`_summarize_checks` classifies it `test_failure` before the reclassification, so
  without the rewrite neither gate could tell a crash from a test failure and give it its one re-run). γ
  swaps the source to `legs` with no change in any answer. A stored boolean would be a second home for a
  fact the category policy owns (heuristic 11); the ruled "timed_out/infra_transient as fields" is met as
  data on the frozen value, and `timed_out` stays a constructor field checked against the legs (C3).
- **D3 — every infra-transient consumer reads one predicate.** The 9 aggregate reads become
  `result.infra_transient`, with one deliberate exception: the cross-check fleet-halt gate in
  `merge_lane/worker.py` (in task 5040's `verify_cross_check.py` once it lands) keeps the aggregate,
  spelled `CATEGORY_POLICY[result.category].is_infra_transient`, because task 6652 records that switching
  it needs human adjudication (ruling S4-3). The merge lane maps `infra_transient` to its existing
  classified infra-transient branch (one in-place retry in the shared budget, then the transient-infra
  hold, never branch blame). A bare crash is not infra-transient, so it never reaches that hold (D17), and
  the fleet-halt gate treats a local crash beside a remote pass as it treats a test failure today. The
  cross-check's trust in a remote green is unchanged: `INDETERMINATE_VERDICT_CATEGORIES` stays exactly
  `{INFRA_KILL}`.
- **D4 — `VerifyContext` carries exactly the ruled five fields (R4).** `retry_owner` replaces
  `max_retries`: `VERIFY` means `run_verification` re-runs a pure-timeout failure up to
  `config.verify_timeout_retries`, `CALLER` means one attempt — the only two values any caller passes
  today. `worktree_provenance` replaces `is_merge_verify`'s cold/timeout/warm-mark meanings and
  `allow_cold_cache` (test-only today): `FRESH` is forced cold, takes the merge-cold timeout track and is
  never warm-marked (only `merge_gate`); `MARKED` is cold iff `.task/` exists without the marker and is
  marked after a completed build (a no-op where `.task/` is absent). `verdict_authority` replaces the
  `role == 'merge' and is_merge_verify` conjunctions. `resource_class` carries scheduling and plan breadth,
  as `role` does today (a main probe run for the merge lane takes merge resources and merge breadth).
  `module_registry` replaces the private channel. The command runner and caches are not context: they
  live on `VerifyHost` (D8). S6 α moves the timeout resolver into its leaf
  `verify_budget.py::resolve_verify_timeout(config, module, *, is_cold, merge_cold)` before β, naming the
  keyword `merge_cold: bool` with no default from the start (lead ruling 2026-11-10); β passes it as
  `context.worktree_provenance is WorktreeProvenance.FRESH`, so the leaf (which `config.py` imports at
  module level) needs no import of `verify_context.py`. If S6 α landed with the old spelling
  `is_merge_verify`, β renames it (β's absent check covers `verify_budget.py` either way).
- **D5 — the registry is explicit and the default is today's.** `context=None` at an entry point means
  `VerifyContext.task(ModuleRegistry.of(config))`, which reads the loading host's discovered registry
  through the public `module_configs_or_empty`. The remote legs pass the spec's registry; nothing writes
  `OrchestratorConfig._module_configs` outside `config.py`. `load_config`'s own discovery write stays: the
  config owns its registry (config shape is S16's). `_discover_module_configs` becomes public
  `discover_module_configs`, so verify's three function-local imports of a config private go.
- **D6 — one scope engine (R4).** `verify_plan.py::derive_verify_plan` renders the fallback branch's
  commands itself, as `VerifyCmd` values built with `verify_cmd.py` primitives, including subproject cd
  scoping, mixed root and subproject scoping, uv-context rescoping of lint and type, and OPAQUE
  first-clause scoping. `run_scoped_verification` executes plan runs on both branches. Behaviour is
  pinned by a characterization corpus captured from `_build_fallback_config` before it is deleted; every
  difference is either a recorded intended delta (tasks 7066, 7211) or a defect in ε.
- **D7 — one no-source-diff resolver (R4).** `verify.py::resolve_no_source_diff(context, has_full_gate,
  task_files, existing_files, worktree, event_store, task_id) -> NoSourceDiffOutcome` returns
  `TRIVIAL_PASS | FULL_GATE | NO_EVIDENCE(reason)` and is the only emitter of `trivial_pass_escalated` for
  that case. Both branches call it with their own `has_full_gate` (any module command vs a global command);
  the global-tail `empty_command_set` backstop calls it too, so double emission is impossible by
  construction rather than by comment.
- **D8 — `VerifyHost` is the process-lifetime collaborator, passed explicitly (ruling S4-1).** One frozen
  value holds the command runner and the owned stores (probe verdicts, main baselines, ruff-escape latch,
  preprovision coalescer, admission executor and gates, archive-prune throttle), each with a stated bound.
  Every verify function that runs a command or reads a store takes `host: VerifyHost` as a REQUIRED
  keyword: the three entry points, `run_node_ids`, the six public probes (`verify_failure_is_preexisting_on_main`,
  `run_main_tip_sweep`, `confirm_main_tip_failure_is_real`, `main_baseline_failing_ids`,
  `confirm_isolated_rerun_verdict`, `apply_merge_flake_suppression`), `LocalRunner` and every internal
  helper between them. No verify signature has a `host` default and no verify function calls
  `VerifyHost.process()`, so pyright flags a helper that forgets to forward its host — the bug a silent
  default would hide, which bites tests (a nested probe runs real commands instead of the scripted ones),
  not production. The process host is resolved ONCE, by `VerifyHost.process()`, at the long-lived boundary
  objects the harness builds: `TaskWorkflow`, `SpeculativeMergeWorker` and `ReviewCheckpoint` each take
  `verify_host: VerifyHost | None = None` and resolve None once at construction, which spares their ~640
  test constructions (126 in 92 files, 500 in 46, 14 in 12, AST at `e634cfe658`) any edit; the eval runner
  passes the host of the workflow it drives; the two remote verify commands, `orchestrator verify-merge`
  and `orchestrator verify-task`, each resolve their own process's host. `Harness` resolves the process host
  once in `__init__` and hands it to the boundary objects it builds, to `AdmissionWatch` (κ) and to its own
  main-tip sweep calls; `TaskVerifyDispatcher` holds no host and forwards the one each workflow call passes.
  The composition roots obtain the host through `VerifyHost.process()`, never `VerifyHost.fresh()`: a
  fresh host built beside the slot would be a second host for every object that resolves None through
  `process()` (row 14c). Guard 2's exemption for `harness.py` and `cli.py` is the ruling's letter; no
  composition root needs to use it.
  Two guards (ι1): no `host` default in any verify signature, and no production module outside the
  composition roots (`harness.py`, `cli.py`) and `verify_host.py` itself calls `VerifyHost.fresh()` — a
  second host would split the admission queue (more verifies than slots) and empty the baseline caches,
  which no signature can prevent. The `contextvars.ContextVar` stays only as the process slot `process()`
  fills once and the per-test reset: the autouse conftest fixture installs
  `VerifyHost.installed(VerifyHost.fresh())` and a `verify_host` fixture yields it; a test that scripts the
  runner passes `VerifyHost.fresh(runner=ScriptedRunner(...))`. Because no verify function reads the slot,
  work handed to the admission executor's threads never depends on context propagation. That one slot
  replaces 12 ad hoc globals and is exempt from κ's guard by type. `VerifyStores.close()` shuts the
  admission executor; `installed()` calls it on exit. Removing the slot as well (a host required on the
  boundary objects too) was weighed and not ruled: about 640 more test edits for the same guarantee.
- **D9 — one isolated-rerun policy table (R4).** The table is keyed by a verify-owned `RerunSite`
  (`MERGE_GATE`, `MAIN_PROBE`, `SWEEP_PREFILTER`, `SWEEP_CONFIRM`); each row names its ledger call site
  (`FlakeCallSite` or None, so the flake ledger's corpus does not change), its timeout, its
  `refuses_partially_measured_sessions` precondition (True for `MERGE_GATE`, `SWEEP_PREFILTER` and
  `SWEEP_CONFIRM`; False for `MAIN_PROBE`, as today) and its log wording. All four sites go through
  `confirm_isolated_rerun_verdict`, so the sweep confirm inherits the `session_truncated` precondition
  (task 6249's defect), and an `unconfirmable` answer keeps the alarm at both sweep rows. Every row builds
  its re-run command with the one helper task 7918 extracted, `verify.py::_node_id_test_config`. The sweep
  confirm worktree is minted by `GitOps.ephemeral_worktree(WorktreeKind.MAIN_SWEEP_CONFIRM, …)` and
  protected by construction.
- **D10 — the spec is the merge-verify profile (ruling S4-4).** `MergeVerifySpec` carries a lossless
  per-module `ModuleVerifyProfile` (prefix, three commands, warm and cold timeouts, `concurrent_verify`,
  `sequential_lint_first`, `verify_env`, `scope_cargo`). `verify_spec_shared.py` owns one
  `apply_merge_verify_spec(config, spec) -> AppliedSpec(config, registry, modules)` that the local and the
  remote legs both call, so a field the spec drops shows up on every local merge, not only on a remote
  one. The task spec reuses the profile. Task 5295's `MERGE_PATH_CONFIG_READS` guard extends to the
  per-module fields. Task 7465 (`scope_cargo` and `git.merge_config_only_full_gate_globs` on the spec) lands
  first; η builds on it.
- **D11 — HostAllocator is the one selection and quarantine home.** `VerifyRunnerPool` collapses into
  `verify_runner.py::dispatch_merge_verify(runner, spec, …)`: dispatch, emit the `merge_verify` event, fall
  back to the local runner on `RunnerUnavailable`. Drift takes its local and remote leases from the
  allocator and quarantines through `HostAllocator.quarantine_and_release`. The quarantine set itself is
  one process-lifetime set that `Harness.__init__` builds (task 6670) and shares by reference with
  `verify_hosts.py::HostPool` (the task-verify pool, which reads it and never writes it), with the
  `DrainParticipant` and with the merge worker (`runner_quarantine=`), whose `HostAllocator` wraps it.
  θ keeps that one set and its construction where they are and makes `HostAllocator` its only writer and
  the merge lane's only reader: the worker's (or, at the leaf's base, task 5040's dispatcher's) direct
  `_runner_quarantine.add(...)` on the bench and cross-check divergence paths and its membership reads go
  through allocator methods (`quarantine_and_release`, a lease-free bench, `is_quarantined`, `readmit`).
  The dead quarantine filter of `verify_hosts.py::build_remote_runners` (no production caller passes it;
  task 8734) goes too, so no third selection site remains. θ does not widen the RunnerUnavailable-class versus
  divergence-class quarantine conflation that task 5291 (pending, medium; it absorbed deferred task 3229)
  fixes by storing a quarantine reason; whichever of θ and 5291 lands second keeps the other's shape.
- **D12 — one artefact grammar (R4).** `verify_artifacts.py::ArtifactName` renders and parses every
  `attempt-*` name; the remote runner's origin gets its own field instead of the prefix slot. The module
  is stdlib-only. `scripts/verify_budget_census.py` imports no first-party module today; λ adds
  `import _checkout as _checkout` before its first first-party import (S25 ε's uniform-binding rule, task
  6760), which binds `orchestrator/src` for the system python (measured: the system python 3.12 imports
  `orchestrator.verify_categories` with only `orchestrator/src` on the path). The summary record carries
  `task_id` and `module_prefix`; the census reads them from the record and parses the path only for
  records written before λ. `verify.py::_warm_marker_name` uses `sanitize_prefix` too.
- **D13 — per-leg facts beside the text; `CheckRun` is the ruled record (R4).** One leg vocabulary:
  `verify_evidence.py::Leg` (task 5652's enum, values `test`/`lint`/`type`, in a leaf module verify already
  imports) replaces `verify.py::LegSlot`, which has the same values (heuristic 11). `CheckRun` gains
  `leg: Leg` (replacing `label: str`; "leg", not "slot", so it cannot be read as the admission
  `slot_wait_secs`, heuristic 1) and `tool: ToolKind`, set at construction where `_tool_for_cmd(cmd)` is
  computed today. `CheckRun.to_dict()` keeps the `'label'` key, now written as `leg.value`, so the summary
  bytes `scripts/verify_budget_census.py` reads (`entry['label']`) are unchanged. `LegOutcome` is the ruled
  `CheckRun`-with-`ToolKind` as published on `VerifyResult`: built only by `LegOutcome.of(check, category,
  cause_hint, failing_test_ids, no_verdict_note)`, so leg and tool are computed once. `CheckRun` stays the
  in-flight execution record. The human-readable output stays in `test_output`/`lint_output`/`type_output`,
  so the 494 test constructions of `VerifyResult` (101 files, AST at `e634cfe658`) keep working. `failing_leg_categories`,
  `failing_test_ids_by_module` and `test_leg_failed` become properties over `legs`; `failing_test_ids` and
  `timed_out` stay stored and are checked against the legs at construction (C3).
- **D14 — prose and tests stance.** Every leaf rehomes the prose of what it touches (see §Sketch); no
  comment or docstring may name a symbol the leaf retires, history goes to the commit message. A leaf that
  retires private patch sites names the public seam they move onto. Behaviour changes are limited to D3
  (rulings S4-3), D16's entry-point branches, D17 (ruling S4-5) and ρ's new command (ruling S4-8); every
  other leaf is behaviour-preserving and says so in its signal.
- **D15 — `run_verdict_parity` stays and gets its command (ruling S4-8).** Two live escalation actions tell
  operators to "re-prove the remote host env (run_verdict_parity)" (the cross-check divergence in
  `merge_lane/worker.py::_run_post_merge_verify`, which task 5040 moves into its `verify_cross_check.py`;
  `DriftDetector.check`'s suggested action), and `docs/verdict-parity-report.md` (2026-06-10) is its
  committed report, produced by hand-running `orchestrator verify-merge` on each host, not by this function,
  which has no code caller. ο keeps `run_verdict_parity`, `render_parity_report`, `VerdictParityReport`,
  `ParityRow` and their codecs and deletes only the sccache helpers and `ColdWarmVerifyDelta`. ρ adds
  `orchestrator verdict-parity --sha <sha> [--sha …] [--expect pass|fail] [--report <path>]`: it defaults to
  the one divergent SHA the escalation names (each SHA costs a full local and a full remote merge verify —
  52 minutes locally for task 7044 on 2026-10-21 — and the command runs while the queue is halted), mints
  and cleans one merge worktree per SHA under `merge_verify_lease` as the drift check does, builds the spec
  with `build_merge_verify_spec` and the remotes with `verify_hosts.py::build_remote_runners` from
  `config.enabled_verify_runners`, gates them on the CLI process's deployed sha the way `orchestrator run`
  captures it, prints `render_parity_report` (and writes it with `--report`), and exits non-zero on any
  disagreement. Both action texts then name the command and the escalation tool
  `clear_verify_host_quarantine` for readmission (θ already replaced the drift action's
  `pool.clear_quarantine`, a method θ deletes). The remote side runs the CLI process's code identity, so an
  operator who runs it from a checkout newer than the halted orchestrator checks newer code than the one
  that diverged; OPERATIONS.md says so.
- **D16 — every entry point branches on `infra_transient` and `bare_crash` (R4, "callers branch"; ruling
  S4-5).** The two gates give a bare crash its one re-run (D17); every advisory caller reads a bare crash
  as it reads an infra-transient result — no verdict — which is what today's raise gave them.

  | Entry point | `result.infra_transient` | `result.bare_crash` |
  |---|---|---|
  | workflow infra-retry loop (`TaskWorkflow._run_scoped_verification_with_infra_retry`, also the path through `TaskVerifyDispatcher`) | one channel: the existing bounded retry (`verify_infra_retry_max_attempts`), then `infra_hold` naming the `InfraFault`s | one re-run (`BARE_CRASH_RERUNS`), consuming no verify attempt; a second crash returns to `_verify_debugfix_loop` as a failure and the debugger receives `repeated_crash_report(first, second)`; never `infra_hold`. A remote task-verify crash is a verdict, not a bench (`TaskVerifyDispatcher._remote_verdict` reads `infra_transient`), so its re-run is dispatched like any verify and may run remote again |
  | workflow post-rebase re-verify (`TaskWorkflow._post_rebase_reverify_passed`, rollback-only: `pre_merge_reverify` is false by default and in this repo's yaml) | a typed outcome (passed \| failed \| infra); `_rebase_before_enqueue` maps infra to today's `infra_issue` block with `escalate_to_human`, the disposition the deleted `VerifyInfraError` row carried | one re-run; a second crash is today's failed branch (`False`, the `post_rebase_verify_fail` waste event), and the merge gate verifies the landing anyway |
  | merge lane (`_run_post_merge_verify`, in task 5040's module once it lands) | D3: one in-place retry in the shared budget, then the transient-infra hold | one in-place re-run on the held host (`BARE_CRASH_RERUNS`, its own count, not the infra budget); a second crash takes today's branch-failure path (discriminator, main-health probe, block, steward) with `repeated_crash_report` in the block detail; the cross-check's local leg is not re-run: a local crash beside a remote pass diverges and halts the fleet as today |
  | main probe (`verify_failure_is_preexisting_on_main`) | `(False, '')` with a WARNING naming the fault, not cached, until task 5292's three-state answer | the same, WARNING naming the crash |
  | baseline probes (`_probe_modules_on_main`, `_whole_tree_main_baseline`) | `None` before any `seed_main_baseline` or `_remember_main_baseline`, so a crash or fault victim is never seeded as red on main | the same |
  | main-tip sweep | no verdict, as today's inline infra sets already treat it | the same |
  | isolated re-runs (`confirm_isolated_rerun_verdict`, `confirm_main_tip_failure_is_real`) and `run_node_ids` | unconfirmable / `'error'`, as a raise gives today | the same |
  | review checkpoint (`ReviewCheckpoint._run_phase1`) | the checkpoint's existing skip path, a new `ReviewOutcome.SKIPPED_NO_VERDICT`, trigger kept; no reviewer receives it | the same |
  | eval runner (`evals/runner.py::collect_metrics`) | the verification cell is recorded Unmeasured (`evals/measurement.py::Unmeasured`, S18 task 6734) instead of scored as a failure | the same |
- **D17 — a bare worker crash is re-run once, then it is a real failure (ruling S4-5, a clarification of
  R4).** A bare crash (`verify.py::_is_bare_xdist_worker_crash`: the xdist worker-death signature with no
  genuine pytest failure surface, only the test leg red) looks the same whether host overload killed the
  worker or a real hang or segfault did. Treating it as infrastructure sends a real crash bug to a human
  after several runs; treating it as a test failure on the first crash sends overload crashes to the
  debugger (task 2365 measured about 8 wasted debugger cycles on task 2286) or the steward (4 merge-gate
  crashes from 2026-10-11 to 2026-11-10, all under heavy CPU pressure, none shown to be a real crash). So
  at both gates the first crash is re-run once (`verify_categories.py::BARE_CRASH_RERUNS = 1`, read by the
  workflow's infra-retry loop and by the merge lane), and a second crash is a real failure for the fix
  loop. `XDIST_WORKER_CRASH` is not infra-transient (C2), so a crash never reaches `infra_hold` or the
  merge lane's transient-infra hold, and the cross-check's fleet halt behaves as today. One renderer,
  `verify.py::repeated_crash_report(first, second) -> str`, writes the report both gates hand on: the test
  worker crashed twice with no failure output; the test running at each crash (the node ids in the crash
  notice, as `_xdist_crash_message` reads them today; one id twice points at that test); the archived logs
  of both runs; "likely a hang or crash bug in the code under test, not host load". The one risk it
  accepts: two overload crashes in a row are blamed on the change; how often that happens is not
  measured. Task 7044's merge-gate crash passed on a re-run 33 minutes later; an immediate re-run under
  the same load might not.

## Contracts

Field names beyond those shown are tactical. Every violation message names the invariant and the
offending values (INV-2).

### C1 — `VerifyContext` and `ModuleRegistry` (lands in β; consumers S6/S7 merge lane, S9 workflow, S8 harness, cli build it)

```python
# shared/src/shared/verify_admission.py
class ResourceClass(StrEnum):
    MERGE = 'merge'; TASK = 'task'; BACKGROUND = 'background'; OFFLINE = 'offline'
def is_gated_role(role: ResourceClass | str) -> bool        # semantics unchanged
def nice_prefix(role: ResourceClass | str) -> list[str]      # semantics unchanged

# orchestrator/src/orchestrator/verify_context.py
class WorktreeProvenance(StrEnum):
    FRESH = 'fresh'           # forced cold, merge-cold timeout track, never warm-marked (merge_gate only)
    MARKED = 'marked'         # cold iff .task/ exists without the marker; marked after a completed build
class VerdictAuthority(StrEnum):
    ADOPTABLE = 'adoptable'   # the merge gate's landable verdict; INV-1 no-evidence escalation applies
    ADVISORY = 'advisory'
class RetryOwner(StrEnum):
    VERIFY = 'verify'         # pure-timeout re-runs up to config.verify_timeout_retries
    CALLER = 'caller'         # one attempt

@dataclass(frozen=True)
class ModuleRegistry:
    modules: Mapping[str, ModuleConfig]           # read-only; {} means "discovery ran, no subprojects"
    @classmethod
    def of(cls, config: OrchestratorConfig) -> 'ModuleRegistry': ...      # config.module_configs_or_empty
    @classmethod
    def of_modules(cls, modules: Iterable[ModuleConfig]) -> 'ModuleRegistry': ...

@dataclass(frozen=True)
class VerifyContext:
    resource_class: ResourceClass
    worktree_provenance: WorktreeProvenance
    verdict_authority: VerdictAuthority
    retry_owner: RetryOwner
    module_registry: ModuleRegistry
    @classmethod
    def merge_gate(cls, registry) -> 'VerifyContext': ...     # MERGE, FRESH, ADOPTABLE, CALLER
    @classmethod
    def main_probe(cls, registry, *, resource_class: ResourceClass) -> 'VerifyContext': ...  # MARKED, ADVISORY, CALLER
    @classmethod
    def task(cls, registry) -> 'VerifyContext': ...           # TASK, MARKED, ADVISORY, VERIFY
    @classmethod
    def background(cls, registry, *, retry_owner: RetryOwner = RetryOwner.VERIFY) -> 'VerifyContext': ...  # BACKGROUND, MARKED, ADVISORY

class InvalidVerifyContext(ValueError): ...
```

Invariants (checked in `__post_init__`, so a hand-built value is checked too): `ADOPTABLE` requires
`MERGE`, `FRESH` and `CALLER`; `FRESH` requires `ADOPTABLE`; `OFFLINE` is never a verify context (the
offline lane does not call verify).

**Site table** (every `run_*` call, measured at `e634cfe658`: 22 production calls of the three entry
points, 16 of them inside verify.py; "today" is the behaviour the old kwargs produce, and every row
resolves the same after β):

| Site | Constructor | Cold rule | Timeout track | Attempts | Warm mark |
|---|---|---|---|---|---|
| `merge_lane/gates.py` per-module merge verify, `merge_lane/history_leg.py`, `verify_runner.py::LocalRunner.run_merge_verify`, `verify.py::_merge_gate_isolated_rerun` | `merge_gate` | forced cold | merge-cold | 1 | never |
| `verify.py::_whole_tree_main_baseline`, `_probe_modules_on_main`, `run_node_ids` (through `_node_id_run`, task 7918) | `main_probe(MERGE)` | marker (no `.task/` → warm) | warm | 1 | no-op |
| `verify.py::verify_failure_is_preexisting_on_main` (main verify), `_run_isolated_confirm_group(_observation)` | `main_probe(TASK)` | marker (warm-seeded `_mainprobe-`, no `.task/`) | warm | 1 | no-op |
| `verify.py::run_main_tip_sweep` (both passes), `_sweep_failure_reproduces_in_isolation` | `background(retry_owner=CALLER)` | marker | warm | 1 | no-op |
| `verify.py::confirm_main_tip_failure_is_real` (provisioning `role='task'`, verify through `_run_isolated_confirm_group`) | `main_probe(TASK)` | marker | warm | 1 | no-op |
| `workflow.py` post-rebase re-verify and infra-retry loop, and `task_verify_dispatch.py::TaskVerifyDispatcher`'s local leg, which serves the loop's call (tasks 6670/7977) | `task` | marker | cold or warm by marker | 1 + `verify_timeout_retries` | after a completed build |
| `task_verify_run.py::run_task_verify_on_worktree` | `task` | marker (fresh `.task/` → cold) | cold | 1 + retries | after a completed build |
| `review_checkpoint.py::ReviewCheckpoint._run_phase1` | `background()` | marker (main checkout, no `.task/`) | warm | 1 + retries | no-op |
| `evals/runner.py` | `task` | marker | by marker | 1 + retries | after a completed build |

Entry points after β and ι1 (signatures re-read at `e634cfe658`; `junit_report` and `admission_plan` are
tasks 7918's and 7977's):

```python
async def run_verification(worktree, config, module_config=None, *, host: VerifyHost,   # host: ι1, required
                           context: VerifyContext | None = None,
                           attempt_id=None, task_id=None, archive_root=None, segment_chained_test=False,
                           admission_request=None, junit_report=None) -> VerifyResult
async def run_scoped_verification(worktree, config, module_configs, task_files=None, *, host: VerifyHost,
                                  context=None, attempt_id=None, task_id=None, archive_root=None,
                                  force_workspace=False, event_store=None, admission_plan=None) -> VerifyResult
async def run_full_verification(project_root, config, *, host: VerifyHost, context=None, event_store=None,
                                force_rediscover=False) -> VerifyResult
async def run_node_ids(tree, config, module_configs, ids, *, host: VerifyHost,
                       concurrent_groups=False) -> Mapping[str, NodeIdStatus]
# removed: role, is_merge_verify, allow_cold_cache, max_retries.  context=None ≡ VerifyContext.task(ModuleRegistry.of(config))
# host has no default here or in any verify signature (D8, ruling S4-1)
```

`MergeVerifySpec.is_merge_verify` is removed; the remote leg builds `merge_gate(spec registry)`. The
summary record's `role` key and the event channel's `role` keep their string values
(`context.resource_class.value`), so `scripts/verify_budget_census.py` and the admission gate read the
same bytes. `verify_plan.derive_verify_plan(..., role: ResourceClass)` and
`effective_merge_module_configs(config, module_configs, registry)` take the context's values. The
task-verify ports (`task_verify_dispatch.py::TaskVerifyPort`, `LocalVerifyPort`, `_VerifyCall`) carry the
context in place of `role: Literal['task']`. `ResourceClass` is also the verify-result cache's role
vocabulary: `verify_cache/modes.py::VerifyRole` and `VERIFY_ROLES` (task 5652) go, the cache's
`VerifyCacheConfig.mode_for`, `policy.py` and `toolchain.py` take `ResourceClass` (its `MERGE`, `TASK` and
`BACKGROUND` members; values equal today's strings, so persisted toolchain records read the same bytes).
`verify.py::merge_verify_command_budget_secs(config, module_configs)` keeps its name, module and public
signature as the worker-side entry over S6's leaf `verify_budget.py::merge_verify_command_budget_secs`
(S6 C6).

### C2 — failure vocabulary and infra-transient data (lands in α; consumers S6 merge lane, S9 workflow, `task_verify_dispatch`)

```python
# verify_categories.py — FailureCategory gains the sentinels as members, each with a full CATEGORY_POLICY row
MERGE_FLAKE_SUPPRESSED, MERGE_NO_EVIDENCE, UNSCOPED_TYPECHECK_FAILED, UNSCOPED_TYPECHECK_TIMEOUT,
FLOCK_CONTENTION, XDIST_WORKER_CRASH, VERIFY_IO_FAULT
INFRA_ERRNOS: frozenset[int]; def is_infra_oserror(exc: object) -> bool      # moved from verify.py, public
BARE_CRASH_RERUNS: Final[int] = 1    # D17: the re-runs either gate gives a bare crash before blaming the change

# on VerifyResult (verify.py until μ0 moves it to verify_result.py)
@dataclass(frozen=True)
class InfraFault:                   # the warm-marker write is the one infra raise site left (D1)
    errno: int
    detail: str                     # the archived-log pointer text the exception used to carry
category: FailureCategory           # coerced from str at construction; an unknown string raises naming it
infra_faults: tuple[InfraFault, ...] = ()
@property
def infra_transient(self) -> bool: ...   # D2
@property
def bare_crash(self) -> bool: ...        # D2, the same per-leg rule

# verify.py
def repeated_crash_report(first: VerifyResult, second: VerifyResult) -> str: ...   # D17's one renderer
```

The new rows (`CategoryPolicy` has six fields and no defaults):

| Member | `severity_rank` | `archive` | `preexisting_probe` | `is_infra_transient` | `verdict_indeterminate` | `retry_kind` |
|---|---|---|---|---|---|---|
| `XDIST_WORKER_CRASH` | immediately after `INFRA_ABORT` | True | True | False | False | `NONE` |
| `VERIFY_IO_FAULT` | immediately after `DISK_FULL` | True | False | True | False | `NONE` |
| `MERGE_FLAKE_SUPPRESSED`, `MERGE_NO_EVIDENCE`, `UNSCOPED_TYPECHECK_FAILED`, `UNSCOPED_TYPECHECK_TIMEOUT`, `FLOCK_CONTENTION` | after `NONE`, in this order | False | True | False | False | `NONE` |

`XDIST_WORKER_CRASH` is not infra-transient (ruling S4-5). Its `preexisting_probe` is True because a
second crash takes the test-failure path, main-health probe included, exactly as a merge-gate crash does
today; its rank sits below `TEST_FAILURE`, so a genuine failure in another leg or module is the
aggregate's headline. Its `retry_kind` stays `NONE`: the one re-run belongs to the gates
(`BARE_CRASH_RERUNS`), not to `run_verification`. The sentinel rows reproduce today's unknown-string
treatment exactly: `should_archive` returns False on the `ValueError` today, an unknown string is not in
the skip or infra sets, and an unknown sorts below every listed entry. Inserting a rank renumbers the rows
below it by one; the relative order of every existing category is unchanged.
`INDETERMINATE_VERDICT_CATEGORIES` stays exactly `{INFRA_KILL}`: a bare crash is a regex match over output
and a hang is branch-causable, so the local crash still vetoes a remote green in the cross-check, and
because it is not infra-transient the fleet halt fires as it does for a test failure today (boundary row
7b). If task 7041 has landed when α runs, `NO_TESTS_RAN` joins the sentinel rows with the same values; if
α lands first, 7041 adds a member, not a string.

Invariants: `infra_faults` non-empty ⇒ `not passed`. The bare-crash precondition is unchanged (only the
test leg red, lint and type clean, no genuine failure marker); when it holds, in every context the merge
gate included, the test leg's recorded category is `XDIST_WORKER_CRASH` (D2). `infra_transient` and
`bare_crash` are never both True. `verify.py::_unmeasured_session_reason` reads an `XDIST_WORKER_CRASH`
leg as `session_truncated`, so the flake ledger's `unconfirmable_reason` for a merge-gate crash that
reaches the discriminator stays byte-identical, as `_unconfirmable_rerun_reason` promises. Error
semantics: no infra raise from any entry point; a programming error still propagates.
`_aggregate_results` concatenates children's `infra_faults`; its `infra_transient` and `bare_crash`
follow D2 over the merged legs.

### C3 — per-leg outcome and the wire codec (lands in γ; consumers S6 worker, verify-result-cache PRD)

```python
@dataclass(frozen=True)
class CheckRun:                        # the in-flight execution record
    leg: Leg                           # verify_evidence.Leg; replaces label: str; verify.py::LegSlot is deleted
    tool: ToolKind                     # computed once, where _tool_for_cmd(cmd) runs today
    cmd: str | None; rc: int; output: str; timed_out: bool; ...   # remaining fields unchanged
    def to_dict(self) -> dict: ...     # keeps the 'label' key, written as leg.value (bytes unchanged)

@dataclass(frozen=True)
class LegOutcome:                      # the ruled CheckRun-with-ToolKind as published on VerifyResult
    module_prefix: str | None          # None = the global commands
    leg: Leg
    tool: ToolKind
    rc: int
    timed_out: bool
    category: FailureCategory          # verify.py::classify_leg's answer; PASSED when rc == 0; XDIST_WORKER_CRASH for a bare crash
    cause_hint: str                    # tool-dispatched; '' when green
    failing_test_ids: tuple[str, ...] | None   # None = not attributable
    no_verdict_note: str | None        # killed / worker-death / unrun-segment note
    @classmethod
    def of(cls, check: CheckRun, *, module_prefix, category, cause_hint, failing_test_ids,
           no_verdict_note) -> 'LegOutcome': ...         # the only constructor production uses

legs: tuple[LegOutcome, ...] | None = None    # on VerifyResult; None = not recorded (fail-closed, never indeterminate)
# derived: failing_leg_categories (None when legs is None), failing_test_ids_by_module, test_leg_failed, timed_out_legs
def to_wire(self) -> dict: ...         # VerifyResult
@classmethod
def from_wire(cls, d: dict) -> 'VerifyResult': ...   # strict: unknown key -> TypeError naming it
```

Invariants, with `failing_legs` as in D2, checked at construction when `legs is not None`:

- `timed_out == (not passed and bool(failing_legs) and all(leg.timed_out for leg in failing_legs))` —
  today's `VerifyAttempt.pure_timeout_failure` and `_aggregate_results` rule, so a lint timeout beside a
  genuine test failure constructs with `timed_out` False;
- `passed == (all(leg.rc == 0 for leg in legs) and not infra_faults)`; the exceptions are exactly
  `MERGE_FLAKE_SUPPRESSED` (passed with red legs, kept as evidence by
  `verify.py::_merge_flake_suppressed_pass`) and `NO_TESTS_RAN` (failed with green legs, task 7041's
  outcome); a sentinel that overrides the verdict joins this list in the same edit;
- `failing_test_ids ⊇ ⋃ leg.failing_test_ids` over attributed legs.

`_aggregate_results` poisons `legs` the way it poisons `failing_leg_categories` and units today: the
merged value is None if any child's `legs` is None, so D2 never runs over a partial set.

`summary` is rendered from legs by one function; `_aggregate_results` never parses a child summary.
`failure_report` reads legs only: a result whose `legs is None` renders its summary and outputs without
per-leg sections. No-verdict is a leg fact: `RemoteRunner` reads `any(leg.no_verdict_note for leg in
result.legs or ())`, and `verify.py::is_no_verdict_summary` and its readers are deleted.
`remote_gate.wire_classes()` includes `LegOutcome` and `InfraFault`. `verify_runner.result_to_dict`/
`result_from_dict` delegate to the codec. A cache-hit unit (verify-result-cache PRD) has a `VerifyUnit`
and no `LegOutcome`: legs are what ran.

### C4 — the one scope engine (lands in δ and ε; consumers verify, verify-result-cache PRD)

`PlannedRun` gains `leg: Leg | None` (`verify_evidence.Leg`; None only for a SKIPPED/TRIVIAL run keyed to
no check), set where `_derive_module_runs` writes the reason prefix today. Executing a plan run selects the
`ModuleConfig` attribute from `leg`; no reason prefix is read anywhere: `verify_plan.py::planned_run_leg`
and its `_LEG_BY_REASON_PREFIX` table (task 5652) go, and both of its readers,
`verify.py::_mc_attr_for_run_or_none` and `verify_cache/keys.py`, read `run.leg`. After ε,
`derive_verify_plan` is the only producer of fallback commands; `run_scoped_verification` executes plan
runs on both branches, and the persisted plan's runs are the executed commands. `plan_hash` and
`verify_evidence.py::run_spec_hash` change once, at δ, because `leg` is a new execution-relevant run key
(`_execution_spec` denies only prose keys, so a new key can only make two hashes more distinct, never
collide them). The verify-result-cache registry exists since task 5652 (done, `957773c10d`) but runs mode
`off` and has no writer until task 5653 (pending); `verify_checkpoint.py` keys on the tip sha. So nothing
is invalidated in practice, and an entry recorded before δ would only miss once. `NoSourceDiffOutcome` per
D7.

### C5 — `VerifyHost` and the command runner (lands in ι1, moved in ι3, extended in κ; consumers verify modules, the boundary objects, `review_pre_test`, `evals/snapshots`, `harness`, `cli`, tests)

```python
# verify_exec.py (in verify.py until ι3)
@dataclass(frozen=True)
class CommandRequest:
    cmd: str; cwd: Path; timeout: float; env: Mapping[str, str] | None; log_path: Path | None
    scope: CgroupScope | None; clock_stop: ClockStopConfig | None
@dataclass(frozen=True)
class CommandOutcome:
    rc: int; output: str; timed_out: bool
class VerifyCommandRunner(Protocol):          # named apart from offline_lane.py::CommandRunner (OQ-5)
    async def run(self, request: CommandRequest) -> CommandOutcome: ...
class SubprocessRunner: ...                   # today's _run_cmd body, unchanged behaviour
def target_subprocess_env(...) -> dict[str, str]  # public (evals/snapshots, verify_runner import it)
async def reap_leftover_verify_scopes(...)    # public; harness.py calls it at startup

# verify_host.py (in verify.py until ι3)
@dataclass(frozen=True)
class VerifyHost:
    runner: VerifyCommandRunner
    stores: VerifyStores                      # κ: probe verdicts, main baselines, ruff latch, preprovision, admission executor and gates, prune throttle
    @classmethod
    def fresh(cls, *, runner: VerifyCommandRunner | None = None) -> 'VerifyHost': ...   # tests; in production only the composition roots (guard 2)
    @classmethod
    def process(cls) -> 'VerifyHost': ...     # the process slot (a ContextVar), filled once; called only by boundary objects and composition roots
    @classmethod
    def installed(cls, host: 'VerifyHost') -> ContextManager['VerifyHost']: ...   # the per-test reset; closes host.stores on exit

# every verify function that runs a command or reads a store:
async def <verify function>(..., *, host: VerifyHost, ...)   # required; never a default (guard 1)

# the boundary objects (outside the verify modules), each resolving the process host once:
TaskWorkflow(..., verify_host: VerifyHost | None = None)        # None -> VerifyHost.process() at construction
SpeculativeMergeWorker(..., verify_host: VerifyHost | None = None)
ReviewCheckpoint(..., verify_host: VerifyHost | None = None)
```

Invariants: no verify signature (`orchestrator/src/orchestrator/verify*.py` and `task_verify_*.py`) has a
`host` default and no verify function calls `VerifyHost.process()` (guard 1, an AST scan); no production
module outside `harness.py`, `cli.py` and the module defining `VerifyHost` calls `VerifyHost.fresh()`
(guard 2); both guards flag a seeded fixture module in their own tests (INV-10). Admission waiters reach a
gate through the host they were given, `host.stores.admission_gate(slots_dir)` (the successor of
`verify_admission_gate.py::gate_for`), and `verify_admission_watch.py::AdmissionWatch` is built with the
host `Harness` gives the workflow, worker and checkpoint, so the watcher's heartbeat and aged-waiter alarm
see the same queue the waiters join; no verify module keeps module-level mutable state other than the one
`ContextVar` (κ's guard); every store states its bound (TTL evicted on write, or a fixed capacity); two
`fresh()` hosts share nothing; `VerifyStores.close()` releases the admission executor. Segmented legs call
`runner.run` per segment.

### C6 — `attempt-*` artefact grammar (lands in λ; consumer `scripts/verify_budget_census.py`, S25)

```python
# verify_artifacts.py — stdlib only
class ArtifactKind(StrEnum): LOG; SUMMARY; PLAN; JUNIT; DISPATCH; STDERR
@dataclass(frozen=True)
class ArtifactName:
    attempt: str          # 'N' or the archive attempt id
    module_prefix: str | None
    origin: str | None    # e.g. remote runner name
    label: str | None
    segment: str | None
    kind: ArtifactKind
    stamp: str | None
    def render(self) -> str: ...
    @classmethod
    def parse(cls, name: str) -> 'ArtifactName | None': ...
def sanitize_prefix(prefix: str) -> str: ...
ARCHIVE_STAMP_FORMAT: str           # '%Y%m%dT%H%M%S_%fZ'
def archive_stamp(now: datetime | None = None) -> str: ...
```

Invariants: `parse(render(n)) == n` for every value; the streamed partial log and the persisted log of the
same leg render the same `ArtifactName`, so `_persist_attempt_logs(skip_if_exists=True)` keeps a killed
leg's streamed log by construction; every summary record carries `task_id` and `module_prefix`.

### C7 — the merge-verify profile and runner dispatch (lands in η and θ; consumer S6 merge lane)

```python
# verify_spec_shared.py
@dataclass(frozen=True)
class ModuleVerifyProfile: ...      # every verdict-relevant ModuleConfig field; lossless both ways
@dataclass(frozen=True)
class AppliedSpec:
    config: OrchestratorConfig; registry: ModuleRegistry; modules: tuple[ModuleConfig, ...]
def apply_merge_verify_spec(config: OrchestratorConfig, spec: MergeVerifySpec) -> AppliedSpec: ...

# verify_runner.py
async def dispatch_merge_verify(runner: VerifyRunner, spec: MergeVerifySpec, *,
                                fallback: LocalRunner | None, event_store, task_id, archive_root,
                                on_unavailable: RunnerUnavailableSink | None,
                                attempt: int = 0, depth: int | None = None, speculative: bool | None = None,
                                chain_items: int = 1, chain_build_ms: int | None = None,
                                dispatch_facts: MergeVerifyDispatchFacts | None = None) -> VerifyResult
```

The forwarded fields are exactly those `merge_lane/worker.py`'s `pool.dispatch` calls pass today
(re-read at `e634cfe658`; `dispatch_facts` is task 8523's `merge_verify_facts.py` record, which the
`merge_verify` event keeps carrying).
Invariants: `profiles_from(modules)` round-trips to equal `ModuleConfig`s; `apply_merge_verify_spec` never
mutates its input config; `LocalRunner` and `run_merge_verify_on_worktree` both run from an `AppliedSpec`;
host selection, quarantine and PARKED live only in `HostAllocator`; `ModuleVerifyProfile` joins
`remote_gate.wire_classes()`.

## Boundary-test sketch

Producers land their own rows; π lands the rows marked ★ (both sides of a cross-module seam) and re-runs
the set.

| # | Contract | Side | Scenario | Pre | Post |
|---|---|---|---|---|---|
| 1 | C1 | verify | build `VerifyContext(MERGE, MARKED, ADOPTABLE, CALLER, r)` | — | `InvalidVerifyContext` naming the fields |
| 2 | C1 | verify | config-only diff: `merge_gate` vs `main_probe(resource_class=MERGE)` | module with a test command | merge gate: `FULL_GATE`, one `trivial_pass_escalated(full_gate)`; probe: trivial pass, no event |
| 2b | C1 | verify | every row of the C1 site table | scripted runner, `.task/` per row | resolved cold rule, timeout and attempt count equal the pre-β values captured by the leaf |
| 3★ | C1 | remote ↔ dispatcher | `orchestrator verify-merge` with spec registry {a}, host discovering {a, b} | real fixture repo | verifies {a} only; the caller's config has the same registry before and after |
| 4 | C2 | workflow | bare xdist crash on a task verify, then a green re-run | scripted runner, lint/type rc 0 | first result `category=XDIST_WORKER_CRASH`, `bare_crash` True, `infra_transient` False, no `InfraFault`, no exception; one re-run consuming no verify attempt; the task proceeds; no debugger, no `infra_hold` |
| 4a | C2 | workflow | bare crash twice on a task verify | scripted runner | after one re-run the second result reaches `_verify_debugfix_loop`; the debugger's failure context is `repeated_crash_report(first, second)`, naming both crashed node ids and both runs' archived logs; never `infra_hold` |
| 4b | C2 | workflow | bare crash twice on the post-rebase re-verify; separately a warm-marker errno there | `pre_merge_reverify: true` | crash: one re-run, then `_post_rebase_reverify_passed` is False with the `post_rebase_verify_fail` waste event, no block; errno: `_rebase_before_enqueue` blocks `infra_issue` with `escalate_to_human` |
| 4c | C2 | review checkpoint | phase 1 crashes; separately phase 1 is infra-transient | — | both: `ReviewOutcome.SKIPPED_NO_VERDICT`, trigger kept, no reviewer dispatched |
| 4d | C2 | evals | the eval runner's verification crashes; separately it is infra-transient | — | both: the cell is `Unmeasured`, not a failed score |
| 5★ | C2 | merge lane ↔ verify | bare crash at the merge gate, then a green re-run | real `LocalRunner` | one in-place re-run on the held host (its own count, not the shared infra budget); the merge lands; no steward, no hold |
| 5b★ | C2 | merge lane ↔ verify | bare crash twice at the merge gate | real `LocalRunner` | the second crash takes the branch-failure path: the discriminator refuses with `session_truncated` (ledger reason byte-identical), the main-health probe runs, the block detail carries `repeated_crash_report`, the steward is dispatched; no `TRANSIENT_INFRA_REASON_PREFIX` hold |
| 6 | C2 | probe | the main probe's own verify crashes; separately it is infra-transient | — | both: `(False, '')`, WARNING naming the crash or fault, verdict not cached |
| 6b★ | C2 | baseline probe | a bare crash in a baseline probe | real `_on_main_probe_worktree`, scripted runner | answer `None`; `cached_main_baseline_failing_ids(main_sha)` is None afterwards |
| 7 | C2 | merge lane | type leg `ENV_TRANSIENT` + genuine test failure | — | `infra_transient` False, branch failure; the cross-check gate unchanged |
| 7b★ | C2 | merge lane ↔ verify | local `XDIST_WORKER_CRASH` versus a remote PASS in the cross-check | two runners | the cross-check vetoes without re-running the local leg; the fleet halt and the blocking escalation fire as for a test failure today; no `verify_cross_check_inconclusive` event |
| 7c | C2 | verify | `MERGE_NO_EVIDENCE` result, no legs | — | `infra_transient` False, `bare_crash` False |
| 7d | C2 | verify | two-module aggregate: crash in A, `TEST_FAILURE` in B | — | `infra_transient` False, `bare_crash` False; aggregate category `TEST_FAILURE` |
| 7e | C2 | verify | warm-marker errno after green legs | — | `VERIFY_IO_FAULT`, `infra_transient` True, one `InfraFault` |
| 8 | C3 | report | lint-only failure, green test leg, `0 errors` type leg | — | `failure_report` has a lint section; no test-failure, test-output or type-error section |
| 8b | C3 | verify | lint timeout + genuine test failure | — | constructs; `timed_out` False |
| 8c | C3 | verify | flake-suppressed pass | red legs retained | constructs; `passed` True with `MERGE_FLAKE_SUPPRESSED` |
| 8d | C3 | verify | no-tests-ran result (once 7041 has landed) | green or skipped legs | constructs; `passed` False with `NO_TESTS_RAN` |
| 9★ | C3 | remote ↔ dispatcher | a result with legs and an `InfraFault` crosses `verify-merge` stdout; a remote result with a killed leg | real subprocess | `from_wire(to_wire(r)) == r`; local and gate `wire_fingerprint` agree and include both classes; the killed-leg result raises `RunnerUnavailable(kind='no_verdict')` on the dispatcher |
| 10 | C4 | plan | characterization corpus of fallback diffs (dark-factory, reify-shaped, scoping-intake-shaped configs) | fixtures captured before ε deletes the engine | plan-rendered commands equal the captured ones, except the listed deltas |
| 11 | C4 | verify | config-only merge diff on a module-config project and on a zero-module project | — | identical `NoSourceDiffOutcome` and identical `trivial_pass_escalated` payloads |
| 12 | C4 | archive | fallback-branch verify | real worktree | `attempt-N.plan.json` runs equal the summary's executed commands |
| 13 | D9 | sweep | sweep failure on a worker-death-truncated session | — | `unconfirmable('session_truncated')`, alarm not suppressed; `_mainsweepconfirm-` worktree minted by `ephemeral_worktree` and in `GitOps.protected_prefixes()` |
| 14 | C5 | tests | `run_scoped_verification(..., host=VerifyHost.fresh(runner=Scripted))` whose verify calls a nested probe | no patch | scripted outcomes drive the verdict, the nested probe included; two fresh hosts share no cache |
| 14b | C5 | guards | guard 1 over a seeded verify module with `host: VerifyHost \| None = None` and one calling `VerifyHost.process()`; guard 2 over a seeded production module calling `VerifyHost.fresh()` | fixture modules | each guard flags its seed and passes the tree |
| 14c★ | C5 | harness ↔ verify | `Harness` built with its default host | real `Harness` wiring | the workflow, worker, checkpoint and admission watch it builds hold the same `VerifyHost` instance as `VerifyHost.process()` |
| 15 | C5 | stores | baseline seeded at sha2 after the TTL | sha1 seeded earlier | sha1 evicted on the write; downgrade counter capped |
| 15b | C5 | admission ↔ watcher | a waiter queued through the host `Harness` gives the workflow | real `AdmissionWatch` built by `Harness` | the waiter appears in `AdmissionWatch.tick`'s `verify_admission_heartbeat` payload |
| 16 | C6 | census | mixed corpus: pre-λ archives, pre-γ and post-γ summaries, and new summaries | real archive tree | each record counted once; a post-γ summary is counted with the same leg label; `task_id`/`module_prefix` read from the record when present |
| 17 | C7 | spec | every `ModuleConfig` in `dark-factory-orchestrator.yaml` | — | `modules(apply(build(spec))) == originals`; a module-level `verify_env` reaches the remote leg |
| 18★ | C7 | merge lane ↔ runner | spec lacks a module that live config has | — | the local leg verifies the spec's set, the same as the remote leg |
| 19★ | C7 | drift ↔ allocator | drift FAIL on a remote | two hosts | `HostAllocator` quarantines it once; the next acquire skips it; no pool-held quarantine; the drift escalation's action names `clear_verify_host_quarantine` |
| 20★ | layering | all | import every `verify_*` module in a fresh interpreter, and an AST scan of their imports | — | no verify module imports one listed below it in §Sketch; no function-local import among them |
| 21 | D15 | CLI ↔ runners | `orchestrator verdict-parity --sha <s>` where the local and a remote verdict agree; again where they disagree | real `LocalRunner`, a remote runner over a fixture repo | agreement: report printed, exit 0; disagreement: report names the SHA and both verdicts, exit non-zero; the minted worktree is removed both times |

## Cross-PRD relationship

| Other PRD / stream / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S6 merge-lane lifecycle (draft) | produces (to) | C1 contexts at `merge_lane/gates.py`, `history_leg.py`, `ports.py::run_scoped`; C2 `infra_transient` branch in `worker.py`; C7 `dispatch_merge_verify` at the pool sites; `drift.py`/`shadow.py` pool construction; μ0's `VerifyResult` import repoint in `disposition.py`, `gates.py`, `history_leg.py`, `ports.py`, `shadow.py`, `types.py`, `worker.py` | S4 | Lead ruling (S6 critic3 N1, 2026-11-02): S6's typed-facts leaves land first on `merge_lane/worker.py` and `merge_lane/gates.py`, so S4 β, S4 θ and S4 μ0 each depend on S6 ζ (S6 C2, `ItemLifecycle` operative), and S4 leaves edit these sites minimally on top. Lead ruling (S6 critic4, same-file residuals; these are the last S6 leaves on those files): S4 α depends on S6 ζ (both edit `worker.py`'s infra-transient region); S4 θ depends on S6 λ (`worker.py` pool sites, `drift.py`, `shadow.py`); S4 μ0 depends on S6 μ (the `VerifyResult` import repoint across `worker.py`, `types.py`, `gates.py`, `shadow.py`). S6 κ depends on S4 β (C1): S6 D12's `_run_post_merge_verify(ctx: VerifyContext…)` consumes C1. S6 α needs no S4 artefact: it moves `_resolve_verify_timeout` into its leaf `verify_budget.py`, and `verify.py::merge_verify_command_budget_secs` keeps its name, module and signature as the worker-side entry (C1, S6 C6); ν imports `resolve_verify_timeout` from S6's leaf and moves nothing of it. C1 `VerifyContext` lands in S4 β, so any S6 leaf that needs it depends on S4 β only. Sitting 2026-11-10, S6-2 = B: task 5040 finishes first and S6's lifecycle task (ζ) waits for it, so every `worker.py` site above is edited where 5040 put it (`merge_lane/verify_dispatch.py` and its sibling modules), read at the leaf's base. ι1 (the `host` argument at the gates, the history leg, the verifier-port calls and the `LocalRunner`/`HistoryLegRunner` constructions) and ρ (the cross-check action text) now edit these files too; both follow S6 ζ through α and β, and the lead places them in the per-file serial order with S6 κ, λ and μ at decompose. |
| S9 workflow spine (draft) | produces (to) | C2: α removes the `except VerifyInfraError` branch of `TaskWorkflow._run_scoped_verification_with_infra_retry`, adds the one bare-crash re-run there (D17), removes the `VerifyInfraError` arm of `TaskWorkflow.run` and the `workflow_types.py` row, makes `_post_rebase_reverify_passed` typed, and repoints `_is_infra_oserror` to `verify_categories.is_infra_oserror`; C1: β edits the two `run_scoped_verification` call sites; C5: ι1 gives `TaskWorkflow` its `verify_host` (resolved once at construction) and passes it at both sites and to the preexisting probe | S4 | S9's κ and λ consume C2 and depend on α; S9 λ's signal (`_is_infra_oserror` gone from `workflow.py`) is delivered by α's repoint — the lead re-anchors that check; ι1's `workflow.py` edit joins the per-file serial order |
| S8 harness | produces (to) | `review_checkpoint.py` builds `VerifyContext.background` (β) and takes the no-verdict skip path (α); `harness.py` resolves `VerifyHost.process()` once and passes it to the `TaskWorkflow`, `SpeculativeMergeWorker`, `ReviewCheckpoint` it builds and to its own `run_main_tip_sweep`/`confirm_main_tip_failure_is_real` calls (ι1), builds `AdmissionWatch(host)` (κ), calls `reap_leftover_verify_scopes` (ι3) and imports the sweep and confirm from their new module (ξ) | S4 | same-file edges with S8 β on `review_checkpoint.py` and with S8's `harness.py` leaves at decompose; when S8's composition-root task moves the harness wiring it carries the one process host, obtained by `VerifyHost.process()` (never `fresh()`, D8), with the other collaborators it builds (one more constructor argument, no new seam); S8 β depends on S4 β and ι3, so it already follows ι1's host resolution in `harness.py`; κ (`AdmissionWatch(host)` in `harness.py`) also follows ι3 and is unordered against S8 β, so the lead places the two in the per-file order |
| S10 task-write contract | adjacent | touches `workflow.py` and `harness.py` | S10 | same-file edges at decompose only |
| S5 GitOps and the git runner (draft) | adjacent | S5 α repoints verify's function-local `git_ops._run` imports (`_derive_task_files_from_git`, `_verify_pipeline_guard_requires_full_gate`, `confirm_main_tip_failure_is_real`); S5 ξ moves `GitOps.ephemeral_worktree`/`WorktreeKind` to `worktrees.py` | S5 | ζ deletes the confirm's hand-rolled worktree and adds `WorktreeKind.MAIN_SWEEP_CONFIRM`: same-file edges, whichever lands second adapts; no contract dependency |
| S25 scripts instruments | produces (to) | C6 parsed by `scripts/verify_budget_census.py` | S4 | λ depends on 6765 (S25 κ); λ's `_checkout` import follows S25 ε (6760) |
| S18 evals measurement | consumes | `Unmeasured` (task 6734, done) for an infra-transient or bare-crash eval verification (D16) | S18 | no dependency |
| S16 config shape / S19 private seams | adjacent | `load_config`'s discovery write stays; β publishes `discover_module_configs` | S16/S19 | xref only |
| S7 gates and target adapters | adjacent | none since the sitting: S7-3 = B deletes reify retry narrowing end to end, so `build_retry_env` and its consumer row are gone (the narrowed-retry arm of the merge lane's in-place retry block that α edits goes with S7's deletion task, which lands before S6's worker leaves and so before α) | S7 | same-file only, through S6 |
| `plans/verify-result-cache-prd.md` (5652 done at `957773c10d`; 5653–5657 pending) | adjacent | consumes `VerifyResult.units`/`plan_hash` and the fan-out seam in `run_scoped_verification`; its landed `verify_cache/` reads `verify_plan.planned_run_leg` (δ replaces it with `PlannedRun.leg`) and spells the role vocabulary `VerifyRole` (β replaces it with `ResourceClass`) | that PRD for the cache decision; S4 for result shape, plan and the role vocabulary | same-file edges with 5653/5655 at decompose; C1, C3 and C4 notes above |
| Task 5291 (pending, medium; absorbed 3229, 4740, 4300) | adjacent | stores a quarantine reason so a divergence-benched host is never readmitted on ssh reachability; its readmission rule is the verdict-parity re-proof ρ makes runnable | 5291 | same-file edges with θ (`verify_runner.py` quarantine API) and ρ (the cross-check action), direction by status at decompose; its `merge_queue.py` anchors are stale (the code is in `merge_lane/` and, after 5040, `verify_dispatch.py`) |

## Existing tasks (contract §8 step 2, confirmed with `get_task`; statuses re-read 2026-11-10)

| Task | Status | Relation | Action |
|---|---|---|---|
| 5506 | pending, low | per-leg infra routing; subsumed by C2's `infra_transient` (D2, ruling S4-3) | cancel → α |
| 7001 | pending, low | the bare crash's own `FailureCategory` row; deletes `_archive_attempt_log(..., infra_reclassified=)` | cancel → α (adjudicated per C2 and ruling S4-5: `is_infra_transient=False`, `verdict_indeterminate=False`) |
| 6652 | pending, low | item 1 (sweep) moves onto `infra_transient` (α); residual items 2 (preexisting-probe skip set) and 3 (cross-check gate, human adjudication) | depends on α |
| 5892 | pending, low | per-leg verdicts in `failure_report`; subsumed by C3 | cancel → γ |
| 7079 | pending, low | unspawnable-tool no-verdict note | depends on γ |
| 5296 | pending, medium | widens the source-file classification that decides the no-source-diff short-circuit | depends on δ |
| 7041 | pending, high | fallback test floor | ε depends on 7041 |
| 7066, 7211 | pending, low | fallback narrowing and reprojection fixes | depend on ε (one site after ε) |
| 6249 | pending, low | sweep confirm on a crash-truncated session; delivered by D9 | cancel → ζ |
| 6506 | pending, low | target-aware `--timeout` injection; one builder after ζ | depends on ζ |
| 5089 | pending | per-task call site in the rerun table | depends on ζ (adds one `RerunSite` row) |
| 5679 | pending | merge-gate isolated re-run as a `VerifyCmd` | depends on ζ |
| 7069 | pending, low | env overlay for `ephemeral_worktree` verifies | depends on ζ |
| 4546 | deferred, low | `x_coalesced_into: 5295` ("RETIRED into 5295 — do not undefer") | cancel: coalesced into 5295 (done); no residual |
| 7465 | pending, low | `scope_cargo` and `git.merge_config_only_full_gate_globs` on `MergeVerifySpec` | η depends on 7465 |
| 7067 | pending, low | carry `verify_allow_no_tests_ran` through both specs | depends on η; cancel if η's census carried it |
| 3229 | deferred, coalesced into 5291 | RunnerUnavailable-class versus divergence-class quarantine conflation | none: 5291 carries it |
| 5291 | pending, medium (depends on 5036, done) | quarantine-reason split (from 3229), `verify_cross_check_remote_green` default flip, cross-check categories | same-file edges with θ and ρ, direction by status at decompose; θ does not widen the conflation it fixes (D11) |
| 5040 | pending, high, plan authored, half-built branch | extracts `merge_lane/verify_dispatch.py` and siblings from `worker.py`, including the worker's reads and writes of the harness-built quarantine set | finishes first (S6-2 = B); S4's worker-site leaves follow it through S6 ζ (§Pre-conditions); θ routes the accesses 5040 moved through `HostAllocator`, the set's only writer (D11) |
| 8734 | pending, low (review suggestion from 6670) | `build_remote_runners`' `quarantine` parameter has no production caller | cancel → θ (θ deletes the parameter and its filter tests, D11) |
| 8731 | pending, low (review suggestion) | `RunnerReason.REMOTE_QUARANTINED` is also stamped by `VerifyRunnerPool.dispatch`'s local fallback | depends on θ (the stamp moves into `dispatch_merge_verify`) |
| 7038 | pending, medium | remote verify-host disk gate in `HostAllocator`/`VerifyRunnerPool`, `cli.py::verify_merge` and the worker's ENOSPC branch | same-file edge with θ, direction by status at decompose |
| 7913 | pending, low | `review_pre_test` imports private `_run_cmd`; moves onto C5 | cancel → ι3 |
| 7339 | pending, low | structural tag on `_run_cmd`'s scope-spawn failure | same-file edge with ι3, direction by status |
| 5292 | pending, medium, unclaimed | probe and baseline hardening (fixture reset, skipped-ids TTL, three-state probe answer) | depends on α; κ depends on 5292 and keeps its eviction policies |
| 5673 | pending, low | move archive writers out of verify.py; duplicate of λ | cancel → λ |
| 5672 | pending, low | archive hygiene (eviction order, task_id sanitising) | depends on λ |
| 6765 | pending, low | S25 κ census policy | λ depends on 6765; same-file edge with ι2 |
| 5895 | pending, low | extract the pytest/xdist output cluster, moving `WORKER_DEATH_SUMMARY_MARKER` with it | cancel → μ |
| 6435 | pending, low | baseline into its own module; delivered by ξ | cancel → ξ |
| 3678 | pending, high, dispatchable | same-signature guard in the infra retry loop α edits | α depends on 3678 |
| 7929, 7666 | pending, low | census report and census test-file split | same-file edges with λ and ι2 |
| 3962, 7295, 6650, 6461, 5294 | pending | edit verify.py/verify_runner.py for other findings | same-file only; the scheduler's file lock serializes them |
| (new) | — | `offline_lane_runs.py::launch` bounds the leg's wall clock from config, `x_finding_key=fk-591206484ea5` | lead files one mechanical task |

## Decomposition plan

Overlay bands: target 300–1,500 changed LOC (a move counts by moved region), ≤ 10–12 declared files; the
leaves over 15 files are one mechanical shape and say so. Same-file leaves are serialized by real edges.
Every leaf stamps `x_finding_run = review-all-dark_factory-20261008`, `source = review-all`, file-level
`metadata.files`, and `x_finding_key` for the keys it closes (leaves with no key carry none and take
priority medium). **High leaves (gate 6698 dependencies at decompose): γ, ε, η, ι2, ξ.** Signal leaf: π.
Every leaf touching `merge_lane/**`, `git_ops.py` or `orchestrator/tests/conftest.py` re-runs
`orchestrator/tests/merge_lane_ratchet_baseline.json`'s guard (cognitive-complexity and file-size
ceilings, ratcheted `patch_targets`/`private_reads`); every leaf editing a line that carries an inline
suppression marker gives it an INV-12 disposition (`scripts/inline_suppression_key.py` keys markers on the
line digest; `verify.py` carries 21, measured at `e634cfe658`).

Chain (ruling S4-7): α → β → γ → δ → ζ → ι1 → ι2 → ι3 → κ → λ → μ → μ0 → ν → ξ → ε; η after ι1 (both edit
`verify_runner.py`'s `LocalRunner` and the worker's runner construction) and γ; θ after η; ο after ξ, θ and
λ; ρ after ο; π after ε, ο and ρ. Cross-stream (lead rulings): α, β, θ and μ0 depend on S6 ζ; θ also on
S6 λ; μ0 also on S6 μ (and on S6 λ through ι3 → θ); S6 κ depends on S4 β (labels confirmed by the lead,
2026-11-10). Through S6 ζ every leaf that edits a `merge_lane/worker.py` site lands after task 5040
(S6-2 = B), and α, the head of the chain, waits on 5040 and S6's chain to ζ: every S4 leaf does.

**α — Failure vocabulary and infra-transient as data; the bare-crash re-run rule; every entry point branches; retire VerifyInfraError** (C2, D16, D17; priority medium; ~1,700 LOC)
- Files: `orchestrator/src/orchestrator/verify_categories.py`, `verify.py`, `verify_runner.py`,
  `workflow.py`, `workflow_types.py`, `merge_lane/worker.py` (or, at the leaf's base, the
  `merge_lane/verify_dispatch.py` family task 5040 moved `_run_post_merge_verify` into),
  `task_verify_dispatch.py`, `flake_ledger.py`, `harness.py` (one comment), `review_checkpoint.py`,
  `evals/runner.py`; the 10 test files referencing `VerifyInfraError` (`test_block_disposition.py`,
  `test_flake_discriminator.py`, `test_flake_ledger.py`, `test_task_verify_dispatch.py`, `test_verify.py`,
  `test_verify_admission_order_integration.py`, `test_verify_env_transient.py`,
  `test_verify_merge_flake_suppression.py`, `test_verify_runner.py`, `test_workflow_verify_infra_resume.py`;
  83 references, 51 in the last, measured at `e634cfe658`) and the 12 files constructing `VerifyResult`
  with non-member category literals (`test_ephemeral_worktree.py`, `test_flake_recorder.py`,
  `test_merge_disposition.py`, `test_merge_queue_flake_recorder.py`, `test_merge_queue_verify_retry.py`,
  `test_multihost_verify_integration.py`, `test_train_moved_main.py`, `test_verify_main_tip_sweep.py`,
  `test_verify_plan_integration.py`, `test_verify_preexisting_main_break.py`, `test_verify_runner.py`,
  `test_workflow_verify_retry.py`; 18 sites), all under `orchestrator/tests/`. Review trigger acknowledged:
  the test edits are one shape (`pytest.raises(VerifyInfraError)` → an asserted result; a literal → a
  member). The leaf is above the 1,500-line band by the 100–200 lines the S4-5 ruling measured for the
  crash rule; it stays whole by lead ruling (2026-11-10: one coherent contract change; a split would
  leave a half-typed result in the tree).
- Keys: `fk-5037fafc755c`, `fk-09ad1a19f587`. Deps: out of batch 3678 (dispatchable, same function);
  cross-stream S6 ζ (lead ruling: both edit `merge_lane/worker.py`'s infra-transient region).
- Work: C2 and the full row table; D2, reading `failing_legs` from `failing_leg_categories` until γ, and
  rewriting the test leg's entry to `XDIST_WORKER_CRASH` when the bare-crash precondition holds, in every
  context; `_unmeasured_session_reason` reads that entry as `session_truncated` (C2); D17: the one re-run
  at both gates (`BARE_CRASH_RERUNS`), the second crash's fall-through to the debugger or the merge lane's
  branch-failure path, and `repeated_crash_report`; D16 for every entry point, including the baseline
  probes' return-`None` before any seed or remember and `TaskVerifyDispatcher._remote_verdict` on
  `result.infra_transient`.
- Deletes: `VerifyInfraError`, both raise sites, the `except` and `isinstance` arms, the `BlockDisposition`
  row, the 8 restatements, `_assert_sentinels_disjoint`, the 4 sentinel string constants and the
  `'merge_flake_suppressed'` literals, the `infra_reclassified` parameter of `_archive_attempt_log`,
  `_xdist_crash_message` (its crashed-id and log-path reading moves into `repeated_crash_report`),
  `_is_infra_oserror` in verify.py (moved, public), 8 of the 9 aggregate `INFRA_TRANSIENT_CATEGORIES` reads.
  No comment or docstring may name `VerifyInfraError` afterwards (D14).
- Signal: boundary rows 4–7e; on main after merge, a task whose verify hits a bare xdist crash logs one
  re-run and then either proceeds or reaches the debugger with the repeated-crash report, and a merge
  whose gate crashes twice is blocked to the steward with that report; no infra hold names a crash, and
  `git grep VerifyInfraError orchestrator/src` is empty.
- G7: INV-11 — the main probe's no-verdict outcome still reads `(False, '')`, waived for this leaf with
  owner 5292 (its three-state answer), logged at WARNING and not cached meanwhile; `_probe_modules_on_main`
  and `_whole_tree_main_baseline` refuse to seed (row 6b), so no new fail-soft is introduced. INV-5/INV-9 —
  one predicate replaces 8 reads, and one renderer writes the crash report both gates hand on. INV-2 —
  category coercion names the bad string; the crash report names both crashed tests and both runs' logs.
  Ratchet baseline and INV-12 markers per the band note.

**β — VerifyContext and ModuleRegistry replace the context kwargs and the private registry channel** (C1; medium; ~1,500 LOC)
- Files: new `verify_context.py`; `shared/src/shared/verify_admission.py`; `verify.py`, `verify_plan.py`,
  `verify_runner.py`, `verify_spec_shared.py`, `verify_budget.py` (S6's leaf; only if S6 α kept the
  `is_merge_verify` spelling the lead ruled out, D4), `task_verify_run.py`, `task_verify_dispatch.py` (the two ports and `_VerifyCall`),
  `verify_cache/modes.py`, `verify_cache/policy.py`, `verify_cache/toolchain.py` (`VerifyRole` →
  `ResourceClass`), `config.py` (`discover_module_configs`; `VerifyCacheConfig.mode_for`), `defaults.yaml`
  and `event_store.py` (comments), `merge_lane/gates.py`, `merge_lane/history_leg.py`,
  `merge_lane/ports.py`, `merge_lane/worker.py` or 5040's modules (comments), `workflow.py` (2 call sites),
  `review_checkpoint.py`, `evals/runner.py`; new `orchestrator/tests/test_verify_context.py`; the test files
  calling an entry point with a removed keyword (30 by AST at `e634cfe658`: role 173, max_retries 78,
  is_merge_verify 26, allow_cold_cache 3 call sites — `test_merge_history_leg_runner.py`,
  `test_remote_verify_dispatch.py`, `test_task_verify_integration.py`, `test_verify.py`,
  `test_verify_admission_integration_gate.py`, `test_verify_admission_order_integration.py`,
  `test_verify_admission_pytest_n.py`, `test_verify_admission_wiring.py`, `test_verify_cancel_unwind.py`,
  `test_verify_cold_preprovision.py`, `test_verify_env_recovery_attempt.py`, `test_verify_env_transient.py`,
  `test_verify_event_channel_wiring.py`, `test_verify_events_units.py`, `test_verify_load_stamp.py`,
  `test_verify_plan_latch.py`, `test_verify_pyright_interpreter_misresolution.py`,
  `test_verify_pytest_n_operator_cap.py`, `test_verify_reverse_dep.py`, `test_verify_ruff_config_boundary.py`,
  `test_verify_scope_cpu_weight.py`, `test_verify_scope_inversion_boundary.py`, `test_verify_scope_kappa.py`,
  `test_verify_scope_lambda.py`, `test_verify_scope_memory.py`, `test_verify_scope_spawn_failure.py`,
  `test_verify_segmented_fallback.py`, `test_verify_sequential_lint_first.py`, `test_verify_summary_facts.py`,
  `tests/scripts/test_verify_budget_census.py`)
  plus the files that reference `is_merge_verify` otherwise (`test_atomic_train_merge.py`,
  `test_flake_discriminator.py`, `test_merge_queue.py`, `test_merge_queue_deep_integration_gate.py`,
  `test_train_integration.py`, `test_verify_merge_flake_suppression.py`, `test_verify_preexisting_main_break.py`,
  `test_verify_runner.py`, `test_warm_lane_pool.py`, `tests/scripts/test_module_verify_budgets.py`,
  `tests/scripts/test_scripts_module_config.py`); orchestrator test paths under `orchestrator/tests/`.
  Review trigger acknowledged: one mechanical shape (kwargs → context) across the files.
- Keys: `fk-0bb03714372a`, `fk-0c5e73f13485`. Deps: α; cross-stream S6 ζ (lead ruling: S6 lands first on
  `merge_lane/worker.py` and `merge_lane/gates.py`).
- Deletes: the `role`/`is_merge_verify`/`allow_cold_cache`/`max_retries` parameters (24 + 5 + 1 + 5 defs,
  measured at `e634cfe658`; `verify_budget.py`'s keyword is already `merge_cold` by the lead ruling, D4), the
  3 conjunctions, `MergeVerifySpec.is_merge_verify`, `verify_cache/modes.py::VerifyRole` and
  `VERIFY_ROLES`, the `_module_configs` write in `apply_spec_to_host_config`, `run_full_verification`'s
  private reads, verify's 3 function-local imports of `config._discover_module_configs`, and every comment
  naming `is_merge_verify` (D14).
- Signal: rows 1–3 and 2b; verify summaries on main keep `role` values byte-identical (census run before
  and after shows the same role split), and `orchestrator verify-cache status` prints the same two role
  modes; `git grep -E 'is_merge_verify|allow_cold_cache'` over every `.py` file under `orchestrator/src`
  plus `defaults.yaml` is empty (a glob, so the sites 5040 moves are covered); the eval corpus under
  `evals/reviewer_trial/corpus/` is JSON history and keeps its strings.
- G7: INV-1 — the pairing rule is a constructor check, not prose. INV-2 — violation names fields. Ratchet
  baseline per the band note.

**γ — Per-leg typed outcome on VerifyResult; report, aggregate, cause hint and remote no-verdict read legs** (C3; high; ~1,300 LOC)
- Files: `verify.py` (the tool-dispatched `cause_hint(tool, output)` sits in the pytest-output region,
  which μ moves to `verify_pytest_output.py`; `verify_classify.py` is untouched), `verify_runner.py` (codec
  delegation; `RemoteRunner` no-verdict from legs), `remote_gate.py` (`wire_classes`),
  `merge_lane/history_leg.py` (`HistoryLegRunner.run_merge_verify` strips attribution by replacing `legs`
  with copies whose `failing_test_ids=None`), tests `test_verify_failure_report.py`,
  `test_verify_summary_facts.py`, `test_verify_leg_classification.py`, `test_remote_gate.py`,
  `test_verify_runner.py`, `test_merge_history_leg_runner.py`, `test_verify_attempt.py` (21 `LegSlot`
  references), and the files constructing the three removed fields (19 sites in 10 files, AST at
  `e634cfe658`).
- Keys: `fk-caa8f680e1c2`, `fk-3b67e8761674`, `fk-a2fb58789c14`. Deps: β.
- Deletes: the substring scan and `removeprefix('Failures: ')` parse in `_aggregate_results`;
  `is_no_verdict_summary`, `_NO_VERDICT_SUMMARY_MARKERS` and their readers; the text re-derivations in
  `failure_report`; the tool-blind `_extract_cause_hint` ladder (the OPAQUE ladder stays for
  `ToolKind.OPAQUE` only); `verify.py::LegSlot` (its 44 uses in verify.py and 28 in two test files read
  `verify_evidence.Leg`); the `CheckRun.label` field (replaced by `leg`; `to_dict()` keeps the `'label'`
  key, bytes unchanged); the constructor fields `failing_leg_categories`, `failing_test_ids_by_module`,
  `test_leg_failed`.
- Signal: rows 8–9; a lint-only red on main produces a block report whose only failure section is lint;
  the archived summary JSON carries a `legs` array with leg, tool, rc and category per leg.
- G7: INV-2 — facts the emitter held in variables reach the report as fields. INV-5 — one summary renderer.

**δ — One no-source-diff resolver and a typed plan leg** (C4 part; medium; ~550 LOC)
- Files: `verify.py`, `verify_plan.py` (`PlannedRun.leg`), `verify_cache/keys.py` (reads `run.leg`),
  `verify_evidence.py` only if `plan_hash`'s run keys need naming, tests `test_verify_plan.py`,
  `test_merge_verdict_integrity_inv1.py`, `test_verify_cache_keys.py` (pins `planned_run_leg`), new
  `orchestrator/tests/test_no_source_diff_resolution.py` (and task 7295's `test_verify_global_tail_label.py`
  if 7295 has landed).
- Keys: `fk-0f9cace0cd00`, `fk-c88f7d3ae54a`. Deps: γ.
- Deletes: the second copy of the no-source-diff block and its duplicated task-2838 comment;
  `verify_plan.py::_LEG_BY_REASON_PREFIX` and `planned_run_leg`'s prefix decoding, `_MC_ATTR_BY_LEG`,
  `_mc_attr_for_run`, `_mc_attr_for_run_or_none`, `_FALLBACK_TOOLKIND_BY_ATTR`. Moves
  `_emit_trivial_pass_escalated` beside the resolver (the ruled pre-split move).
- Signal: row 11; persisted `attempt-N.plan.json` runs carry `leg`; `orchestrator verify-cache status`
  still runs (the cache's keys read the field).

**ζ — One isolated-rerun policy table; sweep confirm on the ephemeral worktree helper** (medium; ~500 LOC)
- Files: `verify.py`, `git_ops.py` (`WorktreeKind.MAIN_SWEEP_CONFIRM`), tests `test_verify_main_tip_sweep.py`,
  `test_flake_discriminator.py`, `test_ephemeral_worktree.py`.
- Keys: `fk-3756ad2f3357` (its policy half; task 7918 fixed its builder half at `fca89bd55b`),
  `fk-f994995001e5`. Deps: δ (ruling S4-7).
- Work: D9, with each row's precondition stated.
- Deletes: `_sweep_failure_reproduces_in_isolation`'s and `confirm_main_tip_failure_is_real`'s own re-run
  paths (both already build their command with `_node_id_test_config`), the hand-rolled worktree
  add/retry/remove, the DD5 comments.
- Signal: row 13.

**η — One merge-verify profile drives both legs** (C7 part; high; ~900 LOC)
- Files: `verify_spec_shared.py`, `verify_runner.py`, `task_verify_spec.py`, `remote_gate.py`
  (`wire_classes`), `merge_lane/worker.py` (spec built once, handed to `LocalRunner`), tests
  `test_verify_runner.py`, `test_task_verify_spec.py`, `test_verify_spec_shared.py`,
  `test_merge_verify_config_fidelity.py`, `test_remote_verify_dispatch.py`.
- Keys: `fk-8dbae7d39adc`. Deps: ι1 (same `LocalRunner` and worker runner construction; it already
  follows γ's codec and β's spec field removal through the chain); out of batch 7465.
- Deletes: `_module_config_from_command`, `module_config_from_command`'s lossy path,
  `apply_spec_to_host_config` (folded into `apply_merge_verify_spec`), the "not from the spec" design note.
- Signal: rows 17–18; a merge verify on the laptop host of a module with its own `verify_env` runs with it.

**θ — Host selection and quarantine live only in HostAllocator** (C7 part; medium; ~650 LOC)
- Files: `verify_runner.py`, `verify_hosts.py` (`build_remote_runners`' dead quarantine filter),
  `merge_lane/drift.py`, `merge_lane/shadow.py`, `merge_lane/worker.py` and, at the leaf's base, task
  5040's `merge_lane/verify_dispatch.py` and `host_quarantine.py` (the pool sites and the quarantine
  reads and writes it moved), tests `test_verify_runner.py`, `test_merge_drift.py`,
  `test_merge_queue_multihost_wiring.py`.
- Keys: `fk-baa454b81f6f`. Deps: η; cross-stream S6 ζ (lead ruling, as for β) and S6 λ (lead ruling: the
  last S6 leaf on `worker.py`'s pool sites, `drift.py` and `shadow.py`); same-file edges with 5291 and 7038,
  direction by status.
- Work: D11: the harness-built set (task 6670) stays one object, and `HostAllocator` becomes its only
  writer and the merge lane's only reader, so the worker's (or 5040's dispatcher's) direct set accesses go
  through the allocator; `HostPool` keeps reading it by reference. The drift escalation's suggested action
  replaces "call pool.clear_quarantine" with readmission through `clear_verify_host_quarantine` (the
  method it named goes here).
- Deletes: `VerifyRunnerPool` (becomes `dispatch_merge_verify`), its `_quarantined`, `quarantine`,
  `clear_quarantine`, `is_quarantined`, `eligible_remote`, `_select_runner`; the `quarantine` parameter
  of `build_remote_runners` and its filter tests (task 8734, cancelled into θ).
- Signal: row 19; after a drift FAIL the worker snapshot's host states show the remote quarantined.
- G7: ratchet baseline per the band note; D11's 5291 note.

**ι1 — The VerifyHost seam in place; `host` required on every verify function** (C5 part, ruling S4-1; medium; ~1,500 LOC; one mechanical shape outside verify.py)
- Files: `verify.py` (`VerifyCommandRunner`, `CommandRequest`, `CommandOutcome`, `SubprocessRunner`,
  `VerifyHost` with its process slot; `host: VerifyHost` required on the entry points, `run_node_ids`, the
  six public probes and every internal helper between them), `verify_runner.py` (`LocalRunner(host=…)`,
  `run_merge_verify_on_worktree`, the `apply_merge_flake_suppression` call), `task_verify_run.py`,
  `task_verify_dispatch.py` (the ports gain `host`; the dispatcher forwards the caller's), `workflow.py`
  (`TaskWorkflow(verify_host=None)` resolved once; both verify sites and the preexisting probe),
  `review_checkpoint.py` (`ReviewCheckpoint(verify_host=None)`), `evals/runner.py` (passes the workflow's
  host), `merge_lane/ports.py` (`VerifyPort` methods gain `host`), `merge_lane/gates.py`,
  `merge_lane/history_leg.py` (`HistoryLegRunner(host=…)`), `merge_lane/worker.py` or, at the leaf's base,
  5040's modules (`SpeculativeMergeWorker(verify_host=None)`; the probe calls, `LocalRunner` and
  `HistoryLegRunner` constructions), `merge_lane/drift.py`, `merge_lane/shadow.py` (their `LocalRunner`
  constructions), `harness.py` (resolves `VerifyHost.process()` once; passes it to the objects it builds and
  to its sweep and confirm calls), `cli.py` (`verify-merge` and `verify-task` resolve their process host);
  `orchestrator/tests/conftest.py` (autouse `VerifyHost.installed(VerifyHost.fresh())` and a `verify_host`
  fixture), new `orchestrator/tests/_verify_runner_fakes.py` (`ScriptedRunner`), new
  `orchestrator/tests/test_verify_host_seam.py` (the two guards with seeded fixtures, rows 14, 14b, 14c);
  and the 55 test files whose 449 calls of an entry point, `run_node_ids`, a public probe or `LocalRunner`
  gain `host=verify_host` (AST at `e634cfe658`; 20 of them are among the 21 files ι2 later edits).
  Review trigger acknowledged: outside verify.py the edit is one shape (thread one required argument).
- No key (prepares `fk-3e806baf4c6e`). Deps: ζ.
- Work: D8 and C5. `SubprocessRunner.run` delegates to module `_run_cmd`, so every dotted patch still
  intercepts until ι2. The boundary objects keep `verify_host: VerifyHost | None = None`, which spares
  their ~640 test constructions; nothing inside the verify modules has a default.
- Signal: rows 14, 14b, 14c; verify summaries on main unchanged; `VerifyHost.process()` logs one INFO
  line when it fills the process slot, and the orchestrator's journal shows it once per process start.
- G7: INV-10 — both guards observe their seeded rejection. INV-11 — no second host can exist in
  production (guard 2), so the admission queue cannot silently split. Ratchet baseline (conftest,
  `merge_lane/**`) per the band note.

**ι2 — Retire every `_run_cmd` patch site** (high; ~1,100 LOC; mechanical)
- Files: `orchestrator/tests/test_verify.py` (118 patches) and the 20 other test files patching `_run_cmd`
  by dotted path, 215 patches in all at `e634cfe658` (`test_verify_env_transient.py`,
  `test_verify_admission_integration_gate.py`, `test_verify_admission_wiring.py`,
  `test_verify_sequential_lint_first.py`, `test_verify_reverse_dep.py`,
  `test_verify_pyright_interpreter_misresolution.py`, `test_verify_cold_preprovision.py`,
  `test_verify_admission_pytest_n.py`, `test_verify_segmented_fallback.py`, `test_verify_scope_kappa.py`,
  `test_merge_verdict_integrity_inv1.py`, `test_verify_load_stamp.py`, `test_verify_leg_classification.py`,
  `test_verify_clock_stop.py`, `test_merge_queue_deep_integration_gate.py`, `test_verify_summary_facts.py`,
  `test_verify_ruff_config_boundary.py`, `test_verify_pytest_n_operator_cap.py`,
  `test_verify_env_recovery_gate.py`, `test_serial_addopts_preservation.py`), the `patch.object` and import
  sites (`test_merge_gate_rerun_end_to_end.py`, `test_verify_merge_flake_suppression.py`,
  `test_verify_run_cmd_reap.py`, `tests/scripts/test_verify_budget_census.py`), `test_shutdown.py` (its
  child-script string imports `_run_cmd`, invisible to AST queries and pyright; the script moves onto the
  public `SubprocessRunner`), the files patching the 10 other dotted targets in the exec group; `verify.py`
  (the delegation deleted); orchestrator test paths
  under `orchestrator/tests/`. Same-file edges with 6765 and 7666 on `tests/scripts/test_verify_budget_census.py`.
  Review trigger acknowledged: one mechanical shape (dotted patch → `host=VerifyHost.fresh(runner=…)`;
  every call site already passes a host after ι1).
- Keys: `fk-3e806baf4c6e`. Deps: ι1.
- Deletes: every `orchestrator.verify._run_cmd` patch; the delegation. The module-level `_run_cmd` name
  stays (it is `SubprocessRunner.run`'s body and `review_pre_test.py` still imports it) until ι3 moves it.
- Signal: `git grep 'orchestrator\.verify\._run_cmd' orchestrator/tests tests` empty.
- G7: INV-10 — the scripted runner drives behaviour; no text matching. Ratchet baseline (conftest).

**ι3 — Move the exec region and the host to verify_exec.py and verify_host.py** (medium; ~1,600 lines moved)
- Files: new `verify_exec.py`, `verify_host.py`; `verify.py`, `verify_runner.py` (`target_subprocess_env`),
  `review_pre_test.py` (repointed to the public runner), `evals/snapshots.py`, `harness.py`
  (`reap_leftover_verify_scopes`), `orchestrator/tests/test_shutdown.py` (child-script import path), test
  imports of the moved names.
- No key. Deps: ι2, θ. Same-file edge with 7339 (direction by status) and with S8's `harness.py` leaves.
- Deletes: the exec region of verify.py; `review_pre_test`'s private import.
- Signal: row 20 for the two modules; `review_pre_test.py` names no `_run_cmd`.

**κ — Owned stores on VerifyHost; the 12 module-level caches and the admission gates go** (C5; low; ~800 LOC)
- Files: new `verify_stores.py` (including `MainShaBaseline`, so stores never import the probe module);
  `verify_host.py`, `verify.py`, `verify_admission_gate.py` (`_GATES` moves onto `VerifyStores`;
  `gate_for` becomes `host.stores.admission_gate(slots_dir)` on the host each waiter was given),
  `verify_admission_watch.py` (`AdmissionWatch(host)`; `tick` reads that host's store, so its heartbeat and
  aged-waiter alarm keep seeing the queue), `harness.py` (builds `AdmissionWatch` with its one process
  host), `orchestrator/tests/test_verify_admission_watch.py`, `orchestrator/tests/conftest.py`, tests `test_verify_preexisting_main_break.py`,
  `test_verify_mu_fork_flake_gate.py`, `test_verify.py` (cache-reference sites), new
  `orchestrator/tests/test_verify_stores.py`, the guard `orchestrator/tests/test_verify_no_module_state.py`.
- Keys: `fk-126a75a57ca5`. Deps: ι3; out of batch 5292.
- Guard rule: in `orchestrator/src/orchestrator/verify*.py`, a module-level name that a function mutates
  (through `global`, a subscript assignment, or `.append/.add/.pop/.clear/.update/.setdefault/.discard/…`)
  is flagged; a `contextvars.ContextVar` is exempt by type; constant tables (`CATEGORY_POLICY`,
  `_CALL_SITE_POLICY`) are not flagged because nothing mutates them. The guard's own fixture test seeds a
  module that mutates a global and expects the flag (INV-10).
- Deletes: the 12 globals, `_suppressed_flake_records` outright (0 readers), `clear_main_probe_caches`,
  `verify_admission_gate.py::_GATES`.
- Signal: rows 15 and 15b; the guard flags its seeded fixture module and passes the tree.
- G7: INV-11 — the admission watcher reads the same gates as the waiters (row 15b), so no heartbeat goes
  silent. Ratchet baseline (conftest) per the band note.

**λ — verify_archive.py and the one attempt-* grammar; the census parses through it** (C6; medium; ~1,100 LOC)
- Files: new `verify_artifacts.py`, `verify_archive.py`; `verify.py`, `verify_runner.py`,
  `task_verify_dispatch.py`, `scripts/verify_budget_census.py` (adds `import _checkout as _checkout`),
  `scripts/suite_census_evidence.py`, tests `test_verify.py` (archive sections), `test_verify_runner.py`,
  `tests/scripts/test_verify_budget_census.py`, new `orchestrator/tests/test_verify_artifacts.py`.
- Keys: `fk-37d63bf83608`. Deps: κ; out of batch 6765.
- Deletes: the 6 sanitiser copies (`_warm_marker_name` uses `sanitize_prefix`), 2 of the 3 stamp
  spellings, `_stream_log_path`'s inline name, the census's private regex for new records.
- Signal: row 16; `scripts/verify_budget_census.py` over `data/verify-logs` reports the same record count as
  before for legacy archives and `where.source: record` for new ones.

**μ — Split I: verify_pytest_output.py and verify_commands.py** (medium; ~1,600 lines moved)
- Files: the two new modules; `verify.py`, `offline_lane_confirm.py`, `offline_lane_runs.py`,
  `task_verify_dispatch.py` (its own `_worktree_reader` copy, with different decode rules, is replaced by
  `verify_commands.worktree_reader`), tests importing the moved private names.
- No key (prepares `fk-73b03674a8e4`). Deps: λ.
- Moves `_worktree_reader` into `verify_commands.py` first (the ruled move) and `WORKER_DEATH_SUMMARY_MARKER`
  with the pytest-output group; names imported from outside become public there. The fallback-only
  rescoper (`_scope_fallback_tool_to_subproject`, `_rescope_clause_to_subproject`, `_cd_clause_target`,
  `_AND_CLAUSE_SPLIT_RE`) stays in `verify.py` for ε (ruling S4-7).
- Signal: row 20 for the two modules; `def _worktree_reader` absent from verify.py and
  `task_verify_dispatch.py`.

**μ0 — Move the result types to verify_result.py and repoint every importer** (ruling S4-6; `complexity='simple'`; low-risk mechanical; ~1,000 lines changed)
- Files: new `verify_result.py`; `verify.py`; the 16 production importers of `VerifyResult` from
  `orchestrator.verify` (`cli.py`, `flake_recorder.py`, `merge_lane/disposition.py`, `merge_lane/gates.py`,
  `merge_lane/history_leg.py`, `merge_lane/ports.py`, `merge_lane/shadow.py`, `merge_lane/types.py`,
  `merge_lane/worker.py`, `remote_gate.py`, `review_checkpoint.py`, `review_working_tree.py`,
  `task_verify_dispatch.py`, `task_verify_run.py`, `verify_runner.py`, `workflow.py`; the same 16 by AST at
  `e634cfe658`, plus whichever of 5040's modules import it at the leaf's base) and the 114 test files
  (AST at `e634cfe658`; enumerated again at decompose by `git grep -l` / the AST query; one import line
  each).
  Review trigger acknowledged: one import-line shape.
- No key. Deps: μ; cross-stream S6 ζ (lead ruling, as for β) and S6 μ (lead ruling: the last S6 leaf on
  `worker.py`, `types.py`, `gates.py` and `shadow.py`).
- Signal: `^class VerifyResult` defined only in `verify_result.py`; row 20.

**ν — Split II: verify_leg.py** (medium; ~2,000 lines moved)
- Files: new `verify_leg.py`; `verify.py`, `evals/runner.py`, `merge_lane/gates.py`,
  `merge_lane/history_leg.py`, tests importing `run_verification` or its helpers (`_xdist_workers`,
  `_load_sample`, `_summarize_checks`, `_build_summary_payload`…).
- No key. Deps: μ0. `merge_verify_command_budget_secs` stays in `verify.py` (C1); `verify_leg.py` imports
  `resolve_verify_timeout` from S6's `verify_budget.py`, which S6 α moved there (§Sketch).
- Signal: row 20; `run_verification` defined in `verify_leg.py` only.

**ξ — Split III: verify_main_probe.py and verify_discriminator.py** (high; ~3,000 lines moved)
- Files: the two new modules; `verify.py`, `verify_runner.py` (`apply_merge_flake_suppression`),
  `merge_lane/worker.py` or 5040's `main_health_probe.py` (`seed_main_baseline`,
  `known_failing_ids_on_main`, the probe), `workflow.py`,
  `harness.py` (`run_main_tip_sweep`, `confirm_main_tip_failure_is_real`), and every other importer
  `git grep` finds for the moved names at the leaf's base; tests of the probe, sweep, discriminator and
  baseline.
- Keys: `fk-73b03674a8e4`, `fk-3e806baf4c6e` (closes it: the last private-name imports become module
  publics here). Priority high. Deps: ν.
- Signal: row 20; `verify_failure_is_preexisting_on_main` and `confirm_isolated_rerun_verdict` imported
  from their modules by `workflow.py`, `merge_lane/worker.py` and `harness.py`.

**ε — The plan executes the fallback branch; delete the second scope engine; final h14 measurement** (C4; high; ~1,400 LOC)
- Files: `verify_plan.py`, `verify_cmd.py`, `verify.py`, tests `test_verify_plan.py`,
  `test_verify_scope_kappa.py`, new `orchestrator/tests/test_fallback_plan_characterization.py` and its
  fixture file, `tests/scripts/test_fallback_verify_config.py`, `tests/scripts/test_pyright_version_pin.py`,
  `tests/scripts/verify_command_invariants.py`, `tests/scripts/test_contributing_type_check_command_drift.py`,
  `tests/scripts/test_verify_command_invariants.py`.
- Keys: `fk-fac2cb6af858`, `fk-d99f146ff901`, `fk-cb29fac47dba`, `fk-a635d54da4ab`. Deps: ξ (ruling S4-7);
  out of batch 7041.
- Step 1 captures the characterization fixtures from `_build_fallback_config` at the leaf's base tree;
  the last step deletes it. The tests/scripts sites move onto `derive_verify_plan`. The commit message
  records the final measurement: raw lines and prose share per verify module, the layering scan (row 20)
  result, and `verify.py` under 2,000 raw lines, else the named partition step that remains.
- Deletes: `_build_fallback_config`, `_select_subproject_pytest_targets`, `_single_subproject_prefix`,
  `_root_plus_single_subproject_prefix`, `_scope_fallback_tool_to_subproject`,
  `_rescope_clause_to_subproject`, `_AND_CLAUSE_SPLIT_RE`, `_cd_clause_target`, `_executed_fallback_plan`,
  `_safe_derive_executed_fallback_plan`, the "NOTE (task γ amendment)" blocks.
- Signal: rows 10 and 12; on main a zero-module project's verify (reify) logs a `Verify plan:` whose runs
  are the commands its summary records.
- G7: INV-5 — removes the lockstep pair. INV-11 — a fallback that cannot render a test target keeps 7041's
  explicit no-tests-ran outcome.

**ο — verify_runner: dead code, reach-backs, HostAllocator and DriftDetector out** (low; ~800 LOC)
- Files: `verify_runner.py`, `verify_hosts.py`, new `verify_drift.py`, `merge_lane/drift.py`, `cli.py`
  (`verify-merge` passes the public port `merge_lane/worker.py::PRODUCTION_VERIFIER.run_unscoped_typechecks` as `run_unscoped`, as the drift path already does), tests `test_verify_runner.py`,
  `test_merge_drift.py`, `test_cli.py`.
- Keys: `fk-e906c4e914c2` (also carried by ρ; closes at ρ). Deps: ξ, θ, λ.
- Work: `run_unscoped` becomes a required keyword of `run_merge_verify_on_worktree`, so `verify_runner.py`
  imports nothing from `orchestrator.merge_lane`; `_run_unscoped_typechecks` stays private in
  `merge_lane/gates.py` (S7's gate) and `cli.py` never imports it.
- Deletes: the sccache helpers, `ColdWarmVerifyDelta` and codec, both function-local imports and the stale
  `type: ignore`. Keeps `run_verdict_parity` and its report types for ρ (D15).
- Signal: row 20 extended to `verify_runner.py`; the commit message records its new line count.
- G7: ratchet baseline per the band note.

**ρ — The `orchestrator verdict-parity` command; both escalation actions name it** (new by ruling S4-8; D15; medium; ~500 LOC, an estimate the sitting did not measure)
- Files: `cli.py` (the `verdict-parity` command), `verify_runner.py` (a small runner-building helper, if one
  is extracted from the drift setup for both callers), `verify_drift.py` (the drift detector's suggested
  action, after ο moved it there), `merge_lane/drift.py` (if the helper is extracted from it),
  `merge_lane/worker.py` or, at the leaf's base, 5040's `verify_cross_check.py` (the cross-check divergence
  action), `OPERATIONS.md` (the re-prove and readmit procedure, including the code-identity caveat),
  `orchestrator/tests/test_cli.py`, new `orchestrator/tests/test_verdict_parity_command.py` (row 21, one
  end-to-end run against real runners on a fixture repo).
- Keys: `fk-e906c4e914c2` (closes it: the parity code ο kept gets its caller). Deps: ο (same `cli.py`,
  `verify_runner.py`, drift module and `test_cli.py`; ο follows θ, which retired the runner pool the drift
  setup used); same-file edges with 5291 (the cross-check action and quarantine) and with S6's later
  `worker.py` leaves in the lead's per-file order.
- Work: D15. The command defaults to the single divergent SHA (`--sha` given once), takes more only on
  request, and prints the per-SHA cost warning before it starts.
- Signal: row 21; after a cross-check divergence on main, the blocking escalation's action text names
  `orchestrator verdict-parity --sha <the merge sha>` and `clear_verify_host_quarantine`, and running that
  command prints the parity report.
- G7: INV-2 — the report names the SHA, both runners and both verdicts. INV-11 — a disagreement exits
  non-zero; the command never readmits a host itself. Ratchet baseline per the band note.

**π — Integration gate: two-way boundary rows for C1–C7 and D15** (medium; no key; ~650 LOC)
- Files: new `orchestrator/tests/test_verify_contracts_boundary.py` (rows ★ plus a re-run of the rest by
  import), fixture helpers under `orchestrator/tests/`.
- Deps: ε, ο, ρ (and through them every leaf).
- Signal: every boundary row green on main; the signal leaf of the sidecar.

## Out of scope

- The cross-check fleet-halt gate's aggregate-vs-per-leg rule (task 6652 item 3, human adjudication).
- The main probe's three-state answer (task 5292).
- `run_verification`'s own 1,034-line body (measured at `e634cfe658`): ν moves it whole; splitting the
  function is heuristic-4 work for a later instrument run.
- `test_verify.py`'s 12,645 lines (measured at `e634cfe658`): the leaves edit it; splitting it is not this
  stream's.
- Merge-lane state, retry budgets and satellites (S6), including the hold-reason wording: every merge-lane
  infra hold reads `TRANSIENT_INFRA_REASON_PREFIX` ("Transient infrastructure failure (disk pressure)")
  whatever the category, a live mislabel for non-disk infra holds that S6's typed outcomes own (no crash
  reaches that hold after ruling S4-5). The unscoped typecheck gate (S7), config shape (S16), the
  `_normalize_cause_hint` function-local import (S9 moves it down).
- Removing the `VerifyHost` process slot altogether (a host required on the boundary objects too): weighed
  at the sitting and not ruled, about 640 more test edits for the same guarantee (D8).

## Open questions (tactical)

1. **Summary record key for the resource class.** Keep `role` (census reads it) or add `resource_class`
   beside it. Suggested: keep `role` only. Decide in β.
2. **Where the characterization fixtures live** (ε): a JSON file beside the test, captured by a script the
   leaf runs once. Suggested: `orchestrator/tests/fixtures/fallback_plan_corpus.json`.

The conftest question the draft carried (one autouse reset plus explicit hosts) is settled by ruling S4-1
(D8). Every design question was ruled at the sitting of 2026-11-10 (§Rulings); the return's
`open_questions` is empty.
