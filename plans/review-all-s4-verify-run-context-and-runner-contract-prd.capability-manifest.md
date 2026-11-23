# Capability manifest — S4 verify run context and runner contract

PRD: `plans/review-all-s4-verify-run-context-and-runner-contract-prd.md`. YAML twin:
`plans/review-all-s4-verify-run-context-and-runner-contract-prd.capability-manifest.yaml` (signal leaf π).

**Code anchors** verified against main `e634cfe658` (2026-11-10), after the rulings of the 2026-11-10
sitting were folded, and the polarity re-run at `bb1c659562` (77 commits later; no anchor changed). The draft was verified at `ce0f71a384` (2026-11-01) and re-checked at `d40ba048e9`
(2026-11-02). Fold-verified at `bb1c659562` (2026-11-10): the lead's sequencing rulings applied, θ's
quarantine-set binding corrected to the harness-built set (task 6670), and one θ check added. Bindings
cite symbols, not lines.

Polarity, measured at `e634cfe658` with `shared/src/shared/delivered_check_polarity.py::lint_delivered_checks`
and a direct `git grep -E` of every pattern: every mechanical check is a producer check that fails today
and passes once its leaf lands (0 vacuous findings across 20 leaves; the remaining lint warnings are
`absent_overbroad`, which the lint raises because it is run without the leaves' declared files). One
check the draft carried had gone vacuous on main: δ's `_MC_ATTR_BY_REASON_PREFIX` (task 5652 moved the
table into `verify_plan.py::_LEG_BY_REASON_PREFIX`); it is replaced. Substrate bindings are recorded `kind:
manual` with the matching grep in their reason: a present-today grep is rejected as vacuous by that
module, so a substrate fact cannot ride the dispatch gate. Directory paths or a `*.py` glob are used where a
later leaf of this PRD (μ, μ0, ν, ξ, ο), S5 ξ or task 5040 moves the symbol, so no check goes red on a move.
λ's sanitiser check names the files λ edits and excludes `verify_artifacts.py`, whose `sanitize_prefix`
body would otherwise match forever. No check names a script.

## α — Failure vocabulary and infra-transient as data; the bare-crash re-run rule; every entry point branches

| Capability | Binding | Verdict |
|---|---|---|
| Category policy table to extend with full rows | substrate — `verify_categories.py::CATEGORY_POLICY`, six-field `CategoryPolicy`, exhaustiveness-validated by `_validate_exhaustive` | PASS |
| Adjudicated indeterminate set stays `{INFRA_KILL}` | substrate — `verify_categories.py::INDETERMINATE_VERDICT_CATEGORIES` derived from `verdict_indeterminate`; the new rows set it False | PASS |
| Infra-transient retry loop that already reads a category | substrate — `workflow.py::TaskWorkflow._run_scoped_verification_with_infra_retry` has the `INFRA_TRANSIENT_CATEGORIES` branch beside its `except`; it is also the path through `TaskVerifyDispatcher` (tasks 6670/7977) | PASS |
| Merge lane has a classified infra-transient branch and an in-place re-dispatch to reuse for the crash re-run | substrate — the `INFRA_TRANSIENT_CATEGORIES` branch of `merge_lane/worker.py::_run_post_merge_verify` (one retry in the shared budget, then the `TRANSIENT_INFRA_REASON_PREFIX` hold; re-dispatch on the held host with `dispatch_facts`), which task 5040 moves into its `merge_lane/verify_dispatch.py` family | PASS |
| The crashed test's node id is readable from a bare crash's output | substrate — `verify.py::_xdist_crash_message` reads it with `_extract_failing_test_ids` today; α moves that reading into `repeated_crash_report` | PASS |
| Review checkpoint has a skip path to extend | substrate — `review_checkpoint.py::ReviewOutcome.SKIPPED_CAPPED` path (skip, trigger kept) | PASS |
| Eval cells can be unmeasured | substrate — `evals/measurement.py::Unmeasured` (S18, task 6734, done) | PASS |
| Post-rebase re-verify site to type | substrate — `workflow.py::TaskWorkflow._post_rebase_reverify_passed` called from `_rebase_before_enqueue` (rollback-only, `pre_merge_reverify: false`) | PASS |
| Per-leg facts to evaluate D2 before γ creates `legs` | substrate — `VerifyResult.failing_leg_categories` (task 3173); None means not recorded and falls back to the aggregate row; α rewrites the bare-crash test leg's entry to `XDIST_WORKER_CRASH` | PASS |
| Ledger reason for a merge-gate crash stays byte-identical | substrate — `verify.py::_unmeasured_session_reason` (α reads `XDIST_WORKER_CRASH` as `session_truncated`), `_unconfirmable_rerun_reason` | PASS |
| Baseline probes return `None` before seeding | substrate — `verify.py::_probe_modules_on_main`, `_whole_tree_main_baseline`, `seed_main_baseline`, `_remember_main_baseline` | PASS |
| S6 lands first on `merge_lane/worker.py`'s infra-transient region (and, through S6 ζ, task 5040 first) | producer: S6 ζ (S6 C2) — upstream of α by lead ruling; wired at decompose | PASS |
| `VerifyInfraError`, `_assert_sentinels_disjoint`, `infra_reclassified` gone; `InfraFault`, sentinel members, `BARE_CRASH_RERUNS`, `SKIPPED_NO_VERDICT` present | produced by α | PASS (α delivers) |

## β — VerifyContext and ModuleRegistry

| Capability | Binding | Verdict |
|---|---|---|
| Public registry accessor for the default context | substrate — `config.py::OrchestratorConfig.module_configs_or_empty` | PASS |
| Resource-class strings used by admission, nice and the event channel | substrate — `shared/verify_admission.py::is_gated_role`, `nice_prefix`; `verify_event_channel.py::encode_channel_open(role=…)`; summary `role` key (task 5671) | PASS |
| The cache's role vocabulary to fold into `ResourceClass` | substrate — `verify_cache/modes.py::VerifyRole`, `VERIFY_ROLES` (task 5652), read by `config.py::VerifyCacheConfig.mode_for`, `verify_cache/policy.py`, `verify_cache/toolchain.py`, `verify.py` | PASS |
| `max_retries` takes only two values | measured at `e634cfe658`: every call passes `max_retries=0` or the default | PASS |
| `allow_cold_cache=False` has no production caller | measured at `e634cfe658`: test files only | PASS |
| Cold/timeout/attempt rules each site resolves today | substrate — `verify.py::_is_verify_cold`, `_resolve_verify_timeout` (moved by S6 α into `verify_budget.py::resolve_verify_timeout` before β), `_mark_verify_warm`, the pure-timeout retry loop in `run_verification`; C1 site table | PASS |
| S6 lands first on `merge_lane/worker.py` and `gates.py` | producer: S6 ζ (S6 C2, `ItemLifecycle` operative) — upstream of β, θ and μ0 by lead ruling; wired at decompose | PASS |
| `VerifyContext`, `ResourceClass`; flags, `VerifyRole` and private registry reads gone; site table preserved | produced by β (the `is_merge_verify` absent check globs every `.py` under `orchestrator/src` plus `defaults.yaml`; the eval corpus under `evals/reviewer_trial/corpus` is JSON history and stays) | PASS (β delivers) |

## γ — Per-leg typed outcome on VerifyResult

| Capability | Binding | Verdict |
|---|---|---|
| Per-leg classifier with tool dispatch | substrate — `verify.py::classify_leg` → `verify_classify.py::classify_failure(tool, …)`; `verify.py::_tool_for_cmd` | PASS |
| One leg vocabulary to adopt | substrate — `verify_evidence.py::Leg` (task 5652; values `test`/`lint`/`type`, equal to `verify.py::LegSlot`'s) | PASS |
| In-flight execution record to type | substrate — `verify.py::CheckRun` (label, cmd, rc, output, timed_out, …); γ replaces `label` with `leg` and adds `tool`; `CheckRun.to_dict()` keeps the `'label'` key that `scripts/verify_budget_census.py` reads (`entry['label']`) | PASS |
| Remote no-verdict decision site | substrate — `verify_runner.py::RemoteRunner` dispatch calls `verify.is_no_verdict_summary` | PASS |
| Wire fingerprint covers the new nested classes | substrate — `remote_gate.py::wire_classes` (hand-kept tuple) / `wire_fingerprint` (task 6565); γ adds `LegOutcome`, `InfraFault` | PASS |
| `LegOutcome`; `LegSlot`, text re-derivation, summary scan and no-verdict summary scan gone | produced by γ | PASS (γ delivers) |

## δ — One no-source-diff resolver and a typed plan leg

| Capability | Binding | Verdict |
|---|---|---|
| INV-1 escalation emitter to centralise | substrate — `verify.py::_emit_trivial_pass_escalated`, `_trivial_pass_reason`, `_merge_no_evidence_fail` | PASS |
| Full-gate override predicates | substrate — `verify.py::_merge_config_only_diff_forces_full_gate`, `_verify_pipeline_guard_requires_full_gate` | PASS |
| The reason-prefix decoder and its readers | substrate — `verify_plan.py::planned_run_leg`, `_LEG_BY_REASON_PREFIX`; readers `verify.py::_mc_attr_for_run_or_none` and `verify_cache/keys.py` | PASS |
| `NoSourceDiffOutcome`, `PlannedRun.leg`; reason-prefix routing gone | produced by δ | PASS (δ delivers) |

## ζ — One isolated-rerun policy table; sweep confirm worktree

| Capability | Binding | Verdict |
|---|---|---|
| Discriminator with the `session_truncated` precondition | substrate — `verify.py::confirm_isolated_rerun_verdict` (task 5492); `_RerunPolicy.refuses_partially_measured_sessions` | PASS |
| One re-run command builder already shared | substrate — `verify.py::_node_id_test_config` (task 7918, `fca89bd55b`) | PASS |
| Ephemeral worktree helper with protected prefixes | substrate — `git_ops.py::GitOps.ephemeral_worktree`, `WorktreeKind`, `PROTECTED_PREFIXES` | PASS |
| Ledger call-site vocabulary each row may name | substrate — `flake_ledger.py::FlakeCallSite` | PASS |
| `RerunSite`, `MAIN_SWEEP_CONFIRM`; the sweep sites' own re-run paths and the hand-rolled worktree gone | produced by ζ | PASS (ζ delivers) |

## η — One merge-verify profile drives both legs

| Capability | Binding | Verdict |
|---|---|---|
| Config-level merge-path field census | substrate — `orchestrator/tests/test_merge_verify_config_fidelity.py::MERGE_PATH_CONFIG_READS` (task 5295, done) | PASS |
| Shared spec vocabulary for both specs | substrate — `verify_spec_shared.py::VerifyCommand`, `apply_spec_to_host_config`; `task_verify_spec.py::apply_task_verify_spec` | PASS |
| Spec fields landed first | `producer:task-7465` (pending) — upstream of η, wired at decompose | PASS |
| Lossless profile, shared apply, lossy reconstruction gone | produced by η | PASS (η delivers) |

## θ — Host selection and quarantine live only in HostAllocator

| Capability | Binding | Verdict |
|---|---|---|
| S6 lands first on `merge_lane/worker.py` | producer: S6 ζ (S6 C2) and S6 λ (drift and shadow own their state; the last S6 leaf on the pool sites, `drift.py`, `shadow.py`) — upstream of θ by lead rulings; task 5040 lands before S6 ζ (S6-2 = B) | PASS |
| Allocator with the shared quarantine set, PARKED and `quarantine_and_release` | substrate — `verify_runner.py::HostAllocator` (acquire stamps `HostLease.reason`, task 8523) | PASS |
| One quarantine set, shared by reference | substrate — `harness.py::Harness.__init__` builds `_runner_quarantine` once (task 6670) and passes it to `verify_hosts.py::HostPool` (reader), the drain participant and the merge worker (`runner_quarantine=`), whose `HostAllocator(…, quarantine=self._runner_quarantine)` wraps it; the worker also writes it directly (`bench_unavailable_runner`), sites task 5040 moves; θ makes the allocator the only writer | PASS |
| Dead selection filter to delete | substrate — `verify_hosts.py::build_remote_runners(…, quarantine=…)`, no production caller passes it (task 8734, cancelled into θ) | PASS |
| Forwarded dispatch fields | substrate — `merge_lane/worker.py`'s `pool.dispatch(merge_sha, spec, attempt=…, depth=…, speculative=…, chain_items=…, chain_build_ms=…, dispatch_facts=…)` calls | PASS |
| Pool retired into `dispatch_merge_verify`; the drift action no longer names `pool.clear_quarantine` | produced by θ | PASS (θ delivers) |

## ι1, ι2, ι3 — VerifyHost, the required host, and the command-runner seam

| Capability | Binding | Verdict |
|---|---|---|
| The subprocess body to wrap | substrate — `verify.py::_run_cmd`, `_run_segmented`, `_target_subprocess_env`, `reap_leftover_verify_scopes` | PASS |
| The boundary objects that resolve the process host once | substrate — `workflow.py::TaskWorkflow`, `merge_lane/worker.py::SpeculativeMergeWorker`, `review_checkpoint.py::ReviewCheckpoint`, each with one production construction (`workflow.py`, `harness.py`, `harness.py`); `evals/runner.py::collect_metrics` holds its workflow; `cli.py` `verify-merge` and `verify-task` | PASS |
| The call sites that must pass a host | measured at `e634cfe658` (AST): 22 production calls of the three entry points (16 inside verify.py), 10 of the six public probes, 7 `LocalRunner` constructions; 449 test calls in 55 files | PASS |
| A delegation that keeps the old patches intercepting until ι2 | produced by ι1 (`SubprocessRunner.run` → module `_run_cmd`), removed by ι2 | PASS (ι1 delivers) |
| Process slot exempt from the module-state guard | substrate — `contextvars.ContextVar` (stdlib) | PASS |
| `VerifyHost`, `VerifyCommandRunner`, the two guards (ι1); dotted and `patch.object` sites gone (ι2); `verify_exec.py`/`verify_host.py`, no private runner in `review_pre_test.py` (ι3) | produced by ι1, ι2, ι3 (the 21 + 3 + 1 test files are ι2's declared files) | PASS (ι1–ι3 deliver) |

## κ — Owned stores on VerifyHost

| Capability | Binding | Verdict |
|---|---|---|
| The structures to retire | measured at `e634cfe658` by the mutation rule: verify.py's 12 (`_suppressed_flake_records`, `_PROBE_CACHE`, `_BASELINE_FAILING_IDS_CACHE`, `_BASELINE_SKIPPED_IDS_CACHE`, `_LAST_PRUNE_AT`, `_RUFF_ESCAPE_REPORTED`, `_RUFF_ESCAPE_PROBE_ATTEMPTS`, `_RUFF_ESCAPE_IN_FLIGHT`, `_admission_executor_singleton`, `_PREPROVISION_LOCKS`, `_PREPROVISION_DONE`, `_MAIN_PROBE_DOWNGRADE_REPEAT_COUNTS`) and `verify_admission_gate.py::_GATES` | PASS |
| Probe hardening upstream | `producer:task-5292` (pending) — upstream of κ, wired at decompose | PASS |
| The admission watcher reads the gates the waiters join | substrate — `verify_admission_watch.py::AdmissionWatch.tick` reads `verify_admission_gate.py::gate_for(slots_dir).snapshot()`; `harness.py` builds `AdmissionWatch()`; κ builds it with the harness's process host and routes both through `host.stores.admission_gate` (row 15b) | PASS |
| Guard flags a seeded module that mutates a global | produced by κ; its own fixture test observes the rejection (INV-10) | PASS (κ delivers) |

## λ — verify_archive.py and the one attempt-* grammar

| Capability | Binding | Verdict |
|---|---|---|
| The census can import an orchestrator module under the system python | substrate — `scripts/_checkout.py::MEMBER_SRC` binds every member's `src` (task 6758); measured: the system python 3.12 imports `orchestrator.verify_categories` with only `orchestrator/src` on the path; the census itself imports no first-party module today, so λ adds the `_checkout` import (S25 ε, task 6760) | PASS |
| Census policy refactor upstream on the same file | `producer:task-6765` (S25 κ, pending) — upstream, wired at decompose | PASS |
| Summary record already carries `role` (precedent for task_id/module_prefix) | substrate — `verify.py::_build_summary_payload` (task 5671) | PASS |
| `ArtifactName`; sanitiser copies, extra stamp spellings and the census sanitiser gone; census binds `_checkout` | produced by λ | PASS (λ delivers) |

## μ, μ0, ν, ξ — the verify.py split

| Capability | Binding | Verdict |
|---|---|---|
| S6 lands first on the `merge_lane/` files μ0 repoints | producer: S6 ζ (S6 C2) and S6 μ (the last S6 leaf on `worker.py`, `types.py`, `gates.py`, `shadow.py`) — upstream of μ0 by lead rulings | PASS |
| Measured partition and its symptoms | substrate — `plans/review-all-dark_factory-20261008.json` `h14_measurements` (seat `orchestrator/verify`), symptoms re-derived by AST at `e634cfe658` | PASS |
| Importers to repoint | measured at `e634cfe658` (AST): 16 production modules and 114 test files import `VerifyResult` from `orchestrator.verify`; `harness.py` reaches `run_main_tip_sweep`, `confirm_main_tip_failure_is_real` | PASS |
| The timeout resolver lives below verify | producer: S6 α (`orchestrator/verify_budget.py::resolve_verify_timeout`, S6 C6), upstream of every S4 leaf; ν imports it | PASS |
| Moves leave no cycle | produced by μ/μ0/ν/ξ; boundary row 20 (fresh-interpreter import + AST layering scan) | PASS (μ–ξ deliver) |

## ε — The plan executes the fallback branch; final measurement

| Capability | Binding | Verdict |
|---|---|---|
| Quote-aware chain and cd primitives to render fallback commands | substrate — `verify_cmd.py::split_top_level_and`, `split_and_chain_segments`, `_literal_cd_target`, `render` | PASS |
| Fallback decision derivation to extend into rendering | substrate — `verify_plan.py::derive_verify_plan` fallback arm, `widen_fallback_for_marker_deselection` | PASS |
| Characterization source captured before deletion | substrate — `verify.py::_build_fallback_config` + `_apply_cargo_scope` at the leaf's base tree | PASS |
| Fallback test floor | `producer:task-7041` (pending, high) — upstream of ε, wired at decompose | PASS |
| Second engine and quote-blind grammar gone; `verify.py` measured | produced by ε (tests/scripts sites included in its files; measurement in the commit message) | PASS (ε delivers) |

## ο — verify_runner dead code and reach-backs

| Capability | Binding | Verdict |
|---|---|---|
| Dead code has no caller | measured at `e634cfe658`: the sccache helpers and `ColdWarmVerifyDelta` are referenced only inside `verify_runner.py` | PASS |
| Parity re-proof stays for ρ | substrate — `verify_runner.py::run_verdict_parity`, `render_parity_report` (no code caller; ruling S4-8 gives them one in ρ) | PASS |
| Unscoped gate is injectable at the composition root | substrate — `run_merge_verify_on_worktree(run_unscoped=…)`; the public port `merge_lane/worker.py::PRODUCTION_VERIFIER.run_unscoped_typechecks` (`merge_lane/ports.py`), already passed by the drift path; `cli.py` `verify-merge` calls `run_merge_verify_on_worktree` without it | PASS |
| Dead code gone, no `orchestrator.merge_lane` import, `verify_drift.py` | produced by ο | PASS (ο delivers) |

## ρ — The `orchestrator verdict-parity` command

| Capability | Binding | Verdict |
|---|---|---|
| The comparison and its report | substrate — `verify_runner.py::run_verdict_parity(corpus, local_runner, remote_runner, spec)`, `render_parity_report`, `VerdictParityReport` | PASS |
| A spec, remotes and a local runner can be built in a CLI process | substrate — `verify_runner.py::build_merge_verify_spec`; `verify_hosts.py::build_remote_runners(config, cwd, *, deployed_sha)` over `config.enabled_verify_runners`; the drift check's worktree-and-`LocalRunner` setup under `git_ops.merge_verify_lease` (`merge_lane/drift.py::_run_drift_check`) | PASS |
| The readmission tool the actions will name | substrate — the escalation server's `clear_verify_host_quarantine` (`escalation/src/escalation/server.py`), documented in `OPERATIONS.md` | PASS |
| The two action texts to rewrite | substrate — the cross-check divergence action in `merge_lane/worker.py::_run_post_merge_verify` ("(run_verdict_parity) and clear its quarantine"), the drift detector's `suggested_action` ("via run_verdict_parity") | PASS |
| Cleanup and host retirement upstream | producer: ο (same files) and, through it, θ (the runner pool retired) | PASS |
| The command, its end-to-end test, both actions naming it | produced by ρ | PASS (ρ delivers) |

## π — Integration gate

| Capability | Binding | Verdict |
|---|---|---|
| Every contract it exercises | `producer:α…ρ` — all upstream of π in this batch | PASS |
| Real `orchestrator verify-merge` subprocess for the wire rows | substrate — `cli.py` `verify-merge` → `verify_runner.py::run_merge_verify_on_worktree` | PASS |
| Boundary suite file and green rows | produced by π | PASS (π delivers) |
