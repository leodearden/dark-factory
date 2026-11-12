# Capability manifest — S7 merge-lane gates, placement mechanisms and target adapters

PRD: `plans/review-all-s7-merge-lane-gates-placement-mechanisms-and-target-adapters-prd.md`.
YAML twin: `plans/review-all-s7-merge-lane-gates-placement-mechanisms-and-target-adapters-prd.capability-manifest.yaml`.
The twin names signal leaves ζ (the integration gate) and δ. Every `task_id` stays `null` until decompose.

**Code anchors** verified against main `bb1c659562` (2026-11-10, fold-verifier pass). Bindings cite
symbols, not lines. Provenance: authored at `ce0f71a384` (2026-11-01), re-checked at `0935e63d88` by the
critic, at `d40ba048e9` by the fold (2026-11-02) and at `e634cfe658` by the ruling fold. The sitting's
rulings of 2026-11-10 are applied: S7-1 = A (deep chain kept), S7-3 = B (retry narrowing deleted, so
δ's rows are new), and the lead's revised invariant order (ζ is INV-17). The lead's sequencing ruling
puts δ first: the chain is δ → β → α1 → α2 → γ → ε → ζ, and the sections below follow it. The lead's
rulings after the fold-verifiers put S6 α first on every shared file: δ, β and ε depend on S6 α, and
S6 α takes no edge on any S7 leaf.

Polarity was re-measured at `bb1c659562` on 2026-11-10, using
`shared/src/shared/delivered_check_polarity.py::evaluate_grep_at_tree` against `main`, and the sidecar
loads through `shared/src/shared/capability_manifest.py::CapabilityManifestDoc`. All 23 mechanical
checks fail at today's tree, as a producer check must:

- each check with `expect: present` has no match today;
- each check with `expect: absent` matches today;
- the path check with `expect: present` names a file that does not exist today, and the one with
  `expect: absent` names a file that does.

`lint_delivered_checks` reports no finding for any block, given each leaf's declared files. Substrate
rows pass today, so they carry no mechanical check: a check that already passes at authoring is vacuous
by the polarity rule. No check is a `kind: script`.

## δ — Delete reify retry narrowing end to end

| Capability | Binding | Verdict |
|---|---|---|
| The deletion changes no live behaviour | 0 of 5,431 `merge_verify` events since 2026-07-20 carry `retry_scope='failed_only'` (3,433 dark-factory, 1,998 reify; read 2026-11-10). No in-repo caller passes `retry_failed_only=True`. The producer runs only on a local dispatch, whose `verify_runner.py::LocalRunner.run_merge_verify` never reads `spec.verify_env` | PASS |
| No reader of the event marker | `retry_scope` and `retry_subset_sizes` are read nowhere outside tests (`git grep` at `bb1c659562`; the two calibration `.jsonl` files under `fused-memory/calibration/` are data, not readers) | PASS |
| The shadow helpers have no other caller | `shadow.py::build_fail_fast_map`, `did_not_pass_subset`, `nextest_filter_ids`, `_nextest_case_is_planned`, `parse_nextest_list_planned`, `parse_failed_run_all_members`, `select_failed_run_all_members` and `merge_retry_shadow_baseline` are called only from the worker's producer section or from one another (measured at `bb1c659562`) | PASS |
| The warm shadow keeps its parser | `merge_lane/shadow.py::parse_per_test_results`, used by `build_warm_shadow_results`, `_run_cold_shadow_verify` and `_run_coarse_shadow_compare` (substrate; kept) | PASS |
| reify needs no change | reify `scripts/verify.sh` reads `REIFY_VERIFY_RETRY_SCOPE` only when set, and reify's `--confirm-failed` mode drives the same variables itself (the sitting's read, 2026-11-10) | PASS |
| Ordering with the escalation-server plan | S3 ε2 moves `merge_request` into `orchestrator/mcp/merge_tools.py`; S3 ε2 depends on δ, so the move carries one parameter fewer (lead edge at decompose) | PASS (lead edge) |
| Same-file order with S6 α | the lead's rulings after the fold-verifiers (2026-11-10): S6 α (INV-15) lands first and takes no edge on δ; δ depends on S6 α on `merge_lane/worker.py`, `shadow.py`, `types.py`, `test_merge_shadow.py`, `test_merge_queue.py` and the ratchet baseline. S6 α waits on no S7 leaf, so no cycle | PASS (lead edge) |
| No edge with 7720 (δ is first) | lead sequencing ruling of 2026-11-10: β depends on δ, and δ's only dependency is S6 α, so it does not wait on 7720. δ and task 7720 (in progress since 2026-11-10T21:02Z) both edit `merge_lane/worker.py` and the ratchet baseline, so whichever lands second rebases | PASS (lead) |
| A fixture with a reader outside the narrowing | `orchestrator/tests/test_verify_event_channel.py`'s D3 corpus reads `fixtures/reify_verify_retry/run_all-failed-marker.txt` at import (measured at `bb1c659562`). δ moves that file into `fixtures/verify_classify_run_all/` and repoints the corpus entry rather than deleting it | PASS (δ moves it) |
| No edge with 5040 | 5040 finishes first and waits on no S7 task (S6-2 = B). δ's absent checks scan all of `orchestrator/src`, so they hold wherever 5040 moved the section (its declared files include `merge_lane/retry_verify_env.py`) | PASS |
| Producer checks | absent after δ: `retry_failed_only` (orchestrator and escalation sources), `_REIFY_ATTEMPT_SIDECAR_RELPATH`, `verify\.sh:[0-9]` in `merge_lane/`, `retry_scope_event_fields`, the three named shadow helper definitions, `NARROWED_RETRIES` and `_post_merge_verify_narrowed_retries`; path `orchestrator/tests/test_merge_queue_verify_retry.py` absent. All seven match today | PASS |

## β — Delete the variable-depth probe and γ2 auto-chain; adjacent placement plus deep chain

| Capability | Binding | Verdict |
|---|---|---|
| Probe is inert | `speculation_probe.probe_fraction: 0.0` in dark-factory's and reify's `dark-factory-orchestrator.yaml` (read 2026-11-10), and in the 9 reify fix, recovery and unblock checkouts (the sitting's read, 2026-11-10); `defaults.yaml` carries no `speculation_probe` key | PASS |
| Auto-chain is dead by constant | `merge_lane/gates.py::AUTO_CHAIN_GENERATIONS_ENABLED: bool = False`. Its only consumer is `_auto_chain_on_equivalence_blocked`. No `post_merge_generation_chained` row in either run store (read 2026-11-10) | PASS |
| The probe's outcome window is probe-only | `_record_verify_outcome` has a single feed in `_finalize_inflight` and a single reader, `_recent_verify_fail_rate`, which only `_probe_verify_placement` calls (measured at `bb1c659562`) | PASS |
| Deep chain is live and kept | `merge_deep.chain_cap: 6` in dark-factory (`2389803498`, 2026-10-27) and reify (`349aa26775`, 2026-10-20); `worker.py::build_chain` and `_deep_chain_placement` present (substrate). Kept by S7-1 = A, an amendment to R11 | PASS |
| A leftover yaml key is already reported | `config.py` unknown-key census, and `harness.py::Harness._file_config_unknown_keys_escalation` (a severity-critical L2 that self-heals once the config is clean). β adds no bespoke WARNING | PASS |
| reify's live yaml no longer carries the key before β deploys | external producer: a reify-side change filed by the lead (OQ-CQ2, ruled by the lead). The stanza is still present on 2026-11-10. Checked with `orchestrator check-config --config /home/leo/src/reify/dark-factory-orchestrator.yaml` | PASS (upstream) |
| `_land_chain_prefix` edit is serialized with 7720 | task 7720 (in progress since 2026-11-10T21:02Z, high, re-planned 2026-11-10) rewrites `_land_chain_prefix`; β depends on it (OQ-CQ7, ruled by the lead) and on δ (chain). 8527 and 8528 (merge-lane policy) are same-file neighbours with no edge | PASS |
| Same-file order with S6 | β depends on S6 α (worker.py, types.py, config.py, `test_merge_queue.py`, `test_merge_queue_deep_integration_gate.py`, the ratchet baseline; S6 α lands first, lead 2026-11-10 late). S6's remaining worker.py leaves depend on β (OQ-CQ5) | PASS (lead edge) |
| Producer checks | after β, these four are absent: `AUTO_CHAIN_GENERATIONS_ENABLED`, `class ProbePlacement`, `class SpeculationProbeConfig`, `_recent_verify_outcomes`. All four match today | PASS |

## α1 — GateMeasurement, ProbeId, UNMEASURED_POLICY, StormEscalationSink and GateUnmeasuredReporter

| Capability | Binding | Verdict |
|---|---|---|
| Storm-counter primitive | `shared/src/shared/storm_counter.py::StormCounter` (observe/prune) | PASS |
| Storm-escape house pattern | `merge_lane/landing_evidence.py::file_landing_git_error_storm_escalation`: strict exceedance, synthetic sentinel task id, category-scoped `has_open_l1` dedupe | PASS |
| Id minting and dedup on the raw queue | `escalation/queue.py::EscalationQueue.make_id(key)`, `has_open_l1(task_id, *, category)` and `submit(escalation)`; `merge_lane/ports.py::EscalationPort` exposes only `file(record)`, hence the three-method `StormEscalationSink` (OQ-CQ4, ruled by the lead) | PASS |
| No import cycle | `merge_lane/ports.py` imports `gates.PostMergePyrightResult` at runtime (still true at `bb1c659562`), so `gate_measurement.py` must not import `ports.py` at runtime (it uses `TYPE_CHECKING` only) | PASS |
| Event taxonomy to extend | `orchestrator/src/orchestrator/event_store.py::EventType`, a `StrEnum`. `merge_gate_unmeasured` is absent today | PASS (α1 delivers) |
| One reporter per lane, reachable from the workflow | `workflow.py::TaskWorkflow.merge_worker` exists today; α1 adds `SpeculativeMergeWorker.gate_reporter` | PASS |
| Producer checks | `class Unmeasurable`, `class ProbeId` and `^UNMEASURED_POLICY` in `merge_lane/gate_measurement.py`; `merge_gate_unmeasured` in `event_store.py`. None matches today | PASS |

## α2 — Convert the 12 fail-open returns and the sub-probes they own; gates_unmeasured on merge_attempt

| Capability | Binding | Verdict |
|---|---|---|
| 12 fail-open returns to convert | AST enumeration re-run at `bb1c659562`: `_check_plan_targets_in_tree` 5, `_check_post_merge_equivalence` 6, `_check_post_merge_pyright` 1. The gate functions are byte-identical to `d40ba048e9` | PASS |
| Directions of the sub-probes to reproduce | fail-closed: `_branch_delta_survives`, `_commit_is_linear`. Fail-open: `_rename_aware_real_drops`, `_rename_aware_compare_set`, `_resolve_second_parent`. These are their own docstrings, read at `0935e63d88` and unchanged since | PASS |
| Exit-code-as-answer probes | `git rev-parse <linear>^2` exits 128, meaning "no second parent". `_resolve_second_parent` first counts parents with `rev-list --parents -n 1`, the call `_commit_is_linear` already makes | PASS |
| A real git fault for the boundary test | measured by the critic: `git clone --depth 2` of a merge tip gives `^2` rc 0 and `merge-base ^1 ^2` rc 1; `--depth 1` fails earlier, at `^2` with rc 128 | PASS |
| Finite caller set to thread through | 4 `_finalize_advanced_merge` call sites and 4 `classify_and_merge` call sites in worker.py (`_merger_loop`, `_remerge_on_tip`, `_remerge_on_main` ×2, after task 8524) (measured at `bb1c659562`) | PASS |
| Producer checks | `return DropGuardResult\(\)` absent from gates.py (5 matches today); `gates_unmeasured` present in gates.py (none today) | PASS |

## γ — Remaining encodings onto GateMeasurement: overlap, rename pair, already-landed, plan-files

| Capability | Binding | Verdict |
|---|---|---|
| Typed failure from GitOps read queries | `producer:S5/C4`, S5 draft leaf ε: `get_files_touched_in_branch` and its siblings raise a typed failure carrying `returncode` and `stderr`. Not on main: `git_ops.py::GitQueryFailed` (task 7485) is message-only, and `get_files_touched_in_branch` still returns `[]` on `rc != 0` (measured at `bb1c659562`). γ depends on S5 ε at decompose, and is held on S5/C4 if S5 has not filed | PASS (upstream) |
| Same query being rewritten | task 3918 (pending, high, dispatchable) rewrites `GitOps.get_files_touched_in_branch` and may add `plan_delivery.py` (absent at `bb1c659562`); γ depends on it | PASS |
| Recheck path for the overlap gate | `merge_lane/reverify_gate_decision.py::ReverifyGateReason.overlap_unknown` | PASS |
| Silent skips in the caller | the `workflow.py` merge phase runs `if rc == 0 and branch_head.strip():` before `_check_plan_files_touched_in_branch`, and `if rc2 == 0 and branch_head2.strip():` before the post-narrowing re-check (both measured at `bb1c659562`) | PASS (γ converts) |
| Producer checks | `_OVERLAP_GIT_ERROR_SENTINEL` and `_RENAME_PROBE_UNMEASURABLE` absent from `orchestrator/src/`. The first has 12 matches in gates.py today (5 of them prose that γ also removes, with the one in `skills/merge-queue/references/two-layer-model.md`); the second has 4 | PASS |

## ε — Resource-audit violations as records fingerprinted on identity, not regex

| Capability | Binding | Verdict |
|---|---|---|
| Producers to convert | `SpeculativeMergeWorker.speculation_accounting_violations`, which has 3 conservation shapes and 3 "check raised" arms, and `worktree_ledger_violations`, which has the unregistered shape (scheduled or overdue) and a "check raised" arm | PASS |
| Frozen rendering to keep | `snapshot()['resource_audit']` is under an additive-only freeze pinned by exact dict equality in `orchestrator/tests/test_merge_queue_invariant_integration_gate.py` (substrate) | PASS |
| Pinned fingerprint behaviour | `TestResourceAuditFingerprint` asserts that grace and disposition are not elided (task 3203). That is why the fingerprint drops only `age_secs` (OQ-CQ6, ruled by the lead as a clarification of R11) | PASS |
| Same-file order on the resource-audit file | S6 α also edits `orchestrator/tests/test_merge_queue_resource_audit.py`; ε depends on S6 α (S6 α lands first, lead 2026-11-10 late) | PASS (lead edge) |
| Coalescer contract unchanged | `orchestrator/src/orchestrator/log_coalesce.py::ExponentialLogCoalescer`, `SpeculativeMergeWorker.RESOURCE_AUDIT_LOG_COALESCE_CAP_POLLS` | PASS |
| Producer checks | `class ResourceAuditViolation` in `merge_lane/resource_audit.py` (no match today); `_RESOURCE_AUDIT_AGE_FIGURE_RE` absent from `orchestrator/src/` (matches today) | PASS |

## ζ — The merge-gate-unmeasured-typed invariant (INV-17), its guard, and the PRD's integration gate

| Capability | Binding | Verdict |
|---|---|---|
| Four-site registration guard | `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES`: design-invariants.md, the fixtures, gates.md, CONTRIBUTING.md (as an absence) and code_quality.md (partial) | PASS |
| INV number | INV-17 under the lead's revised order of 2026-11-10: S6 α INV-15, then S10 θ INV-16, then ζ; ζ depends on S10 θ, which depends on S6 α. The highest invariant on main is INV-14 (measured at `bb1c659562`). Checks anchor on the slug | PASS |
| R11's scope has no false positives | R11's form, run over the four `_check_*` functions plus nested `_tree_paths`, flags only genuine failures (measured at `0935e63d88`; the functions are unchanged at `bb1c659562`). The wider "every git-calling function" form flags three measured negatives | PASS |
| Guard is code, not prose (INV-10) | an AST scan, a policy-completeness check, sentinel absence, and one behavioural row per `GateId`, each tested against fixtures | PASS (ζ delivers) |
| Integration gate (B+H) | boundary rows 1–15 pass together at ζ's tree | PASS (ζ delivers) |
| Producer checks | `merge-gate-unmeasured-typed` in `docs/legibility/design-invariants.md` (no match today); path `orchestrator/tests/test_gate_measurement_invariant.py` (absent today) | PASS |
