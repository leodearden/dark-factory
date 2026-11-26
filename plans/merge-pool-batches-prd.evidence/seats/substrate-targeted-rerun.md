The targeted re-run is buildable, but not as briefed. The parts it needs exist: failing tests as data, a "run exactly these node ids on this tree" function, prefix commits that stay in git, and a legal way to block a link. Five assumptions in the brief are contradicted by the code:

- **The worktree is gone at the verdict.** The tip worktree is removed before the lane ever sees a red verdict.
- **The lease is already released.** A red chain gives back its host lease immediately.
- **There is no remote subset run.** The laptop runner can only run a full plan.
- **"Base alone" is not main.** The chain's base is often an unlanded speculative commit, and the existing control probe runs on current main instead.
- **The flake gate has already filtered F.** It has already turned non-reproducing reds into passes.

Repo `/home/leo/src/dark-factory`, main at `eafdb1ec73`. Code paths are under `orchestrator/src/orchestrator/`. Every cited symbol was checked by grep.

---

## 1. Failing checks as data

**The dark-factory merge gate as configured.** `/home/leo/src/dark-factory/dark-factory-orchestrator.yaml` sets `merge_verify_breadth: "full"` and `merge_verify_max_concurrent_modules: 1`, so modules run one after another. It sets neither `concurrent_verify` nor `sequential_lint_first`. The defaults (`config.py`: `concurrent_verify=True`, `sequential_lint_first=False`) mean each module's three legs run at the same time, with no fail-fast.

- There are 11 modules: client-feed, cockpit, dashboard, escalation, fused-memory, orchestrator, sampler, scripts, shared, tests/repo_guards, tests/scripts (each has an `<m>/orchestrator.yaml`). Each module has three legs:
  - test: `uv run --directory <m> pytest tests/ --tb=short -q --timeout=300`
  - lint: `ruff check` plus a few guard scripts
  - type: `pyright`
- `LocalRunner.run_merge_verify` (`verify_runner.py`) then runs a second, unscoped type-check gate, `merge_lane/gates.py::_run_unscoped_typechecks`, but only if the scoped run passed.
- `merge_deep.chain_cap: 6`, so chains are live.

**What identifies a failure on a red result** (`verify.py::VerifyResult`):

- `failing_test_ids: list[str] | None`
  - Comes from junit, and only for a role='merge', full-breadth, structured-pytest leg (`verify.py::_JunitReportPlan.for_leg` with kind 'attributing'; parsed by `_extract_test_ids_from_junit`).
  - The spelling is junit `classname::name` (`_junit_testcase_id`), for example `tests.test_x::TestY::test_z`. **That is not a pytest node id**, so it cannot be passed to pytest as-is.
  - `None` means "no junit collected", which happens for: an opaque command, a raw-retained `&&` chain, a missing or garbled report, or a leg killed before it wrote one. `[]` means "collected, zero failing".
- `skipped_test_ids` is the same shape and records tests that never ran.
- `failing_test_ids_by_module: dict[prefix, ids] | None`
  - Set per module run, only when that run collected junit.
  - The aggregate falls back to `None` if any collecting child lacks it (`_aggregate_failing_ids_by_module`).
  - Coverage is checked by `_module_attribution_mismatch`.
- `failing_leg_categories: list[str] | None`
  - One `FailureCategory` per failing leg, in test/lint/type order (from `_summarize_checks`).
  - Fails closed: one unrecorded failing child makes the aggregate `None`.
  - It holds categories only; it names neither the module nor the test.
- `units: tuple[VerifyUnit, ...] | None` (`verify_evidence.py::VerifyUnit`)
  - Fields: `unit` (for example `module:orchestrator:lint`, built by `module_leg_name`), `tier` (`UnitTier` TREE/MODULE/TEST), `key`, `outcome` (`UnitOutcome` RAN_GREEN/RAN_RED/CACHE_HIT/WOULD_HIT/UNAVAILABLE), `evidence_event_id`.
  - Only `module_leg_unit` builds units today, so units are module×leg granularity with no duration field.
  - This is the **only structured identifier for a red lint or type leg**: `module:<m>:lint` or `module:<m>:type` with `ran_red`. Runs.db `merge_verify` rows carry `units`, `plan_hash` and `cache_mode`.
- `flake_suppression: FlakeSuppression | None` (`flake_ledger.py`)
  - Fields: `verdict`, `test_ids` (stdout pytest node ids, cwd-relative to the module), `call_site`, `runner`, `unconfirmable_reason`, `evidence`.
  - It survives the remote wire via `flake_ledger.flake_suppression_from_wire`.
  - **This is F in runnable form.**
- Unscoped type-check gate red: the sentinel category `UNSCOPED_TYPECHECK_FAILED_CATEGORY` / `UNSCOPED_TYPECHECK_TIMEOUT_CATEGORY`. The failing subprojects are recovered by splitting `summary` on commas (`verify_runner.py::unscoped_gate_failing_subprojects`), which is a meaningful-string carrier, not structured data.

**When test ids are absent:**
- Opaque or non-pytest commands. Reify is the case here: `/home/leo/src/reify/dark-factory-orchestrator.yaml` has `test_command: ./scripts/verify.sh test …` and `type_check_command: "true"`. `_with_junitxml_str` leaves the command unchanged and logs INFO, so the result is `None`.
- Zero-module merges (docs-only or root-file) at full breadth run the global `&&` chain: no junit, no flake confirmation (task 7616).
- A crashed or killed test leg.
- A collection error, which interrupts the pytest session (per `run_node_ids`'s docstring).
- xdist worker death: the session is truncated (`_is_worker_death_truncated_session`), so the failing set is partial.
- A timeout: `timed_out`, `infra_timeout`.

**Reify's per-test data** is parsed nextest output instead: `merge_lane/shadow.py::parse_per_test_results` / `did_not_pass_subset` / `select_failed_run_all_members`, plus `VerifyResult.run_all_failed_members`.

**"Stops at the first failing leg" means three different things in the code:**
- (a) **Reify**: `concurrent_verify: false` + `sequential_lint_first: true`. `verify.py::run_verification` (`lint_first = role=='merge' and not concurrent and …`) runs lint first and short-circuits test and type on a lint red. **F is incomplete there.**
- (b) **The global fallback `&&` chain**. `_run_segmented` (run every segment) is opted in only for non-merge roles. The merge role deliberately keeps the shell short-circuit (comment at the `run_verification(... fallback ...)` call in `_run_scoped_verification_body`). So on dark-factory's zero-module merges, F is incomplete and opaque.
- (c) **Dark-factory's per-module path does not stop.** All 11 modules and all three legs run, so F is complete except for truncation, timeouts and collection errors.

**PRESENT:** `VerifyResult.failing_test_ids` / `failing_test_ids_by_module` / `skipped_test_ids` / `failing_leg_categories` / `units` / `flake_suppression.test_ids`; `verify_evidence.py::module_leg_name`; `verify.py::red_module_prefixes_of`.

**MISSING:**
- A junit-id → pytest-node-id converter. Today only the reverse exists: `verify.py::_junit_spelling`, used by `_node_id_statuses`.
- Durations per unit or per test in `units`.
- Structured failing subprojects for the unscoped gate.
- Any F for reify beyond nextest ids plus run_all members.

## 2. Existing re-run machinery

**(a) The flake gate**

- Entry point: `verify.py::apply_merge_flake_suppression(failing_result, *, worktree, config, module_configs, _confirm=...) -> VerifyResult`. It is called from `verify_runner.py::LocalRunner.run_merge_verify` on a scoped red, inside the verify, before it returns.
- It calls `confirm_merge_verify_flake_suppressible`, which calls `confirm_isolated_rerun_verdict(worktree, config, module_configs, failing_result, *, call_site, runner='local', now=None) -> FlakeSuppression`.
- The node ids come from **stdout** (`_extract_failing_test_ids(failing_result.test_output)`), not from junit.
- It returns `unconfirmable` for:
  - `merge_no_evidence`
  - any lint or type output (`other_leg_failed`)
  - `_unmeasured_session_reason` (`session_truncated`, `leg_without_verdict:<cat>`)
  - no node ids
  - ids that `_group_node_ids_by_subproject` cannot map (it probes `Path.exists` on the worktree)
- Per module group it builds `_node_id_test_config(mc, ids, tree=, timeout_secs=_MERGE_FLAKE_CONFIRM_TIMEOUT_SECS)`, where the timeout constant is 300. That config is `_scope_to_keyword(test_command,'pytest',ids)` → `_serial_confirm_pytest_str` (no xdist) → `_with_pytest_timeout_str`, with lint and type set to None.
- It runs `_merge_gate_isolated_rerun`, which is `run_verification(worktree, config, scoped_mc, max_retries=0, is_merge_verify=True, role='merge')`. That is single-shot, judged by `_classify_rerun_result`.
- Where it runs: the **same worktree, same host, same sha** as the verify (INV-3 same-tree). That is local, or the laptop when the verify ran there, because `run_merge_verify_on_worktree` builds a `LocalRunner` on the laptop.
- What it returns: on `passes_in_isolation` it flips the result to a PASS (`_merge_flake_suppressed_pass`, category `merge_flake_suppressed`) and the merge proceeds to the unscoped gate. Otherwise it attaches the observation and the merge stays red.
- Recording happens on the dispatcher: `merge_lane/post_merge_verify.py::_PostMergeVerify._record_observation` → `flake_recorder.py::record_merge_flake_suppression`. The tables are `flake_occurrence`, `flake_debt` and `flake_debt_due` in `data/orchestrator/runs.db`, plus a `merge_flake_suppressed` event. None of these records a duration.
- It can only re-run on the tree it is handed. It is not pointable at another commit unless you give it another worktree.

**`verify.py::run_node_ids(tree, config, module_configs, ids, *, concurrent_groups=False) -> Mapping[str, NodeIdStatus]`** (task 7918, done)

- Statuses are 'pass'/'fail'/'skipped'/'error'.
- It shares the same mapping and `_node_id_test_config` (timeout `_NODE_ID_RUN_TIMEOUT_SECS=300`), writes a junit report per group, and bisects runs that pytest refuses (`_bisect_refused_run`).
- **It has no production caller** (only `orchestrator/tests/test_run_node_ids.py`).
- It is the closest existing primitive: "run exactly F on tree T".

**(b) Narrowed retry**

- `merge_lane/post_merge_verify.py::_PostMergeVerify._retry_infra_transient` → `_narrowed_retry_spec`.
- It fires only on an **infra-transient** red, only when `req.retry_failed_only` is set (default `False`, `merge_lane/types.py`), and only for a **local** dispatch (`self.runner is not None` → None).
- `merge_lane/retry_verify_env.py::build_attempt0_payload` + `assemble_retry_verify_env` produce reify-only env keys: `REIFY_VERIFY_RETRY_NEXTEST_FILTER_FILE_*`, `REIFY_RUN_ALL_MEMBER_SUBSET`, and so on, with tree-OID corroboration.
- It re-runs the same `merge_sha` with the same spec plus that env. `verify_runner.py::retry_scope_event_fields` stamps `retry_scope` / `retry_subset_sizes`.
- Dark-factory: `retry_scope` is null on all 2,320 `merge_verify` rows since 2026-10-01. It is not usable for a different commit or for a test-failure red.

**(c) Main-health probe**

- `verify.py::verify_failure_is_preexisting_on_main(worktree, config, module_configs, task_files, failing_result, git_ops, *, flake_sink=None) -> (bool, str)`.
- It probes **current main HEAD** (`git_ops.get_main_sha()`), never the batch base.
- When there are ids, it uses `main_baseline_failing_ids(config, module_configs, git_ops, main_sha, *, red_module_prefixes=None)`:
  - Cache first: `_BASELINE_FAILING_IDS_CACHE`, a 16-sha LRU. It is seeded on every green gate by `seed_main_baseline`.
  - On a miss, a local `git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, main_sha, warm_seed=True)` runs either a whole-tree `run_scoped_verification(... role='merge')` or a per-red-module `run_verification` (`_probe_modules_on_main`).
  - It is **deliberately never narrowed to test ids**: "module granularity keeps the identical command and selection context (task 4585 measured selection-dependent failures)".
- Then a signature comparison (`category` + normalised `cause_hint`), cached in `_PROBE_CACHE` with a 300 s TTL, plus the isolated-flake confirm (`_main_probe_failure_is_isolated_flake`).
- It is leaseless and local-only (never `HostAllocator` or `RemoteRunner`). Lane verifies run it detached, through `main_health_probe.py::spawn_main_health_probe` → `_run_deferred_main_health_probe`.
- Its bound is the module verify timeout. No duration is recorded.

**(d) Other subset / commit entry points**

- `git_ops.py::GitOps.ephemeral_worktree(kind, sha, *, warm_seed=False)` takes **any sha**. `WorktreeKind` has only `MAIN_PROBE` and `MAIN_SWEEP`. `create_ephemeral_verify_worktree(sha)` and `acquire_spec_lane(merge_commit)` also take any sha.
- `scripts/merge_screen.yaml` is a list of node-id "static checks". **Its only reader is the pre-commit hook** (`hooks/project-checks`, `scripts/hook_module_checks.py`). `merge_lane/screen.py` is **ABSENT**; tasks 7919/7922 are pending.
- `plan_hash` (`verify_evidence.py::plan_hash`) identifies a plan only. `cache_mode` is `off`.
- **Remote (laptop) runner.** `verify_runner.py::RemoteRunner._dispatch(sha, subcommand, spec_json, ...)` pushes **any sha** to `refs/merge-verify/<id>` and runs `RemoteVerifySubcommand = Literal['verify-merge','verify-task']`. `verify-merge` always runs the full `LocalRunner` bundle: scoped, then the flake gate, then the unscoped gate. There is no subset subcommand. A spec whose `VerifyCommand.test_command` was pre-scoped to node ids would technically ride the wire, but that is an unvalidated fidelity-guarded path (`test_merge_verify_config_fidelity.py`). It is one in-flight request per runner (`ConcurrentDispatchError`).

**PRESENT:** `run_node_ids`, `_node_id_test_config`, `_group_node_ids_by_subproject`, `confirm_isolated_rerun_verdict`, `ephemeral_worktree(kind, sha, warm_seed=True)`, `main_baseline_failing_ids`.

**MISSING:**
- A production caller of `run_node_ids` against a non-tip commit.
- A `WorktreeKind` band for prefix trees.
- A leg-only rerun for lint or type (it would have to be built as `replace(mc, test_command=None, …)`; nothing does that).
- A remote "run F at sha" protocol.

## 3. The prefix trees

**What a chain build produces.**
- `merge_lane/worker.py::_reserve_and_build_chain` seeds the chain at `item.merge_result.merge_commit`, which is the owner's merge commit (base plus owner). It was built in the owner's own ephemeral `_merge-<uuid>`.
- `build_chain(git_ops, queue_snapshot, head_merge_commit, *, cap, target_depth, verify_is_local)` merges links sequentially in **one** lane (`liveness.py::acquire_chain_build_lane`):
  - local verify: a `_spec-` lane via `acquire_spec_lane`
  - remote verify: a local `create_ephemeral_verify_worktree` (`_merge-`, unseeded)
- `ChainResult.links = [(task_id, cumulative_merge_commit)]`, plus `link_request_ids`, `tip`, `lane`, `lane_warm` and `build_ms`.
- So every prefix tree exists as a commit: the owner's commit, then link 1, …, link k−1. The tip's first-parent chain goes through all of them. `base+member_i` alone exists **nowhere**; building it means a fresh `git merge`, which may conflict or fail on a member stacked on a predecessor.

**Reachability.**
- `InflightVerifyRun._release_chain_resources` ends the reservation, returns the lane (`release_chain_build_lane`) and drops the owner's `_merge-<uuid>` (`_drop_chain_item_ephemeral_wt`).
- No ref points at link commits:
  - `refs/merge-verify/<id>` is deleted after a remote verify.
  - `DeadBaseLedger` (`dead_base_ledger.py`, including `DeadBaseSite.CHAIN_RELEASED`) is in-memory only and records shas, not refs.
  - The only `refs/dark-factory/*` namespaces are rebase-dropped, merge-park and stack-base.
- After the lane goes, the commits are unreferenced loose objects, kept only by a pool lane's per-worktree reflog for as long as that lane exists. They are nonetheless safe in practice:
  - main repo `gc.auto=0`, `maintenance.auto=false` (`git_ops.py::GitOps.disable_shared_repo_auto_maintenance`)
  - the only scheduled maintenance is `scripts/shared_repo_git_maintenance.py`, which runs `repack -d -q -n` (no prune)
  - so they persist indefinitely, which is what task 7922 already relies on ("checks the link commits out by sha, since they are in the shared object store")
  - on the laptop they arrive as ancestors of the pushed tip; the laptop's gc config was not checked
- Re-checkout: `ephemeral_worktree(..., warm_seed=True)` or `acquire_spec_lane(sha)`.

**Recorded cost** (last 14 days, runs.db `merge_verify.chain_build_ms`, which covers lane acquire plus up to 5 merges):

| Runner | Mean | Max |
|---|---|---|
| local | 4–5 s | 19 s |
| laptop-path (ephemeral) | 8–11 s | 125 s |

A cold venv pre-provision showed up in the journal at 13.8 s (2026-11-20 14:10:19, `_merge-961c18b9`).

**Running before the lane is released: not as wired.** `post_merge_verify.py::_PostMergeVerify.run` calls `git_ops.cleanup_merge_worktree(self.merge_wt)` on **every red, before** `red_verify_outcome`. For a chain, `merge_wt` is `chain.lane`, which is not persistent, so it is removed. A prefix run on the same worktree is therefore possible only inside the verify, in the `LocalRunner` slot where the flake gate runs, or after re-acquiring a tree.

**PRESENT:** `ChainResult.links` / `link_request_ids` / `tip`, object-store retention, `ephemeral_worktree(sha)`, `acquire_spec_lane(sha)`.

**MISSING:**
- Threading `ChainResult.links` into the verify (`MergeVerifySpec` and `LocalRunner` do not know the chain).
- Keeping the lane on red.
- A per-prefix tree-prep step that also handles venv drift when a link changes `pyproject`/`uv.lock` (cold pre-provision is about 14 s).

## 4. The host and the lease

**A red chain tip is handled like this:**
- `InflightVerifyRun._settle_chain` sends a red to `custody.requeue_from_verify` → `worker.py::_requeue_from_verify` → `_vacate_inflight_entry(slot, REQUEUED)`. That releases the lease immediately.
- `_verify_releasing_at_verdict` / `_release_lease_at_verdict` (task 8526) return early because `vr.status is not None`.
- A **red single** blocked outcome behind the head has its lease released at the verdict; the head keeps its lease through finalize.
- So nothing holds the host after the verdict. Holding it for minutes means one of:
  - running inside the verify task, before `pool.dispatch` returns, as the flake gate does. That works on the laptop too, but the in-flight no-progress and abandon triggers (`InflightVerifyRun._await_verdict`) watch that time.
  - re-acquiring through `verify_runner.py::HostAllocator.acquire`.

**On the remote:** it can run a full plan at any sha (`RemoteRunner.run_merge_verify(merge_sha, spec)`), but not a subset (see 2d).

**`merge_lane/verify_cross_check.py::cross_check_remote_green`** re-runs the **full** merge verify locally, on a remote **green** only, on the intact local merge worktree. It is **off for dark-factory** (`verify_cross_check_remote_green: false`), it does nothing on a remote red, and it costs one full local verify (mean about 42 min).

**PRESENT:** `HostAllocator`, the `LocalRunner` in-verify hook slot (the precedent is `apply_merge_flake_suppression`).

**MISSING:** a lease hold across the targeted phase, a remote subset subcommand, and a decision on running locally vs on the origin host. The main probe deliberately refuses remote affinity: see the `verify_failure_is_preexisting_on_main` docstring.

## 5. Cost (runs.db, read-only)

- **Not recorded anywhere:** flake-rerun duration, main-probe or baseline duration, and per-unit durations. `merge_flake_suppressed` carries only `node_ids`, `merge_sha` and `measured_at`. `VerifyUnit` has no duration. `merge_verify` row keys are only: runner, merge_sha, passed, duration_ms, attempt, depth, speculative, chain_items, chain_build_ms, retry_scope, retry_subset_sizes, plan_hash, cache_mode, units, runner_reason, base_relation, chain_request_ids, chain_task_ids, chain_depth_reason.
- **Full gate for comparison** (merge_verify, since 2026-11-06):

  | Runner | Verdict | Count | Mean | Range |
  |---|---|---|---|---|
  | laptop | red | 116 | 29.8 min | 10.8–70.5 |
  | laptop | green | 321 | 26.0 min | 15.6–74.9 |
  | local | red | 80 | 41.9 min | 20.3–139.9 |
  | local | green | 273 | 42.2 min | 19.0–148.7 |

  Red chains (chain_items > 1): 56 laptop, 48 local.
- **What F looks like on red chain tips** (merge_verify joined to flake_occurrence by merge_sha, since 2026-10-21):
  - 104 red chain tips. 87 (84%) were `fails_in_isolation`, averaging 2.1 ids. 17 (16%) were `unconfirmable`.
  - Singles: 198 `fails_in_isolation`, 68 `unconfirmable`.
  - Unconfirmable reasons (merge_gate, all rows): `other_leg_failed` 59, `node_ids_unmapped_to_subproject` 41, `session_truncated` 11, `leg_without_verdict:infra_timeout` 2.
- **Timeouts as bounds:** 300 s per test for the flake gate, `run_node_ids` and the main confirm. The command budget is the module's `verify_command_timeout_secs` (600–7200 s warm; cold 7200/10800). Measured pyright per module: orchestrator 220 s, fused-memory 113 s, shared 64 s, dashboard 51 s (config comment above `type_check_command`). Whole-tree-scan tests reach about 132 s each (task 6556).
- **Journal timing (weak):** the 2026-11-20 14:10 local flake confirm logged its two group verdicts at 14:10:26 and 14:10:40, about 14 s for one serial node-id group. Interleaved verifies make the exact attribution uncertain.

## 6. Soundness hooks

**How the lane classifies a red.** `merge_lane/verify_failure_outcome.py::red_verify_outcome(verify, ctx: RedVerifyContext) -> MergeOutcome` checks, in order:
1. flock contention
2. unscoped gate (`_unscoped_gate_outcome`)
3. ENOSPC
4. `INFRA_TRANSIENT_CATEGORIES` (`_transient_infra_outcome`)
5. synchronous `classify_main_health_red` (only when there are no handles)
6. `_task_fault_outcome`

**What `_task_fault_outcome` builds:**
- `MergeOutcome('blocked', reason=<_task_fault_reason + failure_report + skew suffix>, failure_category=verify.category, failure_cause_hint=verify.cause_hint, disposition, failure_diagnostic, skew_evidence, block_category=MergeBlockCategory.POST_MERGE_VERIFY)`
- `MergeBlockCategory` is in `merge_lane/outcome_facts.py`.
- `disposition` needs the `merge_base_sha`/`main_sha` facts (`_skew_attribution`); otherwise it is INDETERMINATE.
- It also spawns a dry run (`ctx.spawn_dry_run`) and the deferred main probe (`spawn_main_health_probe`).

**This path is not chain-aware.** A red chain tip today runs it against the **owner** request, spawning a dry run and a main probe, and then `_settle_chain` discards the outcome.

**Resolving a link that is still buffered.**
- Links are LANE_BUFFERED and only reserved (`merge_lane/chain_reservation.py::ChainReservations`, deadline `merge_deep.chain_reservation_timeout_secs`, default 7200).
- The legal pattern is the one in `worker.py::_land_chain_prefix`: `self._intake.take_buffered(task_id)` (synchronous, verifier-task writer, which may return None, meaning stop) → `_request_abandoned` check → `_resolve_or_drop_abandoned(req, outcome)`. LANE_BUFFERED → TERMINAL is legal in `worker.py::_LEGAL_TRANSITIONS`; no permits are released.
- `intake_buffers.py::remove_buffered` asserts merger ownership (`_assert_merger_owns`), so it is **not** callable from the verifier task.
- The owner itself is in VERIFYING and is resolved through the ordinary `_settle_single`-style outcome.

**PRESENT:** `red_verify_outcome`, `_task_fault_outcome`, `RedVerifyContext`, `take_buffered`, `restore_buffered`, `_resolve_or_drop_abandoned`.

**MISSING:**
- A chain-aware outcome split: owner vs culprit link vs exonerated.
- A reason builder that names the convicting tree.
- A decision on `consecutive_merge_thrash`: deep-merge PRD decision #9 says deep fails never feed it.
- Suppressing the owner-targeted dry run and main probe on a red tip.

## 7. Pending work on the same ground

- **7922** (pending): chain-tip merge screen. Bisect the link commits by sha in a `_screen-chain` lane with static node-id checks, at most ⌈log₂ links⌉+1 screens, and truncate with no outcome. Same prefix search; it **pre-empts** the verify rather than attributing it. Seam: shared lane and prefix search.
- **7919** (pending): screen lanes and manifest; 7922 depends on it.
- **7918** (done): `run_node_ids`, no caller yet. This is the primitive.
- **10069** (gate): design the evidence-chosen batches; suspects and conviction (§6.3–6.6 of `plans/merge-pool-batches-design-notes.md`). Amended 11-18: task 8528 landed **without** the suspect-list bisect. On main, a red tip runs `mark_next_dispatch_unchained`, so every member re-verifies un-chained. This mechanism is the conviction half.
- **8932** (pending): idle-host prefix hedge for chains. Adjacent: prefix trees.
- **7825** (pending): `merge_deep.chain_tip_hosts`. Bears on where targeted runs may run.
- **4585** (pending): selection-dependent tests vs the main baseline. A soundness limit on isolated F.
- **5292**, **6290**, **6506**, **8410**, **6869**, **8695** (pending): harden the discriminator and its timeout injection, single-module mapping, and failure evidence. All are changes to the code this mechanism would reuse.
- **7616** (pending): zero-module merges get no junit or flake confirmation. F is unavailable there.
- **7873** (pending): cold `_merge-*` chain lanes are not on the owned ledger. Holding the lane longer worsens it.
- **8878**, **8912** (pending): `_bisect_refused_run` performance.

---

## Traps (where the brief contradicts the code)

1. **The tip worktree is gone at the verdict.** `_PostMergeVerify.run` removes `merge_wt` (the chain lane) on every red before the lane sees it. Same-worktree prefix checkouts need either an in-verify hook or a re-acquired tree.
2. **No lease at the verdict.** A red chain becomes REQUEUED, and `_vacate_inflight_entry` releases the host at once.
3. **The flake gate already ran F on the tip and filtered it.**
   - Any red whose named tests pass in isolation became a **pass** (`merge_flake_suppressed`) before the lane saw it.
   - So the lane's F means "failed alone on the tip": 84% of red chain tips have it.
   - The other 16% (lint or type co-failure, unmapped ids, truncated sessions) are exactly the cases a targeted test-id re-run also cannot handle.
4. **The test id spellings disagree.**
   - `failing_test_ids` is junit `classname::name`, not runnable.
   - Runnable ids come from stdout (`flake_suppression.test_ids` / `_extract_failing_test_ids`) and are module-cwd-relative.
   - The mapping uses `Path.exists` on the tree. **A test added by member i does not exist on earlier prefixes or the base**, which `run_node_ids` reports as `'error'` / unmapped, not "pass".
5. **Selection dependence.** The main probe refuses to narrow to test ids (task 4585): a test failing only in full-module context passes alone on every prefix and wrongly clears every member. The flake gate makes this mostly moot for tip F, but the same isolated form must be used on every tree for the verdicts to compare.
6. **"Base alone" is not main, and the existing control probes the wrong tree.**
   - The chain base is the owner's `base_sha`. For speculative owners that is a frozen-prefix commit not yet on main (the sample red row: `speculative: true`, `base_relation: stacked`).
   - `verify_failure_is_preexisting_on_main` probes current main HEAD, so it is not this control.
7. **Lint and type legs have no test-level F.** Only `module:<m>:{lint,type}` units exist. Re-running a type leg is a whole-module pyright (orchestrator 220 s), and the unscoped gate's failing subprojects are parsed from `summary`.
8. **Reify's F is incomplete by construction.** Lint-first fail-fast plus an opaque `verify.sh` means no junit. Its subset vocabulary is nextest ids and run_all members, consumed only by reify's own `verify.sh` env contract.
9. **The remote host cannot run a subset.** `verify-merge` is a full-plan bundle. Remote narrowed retries are refused (the spec docstring and `_narrowed_retry_spec`).
10. **`base+member_i`-alone trees do not exist.** Building them may conflict, or fail on stacked members. The prefix trees do exist and stay retrievable by sha: gc.auto=0, no prune.
11. **Blocking a link needs the verifier-side path.** `take_buffered` plus `_resolve_or_drop_abandoned`, while the reservation is live (7200 s deadline) and before expiry hands the link to the merger. `remove_buffered` is merger-only.
12. **The red path is not chain-aware.** A red tip already fires an owner-targeted dry run and main probe whose outcome is thrown away.

## Cost estimate for dark-factory: re-running F on one tree

**About 0.5–2 minutes per tree for a test-id F; up to about 5 minutes when F includes a whole-tree-scan test or a type leg.** Basis:

- **Tree prep, 5–20 s.** Recorded `chain_build_ms` (lane acquire plus merges) averages 4–5 s local and 8–11 s on the remote path, max 125 s. Add about 14 s if the venv is cold, measured in the journal.
- **F size.** About 2 node ids per red tip (flake_occurrence, 2.0–2.1 per sha), so usually 1–2 module groups.
- **Per module group, roughly 10–60 s.** It runs serially through `uv run … pytest <ids>` without xdist. One journal group took about 14 s. Bounds: 300 s per test, and up to 132 s for a whole-tree-scan test.
- **A lint leg costs seconds. A type leg costs 11–220 s** per module (measured pyright).

**For a full attribution:**
- A linear scan (control plus up to 6 prefixes, `chain_cap` 6) takes about 4–14 min.
- Bisection takes about ⌈log₂ k⌉+1 ≈ 4 runs, about 2–8 min.
- Today, every member of a red chain re-verifies the full gate un-chained, at a mean of 26 min (laptop) to 42 min (local) each.

No runs.db field records any of this, so the numbers come from build telemetry, one journal sample and the code's timeouts.
