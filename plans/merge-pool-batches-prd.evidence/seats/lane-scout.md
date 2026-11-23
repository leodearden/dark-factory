# The merge lane on main at bdf7057360 (2026-11-19): scout map for task 10069

The 2026-11-13 lane.md is badly stale on the chain policy. Task 8528 landed without the suspect-list bisect. A red chain now puts an "un-chained mark" on every member (tasks 10617 and 10723), and task 9935 still plans around a suspect list that does not exist. Details are in "Surprises".

All paths are under `/home/leo/src/dark-factory/orchestrator/src/orchestrator/` unless stated otherwise.

---

## 1. Chain (batch) formation today

**One pick site.** `merge_lane/intake_buffers.py::MergeIntake.pop_next_pickable` (`def pop_next_pickable(self) -> MergeRequest | None`).
- It has two callers in `merge_lane/worker.py`: `SpeculativeMergeWorker._acquire_next_request` (the regular pick) and the merger loop's speculative look-ahead (after `_speculation_controller.acquire_for_lookahead()`).
- What it does, in order:
  - returns None while admission is halted;
  - runs `_rebucket_relaned_requests`;
  - walks the lanes high then normal, skipping halted lanes;
  - picks with `_clique_minimal_visible(lane)`: among unreserved requests, the FIFO-earliest one that no same-lane footprint neighbour out-ages;
  - deletes the pick from its buffer and moves it LANE_BUFFERED → MERGING;
  - records `_pick_records.record(...)` (task 9908, see Q4).
- `_aging_key(req)` is now `(req.first_enqueued_at, req.request_id)`. `MergeRequest.first_enqueued_at` is a property: `merge_first_enqueued_at`, falling back to `enqueued_at`.
- The pick does not skip un-chain-marked requests. They are picked normally.

**One dispatch and placement site.**
- Every dispatch goes `worker.py::SpeculativeMergeWorker._dispatch_deciding` → `merge_lane/verify_dispatch.py::VerifyDispatcher.dispatch` → `_dispatch_on_free_host` → `VerifyDispatcher._place`.
- `_place` builds `MergeVerifyDispatchFacts` and calls `ChainPlanning.deep_chain_placement`, which `worker.py` wires to `SpeculativeMergeWorker._deep_chain_placement` (`merge_lane/dispatch_ports.py::ChainPlanning`).
- A gate re-verify of a rebased head goes through the same path (via the redispatch park, `_dispatch_rebased_head`). That is why task 10721 exists.

**Who chooses members and depth:** `_deep_chain_placement(self, item, *, frontier_depth=None, verify_is_local=True) -> DeepPlacement`. Its guards, in order:
1. `may_chain(speculative=..., frontier_depth=...)` (`worker.py::may_chain`): `speculative or frontier_depth == 0`. A head alone in the verify frontier may now chain (task 8527 landed).
2. `merge_deep.chain_cap <= 0`: the kill switch, read live.
   - 2b. `task_id in self._next_dispatch_unchained`: the un-chained mark.
3. `snapshot = self.chain_snapshot()`, then `d, reason = choose_depth(self._chain_policy, 1 + len(snapshot), deep.depth_table)`. If `d < 2` it returns `DeepPlacement(chain=None, depth_reason=reason)`.

After that, `_reserve_and_build_chain`:
- **Members are `snapshot[:d-1]`**, built onto the dispatching item's own merge commit (`item.merge_result.merge_commit`). The dispatching item is member #1.
- `ChainReservations.reserve_candidates`, then `_build_deep_chain` under `asyncio.timeout(chain_build_timeout_secs)`, then `confirm`.

`worker.py::SpeculativeMergeWorker.chain_snapshot` returns high lane then normal, FIFO, deliberately not clique-ordered. It skips:
- halted lanes;
- requests reserved by another chain (`_chain_reservations.holds`);
- requests in `_coalescing_ids`;
- requests whose result future is already done;
- requests whose task is in `_next_dispatch_unchained`.

`worker.py::build_chain(git_ops, queue_snapshot, head_merge_commit, *, cap, target_depth, verify_is_local=True) -> ChainResult` merges sequentially in snapshot order. It truncates at:
- the first refusal (`_link_refusal`: `'train_request'` or `'stacked_on_unlanded_base'`);
- the first non-clean merge (`conflict`, `already_merged` or `merge_error`);
- `lane_base_mismatch`.

A truncated item gets no outcome.

**Task 8528 as landed** (`merge_lane/chain_policy.py`):
- `TipVerdict(depth, passed)` and `ChainPolicyState(recent: tuple[TipVerdict, ...] = ())`. The state has only `recent`, with no suspects. `RECENT_WINDOW = 30`.
- `per_item_pass_estimate(recent) -> float | None` is `pass_rate ** (1 / mean_depth)` over the window, or None when the window is empty.
- `choose_depth(state, queue_depth, depth_table) -> tuple[int, ChainDepthReason]`:
  - with no estimate it uses `depth_table[0]` (basis `NO_ESTIMATE`);
  - otherwise it uses the first row with `estimate >= min_pass_estimate` (basis `PASS_ESTIMATE`);
  - with no matching row the depth is 1;
  - the result is `min(row.depth, queue_depth)`.
- `on_tip_result(state, depth, passed)` appends a verdict and trims the window to 30.
- `seed_from_rows(rows)` seeds from `merge_verify` rows:
  - it keeps the last attempt per `merge_sha`;
  - a row with a null `chain_depth_reason` is excluded;
  - a legacy row with no such key is included;
  - depth comes from `chain_items`.
- Live state is `SpeculativeMergeWorker._chain_policy`, in memory only:
  - `seed_chain_policy()` runs at start and reads `fetch_recent_events_by_type(merge_verify, limit=CHAIN_POLICY_SEED_ROWS)`, where `CHAIN_POLICY_SEED_ROWS = 300`;
  - `_note_tip_result(reason, built_depth, out)` folds in a verdict only when `reason is not None` and `_is_code_verdict(out)`, so a low-disk skip is excluded.
- The window's feeders are `InflightVerifyRun._settle_chain` (built depth) and `_settle_single` (depth 1). Single verifies count only if the policy sized the round.
- **Table** (`config.py::MergeDeepConfig.depth_table`, `list[DepthRow]`, where `DepthRow(min_pass_estimate, depth)`): `[[0.992,12],[0.990,11],[0.988,10],[0.984,9],[0.979,8],[0.972,7],[0.959,6],[0.934,5],[0.876,4],[0.75,3]]`. Below 0.75 the depth is 1.
  - The validator `_validate_depth_table` requires the table to be non-empty, thresholds in [0,1], depths ≥ 2, and both columns strictly descending.
  - `chain_cap` is a kill switch only and bounds nothing. `dark-factory-orchestrator.yaml` sets `chain_cap: 6`, which just means "on", and `chain_build_timeout_secs: 300`.
- `chain_depth_reason`: `merge_verify_facts.py::ChainDepthReason(basis, pass_estimate, row, queue_depth)`. `.wire()` gives `{basis, pass_estimate, row:[thr,depth]|null, queue_depth}`.
  - Live sample: `{"basis":"pass_estimate","pass_estimate":0.8616,"row":[0.75,3],"queue_depth":63}`.
- `select_chain_depth`, `next_halving_state`, `_chain_halving_state` and `_note_chain_outcome` are deleted everywhere, src and tests.

**Helper modules:**
- `merge_lane/chain_policy.py`: the pure depth policy (pass estimate, depth choice, window update, seeding).
- `merge_lane/chain_reservation.py`: `ChainReservations` holds a chain's buffered links from snapshot to round end.
  - It has a provisional reserve, then `confirm` with a deadline from `chain_reservation_timeout_secs`, 7200 s by default.
  - It also provides `release`/`expire`. Memory only.
- `merge_lane/train_selection.py`: pure line-range stackability, `select_train_members`, for trains and coalescing. It has nothing to do with chains.
- `merge_lane/speculation_controller.py`: the merger-side speculation permit state machine (`PermitLedger`, `SpeculationController.base_for`, `on_dequeue`, `acquire_for_lookahead`). K comes from `harness.py::Harness._speculation_k`, which is hosts + 1.
- `merge_lane/lookahead_hold.py`: `LookaheadHold` makes the merger's look-ahead pop wait for the verifier's dispatch decision on the item it just queued. The decision is the successor base, which is the chain tip if the item chained.

## 2. The un-chained mark (tasks 10617 and 10723)

- **State:** `SpeculativeMergeWorker._next_dispatch_unchained: set[str]`, keyed by **task_id**, not request_id. It is in memory only:
  - it is not journaled, and `PersistedMergeRequest` has no field for it;
  - nothing rebuilds it on restart, so after a restart a marked request reloaded from the journal can chain again;
  - no `stop()` clear exists either.
- **Set by** `ChainPlanning.mark_next_dispatch_unchained`, which wraps `self._next_dispatch_unchained.update(ids)`. It has two callers in `merge_lane/inflight_verify.py`:
  - `InflightVerifyRun._settle_chain`, on a red tip, marks `[owner task_id, *link task_ids]`;
  - `InflightVerifyRun._on_verify_error`, on the chain arm only, marks the owner.
- **Cleared only by** `SpeculativeMergeWorker._retire_item`, which discards the task id when the request is retired on any exit: landed, blocked, abandoned or dropped. A verify that the cascade cancels or voids keeps the mark. A resubmission (a new request) may chain again.
- **What a marked request does differently:**
  - it is excluded from `chain_snapshot`, so it can never be a link;
  - guard 2b makes its own dispatch return `NO_DEEP_PLACEMENT` (no chain, and `chain_depth_reason` null);
  - so its solo verdict does not feed the pass-estimate window (`_note_tip_result` returns early on a None reason);
  - it is not reordered, prioritised or front-queued, and `pop_next_pickable` picks it in normal order;
  - "alone" means un-chained only: it still stacks speculatively on in-flight predecessors like any item.
- **A red tip, step by step:**
  1. `InflightVerifyRun.run`'s finally calls `_release_chain_resources`. That calls `end_chain_reservation(owner_rid)` → `_end_chain_reservation(all_landed=False)`, which releases the links and records the **tip** dead (`DeadBaseSite.CHAIN_RELEASED`). It then releases the chain lane and drops the owner's ephemeral worktree.
  2. `_settle_chain` folds the built depth into the window, then marks the owner and all links.
  3. `custody.requeue_from_verify` → `worker.py::SpeculativeMergeWorker._requeue_from_verify` vacates the entry as REQUEUED, then `_requeue_request(item.request)` puts the **owner** back on `_queue`. That queue drains into the **tail** of its lane buffer, so the owner loses its slot. Its aging key is kept.
  4. The **links never left their buffers**, so they keep their exact positions.
  5. When the vacated entry reaches finalize, `_finalize_inflight` treats REQUEUED as did-not-land (`_n_failed=True`, return False). The `_verifier_loop` head-failure cascade then cancels and re-merges every downstream in-flight entry.
  6. The owner's own speculative merge commit is **not** recorded dead. Task 8933, in progress, fixes this.
  7. Nobody gets a blocked outcome, a merge_attempt row or a thrash-ladder entry. Each member later gets its own ordinary verify, and a red there blocks through the normal path in Q3.
- **No suspect list and no bisect code** from 8528 remains in src. The "suspect" hits in `merge_deferred_recovery.py` are unrelated. The ruling is recorded at `/home/leo/src/dark-factory/plans/8528-unchained-mark-study-2026-11-18/REPORT.md`, and the tests are in `/home/leo/src/dark-factory/orchestrator/tests/test_merge_queue_deep_unchained_mark.py`.

## 3. Outcomes

**A request verified alone fails.**
1. `merge_lane/verify_failure_outcome.py::red_verify_outcome(verify, ctx) -> MergeOutcome` takes the first matching arm:
   - flock contention: `blocked`, carrying `failure_category`;
   - unscoped gate, via `_unscoped_gate_outcome`: `block_category=POST_MERGE_VERIFY`;
   - ENOSPC or an infra-transient category, via `_transient_infra_outcome`: TRANSIENT_INFRA prefix, default `MERGE_ERROR`;
   - synchronous `classify_main_health_red`: MAIN_HEALTH_RED;
   - otherwise `_task_fault_outcome`: `MergeOutcome('blocked', reason, failure_category, failure_cause_hint, disposition, failure_diagnostic, skew_evidence, block_category=MergeBlockCategory.POST_MERGE_VERIFY)`. This arm also spawns the dry run and the deferred main-health probe (`spawn_main_health_probe`).
2. `InflightVerifyRun._settle_single` returns the outcome.
3. It is delivered in order at `worker.py::SpeculativeMergeWorker._finalize_inflight`'s FAIL arm through `_resolve_or_drop_abandoned`. That arm emits `merge_attempt(outcome=verify_failed)` only for an INTEGRATION_SKEW or BRANCH_BUG disposition, or when skew evidence exists.
4. The head-failure cascade then re-merges everything downstream.

`MergeBlockCategory` (`merge_lane/outcome_facts.py`) still has three values: POST_MERGE_VERIFY, MERGE_NEEDS_REBASE and MERGE_ERROR.

**Workflow side:** `workflow.py::TaskWorkflow._submit_to_merge_queue` is unchanged from lane.md C4.
- Status arms: wip_halted, done_wip_recovery, wip_recovery_no_advance, unmerged_state, stash_failed, done, already_merged, superseded, conflict.
- For `blocked` it branches on `reason.startswith`: WORKTREE_MISSING, DROPPED_PLAN_TARGETS, POST_MERGE_EQUIVALENCE_FAILED, POST_MERGE_PYRIGHT_BROKEN, TRANSIENT_INFRA, MAIN_HEALTH_RED.
- Then on disposition: INTEGRATION_SKEW.
- Then the generic arm: `_write_merge_failure_review(result.block_category.value, ...)`, the `merge_blocked` event, and `_mark_blocked(..., category=...)`, which files an L0 and starts the steward.
- The thrash ladder is `_check_merge_outcome_thrash` / `_merge_outcome_signature`.

**A chain red** is the same for an owner that is the head (alone in the frontier) and for a speculative owner: the owner is requeued and marked, the links are marked and stay in place, and nobody gets an outcome (Q2). The chain-arm exception arm marks only the owner. A non-chain verify exception resolves `blocked` with the reason `Verification error: ...`.

**`MergeOutcome.kind` does not exist** (task 9144 is pending).
- `merge_lane/types.py::OutcomeKind` exists only as the vocabulary of the `merge_attempt` `data['outcome']` field.
- Its members include `verify_failed`, `main_health_red` and `cas_exhausted`. It has no member for a conviction.

## 4. Recorded rows

**Store:** `/home/leo/src/dark-factory/data/orchestrator/runs.db`, table `events(id INTEGER PK AUTOINCREMENT, timestamp TEXT, run_id TEXT, task_id TEXT, event_type TEXT, phase TEXT, role TEXT, data TEXT DEFAULT '{}', cost_usd REAL, duration_ms INTEGER)`.
- Indexes: `run_id`, `(run_id, task_id)`, `event_type`, `timestamp`.
- There is no project_id column on events. The other tables are invocations, account_events, runs and task_results.

- **`merge_queued`**, emitted by `merge_lane/intake.py::_emit_merge_queued` from `enqueue_merge_request`:
  - always: `branch`, plus `merge_pool_facts.py::queued_request_fields` = `request_id, lane, tier, pinned, head_sha, first_enqueued_at`;
  - optional: `reason, queue_depth, position, source`;
  - tier and pin come from `task_priority.py::read_task_priority` through the injected `Scheduler.task_priority`.
- **`merge_dequeued`**, emitted in `worker.py` in the merger loop:
  - `**self._intake.take_pick_record(rid)` (`PickRecords._snapshot`): `request_id, lane, tier, pinned, in_graph, waiting:[{request_id, task_id, lane, tier, pinned, first_enqueued_at, reserved, in_graph}] (cap WAITING_CAP=200), waiting_dropped, halted_lanes, textual_conflict_pairs, footprint_pair_count`;
  - plus `branch` and `queue_depth`. `queue_depth` is `_queue.qsize()`, the undrained arrivals, not the buffers.
- **`merge_verify`**, emitted at `verify_runner.py` `VerifyRunnerPool.dispatch`, the emit after the verify:
  - `runner` (the host), `merge_sha`, `passed` (the verdict), `duration_ms`, `attempt`, `depth`, `speculative`, `chain_items`, `chain_build_ms`;
  - `runner_reason`, `base_relation` (main | stacked | rebased_head | stale);
  - `chain_depth_reason` (always present, null when the policy was not consulted);
  - `chain_request_ids`, `chain_task_ids` (via `merge_pool_facts.py::with_chain_members`, dispatching request first; present when known);
  - `retry_scope`, `retry_subset_sizes`, `plan_hash`, `cache_mode`, `units`.
  - The row's `task_id` is the owner's.
  - There is **no** lane, tier or pin field, **no** re-verify flag and **no** un-chained flag. An un-chained dispatch is visible only as a null `chain_depth_reason`, and other cases produce that same null (not chain-eligible, a train, a shadow verify, etc.).
- **`merge_reverify_gate`** (`merge_lane/reverify_gate_decision.py::emit_reverify_gate_decision`, `ReverifyGateDecision.event_data`) has `reason, reverify, merge_sha, verified_merge_sha, rebased_from, rebased_onto, overlap, overlap_total, blockers`. It is the only "re-verify after main moved" record, joined by task_id and sha.
- **`merge_finalized`** (`merge_lane/intake.py`, nested `_on_finalized`) has `request_id, branch, state, snapshot_tip, merge_sha, superseded_by, generation, reason, landed_via_chain, absorbed_request_ids, queue_to_land_secs`.
- **`merge_attempt`** (`merge_lane/gates.py`) has `outcome` plus optional `attempt, train_id, member_task_ids, disposition, origin_host, probe_host, gate_clearance`, `failing_tests` / `implicated_commits` / `overlap_files` (+`_total`, skew only) and conflict evidence.
- Other merge events: `merge_verify_cancelled {elapsed_ms, fraction, cause, host}`, `merge_blocked`, `merge_heartbeat`, `merge_coalesced`, `merge_verify_progress_abort`, `merge_flake_suppressed`, `merge_serial_lane_breached`.
- **No event records chain reservation, chain red/requeue, or the setting of the mark.** Those are logger lines only.

## 5. Config

- `config.py::MergeDeepConfig` has `chain_cap`, `depth_table`, `chain_build_timeout_secs` and `chain_reservation_timeout_secs`. It is mounted as `OrchestratorConfig.merge_deep` and its defaults are in `defaults.yaml`.
- Sibling sub-models: `SpeculationProbeConfig` (`speculation_probe`), `VerifyCacheConfig` and `GitConfig` (`git`; several lane knobs are individually reloadable, e.g. `git.merge_park_lock_grace_seconds`). Some top-level merge knobs are reloadable (e.g. `merge_disjoint_skip_requires_verified_drift`, `pre_merge_reverify`); `merge_verify_breadth` is restart-only.
- `RELOADABLE_FIELDS` includes `_submodel_leaf_paths('merge_deep', MergeDeepConfig)` and `_submodel_leaf_paths('speculation_probe', ...)`, so every `merge_deep` leaf is green-tier.
- **`orchestrator/merge_pool_config.py` does not exist.** Task 9933 creates `MergePoolConfig(order, promote_after_secs, bounce_cap)`. `config_sections/` does not exist either.

## 6. The size ratchet

`orchestrator/tests/merge_lane_ratchet_baseline.json`, checked by `orchestrator/tests/test_merge_lane_ratchet.py` (generator `scripts/merge_lane_metrics.py`).
- **Cluster:** `merge_lane/**/*.py`, git_ops.py, lane_lifecycle.py, offline_lane.py, warm_lane_pool.py, recover_main.py, and the test files `_merge_queue_harness.py` and `conftest.py`.
  - **Outside the cluster:** suffix_graph.py, merge_pool_facts.py, merge_verify_facts.py, task_priority.py, config.py, workflow.py.
- **Ratcheted:** per file `cognitive`, `function_local_imports`, `reexport_names`; per function `cognitive`; per test file `patch_targets`, `private_reads`. `lines` and `prose_lines` are reported only.
- **Ceilings for keys absent from the baseline:** a new file may have at most 1500 lines, and a new function at most cognitive 15.
- **Current pins:**

| Function | Pin |
|---|---|
| `pop_next_pickable` | 6 |
| `_clique_minimal_visible` | 7 |
| `take_buffered` | 6 |
| `chain_snapshot` | 7 |
| `build_chain` | 18 |
| `_deep_chain_placement` | 4 |
| `_reserve_and_build_chain` | 4 |
| `_build_deep_chain` | 3 |
| `may_chain` | 1 |
| `_note_tip_result` | 2 |
| `seed_chain_policy` | 3 |
| `choose_depth` | 6 |
| `seed_from_rows` | 8 |
| `_settle_chain` | 1 |
| `_settle_single` | 3 |
| `_on_verify_error` | 1 |
| `_retire_item` | 3 |
| `_land_chain_prefix` | 36 |
| `_finalize_inflight` | 59 |
| `_merger_loop` | 79 |
| `_verifier_loop` | 197 |
| `_acquire_next_request` | 9 |
| `VerifyDispatcher.dispatch` | 6 |

- worker.py is 15,574 lines at file cognitive 1469. intake_buffers.py is 849 lines at cognitive 88.
- **Authorised raise:** `python scripts/merge_lane_metrics.py --write-baseline orchestrator/tests/merge_lane_ratchet_baseline.json --authorize-raise <task-id> --reason "..."`. It writes one new file under `orchestrator/tests/merge_lane_ratchet_raises/` (243 files today). The baseline's `_README` holds the full rules.
- **Enforcement:** `scripts/check_staged_ratchet_raise.py` in pre-commit, and `hooks/reference-transaction` → `scripts/check_ratchet_advance.py` on every move of main. Lowering a measure needs a plain `--write-baseline` in the same commit.

## 7. Planned moves

Only 8933 has a finalized plan.json (in progress, and branch `task/8933` exists). The others have no plan.

- **5041 λ** (pending, no unmet deps, so ready now). Moves into a new `merge_lane/speculation.py` (`ChainPlanner`, `Speculator`):
  - chain build, depth selection and placement, the prefix-landing decision, permits and leases, `_remerge`/`_void_and_remerge`;
  - it absorbs `speculation_controller.py`, and the `_merger_loop` body moves too;
  - "moves them UNCHANGED". Its text still names `select_chain_depth`, which is now deleted. The move would carry `_deep_chain_placement`, `build_chain`, `_chain_policy`/`_note_tip_result` and probably `_next_dispatch_unchained`.
  - Files: worker.py, speculation.py, speculation_controller.py, the baseline.
- **5042 μ** (depends on 5041 and 9265). Moves `_finalize_inflight`, CAS handling, `_land_chain_prefix`, outbox/journal and `_drift_base` into `merge_lane/landing.py` (`Lander`, sole writer of `_inflight`) and decomposes `GitOps.advance_main` in place. It restates "main advances strictly in submission order". Files: worker.py, landing.py, git_ops.py, the baseline.
- **5306 ν+ξ** (depends on 5042). Moves worktree lifecycle into `merge_lane/worktrees.py`, then `snapshot`, heartbeat and escalation emit helpers into `merge_lane/telemetry.py`. 9935 makes `snapshot()` read `pool_order()`. Files: worker.py, worktrees.py, telemetry.py, the baseline.
- **5045 ο** (depends on 5306). Leaves a residual worker.py: composition root, thin `_verifier_loop` and `_merger_loop`, and `stop`. The 45 init-only attributes become frozen config objects in types.py. Files: worker.py, types.py, the baseline.
- **9933 α1** (pending, no unmet deps).
  - Extends `TaskPriority` (`base_tier`, `pin_order`, `age_anchor`) and adds `PINNED`, `GATE_RANK` and the pure `order_key(p, *, now, waited_secs, promote_after_secs, fallback_anchor)`.
  - The lookup answers before the first tick and while paused. Adds an unknown-tier L1 streak.
  - New `merge_pool_config.py::MergePoolConfig`, reloadable. `merge_queued` gains `base_tier, age_secs, age_from`. It changes no ordering.
  - Files: task_priority.py, scheduler.py, merge_pool_facts.py, merge_pool_config.py, config.py, test_task_priority.py.
- **9935 γ** (depends on 9933).
  - New `merge_lane/pool_order.py` + `MergeIntake.pool_order()`. Every order reader switches to it: the pick, `chain_snapshot`, `unfrozen_suffix`, `snapshot()`, graph node order, `take_buffered`, `get_merge_queue`/`merge_status` positions, heartbeat head-of-line, pick records.
  - Under `order: priority`: high lane, then **"task 8528's suspect list"** (which does not exist), then by `order_key`, with no clique rule.
  - The footprint becomes `GitOps.get_branch_changed_files`, with a git error counted as a probe error. `bounce_cap` replaces `MERGE_BOUNCE_CAP=3`. `merge_dequeued` gains rank and age.
  - It also amends the frozen-order line in three plans.
  - Files: intake_buffers.py, pool_order.py, suffix_graph.py, worker.py, merge_pool_facts.py, the test, the baseline, three plans.
- **10721** (pending/high, no deps). A gate re-verify of a rebased head must verify its own tree un-chained. Today `_dispatch_rebased_head` → `rebased_head_item` → ordinary dispatch → `may_chain` lets it chain. This adds a third "never chain this dispatch" case beside guard 2b. Files: worker.py, rebased_head.py, test_merge_queue_deep_dispatch.py.
- **7919** (pending, no unmet deps). Adds a pre-verify static screen in `merge_lane/screen.py`, called from `classify_and_merge` (merger path) and `_do_train_merge`.
  - A finding gives `blocked` with `failure_category='merge_screen'` and a new `OutcomeKind.screen_failed`, with no verify. This is a per-request "convicted without a verify" path that feeds the thrash guard.
  - Files: merge_screen.yaml, screen.py, worker.py, types.py, git_ops.py, workflow.py, config.py, lib_lane_state.sh, outcome_colors.py, the yaml, shared/merge_outcomes.py.
- **7921** (depends on 7919 and human gate 10577; Leo may cancel it). While main is confirmed red, adds `main_red_verdict.py::VerdictBook` and HOLDs requests: a mark over LANE_BUFFERED that is skipped by the pick and excluded from `chain_snapshot`. Per its 9935 note, the skip belongs in `pool_order()`'s filter.
  - Its ratchet pins (`_dispatch_item` 43, `_pop_next_pickable` 13, `chain_snapshot` 3) are stale, and those symbols moved or no longer exist.
  - Files: main_red_verdict.py, screen.py, worker.py, types.py, harness.py, config.py, test_merge_queue_build_chain.py, the yaml.
- **7922** (depends on 7921 and 7259). Screens a built chain tip and truncates before the first failing prefix (`ChainTruncationReason` gains `screen_failed`). It decides chain membership after build. Its text cites halving and a pin of 8, both stale. Files: worker.py, types.py.
- **8933** (in progress, plan finalized).
  - Adds `DeadBaseSite.REQUEUED` and `ABANDONED`, and records the item's own speculative merge commit dead at every requeue or drop exit. That includes the red-tip `_requeue_from_verify`, `_drop_from_verify`, the cascade's manual requeue, the pre-dispatch halt requeue and abandoned exits.
  - Any batch requeue design relies on this.
  - Files: dead_base_ledger.py, worker.py, test_merge_queue_undead_speculative_base.py, test_merge_lane_dead_base_ledger.py, the baseline.
- **9144** (10 unmet deps). Adds `MergeOutcome.kind: OutcomeKind | None` (kw_only) plus 8 terminal members (worktree_missing, transient_infra, needs_rebase, trivial_pass_main_red, train_verify_failed, superseded, shutdown, error) and `GroupMergeRequest.train_kind`.
  - The workflow routes on kind, with reason strings frozen. A "convicted from batch evidence" kind would land here.
  - Files: types.py, gates.py, worker.py, intake_buffers.py, `__init__.py`, workflow.py, harness.py, suffix_graph.py, merge_lane_throughput.py, post_merge_verify.py, verify_dispatch.py, shared/merge_outcomes.py, outcome_colors.py, tests, the baseline.
- **8932** (design milestone, depends on 8532). Covers an idle-host prefix hedge and a dense p̂→depth table. Amended 2026-11-18: no bisect exists, so it is read against the mark's solo verifies. File: plans/merge-lane-policy-prd/idle-hedge-and-depth-table.md.
- **5411** (pending, depends on 5296; pinned 11-19, no shadow). A narrow `docs_gate`/`comments_gate` merge resolution in `verify.py` (merge role), with docs/comments items excluded from trains. Amendment 10-09 says "docs/comments items keep FIFO — no order bypass", which is a constraint on any priority or batch chooser. Files: verify.py, merge_queue.py, config.py, the yaml, plans/merge-verdict-integrity-prd.md.

## 8. Where the design premise ("any chosen subset, in pool order; a request can be forced to verify alone") meets today's lane

1. **Member #1 is fixed and pre-merged.** A batch exists only as an extension of a dispatching item that was already picked by `pop_next_pickable`, merged by the merger onto its speculative base, and holds a SpecPermit and an `_inflight` slot. The other members are buffered requests that hold no permits. No path verifies a batch whose first member was not dispatched.
2. **The base is the owner's merge commit,** which is itself stacked on the frontier tip or on main. Chaining is allowed only when the owner is speculative or alone in the frontier (`may_chain`). A batch cannot pick its own base.
3. **The batch is an ordered prefix that git can silently cut short.** Links are merged in snapshot order and `build_chain` truncates at the first refusal or conflict, so "chosen subset" means "subset up to the first link that does not merge cleanly".
4. **Snapshot order is fixed FIFO, with gaps.** Members are high-then-normal FIFO with exclusions (reserved, coalescing, done, marked), so they need not be adjacent in the buffers. The owner may have been picked out of FIFO order by clique bypass.
5. **Landing assumes that order.** `_land_chain_prefix` CASes links in link order after the owner lands, and aborts silently at the first failed advance.
6. **Depth and membership are coupled.** `choose_depth` returns only a count from one lane-wide p̂, and membership is `snapshot[:d-1]`. Nothing takes a per-request probability, so a chooser has to replace both. `chain_cap` does not cap depth (the table goes to 12).
7. **"Verify alone" is not "verify against main".** The mark means un-chained, still stacked on in-flight predecessors. It is:
   - keyed by task_id and in memory only, so a restart loses it;
   - cleared only at retirement;
   - not a priority boost: the marked request is picked in ordinary order;
   - invisible to the pass estimate.
8. **Lane slots shift after a red chain.** The owner goes to its lane's tail (aging key kept) and the links keep their slots. The owner's own speculative commit is not recorded dead until 8933 lands. The finalize of the vacated owner fires the head-failure cascade and re-merges everything behind it.
9. **No outcome exists for "convicted from batch evidence".** A red tip produces no outcome by design (to keep the thrash ladder clean). A conviction needs a new outcome and route (9144's kind, or a screen-like path as in 7919), and if delivered as `blocked` with POST_MERGE_VERIFY it feeds the L0 steward and the thrash signature. `MergeOutcome` also carries no structured failing-test field.
10. **Reservations and the look-ahead hold sit between the build and the round's end.** Reserved links are invisible to the pick and expire after 7200 s. The merger's next look-ahead waits for the chain decision and stacks on the chain tip.
11. **Trains and branches stacked on an unlanded base cannot be links** (`_link_refusal`). Halted lanes are excluded. A chain may mix high and normal lanes.
12. **The order is frozen in plans.** `plans/merge-lane-quality-prd.md` freezes "main advances strictly in submission order" and the event `data` keys, and 9935 is the task that plans to amend that line. 5411's amendment forbids an order bypass for docs items.
13. **The calibration data is partial.** `merge_verify` rows carry members but no lane, tier, mark or re-verify flag. Re-verifies have to be joined through `merge_reverify_gate`. Red chains, marks and reservations leave no event, only log lines.

## Surprises

- **8528 landed without the suspect-list bisect** (lane.md B2–B6 and B4's `suspects` field are wrong). It landed as the un-chained mark (10617/10723); `ChainPolicyState` has only `recent`. The halving functions are gone.
- **9935 still says "then the requests of task 8528's suspect list in its order".** That premise is dead. 8932 was amended for this; 9935 was not.
- **The depth table differs from the PRD.** The default is a dense 10-row table (0.992→12 … 0.75→3), not `[[0.93,8],[0.85,4],[0.75,2]]`. The DF yaml overrides only `chain_cap` and the build timeout. Task 8932's text quotes the old table.
- **The pass estimate's window feeds only on rounds the policy sized.** Marked solo verdicts are excluded. Seeding reads 300 rows and drops null-reason rows.
- **8527 landed:** a head alone in the frontier chains. Live row 645842 shows `speculative:false, base_relation:"main", chain_items:3`.
- **10721 shows gate re-verifies of passed heads being chained,** losing 4 of 5 landings in the sample measured 11-17/18.
- **9908 landed** (lane.md says it is branch-only). `merge_pool_facts.py`, `task_priority.py`, the pick records, `chain_request_ids`/`chain_task_ids` and the tier/pin on `merge_queued` are all on main. `TaskPriorityLookup` is wired from `Scheduler.task_priority`.
- **5040 landed:**
  - `_run_inflight_verify`, `_dispatch_item` and `_run_post_merge_verify` no longer exist in worker.py. They became `VerifyDispatcher.dispatch` / `run_inflight_verify` / `post_merge_verify`, `merge_lane/inflight_verify.py::InflightVerifyRun`, and `merge_lane/verify_failure_outcome.py::red_verify_outcome`.
  - worker.py shrank from 21,765 to 15,574 lines.
  - The pin numbers that 7921 and 7922 cite are stale: `_deep_chain_placement` is now 4 (was 8), `build_chain` 18 (was 17), and `_pop_next_pickable` does not exist.
- **8526 landed:** the host is released at the verdict (`_verify_releasing_at_verdict`, VERIFIED_PENDING_LANDING), and K = hosts + 1.
- **Unchanged since lane.md:** `PersistedMergeRequest` still lacks `merge_first_enqueued_at`. `MERGE_BOUNCE_CAP` is still 3. The suffix footprint is still the two-dot `get_changed_files`.
- **5041 is unblocked** (it has no unmet deps) and will move the whole chain planner into `merge_lane/speculation.py`. A chooser written into worker.py now would be moved soon after.
- **There is no `config_sections/`** (9709's plan is not on main), and there is no `merge_pool_config.py` yet.
- **Not verified:** whether the look-ahead pick and the prefetched pick each emit exactly one `merge_dequeued`. I saw only the one emit site in the merger loop.
