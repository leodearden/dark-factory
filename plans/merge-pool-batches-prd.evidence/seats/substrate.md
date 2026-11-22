Every symbol cited below was confirmed by grep at main `09bd5bd7b1` (which descends from `bdf7057360`). Code paths are under `/home/leo/src/dark-factory/orchestrator/src/orchestrator/` unless a path says otherwise.

## 1. The tree at dispatch

**What dispatch has in hand**
- `merge_lane/verify_dispatch.py::VerifyDispatcher.dispatch(self, item: SpeculativeItem) -> InflightEntry | NoHost` runs `_dispatch_on_free_host`, then `_launch`, then `_place(self, item: RealMergeItem, lease) -> _Placement`, then `_start_verify`.
- `_place` already reads `self._frontier.frontier_base_commits()`, `frontier_depth()` and `has_inflight_verify()`. These come from the `merge_lane/dispatch_ports.py::VerifyFrontier` port, which is lambdas over the worker.
- The dispatcher has no port that returns the frontier entries themselves.

**The frontier structure**
- `SpeculativeMergeWorker._inflight: collections.deque[InflightEntry]`.
- Readers:
  - `_verify_frontier_entries()`: the finalize head plus `_inflight` entries with `in_verify_frontier`, vacated ones included.
  - `_frozen_inflight_entries()`: the same minus vacated entries.
  - `frozen_prefix_requests()`: the frozen entries' requests.
  - `_frontier_base_commits() -> (stacked_on, rebased_heads)`.
  - `_frozen_base_chain()`: (item, predecessor_commit) pairs.
- `merge_lane/remerge_base.py::entry_successor_base` / `entry_bases` / `frontier_tip`.

**`merge_lane/types.py::InflightEntry` fields**
- `item: SpeculativeItem`, `lease`, `verify_task`, `merge_wt`, `was_speculative`, `passthrough_outcome`, `verify_result`.
- `status: InflightStatus | None`: ABANDONED_/REQUEUED_PREDISPATCH, DROPPED, REQUEUED, VERIFIED_PENDING_LANDING.
- `chain: ChainResult | None`, `chain_adopted`, `verify_base: VerifyBaseFacts | None`, `gate: PendingGate | None`, `verify_base_divergence`, `verdict_host`.
- Properties: `vacated` (REQUEUED/DROPPED), `in_verify_frontier`, `pending_task`.

**`RealMergeItem` fields:** `request`, `merge_result: MergeResult` (`merge_commit`, `pre_merge_sha`, ...), `base_sha`, `speculative`, `merged_branch_tip: str | None`, `remerge_base: RemergeBase | None`, `gate_rebase_of: str | None`, `permit`, `cap_permit`.

**`ChainResult` fields:** `links: list[tuple[task_id, merge_commit]]`, `link_request_ids: tuple[str, ...]` (parallel to `links`), `tip`, `truncated_at`, `truncated_reason`, `lane`, `build_ms`.
- `ChainReservations` records also hold `links ((request_id, task_id), ...)` and `tip`.

**Branch tips per member**
- The dispatching item's tip is `RealMergeItem.merged_branch_tip`, from `MergedOk.branch_tip`, rev-parsed in `classify_and_merge`.
- Chain links: `worker.py::build_chain` calls `git_ops.merge_branch_into_worktree(lane, req.branch.bare_id)` and keeps only the merge commit. **No tip is captured for a link.** It can be derived as `<link merge_commit>^2`, because both paths use `--no-ff`.

**Other sources**
- The `speculative_merge` event carries `base_sha` and `depth` only: no request id, no tip.
- Vacated entries are excluded from the frozen prefix, so new successors never stack on them. Successors already stacked on one die in the head-failure cascade.
- The redispatch park (`MergeIntake._redispatch`) holds built items awaiting a host. They are not in `_inflight`.

PRESENT: `_frozen_inflight_entries`, `_verify_frontier_entries`, `frozen_prefix_requests`, `entry_successor_base`, `entry_bases`, `_frozen_base_chain`, `ChainResult.link_request_ids`, `RealMergeItem.merged_branch_tip`, `merge_commit^2` as a derivation, `classify_base_relation` (MAIN / STACKED / REBASED_HEAD / STALE).
MISSING:
- A dispatch-time function that returns the tested tree's members (request_id plus tip) by walking base ancestry.
- A port to hand it to `VerifyDispatcher`.
- Captured tips for chain links.
- The tree on any event.

## 2. The red-chain path

**What runs when a chain tip comes back red**
1. `InflightVerifyRun._take_verdict` sets `_chain_adopting = chain is not None and out is None`.
2. `run()`'s `finally` runs `_release_chain_resources`, which calls `chain_planning.end_chain_reservation(owner)` so the links return to the buffers, then releases the lane and drops the item's ephemeral worktree.
3. `_settle_chain(out)`:
   - calls `note_tip_result(...)`, which feeds the pass estimate;
   - calls `mark_next_dispatch_unchained([req.task_id, *link task_ids])`;
   - calls `lane.custody.requeue_from_verify(item, slot)`;
   - returns `InflightVerifyResult(status=REQUEUED)`.
4. There is no `custody` module. `requeue_from_verify` is a `DispatchCustody` port field wired to `worker.py::SpeculativeMergeWorker._requeue_from_verify`. That function calls `_vacate_inflight_entry(slot, REQUEUED)` (releases the lease, stamps the status) and then `_requeue_request(req)`, which puts the request back on `_queue` with its future still PENDING.
5. All of the above happens **on the verify task, at verify return**, before the verdicts of the entries ahead are known.

**When the vacated entry reaches the head**
- `_verifier_loop`'s FINALIZE-HEAD branch calls `_finalize_inflight(head)`. Its DROPPED/REQUEUED arm sets `_n_failed_val = True` and returns False; `entry.vacated` skips `_note_requeue`.
- Then the HEAD-FAILURE CASCADE (`if not _head_advanced and self._inflight:`) runs:
  - records the head's commit dead (`DeadBaseSite.HEAD_FAILED`);
  - tears down every downstream entry via `_tear_down_inflight_entry(..., CASCADE)`;
  - re-merges each one onto the redispatch park; an entry already REQUEUED just `continue`s.

**Where the lane knows each fact**
- (a) Verdicts of the entries ahead:
  - An entry ahead failing is known at the cascade (`worker.py::_verifier_loop`), where downstream entries, including vacated chain-red entries, are iterated.
  - The cascade cannot tell "ahead was red" from "ahead was REQUEUED or DROPPED". It only peeks `_head_was_requeued`.
- (b) Entries ahead landed: known when this entry itself reaches FINALIZE-HEAD. Every entry ahead has then advanced main, otherwise the cascade would already have cleared `_inflight`.
- "Landed as built" is not guaranteed. A head whose CAS met a moved main lands a gate-rebased tree (`entry.gate`, `rebased_head.py`, `RealMergeItem.gate_rebase_of`).

**Solo red**
- The outcome stays in `entry.verify_task.result().outcome` and is adopted only in the `_finalize_inflight` FAIL arm.
- If an entry ahead fails, the cascade tears the solo entry down and the red is discarded silently.
- If its base was recorded dead (stragglers, re-merged-away bases), `_chain_dead_link(item, main_sha)` at adoption routes it to `_void_and_remerge(entry, item, dead_link, vr, main_sha)`, which emits `verdict_voided` and re-parks it.

**Smallest change.** Keep the provisional mark at `_settle_chain`, add a field carrying the observation (members plus tips) on `InflightVerifyResult` or `InflightEntry`, and decide at two existing points:
- Confirm at the vacated entry's `_finalize_inflight` REQUEUED arm, which means everything ahead landed. A `gate_rebase_of` check covers "as built".
- Revise at the cascade's `_entry_status == InflightStatus.REQUEUED` skip for vacated entries, which means something ahead did not land. The head's real `vr.outcome` would have to be read there to tell a red from a requeue.

Deferring the requeue itself would touch the `_requeue_request` single-chokepoint guard (`TestRequeueRecipeHasASingleChokepoint`).

PRESENT: `_settle_chain`, `_on_verify_error` (chain arm), `ChainPlanning.mark_next_dispatch_unchained`, `_requeue_from_verify`, `_finalize_inflight`'s REQUEUED arm, the cascade in `_verifier_loop`, `_chain_dead_link`, `_void_and_remerge`.
MISSING: an observation carried on the result or entry; a cascade that distinguishes the head's red from a requeue; any revision of a mark once applied.

## 3. Content under a request id

**The tip is re-read on every attempt.**
- `worker.py::classify_and_merge` runs `git rev-parse HEAD` in `req.worktree` each time and uses `effective_tip = req.snapshot_tip or branch_head`.
- The merge goes by ref name (`merge_to_main(req.worktree, req.branch.full_name, ...)`), so whatever the ref points at then is what merges.
- `worker.py::_unstack_before_merge` calls `git_ops.py::GitOps.unstack_from_unlanded_base`, which rebases the branch in place before an attempt. The lane itself moves the tip.

**The tip is not pinned for workflow submits.**
- `MergeRequest.snapshot_tip` / `PersistedMergeRequest.snapshot_tip` are optional.
- Only three paths set it: the MCP `merge_request`, the stranded reaper and `_maybe_auto_chain_generation`. The workflow submit never sets it.

**Why `merge_queued.head_sha` is null.** `merge_pool_facts.py::queued_request_fields` writes `'head_sha': req.snapshot_tip`, which `intake.py::_emit_merge_queued` spreads into the row. It is null in 1252 of 1300 rows since 11-01.

**How mr-8ddfc972 got two tips.** The request id survives restarts: `queue_store.py::reconstruct_merge_request` keeps `request_id`. The runs.db rows show it re-queued under the same id across an abandoned (restart) finalize, while the branch moved underneath.

**Patch-id.** `git_ops.py` has none. The only existing helper is the private `dry_run_unblock.py::_capture_branch_patch_id(worktree, *, main_branch, task_id) -> str | None` (`git diff --binary main...HEAD | git patch-id --stable`).

PRESENT: `MergedOk.branch_tip`, `RealMergeItem.merged_branch_tip`, `merge_commit^2`, the `_capture_branch_patch_id` pattern, `request_id` stable across a restart.
MISSING: a tip or content id pinned per attempt; the tip on events; a public patch-id helper. Request id alone does not identify the tested content.

## 4. Main is red

**`merge_lane/main_health_probe.py`**
- `classify_main_health_red(git_ops, req, verify, event_store=None, *, escalation_queue=None) -> MergeOutcome | None` is the synchronous path.
- `spawn_main_health_probe(handles, git_ops, req, verify, *, escalation_queue, event_store, origin_is_local) -> bool` detaches `_run_deferred_main_health_probe`, one per task id.
- Both skip on timeout and on infra categories.
- On a confirmed red where `_main_is_still_at(probe_sha)` holds, it emits `merge_attempt outcome=main_health_red` (task_id and hosts only: no sha, no request id, no failing ids). It then calls `worker.py::SpeculativeMergeWorker._auto_heal_main_health_deferred`, which halts lane `normal` for `compile_error` and otherwise escalates.

**`verify_failure_outcome.py::red_verify_outcome`** probes synchronously only when `main_health_probe_handles is None`. The lane always passes handles, so a lane red is first a provisional task fault, and main-red is decided later off the verify path. This applies to chain reds too.

**In-memory state (all lost on restart)**
- `verify.py` `_PROBE_CACHE` (300 s TTL) and `_BASELINE_FAILING_IDS_CACHE` (16-sha LRU).
- `verify.py::known_failing_ids_on_main(main_sha)`: a cache-only lower bound.
- `halt.py::HaltState` (cleared by `unhalt_lanes_owned_by`).
- Nothing clears these explicitly: an entry goes away when main moves or its TTL runs out.

**Known gaps:** the skip guards never probe; a stale-sha drop; deduplication swallows a second red; cancelled verifies leave no row. `plans/merge-pre-verify-screen-prd.md` §2.3 measured 102 verifies started on a sha already judged red, with "no stored red verdict".

**Task 7921** (pending, behind gate 10577, which may cancel it) plans:
- `merge_lane/main_red_verdict.py` with `MainRedVerdict{main_sha, failing_ids, signature, owner_esc_id, confirmed_at, generation}` and a `VerdictBook`, armed only in `_auto_heal_main_health_deferred`;
- a hold skipped at the pick and excluded from `chain_snapshot`;
- `main_red_verdict` and `merge_held` events.

Persisting the verdict across restart is a non-goal.

PRESENT: `classify_main_health_red`, `spawn_main_health_probe`, `_auto_heal_main_health_deferred`, `known_failing_ids_on_main`, `seed_main_baseline`, `HaltState.is_lane_halted`.
MISSING: a queryable or durable "main red at M with ids F"; a main-red row with sha, ids and request; any link from a deferred verdict back to the observation it should void.

## 5. The failure report

**`verify.py::VerifyResult` fields**
- `passed`, `category`, `cause_hint`, `timed_out`, `summary`.
- `failing_test_ids: list[str] | None`: junit node ids, **as data**, but only for role=merge, breadth=full and a junit that parsed. Expect None for opaque runners such as cargo.
- `failing_test_ids_by_module`, `failing_leg_categories`, `test_leg_failed`.
- `units: tuple[VerifyUnit, ...]`, each with an outcome such as `ran_red`.
- `flake_suppression: FlakeSuppression | None`, `archive_log_paths`, `worktree_log_paths`.

**What reaches the lane**
- `post_merge_verify.py::run_post_merge_verify` collapses the result to a `MergeOutcome` (`failure_category`, `failure_cause_hint`, `failure_diagnostic`, `disposition`, reason text). Failing ids survive only inside the reason text.
- The whole `VerifyResult` is visible only through `on_result`, which `InflightVerifyRun._launch` wires on the warm path only.
- The `merge_verify` row (`verify_runner.py`) carries `chain_request_ids` and `units`, but no ids and no category.

**Flake suppression.** `verify.py::apply_merge_flake_suppression` turns a red into a PASS (`merge_flake_suppressed`).

**Logs.** `PostMergeVerifyRun.archive_root = project_root/'data'/'verify-logs'`, and `verify.py::archive_verify_logs` writes under `<archive_root>/<task_id>/`. Logs are keyed by task id, not by request or sha.

PRESENT: `VerifyResult.failing_test_ids`, `units`, `flake_suppression`, `category`; the `on_result` hook.
MISSING: ids and category as data on the lane outcome or on events; `on_result` off the warm path; a log path per verify.

## 6. Blocking a waiting request

**Precedents**
- `suffix_graph.py::SuffixConflictTracker._escalate(self, lane, req, detail, evidence) -> int`:
  - does `self._lane_buffers()[lane].remove(req)` and `self._retire_item(rid)`;
  - resolves `MergeOutcome('blocked', reason=f'{NEEDS_REBASE_REASON_PREFIX}: … (bounce escalation #{n} of tracker {incarnation})', block_category=MERGE_NEEDS_REBASE)`.
  - Its guard is `if self._is_reserved(rid): continue` in the bounce loop (`is_reserved = ChainReservations.holds`).
  - Its reason never repeats, so it never feeds the thrash ladder (`test_merge_queue_bounce.py::TestBounceEscalationsNeverFeedTheThrashLadder`).
- `worker.py::SpeculativeMergeWorker.cancel_request(request_id) -> MergeCancelVerdict` cancels the future in place. The merger drops it later only because `_request_abandoned` checks `result.cancelled()`.
- `intake_buffers.py::MergeIntake.take_buffered(task_id) -> (lane, index, req) | None` with `restore_buffered`: the verifier-task writer, with no ownership assertion.
- `MergeIntake.remove_buffered(request_ids)` asserts merger ownership.
- The twin-drop in `buffer_owned_request` resolves `already_merged`.

**Journal.** `MergeIntake`'s done-callback removes the journal entry on any `set_result` that is not a shutdown outcome.

**Guards against a just-picked or reserved request**
- `pop_next_pickable` deletes from the buffer and transitions to MERGING in one synchronous beat, so a buffer-membership check in the same beat is enough.
- Reserved links are visible through `ChainReservations.holds(request_id)`.
- A plain `set_result` on a buffered request is **not** enough: the merger only skips cancelled futures and would still merge it. It must be removed from the buffer, as `_escalate` does.

**Workflow routing**
- The workflow's merge-outcome dispatch is a chain of reason-prefix `startswith` arms. `MAIN_HEALTH_RED_REASON_PREFIX` goes to `_auto_heal_main_health` and skips the generic fields.
- The generic blocked arm:
  - sets `_last_merge_failure_category` and `_last_merge_failure_cause_hint`;
  - emits `merge_blocked`;
  - calls `_mark_blocked(reason, category=result.block_category.value)`.
- `MergeOutcome.block_category` defaults to `MERGE_ERROR`; the values are POST_MERGE_VERIFY, MERGE_NEEDS_REBASE and MERGE_ERROR (`merge_lane/outcome_facts.py::MergeBlockCategory`).
- The thrash ladder `_check_merge_outcome_thrash` runs after the steward resolves an L0 and the loop resubmits. `_merge_outcome_signature` keys on (failure_category, normalised cause_hint), falling back to `sha256(reason)` when both are empty.

**Keeping a conviction out of the post-merge-verify count.** Either a new reason-prefix arm (the existing pattern, but a string protocol, against heuristic 12) or a new `MergeBlockCategory` / `EscalationCategory` value with its own arm. The outcome must also not copy the verify `failure_category`, or the thrash fingerprint collides with an ordinary verify failure.

PRESENT: `_escalate` as the template; `ChainReservations.holds`; `take_buffered`; the journal done-callback; `_retire_item`; the generic blocked arm.
MISSING: a conviction entry point; a structured conviction category and its workflow arm; a writer that is legal from the verify task.

## 7. Keeping a request out of a batch, and passing it over at the pick

**Today's hooks**
- `worker.py::SpeculativeMergeWorker.chain_snapshot` excludes: halted lanes, `ChainReservations.holds`, `_coalescing_ids`, `req.result.done()`, and `req.task_id in _next_dispatch_unchained`.
- `_deep_chain_placement` guard 2b is `if item.request.task_id in self._next_dispatch_unchained: return NO_DEEP_PLACEMENT`.
- At the pick, `MergeIntake.pop_next_pickable` skips admission-halted, `is_lane_halted(lane)`, and `_clique_minimal_visible`'s `visible = [req … if not self._is_reserved(req.request_id)]`.
- That `visible` filter is where a new skip predicate goes: add a constructor callable beside `is_reserved`.

**Task 9935** (pending):
- adds `merge_lane/pool_order.py::MergeIntake.pool_order()`;
- under 'priority', orders "the requests of task 8528's suspect list";
- says the pick filter "must leave room for task 7921's held-request skip".

**Task 7921** plans a skip at the same pick spot and a `chain_snapshot` exclusion.

PRESENT: the `chain_snapshot` filter, guard 2b, the `_clique_minimal_visible` filter seam, the `may_chain` / `frontier_depth` guard.
MISSING: a suspect predicate; "one suspect per group at a time" (no grouping concept exists at the pick); a coalesce or train exclusion.

## 8. Events and rebuild

**Emitting**
- `event_store.py::EventStore.emit(event_type, *, task_id, phase, role, data, cost_usd, duration_ms)` is synchronous and never raises.
- There is no request_id column; a request id goes in `data`.
- A new type is just a member on `EventType`. `test_event_store.py::test_all_event_types_valid` enforces value == name.
- `merge_pool_facts.py` and `merge_verify_facts.py` build payload fields only. `merge_verify` itself is emitted by `verify_runner.py`.

**Rebuild precedent**
- `SpeculativeMergeWorker.seed_chain_policy` is called synchronously in `run()`. It uses `fetch_recent_events_by_type(type, *, limit=CHAIN_POLICY_SEED_ROWS=300)` (newest N rows across runs, oldest first) and folds them with `chain_policy.py::seed_from_rows`.
- `fetch_events_by_type_all_runs` is unbounded.
- `MergeFinalizedKey` is the precedent for a partial index on a JSON key.
- Rows are never pruned.

**runs.db is per project:** `shared/project_layout.py::runs_db` gives `<root>/data/orchestrator/runs.db`, with a new `run_id` each process.

**Frozen keys:** `plans/merge-lane-quality-prd.md` freezes existing event types and keys (changes must be additive), not new types. The dashboard and `scripts/merge_lane_throughput.py` hardcode event names.

PRESENT: `EventStore.emit`, `fetch_recent_events_by_type`, the `seed_chain_policy` / `seed_from_rows` pattern, `merge_finalized` / `merge_blocked` rows, `merge_verify.chain_request_ids` (own members only).
MISSING: an observation event carrying the whole tested tree with tips and verdict; an explicit choice of rebuild bound; a request-id index.

## 9. Restart inventory

**What survives.** The journal `queue_store.py::MergeQueueStore` (`merge_queue.json`) is restored through `harness.py::Harness._recover_pending_merges`, then `recover_pending_merges`, then `reconstruct_merge_request`. That restores request_id, task_id, branch, snapshot_tip, lane and enqueued_at, in journal order. Trains are not journaled.

Labels: (a) restored from a durable store, (b) rebuilt from event rows, (c) lost. **[B]** marks a loss that changes behaviour after a restart.

**`SpeculativeMergeWorker.__init__`**
- `_next_dispatch_unchained`: (c) **[B]**, marked requests chain again.
- `_inflight`, `_verifier_queue`, `_live_items`, `_lifecycle`: (c); requests rebuilt fresh from the journal.
- `_dead_base_commits` (`DeadBaseLedger`): (c), harmless.
- `_n_failed`: (c), harmless.
- `_cas_retries`, `_gate_retries`: (c) **[B]**.
- `_post_merge_verify_timeouts`, `_enospc_retries`, `_narrowed_retries`: (c) **[B]**.
- `_generation_chain_counts`: (c) **[B]**.
- `_contended_lease_*`: (c) **[B]**, minor.
- `_coalesce_derailed_task_ids`: (a), a `guard_state.py::PersistentSet` file with TTL.
- `_landed_outbox`: (a), `landed_outbox.json`.
- Drift and shadow cadence: (a), their JSON state files.
- `_chain_policy`: (b), `seed_chain_policy`.
- `_recent_verify_outcomes`: (c) **[B]**, deliberate.
- `_probe_round_counter`: (c) **[B]**, minor.
- Speculation and merge-ahead permits, and `_speculation_controller`: (c), harmless.
- `_admission_halt_reason`: (c), by design.
- `auto_heal_registry` (`MainHealthAutoHealRegistry`): (c) **[B]**.
- `main_fix_filer`: (a), from the task store.
- `_chain_reservations`, `_lookahead_hold`: (c), harmless.
- `_verify_dispatch` counters (cold-verify valve, dead-verify aborts): (c) **[B]**.
- Debounce sets: (c), harmless.

**`MergeIntake`**
- `_lane_buffers`: (a) via the journal, but arrival order becomes journal order.
- `_redispatch` park: (c) **[B]**, front priority is lost.
- `_suffix_tracker.bounce_registry`: (c) **[B]**, the bounce cap resets.
- Graph and signature: (c), recomputed.
- `_pick_records`: (c), observability only.

**`HaltState`**
- The WIP halt: (a), via `Harness._rehydrate_merge_halt` from open escalations.
- The operator halt: (c), by design.
- The main-health `normal` lane halt: (c) **[B]**, nothing re-asserts it.

**`HostQuarantine`**
- `_runner_quarantine`, `_runner_unavailable`, `_skew_benches`: (c) **[B]**.

PRESENT: the journal, the `PersistentSet` precedent, the `seed_chain_policy` precedent, the `_rehydrate_merge_halt` pattern.
MISSING: any restore of the un-chained mark or of suspect state; the park's priority; the main-health lane halt.

## 10. Where new code goes

**The size ratchet**
- Baseline: `orchestrator/tests/merge_lane_ratchet_baseline.json`. Gate: `test_merge_lane_ratchet.py`. Logic: `scripts/merge_lane_ratchet.py`.
- New-code ceilings: `scripts/merge_lane_cluster.py::FILE_LINE_CEILING = 1500` and `NEW_FUNCTION_COGNITIVE_CEILING = 15`.
- Totals for the whole cluster (cognitive complexity, private reads, patch targets) are ratcheted. So **any** net-new lane code needs `scripts/merge_lane_metrics.py --write-baseline --authorize-raise <task> --reason`.
- Current scores: `_deep_chain_placement` 4, `_reserve_and_build_chain` 4, `chain_snapshot` 7, `_finalize_inflight` 59, `_settle_chain` 1, `pop_next_pickable` 6.

**Task 5041**
- Pending and dispatchable. There is no branch, no worktree and no `plan.json`.
- It moves `_deep_chain_placement`, `_land_chain_prefix`, `build_chain`, `select_chain_depth`, `_remerge` and `_void_and_remerge` into `merge_lane/speculation.py` (`ChainPlanner`, `Speculator`), and absorbs `speculation_controller.py`.
- It does not name `chain_snapshot` or `_next_dispatch_unchained`; the mark would likely travel with `ChainPlanner`.
- `_settle_chain` stays put; its `ChainPlanning` port gets re-pointed.
- Gate 10573 re-points open tasks that name moved symbols.
- Task 6491's `orchestrator/tests/_merge_queue_scene.py` does not exist yet (its worktree is empty).

**Lane config**
- Lives in `config.py::MergeDeepConfig`, registered in `RELOADABLE_FIELDS`, and is read per request as `item.request.config.merge_deep`.
- `merge_pool_config.py` is **ABSENT**. Task 9933 is pending, pinned, with no branch.

PRESENT: the `MergeDeepConfig` pattern, `--authorize-raise`, the `ChainPlanning` port.
MISSING: `MergePoolConfig` (9933), `speculation.py` (5041), `_merge_queue_scene.py` (6491).

## 11. Tests

**Entry points**
- `orchestrator/tests/_merge_queue_harness.py` drives a single item only (`drive_verify_and_advance`, `finalize_through_gate`).
- Live lane scenes:
  - `orchestrator/tests/_merge_lane_fakes.py::make_lane`, `running_lane`, `FakeVerifier`, `FakeRemoteRunner(name, release=, entered=)`, `passes`, `fails`;
  - `orchestrator/tests/test_merge_queue_deep_reservation.py::_scene`, `_running`, `_until`.

**Scene A ("a chain stacked on an unverified entry goes red, then the entry ahead goes red or green") is scriptable today.** `test_merge_queue_deep_unchained_mark.py::TestAVoidedVerifyKeepsTheMark::test_a_voided_red_is_followed_by_an_unchained_verify_until_the_block` already runs the red-ahead variant with tree-keyed verdicts (`_TreeKeyedRemote` hold and release). The green-ahead variant only needs a different red set. It uses zero private reads.

**Scene B ("restart mid-isolation") has no helper.** Precedents:
- `test_merge_queue_restart_hook.py`: journal, then `recover_pending_merges` into a fresh worker.
- `test_merge_queue_deep_dispatch.py::TestChainPolicySeeding`: the same events db with a new run_id, then `seed_chain_policy`. It assigns the private `_event_store`.

PRESENT: the fakes and the tree-keyed pattern, `recover_pending_merges`, `seed_chain_policy` called from `run()`.
MISSING: a restart-mid-scene helper; a shared home for the tree-keyed fakes (private to one test file today); `_merge_queue_scene.py`.

## Traps

1. **"The members of every in-flight entry ahead" is not the tested tree.** The tree is the ancestry of `item.base_sha`. A recovery re-merge that fell to main (`fell_to_main`, `RemergeBase`; see the `_frozen_base_chain` docstring) sits in the frozen prefix without containing the entries ahead of it. Chained entries ahead also contribute their links (`entry_successor_base`). Walk base links; do not use frontier position.
2. **"Landed as built" is false for gate-rebased heads** (`gate_rebase_of`, `PendingGate`). A landing can remove a request whose tested content differs from what landed.
3. **A chain red is acted on at verify return.** The requeue and the mark both happen then, on the verify task. A solo red waits for in-order finalize and is discarded by the cascade if an entry ahead fails. The cascade cannot tell a red ahead from a requeue or drop ahead, so "a block removes observations" does not cover a requeued or abandoned entry ahead.
4. **Not every red is a test red, and not every pass is green.** `_settle_chain` treats a low-disk `verify_skipped` outcome as red. A flake-suppressed red arrives as a pass.
5. **"Verified alone" today means un-chained, not alone.** The re-dispatch still stacks speculatively on every unlanded entry ahead. Being alone on main would need an empty frontier, as `may_chain` requires for a head.
6. **Batching also happens outside chains, and ignores the mark.** Coalesce trains and declared trains (`merge_train_former_enabled: true` in reify) do not check `_next_dispatch_unchained`.
7. **Identity does not line up.** The mark is keyed by task_id, and `_retire_item` clears it on any exit. Request ids survive restarts while tips move, including through the lane's own unstack rebase. Workflow submits pin no tip.
8. **Conviction is more than resolving a future.** `set_result` on a buffered request does not stop the merger. The request must also be removed from the buffer. Verdicts arrive on the verify task, where `remove_buffered`'s ownership assertion fires.
9. **"Recorded when the verify returns" plus "ignore a red when main was red" conflicts with timing.** The main-red verdict arrives later from a detached probe, keyed by task_id, and only if main has not moved. The ledger must revise an observation it already recorded.
10. **Rebuilding from existing rows is impossible.** `merge_verify.chain_request_ids` excludes the entries ahead and has no tips. A new event is needed at settle time.
11. **Seams with other tasks.** Task 9935 orders "task 8528's suspect list", which does not exist; 8528 delivered only the task-id mark, so this ledger's suspects are what 9935 would order. Task 7921 plans its own skip at the same pick filter and in `chain_snapshot`. Task 5041 moves `_deep_chain_placement` and likely the mark.
12. **Task 7921's text cites stale names:** `_build_main_health_outcome` lives in `main_health_probe.py`, and the pick is `MergeIntake.pop_next_pickable`.

Files used most: `/home/leo/src/dark-factory/orchestrator/src/orchestrator/merge_lane/worker.py`, `.../merge_lane/inflight_verify.py`, `.../merge_lane/verify_dispatch.py`, `.../merge_lane/types.py`, `.../merge_lane/intake_buffers.py`, `.../suffix_graph.py`, `.../workflow.py`, `.../merge_lane/main_health_probe.py`, `.../verify.py`, `.../event_store.py`, `/home/leo/src/dark-factory/orchestrator/tests/merge_lane_ratchet_baseline.json`, `/home/leo/src/dark-factory/orchestrator/tests/test_merge_queue_deep_unchained_mark.py`.
