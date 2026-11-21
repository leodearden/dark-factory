# Merge-pool PRD scout: the lane as it is on main (read-only)

Tree read: main `ec46ac6670` (2026-11-13 14:51Z). Every cited source path was clean (`git status --short` printed nothing for merge_lane/, suffix_graph.py, overlap_footprint.py, workflow.py, harness.py, config.py, git_ops.py and the four plans). The task store was opened read-only (`?mode=ro`). Code is cited by symbol. Where a fact rests on a task's text and not on code, it says so.

---

## A. The pick and the batch today

A1. **There is one pick chokepoint:** `merge_lane/intake_buffers.py::MergeIntake.pop_next_pickable`. Both the merger's regular acquire (`worker.py::SpeculativeMergeWorker._acquire_next_request`) and the merger's speculative look-ahead call it. It returns None while admission is halted. It walks `MERGE_LANES` in order, `high` then `normal`, and skips halted lanes (`HaltState.is_lane_halted`). For the first lane with a visible item it calls `_clique_minimal_visible(lane)`. It deletes the chosen request from that lane's deque and moves its lifecycle state from LANE_BUFFERED to MERGING. It is synchronous, with no await.

A2. **`_clique_minimal_visible` filters, then picks.** It drops reserved requests (`is_reserved` → `ChainReservations.holds`). Of the rest, it returns the FIFO-earliest request x such that no visible `footprint_neighbors(x)` in the same lane has a strictly smaller `_aging_key`. Requests with no footprint neighbours are "vacuously minimal", so a disjoint younger item passes an older contested clique. With an empty graph the pick is pure FIFO. The fallback is `visible[0]`.

A3. **`_aging_key(req)` is `(merge_first_enqueued_at or enqueued_at, request_id)`**, smaller first. `merge_first_enqueued_at` lives in task metadata (stamped once by `workflow.py::TaskWorkflow._stamp_first_merge_enqueue` into the retry ledger). `enqueued_at` is re-stamped on every submission. **MergeRequest has no tier or priority field.** Its only priority is `lane: Literal['normal','high']` (`merge_lane/types.py::MergeRequest`). The lane comes from `worker.py::lane_for_task_metadata` and can be changed by `MergeIntake.relane_waiting`. The scheduler tier exists only in `scheduler.py::Scheduler._last_effective_priorities`, and the lane cannot see it today.

A4. **Lane buffers** are `MergeIntake._lane_buffers: dict[lane, deque[MergeRequest]]`, FIFO within each lane. Their single writer is the merger task (`assert_merger_owns`). `MergeIntake.buffered(lane)` returns a tuple. `buffer_owned_request` appends and journals.

A5. **`take_buffered(task_id)` and `restore_buffered(lane, index, req)`** are the chain walk's pair. `take_buffered` removes the first buffered request for a task, searching high lane first, FIFO, and returns `(lane, idx, req)`. It runs on the verifier task as a documented second writer. `restore_buffered` re-inserts at the exact index, deliberately not via `_requeue_request`, so the request keeps its slot. `request_redispatch(item, front=)` and `next_redispatch` drive the front-priority redispatch park (`_redispatch` deque of SpeculativeItems). Re-merged or redispatched items go there and are dispatched before `_verifier_queue`.

A6. **Chain-link reservation (task 7720, done 2026-11-12)** is `merge_lane/chain_reservation.py::ChainReservations`, created in `SpeculativeMergeWorker.__init__` as `self._chain_reservations`.
   - Representation: `ChainReservation(owner_request_id, owner_task_id, links: ((request_id, task_id),...), tip, deadline)`, kept in `_by_owner` plus an inverse `_owner_of_link`.
   - `reserve_candidates` holds `snapshot[:d-1]` provisionally. A request held by another owner raises ValueError (INV-2).
   - `confirm` narrows the hold to the built links and sets `deadline = monotonic + merge_deep.chain_reservation_timeout_secs` (default 7200).
   - `release` and `expire` drop the hold and fire `on_release` (the merger's `_resume_signal.set`).
   - Release sites: `_deep_chain_placement` on a build error or None result; `_end_chain_reservation(all_landed=...)` at the round's end in `_run_inflight_verify` (non-adopting exits), in `_finalize_inflight`'s finally, at teardown, and after `_land_chain_prefix`; `_expire_chain_reservations`, called from the verifier loop.
   - A confirmed tip whose links did not all land is recorded dead (`DeadBaseSite.CHAIN_RELEASED`).
   - Reserved links stay in the lane buffers. They are hidden from `pop_next_pickable`, `chain_snapshot` and the bounce (`SuffixConflictTracker.bounce_conflicting_suffix_items` skips `is_reserved`).
   - The state is **in memory only**: `clear()` on stop, no journal.
   - `merge_lane/lookahead_hold.py::LookaheadHold` makes the merger's look-ahead pop wait for the chain decision on the item it just queued.

A7. **`worker.py::SpeculativeMergeWorker.chain_snapshot`** returns the unfrozen suffix as request objects: high lane first, FIFO within a lane. It skips halted lanes, reserved links and `_coalescing_ids`. It deliberately does NOT apply clique-minimality ("Do NOT sort by `_aging_key`"), because δ's CAS walk needs a contiguous prefix in submission order. It is pure: no pop, no transition.

A8. **`worker.py::build_chain(git_ops, queue_snapshot, head_merge_commit, *, cap, target_depth)`** takes one chain-build lane (`acquire_chain_build_lane`). It checks that the lane HEAD equals the base (`lane_base_mismatch` otherwise). It then `git merge`s `queue_snapshot[:min(len, cap, target_depth)]` sequentially, in snapshot order. It truncates at the first refused item (a train, or a branch stacked on an unlanded base) or the first non-clean merge, and emits no outcome for the truncated item (deep-merge decision 4). It returns `ChainResult(links=[(task_id, merge_commit)...], tip, truncated_at, truncated_reason, build_ms)`. A non-empty result holds its lane.

A9. **`select_chain_depth(chain_cap, queue_len, halving_state)` and `next_halving_state(passed, dispatched_depth)`** are pure module functions.
   - Depth = `min(queue_len, chain_cap, halving_state)`. It returns None when the cap is 0, when the queue is under 2, or when the result is under 2.
   - On a fail the state becomes `max(1, d//2)`. On a pass it resets to None.
   - Live state: `SpeculativeMergeWorker._chain_halving_state: int | None`, initialised in `__init__`. Its only writer is `_note_chain_outcome(passed, depth)`; its reader is `_deep_target_depth`. It is **in memory only** and resets on restart.
   - `_note_chain_outcome` has two call sites in `_run_inflight_verify`: the chain arm, with the built depth; and the d=1 floor reset arm, which fires only on a passing speculative verify with `chain_cap > 0` and a non-None state.

A10. **`_deep_chain_placement(item)`** decides whether a dispatch builds a chain. Its guards, in order:
   - (1) `not item.speculative` → None: **the head (slot 1) is never chained today**;
   - (2) `merge_deep.chain_cap <= 0` → None, read live;
   - (2b) a one-shot `_chain_error_suppressed` guard;
   - (3) `d = select_chain_depth(...)` with `queue_len = 1 + len(chain_snapshot())`.
   The base is the **dispatching item's own merge commit** (`item.merge_result.merge_commit`), never `frozen_prefix_tip`. It reserves, builds under `asyncio.timeout(chain_build_timeout_secs)` (`_build_deep_chain`), and confirms. Every failure path falls open to the adjacent verify. The build runs inline on `_verifier_loop` (7824, pending, moves it off).

A11. **Chain verdicts.**
   - A green tip goes to `_adopt_head_on_tip_authority` (cancels the head's verify). `_finalize_inflight`'s PASS arm then runs `_land_chain_prefix(entry, head_sha)`.
   - That walk `take_buffered`s each link in order and `_journal_landed_then_advance`s it with `expected_main` = the predecessor's commit and `max_attempts=1`. It aborts silently at the first non-`advanced` result, restoring the link (decision 9). Each landed link carries `MergeOutcome.landed_via_chain=1`.
   - A red tip goes to `_requeue_from_verify`: the dispatching item goes back on `_queue` via `_requeue_request` and loses its lane slot. The links never left their buffers and are released by `_end_chain_reservation`.
   - On a red tip there is **no merge_attempt, no blocked MergeOutcome and no culprit**. The code comments say why: "a red tip names no culprit", and silence keeps `workflow.py`'s `consecutive_merge_thrash` ladder clean.

A12. **What survives a restart:** only the request journal, `merge_lane/queue_store.py::MergeQueueStore`.
   - `PersistedMergeRequest` holds request_id, task_id, branch, worktree, pre_rebased, task_files, snapshot_tip, generation, lane, enqueued_at and module_prefixes.
   - It **does not hold `merge_first_enqueued_at`** (task 5623 owns that gap). Trains are never journaled.
   - The halving state, reservations, suffix graph, bounce registry, redispatch park and `_inflight` are all in memory.

## B. What task 8528 replaces (plans/merge-lane-policy-prd.md D6, D7, §6, §8 ζ)

B1. **Live status (task store):**
   - **8528** ζ is pending/high and depends only on 8527. It has about 200 dependents, mostly pending: the 2026-11-10 "chain parking" of worker.py tasks behind 8528, including 5040–5045, 5306, 7921, 7922, 8933, 9482, 9133, 9144, 9182, 9188.
   - **8527** ε is pending/high and depends on 7720 (done) and 8526 (pending). Its amendment drops 7721 and 7824.
   - **8526** δ is pending/high and depends on 8525 (done). A `task/8526` branch exists with WIP commits; nothing has landed on main.
   - No `task/8527` or `task/8528` branch exists.

B2. **D6 / 8528 work item 2 (quoted)** is the part a probability-driven batch rule supersedes: "On a tip fail at built depth k: requeue the k links at the FRONT in aging-key order, suspects := those links, cancel stacked successors and requeue their items behind the links. While suspects exist, the next dispatch verifies the first ceil(|suspects|/2) as a chain: a passing half lands and is removed; a failing half becomes the suspect set; a lone failing suspect is blocked with its findings. Full depth resumes only when no suspects remain. next_halving_state is deleted."

B3. **D7 / 8528 work item 3 (quoted)** is superseded the same way: "p̂ = per-item pass estimate from the last 30 FRESH tip verdicts (bisect rounds excluded), p̂ = pass_rate^(1/mean k); at start, seed from the last 30 merge_verify rows. Depth = the first row of merge_deep.depth_table whose threshold p̂ meets, default [[0.93, 8], [0.85, 4], [0.75, 2]] else 1, capped only by queue depth. The table is a validated green-tier config list (config.py::MergeDeepConfig, RELOADABLE_FIELDS); chain_cap becomes the kill switch only (0 = off, any positive = on)."

B4. **Planned names (contract §6 "Depth and suspects"):** `ChainPolicyState(suspects: list[task_id], recent: deque[(k, passed)])`; the pure functions `choose_depth(state, queue_len, depth_table) -> (k, reason)` and `on_tip_result(state, links, passed)`. `_chain_halving_state` and `next_halving_state` are deleted.
   - Where the state lives: the PRD says it replaces `_chain_halving_state`, so it is worker state in worker.py (8528's files: worker.py, config.py, dark-factory-orchestrator.yaml, test_merge_queue_deep_dispatch.py).
   - Persistence: §10 open question 1 suggests "in-memory, seeded from the last 30 `merge_verify` rows at start (the probe's `_recent_verify_outcomes` shape)". The task text says "at start, seed from the last 30 merge_verify rows". The suspect list's persistence is not stated anywhere, so read it as in memory.

B5. **New event field:** `merge_verify.chain_depth_reason` records p̂, the row chosen, the queue depth and the suspect state (D7; 8528 item 4). Boundary rows 7 and 8 are 8528's tests.

B6. **8528 still verifies the lone suspect.** "A lone failing suspect is blocked with its findings" means the suspect gets a verify of its own (a depth-1 chain) before it is blocked. The merge pool's "convict at ~0.9 without a verify of its own" removes that step. 8528 names no category or outcome for the block.

B7. **8528 keeps FIFO batches.** Its batches are prefixes of the requeued links "in aging-key order", and full-depth rounds still build from `chain_snapshot` (FIFO), not from probabilities. D14 lists "queue re-ordering" as dropped. §9 puts "queue re-ordering" out of scope.

B8. **Amendments in the 8528 text:** the S16 amendment (dated 2026-11-14 in the store) widens the depth_table check's grep path to `orchestrator/src/orchestrator/` because task 9709 (S16 ζ5a, pending, depends on 8528) moves merge-lane knobs into `config_sections/merge_lane.py`. 9133 notes that if 8528 lands first, 9133 only edits a docstring sentence.

## C. How a failed merge verify gets back to the task

C1. **One request, red verify.** `worker.py::_run_post_merge_verify` builds the outcome.
   - An unscoped type-check failure gives `MergeOutcome('blocked', reason=..., block_category=POST_MERGE_VERIFY)`.
   - Otherwise it first runs a main-health classification. In synchronous mode that is `_classify_main_health_red`. In deferred mode a probe is spawned and the reason gets `MAIN_HEALTH_PROBE_PENDING_NOTE` appended.
   - It then builds `reason = 'Post-merge verification failed: ' + (new failing ids vs a cache-warm main baseline, via describe_new_failing_ids | verify.summary [category])`, followed by `verify.failure_report()` text, and optionally a skew suffix.
   - The result is `MergeOutcome('blocked', reason, failure_category=verify.category, failure_cause_hint=verify.cause_hint, disposition, failure_diagnostic, skew_evidence, block_category=MergeBlockCategory.POST_MERGE_VERIFY)`.
   - It also fires `_spawn_merge_verify_dry_run` (a dry-run investigation) unless the verify timed out.

C2. **A batch tip.** A red tip yields no MergeOutcome for anyone; it requeues (A11). A chain-arm exception also requeues and sets the one-shot `_chain_error_suppressed`. Only an ordinary verify of an item's own tree produces a blocked outcome.

C3. **`MergeBlockCategory`** (`merge_lane/outcome_facts.py`) has exactly three values: `POST_MERGE_VERIFY`, `MERGE_NEEDS_REBASE` and `MERGE_ERROR` (the default). Each is an `EscalationCategory` value. `MergeOutcome.status` literals: done, conflict, blocked, already_merged, wip_halted, done_wip_recovery, wip_recovery_no_advance, unmerged_state, stash_failed, unknown_branch, superseded, error. `MergeFailureDisposition`: main_red, integration_skew, branch_bug, indeterminate.

C4. **The workflow side** is `workflow.py::TaskWorkflow._submit_to_merge_queue`, called from `_run_merge_phase` with `merge_phase=True`.
   - Status dispatch:
     - wip_halted, done_wip_recovery, wip_recovery_no_advance, unmerged_state and stash_failed each go to a handler;
     - done and already_merged → DONE;
     - superseded → `_handle_superseded`;
     - conflict → `_resolve_and_resubmit(conflict_details)`, an in-worktree resolve and resubmit.
   - For `blocked`, routing is by **reason prefix** (`startswith`):
     - WORKTREE_MISSING → a terminal check;
     - DROPPED_PLAN_TARGETS, POST_MERGE_EQUIVALENCE_FAILED and TRANSIENT_INFRA (category `infra_issue`) → `_mark_blocked(..., escalate_to_human=True)`;
     - POST_MERGE_PYRIGHT_BROKEN → `escalate_to_human` plus a dry-run;
     - MAIN_HEALTH_RED → `_auto_heal_main_health`.
   - Then by disposition: INTEGRATION_SKEW → an L1 `integration_skew` with `port_landed_change`.
   - Everything else, **including the normal POST_MERGE_VERIFY red**, takes the generic arm: `_write_merge_failure_review(category, reason)` (a reviews/ JSON), an unconditional `merge_blocked` event (reason[:500], category, failure_category, cause_hint, plus conflict_evidence paths), then `_mark_blocked(reason, merge_phase=True, category=result.block_category.value)`.

C5. **`_mark_blocked` with `merge_phase=True` and no `escalate_to_human`:**
   - It files an **L0** escalation, `category='post_merge_verify'`, with summary = reason[:200] and detail = reason. It files only if no open L1 of the same category exists.
   - It starts the per-task **steward**, which has Read, Edit, Write and Bash in the worktree (`agents/roles.py::STEWARD`), and awaits it.
   - `StewardResolved` → `REQUEUED`, stamping `metadata.merge_retry_pending`.
   - `_run_merge_phase` then checks the merge-outcome signature for thrash (`_check_merge_outcome_thrash`, `consecutive_merge_thrash` vs `max_consecutive_merge_thrash`), runs `_settle_unmergeable_branch`, and **retries the merge in place**: rebase, then resubmit. This repeats up to `max_merge_retries`.
   - The workflow does NOT re-run the implementer or debugger pipeline. The steward is the fixer. The task stays in-progress throughout.

C6. **Evidence that travels:**
   - `reason`, free text including `verify.failure_report()` and, when the main baseline is cache-warm, the NEW failing ids;
   - `failure_category`, `failure_cause_hint`, `disposition`;
   - `failure_diagnostic` (a dict, skew only);
   - `skew_evidence` (`SkewEvidence(implicated_commits, failing_tests, overlap_files)`, only when classification ran);
   - `conflict_evidence` (`ConflictEvidence(conflicted_paths, counterpart_task_ids)`, only for conflict and needs-rebase).
   There is **no structured failing-test-id field on MergeOutcome**. `VerifyResult.failing_test_ids` and `failing_test_ids_by_module` exist in memory but reach the workflow only as reason text.

C7. **Reuse for a conviction by elimination.** The existing generic arm (a blocked outcome with `block_category=POST_MERGE_VERIFY` → L0 → steward fixes in the worktree → in-place retry) is the closest "your branch failed the merge verify, fix it" path. The workflow routes it on category, not prefix. But:
   - (a) it expects the reason to describe the task's own verify;
   - (b) every blocked outcome feeds the `consecutive_merge_thrash` signature. The deep-chain code deliberately avoids rendering outcomes for chain fails for exactly this reason (A11; `_land_chain_prefix` docstring);
   - (c) there is no OutcomeKind or category meaning "convicted from batch evidence".
   Task 9144 (S6 β1, pending) moves workflow routing from reason prefixes to `MergeOutcome.kind`, adding 8 OutcomeKind members. A new conviction kind would land in that vocabulary.
   A precedent for batch attribution exists: `workflow.py::TaskWorkflow._attribute_train_failure` with `worker.py::reverify_member_solo` (solo re-verify of each train member).

C8. **`merge_verify.units`** come from `verify_runner.py::VerifyRunnerPool.dispatch` (the emit site is not in worker.py) via `verify_evidence.py::verify_event_fields`, as `[VerifyUnit.to_dict()]` = `{unit, tier, key, outcome, evidence_event_id}`.
   - The only producer of VerifyUnit is `verify_evidence.py::module_leg_unit`, called from `verify.py`. So the units are **per module leg**: `module:<prefix>:<test|lint|type>`, outcome `ran_green|ran_red`.
   - `UnitOutcome` also has cache_hit, would_hit and unavailable, and `UnitTier.TEST` exists for cache keys, but no producer on main emits test-tier units.
   - **Failing test ids are not on merge_verify.** They appear on events only in `merge_attempt.failing_tests`, and only with skew evidence (`gates.py`), and on the remote no-verdict audit record (`verify_runner.py`, `'failing_test_ids'`).
   - The other merge_verify keys: runner, merge_sha, passed, duration_ms, attempt, depth, speculative, chain_items, chain_build_ms, runner_reason, base_relation, retry_scope fields, plan_hash, cache_mode, units. **There is no list of batch members.** Task 9908 adds `chain_request_ids` and `chain_task_ids`; it exists only on branch `task/9908`, not on main.

## D. Verify slots and the frontier

D1. **K.** `harness.py::Harness._speculation_k` returns `1 + len(config.enabled_verify_runners)`. It sizes the GitOps spec pool and the worker's speculation depth. Two permit ledgers (`speculation_controller.py::PermitLedger`, SpecPermit and CapPermit) size from it. 8526 (pending; WIP on `task/8526`) plans `1 + runners + 1`, with the host lease released at verdict and the entry kept as `verified_pending_landing` (D4).

D2. **Containers.** `SpeculativeMergeWorker._verifier_queue: asyncio.Queue[SpeculativeItem|None]` is fed by the merger in pick order. `_inflight: deque[InflightEntry]` is the dispatched frontier. `_intake._redispatch` is the front park.

D3. **`_verifier_loop`** has two phases.
   - (a) DISPATCH-FILL: take the redispatch park first, then `_verifier_queue`, while `HostAllocator.free_host_count() > 0`; race a persistent getter against running verifies.
   - (b) FINALIZE-HEAD: wait on `_inflight[0]`, `popleft`, `_finalize_inflight`.
   **Main advances strictly in submission order.** This is also a frozen-surface invariant in `merge-lane-quality-prd.md` §Contract.

D4. **Base choice.**
   - The merger (`_merger_loop`) merges each picked request onto `SpeculationController.base_for(actual_main)`: the predecessor's merge commit (`spec_base`) when a speculation permit is held, or a late-arrival attach (`on_dequeue`, task 1862); otherwise actual main (FALLBACK, non-speculative).
   - `frozen_prefix_tip(main)` = `remerge_base.py::frontier_tip(_frozen_inflight_entries())`, the newest entry's chain tip or merge commit.
   - A chain's base is the dispatching speculative item's merge commit (A10).

D5. **Head fails: the cascade** (`_verifier_loop`, `if not _head_advanced and self._inflight:`).
   - The failed head's commit is recorded dead (`DeadBaseSite.HEAD_FAILED`).
   - Each downstream entry's verify is cancelled and its worktree cleaned. It is re-merged via `_remerge` (claim through `remerge_base.py::claim_remerge_base` → `choose_remerge_base`) and front-queued on the redispatch park in submission order.
   - Per two-layer PRD §5.3 (as amended by 8524), the cascade re-merges after the frontier is torn down, so it "structurally falls to main". `choose_remerge_base` returns the live frozen tip only when it is not main, not dead, owned by a frontier entry with a live verify, chain-intact and not vacated, and a permit is available. Otherwise it returns main with a `RemergeReason` (no_tip, tip_dead, tip_vacated, tip_verify_dead, no_permit).
   - A speculative item dispatched after a failed predecessor re-merges (`previous_failed`) at `_dispatch_item`.

D6. **Can the lane dispatch a batch that is not stacked on the frontier tip?**
   - For a **chain**, no. Guard 1 of `_deep_chain_placement` allows chains only on speculative items, and the chain base is that item's merge commit. 8527 lets a lone head chain on main.
   - For a **single item**, yes. A FALLBACK merge on main and a re-merge that fell to main both dispatch a non-speculative verify beside in-flight entries. These are decided in `_merger_loop` (`SpeculationController.on_dequeue` / `base_for`) and `_remerge` (`choose_remerge_base`).
   - Filed defects show verifies dispatched on dead or behind-queued speculative bases (8933; specimens -6, -18, -22, -24, -29/-30). INV-3 (`_void_and_remerge`) voids those verdicts at adoption.

## E. The conflict machinery

E1. **Where it runs.** `suffix_graph.py::SuffixConflictTracker.recompute` is called from `_acquire_next_request` on every acquire, through `MergeIntake.recompute_suffix_conflict_graph`, then `bounce_conflicting_suffix_items`, then `pop_next_pickable`.
   - Suffix = the lane buffers minus frozen requests, in pick order. Debounce signature = `(ordered_request_ids, main_sha)`; it is stored only when no probe raised.
   - **Footprint = `GitOps.get_changed_files(main_sha, head)`**: `git diff --name-only <main_sha> <head>`, a two-dot diff against **current main**, not the merge base. That diff includes main's changes since the branch forked. The docstring admits it is approximate.
   - `get_changed_files` **fails open to `[]`** on a git error, without raising. So a failed diff reads as an empty footprint (no edges) and does not set `_any_probe_error`.
   - Edges come from `overlap_footprint.get_overlap_detector(None)` (always the default path detector) as a non-empty path intersection.

E2. **A merge-base variant exists in git_ops.** `GitOps.get_branch_changed_files(ref)` is `git diff --name-only main...ref -- :!.task/`, a three-dot diff from the fork point. It returns `(files, error)` and never raises; the skew tripwire uses it today. `GitOps.merge_base_with_main(ref)` and `get_changed_line_ranges(ref)` (three-dot) also exist.

E3. **Pairwise probe.** Only footprint-overlapping pairs get `GitOps.merge_tree_conflicts(head_i, head_j)` (`git -c core.quotePath=false merge-tree --write-tree --name-only -z`, at project_root; it raises on a git error). Results are memoised per recompute. A missing ref or a raise counts as a textual edge (fail-open).
   - Each suffix item is also probed against `frozen_prefix_tip(main_sha)`. Unclean or raised → `conflicts_with_main`, with a `ConflictEvidence(conflicted_paths, counterpart_task_ids = frozen task ids)`.
   - The result is the frozen `SuffixConflictGraph(nodes, textual_edges, footprint_edges, conflicts_with_main, conflict_evidence)`.

E4. **Bounce.** `bounce_conflicting_suffix_items` walks each request in `conflicts_with_main`, skipping trains and reserved requests.
   - It calls `bounce_registry.record_bounce(branch)`.
   - If the count exceeds `MERGE_BOUNCE_CAP` (`merge_lane/worker.py`, `= 3`, reached through the deferred import `_worker_constants`), it calls `_escalate`.
   - Otherwise it runs `rebase_onto_main(req.worktree, onto=frozen_tip)`. A clean rebase leaves the request in place; a real conflict calls `_escalate`.
   - `_escalate` removes the request from the lane, retires it, clears its bounce count, and resolves it as `MergeOutcome('blocked', NEEDS_REBASE_REASON_PREFIX…ordinal, block_category=MERGE_NEEDS_REBASE, conflict_evidence)`.
   - `merge_lane/types.py::MergeBounceRegistry` is an in-memory per-branch counter (`count`, `record_bounce`, `clear`).
   - Note: task 6124 (pending) says the bounce rebases onto a speculative frozen tip whose item may later fail.

E5. **Invariant (i).** `worker.py::SpeculativeMergeWorker.two_layer_invariants` reports every edge in `textual_edges - footprint_edges` as "textual⊈footprint contract violated". Invariant (ii): `conflicts_with_main ⊆ nodes`. The results reach the operator through `snapshot()['two_layer_invariants']`.

E6. **Tests that pin the clique-aging pick (task 1891):**
   - `orchestrator/tests/test_merge_queue.py::TestAgingPickOrder`, which includes `test_aging_orders_conflict_clique_by_first_submission`, `test_aging_clique_fallback_to_enqueued_at_when_mfea_none`, `test_aging_tiebreak_equal_timestamp_request_id_lexical` and `test_aging_is_clique_scoped_and_allows_disjoint_bypass`;
   - `test_merge_queue.py::TestLanePickOrderHelpers` and `::TestLanePickIntegration`;
   - `orchestrator/tests/test_merge_queue_two_layer_integration.py::TestScenario4And5AgingAndDisjointBypass`;
   - `orchestrator/tests/test_merge_lane_intake_buffers.py` (pick, halt and reservation cases, e.g. `test_a_reserved_older_neighbour_does_not_block_the_item_behind_it`).
   `pop_next_pickable` is also referenced by test_merge_queue_build_chain.py, test_merge_queue_conflict_graph.py, test_merge_queue_deep_*.py, test_merge_queue_lifecycle_registry.py, test_merge_queue_single_writer_asserts.py and test_merge_speculation.py.

E7. **Tests that pin the bounce cap:** `orchestrator/tests/test_merge_queue_bounce.py` (`TestBouncePrimitives::test_merge_bounce_cap_is_positive_int`, `TestBounceBookkeeping::test_cap_exceeded_escalation_retires_the_request`, and the `TestBounceEscalationsNeverFeedTheThrashLadder` cap-out cases), `test_merge_queue_two_layer_integration.py` (pre-seeds `MERGE_BOUNCE_CAP`), and `test_suffix_conflict_tracker.py`.

## F. Main health in the lane

F1. **Confirmation path.** A red verify goes through the main-health probe: `_classify_main_health_red` (synchronous), or the deferred probe calling `SpeculativeMergeWorker._auto_heal_main_health_deferred`. Once a pre-existing main break is confirmed, the auto-heal-eligible path:
   - records the attempt;
   - calls `halt_lane('normal', ...)` (`merge_lane/halt.py::HaltState`) and sets the lane halt owner to the escalation id from `_file_main_health_escalation`;
   - spawns a HIGH-lane fix task through `main_health_fix_owner.py` (one open fix-main task per sha-independent signature, `metadata.main_health_signature`).
   Non-eligible breaks and attempt-cap cases call `_escalate_owned_main_break` without halting. The workflow-side twin is `workflow.py::TaskWorkflow._auto_heal_main_health`.

F2. **Synchronous signals readable at dispatch today:**
   - (a) `SpeculativeMergeWorker.is_lane_halted('normal')` / `HaltState.is_lane_halted` / `halt_owner_esc_id`. This is set only by the auto-heal path above and by operator or WIP halts.
   - (b) `verify.py::known_failing_ids_on_main(main_sha)`, a cache-only peek: "empty means none known, not green".
   - (c) `verify.py::cached_main_baseline_failing_ids(main_sha, red_module_prefixes=)`, complete-or-None.
   None of these is a "main is confirmed red now" verdict object.

F3. **7921** (pending/medium) depends on 7918 (done), 7919 (pending) and 8528 (pending). It adds `merge_lane/main_red_verdict.py`: `MainRedVerdict{main_sha, failing_ids, signature, owner_esc_id, confirmed_at, generation}` and a `VerdictBook` (arm, clear, degrade, current, held), armed only in `_auto_heal_main_health_deferred`.
   - It runs F (the failing ids) on each item's tree via `verify.run_node_ids`. F passes → verify now. F fails on both the item and bare main → **HOLD** in the lane (a mark over LANE_BUFFERED: skipped by `_pop_next_pickable`, excluded from `chain_snapshot`).
   - Config `merge_main_red_hold.*`; events `main_red_verdict` and `merge_held`.
   - Ratchet pins in its text: `_dispatch_item` 43, `_pop_next_pickable` 13, `chain_snapshot` 3.
   Once it lands, `VerdictBook.current` would be the synchronous "main red" read. 7922 (chain-tip screen) and 8411 depend on 7921.

## G. Config and reload

G1. **Model.** `config.py::MergeDeepConfig` is a plain BaseModel, mutable in place for reload. Fields:
   - `chain_cap: int = 0` (ge 0; 0 is the kill switch; dark-factory-orchestrator.yaml sets 6);
   - `chain_build_timeout_secs: float = 1800` (gt 0; df yaml 300);
   - `chain_reservation_timeout_secs: float = 7200` (gt 0).
   It is mounted as `OrchestratorConfig.merge_deep`.

G2. **Reload.** `RELOADABLE_FIELDS` (config.py, module level) includes `_submodel_leaf_paths('merge_deep', MergeDeepConfig)`, so every leaf, including future ones, is green-tier with no list edit (sibling: `speculation_probe`). The pattern for a new `merge_pool.*` block is a new plain-BaseModel submodel on `OrchestratorConfig` with `Field(default_factory=...)`, a `_submodel_leaf_paths('merge_pool', ...)` entry in RELOADABLE_FIELDS, and a stanza in `orchestrator/src/orchestrator/defaults.yaml` (`merge_deep:` already lives there).
   Pending task 9709 (S16 ζ5a, depends on 8528 and 9708) moves merge-lane knobs into `orchestrator/src/orchestrator/config_sections/merge_lane.py` (the file does not exist on main). A new block would sit there once that lands.

G3. **Kill switch.** `_deep_chain_placement` reads `chain_cap` live per dispatch and checks it before any queue scan. `chain_cap = 0` makes the chain path unreachable, and the floor reset arm checks it too. Under 8528 it becomes "0 = off, any positive = on".

## H. Rules that bind new lane code

H1. **The frozen surface.** `plans/merge-lane-quality-prd.md` §Contract "Frozen public surface" says these are byte-identical before and after every task:
   - `MergeOutcome.status` literals and `reason` strings;
   - the 17 reason-prefix constants;
   - "`EventType` emissions and their `data` keys";
   - `snapshot()` keys (additive only);
   - the journal and outbox schemas;
   - "Ordering invariant: `main` advances strictly in submission order".
   That PRD scopes this to its own refactor tasks (Corrections §8: net-additive work has an authorized-raise path). 9908's text reads the freeze as allowing additive keys ("no existing key changes meaning"). A merge pool that lands out of arrival order would need to amend the ordering line explicitly. Whether "submission order" means pick order is not defined there.

H2. **The ratchet** is `orchestrator/tests/test_merge_lane_ratchet.py` against `orchestrator/tests/merge_lane_ratchet_baseline.json` (`scripts/merge_lane_metrics.py`).
   - Cluster: `merge_lane/**/*.py`, git_ops.py, lane_lifecycle.py, offline_lane.py, warm_lane_pool.py, recover_main.py, plus two test helpers. **`suffix_graph.py` and `overlap_footprint.py` are outside the cluster.**
   - Ratcheted measures:
     - per file: cognitive, function_local_imports, reexport_names (lines and prose_lines are reported, not ratcheted);
     - per function: cognitive;
     - per test file: patch_targets, private_reads.
   - Ceilings apply only to keys **absent from the baseline**: a new file must be ≤ 1500 lines (git_ops.py exempt), a new function ≤ cognitive 15.
   - Raising a measure requires `--write-baseline --authorize-raise <task> --reason`, which writes one JSON under `orchestrator/tests/merge_lane_ratchet_raises/` (about 207 files today). `scripts/check_staged_ratchet_raise.py` (pre-commit) and `hooks/reference-transaction` → `scripts/check_ratchet_advance.py` (on main moves) enforce it.
   - Current pins:

     | Function | Pin |
     |---|---|
     | `MergeIntake.pop_next_pickable` | 6 |
     | `MergeIntake._clique_minimal_visible` | 7 |
     | `MergeIntake.take_buffered` | 6 |
     | `chain_snapshot` | 7 |
     | `build_chain` | 17 |
     | `select_chain_depth` | 4 |
     | `next_halving_state` | 1 |
     | `_note_chain_outcome` | 0 |
     | `_deep_chain_placement` | 8 |
     | `_land_chain_prefix` | 36 |

     worker.py is 21,765 lines at file cognitive 1961; intake_buffers.py is 836 lines; chain_reservation.py is 167 lines.

H3. **Extraction chain** (all pending; all now also depend on 8528):

   | Task | Status | Depends on | Moves | Owns |
   |---|---|---|---|---|
   | **5040** κ | pending/high | 5305 (done), 8528 | to `merge_lane/verify_dispatch.py` (`VerifyDispatcher`): `_run_post_merge_verify`, `_run_inflight_verify`, `_dispatch_item`, host quarantine, main-health probe | `_runner_unavailable`, `_runner_quarantine`, `_remerge_occurred` |
   | **5041** λ | pending/high | 5040, 8528 | to `merge_lane/speculation.py` (`ChainPlanner`, `Speculator`): chain build, depth and placement, the prefix-landing decision, permits and leases, `_remerge`/`_void_and_remerge`; absorbs speculation_controller.py; the `_merger_loop` body moves too. Covers `build_chain`, `select_chain_depth`, `next_halving_state`, `_note_chain_outcome`, `_deep_chain_placement`, "unchanged" | `_n_failed` |
   | **5042** μ | pending/medium | 5041, 8528, 9265 | to `merge_lane/landing.py` (`Lander`): `_finalize_inflight`, CAS handling, **`_land_chain_prefix`**, outbox and journal; decomposes `GitOps.advance_main` in place | `_inflight` |
   | **5306** ν+ξ | pending/medium | 5042, 8528 | to `worktrees.py` and `telemetry.py` (`snapshot`, heartbeat, contended-lease streak) | |
   | **5045** ο | pending/high | 5306, 8528 | residual worker.py: composition root plus two thin loops (`_verifier_loop`, `_merger_loop` shell) and `stop` | |

   `chain_snapshot` is not named by any of them. `MergeIntake.pop_next_pickable` already sits in `intake_buffers.py`, extracted by task 5305, and no listed task moves it. The policy PRD's D11 says its own leaves land in worker.py and κ/λ move them.

## I. Other filed work

The 7720/8526/8527/8528/9908 group is already covered in B1. Each row gives the relation to the merge pool.

| Task | Status / unmet deps | One line | Relation to the merge pool |
|---|---|---|---|
| 9908 | **deferred**/high; no deps; work on branch `task/9908`, not on main | Record lane, tier, pin and head_sha on merge_queued; the waiting set, textual_conflict_pairs and footprint_pair_count on merge_dequeued; chain_request_ids and chain_task_ids on merge_verify. Record-only; adds a tier/pin lookup callable that 8261 and "the later merge-pool pick" are meant to reuse | **Must land before** (the calibration data and the tier lookup) |
| 7921 | pending/medium; 7919, 8528 | Main-red verdict plus F-screen hold | **Edits the same function** (`pop_next_pickable` and `chain_snapshot` skip held items; `_dispatch_item`). Independent policy; seam: held requests must be excluded from pool batches |
| 7922 | pending/medium; 7259, 7921, 8528 | Screen a built chain tip with static checks; truncate before the first defective link | Edits the same function (`build_chain` / post-build). Partly **superseded** if batch membership comes from probabilities (it decides truncation by screen); seam |
| 8932 | pending/medium; 8532 | Design milestone: idle-host prefix hedge and a dense p̂→depth table, after the ι readout | **Superseded** (it designs on 8528's p̂ table) |
| 8532 | pending/medium; 8528 | ι: 7-day canary readout of the lane predicates | Seam: the merge pool must say whether ι reads before or after it; 3190 depends on it |
| 9133 | pending/medium; 8528, 9127, 9140, 9212, 9255 | Delete the speculation probe and γ2 auto-chain; keep the deep chain | Independent but **edits the same functions** (`_note_chain_outcome` docstring, `_land_chain_prefix` chain_ctx, `_finalize_inflight`) |
| 9137 | pending/medium; 5291, 7038, 7267, 7348, 7425, 8528, 9135, 9182, 9197 | Host selection and quarantine live only in `HostAllocator`; adds `verify_runner.py::dispatch_merge_verify(... chain_items ...)` | Seam: a "suspect slot" reservation would live in `HostAllocator`; edits the merge_verify emit (where chain members would go) |
| 9182 | pending/high; 3235, 4833, 4839, 5040, 5245, 7664, 8528, 9152, 9176, 9256 | ItemLifecycle operative in `merge_lane/item_lifecycle.py`; snapshot derived from the registry | **Edits the same files** (intake_buffers.py, suffix_graph.py, the lifecycle transitions in `pop_next_pickable`) |
| 9188 | pending/high; 4324, 4648, 4651, 4833, 7309, 7494, 7657, 8528, 9150, 9166, 9181, 9187, 9191, 9201, 9224, 9287, 9688, 9765 | Move merge_status, merge_cancel, merge_relane and get_merge_queue to `orchestrator/mcp/merge_tools.py`; `lane_resolution.py` moves into merge_lane | Seam: merge_relane/lanes and get_merge_queue "position" semantics change under the pool |
| 9193 | pending/high; 5409, 9127, 9188, 9192, 9201, 9206 | Move merge_request to merge_tools.py; dedupe gate on the native lane port | Independent (intake entry); seam if the pool changes the MergeRequest shape |
| 5623 | pending/low; 5045 | `PersistedMergeRequest` lacks `merge_first_enqueued_at`, so the aging position is lost on restart | **Must land before** if the pool orders by (tier, age) across restarts. Today it is gated behind the whole extraction chain via 5045 |
| 7319 | pending/medium; 9213 | Stranded-verified-green reaper re-enqueues at the normal lane | Independent; seam (tier/lane source for reaper submissions) |
| 6658 | pending/high; 8528 | Derailed coalesce-train members lose their queue place and re-run the whole workflow; trains emit merge_finalized for members | Independent (trains and coalescing are off per 7739); seam on re-dispatch preference |
| 8933 | pending/medium; 8528 | A verify dispatched on a dead or behind-queued speculative base; suggested fix: mark a requeued chain's speculative merge commits dead | **Must land before or with**: the pool's batch requeue would reproduce it. Edits the red-tip requeue arm |
| 9482 | pending/medium; 8528 | The verify-base detector uses the merger's live tip (`choose_remerge_base`) and escalates only the safety class | Independent; seam (shared live-tip helper) |
| 7825 | pending/high; 7721, 8528 | `merge_deep.chain_tip_hosts: any \| remote_only` | Independent knob; seam (host choice for batches) |
| 5388 | pending/high; 8528 | Live shadow-eval (agent shapes) integration gate; touches intake_buffers.py and ports.py | Independent (not a merge-policy shadow) |
| 9144 | pending/medium; 3177, 3918, 4324, 8528, 9133, 9139, 9140, 9173, 9256, 9283, 9717, 9766, 9899 | `MergeOutcome.kind` / OutcomeKind; workflow routes on kind | **Must land before or coordinate**: a conviction outcome needs a kind. Edits intake_buffers.py and `SuffixConflictTracker._escalate` |
| 9152 | pending/medium; 9140, 9175, 9766 | `MergeLaneQueries` port in `merge_lane/queries.py`; dedupe renames in intake_buffers.py | Edits the same file (renames in intake_buffers.py); otherwise independent |
| 5621 | pending/medium; 5040 | A third of DF textual conflicts slip past the suffix bounce | Seam: the narrowed footprint and probe |
| 6124 | pending/low; 8528, 9267 | The bounce rebases onto a speculative frozen tip whose item later fails | Edits the same function (`bounce_conflicting_suffix_items`); possibly superseded |
| 6944 | pending/low; none | Retire the unused project-registry half of `overlap_footprint.py` | Independent; touches the footprint seam |
| 4841, 5697, 9263, 9267, 9272 | all pending; 8528 or S5 chain | Branch-ref resolution; rerere in `rebase_onto_main`; the S5 GitOps query contract and OntoRef (they touch the suffix_graph.py call sites) | Independent; same-file edits in suffix_graph.py (git calls) |
| 6613 | pending/low | Function-local import style sweep (suffix_graph.py `_worker_constants`) | Independent |
| 9723 | pending/medium; 8528, 9150, 9182, 9201, 9722 | Intake-buffer alarms file through the escalation port | Edits intake_buffers.py; independent |
| 7494 | pending/medium; 8528 | Request-identity residuals | Same file (suffix_graph.py); independent |
| 8261 | pending/high; no deps | verify_admission_order `priority_aged` (tier-ranked admission); 9908's lookup is built for it | Seam: share the tier lookup and aging rule |
| 7925, 5622, 7810 | pending; 8528 or 5040 | The request source on merge events; a true queue_depth; head landed-on-tip not stamped | Independent recording work named in 9908's not-in-scope list |

I-2. **Recorded ruling, submission order.** `plans/deep-merge-ahead-prd.md` §"Resolved design decisions" 1 ("Build-on-dispatch, not incremental build-ahead", Leo): "build the chain *right then*: sequentially merge queued items in submission order onto the head's merge commit in **one scratch worktree**, truncating at the first textual conflict, then verify the tip."

I-3. **Recorded rationale, clique aging.** `plans/two-layer-merge-queue-prd.md` §4 decisions 4–5:
   - "Aging gives a *stable* total order (converges; no reorder thrash), and protects exactly the large-slow tasks that cost the most to redo."
   - "Ordering is conflict-clique-scoped, within lane. High lane still beats normal lane. Within a lane: among items that mutually conflict (footprint relation), order by first-submission age; items disjoint from everything ahead may be picked out-of-order (throughput bypass)."
   §5.3: "aging guarantees no item in a conflict clique starves." §5.3 also says the unfrozen suffix "may be reordered/inserted freely".

## J. Shadow precedent

J1. **`merge_lane/shadow.py` is not a policy shadow.** It is the warm-vs-cold verify shadow-compare detective (PRD §10 inv. 6(b)).
   - `_maybe_schedule_shadow_compare` runs a cold verify of a landed warm merge off the lane, through `asyncio.create_task`, every N merges or nightly. Its cadence state is persisted in `data/orchestrator/warm_verify_shadow.json` (`ShadowCompareState`).
   - It diffs per-test results (`diff_per_test_results`), emits `verdict_parity_ok` on agreement, and files an escalation on persistent divergence.
   - It does not evaluate an alternative decision function.

J2. **No in-lane shadow evaluation of a pick or batch policy exists.** The closest shapes are:
   - (a) pure policy functions beside live state (`select_chain_depth`/`next_halving_state` vs `_note_chain_outcome`), which a shadow could call without acting;
   - (b) `shadow_eval.py` (live shadow-eval of agent shapes, task 5388), unrelated to the merge lane.
   9908 explicitly rules "compute no would-be pick in-process; any candidate order is computed offline from these rows". Offline replay over 9908's events is the planned evaluation path. A new pure function could be called in-process and its decision emitted as an event, but nothing on main does this for the lane.

---

## What does not exist (needed by the merge pool, and not delivered by a filed task unless noted)

1. A tier or priority on `MergeRequest`, or any tier input to the pick. Today the pick is two lanes plus aging. 9908 adds a tier/pin lookup for recording only, on a branch.
2. A per-request failure probability, an observation log (which requests were verified together and the verdict), exact inference, or a false-failure-rate parameter. Nothing records batch membership on main (9908 adds it on merge_verify, on a branch). 8528's p̂ is one lane-wide rate, not per request.
3. Batch membership chosen by anything other than a contiguous FIFO prefix of `chain_snapshot` truncated at the first conflict. There is no non-contiguous or priority-ordered batch, and `_land_chain_prefix` assumes a contiguous in-order prefix.
4. Conviction without a verify of its own. A red tip names no culprit by design (A11), 8528 still verifies the lone suspect (B6), and no OutcomeKind, category or reason prefix means "convicted from batch evidence".
5. A structured failing-test field on `MergeOutcome`, or failing test ids on `merge_verify`. Units are module legs only. Evidence for the workflow is reason text.
6. A verify slot reserved for suspects. `HostAllocator` and `free_host_count` have no per-purpose slot, and K counts hosts only.
7. A batch for the head (slot 1). 8527 plans "head chains when alone".
8. A batch base off the frontier. Chains always stack on a speculative item.
9. A per-branch footprint from the merge base in the suffix graph. `get_changed_files(main_sha, head)` is two-dot, though `GitOps.get_branch_changed_files` (three-dot) exists for reuse. Footprint failures read as an empty set.
10. Durable pool state. The halving state, reservations, bounce registry and suffix graph are all in memory. The journal lacks `merge_first_enqueued_at` (5623 owns that, behind 5045).
11. A synchronous "main is confirmed red" verdict object (7921 plans `VerdictBook`).
12. An in-lane policy-shadow mechanism (J2).
13. A `merge_pool` config block, or the `config_sections/merge_lane.py` home (9709 plans it).
14. A recorded amendment of the frozen "main advances strictly in submission order" line (H1).

## Unknowns

- Whether "submission order" in the quality PRD's frozen ordering invariant means arrival order or pick order. Today the two differ only through clique bypass and lanes; no text defines it.
- How 8526's WIP branch changes the verdict and lease flow, or `_speculation_k`. I did not read the branch.
- 8933's claim that a red-tip requeue leaves the dispatching item's speculative merge commit undead. I confirmed only that `_end_chain_reservation(all_landed=False)` records the **chain tip** dead (`DeadBaseSite.CHAIN_RELEASED`). I did not trace whether the dispatching item's own merge commit is recorded dead on that path.
- How reliably the steward fixes a `post_merge_verify` L0 in practice (no outcome data read). Likewise whether the thrash ladder would trip on repeated convictions; that depends on `_merge_outcome_signature` inputs, which I did not read.
- The exact persistence plan for 8528's suspect list. The PRD is silent, apart from p̂ seeding.
- Whether any test pins `MERGE_BOUNCE_CAP == 3` exactly. I saw only "positive int" and pre-seed-to-cap usages.
