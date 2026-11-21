I read all eight items against main `25354e0e09`. Every cited symbol was located by grep, and task text comes from read-only sqlite. None of the tasks named here has a `task/<id>` branch, and only 10215 has a finalized plan.json.

Besides the eight items, three things the draft does not account for:
- `merge_verify` already lists each verify's own members (`chain_request_ids` / `chain_task_ids`, from done task 9908). The draft's `merge_evidence.members` would be a second copy of that fact.
- Task 7824 is dispatchable now (all its dependencies are done) and breaks "tree set once at dispatch".
- The "Restart" row of 10759's contract (`IsolationEnd.shutdown`) contradicts the draft's S3 invariant (a restart keeps the same suspects).

---

## 1. "Same content" on main (task 10215, `plans/one-merge-request-per-task-prd.md`)

**How content is compared.** `merge_lane/landing_evidence.py::range_patch_ids(git_ops, *, target, tip) -> RangePatchIds`:
- It runs `git log -p --no-color --format='commit %H %P' target..tip` and pipes the output to `git patch-id --stable`.
- That gives one patch id per non-merge commit, with git's default context (3 lines, not `-U0`). The ids are sorted into a tuple, a canonical multiset that compares with `==`.
- A merge commit contributes no id, but sets `has_merge_commit=True` (detected from the parent list on its `commit` line).
- Nothing is excluded by path. In practice `.task/` never reaches a commit (it is in the root `.gitignore`, `git_ops.commit()` uses a `:!.task` pathspec, and a pre-commit hook guards it), so the draft's "`.task/` excluded" changes nothing.
- The range is `target..tip`, which the docstring says equals `merge-base(target, tip)..tip` when the merge base is unique.

**The relation.** `merge_lane/content_relation.py::classify_content(git_ops, *, target, old_tip, new_tip) -> ContentRelation` checks, cheapest first:
- `SAME`: the two shas are equal.
- `OLDER`: `new_tip` is an ancestor of `old_tip`.
- `REBASED`: neither range holds a merge commit and the two patch-id tuples are equal. Both ranges are taken against one resolved `target` sha.
- `DIFFERENT`: anything else. A git failure raises `ContentUnclassifiable` and never becomes a relation. Tips must be full 40-hex shas.

**Who consumes it.** Today, only `orchestrator/tests/test_merge_content_relation.py`. `MergeContent` is defined in `merge_lane/types.py::MergeContent` but nothing uses it yet. The planned consumers (PRD M1) are leaf C's `admission.py::decide_admission` and journal recovery, both under gate 10217.

**What 10216 and 10217 change.**
- 10216 (pending; depends on 10093, which is pending) is the lazy restart rejoin (D18/D19). It touches `recovered_merge_owner.py` and `workflow.py`, not content.
- 10217 is the decompose gate for leaves C–G. It waits on 9152, 9182, 9201, 9193 and 9144 (all pending) and 8528 (done). Those leaves bring:
  - **R2 / D11:** a later submit for a waiting request swaps its `MergeContent` under the same `request_id` and raises `content_seq`, through one writer, `MergeIntake.update_waiting`. That writer re-journals the request and clears `SuffixConflictTracker.signature`.
  - **R4:** a request merged speculatively but not being verified counts as waiting, so it is updated and re-merged.
  - **D12:** every merge product records the `content_seq` it was built from.
  - **D20:** links become one frozen record each, `ChainLink(task_id, request_id, merge_commit, content_seq)`. `ChainResult.link_request_ids` is deleted, and `merge_pool_facts.py::_chain_members` reads that field today.
  - **D14:** recovery groups journal records by `lane_task_ids`.

**What this means for a key on the branch's own changes.**
- One `request_id` can carry several contents over its life. The ledger's identity hook (the draft's `note_identity`) belongs at the D11 swap, the only writer, not at enqueue, recovery and merge attempt.
- The draft's `TreeMember(request_id, task_id, tip, merge_commit, …)` overlaps `ChainLink` field for field except `content_seq`. It should be `ChainLink` plus `own` and `content_id`, not a parallel record (heuristic 11, SPOT).

**What "the branch's own delta" is in the stacked and rebased cases.**
- `git_ops.py::GitOps.own_delta_upstream(full_branch) -> str | None` returns main, unless `branch_stack.py::StackBaseLedger` holds a base for the branch that has not landed. In that case it returns `branch_stack.py::foreign_commit_cut`: the newest commit of the contiguous foreign prefix of `main..tip`. A commit is foreign when it is an ancestor of the base, or patch-equivalent to one of its commits (`git cherry base tip main`). The function is read-only and returns None when that prefix cannot be inspected.
- `GitOps.unstack_from_unlanded_base` strips the same prefix by rewriting the branch. `GitOps.unlanded_stack_base` only reports the base.
- **Not covered by any helper:** `suffix_graph.py::SuffixConflictTracker.bounce_conflicting_suffix_items` rebases a branch onto the frozen tip, which contains unlanded in-flight merge commits, and writes no stack-base record. After that bounce:
  - `own_delta_upstream` answers main.
  - `range_patch_ids(main, tip)` carries those merge commits (`has_merge_commit`).
  - `classify_content` answers DIFFERENT.
  - The draft's `merge-base..ref` diff takes in the other entries' changes.
- The draft's "memoised by tip: for a fixed tip the merge base does not move" is false whenever the tip's ancestry holds unlanded commits that later land. The range shrinks, so the id changes while the tip does not.

**Can one function serve both purposes?** Yes, if both use the same base. I recommend a single definition, in `merge_lane/landing_evidence.py` or `content_relation.py` (owned by the 10215/10217 line):

```python
async def own_delta_base(git_ops: GitOps, *, branch: QueuedBranch, tip: str, target: str) -> str
    # target, or the foreign cut (stack ledger), or the newest merge commit on tip's
    # first-parent path within target..tip (a bounce onto an unlanded frozen tip)
async def own_delta_patch_ids(git_ops: GitOps, *, branch: QueuedBranch, tip: str, target: str) -> RangePatchIds
    # range_patch_ids(target=await own_delta_base(...), tip=tip)
```

- The content id is the sorted `patch_ids` tuple, or a hash of it. It is `None` when `has_merge_commit` is still true.
- `classify_content`'s REBASED would compare `own_delta_patch_ids` of both tips. That is a dated amendment to D6, not a second definition.
- Keep per-commit `--stable` with default context, the form that boundary row 18's fixtures pin. The draft's single `-U0` diff would be a second normalisation. If the evidence model needs to stay stable across squashes, that change goes through D6.
- `own_delta_upstream` would need a tip parameter, because it reads the live branch today.

**Seam statement:** The 10215/10217 line owns "a branch's own content" (`own_delta_patch_ids`, and the D11 swap as the only writer of content). This PRD consumes it, keyed by `(task_id, own-delta patch ids)`. Its tree record is `ChainLink` plus `own` and `content_id`. This PRD depends on a dated D6/D20 amendment, or carries `own_delta_base` itself with that PRD's owner named.

## 2. Task 7824: build the chain inside the verify task

- **Status:** pending, merge_lane high. Its dependencies 7720 and 7721 are done, and its dependents 7829 and 8791 are cancelled, so it is dispatchable now. Its `metadata.files` lists only `worker.py`, which is stale since task 5040.
- **What moves:** today `merge_lane/verify_dispatch.py::VerifyDispatcher._place` awaits `chain_planning.deep_chain_placement` (that is, `worker.py::SpeculativeMergeWorker._deep_chain_placement` → `_build_deep_chain`) with the lease held. After that, `_start_verify` builds `InflightEntry(chain=placement.chain)` and the verify task. 7824 moves the build into the verify task (`run_inflight_verify` / `InflightVerifyRun`): links are reserved when the chain is decided, the build runs off the loop, and lease and permit ownership moves to the verify's `finally`.
- **When members become known:** the `InflightEntry` then exists before its chain does. The built members (`ChainResult.links` as `(task_id, merge_commit)`, plus `link_request_ids`) are known when the build returns inside the task, before the verify runs. `merge_pool_facts.py::with_chain_members` already stamps them onto the `merge_verify` facts there (`InflightVerifyRun.__init__`).
- **After it lands:** the set landed is the prefix that `worker.py::SpeculativeMergeWorker._land_chain_prefix` walked. It is known at the end of the walk, which can be shorter than the built set.
- **Consequence for the draft:** "`InflightEntry.tree` set once at dispatch" fails. Use "published once, when the build returns, through the entry slot", the pattern of `types.py::InflightEntrySlot.publish_verify_base`. A successor can only be built on a tip that already exists, so `built_on` stays well-defined.

**Seam statement:** 7824 owns when the chain is built. α owns `InflightEntry.tree`, published once through the slot when the build returns. Either order works if α states that publication point. Add a dated note on 7824.

## 3. Tasks that edit the same symbols

- **9171** (S6 δ; pending; depends on 9165 and 9256, both pending; dependents 9176, 9207, 9725). Creates `merge_lane/main_health.py::MainHealthAutoHeal`. The worker's `_auto_heal_main_health_deferred` and the workflow's `_auto_heal_main_health` both become calls to `auto_heal.heal(...)`. `main_health_probe.py` keeps `classify_main_health_red` and `spawn_main_health_probe`. Today's `main_health_red` row is `_emit_merge_attempt(..., OutcomeKind.main_health_red)`, emitted at two sites, both of which stay in `main_health_probe.py`:
  - `classify_main_health_red`: the synchronous path, no re-check that main has not moved.
  - `_run_deferred_main_health_probe`: re-checks `probe_sha` through `_main_is_still_at`.

  For γ, hook the void at those emit sites, through a sink passed in. Do not hook it in `_auto_heal_main_health_deferred`, which 9171 moves. Then γ and 9171 share a file and nothing else.
- **9156** (S4 ξ; pending; depends on 9153, 3928, 4585 and 9219, all pending). Moves `verify_failure_is_preexisting_on_main`, `main_baseline_failing_ids` and `known_failing_ids_on_main` from `verify.py` to a new `verify_main_probe.py`, and repoints `main_health_probe.py`'s imports. If γ reads failing ids from the main baseline, it imports whichever module holds them when it lands. File contention only.
- **9182** (S6 ζ; pending; 10 dependencies, 9176/4833/3235/5245/7664/4839/9152/9256 still pending). Routes the 32 hand-placed `_retire_item`/`_resolve_or_drop_abandoned`/`_note_requeue` calls and the 21 `.set_result(` sites through `ItemLifecycle.retire(request_id, outcome | Dropped)`, with `on_terminal` hooks. `worker.py::SpeculativeMergeWorker._retire_item` today clears `_next_dispatch_unchained` and knows no exit kind. The draft's `note_gone` and 10759's `note_member_exit` both belong on an `on_terminal` hook once 9182 lands. Until then they are a call in `_retire_item`, which 9182 carries across. β2 most likely lands first.
- **9207** (S9 η; pending; depends on 9202, 4838, 9144, 9171 and 9176, all pending). Puts every merge block in one `OUTCOME_KIND_BLOCK` table keyed by `OutcomeKind`. It converts 9 sites in `TaskWorkflow._submit_to_merge_queue` and 2 in `_check_merge_outcome_thrash`, makes `MergeOutcome` feed the thrash signature, and deletes `_last_merge_block_reason`/`_failure_category`/`_failure_cause_hint`. δ's conviction arm would be a tenth hand-written site. If δ lands first, 9207 must turn it into a table row. A conviction needs an `OutcomeKind` member (9144) for 9207's table to be total.
- **9766** (S22 α; pending; depends on 9140, pending). Moves `EventType` from `event_store.py` to `shared/src/shared/run_events.py` and `OutcomeKind` to `shared/src/shared/merge_outcomes.py`, unchanged. The draft's four new event types should be declared "in `EventType` wherever it lives at dispatch", as 10759 phrases it.
- **10136** (pending, low, no dependencies, files `worker.py` and `intake_buffers.py`). `chain_snapshot` reads only the lane buffers, never `pending_redispatch`. The fix either feeds `may_chain` a "nothing parked behind" fact or changes `chain_snapshot`'s source. β2's suspect filter sits in the same `chain_snapshot` comprehension. Independent predicates, so order-free, but the second lander rebases.
- **8329** (pending; dependencies 8525 and 8528 done, so dispatchable; files `worker.py` and `gates.py`). Changes `_deep_chain_placement`: decline a build when main moved under the head (option A), or cancel the tip verify when the head enters the gate (option D). It also adds `verify_elapsed_ms` to `verdict_voided`. Option D creates verifies that are cancelled, and the draft records no row for those.
- **9133** (S7 β; pending; depends on 9127, 9255, 9140 and 9212, all pending). Deletes `ProbePlacement` and `probe_placement` (used in `VerifyDispatcher._place`, the same function α edits), the `probe_base` threading, `_record_verify_outcome` and the feed in `_finalize_inflight`, and `chain_ctx` in `_land_chain_prefix`. File contention in `verify_dispatch.py` and `worker.py`.

**Cleanest order** (a = α, records at verify return; b = β2, removes the mark; c = δ, conviction and its workflow arm):

| Task | Place relative to a, b, c |
|---|---|
| 7824 | before or after a; a states the slot publication point |
| 10624 | cancelled into b; its item 1 site is a's "record before release" |
| 9171, 9156 | either side of γ; γ hooks at the emit points |
| 9182 | after b; it carries b's `note_gone` call into a hook |
| 9144 | before c, or c gets a dated note to add a `convicted` kind |
| 9207 | after c; it absorbs c's arm as a table row |
| 9766 | free |
| 10136, 8329, 9133 | same-file only; second lander rebases; 8329's option D needs a's "a cancelled verify writes no row" restated |

**Seam statement:** This PRD owns the observation hook at verify return and the suspect predicate in `chain_snapshot`/pick. 9182 owns the terminal edge, and this PRD subscribes to it. 9144 owns the outcome kind and 9207 the block table; δ supplies one row to each. 9171 owns what happens after a red main; γ owns the void, hooked where `main_health_red` is emitted.

## 4. 10759, 10761, 10762 and `plans/merge-chain-visibility-prd.md`

**10624's full text.** Its title is "mark a red chain's links before their reservation is released; test the abandoned-drop discard; make coalescing honour the un-chained mark". Status pending, high. It depends on 10723 (done), and its only dependent is 10759.
- Item 1 (race): the links lose their reservation in the `finally` and are marked only after the awaits. On main this is still live: in `merge_lane/inflight_verify.py::InflightVerifyRun.run` the `finally` calls `_release_chain_resources`, which ends the reservation and then awaits the lane release and worktree drop, before `_settle_chain` sets the mark.
- Item 2 is moot (10723 deleted that discard).
- Item 3: `_maybe_coalesce_waiting_singles` / `_default_coalesce_exclusion_reason` ignore the mark.
- Item 4 was carried as 10723.
- Its file list (`worker.py` only) is stale: the sites are now in `inflight_verify.py`.

**What hard-codes the mark.** I grepped the PRD, both manifest files, and the task texts and `delivered_checks`.
- **Decisions:**
  - D5 ("a marked member's solo verify").
  - D6 ("the lane keeps the mark (`_next_dispatch_unchained`)").
  - D7 (isolation membership until retirement; `rechained` only after a restart or 10624's race).
  - D12 (isolation settled).
  - D19 (`entries[].unchained_mark`, "marked: verifies alone", "the live view of an isolation").
  - D8–D11 and D14–D16 do not depend on the mark.
- **Contract §8.1:**
  - `ChainIsolationSettled`; `IsolationExit.solo_verifies` ("verifies run alone while marked").
  - `IsolationEnd ∈ {members_retired, shutdown}`; `MemberExit ∈ {landed, blocked, dropped, rechained}`.
  - The I4 isolation clause; the INV-7 bound "open isolations ≤ marked members (each member in at most one)".
  - Ledger `open_isolation` ("called only beside the red-tip mark"), `note_solo_verify`, `note_member_exit`.
  - The end-per-exit row "Tip red … beside the mark; `open_isolation`" and the Isolations paragraph ("leaves the lane through `_retire_item`").
  - §6 pre-condition row (mark set and cleared); §7 row for 10624/10721.
- **Boundary rows:** H3, H4, H14, H15 (isolation `shutdown`), H19, H21 (over these), H22 ("H4 under H3"), L6 ("marked: verifies alone").
- **Manifest:**
  - β `unchained-mark` (binding: `_retire_item` clears the mark).
  - β `red-tip-and-finally-sites-settled-first` (producer: 10624).
  - γ `unchained-mark-on-the-wire` and `dashboard-reads-round-view`. These are live in 10761's `delivered_checks` as greps for `unchained_mark` in `worker.py` and in `dashboard/src/dashboard/data/`. They would fail once the key is renamed.
- **Task texts:** 10759's description (isolation definition, D6/D7 lines, contract, H3/H4/H14/H15/H19, "Do not change when the mark is set or cleared"); 10761 (D19, `entries[].unchained_mark`, L6, the two checks); 10762 (isolation definition, H22).

**Smallest coherent revision.**
- **Unchanged:** every round record (`ChainRoundStarted`, `ChainRoundSettled`, `ChainFinal`, `ChainOutcome`, `ChainStop`, `ChainFault`, `MemberFinal`), `chain_id` and `origin_chain_id`, D8–D11, D14–D16, H1–H2, H5–H13, H16–H18, H20, L1–L5, L7–L8.
- **Isolation redefined:** one per red observation (its id stays the red round's `chain_id`). A member exits when that observation no longer implicates it: cleared by a pass, explained by another member's block, or retired.
- **`MemberExit`:** add `released` (live, no longer a suspect) and `convicted` (δ). Drop `rechained`, because one request may be implicated by several open isolations. The bound becomes "open isolations ≤ non-void red observations in the ledger".
- **`IsolationEnd`:** `members_retired` becomes `resolved`. Add `voided` (γ). Drop `shutdown`: the ledger rebuilds, so a restart ends nothing (S3). That is a direct conflict with H15's isolation half as written.
- **`solo_verifies`:** count verifies run "while a suspect".
- **10761:** `unchained_mark: bool` becomes `suspect: bool`, plus the draft's `p_bad` and `implicated_by`, with the delivered-check patterns updated.
- **Edges:** 10759's dependency on 10624 becomes β2.

**Seam statement:** This PRD (β2) owns who is a suspect. 10759 records isolations as defined by it, and depends on β2 instead of 10624. 10761's wire key is `suspect` (dated amendment to D19 and its two delivered checks). Cancelling 10624 needs Leo's word, and 10759's edge must be repointed in the same step.

## 5. Task 7921 and gate 10577

- **7921** (pending; depends on 7919 and 10577, pending, plus 7918 and 8528, done; dependents 7922, 7924, 8411). It adds `merge_lane/main_red_verdict.py` with:
  - `MainRedVerdict {main_sha, failing_ids, signature, owner_esc_id, confirmed_at, generation}` and a `VerdictBook`.
  - Arming only in `_auto_heal_main_health_deferred`, after the `probe_sha` re-check, from structured `probe_sha` and F fields added to the MAIN_RED outcome built by `_build_main_health_outcome` (now in `main_health_probe.py`, not `worker.py` as 7921 says). The synchronous path arms nothing.
  - It also edits `chain_snapshot` and the pick (skip held requests) and `Harness.get_merge_halt_status`.
- **10577** (a decision gate; fires 7 days after 8532) rules build, re-scope or cancel for 7921 and 8411.
- **What the probe actually proves:** `verify_failure_is_preexisting_on_main` returns only `(bool, main_sha)` and compares a `(category, normalised cause_hint)` signature, not test ids. The ids behind "red at M" are in the main-baseline cache, `main_baseline_failing_ids` / `known_failing_ids_on_main(main_sha)`.
- **Who should carry the facts:** today's probe row, not 7921's book.
  - The probe always exists. 7921 may be cancelled.
  - 7921 itself plans to put `probe_sha` and F on the outcome built by the probe, so both designs need the probe to produce them.
  - γ adds `main_sha` and `failing_ids` (and the signature) to the `main_health_red` emit, at the deferred site only. The synchronous site is not checked for freshness.
  - If 7921 survives, `VerdictBook.arm` reads those same fields.

**Seam statement:** The main-health probe owns "main red at M on signature or ids F". γ adds the fields at the deferred `main_health_red` emit and owns the `EvidenceLedger.void` call. 7921, if 10577 keeps it, consumes the same fields and calls the same void (dated note on 7921). No dependency edge is needed either way.

## 6. Adding a block category; who consumes a blocked outcome

- **Steps to add one:**
  - Add the member to `escalation/src/escalation/categories.py::EscalationCategory`. It is `@unique`, and `require_registered_category` enforces it at `submit` and `has_open_l1`.
  - Add it to `merge_lane/outcome_facts.py::MergeBlockCategory`, whose values must be `EscalationCategory.X.value`.
- **Sites that enumerate them:**
  - `orchestrator/tests/test_merge_outcome_recording.py::TestOutcomeFacts` (parametrised over the enum).
  - `escalation/tests/test_category_census.py` (AST census of filers).
  - `merge_lane/__init__.py`'s re-export.
- **Not involved:**
  - No dashboard table keys on block category. 9852's `OUTCOME_FAMILY` keys on `OutcomeKind`, so the dashboard obligation comes only through a new kind (9144/9133 amendments). `shared/src/shared/quality_series/rows.py` filters `merge_blocked` rows on `category = 'post_merge_verify'` and is unaffected.
  - The MCP `merge_status` and `merge_request` tools (`escalation/src/escalation/server.py`) do not expose `block_category`.
- **The three consumers of a blocked outcome:**
  - `workflow.py::TaskWorkflow._submit_to_merge_queue`, the generic arm: `result.block_category.value`, then the merge-failure review, a `merge_blocked` event, and `_mark_blocked`, after the INTEGRATION_SKEW and MAIN_RED dispositions are handled.
  - `recovered_merge_owner.py::RecoveredMergeOwner._file_l1`: an L1 under `outcome.block_category.value`, which is where 10216's `REJOIN_ON` applies.
  - The thrash ladder, through `_last_merge_*` read by `_check_merge_outcome_thrash` (9207 moves this onto the outcome).

**Seam statement:** δ adds `CONVICTED` to both enums and the workflow arm before the generic arm. 9144 owns the `convicted` `OutcomeKind` member, 9852 its colour, 9207 its table row, and 10216 its `REJOIN_ON` membership. Dated notes on all four.

## 7. Task 5623 and `queue_store.py::PersistedMergeRequest`

- **5623** (pending, low; depends on 5045, pending, because of the merge-lane-quality journal freeze). It adds `merge_first_enqueued_at`, writes it in `MergeQueueStore.record`, and restores it in `reconstruct_merge_request`.
- **Fields today:** `request_id`, `task_id`, `branch` (bare id), `worktree`, `pre_rebased`, `task_files`, `snapshot_tip`, `generation`, `lane`, `enqueued_at`, `module_prefixes=None`.
- **Cost of one more optional field:** there is no journal version.
  - Evolution is "new fields last and defaulted". `load` keeps only keys in `_PERSISTED_FIELDS`, which is derived from the dataclass.
  - Unknown keys are dropped with one WARNING, so newer journals load on older binaries from 5063 on.
  - Adding a field costs a defaulted attribute, one line in `record`, one in `reconstruct_merge_request`, and a round-trip test.
- **The draft does not need it:** content ids are recomputed at recovery. 10217's leaf C replaces `snapshot_tip` with `MergeContent` and rewrites recovery (D14), which is the real collision on this file.

**Seam statement:** This PRD adds no journal field. 5623 and 10217-C own the journal schema, and the file lock serialises them.

## 8. `host_quarantine.py` and `Harness._rehydrate_merge_halt`

- **The recorded decision**, from `merge_lane/host_quarantine.py::HostQuarantine.__init__`, quoted:

  > "RESTART RE-TRUST WINDOW: an injected *quarantine* is owned by the harness for the PROCESS lifetime (task 6670 …), shared by reference with the task path's HostPool, so it survives an in-process worker rebuild; the RU tracker below stays this worker's. It still resets to empty on a process restart: a remote that diverged (remote PASS / local FAIL) becomes re-eligible then. The dedup'd L1 blocking escalation (submitted by DriftDetector before quarantine) is the cross-restart guard: an open verify_drift_divergence L1 halts the merge queue across restart via the normal EscalationQueue gate, so the re-trusted remote's verdict is never authoritative for real merges until the escalation is resolved. Rely on that coupling rather than persisting the set."

  `merge_lane/drift.py` says the same.
- **That coupling is not in the code I read.** `verify_runner.py` files `verify_drift_divergence` at level 1, severity blocking. But `_rehydrate_merge_halt` restores only `wip_conflict`/`unmerged_state`/`stash_failed`, and I found no merge-lane read of a pending `verify_drift_divergence` record. θ should verify this rather than rely on it.
- **What `_rehydrate_merge_halt` restores:** startup step `1c0`. It reads pending escalations at level ≥ 1 in `{'wip_conflict', 'unmerged_state', 'stash_failed'}`, takes the newest, and calls `halt_for_wip(reason)` and `set_halt_owner(esc.id)`. The main-health `auto_heal` per-lane hold and quarantines are not restored.
- **Other open tasks on the same restore:**
  - 9176 (S6 θ, pending) turns this into `hold(lane, HaltHolder(kind=wip))` and adds holder kinds, including `auto_heal`.
  - 4838 (pending) owns the "rehydration asymmetry".

**Seam statement:** The draft's θ consumes 9176's holder model and restores the `auto_heal` holder and quarantines through it. It depends on 9176 (and through it 4838), or narrows to an inventory plus whatever 9176 does not own.

---

## Conflicts

1. **Red-batch isolation:** 10624 (open; owns marking order and coalescing exclusion), 10759 (open; its contract records the mark as defined and forbids changing it) and this PRD's β2 (removes the mark). All three think they own when a request verifies alone.
2. **What one verify contained:** `merge_verify.chain_request_ids`/`chain_task_ids` (9908, on main), 10759's `ChainRoundStarted.members` (planned set; D15 says `merge_verify` holds the built set), and the draft's `merge_evidence.members`. That is three records of one fact. The draft should extend `merge_verify` or name it as the owner of the own members.
3. **Content identity:** 10215/10217 D6 (`classify_content` over `target..tip`, per-commit, context 3) and the draft's D2 (one diff of `merge-base..ref`, `-U0`). Two definitions of "same content", and neither handles a bounce onto the frozen tip.
4. **Chain record shape:** 10217-E's `ChainLink(task_id, request_id, merge_commit, content_seq)` and the draft's `TreeMember`.
5. **`chain_snapshot` and pick filters:** β2 (suspects), 7921 (held requests), 10136 (park source), 9935 (pool order, with its note that the filter stays after `pool_order()`). One predicate site with four authors.
6. **`_retire_item` / terminal edge:** 9182 (`ItemLifecycle.retire` + `on_terminal`), 10759 (`note_member_exit` from its callers), the draft (`note_gone`), 10723's mark clear. One hook with three subscribers.
7. **"Main red at M":** today's probe row, 7921's `VerdictBook`, and the draft's γ void. The draft names the probe as owner, and 7921 needs a dated note to use the same fields.
8. **Conviction outcome:** δ (category, field and workflow arm), 9144 (every producer states a kind), 9207 (total table over kinds; deletes the stash δ's arm would sit beside), 10216 (`REJOIN_ON` total over kinds).
9. **Chain build timing:** 7824 (build inside the verify task) and α ("tree set once at dispatch").
10. **Restart-surviving lane state:** the draft's θ, 9176 (holder set, rehydration as holders) and 4838 (rehydration asymmetry) over `halt.py` and `Harness._rehydrate_merge_halt`.

Paths I read: `/tmp/gate-10069/draft/merge-pool-batches-prd.md`, `/home/leo/src/dark-factory/plans/one-merge-request-per-task-prd.md`, `/home/leo/src/dark-factory/plans/merge-chain-visibility-prd.md` (with its `.capability-manifest.md` and `.yaml`), and under `/home/leo/src/dark-factory/orchestrator/src/orchestrator/`: `merge_lane/{content_relation,landing_evidence,inflight_verify,verify_dispatch,worker,types,outcome_facts,main_health_probe,host_quarantine,queue_store,halt}.py`, `git_ops.py`, `branch_stack.py`, `suffix_graph.py`, `merge_pool_facts.py`, `harness.py`, `verify.py`, `workflow.py`, `recovered_merge_owner.py`. Also `/home/leo/src/dark-factory/escalation/src/escalation/categories.py`.
