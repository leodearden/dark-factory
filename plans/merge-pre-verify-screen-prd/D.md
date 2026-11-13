# Seat D: dedup owners for Stream 1 (pinned main 07bd68fd99; reads 2026-10-29 ~17:50Z)

## 0. Headline
- No existing task, PRD or memory owns S1.1 (cheap whole-tree pre-verify screen), S1.2 (no verify on a base judged main-red) or S1.3 (get_changed_line_ranges full ref + fail loud). All three are new; the PRD files them fresh.
- The former-off flip has no owner either. 7739 (deferred carrier) flips only `merge_train_coalesce_enabled`. `dark-factory-orchestrator.yaml` still has `merge_train_former_enabled: true` (yaml line ~1580). Task 5375 (pending) is the closest owner: it records the former as deployed-on with 0 `train_formed` events and asks "wire or retire". The flip should be mapped onto 5375 (see section 1).
- Live evidence for S1.1 is pending escalation esc-3932-9 (L2) and esc-3932-8 (L1): task 3932 thrashed x2 at the merge gate on the INV-12 inline-suppression ratchet (`scripts/tests/test_inline_suppression_ratchet.py::test_the_live_tree_passes_the_gate_once_the_baseline_is_seeded`). It cost several full-gate verifies, and it is exactly the defect a seconds-cheap `inline_suppressions.py --check` would catch.
- A memory constraint binds S1.2: the dispatch-time verify-base guard is "advisory by decision" (tasks 3206, 3981; mem0 27c41813 and graphiti 5ebb26d4). That is about the frozen-prefix base, not main-red. The PRD must say S1.2 is a different predicate (main judged red) and must not reopen the 3206 ruling.
- The chain canary-week measures: measure 2 (items landed per chain verify) is covered by the pending 3190/7722 chain. Measures 1, 3 and 4 have gaps, listed in section 3.

## 1. Owners table

Status read with get_task on 2026-10-29. "Relation" is to S1.1/S1.2/S1.3/former-off.

| task | status | scope (1 line) | relation | recommendation |
|---|---|---|---|---|
| 6450 | pending | Coalesce dissolve/derail re-drive must not retire a train when the member status read fails (merge-deferred strand) | former-off adjacent: coalesce-only. Becomes moot once coalesce is OFF. | unrelated (note: coalesce-off makes it dormant, not closed) |
| 6566 | pending | Declared-train GroupMergeRequest whose tip is already on main: attribute and flip members | Declared trains, which are KEPT. No S1.x relation. | unrelated |
| 6755 | pending | Train verify parity with the single-branch verify (`_lane_post_merge_verify` binding, abort-poll, keep_worktrees); deps 6160 (done), 6278 | Touches the train verify path. If S1.1 or S1.2 add a pre-verify hook, it must go in `_lane_post_merge_verify` so trains inherit it. Do not duplicate the binding. | depend on / coordinate (same file, `merge_lane/worker.py`) |
| 6106 | pending | Laptop verify memory limits as a config field (`systemd-run -p`), retire drop-ins | unrelated (verify host memory) | unrelated |
| 7309 | pending | merge_cancel for trains: safe whole-train cancel and member eject-and-redrive | Declared and coalesced trains, kept. Its text is coalesce-specific (`_maybe_coalesce_waiting_singles`). | unrelated (partly dormant after coalesce-off) |
| 5057 | done (found_on_main 269c584d82 via carrier 7619) | DF deep-chain canary `chain_cap=6`. Description holds the STOP RULE: gates per landed item >1.0, or any chain-attributed red main or rebase-conflict cluster, then revert via `chain_cap: 0`. The DF window began 2026-10-27 ~23:10Z via `reload_config`. | Owns the DF chain canary decision text. Its daily read is human and manual, with no automated task. | map onto (cite as the stop-rule owner); its "daily read" has no task |
| 7619 | done | Carrier for the 5057 yaml flip | context only | unrelated |
| 3774 | done (7394e4d4be) | `_file_main_health_escalation` level and suggested_action fix | S1.2 neighbour: main-health ESCALATION filing. Does not gate verify dispatch. | unrelated (prior art to cite for S1.2) |
| 5471 | done (547cd3ca1a) | Red main of category test_failure now files a normal fix-main task (`main_health_fix_owner`) | S1.2 neighbour: gives a red main an owner after it is judged. It does not stop verifies starting on a red base. | depend on (S1.2 consumes the same main-red verdict; reuse `main_health_actions` / `main_health_fix_owner`, do not make a second classifier) |
| 7739 | deferred (planning carrier) | Lands the MCP-trains memo and flips `merge_train_coalesce_enabled: false` | Owns the coalesce-OFF flip only. Says "Stream 1 goes to /prd in a sibling session". Does NOT touch `merge_train_former_enabled`. | depend on (do not re-file the coalesce flip). The former-off flip is NOT in it. |
| 3188 (ε) | done | Chain telemetry and merge report (`chain_telemetry.py`, digest `## Deep merge-ahead chains`) | Measures 1-2 reader. See section 3. | map onto |
| 3189 (ζ) | done | Reify canary `chain_cap=6` | reify only | unrelated |
| 3190 (η1) | pending, milestone delayed 7d, anchor cleared; deps 3189, 7273, 7720, 7721, 7722 | 7-day canary predicate gate. Its before_done args pool reify AND DF stores, `--min-deep-verifies 8`. | Owns the canary gate. Covers DF only after 7722 lands. | map onto (chain canary stays here) |
| 3191 (η2) | pending, dep 3190 | Promote `chain_cap` 6 to 32. Its before_done args target `/home/leo/src/reify/dark-factory-orchestrator.yaml` port 8100. | reify only. DF's 6 to 32 promotion has no task. | gap, see section 3 |
| 3192 (θ) | pending, dep 3191, milestone 7d | Week-after assessment (halving policy, cap ceiling, retune thresholds) | Covers "chain-attributed reds" only as free-text assessment. It is reify-sequenced, behind 3191. | map onto (chain canary stays here) |
| 5059 (H) | pending, milestone 14d anchored 2026-10-27T23:47Z | 14-day DF measurement of the chain lever via `scripts/merge_lane_throughput.py --chains` (`landed_via_chain`, chain length distribution, lead time, merge_attempt mix) | DF-specific measure owner. Dated report lands about 2026-11-10. | map onto |
| 7720 | pending, high | Chain walk lands 0 links because the merger pops chained followers during the tip verify | Live defect. Measure 2 reads about 1.0 by construction until it lands (DF 6102 on 10-29). | depend on |
| 7721 | pending | Chain build latency (reify 38-53 s, 3 of 5 hit the 120 s timeout) | Feeds measure 1 (chain-build fallbacks). | depend on |
| 7722 | in-progress | Predicate repair: pooled stores, min-n, per-round denominator, same-runner p90, item-attributed failures | Measures 2 and 4 (attributed rate). | map onto |
| 7259 | pending | Carry `chain_truncated_reason` / `chain_truncated_at` on the chain-arm merge_verify event | Measure 4 and fault vs benign truncation. | map onto |
| 7730 | pending | Carry `chain_task_ids` on the chain-arm event for deep-failure attribution | Measure 4. | map onto |
| 6323 | pending | Rename `items_per_deep_verify` (mean dispatched length) vs η1's `items_per_verify` | Naming collision between two measure-2 numbers. | map onto |
| 7729 | pending | Re-transcribe or retire the orchestrator tests' copy of the items_per arithmetic after 7722 | measure 2 | unrelated to Stream 1 |
| 4767 | pending | Retire the legacy `depth` label on merge_verify | telemetry hygiene | unrelated |
| 6132 | done | build_chain un-stacks links | context | unrelated |
| 5375 | pending | Three mechanisms deployed enabled with 0 events: workflow-side train former (`merge_train_former_enabled`, 0 `train_formed` in 399k events), auto-eval dispatch, zero-progress requeue | Former-off. Closest owner. Leo's ruling turns the former OFF; 5375's former row ("wire or retire") is answered by the flip. | map onto: the PRD's former-off task should be filed as a dependent or sub-outcome of 5375, or 5375's former row amended once it lands. Do not file a duplicate. |
| 5373 | pending | Six shipped-document claims contradicted by the deployed system | Docs claim about former opt-in/off. | note: after the flip the docstring "former is opt-in, off by default" becomes true again |
| 3932 | deferred (held_by_session, resumed ruling B, Leo 2026-10-29) | Wire `_df_no_leaked_drain_processes` into 7 conftests. Its diff trips the INV-12 ratchet. | S1.1 live specimen (consumer evidence). Not an owner. | cite as evidence |
| 6997 | pending | Retire the `noqa E402/F401` suppressions in every test-root conftest's df_pytest_isolation wiring | The same suppression family as the 3932 trip. Fixes the cause class (debt), not the late detection. | related, not an owner of S1.1 |
| 5611 | pending | INV-12 integration gate (real-tree report, planted red, live dry-run, runbook) | Plants a red on `--check` and the gate. It is the nearest S1.1 task (`inline_suppressions.py --check` is INV-12's detector) but does not move the check before the merge verify. | depend on (verify before filing S1.1 that the screen reuses 5611's runbook command) |
| 7380 | pending | Main-health probe should re-run the history leg when classifying a red same-tree history leg | S1.2 neighbour (main-health classification precision) | related |
| 3597, 1690, 2590, 5627 | done | Main-health probe and baseline attribution (isolated rerun, per-SHA baseline) | The existing main-red judge that S1.2 would consume | depend on |
| 3206 / 3981 | done / deferred | Verify-base ⊄ frozen-tip guard: ruled advisory by decision; 3981 reframes INV-3 | Constraint on S1.2 | see Memory 4 |
| 1890 | done | Frozen-prefix invariant (verify only on the frozen-prefix tip) | S1.2 distinct: that is base identity, not base health | cite |

Not found (looked): no task by title or description scan for `get_changed_line_ranges` as a defect; none for "pre-verify screen" or "fail fast verify legs". The sqlite scan matched 1832 (done, 2026-06: hardening verify for suite drift), 6405 (census prescreen, unrelated), 7039 and 7573 (pre-verify disk guard, unrelated).

## 2. search_tasks queries (score >= 0.4 unless noted; hits that plausibly overlap)

search_tasks excludes deferred and uncommitted tasks. I back-filled with a read-only sqlite LIKE scan of `/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db` over pending, in-progress, blocked, deferred and merge-deferred rows.

1. "pre-verify screen / cheap check before merge verify": best hits are 1658 (0.545, done, pin verify to the merge SHA), 2828 (0.527, done), 1832 (0.504, done, harden merge-verify for suite drift without pessimising time), 2590, 1724, 2768. None is a pre-verify screen. No overlap with S1.1.
2. "inline suppression ratchet fails at merge gate": 1996 (0.551, done), 5792 (0.542, done, ratchet pre-commit audit false positive), 5722 (0.514, done), 7298 (0.511, done, ratchet ledger). All concern the MERGE-LANE ratchet (`merge_lane_ratchet_baseline.json`), not INV-12 inline suppressions. sqlite scan adds 5611, 6997, 7137, 6542, 6651, 7141, 5289 (none moves the check earlier). Name collision to flag in the PRD: "ratchet" means two things (merge-lane ratchet vs INV-12 inline-suppression baseline).
3. "fail fast verify legs / scripts leg first": 6760 (0.49, pending, S25 bootstrapping), 4320, 3350 (done; budget), 3338 (not returned, but cited by 3350 as the `&&`-chain short-circuit ordering task, done). No open task on leg ORDER or fail-fast across legs. Memory graphiti 3a93c095 and mem0 68db9a38 are generic ordering heuristics, not this.
4. "main-health red base, don't verify on red main": 1690 (0.571, done), 5627 (0.547, done), 7449 (0.536, done, auto-filed fix main), 7380 (0.533, pending), 2590, 4482 (cancelled/retracted), 3597, 6795 (pending, main-probe env), 1724, 1687, 4583. None blocks verify START on a red base. 1690 explicitly scoped itself to classification and signal only.
5. "get_changed_line_ranges / line range stackability": best hit 1878 (0.41, done, train worktree-reuse), 6219, 7577. No task on the ref-ambiguity defect. S1.3 is new. (The earlier line-range task is 1705 A′ β, done.)
6. "train former / merge_train_former_enabled": 1705 (0.528, done), 6566 (0.493, pending), 1719 (done), 1720, 1722, 1707, 7309, 6160, 6755, 6894. sqlite adds 5375, 5373, 6246, 7660, 4651, 4752. Former-off owner: 5375 only.
7. "deep chain link defect / chain halving": 6258 (0.458, pending), 7259, 7720, 6132, 7721, 7730, 4206. None of them is a per-link pre-screen. S1.1's "screen each link before the tip is built" is not owned. Closest, 7259/7730, only add event fields after the fact.
8. "thrash guard": empty at 0.4. At 0.3: 1856, 2757 (done), 1688 (done, merge-outcome thrash signature, 0.591), 2758, 3537 (done), 3003 (done). The thrash guard that tripped 3932 is 1688's `consecutive_merge_thrash`. Nothing open owns reducing wasted verifies before the guard trips.

Metadata scans: `find_tasks_by_metadata(x_finding_key)` was not run as a filter. Instead I scanned `tasks.db` metadata LIKE '%x_finding_key%' with merge/suppress description terms among non-done tasks: hits 6416, 6419, 6432, 6587, 6610, 6613, 6629, 6638, 6716, 6750, 6759. None relates to Stream 1.

## 2b. Dedup protocol and what binds a PRD from a memo
Read `docs/quality-findings-contract.md` §6 and §8 only; I did not read the rest of the contract, so the scope clause is unverified.
- §8 binds "every task filed from a finding" (instrument findings carry `x_finding_key`, `x_finding_run`, `source`). A PRD that comes from an investigation memo has no finding key. §8 steps 1 (key lookup) and 1c do not apply as written. What carries over: step 2 (`search_tasks` paraphrase at `score_threshold >= 0.6`, read plausible matches with `get_task`) and step 3 (curator `candidate_key` as the last net), plus the fact stated in §8 that search_tasks excludes deferred tasks, so a metadata or sqlite scan must back it. I ran all three (section 2).
- Also binding on a /prd decompose: the G4 cross-PRD seam ownership gate (CLAUDE.md names it), and the `docs/task-authoring.md` rules for dependencies (planning_mode, add_dependency, commit_planning) and deferred carriers.
- A deferred planning carrier such as 7739 is invisible to `search_tasks` (memory: feedback "a deferred carrier is invisible to search_tasks"). That is why 7739 and the 3932 hold had to come from get_task and sqlite.

## 3. Item 3: chain canary-week measures and their owners for THIS repo

Method: read `orchestrator/src/orchestrator/chain_telemetry.py`, `digest.py::_render_chain_telemetry_section`, `scripts/merge-deep-canary-predicate.sh`, `event_store.py` merge_finalized note, and sampled `data/orchestrator/runs.db` (mode=ro). Observed event payload keys:
- merge_verify: attempt, cache_mode, chain_build_ms, chain_items, depth, duration_ms, merge_sha, passed, plan_hash, retry_scope, retry_subset_sizes, runner, speculative, units.
- merge_finalized: absorbed_request_ids, branch, generation, landed_via_chain, merge_sha, reason, request_id, snapshot_tip, state, superseded_by.
- merge_queued: branch, queue_depth.

Live DF counts since the 10-27T23:10Z reload: 58 merge_verify rows (56 with chain_items=1, one with 2, one with 3); 2 merge_finalized rows with landed_via_chain=1. rule: all merge_verify rows with timestamp >= 2026-10-27T23:10 in `data/orchestrator/runs.db`, grouped by `chain_items`; landed_via_chain counted by LIKE '%landed_via_chain": 1%' on merge_finalized over the same window. The LIKE assumes the JSON spacing `": 1`; I did not verify it against `landed_via_chain: 2` rows.

| measure (memo §0) | covered for DF? | owner | gap |
|---|---|---|---|
| 1. share of speculative dispatches that build a chain with chain_items >= 2 | Partly. The digest histogram buckets ALL merge_verify rows by chain_items (not split by `speculative`). The predicate's only denominator is n_deep (deep verifies), so it can never express "of N speculative dispatches, k built a chain". The per-dispatch denominator exists in the data (merge_verify.speculative plus chain_items=1 rows) but no reader computes it. | 3188 (histogram), 3190/7722 (min-n only) | (a) No reader of share = deep / speculative verifies. Verified: `chain_telemetry.py` has no use of `speculative` (grepped for the chain-item and landed fields; I did not grep for `speculative` in digest.py or merge_lane_throughput.py, so say unverified there). (b) A chain build that times out and falls back to the adjacent verify (120 s `CHAIN_BUILD_TIMEOUT_SECS`) is only a syslog line (7721 cites it as a log); no event, so these fallbacks are invisible in the denominator. (c) 5059's report (--chains) covers distribution but states no "share" row. |
| 2. items landed per chain verify | Yes in principle, DF included once 7722 lands (pooled `--runs-db` x2). Digest renders `items_landed_per_deep_verify`. | 3188, 7722, 6323 (naming), 5059 | Currently ~1.0 by construction because of 7720 (chain walk lands 0 links); the DF week 10-27 to 11-03 measures the bug, not the lever. 3190's anchor will re-stamp only after 7720/7721/7722 are done, and the DF 7-day clock in 5057 already started 10-27T23:10Z. Two incompatible denominators exist until 6323/7729 land (events vs passed rounds). |
| 3. MCP singles landed via a chain (`landed_via_chain`) | No. | none | merge_finalized has no submitter or `source` key and merge_queued has only branch and queue_depth. `MergeRequest.source` ('mcp' or 'workflow', `merge_lane/types.py`) exists on the request object but is not emitted on any of the three events. Only a branch-name heuristic could split MCP from workflow, and no reader does it. Needs an event field (e.g. `source` on merge_finalized/merge_queued) or a join. No task covers it. |
| 4. chain-attributed reds | Partly. | 7259 (truncated_reason, pending), 7730 (member ids, pending), 7722 item (c) (6 h solo-fail exclusion; only knows the dispatching item), 5057 STOP RULE text | (a) 7722 attribution sees only the dispatcher (stated in 7730). (b) No metric for "red MAIN after a chain landing" (the 5057 stop rule's "chain-attributed red main"). It is a manual daily read with no task or script. (c) 3192 θ is reify-sequenced and free-text. |
| DF cap promotion 6 to 32 | n/a | 3191 args target `/home/leo/src/reify/dark-factory-orchestrator.yaml` port 8100 | No DF-side η2/θ. DF promotion would need a new deploy task. Out of Stream 1 scope; flag for the chain program. |

Leo's ruling (measures stay with η1/θ) is consistent with this table. The gaps for measures 1, 3, 4 mean η1/θ as written cannot deliver them; they need either amendments to 7722/7259/7730/3190 or new telemetry tasks owned by the deep-merge PRD, not Stream 1.

## 4. Memory (search project_id dark_factory), relevant items
- mem0 27c41813: verify-base §5.3 invariant (task 2357 finding): guard log-only; recommendation to enforce is marked [SUPERSEDED 2026-08-09]: task 3206 concluded the guard is ADVISORY BY DECISION; precision as an enforcement predicate measured 0/4; soundness is enforced at adoption by INV-3 (`_void_and_remerge`). Bears on S1.2 (do not conflate with main-red).
- graphiti 5ebb26d4 (2026-08-09): "The dispatch-time verify-base guard is advisory by decision and must not be promoted to hard enforcement." Same ruling.
- mem0 d5390b6e (2026-08-12): under `merge_verify_breadth: "full"` nothing lands over a red main; a wholly pre-existing red yields `MergeOutcome('blocked', MAIN_HEALTH_RED)`, and a red main costs per queued task up to 2 merge attempts, 2x900 s steward waits and 2 full-suite main probes before merge-thrash hard-block (parked 3644). Direct cost evidence for S1.2.
- mem0 73c5050f (2026-09-29): three of five September red-mains in DF were direct single-parent commits to main bypassing the gate (5230, 5425, 5878); the lever is "route automated commits through the merge queue, or run the repo guards pre-commit". Bears on S1.1: a screen at the merge gate does not catch these.
- mem0 256e082f / 1690 done record: main-health probe = classification and signal only; the response is a separate follow-on.
- mem0 75b6b91c (2026-08-28): `_maybe_form_train` (decision-point former) produced ZERO `train_formed` events ever; all 33 trains came from the retroactive coalescer; "either wire it up or retire it" is "a real, separate, unfiled question" (later filed as 5375). Supports former-off.
- mem0 053fab40 (2026-06-10): former fires rarely in reify (needs >=2 mutually-stackable in-progress tasks coinciding at the merge decision point).
- No memory found specifically on a "pre-merge screen" or "fail-fast verify ordering" (search returned only generic ordering heuristics 3a93c095, 68db9a38 and the xdist canonical 8bb3eb15).
- Caution, not verified: the graphiti hits are one-line edges without the source episode; I did not open the provenance episodes.

## 5. Pending escalations (mcp__escalation__get_pending_escalations, compact, read 17:5xZ)
Bearing on Stream 1:
- esc-3932-9 (L2, task_failure, root_cause `task-failure:3932-merge-thrash-inline-suppression-ratchet-duplicate-noqa-key`, updated 10-29T11:51Z) and its member esc-3932-8 (L1, `merge-outcome-thrash:test_failure:failed scripts/tests/test_inline_suppression_ratchet.py::test_the_live_tree_passes_the_gate_once_the_baseline_is_seeded`). Task 3932 shows "merge-phase thrash x2": seven duplicate `# noqa: F401` lines. This is S1.1's motivating specimen; 3932's own block_reason shows the failure arrived at the POST-merge gate after the rest of the suite passed (12136 passed, 1 failed), i.e. a ~5 min leg run after the full verify. Leo's ruling B (10-29) put 3932 on a hand-carry (held_by_session).
Not bearing: esc-7202-1 (lock-limit experiment gate), esc-7194-1, esc-7160-10/11, esc-7709-1 (config knob description, task 7709), esc-3105-3, esc-6431-2/3, esc-unverified-claim-6301-2, write-journal-size-ceiling-1/2, write-journal-insert-rate-ceiling-1. None touches the chain, the former, main-red dispatch or get_changed_line_ranges.

## 6. Unverified / not done
- I did not read plans/deep-merge-ahead-prd.capability-manifest.yaml beyond task-id and binding lines; the η1/η2/θ ids (3190, 3191, 3192) are confirmed by get_task `prd_task_label`.
- I did not check whether `merge_train_former_enabled` is in `RELOADABLE_FIELDS`. config.py declares it at `Field(default=False)` (same as `merge_train_coalesce_enabled`, which 7739 says is not in RELOADABLE_FIELDS). If both are restart-only, the former-off flip shares 7739's restart.
- `find_tasks_by_metadata` was not called; the metadata scan used read-only sqlite on `.taskmaster/tasks/tasks.db` instead.
- I did not grep digest.py or `scripts/merge_lane_throughput.py` for `speculative` (section 3 measure 1 caveat).
- The contract §8 binding analysis rests on §6 and §8 only.
