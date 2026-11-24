# Should dark-factory split its very large test files? (task 5531)

Study 2026-11-19, main at `22ffc3d192`. Read-only. Revised after a critic pass (two blocking errors in the first draft, both fixed below).
Seat reports with full evidence and the rule behind every figure: `/tmp/tlf-5531/seats/*.md`.
Yardstick: `docs/code-quality.md` heuristic 14 (an alarm is a trigger to measure, and a split must pass heuristic 13), heuristic 6 (one purpose per entity), and the Tests stance.

## Short answer

**Split them, but with their source code, not on their own, and not because of size.** The size itself costs almost nothing measurable. These files are big because the code they test is big. Six of the seven each test one oversized source module, and five of those six modules already have planned partitions. The real quality defects are heuristic 6, a test file covering many subjects, and heuristic 13, helpers that other test files import sideways. The fix for both is: when code moves into a new module, its tests move into that module's test file. The planned move tasks mostly don't do that today. The one file that needs a split of its own is `test_stages.py`, which is 17 different suites under one name.

## What the size costs (measured)

1. **Reading: nothing found.** Agents read these files in slices (median 75 to 112 lines per Read) and find their place with `grep -n`: 72% of Reads start at a line grep just printed, against 10% for random offsets. The cost per Read is the same as for 1,500-line control files.
2. **The bug that opened 5531 is gone.** The Bash "Too many AST nodes" failure hit 13 times between 2026-07-23 and 2026-10-07, and never since, although agents kept running `cat`/`sed` on these files. The 256 KB whole-file Read rejection last fired in July.
3. **Locks: almost nothing.** In 90 days, no task was ever skipped *only* because one of these test files was locked. Since the event type started (2026-11-01), one starvation event was caused by a held lock on a test file alone, on `test_merge_queue.py`; a second was a parked reservation, which a split would not remove. Most tasks that declare a test file also declare its source: 71% of 271 open declarers (65% declare the main source module).
4. **Merge conflicts: few.** Hand-resolved overlaps in 90 days: stages 6, interceptor 6 (6 of its 7 hunks in the import block, which a split moves rather than removes), harness 4, merge_queue 1, git_ops 1, others 0. Nobody records how long they took.
5. **Test run time: not a factor.** Each file is under 5% of its package's test time, and xdist spreads tests individually, so no file is pinned to one worker.

So, as with the 2026-11-17 production-file study, the case for splitting is a design case, not an operating-cost case. That matches how `landing_evidence.py` was ruled on 2026-11-17: split later, on heuristics 6 and 12, with 0 lock cost.

## Why they are big: their subjects are big

| Test file | Lines | Main subject (lines) | Planned source partition | Do those tasks move the tests? |
|---|---|---|---|---|
| orchestrator test_merge_queue.py | 23.4k | merge_lane/worker.py (19.6k), plus ~7.6k lines of unit tests for gates, types, intake, tip_relation and liveness | 5041 speculation, 5042 landing, 5306 worktrees + telemetry, 5040 verify dispatch (deferred). For gates/types: 6699, 10585, 10586, 8359, 10588. | **No.** They repoint imports or don't declare the file. 0 lines leave. |
| fused-memory test_harness.py | 22.4k | reconciliation/harness.py (6.3k) | S12 η1 9873, η2 9876, β2 9762, θ 9877 | **Yes:** S12 PRD D11 says so. Up to ~11k leaves; 11k to 16k remains. |
| fused-memory test_stages.py | 19.1k | **17 different modules** (largest: task_knowledge_sync.py, 5.5k) | S13 κ1/κ2 9744/9745 delete a dormant block (up to 5.9k); ι 9743 only repoints | Partly (by deletion). About 13k remains. |
| fused-memory test_task_interceptor.py | 16.0k | middleware/task_interceptor.py (6.9k) | S10 ν 9282, 6332 | **Yes** (9282 names the new test files). ~4.5k to 5k remains. |
| orchestrator test_git_ops.py | 15.7k | git_ops.py (17.0k) | S5: body moves ξ 9269, ο 9271, ρ 9273, plus β | **No.** The S5 codemods only repoint tests; the body moves declare no test file. 0 lines leave. |
| fused-memory test_memory_service.py | 15.2k | services/memory_service.py (11.5k) | S26 ε 9168, τ 9245, υ 9247, φ 9252 (conditional) | **Yes** (they declare new test files). ~3k to 4.3k remains. |
| orchestrator test_scheduler.py | 13.4k | scheduler.py (9.3k) | None ruled. S17 rewrites the tests in place; 7915 retires a re-export. | Not applicable. About 10k remains. |

Every file except stages has clean seams that line up with its source partition. Each seam passes heuristic 13 *only if* the shared fixtures and builders first move down into a `_`-prefixed helper module. Otherwise every new file has to import from the old one. The helpers to move are small: 18 lines for scheduler, about 70 for stages, about 170 for merge_queue, about 200 for git_ops, and 10 fixtures for interceptor.

## The rule to rule once

1. **Tests follow their subject.** A task that moves code into a new module also moves that code's tests into `test_<module>.py`, and those tests drive the new module's public API (S12 D11, made general). Driving the public API is also what lets the harness moves avoid carrying `_make_test_harness`, the private-reach builder that θ deletes.
2. **Tests of another module go to that module's test file** when the task that owns that module touches them (heuristic 6). Cases: scheduler's 299-line lock-table block (7915), memory_service's 210 lines of backend tests, and merge_queue's unit-test blocks.
3. **Shared helpers move down inside the first task that needs them**, never sideways (a test file never imports from a `test_*.py` file). This is not a sweep: 203 such imports exist in 115 files today, and the rule applies only when a task is already moving tests.
4. **No standalone "measure and partition" task for a test file on size alone.** A file that stays big because it tests one big module records its measurement on the source-split task, as heuristic 14 asks. A test file gets its own split task only when it fails heuristic 6 by itself, as stages does.
5. **If a test-file lock ever does start costing waits,** the cheaper fix is a `module_overrides` holder limit on that file, which has precedent: limit 2 on workflow.py, harness.py and config.py, and limit 99 on the ratchet JSON files. Test files are already "rebase-only" under program rule R86. It is not needed today.

Where the rule lives is your call: a sentence under heuristic 14 or the Tests stance in `docs/code-quality.md` (that section is rendered into dispatched prompts, so the edit goes through the merge queue), or a ruling recorded in the task store.

## What happens to the tasks

- **5531: cancel.** Its motivation is gone, and its deferral condition is answered by this study. Note that you deferred it until **/review-all** judged these files; this study is a different instrument, so cancelling means accepting it in review-all's place.
- **Amend the source-move tasks that don't move tests:** the S5 body moves (9269 ξ, 9271 ο, 9273 ρ, and β), *not* the simple codemods 9268/9270, which would lose their fast path. Also merge-lane 5041, 5042 and 5306, and 6699/10585/10586/8359/10588 for the gates/types unit tests now in `test_merge_queue.py`.
- **6491 (merge_queue): re-scope** to the part that waits on nothing: move the helpers into `_merge_queue_scene.py`, fold `_make_request` into `_merge_worker_drive.make_merge_request` (a SPOT duplicate today), and send the unit tests whose subject has no move planned (intake ~2.2k, tip_relation/liveness ~0.7k) to their own test files. It is dispatchable now (0 open ancestors). The lane-behaviour suite follows 5041/5042/5306. Cost of that part: about 3k lines moved, the ratchet baseline and guard pins updated in the same commit, and stale file lists on some of the 53 open declarers.
- **test_stages.py: the one real split.** Recommend a new task now, not after κ, because κ2 has 152 open ancestors. Steps: move the ~70 lines of shared helpers into `tests/reconciliation/stage_fixtures.py`, which retires the two existing sideways imports in `test_live_workflow_section.py` and `test_recon_amend_tool_advertisement.py`. Then move the suites for other modules out: consolidator ~2.9k, live-workflow ~1.3k into its existing test file, tool surface 0.6k, CLI report 0.5k, integrity check 0.6k. Leave the TKS suite and the dormant block for κ to delete. The helper move already exists as 4875's unlanded amendment `4453b0afde`; 7126 is the pending task that finishes it.
- **10745 (interceptor): drop the partition half**; keep the private-patch half for `test_ticket_worker.py`. 9282 and 6332 already move the tests, and its claim that no task partitions the file is wrong. Ordering it after 9282 would need an edge that rests only on a test file, which R86 forbids.
- **9877 (harness θ) and 9282 (interceptor):** record the remainder's measurement there.
- **Scheduler:** 7915 takes the lock-table block. The rest stays whole until a scheduler.py partition is ruled.
- **The other agent-filed test-split tasks** (about 25 real size splits once the seeder-migration and helper-hoist items are removed): one pass against rule 4, in a separate sitting.

## The honest downside

"Follow source" means these files stay big for months: θ 9877 has 150 open ancestors, κ2 152, the S5 body moves about 128, and 9282 46. Harness (11k to 16k) and scheduler (~10k) stay far over the alarm with no end date. Given the measured cost of size is close to zero, waiting is cheap. The expensive mistakes would be splitting them twice (now, then again when their source moves) or splitting them with sideways imports.

## Not checked

- Whether the declarers of each merge_queue seam are disjoint (nobody measured it).
- The time cost of hand-resolved conflicts (not recorded anywhere).
- The merge-lane-quality PRD's module-ownership table for worker.py: the seams were matched to 5041/5042/5306 by task titles and descriptions only.
- The other agent-filed split tasks were inventoried by title, not judged one by one.

## Rulings (Leo, 2026-11-19) and what was applied

- **This study stands in for /review-all's judgement of these files; 5531 closes.** It is the carrier of this commit.
- **The rule lives in `docs/code-quality.md` heuristic 14**, as three sentences at the end of its reading: a test file is sized by its subject and partitioned with the module it tests; it is split on its own only when it tests several subjects (heuristic 6); shared fixtures move down into a helper module first, and a test file never imports from another test file. Rule 5 (a `module_overrides` holder limit before a lock-motivated split) is operational, not quality, and is not in the doc.
- **test_stages.py waits for the S13 κ deletions** (9744/9745). Its split by module is recorded on κ2 9745, to be filed when κ lands, not now.
- Task amendments follow the dispositions above (6491 narrowed, 10745's partition half dropped, the S5 body moves and merge-lane 5041/5042/5306 move their tests, remainder measurements recorded on 9877 and 9282).
- Seat reports (raw evidence) were in `/tmp/tlf-5531/seats/` and are not kept.
