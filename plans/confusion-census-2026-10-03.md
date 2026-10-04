# confusion census 2026-10-03

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
- operator batch cap: 50 batch(es) (not reached -- mining stopped by: saturated)
  - batch 0: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 1: dup_rate=0.90 (total=20, succeeded=20, failed=0, saturated=True)

## Verification

- handed all 3 novel cluster(s) to the verifier; operator verify cap: 150 (not reached).

## Origin x Manifestation Matrix

| origin \ manifested | review |
| --- | --- |
| implement | 1 |

## Synthesis

**Date:** 2026-10-03
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests, per-finding verification against current main (Sonnet), then this synthesis (Fable). One finding reached synthesis. This document adds a read of the orchestrator's event ledger (`data/orchestrator/runs.db`, both tasks, 2026-09-23 to 2026-09-30), task 4880's review artifacts and plan under `.worktrees/.task-meta/4880/`, the task records for 4880, 5800 and 5618, the merge-lane stacking and un-stacking code on main, and the merge history of both branches. Every mechanism claim below names the evidence it rests on.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml`. Dispositions in §2 are inputs to the merger.
**Run notes:** fourteenth completed periodic census, after 2026-09-25 (`census-state.json` reads 2026-09-25). Previous corpora: 09-25 (1 verified finding), 09-20 (9), 09-15 (1), 09-10 (2), 09-05 (1), 08-31 (1), 08-26 (1), 08-21 (2), 08-16 (3), 08-10 (1), 08-05 (0), 07-31 (15 / 4 clusters), 07-24 (52). Saturation statistics and filed-task ids are appended by the runner outside this synthesis. The source session's transcript (`2c2152de`) is not under the main project's transcript directory, and the synthesis sandbox refused the read of the worktree-scoped directory where it would live, so no claim below rests on that transcript. The event ledger and the task's own review and plan artifacts carry the same facts with timestamps, and the quoted evidence is the review text verbatim.

### Corpus

- **1 verified finding, 1 session, 1 sighting, 1 cluster.** The finding is the blocking issue in task 4880's second comprehensive review (`.worktrees/.task-meta/4880/reviews-cycle-1/reviewer_comprehensive.json`, written 2026-09-25 19:20:29Z by an Opus `reviewer_comprehensive` of 14 turns). The verifier's evidence quote is that review's issue text.
- **The finding is already in the codebook, and the verifier presented it as novel.** `cand-20260925-19` ("Branch rebased onto wrong upstream; subsequent rebases carry unrelated task commits silently") records session `e2b6ebd8` under the same review text, stamped `implement × review`, disposition pending. Which orchestrator role each of the two sessions was is not determinable here: the ledger holds no session ids, and four invocations on task 4880 read or wrote that review text between 19:20Z and 02:14Z (reviewer, architect, implementer, second reviewer). The two sightings are two readings of one event, not two events.
- **The candidate's cause text names the wrong actor.** It says "when a developer rebases a task branch onto another task's branch". The event ledger shows the merge lane did it (§1.1). No implementer ran on task 4880 between 2026-09-23 20:03Z and 2026-09-25 19:32Z: both dispatches in that window skipped the execute phase with `no_pending_steps`.
- **Role stamps.** The verifier stamped `implement × review`. The rebase that put the foreign commits on the branch happened inside the merge phase, at train formation. This synthesis refines the origin to `merge`; the runner's matrix renders the verifier's stamp.

### Executive summary (observations)

1. **The rebase onto the sibling branch was the merge lane's train stacker, to the second.** Task 5800 entered the merge queue at 2026-09-24 06:08:13Z and task 4880 at 06:10:35Z. At 07:06:43.823Z the lane emitted `train_coalesced` for `coalesce-4880-2234ead4` with members `[5800, 4880]` and tip 4880. The reflog entry the review cites, `rebase (start): checkout task/5800` at 08:06:43 local time (UTC+1), is that same second. `GitOps.stack_train_branches` (`orchestrator/src/orchestrator/git_ops.py`) rebases each non-anchor member's own worktree onto its predecessor's branch, so task/4880's worktree was rebased onto task/5800's tip 26a7a94207. The train's verify passed at 09:53:31Z after 54.6 minutes and the advance failed on compare-and-swap at 09:53:37Z (`train_derailed`, `cas_failed`).
2. **After the derail nothing restored a main-based tip, and the next rebase onto main kept the foreign commits.** The ledger shows task 4880 re-dispatched at 2026-09-25 10:56Z (lock acquired, execute skipped, verify entered). The reused-worktree path rebases onto main (`git_ops.py` lines 4846 to 4864, `train is None` branch). Its `workflow_verify` at 19:11Z records base 78c1da55d9, where the 09-23 verify had recorded base e9dfdab3dd, so a rebase onto main did occur. Because none of the 18 commits were on main, a plain rebase replays them, which is what the review and the later plan step both describe. Verify passed green on the contaminated branch.
3. **Review caught it with the numbers the branch itself exposes.** The 09-25 review counted 24 commits in `main..HEAD`, 6 own commits touching 4 files (152+/45-) and 18 foreign commits touching 16 fused-memory files (about 4,900 lines), confirmed with `git merge-base --is-ancestor` that none were on main, and read the reflog for the cause. It also noted the copy was stale, missing task/5800's later flake-fix tip 1805651762. The merge-lane pre-merge gate `_check_plan_files_touched_in_branch` (`orchestrator/src/orchestrator/merge_lane/gates.py`) checks that declared plan files were touched; its documented direction is under-delivery, not extra files, so this branch would have passed it.
4. **Recovery was a planned step, executed in 25 minutes.** The architect re-planned at 19:32:50Z, adding `step-5` ("HISTORY REPAIR") with the procedure `git rebase --onto main <cut> task/4880`, a cut located by commit subject rather than remembered SHA, an explicit prohibition on plain `git rebase main`, and four acceptance checks including that `git diff --stat main...HEAD` lists exactly the four watchdog files. The implementer finished at 19:57:09Z with tip 4a047237. Verify passed at 2026-09-26 02:09Z; the review at 02:12Z was `suggestions_only`.
5. **The branch was stacked a second time, and that train also derailed.** On 2026-09-26 at 07:38:19Z the lane coalesced `[4807, 4880, 4978]` into `coalesce-4978-9a04ca2b`, tip 4978. That train derailed at 13:25:11Z on the tip's own rebase conflict. Task 5618's record documents this as its third sighting: task/4978 then carried 26 foreign commits, 6 of them task 4880's. Task 4880 itself landed on 2026-09-30 at 16:58Z (`d5f015e641`) with exactly its 6 own commits and 4 files, after task 5800 had landed solo on 09-26 at 11:24Z (`a11a99178f`).
6. **The defect has an owner, and the fix landed after the sighting.** Task 5618 ("Merge lane never un-stacks a train member whose predecessor leaves the queue without landing") was filed from a 2026-09-18 steward sighting on task 4137 and records three sightings (4137/4480, 5792/5708, 5876, 4978). The 09-24 train on 4880/5800 is not among them. Its fix merged on 2026-10-02 at 15:56 local (`c9ae771e40`): a stack-base ledger of git refs written at stacking time (`orchestrator/src/orchestrator/branch_stack.py`), un-stacking before a derailed member is re-driven (`merge_lane/worker.py::_unstack_redriven_member`) and at solo merge admission (`_unstack_before_merge`). The ledger is written only at stacking time, so a branch stacked before 10-02 has no record and reads `NOT_STACKED`; task 5618's own notes name task/4763 as still pending on a stacked base.

### Origin × manifestation matrix

The runner's matrix renders the verifier's stamps. The table below carries the refined stamp this synthesis establishes (§Corpus).

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| merge | · | · | · | · | 1 | · | · | · | · | **1** |
| **total** | **0** | **0** | **0** | **0** | **1** | **0** | **0** | **0** | **0** | **1** |

Readings, observational. One sighting. The origin is a merge-lane action on 2026-09-24 and the manifestation is a review verdict on 2026-09-25, about 36 hours later, with one green task-verify and one green train-verify in between. This is the first corpus since 07-24 in which the `merge` row is non-zero; the post-07-24 total is now 24 findings. The PRD's architect/implement→merge hypothesis remains untested; this sighting runs the other way, merge→review.

### 1. Verified cluster

#### 1.1 A coalesce-train stacking rebase left task/4880 carrying task 5800's 18 commits after the train derailed, and a later rebase onto main preserved them until review (1 sighting, review of 2026-09-25 19:20Z)

**The trace.** `runs.db` events for tasks 4880 and 5800, 2026-09-24 06:08Z to 09:53Z: `merge_queued` (5800, then 4880 at 06:10:35Z), `train_coalesced` at 07:06:43.823Z, both members `merge_finalized` as `superseded`, both tasks flipped to `merge-deferred` at 07:06:54Z, `train_started` at 08:58:47Z on base cf6184cfe3, `merge_verify` passed at 09:53:31Z, `merge_attempt` `advance_failed`, `train_derailed` with `cas_failed`. The reflog line the review quotes is the stacking rebase's own start record. Task 4880's events from 2026-09-25 10:56Z: `task_started`, execute skipped, verify green at 19:11Z on base 78c1da55d9, review `ISSUES_FOUND` at 19:20Z, architect at 19:32Z, implementer at 19:57Z, verify green at 02:09Z on 09-26 with tip 4a047237, review `suggestions_only` at 02:14Z.

**What the branch state looked like to each consumer.** The task-level verify saw a branch of 24 commits and passed it: the foreign code was task 5800's green work. The reviewer saw `git log main..HEAD` and the reflog and read the stacking from them. The plan-files gate on the merge path would have seen the four declared files touched. No consumer before review reads `main..HEAD` for commits that are not the task's own.

**What the sighting cost.** One review cycle, one architect re-plan, one implementer run of 25 minutes, and one more verify and review. The branch landed six days after the stacking, with its own six commits and nothing else. Task 5800's work landed under its own id four days earlier; the review's stated risk, that 5800's deliverable would land under 4880's merge before 5800's own gate, did not occur.

**Codebook state.** `cand-20260925-19` is the record for this cluster, with a cause text attributing the rebase to a developer. No promoted entry covers the train stacker leaving a derailed member on an unlanded base. The nearest entries by title are "Rebase-orphaned recorded commits misattributed to data loss" and "Verify artifacts invalidated by rebase despite identical content", which concern different rebase consequences.

### 2. Dispositions (inputs to the merger)

- **`cand-20260925-19`**: keep as the canonical record; the sighting is verified against the ledger and the review artifact. The cause text should name the actor the evidence shows: the merge lane's `stack_train_branches`, at coalesce-train formation, with the branch left stacked after `train_derailed`. "Developer rebases" is contradicted by the ledger (no implementer ran in the window). Session `2c2152de` is a second sighting of the same event, not a second event; both quote the review text.
- **Owner**: task 5618 (done 2026-10-02, `c9ae771e40`) is the fix for this mechanism. The candidate's `fix` field should point there. Its coverage boundary is observational, not a defect claim: the stack-base record is written at stacking time, so branches stacked before 10-02 carry no record.
- **The verifier's proposed remediation** (add `git log main..HEAD` and `git diff main...HEAD --name-only` checks to `skills/unblock/SKILL.md` step 5 and section 1d) targets a path this sighting did not take. Task 4880 was never blocked and no escalation or block event appears in its ledger rows; the contamination surfaced in the orchestrator's own review and was repaired by a planned implementer step. Section 1d already instructs `git diff main...HEAD --stat` as the scope read. Whether the same check belongs in the orchestrated verify or review briefing is a question for the owner of those prompts, not a finding of this census.
- **Verifier stamp `implement × review`**: this synthesis adopts `merge × review` on the ledger evidence above.

### 3. Not verified, stated for the record

- Which roles sessions `2c2152de` and `e2b6ebd8` were. Four invocations handled the review text; the transcripts were not readable from this sandbox.
- Whether any task branch is currently stacked on an unlanded base from a pre-10-02 train. Task 5618's notes name task/4763; no bulk check was run here.
- Whether the 09-26 train `coalesce-4978-9a04ca2b` re-stacked task/4880 in its own worktree and, if so, what un-did it before the 09-30 verify. The 09-30 `workflow_verify` records tip afd113ab on base ecdfbff075, which differs from the 09-26 tip 4a047237, so another rebase happened; its reflog was not read.
- Task 4880's escalation history. The escalation reader was not permitted in this sandbox; the ledger rows for the task carry no escalation-type events.


## Filed Tasks

_1 ticket(s) filed -- the curator's create/combine/drop decision is still pending, so no task id exists yet; resolve_ticket returns the task id once it does._

- tkt_0RVBKJ1RBZVY9BR4YJWTST0SY6

## Cost

invoke calls: sonnet miner=40, sonnet verify=3, fable synthesis=1, haiku headroom-probe=2
