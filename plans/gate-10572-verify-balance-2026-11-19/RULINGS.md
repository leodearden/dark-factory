# Rulings on gate task 10572: how task verifies and merge verifies share the hosts

Ruled by Leo on 2026-11-20 from `SITTING.md` in this folder (figures in `EVIDENCE.md`). This file is the
record gate task 10572 is closed against.

## The rulings

1. **What replaces "merge verify is never throttled".** Merge verifies get no cap and no share. No rule
   protects merge verifies from task verifies, and none limits them. While more requests wait than one
   chain can take, spare verify capacity goes to merge verifies. If merge verifies start with shorter
   chains than allowed because too few requests wait, capacity moves back toward task verifies; the
   one-off readout (task 8532) reports that figure.
2. **Task verifies on the laptop.** No change (`verify_task_host_policy: remote_all` stays). Leo: task
   verifies self-choke if they run far enough ahead of merges that all the task slots are waiting on the
   merge queue.
3. **Other projects' load on the two hosts.** No change. No follow-up decision task was filed.
4. **Restarts and the queue.** The task cap mostly limits merge-queue length. Two gaps make that leaky
   across a restart: a recovered merge request holds no file locks and no run slot. Both are tolerable for
   now; the file locks are the more serious. Task 10093 (restore the file locks) is pinned at position 7.
   No task was filed for the run slot.
5. **The task cap.** `max_concurrent_tasks` stays 36. No experiment.
6. **The readout (task 8532).** It reports every figure and escalates a false predicate, but recommends
   switching chains off only for a red main caused by a chain, a frozen-tip violation or an accounting
   violation. Three figures added.
7. **Docs commits.** A docs merge lane: the merge worker takes high-lane merges first, then docs merges,
   then normal-lane merges. Docs-only requests ride it; comments-only requests do not. The queue assigns
   the lane; a docs merge takes the next free merge slot. A new PRD's tasks wait for the PRD to land by a
   dependency on a carrier task. Direct docs commits continue until the migration task lands.
8. **The cheap re-verify after a docs-only move of main (task 6661).** Switched on when it lands, with no
   log-only period. Cut loose from task 9145. A verify stacked behind the re-verifying request is not
   cancelled for a docs-only move.
9. **The task built on the revoked rule (5139) and the documents.** A dated paragraph on the task; dated
   notes in the two plans that state the rule as a hard constraint.
10. **The nightly memory census.** Repaired.

## What was done

| Item | Change |
|---|---|
| Task 10093 | pinned at position 7 |
| Tasks 5296, 5411, 6661 | pinned at positions 8, 9, 10; `merge_lane: high` |
| Task 8532 | amended (ruling 6) |
| Task 5139 | amended: new reason, moved functions, two files added to its declared files |
| Tasks 3589, 3595 | one dated note each |
| Task 6661 | rewritten for on-at-landing; dependency on 9145 removed; priority high |
| Task 5411 | dated amendment: queue order, the re-verify rule, when the direct-commit route ends |
| Task 10996 (new) | the docs merge lane; depends on 5411 and 6661 |
| Task 10997 (new) | `commit_planning` stamps a PRD's capability manifest in a named worktree |
| Task 8260 | amended; now depends on 5411, 6661, 10996, 10997; the adoption gate 8258 cancelled |
| Census timer unit | reinstalled with the 20-hour landing limit (no orchestrator restart) |
| `plans/verify-oversubscription-control-prd.md`, `plans/remote-task-leg-verify-prd.md` | dated notes |

Not changed: orchestrator config; any merge in the queue; code comments that describe what the code does
(`shared/src/shared/verify_admission.py`, `verify.py`, `config.py`, the yaml) and `OPERATIONS.md`, which
stay true under ruling 1.

## Order in which the docs work lands

Task 5296 (prerequisite of the docs gate), task 5411 (the docs gate), task 6661 (cheap re-verify), task
10996 (the docs lane), then task 8260 (skills, documents and the nightly legibility job switch to the
queue). Task 10997 has no prerequisite and is needed by 8260.

## One design point left to the lane task

Task 10996 asks that a docs merge land as soon as its gate is green, even if a code request dispatched
earlier is still verifying on the other host. That lands a request ahead of one dispatched before it,
which the lane's landing-order rule does not allow today. The task is told to stop and escalate with both
options if the rule cannot be relaxed for docs alone.

## Related

A separate session is studying why merge slots are held with no request being verified, and Leo's proposal
to land a chain as soon as a later verify in it is green. Its brief:
`~/.claude/spawn-briefs/merge-slot-holds-and-chain-landing-2026-11-20.md`.
