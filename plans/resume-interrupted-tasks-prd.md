# Resume interrupted tasks with their files after an orchestrator restart

**Status: design choices ruled by Leo on 2026-11-15; not decomposed.** Drafts 1 and 2
were each attacked by a fresh critic and each was blocked (reports:
`plans/lock-handoff-study-2026-11-12/reports/round3/resume-plan-critic.md` and `resume-plan-critic-delta.md`).
This text folds in both, Leo's rulings, and a third, gating critic pass (`resume-plan-critic-3.md`, same
directory; verdict: commit after its listed text fixes, which are applied here). Section 13 lists what changed.
Written in `/prd` author mode. Not decomposed: section 9 says what that waits on.

**Code anchors** verified against main `dc2c55f6d3` (2026-11-15) by the three critics. Main moves fast: every
anchor is a symbol, not a line; re-locate at implementation time.

**Evidence.** `plans/lock-handoff-study-2026-11-12/reports/round2/restart.md` (untracked study directory).
Window: the 9 dark-factory restarts from 2026-10-26 to 2026-11-14. Figures are **measured** from lock, merge and
task events unless marked as a bound.

## 1. Terms

- **Slot.** One of the `max_concurrent_tasks` places a running task occupies (36 since 2026-11-15; 24 or 18 during the
  window measured).
- **File and lock key.** A task declares files; the scheduler locks a key per file (the path, sometimes
  shortened to a directory). A task that declares no files holds one placeholder key until its plan names them.
- **Single-holder file.** A file only one task may hold at a time (almost all files).
- **Mid-work.** Live at the stop and not waiting in the merge queue: planning, implementing, in verify or in
  review.
- **Owner stamp.** The record on a task of which orchestrator run is working on it.
- **Reservation.** Files set aside for a waiting task (today: for pinned tasks) so that nobody else takes them.
- **Start-up steps 2c and 2d.** Named steps in `orchestrator/src/orchestrator/harness.py::Harness.run`. 2c
  (`_recover_crashed_tasks`) decides which worktrees, plans and agent sessions survive. 2d
  (`_reconcile_stranded_at_startup`) settles each in-progress task that has no live owner: done if its branch is
  on main, blocked if it has an open escalation, otherwise back to `pending`. After an unclean stop that comes
  back within 10 minutes, 2d leaves the task in progress, because the dead run's owner stamp still looks fresh.
- **Clean and unclean stop.** A clean stop runs the shutdown path; an unclean one (a crash or a kill) does not.

## 2. Goal

After a restart, a task that was mid-work when the orchestrator stopped carries on where it stopped:

1. it holds the same files again from the end of start-up, before the scheduler starts anything;
2. it is started before any task that was not running at the stop;
3. it keeps its files until it is started. They are not handed to anyone else in between.

What an operator sees: at each start-up one `restart_resume_summary` event lists the tasks whose files were
restored and, for each task that was not, the reason. `get_scheduler_state` shows each restored task as the
holder of the files it held at the stop. No `lock_acquired` by another task appears on a single-holder file while
a restored task holds it.

**Not in the goal, and why.**

- *Skipping the review.* A review killed mid-flight has no verdict, so it runs again. (The reviewer's own agent
  session can be continued, and today sometimes is: 5 times since 2026-10-25, against 11 attempts that fell back
  to a fresh review. Rule: `session_resume` and `session_resume_fallback` events with role
  `reviewer_comprehensive` in `data/orchestrator/runs.db`, read 2026-11-15. Every fallback named a session that
  had not started: it was waiting for an account at the stop, so no transcript existed and no review work was
  lost. Task 4802 owns the telemetry fix; `plans/reviewer-transcripts-study-2026-11-15/REPORT.md` has the
  evidence. It is not part of this plan.)
- *Skipping the verify.* Choice 5 in section 6.
- *Capped agent accounts.* A resumed task whose agent account is capped starts and then waits, holding its
  files, exactly as it would have if there had been no restart. This plan does not change that.

## 3. Background: what a restart does today

A restart already keeps a task's branch, worktree, plan, finished steps and, usually, its agent session
(`Harness._recover_worktree_entry`). It loses two things this plan restores:

- **The files.** Held locks live only in memory (`orchestrator/src/orchestrator/lock_table.py::ModuleLockTable`).
  A clean stop cancels each workflow, which releases them; an unclean stop loses the table.
- **The place in line.** The task goes back to `pending` and competes with all other pending work.

Size of the problem, over the 9 restarts:

| | Count |
|---|---|
| Tasks live at a stop | 181 |
| Of those, waiting in the merge queue (tasks 7277, 10093 and 10216, not this plan) | 54 |
| Of those, mid-work (this plan's population) | 127 |
| Mid-work tasks started again within 1 hour | 40 of 127 |
| Mid-work tasks never started again (as of 2026-11-14) | 8 |
| Mid-work tasks for which another task took a single-holder file before they started again | 51 of 127 |
| Median wait to start again: tasks that lost a file, against the others | 28 h against 0.5 h |
| Conflict before landing: tasks that lost a file, against the others | 10 of 51 (20%) against 4 of 76 (5%) |

The conflict difference is an **association**, not an effect: the tasks that lost a file also waited longer and
were wider. This plan can promise the mechanism. It cannot promise a size for the conflict saving.

Where the mid-work tasks were at the stop: verify (including the wait for a verify slot) 69, review 24, plan 20,
execute 9, merge 3, unknown 2.

How fast tasks are started after a restart, from the last six start-ups: the 18th start came 4 to 24 minutes
after the first scheduler pass. The scheduler starts one task per pass, and the host-load hold paused starts in
up to half of the passes in the first 30 minutes.

## 4. Rulings this plan stands on

- Leo, 2026-11-12: a file lock is never released before its holder's merge lands; whoever takes the file next
  builds on the merged result.
- Leo, 2026-10-04 (task 5371 amendment): a drain does not wait for running agents. Unchanged here.
- Leo, 2026-11-14: write "resume interrupted tasks with their files" as a plan.
- Leo, `plans/one-merge-request-per-task-prd.md`, ruling R5 and decision D19: at start-up the harness launches no
  workflow; a task goes back through the scheduler. A restored hold that waits too long gets one escalation and
  **the hold stays**. This plan takes the same shape for mid-work tasks.
- Leo, 2026-11-14: the trial with the automatic file reservation off runs to about 2026-11-30. This plan does
  not restore reservations.

## 5. Sketch of approach

Three mechanisms. Each has a named consumer.

### 5.1 The resume record

A durable record with one row per task that holds locks through a workflow: task id, the lock keys held, the
time the attempt first started, a count of unclean stops since the task's last outcome, the setting that shapes
lock keys (`lock_depth`), a mark for "restored, not yet started", and a schema version.

- It is written where the lock table changes, inside `orchestrator/src/orchestrator/scheduler.py`, so that no
  path can change a task's files without changing its row: the dispatch acquisitions, plan refinement
  (`Scheduler.handle_blast_radius_expansion`: 516 acquisitions on 457 tasks since 2026-10-25, by far the
  commonest change), implement-phase widening (`Scheduler.acquire_uncontested`), and `Scheduler.release`
  including the forced release by the terminal-status watcher.
- A row is removed on every release **except** one made while the process is shutting down. The one signal is
  `Harness._draining`, set before any workflow is cancelled; the scheduler is told.
- It is a new module, `orchestrator/src/orchestrator/resume_record.py`, with its own table. It is not read back
  from `scheduler_state.json`, which is a snapshot for readers with no version.
- Rows track holds that belong to a workflow, and this plan's own restored holds. Holds taken for a merge
  request with no workflow (tasks 10093 and 10214) get no row.
- Rows are always written, whatever the switch in section 7 says. Every start-up clears the old process's rows
  (5.2), so rows cannot pile up or go stale.

Consumer: the start-up restore step (5.2).

### 5.2 The start-up restore step

The last task-tree step of start-up in `Harness.run`: after step 2d, after task 10093's restore for recovered
merge requests, before the scheduler's first pass.

The step makes no write for a task until tests 1 and 5 below have passed for it. A row's task is a
**candidate** in one of three ways:

- step 2d set it back to `pending` in this start-up (2d is changed to report which tasks it reverted);
- it is still in progress under an owner stamp from an earlier run whose process is gone (the unclean stop that
  restarts within 10 minutes). The step does not decide such a task's fate itself: it asks the same rule 2d uses
  (`orchestrator/src/orchestrator/task_ground_truth.py::TaskGroundTruth.recovery_for`) with the old stamp treated
  as dead, and the task is a candidate only if the answer is "back to `pending`". A task that rule would mark
  done or blocked is marked done or blocked, and nothing is restored;
- it is `pending` and its row carries the mark "restored, not yet started": its files were restored at an
  earlier start-up and the orchestrator stopped again before it was started. Its files are restored again. This
  never adds to its count of unclean stops, since it was not running.

The stop counts as unclean when the previous run's record has no finish written
(`orchestrator/src/orchestrator/run_store.py::RunStore.finish_run` runs in the shutdown path). A run that died
during its own start-up does not count, because no task ran in it.

For each candidate, oldest attempt first, the files are restored only if **all** of these hold:

1. it has no recovered merge request and no rejoin stamp (those belong to tasks 7277, 10093 and 10216);
2. step 2c kept something to continue: a plan with a finished step, a stamped plan, or an agent session;
3. `lock_depth`, the setting that shapes lock keys, is the same as when the row was written;
4. the row's count of unclean stops without an outcome is below the limit (choice 4);
5. it has no pending re-cut ruling (an operator's order to throw the branch away and start again);
6. every key in the row can be acquired under the holder limits the new process loaded. All or nothing.

A task that passes gets the row's keys back through the scheduler's "restore a hold at start-up, with a reason"
entry (section 11), stays `pending` and a normal candidate for starting, and joins the **resumed-first set**
(5.3). Its row is written back with the same first-start time and count and with the mark "restored, not yet
started"; the count goes up by one if the stop was unclean and the task had been running. A task that fails any test is left `pending`, holding nothing. The step then emits one
`restart_resume_summary` event naming every task with its outcome and reason, and never fails the start-up: on
any error it releases whatever it restored, clears the rows, and says so.

With the switch off the step still runs, restores nothing, and clears the rows.

Consumer: every task mid-work at a stop; the operator through the summary event and `get_scheduler_state`.

### 5.3 Resumed tasks are started first, by the scheduler, and keep their files until then

The harness launches nothing. In each pass, `Scheduler.acquire_next` looks at the resumed-first set after the
host-load hold and before pins, oldest attempt first, and starts the first member that every ordinary dispatch
check allows. The task is started the ordinary way, so it gets its owner stamp, slot and heartbeat as any task
does. A restored task is started with **exactly its row's keys**: the restored hold becomes the workflow's hold
and the mark is cleared. Any further file the task's declared list names is taken later by plan refinement, as
for any running task.

A member that cannot be started in this pass for a reason that passes by itself **waits and keeps its files**: a
host-load hold, a paused scheduler, a cooldown. The scheduler's eligibility check is extended to say why it
refused, so the two kinds of refusal can be told apart and the reason can be shown.

A member leaves the set and **all** its files are released when:

- it was started (the workflow now holds the files);
- its status is no longer `pending` (someone cancelled, deferred or blocked it), or its branch was found on main;
- **a dependency of it is no longer done** (it was re-opened while the orchestrator was down), or a delivered
  check it depends on ran and failed. Leo's ruling of 2026-11-15: such a task must lose all its locks, because
  it may be holding a file the task it depends on needs, and the two would block each other. It goes back to
  being an ordinary pending task; its branch and plan are kept for when the dependency lands again.

The dependency release needs **positive evidence**: the dependency's status was read in this pass and is not
done, or the delivered check actually ran and returned false. A status that could not be read, a check that has
not run yet, and a failed lookup of a dependency in another project all mean wait. (The scheduler's dependency
verdict reports a missing status the same way as a re-opened dependency,
`orchestrator/src/orchestrator/dependency_verdict.py`; releasing on that would drop every restored task's files
after one failed read.) A dependency in another project holds no file here, so it never causes a release; the
alert covers it.

Every pass checks **every** member for these release conditions, before the point where a pass with no
candidates ends early; the head of the set is then served after the host-load hold. Each restore emits
`lock_acquired` with the reason `restart_resume`, and each release by the set emits `lock_released` with its
reason, so the goal and the gate can be read from lock events.

Two supporting rules:

- **A reservation must not lock a restored task out of its own files.** Today a pin that is blocked by a held
  file reserves that file, and the lock table then refuses the file to everyone else, the current holder
  included (`ModuleLockTable.try_acquire`). A restored task re-acquires its own keys when it starts, so the lock
  table must skip the reservation check for keys the requester already holds. Task 10216's rejoining tasks are
  in the same position and need the same rule.
- **An alert, not a release.** A background timer of its own in the harness (no branch of the main loop runs
  while the scheduler is paused and while every slot is busy) files one escalation when restored tasks are still not started
  `restart_resume.hold_alert_secs` after start-up (default 2 hours). It names each task and the reason it is
  waiting. The holds stay. A person ends a hold by cancelling or deferring the task.

Consumer: every restored task. Task 10216's rejoining tasks can be ordered by the same set, but they are
released only by that plan's own rules (decision D19), not by the dependency release above.

## 6. Design choices (ruled by Leo on 2026-11-15: all five as recommended, with one amendment to choice 3)

**Choice 1. Which tasks are restored.** Recommended: every mid-work task that passes the tests, whatever its
phase. About a third of pending tasks declare no files until their plan names them; such a task stopped in
planning holds only its placeholder, so for it the restore buys the early start and no files.

**Choice 2. Resumed tasks go ahead of everything, pins and critical tasks included, until each has started.**
Recommended: yes. They had the slots before the stop, so this puts back what the restart took away. Without it,
a low-priority restored task would hold its files for days waiting its turn.

**Choice 3. A restored task that is not started keeps its files; an alert at 2 hours.** Recommended, and changed
from draft 2, which released the files after 30 minutes. Releasing reopens the hole the plan closes and goes
against the 2026-11-12 ruling; the second critic also showed the 30-minute rule would have fired on healthy
start-ups (the host-load hold alone paused starts in up to half the passes, and the 18th start has taken up to
24 minutes). The cost of keeping: a restored task that cannot start for a reason nobody foresaw holds its files
until a person acts on the alert. **Leo's amendment:** a restored task whose dependency was re-opened while the
orchestrator was down is the exception. It loses all its locks at once, because it might otherwise block the
very task it depends on (5.3).

**Choice 4. A task that keeps being there when the process dies.** Recommended: count only **unclean** stops
with no outcome in between (an outcome is a finished plan step, a verify result, a review verdict or a queued
merge request). At 2 the task is not restored and takes today's path. Clean stops, which are the routine
redeploys, never count, so a healthy task in a long phase is not penalised.

**Choice 5. Whether to save the verify of a task stopped after a green verify.** Recommended: do not build it.
When a kept branch is started again, `orchestrator/src/orchestrator/git_ops.py::GitOps.create_worktree` rebases
it onto main before the record of the earlier green verify is read
(`orchestrator/src/orchestrator/verify_checkpoint.py::green_checkpoint_at_tip`); the rebase changes the branch
tip, the record no longer matches, and the verify runs again. Skipping the rebase would save it, but:
- the saving is at most the 24 tasks stopped in review over 9 restarts (a bound);
- it would accept a green taken against an old main. The merge phase runs no verify of its own after its rebase
  (`pre_merge_reverify: false`, task 6580), so the first check against current main would be the merge queue's
  full verify, on the one lane everything shares.

## 7. Contract

`resume_record` (new module; names are for the architect, the rules are fixed):

- `record_held(task_id, keys)`: called by the scheduler after any change to a task's held keys. Creates the row
  (first-start time now, count zero) or updates its keys.
- `record_restored(row, *, unclean_stop: bool)`: writes an old row back with the mark "restored, not yet
  started", keeping its first-start time and count, adding one to the count if `unclean_stop`.
- `record_started(task_id)`: clears the mark when the restored task is started.
- `record_released(task_id, *, shutting_down: bool)`: removes the row unless `shutting_down`.
- `record_outcome(task_id)`: sets the count to zero.
- `take_rows_for_restart() -> list[ResumeRow]`: returns every row, oldest first, and clears the table. An
  unknown schema version returns an empty list, clears the table and logs once.

Scheduler and lock table:

- `restore_hold(task_id, keys, *, reason, all_or_nothing)`: the one start-up restore entry. Task 10093 restores
  what it can for a recovered merge request; this plan restores all or nothing. A restored hold is told apart
  from a workflow's hold and does not mark the task as started. It emits `lock_acquired` with its reason.
- a task that has a restored hold is started with exactly those keys, and the restored hold ends;
- the reservation check is skipped for keys, by exact normalized key, already in the requester's held set;
- the eligibility check can return its reason;
- the resumed-first set: membership, order by first-start time, "next member if the head is refused".

Invariants:

1. After every change to the lock table made for a task, the task's row has exactly its held keys.
2. A row is removed on every release except one made while the process is shutting down.
3. No row from the old process survives a start-up unless the step wrote it back for a restored task.
4. A restored task holds all of its keys or none.
5. A task with a recovered merge request or a rejoin stamp is never restored by this step.
6. A restored hold ends only when its task starts, stops being `pending`, lands, or has a dependency positively
   seen as not done. Time alone never ends it; neither does a refusal that passes by itself, a failed read, or
   another restart.
7. A resumed task passes every dispatch check an ordinary task passes.
8. A task is never refused its own files because of a reservation.
9. The step cannot fail the start-up.

Configuration (new keys): `restart_resume.enabled` (default false until the gate in section 9),
`restart_resume.hold_alert_secs` (7200), `restart_resume.max_unclean_stops_without_outcome` (2).

## 8. Boundary tests (through the start-up sequence and the scheduler, not the lock table's internals)

| Scenario | Before the stop | After the start-up |
|---|---|---|
| Clean stop, task implementing | holds 3 files, plan with a finished step | files restored; started before any other task; a pending task declaring one of the files is not started |
| Unclean stop, restart within 10 minutes | same; the dead run's owner stamp is still fresh | stale stamp cleared; files restored; started first |
| Plan refinement before the stop | the plan swapped one file for another | the restored files are the refined set |
| Widening in the implement phase before the stop | one more file taken | the restored files include it |
| Status changed while down | task set to `deferred` | nothing restored; summary gives the reason |
| Branch already on main | landing raced the stop | step 2d marks it done; nothing restored |
| A dependency was re-opened while down | restored, in the set | at the first scheduler pass: not started; all its files released; `pending`; an event names it; the dependency can take a file they share |
| Host-load hold after three starts | 6 restored | the other 3 keep their files and start when the hold lifts |
| A pin is blocked by a restored file | pin reserves the file | the restored task still starts; the pin gets the file after it |
| A holder limit was lowered | two live tasks share a two-holder file; new limit is 1 | the older attempt is restored; the other holds nothing |
| Scheduler paused at start-up | 2 restored | both keep their files; one alert after the limit; started on resume |
| Paused at start-up, then another restart | 2 restored, not yet started | both restored again; count unchanged |
| Unclean stop, task had an open escalation | in progress under the dead run's stamp | blocked, as step 2d would; nothing restored |
| Dependency status cannot be read in a pass | restored, in the set | keeps its files; waits |
| Declared files name one the row lacks, and a pin reserves it | restored | started with the row's keys; takes the other file later, after the pin |
| Recovered merge request, or rejoin stamp | task waiting in the merge queue | not touched by this step |
| Two unclean stops, no outcome between | count reaches 2 across the restarts | nothing restored the third time |
| An outcome between two unclean stops | a step finished | count back to zero; restored |
| Any number of clean stops | long plan phase | restored every time |
| Owner stamp | resumed task running | its owner is the new run |
| Switch off | rows present | nothing restored; rows cleared |
| Step raises | corrupt row | start-up completes; nothing held; rows cleared; one event |

## 9. Decomposition plan

Four tasks and one gate, serialised because each edits `scheduler.py` or `harness.py`.

- **T1. The resume record.** `resume_record.py` (new), the write hooks at the scheduler's lock changes
  (`scheduler.py`), the shutdown signal passed to the scheduler (`harness.py`), tests. Signal: at a quiet moment
  the rows equal what `get_scheduler_state` reports as held per task; after a clean stop and after a killed
  process the rows are still there. No prerequisite.
- **T2. Restore at start-up.** Step 2d reports the tasks it reverted; the restore step; `restore_hold` with the
  all-or-nothing policy; the three keys (`config.py`); the summary event (`event_store.py`); `OPERATIONS.md`;
  tests. Signal: the restore rows of section 8; with the switch on, a restart emits `restart_resume_summary` and
  restored tasks appear as holders. Prerequisites: T1 and **task 10216** (which itself depends on tasks 7277 and
  10093). 10216 builds the restore entry in the scheduler and the reservation rule; T2 reuses them and adds the
  all-or-nothing policy. T2 also settles how a restored pending task is told apart from a live one by the
  readers of "is this task held" (open question 3).
- **T3. Resumed first.** The set in `Scheduler.acquire_next`, the per-pass release check with its positive
  evidence rule, the start with the row's keys, the release event (`event_store.py`), tests. Signal: the ordering, host-load, pin and dependency rows of
  section 8. Prerequisite: T2.
- **T4. The alert and the count.** The harness timer and its escalation; the outcome hooks
  (`orchestrator/src/orchestrator/mcp/plan_tools.py::mark_step_done`, the verify and review results in
  `workflow.py`, the queued merge in `merge_lane/intake.py`); tests. Signal: the pause, count and outcome rows of
  section 8. Prerequisite: T3.
- **Gate. Switch it on and read two restarts.** `restart_resume.enabled` goes to true by a yaml change after the
  orchestrator has restarted on T1 to T4 (a new key goes into the yaml only once the running process knows it).
  Rows are written whatever the switch says, so the first restart after the change is already a real one. For
  each of two restarts the gate reports, from the `restart_resume_summary` events and the lock events: tasks
  mid-work at the stop; restored; not restored, by reason; minutes until the last restored task started; files
  taken from a restored task by another (expected 0); alerts filed. Leo rules keep or switch off.

## 10. Pre-conditions

- Task 7277 (pending): a recovered merge request's task is owned and not set back to `pending`.
- Task 10093 (pending): a recovered merge request's task holds its files again.
- Task 10216 (pending): the start-up restore entry in the scheduler, and the reservation rule (added to it as a
  note on 2026-11-15). T2 depends on it.

## 11. Cross-plan seams

| Other work | Direction | What crosses | Owner |
|---|---|---|---|
| Tasks 7277, 10093 | beside | tasks with a merge request in the queue; test 1 keeps the populations apart; 10093's restore runs first | those tasks |
| Task 10216 (`plans/one-merge-request-per-task-prd.md`, D18 and D19) | consumes | the restore entry and the reservation rule (in 10216 by a note of 2026-11-15) | 10216 builds them; T2 depends on it. Rejoining tasks may be ordered by the resumed-first set but keep D19's release rules |
| Task 10214, a merge submitted for a task holding no files | beside | it may take a free file during start-up before the restore step runs | accepted: whichever comes first holds the file |
| Task 5153, silent running workflows | beside | a resumed task that goes silent after it starts, a capped account included | 5153, for all tasks |
| Task 7736, a failing rebase when a branch is started again | beside | untouched unless Leo takes choice 5 | 7736 |
| Task 3824, durable fairness state | beside | reservations and skip counts across a restart | 3824 |
| Startup-hardening plan | rule | start-up steps are non-fatal | that plan; invariant 9 |

## 12. Out of scope

- Skipping verify or review, and the reviewer-session fallbacks (section 2, choice 5; task 4802).
- Reservations, skip counts and watchdog counts across a restart (task 3824; the trial runs to 2026-11-30).
- Draining running tasks to a phase boundary before a stop (ruled 2026-10-04).
- Re-attaching to a verify that was running at the stop.
- Capped agent accounts (section 2).

## 13. What changed

From draft 1 (first critic):
- The harness no longer launches workflows; it restores files and the scheduler starts the tasks.
- The record is written at every lock change, plan refinement included.
- The restore runs after step 2d.
- "Skip the verify" left the plan (choice 5).
- The shutdown signal is `Harness._draining`.

From draft 2 (second critic, and Leo's question about the 30-minute limit):
- A restored task keeps its files until it starts; time never releases them; an alert replaces the release.
  This removes the cliff the second critic measured and the false claims about capped accounts and paused
  schedulers.
- A check that fails only for now (host load, pause, cooldown) makes the task wait; it no longer loses its files.
- A reservation no longer locks a restored task out of its own files.
- The count survives a restart (draft 2 reset it every time) and counts only unclean stops.
- The step handles the unclean stop that restarts within 10 minutes, and step 2d reports what it reverted.
- The file-list comparison is replaced by "the settings that shape lock keys are unchanged", and the row's keys
  are the one source for what is restored.
- The restore entry is attributed to whichever of tasks 10093 and 10216 lands it.

From draft 3 (third critic, and Leo's rulings):
- A restored task whose dependency was re-opened loses all its locks (Leo), but only on positive evidence; a
  failed read means wait.
- The fast restart after a crash goes through step 2d's own rule, so a landed or escalated task is not restored.
- A restored task that was not started before the next restart is restored again.
- A restored task is started with exactly its row's keys.
- The unclean-stop signal is named; the restore and release emit lock events; T2 depends on task 10216.

## 14. Open questions (tactical)

1. Where the table lives: its own file, or `orchestrator/src/orchestrator/run_store.py` (also task 3824's
   file). Suggested: its own file. Decide in T1.
2. The alert's default (2 hours). Decide with Leo at the gate.
3. A restored hold makes `Scheduler.is_actively_held` true for a pending task, and the live-owner rule and the
   escalation-promotion check read that as a live owner. How the restored hold is told apart there. Decide in
   T2; task 10216's decision D19 needs the same answer.
4. Whether `lock_depth` needs checking at all: lock keys conflict by prefix across depths, so restoring keys of
   an older depth may be safe. Decide in T2.
