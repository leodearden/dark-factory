# Start a dependent early on its predecessor's branch while the predecessor waits for its merge

**Status: committed 2026-11-16 (draft 9). All seven design choices are ruled by Leo (section 6).** Written in `/prd` author mode on Leo's rulings of
2026-11-16 (section 4). Drafts 1 and 2 were each blocked by a critic; a third, gating pass on draft 3 said "commit
after the listed text fixes", and those fixes are applied here (`prd-critic.md`, `prd-critic-delta.md` and
`prd-critic-3.md` in `plans/lock-handoff-study-2026-11-12/reports/round5/`). Section 14 lists what changed. Section 6 lists seven design choices; Leo ruled all of them on 2026-11-16 (1, 2, 5
and 6 as recommended; 3, 4 and 7 changed), and a fourth critic pass (`prd-critic-4.md`) checked the changes; its fixes are applied.

**Code anchors** verified against main `6d1efef4da` (2026-11-16) by a design scout and re-read at `5f665206d8` by
the critics, last at `783d0ba683` (`design-scout.md` and the four critic reports in `plans/lock-handoff-study-2026-11-12/reports/round5/`, an
untracked study directory). Main moves fast: every anchor is a symbol, not a line; re-locate at implementation time.

**Evidence.** `plans/lock-handoff-study-2026-11-12/reports/round4/early-start-skeptic.md` (the corrected figures).
The first seat's `early-start.md` beside it has a wrong headline; a figure taken from it is marked "first seat" and
comes from a population that was partly wrong. Windows: normal running, 2026-10-25 to 2026-11-08; the current
regime, 2026-11-12T11:26Z to 2026-11-15 (3.2 days). Every gain figure is a **projection and an upper bound**.

## 1. Terms

- **Slot.** One of the `max_concurrent_tasks` places a running task occupies (36 today). In the code it is one
  permit of a counter in the harness, handed to a task as a *lease*; the scheduler does not count slots.
- **File and lock key.** A task declares files; the scheduler locks a key per file. Almost every key allows one
  holder. Two files (`workflow.py`, `harness.py`) allow two; three generated count files allow 99; documentation
  allows any number.
- **Reservation.** Files set aside for a waiting task so that nobody else takes them (today: for the first pinned
  task that is blocked on files).
- **Pinned task.** A task the operator has put ahead of the ranked order.
- **Effective tier.** A task's priority tier as the scheduler ranks it, after boosts and inheritance from the tasks
  that depend on it.
- **Host-load gate.** The scheduler's check of memory and CPU pressure on the machine; while it holds, no new
  agent work is started.
- **Predecessor.** A task whose own workflow has submitted its merge request and is waiting, in its slot and
  holding all its files, for that request to end.
- **Dependent.** A pending task that depends on the predecessor.
- **Early start.** Starting the dependent while its predecessor is still waiting for its merge.
- **Stacked base.** The commit at the tip of the predecessor's branch at the instant of the early start. The
  dependent's branch starts there instead of on main. The **stack-base ledger** is the existing durable record of
  which commit a branch was stacked on.
- **Un-stack.** The existing operation that cuts a branch's recorded base away and replays the branch's own commits
  onto main.
- **Re-stack.** Replaying a stopped dependent's own commits from its old stacked base onto its predecessor's
  current branch tip, and recording that tip as the new stacked base.
- **Co-hold.** The predecessor and its one early-started dependent both hold a key. No third task can take it.
- **Borrowed slot.** The dependent runs without a slot of its own while its predecessor waits. When the
  predecessor's slot ends, that slot becomes the dependent's.
- **Merge entry.** The point in a task's workflow, after review, where it prepares and submits its merge request.
- **Soft cancel, hard cancel.** Asking a running workflow to stop at its next safe point; cancelling it outright.
- **Owner stamp.** The record on a task of which orchestrator run is working on it.
- **Requeue.** A running task going back to `pending` to be started again later; a short cooling-down time and a
  per-task retry limit normally apply.
- **Re-cut.** The existing operation that parks a task's branch, plan and agent session and cuts a fresh branch
  from main. Until now it is issued by a human ruling or by a restart resolution.
- **Ordinary start.** How every task starts today: all dependencies done, all files free, a free slot.

## 2. Goal

A dependent whose only unfinished dependency is waiting in the merge queue starts its work at once, on top of that
predecessor's branch, instead of waiting for the landing. A plan whose tasks form a chain then finishes sooner,
because the dependent's planning, implementation, verify and review overlap the predecessor's wait.

What the operator sees when this is switched on:

1. An event `early_start_dispatched` naming the dependent, its predecessor, the stacked base and the co-held keys.
2. In the scheduler state, the dependent listed as a holder of the shared keys beside its predecessor, and listed
   as an early start with the task whose slot it borrows.
3. When the predecessor lands, an event `early_start_settled` saying how the landing relates to the stacked base
   and what the dependent did; later the dependent's own merge request, queued strictly after the landing.
4. When the predecessor's work does not land, an event `early_start_stopped` with the cause, and the dependent
   back in `pending` holding nothing, its work kept on its branch.
5. One event per early-start selection saying how many predecessors were waiting, how many dependents were
   considered, and the reason each one that was not started was refused.

**Size of the gain** (skeptic report, sections 1, 5 and 7). In the current regime about 33 dependents a week
qualify (15 in 3.2 days). For about 24 a week the predecessor landed in that pass, and each of those would have
started a median 4.8 hours earlier: about 110 dependent-hours a week (11 cases; roughly plus or minus 30%; an
upper bound, because it ignores the host-load gate and the one-dependent-per-predecessor rule). In normal running 46 of 113
ready-but-for-the-predecessor pairs qualified, 21 of them high tier. The gain goes to the dependents and their
plans; an effect on total throughput is not shown.

## 3. Background: what happens today

- A task keeps its slot and its file locks until its merge request ends
  (`orchestrator/src/orchestrator/workflow.py::TaskWorkflow._submit_to_merge_queue` awaits the request inside the
  slot; `orchestrator/src/orchestrator/harness.py::Harness._run_slot` releases through
  `orchestrator/src/orchestrator/slot_lease.py::SlotLease`). Every way a workflow leaves its slot goes through
  that lease, and on a landing the task is written `done` before the lease is released (critic, checked).
- A dependent cannot start until every dependency is `done` or `cancelled`
  (`orchestrator/src/orchestrator/dependency_verdict.py::_local_verdict`). The one exception is for declared merge
  trains, which are switched off and which release the predecessor's locks, so they are no base for this plan.
- While the predecessor waits, its task status is `in-progress` like any running task. The only in-process record
  that it is waiting for a merge is the harness's registry of live merge requests
  (`orchestrator/src/orchestrator/merge_lane/types.py::InFlightMergeRegistry`).
- When every slot is busy, which is the usual state, the harness holds one chosen task and waits for a slot. No
  scheduling pass runs during that wait. An ordinary pass is heavy: 19.9 s on average over 626 passes since
  2026-11-12T11:26Z (maximum 98 s), it begins with an 11.6 MB task fetch, and several of its steps write (they
  change statuses, mark tasks done, use up operator flags, file escalations).
- A branch can already be recorded as stacked: the stack-base ledger
  (`orchestrator/src/orchestrator/branch_stack.py::StackBaseLedger`, task 5618, done). The un-stack
  (`orchestrator/src/orchestrator/git_ops.py::GitOps.unstack_from_unlanded_base`) forgets a base that is on main,
  and otherwise cuts the base away and replays the branch's own commits onto main **whether or not the base's
  content ever reached main**. `GitOps.create_worktree` does not write a ledger record when it stacks a branch
  (task 7660, pending).
- Review, the plan-files check and the scope checks take "the task's own work" to be everything since the branch
  left main (`orchestrator/src/orchestrator/branch_base.py::own_work_base`). On a stacked branch that would include
  the predecessor's whole change. The rule that separates the two already exists for the merge queue
  (`GitOps.own_delta_upstream`).
- Nothing stops or re-checks a running task when one of its dependencies stops being done. A soft or hard cancel
  writes no task status of its own.
- The existing check runner evaluates a delivered check of the `grep` and `path` kinds at any commit; only the
  `script` kind needs a working checkout (`orchestrator/src/orchestrator/delivered_checks.py::run_delivered_check`).
  Of the 1,104 tasks that carry delivered checks, 1,095 use only `grep` and `path`, as do all 249 created since
  2026-10-25 (critic, measured 2026-11-16).
- The predecessor's merge pass fails often enough to design for: in the current regime 3 of 15 qualifying cases
  (post-merge test failures); in normal running after 2026-11-01, 5 of 30 from causes that still exist. A
  predecessor whose pass failed then landed a median 7.8 hours after the dependent would have started in the
  current regime, and 62 hours after in normal running after 2026-11-01, a figure dominated by the night main was
  red.

## 4. Rulings this plan stands on

1. **Locks stay held until the merge lands** (Leo, 2026-11-12). Whoever takes a file next builds on the merged
   result or on something that will become it. This plan lets exactly one task, the declared dependent, build on
   the predecessor's branch, and it may merge only once that branch's content is on main.
2. **The dependent runs in the predecessor's idle slot** (Leo, 2026-11-16), still subject to the host-load gate.
3. **The narrowed tier condition** (Leo, 2026-11-16: "that's what I intended"): a higher-tier task stops an early
   start only if it needs one of the dependent's files and could itself start once the predecessor lands.
4. **Stage 1 only** (Leo, 2026-11-16): the dependent enters the merge queue only after its predecessor has
   landed. The merge queue is not changed. Stage 2 is in section 12.
5. **A task must never block the task it depends on** (Leo, 2026-11-15, ruling on restored tasks). Sections 5.2
   and 5.6 apply the same principle here.
6. **Order within a tier may change.** An early-started dependent passes same-tier tasks that have waited longer
   (a median 118 to 206 runnable same-tier tasks in the two windows). This follows from ruling 3.

## 5. Sketch of approach

### 5.1 Who qualifies

A pair (predecessor, dependent) qualifies when all of these hold.

About the predecessor:
- its own workflow is waiting on a live merge request and it holds a slot;
- it has no early-started dependent running (one per predecessor at a time);
- its worktree is clean and its head is its branch tip;
- none of its delivered checks is of the `script` kind, and every delivered check it carries passes when evaluated
  at that commit (choice 1).

About the dependent:
- it is `pending`, and the predecessor is its **only** unfinished dependency; every other dependency passes the
  ordinary test, delivered checks and external dependencies included;
- it passes `Scheduler._eligible_for_dispatch` with the predecessor set aside, the same predicate the ranked and
  pinned starts use, so a gate added there later applies to early starts too;
- it carries no `milestone` (a dated or delayed milestone acts after its dependencies are done, so such a task is
  never started early);
- it is an ordinary code task. A deterministic task or a human gate is never started early: its whole point is to
  act after the predecessor has landed;
- it has no task branch yet; **or** its branch is one that an earlier early start stacked on this same
  predecessor (the branch has a ledger record and the task's early-start mark names this predecessor) (choice 3). Any other existing branch
  refuses the pair (`has_branch`).

About its files (the dependent's declared lock set, taken all or nothing): every key can be acquired under the
rule of section 5.2, and no other task's reservation covers any of them.

About the rest of the queue (ruling 3). No early start if some task T exists with all of:
- strictly higher effective tier than the dependent (a pinned task counts as higher than every tier). Leo's
  ruling of 2026-11-17, one order at every gate: an unfinished prerequisite of a pinned task counts as pinned
  (once task 9939 of `plans/priority-at-later-gates-prd.md` adds that inheritance); between two pinned-tier
  tasks the lower pin order counts as higher; and a pin set by the starvation watchdog counts as pinned here.
  Tasks 10477 and 9939 carry this as dated notes;
- T needs one of the dependent's keys;
- T's dependencies are all done, **or** the predecessor is T's only unfinished dependency;
- every key T needs is free or held only by the predecessor, and no other task's reservation covers it.
Such a T could start the moment the predecessor lands, and would then find the dependent on its file.

About the system: the switch is on; the host-load gate is not holding; the scheduler is not paused; no review
checkpoint is running (the main loop starts nothing during one); the warm-worktree base is not marked down; the
merge queue is not halted; the usage gate is not paused for every account, and the daily cost ceilings are not
breached (read without acting: the main loop's own check pauses the scheduler and files an escalation, and that
stays in the main loop); the feature has not paused
itself (section 5.7). There is no cap on the number of
live early starts (choice 4).

When several dependents of one predecessor qualify, the one the scheduler ranks highest starts.

### 5.2 Files: one rule, both directions

The lock table records "D is the successor of P" from the early start until either task releases.

- When D or P asks for a key (at the start, or when a plan widens a task's files later), the ordinary count of
  holders applies, with one exception: **on a key that allows one holder, the partner is not counted.** So D can
  share a single-holder key that P holds, and P can widen onto a single-holder key that D holds. Without the second
  half, a refused widening would send P back to pending with all its locks released: the dependent would have
  blocked the task it depends on.
- On a key that allows two holders the ordinary count applies to everyone. D can join P there only if nobody else
  holds it, and the key then has its two holders. There are never three.
- Every other task counts both P and D. A co-held key never looks free.
- Reservations are honoured exactly as today.
- One helper carries this rule for the two acquires and the two read views that explain a refusal, so they cannot
  disagree.
- When P releases, D is the sole holder with nothing to transfer.
- Two accepted residuals. If P widens onto a two-holder file that D and an unrelated task already hold, P is
  refused and goes back to pending, as it would for any two holders (two files, both under gate task 7202's
  re-measure). And the scheduler's estimate of when a blocked task's files will free assumes a key frees when its
  first holder releases; for a co-held key that is too early, which errs on the cautious side (fewer tasks slipped
  in ahead of a reservation).

### 5.3 The start

- **Base.** The stacked base is the predecessor's branch tip at the early start. The queue lands the branch head
  it finds when it picks the request up, so the pair is refused (`predecessor_branch_moved`) unless the
  predecessor's worktree is clean and its head equals its branch tip. A commit the predecessor adds later lands
  with it and is handled as "landed, changed" (section 5.6). The dependent's worktree is created at exactly that commit, and the commit is written to the stack-base
  ledger at the same moment. `GitOps.create_worktree` gains a general "stack on this commit of that branch" input
  (the train path becomes one user of it) and always writes the record. This closes task 7660. The record is read back before the launch; if
  the write or the read-back fails, the early start is abandoned and its claim released. Every reader of the
  record that guards a decision (the settle, the start-time question) tells "no record" apart from "cannot read"
  and treats "cannot read" as "unknown". The launch also writes an **early-start mark** on the dependent's task:
  which task it was stacked on. The commit itself lives only in the ledger. `early_start_dispatched` also records
  main's tip at that moment (section 5.6 uses it).
  A stacked worktree is never taken from the warm-worktree pool: the exclusion that today covers train successors
  in `create_worktree` covers the stack input too, because a released warm worktree does not keep uncommitted
  work. The stop of section 5.6 commits the dependent's work in progress before its workflow exits, so nothing in
  its worktree is left uncommitted when the worktree is released or reused.
- **A second early start: the re-stack.** A dependent that was stopped (section 5.6) keeps its branch, stacked on
  the predecessor's tip of that time. When the predecessor has repaired its work and waits on a new merge
  request, the dependent qualifies again.
  - *Trial before the claim.* In step (b) of section 5.4 the selection tries the replay without touching
    anything: `git merge-tree --write-tree --merge-base <recorded base> <predecessor's tip> <dependent's tip>`.
    If that reports a conflict, the pair is refused (`restack_conflict`). The refusal is remembered in the process
    for that pair of tips, the dependent's and the predecessor's, so the same trial is not repeated; after a
    restart it is tried once more. The dependent then waits for the landing as it does today.
  - *The replay itself.* It is not done in the harness, because the workflow's set-up runs `create_worktree`
    after the launch and would undo it. `GitOps.create_worktree` takes the same "stack on this commit of that
    branch" input as a first early start. On a branch that already exists, each of its three reuse routes (the
    registered cold worktree, the cold re-attach and the warm worktree's reuse) then replaces its rebase onto main
    with the re-stack:
    1. commit any uncommitted work (the routes' existing first step);
    2. compute the dependent's own commits with the rule `GitOps.own_delta_upstream` uses (the foreign-commit
       cut against the recorded base), not by taking the recorded base as given;
    3. replay them onto the predecessor's tip with
       `orchestrator/src/orchestrator/branch_stack.py::rebase_own_delta`;
    4. write the new base to the ledger and read it back.
  - *If the replay fails anyway* (the trial passed but one commit conflicts): it is aborted, the branch and the
    record stay as they were, the refusal is remembered as above, and the workflow exits as a requeue that does
    not count. If git refused to start because the tree was still dirty, or the abort itself failed, the
    dependent is blocked with an escalation instead of being requeued.
  - The landed question of section 5.6 is not asked here: the claim has just established that the predecessor
    is live and unlanded. The same three routes, given no stack input, un-stack before any rebase onto main; the
    registered cold worktree does this too (today it rebases the whole branch, the predecessor's commits
    included).
  - After a re-stack the dependent's workflow resumes as after any requeue, with its plan and its agent session.
    Its task verify and its review run again, because the verify checkpoint is keyed on the branch tip and the
    review verdict on the tree, and the re-stack changes both.
- **Own work.** The workflow's "own work" range goes through the one existing rule that knows about stack bases
  (`GitOps.own_delta_upstream`). Review, the plan-files check and the scope checks then see only the dependent's
  commits, before and after the predecessor lands.
- **Slot.** D's lease takes no permit and names P's lease as its lender. When P's lease is released while D still
  runs, P's permit passes to D's lease instead of going back to the pool. If D ends first, nothing is returned. D
  can lend only after it holds a permit of its own.
- **Bookkeeping.** An early start records the dependent as started exactly as an ordinary start does (the
  scheduler's set of started tasks; its tier at start; the lock event, now carrying `successor_of`; the state
  snapshot; the `in-progress` status and owner stamp written by the shared launch path). It does **not** bump the
  fairness counter for "the top-ranked task was passed over": that counter is bumped only by an ordinary start
  chosen over the ordinary ready list, and a top-ranked task that finds the dependent on a file it needs still has
  that refusal counted in later ordinary passes. Switching the automatic reservation back on (it is off by trial;
  review task 10128) needs no change here.

### 5.4 The early-start selection

It cannot be a step of the ordinary pass, which does not run while every slot is busy, and it must not re-run that
pass's writing steps. It is a **read-only selection** run by a small coroutine of its own in the harness.

- **When.** When a workflow has just submitted its merge request, and on a timer (`early_start.pass_interval_secs`),
  and only while at least one waiting predecessor has no dependent running. Otherwise it does nothing and reads
  nothing.
- **From what: the task picture.** Nothing the ordinary pass computes outlives that pass today, and the last
  ordinary pass is older than 10 minutes for 61% of the time (626 passes since 2026-11-12T11:26Z; no pass runs
  while the harness waits for a slot). So the coroutine builds its own picture and keeps it for
  `early_start.picture_max_age_secs`. A picture is built only by steps that write nothing: the task fetch; a read
  of dependency statuses that bumps no counter; the read of other projects' statuses without the policy step that
  acts on them; the delivered checks of finished dependencies without their cache and counters; the operator's
  priority overrides and pins without their clean-up; a computation of effective tiers that returns its result
  and stores nothing. Five exist today only as steps that also write (all but the task fetch), so each is split
  into a read part, which both passes use, and a writing part, which stays in the ordinary pass. The read parts
  may consult the ordinary pass's caches but never add to them.
  Measured cost of the same reads inside an ordinary pass: about 5 seconds of wall time per picture (fetch 2.3 s
  mean, 69 s worst; dependency records 1.6 s; the rest under 1.5 s), at most one picture per
  `picture_max_age_secs`. How much of that occupies the event loop was not measured; T4 measures it and, if it
  exceeds what the ordinary pass's own fetch already costs, moves the decoding off the loop.
  Locks, reservations, leases and the list of waiting predecessors are never taken from a picture: they are read
  live.
- **In what order.** (a) Shortlist pairs from the picture. (b) For the best pair, with waits allowed: one batched
  read of the live statuses of the dependent, of every task in the picture that needs one of the dependent's
  keys, and of all their dependencies; the predecessor's branch tip, with the check that its worktree is clean;
  the predecessor's delivered checks evaluated at that commit through the existing runner (it reads the commit's tree; nothing is cached or
  counted); for a dependent that has a stacked branch, the trial replay (section 5.3) and the count of its stops
  (section 5.7). (c) In one uninterrupted step, with no wait inside it: check that the predecessor's request is still
  live, apply the file rule and the tier condition against the live lock table and the statuses just read, take
  the dependent's keys, declare the successor relation and mark the dependent started. (d) Launch. If the launch
  fails, the keys, the relation and the mark are released. A re-stack that fails after the claim exits as a
  requeue that does not count (section 5.3), which releases them the ordinary way.
- **Never started twice.** The ordinary pass's own claim (take the keys, mark started) is already one
  uninterrupted step. But today it checks whether a task is already started when it builds its ready list, up to
  tens of seconds before it claims. This plan adds the same check at the claim itself, for ranked and pinned
  starts alike. With that, both claims are uninterrupted and both refuse a task already started, and no lock
  between the two is needed.
- **Residual.** A task filed after the picture was built is not seen until the next picture. A higher-tier task
  filed in that window (10 minutes at most) does not stop an early start.
- The ordinary pass never starts a task early: with the switch on, the dependency test still tells it "not
  satisfied". The per-pass release check of `plans/resume-interrupted-tasks-prd.md` stays in the ordinary pass.
- The selection event is emitted when its outcome changes and at most once per waiting predecessor per interval,
  not on every timer tick.

### 5.5 While both run

The dependent runs the ordinary workflow: plan, implementation, task verify, review. Changes to the predecessor's
branch while it is queued do not touch the dependent: it keeps working on the stacked base and settles once
(section 5.6).

The predecessor's slot is not always idle. After a `conflict` or `blocked` answer its workflow may go back to work
in its slot (the conflict-resolution agent, a retry). For that time two agents run on one permit. The number of such extra agents is at most the number of
predecessors repairing at that moment, one dependent each. The host-load gate holds new early starts while the
machine is under pressure; like every other gate it does not stop agents that already run.

### 5.6 The predecessor's fate

**One question, asked of git: has the predecessor's work landed?** It is asked at two moments: when the
predecessor's slot has ended, and at every later start of a task whose branch still has a ledger record. The
answer is one of four:

- **Landed as built on:** the stacked base is an ancestor of main (the predecessor landed as a merge commit of a
  tip that contains it), or every commit
  between main and the base has its content on main (the containment classifier of task 10215, which has no production caller before this plan,
  `orchestrator/src/orchestrator/merge_lane/landing_evidence.py::classify_containment`).
- **Landed, changed:** neither of those, but the predecessor's work reached main by a merge request submitted
  **after the early start**: a `merge_finalized` event for the predecessor with state `done` or `already_merged`
  and a time after the dependent's most recent `early_start_dispatched`, or the predecessor's done record naming a commit
  that is an ancestor of main and was not yet on main at that early start (the event records main's tip). Any of the predecessor's requests counts, not only the one live at the early
  start, because a resubmission after a conflict is a new request. A landing recorded before the early start
  never counts. An `already_merged` outcome names no commit; it counts, because the queue found the
  predecessor's branch on main. While the predecessor's slot is live its workflow's own result is read; after a
  restart, the event store and the task record. This is the case when the queue re-applied the predecessor's
  commits onto a main that had changed nearby lines, or the predecessor resolved a conflict or added a fix.
- **Not landed:** none of the above.
- **Unknown:** git failed. Ask again; never treat it as an answer. When the answer is unknown at the moment the
  predecessor's slot ends, the question is asked again every `early_start.pass_interval_secs` until it is known;
  the dependent keeps running meanwhile, and after `early_start.unknown_streak_alert` unknown answers in a row
  one escalation is raised.

Deciding this way covers what a task status would get wrong: an operator who marks the predecessor `done` while it
is queued (its merge request is cancelled and nothing lands), and a predecessor that landed but whose `done` write
was refused.

**When the predecessor's slot ends and its work has landed** (either kind), the dependent goes on working. At the
top of its merge phase it asks the landed question itself: on "unknown" it neither replays nor submits and asks
again on the timer; on "not landed" it does not submit and is left to the stop or to the start-time check; on
"landed", at the top of its merge phase (after the merge-phase stamp; before the handling of gating escalations, before the two
guards that can close a task, "already on main" and "nothing left to merge", and before the rebase onto main) it
settles:
- if its base is an ancestor of main, the ledger record is dropped;
- otherwise its own commits are replayed onto main and the record is dropped. A conflict leaves the replay
  paused at the conflicting commit (a variant of the existing replay, which aborts). The merge-conflict agent is
  invoked in the dependent's slot, under its existing attempt limit, with a brief for this case: resolve the
  paused replay and continue it; never merge main in and never rebase the whole branch onto main, either of which
  would bring the predecessor's old commits back. Afterwards the workflow checks that the branch's own work now
  starts at main and that no replay is in progress, and only then drops the record. If the agent cannot resolve
  the conflict, or the check fails, the replay is aborted and the dependent is blocked with an escalation that
  names the conflict. A dirty worktree is committed first.
- It then runs one more task verify **unless** the stacked base is an ancestor of main and the predecessor's
  landing commit is a merge commit whose second parent is the stacked base: the predecessor landed exactly what
  the dependent was built on. In every other case, including every landing re-applied onto a moved main and every
  "landed, changed", it verifies (choice 5).
The dependent never submits a merge request while its ledger record still names a base that is not on main.

If the dependent reaches its merge phase before the predecessor's slot has ended, it waits there, holding its
files and its borrowed slot. The wait has an owner (the predecessor's lease) and an end (the release of that
lease); if it lasts longer than `early_start.merge_entry_alert_secs` it raises one escalation and stays.

**When the predecessor's slot ends and its work has not landed,** the dependent is stopped with a named cause: a
soft cancel, followed by a hard cancel if it has not exited within `early_start.stop_hard_cancel_secs`. On both
arms it is written back to `pending` before its owner stamp is cleared, and it releases its files. The stop is
recorded as a requeue that does not count toward the task's retry limit or the no-progress alert, and it emits
one `early_start_stopped` event. The dependent's uncommitted work is
committed before its workflow exits (section 5.3). Its work stays on its branch with the ledger record. Reason for stopping rather than waiting (choice 2): after a failure the
predecessor needs its files and its slot for the repair, and a dependent that kept them would block the task it
depends on (ruling 5). A stop caused by the orchestrator shutting down is not a stop in this sense: nothing is
emitted or counted.

**At every later ordinary start of the stopped dependent** (once its dependencies are satisfied) the same question
is asked before anything touches its branch. A second early start on the same predecessor does not ask it and
re-stacks instead (section 5.3).

- **landed** (either kind): its commits are replayed onto main by the existing requeue path, and its workflow
  resumes; the task verify runs again on the new tree before it can submit. If that replay conflicts, the
  workflow continues on the old base with the record in place, as today, and the settle at the top of its merge
  phase applies the conflict handling above. The settle runs for every task whose branch has a ledger record,
  not only during an early start, except a declared-train successor, whose stack is kept as it is today;
- **not landed** (the predecessor was cancelled, or marked done without a landing, at the stop or at any time
  after it): the dependent's plan and code were written on top of work that will never land. An automatic re-cut
  is issued: the existing re-cut mechanism (`orchestrator/src/orchestrator/recut_ruling.py`, until now issued by
  a human or by a restart resolution) parks the branch, the plan and the agent session and cuts a fresh branch from main. In the same step, before
  the fresh branch is cut, the dependent's ledger record and its early-start mark are dropped, so the fresh
  branch carries neither and no later start asks the question again. (The record is kept under the branch's
  name, so without this it would sit on the fresh branch and every later start would park the fresh work.) A
  start that finds an executed early-start re-cut naming the same predecessor and base never issues another. The
  ruling names its issuer ("early start: predecessor's work never landed") and the predecessor;
- **unknown:** the dependent goes back to `pending` at once, holding nothing, as a requeue that does not count
  and that arms the ordinary cooling-down time. After `early_start.unknown_streak_alert` such answers in a row for
  one task (default 3), one escalation is raised.
The decision is taken at the start, not at the stop, because the predecessor's fate can change in between: its
pass fails, the dependent is stopped, and a human then cancels the predecessor instead of repairing it.

### 5.7 Bounds and self-pause

- One early start per predecessor at a time, and no cap on how many are live (choice 4). Each borrows the slot of
  a task that is waiting, so tasks running never exceed `max_concurrent_tasks` plus the live early starts, and
  live early starts never exceed the number of predecessors. The host-load gate holds new ones under load.
- No bound on how often one dependent is stopped on one predecessor (ruled by Leo, 2026-11-17; drafts 5 to 9
  had a bound of three). Every stop follows a merge pass of the predecessor that did not land, and the
  predecessor comes back only through one of its own requeue limits or through a resumed escalation, so those
  bound how often a pair can cycle (fourth critic). The self-pause below catches a run of stops across pairs.
- If `early_start.stop_streak_pause` early starts in a row end in a stop (default 5), the feature pauses itself
  and raises one escalation. It resumes when the operator switches `early_start.enabled` off and on again, or at a
  restart.
- A restart turns every early start into an ordinary wait (choice 6).

## 6. Design choices (all ruled by Leo, 2026-11-16)

Ruled as recommended: 1, 2, 5, 6. Ruled differently from the draft's recommendation: 3, 4, 7.

1. **A predecessor that carries delivered checks.** They are normally evaluated on main and so fail until the
   predecessor lands. *Recommended:* evaluate them at the predecessor's branch tip, which the existing check runner
   can do for the `grep` and `path` kinds, and refuse only a pair whose predecessor has a `script` check (9 of
   1,104 tasks with checks; none created since 2026-10-25). *Alternative (draft 1):* refuse every such pair, which
   the first seat's count suggests would give up a third to two thirds of the gain.
2. **The predecessor's work does not land in this pass.** *Recommended:* the dependent stops at once and goes back
   to pending, work kept (section 5.6). *Alternative:* it keeps waiting with its files and slot. Rejected in the
   recommendation because it can block the predecessor's own repair, and because failed predecessors landed a median
   7.8 hours after the dependent would have started in the current regime (62 hours in normal running after
   2026-11-01, dominated by one night).
   Whether the stopped dependent's work is later replayed onto main or cut afresh is decided at its next start,
   from git: replayed if the predecessor's work landed, cut afresh if it never will.
3. **A second early start.** *Ruled by Leo: allowed.* A dependent stopped by a failed predecessor is started
   early again when the predecessor has repaired its work and resubmitted: its own commits are replayed onto the
   predecessor's new tip (the re-stack, section 5.3), as Leo's proposal said ("the dependent will have to rebase
   on the new tip"). A replay that conflicts is not an early start; the dependent waits for the landing. A branch
   that was not stacked on this predecessor by an earlier early start still refuses the pair. (The draft had
   recommended "at most once", to keep a single way of creating a stacked branch.)
4. **Caps.** *Ruled by Leo: no cap on the number of live early starts.* One dependent per predecessor at a time
   stays, because the file sharing and the borrowed slot are between two tasks. The self-pause after 5 stops in
   a row stays. There is no bound on stops per pair (ruled by Leo, 2026-11-17; section 5.7).
   (The draft had recommended a cap of 4 live early starts.)
5. **After the landing.** *Recommended:* one more task verify unless the predecessor landed exactly what the
   dependent was built on (a merge commit of exactly the stacked base), and always after a conflict was
   resolved. No second review. The reason: the dependent was tested against the predecessor's content as it was
   before landing. The merge queue's own verify covers every landing in full. A landing re-applied onto a moved main always costs
   the extra verify, even with the same changes, because what was re-applied is not recorded; how often that
   happens is not known and the dated read reports it.
6. **Restart.** *Recommended:* an early start does not survive a restart as such. The dependent goes back to
   pending with its branch and ledger record and waits for the landing. The committed plan
   `plans/resume-interrupted-tasks-prd.md` already ends in that state (it would restore the dependent's files, or
   fail to, and then release them at its first pass because a dependency is not done). No edit to that plan in
   this commit; a note on its decompose gate (task 10301) asks for one refinement there: do not restore a task
   that has a dependency that is not done, which saves the restore-then-release.
7. **Roll-out.** *Ruled by Leo: enabled by default.* `early_start.enabled` defaults to on, for every project the
   factory operates; a project opts out in its yaml. The build lands in pieces, and an early start must never run
   on an orchestrator that has the launch but not yet the stop, the settle and the start-time question. So the
   default stays off in the code until the last build task (T7), which turns it on. No yaml edit and no human gate
   stand between T7 and live running. A gate two days after T7 lands reads the first real pairs; a second, two
   weeks after, compares the counts with section 2 and recommends on stage 2. (The draft had recommended shipping
   switched off behind a human gate.)

## 7. Contract

Names are proposals; the shapes and the invariants bind.

**Lock table** (`orchestrator/src/orchestrator/lock_table.py::ModuleLockTable`)
- Declaring a successor relation, and removing it on the release of either task.
- One private helper answers "how many holders count against this requester on this key". The start-time acquire,
  the plan-time widening, the in-place widening and the two read views that name blockers all use it. The
  exception of section 5.2 lives in that helper and nowhere else; the reservation paths keep using the plain count.

**Dependency test** (`orchestrator/src/orchestrator/dependency_verdict.py::deps_verdict`)
- A new, opt-in input: the tasks that are waiting predecessors. With it, a task whose only unfinished dependency is
  in the set gets an answer that is **not** "satisfied", names the predecessor, and is given only if every other
  part of the test (external dependencies, scheduled deploys, delivered checks of the finished dependencies)
  passes with the predecessor set aside. Callers that do not pass the input (ordinary starts, pinned starts, the
  clean-up of reservations) get today's answers bit for bit.

**Worktree and git** (`orchestrator/src/orchestrator/git_ops.py::GitOps`)
- `create_worktree` takes "stack on this commit of that branch": the new branch starts at that commit, the ledger
  record is written before the call returns, and the stacked base is returned. The train path uses the same input.
- `TaskWorkflow._own_work_base` answers through `GitOps.own_delta_upstream` when the branch has a ledger record.
- One function answers "has the work under this ledger record landed": landed as built on / landed, changed /
  not landed / unknown, as section 5.6 defines them. It takes the predecessor's recorded landings as an input;
  its callers (the workflow at the settle, the harness at a start) gather them. Both call the same function.
- A re-stack mode of `create_worktree`'s three reuse routes: given the stack input on an existing branch, commit
  the work in progress, cut the branch's own commits with the rule of `own_delta_upstream`, replay them onto the
  named commit, write and read back the record; on a conflict abort and leave branch and record untouched; tell a
  conflict, a refused start and a failed abort apart. A trial of the same replay that touches nothing
  (`merge-tree --write-tree --merge-base`). Given no stack input, all three routes un-stack before a rebase onto
  main.
- A stacked worktree is never taken from the warm-worktree pool.
- A replay entry point: replay a branch's own commits onto main, pause at a conflict without aborting, let the
  caller resolve and continue, and drop the record only when the replay is complete and checked.

**Scheduler and harness**
- The read parts of five pass steps, callable without their writing parts (section 5.4), and a task picture built
  from them.
- The ordinary claim, ranked and pinned, refuses a task that is already started.
- An early-start selection entry on the scheduler that takes a task picture, the statuses re-read in step (b) and
  the live list of waiting predecessors, and returns at most one claim (dependent, predecessor, keys, stacked
  commit), with the keys taken, the relation declared and the dependent marked started; or nothing, with the
  refusal reasons.
- The harness's early-start coroutine (section 5.4) and its launch of a claim with a borrowing lease.
- `SlotLease`: a borrower takes no permit; the lender's release hands its permit to a still-running borrower; every
  permit is released exactly once, on every exit, in either order.
- A per-task "slot ended" signal, set at the release of a lease, that the dependent's workflow and the harness's
  stop logic can await.
- The early-start mark is a new task metadata key, named once and registered in
  `shared/src/shared/task_metadata.py`; the selection reads it, the launch writes it, the automatic re-cut
  clears it.
- At the ordinary start of any task whose branch has a ledger record and whose metadata names an early-start
  predecessor (an early start's own launch re-stacks instead, section 5.3):
  the landed question, placed in `Harness._run_slot` before the existing re-cut gate, which is before anything
  touches the worktree. On "not landed", an automatic re-cut: persist a re-cut ruling whose id is built from the
  dependent, the predecessor and the base (so it cannot be issued twice) through the existing
  `recut_ruling.issue_ruling`, then make the gate see it in this same start (re-read the task, or apply the patch
  to the claim's copy); if the write fails, block the dependent with an escalation, as a refused re-cut is today.
  The re-cut drops the ledger record and the early-start mark. `recut_ruling.py`'s description of a ruling as
  human-issued is changed to name the automatic issuer. On "unknown", a requeue that does not count.

**Workflow** (`orchestrator/src/orchestrator/workflow.py::TaskWorkflow`)
- "Settle the stacked base" in `_run_merge_phase`, after the merge-phase stamp and before the handling of gating
  escalations, before the two guards that can close a task ("already on main" and "nothing left to merge"), and
  before `_rebase_before_enqueue`. It runs for every task whose branch has a ledger record. Merge retries stay inside the loop below it and never pass it again.
- A stop with cause "predecessor's work did not land" that exits through the existing re-pend path
  (`_repend_for_requeue`) as a requeue that does not count, and whose hard-cancel arm writes `pending` before the
  owner stamp is cleared. The stop commits the dependent's work in progress before the workflow exits.

**Config** (`orchestrator/src/orchestrator/config.py`): a group `early_start`, hot-reloadable:
`enabled` (off in the code until T7 turns the default on), `pass_interval_secs` (60), `picture_max_age_secs` (600),
`merge_entry_alert_secs` (43200), `stop_streak_pause` (5), `stop_hard_cancel_secs` (300),
`unknown_streak_alert` (3).

`enabled` gates new early starts only. Switching it off, by reload or by yaml, never touches a live pair: the
stop, the settle, the merge-entry wait and the start-time question run for every live early start and every
branch with a ledger record whatever the switch says.

The `early_start` defaults live in the config model only, not in `defaults.yaml`; T1's test of the default uses
the `code_default_config` fixture, and T7 changes that one value and that one test.

**Events** (new members of the event vocabulary; one writer each)
- `early_start_selection`: waiting predecessors, dependents considered, started, and a count per refusal reason
  (`delivered_check_script`, `delivered_check_failed`, `not_only_dependency`, `has_branch`, `not_code_task`,
  `has_milestone`, `ledger_write_failed`,
  `predecessor_branch_moved`, `key_unavailable`, `key_reserved`, `higher_tier_waiting` with the task that refused
  it, `host_load`, `scheduler_paused`, `review_checkpoint`, `warm_base_down`, `merge_queue_halted`, `usage_paused`, `cost_ceiling`, `predecessor_has_dependent`, `restack_conflict`,
  `self_paused`). Emitted on a change of outcome, at most once per waiting predecessor per interval.
- `early_start_dispatched`: dependent, predecessor, stacked base, main's tip at that moment, co-held keys, own
  keys, and for a re-stack the base it replaced.
- `early_start_settled`: dependent, predecessor, the answer to the landed question, whether its commits were
  replayed, whether a conflict was resolved, whether verify ran again.
- `early_start_stopped`: dependent, predecessor, cause, time worked.
- `early_start_recut`: dependent, predecessor, the ruling issued.
- `early_start_paused`: the stop streak.
- `lock_acquired` for the dependent carries `successor_of`.

**Invariants**
1. A predecessor never gains a co-holder other than its one dependent, and gives up keys only as it does today
   (when its slot ends, or when it narrows its own plan). (Ruling 1.)
2. A key co-held by a predecessor and its dependent is refused to every other task. No key ever has more holders
   than its limit, except a single-holder key held by exactly that pair.
3. The lock table's acquires and its blocker views agree for every requester.
4. Permits are conserved: running tasks never exceed `max_concurrent_tasks` plus live early starts; a lease lends to
   at most one borrower at a time; no permit is released twice or lost.
5. The workflow of an early-started dependent never submits a merge request, and never closes its task, while the
   dependent's ledger record names a base that is not on main. (A `done` written by the operator or by the
   escalation steward is outside the workflow's reach and outside this invariant.)
6. A dependent never blocks its predecessor on a single-holder key: the predecessor's own requests do not count
   it there; and when the predecessor's slot ends with its work not landed, the dependent is `pending` and holds
   nothing within `stop_hard_cancel_secs` plus the cancel's own teardown.
7. A stopped dependent is never left `in-progress` without an owner.
8. The stacked base lives in the stack-base ledger and nowhere else.
9. With the switch off no new early start begins, and the ordinary pass's decisions equal today's; live pairs and
   stacked branches are still stopped, settled and asked the landed question. With it on, ordinary starts, pinned starts and the clean-up of
   reservations never treat an unfinished dependency as satisfied.
10. Review, the plan-files check and the scope checks of a stacked task see its own commits only.
11. A task is started at most once: every claim is one uninterrupted step that refuses a task already started.
12. The early-start selection and the task picture write nothing but the selection's own claim and events.
13. No early start begins while the host-load gate holds, the scheduler is paused, a review checkpoint runs, the
    warm-worktree base is down, the merge queue is halted, or the feature has paused itself.
14. Work written on top of a predecessor never merges without that predecessor: a task with a ledger record is
    replayed onto main only when the landed question answers "landed", is cut afresh, at most once per
    predecessor, when it answers "not landed", and does no work while it answers "unknown".
15. A re-stack either completes (the branch is the predecessor's tip plus the dependent's own commits, and the
    ledger answers that tip), or leaves the branch and the record exactly as they were, or blocks the dependent
    with an escalation when git could not restore them.

## 8. Boundary tests (through the scheduler, the harness loop, a real git repository and a real merge worker)

P is the predecessor, D the dependent.

| # | Scenario | Before | Asserted after |
|---|---|---|---|
| 1 | Early start with full slots | Every slot busy; P waits on a live request; D depends only on P; D's keys free or held by P | D is launched; `early_start_dispatched`; D's branch starts at P's branch tip; the ledger answers that commit; permits in use unchanged |
| 2 | Third task | As 1, then a task X wants a co-held key | X is refused; the blocker view names both P and D; X is admitted after both release |
| 3 | Higher tier ready | A higher-tier task with dependencies done needs one of D's keys; its other keys are free | D is not started; the selection event counts `higher_tier_waiting` and names that task |
| 4 | Higher-tier sibling | A higher-tier task whose only unfinished dependency is P needs one of D's keys and does not itself qualify (it has a branch) | D is not started |
| 5 | Higher tier not startable | As 3, but that task also needs a key held by an unrelated running task | D is started |
| 6 | Higher tier became ready after the picture | The picture shows a higher-tier task waiting on a dependency; that dependency lands before the selection | The re-read in step (b) sees it; D is not started |
| 7 | Two-holder file | (i) P and an unrelated task both hold `harness.py`, D needs it; (ii) P alone holds it | (i) D's acquire is refused (`key_unavailable`); (ii) D is admitted and a third task is refused |
| 8 | Delivered checks | (i) P has a `grep` check that passes at its tip; (ii) one that fails there; (iii) a `script` check | (i) D is started; (ii) refused `delivered_check_failed`; (iii) refused `delivered_check_script` |
| 9 | Deterministic or dated dependent | (i) D is a human gate; (ii) D carries a milestone | Not started: (i) `not_code_task`; (ii) `has_milestone` |
| 10 | Predecessor's worktree not at its branch tip | P has a commit in its worktree that is not on its branch, or uncommitted changes | D is not started (`predecessor_branch_moved`) |
| 11 | System holds | In turn: host-load gate holds; scheduler paused; review checkpoint running; warm-worktree base down; merge queue halted; usage gate paused for every account; a daily cost ceiling breached | No early start; the matching refusal reason each time |
| 12 | Predecessor widens | D co-holds; P's plan widens onto a single-holder key D holds alone | P's widening succeeds; P keeps all its locks |
| 13 | Landing as a merge commit of the base | D finished review; P lands with the stacked base as the merged tip | `early_start_settled`; no replay; no extra verify; D's `merge_queued` is after P's `merge_finalized`; D lands |
| 14 | Landing as a merge commit with a fix on top | P's workflow added a commit after the early start; P lands as a merge commit | No replay; one more task verify runs before D submits |
| 15 | Landing re-applied, same changes | Main moved elsewhere; P lands re-applied | D's commits replayed onto main; one more task verify; D's landed commits do not contain P's old commit ids |
| 16 | Landing re-applied, changed | Main changed lines next to P's edits, or P resolved a queue conflict, so the re-applied commits differ | Answer "landed, changed"; D is **not** stopped; replayed; one more task verify |
| 17 | Replay conflicts | D's own commits conflict with main on the replay | The replay pauses at the commit; the merge-conflict agent resolves and continues it in D's slot; the workflow checks D's own work starts at main, then drops the record; verify follows; D never submits with an unlanded record; P's old commits are not on D's branch |
| 18 | Git fails on the landed question | The containment check errors, or the ledger cannot be read, (i) when P's slot ends, (ii) at the settle | Answer "unknown"; D is neither stopped nor settled; the question is asked again, (i) on the timer until it is known, with one escalation after three unknown answers in a row |
| 19 | D reaches its merge phase first | P still queued | D waits holding its keys and no permit of its own; no `merge_queued` for D; after `merge_entry_alert_secs` one escalation, the wait stays |
| 20 | D still working when P lands | P's slot ends mid-implementation, work landed | D keeps running; P's permit is now D's; permits in use unchanged |
| 21 | Predecessor's pass fails, workflow retries | P's request ends `blocked` or `conflict`; P's workflow resubmits in its slot | D is not disturbed |
| 22 | Predecessor gives up | P's slot ends with its work not landed | D is stopped; `early_start_stopped`; D is `pending` with no owner stamp, holds nothing; branch and ledger record remain; P can re-acquire every key at once; the stop does not count toward D's retry limit |
| 23 | Hard-cancelled stop | As 22, D ignores the soft cancel | After `stop_hard_cancel_secs` D is cancelled, written `pending` before its owner stamp is cleared |
| 24 | After 22, P lands and D was not started early again | D pending with a stacked branch; the feature is paused while P is repaired and lands | D's ordinary start replays its commits onto main and resumes; a task verify runs on the new tip before D submits |
| 25 | After 22, P is cancelled later | D pending with a stacked branch; P cancelled by a human | D's next start issues the automatic re-cut: branch, plan and session parked, fresh branch from main; `early_start_recut` |
| 26 | Operator marks P done while queued | P's request is cancelled; nothing landed | D is stopped, not merged; its next start is the re-cut of 25 |
| 27 | P landed, `done` write refused | P's work on main, P's status not `done` | D is treated as landed (decided from git) and settles |
| 28 | Review subject | D stacked on P, before and after P lands | The review subject and the plan-files check contain no file changed only by P |
| 29 | Closing guards | D's workflow reaches merge entry with a steward `done` to honour while P is queued | The workflow settles first: it does not close D while the record names an unlanded base |
| 30 | Stop streak | Five early starts in a row end stopped | `early_start_paused`; one escalation; no further early start until `enabled` is switched off and on |
| 31 | Shutdown | The orchestrator shuts down with D early-started | No `early_start_stopped`; the streak is unchanged |
| 32 | Switch off | `early_start.enabled: false` | The coroutine reads nothing; start decisions equal today's on a recorded pass |
| 33 | Launch race | P's request ends between the shortlist and the claim | No claim; nothing held |
| 34 | Claim race | An ordinary pass built its ready list with D in it (P was cancelled meanwhile); the selection claims D; the ordinary pass reaches its claim | The ordinary claim refuses D; D is started once |
| 35 | Selection and picture write nothing | A picture is built and a selection starts nobody | No status, counter, flag or escalation written; the next ordinary pass behaves as if neither had run |
| 36 | Restart | Orchestrator restarts with D early-started | D is `pending` (today's start-up path); nothing of D's is held after the first pass; D's next start asks the landed question as in 24 and 25, reading P's landing from the event store and the task record |
| 37 | Re-cut, then requeue | After 25, D's fresh run is requeued before its merge phase | D's next start issues no second re-cut; its fresh branch has no ledger record and D has no early-start mark |
| 38 | Git fails at a later start | The landed question answers "unknown" three times in a row at D's start | Each time D goes back to `pending` holding nothing, not counted toward its retry limit; one escalation after the third |
| 39 | A landing from before the early start | P landed once, was re-opened, and its new work is queued when D starts early; P's new work then fails | The old landing does not count: the answer is "not landed" |
| 40 | Ledger write fails | The record cannot be written or read back at the early start | The early start is abandoned; nothing held; refusal `ledger_write_failed` |
| 41 | On by default | The code after T7; the project's yaml has no `early_start` block | Test 1's early start happens with no yaml key; with `early_start.enabled: false` it does not |
| 42 | Second early start | After 22: P is repaired (its branch rebuilt on a newer main) and waits on a new request | D is started early again; its branch is P's new tip plus D's own commits; the ledger answers the new tip; `early_start_dispatched` names the base it replaced; no commit of P's old branch that is absent from the new tip is on D's branch |
| 43 | Re-stack conflicts | As 42, but D's own commits conflict with P's new tip | The trial reports the conflict; D is not claimed; refusal `restack_conflict`; D's branch tip and ledger record are unchanged; the next selection does not try the same pair of tips |
| 44 | A branch that was not stacked on this predecessor | D has a branch cut from main, or a ledger record with a mark naming another task | D is not started (`has_branch`) |
| 45 | Withdrawn 2026-11-17 | The bound on stops per pair was removed; the number is kept so that later rows keep theirs | Nothing to test |
| 46 | Two dependents of one predecessor | D1 and D2 both qualify on P; D1 ranks higher | D1 is started; D2 is not (`predecessor_has_dependent`); P's lease lends once; after D1 is stopped, D2 may be considered |
| 47 | Re-stack through each reuse route | As 42, with D's worktree (i) still registered, (ii) removed, (iii) a warm worktree | In each case D's branch is P's new tip plus D's own commits and the ledger answers the new tip; no un-stack onto main and no plain rebase onto main runs |
| 48 | Stop with uncommitted work | D has uncommitted edits when P's slot ends not landed; then 42 | After the stop, D's branch carries the edits in a commit; after the re-stack they are on D's branch above P's new tip |
| 49 | Switch off with a live pair | D early-started; `enabled` reloaded to false; P's slot then (i) ends not landed, (ii) lands | (i) D is stopped as in 22; (ii) D settles as in 13; no new early start begins |

## 9. Decomposition plan

Thirteen tasks: eleven build tasks and two gates. Order is forced by shared files, and every "after" below is a
dependency edge. `scheduler.py`: T1, T4, T5a. `git_ops.py` and `branch_stack.py`: T2a, T2b. `harness.py`: T3, T5b,
T6a, T6b, T6c, T7. `workflow.py`: T2a, T2b, T6a, T6b. T1, T2a and T3 touch different files and may overlap.

- **T1. The successor relation in the lock table and the early-start dependency answer.** `lock_table.py`,
  `dependency_verdict.py`, `scheduler.py` (release drops the relation; `successor_of` on the lock event),
  `config.py` (the `early_start` group, with `enabled` defaulting to off until T7, in the config model only), the
  scheduler-state snapshot, tests. Unlocks T4, T5a. Signal: tests 2, 7 (the acquire half, at the scheduler
  boundary) and 12; the dependency half of 32.
- **T2a. Stacked worktree, the re-stack, and own work.** `git_ops.py` (`create_worktree`'s "stack on this commit"
  input with a ledger record and its read-back; a read of the record that tells "no record" from "cannot read"
  (today both read as nothing); stacked worktrees never from the warm pool; the re-stack mode of
  the three reuse routes and its trial; without a stack input all three routes un-stack before a rebase onto
  main), `branch_stack.py`, `workflow.py` (worktree set-up passes the stack target; own work through the existing
  rule), tests on a real repository. Supersedes task 7660; after task 7736 (in progress; it edits the same
  worktree-reuse path). Unlocks T2b, T5b. Signal: test 28; the ledger answers the base of a branch
  `create_worktree` stacked; the git halves of tests 42 and 43; test 47.
- **T2b. The landed question and the replay that pauses.** `git_ops.py`, `branch_stack.py`: the four-valued
  landed question; the replay onto main that pauses at a conflict and can be continued; tests on a real
  repository. After T2a. Unlocks T6a, T6b, T6c. Signal: the landed question answers each of its four values (the
  git halves of tests 13 to 18 and of 39); a paused replay is continued to completion and only then drops the
  record; open question 7.
- **T3. The borrowed slot.** `slot_lease.py`, `harness.py` (launching a task with a borrowing lease; the per-task
  "slot ended" signal), tests through the harness loop. Unlocks T5b. Signal: permits are conserved over every exit
  in both orders (normal end, requeue, forced release, exception, shutdown); the lender's permit becomes the
  borrower's.
- **T4. The task picture and the claim-time check.** `scheduler.py` (the read parts of the five steps split from
  their writing parts; the picture; the refusal of an already-started task at the ranked and pinned claims), tests.
  After T1. Unlocks T5a. Signal: test 35 for the picture; test 34's ordinary half; the ordinary pass's decisions
  are unchanged on a recorded pass; the measured event-loop time of one picture, reported in the task's result.
- **T5a. The early-start selection in the scheduler.** `scheduler.py`: the selection entry (the conditions of
  5.1 through `_eligible_for_dispatch`, the tier condition, the predecessor's delivered checks at its tip, one
  dependent per predecessor, the system holds, the claim in one uninterrupted step, the bookkeeping of 5.3),
  tests at the scheduler boundary. After T1 and T4. Unlocks T5b. Signal: tests 3 to 9, 11, 44, 46 and the early
  half of 34, each refusal with its reason.
- **T5b. The early-start coroutine and launch in the harness.** `harness.py` (waiting predecessors; the
  coroutine; the reads of step (b), the trial replay and its remembered refusal included; the launch on a
  borrowing lease with the stack input, the ledger read-back and the early-start mark), the events
  `early_start_selection` and `early_start_dispatched`, tests. After T5a, T2a and T3. Unlocks T6a. Signal: tests
  1, 10, 33, 34, 40, 42 and 43 with the switch on in a harness test (for 42 and 43 the stopped dependent is built
  by the test: a pending task with a stacked branch, a ledger record and a mark).
- **T6a. Stop when the predecessor's work does not land.** `harness.py`, `workflow.py`, `scheduler.py`: the soft
  then hard stop, the commit of work in progress before the exit, `pending` before the owner stamp is cleared, a
  requeue that does not count, the stop streak and self-pause, the stops-per-pair refusal counted from
  `early_start_stopped` events, the re-asking of an "unknown" answer at slot end, the shutdown exclusion; tests through the harness loop. After T5b and T2b. Unlocks
  T6b. Signal: tests 18 (i), 22, 23, 30, 31, 48, 49 (i), and test 42 run from a real stop.
- **T6b. Settle when it lands.** `workflow.py`, the merge-conflict agent's brief for a paused
  replay: the settle, its use of T2b's replay and the conflict handling, the verify rule, the wait at merge entry and its
  alert; tests with a real merge worker and repository. After T6a. Unlocks T6c. Signal: tests 13 to 17, 18 (ii), 19 to 21, 27, 29,
  49 (ii).
- **T6c. The landed question at every later ordinary start.** `harness.py` (in `_run_slot`, before the re-cut
  gate), `recut_ruling.py` (the automatic issuer): the question, the automatic re-cut with removal of the record
  and the mark, the "unknown" requeue and its alert. After T6b. Unlocks T7. Signal: tests 24, 25, 26, 36 to 39.
- **T7. Operator views, and the default turned on.** The dashboard alarm that compares running tasks with the
  slot count allows for borrowed slots: it reads stored snapshots, not the live state, so each snapshot row gains
  the number of live early starts (a new column, written from the orchestrator's runtime state) and the alarm
  compares running tasks with the slot count plus that number; early starts in the runtime state and the scheduler view;
  `OPERATIONS.md`. Last, `early_start.enabled` defaults to on (choice 7), in its one home, and every existing
  test that then behaves differently is adjusted in the same change. After T6c. Signal: with one early start live
  the alarm is silent and the runtime state names the lender; test 41.
- **T8. Gate: read the first real pairs.** A pure gate (a deterministic task that raises an escalation when it
  fires) with `metadata.milestone: {mode: delayed, after_secs: 172800}` and a dependency on T7: it fires two days
  after T7 is merged. The fleet restarts on merged code by itself within about 8 hours, so by then every project
  runs with the feature on. The gate changes no config. If no pair has completed the sequence below by then, it
  reports the refusal counts from `early_start_selection` and leaves the proof to T9, which then carries it. If a pair breaks the
  sequence, it tells Leo; the operator switches the feature off for a project with `early_start.enabled: false`.
  **Signal (the end-to-end proof):** for one real pair in the event store, in this order:
  `early_start_dispatched` for the dependent naming the predecessor and a stacked base that the landed question
  later answers as landed; `lock_acquired` for the dependent with `successor_of`; no `lock_acquired` by any
  third task on a co-held key before the predecessor's `merge_finalized`; `early_start_settled`; the dependent's
  `merge_queued` strictly after the predecessor's `merge_finalized`; the dependent's own `merge_finalized` with
  state done.
- **T9. Gate: read two weeks of early starts.** A pure gate with `after_secs: 1209600` and a dependency on T7.
  Counts per week and per project of starts (first and second), stops by cause and per pair, re-cuts, refusals by
  reason, hours gained (early start to predecessor landing), verifies re-run, conflicts on a replay or a
  re-stack, how often a landing read as "landed, changed", the peak number of live early starts, the first-build
  cost of a stacked worktree per project, and the task-verify admission wait of ordinary tasks before and after
  T7 (every early start, re-stack and settle adds a verify, and there is no cap); against section 2. If T8 closed without a completed pair, T9 shows T8's sequence for one real pair, or reports
  that no early start has settled in two weeks, which is a finding for Leo. Recommends on stage 2.

## 10. Pre-conditions

- Task 7736 (a failed resume rebase continues on a conflicting base; in progress) lands before T2a, because both
  edit the worktree-reuse path. Task 7660 is delivered by T2a and is closed as superseded at decompose.
- This plan writes no yaml key: the feature is on by the code's default once T7 lands (choice 7). A project that
  opts out writes `early_start.enabled: false` only after its orchestrator has restarted on T1 (an unknown key on
  old code raises an escalation on reload).
- Nothing else blocks T1 to T7.

## 11. Cross-plan seams

| Other work | Seam | Owner | Status |
|---|---|---|---|
| `plans/resume-interrupted-tasks-prd.md` (restart resume; committed, not decomposed) | Its record of held files is written wherever the lock table changes; the early-start claim is one more such place | Whichever lands second hooks it; this plan's claim goes through the scheduler, where that plan writes its rows | Note on its decompose gate, task 10301 |
| Same plan | A restored dependent would be released at its first pass | That plan already ends in the right state; refinement asked for in the note (choice 6) | No edit here |
| Tasks 7277, 10093, 10216 (recovered merge requests after a restart; all pending) | A recovered request has no waiting workflow, so its task is not a predecessor in this plan's sense | This plan (the predecessor test in 5.1) | No change needed there |
| Task 7660 (stack record missing on create; pending, low) | Same code | This plan, T2a | Superseded at decompose |
| Task 7736 (a failed resume rebase continues on a conflicting base; in progress, high) | Same worktree-reuse path as T2a | Task 7736 first | Pre-condition |
| Task 9267 of `plans/review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.md` (one owner for every write to a task branch; pending, high, behind a long chain; task 7660 waited on it) | It routes every rebase of a task branch, the un-stack and `rebase_own_delta` through one owner with a closed set of targets (main, stacked on a commit, a frozen prefix); T2a adds the re-stack and its trial to the same functions | Whichever lands second: if T2a lands first, task 9267 routes the re-stack through its owner (its "stacked on" target fits); if task 9267 lands first, T2a builds on the owner | Note on task 9267 at decompose; no dependency, because task 9267 waits behind a long chain |
| Task 9269 (worktree primitives move out of `git_ops.py`; pending) | Moves the functions T2a changes | Whichever lands second builds on the other | Note on task 9269 at decompose; no dependency |
| Task 9216 (`slot_lease.py` folds into a slot registry; pending, behind task 5153 and others) | T3's borrowing lease and "slot ended" signal live in the file it removes | Whichever lands second: task 9216 carries lender, borrower and the signal into the registry and re-points T3's manifest check | Note on task 9216 at decompose; no dependency |
| Task 9674 (dependency readiness becomes one predicate over a required value; pending, high) | T1's opt-in input to the dependency test | Whichever lands second: the waiting-predecessor input becomes a field of that value | Note on task 9674 at decompose; no dependency |
| Task 7540 (`rebase_own_delta` discards the outcome of its abort; pending, low) | T2a needs that outcome to tell a conflict from a failed abort | This plan, T2a, delivers it | Note on task 7540 at decompose |
| Task 7691 (cold re-attach of a train successor rebases onto main; pending, low) | Same reuse route as T2a | Task 7691; T2a excludes train successors and changes nothing for them beyond writing the record | No change needed there |
| Task 10099 (ledger records do not follow a renamed branch; pending, low) | A record stays with the branch's name, so after a re-cut it would sit on the fresh branch | This plan: the automatic re-cut drops the record itself (section 5.6) | No dependency on task 10099 |
| Task 10211 (dashboard reads every holder of a key; in progress) | Shows a co-held key under both tasks | Task 10211 | Soft pre-condition for T7's view |
| Task 5153 (an idle registered workflow holds its slot for ever; pending, high) | A dependent waiting at merge entry is deliberately idle | Task 5153 reads the early-start entry this plan publishes in the runtime state; if task 5153 lands first, T6b exempts the wait from its reaper | To annotate at decompose |
| Task 6310 (a merge worker that is shutting down answers `blocked`; pending) | A predecessor's workflow may give up on such an answer, and its dependent is then stopped without a real failure | Task 6310 | Residual: wasteful, not unsafe |
| Task 7202 (gate re-measuring the two-holder files; pending) | Holder limits | This plan keeps to them (5.2) | No change needed there |
| `plans/merge-lane-policy-prd.md` (task 8528, batch retry, pending) and `plans/one-merge-request-per-task-prd.md` | None in stage 1: the dependent is never queued before its predecessor lands | n/a | Stage 2 would need them |
| The re-cut mechanism (`orchestrator/src/orchestrator/recut_ruling.py`; issued by a human or by a restart resolution today) | An automatic issuer for "predecessor's work never landed" | This plan, T6c | New |
| Declared trains (`plans/orchestrator-atomic-train-merge-prd.md`; off; gate 7860) | `create_worktree`'s stacking input | This plan generalises it; the train path keeps its behaviour | T2a |
| Allowance reserve (`plans/allowance-reserve-by-account-prd.md`, tasks 10428 to 10436; task 10431, "the scheduler does not start a task the reserve would refuse", pending) | That plan refuses a start the reserve would refuse, at eligibility and at the moment a slot is in hand, and leaves slot lending out on purpose; an early start lends a waiting task's slot and has no slot-in-hand moment | Whichever lands second: task 10431's check goes in `_eligible_for_dispatch` (which the early start uses) and is also asked by the early-start claim | Note on task 10431 at decompose |

## 12. Out of scope

- **Stage 2: the dependent queues its merge behind the predecessor.** Needs four merge-queue changes (strict
  predecessor-first order; a failed predecessor pulls the dependent out instead of re-merging it onto main; urgency
  for a merge a successor waits on; the pair as one batching unit). The first seat found the dependent would have
  finished its own work before the predecessor landed in 1 of 6 and 1 of 4 cases, so the extra gain looks small.
- A dependent with more than one unfinished dependency; early starts deeper than one at a time in a chain (a third
  task starts when the second submits its merge, which is after the first has landed).
- Delivered checks of the `script` kind on a predecessor.
- Skipping the extra verify after a landing re-applied with the same changes: the re-applied range is not
  recorded today.
- Starting early a dependent whose existing branch was not stacked on this predecessor by an earlier early start
  (a branch cut from main, or one stacked on another task).
- Surviving a restart as an early start.
- Any release of a file before its holder's merge lands.
- A trial merge at submit as the trigger. Leo's wording was "not been rejected for an irreconcilable conflict";
  this plan starts on a live request and relies on the stop when the predecessor's work does not land. Conflict
  answers were 16 of 715 and 7 of 245 queue passes (first seat).
- A `done` written on the dependent by the operator or the steward while its predecessor is unlanded.

## 13. Open questions (tactical)

1. **Cross-project allowance weighting and the allowance reserve** with two running tasks on one slot: see the
   allowance-reserve row of section 11; T7 checks the weighting.
2. **Warm worktrees.** Decided: a stacked worktree is cold (section 5.3), because a released warm worktree does
   not keep uncommitted work. T9 measures the first-build cost, per project.
3. **Where the per-task "slot ended" signal lives** (the scheduler's release or the lease). Decide in T3.
4. **`merge_entry_alert_secs`.** Suggested 12 h: the two merge-entry waits the first seat observed were 7.1 h and
   10.8 h. Decide in T6b.
5. **Whether review runs again after a replay at a later start** (test 24 requires only the verify). Suggested:
   no, as for the settle. Decide in T6c.
6. **How often a real landing is re-applied or reads as "landed, changed"** is not known; it decides how often
   the extra verify runs. T9 reports it.
7. **Whether the containment classifier can call a range "contained" when the queue re-applied it with conflicts
   resolved.** It should not, and the plan relies on that; T2b tests it on a real repository.
8. **Stops per pair.** No bound (ruled by Leo, 2026-11-17). T9 reports stops per pair, so that a pair that
   cycles is seen.
9. **Whether the start-up clean-up removes a pending task's cold worktree that still holds uncommitted work.**
   The stop commits first, so only a hard cancel cut short by a restart is exposed. T6a checks.

## 14. What changed

**From draft 1** (first critic, 23 findings, 3 blocking):
- The predecessor's fate is decided from git, not from its task status (findings 1, 20).
- The tier condition also counts a higher-tier task whose only unfinished dependency is the predecessor (2).
- The early-start pass is a read-only selection that never re-runs the ordinary pass's writing steps (3).
- The file rule is stated once, works in both directions, and applies its exception only to single-holder keys
  (4, 13). The separate choice on two-holder files is gone.
- Delivered checks are evaluated at the predecessor's tip; only `script` checks refuse a pair (7).
- The stop to pending on both cancel arms and the shutdown exclusion are specified (6). Invariant 5 is limited to
  what the workflow controls and the settle's place is named (8). Own work goes through the existing rule (18).
- Bookkeeping at an early start, the review-checkpoint and warm-base holds, the not-always-idle slot, the reset of
  the self-pause (12, 14, 15, 19). No edit to the resume plan; seam rows and a note on its gate instead (10).
- Figures corrected (11). T3 split; terms added (16, 23).

**From draft 2** (delta critic, 1 blocking, 8 to fix, 7 minor):
- Whether a stopped dependent is replayed or cut afresh is decided at every start, from git, and the automatic
  re-cut is the mechanism (N1). Invariant 14 states the rule.
- A landing whose commits changed on the way in is "landed, changed", not "not landed"; a git error is "unknown"
  and is retried (N2). The verify rule compares what landed with the stacked base in every case (N3).
- The ordinary claim refuses an already-started task at claim time (N4). The task picture is specified: always
  built by read-only steps, with its measured cost (N5). Statuses of competing tasks are re-read before the claim
  (N6).
- A replay entry point for conflicts (N7). The stop does not count toward the retry limit (N8).
- Task signals moved to the tasks that can produce them; a new task for the picture (N9).
- The stacked base was tied to the predecessor's merge request (N10; replaced in draft 4, F3); the fairness
  reasoning corrected (N11);
  two residuals stated in 5.2 (N12, N13); a clause dropped (N14); the warm-base test and the selection event's
  rate (N15); terms (N16).

**From draft 3** (gating critic: commit after its text fixes; all applied):
- The automatic re-cut drops the ledger record and the early-start mark, so it happens once, not at every later
  start (F1). How the ruling is issued and seen in the same start is specified (F4).
- "Landed, changed" names where a landing is read from, counts only landings after the early start, and counts
  an `already_merged` outcome (F2).
- The stacked base is the predecessor's branch tip with a clean worktree; a merge request records no tip (F3).
- The replay pauses at a conflict, the agent gets a brief for that case, and the result is checked before the
  record is dropped (F5). The extra verify is skipped only for a merge commit of exactly the stacked base (F6).
- The predecessor's-fate task is split in three (F7). "Unknown" at a start is a requeue with an alert (F8).
- The picture's read parts are listed in full; a dependent with a milestone is refused (F9). A failed ledger write
  abandons the early start; "cannot read" is "unknown" (F10). A conflicting replay at a later start is settled at
  merge entry (F11). Two task labels corrected (F12, F13).

**From draft 4** (Leo's rulings of 2026-11-16):
- Choices 1, 2, 5 and 6 stand as recommended.
- Choice 7 is ruled "enabled by default": the last build task turns the default on, the yaml edit and the human
  gate are gone, T8 and T9 are delayed gates after T7, and test 41 is added.
- Choice 3 is ruled "a second early start is allowed": the re-stack (5.3), its refusals, invariant 15, tests 42
  to 44, and a bound of three stops per pair (5.7, test 45).
- Choice 4 is ruled "no cap on live early starts": `max_active` and the `cap` refusal are gone.

**From draft 6** (fourth critic: commit after its text fixes; all applied):
- The re-stack is a mode of `create_worktree`'s three reuse routes, not a step of the launch, which the
  workflow's set-up would have undone; a trial before the claim touches nothing; a failed abort blocks the
  dependent (findings 1, 8, 17). Test 47.
- A stacked worktree is always cold, and the stop commits the work in progress, so a stop loses nothing
  (finding 2). Test 48.
- The switch gates new early starts only; live pairs are still stopped, settled and checked (finding 3). Test 49.
- The dependent passes the scheduler's one eligibility predicate; a seam row for the allowance reserve
  (finding 4). The usage gate and the cost ceilings are system holds (12).
- One dependent per predecessor has a test and a refusal reason (6, test 46). Invariant 15 agrees with section
  5.1 (5). The stops of a pair are counted from events, not kept on the mark (11); `stop_limit` is the reason's
  name (10).
- T2 and T5 are each split in two (7). The defaults have one home (13). T8 and T9 are pure gates with a delayed
  milestone, and T9 also reads the verify admission wait (14). A re-stack re-runs review as well as verify (15).
- Added by the lead on the critic's "could not check": the predecessor's done record counts as a landing only if
  its commit was not yet on main at the early start, and `early_start_dispatched` records main's tip.
- From the manifest seat: an "unknown" answer at slot end is asked again on a timer (T6a); the replay that
  pauses has one owner (T2b); the ledger read tells "no record" from "cannot read" (T2a); the dashboard alarm's
  data source is named (T7); T9 carries the end-to-end proof if T8 could not; two descriptions of existing code
  corrected (the containment classifier's callers, who issues a re-cut today).
- From the critic of the assembled task batch: the settle leaves declared-train successors alone and asks the
  landed question itself; the cost ceilings are read without acting; the early-start mark is a registered
  metadata key; five more seam rows (tasks 9269, 9216, 9674, 7540, 7691).

**From Leo's ruling of 2026-11-17:**
- The bound of three stops per pair is removed: the key `max_stops_per_pair`, the `stop_limit` refusal and
  test 45 are gone (5.1, 5.6, 5.7, invariant 15, the config and event lists). The predecessor's own requeue
  limits and the escalation ladder bound how often a pair can cycle; T9 still reports stops per pair.
