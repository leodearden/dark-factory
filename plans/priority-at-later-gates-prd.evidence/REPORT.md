# Ordering the merge queue by urgency, and its conflict with arrival-order batching

2026-11-12. Read-only follow-up to the three-gates study (`/tmp/team-gates-1112/REPORT.md`). Four evidence seats, one
lead, and one critic who reviewed the assembled text in three passes; its corrections are folded in (section 10). Nothing in the
repository, the task store, the queues or any config was changed. Evidence files are beside this one: `replay.md`,
`code.md`, `urgency.md`, `owners.md`, `critic.md`, `critic-delta.md` and `critic-delta2.md`, each with its scripts. They are cited as `[replay F3]` and so on.

**Terms used throughout.**
- A *request* is one task's branch waiting in the merge queue to be tested and landed on main.
- A *batch* is several queued requests merged together and tested as one run, then landed in order. (The code and the
  plans call this a "chain".)
- A *pinned* task is one placed in the scheduler's pin queue, by the operator or automatically when a task has
  starved. Pinned tasks are served before the rest, in pin order.
- A *main-repair* task is one filed automatically to fix a broken main branch. It is the only kind of work that is
  put in the merge queue's "high" lane automatically; every "high" request is picked before every "normal" one.
- *Urgent* means: the scheduler's priority tier for the task was critical or high when it was queued, or the task was
  pinned, or its lane was "high".
- *Aging* means a waiting request gains rank with time, so that nothing waits for ever. Here: a request is ranked by
  its arrival time plus 2 hours for each tier it is below the top, unless a row says otherwise.
- *Re-verification* means testing again a request that has already passed, because main moved underneath it before
  it could land.
- *Paused window*: 2026-11-09 14:13Z to 2026-11-12 11:30Z, the three other projects paused. *Earlier window*:
  2026-10-25 to 2026-11-08 16:00Z, four projects running.
- *The three pending merge-lane tasks*: task 8526 (prepare a third request while two are being tested, and free a
  test host as soon as its verdict is in; started at 15:33Z today and back to pending at 16:30Z, when its planning step
  found it needed files other tasks hold), task 8527 (batch at the front of the queue
  when nothing else is being tested; pending) and task 8528 (when a batch fails, bisect to find the culprit, and
  choose batch depth from a rolling pass rate; pending).

**Code read.** Main at commit `7848173a7d`. Main has since moved to `0003fc7ec9`; `workflow.py` and `scheduler.py`
changed (a running task can now take extra files), with no change to the queueing, lane or tier code cited here. The running orchestrator started 2026-11-10 23:48Z on commit `1f240e6da2`. One finished merge-lane task is on
main and not yet running: task 8525, which turns a re-verification into an ordinary dispatch on the fastest free
host. `[code §0]`, `[critic B15, C]`, `[critic delta A5]`

## 1. The answer in brief

1. **The conflict is real in mechanism and modest in size.** At today's rate of arrivals an urgency pick is worth
   under an hour to an urgent request in every lane simulated: 0.3 hours on the lane as recorded, and after the three
   pending tasks 0.04 to 0.33 hours on today's mix of work, depending on one assumption (point 2), and up to 0.9 on
   the earlier mix if that assumption fails. The part specific to batching, an urgent request sharing a failed batch with a less important culprit, is of the order of 0.1 to 0.3
   hours per urgent landing.
2. **The merge-lane work is worth more to urgent work than ordering is, but most of that is not batching, and part of
   it is an assumption.** In the simulation the median wait of an urgent request falls from 2.8 hours to 0.8 hours.
   About 1.5 hours of that comes from assuming that re-verification stops entirely, which the plan credits to two
   tasks that are already finished (one running since 11-10, one live at the next restart). The three pending tasks
   add about 0.5 hours to the median on top, or 0.75 hours as a mean per urgent request. On the same lane and the
   same measure an urgency pick is worth 0.14 hours (0.22 if it also sorts the batch). So the three pending tasks are
   worth about five times an urgency pick, and neither needs the other changed.
3. **The margin before the queue grows long again is thin.** The simulated lane keeps a median wait under an hour up
   to about 1.2 times the paused window's arrivals. At 1.3 to 1.4 times the urgent median is 1.3 to 1.9 hours, and at
   1.5 times it is 2.6 hours, the same as today. The task-slot limit returning from 18 to 24 is 1.33 times.
4. **It is tractable, with one barely measured cost.** Which request is picked next is decided in one function, in a file
   few tasks are waiting to edit. But picking by urgency replaces an existing rule, that an older request touching
   the same files goes first. A request that is overtaken and then conflicts with main is rebased, and after three rebases it is escalated to
   a person. At one snapshot 17 of 105 pairs of queued requests changed a common file. A variant that keeps the rule
   exists only if the rule's own test is narrowed first: as it runs today, nearly every pair of waiting requests
   counts as touching the same files.
5. **What to avoid.** Strict priority with no aging starves low-priority work and, with today's handling of failed
   batches, makes one failing request fail batch after batch. Testing urgent requests alone is harmful on today's
   lane. Giving the fastest host to the most urgent request buys minutes.
6. **Urgency should be the definition already ruled for the verify-admission gate** (task 8261: the scheduler's live
   priority tier, read at each pick, with aging). "What is this request blocking upstream" cannot be used as a tier,
   because most requests are blocking something important. As a count of the tasks a request alone is blocking it
   does separate requests, but how often landing one request frees an important task is known only very loosely.
7. **Recommendation for discussion.** Leave the batching work as planned. Record the lane now, and sample from outside what an
   urgency pick would choose and how often queued requests change a common file. Check the simulator against the next restart, which costs nothing. Then
   decide, from that record, between urgency with aging in the pick as it stands and a variant that first narrows the
   same-files test. Nothing in the replay makes waiting for task 8528 a safety requirement, but the change cannot go
   in before the tier can reach the pick (section 7). Detail in section 8.

## 2. How severe is the conflict

### 2a. What was measured before this study

- In the paused window 39% of picks (55 of 142) took a request while a higher-tier or pinned one waited; in the
  earlier window 68% (374 of 549). The extra wait was bounded at about 1.6 and 2.7 hours per affected request.
  `[flow F9]` That is a bound per affected request from a simple attribution rule. It is not comparable with the mean
  gains per urgent request below.
- A September study did weigh priority ordering at the merge queue, when the queue was short (median wait under 15
  minutes). It found 21% of waiting requests overtaken by lower-priority work. It recommended priority with a 4-hour
  aging cap as an optional step, on condition that the would-be pick was logged first, and it rejected deriving the
  lane from priority. It was neither filed nor ruled. `[owners R5, F3]`, `[critic B14]` The parent report's line that
  priority ordering "was not considered" is true of the merge-lane policy plan only.

### 2b. The replay

The replay seat fed the recorded requests (arrival time, tier, pin, final outcome, own test speed, and direct commits
to main) through a simulated lane. Each case was run 30 times with small random changes to test durations; figures
are the median across those runs.

**How far to trust it.**
- The simulated lane reproduces the paused window's recorded wait from first queueing to landing within 16%: median
  2.54 against 2.43 hours, 90th percentile 7.03 against 8.38, mean 3.31 against 3.38 (115 landed requests).
- That fit was tuned. The recorded number of re-verifications per request is an input, and the direct commits to main
  were halved to make the remaining re-verifications match. Without the halving the simulated waits are 13 to 20%
  long. By tier the fit is looser (low tier 1.71 hours recorded against 2.36 simulated).
- It does **not** reproduce the earlier window (simulated median 2.05 against 4.74 hours recorded). Earlier-window
  figures are comparisons between orderings on a different mix of work, not absolute figures.
- The simulation makes about twice as many batches as were recorded, and three times as many failed ones.
- Between runs the urgent median ranges from 0.7 to 1.0 hours on the lane after the pending tasks and from 1.7 to 3.9
  hours on the lane as recorded. Gains are compared within a run, so they are steadier than the medians.
- The lane after the pending tasks is modelled from the plan's text, not from code.

`[replay §1, F1]`, `[critic B11]`, `[critic delta B4]`

**Urgent work is most of the queue in the paused window.** 55% of requests are urgent (83 of 151), against 24% in the
earlier window (154 of 636). When most requests are urgent there is little left to reorder. `[replay F2]`

**Results, paused window.** Hours from queueing to landing. "Gain" is the mean time an urgent request lands sooner
than under arrival order, paired request by request. "Weighted wait" is the change in mean wait with each request
weighted by tier (critical 8, high 4, medium 2, low 1), which is the measure closest to the stated goal.

| Lane | Ordering | Urgent median | Low-tier median | Low-tier worst | Gain per urgent request | Urgent more than 0.5 h sooner / later | Weighted wait |
|---|---|---|---|---|---|---|---|
| As recorded | Arrival order (deployed) | 2.79 | 2.36 | 8.8 | n/a | n/a | n/a |
| As recorded | Urgency picks; batch fills in arrival order | 2.59 | 2.75 | 10.0 | 0.31 | 22% / 14% | −3% |
| As recorded | Urgency picks and sorts the batch | 2.60 | 2.51 | 8.2 | 0.53 | 27% / 14% | −14% |
| As recorded | Two lanes from tier, aging between them | 2.34 | 1.93 | 8.1 | 0.69 | 29% / 10% | −14% |
| As recorded | Urgent requests tested alone | 4.01 | 3.09 | 12.0 | −1.55 | 18% / 56% | +50% |
| After the three pending tasks, no re-verification | Arrival order | 0.83 | 0.93 | 3.3 | n/a | n/a | n/a |
| After the three pending tasks, no re-verification | Urgency picks; batch fills in arrival order | 0.79 | 1.02 | 3.4 | 0.04 | 6% / 3% | −3% |
| After the three pending tasks, no re-verification | Urgency picks and sorts the batch | 0.80 | 1.00 | 3.2 | 0.01 | 6% / 4% | +1% |
| After the three pending tasks, no re-verification | Urgency picks, and urgent work waits for the fast host when that is quicker | 0.78 | 1.04 | 3.4 | 0.06 | 11% / 5% | −7% |

All rows simulated, 66 urgent landings per run. `[replay F3, F4, results tables]`, `[critic B3, B4, B5]`

On the earlier window's mix, where urgent work is a quarter of the queue: 0.41 hours gained per urgent request on the
lane as recorded when urgency picks, 0.08 when it also sorts the batch (and the weighted wait is then 5% worse); 0.15
after the pending tasks, 0.25 with the fast-host rule (an upper bound, since that rule is given perfect knowledge of
when the fast host frees). `[replay F3, F4]`, `[critic B7]`

**The "after" rows rest on one assumption, and the "as recorded" rows are not quite today's lane.** The simulated
lanes differ in two ways, and the larger is not one of the three pending tasks. The lane as recorded replays the 34
re-verifications that the queue's own landings caused in the paused window. The "after" lane sets them to zero, on
the plan's claim that two finished tasks remove them: task 8524 (running since the 11-10 restart) and task 8525
(merged on 11-12, live at the next restart). The critic re-ran the simulator to separate the two effects:

| Simulated lane, paused window, arrival order | Urgent median | Test runs per landing |
|---|---|---|
| As recorded | 2.79 | 2.00 |
| As recorded, with those re-verifications removed and nothing else changed | 1.32 | 1.46 |
| After the three pending tasks, re-verifications kept at the recorded count | 1.73 | 2.35 |
| After the three pending tasks, re-verifications removed | 0.83 | 1.48 |

Simulated; the third row is high, because the simulator cancels work that the real design would keep. `[critic A1]`

- Removing re-verification is worth about 1.5 hours. The three pending tasks are worth about 0.5 hours once it is
  gone, or about 1.1 hours if it persists. Only two of the three are batching.
- The honest range for the urgent median after the three pending tasks is 0.8 to 1.7 hours.
- On the second row's lane, which is the closest to what will run after the next restart, an urgency order gains
  0.14 hours per urgent request (urgency picks), 0.22 (also sorts the batch) or 0.36 (two lanes from tier). On the
  same measure the three pending tasks gain 0.75 hours per urgent request, and removing re-verification 1.55.
  `[critic delta A3]`
- If re-verification persists at its recorded rate after the pending tasks, urgency picking gains 0.33 hours per
  urgent request on the paused window's mix and 0.92 on the earlier one.
- The plan's own pass mark is fewer than 0.3 re-verifications per landing. The paused window recorded 0.45. The
  simulated "after" lane has 0.09. So the plan could pass with more re-verification than the simulation assumes.
- The one observation available does not confirm the assumption. Task 8524 went live inside the paused window. The
  re-verifications caused by the queue's own landings fell from 0.43 to 0.15 per landing across that restart, and
  the recorded median wait did not fall (1.99 hours before, 2.43 after). Arrivals rose from 44 to 60 a day at the
  same time and the urgent share of requests from 40% to 65%, so this refutes nothing either.

### 2c. The batching-specific cost

- An urgent request that shares a batch with a failing, less important request waits for that batch's test run and is
  then tested again. On the simulated lane as recorded that happens about 16 times per run across 67 urgent landings,
  9.7 hours of failed test runs in total; after the pending tasks, 1.7 times and 0.7 hours. This is of the order of
  0.1 to 0.3 hours per urgent landing today. It is not a bound in either direction: the simulation makes three times
  too many failed batches, and it counts only the failed run itself, not the wait to be dispatched again.
  `[replay F6]`, `[critic B2]`
- Recorded, paused window: 32 of 183 merge test runs were batches; 9 failed, carrying 40 of the 128 batched requests.
  Which requests were in each batch is not recorded, so the recorded data cannot say how many of the 40 were urgent.
  `[code F1, U6]`

### 2d. When ordering starts to matter again

The lane after the pending tasks (no re-verification), with arrivals compressed in time:

| Arrivals | Mix | Urgent median, arrival order | Urgent median, urgency picks | Gain per urgent request |
|---|---|---|---|---|
| 1.0 times | Paused window | 0.83 | 0.79 | 0.04 |
| 1.2 times | Paused window | 0.79 | 0.79 | not computed |
| 1.3 times | Paused window | 1.87 | 1.69 | 0.14 |
| 1.4 times | Paused window | 1.34 | 1.26 | not computed |
| 1.5 times | Paused window | 2.62 | 2.11 | 0.28 |
| 2 times | Paused window | 3.70 | 3.47 | 0.62 |
| 1.5 times | Earlier window | 1.48 | 0.87 | 0.64 |
| 2 times | Earlier window | 7.69 | 5.16 | 2.93 |

Simulated projections of loads that have not been observed. The paused-window rows at 1.2, 1.3 and 1.4 times, the
gains at 1.3 and 1.5 times and the earlier-window gain of 0.64 are the critic's re-runs of the seat's simulator.
`[replay F9]`, `[critic A2]`

- The queue grows long somewhere between 1.2 and 1.4 times the paused window's arrivals. The turning point is not
  sharp: the 1.4 times row is lower than the 1.3 times row.
- On the mix of work in the queue today, urgency picking then gains 0.1 to 0.6 hours per urgent request and costs each
  non-urgent request 0.1 to 1.0 hours. The gain of nearly 3 hours appears only on the earlier window's mix.
- With strict priority at 2 times arrivals (earlier mix) the urgent median is 0.93 hours, and the worst low-tier wait
  doubles from 19.5 to 38.9 hours.
- Three things push arrivals up or capacity down, and none is in the replay: the task-slot limit going from 18 to 24
  (ruled today); a faster lane handing slots back to agents sooner; and the other projects resuming, which was
  observed to slow the workstation's merge test runs by up to 1.4 times. Arrivals already rose from 44 to 60 a day
  inside the paused window.

### 2e. What the replay cannot see

| Not modelled | Which way it biases the result |
|---|---|
| Whether a batch fails depends on its contents; a recorded failure may have been a one-off | Overstates the cost of sharing a batch |
| The rule that an older request touching the same files goes first | Hides choice B's cost in rebases. The simulated arrival-order baseline is close to what the rule does today (section 3) |
| New conflicts created by reordering | Unknown, probably small (6 of 151 requests left on a conflict) |
| Landing urgent work sooner frees its files and its task slot sooner, which changes what arrives next | Understates the value of urgency ordering; this is the lock hand-off follow-up's ground |
| Restarts, halts, host outages | Hit every ordering alike |
| The tier is read once at queueing, not at each pick | Small in a short queue |

## 3. How tractable is it

**Where order is decided.** `[code §1]`, checked on main by the critic.

| Decision | Where | Rule today | Changed by the pending work? |
|---|---|---|---|
| Which waiting request is taken next | `merge_lane/intake_buffers.py::MergeIntake.pop_next_pickable`, the only picker | "High" lane first; then, by intent, arrival order except that an older request touching the same files goes first. In effect, oldest first submission first (see below) | Task 7720 (reserving a batch's members for its test run) edits this function |
| Which requests join a batch, in what order | `merge_lane/worker.py::chain_snapshot` and `::build_chain` | "High" lane first, then arrival order | Tasks 8527 and 8528 change who batches and how deep, not the order. A later task moves this code to its own file |
| Whether a request is batched, and how deep | `merge_lane/worker.py::_deep_chain_placement`, `::select_chain_depth` | Depth only | Rewritten by tasks 8527 and 8528 |
| Which host tests it | `verify_runner.py::HostAllocator.acquire` | First free host, laptop preferred; it is told nothing about the request | Task 8526 (pending) edits this |
| Landing order | The verifier loop and `::_land_chain_prefix` | Order of dispatch, then batch order | Moved later, not changed |
| Which lane | `merge_lane/worker.py::lane_for_task_metadata` | One hand-set metadata field | Two tasks from the whole-project review move the operator tools |

**Which rules an ordering change touches.** `[code §2, §3]`

- **Landing in order.** Landing follows the order in which requests were dispatched, not the order they arrived, and
  arrival order is already broken today by the two lanes. Picking or sorting by urgency leaves it intact.
- **An older request touching the same files goes first** (task 1891, done). This is the rule an urgency pick
  replaces, and it is the barely measured cost.
  - Its purpose is that the oldest request, the most expensive to redo, gets the cleanest chance to land.
  - Under an urgency pick the overtaken request meets a changed main. If it then conflicts, it is rebased automatically, and after three
    such rebases it is escalated to a person (`merge_lane/worker.py::MERGE_BOUNCE_CAP`). Aging limits how long a
    request can be overtaken, not how many times.
  - No event records how often two waiting requests touch the same files, and the replay does not model the rule.
    At one snapshot today (16:48Z, 15 queued requests), 17 of 105 pairs changed a common file, counting what each
    branch itself changed. Those are the pairs where overtaking can force a rebase.
  - **How the rule works in practice.** The code calls two requests "touching the same files" when the files that
    differ between current main and each branch overlap. That includes everything main has changed since the branch
    was cut. At the same snapshot all 105 pairs were linked by that test. (The critic applied the code's test to
    the queue's journal, which lists requests being tested as well as waiting ones; the lane's own record of links
    was not read.) So the rule today amounts to "oldest first
    submission first" for the whole lane, which is close to arrival order (it differs for a request queued again
    after a failure, which keeps its first stamp).
  - **A variant that keeps the rule (choice B′)**, in which urgency chooses only among requests with no older waiting
    request touching the same files, needs the test narrowed first. On the test as it runs it would change nothing,
    because only the oldest request has no older linked request. With the test narrowed to each branch's own changes,
    9 of 14 normal-lane requests at the snapshot could be chosen by urgency. That is a change in `suffix_graph.py`
    to a test that also decides today's pick, plus a guard so that an urgent request the test has not yet examined
    cannot jump. Its gain was not simulated.
  `[critic A7]`, `[critic delta A1]`
- **Each request is tested on top of its predecessor.** A batch is rebuilt from scratch at every dispatch and its
  members have no test run of their own, so sorting a batch discards no finished work. Sorting decides which of two
  conflicting requests is dropped from the batch; the batch is cut at the first conflict, so its depth can go either
  way. From code reading, not measured.
- **Blame when a batch fails.** Task 8528's bisect needs one fixed order that nothing jumps between rounds. It does
  not need arrival order. Its text does not yet say how the suspects stay ahead of new arrivals, and any urgency rule
  must rank below them.
- **Nothing in flight is displaced.** Inserting an urgent request ahead of work already merged or being tested would
  need new request states in the lane's bookkeeping and would throw away test runs of 12 to 75 minutes each. It is a
  large change and is not proposed.

**Two findings about the code itself.**
- The order is written out by hand in six places across three files, and one of them, the description on the queue's
  status view, is already wrong. Any change to batch order needs one shared function first. The same fact stated in
  six places is a defect whether or not urgency is adopted (`docs/code-quality.md`, heuristic 11, a single place for
  each fact). `[code F3]`
- The scheduler's priority tier does not reach the merge lane. The task's workflow can see the scheduler, but the
  interface between them has no way to ask for a tier, and the lane deliberately never imports the scheduler. Task
  8261 already specifies the remedy for the verify-admission gate: a small lookup function handed in from outside.
  `[code D10]`

**How contended the files are** (tasks waiting to edit each, at 15:44Z): `merge_lane/intake.py` 8,
`intake_buffers.py` 5, `suffix_graph.py` 12, `merge_lane/types.py` 45, `verify_runner.py` 50, `merge_lane/worker.py` 179, `workflow.py`
243. The pick lives in the least contended. Carrying the tier to it does not: that needs `workflow.py` and
`types.py`, or one line in `worker.py`. `[code F4]`, `[critic A6]`

**A standing ruling constrains all of this.** On 2026-11-10 every pending task that declares `merge_lane/worker.py`
was made to wait for task 8528 (176 tasks depend on it now). A new task touching that file either waits with them or
needs an exemption.

## 4. What "urgency" should mean here

| Candidate | Available when a request is queued? | Verdict |
|---|---|---|
| The scheduler's live priority tier (the task's own tier, any operator boost, and the best tier of anything that depends on it) | Yes, with a lookup function. It is kept current for running tasks: 18 of 18 at one snapshot | **Use this.** It is what task 8261 ruled for verify admission |
| The task's own tier | Yes | Wrong for a third of paused-window requests (50 of 151), which run at a higher inherited tier |
| Pinned | Yes | Add explicitly; pins are not folded into the tier |
| Best tier of the tasks that depend on this one | Already inside the first candidate | Adds nothing (2 of 151 requests) |
| What the request is blocking upstream through the files it holds | No. It needs a new index built by the scheduler on each pass, and must be re-read at each pick | Not ready; see below |
| Position in its PRD | No. Tasks carry no PRD identity | Covers few requests (3 were the last task of their PRD); waits on the PRD work |
| The hand-set lane | Yes | Keep for main-repair tasks, strictly ahead of everything else |

`[urgency §1, F1, F2, F7, F8]`

**On "what is it blocking upstream".** The brief's point stands: a request keeps its task slot and all its file locks
until it lands, so a low-priority request holding a contended file is urgent to land.

- **As a tier it does not work.** By a reconstructed upper bound, almost every request is blocking a ready task (127
  of 130 in the paused window, a median of 32 tasks each). Promoting a request to the best tier among the tasks that
  it alone is blocking would mark 74% of paused-window requests as high, against 57% by the scheduler's tier; in the
  earlier window 57% against 23%. A signal that marks most of the queue urgent orders little. `[urgency F3, F5]`
- **As a count it does separate requests.** At one live snapshot (15:34Z today, the scheduler's own lock table), the
  seven queued requests that held files were the only blocker of 90, 25, 10, 4, 3, 1 and 0 ready tasks. The request blocking 90 (13 of
  them high priority) was the one the operator had already moved to the "high" lane by hand; it has since landed.
  Three more queued requests held no files. Ranking by this count, weighted by tier and on the reconstructed upper
  bound, puts a different request first than ranking by the scheduler's tier does in 51 of 92 contested picks in the
  paused window. `[urgency F5, F9]`, `[critic A4]`, `[critic delta B5, B7]`
- **How often landing one request frees an important task is known only loosely.** Reconstructed upper bound: 22 of
  56 low- and medium-tier requests were the only holder between a high-priority task and its files. Recorded lower
  bound: 0 of 15 (4 of 20 in the earlier window). Those records cover only pinned tasks and tasks starved for days,
  27 waiting tasks behind the 15, which are the wide ones; 13 of the 15 rest on pin records, which are written as soon as a
  pinned task is blocked. So for pinned tasks in the paused window, landing one such
  request was never enough. For other
  high-priority tasks nothing is recorded. `[urgency F4]`, `[critic A4]`, `[critic delta A2]`
- **Its weak points.** It rewards declaring more files (correlation 0.41 between the count and the number of files
  held). Used alone it starves requests that hold no locks, which include hand-carried and recovered requests. Its
  benefit in landings is unmeasured.

This signal is not ready to rank requests, and whether it becomes ready depends on the lock hand-off follow-up, which
decides whether a waiting task can collect its files one at a time. That is the seam between the two follow-ups. If
it is adopted later it should enter the same single definition as a bounded addition, not as a second ordering.

**One definition for both gates.** Rank by arrival time plus a fixed number of hours for each tier below the top,
with the tier read at each pick; a hand-set "high" lane request stays strictly ahead. It means the same at verify
admission, where a waiting task also holds its slot and locks. While the queue is short, 1, 2 or 4 hours per tier and
no aging at all give identical results. When it is long and urgency only picks, the worst low-tier wait is between
unchanged and 40% longer than under arrival order. On the lane as recorded, earlier mix, sorting the batch as well
took it from 14.8 to 20.7 hours. `[replay F10]`, `[critic B7]`, `[critic delta B9]` (The September study's aging rule was different: a 4-hour cap
after which age wins.)

## 5. Design choices and their ramifications

"Today" means the simulated lane as recorded; "after" means after the three pending tasks with no re-verification.
Gains are per urgent request, paused window unless stated.

| Choice | Urgent requests | Everything else | Test runs per landing | The batching work's own targets | What could go wrong |
|---|---|---|---|---|---|
| **A. Do nothing at the merge queue** | Median 2.8 h to between 0.8 and 1.7 h from the merge-lane work, depending on whether re-verification stops | Same | 2.0 to between 1.5 and 2.35 | Untouched | The queue grows long again at 1.2 to 1.4 times arrivals, and urgent work waits in line as today |
| **B. Urgency with aging picks the next request; batches fill in arrival order** | 0.31 h sooner today; 0.04 h after (0.33 h if re-verification persists); 0.1 to 0.6 h if the queue grows long | Today: medium tier 0.7 h slower, low tier 0.4 h slower, worst low-tier wait 8.8 to 10.0 h | Today −3% (earlier mix +4%); after, within 3% | Untouched | Replaces the same-files rule: an overtaken request that then conflicts is rebased, and escalated to a person after three rebases. At one snapshot 17 of 105 pairs of queued requests changed a common file |
| **B′. As B, but only among requests with no older waiting request touching the same files** | None on the same-files test as it runs today; less than B once the test is narrowed; not simulated | Less than B | Not simulated | Untouched | Narrowing the test changes today's pick as well. An urgent request the test has not yet examined must not be allowed to jump |
| **C. Urgency also sorts the batch** | 0.53 h sooner today (0.08 h on the earlier mix, where weighted wait is 5% worse); nothing further after | Low-tier worst wait up on the earlier mix (14.8 to 20.7 h) | Today −4% (earlier mix +4%); after, within 3% | Changes which request is dropped at a conflict, so batch depth may move either way. Changes the letter of the "batches fill in submission order" decision | Needs the shared order function, in the most contended file. Must rank below a failed batch's suspects |
| **D. A batch takes only the top tier present** | Shares no failure with lower tiers | After: low-tier median 0.93 to 1.40 h | After: 1.48 to 1.53 | Shallower batches | A request left out of the batch can be tested on a base that will not be the one it lands on; needs part of task 7720 first |
| **E. Urgent requests tested alone** | **Worse today**: 2.8 to 4.0 h. Neutral after (0.83 to 0.90 h) | Worse today | Up today | Switches batching off for the urgent half of the queue | Harmful on today's lane |
| **F. Lanes derived from tier, with aging between lanes** | Largest gain today (0.69 h) | Low tier slightly faster today, slower after | Today −5% | Batches still fill "high" first | Main-repair work would queue behind all high-tier work. Reverses the documented rule for the "high" lane. Rejected by the September study. The lane is fixed at queueing |
| **G. Fastest host to the most urgent request** | 0.02 to 0.10 h on top of B (upper bound) | Negligible | Slightly down | Touches the host code task 8526 will edit | Little to choose between: usually one ready request and one free host |
| **H. Strict priority, no aging** | Largest gain when the queue is long (7.2 h, earlier mix) | Worst low-tier wait doubles (19.5 to 38.9 h) | Today, earlier mix: +25% | Fights the handling of failed batches (below) | Starvation. Already ruled out at verify admission |

`[replay results tables, F7, F8, F9; code §2, §3d]`, `[critic B6, B8, B9]`

**Priority lanes with aging between them** was a choice the brief asked about. In the form "derive the lane from the
tier" it is choice F. In the form "add aging between the two existing lanes" it was not simulated, and no starvation
of the normal lane by the "high" lane has ever been recorded (3 of 151 requests used the "high" lane). It would
change the documented rule that every "high" request is picked first, which exists so that a main-repair request
never queues behind requests that cannot pass until it lands. If it were wanted, main-repair requests would have to
stay strictly first, which means a third class. That is a ruling for you, not a small step, and it is not proposed.
`[critic A5]`

**Urgency picking and today's handling of failed batches.** Today, when a batch fails, the next batch is half as deep
and no single request is blamed. Under strict priority a failing low-priority request is never picked on its own, so
it is never isolated, and it fails batch after batch: over 30 runs on the earlier window, failed batches per run rise
from 105 to 214 and the worst-hit request fails 32 batches in a typical run. **With aging the effect is absent or
small**: failed batches go from 28 to 27 per run in the paused window and from 105 to 120 in the earlier one. Task 8528's bisect removes the effect
in either case. So the replay gives no safety reason to hold choice B with aging until task 8528. The reasons to wait
are in the code: task 7720 edits the same function, and task 8528 must first say where a failed batch's suspects
rank. `[replay F7]`, `[critic A3]`

## 6. Is it worth doing at all

- **Against.** At today's arrivals the gain is 0.04 to 0.33 hours per urgent request on today's mix of work, and up
  to 0.9 on the earlier mix if re-verification persists. A task whose own priority is high took a median 36 hours from ready to landed in the paused window (21 tasks, parent
  report), so this is a few percent of its delivery time at most. What stops important work is file locking, not
  merge order.
- **For.** The three pending tasks themselves are worth about 0.75 hours per urgent request on the same footing, so
  an urgency pick is about a fifth of them. The margin before the queue grows long is 1.2 to 1.4 times arrivals, and
  three known changes push toward it. The pick is one function in a lightly contended file, though the tier reaches it only
  through `merge_lane/worker.py` or `workflow.py` (section 7), and task 8261 has to build the tier lookup anyway.
  None of the three pending tasks is in progress (section 7), so the batching work may be some way off, and choice B
  is worth 0.14 to 0.31 hours per urgent request on today's mix of work on the lane that runs in the meantime (0.27
  to 0.41 on the earlier mix).
- **Unmeasured in its favour.** A request that lands sooner frees its files and its task slot sooner, and the replay
  cannot credit that.
- **Barely measured against it.** The rebases and escalations from overtaking an older request that touches the same
  files: one snapshot, 17 of 105 pairs.
- **It does not replace setting lanes by hand.** A request moved to the "high" lane by hand was picked within 1 to 15
  minutes; choice B's mean gain is a third of an hour.

## 7. Owners, rulings and sequencing

**Nothing owns this.** No task orders the merge queue by urgency, ages between lanes, chooses a host by urgency, or
picks batch members by urgency. No ruling forbids it. Three pending tasks own neighbouring pieces of the recording in
step 1 below: task 7925 (record where each request came from on the merge events, and add readers for batches), task
7810 (batch landings are under-counted) and task 5622 (the queue-depth field on the merge events always reads zero).
`[owners §D]`, `[critic A6]`

**Rulings and recorded decisions it touches.**
- An older request touching the same files goes first (task 1891, done). Choice B replaces it. Choice B′ keeps its purpose but has to narrow its test first, which changes today's pick. The
  September study asked for a log of the would-be pick before any such change.
- Batches fill in submission order (deep merge-ahead plan, decision of 2026-07-29). Its ground is that a batch must
  land in the order it was built, which any fixed order satisfies. Choice C changes the letter, not the reason.
- The "high" lane is reserved for main-repair and hotfix work, and every "high" request is picked first
  (`docs/task-authoring.md`). Choice F and aging between the lanes both reverse this. Choices B, B′ and C do not, as
  long as "high" stays strictly ahead.
- Tasks that touch `merge_lane/worker.py` wait for task 8528 (2026-11-10).
- Queue re-ordering was dropped from the merge-lane policy plan as worth 0.08 test runs per landing. That figure is
  about not re-picking a request that has just failed until main has changed. It says nothing about urgent work
  landing sooner.
- Priority with aging at verify admission was ruled on 2026-11-02 (task 8261). Its replay notes that the merge queue
  "was not examined".

**What is in the way** (status at 16:40Z to 16:52Z today):
- Task 7720 (reserve a batch's members for its test run) is marked **blocked** after its merge failed on new failing
  tests. Its request was queued again at 15:19Z; a batch of three led by it failed at 16:22Z and it has been picked
  again since. It edits the picking function, and task 8527 needs it.
- Task 8526 started at 15:33Z and went back to pending at 16:30Z without starting implementation. It is first in the
  pin queue and waiting for `config.py`, `harness.py` and `merge_lane/worker.py`. Task 8527 waits on it and on 7720;
  8528 waits on 8527. None of the three is in progress.
- Task 8261 is pending and has not been able to get its files since 11-02: 52 refusals over 9.9 days, blocked on
  `config.py` and `verify.py`. It was moved to the "high" merge lane at 16:02Z and pinned at 16:30Z today.

Task 8526 and the urgency definition (task 8261) are both held up by the file locking the parent study describes;
task 7720 by its own failing tests.
`[critic delta A5]`

**A sequence that does not get in the batching work's way.**

| Step | What | Size | Wait for |
|---|---|---|---|
| 1a | Record the lane on the event written when a request is queued | One line in `merge_lane/intake.py` | Nothing |
| 1b | Record the tier on the merge events, the members of each batch on the test-run event, and what an urgency pick would have chosen and whether the request it passed over touches the same files | Small, but in `merge_lane/worker.py` or `workflow.py` | An exemption from the 11-10 ruling, or task 8528. Fold with tasks 7925, 7810 and 5622 |
| 2 | One urgency function shared by both gates | Small | Task 8261, which specifies it |
| 3 | Urgency with aging in the pick, "high" lane strictly first: choice B, or choice B′ with the same-files test narrowed | B: the pick function and its tests. B′: also the same-files test in `suffix_graph.py` | Task 7720 (same function); step 2; and a way for the tier to reach the pick, which is one line in `merge_lane/worker.py` (needs the same exemption as step 1b) or a field carried through `workflow.py` and `merge_lane/types.py`. Task 8528 by preference, not necessity |
| 4 | One shared order function, then urgency in batch membership (choice C) | Medium | Tasks 8527 and 8528; then before or after the batch code moves to its own file |
| Not proposed | Urgent alone (E), lanes from tier (F), aging between the existing lanes, host by urgency (G), strict priority (H), inserting ahead of work in flight | | |

## 8. Recommendation for discussion

1. **Leave the batching work exactly as planned.** No urgency choice worth having needs it changed, and no urgency
   choice is worth delaying it for.
2. **Check the simulator at the next restart, at no cost, and read it at the arrival rate recorded.** The slot limit
   rises to 24 at the same restart. With task 8525 live and none of the three pending tasks landed, the simulator
   predicts a median wait of 1.0 to 1.8 hours while arrivals stay under about 63 a day, and 1.6 to 4.4 hours at 70 a
   day (the paused window had 52). Only a recorded median well above 1.8 hours at under 63 arrivals a day counts
   against the "after" rows in this report. Arrivals are counted as in section 2b: each queueing after a final result
   is a new request. The wait is from first queueing to landing, for landed requests. `[critic delta A4]`
3. **Do step 1a now, and sample the same-files overlap now.** The overlap can be sampled without any code change,
   by repeating the snapshot in section 3 at intervals (a read-only look at the queue's journal and git). Together
   with the tier the scheduler already publishes, that also shows what an urgency pick would choose at each sample.
   **Decide whether step 1b gets an exemption.** Only 1b can record the tier as read at each pick and the members of
   each batch; without it the cost of a failed batch to its members stays unmeasured. `[critic delta2 A1]`
4. **Adopt task 8261's definition as the single definition of urgency** for both gates, with pins added and the
   "high" lane kept strictly first. Leave "what is it blocking upstream" to the lock hand-off follow-up.
5. **Do not file the pick change yet; decide its form from the overlap samples (and the step 1b record if it exists by
   then).** Choice B is worth 0.14 to 0.31
   hours per urgent request on today's mix before the batching work lands, minutes after it if re-verification really
   stops, and 0.1 to 0.6 hours if the queue grows long. It is paid for by slower medium- and low-tier requests and by
   an unknown number of rebases. Choice B′ avoids the rebases, but as described it changes nothing, and the version
   that would gain something changes the same-files test that today's pick also uses.
6. **Decide choice C from the readout already planned** for seven days after the merge-lane work is deployed (task
   8532). If the queue is short, C is not worth a change in the most contended file.
7. **Do not do** E, F, G, H or aging between the existing lanes.

**Questions for you.**
- Should step 1b be exempt from waiting for task 8528? It is needed for the tier at each pick and the members of each
  batch, not for the same-files overlap, which can be sampled without it.
- When its form is decided, should the pick change go in before task 8528 or after it? Before gives urgent work
  something in the meantime; after avoids one more edit beside the batching work. Either way it needs task 8261
  first and, before 8528, the same exemption as step 1b.
- Do you want the same-files test narrowed to what each branch itself changed? That is a change to today's pick in
  its own right, and choice B′ depends on it.

## 9. Limits of this evidence

- Every wait figure for an ordering is simulated. Only one window is calibrated, with one tuned setting, on 115
  landings over 2.9 days in an unusual regime (projects paused, agent accounts capped).
- The lane after the pending tasks is modelled from the plan, and its headline row assumes re-verification stops.
- Simulated medians vary between runs by more than several of the ordering effects (section 2b).
- Choice B′, and aging between the existing lanes, were not simulated.
- Figures marked as the critic's re-runs come from one harness that nobody else reviewed (in `critic.md` and
  `critic-delta.md`). The same-files snapshot is one moment and 15 requests.
- The lane each request was in is not recorded; current metadata was used.
- "Blocking upstream" figures rest on a reconstructed readiness time that is an upper bound, and on one live
  snapshot.
- Figures that look alike but are not: the plan's 5.4-hour baseline (63 landings, 11-02 to 11-04, counted from the
  last queueing) and this study's 2.43 hours (115 landings, paused window, counted from the first queueing); the
  1.6-hour bound per affected request and the mean gains over all urgent requests; "urgent" here (the scheduler's
  tier, a pin or the lane) and "high priority" in the parent report's 36-hour figure (the task's own tier).

## 10. Team

| Seat | Model | Effort | Outcome |
|---|---|---|---|
| Replay | opus | high | Calibrated simulator, five orderings in three regimes. Did not separate the end of re-verification from the three pending tasks, and reported the failed-batch hazard from a single run |
| Code and tractability | opus | high | Decision-point map, rule-by-rule effects, sequencing; found the six hand-written copies of the order |
| Urgency definition | opus | high | Seven candidates measured; the live snapshot showing that a count of blocked tasks separates requests |
| Owners and rulings | sonnet | medium | Owner and ruling tables; found the September study the parent report missed |
| Critic of the assembled report | fable | high | First pass: 7 answer-changing and 18 minor corrections, with re-runs of the seat's simulator. It changed the headline comparison, the margin, the case for waiting on task 8528, the reading of "blocking upstream", and three proposals. Second pass: 6 answer-changing and 14 minor; it showed that the rule-keeping variant changes nothing as described, and corrected three of its own first-pass items. Third pass: 1 answer-changing and 10 minor; it found that the same-files overlap can be sampled without step 1b, and had nothing further once those were folded in |

Four evidence seats: 1.06M tokens, 44 minutes of wall clock (per-seat split not recorded). Critic: 0.30M tokens and
35 minutes on the first pass, 17 minutes on the second and 4 on the third (0.37M tokens of context by the end).

The lead's first draft made the errors the critic caught at the join: it credited the whole simulated improvement to
the three pending tasks, quoted a single-run figure as a finding, and joined an upper bound to a lower bound in one
sentence.

## 11. Corrections after discussion (2026-11-13)

- **The rebase cap was described wrongly in sections 1, 3 and 5.** What the code does
  (`suffix_graph.py::SuffixConflictTracker.bounce_conflicting_suffix_items`): a waiting request that conflicts with
  what is about to land is rebased automatically. If that rebase fails with a real conflict, the request is sent back
  as blocked **at once**, on the first occurrence. The cap of three (`merge_lane/worker.py::MERGE_BOUNCE_CAP`) only
  limits how many times the same branch may be rebased *cleanly* before it is sent back without another try.
- Recorded, 2026-10-25 to 2026-11-12: the cap was exceeded 0 times. Real conflicts sent a request back 53 times
  (50 tasks). Clean rebases are not recorded as events. Rule: `merge_finalized` events whose reason names the cap or
  a real rebase conflict.
- So the cost of overtaking an older request is not extra trips through the cap. It is that when two waiting requests
  really conflict, one of them has to be reworked whichever goes first; order decides which one. Who does that rework
  (the task's own agent or a person) was not traced.
- Leo's rulings on 2026-11-13: the recording task (step 1b) is to be hand-carried; the pick change goes in after task
  8528; nothing in flight is displaced.
- Further rulings and actions, 2026-11-13: priority is to be strict across tiers at both gates, ordered (tier, age),
  with promotion after a long wait as the backstop (recorded on task 8261 as an amendment); the recording work was
  filed as task 9908 and its hand-carry launched (session `carry-dark_factory-9908-3554830`).
