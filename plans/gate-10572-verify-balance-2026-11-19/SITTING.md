# Sitting: how task verifies and merge verifies share the hosts (gate task 10572)

Date: 2026-11-19. Status: RULED on 2026-11-20; the rulings and what was done are in `RULINGS.md` beside
this file. The text below is the sitting as presented, before any ruling.

Second draft. A critic seat refuted three parts of the first draft (why the queue is deeper than the task
cap, what reify's load explains, and what routing docs through the queue would buy); this version carries
its corrections. The rewrite itself has not had a second review.

Ten questions, each with a recommendation. An unanswered question is taken as accepted as recommended.
The answer sheet is at the end. Figures and their counting rules are in `EVIDENCE.md` beside this file.

## Words used

- **Task verify**: the test run an agent's work must pass before it asks to merge.
- **Merge verify**: the full test run on the merged result before main moves. Two can run at once: one on
  the workstation, one on the laptop.
- **Merge queue**: the requests waiting for a merge verify. **Normal lane** and **high lane** are its two
  priority levels.
- **Chain**: several queued requests verified together in one merge verify.
- **Re-verify**: a merge verify repeated because main moved while it ran.
- **Run slot**: one of the 36 places for a task in progress (`max_concurrent_tasks`). A task keeps its run
  slot, and its locks on the files it edits, until its merge lands.
- **Pinned**: placed by you at the front of the order in which tasks are started.
- **Red main**: main itself fails its tests.
- **Readout**: task 8532, a one-off report on a week of merge-queue figures, due about 26 November.
- **Reify, solar-challenge, solar-challenge-platform**: other projects whose orchestrators share the
  workstation and the laptop.

## Quick view

| # | The choice | Recommendation |
|---|---|---|
| 1 | What replaces the revoked rule "merge verify is never throttled" | No cap and no share on merge verifies. Record that with the queue this full, spare capacity goes to merge verifies |
| 2 | Task verifies were sent to the laptop on 15 November and slow the merge verifies there | A 3-day trial with Dark Factory's task verifies back on the workstation |
| 3 | Other projects' load on the two hosts | No change now; a decision task that opens when per-project CPU recording exists |
| 4 | After a restart the queue grows past the task cap, because recovered requests hold no run slot and no locks | Fix it: pin the filed lock task and file one task for the run slot |
| 5 | The task cap of 36: lower it, test it, or leave it | Leave it, no experiment |
| 6 | The readout will recommend switching chains off, by its own rule | Amend it to report and ask, not recommend the switch-off |
| 7 | How documentation commits reach main once the narrow docs gate lands | Through the queue in the normal lane; a new PRD's tasks wait on its landing by a dependency, so no person waits |
| 8 | The task that makes a docs-only move of main cheap | Cut it loose from its slow prerequisite, pin it behind the docs gate, keep its 7 log-only days |
| 9 | The pinned task whose finished plan is built on the revoked rule, and the documents that state it | One dated paragraph on the task; reword five documents |
| 10 | The nightly memory census has failed to land for four nights | Repair it now |

## What the evidence says

Window: 12 to 19 November, 7.8 days.

1. **The queue is far from draining.** 77 requests wait; the middle one has waited 14.5 hours. Both hosts
   run merge verifies 83 to 88 percent of the time. Task verifies already outnumber merge verifies by 1.0 to
   2.4 to one every day.
2. **Landings are limited by how long a merge verify takes, and the laptop is where it got slow.** Before
   the fleet pause (5 to 7 November) a merge verify took 19 to 20 minutes on the laptop and 39 to 41 on the
   workstation. On 17 and 18 November it took 40 to 44 on the laptop and 43 to 47 on the workstation. The
   workstation is roughly back where it was; the laptop has doubled. The laptop carries about half of all
   landings. (On 19 November the workstation jumped to 94 minutes. That day two more projects resumed and
   Dark Factory restarted; about half the jump is unexplained.)
3. **What changed on the laptop.** On 15 November Dark Factory's task verifies began going there, up to
   three at a time: 2 to 5 hours a day became 9 to 15. From 16 November reify's task verifies went there
   too, 14 to 21 hours a day. Reify's merge verifies were already there before the pause. A Dark Factory
   laptop merge verify takes 38 minutes when few Dark Factory task verifies overlap it, 49 with partial
   overlap and 56 with a full one (27, 15 and 6 verifies; observed, not an experiment).
4. **Dark Factory's own agents are a small cost.** Only 2 to 3 of the 36 run slots hold a task being
   written or reviewed; the rest wait to merge. Agent work adds an estimated 3.5 to 5.5 minutes to a
   47-minute workstation merge verify.
5. **Chains pay, but most verifies are still single-request.** A single lands 0.85 requests per verify-hour;
   a chain of five or more lands about 2.6 (the critic could not reproduce this second figure). Since
   15 November, 102 of 199 verifies were singles although about 50 requests were waiting. 47 of them were
   requests re-tested alone after their chain failed, as ruled on 18 November: 36 hours, a quarter of
   completed verify time.
6. **Failures are the second limit.** Failed verifies plus the verifies they cancel took 43 percent of
   verify time. Most were not the request's fault: 51 of 75 failed singles later landed. Main was red twice.
7. **Main moving under a passed verify cost about 11 percent of verify time.** 66 re-verifies: 32 caused
   only by the queue's own landings (19 hours), 26 only by direct commits (12 hours), 7 by both. At most 19
   of the 66 were docs-only direct commits.
8. **Restarts cost little host time (1 percent) but they are what made the queue deep.** In the one
   44-hour run with no restart the queue peaked at exactly 36. Each of the four restarts since 17 November
   put the waiting requests back in the queue holding no run slot and no file locks, and the new run then
   filled all 36 run slots with fresh tasks. The queue reached 67, 85, 80 and 85.

## Question 1: what replaces "merge verify is never throttled"

**The choice:** whether merge verifies get a cap, a share or a lower priority now that the rule is revoked.

**The issue.** You revoked the rule so that task verifies outrun merge verifies and several requests wait
for each merge verify. That is true today by a wide margin with no limit on merge verifies. Chain size is
currently set by the expected pass rate (3 to 4 requests), not by how many requests wait.

**Option A: no cap; record the direction.** The ruling reads: "No rule protects merge verifies from task
verifies, and none limits them. While more requests wait than one chain can take, spare verify capacity
goes to merge verifies. If merge verifies start with shorter chains than allowed because too few requests
wait, capacity moves back toward task verifies." The readout reports that last figure once; no standing
watcher is built. Cost: none.

**Option B: cap or share merge verifies now.** With 77 waiting and both hosts full, it lands fewer
requests. New code. It solves a shortage that does not exist.

**Recommendation: Option A.**

## Question 2: task verifies on the laptop

**The choice:** whether Dark Factory's task verifies keep going to the laptop.

**The issue.** This is the gate's question in its live form: one host, shared between the two kinds of
verify. The laptop has 8 cores. Since 15 November it runs a Dark Factory merge verify (16 test workers) and
up to three Dark Factory task verifies (8 workers each), beside reify's. Laptop merge verifies have doubled
(points 2 and 3). The setting was made so task verifies would wait less; they now wait 10 to 16 minutes,
and with the queue this full a faster task verify lands nothing sooner.

**Option A: 3-day trial with Dark Factory's task verifies on the workstation only.** One setting
(`verify_task_host_policy: local`), applied without a restart and reversed the same way. Read after 3 days:
laptop and workstation merge-verify minutes, task-verify wait, landings a day. Expected from point 3: laptop
merge verifies about 10 to 18 minutes faster. Cost: the workstation takes 9 to 15 more hours of task
verifies a day, which may slow its merge verifies by a few minutes; task verifies wait longer.

**Option B: trial with one laptop task verify at a time instead of three.** Smaller change, smaller effect,
and the setting is shared with the workstation's count, so it needs an edit on the laptop.

**Option C: leave it.** Laptop merge verifies stay near 40 to 55 minutes.

**Recommendation: Option A.** If you also want reify's task verifies off the laptop, say so: that is a
reify setting, and you have a reify gate (8430) pending on it.

## Question 3: other projects' load

**The choice:** whether Dark Factory's verifies get more of the hosts than other projects' verifies.

**The issue.** Four projects' verifies share each host with equal CPU weight, so Dark Factory's share
shrinks each time a project resumes; two resumed on 19 November. Reify needs 28 to 34 merge-verify hours a
day on the same two hosts. The 12 November study priced unequal weights at 2 to 5 more Dark Factory
landings a day, taken from the others. You declined a follow-up on the laptop slowdown today. Nothing yet
records CPU use per project; the task that would (10456, high) has failed three attempts.

**Option A: no change now, and a decision task that opens when recording exists.** A task that depends on
the recorder task and asks you this question again with a week of measured CPU per project. Cost: equal
shares continue.

**Option B: trial a higher weight for Dark Factory's verify group now.** Can be set without a restart.
Gain 2 to 5 landings a day by the earlier study; the other projects slow by about as much.

**Recommendation: Option A.**

## Question 4: the queue grows past the cap at every restart

**The choice:** whether a request recovered after a restart holds a run slot and its file locks again.

**The issue.** By design a task keeps its run slot and file locks until its merge lands; that is what
balances new work against landings. A restart breaks it (point 8). Consequences: a queue of 77 instead of
about 36; requests that wait a median 14.5 hours; a burst of 10 to 19 new agents after each restart (the
five workstation merge verifies that overlapped a burst took 102 minutes against 47); and 20 of 36
recovered tasks had a file they were editing locked by another task. You ruled on 14 November that the
locks must be restored; that task (10093, high) is filed but not pinned. No task covers the run slot.

**Option A: fix both.** Pin task 10093, and file one task so a recovered request counts against the 36 run
slots until it lands or fails. Effect: the queue stays near 36 and restarts stop starting a burst. Cost in
landings: none while the hosts are full. Risk: after a restart, few or no new tasks start until the queue
drains below 36.

**Option B: fix the locks only (pin 10093).** The queue still grows at each restart.

**Option C: leave it and add a separate limit on queue depth.** A second mechanism for what the run slot
already does.

**Recommendation: Option A.**

## Question 5: the task cap

**The choice:** lower `max_concurrent_tasks` from 36, run an experiment to price it, or leave it.

**The issue.** The cap was proposed as a way to free CPU for merge verifies. It does not do that: only 2 to
3 run slots hold active work (point 4). What the cap sets is how many requests can wait, once question 4 is
fixed. Removing all agent work would save an estimated 3.5 to 5.5 minutes per workstation merge verify;
going from 36 to 24 would save about a third of that. The cap needs a restart to change. One caution on
the estimate: it has no term for memory pressure, which is now the commonest reason the scheduler holds a
new task (338 of 635 holds in the window).

**Option A: leave it at 36, no experiment.**

**Option B: a 4 to 6 day experiment without a restart**, alternating days with a pressure setting that
behaves like a cap of 24. Cost: 10 to 20 fewer new requests on each low day.

**Option C: lower it to 24 now.** The queue ran empty at 24 on 15 November, which is why it was raised.

**Recommendation: Option A.**

## Question 6: the readout will recommend switching chains off

**The choice:** what the readout does when one of its limits is exceeded.

**The issue.** The readout's instructions say: if any of its limits is exceeded, raise it to you and
recommend switching chains off. Four limits are exceeded today: discarded verdicts per landing (0.58
against 0.1, though 195 of 211 were discarded before a host was used); a re-verify following a re-verify
(0.31 against 0.3); median wait to land (4.9 hours against 1); workstation merge-verify hours a week (148
against 60). The limits were set from the days the other projects were paused. The evidence here says
chains raise landings per verify-hour, and the wait is long for the reason in question 4.

**Option A: amend the readout to report every figure and ask you, without recommending the switch-off**,
unless a failure is traced to chains themselves (a red main caused by a chain, or a broken accounting
check). Add three figures: verifies that started with a shorter chain than allowed for lack of waiting
requests (question 1); verifies spent re-testing requests alone after a failed chain; re-verifies caused
by the queue's own landings.

**Option B: leave it.** On 26 November you get a recommendation to switch chains off.

**Recommendation: Option A.**

## Question 7: documentation commits

**The choice:** how docs reach main once the narrow docs gate (task 5411) lands and direct docs commits
stop, as you ruled today.

**The issue.** 101 commits went directly to main in 14 days: 44 from PRD authoring and decomposition, 13
from the nightly legibility job, 12 config edits, 11 study and gate records, 21 others. Three facts bound
the design:

- Routing docs through the queue does not by itself stop re-verifies. Whoever moves main, the request
  verifying on the other host re-verifies in full. Question 8 is what removes that.
- The docs gate keeps queue order. A docs request waits its turn: a median 6.8 hours in the normal lane
  now, less once question 4 shortens the queue.
- A PRD decomposed while its files are still queued fails silently today. The step that copies capability
  checks onto the new tasks looks for the PRD's check file in the main checkout, finds nothing and skips.

**Option A: normal lane, and the tasks wait by a dependency.** The PRD session puts its files on a branch,
submits it under a small carrier task, and files the PRD's tasks depending on that carrier. They start
when the PRD lands; nobody waits at a keyboard. Needs three tasks behind the docs gate: the check-copying
step reads a named branch worktree; the PRD, review and survey skills and `CLAUDE.md` "Working in the main
checkout" switch to this route; the nightly legibility job submits to the queue as the memory census does.
Config edits stay direct (3 of the week's re-verifies). Cost: a new PRD's tasks start hours later than now.

**Option B: as A, in the high lane.** Docs land in 1 to 2.5 hours. It reverses the 9 October ruling that
docs keep their place in line, and jumps docs ahead of code.

**Option C: stop direct docs commits now, before the docs gate lands.** A docs branch would ride a chain,
so the verify cost is small, but every route above has to be built first and the wait is the full queue.

**Recommendation: Option A.** Until the docs gate lands, direct commits continue as now; I propose no
interim rule, because nothing would enforce one. To bring the date forward, move the docs gate and its
prerequisite (26th and 27th in the pinned order) up behind the first six and give both the high lane.

## Question 8: making a docs-only move of main cheap

**The choice:** when task 6661 lands and whether it keeps its trial period.

**The issue.** Task 6661 makes the queue look at what moved main. If the change is docs-only, the request
that was verifying runs only the docs checks instead of a full merge verify. It covers at most 19 of the
week's 66 re-verifies, about 12 verify-hours a week. As filed it only records what it would have done for
7 days, then you judge (your 9 October ruling). It waits on a large refactoring task (9145) that has no
plan yet, only because both edit one function. A separate decision task (8258) still says "adopt the docs
gate after its 30-day trial", which you removed today.

**Option A: cut the link to task 9145, pin 6661 behind the docs gate in the high lane, keep the 7
log-only days, and re-word task 8258 as the decision to switch 6661 on.** Cost: about 12 verify-hours for
the week of logging. Need for the logging: unlike the docs gate, this skips the full test run on a tree
that contains code.

**Option B: as A but switch it on at landing.** Saves that week. It means rewriting the task (its text is
log-only throughout) and settling what happens to a verify stacked behind the re-verifying request.

**Option C: leave as filed.** It lands weeks after the docs gate.

**Recommendation: Option A.**

## Question 9: the task built on the revoked rule, and the documents

**The choice:** what to do with task 5139's finished plan and the places that still state the rule.

**The issue.** Task 5139 (pinned) stops the merge queue's "is main itself broken" check from waiting
behind task verifies, and limits how many of those checks run at once. Under question 1 its design is
still right; its stated reason ("the 7 September directive forbids a cap") is not. Its plan must be
re-checked in any case: the plan no longer matches the task text, and the functions it edits moved when
the merge worker file was split today. The rule is written as standing policy in five documents, in code
comments and three tests, and in two other open tasks (3589 and 3595).

**Option A: one dated paragraph on task 5139** with the new reason and the functions' new file; one dated
line each on tasks 3589 and 3595; reword the five documents in this session's single closing commit, except
the plan document 5139 itself edits, which gets a dated note only. Comments that describe what the code
does stay. The early-start tasks and the done task 5203 need nothing.

**Option B: rewrite task 5139's text.**

**Option C: leave everything.**

**Recommendation: Option A.**

## Question 10: the nightly memory census

**The choice:** repair it now or not.

**The issue.** It lands through the queue, and its request has been abandoned four nights running. The
installed timer unit dates from September and lacks the 20-hour landing limit added on 14 November, so it
gives up after 3 hours. Repair: re-run its install script. No orchestrator restart. Even repaired it may
miss nights while the queue's slowest tenth waits over 20 hours.

**Recommendation: go.**

## Not asked

One direct commit today was made in a way that skips the pre-commit checks (a PRD amendment for task
10782). No damage found; question 7 ends the route.

## After your rulings

As recommended this is a ruling plus five new tasks, not a PRD: one for the run slot (question 4), three
for docs routing (question 7), one decision task (question 3). Amendments to tasks 5139, 3589, 3595, 6661,
8258, 8532 and 5411. Re-pinning as ruled. The laptop trial setting applied and read after 3 days. One
closing commit of this folder and the reworded documents. Gate 10572 closed citing the ruling.

## Answer sheet

Reply with the keys you want to change; anything unlisted is accepted as recommended.

| Key | Recommended | Your answer |
|---|---|---|
| Q1 | A: no cap; record the direction | |
| Q2 | A: 3-day trial, Dark Factory task verifies on the workstation only | |
| Q3 | A: no change; decision task behind the recorder | |
| Q4 | A: pin task 10093 and file the run-slot task | |
| Q5 | A: cap stays 36, no experiment | |
| Q6 | A: readout reports and asks; three figures added | |
| Q7 | A: docs through the normal lane, tasks wait by dependency; move the docs gate up | |
| Q8 | A: cut 6661 from 9145, pin it, keep 7 log-only days, re-word 8258 | |
| Q9 | A: dated paragraph on 5139; reword five documents | |
| Q10 | go | |
