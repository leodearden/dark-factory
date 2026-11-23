# Replacing the project pause with weights between projects: findings, options and decisions

2026-11-12. Read-only study: five evidence seats, one lead, one critic. Evidence files are in
`/tmp/team-weights-1112/` (`host-map.md`, `rulings.md`, `allowance.md`, `mechanisms.md`, `pause.md`, `critic.md`,
cited below as `[file F#]`), each with re-runnable scripts in a directory of the same name. Nothing in any repository,
task store, queue, config or service was changed. No task was filed and no PRD written.

Code read: Dark Factory `main` at `7848173a7d` (the critic at `0f21f5e457`; no difference in the files used here).
The running orchestrators started 2026-11-10 between 20:56Z and 23:48Z and match `main` on everything used here.

**Terms used throughout**
- *Before the pause*: 2026-10-25 to 2026-11-08 16:00Z (14.7 days, all projects running). *During the pause*:
  2026-11-09 14:13Z to 2026-11-12 15:00Z (3.0 days; reify, solar-challenge and solar-challenge-platform halted).
- *Merge verify*: the full test run a branch must pass before it lands. Each project runs one at a time on the
  workstation and one on the laptop. *Task verify*: the test run a task does on its own work before queueing to
  merge. *Background run*: periodic test sweeps of the main branch.
- *Landing*: a task's branch merged to main.
- *Spend*: the notional price the Claude command-line tool reports per call. Nobody is billed it (the accounts are
  flat-rate). It stands in for whatever the usage limit really counts, which could not be determined.
- *Capped*: an account has hit a usage limit and serves nothing until its reset. Each account has a weekly limit
  and a 5-hour limit.
- *CPU pressure*: the kernel's reading of the percentage of time some program was waiting for a CPU.
- Two changes fell inside the pause and confound it: the number of task verifies allowed at once went from 1 to 2
  on 11-09 14:13Z, and new merge-queue code went live on 11-10 23:48Z ("the 11-10 deploy").

## 1. The answer in brief

1. **There are two different things to divide, and the pause acted most on the one nobody was looking at.** CPU is
   a rate: unused CPU is gone the next second, and the kernel can share it by weight with no queue. The agent
   allowance is a weekly budget: seven accounts each receive a weekly lump at a fixed hour, every account reached
   its weekly limit in every week measured, and what one project spends another cannot. The pause moved Dark
   Factory's share of the allowance from 25% to 61% and doubled its spend.
2. **On CPU the pause bought a faster merge verify on the workstation (about 43 to 36 minutes). The landings that
   bought are bounded at 2.5 to 5.6 a day.** Dark Factory's landings did rise, from 30 a day in the 72 hours before
   the pause to 36 and then 44 after the 11-10 deploy, but that rise cannot be split between the larger allowance
   share, the deploy, faster verifies and work in progress.
3. **The cost:** the three paused projects were landing up to about 63 tasks a day between them at their earlier
   rate; 210 have been forgone so far (a bound).
4. **Weights on CPU would recover a small part of the verify speed-up at a small cost.** Projection for the
   recommended design (a project factor on task verifies, background runs and services; merge verifies left alone):
   Dark Factory's workstation merge verify at about 39 to 41 minutes instead of 44, worth about one landing a day at
   most. Nothing changes on the laptop, where projects collide only through merge verifies, and nothing on the
   allowance. The weights need no new code. Whether they work at all is an open experiment that two pending tasks
   already own, and you answered "not now" to it on 10-13.
5. **The allowance is the harder and more valuable half.** Nothing divides it: every program takes the first open
   account in the same list, so a project's share is whatever it asks for. Before the pause 61% of the pool went to
   tasks now ranked low. Interactive sessions now draw on the same accounts, of the order of a tenth of the pool
   since 11-01 (a rough indication), and nobody has measured it.
6. **Memory cannot be weighted with what exists, and it is what holds dispatch.** 57 of the 69 times a scheduler
   declined to start a task during the pause were memory pressure, read from one host-wide signal. Whose memory it
   was is not recorded.
7. **The planned resume is unguarded.** Two projects resume in the same instant with no check on load: up to 38
   more agents and their verifies onto a busy host.
8. **Owners.** No task owns the weight values, a split of the allowance, or an urgency order across projects. Twelve
   decisions are yours before a PRD can be written (section 9).

## 2. What is on the hosts

**Nine orchestrators, not four.** `[host-map F1, unit table]`, `[allowance F2]`

| Project | State during the pause | Finished tasks/day before → during | Share of pool spend before → during |
|---|---|---|---|
| dark-factory | running | 22 → 40 (see note) | 24.7% → 60.6% |
| scoping-intake | running, never paused | 36 → 36 | 20.8% → 22.1% |
| stream-touch | running, never paused | 14 → 36 | 8.1% → 17.2% |
| solar-challenge-platform | halted 11-08 | 26 → 0 | 16.9% → 0 |
| solar-challenge | halted 11-08 | 21 → 0 | 14.7% → 0 |
| reify | halted 11-08; most of its backlog is also held by a gate task that waits for reify's alpha release | 17 → 0 | 12.6% → 0 |
| pump-web-ui | halted since 11-02 by its daily cost ceiling; 54 tasks pending | 4 → 0 | 2.1% → 0 |
| know-live, autopilot-video | no pending work | 0 | about 0 |

Rules. Finished tasks: distinct tasks with outcome "done" per day in each project's own event database. Spend share:
sum of reported cost per project, calls completed in the window, over the nine orchestrators' event databases plus
the memory service's own; one rule in both columns.
Note: Dark Factory's "before" is the two-week mean. By landings (a different count, used in section 3) it was 30 a
day in the 72 hours before the pause, 36 during the pause before the 11-10 deploy and 44 after; by finished tasks
27, 40 and 40. The two counts agree over the whole windows and differ by up to 4 a day over these short ones, so the
size of any step at the deploy is not firm. `[critic A7, delta DM3]`

- scoping-intake and stream-touch are nearly out of work: 22 and 9 tasks pending. Their combined share of pool spend
  by day was 49%, 53%, 28% and 17% from 11-09 to 11-12. `[critic M6]`
- pump-web-ui is a consumer waiting to return: you ruled the cost ceiling effectively off on 11-02 (task 8262,
  raising the default, is pending) and it still needs its scheduler resumed.

**One pool of seven accounts serves all of them**, the two solar projects included: their configs name no accounts
file, so they take the built-in default, which is Dark Factory's file. The critic confirmed it by loading six
projects' configs. `[host-map F11]`, `[allowance F1]`

**How work is grouped for the kernel today.** `[mechanisms F1, M1]`, read from the live system
- Each orchestrator service holds its own program and its agents. All nine have the default weight, 100.
- Each verify is its own group, placed *beside* the services, not under its project. It carries a weight by kind of
  work (merge verify 100, task verify 33, background run 10), the same in every project. Dark-factory, reify and the
  two solar projects do this; the other five run verifies inside their service.
- A weight on a project's service therefore reaches only its program and agents. For reify before the pause that was
  about a quarter of its CPU; verifies were about 70%. `[mechanisms F2]`
- Compiles go through one shared compiler-cache service and are charged to no project (about 0.9 cores on average).
- **Interactive sessions launched from T3 Code** (the desktop app most of your interactive sessions run in) **run
  at weight 10**, a tenth of an orchestrator. T3 Code sets this
  itself each time it launches sessions (its source: "low weights yield CPU and disk to the server under
  contention"); it is a default of the installed version, not an operator choice. Terminal sessions are at 100.
  Your 08-03 ruling that interactive work is deliberately favoured was measured on terminal sessions. `[critic A6]`;
  the lead confirmed the constant in `~/src/t3code/apps/server/src/process/agentScope.ts`.

**The laptop** takes merge verifies from four projects, each of which prefers it, and nothing arbitrates between
them there. Before the pause it ran 73% (reify), 87% (solar-challenge), 82% (solar-challenge-platform) and 53% (Dark
Factory) of each project's passed merge verifies, and each other project's merge verify running there at the same
time added about 4.8 minutes to Dark Factory's. `[critic A1]`, `[pause F3]`, `[mechanisms F6]`

## 3. What the pause bought and what it cost

**CPU: a faster workstation merge verify, visible in a comparison free of the two confounding changes.** Inside the
two weeks before the pause, whenever every account was capped and all agents stopped, Dark Factory's merge verify on
the workstation ran a median 31.4 minutes (22 runs) against 43.4 otherwise (251 runs). This measures "the host with
all agents stopped", Dark Factory's own included, not "the other projects removed", and 14 of the 31 capped hours
are one night. After the pause began the verify fell to about 34 to 36 minutes once the other projects had drained,
before either confounding change. `[pause F1, F2]`, `[critic M15]`

A model fitted on the 273 workstation merge verifies before the pause attributes 10.8 of their 43.9-minute mean to
the other projects. Per run or agent of another project going at the same time (± one standard error): merge verify
6.8 ± 1.0 minutes, task verify 1.4 ± 1.3, background run 0.7 ± 1.7, agent 0.7 ± 0.15. The task-verify and background
terms are not distinguishable from zero. By kind, the 10.8 minutes is: their agents 4.8, their task verifies 3.0,
their merge verifies 2.7, their background runs 0.3. The model explains 31% of the variation, and it is
observational: busy hours raise every term together. Fitted only on that window it gives 37.0 minutes for the pause
(35.9 and 36.6 were observed), one check, and 3.8 of the 37.0 come from Dark Factory's own growth observed during
the pause. `[pause F3, F4]`, `[critic M16]`

**Landings through that channel: 2.5 to 5.6 a day at most.** The merge queue has requests waiting more than 90% of
the time, yet before the 11-10 deploy the laptop was running a Dark Factory merge verify only about a third of the
time. So the queue is not limited by host CPU, and a faster verify helps only through the time each verify takes.
The bound assumes every minute saved becomes landings one for one; 2.5 uses the measured speed-ups, 5.6 assumes all
contention from other projects gone on both hosts. (Scaled from the 72 hours before the pause instead of the two
weeks, 3.2 to 7.3.) `[pause F5, F6]`, `[critic m11]`

**What Dark Factory did overall.** `[critic A7]`, from Dark Factory's own event database; landings = merges recorded
as done; spend = reported cost by completion time, all hours.

| Period | Landings/day | Spend/day |
|---|---|---|
| Two weeks before the pause | 23.7 | $423 |
| 72 hours before the pause | 30.0 | $396 |
| Paused, before the 11-10 deploy (1.4 days) | 36.4 | $757 |
| Paused, after it (1.6 days) | 43.5 | $847 |

Spend doubled while landings rose 21% and then 45% against the 72 hours before. How much of the rise is the larger
allowance share, the deploy, faster verifies or work started and not yet landed cannot be separated.

**What the pause was for.** Its recorded reason is to land the merge-queue policy work "as fast as possible". Three
of that work's six tasks have landed in four days (8524, 8525, 8531). The other three (8526, 8527, 8528) are pending,
one after another, and wait on a blocked task (7720). This study did not establish whether CPU is what holds them.

**Cost.** Before the pause the three projects landed 62.6 tasks a day between them (reify 17.4, solar-challenge
20.4, solar-challenge-platform 24.8). Forgone so far: about 210 landings in 3.97 days, which averages 53 a day
because they landed 38 while draining. Both are bounds (earlier rate times days). About 12 of the 210 would have been
high priority at each project's earlier mix. Solar-challenge was short of work before the pause (3.8 tasks running,
61 ready now), so its part is delay more than loss. Reify and solar-challenge-platform were limited by their own
task-verify queues, which depend on CPU. Reify has 138 tasks it could run now (11 high priority) despite its alpha
gate. `[pause F7]`, `[critic M3]`

**Allowance: the pause freed none and redistributed a lot.** Six of the seven accounts were already at their weekly
limit when it began. Every hour with all accounts capped during the pause ended at an account's weekly reset. The
pool supplies about $12.9k a week if every account reaches its usual weekly cap (projection: the sum of each
account's median spend at its cap). With three projects paused, Dark Factory spent about $1,050 to $1,150 a day
while an account was open, 56 to 62% of that; averaged over the whole pause, stalls included, 45%. It ran 8 to 9
agents against 18 task slots. So at the rate it ran it cannot consume the pool alone. `[allowance F7]`, `[critic M1]`

## 4. What is being divided, and what a weight can guarantee

| Resource | What kind of thing it is | What could weight it | What a weight guarantees | What it cannot do |
|---|---|---|---|---|
| Workstation CPU | A rate; unused CPU is lost at once | Kernel weights on groups | A proportional share among groups that want CPU, only when the host is full; unused share flows to others | Cap anything; help when the host has idle CPU; bound memory; protect a step from its own project's load; stop a starved test from timing out |
| Laptop CPU | Same, 16 threads, also your desktop | The same kernel weights, set in the laptop's own config files. Projects collide there only through merge verifies | Same | Same. A shared queue or lease across projects is ruled out (section 7) |
| Memory | A stock, reclaimed under pressure | Only a protected minimum per project group | That reclaim hits other groups first | Lower the host-wide pressure signal that holds dispatch. A ceiling on one project *raises* that signal and would hold every project |
| Agent allowance | Seven weekly lumps at fixed hours, all spent | Nothing today. A budget rule in each scheduler (section 6) | Depends on the rule | Increase the total. Unspent allowance is lost at each reset |
| The two merge-verify hosts | One merge verify at a time per host per project | Kernel weights when verifies collide | The CPU split during a collision | Decide who goes first. A cross-project turn-taking scheme is ruled out |
| Dispatch | Each scheduler holds itself on host-wide pressure | Nothing weight-aware exists | — | Today whoever has the lowest threshold backs off first (1 running task for every project except Dark Factory and reify, at 3), regardless of importance |

Sources: `[mechanisms M2 to M5 and table]`. Kernel sharing behaviour is taken from its documentation, not measured
on this host; the group tree and every current value were read live.

**What a ratio means, worked.** Two projects weighted 3 to 1 with one group per project, host full, each running one
merge verify and two task verifies: the first project gets 24 of 32 threads and its merge verify 14.5; the second
gets 8 and its merge verify 4.8, about the same as the first project's task verify. If desktop and interactive work
also wants 16 or more threads, all factory work is held to 16 and those figures halve. With today's flat layout
instead (reify at a third would be weights 33 / 11 / 3 against Dark Factory's 100 / 33 / 10), every other program on
the host is in the same sum, so absolute shares are lower, and the low project's task verifies and background runs
are squeezed far harder than its merge verify. (Projection.) `[mechanisms M2]`, `[critic M14]`

**The floor under any ratio differs by project.** A project squeezed to a few threads does not just run slowly: its
tests time out, and the timeout kills the whole test worker. The fix for that (task 5529) has 125 unfinished tasks
ahead of it, so plan as if it will not land soon. At equal weights before the pause, runs took (median / 90th
percentile, minutes; merge verifies are the whole verify on the workstation, the other two columns the test run
alone):

| Project | Merge verify | Task verify | Background run | Time limit per command: normal / merge verify on a cold cache |
|---|---|---|---|---|
| reify | 76 / 124 | 35 / 58 | 74 / 120 (16% already time out) | 120 / 180 |
| solar-challenge-platform | 28 / 39 | 25 / 40 | 30 / 50 | 90 / 120 |
| solar-challenge | 17 / 27 | 17 / 23 | 21 / 30 | 60 / 120 |
| dark-factory | 43 / 56 | 3 / 20 | 2 / 17 | 120 / 120 |

If run time scales inversely with CPU share, a third of the share takes reify's median merge verify to about 230
minutes, past both its limits; puts 22 to 36% of the three projects' task verifies past their limits; and half to
three quarters of their background runs. (Projection; the limit is per command and a run has several, so the
comparison is rough.) The one recorded measurement of starvation killing test workers is on task verifies, in Dark
Factory from 09-25 to 09-28: 14 crashes in about 138 task verifies against none in 76 merge verifies. So no single ratio is safe, and leaving merge
verifies out of the weight does not remove the floor: the ratio has to be set per project and per kind of run.
`[critic A3, delta DA2]`

## 5. The unit of priority, and the CPU design

A single number per project cannot say "Dark Factory's urgent work beats another project's routine work but not its
urgent work". Two layouts are possible on CPU.

**Layout A: weight each run individually (today's layout).** A run's weight = project factor × kind of work, and
optionally × the task's priority tier. All runs from all projects compete side by side.
- Can express an urgency order across projects.
- Needs no new code for the project factor. The three per-kind weights are a setting in each project's config, read
  when each run starts and reloadable without a restart; the critic confirmed this in the code. Verifies that run on
  the laptop take their weights from a separate file there, one per project. Agents are reached by a weight on each
  service. Adding the task tier needs code.
- A change takes effect from the next run to start; a run already going keeps its weight. Each project's config is
  tracked in its repository, so a change and its undo are each a commit on that project's main branch plus a reload.
- Weakness: a project's total share grows with how many runs it has going, so it is not a guaranteed share, and
  every desktop program and service is in the same pool.

**Layout B: one group per project, with the kinds of work inside it** (what task 3394 builds; pending since 09-08,
no prerequisites, blocked on file locks).
- A guaranteed share per project however many runs it has, and one place to measure each project's CPU, memory and
  pressure.
- Cannot express urgency across projects: the project weight dominates everything inside.
- Puts all factory work together against desktop and interactive work, a split that task says must be decided
  openly, not inherited.

**Should a project's weight apply to its merge verifies?** Your 09-07 directive, "merge verify is never throttled",
was recorded against cutting the merge test run's worker count, so it has been applied to CPU and not only to
queues. Leaving merge verifies at today's weight in every project, and weighting only task verifies, background runs
and services, keeps about three quarters of the contention a weight could act on (8.1 of the 10.8 modelled minutes;
it gives up the 2.7 that other projects' merge verifies cost) and stays inside the directive on any reading. It has
two consequences. It does nothing on the laptop, because the three paused projects send the laptop only merge
verifies. And 4.8 of the 8.1 minutes is the agents term, reached only by the weight on each service, which one seat
expects to do almost nothing because agents use little CPU. (Projection from the fitted model.)
`[critic A4, delta DA3]`

**What the recommended design is projected to give.** Same model and assumptions throughout; the landing figure is a
bound (each minute saved becomes landings one for one). `[critic delta DA3]`

| Design | Dark Factory's workstation merge verify, from 43.9 minutes | Dark Factory landings gained |
|---|---|---|
| 3 to 1 on everything, merge verifies included | 36.7 to 40.3 | 0.8 to 1.9 a day |
| 3 to 1 on task verifies, background runs and services | 38.5 to 41.2 | 0.6 to 1.3 a day |
| 1.5 to 2 to 1 on task verifies, background runs left alone, 3 to 1 on services | about 39 to 42 | 0.5 to 1.2 a day |
| The pause, same model and same channel (for scale) | 33.1 to 37.0 (about 36 observed) | 1.8 to 3.1 a day (2.5 to 5.6 when the laptop is counted as well) |

How far task verifies and background runs can be weighted down, as the share of runs the same projection puts past
their time limit:

| Project | Task verifies at 1.5 / 2 / 3 to 1 | Background runs at 1.5 / 2 / 3 to 1 |
|---|---|---|
| reify | 2% / 9% / 36% | 41% / 62% / 76% |
| solar-challenge-platform | 0% / 5% / 31% | 0% / 15% / 50% |
| solar-challenge | 0% / 3% / 22% | 2% / 10% / 55% |

Background runs account for 0.3 of the 10.8 modelled minutes, so leaving them alone gives up nothing measurable.
`[critic delta2 M-a, M-b]`

**Whether weights work at all is an open experiment with owners.**
- Dark Factory task 7112 (pending): why the per-kind weights did not restore reify's merge-verify time, with
  crowding by the nine services as the suspected cause and weighting the services as a candidate fix.
- Reify task 8827 (pending): the same lever as Layout A, as an experiment. It opens "Leo, 2026-10-13: 'not now'",
  requires a comparison whose measures are fixed before it starts, at least a week per setting, and says "The toggle
  is Leo's call."

**Recommendation for discussion.** Layout A with a project factor on task verifies, background runs and services,
and merge verifies left alone. Keep the factor between 1.5 and 2 to 1 on task verifies, leave background runs at
today's weight in all three projects, and set the service weight freely (agents have no time limit to breach). The projected prize is small: 2 to 5 minutes on the workstation merge verify, about one landing a day at
most. Run it as a real comparison after the projects have resumed at equal weights. Verify time was already drifting
by about half a minute a day before the pause, so a baseline week followed by a weights week cannot separate the
effect from the drift; alternate the setting in blocks of a day or two, and switch the service weight separately
from the verify weights so the two can be told apart. This is the experiment task 8827 specifies (it asks for a week
per setting, so the block length is yours), so it needs your go on that task or a replacement for it, and task 7112
should be folded in or closed. Build Layout B for measurement and for a guaranteed floor, and decide then whether
urgency across projects is still wanted on CPU.

If the experiment shows less than about 2 minutes (a threshold to fix before it starts; two weeks of alternating
blocks resolve about 2.5), the conclusion is that weights on task verifies and services are not worth building. It
would say nothing about merge verifies, which are the largest term per run (6.8 minutes each), the only contention
on the laptop, and held out by your directive: a 3-to-1 factor on merge verifies on the laptop alone projects to 0.8
to 1.6 landings a day, as much as the recommended design. Nor would it say anything about memory, the resume guard,
or project groups for measurement. `[critic delta2 M-c]`

How this combines with the first study's recommendations (a priority per PRD, one urgency value carried to every
gate): a project weight sits above them. Inside a project, the urgency value orders work; between projects, the
weight divides capacity. The seam with the merge-queue follow-up: a merge verify's place in the queue is that
study's; its CPU share while it runs is this one's.

## 6. The agent allowance

**How it is consumed.** `[allowance F2 to F8, M1 to M6]`
- Every program that runs agents (nine orchestrators, the memory service, the nightly census, evals) keeps its own
  view of the accounts in memory and takes the first open one in list order. Nothing coordinates them. 88% of agent
  time ran on the first open account even when others were open, and each cap was discovered separately by about
  eight programs.
- The scheduler starts a task without asking whether an account is open. A started task then waits holding its task
  slot and its file locks: 21% of slot time during the pause.
- The limit that stalls the pool is the weekly one. Resets fall at fixed hours (UTC; October times may be an hour
  off): Wednesday 10:00 (max-c) and 13:00 (max-f), Thursday 13:00 (max-d), Friday 12:00 (max-b), Saturday 17:00
  (max-e), Sunday 13:00 (max-g), Monday 21:00 (max-h). The longest gap is Monday 21:00 to Wednesday 10:00, 37 hours,
  where the long stalls fell. An account's weekly lump is spent a median 45 hours after its reset.
- Per account: spend at the weekly cap is about $1.97k (median) on max-d, max-e, max-f, max-g and max-h, varying 7 to
  14% week to week. Each account was capped 48% (max-c) to 82% (max-d) of the time before the pause.
- **Being mostly capped is the normal state.** Before the pause at least 4 of the 7 accounts were capped 90% of the
  time, at least 5 for 63%, at least 6 for 37%, all 7 for 10.2%. During the pause all 7 were capped 22.5% of the
  time. `[critic A5]`
  (The portfolio review's 0.83 hours in 30 days covered the same records but ended on 11-02, the day before the first
  stall with every account on its weekly limit, and undercounted 10-27 and 10-28. Your 11-02 position that single
  caps are negligible unless the whole pool is capped was taken on that figure. From 11-03 to the pause the whole
  pool was capped 17% of the time.) `[critic M4]`
- Before the pause, by current priority: 61% of spend went to low, 20% to medium, 15% to high. By role: implementer
  40%, architect 38%, reviewer 14%.
- Across the ten stalls of half an hour or more, 50 to 62% of the allowance whose absence caused the stall had gone
  to tasks now ranked low. That is an upper bound on what a rule favouring urgent work could have held back. It is
  not stall time removed: held-back work still needs allowance later, and a low task can be a prerequisite of a high
  one.

**Interactive sessions also draw on the pool** (Leo, 2026-11-12, during this study; no seat saw it). Since
2026-11-01 a local proxy lets interactive sessions overflow from their own account, account A, onto the seven
factory accounts in the weeks A runs out. Read by the lead on the live system:
- The proxy tries A first, then the factory accounts in an order of its own (max-c, max-g, max-f, max-e, then max-b
  and max-d, then max-h), filling each before moving on. The factory's own order starts with max-d.
- Its log covers the proxy's whole life and has one line per request with the account chosen. Of 29,749 requests
  served by 17:10Z today, 10,410 (35%) went to factory accounts, 7,116 of them to max-c. It records no tokens or
  cost. Interactive session transcripts do carry token counts, and nobody has joined the two.
- Nearly half of the factory-account requests (4,877) are from today, 11-12: account A ran out on 11-11 16:31Z, and
  this study and this morning's ran through the proxy onto max-c. Expect max-c to cap early this week.
- So the allowance left for the factory is not a fixed quantity. As a rough indication only, interactive overflow
  has taken of the order of a tenth of the pool since 11-01.
- It explains little of the unexplained low weeks. In max-c's low week of 11-04 the proxy sent it 2,046 requests, on
  a rough scale a fifth of a weekly lump; max-c capped just as low the week before with no overflow at all. max-b's
  week of 11-06 (1,271 proxied requests) is the other partly explained one. The four low weeks before that had no
  proxy traffic to speak of. `[critic delta DM1]`

**Can a cap be forecast?** On five accounts, to about 10%, from the factory's own records. Not yet on max-b and
max-c, which in some weeks cap at a third of the usual amount. No warning arrives before a cap and remaining quota
cannot be read. Several consumers record no spend: the memory service's reconciliation, the census, evals, and
interactive overflow.

**What follows for any design.** The weekly total is fixed and all of it is being spent, so a weight changes only
who spends and when. Total stall time will not fall much; what can change is *which* work is stalled. Project shares
can only be shares of what interactive work leaves. And because the pool is short nearly all the time, a rule that
waits for a "shortage" before it acts is either on almost always or acts too late (below).

**Options.**

| | What it is | Needs | Gives | Does not give |
|---|---|---|---|---|
| 1. Dedicated accounts | Each project lists its own subset of the seven | A config edit per project; the mechanism exists | Complete isolation | Urgency. Shares only in sevenths. A project with one account waits up to five days once it is spent, where the shared pool's worst gap is 37 hours. It breaks the "one account pool" rule a pending task (6628) asserts, and moves max-b and max-h out of the shared file |
| 2. A budget per project | Each scheduler compares its own spend over the last 7 days with its configured share. When over its share, it starts only tasks at or above a set tier (a task's own tier, or that of a task depending on it if higher) | Code in the scheduler, built on the trailing-spend query that the existing daily cost ceiling already uses. Each project reads only its own records | An over-budget project's routine work yields first; urgent work in every project keeps running | Full use of the pool when a project under-spends its share: the others hold routine work while accounts stay open, and that allowance is lost at the reset, unless a release is added (below). The share is a dollar figure that moves by two sevenths when max-b and max-h are switched on or off at Anthropic's side |
| 3. A forecast and a reserve | As 2, but restricting only when projected remaining allowance will not last to the next reset | Spend per account from every project's records, read by every scheduler; an estimate per account | Protection sized to the 37-hour gap | Accuracy on max-b and max-c, for unrecorded consumers, or for interactive overflow. A threshold computed from live spend is nearer the forbidden "derived from live load" form, and a shared view of spend is close to the "cap authority" you declined on 11-12 |
| 4. One urgency order for all projects | A single point admits the next agent call across projects | A shared admission point | Strict urgency | Any share for a low project. It is the shared-gate shape rejected for verifies |

**The open design points in option 2 are when the budget applies, and what releases it.** `[critic A5, delta DA1]`

When it applies:
- *All week.* Simplest to reason about. Because the pool is short nearly all the time, any "only in a shortage"
  form is close to this anyway: at least five accounts were capped 63% of the time.
- *Only inside the Monday 21:00 to Wednesday 10:00 gap.* A fixed calendar rule. 7 of the 10 stalls and 37.0 of their
  51.7 hours (72%) fell inside it; the other three stalls ended at the Saturday, Sunday and Monday resets.
- *Only when one account is left*: true 27% of the time (37% counting the time none is left), and at the eight
  stalls before the pause it left a median of about $1,000 of fleet spend to protect, against $12.9k a week. Too late.

What releases it, so that allowance is not thrown away when a project under-spends:
- *A release by calendar before each reset does not work.* (This was the lead's suggestion in the previous draft.)
  In 16 of the 19 weekly resets measured, the account about to reset had been capped for at least the 6 hours before
  it, and the scheduler cannot direct work to a particular account, so a lift before a reset would mostly spend
  other, freshly reset accounts.
- *A release on the program's own view of the accounts* (for example "at least three accounts open", true 37% of
  the time before the pause). The view is each program's own memory, which every program already consults on every
  call. It under-counts caps in a quiet or freshly restarted program, which errs toward releasing. It is a second
  yes/no lever of the shape the load ruling allows once.
- Every account reached its limit in every week measured, so no allowance has yet been lost to a budget. With
  projects idle it would be. The number to watch is allowance unspent at each reset.

**Two adjuncts examined and found weak on this evidence.**
- Choosing accounts evenly instead of first-in-list would have avoided at most 17 of the 23 5-hour caps before the
  pause, none of which caused a stall; the four stalls involving a 5-hour cap happened with one account left. No
  5-hour cap has occurred since 10-30. It would also retire the list order your 10-06 ruling relies on. `[critic M9]`
- The scheduler not starting a task when no account is open frees no capacity during a stall (nothing can run). It
  would let the scheduler choose again when an account reopens. Benefit unmeasured; the seam is with the lock
  follow-up. `[critic M10]`

**Recommendation for discussion.** Option 2, as one deterministic rule with fixed shares and no state shared
between projects. Start it without a release, either all week or only inside the Monday-to-Wednesday gap, and report
allowance unspent at each reset from the first week. Add the own-view release only if that report shows allowance
being lost. The rule needs no knowledge of what interactive work took. It is a second per-project spend rule beside
the daily cost ceiling you ruled effectively off on 11-02, so it should replace or absorb that rule, not sit beside
it. Option 1 is the fallback that needs no code.

## 7. Rulings and owners

**Owners.** No task in the four projects' stores (titles, descriptions and detail fields searched, all statuses)
owns the weight values, an allowance budget or pacing, dispatch gated on account availability, a guarded resume, or
the T3 weight. The task that builds one group per project (3394) excludes choosing values and hands them to "3394 or
its follow-up", never filed. Adjacent owners the PRD must name:
- Dark Factory task 7112 and reify task 8827: whether per-kind weights work (section 5).
- Task 8262 (pending) and your 11-02 ruling: the per-project daily cost ceiling stays as an opt-in tripwire.
- The plan `plans/scheduler-pause-halt-retirement-prd.md` and its task 5313 (pending): operator halts get an owner
  and a declared expiry and never resume automatically. The 11-08 pause is a halt of that class, and the three
  resume tasks are what ends it.
- Task 6628 (pending): one account pool from one file.

**Rulings that bound the design.** `[rulings R1 to R9]`; the critic confirmed each quote in its source.

| Ruling | What it forbids | Where the options stand (lead's reading) |
|---|---|---|
| One shared verify queue across projects, tried and rejected: "Reify's very long (30min+ cargo) verifies starved dark_factory almost entirely … Do not revisit this as an option" (re-affirmed 09-03 and 10-09) | Any shared slot, queue or lease that holds one project's verify for another's turn, on either host | Kernel weights are outside it: they hold nothing and the lower project always keeps a share. Allowance option 4 has the forbidden shape one gate earlier |
| Slot counts are "never derived from instantaneous loadavg" (reify collapsed to one slot under load on 05-28); one yes/no pressure hold in the verify runner is "the one allowed load-shaped lever" | Any count, and by extension any weight or share, computed from live load | Fixed weights, fixed shares and a calendar are fine. Auto-tuned weights are out. The budget test itself compares live spend with a fixed share, the same shape as the daily cost ceiling, and damps instead of feeding itself. A release on open accounts would be a second lever of the allowed shape: yours to allow. Option 3's threshold is nearer the forbidden form |
| Merge verify is "never throttled" (09-07) | Recorded against cutting the merge run's CPU, judged by its run time | A lower project weight on merge verifies is meant to slow them. Leaving them out avoids the question (section 5) |
| Interactive work is "privileged deliberately" (08-03); do not sweep it into project groups | Putting interactive sessions under a project | Stands for terminals. T3 sessions run at a tenth by T3 Code's default |
| "Do not propose cap persistence, a cap authority, or shortening the 5 s failover pause again without new evidence that a run was lost while a seat was open" (11-12) | A shared or remembered record of which accounts are capped | Option 2 needs none. A release on the program's own in-memory view reads no file and is outside the ruling's words. Remembering caps across a restart, sharing them, or option 3's shared view of spend need your ruling |
| max-b and max-h are never disabled or commented out, and stay at the end of the order (10-06); account A is interactive only; evals share the pool | A private reserve account; treating those two as dead | Option 1 and even account selection both disturb the order |
| The daily cost ceiling is an opt-in tripwire, effectively off (11-02) | — | Option 2 is a second spend rule in the same loop |
| Strongest model at PRD and architect (11-04) | Saving allowance with a cheaper model there | No option does this |

**No ranking of the projects is recorded anywhere.** The closest statements are the portfolio review's priorities
(income, the reify alpha, a client-visible demo), your 11-02 note that factory self-improvement is high value because
it compounds, and the pause itself.

## 8. Failure modes, measurement and migration

**Failure modes.**
- *A low project starves into failures*, not slowness (section 4). Guard: a ratio per project and per kind of run,
  and verify timeouts per project as a tripwire.
- *A project's own urgent work stuck behind its project weight.* Layout B cannot avoid it; Layout A with the task
  tier can. On the allowance, option 2's tier threshold is the escape, using the effective tier so prerequisites of
  urgent work are not held.
- *Feedback.* Weights must be fixed settings. A memory ceiling on one project would raise the host-wide signal and
  hold the favoured project too, so no memory ceilings until the hold reads per-project pressure.
- *A starved orchestrator.* A service weight also covers the scheduler and merge worker inside that service; the
  watchdog restarts an unresponsive one.
- *Memory is unweighted and unattributed.* A third of the minutes during the pause were over the hold's threshold.
  The host load recorder's two verify-memory measurements read zero throughout (broken). The recorded signs are
  heavy swapping with 44 to 92 GB still free, not exhaustion. `[mechanisms F4, F5]`
- *Simultaneous resume.* The three resume tasks only un-halt each scheduler and check nothing. The two solar
  projects' resume tasks have identical prerequisites, so they fire together: up to 38 more agents
  (solar-challenge's limit is the default 24, solar-challenge-platform's 14) with 61 and 223 tasks ready, plus their
  verifies. A resumed project is not held by the pressure check until it has one task running, and then every
  project reads the same lagging signal. About 112 held-back Dark Factory tasks become ready at the same moment, but
  Dark Factory's slots are already full, so those change its mix, not its load. The two that resume together are
  the light ones: about 2.4 cores between them before the pause, against reify's 9.4. Reify, the heavy one, resumes
  later, on further prerequisites. Precedents before the pause, both including reify: when three projects restarted
  on 11-07, 50 tasks started in the first hour and CPU pressure went from 34 to 61 in 75 minutes; on 11-04, with
  all four starting from nothing, agents went from 1 to 47 in 30 minutes. `[rulings §3]`, `[pause F10]`,
  `[critic M7, delta DM4]`
- *Work that belongs to no project*: the compiler cache, the memory service, the dashboard, the kernel's own memory
  reclaim. Weights between projects do not see it.

**How to measure it once built**, so the next decision needs no study:
1. Per project per day: CPU used, CPU pressure, memory pressure. Needs one group per project (task 3394); today a
   project's verify CPU cannot be recovered.
2. Per project per week: spend, share of the pool against its configured share, and hours stalled, by priority tier.
3. Pool: hours with every account capped; allowance unspent at each reset; interactive overflow per account
   (requests from the proxy's log today; tokens by joining it to session transcripts).
4. Merge verify duration per project against the other projects' concurrent runs: the fitted model as a standing
   report.
5. Verify timeouts and killed test workers per project (the starvation tripwire).
6. Landings per project weighted by priority: the stated goal, reported nowhere today.

**Migration, without a saturated gap.** Steps 1 and 2 are worth doing whatever is decided on weights.
1. Before the resume tasks' prerequisites land: set reduced task limits in the two solar projects' configs
   (solar-challenge has none of its own) and restart their orchestrators while they are still halted. The limit
   applies only at a restart, so raising it later is another restart, and now, with nothing running, is the cheap
   moment.
2. Put a delay on the second resume task so the two do not resume together. The task system has a "delayed" setting
   that does this with no code; it is documented as set when a task is filed, and whether it can be added to an
   existing task was not checked. The halt-retirement plan (task 5313) is where an owned, expiring halt belongs.
3. Resume at equal weights.
4. With your go on the experiment, alternate each resumed project's task-verify and background weights, and
   separately its service weight, between equal and the chosen fraction of Dark Factory's, in blocks of a day or two
   (section 5). No laptop change is needed while merge verifies are left alone.
5. Land one group per project; move the weights there if step 4 shows a guaranteed share is needed.
6. Build the allowance budget as its own piece of work. Until it exists, resuming the three projects lets them draw
   on the pool again; at their earlier demand that was 44% of it (projection from before the pause; demand has
   changed since). The interim lever is each project's task limit, which is coarse: it limits tasks, not spend.

## 9. Decisions for you

1. **The weights themselves.** No ranking of the projects is recorded. Suggested form: three classes, not tuned
   numbers. What ratio is safe depends on decision 2.
2. **Should a project's weight apply to its merge verifies?** Leaving them out keeps about three quarters of the
   contention a weight could act on and stays inside "merge verify is never throttled", but then weights do nothing
   on the laptop. Including them needs you to say the directive does not cover it, and then a ratio per project
   (reify about 1 to 1, the solar projects about 2 to 1, from workstation run times; their floor on the laptop was
   not examined). Either way the factor on task verifies is limited to about 1.5 to 2 to 1 by the same arithmetic.
3. **Go on the weight experiment?** It is what reify task 8827 specifies and what you answered "not now" to on
   10-13. The projected prize is small: 2 to 5 minutes on the workstation merge verify, about one landing a day at
   most. Alternating the setting in blocks of a day or two is the form that can detect it. Declining is a
   reasonable answer. It would leave this form of CPU weight out of the PRD; the merge-verify question (decision 2),
   the resume guard, memory and project groups stand on their own.
4. **Layout A now and Layout B later, or wait for Layout B?**
5. **An allowance budget per project (option 2): yes, and applied when?** All week, or only inside the
   Monday-to-Wednesday gap. And if under-spending projects leave allowance unused: may a program lift its budget
   when its own view shows several accounts open (a second yes/no lever; it reads no file and remembers nothing
   across a restart)?
6. **Should scoping-intake and stream-touch be inside the scheme?** They took 39% of the pool during the pause, 79%
   and 53% of it on low-ranked tasks, mostly on 11-09 and 11-10; their queues are now nearly empty and their share
   was 17% on 11-12. Your 11-02 note exempted stream-touch from parking as "cheap".
7. **Interactive sessions in T3 run at a tenth of an orchestrator's weight by T3 Code's own default.** Accept it, or
   change the constant in your T3 Code checkout? Deleting the setting does not hold.
8. **Factory work against desktop and interactive work** once project groups exist: equal weight (the default), or
   something else?
9. **Guard the resume now** (migration steps 1 and 2), independently of everything else?
10. **Is the pause still worth holding?** What it was for: landing the merge-queue policy work quickly; three of six
    tasks have landed, the rest wait on a blocked task, and whether CPU holds them is not established. What happened
    to Dark Factory: landings from 30 a day to 36 and then 44, spend doubled; the part through faster verifies is
    bounded at 2.5 to 5.6 a day and the rest cannot be attributed. What it costs: up to about 63 landings a day
    elsewhere, about 4 of them high priority.
11. **Memory**: accept that it stays unweighted for now, and fix the measurement first?
12. **Interactive overflow onto the factory accounts.** Is it a first claim with no limit, so project shares divide
    what is left? Should it be measured (the proxy's log joined to session transcripts)? Should the proxy's account
    order be chosen deliberately? Today its first overflow account is max-c, whose Wednesday 10:00 reset is the one
    that ends the factory's 37-hour gap, so interactive overflow lands on the account the factory most needs then.

## 10. What a PRD would need settled first

- Decisions 1 to 5, 8 and 12.
- The result of the weight experiment, or a decision to skip it.
- Whose memory causes the dispatch holds: one measurement during a daytime episode with the mechanisms seat's
  script, and the load recorder's broken verify-memory measurements repaired.
- Acceptance that spend is a good enough stand-in for what the weekly limit counts, and the cause of the low weeks
  on max-b and max-c (interactive overflow explains part of two of the six at most).
- How much of the slow-down from other projects a CPU weight can reach at all. Only the experiment settles it.
- For the allowance budget, the design rules that gate a PRD (`docs/legibility/design-invariants.md`): every hold
  names who ends it, what bounds it and how it behaves across a redeploy; a count that reads zero after a restart
  must not pass silently as "plenty".
- Dependencies named with their real state: task 3394 (project groups) has no prerequisites and is blocked on file
  locks; task 5529 (test timeouts) has 125 unfinished tasks ahead of it.
- Seams: the halt-retirement plan and task 5313 (owned, expiring halts); the daily cost ceiling (task 8262); the
  lock follow-up (starting tasks when no account is open); the merge-queue follow-up (merge verify order).

## 11. Limits of this evidence

- The pause window is three days in an unusual regime: account caps for a fifth of it, two restarts, a deploy.
- Spend is a stand-in for the real limit. Several consumers record none, interactive overflow among them, and every
  pool figure here excludes them.
- Priority tier is each task's tier today, not when it ran.
- The contention model is observational and two of its four terms are not distinguishable from zero. Kernel sharing
  behaviour is from documentation.
- Forgone landings are bounds: earlier rate times days.
- The proxy figures were read by the lead and re-counted by the critic; its size as a share of the pool is a rough
  indication resting on an unverified assumption (that account A's weekly limit is the size of a factory account's).
- The weights projection is small against day-to-day variation, and most of it rests on the agents term, about which
  two seats disagree.

## 12. Team

| Seat | Model | Effort | Tokens | Minutes | Outcome |
|---|---|---|---|---|---|
| Host map | sonnet | medium | 168k | 27 | Nine units, one pool for all, the T3 weight (cause left open); verify CPU per project unmeasured |
| Rulings and owners | sonnet | high | 299k | 41 | Verbatim rulings, resume mechanics, three constraints the brief lacked; missed four adjacent owners (7112 cited but not as an owner, 8827 listed but not connected, 5313, 8262's ruling) |
| Allowance | opus | high | 235k | 42 | Shares by project and tier, weekly-reset structure, forecast spread, piling onto one account |
| CPU and memory mechanisms | opus | high | 261k | 51 | Live group tree, worked share arithmetic, memory-hold analysis; floor computed for the wrong project |
| Pause counterfactual | opus | high | 238k | 41 | Verify speed-up separated from confounders, landing bound, cost, weights projection; counted four of ten spend records |
| Critic of the assembled report, first pass | fable | high | 361k | 50 | 7 answer-changing, 19 figure and 12 minor corrections; changed the CPU recommendation, the allowance trigger, the floor and the pause question |
| Critic, second pass on the rewrite | fable | high | 396k (cumulative) | 19 | 3 answer-changing, 6 figure and 9 minor corrections, all in text new to the rewrite; refuted the lead's calendar release and resized the weights projection |
| Critic, third pass on the folded edits | fable | high | 433k (cumulative) | 5 | Nothing answer-changing; 3 figure and 7 minor corrections, folded by the lead and not re-checked |

Lead errors the critic caught: recommending a trial with no baseline, a single ratio limit from one project's
numbers, a shortage trigger that is on most of the time, posing the pause question with one benefit channel against
all cost channels, a calendar release that would spend the wrong accounts, and quoting a projection for a design
the report no longer recommended.
