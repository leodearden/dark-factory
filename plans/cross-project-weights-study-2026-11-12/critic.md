# Critic of `/tmp/team-weights-1112/REPORT.md` (2026-11-12, written about 16:50Z)

Read-only. Code read: Dark Factory `main` at `0f21f5e457` (the report read `7848173a7d`; `git diff --stat` between the
two over `verify.py`, `config.py`, `defaults.yaml`, `usage_gate.py`, `scheduler.py` is empty). Scripts and outputs
behind every figure below are in `/tmp/team-weights-1112/critic/` (`capcount.py`, `stallcheck.py`, `sessioncaps.py`,
`joins.py`, `tasks_check.py`, `closure.py`, each with a `*_out.txt`). They read the allowance seat's `raw.pkl`, the
event stores and the task stores read-only. Nothing was changed anywhere.

Severity: **A** changes an answer or a decision. **M** a figure or wording to fix. **m** minor.
Counts: **A 7, M 19, m 12.**

---

## A findings

### A1. Layout A "needs no code, a config edit and a setting on each service" leaves the laptop unweighted, and the laptop is where most of the affected verifies run

- Where: section 1 item 3 ("can be tried today with no new code, because each project's verify weights are already a
  per-project setting"); section 5 Layout A second bullet; section 8 migration step 3.
- What is wrong: the claim is true for the workstation and false as a description of the trial. A verify that runs on
  the laptop takes its weights from a separate file on the laptop, not from the project's
  `dark-factory-orchestrator.yaml`. Each project's laptop runner names its own config
  (`verify_runners[].config_path`, e.g. `/home/leo/.config/orchestrator/reify-laptop.yaml`), and none of the four
  laptop files sets `verify_cgroup_cpu_weight_*`, so every project's laptop scopes run at 100/33/10 whatever the
  workstation file says. Step 3 as written therefore weights the minority of the paused projects' merge verifies.
- Evidence:
  - `orchestrator/src/orchestrator/verify_runner.py` (`['--config', self._config_path]`); reify yaml
    `verify_runners: - name: laptop ... config_path: /home/leo/.config/orchestrator/reify-laptop.yaml`.
  - `ssh leo-laptop 'grep -n verify_cgroup_cpu_weight ~/.config/orchestrator/*-laptop.yaml'` returns nothing; four
    files exist (dark-factory, reify, solar-challenge, solar-challenge-platform), all with `verify_use_cgroup_scope: true`.
  - Share of passed merge verifies that ran on the laptop before the pause (2026-10-25..11-08 16:00Z, `merge_verify`
    events, passed, over 60 s): reify 202 of 275 (73%), solar-challenge 253 of 291 (87%), solar-challenge-platform
    287 of 351 (82%), Dark Factory 308 of 581 (53%). `critic/joins.py others_verify`.
  - The pause seat's own projection states the limit: workstation weights give "none of the laptop relief (F3: 4.75
    min per foreign laptop merge verify, mean 0.57 concurrent)" `[pause F9 assumption 6]`, and in its landing bound
    the laptop term (0.58 x 0.19) is nearly as large as the workstation term (0.41 x 0.33) `[pause F6]`.
- What I could not refute (so the lead can keep it): on the workstation the three weights are read at each scope
  spawn (`orchestrator/src/orchestrator/verify.py::_resolve_scope_cpu_weight`, called where the scope kwargs are
  built), are hot-reloadable (`config.py` `RELOADABLE_FIELDS` lists all three), accept any integer 1..10000
  (`Field(default=100|33|10, ge=1, le=10000)`, no cross-field validator), and all nine units run Dark Factory's code
  with their own yaml (`systemctl --user show`: `WorkingDirectory=/home/leo/src/dark-factory`, `--config
  /home/leo/src/<project>/dark-factory-orchestrator.yaml`). `systemctl --user set-property
  orchestrator-<project>.service CPUWeight=N` is all a service weight needs (the cpu controller is delegated and
  already live under `app.slice`).
- Proposed text for step 3: "Before resuming, set each resumed project's three verify weights to the chosen fraction
  of Dark Factory's **in two places: the project's `dark-factory-orchestrator.yaml` (verifies on the workstation) and
  its file under `~/.config/orchestrator/` on the laptop (verifies on the laptop, which were 73 to 87% of these
  projects' merge verifies before the pause)**, and set `CPUWeight` on each orchestrator service. No code. A verify
  already running keeps the weight it started with."
- Proposed addition to section 1 item 3: "Workstation weights reach only verifies that run on the workstation. The
  laptop, which carried 58% of Dark Factory's landings before the pause and where each other project's merge verify
  added 4.8 minutes, needs the same weights set in its own four config files."

### A2. The Layout A trial cannot answer the question it is set, and it re-proposes a toggle with an existing owner and a recorded "not now"

- Where: section 5 recommendation ("treat it as the experiment that tells us whether weights are a strong enough
  lever"); section 8 migration steps 3 and 4 ("Watch ... for a week. This is the test of whether weights are a strong
  enough lever"); section 1 item 7 ("Nothing here is owned"); section 11 last bullet ("a pending reify task that
  proposes exactly that", unnamed).
- What is wrong, design: step 3 sets the weights at the moment the projects resume. There is then no period with the
  projects running at equal weights on the current merge lane to compare against. The only "before" periods are the
  pause itself and the two weeks before it, which ran a different merge lane, a different verify slot count and a
  different task limit. That is the confounding the first study could not undo (`pause.md` F1 lists four changes in
  three days). A week of watching gives a level, not an effect.
- What is wrong, owners:
  - reify task **8827** (pending, low): "Experiment (needs Leo's go to toggle): favour LOCAL merge verify via
    verify_cgroup_cpu_weight_merge 100→300". Its description opens "Leo, 2026-10-13: 'not now'", requires "a
    pre-registered A/B: weight 100 vs 300 over >= 1 week each, comparing local merge-verify phase walls bucketed by
    concurrent-verify count ... plus the cost side, i.e. task-verify and other projects' wall in the same windows",
    and carries "CONSTRAINT: do NOT call reload_config or edit the yaml weight yourself. The toggle is Leo's call"
    and "Cross-project policy point: a higher reify merge weight outranks other projects' work in app.slice". This is
    the same lever (a per-project, per-kind weight set by config) with a ruled shape for the measurement.
  - Dark Factory task **7112** (pending, low, filed on Leo's 2026-10-18 ruling): investigate why the per-kind weights
    did not restore reify's local merge-verify time; "If confirmed, propose the lever: weight the orchestrator service
    cgroups, or give merge verifies a dedicated slice with a higher weight." The report cites it as "an earlier task
    ... noted"; it is a live, unstarted investigation whose work item is the measurement the trial needs and whose
    candidate lever is the report's "weight on each service".
- Evidence: `critic/tasks_check.py show 7112`; reify store, task 8827 (read with the same script pattern); both
  statuses re-read 16:40Z.
- Proposed text (section 5 recommendation and migration): "Resume first at equal weights under the guard in step 1,
  and hold that for a baseline (the reify experiment task 8827 asks for at least a week per arm; a shorter arm is
  your call). Then set the weights and compare the same measures, bucketed by how many other verifies ran at the same
  time. This is the A/B that reify task 8827 already specifies and that you answered 'not now' to on 10-13; the trial
  needs your go on that task or a replacement for it. Dark Factory task 7112 (pending) already owns the question
  'are per-kind weights a weak lever because every run competes with nine services at 100'; fold the trial into it or
  close it." And in section 1 item 7: "No task owns the weight values, a split of the allowance or an urgency order
  across projects. Two pending tasks own the experiment on whether weights work at all (Dark Factory 7112, reify
  8827)."

### A3. The "floor" and the "keep within 3 to 1 until the test-timeout fix lands" limit are computed for the wrong project and the wrong layout, and the fix is not near

- Where: section 4 "The floor under any ratio"; section 8 first failure mode; section 9 decision 1; section 10 sixth
  bullet ("both were ready and blocked on file locks this morning").
- What is wrong:
  1. The arithmetic uses Dark Factory's local merge verify (28 to 52 minutes) and Dark Factory's two-hour limit. The
     projects that would be weighted down have different numbers. Before the pause, at today's equal weights:

     | Project | Local merge verify, passed: median / 90th pct / max (min) | Runs | Limit per verify command: normal / merge verify on a cold cache |
     |---|---|---|---|
     | reify | 76.2 / 124.2 / 151.8 (23% over 90 min) | 73 | 120 / 180 min |
     | solar-challenge-platform | 27.9 / 39.1 / 43.5 | 64 | 90 / 120 min |
     | solar-challenge | 17.0 / 27.2 / 32.5 | 38 | 60 / 120 min |
     | Dark Factory (for scale) | 42.6 / 56.2 / 90.5 | 273 | 120 / 120 min |

     Under the report's own assumption (run time scales inversely with CPU share) a weight of one third puts reify's
     median local merge verify near 230 minutes, past both of its limits, and the 90th percentile of the two solar
     projects at 82 and 117 minutes, past their normal limits (60 and 90) and inside the cold-cache one. So 3 to 1
     fails reify outright and is not shown safe for the solar pair; reify's 90th percentile is already past two hours
     at equal weights, and a ratio of about 1.6 to 1 takes its median there. (Projection, same assumption as the
     report's. The limits are per command and a verify runs several, so the comparison is rough in the report and
     here alike: `orchestrator/src/orchestrator/verify.py`, the resolver that returns "the effective per-command
     verify timeout", with `verify_command_timeout_secs` and `merge_verify_cold_command_timeout_secs`.)
  2. The 112-to-208-minute figure is the 3-to-1 case **with** desktop work also taking half the host, under the
     one-group-per-project layout (2.41 threads). Without that, 3 to 1 gives 4.82 threads and 56 to 104 minutes, inside
     the limit. And 112 is not past a 120-minute limit. So the sentence "ratios beyond about 3 to 1 risk turning slow
     into fails" does not follow from its own numbers: they show failure at 3 to 1 in one case and none in the other.
  3. The "about 10 threads" baseline is an arithmetic share from a two-project example, not a measurement. The first
     study measured Dark Factory's local merge verify at no more than 4.44 cores averaged over the pause
     (`/tmp/team-gates-1112/capacity.md` F9), about 5.5 while one is running.
  4. "It waits on two unfinished prerequisites": task 5529 has **125 unfinished tasks upstream of it**, not two. Its
     prerequisite 5089 waits on 9132, which waits on four tasks from the whole-project review's refactoring streams,
     which in turn reach the merge-lane chain (8526, 8527, 8528, and the blocked task 7720). "Until the test-timeout
     fix lands" is therefore not a short interim. Section 10's "both were ready and blocked on file locks this
     morning" is wrong for 5529: it is not ready (the first study said so: "it waits on an unfinished prerequisite
     (5089)").
- Evidence: `critic/joins.py others_verify` (others_verify_out.txt); loaded configs
  (`orchestrator.config.load_config` on each project's yaml: `verify_command_timeout_secs` 3600 / 5400 / 7200 / 7200;
  `merge_verify_cold_command_timeout_secs` 10800 in reify's yaml, 7200 from the package `defaults.yaml` elsewhere);
  reify task 8827 reports the same reify figures independently (mean 75.6, p50 75.0, p90 127.8);
  `critic/closure.py 5529` (closure_out.txt: 125 tasks with status other than done or cancelled).
- Proposed text for section 4: "**The floor under any ratio.** A project squeezed to a few threads does not just run
  slowly: its tests time out and the timeout kills the whole test worker. The fix for that (task 5529) has 125
  unfinished tasks ahead of it, so plan as if it will not land soon. The floor differs by project. At equal weights
  before the pause, reify's merge verify on the workstation already took a median 76 minutes (90th percentile 124)
  against limits of two to three hours per command; solar-challenge-platform 28 (39) against 90 minutes;
  solar-challenge 17 (27) against 60. If run time scales inversely with CPU share, a third of the share fails reify
  at the median and brings both solar projects to their normal limits at the 90th percentile. (Projection.) So a
  single ratio limit is not supported; the safe ratio has to be set per project from its own verify time, or merge
  verifies have to be left out of the project weight (see decision 2)."
- Proposed text for decision 1: replace "kept within 3 to 1 until the test-timeout fix lands" with "with merge
  verifies left at today's weight in every project (decision 2), or a ratio per project no larger than its normal
  limit divided by its 90th-percentile verify time (reify about 1 to 1, solar-challenge-platform about 2 to 1,
  solar-challenge about 2 to 1; rough, because the limit is per command)."

### A4. "Merge verify is never throttled": the lead's reading ("no breach") is the weaker one on the record, and there is a design that makes the question moot

- Where: section 7 rulings table row 3; section 9 decision 2.
- What is wrong: the report reads the directive as forbidding "capping or queueing" and offers "no" as the lead's
  reading for a weight. The directive's provenance is task 5203, where it was invoked against cutting the merge test
  run's worker count from 16 to 8, a reduction in CPU given to the merge verify, judged by whether it "costs no
  material wall-clock". The same plan lists "Capping `-n` for the merge role" as "a live-tuning / throughput-tradeoff
  call the operator should make with load data". So the recorded test is the merge verify's wall-clock time, not the
  mechanism, and a lower weight on a project's merge verifies is meant to lengthen them under contention.
- Evidence: task 5203 description ("Leo's 2026-09-07 directive is that the merge verify (the serial landing
  bottleneck) is never throttled, but he suspects 8 costs no material wall-clock ..."); task 5139 details ("merge is
  not, unless Leo rules otherwise"); `plans/cpu-load-robust-verify-prd.md` section 6 (the bullet after the
  TRIED-AND-REJECTED one).
- A design the report does not offer: apply the project factor to task verifies, background runs and the service, and
  leave every project's merge verify at 100. On the pause seat's own model that gives up at most 2.7 of the 10.8
  minutes other projects added to Dark Factory's local merge verify (others' merge verifies 2.7, task verifies 3.0,
  background 0.3, agents 4.8; `pause/projection_out.txt`), stays inside the directive on any reading, and removes the
  merge-verify half of the floor problem in A3.
- Proposed text for decision 2: "**Should a project's weight apply to its merge verifies?** Your 09-07 directive
  ('merge verify is never throttled') was recorded against cutting the merge run's worker count, so it has been
  applied to CPU, not only to queues; a lower weight is meant to slow a project's merge verify when the host is full.
  Option (a): leave merge verifies at today's weight in every project and weight only task verifies, background runs
  and agents. On the fitted model this keeps about three quarters of what weights can reach on the workstation
  (projection) and does not touch the directive. Option (b): weight merge verifies too, which needs you to say the
  directive does not cover it."

### A5. Allowance option 2: the trigger as worded is on most of the time, and one process's view of the accounts is blind exactly where the option needs it

- Where: section 6 options table row 2 ("When it is over its share **and** most accounts are capped ... nothing is
  withheld while the pool is plentiful ... Each project reads only its own records and its own in-memory view of the
  accounts"); the recommendation paragraph; section 7 table row 5 ("Options 1 and 2 share nothing").
- Verdict asked for: **sound with stated changes** (the budget half is sound; the trigger half as worded is not).
- What is wrong:
  1. "Most accounts are capped" is the normal state, not a shortage signal. Before the pause at least 4 of the 7
     accounts were capped in 89.7% of minutes, at least 5 in 63.4%, at least 6 in 36.9%, all 7 in 10.2%; never fewer
     than 3. During the pause at least 5 were capped 97.3% of the time. (Same cap-interval rule as `allowance.md` F5,
     all ten stores; this follows from F5's own "each account alone was capped about three quarters" which the report
     did not join to the trigger.) With the trigger at "most", option 2 is a near-permanent budget: an over-budget
     project's routine work is held most of every week, and "nothing is withheld while the pool is plentiful" is true
     for about a tenth of the time.
  2. If the trigger is moved late enough to mean shortage ("one account left"), little is left to protect. Between
     the minute six accounts were capped and the start of each of the eight stalls before the pause, the whole fleet
     spent $38 to $3,164 (median about $1,000), against about $12.9k a week. (Measured; overlapping where stalls
     follow each other within hours.)
  3. A process learns an account is capped only when it tries that account, and forgets everything at restart
     (`shared/src/shared/usage_gate.py::UsageGate.try_lease` walks the list and skips only accounts this process has
     marked; `::UsageGate.check_at_startup` is a no-op: "caps will be detected reactively"). Measured, in minutes when
     5 or 6 of 7 accounts were capped fleet-wide before the pause: the project's own view also showed 5 or more in 89%
     (Dark Factory), 87% (reify), 83% and 81% (the solar pair), 81% (scoping-intake) and **35% (stream-touch, then
     lightly active)**; it showed five fewer than the truth, that is none or one, in 7 to 8% of those minutes (the
     pattern a restart leaves; Dark Factory's records carry 52 distinct run ids since 09-15). During the pause the two halted solar schedulers saw 5 or more
     in 56% of such minutes. So the view is adequate for a busy project and wrong for a quiet, held-back or freshly
     restarted one, which is the project the option is meant to restrain. It does learn re-openings on its own
     (`::_account_resume_probe_loop`), so the error is one-sided: it under-counts caps.
  4. The obvious repair, remembering caps across a restart from the project's own `account_events`, is cap
     persistence, which the 11-12 ruling names ("No process, long-running or one-shot, reads or writes a cap file";
     `~/.claude/review-all/review-all-dark_factory-20261008/digests/leo-rulings-8253.md`). The report's "share
     nothing" is true of option 2 as written and stops being true if the blind spot is fixed that way.
  5. Not in the report: every scheduler already has a per-project spend ceiling
     (`orchestrator/src/orchestrator/harness.py::Harness._enforce_cost_ceilings`, a trailing-24-hour sum from
     `shared/src/shared/cost_store.py::CostStore.cost_totals_in_window`; it is what halted pump-web-ui). You ruled it
     effectively off on 11-02 (task 8262, pending: "every trip was normal work just over USD 200, pausing dispatch for
     4 h to 10.5 days; none was a runaway ... stays as an opt-in tripwire"). Option 2 is a second per-project spend
     rule in the same loop. It differs (it holds routine work only, and only in a shortage) but it should be built on
     that query, not beside it (heuristic 11, SPOT, `docs/code-quality.md`), and the 11-02 ruling belongs in the
     section 7 table.
  6. The scheduler has no path to the gate's state today (`scheduler.py` never references the gate; the only reads in
     `harness.py` are the startup check). The read surface exists: `UsageGate.capped_account_names`,
     `UsageGate.is_paused`, `UsageGate.soonest_resets_at`.
  7. "Spend over the last 7 days" against seven staggered resets: could not fault it. Any 7-day window contains
     exactly one reset of each account, so a share of a rolling week is well defined; it measures fairness, not what
     is left now (that is option 3).
- Evidence: `critic/capcount.py` (capcount_out.txt); code symbols named above; task 8262 description.
- Proposed text for the option 2 row: "What it is: each scheduler compares its own spend over the last 7 days with
  its configured share. When it is over its share **and a shortage test is true**, it starts only tasks at or above a
  set tier. The shortage test is the open design point: 'most accounts capped' was true 63 to 90% of the time before
  the pause, so it makes this a near-permanent budget; 'one account left' was true 37% of the time and, at the eight
  stalls before the pause, left a median of about $1,000 of fleet spend to protect. A fixed test on the weekly reset
  calendar (for example, inside the Monday 21:00 to Wednesday 10:00 gap) needs no cap state at all. Needs: code in
  the scheduler, built on the existing trailing-spend query behind the daily cost ceiling, and a read of the
  account gate that the scheduler does not have today. Does not give: a correct count after a restart or in a quiet
  project (a process knows only the caps it has itself hit: stream-touch saw the shortage in 35% of the minutes it
  existed, and every process reads zero after a restart); repairing that by remembering caps is inside your 11-12
  ruling."
- For section 10: add "Option 2's hold against the design invariants that gate a PRD: INV-7 (every hold names who
  exits it, what bounds it, and how it behaves across the 8-hour redeploy), INV-11 and INV-13 (a count that reads
  zero after a restart must not pass silently as 'plentiful')" (`docs/legibility/design-invariants.md`).

### A6. The T3 weight of 10 is set by T3 Code itself, on purpose, every time; it is not an anonymous temporary setting and cannot be removed by deleting one setting

- Where: section 2 ("The setting is temporary (a reboot removes it) and nobody recorded making it"); section 7 table
  row 4 ("T3 sessions are at a tenth today, contrary to it"); section 9 decision 6 ("If not, it is one setting to
  remove").
- What is wrong: the T3 Code server applies `CPUWeight=10`, `IOWeight=10` and a task limit to `app-t3code.slice`
  with `systemctl set-property` as part of how it launches agent sessions. Its source states the reason: "Low weights
  yield CPU and disk to the server under contention". The files under `/run` were last written 2026-11-10 11:14:48Z,
  all three in the same instant, with the T3 service up since 11-01. A reboot removes the files and T3 writes them
  again; removing the setting by hand lasts until T3 next applies it.
- Evidence: `/home/leo/src/t3code/apps/server/src/process/agentScope.ts` (`AGENTS_PARENT_SLICE = "app-t3code.slice"`,
  `AGENTS_PARENT_PROPERTIES = ["CPUWeight=10", "IOWeight=10", "TasksMax=25%"]`; checkout at tag `v0.1.6`, the
  installed version); `ls -la --time-style=full-iso /run/user/1000/systemd/user.control/app-t3code.slice.d/`.
- Proposed text, section 2: "**Interactive sessions launched from T3 are weighted at 10**, a tenth of an
  orchestrator. T3 Code sets this itself each time it launches sessions (its source: 'low weights yield CPU and disk
  to the server under contention'); it is a product default of the version installed, not an operator choice.
  Terminal sessions are at 100."
- Proposed text, decision 6: "**Interactive sessions in T3 run at a tenth of an orchestrator's weight, by T3 Code's
  own default.** Your 08-03 ruling (interactive work deliberately favoured) was measured on terminal sessions. To
  bring T3 sessions in line you would have to change T3 Code's constant in your checkout (`~/src/t3code`) or accept
  the default; deleting the setting does not hold."

### A7. Decision 9 ("is the pause still worth holding?") is not posed fairly

- Where: section 9 decision 9; section 3 "Not established"; section 1 item 2.
- What is wrong:
  1. "Its spend in those three days was not extracted" and section 10 lists it as needed. It is one bounded query.
     Dark Factory's own event store (`critic/joins.py df_daily`, df_daily_out.txt):

     | Period | Landings/day | Spend/day | Spend per landing |
     |---|---|---|---|
     | Two weeks before the pause | 23.7 | $423 | $18 |
     | Last 72 hours before the pause | 30.0 | $396 | $13 |
     | Paused, before the 11-10 23:48Z merge-lane deploy (1.40 d) | 36.4 | $757 | $21 |
     | Paused, after it (1.63 d) | 43.5 | $847 | $19 |

     Rules: landings = `merge_finalized` with state done; spend = `invocations.cost_usd` by completion time, Dark
     Factory's store only (the memory server's curator share, about 3%, excluded); all hours, capped ones included.
     So Dark Factory's spend roughly doubled in the pause (+91% against the 72 hours before) while landings rose 21%
     before the deploy and 45% after it. Against the two-week baseline spend rose 79 to 100% and landings 54 to 84%.
     The data cannot split the rise between the larger allowance share, the deploy, faster verifies and work started
     but not yet landed. That is a narrower statement than "not measured", and it is not "no benefit".
  2. The question sets a bound on one channel (CPU, through verify time: 2.5 to 5.6 landings a day) against the
     cost through all channels (other projects' landings), and leaves the benefit through the allowance blank. A
     reader will take it as "the pause bought nothing".
  3. "About 53 landings a day forgone" is the average so far, because the three projects landed 38 tasks while
     draining. For a decision about the next day the rate is their rate before the pause, up to about 63 a day
     (`pause.md` F9 itself uses 62.6), less wherever a project runs out of ready work (solar-challenge had 61 ready).
  4. The pause was not placed to raise landings per day. Its recorded reason is to land the merge-lane policy chain
     "as fast as possible" (`rulings.md` section 3, the pause reason text). Four days in, three of its six tasks are
     done (8524 on 11-10, 8525 on 11-12, 8531 on 11-09); 8526 went back to pending at 16:30Z today; 8527 and 8528 are
     pending; their prerequisite 7720 is blocked. The report does not say whether CPU is what holds those three.
- Proposed text for decision 9: "**Is the pause still worth holding?** What it was for: landing the merge-lane policy
  chain quickly. Three of its six tasks have landed in four days; the other three wait on a blocked task (7720) and on
  each other, and this study did not establish that CPU is what holds them. What it did for Dark Factory overall:
  landings went from 30 a day (the 72 hours before) to 36 (paused, before the 11-10 deploy) and 44 (after it), while
  its spend doubled from about $400 to about $800 a day. Of that, the part that comes through faster verifies is
  bounded at 2.5 to 5.6 a day; the rest cannot be split between the larger allowance share, the deploy and work in
  progress. What it costs: the three paused projects were landing up to about 63 tasks a day between them (about 4 of
  them high priority); 210 have been forgone so far (a bound)."

---

## M findings

**M1. Section 3, "Dark Factory's unrestricted rate is roughly 45 to 60% of it."** The range is two rules stated as
one. 45% = $5.8k a week, Dark Factory's average over the whole pause **including the 22.5% of hours when every
account was capped** (`allowance.md` F2), divided by $12.9k. 56 to 62% = $1,034 to $1,144 a day in minutes when an
account was open, times 7 (`pause.md` F8, whose "open" minutes were defined from four stores). Recomputed on the
ten-store cap rule: $2,515 over 2.35 open days = $1,070 a day = 58%. It is also not "unrestricted": Dark Factory ran
8.0 to 8.7 agents against 18 task slots and shared every open account with two projects that took 39%. And $12.9k is
itself a projection (sum of medians). Proposed: "Counting all ten stores, the pool supplies about $12.9k a week if
every account reaches its usual weekly cap (projection). With the three projects paused, Dark Factory spent about
$1,050 to $1,150 a day while an account was open, about 56 to 62% of that; averaged over the whole pause, stalls
included, 45%. So it cannot consume the pool alone at the rate it ran."

**M2. Section 2 table, Dark Factory "22 → 40" finished tasks a day, beside section 3's "not distinguishable from
zero".** Both are right under their rules (finished tasks: 22.0 → 39.9; landings: 23.7 → 40.2; the two measures agree
to within 2 a day in every window, `df_daily_out.txt`), but the table's "before" is the two-week mean, which
`pause.md` F6 calls the unfair baseline, and its "during" includes the days after the merge-lane deploy. Proposed
footnote: "Dark Factory: 30 a day in the 72 hours before the pause, 36 during it before the 11-10 merge-lane deploy,
44 after. Section 3 explains why little of the rise is attributed to CPU."

**M3. Section 1 item 2 and decision 9, "about 53 landings a day".** 53 = 210 / 3.97 days, net of the 38 tasks landed
while draining. The brief's 63 is the three projects' rate before the pause. Both are bounds. Say which: "about 63 a
day at their earlier rate; 210 so far, which averages 53 a day because they landed 38 while draining."

**M4. Section 6, "(A measurement from early November of 0.83 hours in 30 days ... used one project's records and is
superseded".** The stated cause is wrong. That seat read all nine projects' ledgers
(`plans/portfolio-review-2026-11-01/seats/X7-unused-capacity.md`, "Method and sources": "runs.db for all nine
projects") with the same family of rule. Two things differ. Its window closed at 2026-11-02 10:02Z, a day before the
first stall with every account on its weekly limit (11-03 19:58Z); the stalls of 11-03/04, 11-07 and 11-08 are 23.7
of the 36.0 capped hours before the pause. And for 10-27 it counted about 40 minutes where this study counts 11.6
hours over 10-27 and 10-28; I checked those five stalls and no agent call started inside four of them on any account
(two calls in the fifth), so this study's figure stands (`critic/stallcheck.py`). Proposed: "(The 0.83 hours in 30
days in the portfolio review covered the same nine ledgers but ended on 11-02, the day before the first stall with
every account on its weekly limit; it also undercounted 10-27 and 10-28. Your 11-02 position that single-account
caps are negligible unless the whole pool is capped was taken on that figure. The whole pool has been capped 10% of
the time since 10-25 and 17% of the time from 11-03 to the pause.)"

**M5. Section 6, "23.8% of the pause".** 23.8% is to 11:00Z (68.8 h); to 15:00Z, the window behind every other
during-pause allowance figure in the report (3.03 days), it is 22.5% (`allowance.md` F5; `capcount_out.txt`). Use
22.5%, or state the window.

**M6. Section 9 decision 5 and section 1 item 4, scoping-intake and stream-touch: "39% of the pool during the pause,
84% and 59% of it on low-ranked tasks before the pause."** The during-pause shares exist and should be used: 79% and
53% (`allowance/tiers_out.txt`). More important and missing: their demand is ending. Pending tasks at 15:30Z: 22 and
9 (`host-map.md` F3). Their combined share of pool spend by day: 49% on 11-09, 53% on 11-10, 28% on 11-11, 17% on
11-12. Proposed: "They took 39% of the pool during the pause (79% and 53% of it on low-ranked tasks), mostly on 11-09
and 11-10; their queues are now nearly empty (22 and 9 tasks pending) and their share was 17% on 11-12."

**M7. Section 1 item 6 and section 8, "Two projects and 111 held-back Dark Factory tasks are released in the same
instant".** The 111 add no load. Dark Factory already has about 2,262 ready tasks for 18 slots, so they change which
tasks it runs, not how many (`pause.md` F10 says exactly this). The load at resume is the two solar projects: 61 ready
tasks against a limit of 24, and 223 against 14. The count is also moving: 105 in the brief, 111 in the report, 112
at 16:40Z (176 tasks have an edge on 8528, 169 pending; `critic/tasks_check.py status`). The "nearest precedent"
(11-04, agents 1 to 47) had all four projects starting from nothing, Dark Factory and reify included; the planned
resume adds at most 38 agents from two projects to a running Dark Factory. Proposed: "Two projects resume in the same
instant with no check on load: up to 38 more agents (solar-challenge 24, solar-challenge-platform 14) and their
verifies. About 110 held-back Dark Factory tasks become ready at the same moment, but Dark Factory's slots are already
full, so those change its mix, not its load." Drop "and not in the same hour as the held-back tasks are released"
from migration step 1.

**M8. Section 1 item 3, "would have delivered most of the verify speed-up ... (projection: verify at 34 to 40
minutes, the others losing 0 to 13 landings a day)".** 34 minutes and 13 landings are the 10-to-1 case, which the
report elsewhere rules out. At 3 to 1 the projection is 36.7 to 40.3 minutes against 43.9, which is a third to two
thirds of the modelled 10.8-minute speed-up, and 0 to 3.6 landings a day lost by the others
(`pause/projection_out.txt`). "Most" holds only under the assumption that all contention is CPU time. Proposed:
"(projection at 3 to 1: the workstation verify at 37 to 40 minutes instead of 44, a third to two thirds of what the
pause gave, the others losing 0 to 4 landings a day; nothing on the laptop and nothing on the allowance)".

**M9. Section 6, "choosing accounts evenly instead of first-in-list would mostly remove the 5-hour caps (four of the
eight stalls before the pause involved one)".** The parenthesis implies those four stalls would go. In all of them
the account that hit its 5-hour limit was the only one open; the other six were on weekly limits, so there was no
second account to spread to. Of the 23 5-hour caps before the pause, 6 hit with no other account open, 9 with one, 8
with two or three; all 23 fell between 10-25 and 10-30 and there has been none since (`critic/sessioncaps.py`). Also,
even selection makes the list order meaningless, and the 10-06 ruling records max-b and max-h as kept "at the end of
the order" (accounts file header). Proposed: "Choosing accounts evenly would have avoided at most 17 of the 23 5-hour
caps before the pause, none of which caused a stall; the four stalls that involved a 5-hour cap happened with one
account left. No 5-hour cap has occurred since 10-30. It would also retire the list order your 10-06 ruling relies
on. Low value on this evidence."

**M10. Section 6, "the scheduler not starting a task when no account is open would stop tasks holding slots and file
locks through a stall".** The code claim holds (no reference to the gate anywhere in `scheduler.py`; the only reads
in `harness.py` are at startup). The benefit is not shown. While every account is capped no other task can use the
slot or the file either, since each needs an account at its next agent call; the 21% of slot time is time nothing
could run, not time taken from something else. What the change could alter is the order in which work restarts when
an account reopens. Proposed: "... would not free capacity during a stall (nothing can run); it would let the
scheduler choose again when an account reopens instead of resuming whatever was started before. Benefit unmeasured."

**M11. Section 1 item 7 and section 7 "Owners": "Nothing here is owned."** Beyond 7112 and reify 8827 (A2) and the
daily cost ceiling with its 11-02 ruling (A5): the pause lever itself has an approved PRD,
`plans/scheduler-pause-halt-retirement-prd.md` (2026-08-24), decision 5: "Operator halts get an owner and a declared
expiry, and never auto-resume", carried by task 5313 (pending, medium). The 11-08 pause is an operator hold of
exactly that class, and the three resume tasks are its expiry in practice. Name it in section 7 and in section 10 as
a seam. I searched Dark Factory's store again including the `details` field (and the three other stores) for: weight
values, an allowance budget or pacing, spreading account selection, dispatch gated on account availability, a
guarded or staggered resume, the T3 slice weight. No further owner (`critic/owners_out.txt`).

**M12. Section 7 table row 2, "The yes/no trigger in option 2 is the one allowed load-shaped form."** The ruling
allows one named lever: "a BINARY pressure hold in the runner's health probe is the one allowed load-shaped lever"
(`plans/remote-task-leg-verify-prd.md`, rulings header). Option 2's trigger is a second lever of the same shape, not
that one. Proposed: "The yes/no trigger in option 2 has the same shape as the one load-driven lever the ruling
allows (a yes/no pressure hold in the verify runner); whether a second is allowed is yours." Option 3's threshold,
computed from live spend, is nearer the forbidden form and should say so.

**M13. Section 5 recommendation, "reversible in seconds".** A verify already running keeps its weight
(`config.py` comment beside `RELOADABLE_FIELDS`: "a scope already running keeps the weight it started with"). Reversal
takes effect at the next verify start: up to two hours on reify's local merge verify. The yaml files are tracked in
each project's repository, so the edit is a commit on each project's main plus a reload on each project's own
escalation service. Proposed: "reversible with one reload per project, effective from the next verify to start".

**M14. Section 4 worked case.** It is the one-group-per-project layout, but the recommendation is Layout A. Under
Layout A the same 3-to-1 means weights 100/33/10 against 33/11/3, and every other program in the pool (nine services,
the compiler cache, desktop programs) is in the same sum, so the absolute shares are lower and the low project's task
verify (11) and background run (3) are squeezed far harder than its merge verify. Add one line saying so. Also "about
three times its background run" takes the background figure (1.68 threads) from a different job mix than the rest of
the sentence. And "its two-hour limit" is a per-command limit (`verify_command_timeout_secs`), not a limit on the
whole verify (reify passed merge verifies of up to 152 minutes); it is 60 and 90 minutes on the two solar projects,
and a merge verify on a cold cache has its own (120 minutes from the package default, 180 on reify).

**M15. Section 3, "CPU: a faster verify, measurable without the confounders."** The capped-hours comparison (31.4
against 43.4 minutes) is free of the slot change, the deploy and the laptop move, but has its own: when every account
is capped Dark Factory's own agents and task verifies stop too (worth about 1 to 6 minutes on the seat's own
coefficients), and 14 of the 31 capped hours are one night (11-03 20:00Z to 11-04 10:00Z). It measures "the host with
all agents stopped", not "the other projects removed". Retitle: "CPU: a faster verify, visible in a comparison free
of those three changes".

**M16. Section 3, the fitted model.** Give the uncertainty: merge verify 6.8 ± 1.0, task verify 1.4 ± 1.3,
background 0.7 ± 1.7, agent 0.7 ± 0.15 minutes each (standard errors; R² 0.31). The task-verify and background terms
are not distinguishable from zero, and the task-verify term carries 3.0 of the 10.8 minutes. "Predicted the pause to
within a minute" (section 8) is one point, and the 37.0 includes 3.8 minutes added from exposures observed during the
pause.

**M17. Section 8 migration step 6, "resuming three projects returns them to about 44% of the pool, and Dark
Factory's share falls back toward a quarter."** That is a projection from shares before the pause and should be
labelled. Demand has changed since: scoping-intake and stream-touch (29% then) are nearly out of work, reify has 138
dispatchable tasks, and Dark Factory's task limit is ruled back to 24.

**M18. Section 6 option 1 (dedicated accounts), "Does not give".** Missing its largest cost: the staggered resets.
An account's weekly allowance is spent a median 45 hours after its reset; a project holding one account then waits up
to five days for it. The shared pool's worst gap is 37 hours only because seven resets are spread over the week.
Also unmentioned: task 6628 (pending) asserts the "one account pool from config/usage-accounts.yaml" invariant.

**M19. Section 10, "Tasks 3394 ... and 5529 ... both were ready and blocked on file locks this morning."** True of
3394 (no prerequisites). Not of 5529 (see A3). Today's statuses, re-read 16:40Z: 3394 pending; 5529 pending behind
5089 (pending) behind 9132 (pending); 7112 pending; 8791, 8792, 8793 pending; 8528 and 8527 pending; **8526 back to
pending at 16:30Z** (the rulings seat read it in progress at 15:54Z); 7720 blocked; 5139 pending and, since 16:03Z,
also waiting on 8528; 8262 pending; 8253 pending; 8252 cancelled.

---

## m findings

- m1. Section 1 item 5, "57 of 69" against the brief's "56 of 68": both right; one more memory hold landed at
  14:36Z (68 before 11:49Z, 69 before 14:40Z, none since; `critic/joins.py holds`). Say so in a clause.
- m2. Section 4 table, "the solar pair at 1 task": scoping-intake and stream-touch are also at a floor of 1 (loaded
  config), and they are the ones running now.
- m3. Section 2 table, pump-web-ui "halted since 11-02 by its own daily cost ceiling": you ruled that ceiling
  effectively off on 11-02, and task 8262 notes pump-web-ui "additionally needs resume_scheduler now". It is still
  halted, with 54 tasks pending. Worth one line, since it is a tenth consumer waiting to return.
- m4. Section 1 item 1, "the fleet spent about 96% of the total": that is measured spend over a projected total.
  Every account reached its weekly limit in each week of the window, so "all of it" is the plainer statement.
- m5. Brief question 3 asks for consumption "by project, account and task tier". The report gives project and tier,
  and per-account capped time, but no per-account consumption table.
- m6. Section 6 reset hours are given as fixed in UTC. `allowance.md` U4: either that or the parser drops the time
  zone; October windows may be an hour off.
- m7. The report says Dark Factory has 18 task slots implicitly (section 8); the yaml still reads 18 while the ruled
  value is 24 at the next restart. Say which is meant.
- m8. A service weight also covers the orchestrator's own process (scheduler, merge worker, escalation service), not
  only agents. Inside the group the agents run at lower priority (`shared/src/shared/cli_invoke.py::_cpu_priority_prefix`),
  so the risk is small, but a starved orchestrator is what the watchdog's liveness probe restarts.
- m9. Option 2 should say "effective tier" (a low task that a high task depends on inherits the tier), as task 8261
  does, or it will hold prerequisites of urgent work.
- m10. Option 2's "configured share of the pool" is a dollar figure that changes by two sevenths whenever max-b and
  max-h are switched at Anthropic's side.
- m11. Section 3's landing bound (2.5 to 5.6 a day) is scaled from the two-week landing rate (23.1); scaled from the
  72 hours before the pause (30) it is 3.2 to 7.3.
- m12. Solar-challenge's limit of 24: confirmed by loading its config (`load_config` returns 24; the package
  `defaults.yaml` layer applies before the field default of 3, `config.py::YamlSettingsSource.__call__`). The pause
  seat's "default 3" is wrong; the report is right.

---

## Plain language: sentences a reader with no context could not act on

1. Section 1 item 2, "local merge verify": never defined. Say "the merge verify when it runs on the workstation
   (it also runs on the laptop)".
2. Section 1 item 6 and section 8, "CPU pressure from 25 to 63": no unit. Say "the kernel's CPU pressure reading
   (the percentage of time some task was waiting for a CPU) from 25 to 63".
3. Section 1 item 5, "57 of 69 dispatch holds": unit unclear. "57 of the 69 times the scheduler declined to start a
   task (in 20 separate episodes)".
4. Section 3, "before the verify slot count changed" and "the merge-lane deploy on 11-10": neither is introduced.
   One clause each (the number of task verifies allowed at once went from 1 to 2 on 11-09 14:13Z; new merge-queue
   code went live at 11-10 23:48Z).
5. Section 3, "*Correction made at integration.* The pause seat counted four projects' stores": drafting-process
   language. State the figure and its rule; drop the history.
6. Section 2 rules line, "all ten stores that record it (nine orchestrators plus the memory server's curator)":
   "store" and "curator" are internal labels.
7. Section 6, "the 5-hour caps": the report never says an account also has a 5-hour limit.
8. Section 7 owners paragraph, "section 2 item 3, section 3 decisions D5 and D12, and section 9": plan-internal
   ids; one sentence saying the plan hands the values to "task 3394 or its follow-up, never filed" is enough.
9. Section 7 table, "the one allowed load-shaped form": see M12.
10. Section 8, "the sampler", "swap churn", "kernel reclaim": undefined.
11. Bare numbers without a description at the point of use: "task 5529" in section 8 failure modes and section 9
    decision 1 ("the test-timeout fix" is given once in section 4); "task 3394" in section 8 measure 1 and section 10.
12. Section 5, "A run's weight = project factor × kind of work × (optionally) the task's priority tier": give the
    numbers once (for example reify at a third: 33 / 11 / 3 against Dark Factory's 100 / 33 / 10).

---

## Claims I tried to refute and could not

- The three verify weights are per project, read at each run start, hot-reloadable, and accept values below the
  default (A1, code symbols there). All nine orchestrators run Dark Factory's code against their own yaml.
- All nine projects draw on the same seven accounts (`load_config` on six project yamls returns 7 accounts each).
- The share-of-spend columns in the section 2 table are one rule in both windows (`allowance/tiers_out.txt`); my
  re-query of Dark Factory's store gives $2,963 and $5,637 a week without the curator share, matching $3.0k and $5.8k
  with it.
- The all-capped figures (10.2% before, 22.5% during to 15:00Z) and the stalls behind them: no agent call started in
  nine of the ten stalls on any account, two calls in the tenth.
- Finished tasks (`task_results`, done) and landings (`merge_finalized`, done) agree for Dark Factory in every
  window to within 2 a day.
- "The scheduler starts a task without asking whether an account is open" (no gate reference in `scheduler.py`).
- The resume claims: 8792 and 8793 have identical prerequisites (8528 pending, 8531 done); 8791 also needs 7721 and
  7824; the resume script contains no load check (`scripts/resume_project_scheduler.py`, 136 lines);
  `scheduler.py::Scheduler.resume` only clears the pause; pressure-hold floors are 3 (Dark Factory, reify) and 1
  (everything else); solar-challenge's effective task limit is 24.
- The 3-to-1 arithmetic itself (24 and 8 threads; 14.46, 4.82, 4.77) and the 0.25-CPU / 2.5-second example.
- Every quote in the section 7 table is in its source as quoted: the tried-and-rejected text
  (`plans/cpu-load-robust-verify-prd.md` section 6), "never derived from instantaneous loadavg"
  (`plans/verify-oversubscription-control-prd.md` section 3), the 09-07 directive (tasks 5203, 5139), "privileged
  deliberately" and "deliberately excluded" (task 3394), the 11-12 cap ruling (memory file and digest), max-b and
  max-h (accounts file header).
- Kernel weights are outside the shared-queue rejection; allowance option 4 has the rejected shape.
- No task in the four stores, `details` included, owns the weight values, an allowance budget or pacing, account
  spreading, dispatch gated on account availability, a guarded resume, or the T3 weight (but see A2, A5, M11 for
  adjacent owners the report should name).
- The dispatch-hold count (57 memory, 12 disk I/O, 69 in all to 14:40Z).
- A rolling 7-day spend figure is meaningful against staggered weekly resets (A5 point 7).
