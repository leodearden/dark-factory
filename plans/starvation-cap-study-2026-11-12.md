# Starvation auto-pin cap: what it does, whether raising it helps, what to do instead

As of 2026-11-12 ~13:30Z. Investigation only; nothing was changed.
Data sources: `data/orchestrator/runs.db` events, the task store, `scheduler_overrides.db`,
`scheduler_state.json`, and the escalation archive. Scripts and outputs:
`/tmp/starvation-cap-2026-11-12/{measure,sim,critic}/`.

## Leo's rulings, 2026-11-13

- **Cap stays 3.** No cap-8 trial.
- **Interim: `starvation_watchdog.skip_threshold` 50 → 150** in `dark-factory-orchestrator.yaml`,
  applied by reload_config. It is to be reverted as soon as task 9636 lands.
- **Task 9636 filed:** the cap L1 reports a *change* in starvation, and the watchdog's inputs survive
  restarts. This amends PRD D9 and subsumes 7718 and 7719.
- **Narrower file lists: dropped.** There is no evidence that the architect's file lists are
  wrong. Breadth (~12 files against ~4) shows these tasks touch many files, not that the lists are
  over-declared. Splitting tasks would make throughput worse: the merge queue is close to
  throughput-limited already, so more tasks cost more than waiting on locks.
- **Releasing locks during the merge phase is ruled out** (Leo 2026-11-12: the next holder must
  build on the merged result). In section 4 the merge-phase lever means slots only.
- **Changed since the study, both reducing starvation:**
  - `max_concurrent_tasks` 18 → 24 (d128ff556f; takes effect at the next redeploy).
  - Documentation files take unlimited lock holders (ef7c25fdf6, interim for 5467). This removes
    OPERATIONS.md and docs/task-authoring.md as blockers.

## Words used here

- **Slot.** One of the 18 tasks the orchestrator may run at once (`max_concurrent_tasks: 18`).
- **Lock.** Each task declares the files it will touch and holds a lock on them. Two tasks that need
  the same file cannot run together. A few hub files allow 2 holders.
- **Park.** A soft hold that the fairness logic puts on files for a task that keeps getting skipped,
  so that others cannot grab those files as they free up.
- **Pin.** A manual or automatic "run this next" mark, with a queue position (pin_order).
- **Reservation head.** The single blocked pin (lowest pin_order) allowed to put a park on its files.
- **Backfill.** A rule that lets a small task run in a gap a park is holding, if it will not delay
  the park's owner.
- **PSI hold.** The host is overloaded, so the scheduler briefly dispatches nothing heavy.

## The short answer

Raising `starvation_watchdog.auto_pin_max_concurrent` from 3 to 5 or 8 would not noticeably change
throughput, and it would not quiet the alarm.

- **Starts are bound by slots, not by pins.**
  - All 18 slots are busy 94% of the time, and ~2,200 tasks are ready to run.
  - In 7.5 days there was no moment when no waiting task could take its locks. Every pass that
    dispatched nothing was a PSI hold.
  - So a pin can only change *which* task starts next, not how many start.
- **The pins barely act.**
  - A watchdog pin jumps its task forward only on a pass when all of its files happen to be free.
  - For starving tasks that is rare: 2.5% of (task, dispatch) pairs. It does happen to 80% of them
    at least once, mostly in the first minutes after an orchestrator restart, when every lock and
    park is empty.
  - Of 18 watchdog pins, 9 expired after 24 hours without the task starting.
  - Replayed over the last week, cap 8 gives about +2 extra jumps in 7.5 days compared with cap 3
    (0.27 a day). Cap 5 gives about +0.
- **The alarm cannot drain at any cap below the hundreds.**
  - It auto-closes only when its "waiting for a pin slot" list is empty. That list holds ~290
    tasks and lives in memory.
  - Each restart wipes the list. About a day later the same ~290 tasks re-qualify and are
    re-announced as a burst. Genuinely new starving tasks arrive at only ~9 a day.
  - Replay: alarms per day fall 0–4% at cap 5 and 8–10% at cap 8.
- **The "higher cap hurts throughput" memory is real, but it belongs to a different knob.**
  - The measured harm was −4% to −11% throughput, with gridlock (PRD
    `plans/scheduler-dispatch-scoring-and-lock-layer-prd.md` §7, 2026-08-06).
  - That harm came from automatic parks for *every* starving task. The matching knob is
    `pin_reservation_max_active` (default 1).
  - Nothing says the auto-pin cap hurts throughput, and nothing derives the value 3. It was sized
    as a safety bound after the 09-07 uncapped storm, in which 489 tasks were pinned.
- **Your 11-05 ruling already named the deeper cause, and the data backs it.** Its text:
  "wait (option a). No park eviction and no auto-pin cap change. … The structural cause (slot
  saturation by merge waiters) is owned by task 8480."
  - 41% of all slot time is spent in the merge phase, waiting on the merge queue.
  - A task keeps its file locks through that phase. The final release comes a median 4.9 hours
    after the task enters merge (p90 10 hours).
  - So the hot files the starving tasks wait on are largely held by finished work that is queued
    to merge.

**Recommendation:**
- Keep the cap at 3.
- Stop the alarm from re-announcing the same standing backlog after every restart. There is a
  no-code option and a code option; both are below.
- Put structural effort where the time actually goes: the merge phase's hold on slots
  (task 8480 and related), then the hot files.
- If you want proof rather than a model, run a time-boxed cap-8 trial with `reload_config`. It is
  fully reversible.

## 1. Mechanism: what the cap does

Every scheduler pass, the watchdog checks each pending task whose dependencies are all done. It
calls the task "starving" when it has been ready for at least 72 hours AND one of these holds:
- It was refused its locks in 50 passes while a lower-ranked task was dispatched. 97% of firings
  are this kind.
- It was the top candidate yet was skipped 50 times. This kind is rare.

A third kind, "parked behind a live holder", never pins and files its own L1 escalation.

When a task is called starving:
- **If fewer than 3 watchdog pins exist,** the task is pinned for 24 hours at the *end* of the pin
  queue (`overrides.py::create_watchdog_pin` takes max pin_order + 1). An info escalation is filed.
- **If 3 already exist** (or the cap is 0), the task is not pinned. It goes on an in-memory
  "waiting for a pin slot" list. One shared L1 escalation, `esc-__starvation_auto_pin_cap__-N`,
  names the newly added tasks, and only one such L1 is open at a time.
- The L1 auto-closes only when that list is empty, which has never happened
  (`scheduler.py::_resolve_cap_sentinel_once_drained`).
- When the watcher closes it by hand, the next newly added task files a fresh one.

What a pin changes in scheduling (`scheduler.py::_phase_select_pins`):
- **One dispatch per pass, pins first.** Pinned tasks are tried in pin_order before the normal
  scored order. The first one with all its files free is dispatched and the pass ends. A pinned
  task that is not free still gets its normal turn (with backfill) in the scored order, so a pin
  is an extra try and never a worse position.
- **A pin never takes a held lock.** A blocked watchdog pin also does not get past other tasks'
  parks. Starving tasks own no parks, and only the reservation head gets park precedence.
- **The reservation head.** Only the first *blocked* pin among pins that can dispatch now gets to
  park its files (`pin_reservation_max_active = 1`).
  - Today 4 dispatchable operator pins sit ahead of the watchdog pins (8526, 5653, 5654, 5655). The
    other 9 operator pins are running or waiting on dependencies.
  - A watchdog pin was the head in 2 of 215 samples.
  - **Risk of a higher cap:** whenever no earlier pin is blocked, a watchdog pin becomes the head.
    It then parks its ~12 files while it waits, and those files sit idle. More watchdog pins also
    means more of them ahead of any operator pin you add later.

The harness picks the next task, and takes its locks, before a slot frees
(`harness.py::Harness.run`: `acquire_next()`, then `await sem.acquire()`). That pre-start hold is
a median 6.3 minutes, about 1.4% of a typical 7.7-hour lock hold. It is small next to the merge-phase
hold.

## 2. Throughput impact of raising the cap

| cap | watchdog pins / day | extra starving tasks moved ahead vs cap 3 (7.5 days) | cap alarms / day (model) |
|---|---|---|---|
| 3 (today) | 2.7 | — | 8.0 (measured 7.9) |
| 5 | 4.5 | +0 (with charter-only file lists: +3) | 8.0 (−0 to −4%) |
| 8 | 6.9 | +2 (charter-only: +3) | 7.3 (−8 to −10%) |
| unlimited | 47 | +17 | 0 |

How the replay works:
- It rebuilds locks and parks from the event log for 11-05 to 11-12 (523 dispatches).
- At each real dispatch it asks whether an extra pinned starving task had all its files free and
  would have gone first.
- **Validation:** only 8 of 435 real winners appear blocked at their own dispatch. At cap 3 the
  model picks 16 of the 17 real pins.
- The alarm column is *calibrated* to the real alarm rate through one parameter (how long an alarm
  stays open). Only the relative change is meaningful.
- It does not model knock-on reordering, merge-lane effects, or changes in lock-hold times.

Effect on tasks *landed*, a derived bound rather than replay output:
- (extra jumps per day) × (done-rate gap between other tasks and starving ones) is at most 0.04 a
  day at cap 8, against ~31 a day landed.
- Other paths exist and are not quantified: merge conflicts, supersedes, re-verifies, and requeues
  (35% of starving tasks requeue, against 23% of others).
- At ≤0.3 extra jumps a day, every one of these paths is small.
- If anything, the effect leans negative. At caps 3–8 every replayed jump happened just after a
  restart and displaced high-priority work for a broad task (~12 locked files against ~4). The
  done-rate gap of 46% vs 55% is not statistically significant (n=69).

## 3. The data

**Who is starving.**
- 358 distinct tasks were turned away by the cap since 11-05. 291 are still pending and unstarted.
- They have been ready for at least 221 hours at median. That is a lower bound, because the clock
  was backfilled on 11-01.
- For scale: ~2,230 tasks are ready against ~31 landed a day (11-05..11-11). That is a 70–75 day
  backlog, so multi-day waits are normal. "Starving" here mostly means "waiting on a hot file in a
  deep queue".

**What they wait on.** The count is distinct starving tasks that name the file as a blocker, last
7 days.

| file | tasks |
|---|---|
| harness.py (already allows 2 holders) | 104 |
| workflow.py (already allows 2 holders) | 100 |
| OPERATIONS.md | 97 |
| config.py | 85 |
| fused-memory server/tools.py | 64 |
| merge_lane/worker.py | 43 |
| verify.py | 41 |
| event_store.py | 41 |
| scheduler.py | 39 |
| git_ops.py | 38 |
| docs/task-authoring.md | 32 |

**Who holds those files.** Holders are long runners and tasks waiting in the merge queue. Some park
owners are themselves pending: 5040 has not run since 11-04, and 6718 has not run since 10-26.

**Do pins help the pinned task?** Weakly, and not established.
- Excluding starts within 5 minutes of a restart, 15% of pinned tasks started within 72 hours,
  against 5% of turned-away tasks.
- n=13 with full follow-up, and the confidence intervals overlap.

**Slots and throughput.** 11-05..11-12.

| measure | value |
|---|---|
| mean tasks in flight | 17.8 of 18 |
| share of time at 18 | 94% |
| slot time by phase | merge 41%, verify 21%, plan 15%, review 12%, execute 10% |
| done per day | ~29–32 (two counting rules) |
| starts per day | ~62 |
| passes that dispatched nothing | all were PSI holds |

**The alarm.**
- 57 cap L1s since 11-05; 55 were promoted to L2 and closed by the watcher under your ruling.
- They come in bursts about a day after each restart: per day 10, 12, 1, 1, 0, 20, 1, 12.
- The watchdog's own auto-close has fired 0 times.

## 4. Options

| option | what happens | ramification |
|---|---|---|
| **Keep 3** | Nothing changes. | The alarm keeps bursting after each restart and the watcher keeps closing it, at a cost of watcher tokens and records. No throughput cost. |
| **Raise to 5 or 8** (config, hot) | 2–5 more tasks hold a one-shot 24-hour pin. | Throughput is unchanged (≤0.3 extra jumps a day, weakly negative). The alarm drops ≤10%. A watchdog pin is more often the reservation head and idles ~12 hot files. Operator pins added later queue behind more watchdog pins. There are more info escalations. |
| **Raise to hundreds** | The alarm stops. | ~17 extra jumps a week displace high-priority work, and watchdog pins crowd the pin queue. This is the regime the cap was created to prevent (09-07 storm). |
| **Timed trial: cap 8 for ~3 days** (config, hot) | Measures what the replay predicts. | Reversible with one more reload. The signal is weak (expected +2 jumps), so it mainly confirms "no effect". It costs little. |
| **Cap 0** | No auto-pins. | Every firing goes to the cap L1, so the alarm continues. This only removes the tiny post-restart benefit. |
| **`skip_threshold` raised** (config, hot; e.g. 50 → 150) | Fewer tasks qualify, and a run must last longer before re-announcing. | A no-code way to cut restart bursts. Cost: a real multi-day wedge is detected later. The 72-hour age floor and the 4-hour parked-behind-holder gate still apply. |
| **`enabled: false`** (config, hot) | The watchdog goes silent. | Also silences the useful "parked behind a live holder" L1s. Too blunt. |
| **Shorter `auto_pin_ttl_secs`** (config, hot) | Pin slots turn over faster. | Little gain, since pins rarely act at all. Not worth it alone. |
| **Retune the alarm** (code task) | File the cap L1 only when the waiting list *grows* materially or a new blocker file appears, or report it as a daily digest line. Keep refusal counts and the waiting list across restarts. | Removes ~8 L1 + 8 L2 a day of noise and keeps the signal for real change. **This amends PRD D9** ("hitting the cap is itself the escalation — a systemically starved queue is an operator-grade signal"), so it needs your ruling. Interim step: a watcher rule to close these L1s without L2 promotion. |
| **Hand-pin specific tasks** | An operator pin with a low pin_order becomes the reservation head when it is blocked. | The effective lever for a task you actually need next. Only one head exists at a time, and its files idle while it waits. Use sparingly. |
| **Merge phase's hold on slots and locks** (8480 and related) | Free slots, locks, or both while a finished task waits on the merge queue. 8480 covers reserving a slot or bypassing the cap for main-red heals. A broader version would not count merge-waiters against the slot limit. (Releasing their locks is ruled out; see Leo's rulings.) | The largest lever: 41% of slot time. Locks stay held until the merge lands, by design. It needs design, and it ties into the existing merge-lane PRDs. |
| **Hot files** | Split OPERATIONS.md, config.py and fused-memory tools.py (harness.py and workflow.py already allow 2). Or allow 2 holders on docs-only hot files (OPERATIONS.md, docs/task-authoring.md) via `module_overrides`. | The overrides change needs a restart and brings more merge conflicts on those files. Splits are costly and must pass code-quality heuristics 13 and 14. |
| ~~Narrower file lists~~ | Dropped 2026-11-13: there is no evidence that the lists are wrong, and splitting tasks costs merge-queue throughput. | — |

Do **not** raise `pin_reservation_max_active` for this. That knob was measured as harmful.

## Evidence chain for the recommendation

1. Starts are slot-bound: 94% of time at 18, no pass in which locks blocked everyone, and every
   empty pass was a PSI hold. So a pin only reorders starts.
2. Reordering is tiny: starving tasks' files are rarely free, and blocked watchdog pins get no park
   precedence. The replay gives +0 extra jumps at cap 5 and +2 per 7.5 days at cap 8.
3. The alarm cannot drain: it needs a ~290-task in-memory list to empty, and restarts re-announce
   it. The replay shows at most a −10% change.
4. The remembered throughput harm belongs to `pin_reservation_max_active` (PRD §7), not this cap.
5. The time actually goes to the merge phase holding slots and hot-file locks. That is your 11-05
   ruling's cause, and 8480 owns part of it.
6. Therefore the cap is the wrong lever in both directions. The alarm's design and the merge-phase
   and hot-file structure are the right ones.

## Caveats

- Ready-ages are lower bounds (11-01 backfill). Refusal counts reset on restart.
- 91 of 523 dispatches never emit a start event. 62 are deterministic or human-gate tasks, which
  emit none; ~29 were recovery-vetoed. The replay excludes deterministic winners.
- The pin-effect comparison is small (n=13) and confounded by restarts.
- The alarm model is calibrated, not validated. Its relative changes hold under both settings tried.
