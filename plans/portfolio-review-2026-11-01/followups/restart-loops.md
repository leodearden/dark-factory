# Restart and crash loops: root causes, what is fixed, what remains (follow-up to the 2026-11-01 portfolio review)

Lead: fable `/team` session team-df-903857, 2026-11-02. Read-only investigation; seat outputs under
`followups/restart-loops-seats/` (lead-scout, S1 fate of shutdown-blocked merges, S2 restart-driver
attribution, S3 startup-fatality audit, S4 owners, S5 detection, C1 critic). Every figure carries the rule
it was computed under, in brackets or in the seat section cited. Times are UTC.

## 0. The answer

Leo asked whether the root causes have been found and addressed. Split the three things the review
lumped together:

| incident class | root cause | found? | addressed? |
|---|---|---|---|
| (a) reify crash loop 2026-10-26 | a deterministic startup crash on durable on-disk state: an uncaught `WorktreeConflictError` from the lane-checkout reconciler on `_lane-29`, repeated 193 times at ~100 s per cycle for 5 h 12 m; systemd's StartLimit, the watchdog and every breaker were blind to it by construction | yes, by this follow-up (no seat of the review had it) | the one step is non-fatal since 10-29 (task 4053, deployed 11-01). The CLASS is open: four other startup steps are still fatal on durable state, the merge worker still starts before them, and nothing detects or pages a slow crash loop |
| (b) deliberate restarts (sc 10-15..19, df 94 blocks) | the 8 h fleet-redeploy cadence, and from 10-19 to 10-23 eight explicit deploy tasks plus the coordinator's force-fire, each stop blocking every queued merge with "Merge worker shutting down" | yes | mostly, by the 72 h cadence in force since 10-23 (one block fleet-wide on the 11-01 sweep). The block itself is a re-verify, not a lost landing; the two zero-landing days on sc were merge-queue HALTS, not restarts. Residuals: the drain protects only in-flight verifies; reify's shutdown exceeds TimeoutStopSec=90 on 16 of ~33 stops; the fused-memory tier has no lease |
| (c) `eligible_since` re-stamp 2026-10-23 | a drained fleet sweep (force-fired, "owed 47531 s") restarted df at 05:06 and the in-memory eligibility clock re-anchored every pending task; the 72 h idle backstop then fired for all of them on 10-26 | yes (the review had it) | yes, twice over: task 3821 landed 10-28/29 and (i) anchors are durable in `scheduler_eligibility.db` (1,763 rows on 11-02, oldest 11-01 08:03), forgotten only when a task leaves `pending`; (ii) the age-only idle backstop that fired the flood no longer exists: `starvation_watchdog.py::classify` gates on `park_blocked` or on lock refusals AND age, and starvation alarms no longer feed the EWA breaker's numerator. Deployed 11-01; all 5 STARVATION records since are `park_blocked`. A re-stamp flood cannot recur by the 10-26 route |

The thing that cost whole landing days was neither restarts nor crash loops on their own. The sc zero-landing
days were halts; the reify loop cost 4 tasks ~1.5 h each; the df 94 blocks cost one re-verify each. What
restarts DO cost, and what is still unowned, is in section 5. The ranked options are in section 6.

## 1. What the incidents actually were (corrections to the review's figures)

The review (§3.2 item 4, R2(d)) reads "194 runs a day is ~8 an hour, under the StartLimit" and "no seat found
the loop's driver". Both need correcting.

- **reify 10-26 was a crash loop at ~37 starts/hour, not 8.** [rule: `runs` rows in
  `/home/leo/src/reify/data/orchestrator/runs.db` with `started_at` on 10-26: 194, first 05:33:25, last
  10:45:37, every row but the last with `completed_at` 15–60 s after start and `total_tasks=0`.] The unit
  journal has 193 `Main process exited, code=exited, status=1/FAILURE` lines and one `status=9/KILL` (the
  drained sweep's stop at 05:31–05:33, where `stop-sigterm timed out` at 90 s). All 193 failures end in the
  same exception: `orchestrator.git_ops.WorktreeConflictError: Refusing to commit in
  /home/leo/src/warm-lanes/worktrees/_lane-29: 1 unresolved conflict(s) in the index:
  crates/reify-audit/src/p5_phantom_done.rs`, raised from `harness.py::Harness._reconcile_lane_checkouts`
  → `git_ops.py::GitOps.detach_lane_checkout` → `commit('chore: save WIP before lane-checkout reconcile
  detach')`, after "Lane-checkout reconciler: 8363-rebase absent from a healthy status read (task deleted)".
  [rule: `journalctl --user -u orchestrator-reify.service` 05:30–11:00, lines matching `Main process
  exited` and `Error:`; lead-scout.md items 2–4.]
- **The 579 "merges blocked" were 3–4 tasks blocked once per iteration.** [rule: `merge_finalized` events
  with `Merge worker shutting down` on 10-26 by hour: 48/99/108/114/120/90, distinct `task_id` per hour
  3–4; S1 table B: 4 distinct tasks, all re-queued within 30 min of the loop ending and all landed, median
  1.5 h after the last block.] Each iteration re-enqueued the journaled entries, created merge worktrees,
  started a verify, then blocked them on the crash; `reap_merge_verify_survivors` killed the previous
  iteration's verify subtree at every start.
- **sc 10-15..19 was not a loop.** [rule: sc `runs` rows 10-14..10-20: one run per ~8 h, 17 runs; S2
  cadence table: median gap 8.4 h for all four units 10-12..10-19.] Each 8-hourly stop blocked 13–38
  queued entries on 13–29 distinct tasks [S1 per-restart buckets]. **The two zero-landing days were
  merge-queue halts, not restarts**: `wip_halted` for task 798 from 10-15 04:42 to 10-17 07:00 (~51 h) and
  for task 852 from 10-17 15:00 to 10-19 09:22 (~43 h), both "WIP overlaps merge diff: uncommitted PRD
  docs in project_root", both released by Leo's docket ruling; 10-16 and 10-18 have ZERO `merge_verify`
  events, so a re-verify backlog cannot be what stopped landings. The restarts inflated the block counts
  on exactly those days (99 and 69) because the same journaled entries were blocked at every stop. [rule:
  S1 §B sc table; lead spot-check: `merge_finalized state=wip_halted` events 10-15 04:42 and 10-17 15:00,
  `merge_verify` per day 10-14..10-20 = 14/5/–/32/–/36/52.]
- **df's 94 blocks are the same 8 h mechanism**, 1–16 per day on 10-11..10-23 (3 on the 10-23 sweep),
  then none until one on the 11-01 sweep [rule: `merge_finalized` shutdown blocks per UTC day in df
  runs.db; S1, corrected by C1].
- **The 10-26 df flood is confirmed as the 10-23 restart's re-stamp.** [rule: 2,347 `STARVATION_WATCHDOG`
  records in `data/escalations/archive/2026-10-26/`, filed 05:17–05:50, `dispatch-eligible for N s` with
  N in 259,431..260,166; fire time minus N places every clock origin in 10-23 05:13–05:45, 7–39 min after
  the 05:06:39 restart; S5 §B.] The 10-23 restart was a drained fleet sweep (pid 3887760) that
  force-fired on "pending restart owed 47531s", drained df for 77 min awaiting a merge verify, and
  restarted it at 05:05 [S5 §B].

## 2. (a) The crash loop: driver, why nothing stopped it, what is fixed, what is not

**Driver.** Durable state (a conflicted git index in a warm lane whose task had been deleted) met a startup
repair step that raises on it and is not caught. Deterministic repeat on every start.

**Why it ran for 5 h 12 m with no page** (each is a separate blind spot; all verified in code or journal):

1. **StartLimit cannot catch a slow crash loop.** The unit carries `Restart=on-failure RestartSec=10
   RestartSteps=4 RestartMaxDelaySec=60 StartLimitIntervalSec=600 StartLimitBurst=10`. The backoff steps
   10 → 15.7 → 24.5 → 38.3 → 60 s; once at 60 s every start-to-start cycle exceeds 60 s whatever the crash
   time, so eleven starts can never fit in 600 s. From a fresh start the limit trips only if the crash comes
   within ~15 s of start. On 10-26 the crash came 35–50 s in (after crash recovery of 49 plans and 9
   worktrees). Observed cycle ~100–110 s, ~6 starts per 600 s. The unit comment "blocks a true crash loop"
   is false for any crash after the first ~15 s of startup. [S3 §3a, lead-scout item 3; the same settings
   are on every orchestrator unit.]
2. **The watchdog's liveness probe skipped the unit 295 times.** `STARTUP_GRACE_SECS = 120` in
   `scripts/orchestrator-watchdog.py` skips any unit started less than 120 s ago; the loop cycled every
   ~100 s, so every tick read "started Ns ago; skipping probe (grace window 120s)". Zero dead/failed/wedged
   verdicts, zero revives. It reads neither `NRestarts` nor `Result`. [S5 §A2; S3 §3b.]
3. **Had StartLimit tripped, the watchdog would have undone it**: `restart_unit` runs `reset-failed` on the
   unit and its `.socket` sibling, which by its own docstring "clears the StartLimit state", then
   `restart --no-block`; the orchestrator revive has no minimum interval (fused-memory's has 3600 s). So the
   start limit is at best a 2–3 minute pause every ten starts, never the operator inspection the unit
   comment intends. [S3 §3b, `scripts/orchestrator-watchdog.py::restart_unit` docstring.]
4. **The process records nothing on a startup crash.** The exception leaves `Harness.run` through
   `try/finally`; `cli.py::run` catches only `CancelledError`; `finish_run` stamps `completed_at` with
   `total_tasks=0` (assigned only after `finish_startup()`), and `runs` has no exit-reason column, so a
   crashed start is indistinguishable from a clean idle start in runs.db and the dashboard. No event, no
   escalation, no marker file, no consecutive-failure counter. [S3 §2.]
5. **The breakers watch escalation volume, not restarts.** reify runs.db has 0 pause/resume/breaker events
   on 10-26; the reify escalation queue filed 6 records that day (18 the day before) and only 2 in the
   window, both `stale_l0_strand_sweep`; the three digests covering 10-26 have 0 hits for restart/crash/
   lane-29; the df L2 handover and the L2-watch memory note have 0 mentions. [S5 §A.]

**How it ended.** An interactive `/unblock` session working reify task 8363 found the loop in the journal and
ran `git add` on the marker-free `p5_phantom_done.rs` in `_lane-29` at ~10:45; esc-8363-3's resolution
records it as a side-finding ("crash-looped orchestrator-reify via uncaught WorktreeConflictError in the
lane-checkout reconciler (existing DF #4053); unwedged 10:45Z") and dates the loop from "~10:21Z", i.e. the
session did not know it had been running since 05:33. [S5 §A4; lead read of
`/home/leo/src/reify/data/escalations/archive/2026-10-26/esc-8363-3.json`.]

**Cost.** 5.2 h of no reify dispatch (every run had `total_tasks=0`); 4 merge entries re-verified and
killed ~190 times each, all landed within 1.5 h of the loop ending; ~49 s CPU per iteration of crash
recovery; 193 transcript sweeps and lane reconciles. No landing lost that the data can attribute. [S1 §B;
S5.]

**What is fixed.** Task 4053 (df, done 10-29 13:15, merged 6d2f46a04b): `detach_lane_checkout` returns False
when its WIP snapshot is refused, and a try/except wraps startup step 2c2 (be77e39835 "startup step 2c2 is
non-fatal"); ecda4e7ce5 (10-30) refuses a task branch held by a QUARANTINED lane instead of detaching it.
Deployed with the 11-01 sweep. Note: 4053's record says the defect was verified on 2026-08-12, eleven weeks
before it looped. Follow-ups filed by 4053: 7701 (per-lane fault guard in `_reconcile_lane_checkouts`),
7957 (a refused-WIP lane stays pinned with only a WARNING, no escalation), 7702 (fixture dedupe); all pending.
[S4 §M1.]

**What is not fixed (the class).** S3's audit of `Harness.run`'s startup sequence, verified by reading the
call sites:

| step | can crash the process? | on durable state? | runs after the merge worker starts? |
|---|---|---|---|
| 2c `_recover_crashed_tasks` | yes: no wrapper, no per-worktree isolation; `_sweep_orphaned_transcripts` catches only `OSError`; three `_lane_lifecycle.transition` calls can raise `IllegalLaneTransition` | yes (worktrees, lane records, plan files) | yes |
| 2c2 `_reconcile_lane_checkouts` | no since 4053 (step wrapped; detach refuses) — but 7701 notes one lane's error still abandons later lanes | — | yes |
| 2d `_reconcile_stranded_in_progress` | yes, BY POLICY: catches only `SetTaskStatusRejected`, comment says unexpected exceptions "intentionally propagate"; same unwrapped call mid-run | yes (per-task plan.lock/owner_pid, task-store rows) | yes |
| 2e `_reap_orphan_worktrees`, Diff 7 `_reconcile_terminal_lanes` | no wrapper; callees do not raise today | yes if they regress | yes |
| `_start_merge_worker` `MergeLivenessConfigError` | fatal by design (bad config loops forever) | config | — |
| `mcp.start`, escalation port bind, `get_statuses` | fatal but transient (fused-memory down) | no | — |

The "startup must not fail" rule is enforced by about eleven separate hand-written try/excepts and is
absent at 2c/2d/2e/Diff 7 (code-quality heuristics 10, uniform invariants, and 11, SPOT). The merge worker,
including `reap_merge_verify_survivors`, starts at step 1b via `LifecycleRegistry.start_all`, queued merges
are recovered at 1c0a, and every step from 2c onward runs after that; so any remaining startup crash still
costs one verify kill plus N shutdown blocks per iteration, exactly as on 10-26. [S3 §1, with pins in the
seat report.] **No task owns** a generic startup-quarantine / consecutive-startup-failure / all-reconcilers-
non-fatal change, StartLimit re-sizing against the backoff, NRestarts paging, or orchestrator crash-loop
detection in the watchdog [S4 §M1–M2: terms searched are listed there]. Adjacent pending startup-fatal
fixes: 6615, 6662, 7277.

## 3. (b) Deliberate restarts: drivers, cadence, and what a restart costs

**Who restarted what** [S2; rule: every `runs` row 10-01..11-01 for df/reify/sc/scp attributed from the
unit's systemd lines, the harness's `fleet_drain` line and the watchdog/sweep journals in a ±3 min window;
the 8 starts per project before 10-13 are `unknown_pre_journal` because `journalctl --user` retains nothing
earlier; reify's 10-26 chain collapsed to one row]:

| driver (journal era, 10-13..11-01) | df | reify | sc | scp |
|---|---|---|---|---|
| merge-landed coordinator, drained | 11 | 14 | 21 | 12 |
| coordinator, no `fleet_drain` line (legacy merge-idle gate) | 4 | 2 | 0 | 5 |
| coordinator FORCE-FIRE (all drain outcomes) | 6 | 5 | 2 | 5 |
| explicit deploy task (`orch-redeploy-restart-7201..7355`) | 6 | 7 | 8 | 7 |
| watchdog staleness sweep | 1 | 1 | 1 | 1 |
| sweep, trigger unresolved in two probes | 6 | 4 | 2 | 4 |
| crash exits | 0 | 193 (+1 sweep start) | 0 | 0 |
| liveness revive / module-removal redeploy / operator manual / host boot | 0 | 0 | 0 | 0 |
| stops that needed SIGKILL (`stop-sigterm timed out`, TimeoutStopSec=90) | 4 | **16** | 0 | 2 |

Every deliberate restart in the retained journal went through `restart-all-orchestrators.sh`; the sweep takes
all units, so ~33 sweeps explain ~33 starts per project. Not one watchdog liveness revive occurred.

**Cadence** [S2; rule: gap between consecutive `runs.started_at`, crash chain collapsed]: 10-01..10-12
median 71.6 h; 10-12..10-19 median 8.4 h (the 8 h era); **10-19 12:00..10-23 12:00 median 5.5–6.1 h, 13
sweeps in 3.5 days, i.e. Leo's 10-19 ruling (task 7172) did not take effect until 10-23**: eight
`Deploy: fleet redeploy` tasks (7201, 7213, 7216, 7238, 7282, 7294, 7323, 7331, all done 10-20..10-22, the
review-all sitting deploys) each ran the sweep as their `before_done`, and the coordinator force-fired with
"pending restart owed 6808–57287 s" against its 4500 s bound; 10-23..11-01 median 72.8–73.9 h (10-23 05:06,
10-26 05:57, 10-29 07:30, 11-01 07:57). The 10-26 sweep that opened the reify crash loop and the 10-23 sweep
that re-stamped df's clocks were both this scheduled 72 h redeploy.

**What one restart costs a queued merge** [S1 §A, code; §B, store]:

- `SpeculativeMergeWorker.stop` resolves queued entries with the shutdown outcome and cancels an in-flight
  verify. `intake.py::_on_finalized` emits `merge_finalized state=blocked` for it; a second callback,
  `intake_buffers.py::MergeIntake.buffer_owned_request::_on_terminal`, KEEPS the `merge_queue.json` record
  for this reason only (removes it for every other terminal). The task's status is unchanged (slots are
  cancelled with `block_reason='shutdown_drain'` before the workflow could read the result), no escalation
  is filed, and `Harness._recover_pending_merges` re-enqueues the record at the next start, where it
  re-verifies from scratch.
- So the cost is **one full merge verify per queued-or-in-flight entry per restart** (~15 min median on sc;
  949 s measured on df 6373 on 11-01), plus queue latency. [rule: S1 §B; re-queued within 30 min of the
  next run start: sc 58/60 tasks, df 87/90, reify 4/4; later landed: sc 57/60 (the other 3 are `done`,
  landed another way), df 82/90 (7 pending for other recorded reasons: coalesce supersession,
  `train_incomplete`, `branch_bug`; 2 `iact-*` ids not in the store), reify 4/4.]
- **Human touch: 1 in ~154 tasks**: df 7199 on 10-20, where the startup reconciler attempted a
  Recovery-DONE before the recovered merge reached main, `done_provenance` refused it, the task blocked,
  esc-7199-1 (L1) and -2 (L2) were filed and the L2 watcher stamped it done ~1.5 h later. Task 7277
  (pending) owns the general shape ("startup reconcile re-pends a task whose merge request was just
  recovered"). [S1 §C3; S4 §M3.]
- **Per restart now**: 10-20 (6–7 sweeps that day) df 16, reify 3, scp 5, sc 3, scoping-intake 3; across
  the four 72 h sweeps (10-23, 10-26, 10-29, 11-01) df blocked 3 / 0 / 0 / 1 queued entries, and on the
  11-01 sweep the whole fleet blocked **1** (df 6373, re-landed in 19 min). [S1 per-restart table; C1
  added the 10-23 count.]
- **The drain does not cover queued entries.** `scripts/drain_check.py::classify` returns `idle` once
  admission is halted and no verify is in flight, regardless of queue depth; the 11-01 df drain reported
  `drained` after 1196 s and still blocked the queued 6373. At one entry per 72 h this is negligible; it
  returns as a cost only when a restart meets a deep backlog (a halt that is lifted just before a sweep, a
  crash loop, or a project whose lane runs 71% loaded like df).
- **A misleading signal**: `merge_finalized state=blocked` is emitted for a request the journal keeps alive,
  so any consumer that reads merge_finalized as terminal over-counts blocks (the review's 579 / 286 / 94
  are this count). [S1 §C3.]

**Residuals in the restart machinery itself** [S2 surprises, S4 §M3, §M7; OPERATIONS.md §8]:

- Reify's shutdown exceeds the 90 s `TimeoutStopSec` on 16 of ~33 stops (sc: 0 of ~33), coinciding with
  `sqlite3.OperationalError: database is locked` tracebacks during shutdown; a SIGKILLed stop is the one
  exit no trap can catch, which is how a verify ends as `verifies_killed` and how state can be left
  half-written (the 10-26 loop began with one). Closest owner: 7247 (pending, "bound the live-workflow wait
  for cancel-immune tasks"); no task owns the >90 s reify shutdown as such.
- Stops with no `fleet_drain` line (df 9, reify 7, scp 12, sc 0, mostly 10-13..10-20): the legacy no-drain
  path or an unacknowledged request. The first sweep after a deploy of drain code still meets old-code
  units (OPERATIONS.md "Reading a drain").
- The coordinator's force-fire carried 16 h of "owed" restart debt on 10-23; 4763 (pending; its deps 4754
  and 4755 are done) and 7278/7307/7314 (pending) sit on the coordinator; 5719 (pending, updated 11-01) is
  the fused-memory tier's missing lease.
- Deploy-restart human gates: the deterministic convention (`task_kind=deterministic`, `before_done`
  `restart-all-orchestrators.sh --drain`) already exists and is in use (7828, 7829, 8087, 8130 pending);
  X1's 26 "restart unit X after task N merges" gates have no converter task, and 7276 (pending) records that
  dependents can be satisfied before the restart lands. [S4 §M6.]

## 4. (c) The eligibility-clock re-stamp: fixed, verified in code, not yet exercised

Task 3821 (done 10-29 14:45, merged 8982ed198a; commits 0cc22c2e40, 932cc1947c "scheduler runs the retuned
starvation predicate on a durable clock", ab66f3899e "starvation watchdog alarms no longer feed the EWA
breaker's numerator"). Lead verification of the mechanism in
`orchestrator/src/orchestrator/eligibility_clock.py`: `EligibilityClockStore` persists `(project_root,
task_id, eligible_since)` in sqlite; `EligibilityClock._load` reads all anchors at construction (once per
Scheduler, `scheduler.py` line ~2580, constructed by `harness.py` ~1808); `observe` stamps only candidates
with no anchor; `forget` is called from `scheduler.py::_forget_starvation_inputs`, whose rule is "every task
no longer pending ... a pending task that merely sat out a tick (cooldown, PSI hold) keeps them". So a
restart re-stamps nothing, and a scheduler pause/resume does not reconstruct the clock, which also closes the
10-23→29 watch note's trap "starvation clock re-stamps on resume too" (that note predates 3821). Deployed
11-01.

**The flood's predicate is gone, so the clock alone cannot reproduce it.** C1 corrected the lead's first
draft here, which had the sign backwards: the 11-01 start found an empty store and anchored every pending
task at ~08:03 on 11-01 (store: 1,763 rows, `min(eligible_since)` 2026-11-01T08:03:21), so on 11-04 ~08:03
every anchor crosses 72 h at once, which is exactly the OLD flood condition, and a flood on 11-04 would
have meant the durable clock WORKED (anchors survived the restart) while the predicate was unchanged. What
prevents it is the other half of 3821: `starvation_watchdog.py::classify` (commit 0cc22c2e40) has no
age-only gate any more; it fires `park_blocked` when a task's park is held by another task for
`park_blocked_age_secs`, or the lock-refusal gate when refusals ≥ `skip_threshold` AND age ≥
`idle_only_secs`, and the config docstring says it "replaces the age-only idle_only backstop". All five
STARVATION records filed since 11-01 are `park_blocked`. The right check after the 11-04 sweep is therefore
not "no flood" but "the anchors survived": per-task `eligible_since` identical before and after the restart
for tasks pending on both sides, and `min(eligible_since)` still 11-01 08:03 (option 8; not the row count,
which churns every tick). `classify` also has a `dual` gate (skips ≥ threshold AND age ≥ idle secs); none
of its gates is age-only. C1 verified that the only STARVATION days in the October archive are 10-04 (3
records) and 10-26 (2,347), so the 10-26 flood was the only one even before the fix. C1 also notes, unverified, that the old code did NOT flood on the 10-29 or
11-01 restarts (0 records), so the 10-26 flood was the only one even before the fix; 10-29 is likely
explained by `_rehydrate_starvation_escalated` seeding from the 1,171 then-pending L1s, 11-01 is
unexplained. The EWA breaker still counts escalation volume, so a different mass-filer could still pause
dispatch; the stale-L0 strand sweep records that each restart files (63 L2s in the window per X1) are owned
by 7262 and nits 7788–7790 (pending); 5223 and 4988 (done) route info-L0s. [S4 §M4; S5 §B; C1 §A1.]

**2026-11-04 check (task 8314, option 8): PASS.** The restart checked is the 11-04 sweep: `run-67e561c2224e`
(code 0f0fbc44ae) ended 08:02:34Z, `run-b594c419895e` started 08:04:22Z on b8763514d6, and its first
scheduler tick was 08:06:12.7Z. Basis: the 11-03 07:30:23Z snapshot
`data/orchestrator/eligibility-anchors-snapshot-2026-11-03.json` (1,802 rows), compared read-only at
11:37:52Z against the live `eligibility_anchors` (read through a stat-guarded byte copy, because a `mode=ro`
open of the WAL store is refused in a sandboxed session whenever it is fully checkpointed and has no `-shm`)
and live `tasks.db` status. `min(eligible_since)` is 2026-11-01T08:03:21.302615Z before and after. Of the
1,784 tasks present in both and pending now, 1,771 have a byte-identical `eligible_since`, and each of the 13
mismatches has a `lock_acquired` in runs.db after the snapshot. Eleven were dispatched by the old run after
the snapshot and were either cancelled at its 08:02:33–34Z shutdown (3089, 4650, 5245, 6015, 6145, 6154,
6454, 6455, 6487, 7857) or still held at it with no recorded outcome (6436, dispatched 04:00:27Z); all eleven
were re-stamped at 08:06:12.595585Z by the new process's first tick. One was dispatched 11-03 19:23Z,
requeued 19:52Z (`plan_blast_radius_lock_conflict`) and re-stamped 19:55:45Z, before the restart (4161). One
was dispatched by the new run at 08:06:12.7Z, requeued 11:03:48Z and re-stamped 11:05:12Z (6106). The eleven
first-tick re-stamps are not a restart re-stamp: those tasks were not pending at shutdown, because
`orchestrator/src/orchestrator/scheduler.py::_settle_starvation_on_dispatch` forgets an anchor at dispatch,
so their return to pending started a fresh age. No task that stayed pending across the restart was
re-stamped. Nor did any lose its anchor: each of the 18 snapshot tasks absent from the live store has a
dispatch event after the snapshot, and the only one of them still pending, 6163, was dispatched by the new
run at 11:24:14Z (`lock_acquired`; checked at 12:07Z, when the store had not been written since 11:24:14Z
and so still held the anchors compared at 11:37:52Z). The row count (1,802 in the snapshot, 1,849 live) is
not a criterion; the store churns every tick.

## 5. Owners, summarised (every status read from the df store on 11-02; S4 has the full table)

| mechanism | landed | filed / pending | no owner (terms searched by S4) |
|---|---|---|---|
| startup crash on durable state | 4053 (10-29) | 7701, 7957, 7702; adjacent 6615, 6662, 7277; **7745** (pending, 11-01: startup Recovery releases a `_spec` warm lane the merge worker has just acquired, so a later acquire resets it under a running verify; seen 10-26 10:41–10:45) is the merge-worker-before-recovery ORDERING class that option 2 fixes | generic startup quarantine / consecutive-failure counter / uniform non-fatal wrapper (C1 re-searched: 'crash loop', 'NRestarts', 'start-limit', 'startup failure', 'consecutive', 'shutdown_drain', 'merge_suspended', 'queued entries'; only 7745 was missed) |
| StartLimit, restart-rate paging | unit StartLimit unchanged since 28628d71b8 (05-28); RestartSteps from 3408 (08-01); 3764 fm-liveness cap, 3766 | 5739, 5700 (watchdog logging) | StartLimit sized to the backoff; NRestarts/Result paging; watchdog honouring start-limit-hit |
| shutdown-blocked merges / drain / lease / cadence | 5371, 7297, 4754, 4755, 5020, 7172, 7175 | 5719 (fm lease), 4763, 7278, 7307, 7314, 7830, 7098, 7276, 7275, 5623, 6451, 7277 | drain covering queued entries; `merge_finalized blocked` emitted for a kept request |
| eligibility re-stamp | 3821 (10-29) | 7262, 7788–7790 (stale-L0 sweep), 4691, 7699, 3986 | — |
| verify-survivor reap / verify cost on restart | — | verify-result-cache PRD 5651 done, 5652–5659 pending (1 of 9); 7254 | `reap_merge_verify_survivors` itself |
| TimeoutStopSec / >90 s shutdown | 875, 2700, 3306, 6038 (precedents) | 7247 | reify's locked-db shutdown |
| deploy-restart gates | convention in use (7828, 7829, 8087, 8130) | 7276, 6592, 6529 | converter for X1's 26 gates |

## 6. Options, ranked

Ranking weighs Leo's 11-02 rulings: factory self-work that improves reliability or required supervision is
high value and compounds across every hosted project; reliability and autonomy beat speed; one extra verify
is an accepted price, a silent jam is not. Expected effects are per incident class and carry their rule.

| # | option | mechanism | expected effect | ramifications | needs from Leo |
|---|---|---|---|---|---|
| 1 | **In-process crash-loop breaker with a page** | A durable consecutive-startup-failure journal in `orchestrator_data_dir` (step name + exception type, written by one `except BaseException` around the startup block, cleared at `finish_startup`). On the K-th consecutive failure at the same step (K=3) file one L2 and skip that step (startup quarantine) for this start. Plus an `exit_reason` / `startup_completed_at` column on `runs`. [S3 G3] | Bounds a 10-26-shaped loop to K iterations (~5 min, ~3 kills) instead of 5 h 12 m and 193, and pages within minutes instead of never. Makes a crashed start distinguishable in runs.db. | Skipping a repair step must be safe: holds for the idempotent sweeps (2c, 2c2, 2e, Diff 7); 2d (`_reconcile_stranded_in_progress`) is fatal by stated policy and needs a decision (escalate at startup, propagate mid-run). Does not remove the per-iteration merge cost (see 2). The L2 adds one record per loop, not a flood. | a /prd (G5 design-first: the step list and the K/skip policy); the 2d policy call |
| 2 | **Start the merge worker and the survivor reap after `finish_startup()`** | Construct the worker at 1b so recovery can enqueue, but `create_task(worker.run())` and `reap_merge_verify_survivors` only after the startup block completes. [S3 G2a] Task **7745** (pending) is the same ordering defect seen from the warm-lane side (Recovery releases a `_spec` lane the worker just acquired); fold it in. | A startup crash then costs the merge lane nothing: no verify kill, no shutdown block, no `merge_finalized blocked` per iteration (10-26: 579 events and ~190 verify kills → 0), and 7745's lane reset under a running verify cannot occur. | Merge landing waits ~30–60 s longer after every restart (the startup block's duration). The 1b "worker and reap are live before recovery" contract must be restated. The "→ 0" holds only because `harness.py::_stop_merge_worker` skips `stop()` when no worker task exists; a design that calls `stop()` on the constructed-but-not-started worker would re-emit N blocks per iteration, so state that invariant in the PRD (C1 §A5). Premise to verify in design: enqueue must work with the worker constructed but its loop not running (C1 §B). | approval; same PRD as 1 |
| 3 | **One uniform non-fatal startup wrapper** | `_run_startup_step(name, coro)`: catch `Exception`, log with `exc_info`, emit an event, file/refresh one L1 keyed by step; an explicit opt-in `fatal=True` list (empty but liveness config). Per-entry isolation inside 2c/2e/2d loops so one bad worktree or task is quarantined and the sweep continues (7701 is this for 2c2). [S3 G1] | Closes the class 4053 patched one instance of: 2c, 2d, 2e, Diff 7 stop being able to loop the process. Deletes ~11 ad-hoc try/excepts (heuristics 10, 11). | A step that half-completes and leaves the scheduler unsafe must be named fatal explicitly; today none is except liveness config. Changes 2d's stated policy. Overlaps 7701; fold it in. | approval; same PRD as 1 and 2 (they are one chain: wrapper → ordering → breaker) |
| 4 | **Watchdog reads restart rate and honours start-limit-hit** | `main()` reads `systemctl show -p Result,NRestarts`; on `Result=start-limit-hit` do NOT `reset-failed` and page; alarm when NRestarts grows by ≥5 in 30 min whatever the state; a per-unit liveness revive min-interval mirroring `FM_LIVENESS_RESTART_MIN_INTERVAL_SECS`. [S3 G5] | Independent second detector that works even when the process cannot write (SIGKILL, import error, venv broken): catches the 10-26 shape (6 starts/10 min, never `failed`) that StartLimit misses. | A unit left in start-limit-hit stays down until a human acts; that is the trade the unit comment already claims to make. The revive min-interval delays a genuine wedged-unit revive by that interval. **Not small (C1 §A3)**: the watchdog has no escalation-filing path at all today (it only probes `/alive` over `urllib`, logs, and runs `systemctl`), and a host-level alarm has no obvious home among per-project escalation queues; the review's R2(d) "keep an L2 page when runs/hour exceed a threshold" has the same gap. Design choice needed: file into the dark-factory queue as the host's own, or a new host-level sink. | a design decision on the alarm's home; then one task (not merely behind 5700) |
| 5 | **Size StartLimit to the real cycle** | `StartLimitIntervalSec=1800`, `StartLimitBurst=10` on the orchestrator unit template (catches crashes up to ~120 s into startup: 10×(T+60) < 1800). [S3 G4a] | Backstop only; gives no page and is undone by the watchdog unless 4 lands first. | Cap-rotation churn and deploy sweeps must stay under 10 starts/30 min (they do: one start per sweep). Restart-tier: takes effect at the next unit restart. Without 4 it is cosmetic. | approval; pair with 4 |
| 6 | **Drain covers queued entries, or stop emitting `blocked` for them** | Either (a) `drain_check.classify` treats a non-empty queue as `verifying` up to the per-unit cap (lets the queue drain before restart), or (b) keep the restart but emit a distinct `merge_suspended` instead of `merge_finalized state=blocked` for the shutdown reason, since the journal keeps the request. [S1 §C6] | (a) removes the one re-verify per queued entry per restart (4 entries over the four 72 h sweeps, 3/0/0/1; 13–38 per restart under an 8 h cadence or after a halt). (b) removes a misleading terminal signal that over-counted 579/286/94 in the review and the digests. | (a) lengthens every sweep by up to the queue's verify time and interacts with the sweep-wide ceiling and the 4 h lease; at 72 h cadence the saving is ~1 verify/72 h, so (a) is low value now and would matter only if the cadence tightened or a halt lifts just before a sweep. (b) touches every merge_finalized consumer (digests, burndown, dashboard). | (b) is cheap and clarifying: approve as a task. (a) only if the cadence ever tightens |
| 7 | **Reify's >90 s shutdown** | Root-cause the `database is locked` at shutdown and the unbounded live-workflow wait (7247), then either fix or raise reify's `TimeoutStopSec` to the measured drain (restart-tier). | Ends the SIGKILLed stops (16 of ~33 on reify) that leave half-written state; 10-26 started from one. | A longer TimeoutStopSec delays every sweep by up to that much for reify and must fit the sweep ceiling; fixing the lock is the real repair. | approval to raise 7247's priority and widen it to the reify lock |
| 8 | **Confirm the durable clock survives the 11-04 sweep, nothing else** | Snapshot `(task_id, eligible_since)` from `data/orchestrator/scheduler_eligibility.db` (read-only) just before and after the 11-04 ~08:00 sweep; pass = `min(eligible_since)` is still 2026-11-01T08:03:21Z and, joined on task_id, every task pending on both sides has an identical `eligible_since`. The row COUNT is not a criterion: the store churns every tick (1,582 rows stamped in the 11-01 08:xx hour, 1,763 by 11-02 12:40; non-pending tasks are forgotten), so an unchanged count would be a false test (C1 delta pass 1). A task dispatched and reverted to pending between the two snapshots is legitimately re-stamped; check runs.db for a dispatch before calling such a mismatch a failure. Do NOT read "no STARVATION flood" as the test either: the age-only predicate that fired the flood no longer exists (section 4). | Confirms the anchor half of 3821 on the first restart with a populated store. | None. | nothing now; a two-line check on 11-04 (the L2 watcher can do it) |
| 9 | **Coordinator debt and the fm lease** | 4763 / 7278 / 7307 / 7314 (coordinator), 5719 (fm lease): already filed; raise none above medium. | The force-fire "owed 57287 s" and the fm collision are bounded residuals at 72 h cadence. | Restart-tier work; cost is a sweep each. | nothing; leave as filed |
| 10 | **Do not add more limits** (the review's R2(d) stance, confirmed) | — | — | Another StartLimit or a tighter watchdog without 1–4 would reproduce the 10-26 blind spots. | — |

Recommended bundle: **1 + 2 + 3 as one PRD** (one chain in `harness.py` startup: wrapper → ordering →
breaker; folds in 7701 and 7745; the design questions are 2d's policy, K, and the stop-guard invariant),
**4 + 5 as one watchdog/unit task after a decision on where a host-level alarm is filed**, **6(b) as a
hygiene task**, **7 as a priority bump on 7247**, and **8 as a dated check**. Together they turn a
future startup crash from "5 h, 193 iterations, 4 merges killed 190 times, no page" into "≤3 iterations,
0 merge cost, one L2 within ~5 min, visible in runs.db", on every hosted project at once.

## 6a. Leo's rulings and what was done (2026-11-03)

Leo ruled: PRD yes; alarm question delegated to the watcher-hierarchy redesign session; hygiene yes; no
bump for 7247 (reify is paused); check yes. Applied:

- **PRD (options 1+2+3, and 4+5)**: already filed on 11-02 by a sibling session as
  `plans/startup-hardening-prd.md`, tasks 8267 α runner (folds 7701), 8268 β merge loop after
  `finish_startup` (folds 7745), 8269 γ K=3 breaker + `runs.exit_reason`, 8270 δ watchdog alarm, 8271 ε
  `StartLimitIntervalSec=1800` (in progress), 8273 ζ deploy gate, 8274 η 7-day observation. Nothing to
  author; that session took two provisional rulings (2d non-fatal at startup; alarm into df's queue).
- **Alarm's home**: delegated per Leo to the watcher-design thread (study `followups/watcher-design.md`);
  8270's details now record that its df-queue choice is provisional until that session rules.
- **Hygiene 6(b)**: filed; the curator folded it into existing task **6310** (pending), whose step 2 is
  the same change (a shutdown-resolved request is suspended, not terminal). Owner missed by S4.
- **Bump 7247**: not done, per Leo.
- **Check (option 8)**: task **8314**, dated milestone 2026-11-04T10:00Z, compares the live
  `eligibility_anchors` against the pre-sweep snapshot
  `data/orchestrator/eligibility-anchors-snapshot-2026-11-03.json` (1,802 rows) with the per-task join
  and unchanged-minimum criteria, and appends its verdict to §4 of this report.

## 7. Roster and outcomes

| seat | model / effort | tokens | min | outcome |
|---|---|---|---|---|
| lead scout | fable (session) | — | — | reify loop root-caused from runs.db + journal; StartLimit arithmetic; 4053 found; sc "loop" refuted |
| S1 shutdown-blocked fate | sonnet / high | 127k | 19 | code path + store follow-up for 154 tasks; sc zero-days = halts (answer-changing, lead-verified) |
| S2 restart drivers | sonnet / medium | 166k | 105 | all 361 starts attributed; cadence 10-19→23 was 6 h not 72 h; reify 16 SIGKILLs (slow: journalctl) |
| S3 startup audit | opus / high | 122k | 32 | four steps still fatal; ordering; backoff curve from systemd source; watchdog defeats StartLimit |
| S4 owners | sonnet / medium | 102k | 11 | owners for M1–M7; 4053 defect dated 08-12 |
| S5 detection | sonnet / medium | 114k | 31 | nothing noticed; `/unblock` on 8363 ended it; 10-23 = force-fired drained sweep; re-stamp confirmed to ~1% |
| C1 critic | fable / high | 178k | 26 | five answer-changing findings, all lead-verified and folded in (§8); delta passes below |

Routing lessons: S2's journal attribution took 105 min on sonnet because `journalctl --user` over this host's
journal is slow; it would not have been faster on a higher tier, but it should have been split by unit into
four seats. Nothing was dropped or degraded; S2's 16 "trigger unresolved" starts and 32 pre-journal starts
are the only unattributed rows.

## 8. Unverified, and what the critic changed

Unverified by the lead: S2's SIGKILL counts and driver classes (journal reads, not re-run); S1's
re-queue/landing counts (sqlite rule stated in S1 §B, not re-run); S3's claim that 2e and Diff 7 callees do
not raise today (code reading); whether the 11-01 sweep's other eight units had empty queues (events are
silent on an empty queue); the identity of the `/unblock` session that ended the loop.

Critic fold-in (C1, fable/high, 178k tokens; full critique in `restart-loops-seats/C1.md`). Five
answer-changing findings, each verified by the lead against the store or code before folding in:
(1) §4 and option 8 had the sign of the 11-04 test backwards and watched a predicate 3821 removed; rewritten
around the durable-anchor check. (2) df's per-day block range is 1–16 and the 10-23 sweep blocked 3, so
the four 72 h sweeps cost 3/0/0/1 queued entries (option 6's basis). (3) Option 4 is not small: the
watchdog has no escalation-filing path and a host-level alarm has no home. (4) Task 7745 (pending) is the
ordering class option 2 fixes and was missed by S4; cited in §5 and option 2. (5) Option 2's "→ 0" depends
on `_stop_merge_worker` skipping `stop()` for an unstarted worker; the invariant is now named. C1 verified
as stated: the backoff curve (10.1/15.8/24.6/38.4/60 s per journal, systemd 255), the 193 count, the
watchdog grace and reset-failed, the §2 ordering table, the 3821 mechanism, the sc halts, the 11-01
single block, and every spot-checked task status. Suspected but unverified by C1: the old code did not
flood on 10-29 or 11-01; 14 of S2's 16 reify SIGKILLs; S1's counts; option 2's enqueue-without-loop
premise; option 5's cap-rotation bound; option 1's "skipping 2c is safe" (it lets 2d re-pend tasks).
Delta pass 1 (189k cumulative) found one answer-changing item: option 8's row-count criterion would
false-fail on a healthy, churning store; replaced by the per-task join and the unchanged minimum. Delta
pass 2 covered only that edit.
