# Weighting merge verifies too: what it buys Dark Factory now

2026-11-15. Evidence seat, read-only. Follow-up to `/tmp/team-weights-1112/REPORT.md` (the 11-12 study). Nothing in any
repository, config, task store, queue or service was changed on either host. Scripts and raw outputs are in
`/tmp/team-weights-1112/followup-1115/merge/` (each `<name>.py` writes `<name>_out.txt`).

**Code read.** Dark Factory `main` at `27c93d6f75` (main has since moved to `dc2c55f6d3`; nothing read here changed).
- `orchestrator/src/orchestrator/verify.py::_resolve_verify_timeout`, `::_resolve_scope_cpu_weight`, `::run_verification`
  (a merge verify always takes the cold-cache time limit), `::merge_verify_command_budget_secs`
- `orchestrator/src/orchestrator/verify_runner.py::build_merge_verify_spec`, `::_module_config_from_command`,
  `::_spec_field_overrides`, `::run_merge_verify_on_worktree`: what the laptop gets from the workstation and what it
  takes from its own config file
- `orchestrator/src/orchestrator/defaults.yaml`: the time limits, `merge_deep.chain_cap`, train settings
- `orchestrator/src/orchestrator/workflow.py::_submit_to_merge_queue` and `harness.py::Harness._run_slot`: a task waits
  for its merge result inside its task slot, and the slot is given back only after that
- `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker._maybe_log_queue_heartbeat` (the pipeline
  heartbeat is written every 300 s and only while something is in the pipeline), `chain_telemetry.py` (the meaning of
  `chain_items` and `landed_via_chain`), `event_store.py` (field meanings)
- The four projects' `dark-factory-orchestrator.yaml`, and on the laptop (over ssh, read-only)
  `~/.config/orchestrator/{dark-factory,reify,solar-challenge,solar-challenge-platform}-laptop.yaml` and the
  per-project scope drop-ins under `~/.config/systemd/user/`.

**Data.** Each project's `data/orchestrator/runs.db` up to about 2026-11-15T10:00Z, and the pause seat's per-minute
grid for 10-25 to 11-08 (`/tmp/team-weights-1112/pause/tl.pkl`). Before the pause (W0) = 2026-10-25T00:00Z to
11-08T16:00Z. "Landing" = a `merge_finalized` row with state `done`, the rule of the earlier study.

## 1. The answer in brief

1. **Including merge verifies helps mainly on the laptop.** On the workstation the limit is reify: its local merge
   verify (median 76 minutes, 90th percentile 124) can take only about a 1.25 to 1 factor before runs approach its
   180-minute limit. At that factor it saves Dark Factory 0.4 to 0.9 minutes. On the laptop, reify's merge verifies are
   the only contention that costs Dark Factory anything. A 2.5 to 1 factor there saves Dark Factory 0.8 to 1.5 minutes
   per laptop merge verify by the before-pause model, and 2.3 to 4.5 minutes by a fit on the 21 verifies since reify
   resumed. Reify's laptop runs stay well inside their limit at that factor.
2. **The landings are about twice the earlier study's figure. The cause is the merge lane, not the weights.** Each
   merge verify now lands about one task (1.06 since 11-14) where it used to land about half of one (0.47 before the
   pause). So each minute saved is worth roughly twice as many landings. A floor-respecting factor on the other
   projects' merge verifies and services, with their task verifies left alone, projects to **2 to 4 landings a day**
   (an upper bound). That is 4 to 9% of the 47 a day Dark Factory landed from 11-10 to 11-14. It rises to 3 to 7 a day
   if reify's laptop contention stays at its level since the resume. The earlier study's recommended design gives 1 to 2
   by the same rule.
3. **The lower projects pay in time to land, not in landings, as long as only their merge verifies are weighted.**
   Their merge queues are empty most of the time: reify's 48%, solar-challenge-platform's 72% and solar-challenge's
   86%, before the pause. Their landings are limited by their task verifies. Weighting their task verifies would cost
   them landings, because that is their bottleneck.
4. **The regime described in the revocation is only partly live.** Chains are live and have doubled landings per
   verify. Two changes are merged but run in no orchestrator: K = hosts + 1 with the host released at the verdict
   (8526), and chaining the head (8527). Two more are pending: the depth rule (8528) and the merge pool (9935, gate
   10069). Under the code that is running, a deeper queue does not land more per verify once about six requests are
   waiting.
5. **Dark Factory's supply of merge requests is limited by task slots, not by task-verify CPU.** A task holds its slot
   until its merge lands. Since 11-14, 63% of the 24 slots were held by tasks waiting on the merge lane, and the slots
   were full 91% of the time. Task-verify waiting and running take about 10% of a task's time before it queues (median
   5 minutes waiting, 7 running). Shifting CPU from Dark Factory's merge verifies to its task verifies would not deepen
   the pool. The pool is already deep (empty 4% of the time), and a slower lane would hold more slots. Inside reify and
   solar-challenge-platform the opposite holds: there the task verify is the bottleneck, and the revocation's direction
   is right.

## 2. The merge-lane regime today (question 3)

**What is live, and since when.** The same orchestrator code runs in every project; each project's config decides what
is switched on. Dates are when the running process loaded the code (restart times from each project's `runs` table and
`scripts/orchestrator-watchdog.py --report`, read at 2026-11-15T09:43Z).

| Change | Dark Factory | reify | solar-challenge, solar-challenge-platform |
|---|---|---|---|
| Retroactive "train" coalescing: queued requests batched into one verify | Train events 10-01 to 11-01T07:56Z; switched off in config 10-29 | Never (no train events) | Never |
| Deep merge-ahead chains: one verify covers up to 6 queued requests | `chain_cap: 6` since 10-27 | `chain_cap: 6` since 10-20 | Off (`chain_cap` 0, the default) |
| Recovery re-merges built on the live chain tip, not on main (task 8524) | Running since 11-10T23:48Z | Since the 11-10 fleet restart | Same code; no chains |
| A gate re-verify becomes a normal dispatch, successors re-anchor on it (8525). A chain's links are reserved for its tip verify, and the look-ahead is held (7720) | Since 11-14T00:56Z, with task slots raised from 18 to 24 at the same restart | Since 11-13T23:52Z | Same |
| K = hosts + 1, host released at the verdict (8526); the head chains when alone (8527) | Merged 11-14T13:46Z and 22:40Z. **Not running**: Dark Factory runs `24472e5ad2`, the others `d918f491e7`, both earlier | Not running | Not running |
| Bisect retry and a pass-rate depth table (8528); the pool taken in priority order (9935); batches chosen from evidence (human gate 10069) | Pending | — | — |

Reify's scheduler resumed 11-14T11:00:57Z. Solar-challenge and solar-challenge-platform are still paused.

**Measured, Dark Factory, by window** (`lane_windows_out.txt`; all measured; per day = count divided by window length).

| | Before pause (14.7 d) | Paused, before 11-10 deploy (1.4 d) | 11-10T23:48 to 11-14T00:56 (3.05 d) | Since 11-14T00:56 (1.4 d) |
|---|---|---|---|---|
| Tasks reaching the merge queue for the first time, per day | 30.9 | 35.0 | 45.6 | 56.5 |
| All merge requests (first and repeat), per day | 50.7 | 44.3 | 63.7 | 84.0 |
| Merge verifies per day (local / laptop) | 50.1 (22.2 / 27.8) | 60.7 | 74.5 (32.2 / 42.3) | 63.0 (23.2 / 39.8) |
| Share of merge verifies that passed | 79% | 99% | 80% | 59% |
| Landings per day | 23.7 | 36.4 | 47.3 | 66.6 |
| Landings per merge verify | 0.47 | 0.60 | 0.63 | 1.06 |
| Landings per passed merge verify | 0.60 | 0.61 | 0.80 | 1.80 |
| Share of landings made by a chain | 2% | 14% | 33% | 63% |
| Mean requests per verify (`chain_items`) | 1.06 | 1.22 | 1.70 | 2.26 |
| Pipeline empty, share of minutes | 11% | 12% | 10% | 4% |
| Mean requests in the pipeline | 8.1 | 8.6 | 9.1 | 15.5 |
| Minutes with 3 or more waiting beyond those being verified | 70% | 73% | 80% | 92% |
| Time from queueing to landing, median | 4.4 h | 3.9 h | 3.1 h | 4.2 h |
| Share of time a merge verify is running, local / laptop | 0.68 / 0.36 | 0.75 / 0.37 | 0.71 / 0.61 | 0.68 / 0.66 |

Rules. Pipeline depth: the `merge_heartbeat.depth` of the latest heartbeat at or before each minute, if no more than
330 s old. Otherwise the minute counts as empty: the worker writes nothing when the pipeline is empty and writes every
300 s otherwise. A restart also reads as empty, for a few minutes per window. "Waiting" = depth minus
`occupancy.inflight_total`. Time to land: `queue_to_land_secs` on landing rows. Host busy: the time-weighted share of
minutes in which a merge verify interval (end minus `duration_ms` to end) is open.

Daily, by the same rule (`depth_daily_out.txt`): the pipeline averaged 18.6 requests on 11-14 and was never empty that
day. On 11-15 it drained from 17 to 0.1 between 06:00 and 09:00Z, after three merge verifies landed 18 tasks in two
hours (`df_hourly_1114_out.txt`). 19 tasks started in the hour those slots were freed, and the pipeline then sat nearly
empty until their work could reach it. **This is how the slots couple the pool to supply** (§3).

**Other projects** (measured, same rules). Reify, before the pause: 20.3 merge verifies, 17.4 landings and 17.7 new
requests a day. It landed 0.86 per verify; its pipeline was empty 48% of minutes and had 3 or more waiting 4% of the
time. Since its resume (0.96 day): 29.1 verifies, 23.9 landings, 0.82 per verify, empty 20% of minutes. Solar-challenge,
before the pause: 1.00 landing per verify, empty 86%. Solar-challenge-platform: 0.84 per verify, empty 72%. None of the
three has a deep queue. All three land about as fast as requests arrive.

**Does a deeper pool land more per verify today?** Not beyond about six waiting (`chain_vs_depth_out.txt`; measured).
For each merge verify since 11-14T00:56, waiting was read at its start:

| Waiting at start | 3 to 6 | 6 to 10 | 10 to 15 | 15 or more |
|---|---|---|---|---|
| Verifies (11-10 to 11-14 / since 11-14) | 31 / 2 | 80 / 7 | 78 / 17 | 17 / 57 |
| Mean requests per verify, 11-10 to 11-14 | 1.32 | 1.93 | 1.78 | 1.82 |
| Mean requests per verify, since 11-14 | 2.0 | 2.57 | 2.53 | 2.14 |

Only speculative items chain; the head never does (0 of 143 and 0 of 38 non-speculative verifies chained). That is the
case head chaining (8527) changes. Pass rates fall with chain size: 4 of 15 six-request chains passed since 11-14, and
59% of all verifies. Landings per hour show no rise with depth in either window.

**Is "each minute saved on a merge verify becomes landings one for one" still valid?** It is still an upper bound,
and the reasons it was loose have changed.
- *Closer to binding than before.* Since 11-14 the lane has had work waiting nearly all the time (pipeline empty 4%
  of minutes, against 11% before the pause). The laptop is busy 66% of the time, against 36%.
- *Worth about twice as much per minute.* The bound is landings carried by a host times (old time / new time − 1).
  The landings each verify carries have doubled (0.47 to 1.06 per verify), so the same minutes give about twice the
  landings. That is the main change in this report's table.
- *Still not one for one.* Results land in order, and only two verifies are in flight at a time. The laptop, the faster
  host, can hold a passed result while it waits for the local head (the merge-lane plan measured 33% of heartbeat hours
  before 11-04). So laptop minutes convert less than fully until 8526 runs.
- *What replaces it.* In Dark Factory a task keeps its task slot until its merge lands. So landings are bounded by
  slots ÷ (time before queueing + time in the merge pipeline): 24 ÷ (2.9 h + 4.2 h, both medians since 11-14), about
  81 a day, against 66.6 measured. Under that bound a merge minute saved is worth landings in proportion to the merge
  share of a task's slot time, 4.2 ÷ 7.1 ≈ 0.6, if the pipeline time scales with verify time. So the realised gain is
  between about 0.6 of the one-for-one bound and the bound itself while work is waiting, and zero while the pipeline is
  empty. (Projection, from measured medians.)
- *Once 8526 to 8528 run,* the plan's model expects 1.7 to 3 times today's lane capacity. If the lane then outruns
  supply, verify minutes buy time to land and free slots, not landings directly, and the slot rule above takes over.

## 3. What limits the supply of merge requests (question 4)

`supply_out.txt`, `lane_windows_out.txt`. All measured.

| | Dark Factory, before pause | DF, 11-10 to 11-14 | DF, since 11-14 | reify, before pause | reify, since resume | solar-challenge-platform, before | solar-challenge, before |
|---|---|---|---|---|---|---|---|
| Task slots (`max_concurrent_tasks`) | 24, then 18 from a restart after 11-02 | 18 | 24 | 16 | 16 | 14 | default 24 |
| Task-verify slots (`verify_admission_task_slots`) | 1 | 2 to 3 | 2 | 1 (default) | 1 | 1 | 1 |
| Tasks in flight, mean | 20.1 | 18.0 | 23.2 | (over-counted) | 16.0 | (over-counted) | 3.9 |
| Minutes with slots full (within 1) | 89% | 100% | 91% | 93% | 100% | 95% | 0% |
| Slots held by tasks waiting on the merge lane | 40% of in flight | 50% | 63% of slots | 5% | 13% | 2% | 4% |
| Task-verify test runs per merge verify | 2.43 | 1.47 | 2.46 | 1.64 | 1.11 | 1.57 | 1.72 |
| Task-verify run: wait / run, median per run | 110 / 3.0 min | 1.6 / 3.4 | 4.7 / 3.4 | 529 / 35 | 372 / 33 | 250 / 25 | 10 / 17 |
| Per task before queueing: total task-verify wait, median | 292 min | 2 | 5 | 722 | 286 | 339 | 19 |
| Task-verify wait + run as a share of start-to-queue time, median | 71% | 5% | 10% | 79% | 77% | 78% | 28% |
| Start to first merge request, median | 7.0 h | 2.7 h | 2.9 h | 15.7 h | 7.1 h | 9.3 h | 3.3 h |
| Requeued for a file-lock conflict at planning, per day | 12.2 | 17.1 | 20.3 | 8.2 | 13.5 | 6.6 | 7.5 |
| Dispatch holds on host pressure, per day | 59.7 | 16.1 | 71.9 | 111 | 55 | 309 | 912 |
| Minutes with every account capped | 9% | 14% | 0% | 9% | 0% | 9% | 9% |

Rules. In flight: per-minute count of open task_started..task_completed stints, keyed by (run, task). A stint never
closed ends at its run's last event. Before the pause this over-counts (Dark Factory, reify and solar-challenge-platform
exceed their configured slots), so those columns are approximate. Slots held by merge waits: the per-minute mean of min(pipeline depth, in
flight), divided by slots or by in flight as stated. Task-verify runs: `verify_admission_plan` legs with role `task`
(one leg per module test run; `workflow_verify` rows record passes only). Wait = `waited_secs`, run = `held_secs`. Per
task: the legs recorded between the start of the attempt that queued it and its first `merge_queued`. Lock requeues:
`task_completed` with outcome `requeued` and reason `plan_blast_radius_lock_conflict` or `scope_grant_lock_conflict`.
Holds: `dispatch_deferred` rows. Capped: every pool account capped at once, from the union of the four projects'
account records.

**Reading.**
- **Dark Factory, now: slots, then locks.** The slots are full nearly all the time, and most of them are held by the
  merge pool. A task waits for its merge result inside its slot (`workflow.py::_submit_to_merge_queue`), and the slot
  is released after the workflow returns (`harness.py::Harness._run_slot`). Task verifies are no longer a limit: the
  raise from one task-verify slot to two on 11-09 cut the median wait per task from 292 minutes to 2 to 5. The
  allowance was not limiting in this window (no minute with every account capped). Lock conflicts send about 20
  planned tasks a day back to the queue. Those locks are also held through the merge wait.
- **Dark Factory, before the pause:** its single task-verify slot was the limit (71% of a task's pre-queue time). That
  limit is gone.
- **Reify and solar-challenge-platform: their single task-verify slot.** Tasks waited 5.6 to 12 hours each for it,
  and it was 77 to 79% of their pre-queue time. Their merge lanes were nearly idle. Their supply is CPU-sensitive
  through the task verify.
- **Solar-challenge** was short of work: 3.9 tasks in flight and slots never full.

**Would shifting CPU from merge verifies to task verifies keep Dark Factory's pool deeper?** No, not under the code that
is running. The pool is already deep. Task-verify time is a tenth of the pre-queue cycle. A slower merge lane makes each
task hold its slot longer, which cuts the number of agents running. Dark Factory's own task verifies are also one of
the larger costs on its own merge verify. Each concurrent one of its own task-verify runs added +5.7 ± 2.4 minutes on
the workstation before the pause (`pause/weighttest_out.txt`) and +5.7 ± 1.5 on the laptop since 11-10
(`laptop_fit_recent_out.txt`).

## 4. The floors (question 2)

**Time limits** (from code and config, not observed: no merge verify in these windows hit one). A merge verify always
uses the cold-cache limit (`verify.py::run_verification`), and `merge_verify_cold_command_timeout_secs` takes priority
(`verify.py::_resolve_verify_timeout`).
- Workstation: 180 minutes per command for reify (its yaml sets 10800 s); 120 for the other three (the 7200 s default).
- Laptop: **120 for all four.** None of the four laptop files sets a time limit. The 7200 s default outranks the 10800 s
  that reify's workstation sends with each verify, because that value arrives in the per-module slot
  (`verify_runner.py::_module_config_from_command`). Reify has 60 minutes less on the laptop than on the workstation.
- A timed-out command is retried up to twice (`verify_timeout_retries: 2`).
- The laptop files set no CPU weights, so laptop scopes run at the defaults 100 / 33 / 10. A
  `verify_cgroup_cpu_weight_merge` line in one project's laptop file would set that project's laptop merge weight
  alone, from its next verify; each laptop verify is a fresh process that reads the file. Solar-challenge's laptop
  scopes are also capped at 8 threads (`CPUQuota=800%`).

**Scaling rule (worst case).** Under a factor f against Dark Factory on a full host, a lower project's run takes f
times as long. That assumes every minute of the run is contended and run time is inversely proportional to CPU share.
The whole verify stands in for its longest command, which over-states the command, so the limits err on the safe side.
Largest safe factor = limit ÷ 95th percentile.

| Project, host | Window | n | Median / 90th / 95th / max, minutes | Limit | Past the limit at 1.5 / 2 / 3 / 4 | Largest factor, 5% rule |
|---|---|---|---|---|---|---|
| reify, workstation | before pause | 73 | 76 / 124 / 142 / 152 | 180 | 11% / 23% / 85% / 96% | **1.27** |
| reify, workstation | since 11-14 | 6 | 80 / 118 / 118 / 118 | 180 | 0% / 17% / 67% / 100% | 1.52 (thin) |
| reify, laptop | before pause | 202 | 27.5 / 37.6 / 39.1 / 51.2 | 120 | 0% / 0% / 4% / 43% | **3.07** |
| reify, laptop | since 11-14 | 21 | 38.4 / 47.2 / 48.7 / 49.8 | 120 | 0% / 0% / 43% / 81% | **2.47** |
| solar-challenge-platform, workstation | before pause | 64 | 27.9 / 39.1 / 40.6 / 43.5 | 120 | 0% / 0% / 8% / 47% | **2.96** |
| solar-challenge-platform, laptop | before pause | 287 | 10.8 / 15.5 / 16.9 / 20.0 | 120 | 0% / 0% / 0% / 0% | 7.1 |
| solar-challenge, workstation | before pause | 38 | 17.0 / 27.2 / 29.7 / 32.5 | 120 | 0% / 0% / 0% / 3% | 4.0 |
| solar-challenge, laptop | before pause | 253 | 5.6 / 9.3 / 9.8 / 12.9 | 120 | 0% / 0% / 0% / 0% | 12.3 |
| dark-factory, workstation (for scale) | before pause | 273 | 42.6 / 56.2 / 62.9 / 90.5 | 120 | 0% / 7% / 61% / 95% | 1.91 |
| dark-factory, laptop (for scale) | before pause | 308 | 16.6 / 25.8 / 26.8 / 32.9 | 120 | 0% / 0% / 0% / 1% | 4.5 |

Rule: passed `merge_verify` rows longer than 1 minute, by `runner`, bucketed by end time (`floors_out.txt`). Measured
durations; the shares and factors are projections.

**A less pessimistic check (mirror model, before-pause grid).** This fits each lower project's merge verify duration
against Dark Factory work running at the same time. On the laptop, each concurrent Dark Factory merge verify added
+11.9 ± 1.0 minutes to reify's (mean 29.2, n 202), +3.2 ± 0.3 to solar-challenge-platform's and +2.2 ± 0.2 to
solar-challenge's. Under pure CPU-time sharing, a factor f multiplies that cost by f. At 2.5 to 1, reify's laptop merge
verify gains about 7.5 minutes on average and at most about 18 when fully overlapped: at most about 70 minutes against
120. The workstation fit for reify cannot separate its terms (standard errors larger than the coefficients).
(`floors_out.txt`; ratios with standard errors; the extra minutes are projections.)

**Factors used below:** workstation merge verifies at reify 1.25, solar-challenge 3, solar-challenge-platform 2.5;
laptop merge verifies at reify 2.5, solar-challenge 3, solar-challenge-platform 3. Task verifies, where weighted, at
1.5 (the 11-12 critic's limit: 0 to 2% of runs past their limit). Background runs are left alone. Services are at 3
(agents have no time limit).

## 5. The updated projection (questions 1 and 5)

Model and assumptions (`projection_merge.py`):
- **Workstation:** the pause seat's fit on the 273 workstation merge verifies before the pause (R² 0.32), with the
  other-project merge term split by project: reify +8.0 ± 1.4 minutes per concurrent merge verify, solar-challenge
  +11.0 ± 4.0, solar-challenge-platform +2.7 ± 2.2. Task, background and agent terms stay pooled (+1.1, +0.5, +0.7).
- **Laptop:** 309 laptop merge verifies before the pause (R² 0.57). Reify +7.6 ± 0.4 per concurrent laptop merge
  verify; solar-challenge −0.5 ± 0.9 and solar-challenge-platform +0.1 ± 0.6. On the laptop only reify costs Dark
  Factory anything. The pooled +4.8 in the earlier report averaged reify with two projects that cost nothing.
- **Weights:** a factor f multiplies that project's term by 1/f (pure CPU-time sharing) or by ½ + ½·1/f (half the cost
  not schedulable). Each range below runs from the second assumption to the first.
- **Landings:** a bound, landings carried by each host × (old time ÷ new time − 1). Host shares of landings are matched
  through the verified commit (`landing_host_share_out.txt`): 41 / 58% local / laptop before the pause, 42 / 58% from
  11-10 to 11-14, 30 / 70% since 11-14.
- **Main column:** 11-10 to 11-14 (47.3 a day, 3 days), the longest window under the chained lane.

| Design (factor on the three other projects) | DF workstation merge verify, from 43.9 min | DF laptop merge verify, from 19.1 min | Landings/day gained, lane as before the pause (23.1/d) | **Landings/day gained, chained lane 11-10 to 11-14 (47.3/d)** | Same, since 11-14 (66.6/d, 1.4 days) |
|---|---|---|---|---|---|
| A. 11-12 recommendation: task verifies 1.5, services 3; merge verifies and background left alone | 39.8 to 41.9 | 19.1 | 0.5 to 1.0 | **1.0 to 2.1** | 1.0 to 2.1 |
| B. Merge verifies at each floor, workstation only | 43.1 to 43.5 | 19.1 | 0.1 to 0.2 | **0.2 to 0.4** | 0.2 to 0.4 |
| C. Merge verifies at each floor, laptop only | 43.9 | 17.6 to 18.3 | 0.5 to 1.1 | **1.1 to 2.3** | 1.9 to 4.0 |
| D. Merge verifies at each floor, both hosts | 43.1 to 43.5 | 17.6 to 18.3 | 0.6 to 1.3 | **1.3 to 2.7** | 2.1 to 4.4 |
| E. D + services 3 (task verifies and background left alone) | 39.8 to 41.8 | 17.6 to 18.3 | 1.0 to 2.1 | **2.1 to 4.4** | 2.9 to 6.1 |
| F. D + task verifies 1.5 + services 3 | 39.0 to 41.4 | 17.6 to 18.3 | 1.1 to 2.4 | **2.3 to 4.9** | 3.1 to 6.5 |
| G. 3 to 1 on everything, merge included (ignores reify's floor on both hosts) | 37.1 to 40.5 | 17.4 to 18.2 | 1.4 to 3.0 | **2.9 to 6.3** | 3.8 to 8.2 |
| The pause, same model (all other-project terms removed) | 33.6 | 16.6 | 4.9 | **10.2** | 13.2 |

**Laptop contention since reify resumed (sensitivity, `laptop_fit_since_reify_resume_out.txt`).** Since 11-14T11:01Z,
Dark Factory's laptop merge verify has averaged 30.0 minutes (n 21), against 20.8 between the 11-14 restart and the
resume. A reify laptop merge verify ran during 52% of its time, and each concurrent one added +14.6 ± 3.7 minutes, so
reify accounts for about 7.6 minutes. At reify 2.5 to 1 on the laptop, Dark Factory's laptop merge verify falls to 25.5
to 27.7 minutes. That is worth 2.2 to 4.9 landings a day on the 11-10 to 11-14 lane and 3.8 to 8.3 on the lane since
11-14 (bound). It replaces row C's laptop figure if reify keeps its current laptop load (22.9 laptop merge verifies a
day, busy 59% of the time). Only 13 of the 21 verifies overlapped reify, so this is indicative.

**Cost to the projects weighted down** (projection from the mirror model under pure CPU-time sharing, half under the
other assumption; before-pause exposure):

| Project | Extra minutes per merge verify at the floor factors | What it costs them |
|---|---|---|
| reify | laptop +7.5 on about 29 to 38; workstation about +1 (fit unreliable) | Time to land. Before the pause its queue was empty 48% of minutes, with 0.86 landings per verify; it held 1 to 2 of 16 slots in the merge wait. Its landings are set by its task-verify slot, which this does not touch |
| solar-challenge-platform | laptop +2.9 on 11; workstation +5.4 on 25 | Time to land; queue empty 72% of minutes |
| solar-challenge | laptop +1.6 on 6; workstation +5.5 on 19 | Time to land; queue empty 86%; short of work |
| All three, if their task verifies are also weighted (rows A, F) | Their task verifies are their bottleneck (77 to 79% of pre-queue time for reify and solar-challenge-platform) | Landings. The 11-12 projection for 3 to 1 on everything was up to 3.6 a day lost between reify and solar-challenge-platform; 1.5 to 1 on task verifies was not recomputed here |

**Does including merge verifies change the picture significantly?**
- *For Dark Factory:* on the workstation, no. Reify's floor allows about 1.25 to 1, worth well under a minute. On the
  laptop, yes. The laptop was the host where the 11-12 design did nothing; with merge verifies included it becomes
  the larger half of the prize, and reify's laptop merge verifies are the one thing to weight there.
- *Overall:* row E against row A doubles the bound, from 1 to 2 to 2 to 4 landings a day, and it does so without
  touching anyone's task verifies. With reify's laptop contention at its post-resume level the bound is 3 to 7 (lane
  of 11-10 to 11-14). Against Dark Factory's 47 to 67 landings a day this is a real gain but not a large one, and it is
  a bound (§2: realised between about 0.6 of it and all of it while the pool has work waiting).
- *For the lower projects:* less than before. Their merge verifies were never their bottleneck, so weighting those and
  not their task verifies costs them minutes to land and not landings, down to the floors above.

**What the revocation implies for the per-kind ratio inside a project** (separate from the weight between projects).
- **Dark Factory: keep merge above task for now.** Its task verifies wait about 5 minutes and run about 3. Its merge
  lane is the bottleneck, and the pool holds 63% of the slots. Its own task-verify runs already cost its merge verify
  about +5.7 minutes each. Raising task over merge would lengthen the lane and buy no supply. Reconsider once 8526,
  8527 and 8528 run: if the pipeline is then often empty for long spells, Dark Factory's merge weight can drop at no
  cost in landings. Even then, its supply is set by slots, locks and allowance, not task-verify CPU.
- **Reify and solar-challenge-platform: the revocation's direction is right.** Their task verify is the bottleneck and
  their merge lanes are idle. Inside each project, task verify should be at least level with merge verify. The cheap
  way that does not cost Dark Factory is to lower their merge weight and leave their task weight at today's 33. Reify
  on the laptop would then be about 40 / 33 / 10, task roughly level with merge. Raising their task weight above 33
  instead would speed them, but each concurrent run would cost Dark Factory's merge verify more: 1.4 minutes at weight
  33, up to the merge term's 7 to 8 at weight 100 (an upper bound, because a merge verify is also a bigger job).
- **Solar-challenge** is short of work; its ratio does not matter.
- **Bigger levers, outside CPU weight.**
  - In reify and solar-challenge-platform, the single task-verify slot (`verify_admission_task_slots`, default 1).
    Dark Factory's waits fell from hours to minutes when it went from 1 to 2.
  - In Dark Factory, the slots held through the merge wait. This report does not propose releasing locks earlier;
    Leo's 11-12 ruling stands.

## 6. What could not be determined

- **How completely the lane turns verify minutes into landings** under in-order landing with two verifies in flight.
  The bound stands, and the slot rule (§2) gives about 0.6 of it. Neither was tested.
- **Short and confounded windows.** The chained-lane windows are 3.05 and 1.4 days. The 11-14 restart changed three
  merge-lane behaviours and the slot count at once, and reify resumed 10 hours later. The since-11-14 column has a 59%
  pass rate, against 80% before.
- **How the lane behaves once 8526, 8527 and 8528 run.** None of them runs anywhere yet.
- **Per-command durations.** Merge verify records carry only the whole verify. The floors use it as a stand-in for the
  longest command, which over-states the command.
- **The laptop time limit is from reading the code,** not observed. No laptop merge verify timed out in these windows.
- **Reify's workstation mirror coefficients** are not identifiable (n 73, standard errors larger than the effects).
  The post-resume laptop fit has n 21.
- **The non-schedulable share** of contention (the two assumptions behind each range) is still unmeasured.
- **Merge requests submitted from interactive sessions** are in the pipeline count but hold no task slot. Their number
  was not separated, so "slots held by merge waits" is an upper estimate.
- **In-flight counts before the pause** exceed the configured slots, from stint pairing across restarts or the
  18-slot ruling taking effect at an unrecorded restart. Those cells are approximate.

## 7. Files

`/tmp/team-weights-1112/followup-1115/merge/`:
- `regime_daily.py`: daily merge verifies, chains, landings and requests per project.
- `lane_windows.py`: the §2 window table and the task-verify rows.
- `supply.py`: §3.
- `chain_vs_depth.py`: chain size against the waiting count.
- `depth_daily.py`: daily pipeline depth.
- `floors.py`: §4.
- `laptop_fit_recent.py`: the post-11-10 and post-resume laptop fits.
- `projection_merge.py`: §5.
- `*_out.txt`: printed results, including `df_dur_by_window_out.txt`, `df_hourly_1114_out.txt` and
  `landing_host_share_out.txt`.

Re-checked before writing: the §2 Dark Factory window rows, the §3 slot shares and the §5 table were re-read from their
output files as printed.
