# Evidence for the gate 10572 sitting (2026-11-19)

Lead's digest of five read-only evidence seats. Window unless stated: 2026-11-12T00:00Z to
2026-11-19T19:59Z (7.83 days; 11-19 is partial). Source: `data/orchestrator/runs.db` `events`
(ISO timestamps, UTC), `data/load-samples.db` (`ts` epoch int), main's reflog and first-parent history,
`dark-factory-orchestrator.yaml`. Seat scratch scripts: `/tmp/s2-10572/`, `/tmp/s3-10572/`, `/tmp/s4-10572/`
(volatile).

## A. Merge lane (seat S2)

| Figure | Value | Rule |
|---|---|---|
| Queue depth | 13 at 11-12 00:00Z, 77 at the end | `merge_heartbeat.data.depth`, last heartbeat at or before the instant |
| Landings | 364; per day 52/52/58/44/42/23/48/45 | `merge_finalized` state=done by UTC day; chain members write their own done row; `train_merged` has no rows since 11-01 |
| New arrivals | 426; per day 48/50/52/70/43/57/58/48 | a task's first-ever `merge_queued` |
| Arrivals incl. resubmission after a terminal exit | 536; 72/63/65/89/53/74/65/55 | first `merge_queued` per task, or first after a done/blocked/conflict/already_merged finalize; restart requeues excluded |
| Merge verifies | 453 completed (284 pass, 169 fail), 193 cancelled | `merge_verify` rows; `merge_verify_cancelled` rows; bucketed by start |
| Host busy with merge verifies | workstation 87.8%, laptop 82.9% | union of completed and cancelled verify intervals over 188.0 h per host |
| Mean duration, both hosts | 25.4 min on 11-12/13; 48.7 min on 11-17..19 | completed verifies, by start day |
| Mean duration by host, by end day | workstation 33.1/31.5 (11-12/13) to 43.2/47.3/93.8 (11-17/18/19); laptop 21.6/20.4 to 44.0/39.8/54.0 | completed verifies |
| Landed per verify-hour by chain size | 1: 0.85 (n=285, pass 73.7%); 2: 1.83 (15); 3: 1.16 (56); 4: 1.58 (12); 5+: 2.58 (85, pass 41.2%) | each done landing credited to the latest passed verify ending within 5 s before it whose chain contains its merge commit; 344 of 364 landings attributed; cancelled verifies excluded (no chain size) |
| Where 321.7 verify host-hours went | landed 34.4%; passed, not landed 18.3%; failed 30.3%; cancelled 17.1% (cascade 12.5, re-anchor 2.9, tip adopted 1.7) | verifies started in window |
| Failed + cascade-cancelled | 137.7 h = 43% | sum of the two rows above |
| Failed singles that later landed | 51 of 75 (6 already merged, 17 blocked) | by task, later done finalize |
| Re-verifies after main moved | 66: direct commits only 26, queue landings only 25, both 14, empty range 1 | `merge_reverify_gate` reverify=true; cause from first-parent commits in `rebased_from..rebased_onto`; a commit is a queue landing if it is a done `merge_sha` |
| Passed verifies lost to a main move | direct-only 20 (12.0 h), queue-only 18 (14.2 h), both 13 (8.0 h) | passed verify whose merge sha is a later gate row's `verified_merge_sha` |
| Restart cost | about 3.7 host-hours, 1.1% | drain idle before stop + gap to next verify, six restarts |
| Counterfactual landings | 41.0/day actual on 11-17..19T20 → about 78/day | each verify's duration replaced by the 11-12/13 mean for its host and depth class; hours 122.6 → 64.6 (×1.90); pass rate, depth, cancels unchanged; hosts assumed to stay saturated |
| Task verifies per day | 109/130/123/159/88/152/105/94; ratio to merge verifies started 1.03 to 2.35 | `workflow_verify` rows with a duration |
| Task verify mean duration | 4.0 min (11-13) to 14.9 min (11-19) | same rows |

Readout predicates of task 8532, current values: gates per landing 0.18 (limit 0.3, holds); voids per
landing 0.58 (limit 0.1, fails; 195 of 211 voids were discarded before a host was acquired); P(gate after
gate) 0.31 (limit 0.3, fails); laptop holding a finished verdict 9.2% (limit 10%, holds); median
queue-to-land 4.87 h (limit 1 h, fails); workstation merge-verify hours per week 148 (limit 60, fails);
deep-chain 90th-percentile duration 3,542 s (limit 10,800 s, holds).

## B. Cause of the slowdown and the task cap (seat S4)

Population: workstation merge verifies running the full plan, 11-08..11-19. Observational throughout.

| Figure | Value | Rule |
|---|---|---|
| Cap history | 18 until 11-14 00:56Z; 24 until 11-15 11:00Z; 36 since | run starts and `config_reload` rows; restart-tier key |
| Duration by CPU pressure, same plan | 23.1 min at pressure 0–15 (n=23) to 59.5 min at 45–60 (n=26) | passed full-plan verifies, 30-unit plan, mean `psi_cpu_some_avg10` over the verify's own window |
| Regression of duration | +0.42 min per CPU-pressure point, +1.33 per IO-pressure point, +3.7 for the 33-unit plan; R² 0.73, n=281 | OLS incl. red runs |
| Split of +16.6 min, 11-13 → 11-18 | CPU pressure +9.1, IO pressure +7.8, plan +3.7, unexplained −4.2 | regression terms × change in day means |
| Split of +62.3 min, 11-13 → 11-19 | CPU +11.7, IO +24.0, plan +3.7, unexplained +22.1 | same; n=10 on 11-19 |
| Who raised CPU pressure 11-13 → 11-18 (+22.5 points) | reify +22.4 (task-verify legs +17.1, merge verifies +4.2, agents +1.1); DF sweep +1.8; DF task work −0.3 | 10-min bins, n=1704, pressure regressed on each consumer's concurrent count with day fixed effects, R² 0.71 |
| Cap 18 with reify present (11-08) | 61.4 min (n=19) vs 31–36 min on 11-10..13 | day means |
| Standing cost of DF task work at cap 36 | +3.5 to +5.5 min of about 47 | OLS on n=61 and n=74 verifies since 11-15 11:00Z |
| Tasks being written or reviewed | mean 2.0–2.6 on 11-16..19 (4.95 on 11-13); 27.5–31.1 in the merge phase | `phase_enter`/`phase_exit` pairs, approximate |
| Test functions on main | 77,909 (11-13) → 80,922 (11-19), +3.9% | static count of `def test_` at each day's last first-parent tip |

Not measured: per-consumer CPU use (tasks 10456 and 6355 pending); per-unit timing inside a merge verify;
laptop pressure series; the slow-tests speed-up landed 11-19 18:39Z (no workstation verify completed since).

## C. How capacity is shared today (seat S1)

- Merge verifies: one slot per host (`verify_runner.py::HostAllocator`), laptop tried first
  (`verify_host_policy: prefer_remote`), never gated by task admission
  (`shared/verify_admission.py::acquire_task_slot` returns at once for the merge role), 16 test workers,
  CPU weight 100.
- Task verifies: 3 slots on the workstation (`verify_admission_task_slots: 3`) and a race for laptop turns
  (`verify_task_host_policy: remote_all`), 8 workers, CPU weight 33 per leg; background sweep weight 10.
- Agents and daemons run in `app.slice`, verify scopes in `df.slice`, both weight 100; the four project
  slices under `df.slice` are all 100. Task 9985 (pending) moves daemons into their project slice.
- Reify: own admission, at most one task verify and one merge verify per host.
- `max_concurrent_tasks` is restart-only; the weights, slots, worker counts and chain switch hot-reload.
- No code reads merge-queue depth at dispatch. A task holds its slot and locks until its merge request ends.
- The main-health check's fallback arm runs as a task verify and queues on the task slots (task 5139).

## D. Direct commits (seat S3)

Window 2026-11-05 → 11-19: 1753 first-parent commits = 563 queue landings + 1089 inner commits of
fast-forward landings + 101 direct (rule: main's reflog `commit:` entries plus one `update-ref` with no
`merge_finalized` row; residual 0). Direct by class: PRD flow 44; nightly legibility 13; config 12;
interactive records 11; memory census 9 (queue-routed since 11-15); review-all docs 5; skill edits 4; code
2; legibility census 1.

- Queue wait for a landed request since 11-17: normal lane median 6.8 h, p90 20.7 h (n=107); high lane
  median 2.3 h, p90 3.7 h (n=10, since 11-05). Rule: `merge_finalized` done `queue_to_land_secs`, lane from
  `merge_queued`.
- `merge_lane/gates.py::_disjoint_skip_blockers`: under `merge_verify_breadth: "full"` every rebase onto a
  moved main re-verifies in full; only a byte-identical tree skips.
- `fused_memory/server/manifest_stamping.py::_stamp_sidecar_on_disk` stamps the main checkout and is a
  silent no-op when the sidecar is absent.
- Task 6661 (pending; depends on 5411, 5064 done, 9145): classify the main delta; docs-class → docs
  command set on the rebased tree; log mode 7 days by the 2026-10-09 ruling.
- Memory census: 1 landed (11-15), 4 abandoned at 10,800 s (11-16..19); installed unit lacks
  `CENSUS_LANDING_TIMEOUT_SECS=72000` (commit `9fb020aa1e`).
- Lane idle (no verify in flight): 2.5% of heartbeats.

## E. Where the revoked rule is still written (seat S5)

Live policy: `plans/verify-oversubscription-control-prd.md` §1 item 2 and §3;
`plans/verify-oversubscription-control-prd.capability-manifest.md`; `plans/remote-task-leg-verify-prd.md`
item 16 and its seam row; `plans/verify-admission-fifo-prd.md`; `OPERATIONS.md` "Reading a task-verify
admission wait"; comments in `shared/src/shared/verify_admission.py`, `orchestrator/src/orchestrator/verify.py`,
`config.py`, `dark-factory-orchestrator.yaml`, `shared/orchestrator.yaml`; three tests; tasks 5139, 3589,
3595; five passages of task 5139's plan. Historical (leave): 7 groups of dated studies. Task 5139's
functions now live in `orchestrator/src/orchestrator/merge_lane/main_health_probe.py`.

## G. Critic's corrections (seat S6), which supersede the rows above where they differ

- **Queue depth and the cap.** Recovery at restart (`merge_lane/queue_store.py`, `Harness._recover_pending_merges`)
  re-enqueues journalled requests with no run slot and no locks. Max queue depth per run: 28 (cap 24, from
  11-14 00:56Z), 36 (cap 36, 11-15 11:00Z, no restart for 44 h), then 67, 85, 80, 85 after the restarts of
  11-17 07:13Z, 11-17 20:49Z, 11-18 18:34Z, 11-19 04:25Z (lead re-ran: `max(merge_heartbeat.depth)` by
  `run_id`). Live at 20:27Z: 71 tasks in-progress, 70 of them queued; 41 of 77 queue entries have no
  workflow event in the current run. Task 10093 (pending, high) restores locks only.
- **Baseline.** 11-12/13 was the paused fleet. Pre-pause means (lead re-ran, `merge_verify` by end day):
  workstation 40.5/38.9/41.5 min and laptop 19.4/19.0/20.2 min on 11-05/06/07. So the "78 a day"
  counterfactual in section A is an upper bound against a state that has ended.
- **11-08 comparison invalid**: seven projects ran that day.
- **Laptop.** DF task-verify hours on the laptop 2–5 → 9–15 a day from 11-15 (`remote_all`, 3 slots); reify
  task verifies 14–21 h a day there from 11-16. DF laptop merge verify, 33-unit plan, green, reify present:
  38.0 min with little DF task-leg overlap (n=27), 48.8 partial (n=15), 55.8 full (n=6). Observational.
- **Re-verify split.** 26 direct-only / 32 queue-only / 7 both / 1 empty (S2 had counted inner commits of
  fast-forward landings as direct). Repeated-verify hours: direct-only 12.0, both 3.5, queue-only 19.0;
  34–35 h = 10.6–10.9% of 321.7 h. At most 19 of the 26 direct-only are docs-class under 5411's allowlist.
- **Singles.** 102 of 199 verifies since 11-15 11:00Z were singles, started at median queue depth 51; 47
  were tasks from an earlier red chain (35.9 h, 24% of completed verify hours).
- **Waits.** 11-17..19 landed requests: median 6.65 h, p90 20.7 h. The 77 waiting: median age 14.5 h.
- **Dispatch holds.** `scheduler.py::Scheduler._phase_psi_gate` deferred dispatch 635 times: memory 338, IO
  186, run queue 111; 3–8 a day on 11-12/13, 58–217 a day later.
- **Not reproduced by the critic**: 2.58 landed per verify-hour for chains of five or more; S4's regression
  coefficients; the 78 a day.
- **Docs.** Nothing in `merge_lane/worker.py::may_chain` excludes a docs request from a chain. Task 8258
  (pending) is the now-obsolete "adopt after 30-day shadow" gate; it depends on 5411 and 6661. The 9145
  edge on 6661 only sequences two edits of `_decide_reverify_gate`.

## F. Lead's checks and open joins

- Agents' slice confirmed live: `orchestrator-dark-factory.service` is in `app.slice`.
- Task store: 71 tasks in-progress against a cap of 36, and queue depth 77. The mechanism that lets work
  waiting to merge exceed the cap (parking and backfill) was not traced.
- Re-verify counts differ by window: S2 66 from 11-12T00Z; S3 54 from 11-12T19Z; the brief 55.
