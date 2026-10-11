# Seat `critic` — adversarial pass over SYNTHESIS.md (2026-09-18)

Read-only. No tracked file edited, no git stash, no MCP write, no verify/test command run.
sqlite opened `mode=ro`. The study's stdlib sims were re-run, plus four small critic scripts
(appendix; they import `scripts/sim_lib.py` unchanged). Labels: [M] measured by me now,
[S] sim re-run by me, [C] read from code, [D] derived.

Bottom line: three of the synthesis's load-bearing claims do not survive.
(1) The "unresolved utilisation" is an artefact of pricing a 09-12..09-18 arrival census with
ALL-TIME service cells; era-matched, the census itself gives rho = 0.99. (2) The factory is
measurably CLOSED (WIP 23-24 of 24 for the whole window), and in a closed saturated loop
Little's law pins MEAN plan/task residence under every ordering: the 2-class "-31...48 % mean"
and FIFO "-10...20 % mean" are open-queue artefacts; what survives is FIFO's tail cut and a
2-class REDISTRIBUTION (short -90 %, long +74 %). (3) The merge-queue prize (0.9-4.3 h/day)
is attributed to the wrong policy, the "for free" code claim is false, and the A/B cited as
decisive contradicts itself in its middle arm.

---

## Findings

**1. CLAIM** (s.3 bullet 1; s.7) "Utilisation is unresolved ... census rho >= 0.68; wall and wait fit need 0.95-1.0 ... The bridge is 411 unattributed `run_verification` calls."
**VERDICT: refuted (both halves).**
**EVIDENCE.**
(a) `scripts/sim_lib.py::service_cells` (and `80_replay.py`) pool `data/legs_raw.jsonl` over ALL
time. The dominant cell is non-stationary [M, from `legs_raw.jsonl`, `test.started_at`]:
`orchestrator x full_suite` mean 644 s (Jul, n=18) -> 821 s (Aug 1-14, n=24) -> 1 267 s
(Aug 15-31, n=54) -> 2 877 s (Sep 1-7, n=31) -> 2 597 s (Sep 8-11, n=28) -> **3 066 s in the
census window** (n=38, p50 3 032). The pooled 1 959 s the sims use understates current
service 1.57x. `fused-memory x full_suite` 185 s (old) -> 375 s (window). Re-pricing the SAME
578-arrival census with in-window cells (cells with n>=5 in-window, else all-time) gives
**23.86 h/day offered, rho = 0.994** [D], against the reported 16.36 h/day / 0.68. No dark
load is needed.
(b) The 411 "unattributed" calls are train gate runs, i.e. `role='merge'`, UNGATED.
`scripts/sim_attribute_env.py` attributes merge calls only via `merge_trace.csv` episode
windows, and `data/README.md` "Censoring" item 3 says trains have no episode row. `merge_verify`
events whose start lies outside every episode window, by day 09-13..09-18: 6,1,1,6,3,0 [M,
runs.db ro]; model-sim's unattributed-by-day: 132,29,26,139,77,8. Pearson r = **0.998**, slope
21.5 calls per out-of-episode gate run (9-module fan-out + post-merge type-only gather
`merge_queue.py::_run_unscoped_typechecks`-style + reruns). `train_started` by day 7,1,1,6,3,0
(r = 0.986). This is a daily-count match (n=6), not a timestamp join - the env-line grep was
not persisted in `data/`; the lead can confirm by joining unattributed timestamps to
`train_started..train_merged/derailed` windows.
**SHOULD SAY:** rho is resolved at ~0.99 from the census once service is era-matched; the
0.68 was a pooling error. Delete the "bridge". The finding that outranks everything:
`orchestrator x full_suite` task legs are 130/578 = 22 % of arrivals and **86 % of slot work**
(130 x 3 066 s of 23.86 h/day x 5.375 d), and that leg's service time has roughly TRIPLED
since early August. Every [S] magnitude computed with all-time cells + Poisson "dark" jobs is
mis-specified: the missing load is not an unlabelled stream, it is the LONG class itself.

**2. CLAIM** (s.1 item 3, s.4) "2-class ... cuts mean plan completion 31-48 % with a maximum no worse than FIFO's. ... the long class barely notices."
**VERDICT: refuted as a steady-state mean claim; "barely notices" and "max no worse" hold only in the open-queue model.**
**EVIDENCE.**
(a) The system is closed and full [M, runs.db `task_started`/`task_completed` per run_id]:
time-averaged in-flight dispatches 24.4 / 23.2 / 23.6 over the three runs in the window
(peak 24-25, min 1 only at boot); `dispatch_deferred.in_flight` = 24 in 10/21 events;
`scheduler_tick.candidates` ~ 632, so supply never runs dry. `scripts/sim_lib.py::simulate`
is an OPEN replay (arrivals fixed by the trace regardless of policy); the "closed loop" in
`sim_feedback.py::stage_b` is a back-of-envelope fixed point, never simulated.
(b) Little's law: N = X (Z + R). N = 24 is pinned [M]; X = slot capacity / E[S per plan] is
pinned once the slot is saturated (finding 1) and is order-independent (the synthesis's own
item 5); Z (non-slot time) is not an ordering variable. Therefore **mean R is invariant under
every ordering**. Items 3 and 5 of the synthesis cannot both be true in steady state.
(c) Critic closed-loop sim [S] (appendix `closed.py`/`closed_win.py`: 24 tokens, think
Z = 2.74 h/plan derived from the seat's own 11.92 h residence / 7.62 h wall / 1.57 plans, plans
bootstrapped from the census, era-matched cells, **zero dark load, nothing fitted to waits**),
median of 5 x 55 d:

| policy | plans/day | plan completion mean / p50 / p90 / max (h) | SHORT mean / p90 | LONG mean / p90 / max |
|---|---|---|---|---|
| random | 75.1 | 4.82 / 3.53 / 10.51 / 34.0 | 4.44 / 10.10 | 5.15 / 11.03 / 34.0 |
| leg FIFO | 75.3 | 4.85 / 4.76 / 7.33 / 11.6 | 4.54 / 6.96 | 5.14 / 7.61 / 11.6 |
| 2-class 300 s | 76.6 | **4.78** / 4.73 / 10.84 / **16.4** | **0.46** / 0.87 | **8.95 / 11.84 / 16.4** |

Validation of that model against measurement, with no free parameter: plans/day 75.1 vs 75.7
measured; random-order LEG wait mean 3.95 / p50 2.73 / p90 9.08 h, p50/mean **0.69**, vs
measured in-window 4.85 / 3.54 / 10.85, p50/mean **0.73** - the shape model-sim s.2.2 says its
open sim can never reach (max 0.53) and itself calls "the signature of a WIP-pinned closed
system". The closed model validates where the open one failed.
Result: 2-class mean gain = 0 (4.78 vs 4.82, replicate sd 0.09-0.16); SHORT -90 %; **LONG
mean +74 %, LONG p90 +56 % vs FIFO, max 16.4 h vs FIFO 11.6 h (+41 %)**. With the study's own
all-time cells + 7 h/day dark load the closed run gives the same picture (random 4.59, FIFO
4.60, 2-class 4.71; LONG 4.89 -> 8.48; dark jobs 4.17 -> 8.74 h).
(d) The long-class number was a named BLOCKER in `design-options.md` s.5.5 item 1 / U3 ("must
be answered with a number ... report the long class's own p50/p90") and was never produced:
`sim_lib.py::metrics` has per-tier but no per-class output. Also, `metrics` EXCLUDES dark jobs,
which `two_class` files as LONG (their `plan_exp` = global mean 523 s > 300 s): at the rho 0.98
arm ~30 % of the work pays for the short class's gain off-ledger.
(e) In the OPEN model the claim does reproduce, robustly [S, paired CRN, 15 seeds]: 2-class
-30 % (sd 5 pp) / -42 % (4 pp) / -45 % (3 pp) at rho 0.68/0.91/0.98, and open era-matched LONG
mean 3.82 (FIFO) -> 3.95. So this is a model-class error, not replicate noise.
(f) In-sample leakage (brief item 2): minor. Cells are large (193/158/142) and separated 45x;
the one fragility is `fused-memory x full_suite` (mean 337 all-time, 375 in-window) sitting
on the 300 s threshold - a whole 89-arrival cell flips class with the threshold. Survives.
**SHOULD SAY:** "In the measured regime (WIP pinned, slot saturated) no ordering changes mean
verify residence; 2-class is a redistribution: plans under ~300 s drop from ~4.5 h to ~0.5 h,
orchestrator full-suite plans rise from ~5.1 h to ~9 h (p90 ~12 h, max above FIFO's). Whether
to buy that is a policy choice about WHO waits - it is not a mean-duration lever. The -31...48 %
applies only if WIP stops binding (open regime)." Caveat that cuts both ways: LONG waiters hold
orchestrator module locks (`scheduler.py::ModuleLockTable`), so the scheduler will substitute
non-orchestrator tasks; task-count throughput and mean can then "improve" by composition shift
while orchestrator work (86 % of slot work, capacity ~28 full-suite legs/day) gets slower.

**3. CLAIM** (s.1 item 2; s.2 row 2) "FIFO cuts mean plan completion 10-20 % ... p90 -39 %, max -74 %."
**VERDICT: mean part refuted in the closed regime; tail part survives (smaller). Noise attack fails.**
**EVIDENCE.** Open sim, paired CRN, 15 seeds [S]: FIFO vs random -11 % (sd 4 pp) / -14 % (6) /
-15 % (5), FIFO worse in 0/15 at every arm - the 10-20 % is NOT replicate noise within the open
model (unpaired sd 0.15-0.98 h is the wrong yardstick; `simulate` draws all services before
the loop, so policies share random numbers). Closed validated model (finding 2 table): mean
4.85 vs 4.82 = 0; p90 10.51 -> 7.33 (-30 %); max 34.0 -> 11.6 (-66 %); p50 3.53 -> 4.76 (+35 %).
Gather semantics [C]: `verify.py::run_scoped_verification` gathers `_verify_module` and
aggregates RESULTS; a red leg returns a result, not an exception, so it never cancels
siblings - "gather's cancel-siblings-on-failure not modelled" (s.7) is a non-issue, and plan
completion really is a MAX over legs.
**SHOULD SAY:** FIFO: mean +-0, median worse by ~1/3, p90 -30 %, max -66 %. A pure tail lever.

**4. CLAIM** (s.2 "Frequent restarts vs FIFO: adverse ... at MTBR 8.3 h FIFO is worse than random" => "stamp must be the task's first verify-request time persisted across restart"; s.4; s.5 step 4; s.6 item 3)
**VERDICT: refuted (sim result not reproducible; stated mechanism absent from the sim; restart model contradicts measurement).**
**EVIDENCE.** (a) Paired re-run of `sim_lib.simulate(..., mtbr_s=8.3*3600)`, rho 0.91, 15 seeds
[S]: FIFO vs random median **-17 %**, FIFO worse in 2/15, sd 30 pp. The seat's 2.85 vs 2.45 is
an unpaired median-of-9 under a 30 pp spread. (b) Mechanism [C]: on restart the sim sets every
requeued job's `a`/`plan_a` to `rt + 600`, but `make_key('leg_fifo')` = `(a, pid, seq)` and
`pid` is preserved, so ties break in ORIGINAL arrival order - the "age information is
destroyed" story is not what the code does. (c) Reality [M]: a restart cancels the whole WIP,
not one leg - all 48 DF `task_completed outcome=cancelled` in the window fall in the three
restart minutes (24 @09-12T09:2x, 2 @09-14T12:3x, 22 @09-15T22:5x-23:0x); reify 132/132 in its
restart buckets (38/47/23/24). After that the 24 tasks are re-dispatched in scheduler-score
order and re-run rebase -> verify from attempt 0; none of that is in the sim.
(d) A "task first-verify-request" stamp is a DIFFERENT policy from the simulated plan-arrival
FIFO: a task on its 5th debug-fix/re-dispatch loop holds the oldest stamp forever and jumps the
queue on every re-entry (task 3728: nine dispatches) - oldest-TASK-first gives repeat failers
standing priority on a slot whose work is 86 % orchestrator full-suite legs. Unsimulated.
(e) `design-options.md` s.5.3 item 5 / U16 says "Do not persist the heap ... Blocker"; the
synthesis asks for a persisted stamp without reconciling. If one is wanted, the precedent is the
task-metadata retry ledger (`workflow.py::_stamp_first_merge_enqueue` /
`merge_types.py::MergeRequest.merge_first_enqueued_at`), not gate state.
**SHOULD SAY:** the restart arm shows nothing about FIFO; drop the "must be restart-stable"
requirement as a finding, keep it at most as an open design question, and price step 4 without it.
Also fix s.3 "~50 % of dispatches end cancelled - cause unknown": the cause is the restart
sweep (100 % of cancellations sit on restart boundaries, DF and reify).

**5. CLAIM** (s.1 item 5; s.2) "No ordering policy raises throughput at either queue."
**VERDICT: weakened (first-order true at the slot BECAUSE rho ~ 1, which the synthesis calls unresolved; second-order sign unknown; internally inconsistent with item 3).**
**EVIDENCE.** At the census rho the seat itself shows ordering WOULD raise dispatch 1.4-2.2x until
the slot saturates (model-sim s.5.2) - so invariance is conditional on finding 1. Given
saturation, Little (finding 2b) makes throughput AND mean residence invariant together; the
synthesis keeps one and drops the other. Elastic-demand attack: cancellations are restart
sweeps (finding 4c), not wait-driven, so that channel is closed. Two channels remain open and
unmodelled: (i) 2-class lengthens exactly the orchestrator tasks (+74 %) whose base then ages
past the ~20-commit conflict step (`measure-merge.md` s.3.1) - a plausible throughput LOSS;
(ii) short-first + cancel-still-WAITING-siblings when a short leg goes red would be genuine
demand reduction (today a red cheap leg never stops the 51-min orchestrator leg, finding 3).
**SHOULD SAY:** "To first order, given rho ~ 1, ordering changes neither throughput nor mean
residence. Second-order effects exist in both directions and are unmeasured."

**6. CLAIM** (s.1 item 6; s.4) "effective priority (`scheduler.py::Scheduler._compute_effective_priorities`, which already folds in dependents => most-dependents-first for free) ... targets ... 6.2 dependent-task-h/day; sim bracket 0.9-4.3 h/day saved."
**VERDICT: refuted (code claim false; number belongs to a different policy).**
**EVIDENCE.** [C] `_compute_effective_priorities` is tier inheritance:
`min-rank(own, boost, effective(d) for undone dependents)`. One `high` dependent => `high`; ten
`low` dependents => `low`. It counts nothing. The count is the sibling
`Scheduler._compute_transitive_counts`. Both are underscore-private statics; calling them from
`merge_queue.py` is an interface reach, not SPOT reuse. [S, model-sim s.10.2 table] the
0.9-4.3 h/day is `most_deps` (count-keyed, UNAGED): 0.31x/0.85x. The policy the synthesis
actually recommends - priority + >=4 h aging - is 0.90x/0.94x => **0.4-0.6 h/day**; even
`most_deps + 4 h aging` is 0.76x/0.85x => 0.9-1.5 h/day. The pool itself is soft:
`n_dependents_blocked` and `priority` are CURRENT values (data/README s."Censoring" item 4), so
dependents already unblocked-and-done are dropped and still-pending ones blocked for unrelated
reasons are counted. And under pinned WIP with ~632 candidates, a dependent unblocked an hour
earlier just joins the candidate pool: dependent-task-hours is not dispatch latency.
**SHOULD SAY:** "An aged effective-priority tie-break fixes tier inversions (79->42) and is worth
~0.4-0.6 h/day of a soft 6.2 h/day pool; most-dependents-first needs the transitive count, is
not free, and its larger bracket applies only unaged." That is below the noise of anything else
in s.3 - say so.

**7. CLAIM** (s.2) "Merge order => throughput: absent: gate 74 % busy single server; landings track service time (A/B: gate x1.35 => landings x0.69)."
**VERDICT: weakened (the cited evidence does not show it).**
**EVIDENCE.** `measure-merge.md` s.0 A/B table, middle arm: "pinned 16" has gate p50 51.0 min -
IDENTICAL to the unpinned arm's 51.3 - yet landings fall 14.2 -> 9.0/day (-37 %). Landings
moved with NO change in gate duration. Gates per landing [D from the same table]: 151/106 =
1.4 (arm 1), 46/19 = 2.4 (arm 2), 108/68 = 1.6 (arm 3) - the rework share moved, not the gate.
Confound: `verify_admission_task_slots` 3->1 at 09-08T10:17Z (`measure-sem.md` s.1 timeline), one
hour before the arm boundary at 11:17, and `max_concurrent_tasks` 48->24 on 09-11 - upstream
supply changed at the arm boundaries. A 74 %-busy server is by definition not saturated; its
landings are set upstream (the task slot) and by gates-per-landing (1.4-2.4x), and the latter
is exactly what order/staleness/speculation (14 % discards) can touch. model-sim s.10.2 pt 5:
"Throughput is invariant by construction here (outcomes are replayed), so this model cannot
test it."
**SHOULD SAY:** "No evidence that merge order moves throughput; none that it cannot. Landings
are currently limited upstream and by gates-per-landing, not by gate duration alone. The A/B is
confounded and should not be cited as decisive."

**8. CLAIM** (s.1 item 6) "merge wait adds ~0.5 commits to a base ~25 stale - ~2 % of accrued drift."
**VERDICT: survives.** [M] recomputed from `merge_trace.csv` on aggregates rather than medians:
sum(wait) x 1.58 commits/h / sum(base_age_commits) = **2.0 %** DF (n=294), 0.9 % reify (n=255).
Conflict/needs-rebase episodes have mean base age 323 commits with 1.75 h wait - ancient
branches, not queue victims. Tail note: in 41/294 DF episodes wait is >=20 % of base age.

**9. CLAIM** (s.3) "C-merge-priority has a hole: `verify_failure_is_preexisting_on_main` runs its probe at `role='task'` (lead-verified), so merge-path main-health probes queue on the task slot."
**VERDICT: weakened (line is right; reach is conditional).**
**EVIDENCE.** [C] The `role='task'` call exists and merge-path callers exist
(`merge_queue.py` two sites - the sync classify path and the deferred main-health probe - plus
`workflow.py` for task verifies). But DF sets `merge_verify_breadth: "full"`, under which a
merge-role red carries junit `failing_test_ids`, and the function first takes
`main_baseline_failing_ids(...)` (`role='merge'`) and RETURNS if the baseline is non-None. A
merge-path caller reaches the `role='task'` probe only on the B3 degrade: `failing_test_ids is
None` (lint/type-only red, junit missing/unparseable, non-pytest) or baseline probe failed.
For the `workflow.py` caller `role='task'` is correct. model-sim's 7 wide file-scoped plans in
5.4 d is consistent with a rare degrade path (~3.5 legs/day).
**SHOULD SAY:** "conditional hole on the B3-degrade path; small today; fix is to thread the
caller's role rather than hard-code either value."

**10. CLAIM** (s.3) journal narrowing: "`enqueued_at` IS journaled ... what is lost is the resubmission-lineage advantage for journal-recovered requests."
**VERDICT: survives.** [C] `merge_queue_store.py` rebuilds a recovered `MergeRequest` with
`enqueued_at=persisted.enqueued_at` and no `merge_first_enqueued_at`;
`merge_queue.py` aging key = `merge_first_enqueued_at or enqueued_at`; the first-enqueue stamp
itself is durable in task metadata (`workflow.py::_stamp_first_merge_enqueue`, retry ledger),
so a workflow RE-submission after restart carries it; only the journal-recovered copy falls
back. The lead's narrowing is correct.

**11. CLAIM** (s.5 step 1) "All DF gated acquirers are in-process coroutines of one orchestrator [C], so no cross-process queue is needed."
**VERDICT: survives with two caveats.**
**EVIDENCE.** [C] gated callers found: `workflow.py` (2 sites), `review_checkpoint.py`
(default role), `harness.py` -> `run_main_tip_sweep` (`background`), sweep prefilter
(`background`), `verify.py::_run_isolated_confirm_group_observation` (default `'task'`, logs no
`Verify plan:` line - a gated caller the census cannot see), the B3-degrade probe (finding 9) -
all in the orchestrator process. `cli.py::verify_merge` is merge-role (ungated).
`offline_lane.py` imports only `nice_prefix`. Caveats: (i) `evals/metrics.py::collect` calls
`run_verification(worktree, workflow.config)` at default `role='task'` from the separate eval
CLI process; `evals/runner.py` builds its config with `project_root` from the task fixture, and
the slots dir is a digest of `project_root` (`config.py::_default_verify_admission_slots_dir`) -
an eval run against the DF root is a second-process acquirer that the in-process gate cannot
order (the flock still bounds it; it just wins at random). (ii) restart overlap / drain: two
orchestrators on one root briefly. Neither breaks safety; both break "ordered".
Seven orchestrators run on this host (`systemctl --user`: autopilot-video, dark-factory,
know-live, my-solar-challenge, pump-web-ui, reify, solar-challenge-platform), each with its own
N=1 slot dir - irrelevant to ordering, very relevant to why service time tripled (finding 1).

**12. CLAIM** (s.4 DF) "Order by PLAN, never by leg"; (s.1 item 3) the discriminator "is known before the leg spawns"; step 5 "1-2 ad, static class map in config".
**VERDICT: weakened (over-strong; and the plan key is not at the call site).**
**EVIDENCE.** [C] The acquire site is `verify.py::run_verification::_run_or_skip_timed`, whose
frame holds ONE `module_config`; sibling legs and `verify_plan.py::PlannedRun.scope_kind` live in
`run_scoped_verification`, which reaches `run_verification` through five call sites. A plan-level
key therefore needs a new plan-context argument; `design-options.md` s.5.5's "zero new plumbing"
is about a per-LEG `module_prefix` class. Module alone is not enough (orchestrator file_scoped
43 s vs full_suite 1 959-3 066 s, 69 vs 130 arrivals), so `scope_kind` must be passed as
structured data - the measure/sim seats got it by regex on the command string, which
production must not copy (heuristic 12). [S] But plan-level keying buys nothing for a COARSE
class: adding `two_class_leg` = `(exp<300, a, pid, seq)` to the study's open sim gives mean
0.65/1.33/1.81 h vs plan-level 0.65/1.32/1.81, p90 5.65 vs 5.58, max 9.14 vs 9.00. The
"leg-SEPT harms p90" trap is about a fine-grained leg sort, not a 2-class leg key.
**SHOULD SAY:** "Class by leg from `(module_prefix, scope_kind)`; one new structured kwarg;
plan-level keying is unnecessary for two classes."

**13. CLAIM** (s.4 reify; s.1 item 7) "FIFO only (p99 -51 %, max -73 %, mean +-0) ... no reify-repo change."
**VERDICT: mechanism survives; magnitudes unverifiable.** [C] `map-sem-reify.md` s.1: reify's
orchestrator runs the code-default N=1 DF gate ahead of `verify.sh`'s bash semaphore, so an
in-process ordered gate does order reify task legs. But `sim_reify_sem.py` is open-Poisson on a
system whose restart sweeps cancel 23-47 in-flight dispatches (finding 4c) - i.e. also closed
with a large WIP. Closed-loop FIFO keeps its tail win on DF (finding 3), so direction is safe;
the -51/-73 % figures are from the wrong model class. The reify rho 0.93 uses a current-lane
service sample, so it does not share finding 1's pooling error.

**14. CLAIM** (s.3) "Rankings are stable across rho in {0.68...0.98}; magnitudes are not."
**VERDICT: survives inside the open model, irrelevant outside it.** Shown for the DF semaphore
policies (model-sim s.4.1/s.12, and my paired re-runs). Not shown for merge (bracketed by server
count, not rho). In the closed model all size-blind and 2-class policies TIE on the mean, so
"ranking" on the headline metric does not exist.

**15. CLAIM** (brief item 3) is the wait proxy measuring something other than slot wait?
**VERDICT: proxy survives.** [C] In the concurrent branch `test/lint/type` start in one
`asyncio.gather`; lint/type are ungated; `started_at` for test is stamped inside
`_admission_slot`. `_preprovision_shared_venv` runs BEFORE the gather (cold only), PSI gates are
at dispatch, module locks at dispatch. The only thing between the two stamps is the executor
hop + flock poll. Residual caveat: `_ADMISSION_EXECUTOR_MAX_WORKERS = 64`; beyond 64 blocked
waiters, further acquires (and their `slots_dir.mkdir`) queue FIFO in the pool without polling -
still slot wait, but not "random". 24 tasks x 1.42 legs + an 8-9-module sweep/checkpoint
fan-out sits below that today.

---

## Omissions a decision-maker needs

1. **Task 5139** (pending, high: "`_admission_slot`'s role gate is broken on both ends: merge/mainprobe bypass all concurrency capping AND still contend for the ... 64 workers") is mentioned in NONE of the nine documents. It edits the same function as step 1, its second half is partly overtaken by the inline ungated path already in `_admission_slot` (task 5424), and an asyncio-level ordered gate removes the executor contention by construction. 5413's own text cites 5139. Step 1 must be scoped against it.
2. **The service-time trend is the story** (finding 1): orchestrator full-suite task legs 644 s -> 3 066 s in ten weeks, 86 % of slot work. Candidate causes each already have a knob or owner and none is an ordering matter: `verify_admission_pytest_n: "8"` interim cap (2026-08-19), suite growth, seven orchestrators on one host, merge `-n 16`. Halving that one cell is worth more than any policy here; so is asking why 130 of 199 orchestrator task legs run FULL suite rather than file-scoped (`verify_plan` scope decision).
3. **Pending tasks competing for the same prize, not ranked against the ordering work**: 5410 (D1 disjoint-rebase skip, high; prior art 4.79 h/day of slot demand), 5409 (NULL tip_sha => 8.1 % of greens re-verify on redispatch), 5415/5416 (remote host), the unowned fail-fast gate. The synthesis names D1 and remote host in one clause under reify only. Given finding 2 (ordering cannot move the mean), s.5 should say plainly that steps 4-5 are tail/fairness work and that mean duration and throughput are owned by 5410/5409/remote-host/service-time.
4. **Task-lane fail-fast across a plan's legs** does not exist (finding 3): a red 20 s leg does not stop a sibling's 51-min orchestrator leg, which is usually still WAITING for the slot and could be cancelled for free. With short-first ordering this becomes cheap demand reduction - the one place where 2-class could pay in throughput. Unmentioned.
5. **Restart = whole-WIP cancellation** (24 DF, 23-47 reify per event). The "restart kills exactly one in-flight leg" row is true of slot legs and misleading about cost: 24 tasks re-dispatch, re-rebase and re-verify from attempt 0 (modulo the tip-keyed green checkpoint, which 5409 says fails closed 8.1 % of the time). That is a demand multiplier on a rho ~ 1 slot and belongs in s.3.
6. **The design seat's U3 blocker was closed by assertion.** "watch its p90" (step 5 risk) is not the number the design seat asked for; finding 2 supplies it (+74 % mean, +56 % p90, max above FIFO).
7. `evals/` as an out-of-process gated acquirer when pointed at the DF root (finding 11), relevant to the live-shadow-eval work (5382-5395).

## The three changes that matter most

1. **Replace the open-queue headline with the closed-loop one.** WIP is pinned at 24 [M] and the slot is saturated [M/D]; by Little's law no ordering moves mean plan or task residence. FIFO = tail lever (p90 -30 %, max -66 %, median +35 %, mean 0). 2-class = redistribution (short -90 %; orchestrator full-suite +74 % mean, +56 % p90, max above FIFO), to be chosen or declined as a fairness policy, with the long-class table in the document. Strike "-31...48 % mean", "max no worse than FIFO", "the long class barely notices", and "FIFO -10...20 % mean" as steady-state claims.
2. **Rewrite the utilisation section.** rho ~ 0.99 from the census with era-matched service cells; the 0.68 came from pooling a cell whose mean tripled; the 411 calls are ungated train gates (r = 0.998 vs out-of-episode gate runs). Lead s.3 with the service-time trend and the 86 % work share of orchestrator full-suite legs, and re-rank s.5 so that demand/service work (5410, 5409, full-suite scope, the `-n 8` cap, host contention) sits above every ordering step. Drop the FIFO-inverts-under-restart finding and the restart-stable-stamp requirement (not reproducible; mechanism absent from the code).
3. **Correct the merge section's three errors**: `_compute_effective_priorities` is tier min-rank, not most-dependents; the recommended aged tie-break is worth ~0.4-0.6 h/day (the 0.9-4.3 belongs to unaged count-keyed `most_deps`) of a pool measured with current-status joins and of little dispatch value under pinned WIP; and the xdist A/B is confounded and self-contradicting (same gate p50, landings -37 %), so "order cannot raise throughput" is reasoning, not measurement. Step 6 should be presented as inversion hygiene, explicitly optional.

---

## Appendix - critic scripts (session scratchpad; stdlib; import `scripts/sim_lib.py` unchanged)

`closed.py::run(policy, N=24, Z_h=2.74, dark_hpd, days=60, warm_days=5, thresh=300)` - N tokens;
each: exponential think Z, then submit one plan bootstrapped from `sim_lib.load_plans` (all K
legs at one instant, services drawn from the leg's cell pool), wait for its last leg, repeat;
optional open Poisson dark stream at the pooled mean; one non-preemptive server; selection via
`sim_lib.make_key` (random = uniform pick). Reports plans/day, rho, plan completion
mean/p50/p90/max overall, by class (`plan_exp < thresh`), for dark jobs, and leg wait.
`closed_win.py` - same, with `sim_lib.service_cells` replaced by era-matched cells: per
`(module, scope_kind)`, legs with `test.started_at >= 2026-09-12` when n >= 5, else all-time;
identical exclusions to `sim_lib.service_cells`. Also runs the study's OPEN `simulate` on those
cells with per-class output (random 4.60 / FIFO 3.44 / 2-class 2.23 h; LONG 4.73 / 3.82 / 3.95).
`paired.py` - `simulate` for random / leg_fifo / two_class on identical seeds (15), dark 0/6/7
h/day and `mtbr_s` in {None, 8.3 h, 42 h}; reports per-seed relative differences.
`legclass.py` - monkeypatches `sim_lib.make_key` to add `two_class_leg`.
WIP / cancellation / out-of-episode `merge_verify` queries: `events` and `task_results` in
`data/orchestrator/runs.db` (both projects), `mode=ro`.
Z derivation: N/X = 24 / (212 `task_started` / 6 d) = 16.3 h per dispatch; verify wall 7.62 h x
(290 phase entries / 212 dispatches) = 10.4 h; non-verify 5.9 h / (75.7 plans/day / 35.3
dispatches/day = 2.14 plans) = 2.75 h per plan (the seat's route, 11.92 - 7.62 over 1.57, gives
2.74). The invariance result does not depend on Z; only the absolute level of R does.
