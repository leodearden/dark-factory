PAUSE COUNTERFACTUAL seat, 2026-11-12, read-only. Scripts and raw outputs: /tmp/team-weights-1112/pause/ (load.py builds snap.pkl from the
four event stores; timeline.py builds a per-minute grid 10-25T00Z..11-12T15:22Z; every other script reads that grid; each *_out.txt holds
its printed result). Data cut: event stores 11-12T15:22Z (DF), reify 11-12T05:42Z, sc 11-10T22:00Z, scp 11-10T21:38Z (no events since,
paused). Sampler PSI (capacity seat's psi.pkl) to 11-12T12:00Z; PCP idle/load from 10-28T00:10Z to 11-12T00:10Z; systemd CPU records
10-25..11-12T12:03Z. Code read at main 7848173a7d; the running DF service started 11-10T23:48Z, the three paused services 11-10T20:58-21:00Z.

Paths confirmed: <root>/data/orchestrator/runs.db and <root>/.taskmaster/tasks/tasks.db for dark-factory (df), reify, solar-challenge (sc),
solar-challenge-platform (scp).

"All-capped" = every one of the 7 pool accounts (max-b..max-h) capped. Cap intervals are built from the UNION of all four projects'
account_events and invocations (all four bill the same 7 accounts, F9): cap_hit until the first of resets_at, a 'resumed' event, or a
real (>60 s) turn by that account in any project. W0 = 10-25T00Z..11-08T16:00Z; pause = 11-08T16:04Z..cut.

```findings
F1. BENEFIT BY SUB-INTERVAL (uncapped minutes). The local merge verify fell at the END OF THE DRAIN, before the slot change; slots 1->2 did
    not move it; the laptop task-leg move slowed the laptop merge verify; the landing-rate jump came with the merge-lane deploy, not the pause.
    figure (uncapped rows; h | idle cores | CPU PSI | others' local verifies running | DF local merge p50 min (n) | DF laptop merge p50 (n) |
      local/laptop | DF landings per uncapped hour):
      W0 all       321 h | 1.0  | 49.2 | 2.99 | 43.4 (251) | 16.9 (278) | 2.58 | 0.96
      W0 last 3 d   62 h | 1.1  | 49.0 | 3.25 | 40.1 (64)  | 17.0 (75)  | 2.36 | 1.24
      drain 11-08T16:04..11-09T10   18 h | 1.3 | 46.7 | 2.46 | 46.8 (20) | 25.5 (2) | - | 0.95
      paused, 1 slot ..11-09T14:13  4.2 h | 3.7 | 26.7 | 0.30 | 34.0 (7) | none (laptop not used) | - | 1.66 (7 landings)
      paused, 2+ slots ..11-10T11:22 16 h | 6.4 | 20.3 | 0.00 | 35.9 (16) | 16.2 (18) | 2.21 | 1.42
      paused, DF task legs on laptop ..11-10T23:48  11 h | 3.7 | 26.0 | 0.16 | 36.6 (19) | 19.0 (16) | 1.93 | 1.42
      paused, after merge-lane deploy ..cut  29 h | 5.6 | 21.3 | 0.04 | 33.7 (29) | 21.0 (45) | 1.61 | 2.08 (18% of landings are train members)
      All-capped rows: W0 31 h, idle 8.9, PSI 30.6, others' local verifies 1.67, local 31.4 (22); paused rows idle 14-17, PSI 7-8, local 23.7-24.8.
    rule: merge verifies passed, >60 s, bucketed by END time, counted in the uncapped row only if <50% of the run fell in all-capped minutes.
      Landings = merge_finalized state=done rows (train members carry their own rows). "Others' local verifies" = reify+sc+scp local
      merge_verify intervals plus their admission legs (granted..granted+held), time-averaged. local/laptop = ratio of the two medians, a
      stretch proxy that removes DF tree growth (laptop is NOT a clean control in W0, see F3).
    window: as listed   population: workstation; DF 998 merge verifies since 10-24; four event stores
    kind: measured totals and ratios
    evidence: pause/intervals.py -> intervals_out.txt

F2. NATURAL EXPERIMENT INSIDE W0, free of the slot change, the merge-lane deploy and the laptop leg move: when the pool capped and every
    project's agents stopped, DF's local merge verify ran 28% faster, close to the paused figure.
    figure: W0 uncapped 43.4 min p50 (n=251) vs W0 all-capped 31.4 min (n=22); paused uncapped 33.7-36.6. Host in W0 capped minutes: PSI 30.6,
      idle 8.9 cores, others still running 1.67 local verifies, all agents 0. DF landings 1.27/h in W0 capped minutes vs 0.96/h uncapped (the
      lane keeps draining with no new arrivals).
    rule: as F1   window: W0 (30.6 all-capped hours in 352)   population: DF local merge verifies
    kind: measured totals   evidence: pause/intervals.py (W0 rows)

F3. DOSE-RESPONSE IN W0. Each concurrent other-project local verify adds about 5.7 min to DF's local merge verify; each other-project
    agent adds about 0.66 min. The laptop merge verify ignores workstation load but is slowed 4.75 min by each concurrent OTHER-project
    LAPTOP merge verify, so the laptop shares contention too.
    figure (OLS, DF local merge minutes, n=273): dur ~ k_oth + k_df + day: k_oth +5.70 (se 0.77), k_df (DF's own legs) +5.02 (2.53),
      day -0.51 (0.13), R2 0.22. Adding others' agents: k_oth +4.78 (0.76), ag_oth +0.65 (0.14), R2 0.28. Buckets by k_oth: [1,2) 33.7 (n=18),
      [2,3) 41.4 (132), [3,4) 43.7 (102), [4,5) 52.7 (17). By host PSI: [30,40) 32.5, [40,50) 39.1, [50,60) 47.2, [60,70) 56.1.
      Laptop (n=307): k_oth +0.50 (0.33), R2 0.05; by others' laptop merge verifies: lap_oth +4.75 (0.34), R2 0.41 (buckets 15.9 / 16.5 / 21.6 / 24.5).
      Others' verify-scope CPU from systemd records (credited at scope stop) shows no relation (+0.05, se 0.15): crediting at stop is too coarse.
    rule: exposures time-averaged over each verify's own span from the per-minute grid; passed, 60 s < dur < 120 min, ended in W0; pure-python
      normal equations with intercept. Observational: busy hours raise every exposure together.
    window: W0   population: 273 DF local, 307 DF laptop merge verifies
    kind: ratio (minutes per concurrent unit)   evidence: pause/doseresp.py -> doseresp_out.txt

F4. THE EXISTING ROLE WEIGHTS ALREADY WORK IN THE DIRECTION WEIGHTS PREDICT, and the W0 model predicts the pause's DF verify time out of
    sample. A weight-100 foreign merge verify costs DF's weight-100 merge verify about 4.7x what a weight-33 foreign task leg does.
    figure (OLS n=273, R2 0.31): others' local merge verify (weight 100) +6.75 min (se 0.95); others' task leg (33) +1.43 (1.25); others'
      background leg (10) +0.72 (1.73); others' agent (inside its orchestrator service, default weight 100) +0.66 (0.15); DF's own agents +0.32
      (0.23); DF's own legs +5.69 (2.44); day -0.56 (0.13). Others' contribution at W0 mean exposure C = 2.7 + 3.0 + 0.3 + 4.8 = 10.8 min of a
      43.9 min mean. Removing C and adding DF's own measured growth in the pause (agents 3.7 -> 8.4, legs 0.88 -> 1.3: +3.8 min) predicts
      37.0 min; observed paused pre-deploy uncapped medians 35.9 and 36.6.
    rule: as F3, exposures split by the foreign entity's role weight (verify.py::_resolve_scope_cpu_weight, config verify_cgroup_cpu_weight_*
      merge 100 / task 33 / background 10, per interventions.md F16). Weight and work size are confounded: a merge verify runs a bigger gate
      than a task leg, so the 4.7x is an upper bound on what weight alone buys.
    window: W0   population: 273 DF local merge verifies
    kind: ratio; the 37.0 is a projection checked against a measured total
    evidence: pause/weighttest.py -> weighttest_out.txt; pause/projection.py -> projection_out.txt

F5. WHAT BINDS DF: the merge lane, but not its hosts. Waiters queue behind in-flight verifies in 90%+ of minutes while the laptop, which
    carries most landings and is preferred first (verify_host_policy: prefer_remote), holds a DF merge verify only a third of the time.
    figure (uncapped): share of time a DF merge verify is running, local / laptop: W0 0.70 / 0.36; W0 last 3 d 0.78 / 0.42; paused pre-deploy
      0.82 / 0.33; post-deploy 0.76 / 0.63. Heartbeat minutes with >=1 waiter beyond in-flight: 90%, 93%, 92%, 96% (mean waiters 7.8, 8.4,
      7.8, 6.5). Landings carried by: W0 laptop 58% / local 41%; paused pre-deploy local 52% / laptop 46%; post-deploy laptop 54% / local
      28% / train members 18%. W0 arrivals (first-ever merge_queued) 30.9/day vs landings 23.7/day: DF was not arrival-bound in W0.
      DF agents running in uncapped minutes: W0 4.1, pause 8.0-8.7, against 18 slots.
    rule: running share = time-weighted DF merge_verify intervals per runner; waiters = merge_heartbeat depth - inflight_total, heartbeat
      older than 15 min = no reading; landing attributed to the runner of that task's last passed merge_verify within 6 h before it, else
      train member.   window: as listed   population: DF
    kind: ratios   evidence: pause/binds.py -> binds_out.txt; pause/cost.py (df block); dark-factory-orchestrator.yaml verify_host_policy
    Reading: lane throughput does not scale with host count or CPU (the laptop is idle 60%+ while waiters queue, pre-deploy). The pause can
      only have helped DF landings through verify DURATION on the lane's critical path, which F6 bounds.

F6. BOUND ON DF LANDINGS GAINED FROM THE PAUSE: about +2 to +6 per day through verify speed; the observed change against the comparable
    baseline cannot be distinguished from zero.
    figure: bound (verify-speed channel): gain <= L0 x [s_local x (d0_local/d1_local - 1) + s_laptop x (d0_laptop/d1_laptop - 1)].
      With measured speedups (local 43.4 -> ~36, laptop 16.9 -> 16.2): 23.1 x (0.41 x 0.21 + 0.58 x 0.04) = +2.5/day.
      With F4's full removal of foreign contention and no DF growth (local 43.9 -> 33.1; laptop 16.9 -> 14.2 from F3's 4.75 x 0.57):
      23.1 x (0.41 x 0.33 + 0.58 x 0.19) = +5.6/day. Projection.py's local-only figure: +3.1/day (+1.8 with DF's own growth).
      Observed, uncapped landings/day: W0 23.1 -> paused pre-deploy 34.9, +11.8 (Poisson 95% [+2.5, +24.5]); W0 last 3 d 29.6 -> 34.9, +5.3
      (Poisson [-5.4, +20.6]). Daily counts: W0 10-25..11-07 mean 24.0 (sd 6.9) vs 11-09..11-11 mean 35.3 (sd 2.5), +11.3 [+6.6, +16.0];
      against 11-05..11-07 (33.0, sd 2.6): +2.3 [-1.9, +6.5].
    rule: L0 = W0 uncapped landings/day; shares from F5; lane assumed to convert verify-time savings into landings 1:1 on each host, an
      upper bound because F5 shows hosts are not saturated. Poisson intervals understate spread (train members land together).
    window: W0, 11-05T16..11-08T16, 11-09T10..11-10T23:48 (pre-deploy only; post-deploy landings are the merge-lane change)
    population: DF   kind: bound (and measured totals for the observed rows)   evidence: pause/binds.py, pause/projection.py
    Reading: every project's landings rose in W0's last week (DF 21.8 -> 26.8/day, reify 15.6 -> 20.3, sc 18.8 -> 22.9, scp 23.8 -> 26.3,
      10-25..11-02 vs 11-03..11-08T16), so the W0-wide baseline overstates the pause's effect; the last-3-day baseline is the fair one.

F7. COST, LANDINGS: the three paused projects forwent about 210 landings in 3.97 days (bound), about 12 of them high-tier at W0 tier shares.
    Much of it is deferral, not loss: sc was demand-bound and reify's dispatchable backlog is finite.
    figure (per project: W0 landings/day | W0 first-ever merge_queued/day | W0 running tasks mean | W0 task-leg admission wait median |
      W0 lane waiters | W0 landings by current tier high% | forgone bound | high-tier forgone | pending now | READY now (high) | max_concurrent):
      reify 17.4 | 17.7 | 18.0 | 509 min | 0.6 | 11% | 17.4 x 3.97 - 17 = 52 | 6 | 2,529 | 138 (11) | 16
      sc    20.4 | 20.4 |  3.8 |  10 min | 0.1 |  5% | 20.4 x 3.97 - 8 = 73 (but only ~61-69 were ready) | 3 | 165 | 61 (4) | none in yaml (config default 3)
      scp   24.8 | 25.7 | 17.8 | 247 min | 0.3 |  3% | 24.8 x 3.97 - 13 = 85 | 3 | 562 | 223 (9) | 14
      (DF for scale: 23.7 | 30.9 | 18.6 | 109 min | 7.6 | 19%.)
    rule: landings and first-ever merge_queued (min timestamp per task over all history) from each project's runs.db; running tasks =
      task_started..task_completed pairs on the minute grid (pairing may over-count across restarts); tier = CURRENT tasks.priority of the landed
      task; forgone = W0 rate x pause days - landings since the pause; READY = pending with every local dependency done/cancelled and every
      metadata.external_deps entry done in the named project's store.   window: W0; pause 11-08T16:04..cut
    population: 3 projects   kind: bound   evidence: pause/cost.py -> cost_out.txt
    Reading: reify and scp were bound at their OWN verify admission (CPU-gated: task legs waited 8.5 h and 4.1 h median), not at their lanes
      and not by arrivals, so their throughput is CPU-sensitive. sc was demand-bound (3.8 running, 10-min waits, empty lane): its pause cost is
      delay only. Reify, despite 2,335 tasks parked behind its product gate, has 138 dispatchable now (about 8 days at its W0 rate), so its
      halt is not explained by the product gate alone; once those 138 are done its rate falls to its arrivals.

F8. COST, ALLOWANCE: the pause did not leave allowance unused. It began with 6 of 7 accounts weekly-capped by all four projects' W0 use;
    DF alone exhausted each account that reset within about a day, and since the first resets DF's uncapped spend is about the whole pool's.
    figure: weekly-capped at 11-08T16:04Z: max-h (until 11-09T21), max-f (11-11T13), max-c (11-11T10), max-d (11-12T13), max-b (11-13T12),
      max-e (11-14T17). After the pause: max-g reset 11-08T13:00, capped 11-09T16:01 ($1,010 in 27 h, DF $728); max-h reset 11-09T21:00,
      capped 11-10T22:37 ($1,004 in 26 h, DF $1,002). Dollars per account-week (16 complete account-weeks since 10-24): 458..1,774, median
      about $1,110; latest week per account sums to $7,677/week = about $1,100/day for the pool. Spend, $/day (cost_usd): W0 DF 424 + others
      764 = 1,187, all-capped 8.7% of minutes; pause DF 755, others 29, total 784, all-capped 17.2%. DF in uncapped minutes: W0 $464/day,
      W1 $1,034/day, since 11-11T10:00 $1,144/day (agents 8.7). Agent-hours/day: W0 DF 90 / others 180; W1 DF 150 / others 3.
    rule: weekly cap_hit deduplicated by (account, resets_at) across the four account_events tables; account-week spend = all four projects'
      invocations cost_usd on that account between its previous weekly reset and the cap. cost_usd is the CLI's API-equivalent price, a proxy:
      the weekly limit's own units are not known, and interactive sessions on the same accounts are not in any invocations table.
    window: 10-24..cut   population: 7 accounts, 4 projects   kind: measured totals (spend) and ratio (per account-week)
    evidence: pause/weekly.py -> weekly_out.txt; pause/allowance.py -> allowance_out.txt
    Reading: the W1 cap stalls (24% in capacity.md F3) are largely the hangover of W0 consumption by all four projects, not DF's alone. The
      allowance is zero-sum: DF's share went from 36% of pool spend (W0) to about 96% (pause); at an uncapped $1,034-1,144/day DF alone can
      consume the pool. A weight on allowance is therefore a real choice; a CPU weight does not touch it.

F9. COUNTERFACTUAL UNDER CPU WEIGHTS (projection for W0): DF's local merge verify 34-40 min instead of 44, and the others lose 0-13 landings
    per day instead of 63.
    figure: DF local merge verify (W0 mean 43.9): 3:1 -> 36.7 (A) .. 40.3 (B) min, -16% .. -8%; 10:1 -> 34.2 (A) .. 39.1 (B), -22% .. -11%;
      pause (s=0, no DF growth) 33.1. DF landings gain (local verify channel only): 3:1 +0.8 .. +1.9/day; 10:1 +1.2 .. +2.7/day; pause +1.8 .. +3.1.
      Others' CPU, W0 project supply about 22.2 cores (others used 11.8, DF 10.1): 3:1 per project (DF 3 vs 1+1+1, DF entitled 11.1) others
      lose 0 .. 1.0 cores (0-8%), reify+scp landings lost <= 0 .. 3.6/day; 10:1 per project, or 3:1 DF-vs-all-others (DF entitled 16.7-17.1)
      others lose 0 .. 3.5 cores (0-30%), reify+scp landings lost <= 0 .. 12.5/day; sc about 0 (demand-bound). The pause cost 62.6/day (F7).
    rule/assumptions: (1) F4's linear contention model holds at W0 exposures. (2) Under weights each foreign entity's contention cost is
      multiplied by s: assumption A, pure CPU-time sharing, s = w_other / w_DF; assumption B, half the cost is not schedulable (SMT siblings,
      cache, memory bandwidth), s = 0.5 + 0.5 x w_other/w_DF. The 50% in B is not measured; F4's ordering (weight-33 legs cost far less than
      weight-100 merges) is consistent with A but confounded with work size. (3) CPU weights are work-conserving and bite only under
      contention; W0 uncapped had load1 mean 152 on 32 threads, runqueue_ratio 3.93, busy 30.5 threads, so contention was continuous.
      (4) DF's CPU demand under weights is 10.1 (its W0 use) to 13.6 cores (its paused use plus 1 for uncapped minutes); DF gets
      min(demand, entitlement); others take the rest. (5) reify and scp landings fall at most in proportion to their lost CPU (they were
      admission-bound, F7). (6) Allowance and the laptop are unchanged: CPU weights on the workstation give DF none of the pause's
      allowance gain (F8) and none of the laptop relief (F3: 4.75 min per foreign laptop merge verify, mean 0.57 concurrent).
      Non-project load (8.3 cores: interactive sessions, Chrome, fused-memory, docker) stays outside the groups and ahead of both.
    window: W0   population: modelled   kind: projection   evidence: pause/projection.py -> projection_out.txt; pause/weighttest.py (load block)
    CPU-heavy vs agent-time: reify's verify scopes 6.61 cores in W0 against its orchestrator (agents) 2.37 + other units 0.39; solar verify 1.05
      against orchestrators 1.38 (capacity.md F9). In F4 terms, of the 10.8 min foreign contribution 6.0 came from verify scopes and 4.8 from
      foreign agents (7.3 running on average), so agent time is not CPU-free: an agent at weight 100 costs DF's merge verify as much as a
      half of a weight-33 task leg (0.66 vs 1.43 min).

F10. RESUME SIZING: 111 DF tasks become ready the instant 8528 lands, together with solar-challenge and solar-challenge-platform (8792,
     8793 need only 8528); reify (8791) also needs 7721 and 7824. The last comparable simultaneous start took the host from PSI 25 to 63 in an hour.
     figure: DF pending with an edge on 8528: 168; ready when 8528 is done: 111 (high 18, medium 27, low 66); the other 57 also wait on
       9256, 9182, 9140, 5040 and others. 8528 waits on 8527 (pending). Of the 138 tasks added behind 8528 on 11-10, 131 are still pending,
       0 done. DF already has 2,262 dependency-ready pending tasks for 18 slots, so the 111 change DF's mix, not its load.
       Resuming capacity: scp 223 ready / 14 slots; sc 61 ready / default 3; reify 138 ready / 16 slots; all three at 0 in progress.
       Analogue 1, 11-04T10:00Z pool cap reset with all four projects holding queued work: agents 1.0 -> 47 within 30 min (DF 17.6, reify 13.4,
       scp 12.0, sc 4.2), CPU PSI (15-min means) 25 -> 33 -> 63 at 11:00Z, idle 6.1 -> 0.3 cores. Analogue 2, 11-07T08:06Z restart of
       reify+sc+scp: 50 task starts in the first hour, PSI 34 -> 61 within 75 min, peak 10-min PSI 65.7. Analogue 3, 11-04T07:59Z restart of
       all four while capped: 47 task starts in 15 min, local verifies 0.4 -> 3.6 within 30 min (requeued verifies), PSI 2 -> 36.
     rule: live task store dependencies (status done/cancelled counts as met); analogues = clusters of >=3 projects beginning a new run_id
       within 15 min, and all-capped -> uncapped transitions, from the minute grid; 15-min bins.
     window: live store at 11-12T15:2xZ; W0 analogues   population: DF task store; 4 projects
     kind: measured totals   evidence: pause/resume.py -> resume_out.txt (and the 15-min bins in this seat's transcript, same grid)
```

```mechanisms
M1. Why removing other projects shortened DF's merge verify: DF's local merge verify runs in a transient scope at weight 100, a sibling of
    every orchestrator service (weight 100, holding that project's agents and their tool runs) and of every other project's verify scopes
    (100/33/10 by role) under app.slice (interventions.md F16). In W0 the host ran load1 ~152 on 32 threads, so the CFS share of a weight-100
    scope was cut by every runnable sibling, most by the other weight-100 siblings (F4). Removing three projects removed ~3 local verifies and
    ~7-8 agents' worth of weight-100 and weight-33 siblings.
M2. Why that barely moved landings: the lane keeps waiters 90%+ of the time while its hosts are partly idle (F5); verify duration on the
    critical path is one input among the lane's serial steps. The 11-10T23:48Z merge-lane deploy, not the pause, is where the landing rate
    stepped (1.42 -> 2.08 per uncapped hour, train members 18%). The lane's internal serialisation was not studied here (merge-policy seat).
M3. Why the laptop is a shared resource, not a control: every project runs verify_host_policy: prefer_remote against the same leo-laptop,
    each with its own HostAllocator (verify_runner.py::HostAllocator, one slot per host per worker), so nothing arbitrates the laptop across
    projects; in W0 the others ran 759 passed laptop merge verifies to DF's 308 passed and each concurrent one added 4.75 min to DF's (F3).
M4. Why the allowance is not divided by CPU weights: all four projects draw on the same 7 accounts with weekly limits (F8). Whoever runs
    agents first in a weekly window spends it; caps observed in W1 were set by W0's four-project consumption.
```

```unknowns
U1. The non-schedulable share of contention (SMT siblings, cache, memory bandwidth) is not measured; assumption B's 50% is a placeholder.
    It decides whether 10:1 buys 22% or 11% on DF's local merge verify. Measuring it needs either per-cgroup PSI for verify scopes (the sampler
    has own_cpu_some10 only for orchestrator services) or an A/B with weights applied.
U2. F4's role coefficients mix weight with work size (a foreign merge verify is a bigger job than a task leg). The 4.7x ratio is an upper bound
    on what weight alone buys.
U3. Whether lane throughput converts verify-time savings into landings at all at the margin (F5 shows idle hosts with waiters); F6's bound
    assumes 1:1. The lane's serial structure belongs to the merge-policy seat.
U4. cost_usd is a proxy for the weekly limit; the limit's units and model weighting are unknown, and interactive sessions on the same accounts
    are invisible to the invocations tables. Pool capacity ~$1,100/day is a ratio-derived estimate, range 458-1,774 per account-week.
U5. Laptop CPU arbitration: whether verify scopes on the laptop carry cgroup weights was not read; the laptop utilisation split by project in
    W0 comes only from merge_verify events (task legs on the laptop are not in admission plans).
U6. sc's effective max_concurrent_tasks: its yaml sets none (config.py OrchestratorConfig.max_concurrent_tasks default 3) yet W0 shows 3.8
    running on average, p90 7; the stint pairing may over-count, or the config is supplied another way. Not resolved.
U7. W0 PCP idle/load exists only from 10-28T00:10Z; sc and scp event stores end 11-10T22:00Z / 21:38Z (nothing since, consistent with the halt).
U8. "Forgone" vs "deferred" for reify and sc depends on their arrivals after their ready backlogs drain, which this data cannot project.
```
