# Seat: CPU and memory mechanisms (cross-project weights team, 2026-11-12)

Read-only. Code read: Dark Factory `main` at `7848173a7d` (2026-11-12T13:55Z). The DF service started
2026-11-10T23:48:48Z; the last main commit before that is `1f240e6da2`. `git diff 1f240e6da2 7848173a7d` over
`verify.py`, `config.py`, `cli_invoke.py`, `psi.py` and `scheduler.py` touches only the verify reap path (relocated scopes).
Nothing that sets weights, memory properties or PSI changed, so on these points the running service and `main` agree.
(The deployed sha is assumed to be the last main commit before the restart. I did not read it from the unit.)
Host facts verified live: workstation kernel `7.0.0-31-generic`, systemd 255, 32 threads (AMD Ryzen 9 3950X), cgroup2 mounted
`nsdelegate,memory_recursiveprot`, `user@.service` `Delegate=pids memory cpu`. Laptop kernel `7.0.0-42-generic`, systemd 255,
16 threads, 60 G RAM, 135 G swap.
Scratch and re-runnable scripts are in `/tmp/team-weights-1112/mechanisms/` (`memholds.py`, `cg_delta.py`, `cpu_split.py` and their `*_out*.txt`).

Tags used below: **[host]** means verified on this host by reading the live tree. **[doc]** means documented kernel/systemd
behaviour (cgroup-v2 admin guide, `kernel/sched/fair.c`, `kernel/sched/psi.c`, `mm/vmscan.c`, `mm/memcontrol.c`) that I did
not measure here. **[code]** means read in DF source at the commit above.

## Findings

```
F1. Today every project's verify scope is a direct sibling of all nine orchestrator units under app.slice, and it carries
    the role weight, not a project weight. Units have no CPUWeight. Workstation scopes have no memory limit.
    figure: 9 orchestrator-*.service at cpu.weight 100; live df-verify-dark-factory-* scopes at 100 (merge, 15.5 GB), 33, 33 (task), 10, 10 (background); memory.high=max, memory.max=max on every scope
    rule: cat cpu.weight / memory.* of every child of user@1000.service/app.slice. Role inferred from weight (= verify_cgroup_cpu_weight_{merge,task,background} defaults 100/33/10)
    window: read 2026-11-12T15:2xZ          population: workstation, 87 app.slice children
    kind: measured total
    evidence: /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/*/cpu.weight; verify.py::_scope_launch_argv (systemd-run --user --scope, no --slice);
              verify.py::_resolve_scope_cpu_weight; config.py::VerifyCgroupMemoryConfig.systemd_properties (DF workstation yaml sets no verify_cgroup_memory)

F2. A CPUWeight on orchestrator-<project>.service reaches only the daemon and its agents. In W0 that was about a quarter of
    reify's CPU. Its verify scopes were about 70%, and compiles run by the shared sccache server are charged to no reify cgroup at all.
    figure: reify W0: verify scopes 70.5%, orchestrator-reify.service 25.3%, other reify units 4.2% (of 9.37 cores). sccache.service cgroup used 52.8 CPU-days in compiler children since 2026-09-12 (= 0.86 cores mean)
    rule: ratio of capacity.md F9's W0 core means (6.61 / 2.37 / 0.39). sccache: cgroup cpu.stat usage_usec (61.97 CPU-days) minus the sccache server pid's own CPU time (9.18 days), divided by 61.26 days of unit uptime
    window: W0 (prior seat); sccache 2026-09-12T09:39Z..2026-11-12T15:55Z    population: reify units; one sccache.service
    kind: ratio (reify units' share of reify CPU); measured total (sccache)
    evidence: /tmp/team-gates-1112/capacity.md F9; app.slice/sccache.service/cpu.stat; ps -o time -p 4466; agent placement: claude processes (nice 10 and 0) in app.slice/orchestrator-dark-factory.service/cgroup.procs;
              shared/src/shared/cli_invoke.py::_cpu_govern_prefix, ::_cpu_priority_prefix (agent spawn = nice prefix only, no scope)

F3. Interactive T3 sessions (this one included) are not favoured today. They sit in app-t3code.slice at CPUWeight 10, a sibling
    of the orchestrator units at 100. That weight comes from a runtime drop-in that a reboot drops. Terminal sessions (konsole) sit at 100.
    figure: app-t3code.slice cpu.weight 10 (drop-in /run/user/1000/systemd/user.control/app-t3code.slice.d/50-CPUWeight.conf); t3code.service 1000 with MemoryLow 1536M; konsole scope 100; jcodemunch-watcher 30 with MemoryHigh 8G / MemoryMax 12G (164,786 memory.high events)
    rule: live cpu.weight, systemctl --user cat. /proc/self/cgroup of this session = .../app-t3code.slice/app-t3code-agents.slice/.../t3code-claude-9208f060.scope
    window: read 2026-11-12T15:2xZ           population: workstation
    kind: measured total
    evidence: systemctl --user cat app-t3code.slice t3code.service; app.slice/jcodemunch-watcher.service/memory.events
    note: task 3394's 2026-08-03 ruling ("interactive work is privileged deliberately") was measured on terminal scopes at the default 100. It does not describe T3 sessions today.

F4. Every project's dispatch hold reads the same host-wide /proc/pressure files. In W1, 57 of the 69 holds were the memory-full
    arm, mostly with DF at or near its slot cap. 44 of the 57 fell in 06..16Z. The arm's threshold sits inside the ordinary band: a third of W1 minutes were at or above it.
    figure: 69 dispatch_deferred (57 mem_full_avg10, 12 io_some_avg10); in_flight 16-18 at 56 of 57 (one at 11); mem_full10 at hold p50 4.86, max 19.89; 9 of 57 also over mem_some 15;
            20 episodes (gap > 30 min), largest 24 holds 2026-11-11T11:43..13:25Z; 44 of 57 between 06 and 16Z; host mem_full10 >= 3 in 1,439 of 4,344 W1 minutes (33%)
    rule: runs.db events event_type='dispatch_deferred', timestamp >= 2026-11-09T14:13Z (one event per scheduler pass that skipped a heavy candidate; not ticks, not tasks).
          Minutes: per-minute max of load-samples psi_mem_full_avg10, leaf IS NULL. The brief's 56/68 was counted at about 11:49Z; one more mem hold landed at 14:36Z
    window: 2026-11-09T14:13Z..2026-11-12T14:40Z     population: DF scheduler only (reify and solar halted)
    kind: measured total; ratio (minutes over threshold / W1 minutes)
    evidence: /tmp/team-weights-1112/mechanisms/memholds.py -> memholds_out.txt, memholds_hours.txt; scheduler.py::Scheduler._phase_psi_gate; shared/src/shared/psi.py::_read_host_pressure, ::PsiSample.saturated, ::_ARMS

F5. Whose memory caused the 57 holds cannot be determined from the recorded data. The sampler has no per-cgroup memory series,
    and its two verify-process series read zero for the whole of W1. What is recorded points to host-wide reclaim and swap churn,
    not exhaustion: daytime swap-in ran at 19-35k pages/s with 44-92 GB available. Kernel reclaim threads in the root cgroup are busy,
    and they count toward host PSI but toward no project's cgroup.
    figure: verify_rss_total_bytes and verify_concurrency = 0 in all 4,344 W1 minutes. swap.pagesin 18.7-34.9k/s in 06-15Z 3-h samples on 11-10 and 11-11 vs 1.7-2.5k/s at 18-21Z, mem.util.available 44-92 GB.
            kswapd0 12.5 CPU-days, kcompactd0 7.6 CPU-days over 61.25 days uptime. Lifetime memory "some" stall: root 154,899 s vs user.slice 85,757 s + system.slice 8,800 s
    rule: load-samples per-minute max (leaf IS NULL); PCP pmrep -t 3h on the 20261110/20261111 archives; ps time of kswapd0/kcompactd0; memory.pressure total= of /proc/pressure/memory and each slice.
          Root-minus-children (60,342 s, 39%) is an INDICATION, not a bound: each cgroup's PSI aggregate is weighted by that cgroup's own non-idle time, so the totals do not subtract exactly
    window: W1 (sampler); 11-10, 11-11 (PCP); lifetime to 2026-11-12T15:2xZ (counters)   population: workstation
    kind: measured total; indication
    evidence: memholds_out.txt; pmrep -a /var/log/pcp/pmlogger/leo-MS-7C35/2026111{0,1} swap.pagesin swap.pagesout kernel.all.pressure.memory.full.avg mem.util.available;
              sampler/src/sampler/metrics.py::collect_process_metrics, ::sum_verify_rss (the series exist, so the classifier matches nothing)

F6. On the laptop all four projects prefer it for merge verifies, and nothing arbitrates across projects there. Each project's
    scopes are siblings in the laptop's own app.slice. The per-project limits differ: DF and reify set memory only, and the
    solar projects set static drop-ins, one of them with a CPU cap.
    figure: verify_host_policy prefer_remote in all 4 workstation yamls; laptop DF scopes cpu.weight 100 (merge role), MemoryHigh 24G / Max 40G / SwapMax 32G (from DF laptop yaml verify_cgroup_memory);
            reify 32/48/32 G (yaml); solar-challenge drop-in CPUQuota=800% plus 2/18/32 G; solar-challenge-platform 2/18/32 G. Laptop load 16.6 on 16 threads, cpu some avg10 12.3 at read
    rule: grep of the 4 dark-factory-orchestrator.yaml and 4 laptop yamls; cat of the live laptop scope files; ~/.config/systemd/user/*.scope.d/*.conf
    window: read 2026-11-12T15:43Z          population: laptop, 2 live DF merge scopes
    kind: measured total
    evidence: verify_runner.py::HostAllocator (one slot per host, per orchestrator process; no file lock or lease shared across processes);
              plans/remote-task-leg-verify-prd.md decision 16 ("neither role reads the other's state")
```

F1, F4 and F6 were re-read before return (F4 re-queried at 14:40Z; the tree and laptop reads are as quoted).

## Mechanisms

### M1. Where work sits today (workstation) [host][code]

```
/ (root)                                   cpu.weight n/a; kernel threads (kswapd0, kcompactd0) live here
├─ system.slice                 w100       docker (FalkorDB, Qdrant), system services
└─ user.slice / user-1000.slice  w100
   ├─ session-N.scope            w100      logind login sessions (2)
   └─ user@1000.service          w100      controllers delegated: cpu memory pids (no io, no cpuset)
      ├─ app.slice               w100      <- nearly everything
      │  ├─ orchestrator-<9 projects>.service   w100 each   daemon + agents (claude at nice 10; some at 0)
      │  ├─ df-verify-<project>-<hash>-<id>.scope  w100 merge / 33 task / 10 background   (systemd-run --user --scope, no --slice)
      │  ├─ sccache.service       w100      compiles for every project run as ITS children
      │  ├─ jcodemunch-watcher    w30, MemoryHigh 8G / Max 12G;  fused-memory, dashboard, reify-warm-lane-gc, legibility timers: w100
      │  ├─ app-t3code.slice      w10  (runtime drop-in)  T3 interactive Claude sessions
      │  ├─ t3code.service        w1000, MemoryLow 1.5G   T3 server
      │  └─ konsole / Chrome / Firefox / desktop apps   w100 each
      ├─ session.slice w100 · background.slice w30 · init.scope w100
      ├─ reify.slice/reify-governed.slice  w100  (empty; reify cpu_governance=false) <- SIBLING of app.slice
      └─ dfplan5205probe.slice  w100  (empty probe leftover)
```

- **Verify scopes.** `verify.py::_run_cmd` launches through `_scope_launch_argv`: `systemd-run --user --scope --quiet --collect --unit=… -p CPUWeight=<role weight> [-p MemoryHigh/Max/SwapMax]`.
  It passes no `--slice`, so systemd puts the scope in the user manager's default `app.slice` [host: live scopes are there].
  The weight is read per spawn (`_resolve_scope_cpu_weight`, green tier). A running scope keeps the weight it started with.
  Memory properties come from `verify_cgroup_memory`, one value for every role. The DF workstation yaml does not set it [host: memory.high=max].
- **Nested escape.** When `cpu_governance` is on, `cpu-governed-exec.sh` moves the workload into an inner scope under its own slice,
  so the outer scope's CPUWeight does not reach it (`verify.py::_govern_cpu_str` docstring). DF and reify both have it off today.
- **Agents.** Agents are plain children of the orchestrator unit. `cli_invoke.py` builds `_cpu_govern_prefix(env) + _cpu_priority_prefix(env) + cmd`,
  which is a `nice` prefix and no scope [host: 12 `claude` pids in `orchestrator-dark-factory.service`]. Nice orders threads only
  inside that cgroup. Between cgroups it has no effect.
- **Dispatch hold.** `scheduler.py::Scheduler._phase_psi_gate` reads one `PsiSample` (`shared/psi.py::read_psi_sample`) each time the
  scheduler selects. Host arms: `/proc/pressure/memory` full avg10 and some avg10, `/proc/pressure/io` some avg10, `/proc/pressure/cpu` some avg10 (off),
  and `procs_running / len(sched_getaffinity)` from `/proc/stat`. One own arm reads the own cgroup's `cpu.pressure` (off). The `df-<project_id>.slice`
  ancestor is used when one exists; today `own_cgroup` reads `orchestrator-dark-factory.service`.
  Hold rule: any armed arm at or above its threshold AND `len(_dispatched) >= min_inflight_floor`. Only non-deterministic candidates are held.
  There is no hysteresis (D4). Values:
  DF and reify: mem_full 3, mem_some 15, io 40, runqueue 6.0, floor 3. Solar ×2 have no psi block, so code defaults apply: mem_full 3,
  mem_some 15, io 40, runqueue off, floor 1 (`config.py::PsiAdmissionConfig`).
- **Verify-admission gate.** It reads no load signal. It is a static N per orchestrator process and per slots dir. Merge, offline and
  unknown roles bypass it (admission-policy.md P8; spot-checked `verify.py` leg path `admission = … LegSlot.TEST`).
- **IO is not weighted at all [host].** The io controller is not delegated (`Delegate=pids memory cpu`), so IOWeight on user units
  is inert (`app-t3code.slice` carries IOWeight=10 and has no `io.weight` file). Every NVMe runs scheduler `[none]`, which ignores
  ioprio, so the verify tiers' `ionice -c2 -n7` / `-c3` do nothing [doc for "none ignores ioprio"].

### M2. What a CPU weight guarantees, worked for this tree [doc for semantics; arithmetic is projection]

Semantics (cgroup-v2 `cpu.weight`, range 1..10000, default 100): siblings at one level divide the CPU time their parent receives,
in proportion to weight, **counting only siblings that are runnable**. The scheme is work-conserving: an idle sibling's share flows
to the others. A weight never caps: a lone runnable group may use every CPU. Each level composes multiplicatively.
On one detail the naive arithmetic is wrong. A group's weight is spread over the CPUs where it has runnable threads, in proportion
to its load on each (`fair.c::calc_group_shares`). A group with few threads therefore concentrates its weight on those CPUs.

**(a) Today: `CPUWeight=` on `orchestrator-reify.service`.** It moves the reify daemon and agents only (about 25% of reify's W0 CPU, F2).
Reify's verify scopes keep 100/33/10 as app.slice siblings, and reify's rustc run through sccache stays in `sccache.service`. The
setting is effective only while app.slice is contended and reify's agents are CPU-bound. Agents mostly are not: about 60 agents cost
about 1 core (load-throttle PRD §1). So today it does almost nothing to reify's CPU. The flat, contended case (two projects, each with
one merge and two task verifies, nothing else runnable): Σ = 2×(100+33+33) = 332, so each merge scope gets 9.64 threads, each task
scope 3.18, and each project 16.0. **Project importance cannot enter.** Any other runnable app.slice sibling (orchestrator units,
sccache, desktop) joins the same denominator.

**(b) After task 3394.** Its scope: `--slice=df-<fused_memory.project_id>.slice` at the scope spawn, the per-role `-p CPUWeight=` kept,
agents placed in the same slice, and `CPUWeight=100` set on the slice so that cpu.weight exists. systemd derives the parent from the
name, so `df-dark_factory.slice` sits under `df.slice`, and `df.slice` is a sibling of `app.slice` under `user@1000.service`.
Worked case, all factory work runnable and nothing else contending (32 threads), slice weights A:B = 300:100:

| | A (weight 3) gets 24.0 threads | B (weight 1) gets 8.0 threads |
|---|---|---|
| merge verify (100) | 24×100/166 = **14.46** | 8×100/166 = **4.82** |
| each task verify (33) | 4.77 | 1.59 |
| if A runs merge+task+background (143): background (10) | 24×10/143 = **1.68** | |

- Demand check: a merge verify runs pytest with 16 workers and a task verify with `-n 8`. Every scope above wants more than it gets,
  so the shares bind.
- **B's merge verify (4.82) is roughly equal to A's task verify (4.77). It is 2.9× A's background run and one third of A's merge.**
  The role weights compose inside a slice. Across slices the slice weight dominates.
- **Intra-slice values matter and 3394 does not set them.** If A's daemon unit (default 100) is runnable inside the slice, A's merge
  share falls to 24×100/266 = 9.02. Agent scopes at default 100 each would dilute it further.
- **Interactive and non-project load sit at the top level, above the project split.** At user@1000 level the factory competes as one
  entity: df.slice 100 against app.slice 100, session.slice 100, background.slice 30, and reify.slice 100 if reify's governance were
  turned on. If app.slice has 16 or more runnable threads, the whole factory is held to 16. B's merge then gets 4×100/166 = **2.41**.
  W1's non-pipeline load averaged about 9.3 cores (REPORT/capacity F9), below app.slice's 16-thread entitlement. On average, then,
  the factory keeps about 22-23 threads. But the floor of the arithmetic is 16, not 32, and bursts (an interactive `pytest -n 16`,
  a sccache compile storm) push it toward that floor.
  sccache, jcodemunch, fused-memory, the dashboard and the reify warm-lane gc are factory work that stays in app.slice unless they
  are moved. They count against the factory's top-level share and against no project.
- **What a weight cannot do.**
  1. *Bound memory or IO.* No effect on either (M3; IO is undelegated).
  2. *Help when the host is idle.* Work-conserving: no effect below saturation (24% mean idle in W1).
  3. *Guarantee latency of a single-threaded critical step.* Per-CPU share distribution protects a lone thread. If B has only that
     one thread runnable, B's whole 100 sits on its CPU against A's ≈300/32 ≈ 9.4 there, so the thread gets about 91% of a CPU.
     If B is also running its own verifies (about 17 CPUs occupied), its per-CPU weight is about 5.9 and the thread gets about 39%.
     **The step is starved by its own project's load, which slice weights cannot touch.** That needs nested weights or nice inside
     the slice. [doc, approximate]
  4. *Stop a starved test from timing out (task 5529).* In the 3:1 case, B's 8 threads serve 16 merge workers plus 16 task workers
     plus lint/type, about 0.25 CPU per worker on average and less during the measured runqueue bursts (15-37 runnable per core).
     A 10 s per-test marker then fires on a test needing more than about 2.5 s of CPU, and the merge `--timeout=300` on one needing
     more than about 75 s. pytest-timeout's thread method kills the whole xdist worker (task 5529). If the observed local merge
     verify (28-52 min) ran at about 9.6 threads, then at 2.41 threads it projects to about 112-208 min, past
     `verify_command_timeout_secs` 7200 s. That is a projection, assuming CPU-bound wall time proportional to 1/threads.
  5. *Express "A's urgent work beats B's background but not B's urgent work".* A slice weight is a single scalar per project.
- **Alternatives.**
  - `CPUQuota` / `cpu.max`: a hard cap. It adds a guaranteed ceiling for a project. It breaks work conservation (idle CPU is wasted;
    45% of W1's idle fell in account-cap stalls when a capped project could have used it), and CFS bandwidth throttling stalls the
    whole group until the next period. That is worse for latency and for 5529 than a weight. In use today only per scope on the
    laptop (solar-challenge 800%), and 3394 notes the only throttled cgroups it ever found were leaked reify test slices.
  - `cpu.idle=1`: the group behaves like SCHED_IDLE against its siblings. This is near-strict priority and still work-conserving.
    It is binary, so it cannot say 3:1, and under sustained saturation the idle group gets almost nothing, which means 5529 timeouts.
    A candidate only for background sweeps or a parked project.
  - `cpu.uclamp.{min,max}`: clamps utilisation hints for frequency selection and task placement. It is not a share. The workstation
    runs `amd-pstate-epp` with governor `powersave`, not schedutil [host], so it is inert here [doc].
  - cpusets (`AllowedCPUs=`): a hard partition. It is not work-conserving across partitions, and it is not delegated to the user
    manager (`Delegate=pids memory cpu` [host]), so it needs a host change. It breaks bursting into idle CPUs on both hosts.

### M3. Memory [doc for semantics unless tagged]

- **Mechanisms per project slice once 3394 exists.**
  - `MemoryMin` / `MemoryLow`: protection from reclaim. `memory_recursiveprot` is on [host], so the protection propagates to
    descendants without being set at each level. When the children's protection is overcommitted it is split in proportion to each
    child's usage below its low. Low is best-effort and min is hard; min can push OOM onto unprotected cgroups.
  - `MemoryHigh`: throttle plus forced reclaim above the value.
  - `MemoryMax`: OOM-kill inside the group. A killed verify leg is a no-verdict that falls back to local (task 6667). A kill of the
    daemon or an agent is worse.
  - `MemorySwapMax`: limits the group's swap.
  - Per-slice `memory.pressure` exists for any cgroup.

  **The only one that expresses a "share" is MemoryLow/Min**: a weighted protected working set, so reclaim hits the unprotected
  project first. MemoryHigh and MemoryMax are ceilings, not shares.
- **What the hold reads.** Host-wide `/proc/pressure/memory` only (`psi.py::_read_host_pressure`). The own arm is CPU-only
  (`read_own_cgroup_pressure` reads `cpu.pressure`). There is no own-memory arm. So **any** project's memory stall, and stalls
  belonging to no project, hold **every** project's dispatch, each at its own floor.
- **Two consequences for a design.**
  1. Host memory PSI counts kswapd and kcompactd (root cgroup), page-cache refault stalls of any task, and swap-in stalls (F5). So
     memory protections on project slices would change *whose* pages are reclaimed. They would not change the host signal the holds read.
  2. **MemoryHigh throttling counts as a memory stall** (`mm/memcontrol.c` over-high penalty is wrapped in `psi_memstall_enter`).
     It shows in the throttled group's PSI **and in the host's**. Giving a low-weight project a MemoryHigh would therefore raise the
     host mem PSI and could hold the high-weight project's dispatch. A per-project memory policy needs the hold to read per-slice
     `memory.pressure` (own arm), or it works against itself.
- **The 57 holds (F4, F5).** They held new heavy dispatch while DF had 16-18 of 18 in flight. In 56 of 57 cases that is where the
  floor (3) is irrelevant and DF's own load was at its ceiling. Whether a slot was actually free at the hold is admission-policy U6.
  The memory belonged to whoever was reclaiming at the time, and that is not recorded. Current swap holders [host, memory.swap.current]:
  sccache 6.8 G, t3code slice 5.4 G, konsole 4.9 G, DF service 2.4 G, system.slice 6.7 G. Lifetime swap-ins: konsole 9.4e9 pages,
  32% of app.slice's 29.4e9. Most of the rest is in departed cgroups (verify scopes, earlier service incarnations).
  A live 120 s sample at 15:38Z found the host quiet (mem full 0.36%), with nothing to attribute (`cg_delta_out1.txt`). The tool can
  be re-run during a daytime episode.

### M4. The two merge-verify hosts and the laptop [code][host]

- **Claim path.** Each orchestrator process owns a `verify_runner.py::HostAllocator`: one slot per host, held in process memory.
  `verify_host_policy: prefer_remote` takes the laptop first when it is FREE and not quarantined.
  - All four projects set prefer_remote, so up to four merge verifies (one per project) can run on the laptop at once, plus DF's
    remote task legs (gate turns, decision 16).
  - Laptop-side locks are per project: the verify-admission flock dir is keyed on the sha256 of `project_root`, and the merge role
    never takes it. The `.merge_verify.lock` guards that project's persistent merge worktree.
  - **No cross-project lock, lease or queue exists on the laptop.**
- **Collision.** Two projects' merge scopes are siblings in the laptop's app.slice at weight 100 each, so they split about 8/8 when
  both saturate. A solar-challenge scope is additionally capped at 8 threads (CPUQuota 800%, per scope, not per project).
  Memory: MemoryHigh sum for 4 concurrent merge verifies = 24+32+2+2 = 60 G, the laptop's RAM. MemoryMax sum = 124 G. A second merge
  verify on the laptop was measured at 1.5× contention (REPORT).
- **What could weight it.**
  - (i) The same per-project slices on the laptop. The laptop scopes are spawned by the same `verify.py::_scope_launch_argv`, so
    3394's `--slice=` applies there too. Slice weights would then be set on the laptop's user manager, with the same
    work-conserving semantics and the same blind spots, on 16 threads.
  - (ii) A shared cross-project host lease or slot.
  - (iii) Nothing: leave it to the per-scope weight of 100.
- **The ruling.** `plans/cpu-load-robust-verify-prd.md` §6:
  > "Host-global cross-project verify admission semaphore (shared slots dir) — TRIED-AND-REJECTED, not merely deferred: under a single
  > fair global semaphore, Reify's very long (30min+ cargo) verifies starved dark_factory almost entirely. The correct shape is
  > per-project admission (current design) + per-project off-hot-path serialized non-merge-blocking lanes, NOT a shared global gate.
  > Do not revisit this as an option."

  `plans/remote-task-leg-verify-prd.md` restates it among Leo's 2026-10-09 rulings: "never build host-wide verify admission across
  projects (RED-TIER …); … a BINARY pressure hold in the runner's health probe is the one allowed load-shaped lever."
  - **Inside the ruling:** (ii) in any form. A weighted cross-project token still lets a 30-min cargo verify hold its token, and
    holding time is exactly the stated ground. So is any shared cross-project gate on verify starts on either host.
  - **Outside the ruling:**
    - kernel weights (i), and the workstation slices. They admit nothing and hold no turn, and the starved project always keeps its
      runnable share.
    - Memory protection.
    - The binary remote pressure hold (decision 11, filed deferred).
  - **Ambiguous (needs Leo):** a cross-project dispatch coordinator, meaning one scheduler-level weight over agent starts. It is not
    verify admission, but it is the same "shared global gate" shape one gate earlier.

### M5. Feedback and failure modes [code; behaviour is inference]

- **One signal, many readers.** All projects compare the same host files against the same memory and io thresholds, so they trip
  in the same instant.
  - The project with the lowest floor backs off first. Today that is the two solar projects (floor 1, code default), before DF and
    reify (floor 3). The runqueue arm is armed only in DF and reify.
  - Nothing is weight-aware. A low-weight project's load holds a high-weight project's dispatch exactly as much as the reverse.
    Under weights that is undesired.
  - The own-slice CPU arm (built, off, waiting on 3394 and the ε2 calibration) is the weight-consistent form. A starved low-weight
    project's own `cpu.pressure` rises, so it holds and the favoured one does not. There is no own-memory arm.
- **Slot counts never derive from live load.** `plans/verify-oversubscription-control-prd.md` §3:
  > "C-no-load-derived-count: N and `-n` are fixed config, never derived from instantaneous loadavg (reify positive-feedback-collapse lesson)."

  The lesson is in `plans/occt-throttle-layer-2-3.md`:
  > "The wrapper's load-based formula `N = clamp(nproc - load_1m_int, 1, MAX_CAP)` collapsed N to 1 under sibling-worktree load
  > because it measured run-queue length, which is itself fed by the work being throttled. Every concurrent wrapper then contended
  > for slot-1 alone even when slots 2..MAX_CAP were idle."

  Kernel weights do not fall under this rule, because they are static values the kernel applies. A controller that recomputes
  weights from live load would reproduce the loop, with each project's weight driven by stall that its own weight causes.
- **Starvation of a weight-1 project.** This is covered by M2 limitation 4. Its tests time out before its share recovers, and the
  per-test thread-method kill turns that into worker crashes until task 5529 (which depends on 5089) lands. Weights should not be
  pushed far below the share at which a project's merge verify fits inside 7200 s.
- **Halted schedulers resuming together.** `Scheduler.resume` only clears the pause flag and the blocked-transition window.
  - There is no ramp. The hold is evaluated per selection with `in_flight` read before the dispatch, so a resumed project with fewer
    than `min_inflight_floor` in flight is never held. Reify can start 3 heavy tasks, and solar 1 each, regardless of pressure,
    before PSI applies.
  - After that, every project sees the same lagging avg10 signal. While it is low, all dispatch. Agents and verifies ramp memory
    over minutes, then the signal crosses 3 and every project holds at once. No hysteresis (D4) means the next selection re-opens
    as soon as avg10 dips.
  - Weights do not damp this: they are work-conserving and act on CPU only, while memory caused 57 of 69 holds.
  - The resume burst is bounded by `max_concurrent_tasks` per project and by the account caps, not by any cross-project mechanism.
  - Inference from `_phase_psi_gate` and `_phase_select_scored`; one task is returned per selection pass.

## Table: resource × mechanism

| resource | mechanism | guarantees | cannot | precondition | basis |
|---|---|---|---|---|---|
| workstation CPU | per-project slice `CPUWeight` (df-<p>.slice under df.slice) | proportional share among *runnable* project slices at saturation; role weights compose inside | cap; act when idle; reach sccache/jcodemunch/shared daemons in app.slice; protect a critical thread from its own project's load; bound memory/IO; stop 5529 timeouts | 3394 (topology), intra-slice values for daemon/agents (unowned), sccache placement (unowned) | semantics [doc]; tree [host] |
| workstation CPU | `CPUWeight` on orchestrator-<p>.service today | shifts daemon+agents only | touch verify scopes (about 70% of reify W0 CPU) or sccache compiles | none | [host][code] |
| workstation CPU | df.slice vs app.slice top-level weight | factory share against desktop/interactive | separate projects; interactive T3 is already at w10 inside app.slice | 3394; a stated policy (3394 amendment 4b) | [doc][host] |
| workstation CPU | `CPUQuota`/`cpu.max` | hard ceiling per project | work-conserve; avoid throttle latency | none (systemd property) | [doc] |
| workstation CPU | `cpu.idle` | near-strict "only when free" for one group | ratios; avoid starvation timeouts | 3394 | [doc] |
| workstation CPU | cpusets | hard partition | work-conserve | host change: delegate cpuset | [host] not delegated |
| workstation CPU | uclamp | nothing share-like | anything here (amd-pstate-epp, powersave) | — | [host] driver; effect [doc] |
| laptop CPU | per-project slice weights on the laptop user manager | same as workstation, 16 threads | same; plus solar per-scope CPUQuota already caps solar scopes | 3394's `--slice` reaches laptop spawns (same code path); laptop slice values | [code][host] |
| laptop CPU | shared cross-project host lease | serialisation | — | **inside RED-TIER ruling; forbidden** | ruling text |
| memory | per-slice `MemoryLow`/`Min` | a weighted protected working set (reclaim hits others first) | lower host PSI; affect the hold; stop kswapd/swap churn | 3394; values sized on anon not current (3394 note) | [doc]; recursiveprot [host] |
| memory | per-slice `MemoryHigh`/`Max`/`SwapMax` | ceilings; Max = kill | express a share; MemoryHigh throttling RAISES host PSI and holds every project | 3394 | [doc] |
| memory | dispatch hold on per-slice `memory.pressure` (new own-memory arm) | a project holds on its own memory stall | see root/kernel reclaim | 3394 + new arm (none filed) | [code] (absent) |
| memory | today's hold | host-wide pause of heavy dispatch at mem full avg10 ≥ 3 | attribute; weight; 33% of W1 minutes are over 3 | none | [code][host] |
| merge-verify hosts | HostAllocator per project (today) | one merge verify per host per project | arbitrate between projects | none | [code] |
| merge-verify hosts | kernel weights on each host | CPU split when merge verifies of several projects collide | order who goes first; bound time | 3394 on both hosts | [doc] |
| dispatch | per-project PSI hold (today) | each project backs off on host pressure | weight; who backs off first is set by floor (solar 1 < DF/reify 3), not importance | none | [code] |
| dispatch | own-slice CPU arm | holds follow own stall, consistent with weights | memory | 3394 + ε2 calibration (5220) | [code] built, off |

## Corrections to /tmp/team-gates-1112/admission-policy.md

1. "Nothing arbitrates CPU between projects except the kernel (cgroup weights, nice)" (settled questions, B5). **Nice arbitrates
   nothing between projects.** Each project's verify processes run in their own scopes and its agents in their own unit, and nice
   orders threads only inside one cgroup. This repo's own `config.py` comment on `verify_cgroup_cpu_weight_*` says the same.
   Cross-project arbitration today is cgroup weight alone, and it is by role, not by project.
2. G4 "role nice/ionice". **ionice is inert on this host**: every NVMe uses scheduler `none`, and the io controller is not delegated
   to the user manager. Only nice (intra-cgroup) and the cgroup CPUWeight act.
3. G9 "agent-side nice 10". This orders agents against their own orchestrator daemon only. It gives no priority against verify scopes
   or other projects.
4. "Merge wins … roughly 3:1 per scope." That is correct among the verify scopes, but every runnable app.slice sibling is in the same
   denominator: nine orchestrator units at 100, sccache 100, desktop scopes 100, t3code.service 1000. So the absolute share is lower
   than 100/(100+33+…) suggests.
5. Governor table is missing:
   - app-t3code.slice CPUWeight 10 (interactive T3 sessions) and t3code.service CPUWeight 1000 / MemoryLow 1.5G.
   - jcodemunch-watcher CPUWeight 30 / MemoryHigh 8G / Max 12G (164,786 high events).
   - Laptop solar-challenge CPUQuota=800% per scope.
   - Laptop memory limits: DF 24/40/32 G via yaml (task 8384 landed), reify 32/48/32 G via yaml, solar ×2 2/18/32 G via drop-ins.
   - sccache.service as a shared, unweighted CPU sink, about 0.86 cores mean of compiler children (F2).
6. U8 (how slices weight units against scopes): answered by F1/M1. They are flat siblings in app.slice. The units have no weight
   (100) and the scopes carry the role weight.
7. B5 "a DF priority task gets no protection from sibling-project load". This is true today. After 3394 with slice weights,
   protection is proportional at saturation only. It does not cover memory, IO, sccache-run compiles, or anything in app.slice.

## Unknowns

- U1. **Whose memory** caused the 57 holds. There is no per-cgroup memory series, and `verify_rss_total_bytes`/`verify_concurrency` read 0
  throughout W1. Settle by re-running `mechanisms/cg_delta.py 120` during a 06-16Z episode, or by adding per-slice `memory.pressure` to the sampler.
- U2. The share of host memory PSI owned by kernel reclaim threads. The lifetime root-minus-slices figure (39%) is an indication only (F5 rule).
- U3. Who set `app-t3code.slice` CPUWeight=10 and IOWeight=10 (a runtime `set-property`), and whether that is intended. It is the
  opposite of the 2026-08-03 premise for T3 sessions.
- U4. The laptop holds 153 distinct `/tmp/df-verify-slots-1000-*` dirs, each with only `slot-1`, newest every 15-20 min. If
  `project_root` differs per remote run, the laptop-side n=1 flock does not serialise DF's remote task legs. Not resolved; needs the
  RemoteRunner ssh argv and the laptop config path.
- U5. Whether each acquire_next pass returns at most one task (so a resumed project hits the PSI floor after 3 or 1 dispatches) was
  inferred from `_phase_select_scored`. The harness's call loop was not read.
- U6. Whether a held dispatch at in_flight 18 (= `max_concurrent_tasks`) could have dispatched at all (admission-policy U6).
- U7. The per-CPU share figures (91% / 39%) follow `calc_group_shares` as documented. They were not measured under EEVDF on 7.0.
- U8. The non-factory top-level load figure (about 9.3 cores) comes from capacity F9 and includes factory-shared daemons (sccache,
  jcodemunch, dashboard, fused-memory) that would stay in app.slice after 3394. Their W1 split was not re-derived.
