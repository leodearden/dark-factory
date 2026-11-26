# Host map: what competes for the workstation and the laptop (seat: HOST MAP)

Measured 2026-11-12 ~15:15-15:40Z. Code read: `/home/leo/src/dark-factory` main at `7848173a7d`. The
running Dark Factory service started 2026-11-10T23:48Z and the other eight units 2026-11-10T20:57-21:01Z, so
code behind a unit may lag main. Windows: W0 = 2026-10-25T00:00Z..2026-11-08T16:00Z (14.67 d), W1 =
2026-11-09T14:13Z..2026-11-12T15:20Z (3.05 d). Scripts and raw outputs: `/tmp/team-weights-1112/host-map/`
(`activity.py/.out`, `accounts.py/.out`, `runners.py/.out`, `psi.py/.out`, `cpu_snap.sh`).

## Headline

Nine units, not four, but not nine active projects. Active in W1: Dark Factory, scoping-intake and stream-touch.
scoping-intake and stream-touch were never paused and each lands about 36 tasks/day, the same as Dark Factory
(39.7/day). Dormant: autopilot-video and know-live (empty queues), pump-web-ui (halted by cost ceiling since
2026-11-02). reify, solar-challenge and solar-challenge-platform are the three operator-paused ones. All nine
draw agent accounts from one pool (Dark Factory's file), including the two solar projects, via the package default.

## Findings

F1. Of nine units, three were active in W1 (Dark Factory, scoping-intake, stream-touch); two are dormant by empty queue, one is cost-ceiling halted, three operator-paused.
    figure: done tasks/day W1: DF 39.7, scoping-intake 36.1, stream-touch 36.1; all other six 0.0
    rule: distinct task_id with `task_results.outcome='done'` and `completed_at` in window, / window days
    window: W1 2026-11-09T14:13Z..2026-11-12T15:20Z    population: 9 projects, each its own data/orchestrator/runs.db
    kind: ratio (done tasks to window days)
    evidence: /tmp/team-weights-1112/host-map/activity.py (output activity.out); re-ran the DF/scoping-intake/stream-touch query: 121, 110, 110 tasks = 39.67, 36.07, 36.07 per day

F2. In W0 the same count gives 139.4 tasks/day across nine projects; the three operator-paused projects were 63.0 of it (reify 16.5, solar-challenge 20.8, platform 25.7); scoping-intake (36.2) and stream-touch (14.1) were another 50.3, DF 21.95.
    figure: 139.4 tasks/day total in W0 (63.0 + 50.3 + 21.95 + pump 3.89 + ~0.2 other)
    rule: as F1
    window: W0    population: 9 projects
    kind: ratio
    evidence: activity.out; the lead's "63 a day" for the three paused projects reproduces (16.5+20.8+25.7=63.0)

F3. scoping-intake and stream-touch have almost no queue left, so their demand is finite.
    figure: pending tasks scoping-intake 22 (+1 in-progress), stream-touch 9 (+2 in-progress, 9 deferred), autopilot-video 0, know-live 0 pending (10 deferred), pump-web-ui 54 pending
    rule: `select status,count(*) from tasks group by status` in each checkout's `.taskmaster/tasks/tasks.db`, read-only
    window: snapshot 2026-11-12T15:3xZ    population: 9 stores
    kind: measured total
    evidence: sqlite3 -readonly <project>/.taskmaster/tasks/tasks.db (DF 2898 pending, reify 2529, solar-challenge 165, platform 562)

F4. Verify scopes are SIBLINGS of the orchestrator units under `app.slice`, not children; agents and unscoped verifies sit INSIDE the unit's own cgroup.
    figure: 0 cgroup children under `orchestrator-dark-factory.service` (find -maxdepth 2 shows only itself); 5 live `df-verify-dark-factory-c889f090-*.scope` directly in app.slice (weights seen 100, 33, 10)
    rule: `find <unit cgroup> -maxdepth 2 -type d`; `systemctl --user show <scope> -p ControlGroup,CPUWeight`
    window: snapshot 15:2xZ    population: DF unit + its scopes
    kind: measured total
    evidence: /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/; code `orchestrator/src/orchestrator/verify.py::_scope_launch_argv` (systemd-run --user --scope -p CPUWeight=N, no --slice), `verify.py::_verify_scope_name` (df-verify-{scope_tag}-{uuid}.scope)

F5. The per-kind scope weights are config defaults, same in every project, and only four projects use scopes at all.
    figure: defaults merge 100 / task 33 / background 10; `verify_use_cgroup_scope` default False, set true only in DF, reify, solar-challenge, solar-challenge-platform
    rule: `orchestrator/src/orchestrator/config.py` Field defaults `verify_use_cgroup_scope`, `verify_cgroup_cpu_weight_{merge,task,background}`; grep of each project's dark-factory-orchestrator.yaml
    kind: bound (config reading, not a measurement)
    evidence: scoping-intake, stream-touch, autopilot-video, know-live, pump-web-ui do not set it, so their verifies run inside the orchestrator unit's cgroup (one entity, weight 100 against every sibling)

F6. No project-level weight exists anywhere: all nine orchestrator units have `CPUWeight=[not set]` (kernel default 100), IOWeight unset, MemoryLow 0, MemoryHigh and MemoryMax infinity; no drop-in sets one.
    figure: 9 of 9 units unset; drop-ins present only `orchestrator-reify.service.d/warm-lane.conf` (ordering and one Environment line) and `orchestrator-watchdog.service.d/72h-restart-cadence.conf`
    rule: `systemctl --user show` per unit, grep of `~/.config/systemd/user` excluding .bak
    kind: measured total
    evidence: unit table below; grep -rnE 'CPUWeight|IOWeight|Memory(High|Max|Low)' ~/.config/systemd/user

F7. Weights that DO exist on the host: `app-t3code.slice` CPUWeight=10 and IOWeight=10 (runtime `systemctl set-property` drop-in under /run/user/1000/systemd/user.control, not persistent), which holds all interactive T3 Claude sessions (`t3code-claude-*.scope`); `t3code.service` 1000; `jcodemunch-watcher.service` 30 (MemoryHigh 8G, MemoryMax 12G); `background.slice` 30; app-t3code-agents.slice MemoryHigh 128,523,751,424 B (~119.7 GiB), MemoryMax 130,671,235,072 B.
    figure: interactive T3 Claude sessions (11 live scopes) weigh 10 against 100 for each orchestrator unit
    rule: `systemctl --user cat app-t3code.slice`, `systemctl --user show` on scopes
    kind: measured total
    evidence: files in /run/user/1000/systemd/user.control/app-t3code.slice.d/. Contradicts the "interactive deliberately favoured" ruling recorded in task 3394 as far as T3-hosted sessions go; Konsole/GNOME-terminal sessions are unweighted (100) under app.slice. Whether the set-property was deliberate is unknown.

F8. Everything is a flat sibling set under `app.slice` (weight 100): nine orchestrator units, DF verify scopes, fused-memory, dashboard, sccache, both jobservers, browsers, terminals. docker (falkordb, qdrant, timescale containers, openclaw sandbox) sits under `/system.slice/docker.service`, weight unset; `user.slice` and `system.slice` are both weight 100.
    figure: live cores at one 30 s sample: DF verify scope 13.69 + 1.27, stream-touch unit 4.35, DF unit 2.45, Chrome 1.47, dashboard 0.68, jcodemunch-watcher 0.64, T3 slice 0.39
    rule: cpu.stat usage_usec delta over 30 s / seconds, at 15:2xZ, one sample (stream-touch was 0.04 cores in a 20 s re-sample, so it is bursty)
    kind: measured total (single snapshot, not an average)
    evidence: cpu_snap.sh output; /proc/pressure/cpu some avg10=23.18 at the time, load average 59

F9. Average CPU per unit since start (daemon plus its agents plus unscoped verifies; EXCLUDES the df-verify scopes, which are separate cgroups).
    figure: DF 3.24 cores, reify 0.60, scoping-intake 0.50, stream-touch 0.30, solar-challenge 0.05, autopilot-video 0.03, know-live 0.03, platform 0.03, pump-web-ui 0.03
    rule: `CPUUsageNSec` / seconds since ActiveEnterTimestamp (~39.8-42.6 h, mostly W1)
    window: 2026-11-10T20:57Z..2026-11-12T15:37Z    kind: ratio (cpu seconds to wall seconds)
    evidence: systemctl --user show orchestrator-*.service; DF's verify CPU is not in it: per-project verify CPU cannot be recovered from live cgroups because scopes are transient (`--collect`)

F10. Reify's unit burns 0.60 cores while halted.
    figure: 92,612 core-seconds over 42.6 h, 3 invocations in W1
    rule: as F9    kind: ratio
    evidence: the load is not agent work (invocations W1 = 3, cost 2 USD); likely offline lane and warm-lane GC (`reify-warm-lane-gc.service` 0.10 cores in the sample). Cause not established.

F11. Accounts: ONE shared pool for all nine projects. solar-challenge and solar-challenge-platform DO share Dark Factory's pool.
    figure: both solar projects invoked max-b, max-c, max-d, max-e, max-f, max-g, max-h in W0 (the same names as DF); neither checkout has config/usage-accounts.yaml and neither config has a usage_cap block
    rule: path resolution: no `USAGE_ACCOUNTS_FILE` in any unit, drop-in, DF `.env` (grep count 0) or the two daemons' /proc environ; the package default `orchestrator/src/orchestrator/defaults.yaml` `usage_cap.accounts_file: "${DARK_FACTORY_ROOT:/home/leo/src/dark-factory}/config/usage-accounts.yaml"` applies; invocations table agrees
    kind: measured total (path from code; usage from invocations)
    evidence: resolved file: /home/leo/src/dark-factory/config/usage-accounts.yaml (names in it: max-d, max-e, max-f, max-g, max-b, max-c, max-h, max-a). DF config sets it explicitly (default of `${USAGE_ACCOUNTS_FILE:...}`); reify config names `/home/leo/src/reify/config/usage-accounts.yaml`, a symlink whose `readlink -f` is DF's file (re-checked); pump-web-ui names DF's file. autopilot-video, know-live, scoping-intake, stream-touch have no usage_cap block, so the package default applies as well. Per-process accounting is per process (each unit's own UsageGate with its own in-memory cap state; only the roster is shared). Not established: whether max-a is excluded from automation (pump-web-ui config comment says "A reserved for interactive"; no invocations name it).

F12. The pool is shared in use as well as in roster. W1 invocations by account: DF max-h 459, max-f 456, max-c 124, max-g 42, max-d 32; scoping-intake max-h 339, max-f 249; stream-touch max-h 427, max-f 26. In W1 max-b and max-e (the two with the most W0 use) have zero invocations in any project.
    figure: W1 cost (notional USD): DF 2,458, scoping-intake 907, stream-touch 708, all others about 0
    rule: `invocations` rows with completed_at in window, summed per project/account (names only)
    window: W1    kind: measured total
    evidence: accounts.out, activity.out. So scoping-intake and stream-touch take 39.6% of W1 notional spend (1,615 of 4,075).

F13. Halt state: three operator-paused, one cost-ceiling-halted, DF and four others not halted. Persisted rows in each `runs.db` table `scheduler_state`.
    figure: reify 2026-11-08T16:04:41Z, solar_challenge 16:04:45Z, solar_challenge_platform 16:04:48Z (pause_class `operator`); pump_web_ui 2026-11-02T23:07:50Z pause_class `cost_ceiling`, reason `cost_ceiling_orch_exceeded`; DF, autopilot-video, know-live, scoping-intake, stream-touch have no row
    rule: `select * from scheduler_state` read-only; a missing row means not persisted-paused. In-memory pause not visible here (a non-persisted class would not appear)
    kind: measured total
    evidence: pump-web-ui is halted for a reason separate from the 11-08 pause, and the solar-challenge project_id in the store is `solar_challenge`.

F14. Memory pressure on the workstation fell in W1 but is not low: `some` avg10 p90 7.5 -> 5.1, `full` p90 5.7 -> 3.8.
    figure: psi_mem_some_avg10 mean/p50/p90: W0 2.94 / 1.30 / 7.54, W1 2.27 / 1.01 / 5.05; psi_mem_full_avg10 mean/p50/p90: W0 2.04 / 1.02 / 5.74, W1 1.64 / 0.89 / 3.83; share of samples above 5: some 16.6% -> 10.1%, full 12.3% -> 6.0%; max 94.5 and 93.0 (some)
    rule: `samples.metric` in (psi_mem_some_avg10, psi_mem_full_avg10) with `leaf IS NULL`, ts epoch int in window, nearest-rank percentiles
    window: W0 n=245,441 rows; W1 n=51,257 rows    kind: ratio (distribution of 5-9 s samples)
    evidence: psi.py/psi.out. Current /proc/pressure/memory: some avg10 0.48, avg300 1.30; cpu some avg10 23.18, avg300 25.07; io some avg60 6.62. CPU some mean 47.5 -> 19.2.

F15. Host RAM 125.7 GiB (MemTotal 131,802,932 kB), MemAvailable 73.6 GiB; swap 1,875,000,300 kB (1.75 TiB on five 357.6 GiB NVMe partitions, equal priority), 50.6 GiB used. Memory is not a per-project partition: peak is per unit since start.
    figure: memory.current now (GiB): DF unit 29.5, stream-touch 2.4 (13.3 in the 15:2x sample), reify 3.0, scoping-intake 0.8; memory peak since 11-10 start (GiB): DF 65.2, scoping-intake 44.1, stream-touch 33.3, reify 24.5, autopilot-video 16.2; the unit peaks sum to 195 GiB against 125.7 GiB RAM
    rule: `MemoryCurrent`/`MemoryPeak` of each unit (excludes df-verify scopes, e.g. one DF scope at 12.6 GiB and 9 GiB)
    kind: measured total; the 195 GiB is a sum of non-simultaneous peaks (an upper bound, not an observation)
    evidence: systemctl --user show; /proc/meminfo; `swapon --show`

F16. Merge verifies by runner: the laptop takes work from FOUR projects; the five others verify locally only.
    figure: merge_verify count local/laptop and wall verify-hours, W0: DF 326/408 (240.9/126.7 h), reify 83/215 (106.2/102.3), solar-challenge 38/262 (11.9/26.6), platform 88/346 (30.6/55.9), scoping-intake 526/0 (79.9 h), stream-touch 200/0 (10.2 h), pump-web-ui 55/0 (1.5 h); W1: DF 92/103 (51.1/34.2 h), scoping-intake 110/0 (17.3 h), stream-touch 101/0 (8.5 h)
    rule: `events.event_type='merge_verify'`, `json_extract(data,'$.runner')`, sum of `duration_ms` as wall hours (NOT CPU hours; concurrency within a verify not counted); one 'history-leg' row (DF W0) omitted
    kind: measured total
    evidence: runners.py/runners.out. Task verifies (not merge) were not counted; the `workflow_verify`/`verify_admission_*` events exist but I did not split them by host.

F17. Laptop (leo-laptop, 16 threads, 60 GiB RAM, 136 GiB swap): per-project configs exist for DF, reify, solar-challenge, solar-challenge-platform (`~/.config/orchestrator/<proj>-laptop.yaml`); no CPUWeight, IOWeight or slice is set anywhere there beyond the per-scope weights the orchestrators pass (the live DF merge scope shows 100); only memory drop-ins exist.
    figure: scope memory drop-ins (MemoryHigh/Max/SwapMax GiB): DF 24/40/32 (via config `verify_cgroup_memory`), reify 32/48/32, solar-challenge 2/18/32, platform 2/18/32; `app.slice` CPUWeight 100; load average 16.06 at 15:32Z, cpu some avg10 3.64, memory some 0.00; one live scope (DF merge verify, 14.96 GB current, 5.64e12 ns CPU)
    rule: `ssh leo-laptop` cheap reads: `systemctl --user show`, grep of `~/.config/systemd/user` (no .bak), `/proc/pressure/*`
    kind: measured total (snapshot)
    evidence: the laptop is also Leo's desktop (5 users, Firefox, JetBrains Toolbox, T3 Code, Signal in app-*.scope at weight unset), so interactive use is another unweighted competitor there. Task-verify use of the laptop (admission slots `verify_admission_task_slots` DF 1) was not measured.

## Unit table (one row per unit; CPU/memory per F9, F15; activity per F1/F2)

| unit | checkout | config (ExecStart --config) | max_concurrent_tasks (rule) | halted now | done/day W0 | done/day W1 | avg cores | mem cur/peak GiB | pending | state |
|---|---|---|---|---|---|---|---|---|---|---|
| dark-factory | /home/leo/src/dark-factory | dark-factory/dark-factory-orchestrator.yaml | 18 (explicit) | no | 21.95 | 39.7 | 3.24 | 29.5 / 65.2 | 2898 | active |
| reify | /home/leo/src/reify | reify/dark-factory-orchestrator.yaml | 16 (explicit) | yes, operator 11-08T16:04Z | 16.5 | 0.0 | 0.60 | 3.0 / 24.5 | 2529 | paused (product gate too) |
| my-solar-challenge | /home/leo/src/solar-challenge | solar-challenge/dark-factory-orchestrator.yaml | 24 (package default; config unset) | yes, operator 11-08T16:04Z | 20.8 | 0.0 | 0.05 | 0.4 / 3.4 | 165 | paused |
| solar-challenge-platform | /home/leo/src/solar-challenge-platform | .../dark-factory-orchestrator.yaml | 14 (explicit) | yes, operator 11-08T16:04Z | 25.7 | 0.0 | 0.03 | 0.2 / 4.8 | 562 | paused |
| autopilot-video | /home/leo/src/autopilot-video | .../dark-factory-orchestrator.yaml | 12 (explicit) | no | 0.07 | 0.0 | 0.03 | 0.1 / 16.2 | 0 | dormant (queue empty) |
| know-live | /home/leo/src/know-live | .../dark-factory-orchestrator.yaml | 48 (explicit) | no | 0.14 | 0.0 | 0.03 | 0.1 / 2.6 | 0 (10 deferred) | dormant |
| pump-web-ui | /home/leo/src/pump-web-ui | .../dark-factory-orchestrator.yaml | 24 (explicit) | yes, cost ceiling 11-02T23:07Z | 3.89 | 0.0 | 0.03 | 0.1 / 1.6 | 54 | halted, not by Leo's pause |
| scoping-intake | /home/leo/src/scoping-intake | .../dark-factory-orchestrator.yaml | 24 (package default) | no | 36.2 | 36.1 | 0.50 | 0.8 / 44.1 | 22 | active, nearly drained |
| stream-touch | /home/leo/src/stream-touch | .../dark-factory-orchestrator.yaml | 24 (package default) | no | 14.1 | 36.1 | 0.30 (burst 4.35) | 2.4 / 33.3 | 9 | active, nearly drained |

Common to all rows: unit slice `app.slice`, CPUWeight/IOWeight/MemoryLow/High/Max unset, NRestarts 0. Scheduler halt rule:
`runs.db.scheduler_state` row. `max_concurrent_tasks` default 24 from `orchestrator/src/orchestrator/defaults.yaml` (the
config.py Field default is 3; defaults.yaml is the layer that applies, which I read but did not run through the loader, so
the three "package default" cells are inferred from the file, not from a printed effective config). Accounts file: all nine
resolve to `/home/leo/src/dark-factory/config/usage-accounts.yaml` (F11). Own data path for each: `<checkout>/data/orchestrator/runs.db`
(confirmed present and recently written for all nine).

## Mechanisms

- A user-service orchestrator's agent subprocesses, agent-shell pytest and (when `verify_use_cgroup_scope` is false) its verifies all live
  in the single service cgroup; a weight on the unit would divide host CPU between these as one entity.
- With `verify_use_cgroup_scope: true`, each verify command runs in a transient `systemd-run --user --scope` named
  `df-verify-<project-tag>-<uuid>.scope` that lands in the caller's slice (`app.slice`), carrying CPUWeight 100/33/10 by role. The tag embeds
  the project (e.g. `dark-factory-c889f090`), so a per-project grouping can be derived from the name; the scope is `--collect`ed so its CPU total vanishes at exit.
- `cpu_governance` (reify config, `enabled: false`) would place an inner scope under `reify.slice/reify-governed.slice/reify-governed-agents.slice`; those slices exist but are idle (0 tasks), a leftover.
  Other idle leftovers: `dfplan5205probe.slice`, `reifyexp9698.slice.d`.
- Weights compare only within a parent: today the parent is `app.slice` for every project, so the cross-project split is the sum of entity weights
  each project has runnable at that moment (one unit + N scopes), not a project share. This is why a project that scopes its verifies gets more entities than one that does not.
- Halt persists in `RunStore.save_scheduler_pause` (`harness.py` pause path), cleared by `resume_scheduler`.

## Unknowns

- Per-project verify CPU (not memory) over a window: scopes are transient; I did not mine `own_cpu_some10` per-leaf rows or PCP archives. Per-unit CPU in F9 excludes scope CPU.
- Whether stream-touch's 4.35-core burst is a task verify or agent build; process list inside the unit showed cargo.
- Task-verify host split (local vs laptop) per project; merge verify only (F16).
- Effective (loaded) `max_concurrent_tasks` for the three defaulted units; read from YAML layers, not a runtime dump.
- Why `app-t3code.slice` has weight 10 and whether it survives reboot (it is in /run).
- Whether `max-a` is excluded from automation (no use in either window).
- The cost USD figures are notional (flat-rate Max accounts); no cap-hit share per project was computed here.
