# Measuring CPU and memory by consumer: what we record, what to add, and how hard it is

Follow-up to the 2026-11-12 cross-project weights study (`/tmp/team-weights-1112/REPORT.md` §2, §4, §8;
`mechanisms.md` F4, F5). Read-only seat, 2026-11-15 09:31Z to 10:45Z. Dark Factory `main` at `27c93d6f75`.
Scripts and raw outputs: `/tmp/team-weights-1112/followup-1115/measure/`.

How each statement is sourced: **[live]** read on the running host today; **[code]** read in the source (file and
symbol named); **[doc]** documented kernel or systemd behaviour that I did not measure; **[inference]** my
reasoning from the others.

## 0. The answer in brief

- **Most of what we would need is already counted by the kernel and partly recorded by systemd. Almost none of
  it reaches our own records.** Every cgroup on the workstation already has cumulative CPU time, current and peak
  memory split into program memory and file cache, swap, and stall time for CPU, memory and IO [live]. Systemd
  already writes one journal line per finished verify scope with its CPU time, and for about a quarter of them
  (mostly the long ones) its peak memory and peak swap [live]. Nothing joins those to a project, task or kind of
  verify, and the recorder (`sampler/`) reads none of them.
- **The recorder's two verify-memory series are broken by a one-line matching rule** that can never match on
  this host (section 1.2). The cause is known and already filed as task 7858, but the series measures the wrong
  quantity anyway: it sums process memory, which double-counts shared pages and leaves out file cache and swap.
- **Short live samples taken today point at who is involved in memory pressure, but they are a handful of
  minutes.** Near the hold threshold, the Dark Factory service (its agents and what they run) grew by 4.6 GiB of
  program memory in two minutes; in another 90-second window it took 80% of all swap-ins and 57% of the pages the
  kernel's background reclaim took. The shared code-index watcher (`jcodemunch-watcher.service`, a factory service
  charged to no project) churned constantly against its own 8 GiB memory ceiling and caused 84 to 97% of all pages
  read back after eviction. The kernel's own reclaim thread was using 65% of a core. Section 2.3.
- **The smallest step that answers "whose memory causes the dispatch holds"** is one task in the recorder: add a
  group of per-consumer memory counters (program memory, file cache, swap, the consumer's own memory stall, pages
  reclaimed from it, pages read back, swap-ins), and the reclaim threads' CPU, then join each recorded hold to the
  sample before it. The recorder runs from the main checkout every 5 seconds, so a merge deploys it at the next
  tick with no restart and no host change (section 6).

## 1. What is recorded today, and what is missing or broken

### 1.1 The host load recorder (`sampler/`)

What it is [live, code]. `dark-factory-load-sampler.timer` starts `dark-factory-load-sampler.service` every 5
seconds: `uv run --no-sync --project sampler python -m sampler` with `WorkingDirectory=/home/leo/src/dark-factory`.
Each tick is a fresh process (`sampler/src/sampler/__main__.py::main`). It writes `data/load-samples.db`, table
`samples(ts INTEGER epoch, metric, value, window_mean, window_max, leaf)`; 2.4 GB, oldest row 2026-10-16 (30-day
retention) [live: `measure/db_read.py`, output `db_read.out`].

What it records, from the one metric table `sampler/src/sampler/metrics.py::METRIC_GROUPS` [code, live]:

| Group | Metric | Grain | State today |
|---|---|---|---|
| Host pressure | `psi_{cpu,mem,io}_{some,full}_avg10` | host only | working |
| Process scan | `occt_queue_depth` | host | working (reify's OCCT gate) |
| Process scan | `verify_concurrency` | host | **0 in every row of the database** |
| Process scan | `verify_rss_total_bytes` | host | **0 in every row of the database** |
| Load | `runqueue_ratio`, `runqueue_read_ok` | host | working |
| Load | `own_cpu_some10`, `own_read_ok` | one row per `orchestrator-<project>.service` (or per `df-*.slice` once those exist) | working; CPU stall only |

Not recorded at all: any per-group memory figure (current, program memory, file cache, swap, peak), any
per-group memory or IO stall, any CPU *used* by any group, anything about the compiler cache, the shared
services, Docker, the desktop, or the kernel's reclaim threads. The recorder does not run on the laptop
(`~/src/dark-factory/data/load-samples.db` does not exist there) [live].

### 1.2 Why the two verify-memory series read zero

[code] `sampler/src/sampler/metrics.py::classify_process` marks a process as a verify only when
`os.path.basename(cmdline[0]) == 'verify.sh'`. On this host no verify process ever has that as its first
argument [live, process list of the six live verify scopes at 09:4xZ]:

- Reify's verify runs as `bash ./scripts/verify.sh test --scope branch --include-infra`. A script started through
  its `#!/usr/bin/env bash` line gets `bash` as its first argument, never the script name.
- Dark Factory's verifies are not `verify.sh` at all: `uv run --project shared pytest …`, `… pyright scripts/`,
  launched as `/bin/bash -c …` inside the scope (`orchestrator/src/orchestrator/verify.py::_run_cmd`).

So `count_verify_concurrency` and `sum_verify_rss` sum an empty set on every tick. Measured: 0 in every one of
roughly 17,000 rows a day from 2026-10-16 to today for both series [live, `db_read.out`]. The cause was found on
2026-11-05 and is written into task 7858 (section 5).

Even fixed as matching, the series would measure the wrong thing [inference, doc]: summing each process's
resident memory counts shared pages once per process (16 test workers sharing one Python), and leaves out the
file cache and swap that the kernel charges to the verify. The scope's own `memory.current`, `memory.stat` and
`memory.swap.current` are the right source, and they exist [live, below].

### 1.3 What each orchestrator records about its own verifies and agents

From the Dark Factory `runs.db` and a verify summary file [live, code]:

| Record | What it holds | CPU? | Memory? |
|---|---|---|---|
| `invocations` table / `invocation_end` event (one per agent run) | account, model, role, tokens, cost, duration, turns, success, timed out | no | no |
| `verify_admission_plan` event (task verifies) | per module leg: wait for a slot, time held, exit code, timed out | no | no |
| `merge_verify` event | runner (local or laptop), duration, pass, each unit's outcome | no | no |
| `workflow_verify` event | pass, runner, duration (0 in the sample read) | no | no |
| Verify summary file per attempt (`verify.py::_build_summary_payload`; archived for the budget census) | per command: duration, exit, timed out, slot wait, **host** load at start and end (CPU stall, run-queue ratio, load average), xdist worker count | no | no |
| `dispatch_deferred` event (each held dispatch) | which arm tripped, in-flight count, the host pressure sample, the service's own CPU stall | no | host only |
| `task_results` | total duration, agent runs, verify attempts | no | no |

The name of the verify scope is not written anywhere by the orchestrator [code: no log or event line in
`verify.py` mentions `scope_unit`], so even the journal records below cannot be joined to a task or a kind of
verify except by time.

### 1.4 What the kernel and systemd already account for these units

Controllers [live]: the root enables `cpuset cpu io memory hugetlb pids rdma misc dmem`; `user.slice` passes on
only `cpu memory pids`, and the same three are delegated to the user manager and to `app.slice`. So CPU and
memory accounting is on for every service and scope we own; **IO accounting is not** (no `io.stat` anywhere in the
user tree; `IOAccounting=no`, `IOReadBytes=[not set]` on the services and scopes). Stall files exist regardless of
controllers.

Files present, read on `orchestrator-dark-factory.service` and on a live `df-verify-*.scope` [live]:

| File | Service | Verify scope | Notes |
|---|---|---|---|
| `cpu.stat` (usage, user, system, nice) | yes | yes | cumulative; the DF service shows 417,658 CPU-seconds since its start |
| `memory.current`, `memory.peak` | yes | yes | peak is since the group was created: DF service 61.6 GB |
| `memory.stat` (program memory `anon`, file cache `file`, swap-ins, reclaim counts, refaults, major faults) | yes | yes | |
| `memory.swap.current`, `memory.swap.peak` | yes | yes | DF service peak swap 16.0 GB |
| `memory.pressure`, `cpu.pressure`, `io.pressure` | yes | yes | avg10/60/300 and a cumulative stall total in microseconds |
| `memory.events` (times over the soft ceiling, kills) | yes | yes | |
| `io.stat` | **no** | **no** | needs the io controller delegated (a root change) |

`systemctl --user show <unit>` reports `CPUUsageNSec`, `MemoryCurrent`, `MemoryPeak`, `MemorySwapPeak` for a
live unit [live].

**Systemd's end-of-unit record.** When a unit stops, systemd 255 (both hosts) writes a journal entry with message
id `ae8f7b866b0347b9af31fe1c80b127c0`, fields `USER_UNIT`, `CPU_USAGE_NSEC` and, when it could read them,
`MEMORY_PEAK` and `MEMORY_SWAP_PEAK` [live]. Read by message id it is fast (7 s for a day) [live,
`measure/journal_resources_24h.jsonl`]. Measured over the last 24 h:

| Host | Scopes recorded | CPU recorded | Peak memory recorded |
|---|---|---|---|
| Workstation, Dark Factory verify scopes | 1,754 | 195.5 core-hours | 450 of 1,846 verify scopes (both projects) |
| Workstation, reify verify scopes | 92 | 155.4 core-hours (largest peak 51.9 GiB) | |
| Laptop, Dark Factory | 1,319 | 146.3 core-hours | 163 |
| Laptop, reify | 50 | 114.9 core-hours | 22 |

- Only units that used at least about one CPU-second appear (smallest recorded: 1.003 s) [live].
- Peak memory is present for 81 to 88% of scopes that ran over 10 minutes of CPU, and for under 1% of those under
  10 seconds [live]. Why the rest lack it was not established; most likely systemd's read at stop loses a race
  with the scope's removal [inference].
- Retention: workstation journal back to 2026-10-25 (29.9 GB, size-capped); laptop to 2026-10-20 [live].
- Workstation verify CPU averaged about 14.6 cores over the last day, by hour 4 to 20 cores [live, same file].
- T3 Code already starts each interactive session in its own scope (`t3code-claude-<id>.scope`), so interactive
  sessions get these journal lines too [live].

**PCP.** The workstation's archives keep every process's resident memory and major faults every 5 minutes
(`/var/lib/pcp/config/pmlogger/config.default`, `proc.memory.rss`, `proc.psinfo.maj_flt`) [live]. Instance names
carry the pid and the first argument only (for example `[pytest-xdist` or
`/home/leo/src/dark-factory/.worktrees/5655/…/python`), so some processes can be tied to a project by path and
many cannot; no cgroup, swap or stall per process is kept [live, `measure/pcp_rss_0900.csv`]. Usable for a rough
look back at the 11-09 to 11-12 holds; not an attribution.

## 2. The consumers that matter for allocation decisions

### 2.1 The list, with where each sits and whether it can be separated

Workstation tree (all under `user@1000.service` unless marked) [live, 2026-11-15]:

| Consumer | Where it sits today | Separable from its neighbours? |
|---|---|---|
| Orchestrator daemon of project P | inside `app.slice/orchestrator-P.service` | no: shares the cgroup with P's agents and their children |
| Agent sessions of P | same service cgroup (23 `claude` processes in DF's at 09:4xZ) | no, except per process by sampling |
| Tool servers an agent starts (plan-tools and verdict servers, one `jcodemunch-mcp` per agent, `node`/`npm`) | same service cgroup (19, 12 and 8 processes in DF's) | no, except per process |
| Commands agents run from their shells (pytest, cargo) | same service cgroup (a `python -m pytest` seen in DF's) | no, except per process |
| Task, merge and background verifies of DF, reify and the two solar projects | one transient `app.slice/df-verify-<tag>-<id>.scope` **per command** (test, lint and type legs are separate scopes) | yes per scope; project from the name's tag; kind only from the CPU weight (100/33/10) or by time-joining to events |
| Verifies of scoping-intake, stream-touch, pump-web-ui, know-live, autopilot-video | inside their service cgroup (`verify_use_cgroup_scope` unset, per `host-map.md` F5) | no |
| Compiler cache for every project | `app.slice/sccache.service`; compilers run as its children | no (section 4e) |
| Code-index watcher | `app.slice/jcodemunch-watcher.service`, CPU weight 30, MemoryHigh 8 GiB, MemoryMax 12 GiB | yes, as one shared consumer |
| Memory service, dashboard, duplicate audit, reify warm-lane GC, legibility timers | their own units in `app.slice` | yes, each charged to no project |
| Graph and vector stores and other containers | `/system.slice/docker-<id>.scope` (outside the user tree) | yes per container |
| Interactive T3 sessions | `app.slice/app-t3code.slice/…/t3code-claude-<id>.scope`, weight 10 | yes per session |
| Terminal sessions | `app.slice/app-org.kde.konsole-<id>.scope` etc. | per terminal window only |
| Desktop apps | their own `app-*.scope` | yes |
| Kernel reclaim threads (`kswapd0`, `kcompactd0`) | root cgroup | their CPU from `/proc/<pid>/stat`; their stall counts in the host signal but in no group |

On the laptop the same per-command `df-verify-*.scope` placement applies, with memory limits on each scope
[live; `mechanisms.md` F6]; the desktop and interactive sessions there are unweighted.

### 2.2 Per task

Possible only for verifies, and only by adding a name or a record: today a scope name carries project and a random
id, not task, role or leg [code: `verify.py::_verify_scope_name`]. Agents are per task in our records
(`invocations.task_id`) but not in any cgroup. A cgroup per agent session (section 4d) would make agents per task
too.

### 2.3 Two live samples (whose memory, today)

Four short windows. W1 09:51:35 to 09:53:35Z (`cg_consumers_1.out`, `cg_consumers_1_growth.out`): host memory
"full" stall 3.8% over the window, just over the hold threshold of 3. W2 09:58:19 to 09:59:19Z
(`reclaim_probe_1.out`). W3 10:00:57 to 10:02:27Z (`reclaim_by_group_1.out`): host avg10 "full" 7.3 at the start.
W4 10:27:31 to 10:28:46Z (`reclaim_by_group_2.out`): a quieter host, avg10 0.6. Dark Factory held dispatch on
memory at 09:42 to 09:43 and at 10:23:59, so no hold fell inside any window. [live]

| | Near-threshold windows (W1 to W3) | Quiet (W4) |
|---|---|---|
| Grew most in program memory | W1: DF service +4.6 GiB in 120 s (next: code-index watcher +0.6) | not measured |
| Pages taken by background reclaim | W3: DF service 57%, a reify verify scope 29% | none from any group |
| Swap-ins | W3: DF service 80% (W1: DF service 172 of 2,728 pages/s host-wide) | reify service 27%, dashboard 21%, code-index watcher 14% |
| Pages read back after eviction | W3: code-index watcher 84% | code-index watcher 97% |
| Reclaim forced inside the group (ceiling or direct) | W3: code-index watcher 80% | code-index watcher 100% |
| Groups' own memory "full" stall | W3: reify verify 2.7%, code-index watcher 2.1%, DF service 0.03%; W1: watcher 1.5%, reify service 1.0% | code-index watcher 1.9% |
| `kswapd0` / `kcompactd0` CPU | W2: 65% / 16% of one core | |

The windows do not agree with each other on swap-ins (DF service 6% of the host in W1, 80% in W3), which is itself
the point: one sample per study cannot settle this, and a continuous per-group series can.

What this suggests, and what it does not [inference]:
- Three roles have to be told apart: the group that **grew** (here DF's agents), the groups whose pages were
  **taken** (DF, reify's verify), and the groups that **stalled** (reify's verify, the watcher). The DF service
  barely stalled although it was the main grower and the main victim of reclaim.
- The code-index watcher has hit its 8 GiB ceiling about 8.5 million times since its restart on 11-14 14:36
  (roughly 125 times a second), and re-reads about 120,000 file pages a second whatever the host is doing. On
  11-12 its count was 164,786 [`mechanisms.md` F3]. It is a shared service, so its stall belongs to no project.
- The kernel counts a CPU as memory-"full" when the only thing running on it is stalled reclaim, so
  `kswapd0` at 65% of a core feeds the host "full" figure that the hold reads by itself [doc: the psi
  memory-full rule for reclaiming tasks; not measured]. The 11-12 study put the root's lifetime share of memory
  stall at an indicative 39% (`mechanisms.md` F5).
- A group's stall share cannot be added up to the host figure: each group's figure is weighted by its own busy
  time [doc]. So "whose stall makes up the hold" has no exact answer; who grew and whose pages were taken does.

Two windows are not a pattern. Holds also now come from more than one project: since 2026-11-12 14:40Z,
stream-touch recorded 313 memory holds, Dark Factory 60 and reify 29 (Dark Factory also 37 run-queue holds)
[live, the nine `runs.db`]. Reify is resumed (no paused row in its `scheduler_state`) and its verifies were
running today [live].

## 3. What to record

### 3.1 The counters, and which question each answers

| Counter (cgroup v2 file) | Role | Decision it enables that we cannot make today |
|---|---|---|
| Cumulative CPU time (`cpu.stat` usage, user, system) | used | core-hours per project and per kind of verify; whether a CPU weight changed anyone's share |
| Current memory split into program memory and file cache (`memory.stat` `anon`, `file`), and growth over the window | used / **grew** | which project's growth is driving reclaim when pressure rises (the hold question); sizing any memory protection against program memory, not file cache (task 3394 already warns of this) |
| Swap held (`memory.swap.current`) and swap-ins (`memory.stat` `pswpin`) | used / suffered | whose working set is being swapped back in (the daytime 19k to 35k swap-ins a second in F5) |
| Pages taken by background and forced reclaim (`pgsteal_kswapd`, `pgsteal_direct`) | **victim** | whose pages pay for the pressure; what kernel reclaim is working on |
| Pages read back after eviction (`workingset_refault_anon`, `_file`) | suffered | thrash, such as the code-index watcher above |
| Peak memory and peak swap per run (`memory.peak`, `memory.swap.peak`) | used | what a verify of each kind costs in memory per project; memory limits on the laptop |
| Memory, CPU and IO stall totals per group (`*.pressure` `total=`) | **stalled** | which project's work was actually slowed, by what; whether a weight helped the favoured project |
| Times over a memory ceiling (`memory.events` `high`, `max`, `oom_kill`) | throttled | whether a ceiling is causing stall (a ceiling's throttling counts as memory stall, `mechanisms.md` M3) |
| Reclaim threads' CPU (`/proc/<kswapd pid>/stat`) | nobody's | how much of the host signal is the kernel itself |
| IO bytes (`io.stat`) | used | **not available** without delegating the io controller to the user manager (a root host change); IO stall is available without it |

"Who used it" and "who grew" decide **allocation** (which project costs what; which one to protect or hold).
"Who was stalled" decides **effect** (did a weight or a hold help the work we care about). "Whose pages were
taken" connects the two. Choosing which project to hold under memory pressure needs "who grew" and "whose pages
were taken"; judging whether a CPU weight worked needs "CPU used" and "CPU stalled" per project. Per the standing
rulings, these are for reports and for your decisions, not inputs to an automatic control.

### 3.2 Grain

- **One record per run, written when it ends**: per verify command (CPU, peak memory and swap, stall totals, the
  scope name, role, task, leg) and per agent session once agents have their own scopes. This answers cost
  questions exactly and is small (about 1,800 verify scopes a day on the workstation, 1,400 on the laptop).
- **A periodic sample per consumer**: growth, swap-ins, reclaim, stall, CPU, for each project's service, the sum of
  each project's verify scopes, each shared service, Docker, T3, terminals, the rest of the desktop, and the
  reclaim threads. This is the only grain that can be joined to a hold. 5 to 30 seconds is enough; holds are
  decided on 10-second averages and ticks are about 150 seconds apart (`plans/load-throttle-harmonisation-prd.md`
  §1).
- Store cumulative counters as they are read and let readers take differences, because each tick is a fresh
  process with no memory of the last [code: `sampler/__main__.py` docstring]; a counter going down marks a
  recreated group.

## 4. How difficult each piece is

Summary table first; detail follows.

| Piece | Size | Deploys by | Main risks | Checked live today |
|---|---|---|---|---|
| (a) Repair the verify-memory series | one small task (owned: 7858) | merge to main; next sampler tick | the history stays zero; the dashboard tiles read it; must follow scopes into `df.slice` after 3394 | cause confirmed on all six live scopes |
| (b) Per-consumer sampling in the recorder | one task (plus a hold-join report) | merge to main; next tick | database growth; must keep the one-table vocabulary rule | 218 groups read in 59 ms of CPU; counters all present |
| (c) A record per verify run | one task (owned: 6355, premise stale) plus one for the laptop | orchestrator restart (fleet redeploy) | the scope vanishes at exit; timeouts; keeping it out of the command output | journal already has CPU always and peak for 24% |
| (d) A cgroup per agent session | a few tasks, after 3394 | restart | moving agents changes CPU shares; sandbox; cost of creating scopes | T3 does it already; precedent wrapper inert |
| (e) Compiler cache by project | cannot be done exactly with one server; an estimate is one task | merge | a per-project server is PRD-sized and risky | compiler's working directory names the worktree |
| (f) The laptop | a task plus an operator step | restart / operator | carrying fields through the remote result | journal there has the same records |
| (g) Per-project slices (3394) | owned; pending | restart plus operator reinstall (9985) | | |

### (a) Repairing the verify-memory series

- Site: `sampler/src/sampler/metrics.py::classify_process`, `::count_verify_concurrency`, `::sum_verify_rss`,
  `::collect_process_metrics`.
- Better than fixing the argument match: count `df-verify-*.scope` cgroups and sum their `memory.current`, program
  memory and swap, per project tag. That is a directory glob plus three reads per live scope, cheaper than the
  current whole-process scan, and it measures what the kernel charges [inference]. The glob must cover both
  `app.slice/df-verify-*.scope` today and `df.slice/df-<id>.slice/df-verify-*.scope` after 3394 (3394's plan
  fixes the same glob in `scripts/verify_scope_peaks.py`, step 14, but not in the sampler [live, the plan file]).
- It still misses the five projects that verify inside their service.
- Size: one small task. Deploys at the next 5-second tick after the merge, because the timer runs the main
  checkout [live, unit `WorkingDirectory`]. The zeros since 2026-10-16 stay in the history; the dashboard's
  "Verify concurrency" and "Verify RSS total" tiles (`dashboard/src/dashboard/static/redux/tab_overview.jsx`)
  need to know the change date.

### (b) Per-consumer sampling in the recorder

- Site: a new metric group in `sampler/src/sampler/metrics.py::METRIC_GROUPS` with per-leaf metrics; discovery
  next to `::discover_pressure_cgroups` (which today finds only the nine services or the `df-*.slice`s). The table
  is the single place metric names are declared; `sampler/tests/test_vocabulary_one_table.py` enforces it.
- Leaves: one per consumer class (section 2.1), with verify scopes summed per project rather than one leaf per
  scope, so the number of series stays fixed (about 25).
- Overhead: reading all 218 groups on the host took 59 ms of CPU per snapshot in Python [live]; 25 to 40 groups
  would be well under that. The tick itself already costs about 0.3 CPU-seconds, mostly starting `uv` and Python
  [live, `CPUUsageNSec` of one tick].
- Volume: today about 43 rows a tick, 740,000 a day, 2.4 GB for 30 days. 25 consumers times 8 counters every tick
  would add 200 rows a tick, about five times the database. Every sixth tick (30 s) keeps it about equal
  [inference from the counts].
- Add the reclaim threads' CPU as two host metrics.
- Pair it with a read-only report script that joins every `dispatch_deferred` event in the nine `runs.db` files to
  the sample before it and ranks consumers by growth, pages taken and stall.
- Size: one task. No restart, no host change. Risk is low: the collector's failure handling already drops a
  failing group for the tick without touching the others (`sampler/src/sampler/__main__.py::_collect`).

### (c) A record per verify run

- Does the scope exist when the orchestrator could read it? **No, not reliably.** `systemd-run --scope` hands the
  command the scope; when the command's last process exits the scope stops and, with `--collect`, is removed
  [code: `verify.py::_scope_launch_argv`, `::_run_cmd`; doc]. `scripts/verify_scope_peaks.py` (task 5415) says the
  same and works around it by polling live scopes every 5 seconds.
- Can systemd report totals after exit? Partly: the journal line in section 1.4 always has CPU (for runs over
  about a second) and has peak memory and swap for most long runs. Systemd 255 offers nothing else after exit
  [live]. No stall totals are kept.
- Three ways to get the full record, cheapest first:
  1. Write the scope's name (and role, task, leg) into the verify summary's command entry
     (`verify.py::_build_summary_payload`, beside `load`). The journal can then be joined exactly, which gives
     CPU for every run and peak memory for most long ones, back to 2026-10-25, with no new reading.
  2. Read the counters from inside the scope just before it ends: the scope's shell already runs a short prefix
     before the command (`orchestrator/src/orchestrator/scope_launch_handshake.py::ScopeLaunchHandshake.payload`);
     a matching suffix can read `cpu.stat`, `memory.peak`, `memory.swap.peak` and the three stall totals of its own
     cgroup and write them to a separate descriptor or file, never to the command's output, which the failure
     classifiers parse. For timeouts and cancellations the kill path
     (`verify.py::_reap_command_subtree`) can read the same files before `_kill_cgroup_scope` [code; inference
     that this is sound].
  3. Poll while running, as `scripts/verify_scope_peaks.py` does; it loses the last seconds.
- Laptop: the remote merge path rebuilds its command entries from a hand-built dictionary
  (`_build_summary_payload` docstring), so new fields must also be carried through the remote result [code].
- Five projects run verifies unscoped; turning on `verify_use_cgroup_scope` for them is a one-line config each,
  but it changes their CPU shares (their task verifies drop to weight 33 against everything in `app.slice`), so it
  is a decision, not a free measurement change [inference from `mechanisms.md` M2].
- Size: one task for Dark Factory's own record, one for the remote carry. Deploys with the orchestrators. Task
  6355 already asks for this (section 5); its premise that Dark Factory verifies run unscoped is out of date
  (task 5206 enabled scopes).

### (d) A cgroup per agent session

- Site: the spawn prefix chain in `shared/src/shared/cli_invoke.py` (`_cpu_govern_prefix`, `_cpu_priority_prefix`).
  A precedent exists: `_cpu_govern_prefix` can already wrap an agent in reify's `cpu-governed-exec.sh`, which execs
  `systemd-run --user --scope`; it is inert because no project sets the path [code]. T3 Code runs each interactive
  session in its own scope on this host [live].
- Child processes: everything an agent starts (its shell commands, pytest, cargo, its tool servers) inherits the
  scope by fork and stays there unless it creates its own scope [doc]. So per-agent totals would include its tool
  servers; splitting those out needs per-process sampling inside the scope. Processes left behind after the agent
  exits stay in its scope, which also makes them easy to find and kill [inference].
- **It moves CPU shares.** A scope with no `--slice` lands in `app.slice` at weight 100 [doc; this is how verify
  scopes landed there]. Twenty agents leaving the DF service would become twenty siblings of the merge verify
  instead of sharing one weight-100 group. It has to be placed under the project's slice (after 3394) with a stated
  weight, and 3394's plan currently decides the opposite (agents inherit the service cgroup; no slice variable is
  passed to agents) [live, `.worktrees/3394/.task/plan.json` design decisions].
- Cost of creating scopes: the existing wrapper's docstring records two blocking D-Bus round-trips per launch
  (task 3147); not measured here. A few hundred agent launches a day is small [inference].
- Unknown: whether `systemd-run` can reach the user manager from inside the agent's Landlock sandbox, if the sandbox
  is applied before the wrapper. Not checked.
- Payoff beyond attribution: the journal would then record CPU and (for long ones) peak memory per agent session
  with no further code, if the scope name carries project, task and role.
- Size: a few tasks, after 3394; a PRD if reify's governed path (`cpu-governed-exec.sh`) is to be unified with it.

### (e) Compiler-cache work by project

- One `sccache` server (`app.slice/sccache.service`) runs every compiler as its own child, so the kernel charges
  all compiles to that one group [live: its children; `mechanisms.md` F2 measured about 0.86 cores of compiler
  CPU on average]. By cgroup it **cannot be attributed** while one server serves everyone.
- An estimate is possible: a live compiler's working directory names the worktree
  (`/home/leo/src/warm-lanes/worktrees/_merge-942c612d`, `/home/leo/src/<project>/.worktrees/<task>`) [live], so a
  per-tick scan of the server's children can split its CPU by project, missing compiles shorter than the interval
  and needing a map from warm-lane names to projects. One task.
- An exact split needs one server per project, each started inside its project's slice. That changes how the cache
  is shared on both hosts (task 1701's shared backend) and is a PRD-sized change with real risk to cache hit rates
  [inference].
- Recommended: record its total as a shared consumer now; add the estimate only if a decision needs it.

### (f) The laptop

- No recorder there. Its journal has the same end-of-scope lines (1,369 verify scopes, 261 core-hours in the last
  day) [live]. Per-run records arrive with (c) once the remote result carries them. Per-consumer sampling would
  need the recorder installed there as a user timer: an operator step on the host that is also your desktop.
- Its pressure was zero at the read (memory some 0.00, 53 GB available) [live]; the 11-12 study found the
  collisions there are between projects' merge verifies.
- Size: one task plus one operator step.

### (g) What task 3394's per-project slices give, and what they do not

Gives without any new code [live files exist for every slice; inference for the rest]:
- One cgroup per project holding its daemon, its agents, its in-service verifies and its verify scopes, so
  `cpu.stat`, `memory.current`/`memory.stat`/`memory.swap.current`, `memory.peak` (since the slice was created) and
  all three stall files are per-project totals. This finally makes a project's verify CPU recoverable
  (`host-map.md` F9 could not).
- The recorder's existing `own_cpu_some10` switches from the nine services to the slices automatically
  (`metrics.py::_DF_LEAF_GLOB`).

Does not give:
- Any recording of memory; the recorder still reads only CPU stall per leaf.
- A split inside the slice between daemon, agents and tool servers, or between agents of different tasks.
- The five unscoped projects' verifies as separate groups.
- The compiler cache, the code-index watcher, the memory service, the dashboard, Docker, the desktop, the reclaim
  threads.
- Laptop recording. Whether the laptop's verify scopes will also move into project slices was not checked
  (task 9985's text says verify scopes take `--slice=` from code; whether the remote leg runs that code path
  there is my inference, not verified). Either way nothing on the laptop reads them.
- Any change to what the hold reads (still host-wide pressure).
- It needs the operator reinstall and restart (task 9985) before the daemons move; until then only the verify
  scopes are in the slices.

## 5. Owners already in the store or in `plans/`

Read-only search of `/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db` over titles, descriptions and
details, all statuses (`measure/owners.py`, output `owners.out`), plus `plans/`.

| Task / doc | Status | What it covers | Relation to this report |
|---|---|---|---|
| 3394 Per-project slice topology | pending, high; architect plan written 2026-11-14, branch has no commits | one `df-<project_id>.slice` per project for daemon, agents (by inheritance) and verify scopes; fixes the glob in `verify_scope_peaks.py` | (g). Records nothing. Its plan rules out a slice variable for agents, which (d) would need |
| 9985 Operator: reinstall units and restart | pending, medium, after 3394 | moves the daemons into the slices | rollout of (g) |
| 9984 Re-derive the cross-project CPU split | pending, low, after 3394 | slice weights | a consumer of (b) and (c) data |
| 5220 Own-arm calibration gate | pending, medium | recommends a CPU own-stall threshold from 14 days of slice samples | CPU only; would read (b)'s data if memory were added, but that would be control, out of scope here |
| 3592 Load sampler: run-queue ratio and per-slice own pressure | done | the recorder's current shape | the base (b) extends |
| 3590 Dispatch gate v2 (run-queue and own-slice arms) | done | the hold's arms | the hold still reads host memory pressure |
| 6772, 6773 Sampler leaf column and one metric table | done | the structure a new group must follow | (b) |
| 7858 Seed-cost observability, part 2: `verify_concurrency` always 0 | pending, medium | fix or remove the verify series; cause written in on 2026-11-05 | **owns (a)** |
| 7006 reify verify-plan companion, item 2 | pending, low, waits on 8528 | also re-matches the sampler's verify classifier for reify's coming Python verify | **duplicates part of 7858**; should defer to it |
| 6355 Stamp per-leg CPU-seconds and peak memory on every verify summary | pending, low | per-run CPU and peak in `_build_summary_payload` | **owns (c)**; its premise that DF verifies are unscoped is stale since 5206 |
| 5415 Measure merge-verify peak memory | done | `scripts/verify_scope_peaks.py` poller | prior art for (c) option 3 |
| 5205, 5206 Per-role verify weights; DF verify scopes | done | the current verify placement | |
| 7112 Why per-role weights did not restore reify's merge latency | pending, low | an investigation | needs (b) and (c) data to settle |
| 6987 Host load card reads the sampler table | pending | dashboard | a reader of new metrics |
| 8384 Laptop verify memory limits in config | done | laptop scope memory limits | |
| `plans/load-throttle-harmonisation-prd.md` | the design behind 3590, 3592, 5219, 5220 | per-slice **CPU** stall only; states that own stall means "my work waits", not "I caused it" (§1) | no memory per slice, no per-run resources |
| `plans/verify-parallelism-trial-e1-report.md` | report | the one-off A/B that measured CPU-seconds and peak inside scopes | source of 6355 |

Owned by nobody: per-consumer memory sampling (b); reclaim-thread CPU and per-group reclaim; the journal's
end-of-unit records as a data source; cgroups per agent session (d); compiler-cache attribution (e); a recorder on
the laptop (f); the code-index watcher's constant churn against its ceiling.

## 6. The smallest first step, and the order for the rest

**First step (one task, no restart, no host change):** extend the recorder with a per-consumer memory group and
fold in the verify-series repair:
1. One leaf per consumer class: each `orchestrator-<project>.service` (later each `df-<project_id>.slice`), the
   sum of each project's verify scopes, `sccache`, `jcodemunch-watcher`, the memory service, the dashboard, T3
   Code, terminals, Docker containers, the rest of the desktop.
2. Per leaf: program memory, file cache, swap held, and cumulative swap-ins, pages taken by background and forced
   reclaim, pages read back, memory stall total, times over a ceiling, CPU used.
3. Host: `kswapd0` and `kcompactd0` CPU.
4. Replace `verify_concurrency` / `verify_rss_total_bytes` with the per-project verify-scope sums (closes 7858
   part 2; 7006 item 2 then has nothing left to do).
5. A read-only report script: for every memory hold in the nine `runs.db`, the consumers ranked by growth, pages
   taken and stall in the sample before it.

A week of daytime episodes then answers "whose memory causes the holds" by project and by shared service. While
it is being built, `measure/reclaim_by_group.py 90` and `measure/cg_consumers.py 120` can be run by hand during a
06:00 to 16:00Z episode for a spot answer.

**Then, in order:**
2. Re-scope 6355: write the scope name, role, task and leg into each verify command's summary entry (exact join
   to the journal, history back to 2026-10-25), then the in-scope counter read for full per-run CPU, peak and
   stall; carry them through the laptop's remote result.
3. Land 3394 and its operator rollout 9985; point the new group's leaves at the slices.
4. Decide whether the five unscoped projects should scope their verifies (a share change, so a decision).
5. Agent scopes under the project slice, named by project, task and role (needs an amendment to 3394's agent
   decision).
6. The recorder on the laptop (operator step).
7. Compiler cache: keep it as one shared consumer; add the per-project estimate only if a decision needs it.
8. IO bytes per group only if a decision needs them: it requires delegating the io controller (root).

## 7. What I could not determine

- Whose memory caused any particular past hold. No hold fell inside my two windows, and nothing per group was
  recorded before today. PCP's per-process memory could give a rough look back at 11-09 to 11-12 (section 1.4); I
  did not do it.
- How much of the host "full" signal is the reclaim threads. The psi rule that makes it count is documented, not
  measured.
- Why systemd's end-of-scope record lacks peak memory for three quarters of verify scopes.
- The cost of creating a scope per agent on this host, and whether the agent sandbox blocks `systemd-run`.
- Whether the remote laptop path archives per-command summaries in the same files as local verifies.
- Which kind of verify each journal record is (the scope name does not say).
- Whether the code-index watcher's churn adds meaningfully to the host signal, or only to its own stall.
- The 4.6 GiB growth in the DF service in window A: which process (agent, shell command or tool server) it was.
