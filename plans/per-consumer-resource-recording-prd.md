# PRD — Record CPU and memory use by consumer on the factory hosts

**Status:** active, authored 2026-11-16 in a `/team` + `/prd` session spawned from Leo's 2026-11-16 go. A critic
reviewed it, then re-checked the folded draft, and both rounds of findings are folded in. Leo ruled twice in the
session (D9). Ready for `/prd` decompose.
**Slug:** `per-consumer-resource-recording`. **Path:** `plans/per-consumer-resource-recording-prd.md`.
**Type:** greenfield recording work. It also re-scopes four existing tasks that already own parts of it (§8, "Store
edits"):

- task 7858, seed-cost observability part 2;
- task 7006 item 2;
- task 6355, stamp per-leg CPU and peak memory on verify summaries;
- task 9985, the operator reinstall after the per-project slices.

**Prerequisite (filed, mid-implementation):** task 3394, one cgroup slice per project (`df-<project_id>.slice`), and
its operator reinstall, task 9985. Only the agent split (leaves 6, 7a and 7b) waits on them.

> **Code anchors** verified against main `5f665206d8` (2026-11-16), with the critic re-reading at `ba39519b67`.
> Anchors cite symbols, so re-locate them at implementation time.
>
> **Figure tags.** Each figure is tagged **(measured)** with its read and window, **(bound)**, or **(projection)**.
>
> **Evidence.** The evidence files are untracked copies, so the figures this PRD rests on are restated here. The
> sources are:
> - the 2026-11-15 study, `plans/cross-project-weights-study-2026-11-12/followup-1115/measure.md`;
> - its parent, `plans/cross-project-weights-study-2026-11-12/REPORT.md`;
> - this session's checks in `/tmp/df-percons-prd-1116/` (`recorder.md`, `verify-path.md`, `premise.md`, `owners.md`,
>   `critic.md`, and the two `delegate_probe*.sh` probes; volatile).
>
> **Terms.**
> - A *cgroup* is the kernel's accounting group for a set of processes; every systemd service and scope has one.
> - *Program memory* is the kernel's `anon` figure: heap and stack, the part that has to be swapped to be freed.
> - *File cache* is file pages the kernel can drop and read back.
> - *Stall* is the kernel's pressure figure: time during which work in a group was waiting for CPU, memory or IO.
> - A *memory hold* is a dispatch the scheduler deferred because host memory pressure was over its threshold.
> - *Review-all stream N* names one of the refactoring streams the 2026-10-08 whole-project review filed, each with
>   its own PRD.
> - `uv` is the Python launcher each orchestrator unit starts; the daemon runs as its child.
> - An *inode* is the kernel's identity for a directory. A cgroup that is removed and recreated at the same path gets
>   a new inode.
> - *Landlock* and *bwrap* are the sandboxes an agent can be wrapped in.
> - `cgroup.subtree_control` is the file that says which controllers (CPU, memory, …) apply to a cgroup's children.

---

## 0. Goal and what Leo observes when it lands (G1)

**Goal.** Allocation decisions on the two factory hosts should rest on recorded figures, not one-off samples:

- which project or service causes the memory holds;
- what a verify run costs;
- what a project's agents cost.

A *consumer* is one of these:

- a project's orchestrator daemon;
- its agents, and each agent session;
- its verify runs, and each verify command;
- each shared service: the compiler cache, the memory service, the dashboard, the watchdog, the job servers, the
  account proxy, the quality snapshot, T3 Code with its interactive sessions, terminals, Docker containers, and the
  rest of the desktop;
- the kernel's own work.

**What Leo observes:**

1. **A per-consumer sample every 30 seconds.** Within a minute of the first leaf's merge, `data/load-samples.db`
   holds, for every consumer:
   - program memory, file cache and swap;
   - swap-ins, pages reclaimed from it, and pages read back after eviction;
   - stall;
   - times over a memory ceiling;
   - CPU used.

   It also holds the kernel reclaim threads' CPU.
2. **Two dashboard tiles that read real values.** The host load card's "Verify concurrency" and "Verify RSS total"
   have read 0 in every row of the 30-day store (measured). Two tiles read from the kernel's accounting replace them:
   live verify scopes and their memory.
3. **A hold report.** `scripts/resource_report.py holds --since 7d --summary` lists every memory hold in the nine
   projects' `runs.db` files. For each hold it shows:
   - which consumers grew and which lost pages or stalled in the interval before it;
   - the kernel's share;
   - what could not be attributed.

   It ends with a summary across the window. A hold that predates the recorder says so.
4. **A dated decision task.** A week after the hold report lands, the task brings Leo that summary as the answer to
   "whose memory causes the holds".
5. **A record per verify command, on either host.** Each run leaves one `verify_command_resources` event in its
   project's `runs.db`. The event carries:
   - CPU seconds, peak memory and peak swap;
   - times over a ceiling;
   - the three stall totals;
   - how the figures were captured;
   - the run's project, task, role, module, leg and host.

   The local verify summary and the remote pass summary carry the same record.
6. **A cgroup per agent attempt.** After the operator reinstall (task 9985), each orchestrator service holds:
   - a `daemon/` cgroup, holding the daemon and everything it runs that is not an agent;
   - one cgroup per agent attempt, holding the agent and everything it starts.

   Each agent attempt's `invocations` row carries that cgroup's CPU seconds, peak memory and stall. **CPU sharing does
   not change at all**, by Leo's ruling of 2026-11-16 (D9).
7. **A cost report.** `scripts/resource_report.py costs --since 7d` gives CPU seconds and peak memory over the
   window, on both hosts:
   - per project, verify role, leg and host;
   - per agent role and model.

   A source that has never been written says so.

**Consumers per mechanism (no orphans):**

| Mechanism | Consumer |
|---|---|
| Per-consumer sample group (leaf 1) | the hold report (leaf 2); task 9984 (re-derive the cross-project CPU split once the slice weights exist), which needs CPU used per project; task 7112 (why per-role CPU weights did not restore reify's merge latency), which needs per-group CPU used and stalled under load; Leo's open design for weights between projects (`REPORT.md` §9) |
| Verify-scope tiles (leaf 1) | the dashboard's host load card (`dashboard/src/dashboard/static/redux/tab_overview.jsx`), and task 6987 (the card renders the sampler table's served keys) |
| Hold report (leaf 2) | Leo, through the decision task (leaf 3) and by hand |
| Per-run verify record (leaves 4, 5) | the cost report (leaf 8); the budget census (`scripts/verify_budget_census.py`, which reads the local summaries); task 7616 (zero-module merges have no per-leg cost record); the laptop's verify memory limits (task 8384's `verify_cgroup_memory`, sized today from one poller run) |
| Delegated service subtree, agent cgroups and per-attempt readings (leaves 6, 7a, 7b) | the per-agent columns on `invocations`, the recorder's agents consumer (leaf 1's rules), and the cost report (leaf 8) |
| Cost report (leaf 8) | Leo; task 9984; task 7112 |

---

## 1. Background and measured premise (G6)

**Memory still holds dispatch, and nothing records whose memory it was.** This was re-measured on 2026-11-16
(`premise.md`).

*How holds were counted:* `dispatch_deferred` events in each project's `runs.db`, split by the tripped arm in
`data.metric`. One event is one scheduler pass that skipped a candidate; there is no deduplication.

*The windows:*
- **A** runs from 2026-11-10 to the config commit at 2026-11-15 10:32Z.
- **B** runs from that commit to the code-index watcher's stop at 2026-11-16 06:28Z. Dark Factory took the new
  config only at its restart at 11:00Z, so B's first 28 minutes ran the old config. The slot change at 14:46Z falls
  inside B.

| Project | Memory holds, A | Memory holds, B | Run-queue holds, A / B |
|---|---|---|---|
| Dark Factory | 115 (0.9 an hour) | 80 (4.0 an hour) | 37 / 21 |
| reify | 29 (1.2 an hour from its 11-14 resume) | 30 (1.5 an hour) | 19 / 16 |
| stream-touch | 2,356, none after 11-13 23:01Z | 0 | 0 |
| scoping-intake | 1,747, none after 11-12 10:15Z | 0 | 0 |
| the other five | 0 | 0 | 0 |

(measured) Host memory "full" pressure above the hold threshold of 3 appeared in 7% of 5-second samples on 11-12
and 11-13, and in 24% on 11-15 and 11-16 (measured, `load-samples.db`).

**Why only two projects hold.** The memory arm is on in every project by code default:
`orchestrator/src/orchestrator/config.py::PsiAdmissionConfig` has `enabled` true, memory "full" 3.0 and memory "some"
15.0, and no project's config turns it off. A hold is recorded only when a ready task is turned away, and since
2026-11-13 the other projects have dispatched nothing (measured 2026-11-16 from their `runs.db` and task stores):

- solar-challenge is under Leo's operator pause of 2026-11-08;
- pump-web-ui is paused on its cost ceiling since 2026-11-02;
- stream-touch and scoping-intake have started no task since 11-13 and 11-12;
- know-live and autopilot-video have nothing pending.

**Three things changed after the study:**

- **The code-index watcher** (`jcodemunch-watcher.service`) was stopped at 2026-11-16 06:28Z and is disabled. It
  caused most of the file-cache churn in the study's samples. It also ran on 11-12 to 11-14, when pressure was low,
  and the 42 minutes of data after its stop cannot separate its effect from hourly variation. It is dropped from
  the consumer list.
- **The Dark Factory config changed on 2026-11-15:** `max_concurrent_tasks` went from 24 to 36 and
  `verify_task_host_policy` to `remote_all`, then `verify_admission_task_slots` went from 2 to 3. Dark Factory's
  hold rate rose after these changes. Nothing recorded shows whose memory grew; that gap is this PRD's premise.
- **Leo revoked "merge verify is never throttled"** on 2026-11-15. It does not change what is recorded and is not
  cited as a constraint.

**What the kernel already counts, and nothing reads.** Every cgroup on both hosts has:

- `cpu.stat`;
- `memory.current`, `memory.peak` and `memory.stat`;
- `memory.swap.current` and `memory.swap.peak`;
- `memory.events`;
- the three pressure files, each with a cumulative `total=` stall time.

The root cgroup has only `cpu.stat`, `memory.stat` and the pressure files (measured 2026-11-16 on both hosts;
`premise.md` §4, `verify-path.md` §4.2, `critic.md` m3). Per-group IO bytes (`io.stat`) are absent on both hosts:
the io controller is not delegated to the user manager, and delegating it is a root change. The live picture at
07:11Z on 2026-11-16 (measured, `premise.md` §3):

| Group | Memory | Program memory / file cache | Swap |
|---|---|---|---|
| T3 Code and its sessions (`app-t3code.slice`) | 28.8 GiB | 3.4 / 22.0 GiB | 5.4 GiB |
| Dark Factory service (daemon, agents, tool servers, their commands) | 19.2 GiB | 11.2 / 7.2 GiB | 3.8 GiB |
| five Dark Factory verify scopes together | 26.0 GiB | 25.3 GiB program memory | 0.2 GiB |
| reify service | 8.4 GiB | 0.5 / 7.2 GiB | 0.5 GiB |
| charged to `app.slice` but to none of its live children: pages left by groups that have exited | 2.4 GiB (10:50Z; the user manager had 441 exited groups not yet freed) | — | — |

The kernel's background reclaim thread used 21% of a core over 30 seconds in a quiet sample, and 65% in the study's
busy one on 11-15 (measured, one window each).

**What is recorded today** (`recorder.md`, `verify-path.md`):

- **The recorder.** `sampler/` runs every 5 seconds from the main checkout and writes host pressure, run-queue ratio
  and per-service CPU stall to `data/load-samples.db`. Volume: 29 rows a tick, about 488,000 a day, 2.42 GB at its
  30-day retention (measured over the last 24 h; the study's 43 rows a tick was wrong). It reads no per-group memory
  and no CPU used.
- **The two zero series.** `verify_concurrency` and `verify_rss_total_bytes` are 0 in every row, because
  `sampler/src/sampler/metrics.py::classify_process` matches a first argument of `verify.sh`, which no verify on
  this host has. Even matched, they would sum per-process resident memory, which double-counts shared pages and
  misses file cache and swap. Their only readers are two dashboard tiles. The orchestrator never reads
  `load-samples.db`, and no gate or hold reads these series.
- **systemd's journal line per stopped verify scope.**
  - Scopes that used about a CPU-second or more get one; shorter scopes get none.
  - It carries CPU, but peak memory for only 24% of workstation and 10% of laptop Dark Factory scopes (measured,
    6 h to 07:10Z on 11-16).
  - It carries no task and no kind of verify: the scope name holds the project and a random id only
    (`orchestrator/src/orchestrator/verify.py::_verify_scope_name`).
  - The name cannot be extended, because six consumers pin its shape (`verify-path.md` §6).
- **The verify summary.**
  - It records no CPU or memory.
  - The archive keeps about 5 days in practice: it is over its 500 MB size cap (526 MB; oldest summary 11-11).
  - Remote green verifies leave only a pass summary, which the budget census cannot see.
  - A green local merge verify archives nothing.
- **The laptop** has no recorder.
- **Dark Factory's agents** all share the service's cgroup, with the daemon and everything it runs.

**The laptop now.** `remote_all` makes task verifies *eligible* for the laptop, not forced there: 19 of 60 Dark
Factory task-verify module runs in the 6 h to 07:10Z on 11-16 still ran on the workstation (measured, dispatch
sidecars).

The laptop is CPU-saturated: load 104 on 16 threads, CPU "some" stall 89% (measured 07:13Z). Its memory is fine:
24.5 GB available, memory stall 0. Verify scopes used about 12 of its 16 threads on average over the same 6 h (Dark
Factory 47.6 core-hours, reify 25.7; measured, journal CPU).

The laptop runs no scheduler, so it has no holds. Its factory load is mostly its verify scopes, which leaves 4 and 5
record. Gate builds, ssh sessions and scopes left behind on the laptop (task 7805) stay unrecorded (D11).

---

## 2. Sketch of approach

Four producers, two readers, one decision task.

1. **The recorder gains a consumer group** (leaf 1).
   - **Cadence.** Every 30 seconds.
   - **Walk.** It walks the whole cgroup tree from the root and assigns every *leaf* cgroup (one with no children)
     to exactly one consumer, through one ordered rule table.
   - **Residuals.** Every inner cgroup also contributes a *residual*: its own figures minus its live children's.
     The residual is what was charged to children that have since exited, plus the kernel's own work at the root.
     So consumers add up to the host, and short-lived groups are not lost.
   - **What it writes.** Gauges are written as sums. Cumulative kernel counters are written as increments over the
     interval, computed from per-cgroup state the recorder keeps between its 5-second runs.
   - **Host-level figures.** Two more figures feed the verify tiles: live verify scopes and their memory.
2. **The hold report** (leaf 2) joins each memory hold to the consumer samples before it.
3. **The decision task** (leaf 3) runs the hold report over the week after it lands and escalates the result to Leo.
4. **One record per verify command run** (leaves 4 and 5).
   - **Normal exit.** The command's own scope reads its counters after the command exits.
   - **Kill paths.** The kill paths read them before the kill.
   - **Carrying.** The record rides on the command's run result into the summary, and back from the laptop on the
     existing `VerifyResult` wire.
   - **Storing.** The dispatching orchestrator stores it once, as an event in `runs.db`.
5. **Agent attempts get their own cgroups inside their orchestrator's service** (leaves 6, 7a and 7b), recording
   only, by Leo's two rulings of 2026-11-16 (D9).
   - **The unit.** It delegates the memory controller of its subtree to the orchestrator and starts the whole process
     tree in a `daemon/` child (`Delegate=memory`, `DelegateSubgroup=daemon`).
   - **The daemon.** It turns on only the memory controller for the subtree, and creates one child per agent attempt.
   - **Joining.** The agent joins its child before it execs, through an outermost launch prefix.
   - **Reading.** When the attempt ends, its readings go on its `invocations` row.
   - **CPU sharing.** The CPU controller is not delegated or turned on inside the service, so CPU is shared exactly
     as today.
6. **The cost report** (leaf 8) reads the verify events and the agent rows.

Nothing in this PRD changes:

- what the dispatch hold reads, or its thresholds;
- CPU weights between projects or roles;
- slot counts;
- memory limits.

§6 contract C3 pins this.

---

## 3. Resolved design decisions

**D1 — Recording only.** Every series and record is for reports and Leo's decisions, never an input to an automatic
control, per Leo's standing ruling that slot counts and weights are never derived from live load. The orchestrator
already never reads `load-samples.db`, and this PRD adds no reader there.

**D2 — Members are leaf cgroups; every inner cgroup adds a residual; one ordered rule table names the consumer.**

The table, its walk and the consumer name live in one module, `sampler/src/sampler/consumers.py` (new). A consumer
is a structured value of a *role* and an optional *key*. Its string form in the store's `leaf` column
(`<role>@<key>`, or `<role>` alone) is produced and parsed by one pair of functions in that module, and the reports
import those functions; there is no second parser (heuristic 12).

The roles are `service`, `agents`, `verify`, `project`, `shared`, `interactive`, `system`, `other`, `exited` and
`kernel`. The keys are names the cgroups themselves carry, so no project registry is needed:

- **`service` and `agents`.** Keyed by the project slice `df-<project_id>.slice` when the service sits in one, and
  otherwise by the unit, for example `orchestrator-dark-factory.service`. That is exactly the last segment of the
  hold event's `own_cgroup` in either placement (C2). The key therefore changes once, when task 9985 moves a project
  into its slice, and the change is a real placement change, stated as such.
- **`verify`.** Keyed by the scope tag `<slug>-<hash8>`, derived from the project root by the one shared scope-tag
  rule (leaf 1).
- **`project`.** Keyed by the project slice.

*Anchors* are a fixed list of cgroups: the root, `system.slice`, `user.slice`, `user-<uid>.slice`,
`user@<uid>.service`, `app.slice`, `session.slice` and `background.slice`.

Rules, applied in order to each leaf cgroup by its full path; the first match wins:

1. Inside an orchestrator service `orchestrator-<x>.service`, other than `orchestrator-watchdog.service`:
   - a child `agent-*` gives `agents`;
   - anything else (`daemon/`, systemd's `.control`, or the service itself when it is not split) gives `service`.
2. A verify scope `df-verify-<tag>-<uuid12>.scope`, anywhere, gives `verify` keyed by the tag.
3. Anything else inside a project slice `df-<project_id>.slice`, or inside reify's own `reify.slice`, gives `project`
   keyed by that slice. If reify turns its agent governor on, its governed agents land here rather than under
   `agents`; that is inert today.
4. A named factory service gives `shared` keyed by the unit. The services are `sccache`, `fused-memory`,
   `dark-factory-dashboard`, `orchestrator-watchdog`, `reify-warm-lane-gc`, `reify-jobserver`, `pytest-jobserver`,
   `cliproxyapi`, `dark-factory-load-sampler`, and the quality snapshot's `app-qualityx2dsnapshot.slice`.
5. T3 Code (`t3code.service`, anything in `app-t3code.slice`, and `app-com.t3tools.T3Code-*.scope`) gives
   `interactive@t3code`.
6. Terminal scopes (`app-org.kde.konsole-*`) give `interactive@terminals`; interactive Claude sessions started from
   a terminal land here.
7. `system.slice/docker-*.scope` gives `system@docker`.
8. Everything else gives `other`, keyed by its nearest anchor. This includes root-level leaves such as `init.scope`
   and the `*.mount` cgroups.

Residuals. Every inner cgroup contributes one residual, assigned in this order:

1. A split orchestrator service: its residual is the tails of exited agent attempts, so it joins `agents`.
2. A project slice: it joins `project` for that slice.
3. The root: it goes to `kernel`, which holds the reclaim threads and other kernel work.
4. An inner cgroup that a rule above names, for example `app-t3code.slice`: it joins that consumer.
5. Any other inner cgroup: `exited`, keyed by its nearest anchor (itself, if it is one). The small inner groups under
   `system.slice` therefore fold into `exited@system.slice`.

Before the project slices exist, a project's exited verify scopes leave their tails in `exited@app.slice`, which no
rule can attribute to a project. That is stated, not hidden.

**D3 — Gauges as sums; counters as increments over the interval; state kept per cgroup by path and inode.**

A sum of cumulative counters falls whenever a verify scope or agent attempt exits, so cumulative values cannot be
stored and differenced by readers. Instead, per collection, the recorder works as follows.

1. **Listing and readings, kept separately.** It reads the cgroup listing, each cgroup's path with its inode, and
   each cgroup's counters, and keeps both from the previous collection.
2. **Increments.**
   - A cgroup present in both listings with readable counters contributes current minus previous.
   - A counter that went down, or a new inode at an old path (a restarted service), is a reset: the cgroup
     contributes its current value.
   - A cgroup absent from the previous *listing* is new since then, and contributes its whole value.
   - A cgroup that was listed last time but unreadable is skipped this time and flags its consumer unreadable.
     It is never counted as new.
3. **Residuals.** An inner cgroup's residual increment is its own increment, minus its continuing children's
   increments, minus its new children's whole values. That residual is what exited children used since the previous
   collection. A cgroup that was unreadable last time and readable now has that interval's use land in its parent's
   residual. This is bounded by one interval and flagged.
4. **Exactness.** For CPU, swap-ins, reclaim, read-backs and the memory events, which the kernel counts
   hierarchically, every cgroup's increments, residuals included, sum to the root's increment, up to the skew of
   reading files one after another.
   - Stall totals are not hierarchical sums, and do not add across cgroups. They get no residual and no exactness
     claim (C1).
5. **Pruning.** Cgroups that have left the listing are pruned from the kept state.
6. **Gaps.** After a gap longer than two intervals, the recorder writes gauges only and marks increments unavailable
   for that collection. A gap is never written as a quiet interval (INV-11 `no-silent-fail-soft`: a degraded result
   must be distinguishable from success).
7. **The interval.** The interval's length is written too, so readers can form rates. It is about 30 to 40 seconds,
   since the timer runs every 5 seconds and misses about 2.6% of ticks (measured).

**D4 — The recorder contract gains state and a due time, persisted with the rows.**

- **Today's shape.** Each tick is a fresh process. `MetricGroup.collect` takes no arguments, and
  `sampler/src/sampler/sampler.py::run_tick` writes one transaction per tick.
- **The new shape.** A stateful group declares a due interval. Its collector receives the previous state and returns
  its samples together with the new state. `write_tick` persists the rows, the new state and the next due time in
  the same transaction. A crash mid-tick then neither double counts nor loses an interval.
- **Windowing and the journal.** The group is unwindowed (`windowed=False`), because the store's trailing window is a
  row count and would mean a different span at a 30-second cadence. The tick's journal output gets one summary line
  for the group, never its rows: a per-row print at this volume would push older entries out of the size-capped user
  journal, which holds the verify scopes' end records back to 2026-10-25.
- **Why 30 seconds is enough.** The scheduler decides a hold on the kernel's 10-second averages, and its gate ticks
  about 150 seconds apart (`shared/src/shared/psi.py::resolve_own_cgroup` docstring). A 30-second sample is fine
  enough to join to a hold.

**D5 — Metric names live in `METRIC_GROUPS` and nowhere else** (INV-14 `sampler-vocabulary-one-table`: the sampler's
metric names have one declaring table).

Two new groups are added:

- the consumer group, stateful, every 30 seconds, not served to the dashboard;
- a small verify-scope group with two host-level metrics (live verify scopes, and their summed `memory.current`).
  It is served by adding it to `dashboard/src/dashboard/data/load.py::_SERVED_GROUPS`, on the same due time. Its
  60-sample sparkline therefore spans 30 minutes, against 5 minutes for its neighbours.

The report script imports every metric name it reads from `sampler.metrics`, and is added to the scan in
`sampler/tests/test_vocabulary_one_table.py::_scope`. The card's own list in `tab_overview.jsx` is the known second
spelling the invariant names; leaf 1 swaps the two tiles there unless task 6987 has already removed that list.

**D6 — The two zero series are retired, not redefined.**

- `verify_concurrency` and `verify_rss_total_bytes` leave the table, and their rows age out under the 30-day
  retention.
- The replacements count every project's verify scopes on the workstation, of every role. The scope name carries no
  role, so live merge verifies are answered from the per-run events (leaf 5), which carry role.
- The five projects that verify unscoped are not in the tiles (D11).
- Process-name matching goes, so task 7006 item 2 has nothing left to do.

**D7 — One parser for cgroup files.** The kernel's `memory.stat`, `cpu.stat` and `memory.events` are key-value
lines. One parser in `shared/src/shared/cgroup_stats.py` (new, leaf 1) reads them, along with the pressure files
through the existing `shared.psi.parse_pressure_file`. The recorder, the verify capture and the agent cgroup manager
all import it (SPOT).

**D8 — The per-run verify record is captured inside the scope by a shell suffix and stored once, as an event.** This
supersedes task 6355's 2026-11-12 plan of a Python probe that waits on the command.

- **The capture.** The scope's script, built by `ScopeLaunchHandshake.payload`, runs the command in a subshell,
  `( <cmd> )`. A command that calls `exec` or `exit` therefore cannot skip the capture. Afterwards the script keeps
  the exit code, copies the scope's own `cpu.stat`, `memory.peak`, `memory.swap.peak`, `memory.events` and pressure
  files into a file the orchestrator named, and exits with the command's code.
- **Isolation from the command.** Nothing goes to the command's output, which the failure classifiers parse, and no
  Python starts inside the scope.
- **Timeout, cancel or error.** The kill path (`verify.py::_reap_command_subtree`) copies the same files from the
  scope's sysfs path before it kills the scope.
- **Commands that leave their scope.** A command that moved itself into another scope, which the existing
  `_discover_relocated_scopes` detects, marks the record `relocated` and names the scopes. A near-zero record is then
  never read as cheap.
- **Capture states.** `exit`, `before_kill`, `relocated` or `unavailable`, each with a reason. An unavailable capture
  never fails or slows the verify, so no config switch is needed.
- **Identity.** Identity is added where it is known: role, task, attempt, module, leg, segment, admission plan id,
  scope unit, host and project. For a remote merge verify, the dispatcher stamps task, merge sha and request id,
  because the laptop does not know them.
- **Where the record goes.**
  - It rides on the command's run result (`CheckRun`) into the summary's whitelist, and on a new field of
    `VerifyResult` across the wire.
  - The dispatching orchestrator emits one `verify_command_resources` event per command run into `runs.db`, for local
    and remote runs alike. That event is the durable record; the archive keeps about 5 days.
  - The summary and pass-summary entries are copies written from the same value at the same moment. The value is
    immutable, so they need no reconciler (G7 walk, INV-9).
- **The journal.** systemd's end-of-scope line is not a source.

**D9 — Agent attempts get cgroups inside their service, for recording only; CPU sharing does not change.** Leo ruled
twice on 2026-11-16.

1. Agents are split inside each orchestrator service, not into scopes beside it.
2. Once the session had shown that the split can record CPU without dividing it, Leo chose "record only, no CPU
   change".

The CPU split Leo first accepted would put uv, the daemon, the offline lane, git, ssh, gate builds and five small
projects' in-service verifies together at one agent's weight. That is a floor of 1/(N+1) of the service, with N up
to 36. Equal CPU between agents is therefore left for a later decision, made on this PRD's data. The mechanics below
were measured in this session.

*The unit.*
- Each orchestrator unit carries `Delegate=memory` and `DelegateSubgroup=daemon`. Delegating only the memory
  controller pins "no CPU split" in the unit file itself.
- systemd then starts the unit's whole process tree in a `daemon/` child: `uv`, the daemon it launches, and
  everything they fork. This matters because the unit's main process is `uv`, not the daemon. If only the daemon
  moved itself, `uv` would stay behind, and the kernel would refuse to turn controllers on below a cgroup that still
  holds processes (`critic.md` B1).
- systemd runs the unit's start and stop hooks in a transient `.control` child.
- Measured on 2026-11-16 with a transient unit whose main process launched a child (`delegate_probe2.sh`), run once
  under `Delegate=yes` and once under `Delegate=memory`, both with `DelegateSubgroup=daemon`. Both runs gave the same
  results:
  - both processes started in `daemon/`, and the service cgroup itself held none;
  - writing `+memory` to the service's `cgroup.subtree_control` succeeded;
  - a new child had `cpu.stat`, `memory.peak` and `memory.pressure`, and no `cpu.weight`;
  - a process joined the child before exec;
  - `systemd-run --user --scope` worked from inside both `daemon/` and the agent child, and the new scope landed
    where it would have otherwise;
  - the empty child could be removed.

*The daemon, at startup.* When the daemon's cgroup is `<project slice>/orchestrator-<x>.service/daemon`, and the
subtree is delegated (the service cgroup is writable and offers the memory controller), it writes `+memory`, and
only that, to the service's `cgroup.subtree_control`. It does this whatever the switch says.

Agent children are created only while the per-project switch is on (D10) and no governed agent launch is configured
(`cpu_governance.exec_path` unset).

*Per agent attempt.*
- The daemon creates `agent-<id>` and hands it to the shared spawn path through an injected placement object, never
  an import: shared code imports nothing from the orchestrator.
- A new outermost launch prefix writes its own pid into that child's `cgroup.procs` and execs. It is composed outside
  today's governor and nice prefixes and outside the sandbox wrap. Today's spawn already orders them
  `govern + nice + wrap_argv(cmd)` (`shared/src/shared/cli_invoke.py::run_cli_subprocess`), so the join happens
  before Landlock or bwrap applies.
- Everything the agent starts inherits the child by fork: tool servers, shells, pytest, cargo.
- One child per *spawn*. This matches the review-all stream 20 change to one `invocations` row per attempt (task
  9812 and its writer, §5), so each attempt's reading goes on its own row.

*When the attempt ends.*
- The placement object reads the child's counters through the D7 parser and returns them with the attempt's result,
  so the row writer records them.
- An empty child is removed.
- A child that still holds processes is kept, recorded as `lingering`, and removed by the startup sweep or a periodic
  sweep once empty. Lingering processes are recorded, not killed.

*On failure.* Any failure leaves the agent in `daemon/`, as today, with the reason on the row (INV-11). It never
stops a spawn.

*The share change, stated.* For CPU there is none.
- The CPU controller is neither delegated nor enabled inside the service, so every child shares the service's CPU
  exactly as all its threads do today.
- Shares between projects and roles do not change either.

Turning the memory controller on below the service has one behaviour change. The kernel keeps a separate reclaim
list per child, so under host memory pressure it reclaims from the daemon and from each agent attempt in proportion
to their size, rather than from one list for the whole service. No memory limit or protection is set on any child.

**D10 — The split never moves what the dispatch hold reads, and it can be switched off per project.**

- **Pressure resolution.** `shared/src/shared/psi.py::resolve_own_cgroup` resolves the daemon's own pressure cgroup
  to its project slice when it finds one.
  - The units carry `Slice=` from task 3394, and `Delegate=memory` and `DelegateSubgroup=daemon` from leaf 6. All
    three ride the same reinstall, task 9985.
  - A unit-file test in leaf 6 requires all three lines together, so a daemon is never in `daemon/` outside its
    slice.
- **Slice weight.** Task 3394's `orchestrator/src/orchestrator/project_slice_weight.py::ensure_project_slice_weight`
  checks that the daemon's raw cgroup is a `.service` directly under the slice. From the first start after the
  reinstall, that cgroup is `.../orchestrator-<x>.service/daemon`, so leaf 6 changes the check to accept a `daemon`
  child of such a service. It ships with the units, so 9985's own confirmation that the slice weight reads 100 still
  passes.
- **The switch.** A per-project config switch turns agent-child creation off.
  - It follows the review-all stream 16 layout, defaults on, and is reloadable without a restart: it decides only
    whether new spawns get a child. The memory controller is written at startup regardless (D9).
  - Because it defaults on, no project yaml needs the key. That avoids the trap of putting a new key into yaml before
    the orchestrator runs the code that knows it.
  - With the switch off, new agents stay in `daemon/`, and the service-level figures are unchanged.

**D11 — Left as they are, by decision:**

- **The five projects that verify inside their service** (scoping-intake, stream-touch, pump-web-ui, know-live,
  autopilot-video). Scoping their verifies would change their CPU shares, and none has held dispatch since 11-13.
  Their verify use is counted inside their `service` consumer.
- **The compiler cache.** It stays one shared consumer. A per-project split needs a server per project or an
  estimate, and no pending decision needs it.
- **Interactive sessions.** No per-session split. T3 already gives each session a scope. Leo ruled on 2026-08-03 that
  interactive work is not swept into project slices.
- **A recorder on the laptop.** The laptop has no holds, and its verify scopes carry most of its load (§1).
- **Gate builds, ssh sessions and stranded scopes on the laptop.** They stay unrecorded.

---

## 4. Pre-conditions and substrate (G3)

| Assumed capability | Evidence | State |
|---|---|---|
| Per-cgroup counters (`cpu.stat`; `memory.stat` keys `anon file pswpin pgsteal_kswapd pgsteal_direct workingset_refault_anon workingset_refault_file`; `memory.current`; `memory.swap.current`; `memory.peak`; `memory.swap.peak`; `memory.events`; `*.pressure total=`) | live read 2026-11-16, both hosts; the root has only `cpu.stat`, `memory.stat` and pressure | verified |
| `cpu.stat` exists in a child where the CPU controller is not enabled | `delegate_probe2.sh` (D9) | verified |
| One metric table and its guard | `sampler/src/sampler/metrics.py::METRIC_GROUPS`, `MetricGroup`, `MetricSpec`, `SampleKey`; `sampler/tests/test_vocabulary_one_table.py`; the pins in `sampler/tests/test_metric_groups.py::TestTheLiveTable` | verified; new groups edit `GroupName` and both pins |
| Stateful, due-time collection | `meta` table and `LoadSampleStore._is_due` in `sampler/src/sampler/store.py`; `run_tick` in `sampler/src/sampler/sampler.py` | the pattern exists; the stateful contract is new (D4, leaf 1) |
| A merge deploys the recorder at its next tick | unit `dark-factory-load-sampler.service`, `WorkingDirectory=%h/src/dark-factory`, `uv run --no-sync` | verified; no new third-party dependency (`--no-sync` would not install it) |
| Hold events name the holding orchestrator's cgroup | `dispatch_deferred` payload `sample.own_cgroup` (live, 2026-11-16) | verified; a unit path before the slices, the project slice after |
| A scope script can run a capture after the command, and the kill path can read first | `orchestrator/src/orchestrator/scope_launch_handshake.py::ScopeLaunchHandshake.payload`; `verify.py::_run_cmd`; `verify.py::_reap_command_subtree`; `orchestrator/src/orchestrator/relocated_scopes.py` | verified |
| Summary whitelist and wire carrier | `verify.py::_build_summary_payload`; `orchestrator/src/orchestrator/verify_runner.py::result_from_dict` (strict); `orchestrator/src/orchestrator/remote_gate.py::wire_fingerprint` keeps both hosts on one code version | verified |
| Event registry in `runs.db` | `orchestrator/src/orchestrator/event_store.py` | verified |
| Added nullable columns on `invocations` | `shared/src/shared/cost_store.py` ("Added columns stay LAST and nullable") | verified |
| One `invocations` row per attempt | task 9812 and its writer (review-all stream 20) | **queued prerequisite** for leaf 7a |
| Delegation, `DelegateSubgroup`, memory-only subtree, scope launches from inside | `delegate_probe.sh`, `delegate_probe2.sh`; systemd 255.4 on both hosts | verified |
| Launch prefix runs before the sandbox | `cli_invoke.py::run_cli_subprocess` | verified |
| Project slices exist and daemons live in them | task 3394 (pending; its branch has commits) and task 9985 (operator gate) | **queued prerequisite** for leaves 6, 7a and 7b |

---

## 5. Cross-PRD relationship and seam owners (G4)

| Other PRD or task | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| Task 3394, per-project slices (`docs/cgroup-slice-topology.md` on its branch) | consumes | slice names (`shared/src/shared/cgroup_slice.py::project_slice_name`); verify scopes under `df.slice/df-<id>.slice`; `ensure_project_slice_weight`; the topology doc's statement that agents inherit the service cgroup | 3394 owns the slices. This PRD owns the classifier's coverage of both placements (leaf 1), the weight check's acceptance of `daemon` (leaf 6), and updating the topology doc (leaf 7b) | 3394 pending, mid-implementation |
| Task 9985, operator reinstall | adds a dependency | one reinstall carries `Slice=` (3394) and `Delegate=memory` plus `DelegateSubgroup=daemon` (leaf 6) | 9985's steps are amended at decompose | pending gate |
| `plans/load-throttle-harmonisation-prd.md` | consumes | `METRIC_GROUPS` and INV-14; `resolve_own_cgroup` semantics (its §6.3) | that PRD. This one adds groups and must not move the own reading (C3) | shipped |
| Review-all stream 4, `plans/review-all-s4-verify-run-context-and-runner-contract-prd.md` (splits `verify.py`; `_run_cmd` becomes `verify_exec.py::SubprocessRunner`) | file contention | the capture hook, `CheckRun`, the summary and `VerifyResult` shapes | stream 4 owns the moves. Leaves 4 and 5 put their hooks wherever stream 4 has left that code at dispatch, and are serialized after its pending leaves on the same files | sequencing at decompose |
| Review-all stream 20, `plans/review-all-s20-cli-invocation-and-usage-gate-prd.md` (splits `cli_invoke.py`; the spawn moves to `shared/claude_runner.py`; task 9812 and its writer make `invocations` one row per attempt) | consumes and file contention | the spawn prefix; the attempt row | stream 20 owns the moves and the row grain. Leaf 7a lands on them and depends on 9812 and its writer | sequencing at decompose |
| Review-all stream 16, `plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md` | file contention | the leaf 7b switch's config layout | stream 16 | sequencing at decompose |
| `plans/remote-task-leg-verify-prd.md` | consumes | `VerifyResult` as the only remote carrier; `RemoteRunner` pass summaries | that PRD (shipped). Leaf 5 adds a field | shipped |
| `plans/allowance-reserve-by-account-prd.md` (another session, 2026-11-16; untracked at this writing) | file contention only | it adds keywords to `cli_invoke.py::invoke_with_cap_retry` and reads `invocations`; it lands before stream 20's refactor (its R5) | none semantic. Leaf 7a follows both that PRD's first task and stream 20's attempt rows. That PRD's own §5 should gain this row | stated both ways at decompose |

---

## 6. Contract (contracts plus boundary tests)

This PRD crosses five packages: sampler, dashboard, orchestrator, shared, and the unit files. It touches two
load-bearing seams, the verify path and the agent spawn path. So the seams are written here as contracts, with the
tests that face both sides in §7. Each producer can then be built without its readers in view.

**C1 — The consumer sample** (producer: leaf 1; readers: leaves 2, 3 and 8, and the dashboard).

*Per consumer* (the `leaf` column is the consumer's encoded name, D2):
- gauges, summed over members and residuals:
  - memory (`memory.current`, or `anon + file + kernel` from `memory.stat` where `memory.current` is absent);
  - program memory;
  - file cache;
  - swap held.
- increments over the interval:
  - CPU used (µs);
  - swap-ins (pages);
  - pages reclaimed in the background, and pages reclaimed directly;
  - pages read back after eviction, program memory and file cache separately;
  - times over `memory.high` and over `memory.max`;
  - OOM kills;
  - stall seconds: CPU "some", memory "some" and "full", IO "some".
- members (count).
- a readability flag: 1 when every member was read and its increments are valid.

*Stall seconds.* These are always *member-stall-seconds*: the sum of each member's own stall, which can exceed the
interval when members stall at the same time. They have no residual part, and are outside D3's exactness claim.
Stall says whose work waited, not who caused the pressure (task 5220's own wording); the hold report keeps the two
apart (C2).

*Host-level metrics:*
- the interval in seconds;
- the background reclaim and compaction threads' CPU (µs increments, from `/proc/<pid>/stat`, summed over each
  thread kind);
- live verify scopes and their memory (the served group, D5);
- host program memory and file cache from the root's `memory.stat`, and `MemTotal - MemAvailable` from
  `/proc/meminfo`.

*Rules:*
- a consumer with no members at a collection writes no rows: absent means none existed, never zero use;
- consumers never overlap, and their increments sum to the root's (D3);
- an unreadable member flags its consumer for that collection.

*Volume (projection).* 19 per-consumer metrics every 30 seconds:

- 4 gauges;
- 13 increments;
- the member count;
- the readability flag.

The critic applied D2's rules to the live tree on 2026-11-16 (240 leaf cgroups, 18 inner) and found about 45
consumers today. After the slices and the split, that becomes about 60 to 65. At 65 consumers this is about 205 rows
per 5-second tick, on top of today's 29. That takes the store from 2.4 GB to about 20 GB at 30 days, against 724 GB
free on the volume (measured 2026-11-16).

**C2 — The hold report's mapping and its two questions** (reader side, leaves 2 and 3).

- **Project mapping.** A project is mapped to its consumers by the project root whose `runs.db` holds the event:
  - `service` and `agents`: keyed by the last segment of the hold's `own_cgroup`. That segment is the unit before
    task 9985 and the project slice after, which is exactly how D2 keys these two roles;
  - `verify`: the tag computed from the root by the one shared scope-tag rule (leaf 1);
  - `project`: keyed by the same last segment when it names a project slice (after task 9985). Before that, a
    project has no `project` consumer of its own. reify's own `reify.slice` is listed as its own consumer, unmapped.
    The report imports nothing from task 3394, so it does not wait on the slices.
- **No assumed equality.** The report never assumes the unit stem, the verify slug and the project id are spelled
  alike: `orchestrator-my-solar-challenge.service` verifies as `solar-challenge`.
- **"Who grew"** ranks consumers by program-memory and swap growth over the window before the hold.
- **"Who paid"** ranks consumers by pages reclaimed from them and their own stall.
- **What it prints beside both:** the kernel's share, the unattributed remainder, and the `exited` residuals.
- **The summary.** `--summary` adds, across the window: holds per project, how often each consumer tops each ranking,
  totals per consumer, the unattributed share, and the reclaim threads' share.

**C3 — Recording never moves a control input** (all leaves).

- `shared.psi.resolve_own_cgroup(project_id)` returns the project slice for a daemon in
  `<slice>/orchestrator-<x>.service/daemon`, and `ensure_project_slice_weight` applies to it.
- The units never carry `Delegate=memory` or `DelegateSubgroup=daemon` without `Slice=`, and never delegate the CPU
  controller.
- No leaf edits `shared.psi`'s arms, the scheduler's admission, `psi_admission` thresholds, CPU weights, slot counts
  or memory limits.

**C4 — The per-run verify record** (producer: leaf 4; carrier and store: leaf 5; reader: leaf 8).

*Fields:*
- how it was captured:
  - `capture`: `exit` | `before_kill` | `relocated` | `unavailable`;
  - `capture_reason`;
  - `relocated_scopes`.
- CPU: `cpu_usec`, `user_usec`, `system_usec`.
- memory:
  - `memory_peak_bytes`, which includes file cache, the right figure for sizing a limit;
  - `swap_peak_bytes`;
  - `memory_high_events`, `memory_max_events`, `oom_kills`.
- stall: `cpu_stall_some_usec`, `memory_stall_some_usec`, `memory_stall_full_usec`, `io_stall_some_usec`.
- identity: `scope_unit`, `host`, `project`, `role`, `task_id`, `attempt`, `module`, `leg`, `segment`, `plan_id`.
- timing: `started_at`, `ended_at`.

*Shape.* One JSON-native dataclass in one home, listed in `remote_gate.py::wire_classes()`. Adding it changes the
wire fingerprint, and the laptop's gate checkout rebuilds once.

*Where it rides:* `CheckRun`, the summary whitelist, `VerifyResult`, the remote pass summaries, and the synthetic runs
in `RemoteRunner._archive_failure_streams`.

*The event.* `verify_command_resources` carries the record verbatim, with the dispatcher's runner name and, for
merges, `merge_sha` and `request_id`. Volume: about 4,000 events a day across both hosts (projection, from about 970
Dark Factory scopes per 6 h).

*Rules:* an unavailable capture is a record that says so, never a missing record; a verify never fails or waits
because of capture.

**C5 — The agent cgroup layout and the per-attempt record** (producers: leaves 7a and 7b; readers: leaves 1 and 8).

*Layout.* Inside a split service there are only:
- `daemon/`, holding uv, the daemon, and everything that is not an agent: git, ssh, gate builds, the offline lane,
  and the in-service verifies of the five unscoped projects;
- `.control/`, systemd's start and stop hooks, transient;
- one `agent-<id>/` per live agent attempt.

Processes never sit in the service cgroup itself.

*New nullable `invocations` columns:*
- `cgroup_capture`: `exit` | `lingering` | `unavailable`, with `cgroup_capture_reason`. Null means no placement was
  attempted, for example fused-memory's own rows.
- `cpu_usec`, `memory_peak_bytes`, `swap_peak_bytes`.
- `cpu_stall_some_usec`, `memory_stall_some_usec`, `memory_stall_full_usec`, `io_stall_some_usec`.

*The placement object.* A small protocol declared in shared code:
- before the spawn, it returns the prefix and environment for this attempt;
- after the attempt, it returns the reading.

The orchestrator implements it. `run_cli_subprocess` and `invoke_with_cap_retry` accept it as an optional parameter.

---

## 7. Boundary-test sketch

| Scenario | Preconditions | Postconditions |
|---|---|---|
| A verify scope exits between two collections | fixture tree: scope S present at collection 1 (CPU 10 s) and gone at collection 2; its parent's CPU rose by 4 s; new scope T at collection 2 (CPU 3 s) | `verify` gets 3 s; the parent's residual (`exited`) gets 1 s; no consumer is negative; all CPU increments sum to the root's |
| Stall is not a residual | a parent and two children with stall totals | each consumer's stall is its members' own; no residual stall row is written |
| A restarted service | same path, new inode, CPU counter lower than before | treated as a reset: the service contributes its current value |
| Listed but unreadable last time | member listed at collection 1 with an unreadable `cpu.stat` | not counted as new at collection 2; its consumer is flagged unreadable at collection 2 |
| Recorder gap | previous collection 4 intervals old | gauges written; no increments; every consumer flagged; interval written |
| Crash mid-tick | the process killed after collecting, before commit | the next tick neither double counts nor drops the interval (state and rows commit together) |
| Inner nodes are not members | a tree with `app.slice` → service → `daemon` and `agent-1` | members are `daemon` and `agent-1`; the service and `app.slice` contribute only residuals |
| Watchdog | `orchestrator-watchdog.service` | `shared`, not a project |
| Both verify placements | scopes under `app.slice` and under `df.slice/df-x.slice` | both counted as `verify`; the project slice contributes only its residual |
| Vocabulary | the report reads a metric | names come from `sampler.metrics`; the one-table scan covers the report |
| Hold before go-live | a hold timestamped before the first consumer sample | "recorder not live at this hold", distinct from an empty ranking |
| Hold mapping, both shapes | a hold with `own_cgroup` naming `orchestrator-my-solar-challenge.service`, root slug `solar-challenge`; a hold whose `own_cgroup` names `df-dark_factory.slice`, with consumer rows keyed by that slice | both resolve the service, verify, agents and project consumers per C2 |
| Verify capture at exit | a real scope running a command that allocates 200 MiB and ends with `exec true` | `capture=exit`; `memory_peak_bytes` ≥ 200 MiB; exit code kept; output byte-identical to a run without capture |
| Verify capture on timeout | a scoped command killed by timeout | `capture=before_kill`, with CPU and peak |
| Relocated command | the command starts its own `systemd-run --scope` | `capture=relocated`, with the scope named |
| Capture unavailable | scope off, or sysfs unreadable | `capture=unavailable` with a reason; verify outcome unchanged |
| Remote carry | a laptop task verify | the dispatcher's `runs.db` has one `verify_command_resources` per command, with `runner=laptop` and the task id; the pass summary carries the same records |
| Units carry all three | every `scripts/orchestrator-*.service` | `Slice=`, `Delegate=memory` and `DelegateSubgroup=daemon` together, or none of them; no unit delegates `cpu` |
| Own reading and weight after the split | daemon at `df.slice/df-x.slice/orchestrator-x.service/daemon` | `resolve_own_cgroup('x')` returns the slice; `ensure_project_slice_weight` sets the slice weight |
| Memory controller only | split service | `cgroup.subtree_control` of the service reads `memory`; no `cpu.weight` in any child |
| Agent joins before exec, sandbox on | spawn through the full prefix chain with the sandbox wrap | the agent's pid is in its child before the CLI's first instruction; the stdin-starvation pin (`TestStdinStarvationRace`) still passes |
| Retry spawns | a call that fails over to a second account | two children, two attempt rows, each with its own reading |
| Lingering processes | an agent leaves a background process behind | `cgroup_capture=lingering`; the child kept; the sweep removes it once empty |
| Switch off, or governed project | switch off; or `cpu_governance.exec_path` set | no agent children; spawns as today; rows have null placement columns |
| Switch turned on by reload | daemon started with the switch off | the memory controller is already on; the next spawn gets a child |

---

## 8. Decomposition plan (G2 signals; overlay sizing)

Leaves are numbered; existing tasks keep their ids. Sizes are expected changed lines.

**Leaf 1 — Record every consumer's memory, swap, reclaim, stall and CPU** (new; absorbs task 7858 part 2 and task
7006 item 2; ~1,100–1,400 LOC)

- **Modules:**
  - `sampler/src/sampler/consumers.py` (new: rule table, walk, member and residual state, consumer name encoding);
  - `sampler/src/sampler/metrics.py` (two groups, retire two names);
  - `sampler/src/sampler/sampler.py` and `sampler/src/sampler/store.py` (stateful due-time groups, D4);
  - `sampler/src/sampler/__main__.py` (one journal line per group);
  - `shared/src/shared/cgroup_stats.py` (new, D7);
  - the verify scope-tag rule's one public home in `shared/src/shared/` (new), which builds and parses the tag.
    Leaf 1 does not touch `verify.py`, which task 3394, review-all stream 4 and over a hundred pending tasks hold. A
    cross-check test pins the shared rule against `verify.py::_scope_tag_for` until leaf 4 switches `verify.py` over;
    leaf 4 owns ending that transition;
  - the sampler tests, including `sampler/tests/test_metric_groups.py`;
  - `dashboard/src/dashboard/data/load.py`;
  - `dashboard/src/dashboard/static/redux/tab_overview.jsx`.
- **Measurement during the leaf:** time one retention cleanup against a copy of the store at about 8× its rows, the
  C1 projection. If it exceeds 2 seconds, add an index led by `ts`.
- **Observable signal:**
  - within a minute of the merge, `sqlite3 -readonly data/load-samples.db` returns consumer rows for at least
    `service@orchestrator-dark-factory.service`, `service@orchestrator-reify.service`, `interactive@t3code`,
    `shared@sccache.service`, `system@docker`, `kernel` and `exited@app.slice`;
  - the background reclaim thread's CPU is recorded;
  - while a verify scope is live, the host load card shows non-zero "Verify scopes live" and "Verify scope memory";
  - the journal shows one summary line per collection.
- **Prereqs:** none. Tasks 7858 and 7006 are re-scoped at decompose.

**Leaf 2 — Report which consumers grew and which paid before each memory hold** (new; ~600–900 LOC)

- **Modules:** `scripts/resource_report.py` (new; `holds` subcommand), `scripts/tests/`,
  `sampler/tests/test_vocabulary_one_table.py::_scope`.
- **Observable signal:** `scripts/resource_report.py holds --since 7d --summary` against the live stores of all nine
  projects prints, for every memory hold since leaf 1 went live:
  - the two rankings, and the kernel, unattributed and exited figures;
  - the C2 summary.

  With no holds it prints "0 memory holds since <ts>". Earlier holds are listed as "recorder not live". The output is
  also available as `--json`. The project list comes from repeated `--project-root`, defaulting to the dashboard's
  known project roots.
- **Prereqs:** leaf 1 merged and writing in production (INV-13 `readers-prove-their-producer`: a reader lands only
  once its producer has written).

**Leaf 3 — Decision task: a week of consumer data on the memory holds** (new; `task_kind='deterministic'`,
`always_escalates`, delayed 7 days after leaf 2 is done)

- **Shape:** a pure human gate. Whoever handles its escalation runs
  `scripts/resource_report.py holds --since 7d --summary --json`. A deterministic check cannot name the script at
  filing time, because leaf 2 creates it.
- **Observable signal:** an escalation to Leo carrying that summary: holds per project, the consumers that most often
  topped each ranking, the unattributed share, and the reclaim threads' share.
- **Prereqs:** leaf 2.

**Leaf 4 — Every verify command records its CPU, peak memory and stall from inside its scope** (existing **task
6355**, retitled and re-planned; priority raised from low to medium because leaves 5 and 8 wait on it;
~600–900 LOC)

- **Modules:**
  - `orchestrator/src/orchestrator/scope_launch_handshake.py` (the suffix);
  - `orchestrator/src/orchestrator/verify.py`, or `verify_exec.py` if stream 4 has moved `_run_cmd` (kill-path read,
    `CheckRun`, the summary whitelist, the C4 dataclass's home, and switching `_scope_tag_for` to the shared
    scope-tag rule, which retires leaf 1's cross-check test);
  - `scripts/verify_budget_census.py` (prints CPU and peak per command from local summaries);
  - tests.
- **Observable signal:**
  - in a local Dark Factory task verify's summary under `data/verify-logs/<task>/`, every command entry run in that
    attempt (not carried over) has a C4 record with `capture=exit` and non-zero `cpu_usec` and `memory_peak_bytes`;
  - a verify killed by timeout carries `capture=before_kill`;
  - the census prints both figures.
- **Prereqs:** leaf 1 (the parser), and stream 4's pending leaves on the same files.

**Leaf 5 — Carry verify records back from the laptop, and store each command run as an event** (new; ~500–800 LOC)

- **Modules:**
  - `orchestrator/src/orchestrator/verify_runner.py` (`VerifyResult` field, pass summaries, `_archive_failure_streams`);
  - `orchestrator/src/orchestrator/remote_gate.py`;
  - `orchestrator/src/orchestrator/event_store.py`;
  - `orchestrator/src/orchestrator/task_verify_dispatch.py`, and the local emit point beside the admission-plan event;
  - tests.
- **Observable signal:** on the day it deploys, Dark Factory's `runs.db` holds `verify_command_resources` events with
  `runner=laptop` and with `runner=local`. Each carries a task id (or a merge sha and request id) and a C4 record. A
  remote green pass summary carries the same records.
- **Prereqs:** leaf 4.

**Leaf 6 — Delegate each orchestrator service's subtree, start it in `daemon/`, and keep the slice weight check
working** (new; ~150–300 LOC)

- **Modules:**
  - `scripts/orchestrator-*.service` (`Delegate=memory`, `DelegateSubgroup=daemon`);
  - `tests/scripts/test_orchestrator_service_files.py` (all three lines together);
  - `orchestrator/src/orchestrator/project_slice_weight.py` (accept a `daemon` child);
  - tests for `resolve_own_cgroup` and the weight check on the new path.
- **Observable signal:** an intermediate leaf; it unlocks leaves 7a and 7b and rides task 9985. After 9985's
  reinstall:
  - `systemctl --user show -p DelegateControllers orchestrator-dark-factory.service` reads `memory`;
  - the daemon's `/proc/<pid>/cgroup` ends in `/orchestrator-dark-factory.service/daemon`;
  - the slice's `cpu.weight` still reads 100;
  - new `dispatch_deferred` events' `own_cgroup` names the slice.
- **Prereqs:** task 3394, which edits the same files.
- **Why a separate leaf:** 3394 is mid-implementation on the high-priority merge lane, and adding to its scope would
  force a re-plan.
- **At decompose:** add the dependency of task 9985 on leaf 6, and amend 9985's steps.

**Leaf 7a — The shared spawn path accepts a placement and records each attempt's reading** (new; ~350–550 LOC)

- **Modules:**
  - `shared/src/shared/cli_invoke.py`, or `shared/claude_runner.py` once stream 20 has moved the spawn (the placement
    protocol, the outermost join prefix, the reading returned with each attempt);
  - `shared/src/shared/cost_store.py` (C5 columns);
  - tests, including the stdin-starvation pin.
- **Observable signal:** an intermediate leaf; it unlocks leaf 7b. A CI test spawns a real child through the full
  prefix chain with a fixture placement, and its attempt row carries `cgroup_capture=exit` and a non-zero
  `memory_peak_bytes`.
- **Prereqs:**
  - task 9812 (stream 20's attempt-row schema and readers), and stream 20's per-attempt writer (a separate task;
    name its id at decompose). 9812's deploy note also binds this leaf's rollout: the dashboard restarts after the
    schema lands and before the writer does;
  - the allowance-reserve PRD's first task on `invoke_with_cap_retry`;
  - stream 20's spawn move, if pending.

  All are same-file serialization.

**Leaf 7b — Each agent attempt gets its own cgroup inside its orchestrator service** (new; ~800–1,100 LOC)

- **Modules:**
  - `orchestrator/src/orchestrator/agent_cgroups.py` (new: startup checks, `+memory`, the placement implementation,
    remove, sweep);
  - the harness startup hook;
  - the orchestrator's writers of agent rows, passing the placement (`orchestrator/src/orchestrator/agents/invoke.py`,
    `workflow.py::_invoke`, `review_checkpoint.py`);
  - the per-project switch, in the stream 16 layout;
  - `docs/cgroup-slice-topology.md` (agents no longer inherit the service cgroup);
  - `orchestrator/src/orchestrator/agents/harness_guidance.py`, if its advice to read one's own `memory.events` now
    points at the wrong cgroup;
  - tests.
- **Observable signal:** after task 9985's restart:
  - Dark Factory's service cgroup holds `daemon/` and one `agent-*` child per running agent;
  - `cgroup.subtree_control` reads `memory`;
  - new `invocations` rows carry `cgroup_capture=exit` with non-zero `cpu_usec` and `memory_peak_bytes`;
  - the recorder shows `agents@df-dark_factory.slice`.
- **Prereqs:** leaves 6 and 7a. Its production signal needs task 9985.

**Leaf 8 — Report what verify runs and agent attempts cost** (new; ~400–700 LOC)

- **Modules:** `scripts/resource_report.py` (`costs` subcommand) and tests.
- **Observable signal:** `scripts/resource_report.py costs --since 7d` prints, from live stores:
  - CPU seconds and peak-memory percentiles per project, verify role, leg and host;
  - the same per agent role and model, with peak memory labelled as including file cache.

  A source with no rows says "no producer has written here". Before task 9985, the agent rows say this.
- **Prereqs:** leaves 5 and 7b merged.

**Store edits at decompose (not new tasks).** Apply the trimming rule: fix copied delivered checks and dependency
edges, and force a re-plan wherever the text fingerprint changes.

- **Task 7858.**
  - Remove part 2 (the sampler verify series, its tiles, and its acceptance "tracks live merge verifies"). Live merge
    verifies are answered from leaf 5's events, which carry role.
  - Keep part 1 (seed timing records).
  - Remove `sampler/*`, `tab_overview.jsx`, `dashboard/tests/test_load_data.py` and `shared/verify_role_env.py` from
    its files.
  - Note that its 2026-11-01 architect plan's part 2 is superseded and a re-plan is required.
- **Task 7006.** Drop item 2 with a pointer to leaf 1. Fix its `user_observable_signal`, `consumer_ref` and files
  (drop `sampler/src/sampler/metrics.py`).
- **Task 6355.** Becomes leaf 4:
  - retitle it (the current title still waits on task 5206, done 2026-10-09);
  - priority medium;
  - remove its config kill switch and its rusage arm;
  - add C4 and the suffix design;
  - note the re-plan.
- **Task 9985.**
  - Add the dependency on leaf 6.
  - Amend its steps and its confirmation step to check `Delegate=memory`, `DelegateSubgroup=daemon`, the `daemon/`
    layout, and that the slice weight still reads 100.
  - State that this delays the slice rollout, and with it task 5220's 14-day calibration clock and task 9984, by
    however long leaf 6 takes.
- **Task 3394.** Note that Leo's 2026-11-16 ruling supersedes its "agents inherit the service cgroup" for the long
  run. Its own scope is unchanged.
- **Tasks 9984 and 7112.** Add a pointer to leaves 1 and 8 as their data sources.

**Graph.**

- 1 → 2 → 3
- 2 → 8 (leaf 8 extends leaf 2's script)
- 1 → 4 → 5 → 8
- 3394 → 6 → 7b → 8
- 1 → 7a → 7b (both use leaf 1's cgroup parser)
- 9812 and stream 20's writer → 7a
- 6 → 9985 (an edge on the existing task)

Filed 2026-11-16 as tasks 10456 (leaf 1), 10457 (2), 10458 (3), 6355 (4), 10459 (5), 10460 (6), 10461 (7a),
10462 (7b) and 10463 (8). Each is serialized, where it shares files with pending work, behind:

- leaf 4: tasks 3394 and 8078;
- leaf 5: task 9766;
- leaf 7a: task 9846 (stream 20's writer) and task 9850 (stream 20's spawn move);
- leaf 7b: task 10430 (the allowance-reserve PRD).

Task 6987 now waits on leaf 1.

**G7 walk** (`docs/legibility/design-invariants.md`, all fourteen):

| Invariant | Finding |
|---|---|
| INV-1 contracts machine-checked | C1–C5 are dataclasses and table specs; the consumer name has one encode/parse pair (D2); no contract lives only in prose |
| INV-2 structured facts at failure | every capture failure is a field with a reason, not a log line (C4, C5) |
| INV-3 corroborate before acting | nothing acts on these readings (D1); the agent sweep removes a child only when the kernel reports it empty |
| INV-4 storm escape | a consumer group that fails every collection logs once per collection, about every 30 seconds. Leaf 1 limits it to one line per failure kind per hour, with the throttle's state kept through D4's state channel because each tick is a fresh process. The failure is also recorded as the readability flag |
| INV-5 no lock-step duplication | one cgroup parser (D7); one scope-tag rule (leaf 1), with a stated transition: a cross-check test pins `verify.py`'s copy until leaf 4 removes it; one rule table |
| INV-6 status matches liveness | no status records are added |
| INV-7 holds owned and bounded | a lingering agent child is owned by the daemon's sweep and bounded by its emptiness; it is recorded, not held |
| INV-8 loop-thread occupancy | leaf 7b reads and removes cgroups at agent end off the event loop, as the spawn path does today; the recorder is not async |
| INV-9 one fact, one home | the verify record's home is the `runs.db` event; the summary copies are immutable and written from the same value. The topology doc's agent statement is updated in leaf 7b |
| INV-10 guards exercise behaviour | the boundary tests run against fixture trees and real scopes, never against prose |
| INV-11 no silent fail-soft | gaps, unreadable members, unavailable captures and refused splits are each distinguishable from success (D3, C4, C5) |
| INV-12 exceptions owned or ratified | no allow-list entries or suppressions are planned |
| INV-13 readers prove their producer | leaves 2 and 8 land after their producers write, and print "not live" or "no producer" states |
| INV-14 sampler vocabulary one table | D5 |

---

## 9. Out of scope

- Any change to what the dispatch hold reads, its thresholds, CPU weights between projects or roles, slot counts, or
  memory limits (D1, C3).
- Turning the CPU controller on inside a service, which would give agents equal CPU shares. Leo deferred this on
  2026-11-16 until this PRD's data exists (D9).
- Per-group IO bytes, which need the io controller delegated by root.
- A per-project compiler-cache split.
- A recorder on the laptop.
- Per-session splits of interactive work.
- Scoping the five unscoped projects' verifies (D11).
- Killing leftover processes; they are recorded only.
- Back-filling history.

---

## 10. Open questions (tactical)

1. **Docker per container or as one.** **Suggested resolution:** one `system@docker` consumer. Split per container in
   leaf 1 only if the container's main program can be read cheaply from `/proc/<pid>/cmdline`.
2. **The window before a hold.** **Suggested resolution:** 120 seconds, the four collections before the hold, with a
   `--window` flag. Decide in leaf 2.
3. **Agent child name.** **Suggested resolution:** `agent-<task id>-<role>-<attempt>-<short id>`, sanitized to
   `[a-z0-9-]`. Decide in leaf 7b.
4. **The sweep's period.** **Suggested resolution:** every 10 minutes and at startup. Decide in leaf 7b.
