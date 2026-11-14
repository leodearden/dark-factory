# S24 — Fleet watchdog and redeploy coordination (PRD)

**Status:** draft, authored 2026-11-11 by the S24 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S24, wave 3, size M, risk medium), and
folded the same day with the program critic's findings, folded again on 2026-11-13 with the
sitting's rulings of 2026-11-12/13 (§Rulings), fold-verified the same day at `fceb951e7e`, and aligned the
same day with the lead's rulings after the wave-3 fold-verifiers. B + H: five contracts (C1–C5) and a two-way
boundary-test sketch. G5 fired on blast radius (shared, orchestrator, fused-memory, scripts, bash,
systemd units, docs) and on the seams five other streams read. Thirteen leaves.

**Code anchors** verified against main `a85cb1ad78` (2026-11-13) by the ruling fold and re-checked at
`d538279068` (2026-11-13): between the two, only `dark-factory-orchestrator.yaml`'s `starvation_watchdog`
block, two governed-list keys and `fused_memory/reconciliation/harness.py`'s remediation and judge paths
changed among this plan's paths, none of them a symbol the plan relies on, and every sidecar check keeps
its polarity at `d538279068`. The fold-verifier re-ran every producer check at `fceb951e7e` (no plan path
changed since `d538279068`). The author verified
them at `1f240e6da2` (2026-11-10) and the earlier fold at `ba17934861` (2026-11-10 23:49Z). Counts the
ruling fold re-measured are labelled `measured at a85cb1ad78`; a count still labelled `1f240e6da2` or
`ba17934861` sits in a region no commit since has touched, or is a finding measurement no done-criterion
rests on. Between `ba17934861` and `a85cb1ad78` four landed tasks touched this plan's files:
task 8270 (startup-hardening δ) changed `scripts/orchestrator-watchdog.py` by +230/−94 and added
`scripts/_watchdog_crash_loop.py` (448 lines) and `tests/scripts/test_watchdog_crash_loop.py`; task 6615
moved `orchestrator/src/orchestrator/service_restart.py` onto `shared.proc_liveness.pid_alive` and added
`tests/scripts/test_watchdog_pid_alive_parity.py`; task 8129 rewrote `scripts/restart-fused-memory.sh`'s
argument parser into two axes and added `--rebind-socket` (+120/−37); task 9493 set `KillMode=mixed` in
`scripts/fused-memory.service.template`. The run's anchors were verified at `6551b75aba`. Main moves
fast: cite by symbol and re-locate at implementation time.

Ruling: **R20** (program doc): option 1, one fleet lease file keyed by tier, in one format bash and the
watchdog share; the rest as proposed. Seams owned here (program §Seam ownership): the shared
fleet-coordination leaf, and `scripts/_systemd.py` as the host reader both watchdogs import. The sitting
of 2026-11-12 amended R20's watched-path clause and added three things (§Rulings).

## Rulings (sitting of 2026-11-12)

Leo answered on 2026-11-12 and his residual questions on 2026-11-13
(`digests/wave3-rulings-applied.md` §S24, `digests/discussion/S24.md`).

- **OQ-1** (wedged orchestrator): option B with a 900 s heartbeat age. The watchdog script restarts a
  wedged orchestrator on the third stale one-minute check and files one escalation on every such
  restart, in the same code path as the revive. No orchestrator unit file changes; the page-only soak
  and its 14-day flip task are dropped; systemd's own watchdog stays a possible later change. Ruled by
  Leo (residual R-3 = A). The directory walk behind the long stalls is task 9492, outside this plan and not ordered
  before ε.
- **OQ-2** (watched paths): option B. One code table per tier, and the fused-memory on-merge coordinator
  watches fused-memory's full import closure under a new minimum interval of 4 hours until the shutdown
  fix (task 9493) is in effect and 1 hour after it; the coordinator's restart is a drain restart that lets
  running reconciliation runs finish. Ruled by Leo (residual R-4 = C with changes). "In effect" means
  gate task 9521 is done (its unit check and live one-signal restart), so the 1-hour task depends on
  9493 and 9521, and its one-line runbook edit in `OPERATIONS.md` is exempt from the same-file chain, so
  it lands as soon as 9521 is done (lead ruling after the wave-3 fold-verifiers). The drain waits up to 7,200 s, then restarts and files an urgent
  `fused-memory-drain-timeout` escalation: bounded, so a hung run cannot hold fused-memory drained.
  Ruled by the lead as a reading of Leo's "allows recon runs to finish", and shown to Leo. Amends R20's
  "watched paths derived from orchestrator config": the config default derives from the code table.
- **OQ-3** (reclaim roots): A, the shared known-project-roots reader plus a host drop-in. Ruled by the
  lead (flagged: an operator action).
- **OQ-4** (storm escape per tier): A. Orchestrators page and stop reviving at the ceiling; fused-memory
  pages and keeps reviving under its cap; the dashboard keeps today's page and stop. Ruled by Leo (as
  recommended).
- **OQ-5** (fleet-state spelling guard): A, a family in task 6726's shared spelling-guard test. Ruled by
  the lead.
- **OQ-6** (order with task 7314): A, 7314 waits for γ and is amended. Ruled by the lead.
- **OQ-7** (order with S16's restart-settings task): A, S16 ζ1 waits on γ and ζ. Ruled by the lead.
- **OQ-8** (one sweep at a time): D, a kernel lock held for each restart script's life; a live
  overrunning sweep reads `OVERRUN` and pages hourly. Ruled by the lead (flagged: no displacement of a
  live sweep, and a new hourly page). Implements R20's "acquires the lease under `flock`" literally.
- **OQ-9** (deliberate redeploy during a sweep): B, `--lease-wait SECS` on both restart scripts; the
  watchdog and the coordinator keep the 5 s wait. Ruled by the lead (flagged: `restart-fused-memory.sh
  --now` is refused with exit 75 during a fused-memory sweep unless the flag is passed).

S16's sitting point S16-m14 (γ deletes `orchestrator_restart_lease_max_age_secs`, so S16 ζ1 moves one
setting fewer) is S16's to fold; γ's deletion stands as authored.

## Goal

One supervision policy, written once and parameterised per tier, drives the three supervised kinds of
unit on this host: the orchestrator fleet, `fused-memory.service` and `dark-factory-dashboard.service`.
Each tier is a frozen `TierSpec` row. Its fields are the unit source, start property, watched paths,
deploy clock, min interval, redeploy unit and command, lease tier, streak, revive cap and storm escape.
One liveness pass and one staleness pass consume the rows. Each pass talks to the host only through an
injected `Host`, so its tests use a fake host and assert on what the pass did.

The facts the fleet's redeploy deciders share now have one home: `shared/fleet_coordination.py`, a
stdlib-only leaf. These facts are the deploy clocks, the lease file, the heartbeat directory, each
watchdog tier's state file, the lease ages and each tier's watched code closure. The Python tiers import
it, and bash reads it through `python3 -m`. The in-flight lease is one file keyed by tier. Each restart
script holds an exclusive kernel `flock` on its tier's sweep lock for its whole life, so a second
concurrent sweep is refused at the writer instead of trusted to callers, and the kernel releases the
lock on any exit, SIGKILL included.

What a user observes when this lands:

- `scripts/orchestrator-watchdog.py` revives all nine installed orchestrator units on this host. Today
  its liveness loop walks the hand-kept `WATCHED` list and never probes
  `orchestrator-scoping-intake.service` or `orchestrator-stream-touch.service`; staleness and
  `--report` already enumerate every running unit (`scripts/orchestrator-watchdog.py::report` reads
  `_enumerate_running_units`). `--report` gains a LIVENESS column fed by the same installed-socket
  derivation the liveness pass uses, so both units show as probed, and one `FLEET-LEASE:` line per tier.
- Starting `scripts/restart-all-orchestrators.sh --drain` while another sweep holds the
  orchestrator-fleet lease exits 75 before writing any drain request. It names the holder's pid,
  current units and age. A second `restart-fused-memory.sh` is refused the same way, and the two tiers
  never refuse each other. A deliberate deploy (an explicit deploy task or an operator's restart) passes
  `--lease-wait SECS` and waits for the holder instead of failing. A broken lease CLI aborts
  the script with exit 70 before it writes anything.
- A wedged orchestrator whose event loop stopped refreshing its fleet heartbeat is restarted by the
  watchdog about 17 minutes into the wedge: a heartbeat older than 900 s on 3 consecutive one-minute
  checks, counted by a streak of its own, separate from the dead-unit revive streak. Every such restart
  files one critical escalation, `orchestrator-liveness-heartbeat-<unit>`, naming the heartbeat age, the
  threshold and where the stall's stack lines are, in the same code path as the restart. Today the port
  probe and ActiveState both stay green, so nothing notices. Every liveness tier escalates when its
  revives storm (INV-4); an orchestrator stops being revived at that ceiling and fused-memory does not.
- The fused-memory on-merge coordinator in dark-factory's orchestrator restarts fused-memory for any
  merge into fused-memory's import closure (fused-memory, shared, escalation and orchestrator code and
  their packaging files), at most once per 4 hours until task 9493's stop fix is in effect on the host
  and once per hour after that. Its restart is a drain restart: no new reconciliation run starts, and
  the restart waits for the runs in flight to finish, up to 2 hours, which covers every completed full
  run of the last 30 days; a run still going at that bound is cut and an escalation says so. The
  restart runs as its own transient user unit, so an orchestrator restart during the drain cannot kill
  it, and once the drain signal is sent every exit path restarts fused-memory: a drained harness is
  never abandoned.
- The nightly orphan-worktree reclaim sweeps every known project root. Today it sweeps dark-factory's
  only. Measured at `a85cb1ad78` (2026-11-13), `solar-challenge-platform/.worktrees-orphaned` holds 16
  parkings (1.4 G), `solar-challenge` 5 (1.4 G) and `scoping-intake` 1 (110 M), unchanged since
  `1f240e6da2`.
- Adding a project needs no watchdog edit, and a shared-only change makes the orchestrator fleet stale.
  Changing a lease age or a state path is one edit.

## Background

The ten-day-old run measured a watchdog that copies its policy per unit kind
(`fk-906957d2ec84`). It keeps a hand-maintained liveness set (`fk-c9390bf37295`). It patches private
names in its tests because it has no host seam (`fk-3830e2a6de79`). Its prose contradicts its code in
four places (`fk-ed19c0c4d4df`). Its watched paths are five literals (`fk-68efae8a36fc`). Its orchestrator
liveness tier cannot see a wedged event loop and has no storm escape (`fk-1cc71283157c`). Fleet mutual
exclusion is a check-then-act lease that the writer overwrites (`fk-4c21aaf01b82`). The orphan reclaim
covers one root of nine producers (`fk-47a7eb8c020d`). Each fleet fact has 3–4 copies (`fk-63633270af4e`).
All nine re-measure as present at `a85cb1ad78` (§Findings disposition).

Prior work this PRD builds on and does not redo: the per-unit drain (tasks 5371, 7297), the in-flight
lease (4755), the head start anchored on min-interval expiry (4754), the module-removal waiver (7198),
fm's streak and revive cap (3764, 4131), the `/alive` route (3765), the dashboard's hysteresis and storm
escape (`plans/dashboard-availability-prd.md`), the state primitives (6448), S25's
`scripts/_checkout.py` bootstrap (S25 γ 6758, done), the reconciliation drain that finishes current
cycles (2702, 1053), the cycle-aware restart script (2703), the restart script's two-axis parser and
`--rebind-socket` (8129, done), and three tasks that landed while this plan was in review: task 8270's
crash-loop alarm and per-unit revive cap (done, merged `7ddc80450e`), task 6615's one pid predicate
`shared.proc_liveness.pid_alive` (done, merged 2026-11-11, `f750373949`), and task 9493's single stop
signal for fused-memory (done, merged `1d0ae21e44`; in effect on the host only once operator task 9521
reinstalls the unit).

## Findings disposition

Re-measured by the ruling fold at `a85cb1ad78` unless a cell says otherwise.

| Key | Sev | Disposition | Re-measured |
|---|---|---|---|
| `fk-906957d2ec84` policy copied per unit kind | high | leaf δ3 (closing; β, δ1, δ2a and δ2b carry their parts) | Still present. There are now 5 hand `split("=", 1)` parsers (4 in `orchestrator-watchdog.py`, task 8270's `_unit_restart_status` among them, + 1 in `dashboard-watchdog.py`). `dashboard-watchdog.py` still holds same-named copies of `log`, `_JournalLog`, `probe_health`, `restart_unit` and `_unit_active_enter_epoch`. The lease and the 4755 re-check exist only in `staleness_pass`. The "three-phase stop" prose is still in `dashboard-watchdog.py::restart_unit`. The 8 fleet/fm sibling pairs held 1,004 lines at `1f240e6da2` (AST spans, prose included; not re-measured, no done-criterion rests on it). |
| `fk-c9390bf37295` hand-kept `WATCHED` | medium | leaf δ2a | Still present: `WATCHED` has 7 pairs, and 9 `orchestrator-*.socket` units are installed and enabled. `scoping-intake` (8109) and `stream-touch` (8110) are missing, and since task 8270 they also miss the crash-loop sampling, which walks `WATCHED`. Task 9321 (pending, low) adds the two pairs as an interim fix. Three tests pin `WATCHED`: `test_main_targets_expected_pairs`, `test_watched_ports_match_configured_escalation_ports` and `tests/scripts/test_orchestrator_socket_units.py`; a fourth, `test_restart_orchestrator_unit_default_matches_across_tiers`, reads it. |
| `fk-ed19c0c4d4df` prose share and false claims | medium | leaf ω | Three of the four false claims are still present: "ONLY by main()", "dark-factory + reify" and "mirrors orchestrator/config.yaml". "only uncapped" went with task 8270's revive cap (`d5020b8555`). The "ZERO watched-path commits" paragraph has 4 copies. The file is 3,214 lines; its prose share was 1,856 of 3,078 at `1f240e6da2` and ω re-measures it. |
| `fk-3830e2a6de79` no host seam; private patches | medium | leaf δ2b (δ1 carries the fm half, δ2a the fleet-liveness half) | `tests/scripts/test_orchestrator_watchdog.py` has 12,809 lines and 394 tests, with 595 `monkeypatch.setattr(wdog, …)` sites. The watchdog test files hold 220 private-name patch sites (`wdog, "_…"`: 213 + 4 + 3), down from 237 at `1f240e6da2` because task 8270 moved `main()` onto an injected status reader. 7 docstrings carried the "monkeypatch the module global" clause at `1f240e6da2`. |
| `fk-68efae8a36fc` watched paths in five lists | medium | leaf ζ (ruled OQ-2 B) | Still present: `WATCHED_PATHS` has no `shared/src/`, the config default is `['orchestrator/src/']`, and the yaml sets the 5-entry list. `MODULE_REMOVAL_WATCHED_PATHS` adds `shared/src/` for the waiver only. `WATCHED_PATHS` watches `orchestrator/uv.lock`, not the workspace's root `uv.lock`. The fused-memory coordinator's default is `['fused-memory/src/']` with no min interval (`harness.py::Harness._build_service_restart_coordinator` passes no `min_interval_secs` or `state_path`). |
| `fk-1cc71283157c` orchestrator liveness fails INV-4 | high | leaf ε (ruled OQ-1 B, OQ-4 A) | `main()` still decides on `probe_port` and `_unit_active_state` only and never reads the fleet heartbeat. It has no storm escape. Task 8270 (done, merged `7ddc80450e`) added a per-unit revive cap (`ORCH_LIVENESS_RESTART_MIN_INTERVAL_SECS`, 900 s) and the crash-loop alarm (`scripts/_watchdog_crash_loop.py`), but no wedge signal and no escape. |
| `fk-4c21aaf01b82` advisory check-then-act lease | high | leaf γ | `lease_acquire` stamps and `mv -f`s unconditionally. `scripts/restart-all-orchestrators.sh` and `restart-fused-memory.sh` contain no `flock`; `restart-fused-memory.sh` guards only `--rebind-socket` against a running staleness redeploy (`refuse_while_staleness_redeploy_runs`, task 8129). |
| `fk-47a7eb8c020d` reclaim covers one root | medium | leaf η (ruled OQ-3 A) | `scripts/reclaim-orphaned-worktrees.sh` still pins `REPO=/home/leo/src/dark-factory`. The live parkings are listed in §Goal. |
| `fk-63633270af4e` fleet facts copied (**weakened**, R23) | low | leaf α (γ finishes the bash and `service_restart` copies) | Re-verified under G3/G6 and kept. The lease path has 4 copies (`service_restart.py`, the watchdog, `df_pytest_isolation.py`, the bash script). The deploy clock has 4. The heartbeat dir has 4 (`fleet_heartbeat.py`, `drain_check.py`, `df_pytest_isolation.py`, bash). The lease max-age has 3 code copies (`config.py`, `service_restart.py`, the watchdog). The skeptic's two objections no longer hold. (1) "Needs sys.path wiring in a deployed stdlib script": `scripts/_checkout.py` is that wiring. Measured: `/usr/bin/python3 -I` (3.12.3) imports a stdlib `shared` leaf through it without loading aiosqlite, pydantic or yaml. A plain `PYTHONPATH=shared/src python3` does too. (2) "df_pytest_isolation cannot import shared": escalation's conftest stub sets `__path__` to `shared/src/shared` before it imports `df_pytest_isolation`, so a stdlib submodule resolves (measured with the stub reproduced). The "imports eagerly" docstring the finding cited is already gone from `sweep_toolcall_markup.py`. |

None of the nine keys has a task (contract §8 step 1, `find_tasks_by_metadata`, all nine empty again on
2026-11-13). None is in the report's `## Mechanical filings` table (all structural). Step 2 found the live
tasks that §Cross-PRD relationship orders or folds in; the ruling fold's re-run found three more: task
9321 (the two missing `WATCHED` pairs), task 9389 (a refusal in `restart-fused-memory.sh` that leaves
reconciliation drained) and task 9521 (the operator reinstall that puts 9493's stop fix into effect).

## Pre-conditions for activating

- **S25 `scripts/_checkout.py` (S25/C1, task 6758, done).** It is what lets the stdlib watchdog and
  `drain_check.py` import `shared.fleet_coordination` (G3 verified above).
- **S14 `shared.project_layout` (task 6710, done).** C1 derives the fleet home from
  `default_project_root()`, which returns the main checkout from a linked worktree. η reads
  `known_project_roots()`.
- **Task 8270** (startup-hardening δ, done, merged `7ddc80450e`). It added `NRestarts` sampling, the
  start-limit-hit rule and the per-unit revive cap to `main()`, with the crash-loop decision in
  `scripts/_watchdog_crash_loop.py` (pure `observe_unit`, injected reader, its own state file
  `data/orchestrator/watchdog-restart-rate.json`) and the cap's clocks under
  `data/orchestrator/liveness-restart/`. Its sampling runs "before every liveness gate, enabled and
  startup grace included"; C4's gate order keeps that. The edges β → 8270, ζ → 8270 and δ2a → 8270 are
  satisfied; δ2a ports the landed code onto the spec.
- **Task 6615** (S19 C1, done, merged 2026-11-11, `f750373949`). It added
  `shared/src/shared/proc_liveness.py::pid_alive`, moved `service_restart.py::lease_is_live` onto it,
  rewrote `scripts/orchestrator-watchdog.py::_pid_alive` as "a deliberate COPY of
  shared/src/shared/proc_liveness.py::pid_alive", and added `tests/scripts/test_watchdog_pid_alive_parity.py`
  to pin that copy. γ → 6615 is satisfied: γ binds the watchdog and C2 to `shared.proc_liveness` and
  deletes the copy and its parity test (the cross-stream "one public liveness check", confirmed by Leo
  2026-11-13).
- **Task 9140** (S6 α, the INV-15 validator, pending, not parked). R54 puts it first on every file it
  shares with another plan. γ, ζ, η and ι depend on it for `orchestrator/src/orchestrator/config.py`,
  `orchestrator/src/orchestrator/harness.py` and `OPERATIONS.md`.
- **Task 6726** (the shared spelling-guard test, S14, pending). ω depends on it.
- **Task 9493** (fused-memory's single stop signal, done, merged `1d0ae21e44`) and **task 9521** (its
  operator follow-up, a deterministic human gate, blocked at L2 since 2026-11-13T01:29Z): 9521 reinstalls
  `~/.config/systemd/user/fused-memory.service` with `KillMode=mixed` and checks one stop signal on a
  live restart. Measured at `a85cb1ad78` (2026-11-13): `systemctl --user show fused-memory -p KillMode`
  prints `control-group` and `python3 scripts/check_fused_memory_unit_parity.py` exits 1 (drift:
  `KillMode=mixed` missing), so the fix's unit half is not yet in effect; re-measured unchanged at
  `fceb951e7e`. θ, the 1-hour interval flip, depends on both. **How "in effect" is known:** 9521 is
  done. Its own steps are the check: step (4) confirms `systemctl --user show fused-memory -p KillMode`
  prints `KillMode=mixed`, and step (6) confirms a live restart logged exactly one `Received SIGTERM`
  and "Application shutdown complete." By then both halves of 9493 are live, the unit setting and the
  server's repeat-signal guard. The scheduler enforces this through θ → 9521, and θ does not re-probe the
  host.
- **Task 9492** (pending, medium, parked behind 8528) moves the directory walk behind the measured long
  loop stalls (`merge_lane/liveness.py::newest_content_mtime`) off the event loop. ε does not wait on
  it (ruled OQ-1, R-3 = A); 900 s leaves about 384 s over the longest measured stall either way.
- No leaf declares `orchestrator/src/orchestrator/merge_lane/worker.py`, so none is parked behind 8528.
  No leaf touches verify or the merge lane, so none waits on 5040.

## Sketch of approach

**The shared leaf (C1, then C2 and C5).** `shared/fleet_coordination.py` is stdlib-only. α's C1 imports
only `shared.project_layout`; γ adds `shared.safe_io` and `shared.proc_liveness` (task 6615; used by the
transition branch only, which ω deletes). Measured at `ba17934861`: `project_layout` and `safe_io` are
listed in `shared/tests/test_pure_stdlib_leaves.py::PURE_STDLIB_LEAVES`; `proc_liveness` imports only
`os` (landed with 6615). It tabulates each piece of fleet state on disk as a
`FleetFile` (relpath, env override, redirect-only flag), including each watchdog tier's state file. It
resolves each one against the fleet home, which is the dark-factory main checkout and machine-global,
never a foreign project's root. It also holds the per-tier lease ages; after ζ, the per-tier code
closure and the fused-memory on-merge coordinator's minimum interval; after ι, the fused-memory drain's
bound. A `python3 -m shared.fleet_coordination` CLI gives bash the same answers: `path …`, `lease …`
and, after ι, `drain max-wait`. That is how R20's "bash reads via `python3 -c`" is met without Python
source inside a bash file. These restart defaults live here under the cross-package defaults rule (lead
S16/OQ-4, sitting point S24-m7): one dependency-free shared module per default, and the orchestrator
config field's default names the constant.

**The keyed lease (C2).** One file, the existing `data/orchestrator/fleet_redeploy_lease.json`, keyed
by tier, plus one sweep lock file per tier. Each restart script takes an exclusive kernel `flock` on its
tier's sweep lock before anything else and holds it for its whole life; the kernel releases it on any
exit. Readers judge "in flight" by the same lock, probed without blocking, and read the JSON
entry only for the holder's pid, units and age. "At most one live sweep per tier" is enforced where a
breach can be refused (heuristic 10), with one predicate for writer and readers (heuristic 11). Both
restart scripts publish their entry and current units after taking the lock, and release the entry on
every catchable exit. The watchdog, `service_restart` and the harness's two coordinators read through C2. The
reader-side checks stay as cheap early exits that save a doomed `systemd-run`, not as the guarantee.

**The host reader (C3).** `scripts/_systemd.py` is the watchdogs' one boundary with the systemd user
manager. It covers `show` with one key=value parser, the running and installed units, enablement,
revive with a typed outcome, transient-unit registration and the journal. Both watchdogs import it.

**The supervision engine (C4).** `scripts/_supervision.py` holds `TierSpec`, its policy values, the
`Host` protocol and the two passes. `scripts/_watchdog_host.py` holds the real `Host`, composing C3, the
port and HTTP probes, git, the clock, the per-tier state files, C2's lease reader and the
`escalation submit` page. `tests/scripts/fake_watchdog_host.py` holds the test double. The entry
scripts shrink to their spec rows and `_cli`: `orchestrator-watchdog.py` has two tiers and
`dashboard-watchdog.py` one. The fleet tier's liveness set is every installed `orchestrator-*.socket`'s
service, read through C3, and its staleness set is those of them that are running (today's
`_enumerate_running_units`), so `WATCHED` and its pinning tests go.

**The fused-memory drain restart (ι).** `scripts/restart-fused-memory.sh --drain` already does most of
what Leo asked: it sends SIGUSR1, `fused_memory/reconciliation/harness.py::ReconciliationHarness.drain`
stops starting new cycles, each `_project_loop` finishes the cycle it is running and exits, and the
script restarts on the "Harness fully drained" journal line. What is missing is the bound and its
visibility: the script gives up after `DRAIN_TIMEOUT` (2,100 s) and restarts anyway, cutting the run
still going, with only a WARNING. The next instance resumes or recovers the interrupted run (tasks 2717,
2711) and files nothing (`ReconciliationHarness._recover_stale_runs` suppresses `recon_stale_run` for a
killed owner by design), so the cut is silent. ι raises the bound to
7,200 s, the shared leaf's `FM_DRAIN_MAX_WAIT_SECS`, and files one escalation when a drain still times
out. The coordinator's restart is pinned to `--drain` in ζ.

A drain must never be abandoned between the drain signal and the restart, because
`ReconciliationHarness.drain` is one-way until the process restarts. Three paths lose the restart today,
and a longer bound widens each of them:
- The coordinator spawns the script as a plain child process
  (`service_restart.py::StaleServiceRestartCoordinator._default_restart_executor`). That child lives in
  the orchestrator unit's cgroup (`KillMode=control-group`, measured on `orchestrator-dark-factory.service`).
  Any stop of that orchestrator during the wait kills the script after SIGUSR1: a fleet sweep, its
  on-merge self-restart, or a crash. ζ's closure makes the overlap likelier, because one merge into
  `orchestrator/src/` now arms both coordinators.
- γ's `trap 'exit 143' TERM` and `trap 'exit 130' INT` end the script without restarting.
- A hung `escalation submit` on the timeout path would hold the restart behind it.

So ζ fires the script as a cgroup-escaping transient user unit, the shape the orchestrator coordinator
already uses (`harness.py::Harness._build_orchestrator_restart_coordinator` over
`service_restart.py::schedule_detached_systemd_restart`), with an on-failure escalation. ι makes every
exit after SIGUSR1 that has not reached the restart step issue `systemctl --user restart --no-block
fused-memory`, and bounds the submit with `timeout 60`. Measured: the 81 drains of 2026-11-01..13 each
reached a restart, so this risk has not yet fired; the fix closes it before the bound grows to 2 hours.

**Order.** Edge notation: "X → Y" means X depends on Y. γ → α; ζ → γ; β → ζ; ι → ζ; η → ι;
δ1 → β; δ1 → η; δ2a → δ1; δ2b → δ2a; δ3 → δ2b; ε → δ3; ω → ε, ω → θ; θ → ζ, θ → ι, θ → 9493,
θ → 9521.
Two tasks of this plan that edit the same non-test file are ordered by an edge; tasks that share only a
test file are not, and file locks serialize them (lead rulings of 2026-11-12, restating R54). The one
exemption is θ's one-line runbook edit in `OPERATIONS.md` (lead ruling after the wave-3 fold-verifiers):
it takes no edge, file locks serialize it, and the later task rebases one line. Each shared non-test file
is then edited in one order:
- `scripts/orchestrator-watchdog.py`: α, γ, ζ, β, δ1, δ2a, δ2b, ε, ω.
- `OPERATIONS.md`: γ, ζ, ι, η, δ1, δ2a, δ2b, ε, ω; θ's one line is exempt and lands whenever θ does.
- `shared/src/shared/fleet_coordination.py`: α, γ, ζ, ι, θ, ω.
- The engine files: δ1 through ω.
- `restart-fused-memory.sh`: γ, ι. `config.py` and `harness.py`: γ, ζ. `dashboard-watchdog.py`: β, δ3.

ζ sits right after γ, so the config-shape plan's ζ1 waits only on this plan's second and third tasks.

ι and η run beside β. δ1 waits on β and on η, and β (about 900 changed lines) is roughly as long as
ι followed by η (about 800), so the two short tasks add little to the engine chain. They also land
Leo's drain ruling and R20's reclaim early rather than after the whole chain. The lead confirmed this
placement (the drain task and the reclaim task stay before the engine tasks).

θ lands as soon as operator gate 9521 is done, as Leo ruled ("until the shutdown fix, then A"). Its
runbook line in `OPERATIONS.md` is exempt from the same-file chain, so θ keeps only its real edges: ζ
(the watched-path task that creates the constant, named by the ruling), ι (the task before it on
`shared/fleet_coordination.py`; ζ is reached through it too), 9493 and 9521. ω also edits
`shared/fleet_coordination.py` (it deletes C2's legacy branch) and no exemption covers that file, so the
two are chained, and the direction the ruling leaves is ω → θ. **Consequence, recorded:** ω, the
integration gate, now waits on human gate 9521 through θ. Nothing in this plan or any sibling plan
depends on ω, and no high leaf waits on it.

## Resolved design decisions

1. **One lease file keyed by tier (R20).** It keeps the path `data/orchestrator/fleet_redeploy_lease.json`
   and the env override `ORCH_FLEET_LEASE`. Spawners, the `df_pytest_isolation` redirect and the
   operator's muscle memory carry over. The four-tier drift test collapses into C1. A third tier is a
   `Tier` member and a `LEASE_MAX_AGE_SECS` entry.
2. **Exclusion is enforced at the writer, and the CLI's failures are loud.** A restart script whose
   tier's sweep lock is held by another sweep is refused (C2 L1). Refusal exits 75 (`EX_TEMPFAIL`), not 0. A deploy task, an operator or the
   coordinator must never read "another sweep is running" as "your redeploy happened" (INV-11). The
   refused run writes no drain request and releases nothing it does not hold (L5). Every other CLI exit
   status is an infrastructure failure (an `ImportError` after a bad merge to the leaf,
   `CheckoutRootError`, a missing `python3`): the bash script aborts before any drain request, echoes
   the CLI's stderr and exits 70 (`EX_SOFTWARE`). This reverses a deliberate choice:
   `scripts/restart-all-orchestrators.sh::lease_release` kept the lease path coreutils-only because "a
   python3 dependency on a path that EVERY invocation takes would be new". After γ every invocation of
   both restart scripts depends on `python3` and `shared/src`. The reversal is accepted because the
   alternative is a fifth bash copy of each fact, and a broken CLI now fails loud (exit 70, no drain
   request, no lease) rather than silently. A deliberate deploy passes `--lease-wait SECS` and waits on
   the sweep lock, bounded by the holder tier's max-age; the watchdog and the coordinator never pass it
   and keep the default 5 s wait.
3. **Transition (one deploy cycle).** A sweep started on pre-γ code can still be running when γ lands,
   writing the legacy single-tier body. C2's reader accepts that body as the orchestrator-fleet tier's
   lease (`current_unit` → `current_units`), judged by pid and age as today, so a post-γ watchdog still
   honours it; and the new script's `lease acquire` refuses with 75 while that legacy entry is live,
   because a pre-γ sweep takes no kernel lock. ω deletes that branch. The other direction is
   DF's long-running coordinator, which runs pre-γ code until the next fleet redeploy. It reads the keyed
   body as "no lease" (fail-open, its `lease_is_live` finds no `pid`) and could fire into a live sweep.
   The new script refuses with 75, and the coordinator's `--on-failure` escalation files one "fire-time
   failure" page. That residual lasts at most one 72 h window. It is accepted and named in OPERATIONS.md,
   not engineered around.
4. **No pid identity for a live sweep; `shared.proc_liveness.pid_alive` for the transition only (task
   6615, done; γ → 6615 satisfied).** The kernel lock is the one in-flight predicate for writer and readers, so
   no live sweep is judged by pid, start time or age. Only the transition branch (decision 3) needs a pid
   predicate, and it uses `pid_alive`: public, stdlib-only, never raises, rejects anything that is not a
   positive non-bool int before `os.kill`. γ deletes the watchdog's `_pid_alive` (a declared copy since
   6615; every caller is a lease read that becomes a C2 call) together with 6615's
   `tests/scripts/test_watchdog_pid_alive_parity.py`, and deletes `service_restart.lease_is_live` with its
   `shared.proc_liveness` import.
5. **Fleet state is machine-global.** `service_restart` today joins the clock and lease relpaths to
   each orchestrator's own `project_root`, so a foreign project's coordinator would read
   `<foreign>/data/orchestrator/…`. After γ every reader resolves the fleet home through C1. On DF it
   resolves to the same file as today. This reverses a deliberate per-checkout scope: the comments above
   `scripts/restart-all-orchestrators.sh::CLOCK_FILE` and `LEASE_FILE` scope the clock and lease per
   checkout. After γ a sweep or test run from a worktree that does not set the env overrides touches the
   MAIN checkout's live lease and clock. Tests stay contained because `df_pytest_isolation` sets the
   overrides for every suite (unchanged) and its deploy-clock guard already watches the main checkout
   (`df_pytest_isolation.py::deploy_clock_guard_roots`). A hand-run sweep from a worktree now honours the
   live lease, which is the point. **Accepted residual:** on a remote-verify host,
   `orchestrator/src/orchestrator/remote_gate/ensure-gate.sh` makes gate checkouts with
   `git clone --local`, so `fleet_home()` there answers the clone itself, and `df_pytest_isolation`'s
   `LIVE_FLEET_DIR` names the clone's own `data/fleet`, not the host checkout's. The suite there is still
   redirected by `ORCH_FLEET_DIR`; only the forgetful-spawner detector watches the clone's directory
   instead of the host's.
6. **The fm tier gets the same lease (subsumes task 5719).** Its max-age is 3,600 s from γ and 7,800 s
   from ι. Derivation at γ: `restart-fused-memory.sh` can hold `RECON_GATE_TIMEOUT` or `DRAIN_TIMEOUT`
   (both 2,100 s), then a stop of up to `TimeoutStopSec` 90 s (unchanged: S15-1 keeps it), then
   `HEALTH_TIMEOUT` 30 s, about 2,220 s, so 3,600 s clears its worst legitimate run. ι raises the drain's
   bound to `FM_DRAIN_MAX_WAIT_SECS` 7,200 s (decision 17). The worst legitimate run becomes 7,380 s:
   the bound, then the timeout escalation's 60 s submit bound, the stop and the health wait. So the
   max-age becomes 7,800 s, and a test in `shared/tests/test_fleet_coordination.py` pins
   `LEASE_MAX_AGE_SECS[FUSED_MEMORY] >= FM_DRAIN_MAX_WAIT_SECS + 60 + 90 + 30`. Under the kernel lock the
   max-age only pages (`OVERRUN`); it never frees the tier. The fleet tier keeps 14,400 s. **L1 holds only while the bash ceiling
   derivation holds, and at nine units it does not:** the derivation above
   `scripts/restart-all-orchestrators.sh::DRAIN_SWEEP_VERIFY_MAX_WAIT_SECS` assumes 8 units (7 later
   ones), and with 9 installed, 10,300 + 1,080 + 8 × 420 = 14,740 s > 14,400 s, so a sweep can outlive
   its lease by about 340 s (measured at `ba17934861`). Task 7307 owns re-deriving that ceiling from the
   live unit count; γ gives it the bound through `python3 -m shared.fleet_coordination lease max-age
   --tier orchestrator-fleet` (7307 → γ), and the bash comment points at C1. Under OQ-8 D, exclusion is
   the kernel lock and no longer depends on that arithmetic at all; the age bound only pages: a sweep
   still holding its lock past the tier max-age reads `OVERRUN`, stays in flight for every reader, and
   the watchdog files one `<tier>-lease-overrun` page per hour per tier.
7. **Env knob names are operator interface and do not change.** `ORCH_RESTART_MIN_INTERVAL_SECS` (set
   by the installed `72h-restart-cadence.conf` drop-in), `STALENESS_GRACE_SECS`, `FM_*`,
   `ORCH_LIVENESS_RESTART_MIN_INTERVAL_SECS` (8270), `DASHBOARD_WATCHDOG_*` (including
   `DASHBOARD_WATCHDOG_STATE`, which keeps naming the dashboard tier's state file), `ORCH_FLEET_*`
   (including the bash script's `ORCH_FLEET_LEASE_MAX_AGE_SECS`) and `FM_DEPLOY_CLOCK` all keep their
   names. New: `FM_LEASE_MAX_AGE_SECS`, `ORCH_WATCHDOG_STATE`, `FM_WATCHDOG_STATE`, `FM_ON_MERGE_CLOCK`
   (ζ). The spec rows read them. Task 8270's `ORCH_RESTART_RATE_STATE` and
   `ORCH_LIVENESS_RESTART_CLOCK_DIR` name files δ2a folds into the fleet tier's `TierState`; they go with
   those files, and `ORCH_WATCHDOG_STATE` overrides the one that replaces them. `ORCH_CRASH_LOOP_QUEUE_DIR`
   keeps naming the queue every watchdog page goes to. The names `FAIL_STREAK` and `MAX_RESTARTS` also
   stay spelled in `dashboard-watchdog.py`: a landed task's delivered check greps `scripts/` for
   `FAIL_STREAK`.
8. **One gate sequence for every tier (C4).** The 4755 read-then-act re-check, the lease gate and the
   current-units skip now apply to every tier with a lease tier, not to fleet staleness only. The
   per-unit enabled and startup-grace gates that both staleness passes apply today stay in the staleness
   sequence. The streak, revive cap and storm escape apply to every tier whose spec sets them. Policy
   values differ per tier, and a tier that lacks a policy says `None` in its row. The feature matrix the
   finding measured becomes visible in one table (heuristic 3).
9. **Persisted per-tier state replaces three ad-hoc files.** Each tier has one `TierState` JSON file, a
   C1 `FleetFile` (so it resolves through `resolve()` and inherits `df_pytest_isolation`'s redirect),
   written atomically through `_watchdog_state.atomic_write_json`. It carries per-unit streaks, instance
   start, revive epochs, the escape episode, crash-loop samples and per-sentinel page epochs. The first
   tick after δ1, δ2a or δ2b deploys starts the fm and fleet tiers from empty state (fail toward not
   acting): the worst case is one revive that the old cap clock would have held, and a crash loop already
   in progress is paged up to one alarm window (1,800 s) late, because 8270's
   `data/orchestrator/watchdog-restart-rate.json` and `data/orchestrator/liveness-restart/<unit>.json`
   are not migrated.
   The dashboard tier keeps today's path, `data/dashboard-watchdog/state.json`, and δ3's reader accepts
   today's body (`streak`, `restarts`, `ceiling_open`, `last_escalation_epoch`) once, so an open ceiling
   episode and the restart list survive the deploy: no extra revives, no re-page. `shared.streak.KeyedStreaks`
   is not used because it is an in-process counter and these streaks must survive a oneshot's exit. They
   also need instance-boundary and max-age expiry, which `KeyedStreaks` does not model.
10. **Pages go through `escalation submit`.** That is the dashboard watchdog's path today and the one
    8270 landed (`scripts/_watchdog_crash_loop.py::crash_loop_submit_argv`). The real host has one
    `file_page(Page)`. Agent roles stay `orchestrator-watchdog` and
    `harness-dashboard-watchdog`, which `escalation/src/escalation/authority.py` and the
    watcher-design ruling on 8270 pin. When S1 lands its filing entry point, the CLI routes through it
    and this code does not change.
11. **A stale heartbeat restarts the orchestrator, and every such restart escalates (ruled OQ-1 B,
    R-3 = A).** The fleet heartbeat (`data/fleet/<unit>.json`) is written every
    `merge_heartbeat_interval_secs` = 15 s by the `merge-heartbeat` background service, which
    `Harness._start_lifecycle_services` starts at startup step 1b, on the event loop, just as fm's
    `/alive` is served. Measured at `a85cb1ad78` (2026-11-13): all nine units' `ts_epoch` ages were
    4.7–14.9 s at one read. A check reads stale when the heartbeat is older than 900 s; the age is counted
    from the later of `ts_epoch` and the unit's current instance start, so a file left by the previous
    instance never counts against a new one. The probe acts only after `HeartbeatProbe.streak` = 3
    consecutive stale one-minute checks, counted by a heartbeat streak persisted apart from the dead-unit
    revive streak, which stays `Streak(1, 300)` (today's immediate revive) (INV-3). At the third stale
    check the unit joins the dead-unit path: storm escape, then task 8270's per-unit revive cap (one
    revive per 900 s), then the revive. So a true wedge is restarted about 1,020 s (17 minutes) after the
    loop stopped. The longest legitimate stall measured is 635.7 s (`orchestrator-reify.service`,
    2026-10-29..11-01 sample, `digests/discussion/S24.md` §S24-1), which leaves about 384 s of margin
    before task 9492 moves its cause off the loop.
    - **One escalation per wedge revive, in the revive's own code path.** The liveness pass files it in
      the revive's `finally`, after the restart attempt, so a restart that raises or times out still
      files it, and a filing that fails is logged and never undoes the revive. Sentinel
      `orchestrator-liveness-heartbeat-<unit>`, severity critical, category `infra_issue`, agent role
      `orchestrator-watchdog`; its detail names the heartbeat age, the 900 s threshold, the revive
      outcome and the recipe (`journalctl --user -u <unit>` around the stall, and its `loop_stall` stack
      lines). There is no once-per-window setting: pages are bounded by the revive cap and by the storm
      ceiling (3 revives in an hour, then one ceiling page and no further revive).
    - A missing heartbeat file never counts as stale: `--report` renders `absent`, and the pass logs one
      bucketed WARNING per active unit past its startup grace that has no heartbeat file (INV-13: the
      reader proves its producer).
    - A wedge restart is a hard kill: a wedged loop cannot run its SIGTERM handler, so `systemctl
      restart` ends in SIGKILL after the unit's 90 s stop timeout, and in-flight merge verifies and agent
      sessions die with it. That is why the threshold sits well above the longest measured stall.
    - No orchestrator unit file changes. systemd's own watchdog (`WatchdogSec=` with an on-loop ping)
      stays a possible later change (§Out of scope); nothing here prevents it. The unit files this plan
      does edit are unrelated to the wedge: δ2a's `TimeoutStartSec` comment in the watchdog's own unit,
      and η's `Environment=` line in the reclaim unit.
    - After a wedge revive, both of the unit's streaks are consumed (heartbeat and dead-unit). A unit
      still wedged after the restart is not stale again until its new instance is 900 s old, and it then
      needs 3 more stale checks. Revives stay at most one per 900 s under 8270's cap (row 22c).
12. **Prose goes to its homes.** I5 (liveness is not a deploy), I6 (stateless between ticks, except
    each tier's declared state) and I7 (`--report` mutates nothing) get one statement each in
    OPERATIONS.md §8 "Supervision invariants". The fail directions do too. `TierSpec`'s docstring
    points there. Measurements, rejected alternatives and task history leave the code for the commit
    messages and the PRDs they came from (Comments stance). Each migration leaf writes its new code
    low-comment. ω measures what remains and moves the rest.
13. **Same-file serialization is a chain** (§Decomposition plan). The tests that pin the old structure
    are retired with it: a test that patches a private name is rewritten onto `FakeHost` if it pins
    behaviour (an action taken or withheld) and deleted if it pins structure (which helper ran). The
    leaf's commit message carries the mapping (old test → new test or "structure pin, retired").
14. **The journal stays quiet when nothing needs doing.** `scripts/orchestrator-watchdog.py::SKIP_LOG_INTERVAL_SECS`
    exists because ungated skip lines wrote "~4,320 near-identical lines" over a window, and
    `scripts/_watchdog_dashboard_parity.py` gates for the same reason ("~1440 near-identical WARNINGs a
    day"); healthy liveness ticks in `main()` are silent today. One line per decision would be about 10
    tier units every 60 s plus the dashboard every 30 s, roughly 17,000 lines a day. C4 therefore logs
    every actionable decision and keeps non-actionable verdicts silent or bucketed, as today.
15. **Who decides each supervision case (`one-decision-partition-per-area`, INV-23; Leo, 2026-11-12).** Every case this
    plan touches is settled by facts code holds, so code decides all of them, in one place each, and no
    case is handed to an LLM. Where a human must act, code files a born-at-L2 page and the human is the
    escape hatch.

    | Case | Facts | Decided by (where) | Action | Human path |
    |---|---|---|---|---|
    | Unit dead (port down, or ActiveState inactive/failed) | probe, systemd | code (`_supervision.py::liveness_pass`) | revive, under streak, escape and cap | crash-loop page (8270) |
    | Orchestrator heartbeat older than 900 s on 3 checks | heartbeat file, instance start | code (same pass, same revive path) | revive and file one `orchestrator-liveness-heartbeat-<unit>` | that escalation |
    | Revives reach the storm ceiling | revive epochs in `TierState` | code (same pass) | orchestrators and dashboard stop reviving; fused-memory keeps reviving under its cap | one ceiling page |
    | Restart count grows by 5 in 30 min, or `Result=start-limit-hit` | `NRestarts`, `Result` | code (8270's `_watchdog_crash_loop.py::observe_unit`, called by the pass) | page; start-limit-hit is left down | the crash-loop page |
    | A second same-tier sweep starts | kernel lock | code (`flock` in each restart script) | refuse, exit 75 (or wait with `--lease-wait`) | none needed |
    | A sweep holds its lock past the tier max-age | kernel lock, lease entry age | code (`staleness_pass`) | stays in flight; page hourly | the `<tier>-lease-overrun` page |
    | A merge touches a tier's code closure | landed diff, `WATCH_CLOSURE` | code (coordinator, watchdog staleness) | restart after the min interval | none needed |
    | A fused-memory drain still has a run in flight at 7,200 s | journal drained marker, `/health` `recon_busy` | code (`restart-fused-memory.sh`) | restart, cutting the run | one `fused-memory-drain-timeout` escalation |
    | The restart script exits after SIGUSR1 without reaching its restart step (signal, error) | the script's own "signal sent, restart not run" flag | code (its EXIT trap) | `systemctl --user restart --no-block fused-memory` | the on-failure escalation below |
    | The fused-memory on-merge restart exits non-zero (refused 75, health failure, abandoned drain) | the transient unit's exit status | code (the on-failure wrapper `schedule_detached_systemd_restart` builds) | none beyond logging | one born-at-L2 escalation |
16. **The fused-memory on-merge coordinator (ruled OQ-2 B, R-4 = C with changes; lands in ζ, flipped in
    θ).** It keeps today's machinery (`service_restart.py::StaleServiceRestartCoordinator`, which already
    takes `min_interval_secs`, `state_path` and `stamp_clock_on_fire` and never lets force-fire bypass the
    interval) and changes six inputs:
    - **Watched paths:** its config default becomes `list(WATCH_CLOSURE[Tier.FUSED_MEMORY])`, the
      closure row the watchdog's fm staleness reads, and a test pins equality (W3).
    - **Interval:** new restart-only field `fused_memory_restart_min_interval_secs`, default
      `FM_ON_MERGE_MIN_INTERVAL_SECS` = 14,400 s (4 hours) in `shared/fleet_coordination.py`. Measured
      over the 30 days to 2026-11-12 (`digests/discussion/S24.md` §S24-2): 150 coordinator fires, about 5
      a day; replaying 993 closure merges at 4 hours models about today's rate, at 1 hour about 11 a day.
      θ sets it to 3,600 s (1 hour) once task 9493's fix is in effect on the host, which is known by
      gate task 9521 being done (§Pre-conditions).
    - **Clock:** a new `FleetFile`, `FM_ON_MERGE_CLOCK` (`data/fused-memory/last_on_merge_restart_fused_memory.json`,
      env `FM_ON_MERGE_CLOCK`), change-detected by `df_pytest_isolation` like the deploy clocks, since a
      test that stamped the live file would hold fused-memory's restarts for up to 4 hours. The
      coordinator stamps it on a successful fire (`stamp_clock_on_fire=True`, the class default): the
      interval caps attempts. A fire whose script later fails files its on-failure escalation (below),
      and the next closure merge after the interval, or the watchdog's 8-hour fm staleness backstop with
      its own clock, retries it.
    - **Restart mode:** the builder passes `script_args=['--drain']` explicitly instead of inheriting the
      class default, and a harness test pins that the fused-memory coordinator fires
      `scripts/restart-fused-memory.sh --drain` (decision 17 says what the drain waits for).
    - **Where the restart runs:** the builder passes a `restart_executor` that registers the script as
      a transient user unit `fm-on-merge-restart-<n>.service` through
      `service_restart.py::schedule_detached_systemd_restart`, the shape
      `Harness._build_orchestrator_restart_coordinator` uses for `orch-selfrestart-on-merge-<n>.service`.
      Its `on_failure_escalation` files one born-at-L2 escalation when the script exits non-zero. Without
      this, an orchestrator stop during a drain of up to 2 hours kills the script after SIGUSR1 and leaves
      fused-memory drained (§Sketch, "The fused-memory drain restart").
    - **Where it runs:** `fused_memory_restart_on_merge_enabled` defaults to false, and
      `dark-factory-orchestrator.yaml` sets it true. Chosen over "apply the closure only where the
      project root is the dark-factory checkout", which would need a checkout-identity test in code. The
      fact "the project whose merges change fused-memory restarts it" is stated once, in that project's
      config. Measured at `a85cb1ad78`: no other project yaml sets any `fused_memory_restart_*` key; five
      other projects have a root `uv.lock` (autopilot-video, know-live, scoping-intake, solar-challenge,
      solar-challenge-platform), and none has `scripts/restart-fused-memory.sh`, so their coordinators
      could never restart fused-memory and would only log a WARNING per matching merge. Turning them off
      changes no restart anywhere.

    Why the 4-hour to 1-hour flip is a task and not a config value: the value is a code default in one
    shared constant, pinned by a test and described in OPERATIONS.md. A yaml override would make a second
    home for it (heuristic 11, S16 C8's one-home rule), and the flip's trigger is a fact the scheduler can
    enforce as dependencies (9493 merged, 9521 done) rather than an edit someone must remember.
17. **The drain restart lets reconciliation runs finish (ruled R-4, bound confirmed by the lead; lands
    in ι).**
    - Drain timeouts, measured from the journal (`/var/log/syslog`, 2026-11-01..13): 81 `--drain`
      restarts. 74 ended on the drained marker (median 11 s, p90 1,253 s, longest 1,801 s); 7 timed out
      at 2,100 s and cut a running run.
    - Run lengths, re-measured by the fold-verifier from `data/reconciliation/reconciliation.db` `runs`
      (completed rows, `completed_at − started_at`) over 2026-10-14T07:00Z to 2026-11-13T07:00Z:
      - full runs: 106 of 1,766 took longer than 2,100 s, and the longest took 6,407 s;
      - remediation runs: the longest of 27 took 6,897 s.

      So today's bound cuts runs Leo asked to finish. ι sets `FM_DRAIN_MAX_WAIT_SECS` = 7,200 s, which
      covers every completed run in that window, with 303 s to spare over the longest.
    - The window is edge-sensitive. Moving its start back one day, to 2026-10-13, adds five longer runs,
      all finishing on 2026-10-14 between 06:08 and 06:58Z: four full runs of 7,386–9,750 s and one
      remediation run of 16,890 s. The bound would have cut those.
    - `restart-fused-memory.sh --drain` reads the bound through
      `python3 -m shared.fleet_coordination drain max-wait`. A drain still running at the bound restarts
    anyway, because a hung run must not hold fused-memory drained and stale indefinitely (INV-7), and it
    files one escalation, sentinel `fused-memory-drain-timeout`, severity urgent, category `infra_issue`,
    agent role `harness-fused-memory-restart`, naming the bound and the runs `/health`'s `recon_busy`
    listed, so the cut is not silent. The default path's `RECON_GATE_TIMEOUT` (the watchdog's 8-hour fm
    staleness backstop uses it) and `--now` are unchanged.
    - The timeout path cannot lose the restart. The submit is bounded by `timeout 60` and ends `|| true`.
      The `/health` read is bounded by `CURL_MAX_TIME`. The restart follows unconditionally.
    - A drain is never abandoned. Once SIGUSR1 has been sent, any exit that has not reached the restart
      step issues `systemctl --user restart --no-block fused-memory` and logs why: a TERM or INT trap, or
      an error under `set -e`. Only a SIGKILL can still skip it, and since ζ the script runs in its own
      transient unit, so only an operator stopping that unit can deliver one.
    - Cost: while draining, no new reconciliation cycle starts for any project. At today's figures that
      is 11.2 hours of drain over 12 days; the 7 cut runs would have held it longer, by at most 5,100 s
      each.
    - γ publishes `fused-memory.service` as the lease's current unit only for the restart step itself
      (stop, start and health wait, at most 120 s), not for the drain wait, so the watchdog's fm liveness
      revive is skipped for that step only and not for up to 2 hours.

## Contracts

Signatures are the contract; bodies are the implementer's. Every function is total over its stated
domain and raises only where stated.

### C1 — fleet state locations and lease ages (`shared/src/shared/fleet_coordination.py`; lands in α)

```python
# α: stdlib + shared.project_layout only
# γ adds shared.safe_io and shared.proc_liveness (task 6615; transition branch only, deleted in ω)
class Tier(StrEnum):
    ORCHESTRATOR_FLEET = 'orchestrator-fleet'
    FUSED_MEMORY = 'fused-memory'

@dataclass(frozen=True)
class FleetFile:
    relpath: str          # under fleet_home()
    env_var: str          # a non-empty value overrides the resolved path, for every reader and writer
    redirect_only: bool   # df_pytest_isolation: redirect, but no change detection (the lease and the state files)

FLEET_DEPLOY_CLOCK: FleetFile   # data/orchestrator/last_redeploy_orchestrator.json, ORCH_FLEET_DEPLOY_CLOCK
FM_DEPLOY_CLOCK: FleetFile      # data/fused-memory/last_redeploy_fused_memory.json, FM_DEPLOY_CLOCK
REDEPLOY_LEASE: FleetFile       # data/orchestrator/fleet_redeploy_lease.json, ORCH_FLEET_LEASE, redirect_only
SUPERVISION_STATE: Mapping[str, FleetFile]     # one TierState file per watchdog tier, all redirect_only:
    # 'orchestrator-fleet' → data/orchestrator/watchdog-orchestrator-fleet.json, ORCH_WATCHDOG_STATE
    # 'fused-memory'       → data/fused-memory/watchdog-fused-memory.json, FM_WATCHDOG_STATE
    # 'dashboard'          → data/dashboard-watchdog/state.json, DASHBOARD_WATCHDOG_STATE (today's path and name)
FM_ON_MERGE_CLOCK: FleetFile    # ζ: data/fused-memory/last_on_merge_restart_fused_memory.json, FM_ON_MERGE_CLOCK (change-detected)
PROTECTED_FILES: tuple[FleetFile, ...]   # the files above plus the three state files, in df_pytest_isolation's reporting order
HEARTBEAT_DIR: FleetFile        # data/fleet, ORCH_FLEET_DIR

def fleet_home() -> Path                       # shared.project_layout.default_project_root(); raises CheckoutRootError
def resolve(file: FleetFile, environ: Mapping[str, str] = os.environ) -> Path
def deploy_clock(tier: Tier) -> FleetFile
LEASE_MAX_AGE_SECS: Mapping[Tier, int]         # {ORCHESTRATOR_FLEET: 14400, FUSED_MEMORY: 3600}; ι: FUSED_MEMORY 7800
def lease_max_age_secs(tier: Tier, environ: Mapping[str, str] = os.environ) -> int
    # env ORCH_FLEET_LEASE_MAX_AGE_SECS / FM_LEASE_MAX_AGE_SECS; a non-integer or <= 0 override is ignored with a WARNING
FM_ON_MERGE_MIN_INTERVAL_SECS: Final[int]      # ζ: 14400 (4 h); θ: 3600 (1 h). The default of
                                               # OrchestratorConfig.fused_memory_restart_min_interval_secs
FM_DRAIN_MAX_WAIT_SECS: Final[int]             # ι: 7200; restart-fused-memory.sh --drain's bound
```

CLI: `python3 -m shared.fleet_coordination path {deploy-clock,lease,heartbeat-dir} [--tier T]` prints
one absolute path and exits 0; after ι, `drain max-wait` prints `FM_DRAIN_MAX_WAIT_SECS` as an int and
exits 0. A usage error exits 2 and names the bad argument. Invariants:
**P1**, every reader and writer of a `FleetFile` resolves it through `resolve()` (env override first),
and no other module spells its relpath (ω's guard). **P2**, `fleet_home()` answers the main checkout
from any linked worktree (a `git clone --local` gate clone answers itself; decision 5). **P3**, the
module imports under `/usr/bin/python3 -I` with only `shared/src` on `sys.path`, and loads no
third-party module (a test runs it in a subprocess and asserts on `sys.modules`).

### C2 — the keyed lease (same module; lands in γ)

```python
@dataclass(frozen=True)
class Lease:
    tier: Tier
    pid: int
    started_ts: float
    current_units: tuple[str, ...]

class LeaseState(StrEnum):            # extends today's six --report states (scripts/orchestrator-watchdog.py::_format_fleet_lease, task 4755)
    NONE = 'none'                     # sweep lock free, no entry
    LIVE = 'live'                     # sweep lock held, age within the tier max-age; or held with no entry yet (lease None)
    OVERRUN = 'overrun'               # sweep lock held, age >= the tier max-age: still in flight; the age bound only pages
    FUTURE_DATED = 'future-dated'     # sweep lock held, started_ts ahead of this clock: still in flight; one WARNING
    UNREADABLE = 'unreadable'         # body corrupt, unreadable or non-finite started_ts: in flight iff the sweep lock is held
    HOLDER_GONE = 'holder-gone'       # sweep lock free, an entry left by a dead holder: not in flight
    EXPIRED = 'expired'               # legacy entry only: pid alive but past max-age (today's rule): not in flight

class LeaseReading(NamedTuple):
    state: LeaseState
    lease: Lease | None
    held: bool                        # in flight: what every reader gates on

class Acquired(NamedTuple):
    lease: Lease
    displaced: Lease | None           # an entry a dead holder left behind, replaced (logged)

class Refused(NamedTuple):
    holder: Lease                     # a live legacy entry (transition only)
    age_secs: float

def sweep_lock(tier: Tier, environ=os.environ) -> Path        # '<lease path>.<tier>.sweep.lock', one per tier
def sweep_held(tier: Tier, environ=os.environ, *, log: Callable[[str], None]) -> bool
    # non-blocking fcntl.flock(LOCK_SH | LOCK_NB) on a fresh descriptor, released at once;
    # a missing lock file is not held; an unexpected OSError logs one WARNING and reads as not held
def read_leases(environ=os.environ, *, log: Callable[[str], None]) -> Mapping[Tier, Lease]
def classify(tier: Tier, *, now: float, environ=os.environ, log=...) -> LeaseReading
def acquire(tier: Tier, *, pid: int, units: Sequence[str] = (), now: float,
            environ=os.environ, log=...) -> Acquired | Refused  # records the entry; the caller already holds the tier's sweep lock
def set_current_units(tier: Tier, *, pid: int, units: Sequence[str], environ=os.environ) -> bool
def release(tier: Tier, *, pid: int, environ=os.environ) -> bool
def describe(lease: Lease | None, *, now: float) -> str # 'tier T, pid N, units U, age Xs', or 'holder entry not yet written' for None — every skip line and --report use it
```

File body: `{"schema": 1, "tiers": {"<tier>": {"pid": int, "started_ts": float, "current_units": [str]}}}`.
Two kinds of lock, never confused:
- each tier's **sweep lock**, `sweep_lock(tier)`, held by a restart script for its whole life;
- one **body lock**, `<lease path>.lock`, taken by `acquire`, `set_current_units` and `release` for a
  single read-modify-write and never held across a wait or a sleep. Body writes go through
  `shared.safe_io.atomic_write_text` under it; there is no hand-rolled temp-and-rename
  (`tests/scripts/test_atomic_write_regrowth.py::test_no_unapproved_renamers_in_source_trees` rejects one
  in `shared/src`). `read_leases` takes neither lock.

- **L1** At most one sweep per tier. Before anything else, each restart script opens its tier's
  sweep lock on fd 9 and takes an exclusive kernel lock: `exec 9>"$(… path sweep-lock --tier T)"` then
  `flock -w "$LEASE_WAIT_SECS" 9` (`/usr/bin/flock`, util-linux 2.39.3, already used by
  `scripts/deploy/enable_laptop_persistent_worktree.sh`). The default wait is 5 s, so a reader's instant
  shared probe never refuses a starting sweep; `--lease-wait SECS` raises it, capped at the tier's
  max-age. On timeout the script prints `lease show --tier T`'s holder line and exits 75. The
  kernel releases the lock on every exit, SIGKILL included, so there is no displacement rule, no pid
  identity and no dependence on the drain arithmetic. The script closes fd 9 for any detached child it
  starts (none today). `acquire` exits 2 when the tier's sweep lock is not held (a caller that skipped
  the lock).
- **L2** A tier is in flight for every reader iff `classify(...).held`, which is true iff its sweep lock is
  held (`sweep_held`): the same predicate the writer uses (heuristic 11). The entry supplies pid, units
  and age for skip lines, `--report` and the current-units skip. Held past the tier max-age is
  `OVERRUN`: still in flight, so the coordinator defers, the staleness backstop skips and liveness keeps
  skipping `current_units`; the age bound only pages (C4). A legacy entry (transition branch) is judged
  by `shared.proc_liveness.pid_alive(pid)` and `0 <= now - started_ts < lease_max_age_secs(tier)` as
  today, giving `LIVE`, `HOLDER_GONE` or `EXPIRED`.
- **L3** Tiers are independent: separate sweep locks, separate entries. Holding one never refuses,
  releases or rewrites another's.
- **L4** Readers fail open on the body: a missing file is silent; a corrupt file, unreadable file or bad
  entry logs one WARNING naming the path and reads `UNREADABLE`, in flight iff the sweep lock is held.
  `acquire` over a corrupt body rewrites it and logs the displaced bytes' length; over an entry a dead
  holder left (`HOLDER_GONE`) it replaces the entry and returns it as `displaced`.
- **L5** `release` and `set_current_units` act only on an entry whose pid matches. Otherwise they
  return False and change nothing. The file is removed when no tier remains.
- **Transition branch:** a legacy body (top-level `pid`, `started_ts`, `current_unit`) reads as the
  orchestrator-fleet entry, judged per L2. `acquire` returns `Refused` (75) while it is `LIVE`, because a
  pre-γ sweep takes no sweep lock. It is deleted in ω (decision 3).

CLI (bash's only interface):
- `lease acquire --tier T --pid P [--units U …]` exits 0 and prints the recorded entry as one JSON line.
  It exits 75 and prints `REFUSED: <describe(holder)>` on stderr only for a live legacy entry, and 2 when
  the tier's sweep lock is not held.
- `lease set-units --tier T --pid P U …` exits 0, or 1 when P is not the holder.
- `lease release --tier T --pid P` exits 0 (released, or nothing held), or 1 when another pid holds it.
- `lease max-age --tier T` prints `lease_max_age_secs(T)` as an int and exits 0.
- `lease show [--tier T]` prints one `FLEET-LEASE: <tier> <LeaseState> …` line per tier, through `classify`.
- `path sweep-lock --tier T` prints the tier's sweep lock path (γ adds it to C1's `path` verb).
- **Exit status contract.** 0 = done or recorded, 75 = refused (`acquire`, and the script itself when
  `flock -w` times out), 1 = not the holder (`set-units`, `release`), 2 = usage error or sweep lock not
  held. Any other status is an infrastructure failure. The bash callers abort before any drain request,
  print the CLI's stderr and exit 70 on any status outside the verb's documented set. Every call on the
  EXIT/TERM/INT trap path ends `|| true`, so a trap never replaces the status the run earned (a refused
  run keeps 75).

The config field `orchestrator_restart_lease_max_age_secs` is deleted in γ: the orchestrator
coordinator and `DrainIdentity` (built in `Harness.__init__`, `orchestrator/src/orchestrator/fleet_drain.py::DrainIdentity`)
both read `lease_max_age_secs(Tier.ORCHESTRATOR_FLEET)`, so no yaml can move one bound without the other
(INV-9). The field is flat today, and S16's config-shape leaf ζ1 does not move it (lead OQ-7 A; S16
records the deletion as its sitting point S16-m14).

### C3 — the host reader (`scripts/_systemd.py`; lands in β)

```python
Runner = Callable[[Sequence[str], float], RunResult]   # injected; the default wraps subprocess.run
@dataclass(frozen=True)
class RunResult:
    rc: int | None             # None: timed out or could not execute
    stdout: str
    stderr: str
    failure: str | None        # 'timeout' | 'not-found' | '<ExceptionName>: <msg>'

@dataclass(frozen=True)
class UnitShow:
    unit: str
    props: Mapping[str, str]                        # from one `show --timestamp=unix -p A -p B …` call
    def epoch(self, prop: str) -> int | None        # '@<n>' → n; '', '0', '@0', unparseable → None
    def monotonic_secs(self, prop: str) -> float | None   # µs → s; 0 → None
    def integer(self, prop: str) -> int | None
    def text(self, prop: str) -> str | None

@dataclass(frozen=True)
class SocketUnit:
    socket: str      # orchestrator-reify.socket
    service: str     # orchestrator-reify.service
    port: int        # from `Listen` '127.0.0.1:8100 (Stream)'

class Phase(StrEnum): RESET_FAILED = 'reset-failed'; RESTART = 'restart'
@dataclass(frozen=True)
class ReviveOutcome:
    unit: str
    timed_out: tuple[Phase, ...]
    failed: tuple[Phase, ...]           # non-zero rc
    @property
    def clean(self) -> bool

class Registration(StrEnum): REGISTERED = 'registered'; ALREADY_RUNNING = 'already-running'; FAILED = 'failed'
@dataclass(frozen=True)
class TransientOutcome:
    unit: str
    registration: Registration
    detail: str

class Systemd:
    def __init__(self, *, run: Runner = default_runner, log: Callable[[str], None]) -> None
    def show(self, unit: str, *props: str) -> UnitShow | None
    def running_units(self, glob: str) -> tuple[str, ...]
    def installed_sockets(self, glob: str) -> tuple[SocketUnit, ...]
    def is_enabled(self, unit: str) -> bool
    def is_in_flight(self, unit: str) -> bool
    def revive(self, unit: str, *, no_block: bool, reset_socket: bool) -> ReviveOutcome
    def register_transient(self, unit: str, argv: Sequence[str], *, setenv: Mapping[str, str] = {}) -> TransientOutcome

class JournalLog:     # systemd-cat -t <tag>, bounded 5 s; stderr fallback suppressing (OSError, ValueError)
    def __init__(self, tag: str, *, run: Runner = default_runner) -> None
    def __call__(self, line: str) -> None
    def warning(self, line: str) -> None
```

Error semantics, uniform across the class:
- A failed or timed-out call logs one WARNING with the argv, rc and failure, then returns the fail-safe
  value. `show` returns None, the unit lists return `()`, `is_enabled` returns False ("skip"),
  `is_in_flight` returns False ("say something"; today's `_unit_is_active` direction), and `revive`
  returns the outcome with the phase named. No method raises.
- `installed_sockets` refuses one socket loudly (WARNING naming the unit) when it lacks exactly one
  numeric `Listen` port, and keeps the rest.
- `revive` never issues `systemctl stop` (the socket-activation rule in today's `restart_unit`).
- `is_enabled` reads `systemctl --user is-enabled` with today's semantics.

### C4 — tier spec, host protocol and the two passes (`scripts/_supervision.py`; lands in δ1, extended by δ2a, δ2b, δ3, ε)

```python
@dataclass(frozen=True)
class StaticUnits: units: tuple[str, ...]
@dataclass(frozen=True)
class InstalledSockets: glob: str     # 'orchestrator-*.socket': liveness walks every installed socket's service;
                                      # staleness walks those of them that are running (today's _enumerate_running_units)
UnitSource = StaticUnits | InstalledSockets

class StartProperty(StrEnum): EXEC_MAIN_START = 'ExecMainStartTimestamp'; ACTIVE_ENTER = 'ActiveEnterTimestamp'

@dataclass(frozen=True)
class PortAndActiveState: pass                        # orchestrator: port LISTEN, else ActiveState in {inactive, failed} → dead
@dataclass(frozen=True)
class PortAndHttp: url: str; timeout_secs: float      # fm: port down → 'port-down'; any HTTP response (503 included) → healthy
@dataclass(frozen=True)
class HttpOk: url: str; timeout_secs: float           # dashboard: only a 200 within the timeout is healthy
Probe = PortAndActiveState | PortAndHttp | HttpOk

@dataclass(frozen=True)
class HeartbeatProbe:                                 # ε (ruled OQ-1 B)
    max_age_secs: int        # a check is stale when the heartbeat, counted from the later of its ts_epoch
                             # and the instance start, is older than this (900)
    streak: int              # consecutive stale checks before the unit joins the dead-unit path (3); own persisted streak
    escalation_prefix: str   # 'orchestrator-liveness-heartbeat-': one escalation per wedge revive, filed in the revive's finally
    agent_role: str          # 'orchestrator-watchdog'
    # a missing heartbeat file is never stale (decision 11)

@dataclass(frozen=True)
class Streak: threshold: int; max_age_secs: int       # persisted; instance-boundary expiry always on; max_age_secs 0 = no age expiry
@dataclass(frozen=True)
class ReviveCap: min_interval_secs: int               # per unit, own clock in TierState; <= 0 disables
@dataclass(frozen=True)
class StormEscape:
    max_revives: int; window_secs: int; stop_reviving: bool
    page_sentinel: str; agent_role: str
    # an episode: opens when the unit's revives inside window_secs reach max_revives, and pages once per opening;
    # closes on the unit's first healthy verdict; while open and stop_reviving, no revive, even after the window ages out
@dataclass(frozen=True)
class CrashLoopPolicy: count: int; window_secs: int; page_sentinel: str
    # task 8270's rule, decided by scripts/_watchdog_crash_loop.py::observe_unit (reused, not rewritten):
    # NRestarts samples persisted in TierState; growth >= count within window_secs, or Result=start-limit-hit,
    # pages once per window; start-limit-hit also stops the unit's pass (no reset-failed, no revive)
@dataclass(frozen=True)
class Revive: no_block: bool; reset_socket: bool

@dataclass(frozen=True)
class LivenessPolicy:
    probe: Probe
    wedge_signal: HeartbeatProbe | None               # ε; at its streak the unit joins the dead-unit path
    streak: Streak                                    # the dead-unit streak
    revive: Revive
    revive_cap: ReviveCap | None
    storm_escape: StormEscape | None
    crash_loop: CrashLoopPolicy | None

@dataclass(frozen=True)
class Redeploy: transient_unit: str; argv: tuple[str, ...]; setenv: Mapping[str, str]
@dataclass(frozen=True)
class RemovalWaiver: watched_paths: tuple[str, ...]; grace_secs: int   # task 7198, fleet tier only

@dataclass(frozen=True)
class StalenessPolicy:
    start_property: StartProperty
    watched_paths: tuple[str, ...]
    clock: FleetFile
    min_interval_secs: int
    head_start_secs: int
    redeploy: Redeploy
    removal_waiver: RemovalWaiver | None

@dataclass(frozen=True)
class TierSpec:
    name: str
    units: UnitSource
    startup_grace_secs: int
    state_file: FleetFile                             # a C1 SUPERVISION_STATE entry
    lease_tier: Tier | None
    liveness: LivenessPolicy | None
    staleness: StalenessPolicy | None

class UnitOutcome(StrEnum):
    SKIPPED_DISABLED = 'skipped-disabled'; IN_GRACE = 'in-grace'; LEASED = 'leased'; HEALTHY = 'healthy'
    STREAK_PENDING = 'streak-pending'; CAPPED = 'capped'; ESCAPE_OPEN = 'escape-open'
    CRASH_LOOP_PAGED = 'crash-loop-paged'; REVIVED = 'revived'; ERROR = 'error'
    # a wedge revive is REVIVED with gate 'heartbeat'; its escalation is part of that decision
QUIET_OUTCOMES: frozenset[UnitOutcome]                # {HEALTHY, SKIPPED_DISABLED, IN_GRACE}: silent or bucketed

@dataclass(frozen=True)
class UnitDecision:
    tier: str; unit: str; kind: UnitOutcome; gate: str
    detail: str                                       # the values the deciding gate compared

class StalenessOutcome(StrEnum):
    MIN_INTERVAL = 'min-interval'; HEAD_START = 'head-start'; LEASED = 'leased'; NO_COMMIT = 'no-commit'
    FRESH = 'fresh'; RECHECK_HELD = 'recheck-held'; DELEGATED = 'delegated'
    DELEGATE_FAILED = 'delegate-failed'; ERROR = 'error'
QUIET_STALENESS: frozenset[StalenessOutcome]          # {MIN_INTERVAL, HEAD_START, NO_COMMIT, FRESH}: bucketed

@dataclass(frozen=True)
class TierDecision:
    tier: str; kind: StalenessOutcome; stale_units: tuple[str, ...]; waived: bool; detail: str

@dataclass(frozen=True)
class Page:                                           # one `escalation submit` call; the real host builds the argv
    sentinel: str        # --task, e.g. orchestrator-crash-loop-<unit>
    severity: str        # --severity; 'critical' for every page this PRD adds
    category: str        # --category; 'infra_issue'
    agent_role: str      # --agent-role; 'orchestrator-watchdog' | 'harness-dashboard-watchdog'
    summary: str
    detail: str

@dataclass(frozen=True)
class UnitState:
    streak: int; streak_instance: int | None; streak_started: float | None
    heartbeat_streak: int
    revive_epochs: tuple[float, ...]                  # revives inside the escape window
    last_revive: float | None                         # the revive cap's clock
    escape_open: bool
    restart_samples: tuple[tuple[float, int], ...]    # (epoch, NRestarts) inside the crash-loop window
    last_page: Mapping[str, float]                    # page sentinel → epoch, for once-per-window

@dataclass(frozen=True)
class TierState:
    schema: int
    units: Mapping[str, UnitState]
    last_page: Mapping[str, float]                    # tier-level page sentinel → epoch (the lease-overrun page)

class SystemdPort(Protocol):                          # C3's Systemd as the engine sees it
    def show(self, unit: str, *props: str) -> UnitShow | None
    def running_units(self, glob: str) -> tuple[str, ...]
    def installed_sockets(self, glob: str) -> tuple[SocketUnit, ...]
    def is_enabled(self, unit: str) -> bool
    def is_in_flight(self, unit: str) -> bool
    def revive(self, unit: str, *, no_block: bool, reset_socket: bool) -> ReviveOutcome
    def register_transient(self, unit: str, argv: Sequence[str], *, setenv: Mapping[str, str] = {}) -> TransientOutcome

class Host(Protocol):
    systemd: SystemdPort
    def port_listening(self, port: int) -> bool | None
    def http_status(self, url: str, timeout_secs: float) -> int | None      # None: no response
    def heartbeat_age(self, unit: str) -> float | None                      # ε; via C1 HEARTBEAT_DIR; None = no file
    def newest_commit_epoch(self, paths: Sequence[str]) -> int | None
    def removed_modules_since(self, epoch: int, paths: Sequence[str]) -> tuple[str, ...]
    def now(self) -> float
    def monotonic(self) -> float
    def read_clock(self, file: FleetFile) -> float | None
    def read_state(self, spec: TierSpec) -> TierState
    def write_state(self, spec: TierSpec, state: TierState) -> bool
    def lease_reading(self, tier: Tier) -> LeaseReading                     # C2 classify
    def file_page(self, page: Page) -> bool
    def log(self, line: str) -> None

def liveness_pass(spec: TierSpec, host: Host) -> tuple[UnitDecision, ...]
def staleness_pass(spec: TierSpec, host: Host) -> TierDecision
def report(specs: Sequence[TierSpec], host: Host, out: TextIO) -> int        # 0 all fresh, 1 any stale
```

Gate order. One sequence each; a gate that has no policy in the spec is a no-op.
- **Liveness, per unit of the unit source.** enabled → crash-loop sample and alarm (every tick, before
  the grace gate, task 8270 item 1: record `NRestarts`/`Result` in `TierState`; growth ≥ `count` within
  `window_secs`, or `Result=start-limit-hit`, pages once per window; on start-limit-hit the unit's pass
  stops here with no reset-failed and no revive) → startup grace (resets the unit's streaks) → in a
  held tier lease's `current_units` (any held state, `OVERRUN` included) → probe and wedge signal (the heartbeat streak advances on a stale check,
  clears on a fresh one, and at its threshold the unit joins the dead-unit path with gate `heartbeat`) →
  dead-unit streak (persisted, with instance-boundary and max-age expiry; a healthy verdict clears it and
  closes an open escape episode) → storm escape (at the ceiling: open the episode and page once; while
  open and `stop_reviving`, stop) → revive cap → revive. Bookkeeping runs in a `finally`: the cap clock
  and revive epoch are recorded and the streak consumed, on the raising path too (task 4131's rule); a
  revive whose gate is `heartbeat` also files its one wedge escalation there, carrying the
  `ReviveOutcome`.
- **Staleness, per tier.** lease overrun (evaluated every tick, before any clock gate: an `OVERRUN`
  reading pages `<tier>-lease-overrun`, critical, `infra_issue`, naming `describe(holder)`, once per
  hour per tier, recorded in `TierState.last_page`; the pass continues) → min interval on the tier
  clock → head start (`min_interval + head_start_secs`, task 4754) → removal waiver (crosses the two
  clock gates only, never the lease) → held lease (skip) → newest commit over the watched paths, plus
  commit grace → per running unit of the unit
  source: enabled → startup grace → start property older than that commit → re-check of the min
  interval and the lease (read-then-act, task 4755) → delegate once via `register_transient`.

Invariants:
- **I5** `liveness_pass` never reads or stamps a deploy clock and never delegates. `staleness_pass`
  never calls `revive`.
- **I6** No state outlives a tick except the tier's `TierState` file, the deploy clocks and the lease.
- **I7** `report` makes no mutating host call. `FakeHost` counts mutating calls and asserts zero.

Error semantics. An exception while evaluating one unit is logged with tier, unit and gate. It becomes
`UnitDecision(kind=ERROR)` and stops neither the other units nor the next pass. Actionable decisions
always write one journal line naming tier, unit, gate and the values compared (INV-2): every revive,
page and error, a failing probe below the streak threshold, and every skip for a lease, cap, open escape
or crash loop. The outcomes in `QUIET_OUTCOMES` and `QUIET_STALENESS` (healthy, disabled, inside grace,
min-interval or head-start holds, nothing newer, all fresh) are silent or bucketed by
`SKIP_LOG_INTERVAL_SECS` as today. `FakeHost` asserts both halves.

### C5 — one code closure per tier, and the fused-memory coordinator's inputs (`shared/fleet_coordination.py`; lands in ζ; ruled OQ-2 B)

```python
WATCH_CLOSURE: Mapping[Tier, tuple[str, ...]]
# member src dirs the tier's process loads, plus the packaging files that change its environment
```

- **W1** For each tier, the `<member>/src/` entries equal the AST import closure of the tier's root
  member (fused-memory or orchestrator) at member granularity. Imports count whether module-level,
  function-local or TYPE_CHECKING, as in `tests/scripts/test_fm_watched_paths_import_closure.py`
  today. The packaging entries are the root `uv.lock` and each closure member's `pyproject.toml`.
  `orchestrator/uv.lock` is dropped.
- **W2** The default of `OrchestratorConfig.orchestrator_restart_watch_prefixes` (the flat field until
  S16 ζ1 moves it) is `list(WATCH_CLOSURE[Tier.ORCHESTRATOR_FLEET])`, and the committed
  `dark-factory-orchestrator.yaml` does not set the key.
- **W3** The default of `OrchestratorConfig.fused_memory_restart_watch_prefixes` (the flat field until
  S16 ζ1) is `list(WATCH_CLOSURE[Tier.FUSED_MEMORY])`, and a test pins it equal to the table row.
- **W4** The watchdog's staleness rows and the removal waiver read the table. `WATCHED_PATHS`,
  `FM_WATCHED_PATHS` and `MODULE_REMOVAL_WATCHED_PATHS` go.
- **W5** `OrchestratorConfig.fused_memory_restart_min_interval_secs` (new, restart-only: not in
  `RELOADABLE_FIELDS`, like its sibling `orchestrator_restart_min_interval_secs`) defaults to C1's
  `FM_ON_MERGE_MIN_INTERVAL_SECS`; a test pins the field default to the constant, and
  `shared/tests/test_fleet_coordination.py` pins the constant's value (14,400 until θ, 3,600 after).
- **W6** `Harness._build_service_restart_coordinator` passes `min_interval_secs` from W5,
  `state_path=resolve(FM_ON_MERGE_CLOCK)`, `stamp_clock_on_fire=True` and `script_args=['--drain']`, and
  the fm coordinator's force-fire never bypasses the interval (today's
  `StaleServiceRestartCoordinator.maybe_restart` rule). It also passes a `restart_executor` that registers
  `fm-on-merge-restart-<n>.service` through `schedule_detached_systemd_restart`, with an
  `on_failure_escalation` (decision 16). A harness test pins the interval, the clock, the transient unit
  and the `--drain` argument at the fired argv.
- **W7** `OrchestratorConfig.fused_memory_restart_on_merge_enabled` defaults to false, and the committed
  `dark-factory-orchestrator.yaml` sets it true (decision 16).

## Boundary-test sketch

Rows 1–22c and 26–28 are unit-level, against fakes (rows 26–27 run the real bash script with a fake
`systemctl` and `journalctl` on PATH, row 26b included). Rows 23–25 cross real processes and are ω's signal.

| # | Seam | Side | Scenario | Pre | Post |
|---|---|---|---|---|---|
| 1 | C1 | bash ↔ Python | `python3 -m shared.fleet_coordination path lease` vs `resolve(REDEPLOY_LEASE)` | env override set / unset; run from a linked worktree | identical absolute paths; unset resolves under the main checkout |
| 2 | C1 | import | `/usr/bin/python3 -I` with only `shared/src` on `sys.path` imports the module | clean interpreter | no third-party module in `sys.modules` |
| 3 | C2 ↔ bash | writer | two restart scripts of the same tier start concurrently | no sweep | exactly one proceeds; the other exits 75 after the default 5 s wait, printing the holder's pid, units and age |
| 3b | C2 ↔ bash | writer | a second script with `--lease-wait 30` while the holder exits after 10 s / holds past 30 s; meanwhile the holder runs `lease set-units` and `lease release` | live sweep | proceeds once the holder exits / exits 75 at 30 s; the holder's `set-units` and `release` each return within 1 s while the waiter waits |
| 4 | C2 | writer | `acquire` on two different tiers | each caller holds its own tier's sweep lock | both `Acquired`; the file holds both entries |
| 5 | C2 ↔ bash | writer | holder SIGKILLed mid-sweep / holder alive past the tier max-age | stale entry / old live sweep | the kernel freed the lock: the next script proceeds at once and `acquire` returns `Acquired(displaced=<old>)` with a WARNING / the next script exits 75 and every reader reads `OVERRUN` |
| 6 | C2 | writer | `release` / `set_current_units` by a non-holder pid | live entry | returns False; file byte-identical |
| 6b | C2 | writer | `lease acquire` while the tier's sweep lock is not held | no sweep | exit 2; file unchanged |
| 7 | C2 | reader | lock free and no entry; lock free with an entry; lock held with no entry, a fresh entry, an entry past max-age, a future-dated entry, a corrupt body, NaN `started_ts`; legacy entry with a live pid, a dead pid, past max-age | each | `classify` returns NONE, HOLDER_GONE, LIVE with `lease` None (`describe` prints "holder entry not yet written"), LIVE, OVERRUN, FUTURE_DATED, UNREADABLE, UNREADABLE (all six held), LIVE (held), HOLDER_GONE, EXPIRED; one WARNING per corrupt read |
| 8 | C2 | reader | legacy single-tier body (transition) | old writer's body | read as an orchestrator-fleet entry with `current_units=(current_unit,)` |
| 9 | C2 ↔ bash | writer | a second `restart-all-orchestrators.sh --drain` while one holds the lease (fake systemctl) | live lease | exit 75 before any drain request file exists; holder's lease and drain requests untouched; the EXIT trap's `lease release` (exit 1, not the holder) does not change the 75 |
| 9b | C2 ↔ bash | writer | the CLI is broken (`PYTHONPATH` points at an empty directory) for each restart script | no lease | exit 70; no drain request; no lease file; the CLI's stderr echoed |
| 10 | C2 ↔ bash | writer | `restart-fused-memory.sh --drain` while the orchestrator-fleet lease is held | fleet tier held | proceeds (tier independence); its entry has no current unit during the drain wait, names `fused-memory.service` only from the restart step to the health check, and is removed at EXIT |
| 10b | C2 ↔ bash | writer | `restart-fused-memory.sh --rebind-socket` while another `restart-fused-memory.sh` (the staleness redeploy's) holds the fused-memory sweep lock | fm tier held | exits 75 before any SIGUSR1, socket install or restart; no `refuse_while_staleness_redeploy_runs` probe remains (task 9389's drained-and-refused state cannot occur) |
| 11 | C2 ↔ coordinator | reader | `StaleServiceRestartCoordinator` fires with a live lease on its own tier / on the other tier only | each | defers keeping `pending` / fires |
| 12 | C3 | reader | `show` over canned stdout: `@1794…`, `0`, `@0`, a missing prop, rc 1, timeout | fake runner | typed values / None; exactly one WARNING per failed call |
| 13 | C3 | actuator | `revive` where the restart phase times out | fake runner | `ReviveOutcome.timed_out == (RESTART,)`; the caller's journal line says the phase timed out (task 5700) |
| 14 | C3 | reader | `installed_sockets` over `list-unit-files` + `Listen` output, one socket with two `Listen` lines | fake runner | ports parsed; the two-Listen socket refused by name, others kept |
| 15 | C4 fm | liveness | verdicts unhealthy ×2 → no revive; ×3 → revive and cap; within cap → withheld, streak kept; healthy → cleared; new instance → streak expired | `FakeHost` | recorded actions match; bookkeeping runs when `revive` raises |
| 15b | C4 | journal | a healthy tick; ticks inside grace and inside the min interval; a revive, a page, an error and a lease skip | `FakeHost` | healthy writes nothing; the holds write at most one bucketed line per `SKIP_LOG_INTERVAL_SECS`; each actionable decision writes one line naming tier, unit, gate and compared values |
| 16 | C4 fm | staleness | min interval / head start / live fm lease / clock stamped mid-pass / disabled unit / unit inside grace / stale | `FakeHost` | skip / skip / skip / skip (re-check) / no delegation / no delegation / one delegation with `FM_DEPLOY_CLOCK` in `setenv` |
| 17 | C4 fleet | liveness | installed sockets {A, B, C}, C's service disabled; a fourth socket appears | `FakeHost` | A and B probed, C skipped; the fourth probed on the next tick with no code change; `report`'s LIVENESS column shows A and B probed and C disabled |
| 18 | C4 fleet | liveness | a dead unit in the lease's `current_units` / a dead unit outside it | live lease | skipped / revived |
| 19 | C4 fleet | staleness | removal due while the min interval holds; lease live / a disabled unit whose start precedes the newest commit / a running unit inside startup grace | `FakeHost` | waiver crosses the clock gates, never the lease / no fleet redeploy / no fleet redeploy |
| 19b | C4 | staleness | a stale fleet with a live holder past max-age (`OVERRUN`), for 90 one-minute ticks / the same while the min interval still holds | `FakeHost` | no delegation on any tick; exactly one `orchestrator-fleet-lease-overrun` page per hour, naming the holder / the same: exactly one overrun page per hour and no delegation |
| 20 | C4 fleet | liveness | NRestarts grows by 5 in 1,800 s; `Result=start-limit-hit` (8270's rows), unit outside grace | `FakeHost` | one page per window; start-limit-hit gets no reset-failed or revive |
| 20b | C4 fleet | liveness | a unit 30 s after start (inside grace) whose NRestarts grew by 5 in 1,800 s | `FakeHost` | exactly one `orchestrator-crash-loop-<unit>` page |
| 21 | C4 dashboard | liveness | 3 failed probes, grace 60 s, 3 revives in 1 h; then the window ages out with the probe still failing; then one healthy probe | `FakeHost` | revive on the third failure; at the ceiling one page and revives stop; still no revive after the window ages out; the healthy probe closes the episode (today's B-rows) |
| 21b | C4 dashboard | state | today's `state.json` body with `ceiling_open: true` and 3 restarts | `FakeHost` | read as an open episode with 3 revive epochs; no revive, no new page |
| 22 | C4 ε | liveness | heartbeat 950 s old on checks 1 and 2, then check 3 / heartbeat 30 s / heartbeat file absent past grace / a heartbeat left by the previous instance, 950 s old, 200 s after the unit's restart / fm revives 3 in 6 h | `FakeHost` | nothing on checks 1–2; on check 3 exactly one revive and exactly one `orchestrator-liveness-heartbeat-<unit>` escalation (critical, `infra_issue`, `orchestrator-watchdog`) whose detail names the age, the 900 s threshold and the recipe / nothing / never stale, one bucketed WARNING / not stale (age counted from the instance start) / fm pages once and keeps reviving under its cap |
| 22b | C4 ε | liveness | check 3's revive raises / times out in its restart phase | `FakeHost` | the escalation is still filed, naming the failed phase; the cap clock and revive epoch are recorded |
| 22c | C4 ε | liveness | the unit stays wedged after each revive, for 60 one-minute checks | `FakeHost` | one escalation per revive, revives at most one per 900 s (8270's cap); at the third revive in an hour one `orchestrator-liveness-ceiling` page and no further revive or wedge escalation |
| 23 | C2 × C4 | cross | a real bash sweep (fake `systemctl` on PATH) holding the lease, and one real watchdog tick with `_watchdog_host` | sweep mid-unit | the tick skips the sweep's current units and its staleness pass logs "already in flight" |
| 24 | C5 | config ↔ watchdog | the closure table vs the AST closure; both coordinators' `OrchestratorConfig` defaults; the interval default; the committed yaml | committed tree | W1–W5 and W7 hold |
| 25 | η | reclaim | roots {R1 with 2 old parkings, R2 with no parking dir, R3 not a git toplevel} | env lists all three | R1 swept, R2 a no-op, R3 `refused_target`; one report per root, exit 0 |
| 26 | ι | bash | `restart-fused-memory.sh --drain` with a fake journal that prints the drained marker after 3 s / never prints it, with `DRAIN_TIMEOUT=10` in the environment (the override today's tests use) | fake `systemctl`, `journalctl`, `curl` | restarts after the marker, files nothing / restarts at 10 s and files exactly one `fused-memory-drain-timeout` escalation naming the bound and the `recon_busy` runs; a filing failure, or a fake `escalation submit` that hangs, is logged and the restart still runs (the hang is cut at the submit bound) |
| 26b | ι | bash | after SIGUSR1, during the drain wait: the script gets SIGTERM / SIGINT / a failing command under `set -e` | fake `systemctl`, `journalctl` | each exit issues exactly one `systemctl --user restart --no-block fused-memory` and logs why; exit statuses stay 143 / 130 / the failing status; a refusal before SIGUSR1 (exit 75) issues no restart (`test_drain_is_never_abandoned_after_the_signal`) |
| 27 | ι | bash | no `DRAIN_TIMEOUT` in the environment | fake CLI answering `drain max-wait` | the script waits the CLI's value; a broken CLI exits 70 before SIGUSR1 |
| 28 | ζ | coordinator | two closure merges 30 min apart, then one 5 h later; a merge after a fire whose clock file is corrupt | `StaleServiceRestartCoordinator` built by the harness builder, fake clock, recording executor | one fire, registered as a transient unit `fm-on-merge-restart-<n>.service` whose argv ends `scripts/restart-fused-memory.sh --drain` and which carries an on-failure escalation; the second held by the 14,400 s interval; the third fires under a new unit name; the clock is persisted to `FM_ON_MERGE_CLOCK`; a corrupt clock reads as no fire (fail open), as `_load_last_fire_wall` does today |

## Cross-PRD relationship

Edge notation: "X → Y" means X depends on Y (Y lands first).

| Other PRD / task | Direction | Seam / file | Owner | Order |
|---|---|---|---|---|
| S14 `plans/review-all-s14-shared-leaves-prd.md` | consumes | `shared.project_layout.default_project_root`, `known_project_roots` (S14 α 6710, done) | S14 | none needed (landed) |
| task 6726 (the shared spelling-guard test, S14, pending) | consumes | ω adds the fleet-state family to `shared/tests/test_shared_leaf_spellings.py` | S14 | ω → 6726 (lead OQ-5 A) |
| S25 `plans/review-all-s25-scripts-instruments-prd.md` C1 | consumes | `scripts/_checkout.py` (6758, done) | S25 | none needed |
| S19 C1 `shared.proc_liveness.pid_alive`, task 6615 (done, merged 2026-11-11, `f750373949`) | consumes | γ binds C2 and the watchdog to it and deletes the watchdog's copy and `tests/scripts/test_watchdog_pid_alive_parity.py` (cross-stream "one public liveness check", Leo 2026-11-13) | S19 (6615) | γ → 6615, satisfied |
| S19 η2 (wave 3: session_registry split, step 2) | none | `orchestrator/src/orchestrator/service_restart.py` | — | no edge: since 6615, `service_restart.py` has no `session_registry` reference, so S19 η2 does not declare the file (S19's fold); the earlier "S19 η2 → γ" ask is withdrawn |
| S16 ζ1 (wave 3: supervision knobs become submodels) | ordered (lead OQ-7 A) | `config.py`, `harness.py`, `service_restart.py`, `scripts/orchestrator-watchdog.py`: ζ1 moves the restart knobs into `service_restart[…]` and `fleet_redeploy` | S16 owns the config shape; S24 owns the defaults' values (C1 `LEASE_MAX_AGE_SECS`, `FM_ON_MERGE_MIN_INTERVAL_SECS`, C5 `WATCH_CLOSURE`) | S16 ζ1 → γ and S16 ζ1 → ζ (this plan's second and third tasks, so ζ1 never waits on the engine leaves); γ deletes `orchestrator_restart_lease_max_age_secs`, so ζ1 does not move it (S16-m14); ζ1 carries ζ's new `fused_memory_restart_min_interval_secs` into the fused-memory sub-section with its default still naming C1's constant (cross-stream, rulings §S16/S24), keeps both watch-prefix defaults naming `WATCH_CLOSURE`, and moves `fused_memory_restart_on_merge_enabled` (default false after ζ; the dark-factory yaml's `true` moves with it). θ changes only the constant, so ζ1 takes no edge on θ |
| S15 (wave 3: fused-memory server boundary) | none | `TimeoutStopSec` stays 90 s (S15-1 D); S15's shutdown task lengthens fused-memory's HTTP drain inside its phased shutdown, not the reconciliation drain ι bounds | S15 | no edge; decision 6's arithmetic restates 90 s and needs no change (cross-stream) |
| task 9493 (done, `1d0ae21e44`) and task 9521 (operator reinstall, deterministic human gate, blocked at L2) | ordered | the 1-hour interval applies once the stop fix is in effect on the host | outside every plan | θ → 9493, θ → 9521 |
| task 9492 (pending, medium, parked behind 8528) | xref | moves `merge_lane/liveness.py::newest_content_mtime` off the event loop, the cause of the measured long stalls | outside every plan | none: ε does not wait on it (R-3 = A) |
| startup-hardening δ, task 8270 (done, merged `7ddc80450e`) | ordered | `main()` in `scripts/orchestrator-watchdog.py` and `scripts/_watchdog_crash_loop.py`; its NRestarts sampling, start-limit rule and revive cap become spec policy in δ2a, with the crash-loop step ahead of the grace gate | startup-hardening PRD | ζ → 8270, β → 8270, δ2a → 8270, all satisfied |
| task 9321 (pending, low) | partly subsumed | adds the two missing pairs to `WATCHED`, an interim fix δ2a deletes with `WATCHED` | 9321 | none; if 9321 is still unclaimed when δ2a is claimed, it is cancelled into δ2a (lead action) |
| task 9389 (pending, low) | subsumed | `restart-fused-memory.sh`'s second staleness probe can refuse after SIGUSR1 and leave reconciliation drained | γ | cancelled into γ: γ's sweep lock is taken before anything else and replaces `refuse_while_staleness_redeploy_runs` |
| S6 α, task 9140 (wave 2, pending) | ordered | `orchestrator/src/orchestrator/config.py`, `harness.py`, `OPERATIONS.md` (R54: S6 α first on every shared file) | S6 | γ → 9140, ζ → 9140, η → 9140, ι → 9140 |
| S5 ε, task 9263 (wave 2) | ordered | `orchestrator/src/orchestrator/service_restart.py`: S5 changes the `get_merge_diff_files` call, γ changes the lease and clock constants (disjoint regions) | S5 | 9263 → γ (9263 already waits on 5040 through S5's chain, so the edge costs it nothing). S5 ρ0 9272 follows 9263 in S5's chain. |
| S8 (wave 2) | none | `orchestrator/src/orchestrator/harness.py`: γ edits `Harness._build_orchestrator_restart_coordinator`, `_build_service_restart_coordinator` and the `DrainIdentity` bound in `Harness.__init__` only; no S8 leaf's plan names them | — | no edge; file locks serialize |
| S7 β 9133, S8 δ 9219, S8 θ 9229 (wave 2) | none | `dark-factory-orchestrator.yaml`: ζ deletes the `orchestrator_restart_watch_prefixes` key and adds `fused_memory_restart_on_merge_enabled: true` only | — | no edge (disjoint keys) |
| task 7307 (sweep ceiling from the live unit count) | ordered | `scripts/restart-all-orchestrators.sh`: 7307 re-derives `DRAIN_SWEEP_VERIFY_MAX_WAIT_SECS` from the live unit count | 7307 | 7307 → γ; 7307 amended to read the lease bound through `lease max-age --tier orchestrator-fleet` instead of restating 14400 |
| task 7314 (concurrent restarts within one sweep) | ordered | `scripts/restart-all-orchestrators.sh`; 7314's lease step 3 (the lease names the SET of units) is C2's `current_units` | this PRD (C2) | 7314 → γ; 7314 amended to call `lease set-units`, to drop its watchdog, `service_restart` and `config.py` edits, and to point its step-3 max-age copies at C1's `LEASE_MAX_AGE_SECS` (lead OQ-6 A) |
| task 5719 | subsumed | fm-tier lease | γ | cancelled into γ |
| task 6543 | subsumed | import-closure watched paths for the orchestrator tier and the fm coordinator (ζ delivers both items in full under OQ-2 B) | ζ | cancelled into ζ |
| tasks 5700, 4253 | subsumed | `restart_unit`'s log after timeouts; `log()`'s closed-stderr suppress in both watchdogs | β (C3's `ReviveOutcome`, `JournalLog`) | cancelled into β |
| task 8362 | subsumed | one test loader per hyphen-named watchdog | ω | cancelled into ω |
| task 7517 | partly subsumed | its watchdog half moves to δ2a in socket-template form (`test_socket_port_matches_configured_escalation_port`); its `scripts/run_vllm_eval.py` half stays | 7517 | amended to the `run_vllm_eval.py` half |
| task 7310 | partly subsumed | its `service_restart.py` `FLEET_*_RELPATH` items | γ | amended to drop them |
| task 7139 | partly subsumed | the `scripts/orchestrator-watchdog.py::_unit_start_elapsed_secs` allowlist entry | β | amended to drop that entry and file; the list's count drops to 11 with β |
| task 8295 | ordered | `df_pytest_isolation.py` partition | 8295 | 8295 → α (α's literal block moves first); 8295 amended to drop "keep the STDLIB+PYTEST-ONLY import constraint", which α relaxes to the stdlib `shared` leaves |
| S1 (wave 3) | xref | pages go through `escalation submit`, now including ε's wedge escalation and ι's `fused-memory-drain-timeout` filed from `restart-fused-memory.sh`; S1 routes the CLI through its filing entry point | S1 | none |

## Decomposition plan

Overlay bands: target ~300–1,500 changed LOC, ≤ 10–12 declared files, chained where files are shared.
Several leaves are deletion-dominated: they retire old tests and prose, so their changed-LOC counts
overstate the authored code. Priority is the highest severity of the keys a leaf closes; ι and θ carry
Leo's R-4 additions and take `medium` and `low`. The `high` leaves (β, γ, δ1, δ2a, δ2b, δ3, ε) become
dependencies of trigger-chain gate 6698 (pending) at decompose.
**Priority inversion, recorded:** α is `low` (its one key, `fk-63633270af4e`, is low) and is a
prerequisite of all seven high leaves. As with S11 α in wave 1, the inversion is recorded with no
override (contract §4). The same holds for ζ (`medium`), which precedes β–ε so that the config-shape
plan waits only on this plan's second and third tasks (lead OQ-7 A). It also holds for ι and η (both
`medium`), which precede δ1–ε on `OPERATIONS.md` and run beside β (§Sketch, Order).

**Band exceptions, recorded.** γ declares 19 files: the writer and every reader of the lease body must
change together, because a keyed body read by a pre-γ reader is "no lease" (decision 3); eleven of the
19 are test or doc files whose edit is the lease fixture or one sentence. ζ declares 14 files because
the ruling put the fused-memory coordinator's closure default, interval, clock and drain pin in the
watched-path task (rulings §S24 OQ-2) and S16 ζ1 waits on that task for the new field; nine of the 14
are tests or docs. δ2a declares 14 files, two of them task 8270's crash-loop module and its test. δ1,
δ2a and δ2b exceed ~1,500 changed LOC because they rewrite or retire tests. Measured at `ba17934861`
over `tests/scripts/test_orchestrator_watchdog.py` (then 12,192 lines, 375 tests) with a name-regex
classifier: 132 fm-only tests (2,599 lines) go to δ1; 39 fleet-liveness-only tests (1,022 lines) to
δ2a; 56 fleet-staleness-or-report-only tests (1,780 lines) to δ2b; 51 fleet tests touching both (1,743
lines) are split between δ2a and δ2b by the behaviour they pin; 97 tests (2,250 lines) touch neither or
both tiers and move only where they patch a private name. Task 8270 has since added 19 tests (394 at
`a85cb1ad78`): 18 pin `main()`'s revive cap and crash-loop sampling (δ2a) and one the `--report`
RESTARTS column (δ2b). The critic's AST classifier at `ba17934861` gave 136 fleet-only (4,391 lines),
124 fm-only (2,258) and 115 mixed or neither (2,745); the exception holds under either.

**α — Fleet state paths and lease ages get one home, `shared/fleet_coordination.py` (C1), and every
Python reader uses it** (`task_kind=normal`, priority low, est. ~700 LOC)
- Files: `shared/src/shared/fleet_coordination.py` (new), `shared/tests/test_fleet_coordination.py` (new),
  `scripts/orchestrator-watchdog.py`, `scripts/drain_check.py`, `orchestrator/src/orchestrator/fleet_heartbeat.py`,
  `df_pytest_isolation.py`, `tests/scripts/test_orchestrator_watchdog.py`, `tests/scripts/test_deploy_clock_isolation.py`,
  `tests/scripts/test_fleet_dir_isolation.py`, `orchestrator/tests/test_fleet_heartbeat.py`,
  `orchestrator/tests/test_fleet_dir_isolation.py`, `scripts/tests/test_drain_check.py`
- Deps: none
- Keys: `fk-63633270af4e`.
- Work: write C1 (with `SUPERVISION_STATE`) and its CLI `path` verb. Then repoint the Python readers.
  Task 8295 is ordered after α (lead action).
  - The watchdog: `import _checkout` before its first `shared` import. Its clock, lease and fm-clock
    paths and the lease max-age default come from C1.
  - `drain_check.py` and `fleet_heartbeat.py`: `DEFAULT_FLEET_DIR` goes. Each keeps
    `resolve_fleet_dir(env=None)` as a one-line delegate to `resolve(HEARTBEAT_DIR, env)`, so its
    importers (`orchestrator/src/orchestrator/harness.py`'s module import, `DrainParticipant(fleet_dir=resolve_fleet_dir)`
    and `Harness._write_merge_heartbeat`; the watchdog's `drain_check.resolve_fleet_dir()` calls) are
    untouched. The `DEFAULT_FLEET_DIR` importers (`orchestrator/tests/test_fleet_heartbeat.py`,
    `orchestrator/tests/test_fleet_dir_isolation.py`, `scripts/tests/test_drain_check.py`) re-pin to
    `resolve(HEARTBEAT_DIR, {})`.
  - `df_pytest_isolation.py`: `PROTECTED_DEPLOY_CLOCK_ENV_VARS`, `REDIRECT_ONLY_RELPATHS` and
    `LIVE_FLEET_DIR` derive from `PROTECTED_FILES` and `HEARTBEAT_DIR`, so the three supervision state
    files are redirected from α on. `LIVE_FLEET_DIR` is resolved with an empty environ, so it always
    names the live directory (decision 5's residual for gate clones). Its "STDLIB + PYTEST ONLY"
    paragraph becomes "stdlib, pytest and the stdlib `shared` leaves", with the escalation-stub fact
    stated, and `tests/scripts/test_fleet_dir_isolation.py::test_df_pytest_isolation_stays_stdlib_and_pytest_only`
    adds the `shared` root to its allowlist: any third-party module a `shared` leaf pulled in would still
    appear under its own top-level name and fail the test. Its two "WATCHED table" comments are reworded
    without naming `WATCHED` ("a drift pin against the installed fleet"), so task 8295's partition, which
    follows α, never carries them.
  - `orchestrator/src/orchestrator/service_restart.py` keeps its two relpath constants, and
    `scripts/restart-all-orchestrators.sh` its three defaults, until γ. One interim pin in
    `tests/scripts/test_orchestrator_watchdog.py`, where the three across-tiers pins it replaces live
    (they already import `orchestrator.service_restart` and read the bash script), pins them to C1; γ
    deletes it.
- Deletes: `DEFAULT_FLEET_DIR` (both copies), `test_fleet_lease_path_matches_across_tiers`,
  `test_fleet_deploy_clock_path_matches_across_tiers`, `test_fleet_dir_default_matches_across_tiers`
  (four-way pins collapse into one interim pin), `scripts/tests/test_drain_check.py::test_default_fleet_dir_matches_orchestrator_fleet_heartbeat`,
  and the "FOUR copies of this literal" prose at each former Python copy.
- Signal: boundary rows 1–2. `python3 -m shared.fleet_coordination path deploy-clock --tier
  orchestrator-fleet` on the host prints `/home/leo/src/dark-factory/data/orchestrator/last_redeploy_orchestrator.json`.
  `scripts/orchestrator-watchdog.py --report` is unchanged. `uv run --project escalation pytest
  escalation/tests -q` passes, which shows `df_pytest_isolation` imports `shared.fleet_coordination`
  under escalation's stub.
- G7: INV-5/INV-9 (four copies → one home). INV-13: the readers' producer is unchanged (same files).

**γ — Redeploy lease keyed by tier and acquired under flock; a second concurrent sweep is refused at
the writer (C2)** (`task_kind=normal`, priority **high**, est. ~1,400 LOC; band exception above)
- Files: `shared/src/shared/fleet_coordination.py`, `shared/tests/test_fleet_coordination.py`,
  `scripts/restart-all-orchestrators.sh`, `scripts/restart-fused-memory.sh`, `scripts/orchestrator-watchdog.py`,
  `orchestrator/src/orchestrator/service_restart.py`, `orchestrator/src/orchestrator/harness.py`,
  `orchestrator/src/orchestrator/config.py`, `scripts/tests/test_restart_all_orchestrators.py`,
  `scripts/tests/test_restart_fused_memory.py`, `tests/scripts/test_restart_all_orchestrators.py`,
  `tests/scripts/test_orchestrator_watchdog.py`, `tests/scripts/test_watchdog_pid_alive_parity.py` (deleted),
  `orchestrator/tests/test_service_restart.py`, `orchestrator/tests/test_harness_service_restart.py`,
  `orchestrator/tests/test_harness_fleet_drain.py`, `orchestrator/tests/test_config.py`,
  `OPERATIONS.md`, `CLAUDE.md`
- Deps: α, 9140, 6615 (done; satisfied)
- Keys: `fk-4c21aaf01b82`.
- Work: C2, its `lease` CLI verbs (`lease max-age` included) and the exit-status contract. Tasks 7307,
  7314 and 9263 are ordered after γ (lead actions); task 9389 is cancelled into γ.
  - `restart-all-orchestrators.sh`: before anything else it opens the orchestrator-fleet sweep lock on
    fd 9 and runs `flock -w "$LEASE_WAIT_SECS" 9` (C2 L1, lead OQ-8 D); on timeout it prints the holder line
    and exits 75 before any drain request is written. Its `FLEET_DIR`, `CLOCK_FILE` and `LEASE_FILE`
    defaults come from `python3 -m shared.fleet_coordination path …` with
    `PYTHONPATH="$REPO_DIR/shared/src"`; the env overrides keep working because C1 honours them.
    `lease_acquire`, `lease_set_current_unit` and `lease_release` become calls to the CLI. Any status
    outside a verb's documented set aborts with 70 and the CLI's stderr (decision 2). The EXIT trap
    ordering is kept, every trap-path call ends `|| true`, and the refused run's trap releases nothing it
    does not hold (L5). `current_units` is published per unit (still one unit at a time until 7314). Its
    argument parser accepts `--lease-wait SECS` (today it rejects any extra argument with "ERROR:
    unexpected argument"), which sets `LEASE_WAIT_SECS`, capped at `lease max-age` (lead OQ-9 B).
  - `restart-fused-memory.sh` takes the fused-memory sweep lock the same way, before the `--rebind-socket`
    checks and before any SIGUSR1, and releases it the same way: trap EXIT plus `trap 'exit 143' TERM`
    and `trap 'exit 130' INT`, and the same 70/75 contract. It records its entry with no current unit,
    and runs `lease set-units --tier fused-memory --pid $$ fused-memory.service` just before its restart
    step (the `systemctl --user restart`, or `--rebind-socket`'s stop and start) and keeps it through the
    health check, so C4's current-units gate skips fm only for that step (at most 120 s), never for the
    reconciliation gate or drain wait (decision 17). Its two-axis parser (task 8129: one of
    `--now|--drain`, plus `--rebind-socket`) also accepts `--lease-wait SECS`. The lock makes
    `refuse_while_staleness_redeploy_runs` redundant, since the staleness redeploy runs this same script
    under the same lock: γ deletes it and its comment "The fused-memory tier has no restart lease", so
    `--rebind-socket` during any fused-memory sweep now exits 75 (not 3) before it drains, installs or
    restarts anything, which also removes task 9389's drained-then-refused state. Both scripts spell
    their tier literally in the acquire call (`lease acquire --tier orchestrator-fleet`,
    `lease acquire --tier fused-memory`).
  - The watchdog's `_read_fleet_lease`, `_fleet_lease_age_secs`, `_live_fleet_lease`,
    `_describe_lease`, `_pid_alive` and the lease half of `_format_fleet_lease` become C2 calls
    (`classify` renders the `LeaseState` values). `main()` skips every unit in a held lease's
    `current_units`, and both staleness passes treat any held state, `OVERRUN` included, as in flight;
    the overrun page arrives with the engine (δ1 for fused-memory, δ2b for the fleet), and until then
    `--report` shows `OVERRUN`. `fused_memory_staleness_pass` gains the fused-memory lease gate (5719's
    residual). 6615's parity test goes with the copy.
  - `service_restart.lease_is_live` and its `shared.proc_liveness` import (added by 6615) go.
    `StaleServiceRestartCoordinator` takes `lease_tier: Tier | None` in place of `lease_path` and
    `lease_max_age_secs`, and resolves the clock through C1. The fm coordinator gets
    `Tier.FUSED_MEMORY`.
  - `harness.py`: the two builders pass the tiers, and `Harness.__init__`'s
    `DrainIdentity.from_environ(os.environ, max_age_secs=…)` reads
    `lease_max_age_secs(Tier.ORCHESTRATOR_FLEET)` (about 8 lines).
  - `config.py`: the field `orchestrator_restart_lease_max_age_secs` is deleted (C2's last paragraph).
    `OrchestratorConfig` ignores unknown yaml keys, and no host yaml sets this one (measured at
    `a85cb1ad78` over `~/.config/orchestrator/` and every `/home/leo/src/*/dark-factory-orchestrator.yaml`).
    `orchestrator/tests/test_config.py`'s default and reload-tier pins of the field go; the
    `spec_set` mocks in `orchestrator/tests/test_harness_fleet_drain.py` and
    `orchestrator/tests/test_harness_service_restart.py` stop setting it.
  - `--report` prints one `FLEET-LEASE:` line per tier.
  - CLAUDE.md §Orchestrator Fleet Redeploy says "the fused-memory tier has no lease at all" and names
    the deleted field; γ rewrites both sentences. OPERATIONS.md's three mentions of
    `orchestrator_restart_lease_max_age_secs` (§8 and "Reading a drain") point at C1's
    `LEASE_MAX_AGE_SECS` / `lease max-age`, and its two "the fused-memory tier has no (restart) lease"
    sentences (§8 and the socket-activation `--rebind-socket` paragraph) are rewritten. In that paragraph
    "It refuses with exit 3 while `fm-staleness-redeploy.service` runs" becomes exit 75, during any
    fused-memory sweep, before the gate (lead ruling of 2026-11-12). The script's own usage header, which
    says "(exit 3)", is rewritten the same way. OPERATIONS.md §8
    gets the refusal, the 70/75 statuses, the `OVERRUN` page and the transition notes (decision 3), and
    its explicit-deploy pattern (`scripts/restart-all-orchestrators.sh --drain`, as tasks 2776/5049/7175
    did) gains `--lease-wait`.
- Deletes: `service_restart.FLEET_LEASE_RELPATH`, `FLEET_DEPLOY_CLOCK_RELPATH`,
  `DEFAULT_FLEET_LEASE_MAX_AGE_SECS`, `lease_is_live`; `OrchestratorConfig.orchestrator_restart_lease_max_age_secs`;
  the watchdog's five lease helpers and `_pid_alive`; `tests/scripts/test_watchdog_pid_alive_parity.py`;
  the bash `mktemp … mv -f` writer and its hard-coded path defaults; α's interim pin;
  `test_fleet_lease_max_age_matches_config_default` and
  `test_coordinator_lease_max_age_default_matches_the_config_default`;
  `restart-fused-memory.sh::refuse_while_staleness_redeploy_runs` and its exit 3 (8129's tests of it
  become lock tests).
- Signal: boundary rows 3–11 and 10b. Live, on the host: during a running `--drain` sweep,
  `flock -n "$(python3 -m shared.fleet_coordination path sweep-lock --tier orchestrator-fleet)" true`
  exits 1 (the kernel lock is held), and `scripts/orchestrator-watchdog.py --report` shows the holder on the
  orchestrator-fleet `FLEET-LEASE:` line.
- G7: INV-1 (the exclusion is machine-checked at the writer), INV-2 (a refusal names tier, pid, units
  and age), INV-6 (a sweep is in flight iff its kernel lock is held, for writer and readers alike),
  INV-7 (every hold ends with the holder's process; an overrun of the tier max-age pages),
  INV-9 (one home for the max-age), INV-11 (refusal is exit 75, a broken CLI exit 70, never a silent 0;
  corrupt reads WARN).

**β — `scripts/_systemd.py`: one host reader for both watchdogs (C3); the dashboard watchdog's copies
and the five hand parsers go** (`task_kind=normal`, priority **high**, est. ~900 LOC)
- Files: `scripts/_systemd.py` (new), `tests/scripts/test_systemd_host.py` (new), `scripts/orchestrator-watchdog.py`,
  `scripts/dashboard-watchdog.py`, `tests/scripts/test_orchestrator_watchdog.py`, `tests/scripts/test_dashboard_watchdog.py`,
  `shared/tests/silent_fallthrough_allowlist.py`, `shared/tests/test_shared_governed_lists.py`
- Deps: ζ, 8270 (done; satisfied)
- Keys: `fk-906957d2ec84` (closes its parsers, same-named-defs and stale-prose measurements).
- Work: C3.
  - The orchestrator watchdog's systemctl helpers (`_unit_start_elapsed_secs`, `_unit_start_epoch`,
    `_unit_active_enter_epoch`, `_unit_active_state`, `is_unit_enabled`, `_unit_is_active`,
    `_enumerate_running_units`, `restart_unit`, `_register_transient_unit`'s runner, and 8270's
    `_unit_restart_status`, the `NRestarts`/`Result` reader `main()` takes as `read_status`) keep their
    names for now. Their bodies become one call into `Systemd` each, so the existing tests still reach
    them; δ1, δ2a and δ2b delete them.
  - `log` and `_JournalLog` become `JournalLog`, with the `(OSError, ValueError)` suppress from 4253.
  - Each caller of `restart_unit` logs the `ReviveOutcome`, never an unconditional "restart issued"
    (5700).
  - `dashboard-watchdog.py` drops its `log`, `_JournalLog`, `restart_unit` (it becomes
    `revive(no_block=False, reset_socket=False)`, the dashboard's contract) and
    `_unit_active_enter_epoch`, along with the stale "three-phase" prose.
  - The silent-fallthrough allowlist entry for `_unit_start_elapsed_secs` is deleted: the swallow now
    logs a WARNING in `Systemd.show`. `shared/tests/test_shared_governed_lists.py`'s
    `default_covers` for that list drops from 12 (measured at `a85cb1ad78`) to 11 (or by one from
    whatever 7139 has left), since `shared/src/shared/governed_exceptions.py` raises
    `MalformedDeclaration` when the list shrinks and the literal does not.
- Deletes: 5 `split("=", 1)` parsers (4 in `orchestrator-watchdog.py`, 8270's among them, and 1 in
  `dashboard-watchdog.py`, measured at `a85cb1ad78`), 4 of dashboard-watchdog's 5
  same-named copies (`probe_health` goes in δ3), the 7 "monkeypatch the module global" docstring clauses
  where their functions shrink.
- Signal: boundary rows 12–14. The journal tag is unchanged: `journalctl --user -t orchestrator-watchdog
  --since -1h` and `-t dashboard-watchdog` still show the lines they showed before β. `--report` output
  is unchanged except for any revive-outcome wording.
- G7: INV-11 (each probe failure is logged, no silent `None`), INV-12 (the allowlist entry is retired,
  not re-owned).

**δ1 — Supervision engine with one liveness and one staleness pass (C4); the fused-memory tier moves
onto it** (`task_kind=normal`, priority **high**, deletion-heavy; band exception above)
- Files: `scripts/_supervision.py` (new), `scripts/_watchdog_host.py` (new), `tests/scripts/fake_watchdog_host.py` (new),
  `tests/scripts/test_supervision.py` (new), `scripts/orchestrator-watchdog.py`, `tests/scripts/test_orchestrator_watchdog.py`,
  `tests/scripts/test_watchdog_dashboard_parity.py`, `scripts/_watchdog_state.py`, `OPERATIONS.md`
- Deps: β, η
- Keys: `fk-906957d2ec84`, `fk-3830e2a6de79` (fm halves).
- Work: δ1 waits on η only for `OPERATIONS.md` (the same-file rule), and through η on ι. C4's types and
  vocabularies, `Host`, `liveness_pass`, `staleness_pass` and `report`, with the
  journal rule of C4's error semantics. `_watchdog_host.py` holds the real host; its state reads and
  writes resolve C1's `SUPERVISION_STATE`. `FakeHost` records every call and every journal line and
  counts the mutating calls. The fm tier becomes one `TierSpec` row in `orchestrator-watchdog.py`:
  - `PortAndHttp` on `/alive`; `Streak(3, 300)`; `ReviveCap(3600)`;
    `Revive(no_block=True, reset_socket=True)`;
  - staleness on `ActiveEnterTimestamp` with fm's clock, 28,800 s and the `fm-staleness-redeploy`
    `bash -c` command, with the per-unit enabled and startup-grace gates;
  - `lease_tier=FUSED_MEMORY`; `state_file=SUPERVISION_STATE['fused-memory']`.

  The fm tier gains the read-then-act re-check (decision 8). `_cli` runs the fm row through the passes.
  `--stamp-fm-deploy-clock` is kept. The row's watched paths are
  `WATCH_CLOSURE[Tier.FUSED_MEMORY]` (C5, landed in ζ). `_print_fused_memory_liveness` becomes `report`'s fm section. 132 fm-only tests
  (2,599 lines, measured at `ba17934861`) are rewritten onto `FakeHost` or retired under decision 13.
- Deletes: `fused_memory_liveness_pass`, `fused_memory_staleness_pass`, `_delegate_fm_restart`,
  `_newest_fm_watched_commit_epoch`, `_within_fm_staleness_head_start`, `_within_fm_deploy_min_interval`,
  `_read_last_fm_deploy_epoch`, `_stamp_fm_liveness_restart_clock`, `_read_last_fm_liveness_restart_epoch`,
  `_within_fm_liveness_restart_min_interval`, the four fm streak helpers, `_fused_memory_liveness_verdict`,
  the fm halves of the sibling pairs, `_neutralize_fm_clock_gates`, and every `wdog, "_…fm…"` patch.
  `data/fused-memory/fm_liveness_streak.json` and `last_liveness_restart_fused_memory.json` are no
  longer written (decision 9).
- Signal: boundary rows 15, 15b and 16. Live: on a healthy host the fm tier writes no journal line per
  tick, any fm line names tier `fused-memory`, and `--report` shows the fm section.
- G7: INV-3 (streak before a revive), INV-10 (the passes are tested through `FakeHost` actions, not
  patched privates).

**δ2a — The orchestrator fleet's liveness runs on the engine; its unit set is derived from installed
`orchestrator-*.socket` units and `WATCHED` goes** (`task_kind=normal`, priority **high**,
deletion-heavy; band exception above)
- Files: `scripts/orchestrator-watchdog.py`, `scripts/_supervision.py`, `scripts/_watchdog_host.py`,
  `tests/scripts/fake_watchdog_host.py`, `tests/scripts/test_supervision.py`, `tests/scripts/test_orchestrator_watchdog.py`,
  `tests/scripts/test_orchestrator_socket_units.py`, `scripts/restart-orchestrator.sh`,
  `skills/factory-init/SKILL.md`, `skills/factory-init/references/supervised-unit.md`,
  `scripts/orchestrator-watchdog.service`, `OPERATIONS.md`, `scripts/_watchdog_crash_loop.py`,
  `tests/scripts/test_watchdog_crash_loop.py`
- Deps: δ1, 8270 (done; satisfied)
- Keys: `fk-c9390bf37295`; fleet-liveness halves of `fk-906957d2ec84` and `fk-3830e2a6de79`.
- Work: the fleet tier's liveness row:
  - `InstalledSockets('orchestrator-*.socket')`; `PortAndActiveState`; `Streak(1, 300)` (today's
    immediate revive); `Revive(no_block=True, reset_socket=True)`;
    `state_file=SUPERVISION_STATE['orchestrator-fleet']`; `lease_tier=ORCHESTRATOR_FLEET`;
  - 8270's per-unit `ReviveCap(ORCH_LIVENESS_RESTART_MIN_INTERVAL_SECS)` (900 s) and
    `CrashLoopPolicy(RESTART_RATE_ALARM_COUNT, RESTART_RATE_ALARM_WINDOW_SECS, 'orchestrator-crash-loop-')`,
    with the crash-loop step evaluated before the startup-grace gate, as `main()` runs
    `crash_loop_pass` today. The engine's crash-loop gate calls
    `scripts/_watchdog_crash_loop.py::observe_unit` and `mark_page_landed` (pure, kept) on a history
    held in `TierState`; that module's file I/O and filer (`crash_loop_pass`, `load_restart_histories`,
    `save_restart_histories`, `restart_rate_state_path`, `file_crash_loop_escalation`) go, and the page
    goes through `Host.file_page` with the module's sentinel, role, severity and category constants. The
    revive cap's clock files under `ORCH_LIVENESS_RESTART_CLOCK_DIR` become `UnitState.last_revive`.

  `main()`'s `WATCHED` loop (and its `crash_loop_pass` call) becomes `liveness_pass` over the row.
  `--report` gains a LIVENESS column fed by the same `InstalledSockets` derivation (probed / disabled /
  in grace / leased) and keeps 8270's RESTARTS column. The tests pinning `WATCHED` go, and 39
  fleet-liveness-only tests (1,022 lines, measured at `ba17934861`), the liveness half of the 51 mixed
  fleet tests, and 8270's 18 `main()` cap and crash-loop tests move onto `FakeHost`;
  `tests/scripts/test_watchdog_crash_loop.py` keeps its `observe_unit` cases and loses those of the
  deleted I/O. Task 9321's two interim `WATCHED` pairs, if it landed, go with `WATCHED`.
  `tests/scripts/test_orchestrator_socket_units.py` keeps its committed-template checks (every service
  has a socket, one 127.0.0.1 `ListenStream` per socket, ports unique) without importing `WATCHED`, and
  gains `test_socket_port_matches_configured_escalation_port`: for each committed
  `scripts/orchestrator-*.service` it reads the `--config` path from `ExecStart`, loads that yaml's
  `escalation.port` and asserts it equals the socket template's `ListenStream` port; a missing project
  yaml is a per-project skip with a reason, never a silent pass (the watchdog half of task 7517, in
  socket-template form). `test_restart_orchestrator_unit_default_matches_across_tiers` re-pins
  `scripts/restart-orchestrator.sh`'s `SERVICE` default against `SELF_UNIT` and the installed socket
  set, and the script's "WATCHED table" comment is rewritten. The watchdog unit's `TimeoutStartSec`
  comment re-derives its worst case from the installed unit count. OPERATIONS.md §8 liveness bullet:
  "installed sockets", and its "Known gap: the watched list is hardcoded" subsection is deleted.
  factory-init's watchdog step becomes "nothing to do: installing the `.socket` is enough".
- Deletes: `WATCHED`, `main`'s body (the entry keeps `_cli`), `_unit_active_state`,
  `test_main_targets_expected_pairs`, `test_watched_ports_match_configured_escalation_ports`, the
  `module.WATCHED` reads in `test_orchestrator_socket_units.py`, `_revive_if_dead`,
  `_orch_liveness_restart_clock_path`, `_stamp_orch_liveness_restart_clock`,
  `_orch_liveness_revive_wait_secs`, and `_watchdog_crash_loop.py`'s state-file and filer functions
  named above.
- Signal: boundary rows 17, 18, 20 and 20b. Live: `scripts/orchestrator-watchdog.py --report`'s LIVENESS
  column shows `orchestrator-scoping-intake.service` and `orchestrator-stream-touch.service` as probed.
- G7: INV-5/INV-9 (one unit-set derivation, the one staleness already used), INV-1 (socket port ↔
  configured escalation port stays machine-checked), INV-10.

**δ2b — The orchestrator fleet's staleness and `--report` run on the engine; the fleet sibling helpers
and the remaining private patches go** (`task_kind=normal`, priority **high**, deletion-heavy; band
exception above)
- Files: `scripts/orchestrator-watchdog.py`, `scripts/_supervision.py`, `scripts/_watchdog_host.py`,
  `tests/scripts/fake_watchdog_host.py`, `tests/scripts/test_supervision.py`, `tests/scripts/test_orchestrator_watchdog.py`,
  `tests/scripts/test_orchestrator_watchdog_module_removal.py`, `OPERATIONS.md`
- Deps: δ2a
- Keys: `fk-3830e2a6de79` (closing); fleet-staleness half of `fk-906957d2ec84`.
- Work: the fleet tier's staleness row: `ExecMainStartTimestamp` with the fleet clock,
  `ORCH_RESTART_MIN_INTERVAL_SECS`, `STALENESS_GRACE_SECS` and the `orch-fleet-staleness-redeploy`
  `--drain` command, with the per-unit enabled and startup-grace gates over the running installed units;
  the module-removal waiver as `RemovalWaiver`. The row and the waiver read
  `WATCH_CLOSURE[Tier.ORCHESTRATOR_FLEET]` (C5, landed in ζ). `report()` renders through C4 for both tiers, keeps
  the LIVENESS column, and its MERGE-IDLE column keeps reading `drain_check`. 56 fleet
  staleness-or-report-only tests (1,780 lines, measured at `ba17934861`), 8270's RESTARTS-column test,
  the staleness half of the 51 mixed fleet tests, and every remaining private patch move onto
  `FakeHost` or are retired.
- Deletes: `staleness_pass`, `_delegate_fleet_restart`, `_register_transient_unit`, `_fleet_clock_hold`,
  `_staleness_mode` and its `_StalenessMode` family, `_newest_watched_commit_epoch`,
  `_read_last_fleet_deploy_epoch`, `_within_fleet_deploy_min_interval`, `_within_fleet_staleness_head_start`,
  `_unit_start_epoch`, `_unit_start_elapsed_secs`, `_enumerate_running_units`, `_BucketedWarnings`,
  `_neutralize_fleet_clock_gates`, and every remaining `wdog, "_…"` patch (220 across the watchdog test
  files at `a85cb1ad78`, before δ1 and δ2a retire theirs).
- Signal: boundary rows 19 and 15b. Live: `--report` prints the same staleness verdicts and columns as
  before δ2b, plus LIVENESS.
- G7: INV-10 (no test reaches a private name), INV-5 (one staleness sequence for every tier).

**δ3 — The dashboard watchdog runs on the engine; its tick, state and ceiling code go**
(`task_kind=normal`, priority **high**, est. ~1,100 LOC, deletion-heavy)
- Files: `scripts/dashboard-watchdog.py`, `scripts/_supervision.py`, `scripts/_watchdog_host.py`,
  `tests/scripts/fake_watchdog_host.py`, `tests/scripts/test_supervision.py`, `tests/scripts/test_dashboard_watchdog.py`,
  `dashboard/tests/test_dashboard_watchdog_storm_escape.py`, `tests/scripts/test_atomic_write_regrowth.py`
- Deps: δ2b
- Keys: `fk-906957d2ec84`.
- Work: the dashboard tier becomes one row:
  - `StaticUnits(('dark-factory-dashboard.service',))`; `HttpOk(PROBE_URL, PROBE_TIMEOUT)`;
    `Streak(FAIL_STREAK, 0)` (no age expiry, as today); grace `GRACE_SECS`;
  - `Revive(no_block=False, reset_socket=False)`;
  - `StormEscape(MAX_RESTARTS, RATE_WINDOW_SECS, stop_reviving=True,
    'dashboard-watchdog-restart-ceiling', 'harness-dashboard-watchdog')`, an episode that closes on the
    first healthy verdict, as `scripts/dashboard-watchdog.py::tick`'s `ceiling_open` does today;
  - `state_file=SUPERVISION_STATE['dashboard']` (today's `data/dashboard-watchdog/state.json`);
  - no staleness, no lease.

  The real host's state reader accepts today's body once (decision 9). The storm-escape test keeps
  asserting against the real `escalation submit` writer, now through `_watchdog_host.file_page`.
  `tests/scripts/test_atomic_write_regrowth.py::_ALLOWED_RENAMERS` loses its
  `('scripts/dashboard-watchdog.py', 'save_state')` entry, which `test_no_unapproved_renamers_in_source_trees`
  would otherwise fail as stale.
- Deletes: `tick`, `load_state`, `save_state`, `default_state`, `_normalise_state`, `_prune_restarts`,
  `file_ceiling_escalation`, `probe_health` (the fifth copy).
- Signal: boundary rows 21 and 21b (today's B-rows replayed through `FakeHost`). Live: a healthy
  dashboard writes no line per 30 s tick, and any dashboard-watchdog line names tier `dashboard`.
- G7: INV-4 (the escape keeps its page and stop), INV-10.

**ε — The watchdog restarts an orchestrator whose event loop has wedged, filing one escalation per
restart, and every liveness tier escalates a revive storm (INV-4)** (`task_kind=normal`, priority
**high**, est. ~650 LOC; ruled OQ-1 B with R-3 = A, OQ-4 A)
- Files: `scripts/_supervision.py`, `scripts/_watchdog_host.py`, `scripts/orchestrator-watchdog.py`,
  `tests/scripts/fake_watchdog_host.py`, `tests/scripts/test_supervision.py`, `tests/scripts/test_orchestrator_watchdog.py`,
  `OPERATIONS.md`
- Deps: δ3
- Keys: `fk-1cc71283157c`.
- Work (decision 11). ε is not ordered after task 9492 (R-3 = A); nothing in this plan waits on it.
  - `HeartbeatProbe` (with its own persisted streak) in C4. `Host.heartbeat_age` reads
    `resolve(HEARTBEAT_DIR) / '<unit>.json'` `ts_epoch` and returns the age counted from the later of
    that stamp and the unit's current instance start; a missing or unparseable file is `None`, which
    never counts as stale (fail toward not acting). The pass logs one bucketed WARNING per active unit
    past its startup grace whose heartbeat file is missing, and `--report`'s HEARTBEAT-AGE column renders
    `absent` for it.
  - The fleet row sets `HeartbeatProbe(900, 3, 'orchestrator-liveness-heartbeat-', 'orchestrator-watchdog')`
    and `StormEscape(3, 3600, stop_reviving=True, 'orchestrator-liveness-ceiling', 'orchestrator-watchdog')`,
    and keeps `Streak(1, 300)` for dead units. At the heartbeat streak the unit takes the dead-unit
    path: storm escape, 8270's revive cap, revive.
  - A wedge revive files exactly one escalation in the revive's `finally`: sentinel
    `orchestrator-liveness-heartbeat-<unit>`, critical, `infra_issue`, role `orchestrator-watchdog`, its
    detail naming the heartbeat age, the 900 s threshold, the `ReviveOutcome` and the recipe (the unit's
    journal and its `loop_stall` stack lines). A restart that raises or times out still files it; a
    filing that fails is logged and does not undo the revive. No once-per-window setting.
  - The fm row sets `StormEscape(3, 21600, stop_reviving=False, 'fused-memory-liveness-ceiling', 'orchestrator-watchdog')`.
  - Tests in `tests/scripts/test_supervision.py` include
    `test_third_stale_heartbeat_check_revives_once_and_escalates_once` (row 22) and
    `test_wedge_escalation_is_filed_when_the_restart_raises` (row 22b).
  - OPERATIONS.md §8: what the wedge escalation and each ceiling page mean, the recipe, that a wedge
    restart is a hard kill after the 90 s stop timeout, and task 9492 as the cause of the measured long
    stalls.
- Signal: boundary rows 22, 22b and 22c. Live: `--report` shows a HEARTBEAT-AGE column for all 9 units,
  all under 30 s on a healthy fleet (measured 4.7–14.9 s at `a85cb1ad78`).
- G7: INV-3 (a stale heartbeat must hold for 3 checks on its own streak), INV-4 (every liveness tier
  has a rate escape), INV-13 (the reader's producer is the harness's `merge-heartbeat` service,
  `Harness._write_merge_heartbeat`, measured live on 9 of 9 units; a missing producer file is reported,
  not silent), INV-2 (each wedge restart names its facts in its escalation).

**ζ — One code closure per tier feeds the watchdog and both on-merge coordinators; the fused-memory
coordinator watches fused-memory's full import closure at most once per 4 hours, with a drain restart
(C5)** (`task_kind=normal`, priority medium, est. ~650 LOC; band exception above; ruled OQ-2 B, R-4)
- Files: `shared/src/shared/fleet_coordination.py`, `shared/tests/test_fleet_coordination.py`,
  `orchestrator/src/orchestrator/config.py`, `orchestrator/src/orchestrator/harness.py`,
  `dark-factory-orchestrator.yaml`, `scripts/orchestrator-watchdog.py`,
  `tests/scripts/test_fm_watched_paths_import_closure.py` (deleted), `tests/scripts/test_watched_paths_import_closure.py` (new),
  `tests/scripts/test_orchestrator_restart_config_drift.py`, `orchestrator/tests/test_harness_service_restart.py`,
  `tests/scripts/test_deploy_clock_isolation.py`, `OPERATIONS.md`,
  `tests/scripts/test_orchestrator_watchdog.py`, `tests/scripts/test_orchestrator_watchdog_module_removal.py`
- Deps: γ, 9140, 8270 (done; satisfied)
- Keys: `fk-68efae8a36fc`.
- Work: `harness.py` is already behind 9140 through that edge. `WATCH_CLOSURE` and W1–W7, on today's flat config fields (lead OQ-7 A). ζ lands right after γ
  and before the engine, so today's `staleness_pass`, `fused_memory_staleness_pass` and module-removal
  waiver read the table in place of the three constants; δ1 and δ2b then build their rows on it.
  - The orchestrator coordinator and the watchdog: as W1, W2 and W4. The yaml comment describing the
    explicit-deploy convention (`scripts/restart-all-orchestrators.sh --drain`, the pattern of tasks
    2776/5049/7175) gains `--lease-wait` (lead OQ-9 B).
  - The fused-memory coordinator (decision 16): W3's closure default, W5's restart-only
    `fused_memory_restart_min_interval_secs` with its default `FM_ON_MERGE_MIN_INTERVAL_SECS` = 14,400 in
    the shared leaf, C1's `FM_ON_MERGE_CLOCK` added to `PROTECTED_FILES` (so `df_pytest_isolation`
    redirects and change-detects it from this commit on), W6's builder wiring
    (`Harness._build_service_restart_coordinator` passes the interval, the clock,
    `script_args=['--drain']` and a cgroup-escaping `restart_executor`: per fire, a transient unit
    `fm-on-merge-restart-<n>.service` registered through `service_restart.py::schedule_detached_systemd_restart`
    with an on-failure escalation, as `Harness._build_orchestrator_restart_coordinator` already does for
    the orchestrator. The builder's docstring line "No min_interval cap (state_path=None) throttles it",
    and the config comment saying the same, are rewritten), and W7's enable flag (default false; the
    dark-factory yaml sets `fused_memory_restart_on_merge_enabled: true`). Task 6543 is cancelled into ζ.
  - Tests: the closure guard generalizes (`tests/scripts/test_watched_paths_import_closure.py` replaces
    the fm-only one and adds the orchestrator tier); the config drift test asserts W2, W3, W5's field
    default and W7; `shared/tests/test_fleet_coordination.py` pins `FM_ON_MERGE_MIN_INTERVAL_SECS ==
    14400` and that `FM_ON_MERGE_CLOCK` is in `PROTECTED_FILES`; `orchestrator/tests/test_harness_service_restart.py`
    pins boundary row 28 (interval, clock, transient unit and `--drain` at the fired argv, observed
    through the injected `systemd-run` runner rather than the coordinator's private `_script_args`).
  - The two watchdog test files still read the constants ζ deletes, and ζ re-pins those reads onto
    `WATCH_CLOSURE`: 10 lines in `tests/scripts/test_orchestrator_watchdog.py` (among them
    `_EXPECTED_WATCHED_PATHS`, the hand-kept copy of today's five entries, the FM_WATCHED_PATHS-entries
    test and the pathspec assertion `watched_args == list(wdog.FM_WATCHED_PATHS)`) and 4 in
    `tests/scripts/test_orchestrator_watchdog_module_removal.py` (among them
    `wdog.MODULE_REMOVAL_WATCHED_PATHS` and `set(wdog.WATCHED_PATHS) <= set(watched)`), measured at
    `a85cb1ad78` (unchanged since `ba17934861`).
  - OPERATIONS.md: the fused-memory on-merge coordinator's closure trigger, its 4-hour interval and
    clock (and that θ lowers it to 1 hour once task 9521 confirms the stop fix on the host), its drain
    restart in a `fm-on-merge-restart-<n>.service` transient unit, how to read that unit's journal, and
    that only dark-factory's orchestrator runs it.
  - G6 for the orchestrator tier, measured at `1f240e6da2` over the last 30 days: every one of the ten
    72 h windows already had an orchestrator-tier commit, so adding `shared/src/` changes how many fleet
    redeploys happen by zero; the value is correctness. For fused-memory the 4-hour interval keeps the
    modelled restart rate at about today's 5 a day (decision 16), while a shared, escalation or
    orchestrator change now reaches fused-memory within about 4 hours instead of waiting for the 8-hour
    watchdog backstop.
- Deletes: `WATCHED_PATHS`, `FM_WATCHED_PATHS`, `MODULE_REMOVAL_WATCHED_PATHS`, the yaml key
  `orchestrator_restart_watch_prefixes`, `orchestrator/uv.lock` from the watched set, the fm
  coordinator's `['fused-memory/src/']` default, and `tests/scripts/test_fm_watched_paths_import_closure.py`.
- Signal: boundary rows 24 and 28. Live: `--report`'s newest-watched-commit column moves when a
  `shared/src/`-only commit lands; after a `shared/src/`-only merge, the orchestrator journal shows the
  fused-memory coordinator registering one `fm-on-merge-restart-<n>.service` that runs
  `restart-fused-memory.sh --drain`, and
  `data/fused-memory/last_on_merge_restart_fused_memory.json` holds that fire's epoch.
- G7: INV-9 (one closure home; one home for the interval), INV-10 (the guard derives from source, with
  a seeded fixture member), INV-7 (the interval bounds how often a restart can cut sessions and
  reconciliation).

**ι — fused-memory's drain restart waits for every measured reconciliation run, and a drain that still
times out files an escalation** (`task_kind=normal`, priority medium, est. ~300 LOC; Leo's R-4 "drain
restart that allows recon runs to finish")
- Files: `scripts/restart-fused-memory.sh`, `scripts/tests/test_restart_fused_memory.py`,
  `shared/src/shared/fleet_coordination.py`, `shared/tests/test_fleet_coordination.py`, `OPERATIONS.md`
- Deps: ζ, 9140
- Keys: none (ruling addition; it serves `fk-68efae8a36fc`'s widened trigger).
- Work (decision 17):
  - C1 gains `FM_DRAIN_MAX_WAIT_SECS = 7200` and the CLI verb `drain max-wait`;
    `LEASE_MAX_AGE_SECS[FUSED_MEMORY]` rises from 3,600 to 7,800, with a test pinning
    `>= FM_DRAIN_MAX_WAIT_SECS + 60 + 90 + 30` (submit bound, stop timeout and health wait; decision 6).
  - `restart-fused-memory.sh`: `DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-<drain max-wait from the CLI>}"`, read
    after the sweep lock and under γ's 70 rule. On a drain timeout it reads `/health` once
    (`recon_busy`, through `recon_busy_check.py` as the default path does), logs the WARNING, files one
    escalation through `escalation submit` (sentinel `fused-memory-drain-timeout`, severity urgent,
    category `infra_issue`, agent role `harness-fused-memory-restart`; the detail names the bound, the
    elapsed time and the busy runs), then restarts. The filing runs the same way
    `scripts/_watchdog_crash_loop.py::crash_loop_submit_argv` does (absolute `uv`, the repo's escalation
    queue). It runs under `timeout 60`, the same bound as that module's `CRASH_LOOP_SUBMIT_TIMEOUT_SECS`,
    and ends `|| true` after logging a failed or timed-out submit, so it never blocks the restart.
  - A drain is never abandoned (decision 17). The script sets a flag when SIGUSR1 has been sent and
    clears it at its restart step. If the flag is set, its EXIT path, which γ's `TERM`/`INT` traps reach,
    runs `systemctl --user restart --no-block fused-memory || true` and logs that the drain was
    abandoned. The script's exit status is unchanged. A refusal or CLI failure before SIGUSR1 (75, 70)
    restarts nothing. Test: `test_drain_is_never_abandoned_after_the_signal` (row 26b).
  - The comment above `DRAIN_TIMEOUT` (today "cycle-scale wait … replaces the old 120s") states the new
    basis in one line and points at C1.
  - OPERATIONS.md §8 (fused-memory restarts): what `--drain` waits for, the 2-hour bound, the
    escalation, that an abandoned drain still restarts fused-memory, and that `RECON_GATE_TIMEOUT` (the
    default path) and `--now` are unchanged.
- Deletes: the literal `DRAIN_TIMEOUT` default 2100.
- Signal: boundary rows 26, 26b and 27. Live: the next coordinator-fired drain that overlaps a running full
  reconciliation run logs "Harness fully drained" after that run completes, with no "Drain timed out"
  line, for any run shorter than 7,200 s.
- G7: INV-7 (the drain's hold is bounded, and its end is visible), INV-2 (the escalation names the bound
  and the runs), INV-9 (the bound has one home, read by bash through the CLI).

**θ — fused-memory's on-merge restart interval drops from 4 hours to 1 hour once its stop fix is in
effect on the host** (`task_kind=normal`, `complexity='simple'`, priority low, est. ~30 LOC; Leo's R-4
"until the shutdown fix (then A)")
- Files: `shared/src/shared/fleet_coordination.py`, `shared/tests/test_fleet_coordination.py`, `OPERATIONS.md`
- Deps: ζ, ι, 9493, 9521
- Keys: none (ruling follow-up).
- Work: θ lands as soon as 9521 is done (lead ruling after the wave-3 fold-verifiers). Its one-line
  `OPERATIONS.md` edit is exempt from the same-file chain, so it takes no edge for that file. It follows
  ι on `shared/fleet_coordination.py`, and ω follows θ on that file (ω → θ; §Sketch, Order). 9493 is
  done; 9521 is the operator reinstall of the fused-memory unit with `KillMode=mixed` and its live
  check, a deterministic human gate blocked at L2.
  - `FM_ON_MERGE_MIN_INTERVAL_SECS` goes from 14,400 to 3,600, and the value pin in
    `shared/tests/test_fleet_coordination.py` follows.
  - OPERATIONS.md's 4-hour sentence becomes 1 hour and drops the "until 9521" clause: one line, the
    exempt edit.
  - "In effect" is known by 9521 being done (§Pre-conditions): its step (4) `KillMode=mixed` check and
    its step (6) live one-signal restart, both recorded at its resolution. θ does not re-probe the host.
- Why a task and not a config value: decision 16's last paragraph. Why it depends on 9521 and not only
  9493: 9493 merged on 2026-11-13, but the installed unit still reads `KillMode=control-group` (measured
  at `a85cb1ad78` and again at `fceb951e7e`), so the fix's unit half, the one that spares running tool
  calls, takes effect only when 9521 is done.
- Signal: none beyond the constant (no boundary row). After the next dark-factory orchestrator restart,
  the orchestrator journal shows fused-memory on-merge fires as little as 1 hour apart, where the
  minimum had been 4 hours.
- G7: INV-9 (one value, one home).

**η — The orphan-worktree reclaim sweeps every known project root** (`task_kind=normal`, priority
medium, est. ~500 LOC; lead OQ-3 A)
- Files: `scripts/reclaim_orphaned_worktrees.py`, `scripts/reclaim-orphaned-worktrees.sh`,
  `scripts/reclaim-orphaned-worktrees.service`, `scripts/tests/test_reclaim_orphaned_worktrees.py`,
  `scripts/tests/test_reclaim_orphaned_worktrees_wrapper.py`, `OPERATIONS.md`
- Deps: ι, 9140
- Keys: `fk-47a7eb8c020d`.
- Work: η shares only `OPERATIONS.md` with the rest of the batch. It follows ι on that file, γ and ζ
  through ι, and δ1 waits on it (§Sketch, Order).
  - `main` loops `shared.project_layout.known_project_roots()` (bound through `scripts/_checkout.py`)
    and sweeps each root's `<root>/.worktrees-orphaned`. `--repo` still narrows to one root.
  - A root that is not a verified git toplevel is reported `refused_target` (today's fail-closed
    verify), per root.
  - With the env unset, it sweeps the main checkout only and logs one WARNING naming
    `DASHBOARD_KNOWN_PROJECT_ROOTS`.
  - The committed unit carries `Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory`,
    as `dashboard/dark-factory-dashboard.service` does. The host's full list goes in a drop-in,
    `reclaim-orphaned-worktrees.service.d/known-project-roots.conf`, the shape `return-brief.service.d`
    already uses. That drop-in survives `scripts/install-reclaim-orphaned-worktrees-timer.sh`, which
    copies the unit verbatim. OPERATIONS.md gives the operator step, and the lead files it as a lead
    action.
  - The JSON report becomes one object per root.
  - Out of scope: a project whose `worktree_base` is relocated parks at `<base>-orphaned`. Measured:
    `/home/leo/src/warm-lanes/worktrees-orphaned` holds 1 entry of 0 bytes.
- Signal: boundary row 25. Live: `systemctl --user start reclaim-orphaned-worktrees.service`, after the
  operator re-installs the unit, journals one report per known root, and
  `solar-challenge-platform`'s parkings older than 48 h are reclaimed.
- G7: INV-7 (every producer's quarantine hold has a reclaimer), INV-2 (per-root refusal named).

**ω — Integration gate: cross-seam boundary tests, the supervision invariants' one home, the legacy
lease branch and the spelling-guard family** (`task_kind=normal`, priority medium, est. ~700 LOC)
- Files: `tests/scripts/test_fleet_supervision_boundary.py` (new), `shared/src/shared/fleet_coordination.py`,
  `shared/tests/test_shared_leaf_spellings.py`, `OPERATIONS.md`, `scripts/orchestrator-watchdog.py`,
  `scripts/_supervision.py`, `tests/scripts/conftest.py`, `tests/scripts/test_orchestrator_watchdog.py`,
  `tests/scripts/test_orchestrator_watchdog_module_removal.py`, `tests/scripts/test_watchdog_dashboard_parity.py`,
  `tests/scripts/test_dashboard_watchdog.py`
- Deps: ε, θ, 6726
- Keys: `fk-ed19c0c4d4df`.
- Work: η and ι come before ω through δ1. ω follows θ on `shared/fleet_coordination.py` (the same-file
  rule; θ's exemption covers only its `OPERATIONS.md` line), so ω waits on operator gate 9521 through θ
  (§Sketch, Order).
  - Boundary rows 23 and 24 run across real processes: bash with a fake `systemctl` on PATH, and
    `_watchdog_host`.
  - OPERATIONS.md §8 gains `### Supervision invariants`, with I5, I6, I7 and the fail directions, each
    stated once. `TierSpec`'s docstring and the entry scripts' module docstrings become one-line
    pointers.
  - Re-measure the watchdog files' prose and move what is still reviewer-answering (Comments stance).
  - Delete C2's legacy-body branch once `--report` shows no legacy body on the host (decision 3).
  - Add the fleet-state family (`FleetFile` relpaths, `data/fleet`) to the shared spelling-guard test
    from task 6726, with a seeded fixture (lead OQ-5 A).
  - One `load_orchestrator_watchdog()` in `tests/scripts/conftest.py` replaces the remaining
    orchestrator `_load_watchdog` copies, and one `load_dashboard_watchdog()` there replaces
    `tests/scripts/test_dashboard_watchdog.py`'s variant (8362).
- Signal: boundary rows 23–24 pass in the project verify. Live: `--report` shows 9 units, one
  `FLEET-LEASE:` line per tier, the LIVENESS column and the HEARTBEAT-AGE column. The prose measure is
  recorded in the commit message.
- G7: INV-1 (cross-seam behaviour is machine-checked), INV-9, INV-10 (the guard's seeded fixture).

## Design questions for the author-pass sitting

All nine were ruled at the sitting of 2026-11-12 and its residual answers of 2026-11-13; §Rulings
lists them. No question remains open, and no open-question marker remains in the text.

## Out of scope

- The bash sweep's own prose (530 of 894 lines are comments, measured at `1f240e6da2`). Only the regions
  γ edits are rewritten.
- Concurrent restarts within one sweep (task 7314) and the ceiling derivation (task 7307). C2 carries
  7314's unit set and gives 7307 the bound; those tasks keep the rest.
- `dashboard_restart_watch_prefixes` and the dashboard on-merge coordinator.
- The unit-parity pass (`scripts/_watchdog_dashboard_parity.py`) and S25's `run_parity`.
- A relocated `worktree_base`'s parking directory (η, measured empty).
- `DASHBOARD_KNOWN_PROJECT_ROOTS`'s copies in other installed units (fused-memory, dashboard,
  return-brief).
- Reify-side supervision. reify's units are on this host and covered by the installed-socket set; its
  repo is untouched.
- A second `systemctl --user show` parser already exists in the orchestrator package:
  `orchestrator/src/orchestrator/systemd_inspect.py::inspect_systemd_unit`. R20 places C3 in `scripts/`
  for the stdlib watchdogs; unifying the two is a later candidate.
- systemd's own watchdog for orchestrators (`WatchdogSec=` with an on-loop `sd_notify` ping, the
  fused-memory lesson of task 1731: the ping must come from the loop). Leo preferred it but accepted the
  watchdog-script restart because one tested code path both restarts and escalates; it would need nine
  hand edits to drifted installed units, an `ExecStopPost=` filer and a backstop for that filer
  (`digests/discussion/S24.md` §S24-1). It stays a possible later change, and nothing in ε prevents it.
- Moving the directory walk behind the long loop stalls off the event loop: task 9492, outside this
  plan.
- `restart-fused-memory.sh`'s default path (`RECON_GATE_TIMEOUT`, 2,100 s), which the watchdog's 8-hour
  fm staleness backstop uses, and `--now`. Leo's drain ruling concerns the on-merge coordinator's
  restart; ι changes only `--drain`.
- An "undrain" path for fused-memory's reconciliation harness (`ReconciliationHarness.drain` is one-way
  until restart). No S24 path leaves the harness drained without a restart, for three reasons. γ takes
  the sweep lock before any SIGUSR1. ζ runs the coordinator's restart in its own transient unit, outside
  the orchestrator's cgroup. ι restarts at the bound, and it restarts on every exit after SIGUSR1 short
  of a SIGKILL (decision 17).

## Open questions (tactical)

1. **CLI verb spelling.** `lease show` vs folding it into `--report` only. Decide in γ. The `FLEET-LEASE:`
   prefix is fixed either way, since CLAUDE.md tells operators to read it.
2. **Where the moved measurements land.** Commit messages by default, or a `plans/` note when a
   measurement is still a decision input (the 2026-08-25 boundary measurement backs task 4754's head
   start, so a `plans/` pointer is fine). Decide per leaf.
3. **The `TimeoutStartSec=1800` re-derivation** in `scripts/orchestrator-watchdog.service` with 9 units.
   δ2a recomputes it from the per-call bounds. A value change is a restart-only unit edit, which the
   lead installs.
