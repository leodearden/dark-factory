# Capability manifest — S24 fleet watchdog and redeploy coordination

PRD: `plans/review-all-s24-fleet-watchdog-and-redeploy-coordination-prd.md`.
YAML twin: `plans/review-all-s24-fleet-watchdog-and-redeploy-coordination-prd.capability-manifest.yaml`.
The twin names three signal leaves: ω (the integration gate), γ (the live lease refusal) and η (the live
per-root reclaim). Every `task_id` stays `null` until decompose.

**Code anchors** verified against main `a85cb1ad78` (2026-11-13) by the ruling fold and re-checked at
`d538279068` (no symbol the plan relies on moved; every check keeps its polarity); the author verified
them at `1f240e6da2` and the first fold at `ba17934861`. Bindings cite symbols, not lines. The fold
applied the sitting's rulings of 2026-11-12/13 (PRD §Rulings): ε restarts a wedged orchestrator at a 900 s
heartbeat age and escalates on every such restart (OQ-1 B); ζ gives the fused-memory on-merge coordinator
its full closure, a 4-hour interval, a clock and an explicit drain restart (OQ-2 B, R-4); two new leaves,
ι (the drain waits for every measured reconciliation run and escalates a timeout) and θ (the 4-hour to
1-hour flip, after tasks 9493 and 9521).

The fold-verifier (2026-11-13, at `fceb951e7e`) made four changes:
- It chained every pair of leaves that share a non-test file: ι → ζ, η → ι, δ1 → η.
  The alignment pass (2026-11-13) applied the lead's ruling that θ's one-line `OPERATIONS.md` edit is
  exempt from the same-file chain: θ → ζ, θ → ι, θ → 9493, θ → 9521, and ω → θ on
  `shared/fleet_coordination.py`. The flip lands as soon as 9521 is done; ω waits on 9521 through θ.
- ζ fires the fused-memory restart in a transient user unit, outside the orchestrator's cgroup.
- ι never abandons a drain once SIGUSR1 has been sent.
- It added two producer checks.

Polarity was measured at `a85cb1ad78` and re-run in full at `fceb951e7e`. Grep checks went through
`shared/src/shared/delivered_check_polarity.py::evaluate_grep_at_tree` and path checks through
`evaluate_path_at_tree`, both at that sha. The sidecar loads through
`shared/src/shared/capability_manifest.py::CapabilityManifestDoc` (13 tasks). All 76 mechanical checks
fail at today's tree, as a producer check must:

- each `expect: present` grep has no match today;
- each `expect: absent` grep matches today;
- each `expect: present` path names a file that does not exist today, and the one `expect: absent`
  path names a file that does.

Substrate rows carry no mechanical check. Each is named `*-substrate`, and all pass today
(`proc-liveness-predicate-substrate` since task 6615 merged on 2026-11-11). Every `OPEN` row (six)
carries a `kind: manual` check whose `reason` says what a reviewer confirms. There are no
`kind: script` checks. No check's value is one a later leaf of this plan deliberately changes: ζ's
interval check names the constant, not its 14,400 value, because θ sets 3,600.

`lint_delivered_checks`, run with each leaf's declared files, reports two `absent_overbroad`
warnings and nothing else. Both are matches an earlier leaf in the chain removes:

- δ2b `private-watchdog-patches-gone` matches `tests/scripts/test_watchdog_dashboard_parity.py`'s three
  `_print_fused_memory_liveness` patches. δ1 declares that file and removes them, and δ2b depends on δ1
  through δ2a.
- ω `single-watchdog-loader` matches `tests/scripts/test_fm_watched_paths_import_closure.py` and
  `tests/scripts/test_watchdog_pid_alive_parity.py`. ζ and γ delete them, and ω depends on both
  through ε.

## α — Fleet state paths and lease ages get one home (C1)

| Capability | Binding | Verdict |
|---|---|---|
| `shared/src/shared/fleet_coordination.py` exists | absent today | PASS (producer check) |
| Each watchdog tier's state file is a `FleetFile` (`SUPERVISION_STATE`) | absent today | PASS (producer check) |
| The watchdog stops spelling the three fleet JSON names | `FLEET_DEPLOY_CLOCK_PATH`, `FLEET_LEASE_PATH`, `FM_DEPLOY_CLOCK_PATH` built from literals today | PASS (producer check) |
| `df_pytest_isolation.py` derives its protected-file table and `LIVE_FLEET_DIR` from C1 | four literal assignments today | PASS (producer check) |
| One heartbeat-dir default | `DEFAULT_FLEET_DIR` in `scripts/drain_check.py` and `orchestrator/fleet_heartbeat.py` today | PASS (producer check) |
| The four-way drift pins collapse | three across-tiers tests in `tests/scripts/test_orchestrator_watchdog.py` today | PASS (producer check) |
| The system python can import a stdlib `shared` leaf | `scripts/_checkout.py` (S25 C1, task 6758 done); measured: `/usr/bin/python3 -I` imports `shared.project_layout` with no aiosqlite, pydantic or yaml loaded | PASS (substrate) |
| `df_pytest_isolation` can import a `shared` submodule under escalation's stub | the stub sets `__path__` to `shared/src/shared` before the import; measured with the stub reproduced | PASS (substrate) |

## γ — Redeploy lease keyed by tier, acquired under flock (C2)

| Capability | Binding | Verdict |
|---|---|---|
| The sweep acquires through C2 | `lease acquire --tier orchestrator-fleet` absent today; `lease_acquire` stamps unconditionally | PASS (producer check) |
| The fm restart acquires its tier | `restart-fused-memory.sh` has no lease today | PASS (producer check) |
| The fm restart publishes `fused-memory.service` as its current unit, for its restart step only (`lease set-units`) | absent today | PASS (producer check) |
| The lock replaces `refuse_while_staleness_redeploy_runs` (task 8129; task 9389's drained-then-refused state) | present in `restart-fused-memory.sh` today | PASS (producer check) |
| Each restart script holds a kernel `flock` on its tier's sweep lock for its whole life (lead OQ-8 D) | `flock -w` absent in both scripts today | PASS (producer checks, one per script) |
| Each restart script accepts `--lease-wait SECS` (lead OQ-9 B) | `restart-all-orchestrators.sh` rejects extra arguments and `restart-fused-memory.sh` accepts only `--now`, `--drain` and `--rebind-socket` today | PASS (producer checks, one per script) |
| Each restart script exits 70 on a CLI infrastructure failure | `exit 70` absent in both scripts today | PASS (producer checks, one per script) |
| The bash sweep reads its path defaults from C1 | `FLEET_DIR`, `CLOCK_FILE` and `LEASE_FILE` hard-code their defaults today | PASS (producer check) |
| Exclusion is held under `fcntl.flock` | no `flock` anywhere in the fleet today | PASS (producer check) |
| The unconditional `mktemp`/`mv -f` writer goes | present in `lease_set_current_unit` today | PASS (producer check) |
| `service_restart.py`'s fleet literals and max-age go | three assignments today | PASS (producer check) |
| `service_restart.py`'s own lease predicate goes | `service_restart.py::lease_is_live` present today | PASS (producer check) |
| The lease max-age config field goes; the coordinator and `DrainIdentity` read C1 | `orchestrator_restart_lease_max_age_secs` in `config.py` and twice in `harness.py` today | PASS (producer check) |
| The watchdog's `_pid_alive` copy goes | present today, a declared copy since 6615 | PASS (producer check) |
| The pid predicate C2 uses exists and is safe | `shared.proc_liveness.pid_alive`, stdlib-only, never raises; landed with 6615 (`f750373949`) | PASS (substrate) |
| A second concurrent sweep is refused on the host | live confirmation: during a sweep, `flock -n` on the orchestrator-fleet sweep lock exits 1 and `--report` names the holder | OPEN (manual) |
| Ordering | γ → α, γ → 9140, γ → 6615 (satisfied); ζ → γ; 7307 → γ, 7314 → γ and 9263 → γ (lead actions); 5719 and 9389 cancelled into γ; S16 ζ1 → γ (lead OQ-7 A); the earlier S19 η2 → γ ask is withdrawn | PASS (lead edges) |

## β — One host reader for both watchdogs (C3)

| Capability | Binding | Verdict |
|---|---|---|
| `scripts/_systemd.py` exists | absent today | PASS (producer check) |
| The hand show parsers go, in either quote style | 4 in `orchestrator-watchdog.py` (task 8270's among them) and 1 in `dashboard-watchdog.py` today | PASS (producer check) |
| The dashboard watchdog's copies of `log`, `_JournalLog`, `restart_unit` and `_unit_active_enter_epoch` go | present today | PASS (producer check) |
| The orchestrator watchdog's `log` and `_JournalLog` become `JournalLog` | present today | PASS (producer check) |
| The stale "three-phase stop" prose goes | present in `dashboard-watchdog.py::restart_unit` today | PASS (producer check) |
| The `_unit_start_elapsed_secs` silent-fallthrough entry is retired | present in `shared/tests/silent_fallthrough_allowlist.py` today; 7139 amended; β decrements the governed list's `default_covers` | PASS (producer check) |
| Ordering | β → ζ (same file), β → 8270 (satisfied) | PASS |

## δ1 — Supervision engine and the fused-memory tier (C4)

| Capability | Binding | Verdict |
|---|---|---|
| `scripts/_supervision.py` exists | absent today | PASS (producer check) |
| `tests/scripts/fake_watchdog_host.py` exists | absent today | PASS (producer check) |
| The fm halves of the sibling pairs go | eight fm functions present today | PASS (producer check) |
| No test patches an fm private on the watchdog | e.g. `_read_last_fm_deploy_epoch` 15 sites and `_print_fused_memory_liveness` 7, today | PASS (producer check) |

## δ2a — The fleet's liveness on the engine; installed sockets replace `WATCHED`

| Capability | Binding | Verdict |
|---|---|---|
| `WATCHED` goes | 7 pairs today, while 9 sockets are installed | PASS (producer check) |
| Its pinning tests go | two tests plus `test_orchestrator_socket_units.py`'s `module.WATCHED`, today | PASS (producer check) |
| Socket port ↔ configured `escalation.port` stays checked, in socket-template form | `test_socket_port_matches_configured_escalation_port` absent today | PASS (producer check) |
| `--report` has a LIVENESS column | absent today | PASS (producer check) |
| 8270's crash-loop file I/O and filer fold into the tier state and `Host.file_page`; `observe_unit` stays | four functions in `scripts/_watchdog_crash_loop.py` today | PASS (producer check) |
| factory-init stops telling an agent to edit `WATCHED` | two skill files today | PASS (producer check) |
| The host has the units the derivation reads | 9 `orchestrator-*.socket` installed and enabled; `show -p Listen --value` answers `127.0.0.1:<port> (Stream)` | PASS (substrate) |
| The LIVENESS column shows scoping-intake and stream-touch as probed | live confirmation after merge | OPEN (manual) |
| Ordering | δ2a → δ1, δ2a → 8270 (satisfied); task 9321's interim pairs go with `WATCHED` | PASS |

## δ2b — The fleet's staleness and `--report` on the engine

| Capability | Binding | Verdict |
|---|---|---|
| The fleet halves and the systemctl helpers go | present today | PASS (producer check) |
| No test patches a watchdog private | 220 sites across the watchdog test files today | PASS (producer check) |
| Ordering | δ2b → δ2a | PASS |

## δ3 — The dashboard tier on the engine

| Capability | Binding | Verdict |
|---|---|---|
| The dashboard's tick, state, ceiling and probe functions go | 8 functions present today | PASS (producer check) |
| The renamer allowance for `save_state` goes with it | `('scripts/dashboard-watchdog.py', 'save_state')` in `_ALLOWED_RENAMERS` today | PASS (producer check) |
| `FAIL_STREAK` stays spelled in the dashboard watchdog | a landed task's check greps `scripts/` for it | PASS (substrate) |

## ε — A wedged orchestrator is restarted, one escalation per restart; storm escapes (INV-4)

Ruled OQ-1 B by Leo (R-3 = A): restart on the third stale one-minute check at a 900 s heartbeat age, one
escalation per restart in the revive's own code path; no orchestrator unit-file change, no soak, no flip
task. OQ-4 A.

| Capability | Binding | Verdict |
|---|---|---|
| `HeartbeatProbe` exists | no pass reads the fleet heartbeat today | PASS (producer check) |
| The orchestrator tier has a storm escape | sentinel `orchestrator-liveness-ceiling` absent today | PASS (producer check) |
| The fm tier has a storm escape | sentinel `fused-memory-liveness-ceiling` absent today | PASS (producer check) |
| The wedge-revive escalation sentinel exists | `orchestrator-liveness-heartbeat` absent today | PASS (producer check) |
| On the third stale check: one revive, one escalation (row 22) | `test_third_stale_heartbeat_check_revives_once_and_escalates_once` absent today | PASS (producer check) |
| A revive whose restart raises still escalates (row 22b) | `test_wedge_escalation_is_filed_when_the_restart_raises` absent today | PASS (producer check) |
| The heartbeat has a live producer (INV-13) | the harness's `merge-heartbeat` service every 15 s; ages 4.7–14.9 s on 9 of 9 units at one read on 2026-11-13 | PASS (measured) |
| Ordering | ε → δ3; not after task 9492 (R-3 = A) | PASS |

## ζ — One code closure per tier (C5)

| Capability | Binding | Verdict |
|---|---|---|
| The yaml stops overriding the orchestrator watch prefixes | 5-entry key, without `shared/src/`, today | PASS (producer check) |
| The config default derives from `WATCH_CLOSURE` | default is `['orchestrator/src/']` today | PASS (producer check) |
| The watchdog's three path lists go | present today | PASS (producer check) |
| One closure guard covers both tiers | absent today | PASS (producer check) |
| The fm coordinator's default is the closure, not `['fused-memory/src/']` (W3) | narrow default in `config.py` today | PASS (producer check) |
| The restart-only `fused_memory_restart_min_interval_secs` field exists (W5) | absent today | PASS (producer check) |
| The harness builder passes the interval (W6) | absent in `harness.py` today | PASS (producer check) |
| The fm builder passes `--drain` explicitly (W6) | absent today; inherited from the class default | PASS (producer check) |
| The fm restart runs in a transient unit `fm-on-merge-restart-<n>.service`, outside the orchestrator's cgroup, with an on-failure escalation (W6) | today the script is a plain child in the orchestrator's `KillMode=control-group` cgroup | PASS (producer check) |
| The interval constant and `FM_ON_MERGE_CLOCK` live in the shared leaf | absent today | PASS (producer check) |
| The fm coordinator is off by default (W7) | `Field(default=True)` today | PASS (producer check) |
| The dark-factory yaml turns it on (W7) | key absent today | PASS (producer check) |
| After a `shared/src/`-only merge the fm coordinator fires `--drain` once and stamps its clock | live confirmation after deploy | OPEN (manual) |
| The fm-only guard is retired | present today | PASS (producer check) |
| Redeploy cadence is unchanged | 10 of 10 72 h windows in the last 30 days already had an orchestrator-tier commit | PASS (measured) |
| Ordering | ζ → γ, ζ → 9140, ζ → 8270 (satisfied); third in the chain, so S16 ζ1 waits only on γ and ζ (lead OQ-7 A) and carries the new field | PASS |

## ι — The fused-memory drain waits for every measured reconciliation run

Leo's R-4 addition. Bound 7,200 s (every completed full run in the 30 days to 2026-11-13 took at most
6,407 s; today's 2,100 s cut 7 of 81 drains since 2026-11-01).

| Capability | Binding | Verdict |
|---|---|---|
| The script reads its drain bound from the shared leaf (`drain max-wait`) | absent today | PASS (producer check) |
| The literal `DRAIN_TIMEOUT` default 2100 goes | present today | PASS (producer check) |
| A drain timeout files one `fused-memory-drain-timeout` escalation | absent today (WARNING only) | PASS (producer check) |
| A drain is never abandoned: after SIGUSR1 a TERM, an INT or an error still restarts fused-memory (row 26b) | `test_drain_is_never_abandoned_after_the_signal` absent today | PASS (producer check) |
| `FM_DRAIN_MAX_WAIT_SECS` lives in the shared leaf | absent today | PASS (producer check) |
| A drain overlapping a running full run ends on the drained marker after the run completes | live confirmation | OPEN (manual) |
| Ordering | ι → ζ, ι → 9140; η → ι and δ1 → η order it before the engine leaves on `OPERATIONS.md`; θ → ι on `shared/fleet_coordination.py` | PASS |

## θ — The fused-memory on-merge interval drops to 1 hour

Leo's R-4 follow-up. A task, not a config value (PRD decision 16).

| Capability | Binding | Verdict |
|---|---|---|
| `FM_ON_MERGE_MIN_INTERVAL_SECS` is 3,600 | the shared leaf is absent today; ζ sets 14,400 | PASS (producer check) |
| The stop fix is in effect on the host | task 9521's own check (`KillMode=mixed`); `control-group` on 2026-11-13 | dependency (θ → 9521), not a check |
| "In effect" is known | task 9521 done: its step-4 `KillMode=mixed` check and step-6 live one-SIGTERM restart | dependency (θ → 9521), not a check |
| Ordering | θ → ζ, θ → ι (`shared/fleet_coordination.py`), θ → 9493 (done), θ → 9521; its one-line `OPERATIONS.md` edit is exempt from the same-file chain (lead ruling), so it lands as soon as 9521 is done; ω → θ | PASS |

## η — Reclaim over known roots

| Capability | Binding | Verdict |
|---|---|---|
| The reclaim reads `known_project_roots` | absent today | PASS (producer check) |
| The single-root pin goes | `REPO:-/home/leo/src/dark-factory` in the wrapper today | PASS (producer check) |
| One parser of the known-roots env exists | `shared.project_layout.known_project_roots` (S14 α 6710) | PASS (substrate) |
| A timer run reports per root and reclaims stale parkings | live confirmation after the operator installs the drop-in | OPEN (manual) |
| Ordering | η → ι, η → 9140; δ1 → η (`OPERATIONS.md`, the same-file rule); lead OQ-3 A | PASS |

## ω — Integration gate

| Capability | Binding | Verdict |
|---|---|---|
| The cross-seam boundary suite exists | absent today | PASS (producer check) |
| The duplicated measurement paragraph goes | 4 copies today | PASS (producer check) |
| The false claims go | three of four present today ("only uncapped" went with task 8270) | PASS (producer check) |
| One loader per hyphen-named watchdog | 6 `_load_watchdog` copies today, one of them the dashboard's (task 8362) | PASS (producer check) |
| `load_dashboard_watchdog()` lives in `tests/scripts/conftest.py` | absent today | PASS (producer check) |
| The supervision invariants have one home | no `### Supervision invariants` in OPERATIONS.md today | PASS (producer check) |
| The live `--report` shows the whole picture | live confirmation after merge | OPEN (manual) |
| Ordering | ω → ε, ω → θ (`shared/fleet_coordination.py`; ω therefore waits on 9521 through θ), ω → 6726 | PASS |
