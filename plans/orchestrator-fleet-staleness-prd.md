# PRD: Orchestrator fleet staleness — event-driven restart-all + watchdog backstop

**Status:** active — authored 2026-07-02 (design session; user AFK, brief at
`~/.claude/spawn-briefs/prd-2003-orchestrator-staleness.md` folded in;
recommended defaults adopted for surfaced choices, all noted in §Resolved).
**Project:** dark_factory. **Supersedes:** the design half of task 2003
(blocked, G5-routed here) and esc-1969-36's deferred U2 activation.
**Escalation:** esc-2003-58 stays pending — the human resolves it against this
PRD. **Approach:** light B+H (contract + boundary-test sketch; G5 heuristic:
operationally high-stakes seam — auto-restart of every orchestrator unit).

## Goal

After a merge touching orchestrator-core code (`orchestrator/src/**`,
`escalation/src/**`) lands on dark-factory main, **every** running
`orchestrator-*.service` user unit — not just orchestrator-dark-factory — is
restarted onto the new code automatically, politely (debounced, df-idle,
merge-drained), and verifiably (fresh `ActiveEnterTimestampMonotonic` per
unit). Independently, a periodic staleness backstop catches every path the
event-driven hook can miss, and an operator can run a read-only report that
enumerates the fleet and flags stale daemons.

User-observable surfaces:
- Post-merge: journal `service_restart` flow + all six units showing fresh
  `ActiveEnterTimestamp` newer than the merge.
- `scripts/orchestrator-watchdog.py --report`: per-unit staleness table,
  exit 0 all-fresh / 1 any-stale.
- Watchdog journal (`systemd-cat -t orchestrator-watchdog`): staleness-restart
  WARNING lines when the backstop acts.

## Background — incidents and existing mechanism inventory

Two confirmed live incidents (task 2003): orchestrator-know-live.service ran
stale orchestrator-core bytecode across a rename
(`classify_simple_task` → `is_declared_simple_task`, 8620f2999a), throwing
dispatch-time ImportError on every triage for ~3.5 h until a human bounced it
(2026-05-31 PID 29355; 2026-07-02 PID 5560). All six per-project daemons run
the SAME checkout (`WorkingDirectory=/home/leo/src/dark-factory`,
`uv run --frozen --project orchestrator`, verified in every unit file), so one
core merge staleness-invalidates the whole fleet at once. Remediation today is
manual per-incident capstone tasks (1800/1858/1863/1866/1875 precedent).

What already exists on main (all verified this session):

| Mechanism | Where | State |
|---|---|---|
| `StaleServiceRestartCoordinator` U2 instance (self-restart on core merge) | `harness.py:5289` `_build_orchestrator_restart_coordinator`, fan-in at `_note_merge_all` (harness.py:5047/5067), fired from `_maybe_restart_stale_service` (harness.py:5352) | wired, **dormant**: `orchestrator_restart_on_merge_enabled` defaults False (config.py:1741) and df's `orchestrator/config.yaml` doesn't set it |
| Cgroup-escaping detached restart | `service_restart.py:78` `schedule_detached_systemd_restart` (task 1973) | merged, exercised |
| Fleet restart script | `scripts/restart-all-orchestrators.sh` (100755, blob c01fad7166): runtime enumeration of running `orchestrator-*.service`, per-unit fresh-start verify, SELF_UNIT deferred last | merged; used by deploy capstones df 2002/2009 |
| Single-unit restart script (U1) | `scripts/restart-orchestrator.sh` (task 1969) | merged, operator-proven |
| Liveness watchdog | `scripts/orchestrator-watchdog.py` + `orchestrator-watchdog.timer` (60 s, ACTIVE) | running; **WATCHED hardcodes only 3 of 6 units** — know-live (both incidents!), autopilot-video, solar-challenge-platform are unwatched |
| DeterministicRunner own-unit detached deploy path | deterministic_runner.py; terminal-transition stall fixed by task 2004 (merged 3590d655) | merged (live daemon may predate the fix until next restart) |

The gap: the coordinator's script restarts only the df unit, and nothing at
all watches the other five daemons for staleness. `orchestrator/src` merges
land **only** through the df merge queue — no other daemon can observe them.

## Architecture decision

**Chosen: two-tier synthesis — (1) event-driven fleet restart via the existing
U2 coordinator repointed at `restart-all-orchestrators.sh` (a pure config
change), plus (2) a periodic staleness backstop + read-only doctor in the
existing watchdog.**

Why this shape:
- The df daemon is the **unique observer** of core merges (they exist only in
  its merge queue), so central event-driven detection is structurally correct;
  the coordinator already provides debounce, idle-gating, merge-drain
  precondition, transient-failure retry, and the cgroup-escaping detached
  executor. Scope-widening it is a config repoint, zero orchestrator/src code.
- The event path has known blind spots, each covered by the backstop:
  df daemon down/crashed with the in-memory pending flag lost; direct-to-main
  commits that bypass the merge worker (they never reach `on_merge_landed`);
  fire-time script failures inside the transient unit (accepted 1973 gap —
  journald-only); the coordinator's give-up-after-3-transient-failures path;
  the knob accidentally off. A 60 s-timer staleness probe converges all of
  them within minutes, and its restart makes the unit fresh, so it is
  self-clearing and cannot flap.

Rejected as primary:
- **A — per-daemon self-check:** six independent decision-makers polling git
  and racing each other and the coordinator; requires new in-daemon machinery
  plus config edits in five foreign project repos, for a fleet co-located on
  one host reading one repo. Strictly more moving parts for the same effect.
- **C — doctor only:** detection without remediation re-inserts a human into
  a fully mechanizable action, contrary to the standing autonomy directive.
  Absorbed: the backstop's probe IS the doctor; `--report` exposes it
  read-only.
- **D — auto-file a deterministic deploy per core merge:** duplicates the
  coordinator's exact function with more moving parts (filer, task churn per
  merge, dispatch latency, runner) and no politeness gates; its one advantage
  (failure → born-at-L2) is substantially recovered by the backstop + the
  liveness watchdog reviving failed-to-return units within ~2 min.
  Deterministic deploys remain the right tool for *batch capstones* (2002,
  2009, and ε below) — just not as the standing per-merge mechanism.

## Resolved design decisions

1. **Fleet restart is config-only.** γ edits `orchestrator/config.yaml`
   (dark-factory's own daemon config) to set
   `orchestrator_restart_on_merge_enabled: true`,
   `orchestrator_restart_script: scripts/restart-all-orchestrators.sh`, and
   `orchestrator_restart_watch_prefixes: [orchestrator/src/, escalation/src/,
   orchestrator/pyproject.toml, orchestrator/uv.lock,
   escalation/pyproject.toml]` (escalation runs in-process — harness.py:5381;
   dependency-manifest changes also invalidate the frozen env; exact-path
   match is supported — service_restart.py:60). All other knobs keep defaults
   (debounce 300 s, on_active 10 s). No coordinator/harness code change; the
   politeness gates and burst-coalescing are existing tested behavior.
2. **Operator sign-off = resolving esc-2003-58.** config.py:1748 requires an
   operator action to flip the knob after sign-off; the flip ships as a
   reviewed commit (γ) and only ACTIVATES when the daemon restarts (ε), after
   the human has resolved esc-2003-58 against this PRD. Soak observation:
   `journalctl --user -u 'orch-selfrestart-on-merge-*'` for fire-time
   failures, plus reconciliation's existing staleness detection (the path
   that filed task 2003) as the independent detector of residual gaps.
3. **Busy-daemon restart semantics: SIGTERM graceful drain, accepted.** The
   df-side gates (idle + merge-drained + debounce) make the *fire* polite for
   df; other daemons get `systemctl restart` = SIGTERM with
   `TimeoutStopSec=90` (shutdown cancels in-flight tasks, reaps agents,
   releases locks; state re-dispatches from fused-memory). Identical to the
   already-sanctioned restart-all deploy capstones (2002/2009). Cross-daemon
   idle coordination is out of scope v1.
4. **Staleness criterion:** unit `ExecMainStartTimestamp` (realtime; query
   precedent at orchestrator-watchdog.py:149 for the monotonic twin) older
   than `git log -1 --format=%ct HEAD -- <watched paths>` (max committer
   time) in /home/leo/src/dark-factory. Committer time ≈ landing time because
   the merge queue rebases immediately before merging and direct-to-main
   commits are committed at landing. File mtimes rejected (editor/checkout
   perturbation, no provenance).
5. **Backstop restraint (anti-flap):** the staleness pass restarts a unit
   only when ALL hold: stale per (4); newest watched commit older than
   `STALENESS_GRACE_SECS = 1800` (30 min head start for the polite event
   path); outside the existing `STARTUP_GRACE_SECS = 120`; unit enabled
   (`is_unit_enabled` respected, existing). A restart refreshes the start
   timestamp, so staleness self-clears — no stored state, no flap loop.
6. **Enumeration:** the staleness pass enumerates dynamically
   (`systemctl --user list-units 'orchestrator-*.service'` — same source as
   restart-all-orchestrators.sh) so new projects are covered automatically.
   The liveness port list stays hardcoded + drift-tested (ports live in
   foreign repos' configs; runtime parsing adds failure modes for zero drift
   benefit given the test) and is extended 3→6: df 8102, reify 8100,
   my-solar-challenge 8106 (existing) + know-live 8105, autopilot-video 8101,
   solar-challenge-platform 8107 (verified against each project's committed
   orchestrator yaml this session).
7. **Watchdog stays a stdlib-only oneshot.** New code lands in the same
   script; the timer re-execs it every 60 s, so script changes deploy on the
   next tick with no unit-file edits and no daemon restarts.
8. **Failure visibility:** coordinator fire-time failures remain
   journald-only (accepted 1973 gap), but the backstop re-converges any
   still-stale unit ≤ grace + one tick later, and the (now fleet-wide)
   liveness probe revives any unit that fails to come back within ~2 min.
   Backstop actions log WARNING via the existing `systemd-cat -t
   orchestrator-watchdog` tag. Watchdog→L2 escalation wiring: out of scope
   v1 (recon remains the independent detector).

## Contract (invariants)

- **I1 single observer.** Only the df daemon's coordinator fires event-driven
  fleet restarts; other daemons keep `orchestrator_restart_on_merge_enabled`
  false (their merge diffs contain no watched paths regardless).
- **I2 coalescing.** A burst of core merges produces exactly one restart-all
  fire (existing debounce/pending semantics, re-asserted under the new
  config).
- **I3 politeness unchanged.** Fire only at agents-idle + merge pipeline
  drained + debounce elapsed — the config repoint must not alter coordinator
  gate behavior.
- **I4 self-last.** restart-all defers SELF_UNIT
  (orchestrator-dark-factory.service) to last, so a mid-script df death
  cannot strand other units unrestarted (script-guaranteed, on main).
- **I5 backstop restraint.** The staleness pass never restarts a unit that is
  fresh, within startup grace, disabled, or stale only w.r.t. a commit
  younger than STALENESS_GRACE_SECS (negative assertion — mechanism authored
  in α, observed firing in α tests + δ scenarios 2–4).
- **I6 convergence, no stored state.** Staleness is computed from live
  systemd + git state each tick; any successful restart (coordinator,
  backstop, capstone, operator U1) makes the unit read fresh immediately.
- **I7 read-only doctor.** `--report` performs zero mutating systemctl calls
  (asserted via injected command recorder).
- **I8 idempotent overlap.** All restart paths converge on per-unit systemctl
  jobs, which systemd serializes; overlapping invocations degrade to at most
  one redundant restart (made unlikely by I5's grace window).

## Pre-conditions for activating

None external — every assumed capability exists on main today (G3 verified;
see the capability manifest beside this PRD). The one live-process caveat:
the running df daemon may predate task 2004's own-unit stall fix (merged
3590d655 after the daemon's last restart), so ε carries the known stall
signature + remediation in its description.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/merge-queue-modularization-invariants-prd.md` (df 1985–2002, in flight) | consumes | `on_merge_landed` → `Harness._note_merge_all` → `coordinator.note_merge` chokepoint | other PRD (its behavior-preservation contract keeps the chokepoint stable); this PRD is a read-only consumer, touches no orchestrator/src file | wired (on main) |
| `plans/config-hot-reload-prd.md` (df 2005–2009, in flight) | adjacency only | none — `orchestrator_restart_*` knobs are NOT in the reload allowlist, so the γ flip activates via restart (ε), not reload; we edit `orchestrator/config.yaml`, they edit `orchestrator/src/orchestrator/config.py` | n/a | n/a |

Both in-flight batches end in restart-all deploy capstones (2002, 2009);
ours (ε) is idempotent with them (I8) — whichever lands last simply verifies
already-fresh units.

## Decomposition plan

DAG: α → β → δ; γ → δ; δ → ε. (α/β share files, serialized; γ independent.)

- **α — watchdog staleness probe + `--report` doctor mode** (intermediate →
  unlocks β, δ; carries its own observable signal). Extend
  `scripts/orchestrator-watchdog.py`: dynamic unit enumeration, staleness
  criterion per §Resolved 4, restraint gates per §Resolved 5, restart via the
  existing `restart_unit`, WARNING journal lines; `--report` prints a
  per-unit table (unit, start time, newest watched commit, verdict) and exits
  0/1 without mutating (I7). **Signal:** `--report` CLI output + exit code;
  with injected fake systemctl/git, a stale-beyond-grace unit is restarted on
  the timer path and the next probe reports fresh. Modules:
  scripts/orchestrator-watchdog.py, tests/scripts/test_orchestrator_watchdog.py.
- **β — liveness WATCHED 3→6** (leaf). Add (8105, know-live), (8101,
  autopilot-video), (8107, solar-challenge-platform) to WATCHED; extend the
  existing port-parity drift test to all six. **Signal:** extended drift test
  green in CI; post-deploy, a down know-live escalation port is revived
  within ~2 min (soak). Depends: α (same files). Consumer: the active
  orchestrator-watchdog.timer. `complexity=simple`.
- **γ — df config flip + config-integrity drift test** (leaf). Edit
  `orchestrator/config.yaml` per §Resolved 1; add a drift test asserting the
  committed df config sets the knob true, the configured script exists and is
  executable on main, and every watched prefix/path exists (a typo'd script
  path fails FileNotFoundError-fail-open at fire time — the test makes that
  impossible to ship). **Signal:** drift test green in CI; post-activation,
  the next core merge logs `orchestrator restart pending: merge …` in the df
  journal. `complexity=simple`.
- **δ — integration gate + operator docs** (leaf; boundary-test sketch below
  is its signal). End-to-end scenarios 1–10; coordinator-composition test
  loading the committed `orchestrator/config.yaml` and asserting
  `_build_orchestrator_restart_coordinator` plumbs the restart-all script
  path + prefixes through (ties deployment config to code path); operator
  docs (CLAUDE.md orchestrator/ops note: two-tier semantics, when to run
  `--report`, soak-watch pointers). Depends: α, β, γ.
- **ε — deterministic deploy capstone** (leaf; `task_kind='deterministic'`,
  mirrors df 2009). `before_done.script =
  scripts/restart-all-orchestrators.sh`, `timeout_secs=900`,
  `target_unit=orchestrator-dark-factory.service` (own unit → detached
  systemd-run path, done='scheduled'), `always_escalates=false`. Restarts the
  whole fleet: activates γ's flip in the df daemon, brings all six units onto
  current HEAD (clearing today's accumulated staleness), and loads task
  2004's runner fix. **Signal:** `done_provenance
  kind='deterministic-deploy-scheduled'`; journal shows every running
  orchestrator unit verified fresh. Depends: δ. Description carries the
  1976/1982 stall signature + manual remediation (run script, verify, set
  done) in case the live runner predates the 2004 fix.

G2 note: α is an intermediate with a genuine user-observable signal of its
own (the doctor CLI); β/γ/δ/ε are leaves. G6: both negative assertions (I5,
I7) are backed by mechanisms produced upstream-or-same-task and observed
firing in α's tests and δ's scenarios; no numeric-accuracy premises (grace
values are policy constants, not measurement claims).

## Boundary-test sketch (δ's signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Stale unit beyond grace | fake systemctl/git: start < commit, commit age > 1800 s | timer path restarts it; WARNING journal line; next probe fresh (I6) |
| 2 | Stale unit, commit younger than grace | commit age 300 s | no restart (I5) |
| 3 | Fresh unit | start > commit | untouched (I5) |
| 4 | Disabled stale unit | is-enabled non-zero | skipped (I5) |
| 5 | `--report`, mixed fleet | 6 fake units, one stale | table lists all six with verdicts; exit 1; recorder shows zero stop/start/restart calls (I7) |
| 6 | `--report`, all fresh | — | exit 0 |
| 7 | Config-integrity drift (γ) | committed orchestrator/config.yaml | knob true; script exists + 100755; every watched path exists |
| 8 | WATCHED parity (β) | committed watchdog | six entries; ports match the drift-test fixture |
| 9 | Coordinator composition | Config loaded from committed df yaml | builder yields coordinator with restart-all script path + new prefixes; executor registers a `systemd-run` transient targeting it (existing executor tests re-parametrized) (I3) |
| 10 | Burst coalescing under new config | two note_merge calls inside debounce | exactly one fire (I2) |

## Out of scope

- Cross-daemon idle coordination before fleet restart (per-unit drain via
  each escalation MCP) — future refinement if SIGTERM-drain proves too
  disruptive in practice.
- Watchdog → escalation-queue (L2) wiring; journald WARNING + recon remain
  the alerting path in v1.
- Hot code reload (config hot-reload PRD explicitly excludes code; restarts
  remain the code-deploy mechanism).
- Editing foreign project repos' orchestrator configs (their coordinators
  stay dormant; I1).
- System-level (non-user) systemd; multi-host fleets.

## Open questions (surfaced but not decided in this session)

1. **STALENESS_GRACE_SECS tunability.** Constant vs env override. **Suggested
   resolution:** module constant with env override mirroring
   `RESTART_VERIFY_TIMEOUT`'s pattern in restart-all. Decide in α.
2. **Dependency-manifest watch set.** Whether `orchestrator/uv.lock` churn is
   too noisy a restart trigger (lockfile edits without src changes are rare
   but possible). **Suggested resolution:** keep it — a frozen-env change IS
   staleness; γ's drift test enforces only that listed paths exist. Decide in
   γ.
3. **`--report` also showing liveness.** The doctor could merge the port
   probe into its table. **Suggested resolution:** yes if free, staleness-only
   otherwise. Decide in α.
4. **Doc placement.** CLAUDE.md ops note vs skills/escalation-watcher
   reference vs both. Decide in δ.
