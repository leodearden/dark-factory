# PRD: Orchestrator fleet redeploy — throughput-preserving (shared 8h bound + drain-aware restart + fire-while-busy)

**Status:** active — authored 2026-07-09 (design session; root cause + policy
decided with the operator, recorded in memory
`project_orch_restart_churn_rca_2026_07_08.md`).
**Project:** dark_factory.
**Supersedes/extends:** the two-tier design in
`plans/orchestrator-fleet-staleness-prd.md` (event coordinator + watchdog
backstop) and the task-2371 8h self-redeploy rate cap (which today is
coordinator-only and defeated in practice).
**Approach:** B + H (contracts + two-way boundary tests) — G5 high-stakes:
this changes *when and how every orchestrator unit on the host is restarted*.

## Goal

Stop the orchestrator fleet from restart-thrashing its own (and especially
**Reify's**) in-flight merges. Today a df `orchestrator/src` / `escalation/src`
merge staleness-invalidates the whole shared-checkout fleet, and the watchdog
staleness backstop SIGTERM-restarts every unit ~once every 2.5 h with no drain
and no rate cap — killing Reify verifies that take >30 min, which re-enqueue
and re-verify from scratch, livelocking Reify throughput and oversubscribing
the box.

After this PRD:
1. **Rate:** the fleet is redeployed for staleness **at most once per 8 h**
   (`orchestrator_restart_min_interval_secs`, default 28800), enforced by a
   **single persisted fleet-deploy clock honored by BOTH tiers** — the
   event-driven coordinator and the watchdog backstop. (The 8h cap is
   currently wired into the coordinator only and completely ignored by the
   backstop.)
2. **Drain:** a fleet redeploy **does not kill an in-flight merge** — each unit
   is restarted only at a merge-idle moment, with a bounded 75-min drain-grace
   before a still-continuously-busy unit is force-restarted.
3. **Promptness (secondary):** df's own once-per-8h deploy fires promptly and
   politely from the event coordinator **even while the fleet is busy**
   (fire-while-busy), instead of being starved of the idle quiet-window and
   always losing to the 30-min backstop.
4. **Liveness is untouched:** a down / wedged escalation port still triggers an
   immediate per-unit revive, uncapped and not drain-gated (brokenness ≠ a
   scheduled deploy).

**Accepted cost (operator-ratified):** a unit may run up to ~8 h stale between
deploys; the task-2003 "healthy daemon running stale bytecode" class (which the
liveness probe cannot catch) can persist up to ~8 h; and a unit continuously
merging for the full 75-min drain-grace loses at most **one** re-verified merge
per 8 h window (which the current backstop already incurs every 2.5 h).

## User-observable surfaces (consumer — G1)

The consumer is the **operator + fleet task throughput**, especially Reify's
merge completion rate. Observable signals:
- **Watchdog journal** (`journalctl --user -t orchestrator-watchdog`): staleness
  restarts drop to **≤1 per 8 h**; a "skip: <8h since last fleet deploy" line
  when gated; a "deferring restart of <unit>: mid-merge (grace 75m)" line when
  drain-gated.
- **`scripts/orchestrator-watchdog.py --report`** gains two columns: per-unit
  **merge-idle** and **age-since-last-fleet-deploy**, plus a per-unit
  would-defer verdict.
- **Reify:** `orchestrator-reify.service` staleness restarts fall from ~1/2.5 h
  to ≤1/8 h; a Reify merge in flight at a deploy tick survives (heartbeat
  non-idle → deferred).
- **df coordinator:** on a saturated fleet, an `orch-selfrestart-on-merge-*`
  transient unit appears within ~75 min of fire-eligibility (today: never).
- **Fleet-deploy clock file** present and advanced only on a *verified* fleet
  restart.

## Background — what is actually broken (verified this session)

The `orchestrator-fleet-staleness` PRD shipped a two-tier design: (1) an
event-driven `StaleServiceRestartCoordinator` (U2) that fires
`restart-all-orchestrators.sh` at df's idle + merge-drained quiet-window, and
(2) a watchdog `staleness_pass()` backstop that restarts any unit older than
the newest watched commit after `STALENESS_GRACE_SECS=1800` (30-min head
start). Task 2371 later added an 8h `min_interval` rate cap.

Three defects compound (all evidenced in the RCA memory):

1. **Idle-window starvation.** The coordinator's fire path
   (`maybe_restart(agents_idle=True)`) is reachable **only** from the run-loop
   idle branch (`harness.py:1558`), entered only when `acquire_next() is None
   AND not active`. The fleet has been continuously saturated since 2026-07-07
   (idle-branch log line count: 187 on Jul 06 → **0** on Jul 07/08/09), so the
   coordinator has not fired since Jul 06 09:59. `require_idle=True` plus the
   `_merge_pipeline_idle` precondition make its fire condition effectively
   unsatisfiable under saturation.
2. **The 8h cap is defeated.** `min_interval` (task 2371) was wired into the
   coordinator only (`config.py` / `service_restart.py` / `harness.py`); the
   watchdog has **zero** awareness of it (no `min_interval` / `last_redeploy`
   reference — only `STALENESS_GRACE_SECS`). Empirically the backstop issued
   **20 df staleness restarts in 49 h (~1/2.5 h)**, ~8× the intended 1/8 h. The
   two tiers share no state, so the cap throttles nothing the fleet actually
   does.
3. **No drain.** `restart-all-orchestrators.sh` accepts-and-**ignores**
   `--drain`; it `systemctl restart`s each unit (SIGTERM, `TimeoutStopSec`),
   cancelling in-flight work. Reify verifies (>30 min) are repeatedly killed
   and re-enqueued (`recover_pending_merges` re-verifies from scratch) →
   livelock + wasted CPU → box oversubscription that further slows verifies.

Crash-safety (verified): a mid-merge restart is **safe** —
`recover_pending_merges` (`merge_queue_store.py:247`) replays the durable
journal on boot and drops any record whose branch is gone / already an ancestor
of main (idempotency, no double-merge) / whose worktree was pruned; otherwise
re-enqueues. So "never fire mid-merge" is a **throughput/efficiency** property,
not a correctness one — which is why drain-awareness (not a hard prohibition)
is the right tool.

## Architecture decision

**Single restart chokepoint, two decision-makers, one shared clock, one drain
gate, one heartbeat producer.**

- **`scripts/restart-all-orchestrators.sh` becomes the sole fleet-restart
  chokepoint.** It gains (a) a **per-unit merge-drain gate** — before
  restarting unit U, read U's merge heartbeat; if U is mid-merge and its
  heartbeat is fresh, defer U (skip this pass) up to a 75-min continuous-busy
  drain-grace, then force-restart; (b) **stamp-on-verified-success** — on the
  existing all-units-verified-fresh exit-0 path, atomically write the shared
  fleet-deploy clock. SELF_UNIT-last ordering and per-unit fresh-timestamp
  verify are retained.
- **Both tiers invoke this one script and gate on the one shared clock.** The
  coordinator (event-driven, fire-while-busy) and the watchdog backstop
  (staleness, after the coordinator's head-start grace) each check
  `now − last_fleet_deploy ≥ 8h` before invoking; neither redeploys inside the
  window. The **backstop stops doing per-unit `restart_unit` for staleness** and
  delegates to the drain-aware script instead — so drain + stamping are
  identical regardless of which tier triggers.
- **A merge-idle heartbeat producer** (new): every orchestrator writes, each
  run-loop tick, a tiny per-unit file to a fleet-common directory keyed by
  `ORCH_UNIT`, carrying `{unit, merge_idle, depth, queue_empty, ts_epoch}` from
  the existing `Harness._merge_pipeline_idle()`. Fleet-common + unit-keyed so
  the stdlib watchdog and the bash script read `<dir>/<unit>.json` by name
  without parsing six foreign project configs.
- **Coordinator fire-while-busy** (secondary): a non-resetting
  `first_pending_monotonic` + `orchestrator_restart_force_fire_after_secs`
  (default 4500 = 75 min); once the 8h clock is open and pending is set, fire at
  the next clean window or force-fire after 75 min of eligibility, bypassing the
  `agents_idle` and 300-s debounce gates. Per-unit drain safety (including df,
  restarted self-last) is delegated to the script's drain gate, so the
  coordinator no longer needs `_merge_pipeline_idle` as a hard precondition.
- **Liveness stays exactly as-is:** `main()`'s port-probe revive is per-unit,
  uncapped, not drain-gated, and does **not** stamp the fleet-deploy clock (a
  single wedged-unit revive is not a fleet deploy).

Why this shape: the shared clock is the only correct way to make "≤1 fleet
deploy per 8h" a *fleet* property rather than a per-tier one; funneling both
tiers through one script means drain + stamping are defined once (no drift);
the heartbeat is the minimal new substrate that makes drain externally
observable to a stdlib reader; fire-while-busy is additive and orthogonal.

## Resolved design decisions

1. **Shared fleet-deploy clock path.** Reuse the coordinator's existing
   persisted file `<df_project_root>/data/orchestrator/last_redeploy_orchestrator.json`
   (`{ts, iso}`) as the single fleet-deploy clock. The watchdog (REPO_DIR is
   the df checkout) reads/writes the same path. Written **only** by a *verified*
   fleet restart (the script's exit-0 path) — never on mere fire/registration,
   closing the "failed detached deploy silences the backstop 8h" hole.
2. **8h cap is the top-priority bound, honored by both tiers.** Neither the
   coordinator nor the backstop restarts for staleness within
   `orchestrator_restart_min_interval_secs` of the stamp. `0` disables (test /
   emergency). The backstop reads it from the same config value (or an env
   mirror for the stdlib watchdog — see Open Q1).
3. **Drain gate = defer-then-force at 75 min of continuous busy.** A unit whose
   fresh heartbeat says `merge_idle:false` is deferred; once a unit has been
   continuously non-idle for `orchestrator_restart_force_fire_after_secs`
   (75 min) it is force-restarted (one re-verified merge, accepted). Reify's
   near-continuous stream still has sub-second inter-merge gaps, so in practice
   the gate catches a gap well inside 75 min and kills ~0 merges most windows.
4. **Missing / stale heartbeat ⇒ proceed after a short grace (fail-toward-
   convergence).** A unit not writing a fresh heartbeat (pre-producer code, or
   wedged) must not block staleness forever; treat unknown as restartable after
   a bounded grace. Rationale: the 8h rate bound already makes restarts rare, and
   a wedged unit is the liveness tier's job. (Symmetric-guard note: this is the
   opposite fail-direction from the drain gate's fresh-non-idle case, and
   deliberately so.)
5. **`force_fire_after_secs` is one knob, double duty:** the coordinator's max
   wait for a clean initiate window AND the per-unit drain gate's max defer.
   Default 4500 (75 min), env/config-overridable.
6. **Coordinator drops `_merge_pipeline_idle` as a hard precondition;** keeps an
   idle *preference*. Merge-drain safety for every unit (df included, self-last)
   is the script's job now. This removes the redundant double-gate and the last
   structural reason the coordinator can't fire under load.
7. **Liveness uncapped + non-stamping** (see Architecture). The three concerns
   stay orthogonal: liveness = brokenness (immediate, per-unit), staleness =
   scheduled fleet deploy (8h-capped, drain-aware), the coordinator = the
   polite event-driven trigger for the same deploy.
8. **Watchdog stays a stdlib-only oneshot;** it shells out to the (bash) script
   for the actual drain-aware restart + stamp, and reads JSON heartbeats via
   `json` (stdlib). The drain gate's JSON parsing inside the bash script uses
   `python3 -c` (already required on-host), not `jq`.

## Contract (invariants)

- **I1 one clock, both tiers.** A fleet staleness redeploy occurs only when
  `now − last_fleet_deploy ≥ min_interval`, checked by BOTH the coordinator and
  the watchdog against the same persisted file. (positive assertion — tested
  with an injected clock + fake stamp.)
- **I2 stamp-on-verify only.** The clock advances iff a fleet restart verified
  all units fresh (script exit 0); a fire-registration or a failed/partial
  restart does NOT advance it. (negative assertion — recorder test: failed
  verify ⇒ file unchanged.)
- **I3 drain, bounded.** A unit with a fresh `merge_idle:false` heartbeat is not
  restarted until it goes idle OR has been continuously busy ≥
  `force_fire_after_secs`; then it is. (two-way: deferred-when-busy AND
  force-restarted-after-grace both asserted.)
- **I4 heartbeat freshness semantics.** A stale/absent heartbeat is treated as
  restartable after the unknown-grace (I5-of-old-PRD analog); a fresh one is
  authoritative. (asserted both directions.)
- **I5 liveness untouched + non-stamping.** A port-down unit is revived
  immediately regardless of the 8h clock or drain state, and that revive does
  not advance the fleet-deploy clock. (asserted.)
- **I6 convergence, no livelock.** Every stale unit is redeployed within
  `min_interval + force_fire_after_secs + one tick`; staleness self-clears on
  restart; no stored state beyond the single clock file. (asserted via the
  end-to-end scenario.)
- **I7 self-last preserved.** SELF_UNIT (df) is still restarted last so a
  mid-script df death can't strand other units (existing script invariant, kept
  under the new drain gate).
- **I8 read-only `--report`.** The extended report performs zero mutating
  systemctl calls and no clock write. (recorder test.)
- **I9 crash-safe merges.** A force-restart mid-merge never double-lands and
  never loses a task (relies on `recover_pending_merges`; asserted by an
  existing-behavior link test, not re-implemented).

## Pre-conditions / substrate (G3)

- **Missing today, must be built first:** an externally-observable per-unit
  merge-idle signal. The scheduler state snapshot (`scheduler.py`
  `_write_snapshot_best_effort`) carries holders/locks/pause but **not** merge
  state; the escalation server exposes no plain-HTTP health route. → the
  heartbeat producer (task α) is a hard prerequisite of the drain gate (task γ).
- **Exists (verified):** `Harness._merge_pipeline_idle()` (the truth source for
  the heartbeat); atomic write pattern (`_write_state_snapshot_raw`);
  `recover_pending_merges` idempotent recovery; `restart-all-orchestrators.sh`
  per-unit fresh-timestamp verify + exit-0 contract + SELF_UNIT-last;
  `ORCH_UNIT` env on every unit; `time.clock_gettime`/`ExecMainStartTimestamp`
  reads in the watchdog; `python3` on-host.
- **Assumption to monitor (G6):** the 8h window must exceed the longest single
  merge-verify, or a merge started right after a deploy could still be killed at
  the next boundary before completing. Reify verifies are >30 min but < 8 h
  today; if a verify ever approaches 8 h the anti-livelock guarantee weakens →
  surface as a soak signal, not a code premise.

## Cross-PRD relationship

| Other PRD | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| `plans/orchestrator-fleet-staleness-prd.md` | supersedes | this PRD replaces that PRD's uncapped 30-min backstop policy + coordinator idle-only fire with the shared-clock + drain + fire-while-busy model; the two-tier *structure* and `--report` are retained and extended | this PRD | landed; being modified here |
| task-2371 8h cap | extends | the coordinator's `min_interval` + persisted clock is generalized into the shared fleet clock both tiers honor | this PRD | landed; extended here |
| `plans/verify-oversubscription-control-prd.md` (PSI worker-admission) | complements | both reduce wasted verify CPU under load; this removes restart-induced re-verify churn, that bounds concurrent verify admission — independent levers on the same throughput pain | that PRD | in flight |

## Decomposition plan

DAG: α → γ; α → ε; β → ε; γ → ε; δ → ε; ε → ζ. (β independent of α; δ depends on
β for the shared clock and may use γ.)

- **α — merge-idle heartbeat producer** (intermediate; unlocks γ; own signal).
  Each orchestrator writes `<fleet_heartbeat_dir>/<ORCH_UNIT>.json` =
  `{unit, merge_idle, depth, queue_empty, ts_epoch}` every run-loop tick from
  `Harness._merge_pipeline_idle()`, atomic write. **Signal:** file appears +
  `ts_epoch` advances for each running unit; forcing a synthetic in-flight merge
  flips `merge_idle:false`. Modules: `harness.py`, a small heartbeat writer +
  test.
- **β — shared fleet-deploy clock, both tiers honor + stamp-on-verify** (leaf).
  `restart-all-orchestrators.sh` writes the clock on its exit-0 verified path;
  the watchdog reads it and skips staleness when `<min_interval`; the
  coordinator reads the same path. Watchdog no longer per-unit-`restart_unit`s
  for staleness — it invokes the script. **Signal:** with an injected clock +
  fake systemctl, a stale unit inside the 8h window is NOT restarted and a
  failed verify leaves the clock unchanged (I1/I2). Modules:
  `orchestrator-watchdog.py`, `restart-all-orchestrators.sh`,
  `service_restart.py` (clock path unification), tests.
- **γ — drain gate in the restart chokepoint** (leaf; consumes α). Before
  restarting unit U, `restart-all-orchestrators.sh` reads U's heartbeat (via
  `python3 -c`); defers a fresh-non-idle U; force-restarts after 75 min
  continuous-busy; proceeds on stale/absent heartbeat after the unknown-grace.
  **Signal:** a unit with `merge_idle:false` is skipped with a journal line and
  restarted once it idles; a synthetic 75-min-busy unit is force-restarted
  (I3/I4). Modules: `restart-all-orchestrators.sh`, a python drain-check helper,
  tests.
- **δ — coordinator fire-while-busy** (leaf; secondary; depends β). Non-resetting
  `first_pending_monotonic`; `orchestrator_restart_force_fire_after_secs`
  (4500); force-fire bypasses `agents_idle` + debounce; drop
  `_merge_pipeline_idle` hard precondition (keep as preference); honor the shared
  8h clock. **Signal:** on a saturated fleet an `orch-selfrestart-on-merge-*`
  transient unit is registered within 75 min of eligibility (I6). Modules:
  `service_restart.py`, `harness.py`, `config.py`, tests.
- **ε — integration gate + `--report` extension + operator docs** (leaf;
  boundary-test sketch below is its signal). End-to-end scenarios 1–10;
  `--report` adds merge-idle + deploy-age + would-defer columns (I8 read-only);
  CLAUDE.md ops note (three orthogonal concerns; when to read `--report`; soak
  pointers). Depends α, β, γ, δ.
- **ζ — deterministic deploy capstone** (leaf; `task_kind='deterministic'`,
  `target_unit=orchestrator-dark-factory.service`, `always_escalates=false`,
  `before_done.script=scripts/restart-all-orchestrators.sh`, `timeout_secs=900`).
  Activates the whole change (config knobs load, heartbeats start, clock begins
  gating) via one drain-aware fleet restart onto current HEAD. **Signal:**
  `done_provenance kind='deterministic-deploy-scheduled'`; journal shows the
  drain-aware restart + a fresh clock stamp. Depends ε.

## Boundary-test sketch (ε's signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Staleness inside 8h window | fake clock: last_deploy 2h ago; a stale unit | neither tier restarts; watchdog logs "skip: <8h" (I1) |
| 2 | Staleness past 8h, all idle | last_deploy 9h ago; heartbeats idle | script restarts all, verifies fresh, stamps clock (I1/I2/I6) |
| 3 | Stamp-on-verify only | restart where one unit fails to verify fresh | script exits 1; clock file UNCHANGED (I2) |
| 4 | Drain-defer | past 8h; unit R heartbeat `merge_idle:false`, fresh | R skipped; "deferring R: mid-merge" line; others restart (I3) |
| 5 | Drain force after grace | R continuously busy ≥75 min | R force-restarted; one merge re-verified (I3) |
| 6 | Absent/stale heartbeat | unit has no fresh heartbeat | restarted after unknown-grace (I4) |
| 7 | Liveness during window | port down, last_deploy 1h ago | unit revived immediately; clock NOT advanced (I5) |
| 8 | Coordinator fire-while-busy | saturated (never idle); pending; 8h open | force-fire after 75 min eligibility → transient unit (I6) |
| 9 | `--report` mixed fleet | one stale, one mid-merge, deploy 3h ago | table shows verdicts + merge-idle + deploy-age; zero mutating calls (I8) |
| 10 | Crash-safe force-restart | force-restart a unit mid-merge | on recovery: no double-land, task not lost (I9, via existing-behavior link) |

## Out of scope

- Graceful in-daemon SIGTERM merge-drain (finish-then-exit with a long
  `TimeoutStopSec`) — the external defer-then-force gate is chosen instead; a
  wedged unit must still die on the systemd timeout.
- Watchdog → escalation-queue (L2) wiring for staleness (recon + journald
  remain the alerting path).
- Cross-daemon *idle* (agent-level) coordination — only *merge*-drain is gated.
- Hot code reload; foreign-project config edits (their coordinators stay
  dormant); non-user systemd; multi-host.

## Open questions (tactical — decide at implementation)

1. **Watchdog's `min_interval` source.** Read df's `orchestrator/config.yaml`
   (adds a yaml read to the stdlib oneshot) vs an env mirror
   (`ORCH_RESTART_MIN_INTERVAL_SECS`, matching the existing
   `STALENESS_GRACE_SECS` env pattern) vs a constant. Suggested: env mirror with
   the 28800 default, drift-tested against the config value. Decide in β.
- 2. **Heartbeat directory.** `<df_repo>/data/fleet/` (all units write there,
  same host/user) vs each unit's own `data/orchestrator/`. Suggested: fleet-common
  `data/fleet/<unit>.json` so the reader needs no per-project path discovery.
  Decide in α.
3. **Unknown-heartbeat grace length.** How long to treat an absent/stale
   heartbeat as "wait" before proceeding. Suggested: 2× the run-loop tick or a
   small constant (e.g. 120 s). Decide in γ.
4. **`--report` liveness merge.** Fold the port probe into the report table or
   keep staleness/merge-only. Suggested: add merge-idle + deploy-age; leave
   liveness to the timer path. Decide in ε.
