# Capability manifest — orchestrator-fleet-redeploy-throughput-prd

Mechanizes G3 + G6 per leaf for `plans/orchestrator-fleet-redeploy-throughput-prd.md`.
Built at decompose time (2026-07-09) by re-verifying every asserted capability
against `main`. Evidence forms: `grep:<file>:<line>` (wired on main),
`producer:task-<label>` (delivered by an **upstream** task in the dep closure),
`existing` (verified-present substrate), `self` (the task builds it and its own
signal observes it firing). **No FAIL bindings** — batch clears the manifest gate.

DAG (consumer depends_on producer): γ→α; ε→α; ε→β; ε→γ; ε→δ; ζ→ε; δ→β.

---

## α — merge-idle heartbeat producer  *(intermediate; consumers γ, ε)*

Signal: `<fleet_heartbeat_dir>/<ORCH_UNIT>.json` appears and `ts_epoch` advances
per running unit; a forced synthetic in-flight merge flips `merge_idle:false`.

| Capability | Evidence | Verdict |
|---|---|---|
| `Harness._merge_pipeline_idle()` — truth source for `merge_idle`/`depth`/`queue_empty` | `grep:orchestrator/src/orchestrator/harness.py:6672` (defined; wired/called at `:6813`) | PASS |
| Atomic write pattern for the per-tick file | `grep:orchestrator/src/orchestrator/scheduler.py:4493` (`_write_state_snapshot_raw`) | PASS |
| `ORCH_UNIT` env available to key the file | `grep:orchestrator/src/orchestrator/deterministic_runner.py:336` (`os.environ.get('ORCH_UNIT')`) — set in each unit's `[Service] Environment` | PASS (existing) |
| `<fleet_heartbeat_dir>` (`data/fleet/`) — the new fleet-common dir | `self` (α creates it; Open-Q2 resolution = fleet-common `data/fleet/<unit>.json`) | PASS |
| Run-loop tick hook to emit each tick | `grep:orchestrator/src/orchestrator/harness.py` run-loop (idle+busy branches per RCA `:1466-1575`) | PASS |

*G3 note:* the externally-observable per-unit merge signal is genuinely absent
today — `grep merge_idle\|merge_pipeline\|merge_depth orchestrator/src/orchestrator/scheduler.py`
returns nothing (snapshot carries holders/locks/pause only), and the escalation
server exposes no plain-HTTP health route. α is therefore the one novel
substrate and a hard prerequisite of γ (α→γ wired).

---

## β — shared fleet-deploy clock, both tiers honor + stamp-on-verify  *(intermediate; consumers δ, ε)*

Signal: with an injected clock + fake systemctl, a stale unit inside the 8h
window is NOT restarted (I1); a failed verify leaves the clock file UNCHANGED (I2).

| Capability | Evidence | Verdict |
|---|---|---|
| Clock file `data/orchestrator/last_redeploy_orchestrator.json` (reused as the single fleet clock) | `grep:orchestrator/src/orchestrator/harness.py:6799` (coordinator already writes it) | PASS |
| `orchestrator_restart_min_interval_secs` config (28800 default) | `grep:orchestrator/src/orchestrator/config.py:1922` | PASS |
| Coordinator honors `min_interval` (extend to shared clock) | `grep:orchestrator/src/orchestrator/service_restart.py:437-446` | PASS |
| Script exit-0 verified path to hook stamp onto | `grep:scripts/restart-all-orchestrators.sh:132` (`exit 0`) + per-unit fresh-mono verify `:77-93` | PASS |
| Watchdog reads/writes the clock + delegates staleness to the script (NEW wiring) | `self` — watchdog today has ZERO clock-awareness (`grep min_interval\|last_redeploy scripts/orchestrator-watchdog.py` = none; only `STALENESS_GRACE_SECS`, `:90`); β builds it | PASS |
| **I2 rejection** — failed/partial verify does NOT advance clock | `rejection-check:self` — β adds stamp-on-verified-success (exit-0 gate); β's recorder test authors a failed verify and observes the file unchanged | PASS |
| Watchdog `staleness_pass()` / `restart_unit()` seam to redirect through the script | `grep:scripts/orchestrator-watchdog.py:469` (`staleness_pass`), `:190` (`restart_unit`), `:493` (grace gate) | PASS |

---

## γ — drain gate in the restart chokepoint  *(intermediate; consumer ε; consumes α)*

Signal: a unit with `merge_idle:false` is skipped with a journal line and
restarted once it idles; a synthetic 75-min-busy unit is force-restarted (I3/I4).

| Capability | Evidence | Verdict |
|---|---|---|
| Read unit U's heartbeat `<dir>/<unit>.json` | `producer:task-α` (upstream; α→γ wired) | PASS |
| `python3` on-host for the in-bash JSON parse (not `jq`) | `existing` (PRD-verified; `python3 -c` used across scripts) | PASS |
| Restart chokepoint to insert the gate (the `--drain` no-op is the seam) | `grep:scripts/restart-all-orchestrators.sh:41` (`--drain` accepted-and-ignored → real drain lands here) | PASS |
| `force_fire_after_secs` (75-min defer bound) read as env/constant, default 4500 | `existing` — the drain gate reads the value independently (env/bash default), NOT via δ's coordinator code; consistent with the DAG (γ does **not** depend on δ). "One knob, double duty" is semantic, realized as two independent readers | PASS |
| **I4** stale/absent heartbeat ⇒ restartable after unknown-grace (fail-toward-convergence) | `self` — γ's gate + test assert the absent-heartbeat direction; Open-Q3 resolution = ~120s / 2× tick | PASS |

---

## δ — coordinator fire-while-busy  *(intermediate; consumer ε; depends β)*

Signal: on a saturated (never-idle) fleet with pending set and the 8h clock
open, an `orch-selfrestart-on-merge-*` transient unit is registered within 75
min of eligibility (I6).

| Capability | Evidence | Verdict |
|---|---|---|
| Coordinator `note_merge` / `maybe_restart(agents_idle)` / `_pending` / `require_idle` | `grep:orchestrator/src/orchestrator/service_restart.py:346,398,319,295` | PASS |
| Non-resetting `first_pending_monotonic` (owed-age measure) | `self` — δ builds it; scaffold/intent already noted `grep:service_restart.py:325` ("NOT reset by note_merge re-arming") | PASS |
| Shared 8h clock to honor | `producer:task-β` (upstream; δ→β wired) | PASS |
| `orchestrator_restart_force_fire_after_secs` config knob (4500) | `self` — δ owns the coordinator-side config addition (`config.py`) | PASS |
| Drop `_merge_pipeline_idle` hard precondition, keep as preference | `grep:orchestrator/src/orchestrator/harness.py:6760,6813` (current hard precondition wiring δ relaxes) | PASS |

---

## ε — integration gate + `--report` extension + operator docs  *(intermediate; consumer ζ; depends α,β,γ,δ)*

Signal: end-to-end boundary scenarios 1–10 (PRD §Boundary-test sketch);
`--report` adds merge-idle + deploy-age + would-defer columns with zero mutating
systemctl calls (I8).

| Capability | Evidence | Verdict |
|---|---|---|
| All of α,β,γ,δ landed (the scenarios exercise them) | `producer:task-α,β,γ,δ` (all upstream; ε deps wired) | PASS |
| `--report` read-only doctor mode to extend | `grep:scripts/orchestrator-watchdog.py:536` (`report()`), `:589-601` (never mutates systemd) | PASS |
| **I8** read-only — zero mutating systemctl calls / no clock write | `existing` — `report()` is already non-mutating (`:594` "not invoked, so this path never mutates systemd") | PASS |
| **I9** crash-safe force-restart (existing-behavior link test) | `grep:orchestrator/src/orchestrator/merge_queue_store.py:247` (`recover_pending_merges`) | PASS |
| CLAUDE.md ops note (three orthogonal concerns; when to read `--report`; soak pointers) | `self` — docs deliverable | PASS |

---

## ζ — deterministic deploy capstone  *(leaf; `task_kind='deterministic'`)*

Signal: `done_provenance kind='deterministic-deploy-scheduled'`; journal shows
the drain-aware restart + a fresh clock stamp.

| Capability | Evidence | Verdict |
|---|---|---|
| DeterministicRunner self-restart path (`target_unit==own` → detached `systemd-run --user`, done=`scheduled`, `kind='deterministic-deploy-scheduled'`) | `grep:orchestrator/src/orchestrator/deterministic_runner.py:330-336` + CLAUDE.md "Runner stamps" | PASS |
| Detached path honors `cwd` (relative script no longer 127s) | `producer:task-2105` (done, merged `3846b830c0`) | PASS |
| `before_done.script = scripts/restart-all-orchestrators.sh` exists & executable | `existing` — `ls` = `-rwxrwxr-x` (100755) | PASS |
| Drain-aware restart embedded in that script | `producer:task-γ` (transitively upstream via ε) | PASS |
| Fresh clock stamp on the exit-0 path | `producer:task-β` (transitively upstream via ε) | PASS |
| Journal observability of the restart + stamp | `existing` (journalctl operator surface) | PASS |

---

**Gate result:** every leaf/intermediate capability binds to PASS evidence.
No `declared-only` / `test-only` / `producer-downstream` / `producer-absent` /
`producer-extent-short` / `rejection-absent` / `bound≤floor` bindings. Batch is
clear to queue.
