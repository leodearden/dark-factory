# Capability manifest — plans/orchestrator-fleet-staleness-prd.md

Per-leaf capability→evidence bindings (mechanizes G3+G6). Verified on main
2026-07-02, PRD commit c631daa063. All bindings PASS — no declared-only /
test-only / producer-downstream / rejection-absent findings.

## α — watchdog staleness probe + --report doctor

- runtime unit enumeration (`systemctl --user list-units 'orchestrator-*.service'`)
  → grep:scripts/restart-all-orchestrators.sh (mapfile running_units; blob
  c01fad7166, 100755, production caller df 2002/2009 capstones) — wired pattern
- unit start-time query → grep:scripts/orchestrator-watchdog.py:149
  `_unit_start_elapsed_secs` (invoked from main() timer path) — wired;
  realtime twin `ExecMainStartTimestamp` via the same `systemctl show` call
- newest-core-commit query (`git log -1 --format=%ct HEAD -- <paths>`) → git
  present; committer-time ≈ landing time because the merge queue rebases
  immediately pre-merge (rebase-before-verify convention) — stated basis
- restart mechanism → grep:scripts/orchestrator-watchdog.py:111 `restart_unit`
  (stop → reset-failed → start), wired in main() — wired
- enabled-respect gate → grep:scripts/orchestrator-watchdog.py:194
  `is_unit_enabled`, wired in main() — wired
- rejection-check (I5 restraint, I7 read-only): mechanism authored in α
  itself; asserted diagnostics (no-restart branches, zero mutating calls)
  observed to fire in α unit tests + δ scenarios 2–6 — producer = this leaf,
  upstream of δ

## β — liveness WATCHED 3→6

- ports: know-live 8105 / autopilot-video 8101 / solar-challenge-platform 8107
  → read this session from each project's committed orchestrator yaml
  (`/home/leo/src/know-live/orchestrator.yaml`, `…/autopilot-video/orchestrator-config.yaml`,
  `…/solar-challenge-platform/orchestrator.yaml`); all six unit files present
  in ~/.config/systemd/user and running — verified
- drift test to extend → grep:tests/scripts/test_orchestrator_watchdog.py
  (committed) — wired
- consumer → orchestrator-watchdog.timer ACTIVE (`systemctl --user is-active`
  = active, 60 s cadence) — wired

## γ — df config flip + config-integrity drift test

- knobs → grep:orchestrator/src/orchestrator/config.py:1741-1756
  (`orchestrator_restart_on_merge_enabled/debounce/watch_prefixes/script/on_active_secs`)
  — declared AND wired: read by
  grep:orchestrator/src/orchestrator/harness.py:5337-5350
  (`_build_orchestrator_restart_coordinator`), fanned in at harness.py:5047
  (`on_merge_landed=self._note_merge_all`), fired from harness.py:5352
  (`_maybe_restart_stale_service`, run-loop idle+busy branches)
- exact-path prefix match (pyproject/uv.lock entries)
  → grep:orchestrator/src/orchestrator/service_restart.py:60-70 (`p == q`
  branch of `diff_touches_watched_paths`) — wired
- target script exists+executable → git ls-files -s
  scripts/restart-all-orchestrators.sh = 100755 c01fad7166 — verified
- every watched path exists on main: orchestrator/src/, escalation/src/,
  orchestrator/pyproject.toml, orchestrator/uv.lock,
  escalation/pyproject.toml — verified via ls this session
- df config file committed → git ls-files -s orchestrator/config.yaml =
  100644 37016c84 (currently sets NO restart knobs → defaults, enabled=False)
  — verified; γ's diff is additive

## δ — integration gate + operator docs

- coordinator test precedent → grep:orchestrator/tests/test_service_restart.py,
  test_harness_service_restart.py, test_merge_queue_restart_hook.py
  (committed) — wired; scenario 9/10 re-parametrize these against the
  committed df yaml
- burst-coalescing + politeness gates (I2, I3) → existing coordinator
  behavior, service_restart.py:306-479 — wired, config repoint does not touch
  the gate code

## ε — deterministic deploy capstone

- own-unit detached self-restart path → DeterministicRunner
  (orchestrator/src/orchestrator/deterministic_runner.py); terminal-transition
  stall fixed by task 2004, merged 3590d655 (done_provenance kind='merged') —
  wired; live-daemon-predates-fix caveat + 1976/1982 remediation carried in
  ε's description
- before_done submit-time validation (script exists + executable) →
  producer: fused-memory guard (validated at submit_task; exercised by 1956
  ε2 deferred-filer precedent) — wired
- spec precedent → df 2009 (timeout_secs=900 > 6 units × (90 s stop + 30 s
  verify) = 720 s; target_unit=orchestrator-dark-factory.service;
  always_escalates=false) — verified via get_task(2009)
- numeric bound: 900 s > 720 s worst-case floor, stated above — floor check
  PASS
