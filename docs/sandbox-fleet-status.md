# OS Sandbox — Fleet Status

The single statement of where OS-level agent sandboxing stands on this fleet.
`ARCHITECTURE.md`, `README.md` and the `SandboxConfig` docstring cite this file
rather than restating it. Decisions (D1–D13) are those of
`plans/os-sandbox-worktree-containment-prd.md`.

## Status

**Enabled across the fleet** for the sandboxed roles: those with
`sandboxed=True` in `orchestrator/src/orchestrator/agents/roles.py`, which today
are implementer, debugger and simple_task (D3). Dispatch goes through
`orchestrator/src/orchestrator/workflow.py::_guard_sandbox`. The production
EACCES denials are recorded in `docs/sandbox-containment-probe-report.md` (γ4).

## Backend

**Landlock** is the supported fleet backend, and every factory target pins
`backend: landlock` explicitly. `auto` would take the first available of
landlock, bwrap and none
(`orchestrator/src/orchestrator/agents/sandbox_dispatch.py::resolve_active_backend`),
so a host that lost landlock would drop to bwrap, under which Bun v1.3.13
segfaulted on kernel 6.17. bwrap is **legacy passthrough** (D10): it inherits
the write set through the backend-agnostic `writable_modules`/`writable_extras`
params, but gets no parity work and is not what the fleet runs.

## Architecture

- Landlock enforcement is **validated on x86_64 only**; the fleet host reports
  `uname -m` = `x86_64`. Both implementations hardcode syscall numbers 444–446
  and declare an x86_64 assumption, with no runtime architecture gate
  (`orchestrator/src/orchestrator/agents/landlock.py::SYS_landlock_create_ruleset`,
  `orchestrator/src/orchestrator/agents/landlock_exec.py`).
- 444–446 are the kernel's unified numbers, the same in `asm-generic/unistd.h`
  (aarch64, riscv64, …) and on i386, so the constants are probably not what a
  port must change; validation and an architecture gate are. x32, alpha and
  mips number them differently.

## Failure posture

Fail-closed on any host. If the landlock ABI probe fails (kernel older than
5.13, landlock absent from the active LSM list, …),
`orchestrator/src/orchestrator/agents/sandbox_dispatch.py::resolve_backend_or_refuse`
raises `SandboxUnavailable` and the sandboxed-role dispatch is **refused**
rather than run unconfined, with one escalation per backend-state change (D4).
`backend: none` is the explicit escape hatch.

## What it does not cover (PRD §"Out of scope (explicit)")

- Writes only: the whole filesystem stays readable, and network is unscoped.
- The write scope is the whole worktree (D1). Tightening it to the plan's files
  was declined, not deferred: task 2916 (δ1) records why.
- merger, steward and architect Bash are not confined.

## Census

Measured 2026-09-23 09:04Z on the fleet host, from `DASHBOARD_KNOWN_PROJECT_ROOTS`
in the installed `dark-factory-dashboard.service` (9 roots, primary included).

| Project | Config lives in | Flipped by | Status |
|---|---|---|---|
| dark-factory | this repo: `dark-factory-orchestrator.yaml` | γ2 (task 2911) | landlock |
| dark-factory dashboard | this repo: `dashboard/orchestrator.yaml` | γ7 (task 2915) | landlock |
| reify | its own repo and registry | γ6a/γ6b (`reify:5332`, `reify:5333`) | landlock |
| autopilot-video | its own repo and registry | γ7a (`autopilot_video:649`) | landlock |
| know-live | its own repo and registry | γ7b (`know_live:592`) | landlock |
| pump-web-ui | its own repo and registry | γ7c (`pump_web_ui:16`) | landlock |
| solar-challenge | its own repo and registry | γ7d (`solar_challenge:95`) | landlock |
| solar-challenge-platform | its own repo and registry | γ7e (`solar_challenge_platform:164`) | landlock |

- **Dashboard row.** No unit runs `dashboard/orchestrator.yaml` as a project
  config, so its flip changed no running daemon; one that did would start confined.
- **Not factory targets.** autotrade and mission-control are recon-only
  registrations with no orchestrator config.
- **solar-challenge path trap.** Its config is
  `/home/leo/src/solar-challenge/dark-factory-orchestrator.yaml`; "my-" survives
  only in the legacy unit name `orchestrator-my-solar-challenge.service`, and
  `/home/leo/src/my-solar-challenge` exists but holds no config.

```bash
# Re-run the census:
for p in $(grep -o 'DASHBOARD_KNOWN_PROJECT_ROOTS=[^"]*' ~/.config/systemd/user/dark-factory-dashboard.service | cut -d= -f2 | tr , ' '); do echo "== $p"; f="$p/dark-factory-orchestrator.yaml"; if [ -f "$f" ]; then grep -n -A2 '^sandbox:' "$f" || echo '  (config present, NO sandbox block)'; else echo '  (no dark-factory-orchestrator.yaml)'; fi; done
```

## Standing exceptions

- The shipped default `orchestrator/src/orchestrator/defaults.yaml` stays
  `enabled: false` (D7): enablement is explicit per-project, restart-tier config.
- The eval runner forces `SandboxConfig(enabled=False)` (D12): eval worktrees
  live under `/tmp`, where the restrictions are nullified anyway.

## Enforcement

- `orchestrator/tests/test_sandbox_fleet_census.py` holds the in-repo half: both
  factory targets pinned, completeness over the repo's orchestrator configs, D7.
- `orchestrator/tests/test_eval_profile.py` holds D12.
- Each sibling's own registry holds its row; nothing in this repo does.
