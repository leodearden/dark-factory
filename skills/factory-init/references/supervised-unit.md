# Supervised Orchestrator Unit (going unattended)

Stage 8 — **optional**, and a deliberate operational commitment: it puts the project into the always-on autonomous workload (it will consume API budget and change code without a human in the loop). Only do this when the user explicitly wants unattended operation, and **only after the project has `pending` tasks** (see the hazard below).

A supervised unit is three independent layers. Layer 1 is what makes the project run unattended; Layer 2 makes that durable; Layer 3 is optional extra liveness.

## The cold-start hazard (why timing matters)

An orchestrator started against a task tree with **no `pending` task** exits with "No pending tasks found." If the unit is enabled and the watchdog is probing it, that becomes a 60s crash-loop. So:

- Create + enable + start the unit **only after** Stage 7 has queued tasks (a `/prd` batch landed, or `/review` filed work). Confirm `orchestrator status --config <CONFIG>` shows ≥1 `pending`.
- The watchdog **skips disabled units**, so if you must install the unit before tasks exist, leave it `disabled` and enable it later.

## Layer 1 — the unit (makes it always-on)

Concrete file (not templated — the `scripts/orchestrator-*.service` files are `cp`'d verbatim by `setup-host.sh`, not `sed`-rendered). Model on `scripts/orchestrator-reify.service` / `orchestrator-autopilot-video.service`. Placeholders: `<NAME>` = the project's hyphenated name (e.g. `my-solar-challenge`), `<DESC>` a human label, `<DF>` = dark-factory root, `<CONFIG>` = `<target>/orchestrator.yaml`.

```ini
[Unit]
Description=<DESC> Orchestrator (supervised)
After=network.target fused-memory.service
# Wants=, not Requires=: a Requires= turns a single fused-memory boot-race failure
# into a PERMANENT cancel of our start job. Wants= orders us after it but lets us
# proceed; ExecStartPre waits for port 8002 and Restart=on-failure self-heals.
Wants=fused-memory.service
# Restart-rate guard — MUST be under [Unit] (silently ignored under [Service]).
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple
# CWD must be dark-factory so `uv run --project orchestrator` resolves the package.
WorkingDirectory=<DF>
ExecStartPre=/home/leo/bin/wait-for-port.py --timeout 280 127.0.0.1:8002
ExecStart=/home/leo/.local/bin/uv run --frozen --project orchestrator orchestrator run --config <CONFIG>
# Replicate PATH so verify subprocesses find their toolchain (a user service gets a minimal PATH).
Environment=PATH=/home/leo/.cargo/bin:/home/leo/.local/npm-global/bin:/home/leo/.local/bin:/home/leo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin
Environment=LANG=en_US.UTF-8
Restart=on-failure
RestartSec=10
RestartMaxDelaySec=60
TimeoutStopSec=90      # async shutdown reaps agents + releases the lock
TimeoutStartSec=300    # must exceed the 280s ExecStartPre budget (one start attempt)
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

Install:
```bash
cp <unit> ~/.config/systemd/user/orchestrator-<NAME>.service
systemctl --user daemon-reload
systemctl --user enable --now orchestrator-<NAME>.service   # --now starts it; only once tasks are pending
journalctl --user -u orchestrator-<NAME>.service --since '1 min ago'   # confirm it engaged, not "No pending tasks"
```

## Layer 2 — persistence (so a re-provision keeps it)

The live unit alone is lost if `setup-host.sh` re-provisions the host. Persist it:

1. Copy the unit into the repo: `<DF>/scripts/orchestrator-<NAME>.service` (concrete, verbatim).
2. In `<DF>/scripts/setup-host.sh`, add a `cp "$REPO_ROOT/scripts/orchestrator-<NAME>.service" "$UNIT_DIR/"` next to the other orchestrator `cp` lines, and a `systemctl --user enable orchestrator-<NAME>.service` next to the other `enable` lines.
3. Commit both to dark-factory.

## Layer 3 — watchdog port-probe (optional)

The `orchestrator-watchdog` actively probes each orchestrator's escalation port and restarts a unit that's wedged-but-not-crashed. Its `WATCHED` list is **hardcoded** in `<DF>/scripts/orchestrator-watchdog.py`, and a drift test (`tests/scripts/test_orchestrator_watchdog.py`) asserts those ports match each orchestrator's `escalation.port`.

This layer is **optional** — `autopilot-video` runs enabled *without* watchdog coverage, relying on `Restart=on-failure` alone. Add it only if you want active liveness probing:

1. Add `(<escalation_port>, "orchestrator-<NAME>.service")` to `WATCHED` in `orchestrator-watchdog.py`.
2. Update the drift test so it stays green (it cross-checks `WATCHED` ports against the configs).
3. Run `cd <DF> && uv run --project orchestrator pytest tests/scripts/test_orchestrator_watchdog.py` before committing.

Skipping Layer 3 is fine and common; `Restart=on-failure` covers crashes, just not silent wedges.
