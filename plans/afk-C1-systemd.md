# AFK C1: Supervise both orchestrators under systemd with kill+restart watchdog

## Problem

Both orchestrators run as foreground `uv run` processes. No supervisor, no auto-restart, no journal logging. A power blip / kernel OOM / unhandled exception kills them silently; AFK = no work for the rest of the window. Crash is rare, but the asymmetry between cost-of-coverage (~1 hour) and cost-of-loss (days of stalled progress) justifies the defensive bet.

## Solution

Two non-templated user services + a watchdog timer/service pair.

### Files at `~/.config/systemd/user/`

`orchestrator-dark-factory.service`:
```ini
[Unit]
Description=Dark Factory Orchestrator (supervised)
After=network.target fused-memory.service reify-jobserver.service pytest-jobserver.service
Requires=fused-memory.service

[Service]
Type=simple
WorkingDirectory=/home/leo/src/dark-factory
ExecStart=/home/leo/.local/bin/uv run --project orchestrator orchestrator run --config /home/leo/src/dark-factory/orchestrator/config.yaml
Restart=on-failure
RestartSec=10
RestartMaxDelaySec=60
StartLimitIntervalSec=600
StartLimitBurst=10
TimeoutStopSec=90
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

`orchestrator-reify.service`: same shape with `WorkingDirectory=/home/leo/src/reify` and `--config /home/leo/src/reify/orchestrator.yaml`.

`orchestrator-watchdog.timer`:
```ini
[Unit]
Description=Orchestrator escalation-MCP health probe (every 60s)

[Timer]
OnBootSec=30
OnUnitActiveSec=60

[Install]
WantedBy=timers.target
```

`orchestrator-watchdog.service`:
```ini
[Unit]
Description=Probe escalation MCP ports; kill+restart unresponsive orch

[Service]
Type=oneshot
ExecStart=/home/leo/bin/orchestrator-watchdog.sh
```

### Watchdog script at `~/bin/orchestrator-watchdog.sh`

For each orchestrator (df=8102, reify=8100):
- `curl -sf --max-time 2 http://127.0.0.1:<port>/mcp/`
- On failure: `systemctl --user kill -KILL orchestrator-<name>.service && systemctl --user start orchestrator-<name>.service`

No alert-only mode (zero-oversight policy).

## Migration plan

1. Write the four files
2. `systemctl --user daemon-reload`
3. SIGTERM the existing PIDs (`499514`, `1245994`); wait for clean exit (orch's signal handlers run the asyncio `finally` blocks; lock files released)
4. `systemctl --user enable --now orchestrator-dark-factory.service orchestrator-reify.service orchestrator-watchdog.timer`
5. Verify: `journalctl --user -u orchestrator-dark-factory -f` and `systemctl --user list-timers`

## Acceptance criteria

- Both orchestrators start under systemd and survive `systemctl --user kill -KILL` (auto-restart within ≤15s)
- Logs reach journalctl
- Watchdog timer ticks every 60s; manually wedging an escalation port (e.g., `iptables -A INPUT -p tcp --dport 8102 -j DROP` for the test, then revert) triggers kill+restart within one tick
- Lock files cleaned on shutdown (fcntl releases on process death)
- StartLimitBurst stops auto-restart loop on broken config (10 restarts in 10min → manual `systemctl start` required)

## Risks

- **Old uv process not fully exited when systemd starts** — fcntl is atomic; second instance fails immediately and logs to journal.
- **Watchdog over-aggressive on transient localhost stall** — 2s curl timeout is generous; StartLimitBurst caps the damage.
- **Display-manager dependency strands** — none; no `WantedBy=graphical-session.target`; uses `default.target`.

## Out of scope (post-AFK candidate)

Move escalation MCP server out of the orch process and into `fused-memory.service` so session connections survive orch restarts. Larger refactor; addresses the more-common "session MCP glitch" failure mode separately from this AFK supervision.
