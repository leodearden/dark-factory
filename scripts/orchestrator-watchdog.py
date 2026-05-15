#!/usr/bin/env python3
"""Orchestrator escalation-MCP watchdog.

Probes the escalation-MCP TCP ports for the dark-factory and reify
orchestrators via ``ss``, then performs a two-phase
``systemctl --user stop`` → ``systemctl --user start`` on any that are not
listening.  Runs as a oneshot systemd service on a 60-second timer.

Invoked by scripts/orchestrator-watchdog.service (launched via
scripts/orchestrator-watchdog.timer).
"""

import subprocess
import sys

# (port, systemd unit name) pairs to watch.  Port order matches the
# config files: dark-factory=8102 (orchestrator/config.yaml:38),
# reify=8100 (/home/leo/src/reify/orchestrator.yaml:87).
WATCHED = [
    (8102, "dark-factory-orchestrator.service"),
    (8100, "reify-orchestrator.service"),
]


def probe_port(port: int) -> bool:
    """Return True iff a process is listening on *port* (TCP, local).

    Runs ``ss -ltn "sport = :<port>"`` and checks whether any output line
    contains a local-address field whose port component is exactly *port*
    (robust against substring matches such as :81020 or :48102).
    """
    result = subprocess.run(
        ["ss", "-ltn", f"sport = :{port}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    for line in result.stdout.splitlines():
        # ss -ltn columns: Netid State Recv-Q Send-Q Local-Address:Port Peer-Address:Port
        # The local address is the 5th whitespace-delimited field (index 4).
        parts = line.split()
        if len(parts) < 5:
            continue
        local = parts[4]  # e.g. "127.0.0.1:8102" or "*:8102"
        # Split on the LAST ':' so IPv6 addresses like "[::1]:8102" work too.
        colon_idx = local.rfind(":")
        if colon_idx == -1:
            continue
        try:
            local_port = int(local[colon_idx + 1:])
        except ValueError:
            continue
        if local_port == port:
            return True
    return False


def restart_unit(unit: str) -> None:
    """Two-phase restart: stop then start *unit* via ``systemctl --user``.

    systemd's TimeoutStopSec=30 escalates SIGTERM→SIGKILL so in-flight work
    gets a grace period; we never invoke ``systemctl kill`` directly.
    check=False so a stop failure (e.g. unit already dead) does not abort the
    subsequent start.
    """
    subprocess.run(["systemctl", "--user", "stop", unit], check=False)
    subprocess.run(["systemctl", "--user", "start", unit], check=False)


def log(msg: str) -> None:
    """Write *msg* to the systemd journal tagged as ``orchestrator-watchdog``."""
    subprocess.run(
        ["systemd-cat", "-t", "orchestrator-watchdog"],
        input=msg,
        text=True,
        check=False,
    )


def main() -> None:
    """Probe each watched port; restart the unit if the port is not listening."""
    for port, unit in WATCHED:
        try:
            if not probe_port(port):
                log(f"{unit} escalation port {port} not listening; restarting")
                restart_unit(unit)
                log(f"{unit} restart issued")
        except Exception as exc:  # noqa: BLE001
            log(f"watchdog error for {unit} (port {port}): {exc}")


if __name__ == "__main__":
    main()
