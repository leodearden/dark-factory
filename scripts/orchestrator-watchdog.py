#!/usr/bin/env python3
"""Orchestrator escalation-MCP watchdog.

For each enabled orchestrator (dark-factory + reify): probes its escalation-MCP
TCP port via ``ss``, and performs a three-phase ``systemctl --user stop`` →
``reset-failed`` → ``start`` if the port is not listening. The probe-fails →
restart path covers BOTH failure modes:

  * **Wedged** (alive-but-hung): the unit is active but the port stopped
    answering — systemd Restart= won't fire. The watchdog forces a restart.
  * **Dead-but-enabled** (e.g. a boot-race dependency-cancel that systemd
    never retries, or a unit that gave up after exhausting StartLimitBurst):
    the unit is inactive, the port isn't listening, ``stop`` is a no-op, and
    ``reset-failed`` + ``start`` revives it. This is what self-heals the
    2026-05-27 powercut failure mode.

Disabled units are skipped — disabling is explicit operator intent. Runs as a
oneshot systemd service on a 60-second timer.

Invoked by scripts/orchestrator-watchdog.service (launched via
scripts/orchestrator-watchdog.timer).
"""

import subprocess
import time

# (port, systemd unit name) pairs to watch.  Port values match each
# orchestrator's configured escalation.port (guarded by the drift test in
# tests/scripts/test_orchestrator_watchdog.py).
WATCHED = [
    (8102, "orchestrator-dark-factory.service"),
    (8100, "orchestrator-reify.service"),
    (8106, "orchestrator-my-solar-challenge.service"),
]

# Skip the port probe for a unit that started within this many seconds.
# Prevents the watchdog from stop→starting an orchestrator that is still
# binding its escalation-MCP port after a fresh (re)start — which would
# produce an indefinite restart loop and quickly exhaust StartLimitBurst.
# Value = 2 × probe interval (60s) to cover worst-case startup latency.
STARTUP_GRACE_SECS = 120


def log(msg: str) -> None:
    """Write *msg* to the systemd journal tagged as ``orchestrator-watchdog``."""
    subprocess.run(
        ["systemd-cat", "-t", "orchestrator-watchdog"],
        input=msg,
        text=True,
        check=False,
    )


def probe_port(port: int) -> bool:
    """Return True iff a process is listening on *port* (TCP, local).

    Runs ``ss -ltn "sport = :<port>"`` and checks whether any output line
    contains a field whose port component is exactly *port* (robust against
    substring matches such as :81020 or :48102).

    Column-independent scan: for each LISTEN line every whitespace-delimited
    field is checked.  Any field containing ':' is split on the LAST ':' and
    the trailing component is compared as an int to *port*.  This handles both
    the legacy Netid-prefixed layout (local addr at index 4) and the no-Netid
    layout from iproute2-6.1.0 / systemd 255 (local addr at index 3); the
    peer ``0.0.0.0:*`` field is harmlessly skipped (``int('*')`` raises
    ValueError).

    If ``ss`` exits non-zero (permission issue or filter-syntax difference
    across iproute2 versions), a diagnostic is logged and True is returned.
    If ``ss`` is not installed (FileNotFoundError) or the probe exceeds its
    5-second timeout (subprocess.TimeoutExpired), the exception is caught,
    a diagnostic is logged, and True is returned.  In all error cases the
    safe default is True — a tooling failure must not trigger a spurious
    restart on a healthy unit.
    """
    try:
        result = subprocess.run(
            ["ss", "-ltn", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log(
            f"ss probe for port {port} could not complete ({type(exc).__name__}); "
            "assuming port is up to avoid a false restart"
        )
        return True
    if result.returncode != 0:
        log(
            f"ss probe for port {port} exited with code {result.returncode}; "
            "assuming port is up to avoid a false restart"
        )
        return True
    for line in result.stdout.splitlines():
        if "LISTEN" not in line:
            continue
        for field in line.split():
            colon_idx = field.rfind(":")
            if colon_idx == -1:
                continue
            try:
                if int(field[colon_idx + 1 :]) == port:
                    return True
            except ValueError:
                continue
    return False


def restart_unit(unit: str) -> None:
    """Three-phase restart: stop → reset-failed → start *unit* via ``systemctl --user``.

    - ``stop`` allows systemd's TimeoutStopSec=30 to escalate SIGTERM→SIGKILL
      gracefully so in-flight work gets a 30-second grace period; we never
      invoke ``systemctl kill`` directly.
    - ``reset-failed`` clears the StartLimit state (StartLimitBurst / start-limit-hit)
      so the subsequent start is not a silent no-op on a rate-limited unit.
    - ``start`` re-launches the unit.

    An explicit timeout of 45s (comfortably above TimeoutStopSec=30) prevents
    the oneshot watchdog from hanging indefinitely if systemctl blocks.
    TimeoutExpired is caught and logged; the remaining phases still execute.
    """
    try:
        subprocess.run(
            ["systemctl", "--user", "stop", unit], check=False, timeout=45
        )
    except subprocess.TimeoutExpired:
        log(f"systemctl stop {unit} timed out after 45s")

    # Always run reset-failed regardless of stop outcome so a rate-limited unit
    # (StartLimitBurst exhausted) can be recovered even after the timeout path.
    try:
        subprocess.run(
            ["systemctl", "--user", "reset-failed", unit], check=False, timeout=10
        )
    except subprocess.TimeoutExpired:
        log(f"systemctl reset-failed {unit} timed out after 10s")

    try:
        subprocess.run(
            ["systemctl", "--user", "start", unit], check=False, timeout=45
        )
    except subprocess.TimeoutExpired:
        log(f"systemctl start {unit} timed out after 45s")


def _unit_start_elapsed_secs(unit: str) -> float | None:
    """Seconds since *unit*'s main process started (monotonic clock), or None.

    Queries ``ExecMainStartTimestampMonotonic`` (microseconds since boot) via
    ``systemctl --user show`` and compares it against
    ``time.clock_gettime(time.CLOCK_MONOTONIC)`` — the same CLOCK_MONOTONIC
    source systemd uses — so a host suspend no longer skews the elapsed
    estimate (unlike /proc/uptime which advances during suspend).

    Returns None if the unit has no recorded start time, the value cannot be
    parsed, or any subprocess/OS error occurs — callers must treat None as
    "grace window does not apply, proceed with probe".
    """
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=ExecMainStartTimestampMonotonic",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            val = line.split("=", 1)[1].strip()
            try:
                start_mono_us = int(val)
            except ValueError:
                return None
            if start_mono_us == 0:
                return None  # unit has never started (or no PID recorded)
            now_secs = time.clock_gettime(time.CLOCK_MONOTONIC)
            return max(0.0, now_secs - start_mono_us / 1_000_000)
        return None
    except Exception:  # noqa: BLE001
        return None


def is_unit_enabled(unit: str) -> bool:
    """Return True iff ``systemctl --user is-enabled <unit>`` exits 0.

    Disabling a unit is explicit operator intent (a staged-but-not-yet-active
    deployment, or a temporarily-disabled service) — the watchdog must respect
    it and not auto-revive. ``is-enabled`` exits 0 for ``enabled`` /
    ``enabled-runtime`` / ``static`` / ``alias``; non-zero for ``disabled`` /
    ``masked`` / unknown. Subprocess errors fall safe to False (skip).
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-enabled", "--quiet", unit],
            check=False,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log(
            f"is-enabled probe for {unit} could not complete "
            f"({type(exc).__name__}); skipping unit"
        )
        return False


def main() -> None:
    """Probe each watched port; restart the unit if the port is not listening."""
    for port, unit in WATCHED:
        try:
            if not is_unit_enabled(unit):
                # Disabled (or unknown) — respect operator intent, skip silently.
                continue
            elapsed = _unit_start_elapsed_secs(unit)
            if elapsed is not None and elapsed < STARTUP_GRACE_SECS:
                log(
                    f"{unit} started {elapsed:.0f}s ago; "
                    f"skipping probe (grace window {STARTUP_GRACE_SECS}s)"
                )
                continue
            if not probe_port(port):
                # Covers both wedged-active and dead-enabled (boot-race
                # cancelled, or StartLimit-exhausted): restart_unit's
                # stop+reset-failed+start sequence revives either case.
                log(f"{unit} escalation port {port} not listening; restarting")
                restart_unit(unit)
                log(f"{unit} restart issued")
        except Exception as exc:  # noqa: BLE001
            log(f"watchdog error for {unit} (port {port}): {exc}")


if __name__ == "__main__":
    main()
