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

Each timer tick also runs a fleet-wide staleness pass (staleness_pass()):
any running orchestrator-*.service unit whose realtime start time predates
the newest commit touching a small set of watched paths is restarted, as a
backstop for the event-driven restart coordinator
(plans/orchestrator-fleet-staleness-prd.md). Pass ``--report`` for a
read-only doctor mode that prints a per-unit staleness table and performs
no mutating systemctl calls.

Invoked by scripts/orchestrator-watchdog.service (launched via
scripts/orchestrator-watchdog.timer).
"""

import os
import subprocess
import sys
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

# The watchdog's own systemd unit. It matches the `orchestrator-*.service`
# glob that _enumerate_running_units filters on, but must never be treated as
# a fleet member to probe or restart. Today it is excluded incidentally
# because a Type=oneshot unit's SUB state while executing is 'start', not
# 'running' (see orchestrator-watchdog.service), so --state=running filters
# it out on its own. That is a fragile invariant to depend on implicitly --
# if the unit were ever changed to RemainAfterExit=yes or a non-oneshot Type,
# staleness_pass() could enumerate and blocking-restart itself mid-pass.
# _enumerate_running_units excludes it explicitly so the exclusion does not
# rely on systemd oneshot SUB-state semantics.
WATCHDOG_UNIT_NAME = "orchestrator-watchdog.service"

# Working directory shared by every orchestrator-*.service unit; the repo the
# staleness pass diffs against.
REPO_DIR = "/home/leo/src/dark-factory"

# Paths whose newest commit defines "fresh" for the staleness pass (mirrors
# orchestrator/config.yaml's orchestrator_restart_watch_prefixes — see
# plans/orchestrator-fleet-staleness-prd.md §Resolved 1/4).
WATCHED_PATHS = [
    "orchestrator/src/",
    "escalation/src/",
    "orchestrator/pyproject.toml",
    "orchestrator/uv.lock",
    "escalation/pyproject.toml",
]

# Fleet-wide head start for the polite event-driven restart coordinator
# (PRD: orchestrator-fleet-staleness) before the backstop staleness pass acts
# on the same commit. Env-overridable, mirroring restart-all-orchestrators.sh's
# RESTART_VERIFY_TIMEOUT env-with-default pattern (PRD Open Question 1). A
# missing or malformed value falls back to the default — a typo'd env var
# must not crash the oneshot watchdog.
try:
    STALENESS_GRACE_SECS = int(os.environ["STALENESS_GRACE_SECS"])
except (KeyError, ValueError):
    STALENESS_GRACE_SECS = 1800


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


def _enumerate_running_units() -> list[str]:
    """Return the names of all running ``orchestrator-*.service`` units.

    Runs the same enumeration as scripts/restart-all-orchestrators.sh
    (``systemctl --user list-units 'orchestrator-*.service' --state=running
    --no-legend --plain``) so the staleness pass and ``--report`` cover
    exactly the units restart-all touches — new projects are covered
    automatically with no watchdog code change.

    Returns [] on a non-zero exit, a subprocess error (missing binary /
    timeout), or empty output — a tooling failure yields no units to act on
    rather than a crash.

    Explicitly excludes WATCHDOG_UNIT_NAME (the watchdog's own unit) even
    though it should also match `orchestrator-*.service`, so callers never
    have to rely on oneshot SUB-state timing to keep the watchdog from
    probing or restarting itself.
    """
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "list-units",
                "orchestrator-*.service",
                "--state=running",
                "--no-legend",
                "--plain",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [
        fields[0]
        for line in result.stdout.splitlines()
        if (fields := line.split()) and fields[0] != WATCHDOG_UNIT_NAME
    ]


def _unit_start_epoch(unit: str) -> int | None:
    """Return *unit*'s realtime start epoch (Unix seconds since epoch), or None.

    Queries ``ExecMainStartTimestamp`` via ``systemctl --user show
    --timestamp=unix``, which yields a clean, timezone-independent ``@<epoch>``
    value directly comparable to git's ``%ct`` committer epoch. Deliberately
    does NOT parse systemd's human-readable timestamp string (locale/TZ
    fragile) and does NOT derive from the monotonic twin
    _unit_start_elapsed_secs (a different clock domain, unusable here).

    Returns None if the unit has no recorded start time (the ``@0`` sentinel),
    the value cannot be parsed, or any subprocess/OS error occurs — callers
    must treat None as "staleness cannot be determined for this unit".
    """
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--timestamp=unix",
                "-p",
                "ExecMainStartTimestamp",
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
            if val.startswith("@"):
                val = val[1:]
            try:
                epoch = int(val)
            except ValueError:
                return None
            if epoch == 0:
                return None  # unit has never started (or no PID recorded)
            return epoch
        return None
    except Exception:  # noqa: BLE001
        return None


def _newest_watched_commit_epoch() -> int | None:
    """Return the newest committer epoch touching WATCHED_PATHS on HEAD, or None.

    Runs ``git -C REPO_DIR log -1 --format=%ct HEAD -- <WATCHED_PATHS>``.
    Committer time approximates landing time (the merge queue rebases
    immediately before merging; direct-to-main commits are committed at
    landing) — file mtimes are rejected as a source since they drift under
    editor/checkout perturbation with no provenance.

    Returns None if no commit touches the watched paths (confirmed real
    behavior: git exits 0 with EMPTY stdout in this case — this must be
    treated as undeterminable, not epoch 0, or every unit would look
    infinitely stale), on a non-zero exit, on unparseable stdout, or on any
    subprocess/OS error. Callers must treat None as "staleness cannot be
    determined this tick".
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                REPO_DIR,
                "log",
                "-1",
                "--format=%ct",
                "HEAD",
                "--",
                *WATCHED_PATHS,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        stdout = result.stdout.strip()
        if not stdout:
            return None
        try:
            return int(stdout)
        except ValueError:
            return None
    except Exception:  # noqa: BLE001
        return None


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


def staleness_pass() -> None:
    """Restart any running orchestrator unit stale w.r.t. the newest watched commit.

    Fleet-wide backstop for the event-driven restart-all coordinator
    (plans/orchestrator-fleet-staleness-prd.md): a unit is stale when its
    realtime start epoch predates the newest commit touching WATCHED_PATHS.
    Stateless — staleness is recomputed from live systemd + git state on
    every call, so a successful restart (from this pass, the coordinator, a
    deploy capstone, or manual operator action) makes the unit read fresh on
    the very next call. No stored state, no flap loop (I6).

    Known limitation: the commit-grace gate below keys on the age of the
    *newest* watched commit only, not on how far behind any individual unit
    is. During a burst of continuous landings where a watched-path commit
    lands more often than every STALENESS_GRACE_SECS, the newest commit is
    perpetually inside the grace window, so this backstop is inhibited for
    the whole burst even if some running unit is far behind an older commit.
    This is an accepted trade-off (never race the event-driven coordinator)
    rather than a bug; use `--report` to inspect actual per-unit staleness
    while a burst is in progress.
    """
    commit_epoch = _newest_watched_commit_epoch()
    if commit_epoch is None:
        return  # undeterminable — fall safe, no restarts this tick
    if time.time() - commit_epoch < STALENESS_GRACE_SECS:
        # Give the polite event-driven restart coordinator its head start.
        # NOTE: gates on the newest commit's age alone — see the "Known
        # limitation" paragraph above for the rapid-landing suppression case.
        return

    for unit in _enumerate_running_units():
        try:
            if not is_unit_enabled(unit):
                # Disabled (or unknown) — respect operator intent, skip silently.
                continue
            # Two separate `systemctl show` calls per unit (elapsed here,
            # start_epoch below) — could be merged into one
            # `show -p ExecMainStartTimestampMonotonic -p ExecMainStartTimestamp`
            # call to halve the subprocess count. Left as-is: low priority
            # given the 60s timer cadence, and keeps this function reusing
            # _unit_start_elapsed_secs verbatim rather than parsing both
            # properties out of a combined call.
            elapsed = _unit_start_elapsed_secs(unit)
            if elapsed is not None and elapsed < STARTUP_GRACE_SECS:
                # None => grace does not apply, proceed (mirrors main()).
                continue
            start_epoch = _unit_start_epoch(unit)
            if start_epoch is None:
                continue  # undeterminable for this unit — skip, don't guess
            if start_epoch < commit_epoch:
                log(
                    f"WARNING: {unit} started at {start_epoch} before the newest "
                    f"watched commit ({commit_epoch}); restarting for staleness"
                )
                restart_unit(unit)
                log(f"{unit} staleness restart issued")
        except Exception as exc:  # noqa: BLE001
            log(f"staleness probe error for {unit}: {exc}")


def _format_epoch(epoch: int | None) -> str:
    """Render a Unix epoch as a UTC timestamp string, or 'unknown' for None."""
    if epoch is None:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(epoch))


def report() -> int:
    """Print a per-unit staleness table for the running fleet; return 1 iff any unit is stale.

    Read-only doctor mode (I7): performs zero mutating systemctl calls, only
    the same reads staleness_pass uses (list-units / show / git log) via
    _enumerate_running_units, _newest_watched_commit_epoch, and
    _unit_start_epoch. The row set is the dynamically-enumerated running
    fleet — decoupled from the static WATCHED liveness port list (PRD Open
    Question 3).

    A unit's verdict is 'unknown' (not 'stale') when either epoch is
    undeterminable; an 'unknown' verdict does not force a non-zero exit —
    only a confirmed-stale unit does.

    IMPORTANT: the verdict reflects raw start_epoch-vs-commit_epoch staleness
    only. It does NOT evaluate the is_unit_enabled, STARTUP_GRACE_SECS, or
    STALENESS_GRACE_SECS restraint gates that staleness_pass() applies before
    actually restarting a unit — a unit reported 'stale' here may be one that
    staleness_pass() will (correctly) leave alone this tick because it is
    disabled, within its startup grace window, or the newest watched commit
    is still within the fleet-wide commit-grace window. Treat 'stale' as
    "not running code from the newest watched commit", not as a prediction
    that a restart is imminent.
    """
    commit_epoch = _newest_watched_commit_epoch()
    units = _enumerate_running_units()

    commit_str = _format_epoch(commit_epoch)
    print(
        "NOTE: verdict reflects raw start-time-vs-commit staleness only; it "
        "does not account for the enabled / startup-grace / commit-grace "
        "restraint gates staleness_pass() applies before actually restarting "
        "a unit."
    )
    print(f"{'UNIT':<50} {'START':<24} {'NEWEST WATCHED COMMIT':<24} VERDICT")

    any_stale = False
    for unit in units:
        start_epoch = _unit_start_epoch(unit)
        start_str = _format_epoch(start_epoch)
        if start_epoch is None or commit_epoch is None:
            verdict = "unknown"
        elif start_epoch < commit_epoch:
            verdict = "stale"
            any_stale = True
        else:
            verdict = "fresh"
        print(f"{unit:<50} {start_str:<24} {commit_str:<24} {verdict}")

    return 1 if any_stale else 0


def _cli(argv: list[str] | None = None) -> int:
    """Dispatch the CLI: ``--report`` routes to the read-only doctor mode.

    With no ``argv`` argument, reads ``sys.argv[1:]``. If ``--report`` is
    present, runs ONLY the read-only report() and returns its exit code
    (0 = all fresh, 1 = at least one stale unit) — main() and
    staleness_pass() are not invoked, so this path never mutates systemd
    state (I7 at the CLI boundary). Otherwise runs the existing liveness
    main() followed by staleness_pass() (the timer path) and returns 0.
    Unknown flags are not treated as an error — they fall through to the
    timer path.
    """
    argv = sys.argv[1:] if argv is None else argv
    if "--report" in argv:
        return report()
    main()
    staleness_pass()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
