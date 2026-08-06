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

import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# (port, systemd unit name) pairs to watch.  Port values match each
# orchestrator's configured escalation.port (guarded by the drift test in
# tests/scripts/test_orchestrator_watchdog.py).
WATCHED = [
    (8102, "orchestrator-dark-factory.service"),
    (8100, "orchestrator-reify.service"),
    (8106, "orchestrator-my-solar-challenge.service"),
    (8105, "orchestrator-know-live.service"),
    (8101, "orchestrator-autopilot-video.service"),
    (8107, "orchestrator-solar-challenge-platform.service"),
    (8108, "orchestrator-pump-web-ui.service"),
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

# fused-memory liveness (task 2713, ξ of the fm-restart survey). fused-memory
# is the single shared MCP server all 7 orchestrators depend on — it is
# deliberately NOT added to WATCHED/main()'s loop (both are pinned by
# exact-equality drift tests). It needs a richer probe than a bare port
# check: the in-process systemd watchdog thread pings WATCHDOG=1 from a
# dedicated OS thread unconditionally (task 1731), so a wedged asyncio loop
# still looks "healthy" to systemd forever — only an HTTP /health fetch can
# catch that. See fused_memory_liveness_pass() / probe_health() below.
FUSED_MEMORY_UNIT = "fused-memory.service"
# Port value matches fused-memory/config/config.yaml's server.port (guarded
# by the drift test in tests/scripts/test_orchestrator_watchdog.py).
FUSED_MEMORY_PORT = 8002
FUSED_MEMORY_HEALTH_URL = f"http://127.0.0.1:{FUSED_MEMORY_PORT}/health"
# /health (fused_memory/server/tools.py:701) runs two sequential backing-store
# probes (FalkorDB, Qdrant), each wrapped in asyncio.timeout(5), so a
# degraded-but-alive server can take ~10s to return its 503. 15s clears that
# ~10s worst case with margin, so only a genuinely wedged loop (no response
# at all) times out here.
FUSED_MEMORY_HEALTH_TIMEOUT_SECS = 15

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

# Minimum wall-clock seconds between successive FLEET-WIDE redeploys, shared
# with the event-driven restart coordinator
# (orchestrator.service_restart.StaleServiceRestartCoordinator's
# min_interval_secs, sourced from
# OrchestratorConfig.orchestrator_restart_min_interval_secs, default 28800.0
# = 8h). The watchdog is a stdlib-only systemd oneshot that cannot import the
# orchestrator package, so this is a hardcoded env-mirror of the config
# default (drift-tested in tests/scripts/test_orchestrator_watchdog.py
# against a live OrchestratorConfig, task 2396 Open-Q1) rather than a live
# read of it. 0 disables the cap entirely. Mirrors STALENESS_GRACE_SECS's
# env-with-default try/except pattern immediately above — a typo'd env var
# must not crash the oneshot watchdog.
try:
    ORCH_RESTART_MIN_INTERVAL_SECS = int(os.environ["ORCH_RESTART_MIN_INTERVAL_SECS"])
except (KeyError, ValueError):
    ORCH_RESTART_MIN_INTERVAL_SECS = 28800

# Path to the shared fleet-deploy clock file: the SAME file
# restart-all-orchestrators.sh stamps (atomically, only on its verified-fresh
# exit-0 path) and the orchestrator's own StaleServiceRestartCoordinator
# reads/seeds from (state_path, task 2371/2396). Mirrors
# orchestrator.service_restart.FLEET_DEPLOY_CLOCK_RELPATH — neither this
# stdlib script nor that constant can import each other, so the literal path
# is duplicated here and guarded by a drift/consistency test. Env-overridable
# so tests can point every tier at a tmp file without touching real data/.
FLEET_DEPLOY_CLOCK_PATH = os.environ.get(
    "ORCH_FLEET_DEPLOY_CLOCK",
    os.path.join(REPO_DIR, "data", "orchestrator", "last_redeploy_orchestrator.json"),
)

# staleness_pass() is a stateless oneshot: every ~60s timer tick
# (orchestrator-watchdog.timer's OnUnitActiveSec=60) is a FRESH process (see
# module docstring), so there is no cross-tick memory to log the fleet-deploy
# min-interval skip line only once per window. Logging on EVERY tick would
# write ~480 near-identical lines to the journal over one full
# ORCH_RESTART_MIN_INTERVAL_SECS (8h default) window, burying genuinely
# actionable watchdog output. Re-emit the skip line at most once per this many
# wall-clock seconds instead, bucketed purely off time.time() (no persisted
# state needed) — deliberately much coarser than the ~60s tick cadence so most
# ticks land inside an already-logged bucket and stay silent.
SKIP_LOG_INTERVAL_SECS = 1800

# Freshness window for report()'s MERGE-IDLE column (_classify_unit_heartbeat
# below), mirroring restart-all-orchestrators.sh's ORCH_DRAIN_FRESH_WINDOW_SECS
# default so the report's classification matches the drain gate exactly. A
# missing/malformed value falls back to the default — mirrors
# STALENESS_GRACE_SECS's try/except pattern above: a typo'd env var must not
# crash the oneshot watchdog (or report()).
try:
    DRAIN_FRESH_WINDOW_SECS = int(os.environ["ORCH_DRAIN_FRESH_WINDOW_SECS"])
except (KeyError, ValueError):
    DRAIN_FRESH_WINDOW_SECS = 120

# --- fused-memory staleness backstop (task 2714, ο of the fm-restart survey) ---
# fused-memory needs the SAME staleness backstop the orchestrator fleet has
# (staleness_pass below): a merged fm change otherwise sits undeployed until a
# human/PRD files a deploy task (survey finding A2). fused_memory_staleness_pass()
# is a single-unit mirror of staleness_pass() over FUSED_MEMORY_UNIT; the
# constants below are fm siblings of WATCHED_PATHS / FLEET_DEPLOY_CLOCK_PATH /
# ORCH_RESTART_MIN_INTERVAL_SECS, kept deliberately independent so fm's redeploy
# cadence never couples to the orchestrator fleet's.

# Paths whose newest commit defines "fresh" for the fm staleness pass. Includes
# shared/src/ because fused-memory imports shared.* (e.g. shared.task_metadata),
# so a shared change can alter fm's behavior and must count toward fm staleness.
FM_WATCHED_PATHS = ["fused-memory/src/", "shared/src/"]

# fused-memory's OWN deploy-clock file — a DIFFERENT file than the orchestrator
# fleet clock (FLEET_DEPLOY_CLOCK_PATH), so an orchestrator fleet redeploy never
# resets fm's min-interval window and vice-versa. Unlike
# restart-all-orchestrators.sh (which stamps the fleet clock itself),
# restart-fused-memory.sh does NOT write this file — the watchdog does, via the
# `--stamp-fm-deploy-clock` subcommand chained after a verified restart (see
# _delegate_fm_restart / _stamp_fm_deploy_clock below). Env-overridable so tests
# can point it at a tmp file without touching real data/.
FM_DEPLOY_CLOCK_PATH = os.environ.get(
    "FM_DEPLOY_CLOCK",
    os.path.join(REPO_DIR, "data", "fused-memory", "last_redeploy_fused_memory.json"),
)

# Minimum wall-clock seconds between successive fm redeploys — fm's own
# independent cap, mirroring ORCH_RESTART_MIN_INTERVAL_SECS's 8h default and its
# env-with-try/except-fallback pattern (a typo'd env var must not crash the
# oneshot watchdog). 0 disables the cap entirely.
try:
    FM_RESTART_MIN_INTERVAL_SECS = int(os.environ["FM_RESTART_MIN_INTERVAL_SECS"])
except (KeyError, ValueError):
    FM_RESTART_MIN_INTERVAL_SECS = 28800

# Fixed transient unit name for the detached fm staleness redeploy — the natural
# overlap guard (a second tick fails to re-register the same unit name while a
# redeploy is still running), sibling of orch-fleet-staleness-redeploy.service.
FM_STALENESS_REDEPLOY_UNIT = "fm-staleness-redeploy.service"


# --- fm liveness streak + restart cap (task 3764) ---
# fused_memory_liveness_pass() below used to restart fused-memory on a SINGLE
# non-healthy verdict, uncapped. The measured evidence says that is wrong: a
# controlled experiment (2026-08-06 07:14-09:14Z, kill path disconnected)
# recorded six /health stalls >=8s, four >15s, three exceeding a 25s probe cap
# — and every one SELF-RECOVERED, with 0 restarts needed over 20 cycles. The
# knobs below add the missing TEMPORAL dimension to the kill decision in two
# independent layers: a streak (make a wrong verdict RARE) and a min-interval
# cap on the restart itself (make a wrong verdict HARMLESS).

# How many CONSECUTIVE non-healthy verdicts are required before a restart.
# Derivation, not a guess: orchestrator-watchdog.timer is OnUnitActiveSec=60,
# so N=3 demands ~180s of CONTINUOUS non-response — ~7x the longest observed
# (25s+) self-recovering stall, so no observed transient can reach it, while a
# genuinely wedged asyncio loop (which stays wedged by definition) reaches it
# on the third tick. Env-with-try/except-fallback exactly like
# FM_RESTART_MIN_INTERVAL_SECS above: a typo'd env var must not crash the
# oneshot watchdog.
try:
    FM_LIVENESS_STREAK_THRESHOLD = int(os.environ["FM_LIVENESS_STREAK_THRESHOLD"])
except (KeyError, ValueError):
    FM_LIVENESS_STREAK_THRESHOLD = 3

# Maximum age of a persisted streak entry before it is treated as expired and
# the count restarts at 1. 300s = 5x the 60s tick cadence: a larger gap means
# >=4 ticks were missed (watchdog stopped, timer disabled, host suspended,
# unit disabled and re-enabled), so the "CONSECUTIVE" claim is no longer true
# and continuity must be re-established from scratch. This is what stops a
# streak from hours ago masquerading as a fresh one — and it fails in the safe
# direction: an unusually slow tick sequence expires the streak and SUPPRESSES
# a restart rather than manufacturing one.
try:
    FM_LIVENESS_STREAK_MAX_AGE_SECS = int(os.environ["FM_LIVENESS_STREAK_MAX_AGE_SECS"])
except (KeyError, ValueError):
    FM_LIVENESS_STREAK_MAX_AGE_SECS = 300

# Minimum wall-clock seconds between successive watchdog-initiated fm LIVENESS
# restarts — the second layer, bounding the blast radius of a verdict that is
# wrong anyway. 3600s strictly exceeds the 3180s (53 min) worst observed
# pathological instance lifetime, so even a wrong verdict cannot reproduce that
# pathology, and it bounds watchdog-initiated fm restarts to <=24/day versus
# today's unbounded. Deliberately 8x SHORTER than the staleness pass's
# FM_RESTART_MIN_INTERVAL_SECS (28800s): a brokenness revive must react faster
# than a scheduled deploy (I5). 0 disables the cap entirely.
try:
    FM_LIVENESS_RESTART_MIN_INTERVAL_SECS = int(
        os.environ["FM_LIVENESS_RESTART_MIN_INTERVAL_SECS"]
    )
except (KeyError, ValueError):
    FM_LIVENESS_RESTART_MIN_INTERVAL_SECS = 3600

# Where the consecutive-failure streak is PERSISTED. Persistence is mandatory
# rather than stylistic: orchestrator-watchdog.service is Type=oneshot on a 60s
# timer, so the process EXITS between probes and an in-memory counter could
# never accumulate. Same env-overridable REPO_DIR/data/fused-memory/ shape as
# FM_DEPLOY_CLOCK_PATH so tests can point it at a tmp file without touching
# real data/.
FM_LIVENESS_STREAK_PATH = os.environ.get(
    "FM_LIVENESS_STREAK",
    os.path.join(REPO_DIR, "data", "fused-memory", "fm_liveness_streak.json"),
)

# The liveness restart cap's OWN clock — a DIFFERENT file from
# FM_DEPLOY_CLOCK_PATH, preserving I5 in BOTH directions. Sharing one file
# would mean a liveness revive silences the fm STALENESS backstop for 8h (a
# merged fm change sitting undeployed because the watchdog once revived a
# wedge), and conversely a scheduled deploy would license an immediate extra
# liveness kill. Stamped ONLY by a liveness revive, read ONLY by the liveness
# pass.
FM_LIVENESS_RESTART_CLOCK_PATH = os.environ.get(
    "FM_LIVENESS_RESTART_CLOCK",
    os.path.join(
        REPO_DIR, "data", "fused-memory", "last_liveness_restart_fused_memory.json"
    ),
)


def log(msg: str) -> None:
    """Write *msg* to the systemd journal tagged as ``orchestrator-watchdog``.

    Falls back to stderr when ``systemd-cat`` is unavailable (e.g. a test
    environment, or a systemd-less host) or wedged. That is not a silent
    swallow: the unit sets ``StandardError=journal``, so the message still
    reaches the same journal by the other route.

    BOUNDED, like every other subprocess in this file (probe/show 5s,
    reset-failed 10s, restart 45s, the ``systemd-run --no-block`` delegations
    10s) — this one matters MORE than it looks. ``orchestrator-watchdog.timer``
    is ``OnUnitActiveSec=60``, measured from this unit's last activation, and
    systemd disables ``TimeoutStartSec`` for ``Type=oneshot`` by default — so
    a systemd-cat blocked on a stuck journald or a full /run would hang the
    tick forever, the timer would never re-trigger, and supervision would stop
    entirely with no signal. That is precisely the silent degradation this
    watchdog exists to catch elsewhere, arriving through its own LOGGING path
    (same defect fixed for scripts/dashboard-watchdog.py in 87ff5d1870).
    ``TimeoutStartSec`` in the unit file is the belt-and-braces second bound.

    NEVER raises from either journal route. A tooling failure in the logging
    path must not become a control-flow event at a call site — main()'s
    per-unit ``except Exception`` handler calls log() from INSIDE the except
    block, so an exception escaping here would abort the for-loop and leave
    the remaining WATCHED units unprobed for that tick. (An unrelated
    non-tooling exception type is still deliberately propagated — see the
    narrow except clause below.)
    """
    try:
        subprocess.run(
            ["systemd-cat", "-t", "orchestrator-watchdog"],
            input=msg,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # systemd-cat missing/unexecutable (OSError) or wedged past the bound
        # (TimeoutExpired, a SubprocessError) — still emit, just via stderr.
        # The fallback is itself best-effort: writing to stderr raises on a
        # broken pipe or a full/failing journal socket, and that OSError
        # would otherwise escape log() and abort a caller's tick (see the
        # never-raises contract in the docstring). Both journal routes are
        # gone at this point, so there is nothing left to report WITH —
        # dropping the message is the only remaining option, and it is
        # strictly better than dropping the rest of the tick with it.
        with contextlib.suppress(OSError):
            print(
                f"orchestrator-watchdog: {msg} [systemd-cat unusable: {exc!r}]",
                file=sys.stderr,
            )


class _JournalLog:
    """Route ``.warning(...)`` through the systemd-cat ``log()`` helper (single
    'orchestrator-watchdog' tag).

    Itself fail-soft: a journald-write failure must never convert a probe's
    return-None contract into a raised exception.

    Intentionally exposes ONLY ``.warning()`` — the minimal attribute-call
    surface required by the silent-fallthrough gate's WARN_METHODS check
    (see shared/tests/silent_fallthrough_scan.py::_handler_has_warn_log).
    This is not a general-purpose logging facade: other call sites in this
    module (e.g. probe_port's diagnostics) intentionally keep calling the
    bare ``log()`` helper directly, since those are not silent-fallthrough
    sig-b handlers and do not need gate exoneration. Do not add ``.info()``/
    ``.error()``/etc. here without re-checking whether they're actually
    needed for a gated handler.
    """

    def warning(self, msg: str) -> None:
        # Broad by design: logging must never break a fail-soft probe.
        with contextlib.suppress(Exception):
            log(f"WARNING: {msg}")


logger = _JournalLog()


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


def probe_health(
    url: str = FUSED_MEMORY_HEALTH_URL, timeout: float = FUSED_MEMORY_HEALTH_TIMEOUT_SECS
) -> bool:
    """Return True iff *url* returns ANY HTTP response within *timeout* seconds.

    Deliberately inverted fail-direction vs. probe_port(): probe_port defaults
    to True (assume healthy) on a tooling failure so a flaky `ss` invocation
    never triggers a restart. probe_health instead treats an HTTP response —
    including an error status such as 503 (surfaced by urlopen as
    urllib.error.HTTPError) — as the positive/alive signal, and returns False
    only when NO response is received at all (connection-refused, DNS
    failure, or a timeout).

    This asymmetry is intentional: fused-memory's /health returns 503 when a
    backing store (FalkorDB/Qdrant) is degraded, but a 503 still means the
    asyncio event loop IS serving requests — restarting the process would not
    fix a down store, would flap the single shared instance all 7
    orchestrators depend on, and would cancel expensive in-flight
    reconciliation work for nothing. The wedge this probe must actually catch
    (task 1731: an OS thread pings systemd's WATCHDOG=1 unconditionally, so a
    hung asyncio loop never trips systemd's own watchdog) manifests as no
    HTTP response at all within the timeout — that is the only case that
    should return False here.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        # The server responded (even with an error status) — the loop is
        # alive. A degraded backing store is not something a restart fixes.
        return True
    except Exception as exc:  # noqa: BLE001 -- any other failure means no response
        log(f"probe_health({url!r}) got no response ({type(exc).__name__}: {exc})")
        return False


def _fused_memory_liveness_verdict() -> str:
    """Classify fused-memory.service liveness as 'port-down' / 'healthy' / 'wedged'.

    Pure read-only (no restart, no logging) — the single source of truth for
    fused-memory's liveness, consumed by both fused_memory_liveness_pass()
    (which restarts on anything but 'healthy') and _print_fused_memory_liveness()
    (the --report row).

    The port probe runs first and short-circuits before the (up to 15s)
    health fetch, so a fully-dead unit is classified quickly without waiting
    on probe_health()'s timeout.
    """
    if not probe_port(FUSED_MEMORY_PORT):
        return "port-down"
    if probe_health():
        return "healthy"
    return "wedged"


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
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"_unit_start_epoch({unit!r}): swallowed {exc!r}; "
            "returning None (staleness undeterminable this tick)"
        )
        return None


def _unit_active_enter_epoch(unit: str) -> int | None:
    """Return *unit*'s realtime ActiveEnterTimestamp epoch (Unix seconds), or None.

    fm sibling of _unit_start_epoch, structurally identical but querying
    ``ActiveEnterTimestamp`` instead of ``ExecMainStartTimestamp``.
    fused_memory_staleness_pass() (task 2714) uses this field per the task's
    explicit choice: ActiveEnterTimestamp marks when a Type=notify service
    signalled readiness (= "when did this version start serving"), which is
    the semantically-correct staleness anchor for fused-memory and is exactly
    the field restart-all-orchestrators.sh's own freshness verify reads.

    Queries ``systemctl --user show --timestamp=unix`` which yields a clean,
    timezone-independent ``@<epoch>`` value directly comparable to git's
    ``%ct`` committer epoch. Returns None if the unit has never activated (the
    ``@0`` sentinel), the value cannot be parsed, or any subprocess/OS error
    occurs — callers must treat None as "staleness cannot be determined for
    this unit" (skip, don't guess).
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
                "ActiveEnterTimestamp",
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
                return None  # unit has never activated (no readiness signal recorded)
            return epoch
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"_unit_active_enter_epoch({unit!r}): swallowed {exc!r}; "
            "returning None (staleness undeterminable this tick)"
        )
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
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"_newest_watched_commit_epoch: swallowed {exc!r}; "
            "returning None (staleness undeterminable this tick)"
        )
        return None


def _newest_fm_watched_commit_epoch() -> int | None:
    """Return the newest committer epoch touching FM_WATCHED_PATHS on HEAD, or None.

    fm sibling of _newest_watched_commit_epoch, identical body but diffing
    FM_WATCHED_PATHS (fused-memory/src/ + shared/src/) instead of WATCHED_PATHS.
    fused_memory_staleness_pass() (task 2714) compares this against
    fused-memory.service's ActiveEnterTimestamp to detect a merged-but-
    undeployed fm change.

    Returns None if no commit touches the fm-watched paths (git exits 0 with
    EMPTY stdout — must be treated as undeterminable, not epoch 0, or
    fused-memory would look infinitely stale), on a non-zero exit, on
    unparseable stdout, or on any subprocess/OS error. Callers must treat None
    as "staleness cannot be determined this tick".
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
                *FM_WATCHED_PATHS,
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
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"_newest_fm_watched_commit_epoch: swallowed {exc!r}; "
            "returning None (fm staleness undeterminable this tick)"
        )
        return None


def _read_last_fleet_deploy_epoch() -> float | None:
    """Return the last verified fleet-deploy epoch from FLEET_DEPLOY_CLOCK_PATH, or None.

    Reads the same ``{ts, iso}`` JSON schema
    ``StaleServiceRestartCoordinator._load_last_fire_wall`` reads and
    ``restart-all-orchestrators.sh``'s ``stamp_fleet_deploy_clock`` writes, so
    all three tiers agree on the shared clock's format.

    Fail-open, mirroring ``_load_last_fire_wall``: returns None (never
    raises) when the file is missing (no fleet deploy has ever verified
    fresh, or a fresh checkout with no data/ yet), or when it is corrupt,
    unreadable, or missing its ``ts`` key. Callers must treat None as
    "the min-interval cap does not apply" (fail toward restarting, not
    toward silence).
    """
    try:
        with open(FLEET_DEPLOY_CLOCK_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return float(raw["ts"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log(
            f"ignoring unreadable/corrupt fleet-deploy clock at "
            f"{FLEET_DEPLOY_CLOCK_PATH}: {exc!r}"
        )
        return None


def _within_fleet_deploy_min_interval() -> bool:
    """Return True iff we are still inside the shared fleet-deploy min-interval window.

    Reads the same on-disk clock ``restart-all-orchestrators.sh`` stamps only
    on a verified-fresh fleet restart (never on mere fire/registration — see
    the coordinator's ``stamp_clock_on_fire=False`` for the orchestrator's own
    coordinator, task 2396). ORCH_RESTART_MIN_INTERVAL_SECS<=0 disables the
    cap outright (the clock is not even read). A missing/unreadable clock
    (``_read_last_fleet_deploy_epoch`` returns None) is treated as "outside
    the window" — fail toward letting the backstop run, not toward silencing
    it indefinitely.
    """
    if ORCH_RESTART_MIN_INTERVAL_SECS <= 0:
        return False
    last = _read_last_fleet_deploy_epoch()
    if last is None:
        return False
    return (time.time() - last) < ORCH_RESTART_MIN_INTERVAL_SECS


def _atomic_write_json(path: str, payload: dict) -> None:
    """Atomically write *payload* as JSON to *path*, creating its parent dir.

    The shared write primitive behind every piece of persisted watchdog state
    (fm deploy clock, fm liveness streak, fm liveness restart clock) — extracted
    in task 3764 so the mkdir/mktemp/write/rename dance is defined ONCE rather
    than copy-pasted per state file. Python analogue of
    restart-all-orchestrators.sh's stamp_fleet_deploy_clock: mkdir -p, mktemp a
    sibling in the SAME directory (so os.replace is a same-filesystem atomic
    rename), write, rename over the target.

    Fail-soft: makedirs / temp-file / rename errors are logged and swallowed
    (the temp file, if created, is unlinked) rather than raised. A persistence
    hiccup must never crash the Type=oneshot watchdog or abort the rest of the
    tick. Every caller's contract must therefore tolerate the write silently
    not having happened — see _record_fm_liveness_failure, where a failed write
    means the streak counter cannot accumulate and so fails SAFE (toward never
    restarting).
    """
    tmp_path: str | None = None
    try:
        target_dir = os.path.dirname(path)
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="." + os.path.basename(path) + ".", dir=target_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        os.replace(tmp_path, path)
        tmp_path = None  # renamed away — nothing to clean up
    except Exception as exc:  # noqa: BLE001
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        log(f"_atomic_write_json: failed to write {path}: {exc!r}")


def _read_json_state(path: str) -> dict | None:
    """Return the parsed JSON dict at *path*, or None if it cannot be read.

    The shared fail-open read primitive paired with _atomic_write_json
    (task 3764). Returns None — never raises — when the file is missing,
    corrupt, truncated (a torn concurrent write), or unreadable.

    A MISSING file is the normal "no state yet" case (fresh checkout, never
    stamped, streak just cleared) and is deliberately SILENT: logging it would
    spam the journal on every 60s tick. Corrupt/unreadable files ARE logged —
    those are genuine degradations, and swallowing them silently is exactly the
    silent-degradation failure mode the project's norms forbid.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        log(f"ignoring unreadable/corrupt watchdog state at {path}: {exc!r}")
        return None


def _read_last_fm_deploy_epoch() -> float | None:
    """Return the last verified fm-deploy epoch from FM_DEPLOY_CLOCK_PATH, or None.

    fm sibling of _read_last_fleet_deploy_epoch: reads the same ``{ts, iso}``
    JSON schema _stamp_fm_deploy_clock writes (which itself mirrors
    restart-all-orchestrators.sh's stamp_fleet_deploy_clock), so ``float(ts)``
    reads it identically.

    Fail-open: returns None (never raises) when the file is missing (no fm
    deploy has ever verified fresh, or a fresh checkout with no data/ yet), or
    when it is corrupt, unreadable, or missing its ``ts`` key. Callers must
    treat None as "the min-interval cap does not apply" (fail toward running
    the backstop, not toward silence).

    Thin wrapper over the shared _read_json_state helper (task 3764), which
    owns the missing/corrupt/unreadable fail-open branches; only the ``ts``
    extraction is fm-deploy-specific. Reads FM_DEPLOY_CLOCK_PATH at CALL time,
    not at def time, so tests that monkeypatch the module global still work.
    """
    state = _read_json_state(FM_DEPLOY_CLOCK_PATH)
    if state is None:
        return None
    try:
        return float(state["ts"])
    except (KeyError, TypeError, ValueError) as exc:
        log(
            f"ignoring fm-deploy clock at {FM_DEPLOY_CLOCK_PATH} "
            f"with unusable 'ts': {exc!r}"
        )
        return None


def _within_fm_deploy_min_interval() -> bool:
    """Return True iff we are still inside fm's own deploy min-interval window.

    fm sibling of _within_fleet_deploy_min_interval, reading fm's OWN clock
    (FM_DEPLOY_CLOCK_PATH) and honoring fm's OWN cap (FM_RESTART_MIN_INTERVAL_SECS)
    — independent of the orchestrator fleet clock. FM_RESTART_MIN_INTERVAL_SECS<=0
    disables the cap outright (the clock is not even read). A missing/unreadable
    clock (_read_last_fm_deploy_epoch returns None) is treated as "outside the
    window" — fail toward letting the backstop run, not toward silencing it
    indefinitely.
    """
    if FM_RESTART_MIN_INTERVAL_SECS <= 0:
        return False
    last = _read_last_fm_deploy_epoch()
    if last is None:
        return False
    return (time.time() - last) < FM_RESTART_MIN_INTERVAL_SECS


def _stamp_fm_deploy_clock() -> None:
    """Atomically stamp FM_DEPLOY_CLOCK_PATH with the current ``{ts, iso}`` time.

    Python analogue of restart-all-orchestrators.sh's stamp_fleet_deploy_clock
    (mkdir -p, mktemp a sibling, write, atomic rename). Needed because
    restart-fused-memory.sh — unlike restart-all-orchestrators.sh — does NOT
    stamp its own clock; the watchdog owns fm's clock. Called ONLY from the
    ``--stamp-fm-deploy-clock`` CLI subcommand, which _delegate_fm_restart
    chains after restart-fused-memory.sh's verified exit-0 (``&&``), so the fm
    clock advances only after a genuinely-verified restart (I2: a failed fm
    restart can never silence the backstop).

    Fail-soft: makedirs / temp-file / rename errors are logged and swallowed
    by _atomic_write_json (the temp file, if created, is unlinked) rather than
    raised — a stamp failure must not crash the detached unit. The backstop
    still self-heals via ActiveEnterTimestamp next tick, so the clock is only a
    secondary flap-guard.

    Thin wrapper over the shared _atomic_write_json helper (task 3764): this
    function owns only the ``{ts, iso}`` payload schema. Reads
    FM_DEPLOY_CLOCK_PATH at CALL time, not at def time, so tests that
    monkeypatch the module global still work.
    """
    now = time.time()
    _atomic_write_json(
        FM_DEPLOY_CLOCK_PATH,
        {
            "ts": int(now),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now)),
        },
    )


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


def fused_memory_liveness_pass() -> None:
    """Probe fused-memory.service liveness; restart it if the verdict isn't 'healthy'.

    Single-unit analogue of main()'s per-unit body, targeting FUSED_MEMORY_UNIT
    only — fused-memory is deliberately NOT added to WATCHED (see the
    module-level comment above FUSED_MEMORY_UNIT's definition), since both
    WATCHED and main()'s loop are pinned by exact-equality drift tests. This
    sibling pass reuses is_unit_enabled + STARTUP_GRACE_SECS gating verbatim
    from main(), and _fused_memory_liveness_verdict() (B2) for the combined
    port+/health verdict.

    Both 'port-down' and 'wedged' trigger a restart via restart_unit() called
    DIRECTLY — uncapped, with no fleet-deploy clock and no delegation to
    restart-all-orchestrators.sh (I5: brokenness is not a scheduled deploy),
    exactly like main()'s wedged-orchestrator revive path.

    Wrapped in a single try/except Exception (mirroring main()'s per-unit
    isolation) so a probe/verdict failure is logged and swallowed rather than
    raising — a hiccup here must never crash the oneshot watchdog or prevent
    staleness_pass() from running afterward.
    """
    try:
        if not is_unit_enabled(FUSED_MEMORY_UNIT):
            # Disabled (or unknown) — respect operator intent, skip silently.
            return
        elapsed = _unit_start_elapsed_secs(FUSED_MEMORY_UNIT)
        if elapsed is not None and elapsed < STARTUP_GRACE_SECS:
            log(
                f"{FUSED_MEMORY_UNIT} started {elapsed:.0f}s ago; "
                f"skipping probe (grace window {STARTUP_GRACE_SECS}s)"
            )
            return
        verdict = _fused_memory_liveness_verdict()
        if verdict != "healthy":
            log(f"{FUSED_MEMORY_UNIT} liveness verdict '{verdict}'; restarting")
            restart_unit(FUSED_MEMORY_UNIT)
            log(f"{FUSED_MEMORY_UNIT} restart issued")
    except Exception as exc:  # noqa: BLE001
        log(f"watchdog error for {FUSED_MEMORY_UNIT} (port {FUSED_MEMORY_PORT}): {exc}")


def _delegate_fleet_restart() -> None:
    """Delegate a fleet-wide staleness redeploy to restart-all-orchestrators.sh --drain.

    Fires a detached, named transient unit via ``systemd-run --user`` rather
    than restarting units in-process (task 2396, fleet-redeploy β): drain and
    clock-stamping are then defined ONCE in the script and identical whether
    the fleet restart was triggered by this backstop or by the event-driven
    coordinator / an operator.

    - ``--unit=orch-fleet-staleness-redeploy.service`` is a FIXED transient
      unit name — the natural overlap guard. A second staleness_pass tick
      while a redeploy is still running fails to re-register the same unit
      name (systemd-run exits non-zero) and no-ops, so this stateless
      oneshot needs no cross-tick bookkeeping to avoid piling up concurrent
      fleet restarts.
    - ``--collect`` removes the transient unit once it exits (success or
      failure) so a LATER tick can re-register the same name.
    - ``--no-block`` detaches: this call returns as soon as the transient
      unit is *registered*, without waiting for restart-all-orchestrators.sh
      to finish. Essential once γ's per-unit merge-drain gate can defer a
      restart for up to ORCH_RESTART_FORCE_FIRE_AFTER_SECS (75 min default)
      — the 60s oneshot watchdog (and its liveness pass) must never block on
      that.
    - ``--drain`` enables γ's per-unit merge-drain gate, so a watchdog-
      initiated fleet restart drains + stamps identically to an operator- or
      coordinator-driven ``restart-all-orchestrators.sh --drain``.

    Fail-soft: a missing systemd-run binary, a timeout, or any other
    registration error is logged and swallowed, never raised — a
    registration hiccup must not crash the oneshot watchdog. The NEXT tick's
    staleness_pass will simply try again (stateless — I6).
    """
    try:
        subprocess.run(
            [
                "systemd-run",
                "--user",
                "--collect",
                "--no-block",
                "--unit=orch-fleet-staleness-redeploy.service",
                os.path.join(REPO_DIR, "scripts", "restart-all-orchestrators.sh"),
                "--drain",
            ],
            check=False,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"_delegate_fleet_restart: systemd-run registration failed: {exc!r}")


def _delegate_fm_restart() -> None:
    """Delegate a fused-memory staleness redeploy to restart-fused-memory.sh (task 2714).

    fm sibling of _delegate_fleet_restart: fires a detached, named transient
    unit via ``systemd-run --user`` so the defer-if-busy restart and the
    exit-0-gated clock stamp run identically whether triggered by this backstop
    or (later) any other caller.

    - ``--unit=fm-staleness-redeploy.service`` (FM_STALENESS_REDEPLOY_UNIT) is a
      FIXED transient unit name — the natural overlap guard. A second tick while
      a redeploy is still running fails to re-register the same unit name and
      no-ops, so this stateless oneshot needs no cross-tick bookkeeping.
    - ``--collect`` removes the transient unit once it exits so a LATER tick can
      re-register the same name.
    - ``--no-block`` detaches: this call returns as soon as the transient unit
      is *registered*, without waiting for restart-fused-memory.sh to finish.
      Essential because restart-fused-memory.sh's default defer-if-busy path can
      hold up to RECON_GATE_TIMEOUT (35 min) while a full recon cycle runs — the
      60s oneshot watchdog must never block on that (task 2703 δ).
    - ``--setenv=FM_DEPLOY_CLOCK=<path>`` forwards the reader's RESOLVED clock
      path into the detached unit. ``systemd-run --user`` runs the transient
      unit under the systemd user *manager's* environment, NOT this watchdog
      process's — so without this, the chained ``--stamp-fm-deploy-clock`` would
      recompute FM_DEPLOY_CLOCK_PATH from the session env and default it,
      silently diverging from the path the reader
      (_within_fm_deploy_min_interval, in THIS process) consults whenever
      FM_DEPLOY_CLOCK is overridden. Pinning the stamp to FM_DEPLOY_CLOCK_PATH
      keeps writer==reader in all cases (a no-op at the default path;
      load-bearing under an operator/test override).

    KEY divergence from _delegate_fleet_restart (task ο design decision):
    restart-fused-memory.sh is invoked in its DEFAULT defer-if-busy mode (NO
    --now / --drain), and — because it does NOT stamp its own clock (unlike
    restart-all-orchestrators.sh) — the command chains ``&& <self>
    --stamp-fm-deploy-clock`` inside the SAME detached ``bash -c`` payload. The
    ``&&`` short-circuit means the fm clock is stamped ONLY on the restart
    script's verified exit-0, and the whole thing stays non-blocking (the stamp
    runs inside the detached unit, not inline in the watchdog).

    Fail-soft: a missing systemd-run binary, a timeout, or any other
    registration error is logged and swallowed, never raised — a registration
    hiccup must not crash the oneshot watchdog. The NEXT tick's
    fused_memory_staleness_pass will simply try again (stateless — I6).
    """
    restart_script = os.path.join(REPO_DIR, "scripts", "restart-fused-memory.sh")
    stamp_cmd = (
        f"{shlex.quote(sys.executable)} "
        f"{shlex.quote(os.path.abspath(__file__))} --stamp-fm-deploy-clock"
    )
    try:
        subprocess.run(
            [
                "systemd-run",
                "--user",
                "--collect",
                "--no-block",
                # Pin the detached stamp to the SAME clock file the reader
                # consults — systemd-run --user does not propagate this
                # process's env, so the chained --stamp-fm-deploy-clock would
                # otherwise default the path (see docstring). No-op at the
                # default path; load-bearing under a FM_DEPLOY_CLOCK override.
                f"--setenv=FM_DEPLOY_CLOCK={FM_DEPLOY_CLOCK_PATH}",
                f"--unit={FM_STALENESS_REDEPLOY_UNIT}",
                "/bin/bash",
                "-c",
                f"{shlex.quote(restart_script)} && {stamp_cmd}",
            ],
            check=False,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"_delegate_fm_restart: systemd-run registration failed: {exc!r}")


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

    Shared fleet-deploy clock (task 2396, fleet-redeploy β): checked FIRST,
    ahead of the commit-grace gate below — a top-priority, fleet-wide
    restraint that caps this backstop (like the event-driven coordinator) to
    at most once per ORCH_RESTART_MIN_INTERVAL_SECS, honoring a redeploy
    verified by EITHER tier (restart-all-orchestrators.sh is the sole on-disk
    writer, stamped only on its verified-fresh exit-0 path). The skip line is
    itself rate-limited to at most once per SKIP_LOG_INTERVAL_SECS (see its
    module-level docstring) — the gate check still runs every tick, only the
    log emission is throttled.

    Delegation (task 2396): once ANY eligible unit is found stale, the
    per-unit loop below no longer restarts it directly — instead the whole
    fleet-wide restart is delegated ONCE, after the loop, to
    _delegate_fleet_restart(). restart_unit() remains used ONLY by main()
    (liveness stays uncapped, non-clock-gated, and non-stamping — I5:
    brokenness is not a scheduled deploy).
    """
    if _within_fleet_deploy_min_interval():
        # Bucket on wall-clock time (not elapsed-since-deploy) so this needs
        # no extra clock-file read beyond the one _within_fleet_deploy_min_
        # interval() already did — see SKIP_LOG_INTERVAL_SECS above.
        if time.time() % SKIP_LOG_INTERVAL_SECS < 120:
            log(
                "skip: within fleet-deploy min-interval "
                f"({ORCH_RESTART_MIN_INTERVAL_SECS}s) since last deploy"
            )
        return

    commit_epoch = _newest_watched_commit_epoch()
    if commit_epoch is None:
        return  # undeterminable — fall safe, no restarts this tick
    if time.time() - commit_epoch < STALENESS_GRACE_SECS:
        # Give the polite event-driven restart coordinator its head start.
        # NOTE: gates on the newest commit's age alone — see the "Known
        # limitation" paragraph above for the rapid-landing suppression case.
        return

    stale_found = False
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
                    f"watched commit ({commit_epoch}); flagging for fleet-wide staleness redeploy"
                )
                stale_found = True
        except Exception as exc:  # noqa: BLE001
            log(f"staleness probe error for {unit}: {exc}")

    if stale_found:
        log("delegating fleet-wide staleness redeploy to restart-all-orchestrators.sh --drain")
        _delegate_fleet_restart()


def fused_memory_staleness_pass() -> None:
    """Restart fused-memory.service if it is stale w.r.t. the newest fm commit.

    Single-unit mirror of staleness_pass() targeting FUSED_MEMORY_UNIT — closes
    survey finding A2 (no staleness signal drives fused-memory redeploys, so a
    merged fm change sits undeployed until a human/PRD files a deploy task).
    fused-memory has an on-merge event-driven StaleServiceRestartCoordinator
    (survey finding A7) that can starve under load; this watchdog pass is the
    BACKSTOP for it, exactly as staleness_pass() backstops the orchestrator
    fleet's coordinator.

    Gate order (mirrors staleness_pass): (1) fm-deploy min-interval clock cap
    FIRST (fm's OWN clock, throttled skip-log); (2) newest fm-watched commit,
    None -> no-op; (3) commit-grace head-start reusing STALENESS_GRACE_SECS so
    the polite fm coordinator gets its head start before the backstop acts;
    (4) enabled / startup-grace / ActiveEnterTimestamp-vs-commit -> delegate
    once via _delegate_fm_restart().

    Stateless (I6): staleness is recomputed from live systemd + git each tick,
    so a successful restart (from this pass or the fm coordinator) advances
    ActiveEnterTimestamp past the commit and the unit reads fresh on the very
    next call — no stored state, no flap loop.

    Wrapped in a single try/except Exception around the per-unit probes
    (mirroring fused_memory_liveness_pass()'s isolation) so a probe hiccup is
    logged and swallowed rather than crashing the oneshot watchdog. Uses
    _delegate_fm_restart() (detached defer-if-busy chokepoint), never
    restart_unit() directly — liveness stays the only uncapped, non-clock-gated
    revive path (I5: brokenness is not a scheduled deploy).
    """
    if _within_fm_deploy_min_interval():
        # Bucket on wall-clock time (no extra clock read beyond the gate's) —
        # see SKIP_LOG_INTERVAL_SECS. The gate check still runs every tick;
        # only the log emission is throttled.
        if time.time() % SKIP_LOG_INTERVAL_SECS < 120:
            log(
                "skip: within fm-deploy min-interval "
                f"({FM_RESTART_MIN_INTERVAL_SECS}s) since last deploy"
            )
        return

    commit_epoch = _newest_fm_watched_commit_epoch()
    if commit_epoch is None:
        return  # undeterminable — fall safe, no restart this tick
    if time.time() - commit_epoch < STALENESS_GRACE_SECS:
        # Give the polite event-driven fm coordinator its head start.
        return

    try:
        if not is_unit_enabled(FUSED_MEMORY_UNIT):
            # Disabled (or unknown) — respect operator intent, skip silently.
            return
        elapsed = _unit_start_elapsed_secs(FUSED_MEMORY_UNIT)
        if elapsed is not None and elapsed < STARTUP_GRACE_SECS:
            # None => grace does not apply, proceed (mirrors main()).
            return
        active = _unit_active_enter_epoch(FUSED_MEMORY_UNIT)
        if active is None:
            return  # undeterminable — skip, don't guess
        if active < commit_epoch:
            log(
                f"WARNING: {FUSED_MEMORY_UNIT} activated at {active} before the newest "
                f"fm-watched commit ({commit_epoch}); delegating fused-memory staleness redeploy"
            )
            _delegate_fm_restart()
    except Exception as exc:  # noqa: BLE001
        log(f"fm staleness probe error for {FUSED_MEMORY_UNIT}: {exc}")


def _format_epoch(epoch: int | None) -> str:
    """Render a Unix epoch as a UTC timestamp string, or 'unknown' for None."""
    if epoch is None:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(epoch))


def _classify_unit_heartbeat(unit: str, now: float) -> str:
    """Classify *unit*'s merge-idle heartbeat for report()'s MERGE-IDLE column.

    Reuses scripts/drain_check.py's classify()/_read_heartbeat()/
    heartbeat_path()/resolve_fleet_dir() (task 2397, γ) via a lazy import, so
    this column predicts restart-all-orchestrators.sh's drain gate exactly
    rather than risking a reimplementation drifting from it. report() (doctor
    mode) is the ONLY caller — the liveness/staleness timer path (main() /
    staleness_pass()) never imports drain_check at all.

    Fail-soft: any exception (drain_check missing/unimportable, an unreadable
    fleet dir, or anything else) is swallowed and degrades this single column
    to 'unknown' rather than breaking report() as a whole.

    Inserts this module's own directory onto sys.path (guarded — a no-op if
    already present, e.g. via tests/scripts/conftest.py under test) so
    ``import drain_check`` resolves regardless of how this script was
    invoked.
    """
    try:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import drain_check  # noqa: PLC0415

        fleet_dir = drain_check.resolve_fleet_dir()
        path = drain_check.heartbeat_path(fleet_dir, unit)
        heartbeat = drain_check._read_heartbeat(path)
        return drain_check.classify(heartbeat, now, DRAIN_FRESH_WINDOW_SECS)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"_classify_unit_heartbeat({unit!r}): swallowed {exc!r}; returning 'unknown'"
        )
        return "unknown"


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

    DEPLOY-AGE is a single fleet-wide value (age since the shared
    fleet-deploy clock read via _read_last_fleet_deploy_epoch, task 2396 β)
    rendered as hours-to-one-decimal and repeated on every row, like the
    existing NEWEST WATCHED COMMIT column; 'unknown' when the clock is
    absent/unreadable.

    MERGE-IDLE is each unit's per-unit merge-idle heartbeat verdict
    (idle/busy/stale/absent/unknown), classified by _classify_unit_heartbeat
    via a lazy reuse of scripts/drain_check.py (task 2397 γ) so it predicts
    restart-all-orchestrators.sh's drain gate exactly. WOULD-DEFER is 'yes'
    iff MERGE-IDLE is 'busy' — the only verdict the drain gate actually
    defers on; idle proceeds immediately and stale/absent proceed after the
    gate's short unknown-grace.

    Read-only: report() never writes the fleet-deploy clock file and issues
    zero mutating systemctl calls (I8).
    """
    commit_epoch = _newest_watched_commit_epoch()
    units = _enumerate_running_units()
    now = time.time()
    deploy_epoch = _read_last_fleet_deploy_epoch()
    deploy_age_str = (
        f"{(now - deploy_epoch) / 3600:.1f}h" if deploy_epoch is not None else "unknown"
    )

    commit_str = _format_epoch(commit_epoch)
    print(
        "NOTE: verdict reflects raw start-time-vs-commit staleness only; it "
        "does not account for the enabled / startup-grace / commit-grace "
        "restraint gates staleness_pass() applies before actually restarting "
        "a unit."
    )
    print(
        f"{'UNIT':<50} {'START':<24} {'NEWEST WATCHED COMMIT':<24} {'VERDICT':<10} "
        f"{'DEPLOY-AGE':<12} {'MERGE-IDLE':<12} WOULD-DEFER"
    )

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
        merge_verdict = _classify_unit_heartbeat(unit, now)
        would_defer = "yes" if merge_verdict == "busy" else "no"
        print(
            f"{unit:<50} {start_str:<24} {commit_str:<24} {verdict:<10} "
            f"{deploy_age_str:<12} {merge_verdict:<12} {would_defer}"
        )

    return 1 if any_stale else 0


def _fused_memory_recon_busy_verdict() -> str:
    """Classify fused-memory's in-flight reconciliation state for ``--report``.

    Reuses scripts/recon_busy_check.py's parse_health()/classify() (task 2703
    δ) via a lazy import — the SAME busy/idle/unreachable gate
    restart-fused-memory.sh's default defer-if-busy path consumes — so this
    ``--report`` column predicts the restart script's recon gate exactly
    rather than risking a reimplementation drifting from it. Mirrors
    _classify_unit_heartbeat's lazy-reuse-of-a-sibling-script pattern.

    Fetches FUSED_MEMORY_HEALTH_URL's body and runs it through
    parse_health()+classify():
      - 'busy'        — a full reconciliation cycle is in flight
      - 'idle'        — no cycle running (recon_busy empty/absent)
      - 'unreachable' — the endpoint answered with a blank/unparseable body
      - 'unknown'     — the fetch/import itself failed (fail-soft, logged)

    Read-only: no restart, no clock write, no mutating call. Any exception
    (recon_busy_check missing/unimportable, the /health fetch failing, or
    anything else) is swallowed and degrades this single column to 'unknown'
    rather than breaking ``--report``.

    Inserts this module's own directory onto sys.path (guarded — a no-op if
    already present, e.g. via tests/scripts/conftest.py under test) so
    ``import recon_busy_check`` resolves regardless of how this script was
    invoked.
    """
    try:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import recon_busy_check  # noqa: PLC0415

        with urllib.request.urlopen(
            FUSED_MEMORY_HEALTH_URL, timeout=FUSED_MEMORY_HEALTH_TIMEOUT_SECS
        ) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return recon_busy_check.classify(recon_busy_check.parse_health(body))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"_fused_memory_recon_busy_verdict: swallowed {exc!r}; returning 'unknown'"
        )
        return "unknown"


def _print_fused_memory_liveness() -> None:
    """Print a single labelled fused-memory row for ``--report``.

    Strictly read-only, mirroring report()'s I7/I8 guarantee. Three fields,
    all read-only:
      - liveness verdict via _fused_memory_liveness_verdict() (port probe +
        /health fetch only — no is_unit_enabled/STARTUP_GRACE_SECS gating,
        since report() likewise shows every unit's raw verdict
        unconditionally);
      - DEPLOY-AGE from fused-memory's OWN deploy clock
        (_read_last_fm_deploy_epoch), rendered hours-to-one-decimal exactly
        like report()'s DEPLOY-AGE column, or 'unknown' when the fm clock has
        never been stamped / is unreadable (fail-open);
      - recon-busy via _fused_memory_recon_busy_verdict() (fail-soft
        'unknown').
    No systemctl mutation, no clock write, no restart — the DEPLOY-AGE read
    and the recon-busy /health fetch are both read-only.

    Informational only: called from _cli's --report branch AFTER report()'s
    own staleness table, and never affects report()'s staleness-only exit
    code (see _cli's docstring below).

    Wrapped in a try/except Exception mirroring fused_memory_liveness_pass()'s
    per-unit isolation (B3): probe_port only catches FileNotFoundError/
    TimeoutExpired, so an unusual subprocess failure (e.g. PermissionError)
    could otherwise propagate out of the verdict chain and crash --report
    after report() has already computed its exit code. An unexpected failure
    here degrades to a logged 'unknown' liveness verdict instead; DEPLOY-AGE
    (fail-open None) and recon-busy (fail-soft 'unknown') never raise.
    """
    try:
        verdict = _fused_memory_liveness_verdict()
    except Exception as exc:  # noqa: BLE001
        log(f"watchdog error printing {FUSED_MEMORY_UNIT} liveness row: {exc}")
        verdict = "unknown"
    deploy_epoch = _read_last_fm_deploy_epoch()
    deploy_age_str = (
        f"{(time.time() - deploy_epoch) / 3600:.1f}h"
        if deploy_epoch is not None
        else "unknown"
    )
    recon_busy = _fused_memory_recon_busy_verdict()
    print(
        f"{FUSED_MEMORY_UNIT} liveness (port {FUSED_MEMORY_PORT} + /health): "
        f"{verdict} | DEPLOY-AGE: {deploy_age_str} | recon-busy: {recon_busy}"
    )


def _cli(argv: list[str] | None = None) -> int:
    """Dispatch the CLI: ``--stamp-fm-deploy-clock`` / ``--report`` / timer path.

    With no ``argv`` argument, reads ``sys.argv[1:]``.

    ``--stamp-fm-deploy-clock`` (checked first) stamps fused-memory's own deploy
    clock and returns 0, running none of the liveness/staleness/report passes —
    it is the subcommand _delegate_fm_restart chains after a verified
    restart-fused-memory.sh exit-0 (task 2714).

    If ``--report`` is present, runs the read-only report() followed by the
    read-only _print_fused_memory_liveness() (B4) and returns report()'s OWN
    exit code (0 = all fresh, 1 = at least one stale unit) — the fm row is
    informational only and never alters this exit code. main(),
    fused_memory_liveness_pass(), staleness_pass(), and
    fused_memory_staleness_pass() are NOT invoked under --report, so this path
    never mutates systemd state (I7 at the CLI boundary): the fm staleness
    backstop runs on the timer path only.

    Otherwise runs the timer path: the liveness passes first (main() =
    orchestrator liveness, then fused_memory_liveness_pass() = fm liveness),
    then the scheduled clock-gated deploys (staleness_pass() = orchestrator
    fleet, then fused_memory_staleness_pass() = fm backstop, task 2714), and
    returns 0 — grouping immediate brokenness-revives ahead of scheduled
    deploys (I5). Unknown flags are not treated as an error — they fall through
    to the timer path.
    """
    argv = sys.argv[1:] if argv is None else argv
    if "--stamp-fm-deploy-clock" in argv:
        # Checked FIRST (before --report): this subcommand does exactly one
        # thing — stamp fm's deploy clock — and is chained after a verified
        # restart-fused-memory.sh exit-0 inside the detached _delegate_fm_restart
        # unit. It runs none of the liveness/staleness/report passes.
        _stamp_fm_deploy_clock()
        return 0
    if "--report" in argv:
        rc = report()
        _print_fused_memory_liveness()
        return rc
    main()
    fused_memory_liveness_pass()
    staleness_pass()
    fused_memory_staleness_pass()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
