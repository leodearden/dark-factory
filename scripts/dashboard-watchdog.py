#!/usr/bin/env python3
"""Dashboard availability watchdog — hysteresis probe with a storm escape.

Replaces the single-sample inline shell that shipped as
``dark-factory-dashboard-watchdog.service``'s ``ExecStart``: a one-line
``curl ... || systemctl --user restart`` that probed the DEEP, DB-touching
health endpoint (three 5s DB probes) and restarted the dashboard on a SINGLE
miss, with no rate ceiling — which on 2026-07-30 produced 192 restarts in 3
hours (~27% downtime) from a dashboard that was merely slow, not dead. It was
also a contract-in-prose (INV-1): the supervision policy lived inside an
``sh -c`` string with no test able to reach it.

The retired endpoint is named nowhere in this file on purpose — its literal
absence is asserted by
``tests/scripts/test_dashboard_watchdog.py::test_healthz_appears_nowhere_in_the_source``
so no future edit can quietly route the probe back to it.

The real contract this file implements is
``plans/dashboard-availability-prd.md`` §Contract — the supervision seam.
Summarised:

  probe      GET http://127.0.0.1:8080/api/health (SHALLOW — a bare
             ``{'status': 'ok'}`` handler with no DB access), 200 within 5s
             is the only success. A non-200 is a FAILURE: this is the
             deliberate INVERSE of ``scripts/orchestrator-watchdog.py``'s
             probe_health, which treats a 503 as alive. Different targets,
             different contracts — see probe_health() below.
  hysteresis FAIL_STREAK consecutive failed probes are required before any
             actuation, so a single transient miss never restarts anything.
  grace      A unit that activated less than GRACE_SECS ago is not probed at
             all, and its streak is reset — a fresh activation invalidates any
             pre-restart streak.
  ceiling    At most MAX_RESTARTS restarts within a rolling RATE_WINDOW_SECS.
             On reaching the ceiling the watchdog files ONE born-at-L2
             escalation and STOPS restarting (the storm escape) rather than
             continuing to flap a service that restarting plainly does not fix.
  fail-soft  Every tooling failure (systemctl, state IO, the escalation
             subprocess) is warn-logged and swallowed. The oneshot never exits
             non-zero and never acts on a probe it did not actually run.

PRD open question 2 — "shell or Python?" — is resolved HERE as **Python**,
stdlib-only, modelled on ``scripts/orchestrator-watchdog.py``. The timer fires
a FRESH ``Type=oneshot`` process every 30s, so the failure streak, the rolling
restart-timestamp window and the escalation-dedup flag must all be persisted to
disk and re-read each tick; shell would need an ad-hoc file format and
hand-rolled atomic writes. The escalation call additionally needs precise argv
construction against ``escalation submit``'s enforced argparse boundary. The
PRD document itself is deliberately not edited to record this — sibling tasks
are resolving their own open questions in the same file — so this docstring and
the task's plan design-decisions are the record.

Invoked by ``dashboard/dark-factory-dashboard-watchdog.service`` (launched via
``dashboard/dark-factory-dashboard-watchdog.timer``, OnUnitActiveSec=30). The
file must stay mode 100755: ExecStart runs it directly, mirroring
``scripts/orchestrator-watchdog.service``.

Tested by ``tests/scripts/test_dashboard_watchdog.py`` (the bulk: probe, state,
streak, grace, ceiling, unit-file contract) and
``dashboard/tests/test_dashboard_watchdog_storm_escape.py`` (the storm escape
asserted against the REAL born-at-L2 writer, which ``tests/scripts/`` cannot
import).
"""

import os
import subprocess
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Contract constants — plans/dashboard-availability-prd.md §Contract.
#
# Numeric values are env-overridable using the same
# ``try: int(os.environ[...]) except (KeyError, ValueError)`` shape as
# scripts/orchestrator-watchdog.py's STALENESS_GRACE_SECS: a typo'd env var
# must fall back to the default, never crash the oneshot.
# ---------------------------------------------------------------------------

#: Repo checkout the watchdog supervises. Hardcoded rather than derived from
#: __file__ so a stray copy of this script cannot silently supervise a
#: different tree; mirrors orchestrator-watchdog.py:REPO_DIR.
REPO_DIR = "/home/leo/src/dark-factory"

#: §Contract "target": the unit this watchdog restarts. Its drain is bounded by
#: task 3306 (--timeout-graceful-shutdown 8 under TimeoutStopSec=15).
DASHBOARD_UNIT = "dark-factory-dashboard.service"

#: §Contract "probe": the SHALLOW liveness endpoint — dashboard/src/dashboard/
#: app.py's ``@app.get('/api/health') -> {'status': 'ok'}``, no DB access. The
#: deep DB-probing sibling immediately below it in that file is what the
#: retired inline shell used; see probe_health() for the fail-direction.
PROBE_URL = os.environ.get(
    "DASHBOARD_WATCHDOG_PROBE_URL", "http://127.0.0.1:8080/api/health"
)

#: §Contract "probe": seconds a single probe may take before it counts as a
#: failure. Well above the shallow handler's real latency (no I/O), well below
#: the timer's 30s cadence so a hung probe cannot overlap the next tick.
try:
    PROBE_TIMEOUT = int(os.environ["DASHBOARD_WATCHDOG_PROBE_TIMEOUT"])
except (KeyError, ValueError):
    PROBE_TIMEOUT = 5

#: §Contract "grace": a unit that activated less than this many seconds ago is
#: not probed at all (invariant I5 — never act on a probe that was not run),
#: and its streak is reset. Covers uvicorn startup plus the SPA mount.
try:
    GRACE_SECS = int(os.environ["DASHBOARD_WATCHDOG_GRACE_SECS"])
except (KeyError, ValueError):
    GRACE_SECS = 60

#: §Contract "hysteresis": consecutive failed probes required before ANY
#: actuation. 3 × the timer's OnUnitActiveSec=30 sets the ~90s
#: sustained-outage detection latency. This is the constant name the task's
#: sidecar delivered_check greps ``scripts/`` for — do not rename it.
try:
    FAIL_STREAK = int(os.environ["DASHBOARD_WATCHDOG_FAIL_STREAK"])
except (KeyError, ValueError):
    FAIL_STREAK = 3

#: §Contract "ceiling": restarts permitted inside one rolling
#: RATE_WINDOW_SECS. Deliberately a SEPARATE constant from FAIL_STREAK even
#: though both are 3 today — one counts consecutive failed probes, the other
#: counts restarts in a window. Reaching this ceiling files one born-at-L2
#: escalation and STOPS restarting (INV-4, the storm escape).
try:
    MAX_RESTARTS = int(os.environ["DASHBOARD_WATCHDOG_MAX_RESTARTS"])
except (KeyError, ValueError):
    MAX_RESTARTS = 3

#: §Contract "ceiling": width of the ROLLING window the restart count is
#: measured over (1h). Rolling, not lifetime: epochs older than this are
#: pruned, so a service that misbehaves once a day never trips the ceiling.
try:
    RATE_WINDOW_SECS = int(os.environ["DASHBOARD_WATCHDOG_RATE_WINDOW_SECS"])
except (KeyError, ValueError):
    RATE_WINDOW_SECS = 3600

#: Persisted tick state: ``{"streak": int, "restarts": [epoch, ...],
#: "ceiling_open": bool}``. Every timer tick is a FRESH oneshot process, so
#: none of this can live in memory. Sits under the root-anchored ``/data/``
#: gitignore.
STATE_PATH = os.environ.get(
    "DASHBOARD_WATCHDOG_STATE",
    os.path.join(REPO_DIR, "data", "dashboard-watchdog", "state.json"),
)

#: File-backed escalation queue the storm escape writes into — the same
#: directory dark-factory-orchestrator.yaml:116 configures as
#: ``escalation.queue_dir``, so the record is visible to the dashboard and the
#: escalation server without an MCP round-trip.
ESCALATION_QUEUE_DIR = os.environ.get(
    "DASHBOARD_WATCHDOG_QUEUE_DIR", os.path.join(REPO_DIR, "data", "escalations")
)

#: String-sentinel task id for an infra escalation with no owning task,
#: following the ``pipeline-landing-tripwire-{sha12}`` precedent in
#: orchestrator/src/orchestrator/merge_skew_tripwire.py — so the L2 is not
#: misattributed to an unrelated numeric task.
ESCALATION_TASK_SENTINEL = "dashboard-watchdog-restart-ceiling"

#: Must carry a ``harness-``/``orchestrator-`` prefix: escalation/src/
#: escalation/submit.py REJECTS a non-sentinel role at the argparse boundary,
#: because the CLI stamps level=2 directly and bypasses the server's chokepoint.
ESCALATION_AGENT_ROLE = "harness-dashboard-watchdog"

#: Must be a member of escalation.models.BORN_AT_L2_SEVERITIES ('critical',
#: 'urgent') — submit.py restricts --severity to those via argparse choices.
ESCALATION_SEVERITY = "critical"

#: Escalation category. 'infra_issue' matches the escalate_* category
#: vocabulary a dashboard/steward reader expects for a supervision failure.
ESCALATION_CATEGORY = "infra_issue"

#: uv launcher used to run ``escalation submit`` in the escalation project's
#: own environment — this script is stdlib-only and cannot import escalation.
#: Default matches dashboard/dark-factory-dashboard.service's ExecStart.
UV_BIN = os.environ.get("DASHBOARD_WATCHDOG_UV_BIN", "/home/leo/.local/bin/uv")

#: systemd-cat tag every log line carries — ``journalctl --user -t
#: dashboard-watchdog`` is the single place to read this watchdog's decisions.
LOG_TAG = "dashboard-watchdog"


# ---------------------------------------------------------------------------
# Logging (reused idiom-for-idiom from scripts/orchestrator-watchdog.py)
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    """Write *msg* to the systemd journal tagged as ``dashboard-watchdog``.

    Falls back to stderr when ``systemd-cat`` is unavailable (e.g. a test
    environment, or a systemd-less host). That is not a silent swallow: the
    unit sets ``StandardError=journal``, so the message still reaches the same
    journal by the other route.
    """
    try:
        subprocess.run(
            ["systemd-cat", "-t", LOG_TAG],
            input=msg,
            text=True,
            check=False,
        )
    except OSError:
        # systemd-cat missing/unexecutable — still emit, just via stderr.
        print(f"{LOG_TAG}: {msg}", file=sys.stderr)


class _JournalLog:
    """Route ``.warning(...)`` through the systemd-cat ``log()`` helper.

    Itself fail-soft: a journald-write failure must never convert a fail-soft
    handler's swallow-and-continue contract into a raised exception.

    Intentionally exposes ONLY ``.warning()`` — the minimal attribute-call
    surface required by the silent-fallthrough gate's WARN_METHODS check
    (shared/tests/silent_fallthrough_scan.py::_handler_has_warn_log, whose
    _SCOPE_ROOTS includes ``scripts``). This is not a general-purpose logging
    facade; non-handler call sites in this module keep calling bare ``log()``.
    Copied deliberately from scripts/orchestrator-watchdog.py so the two
    watchdogs' journal behaviour stays identical.
    """

    def warning(self, msg: str) -> None:
        try:
            log(f"WARNING: {msg}")
        except Exception:  # noqa: BLE001 -- logging must never break a fail-soft path
            pass


logger = _JournalLog()


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def probe_health(url: str = PROBE_URL, timeout: float = PROBE_TIMEOUT) -> bool:
    """Return True iff *url* answers HTTP 200 within *timeout* seconds.

    §Contract "probe". 200 is the ONLY success: a non-200 status, a connection
    refusal and a timeout are all failures. Note that a failure here does not
    actuate anything on its own — FAIL_STREAK consecutive failures are
    required before any restart, which is the whole point of this rewrite.

    DELIBERATE INVERSION vs scripts/orchestrator-watchdog.py's probe_health,
    which returns True on a 503. Do not "fix" one to match the other:

      * That probe guards fused-memory, whose /health returns 503 when a
        backing STORE (FalkorDB/Qdrant) is degraded. Restarting the process
        would not fix a down store and would flap the single shared instance
        all seven orchestrators depend on — so there, ANY HTTP response means
        the event loop is alive, and only silence is a failure.
      * This probe targets the dashboard's shallow ``{'status': 'ok'}``
        handler, which performs no I/O whatsoever. If it answers anything
        other than 200, the ASGI app itself is broken — and that a restart
        CAN fix. Accepting a 503 here would mean never noticing the exact
        failure the watchdog exists to catch.

    Fail-soft: every exception is caught and reported as "not healthy". A
    tooling failure inside the probe can therefore contribute to a streak, but
    still cannot by itself trigger a restart.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        # The server answered, but not with 200 — the shallow handler cannot
        # legitimately return anything else, so this is a real failure.
        log(f"probe_health({url!r}) got HTTP {exc.code} (expected 200)")
        return False
    except Exception as exc:  # noqa: BLE001 -- any other failure means no 200
        logger.warning(
            f"probe_health({url!r}) got no usable response "
            f"({type(exc).__name__}: {exc}); counting as a failed probe"
        )
        return False
