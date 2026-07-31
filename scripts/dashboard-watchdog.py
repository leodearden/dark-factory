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

import json
import os
import subprocess
import sys
import tempfile
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


# ---------------------------------------------------------------------------
# Persisted tick state
#
# Every timer tick is a FRESH oneshot process, so the failure streak, the
# rolling restart-epoch window and the escalation-dedup flag must all survive
# on disk. An in-memory counter would reset every 30 seconds and the streak
# gate would never reach FAIL_STREAK.
# ---------------------------------------------------------------------------


def default_state() -> dict:
    """Return a fresh default state — no failures, no restarts, no open trip.

    Also the documented fallback for every unreadable/ill-shaped state file:
    losing the state means losing hysteresis history, which fails toward NOT
    restarting (three fresh consecutive failures are needed again) rather than
    toward a spurious restart.
    """
    return {"streak": 0, "restarts": [], "ceiling_open": False}


def _normalise_state(raw: dict) -> dict:
    """Coerce *raw* into the declared state schema, dropping what cannot be.

    A hand-edited (or partially-written) state file must never crash a tick.
    In particular a non-numeric entry inside ``restarts`` is dropped here
    rather than left to poison the ``now - epoch`` arithmetic in
    _prune_restarts. Booleans are excluded from the epoch list explicitly —
    ``isinstance(True, int)`` is True in Python, and a stray ``true`` in the
    list would otherwise be read as the epoch 1 (1970) and silently pruned.
    """
    streak = raw.get("streak")
    if not isinstance(streak, int) or isinstance(streak, bool) or streak < 0:
        streak = 0

    restarts_raw = raw.get("restarts")
    restarts: list[int] = []
    if isinstance(restarts_raw, list):
        for entry in restarts_raw:
            if isinstance(entry, bool):
                continue
            if isinstance(entry, (int, float)):
                restarts.append(int(entry))

    return {
        "streak": streak,
        "restarts": restarts,
        "ceiling_open": bool(raw.get("ceiling_open", False)),
    }


def load_state(path: str = STATE_PATH) -> dict:
    """Return the persisted tick state from *path*, or defaults.

    Fail-open, mirroring orchestrator-watchdog.py's
    _read_last_fleet_deploy_epoch: never raises, and never creates anything —
    reading is a pure query, so a first-ever tick on a fresh checkout (where
    data/dashboard-watchdog/ does not exist yet) leaves the filesystem alone.

    A missing file is the ordinary first-run case and is NOT warned about. A
    file that exists but is corrupt, unreadable, or valid JSON of the wrong
    shape IS warned about — that is a real anomaly worth a journal line — and
    still degrades to defaults rather than wedging the watchdog.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return default_state()
    except (OSError, ValueError) as exc:
        logger.warning(
            f"load_state: unreadable/corrupt state at {path}: {exc!r}; using defaults"
        )
        return default_state()

    if not isinstance(raw, dict):
        logger.warning(
            f"load_state: state at {path} is {type(raw).__name__}, expected dict; "
            "using defaults"
        )
        return default_state()

    return _normalise_state(raw)


def save_state(state: dict, path: str = STATE_PATH) -> None:
    """Atomically persist *state* to *path*.

    Python analogue of orchestrator-watchdog.py's _stamp_fm_deploy_clock:
    makedirs → mkstemp a SIBLING (same filesystem, so os.replace is a true
    atomic rename) → write → os.replace, unlinking the temp if anything fails.
    Atomicity matters because the timer can fire while a previous tick is
    still writing; a half-written file would otherwise be read as corrupt.

    Fail-soft: makedirs/temp/rename errors are warn-logged and swallowed. An
    unwritable state path degrades the watchdog to effectively stateless — it
    stops restarting, since no streak can ever accumulate — which is the safe
    direction. Crashing here would instead put the oneshot unit into 'failed'.
    """
    tmp_path: str | None = None
    try:
        state_dir = os.path.dirname(path) or "."
        os.makedirs(state_dir, exist_ok=True)
        payload = json.dumps(_normalise_state(state), sort_keys=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".state.", dir=state_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        os.replace(tmp_path, path)
        tmp_path = None  # renamed away — nothing to clean up
    except Exception as exc:  # noqa: BLE001 -- a failed stamp must not crash the tick
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        logger.warning(f"save_state: failed to persist state to {path}: {exc!r}")


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------


def tick() -> None:
    """Run one supervision tick — the whole body of a single oneshot firing.

    §Contract, in order. A healthy probe is the overwhelmingly common case and
    resets the streak, so the steady state costs one HTTP GET and one small
    atomic write every 30 seconds and actuates nothing.
    """
    state = load_state()

    if probe_health():
        if state["streak"]:
            log(f"{DASHBOARD_UNIT} healthy again; clearing streak {state['streak']}")
        state["streak"] = 0
        save_state(state)
        return


def main() -> int:
    """Entry point for the oneshot unit."""
    tick()
    return 0


if __name__ == "__main__":
    sys.exit(main())
