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
             On reaching the ceiling the watchdog files at most ONE born-at-L2
             escalation per rolling RATE_WINDOW_SECS and STOPS restarting (the
             storm escape) rather than continuing to flap a service that
             restarting plainly does not fix.
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
import time
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
#: "ceiling_open": bool, "last_escalation_epoch": int}``. Every timer tick is a
#: FRESH oneshot process, so none of this can live in memory. Sits under the
#: root-anchored ``/data/`` gitignore.
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

    ``last_escalation_epoch`` 0 is the "never filed" sentinel and always
    permits filing — which is also what an on-disk state file written before
    this key existed normalises to, so no migration is needed.
    """
    return {
        "streak": 0,
        "restarts": [],
        "ceiling_open": False,
        "last_escalation_epoch": 0,
    }


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

    # Same coercion as a restarts entry, for the same reason: this epoch feeds
    # a ``now - epoch`` comparison in tick()'s escalation gate. Anything
    # unusable — including a negative, which would read as "filed before 1970"
    # and permanently satisfy the window — degrades to the 0 "never filed"
    # sentinel, i.e. toward filing the L2 rather than toward silence.
    last_escalation = raw.get("last_escalation_epoch", 0)
    if (
        isinstance(last_escalation, bool)
        or not isinstance(last_escalation, (int, float))
        or last_escalation < 0
    ):
        last_escalation = 0

    return {
        "streak": streak,
        "restarts": restarts,
        "ceiling_open": bool(raw.get("ceiling_open", False)),
        "last_escalation_epoch": int(last_escalation),
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
# Startup grace
# ---------------------------------------------------------------------------


def _unit_active_enter_epoch(unit: str = DASHBOARD_UNIT) -> int | None:
    """Return *unit*'s ActiveEnterTimestamp as a Unix epoch, or None.

    Near-verbatim reuse of scripts/orchestrator-watchdog.py's function of the
    same name, so the two watchdogs read systemd the same way.

    ``ActiveEnterTimestamp`` — not ``ExecMainStartTimestamp`` — is the field
    the §Contract "grace" row is written against: it marks when the unit
    reached *active*, i.e. when the grace window for THIS activation began.
    ``--timestamp=unix`` is what makes the value a timezone-independent
    ``@<epoch>`` integer rather than a locale-formatted date string that
    int() would reject (which would silently disable the gate).

    Returns None — meaning "the activation time cannot be determined" — for
    the ``@0`` never-activated sentinel, an unparseable value, a non-zero
    returncode, or any OS/subprocess error. Callers must treat None as
    "grace does not apply" and go on to probe, rather than guessing: failing
    the other way would let a broken systemctl query permanently disarm the
    watchdog, leaving a genuinely dead dashboard dead forever.
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
            logger.warning(
                f"_unit_active_enter_epoch({unit!r}): systemctl show exited "
                f"{result.returncode}; grace cannot be determined this tick"
            )
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
                logger.warning(
                    f"_unit_active_enter_epoch({unit!r}): unparseable "
                    f"ActiveEnterTimestamp {val!r}; grace does not apply"
                )
                return None
            if epoch == 0:
                # systemd's never-activated sentinel. Read as an epoch it
                # would place activation in 1970 and grace would never apply.
                return None
            return epoch
        return None
    except Exception as exc:  # noqa: BLE001 -- a query failure must not crash the tick
        logger.warning(
            f"_unit_active_enter_epoch({unit!r}): swallowed {exc!r}; "
            "returning None (grace undeterminable this tick)"
        )
        return None


# ---------------------------------------------------------------------------
# Actuation
# ---------------------------------------------------------------------------


def restart_unit(unit: str = DASHBOARD_UNIT) -> None:
    """Restart *unit* via the USER manager: ``reset-failed`` then ``restart``.

    §Contract "actuation". Two deliberate divergences from
    scripts/orchestrator-watchdog.py's restart_unit, which does a three-phase
    stop → reset-failed → start:

      * ``restart`` rather than stop+start. Task 3306 bounded this unit's
        drain explicitly (``--timeout-graceful-shutdown 8`` inside
        ``TimeoutStopSec=15``) and PRD row B7 states its expectation against
        ``systemctl restart``, so the single verb is the contract the rest of
        the batch is written against. It is also atomic from systemd's point
        of view: a stop+start pair can leave the dashboard DOWN if the
        oneshot is killed between the two calls, which is the worse failure
        for an availability watchdog to own.
      * ``reset-failed`` still runs FIRST, and unconditionally. Without it a
        unit that has exhausted StartLimitBurst silently ignores the restart:
        systemctl returns, the watchdog logs "restart issued", the streak
        resets — and nothing actually happened. That is a fail-soft hole that
        makes the gate LOOK like it fired, so it is pinned by
        test_b3_reset_failed_precedes_the_restart.

    Timeouts are bounded (10s for the cheap reset, 45s for the restart —
    comfortably above the unit's TimeoutStopSec=15) so a wedged systemctl
    cannot leave the oneshot running into the next 30s tick. A TimeoutExpired
    is warn-logged and swallowed: the restart is still RECORDED by the caller
    (a timed-out restart very often still took effect, and counting it is the
    fail-direction that leads to the storm escape rather than to a flap).
    """
    try:
        subprocess.run(
            ["systemctl", "--user", "reset-failed", unit], check=False, timeout=10
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            f"systemctl reset-failed {unit} timed out after 10s; "
            "attempting the restart anyway"
        )
    except Exception as exc:  # noqa: BLE001 -- actuation must never crash the oneshot
        logger.warning(f"systemctl reset-failed {unit} failed: {exc!r}")

    try:
        subprocess.run(
            ["systemctl", "--user", "restart", unit], check=False, timeout=45
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"systemctl restart {unit} timed out after 45s")
    except Exception as exc:  # noqa: BLE001 -- actuation must never crash the oneshot
        logger.warning(f"systemctl restart {unit} failed: {exc!r}")


# ---------------------------------------------------------------------------
# Storm escape (INV-4)
# ---------------------------------------------------------------------------


def _prune_restarts(restarts: list[int], now: int) -> list[int]:
    """Return the epochs in *restarts* that fall inside the rolling window.

    The window is ROLLING, not a lifetime counter: without the prune, a
    service that misbehaves once a month would eventually accumulate
    MAX_RESTARTS epochs, trip the ceiling, stop being supervised, and file an
    L2 about a problem that had already gone away years earlier.
    """
    return [epoch for epoch in restarts if now - epoch < RATE_WINDOW_SECS]


def file_ceiling_escalation(restart_count: int, window_secs: int) -> None:
    """File a born-at-L2 escalation reporting the restart ceiling.

    Shells out to ``escalation submit`` through ``uv run --project`` rather
    than importing it: this script is stdlib-only by design (it must run as a
    bare ``Type=oneshot`` with no environment of its own), and ``escalation``
    is not on its import path.

    The argv is built against submit.py's ENFORCED argparse boundary, not
    against convenience. That CLI constructs ``Escalation(level=2, ...)``
    directly — bypassing the escalation server's severity chokepoint — so it
    restricts ``--severity`` to BORN_AT_L2_SEVERITIES and REJECTS an
    ``--agent-role`` without a ``harness-``/``orchestrator-`` sentinel prefix.
    Both are satisfied by ESCALATION_SEVERITY and ESCALATION_AGENT_ROLE, and
    pinned by tests, because a rejected submit would leave the watchdog quiet
    at the ceiling with nobody told: the worst of both behaviours.

    ``--task`` is a STRING SENTINEL rather than a numeric task id, following
    the ``pipeline-landing-tripwire-{sha12}`` precedent in
    orchestrator/src/orchestrator/merge_skew_tripwire.py — there is no owning
    task for an infra failure, and attaching one would misattribute the L2.

    DEDUP is the CALLER's job and lives in tick(): at most one L2 per rolling
    RATE_WINDOW_SECS, keyed on the persisted ``last_escalation_epoch``. That is
    deliberately the SAME window the ceiling itself is measured over, so
    "the ceiling is saturated" and "an L2 about this saturation is already on
    file" cannot disagree.

    It is keyed on that epoch rather than on a
    ``queue.get_by_task(..., status='pending')`` query — the way the merge
    tripwire does it — precisely because this script cannot import escalation
    to run that query.

    An earlier version keyed the dedup on the ``ceiling_open`` flag instead,
    reasoning that one L2 per ceiling EPISODE was the same thing. It is not: a
    single successful probe ends an episode, while the restart window it is
    measured against is deliberately preserved across that recovery. A
    flapping dashboard — one healthy tick, then FAIL_STREAK failing ones, four
    ticks and ~120s at the timer's cadence — therefore closed the episode and
    immediately re-tripped the still-saturated ceiling, filing a fresh
    paging-severity L2 roughly every two minutes. The escape has to bound the
    escalation rate over the same horizon as the restart rate, or it just
    trades a restart storm for an escalation storm.

    Fail-soft: a missing uv, a non-zero exit, or a timeout is warn-logged and
    swallowed. A failed submit must not leave the watchdog restarting again —
    the ceiling stays tripped either way, which fails toward "quiet and
    logged" rather than toward resuming the flap.
    """
    summary = (
        f"{DASHBOARD_UNIT} hit the watchdog restart ceiling: {restart_count} "
        f"restarts in the last {window_secs}s. Restarting is not fixing it; "
        "the watchdog has STOPPED restarting and needs a human."
    )
    detail = (
        f"scripts/dashboard-watchdog.py restarted {DASHBOARD_UNIT} "
        f"{restart_count} times within the rolling {window_secs}s window "
        f"(MAX_RESTARTS={MAX_RESTARTS}), each after {FAIL_STREAK} consecutive "
        f"failed probes of {PROBE_URL}. Reaching the ceiling is evidence that "
        "a restart does not repair the fault, so per INV-4 the watchdog has "
        "stopped actuating rather than adding downtime (the 2026-07-30 "
        "incident was 192 restarts in 3h, ~27% downtime). It keeps probing: "
        "if the dashboard recovers on its own the episode ends and normal "
        "supervision resumes. Investigate with `journalctl --user -t "
        f"{LOG_TAG}` and `journalctl --user -u {DASHBOARD_UNIT}`. State file: "
        f"{STATE_PATH}."
    )

    argv = [
        UV_BIN,
        "run",
        "--project",
        os.path.join(REPO_DIR, "escalation"),
        "escalation",
        "submit",
        "--queue-dir",
        ESCALATION_QUEUE_DIR,
        "--task",
        ESCALATION_TASK_SENTINEL,
        "--severity",
        ESCALATION_SEVERITY,
        "--category",
        ESCALATION_CATEGORY,
        "--summary",
        summary,
        "--detail",
        detail,
        "--agent-role",
        ESCALATION_AGENT_ROLE,
    ]

    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning(
                f"file_ceiling_escalation: `escalation submit` exited "
                f"{result.returncode}: {result.stderr.strip()[:500]!r}. The "
                "ceiling stays tripped (no further restarts) regardless."
            )
            return
        log(f"filed born-at-L2 restart-ceiling escalation for {DASHBOARD_UNIT}")
    except Exception as exc:  # noqa: BLE001 -- a failed submit must not resume the flap
        logger.warning(
            f"file_ceiling_escalation: could not run `escalation submit` "
            f"({type(exc).__name__}: {exc}). The ceiling stays tripped, so the "
            "watchdog is quiet but the L2 was NOT filed — check the journal."
        )


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

    # §Contract "ceiling", tripped state. INV-4: once the ceiling has been
    # reached, no further restarts happen until the episode ends. This branch
    # never files anything at all — the L2 was filed on the tick that tripped
    # the ceiling, and its rate limit (one per rolling RATE_WINDOW_SECS) is
    # enforced there, not here.
    #
    # The tick still PROBES, because the episode has to be able to end. Taken
    # literally, I4's "until an operator intervenes" means a state file
    # someone deletes by hand — so a dashboard that recovered on its own at
    # 3am would stay unsupervised until a human noticed a stale flag they do
    # not know exists. A successful probe is unambiguous evidence the episode
    # is over: the service is answering, there is nothing left to restart, and
    # normal supervision (which still requires FAIL_STREAK fresh consecutive
    # misses before it acts) can safely resume. A dashboard that stays down
    # stays cleanly down, with its L2 on file, for as long as the fault lasts.
    if state["ceiling_open"]:
        if probe_health():
            logger.warning(
                f"{DASHBOARD_UNIT} is healthy again; closing the restart-ceiling "
                "episode and resuming normal supervision. The L2 already on "
                "file still describes what happened and is not withdrawn."
            )
            state["ceiling_open"] = False
            state["streak"] = 0
            # The restart epochs are deliberately NOT cleared. They are the
            # evidence the rolling ceiling is measured against; wiping them
            # here would let a dashboard alternate crash → recover → crash and
            # earn a fresh MAX_RESTARTS allowance every cycle — a restart storm
            # with extra steps.
            save_state(state)
            return

        # Still down. Log and leave the streak at 0: banking a streak while
        # nobody is acting on it would mean the first miss after a recovery
        # instantly satisfies FAIL_STREAK, bypassing the hysteresis gate.
        logger.warning(
            f"{DASHBOARD_UNIT} restart ceiling is OPEN and the probe is still "
            "failing; an L2 is on file, taking no action"
        )
        return

    # §Contract "grace". A unit that activated less than GRACE_SECS ago is not
    # probed AT ALL (invariant I5 — never act on a probe that was not run, so
    # do not run one whose answer must be discarded): uvicorn needs a few
    # seconds to bind and mount the SPA, and a probe during that window fails
    # for a perfectly healthy service.
    #
    # The streak is RESET here rather than merely left alone. A fresh
    # activation invalidates any pre-restart streak: without this, the ticks
    # that caused a restart would combine with the first post-grace miss to
    # restart again after a single failed probe — the incident's per-tick
    # behaviour, reintroduced through the back door.
    active_enter = _unit_active_enter_epoch(DASHBOARD_UNIT)
    if active_enter is not None and int(time.time()) - active_enter < GRACE_SECS:
        if state["streak"]:
            log(
                f"{DASHBOARD_UNIT} activated <{GRACE_SECS}s ago; inside startup "
                f"grace, clearing streak {state['streak']} and skipping the probe"
            )
        state["streak"] = 0
        save_state(state)
        return

    if probe_health():
        if state["streak"]:
            log(f"{DASHBOARD_UNIT} healthy again; clearing streak {state['streak']}")
        state["streak"] = 0
        save_state(state)
        return

    # Failed probe. Advance the streak and persist it — the next tick is a
    # different PROCESS, so this write is the only thing that carries the
    # count forward.
    state["streak"] += 1
    save_state(state)

    if state["streak"] < FAIL_STREAK:
        # §Contract "hysteresis". Below the gate nothing is actuated: this is
        # precisely where the retired inline shell restarted, and where the
        # 2026-07-30 storm began.
        logger.warning(
            f"{DASHBOARD_UNIT} probe failed ({state['streak']}/{FAIL_STREAK} "
            "consecutive); below the streak gate, taking no action"
        )
        return

    # §Contract "ceiling". The streak is complete, so a restart is warranted —
    # but first check whether restarting is actually WORKING. Prune the rolling
    # window, then compare what remains against MAX_RESTARTS.
    now = int(time.time())
    state["restarts"] = _prune_restarts(state["restarts"], now)

    if len(state["restarts"]) >= MAX_RESTARTS:
        # INV-4, the storm escape. MAX_RESTARTS restarts inside one window is
        # evidence that a restart does not repair this fault, so continuing
        # would only add downtime — which is exactly what 2026-07-30 was.
        # File ONE L2 and stop actuating.
        logger.warning(
            f"{DASHBOARD_UNIT} hit the restart ceiling "
            f"({len(state['restarts'])} restarts in {RATE_WINDOW_SECS}s >= "
            f"MAX_RESTARTS={MAX_RESTARTS}); NOT restarting"
        )

        # The L2 is rate-limited over the SAME rolling window the ceiling is
        # measured over. Not over the ceiling episode: a single healthy probe
        # ends an episode while these restart epochs survive it, so a flapping
        # dashboard re-reaches this branch every ~120s and would page a human
        # every time. 0 means never filed and always permits filing, so a
        # pre-existing state file from before this key existed still escalates.
        #
        # The stamp is written when the submit is ATTEMPTED, not when it
        # succeeds — file_ceiling_escalation is fail-soft, and stamping on
        # success would make a persistently broken submit path retry on every
        # re-trip, which is the same storm wearing a different costume (uv
        # spawns instead of records) at the moment the host is least healthy.
        # The L2 is still retried, just at window cadence, and the failure is
        # already in the journal.
        last_escalation = state["last_escalation_epoch"]
        if last_escalation == 0 or now - last_escalation >= RATE_WINDOW_SECS:
            state["last_escalation_epoch"] = now
            file_ceiling_escalation(len(state["restarts"]), RATE_WINDOW_SECS)
        else:
            log(
                f"{DASHBOARD_UNIT} re-reached the restart ceiling, but an L2 "
                f"filed {now - last_escalation}s ago already describes it "
                f"(one per {RATE_WINDOW_SECS}s window); not re-filing"
            )

        state["ceiling_open"] = True
        state["streak"] = 0
        save_state(state)
        return

    # §Contract "hysteresis", gate reached: FAIL_STREAK consecutive failed
    # probes. Actuation is per COMPLETED STREAK, never per tick (invariant
    # I3) — the streak is reset to 0 below, so the NEXT restart needs another
    # FAIL_STREAK consecutive misses rather than firing again in 30 seconds.
    logger.warning(
        f"{DASHBOARD_UNIT} failed {state['streak']} consecutive probes "
        f"(>= FAIL_STREAK={FAIL_STREAK}); restarting"
    )
    restart_unit(DASHBOARD_UNIT)

    # Record the restart BEFORE resetting the streak, and record it even if
    # systemctl misbehaved: the rolling window that feeds the storm escape is
    # measured against these epochs, so an unrecorded restart would make the
    # ceiling unreachable and re-open the flap the escape exists to stop.
    state["restarts"].append(now)
    state["streak"] = 0
    save_state(state)


def main() -> int:
    """Entry point for the oneshot unit. ALWAYS exits 0.

    Mirrors orchestrator-watchdog.main()'s isolation. A watchdog that exits
    non-zero puts its oneshot into `failed`, where it supervises nothing, is
    silently skipped by some restart tooling, and shows up as a red unit an
    operator must clear by hand before supervision resumes. That is a far
    worse outcome than one skipped tick 30 seconds before the next one.

    Swallowing here is safe precisely because it cannot cause an actuation:
    an exception unwinding out of tick() means the tick did NOT reach the
    restart branch (invariant I5 — never act on a probe that was not run).
    Failing quietly therefore always fails toward doing nothing.
    """
    try:
        tick()
    except Exception as exc:  # noqa: BLE001 -- a wedged tick must not fail the unit
        logger.warning(
            f"tick() raised {type(exc).__name__}: {exc}; skipping this tick. "
            "The unit still exits 0 so the timer keeps firing."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
