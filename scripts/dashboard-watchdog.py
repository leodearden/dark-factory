#!/usr/bin/env python3
"""Dashboard availability watchdog — hysteresis probe with a storm escape.

Replaces the single-sample inline shell that shipped as
``dark-factory-dashboard-watchdog.service``'s ``ExecStart``:

    /bin/sh -c 'curl -sf --max-time 5 http://127.0.0.1:8080/healthz ||
                systemctl --user restart dark-factory-dashboard.service'

That one-liner probed the DEEP ``/healthz`` endpoint (three 5s DB probes) and
restarted the dashboard on a SINGLE miss, with no rate ceiling — which on
2026-07-30 produced 192 restarts in 3 hours (~27% downtime) from a dashboard
that was merely slow, not dead. It was also a contract-in-prose (INV-1): the
supervision policy lived inside an ``sh -c`` string with no test able to reach
it.

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
