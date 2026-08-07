"""Fan-out probe of per-task runtime state across all known orchestrators.

Each orchestrator's escalation MCP server owns the live runtime snapshot for
its own worktrees (loops/attempts/started/lane/phase/lane_state — see
``shared.task_runtime_state``). To surface that on the dashboard we fan out
``get_task_runtime_state`` calls across every known escalation URL
concurrently and aggregate the results into a ``{project_label:
TaskRuntimeSnapshot}`` dict.

Clones the ``dashboard.data.merge_halt`` fan-out pattern: per-call
``asyncio.wait_for`` timeout (the default ``mcp_tool_call`` timeout is 10s —
too slow for a fast-polling dashboard), ``asyncio.gather``, and per-project
offline degradation so one unreachable/misbehaving orchestrator never takes
down the others.

The wire payload is decoded through ``TaskRuntimeSnapshot.model_validate`` —
the SAME shared model the server projects into (INV-1 contracts-machine-
checked; no structurally-mirrored second copy). Any failure mode degrades
that ONE project to ``TaskRuntimeSnapshot(offline=True, offline_reason=...)``
— the model's own fields, synthesized client-side exactly as its docstring
prescribes — and names the FAULT DOMAIN of the failure (task 3517):

- ``'deadline_exceeded'`` — our OWN ``per_call_timeout`` fired. This says
  nothing about the orchestrator's health: under a starved dashboard event
  loop the probe never got enough CPU to complete, so the fault is the
  DASHBOARD's. See the aggregate warning in ``fetch_task_runtime``.
- ``'unreachable'`` — ANY ``httpx.TransportError`` (connect refused, a
  connection reset or protocol error mid-response, a proxy failure), an HTTP
  error status, or a malformed payload. The orchestrator IS the fault domain
  here and the operator's next action is to go look at it. Malformed-payload
  deliberately shares this member: same fault domain, same next action; the
  finer detail survives in that path's own WARNING.

``offline=True`` alone cannot separate those, which is why the 2026-07-30
event — a starved dashboard loop — was misdiagnosed as an orchestrator
outage.

Those two clauses between them catch EVERYTHING ``mcp_tool_call`` can raise
on the wire, which is load-bearing rather than merely tidy: ``asyncio.gather``
runs with ``return_exceptions=False``, so anything escaping them propagates
all the way out and 500s the whole tasks endpoint — turning one orchestrator's
hiccup into a total loss of every project's rows. See the tuple comments.

Per-project failures are logged at WARNING on a STATE TRANSITION only (see
``_log_probe_failure``): this endpoint is polled every few seconds per open
browser tab, so unconditional WARNINGs would bury the diagnosis under
thousands of identical lines an hour.

``DEFAULT_PER_CALL_TIMEOUT`` deliberately STAYS 2.0. A healthy call measures
0.2-0.4s, so 2.0s is already ~5-10x headroom; the 2026-07-30 failure was
event-loop starvation (since fixed by tasks 3304/3305/3309), and widening
the deadline would only paper over the next starvation instead of reporting
it.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from shared.task_runtime_state import TaskRuntimeSnapshot

from dashboard.data.memory import mcp_tool_call

logger = logging.getLogger(__name__)

DEFAULT_PER_CALL_TIMEOUT = 2.0

# ORDERING IS LOAD-BEARING — _DEADLINE_EXC MUST be caught BEFORE
# _UNREACHABLE_EXC. On this Python, builtin ``TimeoutError`` IS a subclass of
# ``OSError`` (and ``asyncio.TimeoutError is TimeoutError``), so an
# ``OSError``-bearing clause placed first would swallow every fired deadline
# and confidently mislabel it ``'unreachable'`` — reintroducing the exact
# 2026-07-30 misdiagnosis this taxonomy exists to prevent, but harder to spot
# because the field would be populated with a wrong value rather than left
# empty. Do not reorder these two except-clauses, and do not merge them.
#
# ``httpx.ConnectTimeout`` subclasses ``httpx.TimeoutException`` (NOT
# ``httpx.ConnectError``), so a connect-timeout lands in the deadline bucket
# by design: a timeout is a timeout, and under a starved loop even the connect
# leg can time out through no fault of the orchestrator.
#
# The unreachable tuple names the ``httpx.TransportError`` BASE, not a hand-
# picked list of leaves. Enumerating leaves silently missed the rest of the
# family: ``httpx.ReadError`` and ``httpx.RemoteProtocolError`` are neither
# ``OSError`` nor ``TimeoutException`` subclasses, so nothing caught them and
# — with ``asyncio.gather(return_exceptions=False)`` — one connection reset
# mid-response (exactly what an orchestrator restart on the fleet-redeploy
# path produces) propagated out of ``fetch_task_runtime`` and 500'd the whole
# tasks endpoint, losing EVERY project's rows rather than degrading the one
# that failed. ``httpx.TimeoutException`` is itself a ``TransportError``
# subclass, so widening to the base makes the clause ORDER above even more
# load-bearing, not less: deadlines are only kept out of this bucket by being
# caught first.
_DEADLINE_EXC = (TimeoutError, httpx.TimeoutException)
_UNREACHABLE_EXC = (httpx.TransportError, httpx.HTTPStatusError, ValueError, OSError)

# ── Bounded WARNING logging ──
# ``/api/v2/dashboard/tasks`` is uncached and the React app re-polls it every
# few seconds, once per open browser tab. Logging every probe failure at
# WARNING would emit an identical line per project per poll — on the order of
# a thousand an hour for ONE down orchestrator — burying the very signal this
# taxonomy exists to surface. So WARNING fires on a STATE TRANSITION only (a
# URL started failing, or its fault domain changed); a steady-state repeat
# drops to DEBUG, and a recovery logs one INFO and re-arms the WARNING. Same
# trade-off ``dashboard.config._discover_escalation_urls`` already reasons
# about: a bounded reminder, not log spam.
#
# Keyed by base_url and holding a LOG key, not the wire ``offline_reason``:
# connect-refused and malformed-payload share the ``'unreachable'`` member but
# have different operator detail, so a flip between them is a real transition
# worth one more line.
_LAST_PROBE_LOG_KEY: dict[str, str] = {}
# Labels that were ALL deadline-exceeded on the previous pass; the aggregate
# WARNING re-fires only when that set changes.
_LAST_AGGREGATE_LABELS: set[str] = set()


def reset_probe_log_state() -> None:
    """Forget the WARNING-dedupe memory (process-global).

    Exists so tests can assert on transition logging without depending on
    which test ran first. Production never needs to call it — the state is
    self-maintaining across polls.
    """
    _LAST_PROBE_LOG_KEY.clear()
    _LAST_AGGREGATE_LABELS.clear()


def _log_probe_failure(base_url: str, log_key: str, msg: str, *args: object) -> None:
    """WARNING on a transition for ``base_url``, DEBUG while it stays put."""
    changed = _LAST_PROBE_LOG_KEY.get(base_url) != log_key
    _LAST_PROBE_LOG_KEY[base_url] = log_key
    (logger.warning if changed else logger.debug)(msg, *args)


def _note_probe_success(base_url: str) -> None:
    """Clear the dedupe memory for a recovered URL so the next failure warns."""
    previous = _LAST_PROBE_LOG_KEY.pop(base_url, None)
    if previous is not None:
        logger.info(
            'get_task_runtime_state: %s recovered (was failing: %s)',
            base_url, previous,
        )


async def _probe_one(
    client: httpx.AsyncClient,
    base_url: str,
    timeout: float,
) -> TaskRuntimeSnapshot:
    try:
        result = await asyncio.wait_for(
            mcp_tool_call(client, base_url, 'get_task_runtime_state', {}),
            timeout=timeout,
        )
    except _DEADLINE_EXC as exc:
        # OUR budget expired, not necessarily their outage — see module docstring.
        _log_probe_failure(
            base_url, 'deadline_exceeded',
            'get_task_runtime_state: deadline_exceeded for %s after %.2fs budget '
            '(the orchestrator may be healthy; a starved dashboard event loop '
            'produces this too): %s',
            base_url, timeout, exc,
        )
        return TaskRuntimeSnapshot(offline=True, offline_reason='deadline_exceeded')
    except _UNREACHABLE_EXC as exc:
        _log_probe_failure(
            base_url, 'unreachable',
            'get_task_runtime_state: unreachable for %s (transport/HTTP failure): %s',
            base_url, exc,
        )
        return TaskRuntimeSnapshot(offline=True, offline_reason='unreachable')
    try:
        snapshot = TaskRuntimeSnapshot.model_validate(result)
    except ValueError as exc:
        # pydantic v2 ValidationError subclasses ValueError. Same fault domain
        # as a connect failure — the orchestrator is the problem — so it shares
        # the 'unreachable' member; the finer detail stays in this WARNING.
        _log_probe_failure(
            base_url, 'unreachable:malformed',
            'get_task_runtime_state: malformed payload from %s, degrading to '
            'offline (unreachable): %s',
            base_url, exc,
        )
        return TaskRuntimeSnapshot(offline=True, offline_reason='unreachable')
    _note_probe_success(base_url)
    return snapshot


async def fetch_task_runtime(
    client: httpx.AsyncClient,
    escalation_urls: dict[str, str],
    *,
    per_call_timeout: float = DEFAULT_PER_CALL_TIMEOUT,
) -> dict[str, TaskRuntimeSnapshot]:
    """Probe every escalation URL concurrently; return ``{label: TaskRuntimeSnapshot}``.

    Per-project labels match the keys used elsewhere in the dashboard
    (project basename — see ``active_tasks._project_label``). A failure
    degrades that project's entry to ``TaskRuntimeSnapshot(offline=True,
    offline_reason=...)`` so the caller can render an honest ``—`` rather
    than a fabricated zero, AND say which fault domain produced it —
    ``'deadline_exceeded'`` (our probe budget fired) vs ``'unreachable'``
    (any transport error / HTTP error / malformed payload). No wire failure
    escapes that split, so one sick orchestrator can never take the whole
    endpoint down with it. See the module docstring for the full taxonomy.

    Beyond the per-project reasons, this emits ONE aggregate WARNING when
    EVERY probed project (>= 2 of them) exceeded the deadline at once. That
    pattern is the 2026-07-30 discriminator: a real outage is per-project,
    so all-at-once points at the dashboard's own event loop rather than at
    the orchestrators. The returned mapping is unaffected — the signal is
    purely additive.

    Both the per-project and the aggregate lines are TRANSITION-logged: the
    first pass of a new failure mode is a WARNING, steady-state repeats drop
    to DEBUG, and a recovery re-arms the WARNING. This endpoint is polled
    every few seconds per open browser tab, so unconditional WARNINGs would
    bury the diagnosis they exist to deliver.
    """
    if not escalation_urls:
        return {}
    labels = list(escalation_urls.keys())
    urls = [escalation_urls[lbl] for lbl in labels]
    base_urls = [u.removesuffix('/mcp').rstrip('/') for u in urls]
    results = await asyncio.gather(
        *(_probe_one(client, base, per_call_timeout) for base in base_urls),
        return_exceptions=False,
    )
    # A genuine orchestrator outage is PER-PROJECT. Every probed project
    # blowing the deadline in the same pass is the signature of OUR event loop
    # being starved — say so, so an operator is not sent chasing N healthy
    # orchestrators (the 2026-07-30 misdiagnosis).
    #
    # The >= 2 threshold is load-bearing: with a single configured project the
    # pattern carries no information at all — "all projects timed out" is then
    # exactly "that one orchestrator might be down", and claiming the dashboard
    # is at fault would be the same unfounded diagnosis in the other direction.
    # dashboard.static.redux.runtime_format.rtProbeSummary applies the IDENTICAL
    # threshold so the log and the UI can never disagree.
    deadline = [
        lbl for lbl, snap in zip(labels, results, strict=True)
        if snap.offline_reason == 'deadline_exceeded'
    ]
    #
    # Like the per-project lines, this is TRANSITION-logged: a starvation event
    # spanning many polls would otherwise repeat this paragraph every few
    # seconds per open browser tab. It re-fires when the affected label set
    # changes (a project joining or leaving the pattern is new information).
    firing = len(labels) >= 2 and len(deadline) == len(labels)
    previous = set(_LAST_AGGREGATE_LABELS)
    _LAST_AGGREGATE_LABELS.clear()
    if firing:
        _LAST_AGGREGATE_LABELS.update(deadline)
        emit = logger.warning if previous != _LAST_AGGREGATE_LABELS else logger.debug
        emit(
            'get_task_runtime_state: all %d probed projects (%s) exceeded the '
            '%.2fs probe deadline in the same pass. A real orchestrator outage '
            'is per-project, so this pattern points at the DASHBOARD\'s own '
            'event loop being starved rather than at the orchestrators, which '
            'may be healthy — see the 2026-07-30 incident. Look at what is '
            'blocking the dashboard event loop before restarting anything.',
            len(labels), ', '.join(deadline), per_call_timeout,
        )
    return dict(zip(labels, results, strict=True))
