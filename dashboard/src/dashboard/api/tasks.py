"""`/api/v2/dashboard/tasks` — the active-task table and its done counts.

Fans out over every known project root, reporting four distinct failure
facts separately (offline / degraded / count-unknown / no rows anywhere)
rather than collapsing them into one "empty" — see the handler docstring
for why each is its own signal.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import (
    _all_project_roots,
    _project_label,
    collect_tasks_with_counts,
    shape_terminal_rows,
)
from dashboard.data.datum import Datum, DatumContractError, DatumState, validate_datum
from dashboard.data.task_snapshot import (
    FRESHNESS_BOUND_SECONDS,
    SnapshotFailure,
    SnapshotHealth,
    TaskSnapshot,
    acquire_terminal_window,
    as_served,
    classify,
    measured_terminal_total,
    unmeasured_snapshot,
)
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

router = APIRouter()


def _validated(
    label: str, snapshot: TaskSnapshot, served_at: datetime,
) -> TaskSnapshot:
    """*snapshot*, or a fully-unknown stand-in if it breaks the envelope contract.

    The producer already validates what it builds, so reaching this fallback
    means a datum crossed a seam it should not have. Degrading THAT ROOT and
    logging it applies the rule ``collect_tasks_with_counts._one`` already
    enforces one layer down — one bad root must never blank the whole tab — at
    the layer where the break would otherwise 500 the entire fan-out.

    The stand-in is unknown on both halves rather than a partial copy: the
    break is in the envelope itself, so no half of it can be trusted, and the
    reason names the invariant so a reader is not left guessing which.
    """
    try:
        validate_datum(snapshot.census, served_at)
        validate_datum(snapshot.rows, served_at)
    except DatumContractError as broken:
        logger.warning(
            'project %s: its task snapshot breaks the %s datum invariant and '
            'is served as UNKNOWN for this render; this is a BUG, not an outage',
            label, broken.invariant.value, exc_info=True,
        )
        return unmeasured_snapshot(
            label, now=served_at,
            reason=f'snapshot broke the {broken.invariant.value} invariant: {broken}',
            failure=SnapshotFailure.UNREACHABLE,
        )
    return snapshot


async def _terminal_window(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    label: str,
    snapshots: dict[str, TaskSnapshot],
    *,
    served_at: datetime,
) -> Datum[list[dict]]:
    """The terminal rows ``?terminal=<label>`` asked for, as one ``Datum``.

    THE ONLY read the default render does not make, spent here because the
    client that wants terminal rows is the only one that should pay for them.
    ``task_snapshot.acquire_terminal_window`` owns what the window bounds and
    why its answer is a ``LOWER_BOUND``; this resolves the label the request
    names and hands the rows to the shaper every other task row goes through.

    An unresolvable *label* is ANSWERED — never raised on, and never answered
    with an empty list. A 500 would lose the rest of a payload that is
    otherwise fine, and an empty success reads as "this project has no
    terminal tasks", which is a measurement nobody made.
    """
    root = next(
        (r for r in _all_project_roots(config) if _project_label(r) == label), None
    )
    if root is None:
        return Datum(
            None, None, DatumState.UNKNOWN,
            f'no configured project root is labelled {label!r}',
            FRESHNESS_BOUND_SECONDS,
        )
    snapshot = snapshots.get(label)
    window = await acquire_terminal_window(
        client, config, root, now=served_at,
        # None when this root's census is not fresh, which is exactly what
        # makes the window unpositionable: the count the offset is computed
        # from IS the census's, so the two facts cannot disagree.
        terminal_total=None if snapshot is None else measured_terminal_total(snapshot),
    )
    if window.value is None:
        return window
    return replace(
        window, value=shape_terminal_rows(Path(root), window.value, now=served_at),
    )


def _terminal_key(project: str) -> str:
    """The payload key the ``?terminal=<project>`` window answers under.

    The other half of this contract is ``data.js::ON_DEMAND_KEYS.terminal.key``:
    the client reads the body under the key it built itself and stores the
    value under that same name. A body that lacks the key is still reported
    ``applied``, so a mismatch is silent. The two halves are guarded together by
    ``test_app.py::TestTerminalWindow::test_the_real_client_applies_the_window_this_endpoint_serves``,
    which runs the real client over this endpoint's real body.
    """
    return f'TASKS_TERMINAL:{project}'


@router.get('/api/v2/dashboard/tasks')
async def api_tasks(request: Request) -> JSONResponse:
    """ACTIVE_TASKS (lock state surfaced via the scheduler endpoint — see /api/v2/dashboard/scheduler).

    Each task in ACTIVE_TASKS includes a ``meta_files`` field (taskmaster
    ``metadata.files``) that is retained on the wire for debugging and tooling.
    No frontend UI reads it directly — lock display routes through D.SCHEDULER.

    **Four distinct failure facts (plus a denominator), deliberately not
    collapsed:**

    - ``TASKS_OFFLINE`` — NO root produced rows and at least one root
      DEMONSTRABLY failed. One fused-memory URL serves every root, so that is
      the observable proxy for "fused-memory itself is unreachable", and it is
      the only state the global banner's copy ("fused-memory offline — task
      data unavailable") actually describes.

      The demonstrably-failed conjunct is what keeps a pure budget expiry
      (every root merely degraded, nothing proven down) from claiming an
      outage. The no-root-succeeded conjunct is why the test is *not* the
      tighter ``len(offline) == total_roots``: the handler's own budget caps
      how many roots can even reach the offline state. In the hang case each
      root burns up to ``_TASKS_PER_PROJECT_BUDGET`` before ``wait_for`` cuts
      it, and a cut root lands in ``degraded``, not ``offline`` — so with
      ``_TASKS_TOTAL_BUDGET / _TASKS_PER_PROJECT_BUDGET`` under three, at most
      a couple of roots per render can ever be marked offline. Requiring ALL
      of them to be would have made this flag unreachable on a nine-root
      config for the most likely total outage, leaving the payload to say
      "unavailable for 2 of 9" plus "timed out for 7 of 9" and never the
      thing that was actually true.
    - ``TASKS_OFFLINE_PROJECTS`` — the roots whose fetch DEMONSTRABLY failed.
      Non-empty with ``TASKS_OFFLINE`` false is the normal partial case.
    - ``TASKS_COUNT_UNKNOWN_PROJECTS`` — roots whose ACTIVE rows loaded fine
      but whose compact status map did not, so the census is UNKNOWN. Neither
      offline nor degraded: without a list of their own they would render as a
      healthy project with a confident "0 done".
    - ``TASKS_DEGRADED_PROJECTS`` — roots the handler ran out of budget for
      (see ``collect_tasks_with_counts``). Their state is UNKNOWN, not bad:
      nothing was proven unreachable, so degradation ALONE never raises the
      offline flag, not even when every root degrades. It can only ever fail
      to VETO the flag, alongside a root that did demonstrably fail.
    - ``TASKS_PROJECT_COUNT`` — N: how many roots were fanned out over. The
      banner's "k of N" phrasing needs a denominator drawn from the SAME
      population as its numerator, and the client's only other candidate
      (``PROJECTS``, from /api/v2/dashboard/orchestrators) is a different one —
      a root with no orchestrator, or an orchestrator with no task root, makes
      the two diverge and the notice understate the outage. The handler must
      compute this anyway to decide ``TASKS_OFFLINE``, so emitting it costs
      nothing and removes a client-side re-derivation that could drift.

    ``TASKS_OFFLINE`` used to be ``bool(offline_projects)``. That is what made
    the banner claim a total outage over eight healthy projects' rows carried
    in the very same payload — one unreachable root out of nine was enough.
    Collapsing any of these four into the others reintroduces that lie.

    **All three lists are DERIVED, in one pass, from the per-root units.** The
    collector returns a ``TaskSnapshot`` per root and ``task_snapshot.classify``
    routes each to exactly one banner from its STRUCTURED failure kind and its
    census's state — never from ``Datum.reason``, which is by contract the
    producer's verbatim failure text, so keying on it would be an ad-hoc parser
    over a meaningful string and one reworded fan-out message would silently
    move a project between banners. Because the lists and ``TASKS_SNAPSHOT``
    are read off the same records, a project cannot be named offline beside an
    entry claiming its census was measured.

    ``DONE_COUNTS`` is GONE, replaced by ``TASKS_SNAPSHOT[p].census`` — a
    ``Datum`` that says when the count was measured and, when it was not, why.
    The key is ABSENT rather than empty: ``data.js::applyKey`` returns early on
    a missing key, so each client surface keeps its seeded default, whereas
    ``{}`` would read as a measured "no project has any done tasks".

    **``?terminal=<project>`` — terminal rows, on request only.** The default
    render stopped fetching them, so its cost no longer grows with the terminal
    tree at all; a client that wants them asks, and gets ONE ``Datum`` in the
    ``lower_bound`` state under the flat top-level key
    ``TASKS_TERMINAL:<project>`` (``_terminal_key``). The key is flat because
    the client registry applies body keys VERBATIM: the name it reads from the
    body is the name it stores under, so a per-project key cannot be nested
    inside a ``TASKS_TERMINAL`` map. Absent the parameter no terminal key is
    present at all, for the same reason ``DONE_COUNTS`` is absent.

    That ``lower_bound`` state IS the disclosure PRD decision 5 substitutes for
    the retired live-PRD terminal-member exemption. Task 4416 asked how a tab
    showing only a window of terminal rows can honour "all live members of a
    live PRD are shown"; the answer adopted is option (a) — live members inside
    the fetched window — with the under-count DISCLOSED rather than sanctioned
    silently. The exemption machinery that used to widen the window per-PRD is
    gone with it: its own constant comment recorded that it could only ever
    exempt rows that were FETCHED, so it never met that contract either. The
    CLIENT half — reading the state and rendering the disclosure — belongs to
    leaf γ3, which migrates the Tasks tab.
    """
    config = request.app.state.config
    http_client = request.app.state.http_client
    # The one request-scoped input this handler takes. Absent is not the same
    # as empty: no parameter means no terminal read and no terminal key.
    terminal = request.query_params.get('terminal')
    # TWO instants, in this order. render_at is the producer's: it stamps what
    # this render measures and ages its rows, one clock read for all of them.
    # served_at is resolved only AFTER the fan-out returns, and every Datum on
    # the payload is validated against it.
    #
    # The order is what makes validate_datum's refusal of a NEGATIVE age hold
    # by construction. The unit cache is shared with every other render and
    # with the scheduler route, and a unit another request refreshed carries
    # THAT request's instant, which can be later than this render's own
    # render_at. But any unit this render read was stored before it read it,
    # and so stamped before served_at. Validating against render_at instead
    # made a healthy root read as a contract break: _validated swapped it for
    # an unknown unit and classify listed it in TASKS_OFFLINE_PROJECTS.
    #
    # The fan-out spends TWO bounded operations per project: one `statuses`-
    # narrowed active fetch and one compact get_statuses walk, concurrent and
    # under one TTL. The third slot in task_snapshot.PER_PROJECT_MCP_CALLS is
    # the terminal window, which only the `?terminal=` request spends.
    render_at = resolve_now(None)
    active, snapshots = await collect_tasks_with_counts(
        http_client, config, resolve_external=True, now=render_at,
    )
    served_at = resolve_now(None)
    offline_projects: list[str] = []
    degraded_projects: list[str] = []
    count_unknown_projects: list[str] = []
    wire_snapshots: dict[str, dict] = {}
    banners = {
        SnapshotHealth.OFFLINE: offline_projects,
        SnapshotHealth.DEGRADED: degraded_projects,
        SnapshotHealth.COUNT_UNKNOWN: count_unknown_projects,
    }
    for label, acquired in snapshots.items():
        snapshot = _validated(label, as_served(acquired, served_at), served_at)
        wire_snapshots[label] = snapshot.to_wire()
        # SnapshotHealth.OK routes nowhere, which is the whole point: a healthy
        # root is named in no banner list.
        banner = banners.get(classify(snapshot))
        if banner is not None:
            banner.append(label)
    # Same enumerator the collector walks, so N here is the same N it fanned
    # out over. ``bool(total_roots)`` guards the degenerate no-roots config:
    # 0 == 0 would otherwise declare an outage with nothing configured to fail.
    total_roots = len(_all_project_roots(config))
    # "No root succeeded" — the three lists are disjoint by construction (each
    # root appends to exactly one of them, then ``continue``s), so a root that
    # is in neither of these two either produced rows or produced rows with an
    # unknown count; both veto the flag. A set, not a sum, so a duplicate
    # label can only ever UNDERcount and fail safe (flag stays False).
    no_rows_anywhere = (
        len(set(offline_projects) | set(degraded_projects)) == total_roots
    )
    payload: dict[str, object] = {}
    if terminal is not None:
        window = await _terminal_window(
            http_client, config, terminal, snapshots, served_at=served_at,
        )
        # Validated here like every other emitted datum, and by the same rule:
        # a break is a BUG in this layer, so it degrades this one key rather
        # than 500-ing a payload whose other facts are fine.
        try:
            validate_datum(window, served_at)
        except DatumContractError as broken:
            logger.warning(
                'terminal window for %s breaks the %s datum invariant and is '
                'served as UNKNOWN; this is a BUG, not an outage',
                terminal, broken.invariant.value, exc_info=True,
            )
            window = Datum(
                None, None, DatumState.UNKNOWN,
                f'terminal window broke the {broken.invariant.value} '
                f'invariant: {broken}',
                FRESHNESS_BOUND_SECONDS,
            )
        payload[_terminal_key(terminal)] = window.to_wire()
    # ...and the same N goes on the wire as TASKS_PROJECT_COUNT, so the banner
    # denominates over the population its numerator is drawn from.
    return JSONResponse(
        {
            **payload,
            'ACTIVE_TASKS': active,
            'TASKS_SNAPSHOT': wire_snapshots,
            'TASKS_OFFLINE': (
                bool(total_roots) and bool(offline_projects) and no_rows_anywhere
            ),
            'TASKS_OFFLINE_PROJECTS': offline_projects,
            'TASKS_DEGRADED_PROJECTS': degraded_projects,
            'TASKS_COUNT_UNKNOWN_PROJECTS': count_unknown_projects,
            'TASKS_PROJECT_COUNT': total_roots,
            'served_at': served_at.isoformat(),
        }
    )
