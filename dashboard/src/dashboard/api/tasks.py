"""`/api/v2/dashboard/tasks` — the active-task table and its done counts.

Fans out over every known project root, reporting four distinct failure
facts separately (offline / degraded / count-unknown / no rows anywhere)
rather than collapsing them into one "empty" — see the handler docstring
for why each is its own signal.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import (
    _MAX_CANCELLED_PER_PROJECT,
    _MAX_DONE_PER_PROJECT,
    _all_project_roots,
    collect_tasks_with_counts,
)

router = APIRouter()


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
      but whose compact status map did not, so the done count is UNKNOWN and
      the terminal window was skipped. Neither offline nor degraded: without
      a list of their own they would render as a healthy project with a
      confident "0 done".
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
    """
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    # Bounded fan-out, two-to-three MCP calls per project: one `statuses`-
    # narrowed active fetch and one compact get_statuses map (concurrent),
    # plus a page_size/offset window over terminal rows only when a terminal
    # cap is actually requested. See _shape_one_project's docstring for why
    # each call is needed and what the window's disclosed narrowing costs.
    #
    # This comment used to read "single-pass: fetch_tasks once per project ...
    # no second fetch_statuses round-trip". That described the unnarrowed
    # whole-tree fetch this handler was changed to stop issuing, and
    # fetch_statuses is now exactly the second round-trip it denied.
    (
        active, offline_projects, done_counts,
        degraded_projects, count_unknown_projects,
    ) = await collect_tasks_with_counts(
        http_client, config,
        max_done_per_project=_MAX_DONE_PER_PROJECT,
        max_cancelled_per_project=_MAX_CANCELLED_PER_PROJECT,
        resolve_external=True,
    )
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
    # ...and the same N goes on the wire as TASKS_PROJECT_COUNT, so the banner
    # denominates over the population its numerator is drawn from.
    return JSONResponse(
        {
            'ACTIVE_TASKS': active,
            'TASKS_OFFLINE': (
                bool(total_roots) and bool(offline_projects) and no_rows_anywhere
            ),
            'TASKS_OFFLINE_PROJECTS': offline_projects,
            'TASKS_DEGRADED_PROJECTS': degraded_projects,
            'TASKS_COUNT_UNKNOWN_PROJECTS': count_unknown_projects,
            'TASKS_PROJECT_COUNT': total_roots,
            'DONE_COUNTS': done_counts,
        }
    )
