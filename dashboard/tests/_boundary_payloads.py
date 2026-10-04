"""The served bodies the boundary suite carries across the wire.

Each scenario drives a REAL dashboard route over a fixture substrate and keeps
the body that route served. ``test_boundary_js.py`` asserts the server-side
rows on those bodies and writes them where ``js/boundary_*.test.mjs`` read
them, so the client half never applies a hand-built payload.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from _canned_mcp import CannedMCP, _raw_row
from fastapi.testclient import TestClient
from shared.task_statuses import TaskStatus

import dashboard.data.active_tasks as active_tasks_mod
import dashboard.data.task_snapshot as snapshot_mod
from dashboard.app import app
from dashboard.config import DashboardConfig

T0 = datetime(2026, 10, 4, 9, 0, 0, tzinfo=UTC)
"""The instant every scenario's first render is served at."""

TASKS = '/api/v2/dashboard/tasks'

# in_flight 43 (running 25), backlog 1310, terminal 4106: total 5459.
DARK_FACTORY_COUNTS: Mapping[TaskStatus, int] = {
    TaskStatus.IN_PROGRESS: 25,
    TaskStatus.BLOCKED: 10,
    TaskStatus.REVIEW: 4,
    TaskStatus.MERGE_DEFERRED: 2,
    TaskStatus.INFRA_HOLD: 2,
    TaskStatus.PENDING: 1200,
    TaskStatus.DEFERRED: 110,
    TaskStatus.DONE: 4000,
    TaskStatus.CANCELLED: 106,
}

REIFY_COUNTS: Mapping[TaskStatus, int] = {
    TaskStatus.IN_PROGRESS: 2,
    TaskStatus.PENDING: 3,
    TaskStatus.DONE: 7,
}

NINE_MEMBERS: Mapping[TaskStatus, int] = {status: 1 for status in TaskStatus}


def _substrate(counts: Mapping[TaskStatus, int]) -> CannedMCP:
    """One project's task tree: *counts* tasks per status, ids ascending from 1."""
    statuses = [status for status, n in counts.items() for _ in range(n)]
    pairs = list(enumerate(statuses, start=1))
    return CannedMCP(
        rows=[_raw_row(task_id, status.value) for task_id, status in pairs],
        status_map={task_id: status.value for task_id, status in pairs},
        status_page_size=2000,
    )


def _fail_status_map(substrate: CannedMCP) -> None:
    """Every later ``get_statuses`` on *substrate* raises ``httpx.ReadTimeout``."""
    substrate.fail_when = lambda call: call['tool'] == 'get_statuses'


def _heal(substrate: CannedMCP) -> None:
    substrate.fail_when = lambda call: False


class _ByRoot:
    """One canned substrate per project root, routed by the read's ``project_root``."""

    def __init__(self, by_label: Mapping[str, CannedMCP]) -> None:
        self._by_label = by_label

    async def __call__(self, client, url, tool, args, **kwargs):
        substrate = self._by_label[Path(args['project_root']).name]
        return await substrate(client, url, tool, args, **kwargs)


@contextmanager
def _tasks_route(
    client: TestClient, workdir: Path, substrates: Mapping[str, CannedMCP],
) -> Iterator[Callable[[datetime], dict[str, Any]]]:
    """GET /tasks over one tmp root per substrate, served at the instant the caller names.

    The unit TTL is zero, so every render reads the substrate and a failure
    lands on the very next render; the last good store is what survives.
    """
    roots = [workdir / label for label in substrates]
    for root in roots:
        root.mkdir(parents=True)
    app.state.config = DashboardConfig(project_root=roots[0], known_project_roots=roots[1:])
    snapshot_mod._snapshot_cache_clear()
    active_tasks_mod._reset_root_rotation()

    def render(at: datetime) -> dict[str, Any]:
        with patch('dashboard.api.tasks.resolve_now', new=lambda _now: at):
            resp = client.get(TASKS)
        assert resp.status_code == 200, resp.text
        return resp.json()

    try:
        with (
            patch('dashboard.data.tasks.mcp_tool_call', new=_ByRoot(substrates)),
            patch('dashboard.data.active_tasks.fetch_task_runtime', new=AsyncMock(return_value={})),
            patch.object(snapshot_mod, 'SNAPSHOT_TTL_SECONDS', 0.0),
        ):
            yield render
    finally:
        snapshot_mod._snapshot_cache_clear()


def _census_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """Sketch #1-#5 and #13: the census every census surface reads."""
    bodies: dict[str, Any] = {}

    two_projects = {'dark-factory': _substrate(DARK_FACTORY_COUNTS), 'reify': _substrate(REIFY_COUNTS)}
    with _tasks_route(client, workdir / 'census_fresh', two_projects) as render:
        bodies['census_fresh'] = render(T0)

    with _tasks_route(client, workdir / 'census_nine', {'dark-factory': _substrate(NINE_MEMBERS)}) as render:
        bodies['census_nine'] = render(T0)

    cold = {'dark-factory': _substrate(DARK_FACTORY_COUNTS), 'reify': _substrate(REIFY_COUNTS)}
    _fail_status_map(cold['dark-factory'])
    with _tasks_route(client, workdir / 'census_unknown', cold) as render:
        bodies['census_unknown'] = render(T0)

    aging = _substrate(DARK_FACTORY_COUNTS)
    with _tasks_route(client, workdir / 'census_stale_3h', {'dark-factory': aging}) as render:
        render(T0)
        _fail_status_map(aging)
        bodies['census_stale_3h'] = render(T0 + timedelta(hours=3))

    transient = _substrate(DARK_FACTORY_COUNTS)
    with _tasks_route(client, workdir / 'census_transient', {'dark-factory': transient}) as render:
        bodies['census_transient_before'] = render(T0)
        _fail_status_map(transient)
        bodies['census_transient_during'] = render(T0 + timedelta(seconds=20))
        _heal(transient)
        bodies['census_transient_after'] = render(T0 + timedelta(seconds=40))

    return bodies


def build_all(workdir: Path) -> dict[str, Any]:
    """Every scenario's served body, keyed by the name the node half reads it under.

    *workdir* is an empty directory the scenarios root their substrates in.
    """
    with TestClient(app) as client:
        return {**_census_family(client, workdir)}
