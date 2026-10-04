"""The served bodies the boundary suite carries across the wire.

Each scenario drives a REAL dashboard route over a fixture substrate and keeps
the body that route served. ``test_boundary_js.py`` asserts the server-side
rows on those bodies and writes them where ``js/boundary_*.test.mjs`` read
them, so the client half never applies a hand-built payload.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
from _canned_mcp import CannedMCP, _raw_row
from _dt_helpers import make_fixed_datetime_cls
from fastapi.testclient import TestClient
from shared.task_statuses import TaskStatus

import dashboard.data.active_tasks as active_tasks_mod
import dashboard.data.task_snapshot as snapshot_mod
from dashboard.app import app
from dashboard.config import DashboardConfig
from dashboard.data.burndown import BURNDOWN_SCHEMA, collect_snapshot

T0 = datetime(2026, 10, 4, 9, 0, 0, tzinfo=UTC)
"""The instant every scenario's first render is served at."""

TASKS = '/api/v2/dashboard/tasks'
BURNDOWN = '/api/v2/dashboard/burndown'

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

# DARK_FACTORY_COUNTS once two pending tasks have started.
AFTER_T_COUNTS: Mapping[TaskStatus, int] = {
    **DARK_FACTORY_COUNTS,
    TaskStatus.IN_PROGRESS: DARK_FACTORY_COUNTS[TaskStatus.IN_PROGRESS] + 2,
    TaskStatus.PENDING: DARK_FACTORY_COUNTS[TaskStatus.PENDING] - 2,
}

RAGGED = 'burndown_ragged'
"""Two projects sampled daily for eight days; B's newest sample is a gap."""
RAGGED_T2 = T0
"""The ragged store's newest instant: A is measured at it, B is a gap row."""
RAGGED_T1 = RAGGED_T2 - timedelta(days=1)
"""B's last measured instant."""
RAGGED_DAYS = 8
RAGGED_CAP = 24
"""Both ragged projects' ``max_concurrent_tasks``."""


def ragged_day(index: int) -> datetime:
    """The instant of daily sample *index*: 7 is t2, 6 is t1."""
    return RAGGED_T2 - timedelta(days=RAGGED_DAYS - 1 - index)


def ragged_a(index: int) -> Mapping[TaskStatus, int]:
    """Project A's tree on day *index*: measured every day, never over its cap."""
    return {
        TaskStatus.IN_PROGRESS: 3,
        TaskStatus.BLOCKED: 1,
        TaskStatus.PENDING: 40 - 2 * index,
        TaskStatus.DONE: 10 + 2 * index,
    }


def ragged_b(index: int) -> Mapping[TaskStatus, int]:
    """Project B's tree on day *index*: 30 live against its cap at t1, 5 otherwise."""
    return {
        TaskStatus.IN_PROGRESS: 30 if ragged_day(index) == RAGGED_T1 else 5,
        TaskStatus.PENDING: 20 - index,
        TaskStatus.DONE: 100 + index,
    }


def _stock(
    substrate: CannedMCP, counts: Mapping[TaskStatus, int], *, live_at: datetime | None = None,
) -> CannedMCP:
    """Replace *substrate*'s task tree: *counts* tasks per status, ids ascending from 1.

    With *live_at*, every in-progress row carries a claimant heartbeating at
    that instant, so the sampler's strand split counts it live.
    """
    statuses = [status for status, n in counts.items() for _ in range(n)]
    pairs = list(enumerate(statuses, start=1))
    claimant = {} if live_at is None else {'claimant_run_id': 'run-boundary', 'heartbeat_at': live_at.isoformat()}
    substrate.rows = [
        _raw_row(task_id, status.value, **(claimant if status is TaskStatus.IN_PROGRESS else {}))
        for task_id, status in pairs
    ]
    substrate.status_map = {task_id: status.value for task_id, status in pairs}
    return substrate


def _substrate(counts: Mapping[TaskStatus, int], *, live_at: datetime | None = None) -> CannedMCP:
    """One project's task tree: *counts* tasks per status, ids ascending from 1."""
    return _stock(CannedMCP(status_page_size=2000), counts, live_at=live_at)


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

    The unit TTL is zero, so every acquisition, a render or a sampler tick,
    reads the substrate and a failure lands on the very next one; the last
    good store is what survives.
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


def _sample(client: TestClient, at: datetime) -> None:
    """One tick of the REAL burndown sampler at *at*, into the primary root's store.

    It runs on the app's own event loop, the one the routes run on, so the
    snapshot unit it shares with /tasks never crosses loops. It discovers no
    orchestrator: the sampled roots are exactly the configured ones.
    """
    assert client.portal is not None, 'the sampler runs inside a live TestClient'
    with (
        patch('dashboard.data.burndown.datetime', make_fixed_datetime_cls(at)),
        patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]),
    ):
        client.portal.call(_collect_into_store, app.state.config, app.state.http_client)


async def _collect_into_store(config: DashboardConfig, http_client: httpx.AsyncClient) -> None:
    config.burndown_db.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(config.burndown_db) as conn:
        await conn.executescript(BURNDOWN_SCHEMA)
        await conn.commit()
        await collect_snapshot(conn, config, http_client)


def _render_burndown(client: TestClient, at: datetime, window: str) -> dict[str, Any]:
    """GET /burndown?window=*window* on the real route, its single clock capture at *at*."""
    with patch('dashboard.api.burndown.datetime', make_fixed_datetime_cls(at)):
        resp = client.get(BURNDOWN, params={'window': window})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _cap_every_root(cap: int) -> None:
    """Give every configured root an orchestrator config capping it at *cap* concurrent tasks."""
    config: DashboardConfig = app.state.config
    for root in (config.project_root, *config.known_project_roots):
        (root / 'dark-factory-orchestrator.yaml').write_text(f'max_concurrent_tasks: {cap}\n')


def _burndown_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """Sketch #6 and #7: the series the real sampler persists, beside the census it sampled."""
    bodies: dict[str, Any] = {}

    tree = _substrate(DARK_FACTORY_COUNTS)
    with _tasks_route(client, workdir / 'census_at_t', {'dark-factory': tree}) as render:
        _sample(client, T0)
        bodies['census_at_t'] = render(T0)
        bodies['burndown_at_t'] = _render_burndown(client, T0, '24h')
        _stock(tree, AFTER_T_COUNTS)
        bodies['census_after_t'] = render(T0 + timedelta(minutes=10))

    a, b = CannedMCP(status_page_size=2000), CannedMCP(status_page_size=2000)
    with _tasks_route(client, workdir / RAGGED, {'A': a, 'B': b}):
        _cap_every_root(RAGGED_CAP)
        for index in range(RAGGED_DAYS):
            at = ragged_day(index)
            _stock(a, ragged_a(index), live_at=at)
            if at == RAGGED_T2:
                _fail_status_map(b)
            else:
                _stock(b, ragged_b(index), live_at=at)
            _sample(client, at)
        bodies[RAGGED] = _render_burndown(client, RAGGED_T2, '30d')

    return bodies


def ragged_store_rows(workdir: Path) -> list[dict[str, Any]]:
    """Every row the real sampler wrote in the ragged scenario, oldest first.

    Each row carries ``project``, the project's label, beside its stored columns.
    """
    store = DashboardConfig(project_root=workdir / RAGGED / 'A').burndown_db
    with closing(sqlite3.connect(f'file:{store}?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM snapshots ORDER BY ts, project_id').fetchall()
    return [{**dict(row), 'project': Path(row['project_id']).name} for row in rows]


def build_all(workdir: Path) -> dict[str, Any]:
    """Every scenario's served body, keyed by the name the node half reads it under.

    *workdir* is an empty directory the scenarios root their substrates in.
    """
    with TestClient(app) as client:
        return {**_census_family(client, workdir), **_burndown_family(client, workdir)}
