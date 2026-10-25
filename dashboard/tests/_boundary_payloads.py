"""The served bodies the boundary suite carries across the wire.

Each scenario drives a REAL dashboard route over a fixture substrate and keeps
the body that route served. ``test_boundary_js.py`` asserts the server-side
rows on those bodies and writes them where ``js/boundary_*.test.mjs`` read
them, so the client half never applies a hand-built payload.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
from _canned_mcp import CannedMCP, _raw_row
from _dt_helpers import make_fixed_datetime_cls
from escalation.archive import archive_dir_for_date
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from fastapi.testclient import TestClient
from shared.locking import files_to_modules
from shared.task_statuses import TaskStatus

import dashboard.data.active_tasks as active_tasks_mod
import dashboard.data.task_snapshot as snapshot_mod
from dashboard.api.escalations import _analytics_memo_clear
from dashboard.app import app
from dashboard.config import DashboardConfig
from dashboard.data import escalation_corpus
from dashboard.data.burndown import BURNDOWN_SCHEMA, collect_snapshot
from dashboard.data.scheduler import _scheduler_cache_clear

T0 = datetime(2026, 10, 4, 9, 0, 0, tzinfo=UTC)
"""The instant every scenario's first render is served at."""

TASKS = '/api/v2/dashboard/tasks'
BURNDOWN = '/api/v2/dashboard/burndown'
COSTS = '/api/v2/dashboard/costs'
MERGE_QUEUE = '/api/v2/dashboard/merge-queue'
ESCALATIONS = '/api/v2/dashboard/escalations'
ESCALATION_ANALYTICS = '/api/v2/dashboard/escalation-analytics'
MEMORY_GRAPHS = '/api/v2/dashboard/memory-graphs'
SCHEDULER = '/api/v2/dashboard/scheduler'

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


class MergeAttempt(NamedTuple):
    """One merge_attempt event, *age* before T0; a NULL or zero duration is untimed."""

    age: timedelta
    outcome: str
    duration_ms: int | None


MERGE_ATTEMPTS: tuple[MergeAttempt, ...] = (
    MergeAttempt(timedelta(hours=2), 'done', 1200),
    MergeAttempt(timedelta(hours=5), 'conflict', None),
    MergeAttempt(timedelta(hours=20), 'done', 900),
    MergeAttempt(timedelta(hours=30), 'done', None),
    MergeAttempt(timedelta(hours=50), 'blocked', 0),
    MergeAttempt(timedelta(hours=70), 'done', 3000),
)
"""One project's merges over three days: three inside 24h, all six inside 7d."""

# The orchestrator's runs.db events table, as orchestrator/src/orchestrator/
# event_store.py::_SCHEMA creates it; the dashboard does not depend on that package.
_EVENTS_DDL = """\
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);
"""


ESCALATIONS_FILED = 5
ESCALATIONS_ARCHIVED = 3
"""Pending L1 records filed through the real queue, and how many then sit in its archive."""

CORPUS_TTL_GENERATIONS: Mapping[str, float] = {'a': escalation_corpus.CORPUS_TTL_SECONDS, 'b': 3600.0}
"""Each corpus generation's TTL; the counts must not depend on it."""

MEMORY_OPS: tuple[tuple[str, str], ...] = (
    ('search', 'read'),
    ('search', 'read'),
    ('get_entity', 'read'),
    ('add_memory', 'write'),
    ('delete_memory', 'write'),
    ('compact', 'maintenance'),
)
"""(operation, kind) per journal row in the window: one kind is neither read nor write."""

_WRITE_OPS_DDL = """\
CREATE TABLE write_ops (
    id         TEXT PRIMARY KEY,
    operation  TEXT,
    project_id TEXT,
    agent_id   TEXT,
    kind       TEXT NOT NULL DEFAULT 'write',
    created_at TEXT NOT NULL
);
CREATE INDEX idx_wo_created ON write_ops(created_at);
"""
"""The columns and index the ops read touches of fused-memory's write_ops
(fused-memory/src/fused_memory/services/write_journal.py::SCHEMA_SQL)."""

OFFLINE_SCHEDULER = 'P'
"""The project whose scheduler fan-out fails; its task reads stay healthy."""
HELD = 'H'
"""The healthy project: task 1 holds :data:`LOCKED_MODULE` and live task 2 is parked on it."""
LOCKED_FILE = 'src/m/x.py'
LOCK_DEPTH = 2
LOCKED_MODULE = files_to_modules([LOCKED_FILE], LOCK_DEPTH)[0]


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


def _get(client: TestClient, path: str, **params: str) -> dict[str, Any]:
    resp = client.get(path, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


@contextmanager
def _canned_roots(
    client: TestClient, workdir: Path, substrates: Mapping[str, CannedMCP],
) -> Iterator[Callable[[datetime], dict[str, Any]]]:
    """Configure the app over one tmp root per substrate; yield GET /tasks at a named instant.

    Every task read under it, by any route or by the sampler, is answered by
    the substrate its root is labelled with. The unit TTL is zero, so every
    acquisition reads the substrate and a failure lands on the very next one;
    the last good store is what survives.
    """
    roots = [workdir / label for label in substrates]
    for root in roots:
        root.mkdir(parents=True)
    app.state.config = DashboardConfig(project_root=roots[0], known_project_roots=roots[1:])
    snapshot_mod._snapshot_cache_clear()
    active_tasks_mod._reset_root_rotation()

    def render(at: datetime) -> dict[str, Any]:
        with patch('dashboard.api.tasks.resolve_now', new=lambda _now: at):
            return _get(client, TASKS)

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
    with _canned_roots(client, workdir / 'census_fresh', two_projects) as render:
        bodies['census_fresh'] = render(T0)

    with _canned_roots(client, workdir / 'census_nine', {'dark-factory': _substrate(NINE_MEMBERS)}) as render:
        bodies['census_nine'] = render(T0)

    cold = {'dark-factory': _substrate(DARK_FACTORY_COUNTS), 'reify': _substrate(REIFY_COUNTS)}
    _fail_status_map(cold['dark-factory'])
    with _canned_roots(client, workdir / 'census_unknown', cold) as render:
        bodies['census_unknown'] = render(T0)

    aging = _substrate(DARK_FACTORY_COUNTS)
    with _canned_roots(client, workdir / 'census_stale_3h', {'dark-factory': aging}) as render:
        render(T0)
        _fail_status_map(aging)
        bodies['census_stale_3h'] = render(T0 + timedelta(hours=3))

    transient = _substrate(DARK_FACTORY_COUNTS)
    with _canned_roots(client, workdir / 'census_transient', {'dark-factory': transient}) as render:
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
        return _get(client, BURNDOWN, window=window)


def _cap_every_root(cap: int) -> None:
    """Give every configured root an orchestrator config capping it at *cap* concurrent tasks."""
    config: DashboardConfig = app.state.config
    for root in (config.project_root, *config.known_project_roots):
        (root / 'dark-factory-orchestrator.yaml').write_text(f'max_concurrent_tasks: {cap}\n')


def _burndown_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """Sketch #6 and #7: the series the real sampler persists, beside the census it sampled."""
    bodies: dict[str, Any] = {}

    tree = _substrate(DARK_FACTORY_COUNTS)
    with _canned_roots(client, workdir / 'census_at_t', {'dark-factory': tree}) as render:
        _sample(client, T0)
        bodies['census_at_t'] = render(T0)
        bodies['burndown_at_t'] = _render_burndown(client, T0, '24h')
        _stock(tree, AFTER_T_COUNTS)
        bodies['census_after_t'] = render(T0 + timedelta(minutes=10))

    a, b = CannedMCP(status_page_size=2000), CannedMCP(status_page_size=2000)
    with _canned_roots(client, workdir / RAGGED, {'A': a, 'B': b}):
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


def _write_merge_attempts(root: Path) -> None:
    """*root*'s runs.db, holding :data:`MERGE_ATTEMPTS` as task ids 1, 2, ..."""
    store = root / 'data' / 'orchestrator' / 'runs.db'
    store.parent.mkdir(parents=True)
    with closing(sqlite3.connect(store)) as conn:
        conn.executescript(_EVENTS_DDL)
        conn.executemany(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, data, duration_ms) '
            "VALUES (?, ?, ?, 'merge_attempt', 'merge', ?, ?)",
            [
                ((T0 - attempt.age).isoformat(), f'run-{task_id}', str(task_id),
                 json.dumps({'outcome': attempt.outcome}), attempt.duration_ms)
                for task_id, attempt in enumerate(MERGE_ATTEMPTS, start=1)
            ],
        )
        conn.commit()


def _window_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """Sketch #8 and #9: windowed routes, and the window each says it served."""
    bodies: dict[str, Any] = {}

    with _canned_roots(client, workdir / 'costs_90d', {'dark-factory': _substrate(REIFY_COUNTS)}):
        bodies['costs_90d'] = _get(client, COSTS, window='90d')

    with _canned_roots(client, workdir / 'merge', {'dark-factory': _substrate(REIFY_COUNTS)}):
        _write_merge_attempts(app.state.config.project_root)
        for window in ('7d', '24h'):
            with patch('dashboard.api.merge_queue.resolve_now', new=lambda _now: T0):
                bodies[f'merge_{window}'] = _get(client, MERGE_QUEUE, window=window)

    return bodies


def _clear_corpus() -> None:
    """Forget the corpus walk and the analytics derived from it, through their own hooks."""
    escalation_corpus._corpus_cache_clear()
    _analytics_memo_clear()


def _file_escalations(config: DashboardConfig) -> None:
    """File :data:`ESCALATIONS_FILED` pending records, then move the last few under the archive.

    A pending record in the archive is how a stranded one sits there: the
    live queue no longer lists it, the history still holds it open.
    """
    queue = EscalationQueue(config.escalations_dir)
    for n in range(1, ESCALATIONS_FILED + 1):
        queue.submit(Escalation(
            id=f'esc-b{n}-1',
            task_id=f'b{n}',
            agent_role='boundary-suite',
            severity='blocking',
            category='cleanup_needed',
            summary=f'boundary escalation {n}',
            timestamp=(T0 - timedelta(hours=n)).isoformat(),
            status='pending',
            level=1,
        ))
    archived = archive_dir_for_date(queue.queue_dir, T0.isoformat())
    archived.mkdir(parents=True, exist_ok=True)
    for n in range(ESCALATIONS_FILED - ESCALATIONS_ARCHIVED + 1, ESCALATIONS_FILED + 1):
        os.replace(queue.queue_dir / f'esc-b{n}-1.json', archived / f'esc-b{n}-1.json')
    config.reconciliation_escalations_dir.mkdir(parents=True, exist_ok=True)


def _write_memory_ops(config: DashboardConfig) -> None:
    """The write journal, holding :data:`MEMORY_OPS` inside the hour before T0."""
    journal = config.write_journal_db
    journal.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(journal)) as conn:
        conn.executescript(_WRITE_OPS_DDL)
        conn.executemany(
            'INSERT INTO write_ops (id, operation, project_id, agent_id, kind, created_at) '
            "VALUES (?, ?, 'dark_factory', 'boundary-suite', ?, ?)",
            [
                (f'op-{n}', operation, kind, (T0 - timedelta(minutes=n)).isoformat())
                for n, (operation, kind) in enumerate(MEMORY_OPS, start=1)
            ],
        )
        conn.commit()


@contextmanager
def _memory_graphs_at_t0() -> Iterator[None]:
    """Both clocks a /memory-graphs request reads, pinned to T0: the journal
    read's as_of and the route's served_at."""
    with (
        patch('dashboard.data.write_journal.resolve_now', new=lambda now: now or T0),
        patch('dashboard.app.resolve_now', new=lambda now: now or T0),
    ):
        yield


def _held_tree() -> CannedMCP:
    """H's two in-progress tasks, both footprinting :data:`LOCKED_FILE`."""
    footprint = {'metadata': {'files': [LOCKED_FILE]}}
    return CannedMCP(
        rows=[_raw_row(1, 'in-progress', **footprint), _raw_row(2, 'in-progress', **footprint)],
        status_map={1: 'in-progress', 2: 'in-progress'},
        status_page_size=2000,
    )


_HELD_SNAPSHOT: Mapping[str, Any] = {
    'lock_depth': LOCK_DEPTH,
    'current_holders': {LOCKED_MODULE: '1'},
    'park_stacks': {LOCKED_MODULE: [{'owner': '2', 'installed_at': T0.isoformat()}]},
    'parks': {'2': {'modules': [LOCKED_MODULE], 'installed_at': T0.isoformat()}},
}
"""H's scheduler: task 1 holds the module and task 2 is parked on it."""


async def _scheduler_mcp(_client, _url, tool, args):
    """The scheduler fan-out: unreachable for P, the held lock and no events for H."""
    if Path(args['project_root']).name == OFFLINE_SCHEDULER:
        raise httpx.ConnectError('canned scheduler unreachable')
    return dict(_HELD_SNAPSHOT) if tool == 'get_scheduler_state' else []


def _corpus_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """Sketch #10, #11 and #12: the escalation corpus, memory ops and locks."""
    bodies: dict[str, Any] = {}

    with _canned_roots(client, workdir / 'escalations', {'dark-factory': _substrate(REIFY_COUNTS)}):
        _file_escalations(app.state.config)
        for hours, (generation, ttl) in enumerate(CORPUS_TTL_GENERATIONS.items()):
            at = T0 + timedelta(hours=hours)
            _clear_corpus()
            with (
                patch.object(escalation_corpus, 'CORPUS_TTL_SECONDS', ttl),
                patch('dashboard.api.escalations.resolve_now', new=lambda _now, at=at: at),
            ):
                bodies[f'escalations_{generation}'] = _get(client, ESCALATIONS)
                bodies[f'analytics_{generation}'] = _get(client, ESCALATION_ANALYTICS)
        _clear_corpus()

    with _canned_roots(client, workdir / 'memory', {'dark-factory': _substrate(REIFY_COUNTS)}):
        _write_memory_ops(app.state.config)
        with _memory_graphs_at_t0():
            bodies['memory_ops'] = _get(client, MEMORY_GRAPHS)
    with (
        _canned_roots(client, workdir / 'memory_missing', {'dark-factory': _substrate(REIFY_COUNTS)}),
        _memory_graphs_at_t0(),
    ):
        bodies['memory_ops_missing'] = _get(client, MEMORY_GRAPHS)

    offline_tree = _substrate({TaskStatus.IN_PROGRESS: 3, TaskStatus.PENDING: 2})
    with _canned_roots(client, workdir / 'locks', {HELD: _held_tree(), OFFLINE_SCHEDULER: offline_tree}) as render:
        _scheduler_cache_clear()
        try:
            with patch('dashboard.data.scheduler.mcp_tool_call', new=_scheduler_mcp):
                bodies['scheduler_offline'] = _get(client, SCHEDULER)
        finally:
            _scheduler_cache_clear()
        bodies['tasks_with_p'] = render(T0)

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
        return {
            **_census_family(client, workdir),
            **_burndown_family(client, workdir),
            **_window_family(client, workdir),
            **_corpus_family(client, workdir),
        }
