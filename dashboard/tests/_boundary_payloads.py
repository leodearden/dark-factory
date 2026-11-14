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
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
from _canned_mcp import ByRoot, CannedMCP, _raw_row, is_metadata_projection
from _dt_helpers import fixed_clock, installed_clock
from _prd_lanes_tree import (
    CARRIER,
    LaneTask,
    Scenario,
    holder_processes,
    lanes_tree,
    prd_key,
    sidecar,
    tree_tasks,
)
from _prd_tree import prd_tree_tasks
from _quality_rows import (
    NO_COST,
    NO_OUTCOMES,
    epochs_from,
    history,
    train,
    write_landing_events,
    write_marker_state_db,
    write_quality_db,
)
from _runs_db_fixture import EVENTS_DDL, EventRow, runs_db_path, write_runs_db
from _served_reads import forget_escalation_corpus, forget_served_reads
from escalation.archive import archive_dir_for_date
from escalation.models import DeliveredCheckOwner, Escalation
from escalation.queue import EscalationQueue
from fastapi.testclient import TestClient
from shared.capability_manifest import sidecar_path
from shared.locking import files_to_modules
from shared.project_id import project_id_for_root
from shared.quality_series.git_lines import ModuleLines
from shared.quality_series.landings import GitCrossCheck, LinesBase
from shared.quality_series.rows import Cost, LandingRow, Outcomes
from shared.task_statuses import TaskStatus

import dashboard.data.task_snapshot as snapshot_mod
from dashboard.app import app
from dashboard.config import DashboardConfig
from dashboard.data import escalation_corpus
from dashboard.data.burndown import BURNDOWN_SCHEMA, collect_snapshot
from dashboard.data.quality_markers import Marker, MarkerKind, MarkerSource, SourceType
from dashboard.data.quality_sampler import QUALITY_SAMPLE_INTERVAL_SECONDS
from dashboard.data.quality_store import QualitySample
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
PRDS = '/api/v2/dashboard/prds'
QUALITY = '/api/v2/dashboard/quality'

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

ESCALATIONS_FILED = 5
ESCALATIONS_ARCHIVED = 3
"""Pending L1 records filed through the real queue, and how many then sit in its archive."""

CHECK_OWNER: DeliveredCheckOwner = {
    'task_id': '4',
    'check_name': 'boundary-capability',
    'sidecar': {'path': 'plans/boundary.capabilities.yaml', 'on_main': 'declares'},
}
"""The delivered_check_owner of the one dependency_capability record the check-owner scenario files."""

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

SCHEDULER_SNAPSHOT_AT = T0 - timedelta(hours=2)
"""When every canned root's scheduler last wrote its state file: hours before any read of it."""
CANNED_SCHEDULER_STATE: Mapping[str, Any] = {
    'lock_depth': LOCK_DEPTH,
    'current_holders': {LOCKED_MODULE: '1'},
    'snapshot_at': SCHEDULER_SNAPSHOT_AT.isoformat(),
}
"""The scheduler state every canned root serves unless its scenario names another."""


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


class _SchedulerByRoot:
    """The scheduler fan-out, one canned state per root label; a label mapped to ``None`` is
    unreachable. Every other root serves :data:`CANNED_SCHEDULER_STATE`, and no root has events."""

    def __init__(self, states: Mapping[str, Mapping[str, Any] | None]) -> None:
        self._states = states

    async def __call__(self, _client, _url, tool, args, **_kwargs):
        state = self._states.get(Path(args['project_root']).name, CANNED_SCHEDULER_STATE)
        if state is None:
            raise httpx.ConnectError('canned scheduler unreachable')
        return dict(state) if tool == 'get_scheduler_state' else []


def quiet_ledgers(root: Path) -> None:
    """An empty run ledger and an empty escalation queue: a project that has recorded nothing."""
    write_runs_db(root)
    (root / 'data' / 'escalations').mkdir(parents=True, exist_ok=True)


def _quietly(labels: Iterable[str]) -> dict[str, Callable[[Path], object]]:
    return dict.fromkeys(labels, quiet_ledgers)


def _get(client: TestClient, path: str, **params: str) -> dict[str, Any]:
    resp = client.get(path, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


@contextmanager
def _canned_roots(
    client: TestClient,
    workdir: Path,
    substrates: Mapping[str, CannedMCP],
    *,
    schedulers: Mapping[str, Mapping[str, Any] | None] | None = None,
    furnish: Mapping[str, Callable[[Path], object]] | None = None,
) -> Iterator[Callable[[datetime], dict[str, Any]]]:
    """Configure the app over one tmp root per substrate; yield GET /tasks at a named instant.

    Every task read under it, by any route or by the sampler, is answered by
    the substrate its root is labelled with, and every scheduler read by
    *schedulers* (see :class:`_SchedulerByRoot`), so no read reaches a live
    fused-memory. *furnish* lays out a labelled root's files before the first
    read. The unit TTL is zero, so every acquisition reads the substrate and a
    failure lands on the very next one; the last good store is what survives.
    """
    roots = [workdir / label for label in substrates]
    for root in roots:
        root.mkdir(parents=True)
        if furnish is not None and root.name in furnish:
            furnish[root.name](root)
    app.state.config = DashboardConfig(project_root=roots[0], known_project_roots=roots[1:])
    forget_served_reads()

    def render(at: datetime) -> dict[str, Any]:
        with installed_clock(app, fixed_clock(at)):
            return _get(client, TASKS)

    try:
        with (
            patch('dashboard.data.tasks.mcp_tool_call', new=ByRoot(substrates)),
            patch('dashboard.data.scheduler.mcp_tool_call', new=_SchedulerByRoot(schedulers or {})),
            patch('dashboard.data.active_tasks.fetch_task_runtime', new=AsyncMock(return_value={})),
            patch.object(snapshot_mod, 'SNAPSHOT_TTL_SECONDS', 0.0),
        ):
            yield render
    finally:
        forget_served_reads()


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
    with patch('dashboard.data.burndown.find_running_orchestrators', return_value=[]):
        client.portal.call(_collect_into_store, app.state.config, app.state.http_client, at)


async def _collect_into_store(
    config: DashboardConfig, http_client: httpx.AsyncClient, at: datetime,
) -> None:
    config.burndown_db.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(config.burndown_db) as conn:
        await conn.executescript(BURNDOWN_SCHEMA)
        await conn.commit()
        await collect_snapshot(conn, config, http_client, now=at)


def _render_burndown(client: TestClient, at: datetime, window: str) -> dict[str, Any]:
    """GET /burndown?window=*window* on the real route, its single clock capture at *at*."""
    with installed_clock(app, fixed_clock(at)):
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
    store = runs_db_path(root)
    store.parent.mkdir(parents=True)
    with closing(sqlite3.connect(store)) as conn:
        conn.executescript(EVENTS_DDL)
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
            with installed_clock(app, fixed_clock(T0)):
                bodies[f'merge_{window}'] = _get(client, MERGE_QUEUE, window=window)

    return bodies


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
            forget_escalation_corpus()
            with (
                patch.object(escalation_corpus, 'CORPUS_TTL_SECONDS', ttl),
                installed_clock(app, fixed_clock(at)),
            ):
                bodies[f'escalations_{generation}'] = _get(client, ESCALATIONS)
                bodies[f'analytics_{generation}'] = _get(client, ESCALATION_ANALYTICS)
        forget_escalation_corpus()

    with _canned_roots(client, workdir / 'memory', {'dark-factory': _substrate(REIFY_COUNTS)}):
        _write_memory_ops(app.state.config)
        with installed_clock(app, fixed_clock(T0)):
            bodies['memory_ops'] = _get(client, MEMORY_GRAPHS)
    with (
        _canned_roots(client, workdir / 'memory_missing', {'dark-factory': _substrate(REIFY_COUNTS)}),
        installed_clock(app, fixed_clock(T0)),
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


def _check_owner_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """An /escalations body whose one row carries a delivered_check_owner (task 7725).

    The record is filed as orchestrator/src/orchestrator/harness.py files a
    failing delivered check, through the real queue, so the render tests'
    owner-carrying row is checked against a shape the route really serves.
    """
    with _canned_roots(client, workdir / 'check_owner', {'dark-factory': _substrate(REIFY_COUNTS)}):
        forget_escalation_corpus()
        config = app.state.config
        queue = EscalationQueue(config.escalations_dir)
        queue.submit(Escalation(
            id=queue.make_id('3'),
            task_id='3',
            agent_role='orchestrator-scheduler',
            severity='critical',
            category='dependency_capability',
            summary='delivered check boundary-capability fails on main',
            detail='the owner block the harness renders',
            suggested_action='manual_intervention',
            timestamp=T0.isoformat(),
            status='pending',
            level=2,
            delivered_check_owner=CHECK_OWNER,
        ))
        config.reconciliation_escalations_dir.mkdir(parents=True, exist_ok=True)
        try:
            with installed_clock(app, fixed_clock(T0)):
                return {'escalations_check_owner': _get(client, ESCALATIONS)}
        finally:
            forget_escalation_corpus()


PRD_C = 'plans/c-prd.md'
OVERLAY_ROW = 11
ORPHAN = 12


def prd_tree() -> CannedMCP:
    """``_prd_tree.py``'s fixture tree at T0, served raw as one project's substrate."""
    rows = [
        _raw_row(
            task_id, task.status, metadata=dict(task.metadata),
            **({} if task.priority is None else {'priority': task.priority}),
        )
        for task_id, task in prd_tree_tasks(T0).items()
    ]
    return CannedMCP(rows, {int(row['id']): row['status'] for row in rows}, status_page_size=2000)


def _grow_past_lineage(substrate: CannedMCP) -> None:
    """A new active row naming its own PRD, and an id only the status map knows."""
    substrate.rows.append(_raw_row(OVERLAY_ROW, 'pending', priority='medium', metadata={'prd_path': PRD_C}))
    substrate.status_map[OVERLAY_ROW] = 'pending'
    substrate.status_map[ORPHAN] = 'pending'


def _render_prds(client: TestClient, at: datetime) -> dict[str, Any]:
    with installed_clock(app, fixed_clock(at)):
        return _get(client, PRDS)


@contextmanager
def canned_prd_roots(
    client: TestClient,
    workdir: Path,
    substrates: Mapping[str, CannedMCP],
    *,
    schedulers: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> Iterator[Callable[[datetime], dict[str, Any]]]:
    """:func:`_canned_roots` over quiet roots (:func:`quiet_ledgers`), yielding GET /prds at a
    named instant."""
    with _canned_roots(
        client, workdir, substrates, schedulers=schedulers, furnish=_quietly(substrates),
    ):
        yield lambda at: _render_prds(client, at)


def prd_view_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """Boundary rows 8, 9 and 12: the PRD view over the census join.

    ``prd_view_map_failed_tasks`` is the /tasks body served at the same
    instant as ``prd_view_map_failed``, so its census reason is the one the
    failed map half carried.
    """
    bodies: dict[str, Any] = {}

    two_projects = {'dark-factory': prd_tree(), 'reify': _substrate(REIFY_COUNTS)}
    with _canned_roots(
        client, workdir / 'prd_view_fresh', two_projects, furnish=_quietly(two_projects),
    ):
        bodies['prd_view_fresh'] = _render_prds(client, T0)

    growing = prd_tree()
    with _canned_roots(
        client, workdir / 'prd_view_overlay', {'dark-factory': growing},
        furnish=_quietly(['dark-factory']),
    ):
        _render_prds(client, T0)
        _grow_past_lineage(growing)
        bodies['prd_view_overlay'] = _render_prds(client, T0 + timedelta(seconds=20))

    failing = prd_tree()
    with _canned_roots(
        client, workdir / 'prd_view_map_failed', {'dark-factory': failing},
        furnish=_quietly(['dark-factory']),
    ) as render_tasks:
        _render_prds(client, T0)
        _fail_status_map(failing)
        failed_at = T0 + timedelta(seconds=20)
        bodies['prd_view_map_failed'] = _render_prds(client, failed_at)
        bodies['prd_view_map_failed_tasks'] = render_tasks(failed_at)

    return bodies


LANES_ROOT = 'dark-factory'
"""The lanes family's root holding the lanes tree, its run ledger, queue, sidecars and PRD files."""
QUEUELESS_ROOT = 'reify'
"""The lanes family's second root: a run ledger but no escalation queue, so the corpus as a
whole is a lower bound while :data:`LANES_ROOT`'s own queue is read whole."""


def lane_title(key: str) -> str:
    """The H1 of the fixture PRD file at *key*."""
    return f'Lanes fixture {key}'


def lanes_substrate(tree: Mapping[str, Scenario]) -> CannedMCP:
    """*tree*'s tasks, the outside tasks included, served raw as one project's substrate."""
    tasks = tree_tasks(tree)
    rows = [
        _raw_row(
            task_id, task.status, priority=task.priority,
            dependencies=list(task.dependencies), metadata=dict(task.metadata),
        )
        for task_id, task in tasks.items()
    ]
    return CannedMCP(rows, {task_id: task.status for task_id, task in tasks.items()}, status_page_size=2000)


def _lane_events(tree: Mapping[str, Scenario]) -> list[EventRow]:
    events: list[EventRow] = []
    for scenario in tree.values():
        events.extend((at, str(task_id), 'task_started', {}) for task_id, at in scenario.started.items())
        events.extend(
            (at, str(task_id), 'task_completed', {'outcome': 'done'})
            for task_id, at in scenario.completed.items()
        )
    return events


def _file_lane_escalations(root: Path, tree: Mapping[str, Scenario]) -> None:
    queue = EscalationQueue(root / 'data' / 'escalations')
    for scenario in tree.values():
        for filed in scenario.escalations:
            queue.submit(Escalation(
                id=filed.escalation_id,
                task_id=str(filed.task_id),
                agent_role='boundary-suite',
                severity='blocking',
                category='cleanup_needed',
                summary=f'lanes escalation on #{filed.task_id}',
                timestamp=(T0 - timedelta(hours=1)).isoformat(),
                status='pending',
                level=filed.level,
            ))


def _furnish_lanes(
    tree: Mapping[str, Scenario], *, ledger_readable: bool = True, queue: bool = True,
) -> Callable[[Path], None]:
    """Lay out :data:`LANES_ROOT`: its run ledger (or bytes that are no database), its escalation
    queue unless *queue* is False, and each scenario's PRD file and sidecar."""

    def furnish(root: Path) -> None:
        if ledger_readable:
            write_runs_db(root, _lane_events(tree))
        else:
            runs_db_path(root).parent.mkdir(parents=True)
            runs_db_path(root).write_bytes(b'this is not a sqlite database\n' * 64)
        if queue:
            _file_lane_escalations(root, tree)
        for scenario in tree.values():
            prd_file = root / scenario.key
            prd_file.parent.mkdir(parents=True, exist_ok=True)
            prd_file.write_text(f'# {lane_title(scenario.key)}\n\nA lanes fixture PRD.\n')
            if (raw := sidecar(scenario)) is not None:
                (root / sidecar_path(scenario.key)).write_text(json.dumps(raw))

    return furnish


def _lineage_read(call: dict) -> bool:
    return call['tool'] == 'get_tasks' and is_metadata_projection(call['args'])


CONTENDED_IDS = (950, 951, 952)
"""Ready high-priority tasks outside the lanes tree, each footprinting :data:`LOCKED_FILE`."""
_CARRIER_HOLDS_LOCKED_MODULE: Mapping[str, Any] = {
    **CANNED_SCHEDULER_STATE,
    'current_holders': {LOCKED_MODULE: str(CARRIER)},
}
"""A scheduler whose :data:`LOCKED_MODULE` is held by :data:`CARRIER`, a pending task the store holds."""


def _contend(substrate: CannedMCP) -> None:
    """Add :data:`CONTENDED_IDS` to *substrate*: pending, dependency-free, high priority, each
    naming :data:`LOCKED_FILE`, so the contention line's ``top`` has a key ready tasks want."""
    for task_id in CONTENDED_IDS:
        substrate.rows.append(
            _raw_row(task_id, TaskStatus.PENDING.value, priority='high', metadata={'files': [LOCKED_FILE]}),
        )
        substrate.status_map[task_id] = TaskStatus.PENDING.value


SECOND_TAIL = 'second_tail'
"""A second delivered-signal PRD beside the lanes tree's one, so the collapsed Tail line lists
more than one title (boundary row 20)."""


def _second_tail() -> Scenario:
    """Its signal leaf 961 is done and its one follow-up, 962, is open."""
    key = prd_key(SECOND_TAIL)
    return Scenario(
        {961: LaneTask('done', {'prd_path': key}), 962: LaneTask('pending', {'spawned_from': 961})},
        signal_leaves={'gate': 961},
        key=key,
    )


def prd_lanes_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """Boundary rows 13–21 end to end: the lanes tree served through every γ2 input.

    :data:`LANES_ROOT` serves the lanes tree plus :data:`SECOND_TAIL` with its
    run ledger, escalation queue, sidecars and PRD files, and :data:`QUEUELESS_ROOT` a plain tree with
    a run ledger and no queue. The live session holder is a real process alive
    while every body is served. ``prd_lanes_tasks`` is the /tasks body served
    over the same substrate at the same instant as ``prd_lanes``.
    ``prd_lanes_scheduler_offline`` has an unreachable scheduler on
    :data:`LANES_ROOT`; ``prd_lanes_runs_db_unreadable`` a run ledger that is no
    database; ``prd_lanes_lineage_failed`` a lineage read that fails;
    ``prd_lanes_contended`` adds :data:`CONTENDED_IDS` under a scheduler whose
    lock :data:`CARRIER` holds; ``prd_lanes_queueless`` lays out no escalation
    queue, so its view is a lower bound that still draws every row.
    """
    bodies: dict[str, Any] = {}
    with holder_processes() as (live, dead):
        tree = {**lanes_tree(T0, live_holder=live, dead_holder=dead), SECOND_TAIL: _second_tail()}

        def serve(
            name: str,
            *,
            schedulers: Mapping[str, Mapping[str, Any] | None] | None = None,
            ledger_readable: bool = True,
            lineage_fails: bool = False,
            contended: bool = False,
            tasks_body: str | None = None,
            queue: bool = True,
        ) -> None:
            substrate = lanes_substrate(tree)
            if lineage_fails:
                substrate.fail_when = _lineage_read
            if contended:
                _contend(substrate)
            substrates = {LANES_ROOT: substrate, QUEUELESS_ROOT: _substrate(REIFY_COUNTS)}
            furnish = {
                LANES_ROOT: _furnish_lanes(tree, ledger_readable=ledger_readable, queue=queue),
                QUEUELESS_ROOT: write_runs_db,
            }
            with _canned_roots(
                client, workdir / name, substrates, schedulers=schedulers, furnish=furnish,
            ) as render_tasks:
                bodies[name] = _render_prds(client, T0)
                if tasks_body is not None:
                    bodies[tasks_body] = render_tasks(T0)

        serve('prd_lanes', tasks_body='prd_lanes_tasks')
        serve('prd_lanes_scheduler_offline', schedulers={LANES_ROOT: None})
        serve('prd_lanes_runs_db_unreadable', ledger_readable=False)
        serve('prd_lanes_lineage_failed', lineage_fails=True)
        serve(
            'prd_lanes_contended', contended=True,
            schedulers={LANES_ROOT: _CARRIER_HOLDS_LOCKED_MODULE},
        )
        serve('prd_lanes_queueless', queue=False)
    return bodies


QUALITY_AT = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
"""The instant the quality scenarios are read at; their stored landings end a day before it."""
QUALITY_WINDOW = 100
QUALITY_STORED = 150
"""Landings a full quality scenario stores: past the window and past a baseline's 100."""
QUALITY_SMALL_N = 40
QUALITY_OFF_LINE = QUALITY_STORED - 3
QUALITY_LATE_CLAIM = QUALITY_STORED - 2
"""The ordinals of the fresh scenario's landing off main's first parent and of its late claim,
which re-claims the sha landed :data:`QUALITY_RECLAIMED_BACK` units before it."""
QUALITY_RECLAIMED_BACK = 3
QUALITY_GIT_CHECK = GitCrossCheck(off_first_parent=1, merges_without_landing=1, late_claims=1)
"""The fresh scenario's git cross-check: its two landings off the line, and a merge commit
runs.db records no landing for."""
QUALITY_SOURCE_EPOCH = datetime(2026, 10, 21, tzinfo=UTC)
"""review_completed's first row in the before-epoch scenario, after every landing it stores."""
QUALITY_PENDING_TRAIN = ('7001', '7002', '7003')
"""A train runs.db records after the main-ahead scenario's last tick, which quality.db lacks."""
QUALITY_TICK = timedelta(seconds=QUALITY_SAMPLE_INTERVAL_SECONDS)
QUALITY_MODULE = 'dashboard'
QUALITY_MODULE_ORDINALS = tuple(range(0, QUALITY_STORED, 3))
"""The module the drill-down scenario is read for, and the stored landings touching it."""
QUALITY_SHIFT = 300
"""The landing at which the alarm scenarios' rejection rate jumps from 3 in 10 to every landing:
past the 200 a frozen baseline needs before the marker nearest it."""
QUALITY_SHIFTED = 15
"""Landings the alarm scenarios store from the shift on: enough for the up-CUSUM to cross h, too
few to give the explaining marker's own segment a baseline of its own."""
_NO_DISAGREEMENT = GitCrossCheck(0, 0, 0)


def _quality_outcome(ordinal: int) -> Outcomes:
    """Every landing reviewed; every fifth rejected and every fourth blocked in review."""
    rejections = int(ordinal % 5 == 0)
    return replace(NO_OUTCOMES, rejected=rejections > 0, rejection_events=rejections,
                   reviewed=True, review_blocking=ordinal % 4 == 0)


def _quality_cost(ordinal: int) -> Cost:
    usd = 2.0 + (ordinal % 7) / 2
    return replace(NO_COST, all_role_usd=usd, by_role=(('implementer', usd),),
                   implementer_invocations=1 + ordinal % 3, turns=40 + ordinal % 11)


def _quality_history(
    n: int, *, cost: Callable[[int], Cost] = lambda ordinal: NO_COST,
    outcome: Callable[[int], Outcomes] = _quality_outcome,
    modules: Callable[[int], tuple[ModuleLines, ...]] = lambda ordinal: (),
) -> list[LandingRow]:
    """*n* landings two hours apart, the newest a day before :data:`QUALITY_AT`."""
    return history(n, start=QUALITY_AT - timedelta(days=1, hours=2 * (n - 1)),
                   outcome=outcome, cost=cost, modules=modules)


def _shifted_outcome(ordinal: int) -> Outcomes:
    """The first three of every ten landings rejected until :data:`QUALITY_SHIFT`, every one from it."""
    rejected = ordinal >= QUALITY_SHIFT or ordinal % 10 < 3
    return replace(NO_OUTCOMES, rejected=rejected, rejection_events=int(rejected))


_MODULE_TOUCHED = frozenset(QUALITY_MODULE_ORDINALS)


def _module_lines(ordinal: int) -> tuple[ModuleLines, ...]:
    module = QUALITY_MODULE if ordinal in _MODULE_TOUCHED else 'orchestrator'
    return (ModuleLines(module, 40),)


def _off_the_line(rows: list[LandingRow]) -> list[LandingRow]:
    """*rows* with :data:`QUALITY_OFF_LINE` off main's first parent and :data:`QUALITY_LATE_CLAIM`
    a late claim on an earlier landing's sha, as α records each."""
    served = list(rows)
    off = served[QUALITY_OFF_LINE]
    served[QUALITY_OFF_LINE] = replace(off, landing=replace(off.landing, git_first_parent=False))
    late = served[QUALITY_LATE_CLAIM]
    sha = served[QUALITY_LATE_CLAIM - QUALITY_RECLAIMED_BACK].landing.merge_sha
    served[QUALITY_LATE_CLAIM] = replace(late, landing=replace(
        late.landing, merge_sha=sha, unit_id=f'{sha}:{late.landing.task_id}', late_claim=True,
        lines_base=LinesBase.UNMEASURABLE, base_sha=None,
    ))
    return served


def _quality_root(workdir: Path) -> Path:
    """Configure the app over one root named dark_factory under *workdir*."""
    root = workdir / 'dark_factory'
    root.mkdir(parents=True)
    app.state.config = DashboardConfig(project_root=root)
    return root


def _store_quality(
    client: TestClient, root: Path, rows: list[LandingRow], *, sampled_at: datetime,
    git_check: GitCrossCheck = _NO_DISAGREEMENT,
) -> None:
    """Store *rows* in the app's quality.db with one value sample at *sampled_at* whose cursor is
    the newest row, as a sampler tick that measured them would."""
    sample = QualitySample.value(
        project_id_for_root(root), sampled_at, cursor_sha=rows[-1].landing.merge_sha,
        landings_sampled=len(rows), landings_deferred=0, git_check=git_check,
    )
    assert client.portal is not None, 'quality.db is written inside a live TestClient'
    client.portal.call(write_quality_db, app.state.config.quality_db, rows, [sample])


def _render_quality(client: TestClient, at: datetime, **module: str) -> dict[str, Any]:
    """GET /quality?project=dark_factory&window=100 on the real route, its clock at *at*; a
    ``module=`` keyword drills down to that module."""
    with installed_clock(app, fixed_clock(at)):
        return _get(client, QUALITY, project='dark_factory', window=str(QUALITY_WINDOW), **module)


def _store_routing_marker(client: TestClient, at: datetime, *, refreshed_at: datetime) -> None:
    """Store one derived reviewer effort change at *at* as dark_factory's marker derivation."""
    marker = Marker(
        marker_id=f'routing:reviewer_comprehensive:{at.isoformat()}', at=at, kind=MarkerKind.EFFORT,
        scope='dark_factory', series=None, until=None,
        label='reviewer_comprehensive effort high → xhigh',
        source=MarkerSource(SourceType.ROUTING, role='reviewer_comprehensive'),
    )
    assert client.portal is not None, 'quality.db is written inside a live TestClient'
    client.portal.call(partial(write_marker_state_db, app.state.config.quality_db, 'dark_factory',
                               (marker,), refreshed_at=refreshed_at))


def _alarm_scenario(
    client: TestClient, workdir: Path, *, marker_before: int, sampled_at: datetime,
) -> dict[str, Any]:
    """Sketch row 9: the rejection rate shifts at :data:`QUALITY_SHIFT`, and the one derived
    marker is dated at the landing *marker_before* landings before it."""
    root = _quality_root(workdir)
    rows = _quality_history(QUALITY_SHIFT + QUALITY_SHIFTED, outcome=_shifted_outcome)
    write_landing_events(root, rows)
    _store_quality(client, root, rows, sampled_at=sampled_at)
    _store_routing_marker(client, rows[QUALITY_SHIFT - marker_before].landing.landed_at,
                          refreshed_at=sampled_at)
    return _render_quality(client, QUALITY_AT)


def _quality_family(client: TestClient, workdir: Path) -> dict[str, Any]:
    """Quality sketch rows 3, 4, 5, 9, 10 and 12 and the module drill-down: one project's
    per-landing series as /quality serves it.

    Each scenario stores rows in quality.db and records the same landings in runs.db, so the
    read side's freshness is runs.db's verdict on the store, never a fixture's say-so.
    """
    bodies: dict[str, Any] = {}
    sampled = QUALITY_AT - timedelta(minutes=5)

    root = _quality_root(workdir / 'quality_fresh')
    fresh = _off_the_line(_quality_history(QUALITY_STORED, cost=_quality_cost))
    write_landing_events(root, fresh)
    _store_quality(client, root, fresh, sampled_at=sampled, git_check=QUALITY_GIT_CHECK)
    bodies['quality_fresh'] = _render_quality(client, QUALITY_AT)
    bodies['quality_fresh_again'] = _render_quality(client, QUALITY_AT + timedelta(minutes=1))

    root = _quality_root(workdir / 'quality_before_epoch')
    start = QUALITY_SOURCE_EPOCH - timedelta(days=40)
    before = history(QUALITY_STORED, start=start, outcome=_quality_outcome,
                     epochs=replace(epochs_from(start), review_completed=QUALITY_SOURCE_EPOCH))
    write_landing_events(root, before)
    _store_quality(client, root, before, sampled_at=sampled)
    bodies['quality_before_epoch'] = _render_quality(client, QUALITY_AT)

    root = _quality_root(workdir / 'quality_small_n')
    small = _quality_history(QUALITY_SMALL_N)
    write_landing_events(root, small)
    _store_quality(client, root, small, sampled_at=sampled)
    bodies['quality_small_n'] = _render_quality(client, QUALITY_AT)

    root = _quality_root(workdir / 'quality_main_ahead')
    stored = _quality_history(QUALITY_STORED)
    pending = train(QUALITY_PENDING_TRAIN, 'sha-pending-train', QUALITY_STORED,
                    sampled + timedelta(minutes=2), epochs=stored[-1].rule_vintage)
    write_landing_events(root, [*stored, *pending])
    _store_quality(client, root, stored, sampled_at=sampled)
    bodies['quality_main_ahead'] = _render_quality(client, QUALITY_AT)
    next_tick = sampled + QUALITY_TICK
    _store_quality(client, root, [*stored, *pending], sampled_at=next_tick)
    bodies['quality_main_ahead_cleared'] = _render_quality(client, next_tick + timedelta(minutes=1))

    bodies['quality_alarm_explained'] = _alarm_scenario(
        client, workdir / 'quality_alarm_explained', marker_before=10, sampled_at=sampled)
    bodies['quality_alarm_unexplained'] = _alarm_scenario(
        client, workdir / 'quality_alarm_unexplained', marker_before=100, sampled_at=sampled)

    root = _quality_root(workdir / 'quality_module')
    touched = _quality_history(QUALITY_STORED, cost=_quality_cost, modules=_module_lines)
    write_landing_events(root, touched)
    _store_quality(client, root, touched, sampled_at=sampled)
    bodies['quality_module'] = _render_quality(client, QUALITY_AT, module=QUALITY_MODULE)

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
            **_check_owner_family(client, workdir),
            **prd_view_family(client, workdir),
            **prd_lanes_family(client, workdir),
            **_quality_family(client, workdir),
        }
