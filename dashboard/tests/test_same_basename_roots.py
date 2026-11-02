"""Two configured roots sharing a basename stay distinct on every per-project surface."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot

from dashboard.config import DashboardConfig
from dashboard.data import escalation_corpus, redux_api
from dashboard.data.active_tasks import (
    collect_census_snapshots,
    collect_tasks_with_counts,
    shape_terminal_rows,
)
from dashboard.data.datum import Datum, DatumState
from dashboard.data.escalation_corpus import QueueKind, corpus_queues
from dashboard.data.root_invariants import RootInvariants, ScanStatus
from dashboard.data.scheduler import collect_scheduler_state
from dashboard.data.task_snapshot import (
    FRESHNESS_BOUND_SECONDS,
    SnapshotFailure,
    unmeasured_snapshot,
)
from dashboard.project_labels import ProjectLabels

pytestmark = pytest.mark.usefixtures('fresh_task_snapshot_state')

A_LABEL = 'a/proj'
B_LABEL = 'b/proj'


@pytest.fixture
def roots(tmp_path) -> tuple[Path, Path]:
    a = tmp_path / 'a' / 'proj'
    b = tmp_path / 'b' / 'proj'
    for root in (a, b):
        root.mkdir(parents=True)
    return a.resolve(), b.resolve()


@pytest.fixture
def config(roots) -> DashboardConfig:
    a, b = roots
    config = DashboardConfig(project_root=a, known_project_roots=[b])
    assert (config.project_labels.of(a), config.project_labels.of(b)) == (A_LABEL, B_LABEL)
    return config


def _substrate_with_one_task_per_root(titles: dict[Path, str]):
    """A fused-memory stand-in whose one in-progress task 1 is titled per root."""

    async def _mcp(client, url, tool, args, **_kw):
        title = titles[Path(args['project_root'])]
        if tool == 'get_statuses':
            return {'statuses': {'1': 'in-progress'}}
        if tool == 'get_tasks':
            return {'tasks': [{
                'id': '1', 'title': title, 'status': 'in-progress',
                'dependencies': [], 'metadata': {},
                'updatedAt': '2026-01-01T00:00:01+00:00',
            }]}
        raise AssertionError(f'unexpected tool {tool!r}')

    return _mcp


def _runtime_for_b_only(monkeypatch) -> None:
    snapshot = TaskRuntimeSnapshot(tasks=[TaskRuntimeEntry(
        task_id=1, has_worktree=True, loops=3, attempts=1, started=None,
        lane='lane-b', phase=None, lane_state=None,
    )])

    async def _fetch_task_runtime(client, escalation_urls):
        return {B_LABEL: snapshot}

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', _fetch_task_runtime)


class TestTasksCollector:
    @pytest.fixture(autouse=True)
    def _canned_reads(self, roots, monkeypatch):
        a, b = roots
        monkeypatch.setattr(
            'dashboard.data.tasks.mcp_tool_call',
            _substrate_with_one_task_per_root({a: 'from-a', b: 'from-b'}),
        )
        _runtime_for_b_only(monkeypatch)

    async def test_each_root_contributes_its_own_row_once(self, config, dummy_client):
        rows, _snapshots = await collect_tasks_with_counts(dummy_client, config)

        assert sorted((row['id'], row['project'], row['title']) for row in rows) == [
            ('a/proj/T-1', A_LABEL, 'from-a'),
            ('b/proj/T-1', B_LABEL, 'from-b'),
        ]

    async def test_each_root_keeps_its_own_snapshot(self, config, dummy_client):
        _rows, snapshots = await collect_tasks_with_counts(dummy_client, config)

        assert set(snapshots) == {A_LABEL, B_LABEL}

    async def test_a_roots_runtime_reaches_only_that_roots_rows(self, config, dummy_client):
        rows, _snapshots = await collect_tasks_with_counts(dummy_client, config)
        by_project = {row['project']: row for row in rows}

        assert by_project[B_LABEL]['lane'] == 'lane-b'
        assert by_project[B_LABEL]['runtime_status'] == 'ok'
        assert by_project[A_LABEL]['lane'] is None
        assert by_project[A_LABEL]['runtime_status'] == 'not_configured'

    async def test_the_census_keeps_each_root(self, config, dummy_client):
        snapshots = await collect_census_snapshots(dummy_client, config)

        assert set(snapshots) == {A_LABEL, B_LABEL}


def test_terminal_rows_are_identified_by_the_label_they_are_shaped_under():
    raw = {
        'id': 7, 'title': 'finished', 'status': 'done', 'dependencies': [],
        'metadata': {}, 'updated_at': '2026-01-01T00:00:07+00:00',
    }

    (row,) = shape_terminal_rows(B_LABEL, [raw], now=datetime(2026, 1, 2, tzinfo=UTC))

    assert (row['id'], row['project']) == ('b/proj/T-7', B_LABEL)


def test_the_terminal_window_reads_the_root_its_label_names(client, roots):
    a, b = roots
    client.app.state.config = dataclasses.replace(
        client.app.state.config, project_root=a, known_project_roots=[b],
    )
    window = AsyncMock(return_value=Datum(
        None, None, DatumState.UNKNOWN, 'canned', FRESHNESS_BOUND_SECONDS,
    ))
    with patch(
        'dashboard.api.tasks.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], {})),
    ), patch('dashboard.api.tasks.acquire_terminal_window', new=window):
        resp = client.get('/api/v2/dashboard/tasks?terminal=b%2Fproj')

    assert resp.status_code == 200, resp.text
    assert f'TASKS_TERMINAL:{B_LABEL}' in resp.json()
    (acquisition,) = window.await_args_list
    assert acquisition.args[2] == b


_NOW = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
_FIVE_MINUTES_AGO = (_NOW - timedelta(minutes=5)).isoformat()


def _active_row(label: str, task_id: int) -> dict:
    """An active row as ``collect_tasks_with_counts`` emits it, locking ``src/utils.py``."""
    return {
        'id': f'{label}/T-{task_id}', 'project': label, 'title': f'task {task_id}',
        'priority': 'medium', 'status': 'in-progress', 'started': 5,
        'meta_files': ['src/utils.py'],
    }


@dataclasses.dataclass
class _SchedulerSubstrate:
    """A fused-memory stand-in answering the scheduler's three reads per root."""

    skips: dict[Path, int]
    recovery_type: dict[Path, str]
    state_reads: list[Path] = dataclasses.field(default_factory=list)

    async def __call__(self, client, url, tool, args, **_kw):
        root = Path(args['project_root'])
        if tool == 'get_scheduler_state':
            self.state_reads.append(root)
            return {
                'skip_counts': {}, 'parks': {}, 'effective_priorities': {},
                'pin_queue': [], 'overrides': {}, 'current_holders': {},
            }
        if args['event_types'] == ['task_skipped']:
            return [
                {'event_type': 'task_skipped', 'task_id': '7', 'timestamp': _FIVE_MINUTES_AGO}
                for _ in range(self.skips.get(root, 0))
            ]
        return [{
            'event_type': self.recovery_type[root], 'task_id': '7',
            'timestamp': _FIVE_MINUTES_AGO,
        }]


class TestSchedulerCollector:
    @pytest.fixture
    def scheduler_roots(self, tmp_path) -> tuple[Path, Path]:
        x = tmp_path / 'x' / 'proj'
        y = tmp_path / 'y' / 'proj'
        for root in (x, y):
            root.mkdir(parents=True)
        return x.resolve(), y.resolve()

    @pytest.fixture
    def substrate(self, scheduler_roots) -> _SchedulerSubstrate:
        x, y = scheduler_roots
        return _SchedulerSubstrate(
            skips={x: 2, y: 3},
            recovery_type={x: 'recovery_vetoed', y: 'strand_converted'},
        )

    @staticmethod
    async def _collect(client, roots, substrate, active, snapshots=None):
        config = DashboardConfig(project_root=roots[0], known_project_roots=list(roots[1:]))
        with patch('dashboard.data.scheduler.mcp_tool_call', new=substrate), patch(
            'dashboard.data.scheduler.collect_tasks_with_counts',
            new=AsyncMock(return_value=(active, snapshots or {})),
        ):
            return await collect_scheduler_state(client, config, now=_NOW)

    async def _both_online(self, dummy_client, scheduler_roots, substrate):
        return await self._collect(
            dummy_client, scheduler_roots, substrate,
            [_active_row('x/proj', 7), _active_row('y/proj', 7)],
        )

    async def test_each_task_row_is_composed_once_with_its_own_root(
        self, dummy_client, scheduler_roots, substrate,
    ):
        x, y = scheduler_roots
        rows, *_rest = await self._both_online(dummy_client, scheduler_roots, substrate)

        assert len(rows) == 2, rows
        assert {(row['project'], row['task_id'], row['project_root']) for row in rows} == {
            ('x/proj', '7', str(x)), ('y/proj', '7', str(y)),
        }

    async def test_each_roots_skip_sparkline_keeps_its_own_key(
        self, dummy_client, scheduler_roots, substrate,
    ):
        _rows, _modules, _pins, events_by_task, *_rest = await self._both_online(
            dummy_client, scheduler_roots, substrate,
        )

        assert {key: sum(spark['values']) for key, spark in events_by_task.items()} == {
            'x/proj/7': 2, 'y/proj/7': 3,
        }

    async def test_each_roots_recovery_events_keep_their_own_key(
        self, dummy_client, scheduler_roots, substrate,
    ):
        *_head, recovery_events, _degraded = await self._both_online(
            dummy_client, scheduler_roots, substrate,
        )

        assert {
            label: [(row['event_type'], row['project']) for row in events]
            for label, events in recovery_events.items()
        } == {
            'x/proj': [('recovery_vetoed', 'x/proj')],
            'y/proj': [('strand_converted', 'y/proj')],
        }

    async def test_each_roots_module_contention_is_counted_apart(
        self, dummy_client, scheduler_roots, substrate,
    ):
        _rows, modules, *_rest = await self._both_online(dummy_client, scheduler_roots, substrate)

        assert sorted(
            (module['project'], module['contention'])
            for module in modules if module['path'] == 'src/utils.py'
        ) == [('x/proj', 1), ('y/proj', 1)]

    async def test_an_offline_root_skips_only_itself(
        self, dummy_client, scheduler_roots, substrate,
    ):
        x, y = scheduler_roots
        offline_x = unmeasured_snapshot(
            x, now=_NOW, reason='test', failure=SnapshotFailure.UNREACHABLE,
        )
        rows, _modules, _pins, _events, offline_projects, _paused, recovery_events, _degraded = (
            await self._collect(
                dummy_client, scheduler_roots, substrate,
                [_active_row('y/proj', 7)], {'x/proj': offline_x},
            )
        )

        assert offline_projects == ['x/proj']
        assert substrate.state_reads == [y]
        assert [row['project'] for row in rows] == ['y/proj']
        assert set(recovery_events) == {'y/proj'}

    async def test_a_label_containing_the_uid_separator_keeps_its_task_id(
        self, dummy_client, tmp_path,
    ):
        m = tmp_path / 'm' / 'T-x' / 'proj'
        n = tmp_path / 'n' / 'T-x' / 'proj'
        for root in (m, n):
            root.mkdir(parents=True)
        roots = (m.resolve(), n.resolve())
        substrate = _SchedulerSubstrate(
            skips={}, recovery_type=dict.fromkeys(roots, 'recovery_left'),
        )

        rows, *_rest = await self._collect(
            dummy_client, roots, substrate, [_active_row('m/T-x/proj', 5)],
        )

        assert [(row['project'], row['task_id']) for row in rows] == [('m/T-x/proj', '5')]


def _write_orchestrator_config(root: Path, escalation_port: int) -> None:
    (root / 'dark-factory-orchestrator.yaml').write_text(
        yaml.safe_dump({'escalation': {'port': escalation_port}}),
    )


def test_each_orchestrator_queue_is_labelled_by_its_escalation_url_key(roots):
    a, b = roots
    _write_orchestrator_config(a, 9101)
    _write_orchestrator_config(b, 9102)
    config = DashboardConfig(project_root=a, known_project_roots=[b])

    queue_labels = {
        queue.label for queue in corpus_queues(config) if queue.kind is QueueKind.ORCHESTRATOR
    }

    assert queue_labels == {A_LABEL, B_LABEL} == set(config.escalation_urls)


def test_each_roots_invariants_record_names_its_own_project(config, roots):
    records = [
        RootInvariants(
            root=root, status=ScanStatus.BLIND, measured_at=_NOW, rows_scanned=None,
            error='x', terminal_tier=None, hygiene_tier=None, chains=None,
        )
        for root in roots
    ]

    assert [record.to_wire(config.project_labels)['project'] for record in records] == [
        A_LABEL, B_LABEL,
    ]


def test_the_invariants_route_names_each_root_by_its_own_label(client, roots):
    a, b = roots
    escalation_corpus._corpus_cache_clear()
    try:
        client.app.state.config = dataclasses.replace(
            client.app.state.config, project_root=a, known_project_roots=[b],
        )
        resp = client.get('/api/v2/dashboard/invariants')
    finally:
        escalation_corpus._corpus_cache_clear()

    assert resp.status_code == 200, resp.text
    assert [(record['root'], record['project']) for record in resp.json()['INVARIANTS']] == [
        (str(a), A_LABEL), (str(b), B_LABEL),
    ]


_TEAM_A = '/srv/team-a/app'
_TEAM_B = '/srv/team-b/app'
_TEAM_LABELS = ProjectLabels.of_roots([_TEAM_A, _TEAM_B])


def _measured_merge_queue(depth: int) -> dict:
    """One project's merge-queue aggregate, with the fields the route always resolves."""
    return {
        'depth_timeseries': {'labels': [0], 'values': [depth]},
        'outcomes': {'labels': [], 'values': []},
        'latency': {},
        'recent': [],
        'active': [],
        'in_queue': Datum(0, _NOW, DatumState.FRESH, None, 30),
        'live_probe_configured': True,
        'speculative': Datum(
            {'hit_count': 0, 'discard_count': 0, 'total': 0, 'hit_rate': None},
            _NOW, DatumState.FRESH, None, 30,
        ),
        'recent_total': Datum(0, _NOW, DatumState.FRESH, None, 30),
    }


class TestReduxShapers:
    def test_orchestrators_and_projects_name_each_root_by_its_own_label(self):
        raw = [
            {'pids': [pid], 'project_root': root, 'running': True, 'started': ''}
            for pid, root in ((1, _TEAM_A), (2, _TEAM_B))
        ]

        body = redux_api.shape_orchestrators(raw, labels=_TEAM_LABELS, running_spark=None)

        assert {orch['project'] for orch in body['ORCHESTRATORS']} == {'team-a/app', 'team-b/app'}
        assert sorted((p['id'], p['name'], p['active']) for p in body['PROJECTS']) == [
            ('team-a/app', 'team-a/app', True), ('team-b/app', 'team-b/app', True),
        ]

    def test_a_discovered_orchestrator_is_told_apart_from_the_configured_root_it_shares_a_basename_with(self):
        raw = [
            {'pids': [pid], 'project_root': root, 'running': True, 'started': ''}
            for pid, root in ((1, '/srv/app'), (2, '/tmp/wt/app'))
        ]

        body = redux_api.shape_orchestrators(raw, labels=ProjectLabels.of_roots(['/srv/app']))

        assert [orch['project'] for orch in body['ORCHESTRATORS']] == ['app', 'wt/app']
        assert sorted(p['id'] for p in body['PROJECTS']) == ['app', 'wt/app']

    def test_merge_queue_keeps_each_roots_depth_and_halt(self):
        halt_status = {
            'team-a/app': {'wired': True, 'halted': True, 'owner_esc_id': 'esc-a', 'offline': False},
            'team-b/app': {'wired': True, 'halted': False, 'owner_esc_id': None, 'offline': False},
        }

        mq = redux_api.shape_merge_queue(
            {_TEAM_A: _measured_merge_queue(3), _TEAM_B: _measured_merge_queue(5)},
            labels=_TEAM_LABELS, halt_status=halt_status, served_at=_NOW,
        )['MERGE_QUEUE']

        assert {
            label: (block['depth']['values'], block['halt']['halted'])
            for label, block in mq.items()
        } == {'team-a/app': ([3], True), 'team-b/app': ([5], False)}

    def test_burndown_keeps_and_sums_each_root(self):
        sampled_at = (_NOW - timedelta(minutes=1)).isoformat()
        series = {
            _TEAM_A: {'labels': [sampled_at], 'done': [10], 'pending': [1]},
            _TEAM_B: {'labels': [sampled_at], 'done': [32], 'pending': [2]},
        }

        body = redux_api.shape_burndown(series, labels=_TEAM_LABELS, served_at=_NOW)

        assert set(body['BURNDOWN_BY_PROJECT']) == {'team-a/app', 'team-b/app'}
        assert body['BURNDOWN']['latest']['value']['counts']['done'] == 42

    def test_burndown_keeps_a_discovered_roots_series_beside_the_configured_one(self):
        sampled_at = (_NOW - timedelta(minutes=1)).isoformat()
        series = {
            '/srv/app': {'labels': [sampled_at], 'done': [10], 'pending': [1]},
            '/tmp/wt/app': {'labels': [sampled_at], 'done': [32], 'pending': [2]},
        }

        body = redux_api.shape_burndown(series, labels=ProjectLabels.of_roots(['/srv/app']), served_at=_NOW)

        assert set(body['BURNDOWN_BY_PROJECT']) == {'app', 'wt/app'}
        assert body['BURNDOWN']['latest']['value']['counts']['done'] == 42


def test_the_merge_queue_route_reads_each_roots_own_live_probe(client, roots):
    a, b = roots
    client.app.state.config = dataclasses.replace(
        client.app.state.config, project_root=a, known_project_roots=[b],
    )
    entry = {
        'task_id': '7', 'branch': 'task/7', 'state': 'queued',
        'age_secs': 60.0, 'position': 1, 'waiter_alive': True,
    }
    live_map = {
        A_LABEL: {'reachable': True, 'entries': [entry], 'metrics': {'probe': 'a'}},
        B_LABEL: {'reachable': True, 'entries': [entry, entry], 'metrics': {'probe': 'b'}},
    }
    per_root = {str(root): _measured_merge_queue(0) for root in roots}
    with (
        patch('dashboard.api.merge_queue.build_per_project_merge_queue',
              new=AsyncMock(return_value=per_root)),
        patch('dashboard.api.merge_queue.get_merge_halt_status', new=AsyncMock(return_value={})),
        patch('dashboard.api.merge_queue.fetch_live_merge_queues',
              new=AsyncMock(return_value=live_map)),
        patch('dashboard.api.merge_queue.lookup_tasks', new=AsyncMock(return_value={})),
    ):
        resp = client.get('/api/v2/dashboard/merge-queue')

    assert resp.status_code == 200, resp.text
    mq = resp.json()['MERGE_QUEUE']
    assert {
        label: (len(block['active']), block['in_queue']['value'], block['metrics'])
        for label, block in mq.items()
    } == {
        A_LABEL: (1, 1, {'probe': 'a'}),
        B_LABEL: (2, 2, {'probe': 'b'}),
    }
