"""Reusing one unit's shaped rows across renders, at the collector's public seam.

Two renders inside one ``SNAPSHOT_TTL_SECONDS`` are served the SAME cached
unit. Whatever the collector reuses between them, each render's rows must
still carry that render's own runtime probe, clock and external-dep read, and
serialise exactly as a freshly shaped row would.

What may be reused is the unit-derived half of each row, memoised per unit by
``UnitRowMemo``; its own contract is pinned on a test-owned instance below.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta

import pytest
from _canned_mcp import ByRoot, CannedMCP, _raw_row
from _task_units import measured_unit
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import UnitRowMemo, collect_tasks_with_counts

pytestmark = pytest.mark.usefixtures('fresh_task_snapshot_state')

NOW = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
LATER = NOW + timedelta(minutes=3)
STARTED = (NOW - timedelta(minutes=10)).isoformat()
OFFLINE = TaskRuntimeSnapshot(offline=True, offline_reason='unreachable')


class _RuntimeProbe:
    """A stand-in for ``fetch_task_runtime`` whose answer a test changes between renders."""

    def __init__(self) -> None:
        self.answer: dict[str, TaskRuntimeSnapshot] = {}

    async def __call__(self, client, escalation_urls):
        return dict(self.answer)


def _entry(task_id: int, *, loops: int, phase, started: str | None = STARTED) -> TaskRuntimeEntry:
    return TaskRuntimeEntry(
        task_id=task_id, has_worktree=True, loops=loops, attempts=1,
        started=started, lane='_lane-1', phase=phase, lane_state='assigned',
    )


@pytest.fixture()
def probe(monkeypatch) -> _RuntimeProbe:
    fake = _RuntimeProbe()
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', fake)
    return fake


@pytest.fixture()
def alpha_config(tmp_path) -> DashboardConfig:
    root = tmp_path / 'alpha'
    root.mkdir()
    return DashboardConfig(project_root=root)


def _serve(monkeypatch, by_label: dict[str, CannedMCP]) -> None:
    monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', ByRoot(by_label))


def _assert_one_unit(alpha: CannedMCP) -> None:
    """Every render of the test was served the one unit the first render read."""
    reads = alpha.calls_to('get_tasks')
    assert len(reads) == 1, (
        f'{len(reads)} row reads: a later render missed the unit cache, so it '
        'reshaped a new unit and reuse was never exercised'
    )


async def _alpha_row(client, config, *, now: datetime) -> dict:
    snapshots = await collect_tasks_with_counts(client, config, resolve_external=True, now=now)
    (row,) = snapshots['alpha'].rows.value or []
    return row


async def test_each_render_carries_its_own_runtime_fields_and_started(
    dummy_client, alpha_config, probe, monkeypatch,
):
    alpha = CannedMCP(rows=[_raw_row(1, 'in-progress')], status_map={1: 'in-progress'})
    _serve(monkeypatch, {'alpha': alpha})

    probe.answer = {'alpha': TaskRuntimeSnapshot(tasks=[_entry(1, loops=1, phase='PLAN')])}
    first = await _alpha_row(dummy_client, alpha_config, now=NOW)
    probe.answer = {'alpha': TaskRuntimeSnapshot(tasks=[_entry(1, loops=2, phase='EXECUTE')])}
    second = await _alpha_row(dummy_client, alpha_config, now=LATER)
    probe.answer = {'alpha': OFFLINE}
    offline = await _alpha_row(dummy_client, alpha_config, now=LATER)

    _assert_one_unit(alpha)
    assert (first['loops'], first['phase'], first['started']) == (1, 'PLAN', 10)
    assert (second['loops'], second['phase'], second['started']) == (2, 'EXECUTE', 13), (
        f'the second render carried {second}: a runtime fact or the clock was reused '
        'from the first render of the same unit'
    )
    assert (offline['runtime_offline'], offline['runtime_status'], offline['loops']) == (
        True, 'unreachable', None,
    ), f'an offline probe after an online render of the same unit must read offline: {offline}'
    assert (first['loops'], first['phase']) == (1, 'PLAN'), (
        'a later render changed a row an earlier render already returned'
    )


async def test_an_external_dep_status_never_leaks_into_a_later_render(
    dummy_client, alpha_config, probe, monkeypatch,
):
    alpha = CannedMCP(
        rows=[_raw_row(1, 'pending', metadata={'external_deps': ['beta:5']})],
        status_map={1: 'pending'},
    )
    beta = CannedMCP(status_map={5: 'done'})
    _serve(monkeypatch, {'alpha': alpha, 'beta': beta})
    probe.answer = {'alpha': TaskRuntimeSnapshot()}

    first = await _alpha_row(dummy_client, alpha_config, now=NOW)
    beta.status_map[5] = 'in-progress'
    second = await _alpha_row(dummy_client, alpha_config, now=LATER)

    _assert_one_unit(alpha)
    assert second['external_deps'] == [{'id': 'beta:5', 'status': 'in-progress'}]
    assert first['external_deps'] == [{'id': 'beta:5', 'status': 'done'}], (
        f'the first render\'s row now says {first["external_deps"]}: the two renders '
        'share one external_deps entry, so the second read overwrote the first'
    )


EVERY_FIELD_GROUP_WIRE = {
    'id': 'alpha/T-1',
    'project': 'alpha',
    'title': 'task 1',
    'status': 'in-progress',
    'agent': 'claude-task-1',
    'loops': 2,
    'attempts': 1,
    'lane': '_lane-1',
    'phase': 'EXECUTE',
    'lane_state': 'assigned',
    'runtime_offline': False,
    'runtime_status': 'ok',
    'claimant_run_id': 'run-1/sess-1/pid=42',
    'heartbeat_at': '2026-11-22T11:59:00+00:00',
    'stranded': False,
    'created_at': '2026-11-12T09:10:17.253Z',
    'created_at_source': 'stamped',
    'meta_files': ['a.py', 'b.py'],
    'train': {'id': 'train-1', 'order': 2},
    'external_deps': [{'id': 'beta:5', 'status': 'done'}],
    'prd': None,
    'started': 13,
    'deps': [
        {'id': 'alpha/T-2', 'title': 'task 2', 'done': False},
        {'id': 'alpha/T-3', 'title': '', 'done': True},
    ],
}


async def test_a_reused_render_serialises_byte_identically(
    dummy_client, alpha_config, probe, monkeypatch,
):
    row = _raw_row(
        1, 'in-progress',
        dependencies=['2', '3'],
        claimant_run_id='run-1/sess-1/pid=42',
        heartbeat_at='2026-11-22T11:59:00+00:00',
        created_at='2026-11-12T09:10:17.253Z',
        created_at_source='stamped',
        metadata={
            'files': ['a.py', 'b.py'],
            'train': {'id': 'train-1', 'order': 2},
            'external_deps': ['beta:5'],
        },
    )
    alpha = CannedMCP(
        rows=[row, _raw_row(2, 'pending')],
        status_map={1: 'in-progress', 2: 'pending', 3: 'done'},
    )
    _serve(monkeypatch, {'alpha': alpha, 'beta': CannedMCP(status_map={5: 'done'})})
    probe.answer = {'alpha': TaskRuntimeSnapshot(tasks=[_entry(1, loops=2, phase='EXECUTE')])}

    await collect_tasks_with_counts(dummy_client, alpha_config, resolve_external=True, now=NOW)
    snapshots = await collect_tasks_with_counts(
        dummy_client, alpha_config, resolve_external=True, now=LATER,
    )
    served = (snapshots['alpha'].rows.value or [])[0]
    _assert_one_unit(alpha)

    def wire(value) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

    assert wire(served) == wire(EVERY_FIELD_GROUP_WIRE)


# ---------------------------------------------------------------------------
# UnitRowMemo: the unit-derived half of a unit's rows, shaped once per unit.
# ---------------------------------------------------------------------------

RUNTIME_PLACEHOLDERS = {
    'agent': None, 'loops': None, 'attempts': None, 'lane': None, 'phase': None,
    'lane_state': None, 'started': None, 'runtime_offline': False, 'runtime_status': 'ok',
}


def _unit(*task_ids: int, external_deps: tuple[str, ...] = ()):
    return measured_unit(rows=[
        {
            'id': task_id, 'title': f'task {task_id}', 'status': 'pending',
            'dependencies': [], 'metadata': {'external_deps': list(external_deps)},
        }
        for task_id in task_ids
    ])


def test_the_same_unit_is_shaped_once():
    memo, unit = UnitRowMemo(), _unit(1, 2)

    assert memo.rows_for('alpha', unit) is memo.rows_for('alpha', unit)


def test_a_new_unit_is_reshaped_even_with_equal_content():
    memo, unit = UnitRowMemo(), _unit(1, 2)
    first = memo.rows_for('alpha', unit)

    refreshed = memo.rows_for('alpha', dataclasses.replace(unit))

    assert refreshed is not first, (
        'a refreshed unit is a new object, so it must be reshaped: equal content '
        'is not evidence it is the same generation'
    )
    assert refreshed == first


def test_labels_are_memoised_independently():
    memo, alpha, beta = UnitRowMemo(), _unit(1), _unit(2)
    alpha_rows, beta_rows = memo.rows_for('alpha', alpha), memo.rows_for('beta', beta)

    assert memo.rows_for('alpha', alpha) is alpha_rows
    assert memo.rows_for('beta', beta) is beta_rows


def test_clear_forgets_every_unit():
    memo, unit = UnitRowMemo(), _unit(1)
    first = memo.rows_for('alpha', unit)

    memo.clear()

    assert memo.rows_for('alpha', unit) is not first


def test_unit_rows_carry_no_per_render_fact():
    rows = UnitRowMemo().rows_for('alpha', _unit(1, 2, external_deps=('beta:5',)))

    assert [row.task_id for row in rows] == [1, 2]
    for row in rows:
        runtime = {key: row.fields[key] for key in RUNTIME_PLACEHOLDERS}
        assert runtime == RUNTIME_PLACEHOLDERS, (
            f'task {row.task_id} holds runtime facts {runtime}: a memo shared '
            'across renders must hold only placeholders for them'
        )
        assert row.fields['external_deps'] == [{'id': 'beta:5', 'status': 'unknown'}]
