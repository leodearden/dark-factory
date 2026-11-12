"""The ``created_at`` / ``created_at_source`` filing-time pair through the dashboard's row seams.

PRD ``plans/instruments-honesty-prd.md`` task α; the source vocabulary is
``shared/src/shared/task_created_at.py::CreatedAtSource``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _boundary_payloads import CANNED_SCHEDULER_STATE, T0, canned_prd_roots
from _canned_mcp import CannedMCP, _raw_row, is_metadata_projection
from _dt_helpers import fixed_clock, installed_clock
from fastapi.testclient import TestClient

from dashboard.app import app
from dashboard.data.active_tasks import _build_task_row, shape_terminal_rows, task_uid
from dashboard.data.tasks import _shape_task

pytestmark = pytest.mark.usefixtures('fresh_task_snapshot_state')

_FILED = '2026-11-12T09:10:17.253Z'
_NOW = datetime(2026, 11, 12, 12, 0, 0, tzinfo=UTC)

ROOT = 'dark-factory'
TASKS = '/api/v2/dashboard/tasks'


@pytest.fixture(scope='module', autouse=True)
def _module_samplers_off() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv('DASHBOARD_SAMPLERS', 'off')
        yield


def _row(**overrides) -> dict:
    """A minimally-valid MCP ``get_tasks`` row."""
    row = {
        'id': '7',
        'title': 'a task',
        'status': 'in-progress',
        'priority': 'medium',
        'dependencies': [],
        'metadata': {},
        'updatedAt': '2026-11-12T10:00:00.000Z',
    }
    row.update(overrides)
    return row


class TestShapeTaskCarriesFilingTime:
    @pytest.mark.parametrize('source', ['stamped', 'first_event'])
    def test_preserves_created_at_and_its_source(self, source):
        shaped = _shape_task(_row(created_at=_FILED, created_at_source=source))

        assert shaped is not None
        assert shaped['created_at'] == _FILED
        assert shaped['created_at_source'] == source

    def test_absent_pair_surfaces_none_with_keys_present(self):
        shaped = _shape_task(_row())

        assert shaped is not None
        assert 'created_at' in shaped
        assert 'created_at_source' in shaped
        assert shaped['created_at'] is None
        assert shaped['created_at_source'] is None

    def test_null_pair_on_an_unbackfilled_row_stays_none(self):
        shaped = _shape_task(_row(created_at=None, created_at_source=None))

        assert shaped is not None
        assert shaped['created_at'] is None
        assert shaped['created_at_source'] is None


def _task(**overrides) -> dict:
    """A dashboard-shaped task row as ``_shape_task`` emits it."""
    task = {
        'id': 42,
        'title': 'a task',
        'status': 'in-progress',
        'priority': 'medium',
        'dependencies': [],
        'metadata': {},
        'updated_at': '2026-11-12T11:00:00.000Z',
        'claimant_run_id': None,
        'heartbeat_at': None,
        'created_at': _FILED,
        'created_at_source': 'stamped',
    }
    task.update(overrides)
    return task


class TestBuildTaskRowCarriesFilingTime:
    def test_active_row_carries_the_pair(self):
        row = _build_task_row(
            'p', _task(created_at=_FILED, created_at_source='pending_since'), 42, {}, 'p/T-42',
            now=_NOW,
        )

        assert row['created_at'] == _FILED
        assert row['created_at_source'] == 'pending_since'

    def test_terminal_row_carries_the_pair(self):
        [row] = shape_terminal_rows(
            'p', [_task(status='done', created_at=_FILED, created_at_source='updated_at')], now=_NOW,
        )

        assert row['created_at'] == _FILED
        assert row['created_at_source'] == 'updated_at'

    def test_absent_pair_degrades_to_none(self):
        task = _task()
        del task['created_at'], task['created_at_source']

        row = _build_task_row('p', task, 42, {}, 'p/T-42', now=_NOW)

        assert 'created_at' in row
        assert 'created_at_source' in row
        assert row['created_at'] is None
        assert row['created_at_source'] is None


IN_PROGRESS, PENDING, DONE = 9101, 9102, 9111
SERVED_PAIRS: dict[int, tuple[str, str]] = {
    IN_PROGRESS: ('2026-10-01T08:00:00.000Z', 'stamped'),
    PENDING: ('2026-09-15T17:30:00.125Z', 'pending_since'),
    DONE: ('2026-08-02T23:59:59.999Z', 'first_event'),
}
STATUSES = {IN_PROGRESS: 'in-progress', PENDING: 'pending', DONE: 'done'}


def _substrate() -> CannedMCP:
    rows = [
        _raw_row(task_id, STATUSES[task_id], created_at=created_at, created_at_source=source)
        for task_id, (created_at, source) in SERVED_PAIRS.items()
    ]
    return CannedMCP(rows=rows, status_map=dict(STATUSES), status_page_size=2000)


def _served(client: TestClient, substrate: CannedMCP, tmp_path: Path):
    return canned_prd_roots(
        client, tmp_path, {ROOT: substrate}, schedulers={ROOT: CANNED_SCHEDULER_STATE},
    )


def _get(client: TestClient, **params: str) -> dict[str, Any]:
    with installed_clock(app, fixed_clock(T0)):
        response = client.get(TASKS, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _pairs_by_uid(rows: list[dict[str, Any]]) -> dict[str, tuple[Any, Any]]:
    return {row['id']: (row['created_at'], row['created_at_source']) for row in rows}


class TestTheTasksRouteServesFilingTime:
    def test_the_tasks_rows_carry_the_served_pair(self, client, tmp_path):
        with _served(client, _substrate(), tmp_path):
            body = _get(client)

        rows = body['TASKS_SNAPSHOT'][ROOT]['rows']
        assert rows['state'] == 'fresh', rows['reason']
        served = _pairs_by_uid(rows['value'])
        for task_id in (IN_PROGRESS, PENDING):
            assert served[task_uid(ROOT, task_id)] == SERVED_PAIRS[task_id]

    def test_the_terminal_window_carries_the_served_pair(self, client, tmp_path):
        with _served(client, _substrate(), tmp_path):
            body = _get(client, terminal=ROOT)

        window = body[f'TASKS_TERMINAL:{ROOT}']
        assert window['value'], window['reason']
        assert _pairs_by_uid(window['value'])[task_uid(ROOT, DONE)] == SERVED_PAIRS[DONE]

    def test_reading_the_pair_reads_nothing_unprojected(self, client, tmp_path):
        substrate = _substrate()
        substrate.audit_projection = True

        with _served(client, substrate, tmp_path):
            _get(client)
            _get(client, terminal=ROOT)

        row_reads = [call['args'] for call in substrate.calls_to('get_tasks')
                     if not is_metadata_projection(call['args'])]
        assert row_reads
        assert all(args.get('fields') is not None for args in row_reads), row_reads
        assert substrate.unprojected_reads == set()
