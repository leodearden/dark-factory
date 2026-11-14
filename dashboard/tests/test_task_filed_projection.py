"""The ``created_at`` / ``created_at_source`` filing-time pair through the dashboard's row seams.

PRD ``plans/instruments-honesty-prd.md`` task α; the source vocabulary is
``shared/src/shared/task_created_at.py::CreatedAtSource``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from _boundary_payloads import CANNED_SCHEDULER_STATE, T0, canned_prd_roots
from _canned_mcp import CannedMCP, _raw_row
from _dt_helpers import fixed_clock, installed_clock
from fastapi.testclient import TestClient

from dashboard.app import app
from dashboard.data.active_tasks import task_uid

pytestmark = pytest.mark.usefixtures('fresh_task_snapshot_state')

ROOT = 'dark-factory'
TASKS = '/api/v2/dashboard/tasks'


@pytest.fixture(scope='module', autouse=True)
def _module_samplers_off() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv('DASHBOARD_SAMPLERS', 'off')
        yield


IN_PROGRESS, PENDING, UNBACKFILLED, DONE = 9101, 9102, 9103, 9111
UNFILED_ACTIVE, UNFILED_DONE = 9104, 9112
"""Rows that carry no pair at all, as a store without the v6 columns serves them."""
SERVED_PAIRS: dict[int, tuple[str | None, str | None]] = {
    IN_PROGRESS: ('2026-10-01T08:00:00.000Z', 'stamped'),
    PENDING: ('2026-09-15T17:30:00.125Z', 'pending_since'),
    UNBACKFILLED: (None, None),
    DONE: ('2026-08-02T23:59:59.999Z', 'first_event'),
}
STATUSES = {
    IN_PROGRESS: 'in-progress', PENDING: 'pending', UNBACKFILLED: 'pending', DONE: 'done',
    UNFILED_ACTIVE: 'in-progress', UNFILED_DONE: 'done',
}


def _substrate() -> CannedMCP:
    rows = []
    for task_id, status in STATUSES.items():
        if task_id in SERVED_PAIRS:
            created_at, source = SERVED_PAIRS[task_id]
            rows.append(_raw_row(task_id, status, created_at=created_at, created_at_source=source))
        else:
            rows.append(_raw_row(task_id, status))
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


def _rows_by_uid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row['id']: row for row in rows}


def _assert_pair_absent_as_none(row: dict[str, Any]) -> None:
    assert 'created_at' in row
    assert 'created_at_source' in row
    assert row['created_at'] is None
    assert row['created_at_source'] is None


class TestTheTasksRouteServesFilingTime:
    def test_the_tasks_rows_carry_the_served_pair(self, client, tmp_path):
        with _served(client, _substrate(), tmp_path):
            body = _get(client)

        rows = body['TASKS_SNAPSHOT'][ROOT]['rows']
        assert rows['state'] == 'fresh', rows['reason']
        served = _pairs_by_uid(rows['value'])
        for task_id in (IN_PROGRESS, PENDING, UNBACKFILLED):
            assert served[task_uid(ROOT, task_id)] == SERVED_PAIRS[task_id]

    def test_the_terminal_window_carries_the_served_pair(self, client, tmp_path):
        with _served(client, _substrate(), tmp_path):
            body = _get(client, terminal=ROOT)

        window = body[f'TASKS_TERMINAL:{ROOT}']
        assert window['value'], window['reason']
        assert _pairs_by_uid(window['value'])[task_uid(ROOT, DONE)] == SERVED_PAIRS[DONE]

    def test_a_tasks_row_without_the_pair_serves_none_with_the_keys_present(self, client, tmp_path):
        with _served(client, _substrate(), tmp_path):
            body = _get(client)

        rows = body['TASKS_SNAPSHOT'][ROOT]['rows']
        assert rows['state'] == 'fresh', rows['reason']
        _assert_pair_absent_as_none(_rows_by_uid(rows['value'])[task_uid(ROOT, UNFILED_ACTIVE)])

    def test_a_terminal_row_without_the_pair_serves_none_with_the_keys_present(self, client, tmp_path):
        with _served(client, _substrate(), tmp_path):
            body = _get(client, terminal=ROOT)

        window = body[f'TASKS_TERMINAL:{ROOT}']
        assert window['value'], window['reason']
        _assert_pair_absent_as_none(_rows_by_uid(window['value'])[task_uid(ROOT, UNFILED_DONE)])
