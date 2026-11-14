"""The drift guard for ``task_snapshot.ROW_READ_FIELDS``: no reader reads a key it does not ask for.

The routes run over an audited :class:`_canned_mcp.CannedMCP`, which serves every projected
row whole and records each read of a key the projection would have dropped. A reader that
reads such a key would see it missing in production, so every test asserts the record stays
empty.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from _boundary_payloads import (
    CANNED_SCHEDULER_STATE,
    LOCKED_FILE,
    T0,
    canned_prd_roots,
    lanes_substrate,
)
from _canned_mcp import CannedMCP, _raw_row, is_metadata_projection
from _dt_helpers import fixed_clock, installed_clock
from _prd_lanes_tree import STAMPED_BY, holder_processes, lanes_tree, prd_key
from fastapi.testclient import TestClient
from shared.task_deferral import CarriedByDeferral, HeldBySessionDeferral, record_to_json

from dashboard.app import app
from dashboard.data.active_tasks import task_uid
from dashboard.data.task_snapshot import ROW_READ_FIELDS

pytestmark = pytest.mark.usefixtures('fresh_task_snapshot_state')

ROOT = 'dark-factory'
TASKS = '/api/v2/dashboard/tasks'

CLAIMED, READY, CARRIED, HELD, BLOCKED = 9001, 9002, 9003, 9004, 9005
TERMINAL = (9011, 9012, 9013, 9014)
APPENDED = (CLAIMED, READY, CARRIED, HELD, BLOCKED, *TERMINAL)


@pytest.fixture(scope='module', autouse=True)
def _module_samplers_off() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv('DASHBOARD_SAMPLERS', 'off')
        yield


def _unprojected(task_id: int, status: str, metadata: dict[str, Any], **row: Any) -> dict[str, Any]:
    """A raw row that also carries every key a row read must not ask for."""
    return _raw_row(
        task_id, status,
        description=f'why {task_id} exists', details='how', testStrategy='prove it',
        metadata={**metadata, 'routing': {'model': 'm'}, 'memory_hints': ['h']},
        **row,
    )


def _appended(live_holder) -> list[dict[str, Any]]:
    stamp = {'deferred_at': T0 - timedelta(days=1), 'stamped_at': T0 - timedelta(days=1),
             'stamped_by': STAMPED_BY}
    return [
        _unprojected(CLAIMED, 'in-progress', {'infra_hold': {'reason': 'disk'}},
                     claimant_run_id='run-9001/sess/pid=1', heartbeat_at=T0.isoformat()),
        _unprojected(READY, 'pending', {
            'external_deps': ['reify:12'], 'train': {'id': 'tr-1', 'order': 1},
            'files': [LOCKED_FILE], 'prd': prd_key('ruling_over_tail'),
        }, priority='high', dependencies=['131', '101']),
        _unprojected(CARRIED, 'deferred', {
            'deferral': record_to_json(CarriedByDeferral(carrier_task_id=READY, **stamp)),
        }),
        _unprojected(HELD, 'deferred', {
            'deferral': record_to_json(HeldBySessionDeferral(holder=live_holder, **stamp)),
        }),
        _unprojected(BLOCKED, 'blocked', {'source': 'agent-followup'}),
        _unprojected(TERMINAL[0], 'done', {'files': ['a.py']}),
        _unprojected(TERMINAL[1], 'done', {'prd_path': prd_key('ruling_over_tail')}),
        _unprojected(TERMINAL[2], 'cancelled', {'spawned_from': READY}),
        _unprojected(TERMINAL[3], 'cancelled', {'files': ['b.py']}),
    ]


@contextmanager
def _audited_routes(client: TestClient, tmp_path: Path):
    """The lanes tree plus :func:`_appended`, audited, served to every route."""
    with holder_processes() as (live, dead):
        substrate = lanes_substrate(lanes_tree(T0, live_holder=live, dead_holder=dead))
        for row in _appended(live):
            substrate.rows.append(row)
            substrate.status_map[int(row['id'])] = row['status']
        substrate.audit_projection = True
        with canned_prd_roots(
            client, tmp_path, {ROOT: substrate}, schedulers={ROOT: CANNED_SCHEDULER_STATE},
        ) as render_prds:
            yield substrate, render_prds


def _get(client: TestClient, **params: str) -> dict[str, Any]:
    with installed_clock(app, fixed_clock(T0)):
        response = client.get(TASKS, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _row_reads(substrate: CannedMCP) -> list[dict[str, Any]]:
    return [call['args'] for call in substrate.calls_to('get_tasks')
            if not is_metadata_projection(call['args'])]


def _assert_nothing_unprojected(substrate: CannedMCP) -> None:
    row_reads = _row_reads(substrate)
    unaudited = [args for args in row_reads if args.get('fields') is None]
    assert row_reads and not unaudited, f'row reads the audit never saw: {unaudited}'
    assert substrate.unprojected_reads == set(), (
        f'a reader of the dashboard row read read {sorted(substrate.unprojected_reads)}, which '
        'the read does not ask for: add them to task_snapshot.ROW_READ_FIELDS'
    )


def test_the_tasks_render_reads_only_projected_keys(client, tmp_path):
    with _audited_routes(client, tmp_path) as (substrate, _):
        body = _get(client)

    rows = body['TASKS_SNAPSHOT'][ROOT]['rows']
    assert rows['state'] == 'fresh', rows['reason']
    served = {row['id'] for row in rows['value']}
    expected = {task_uid(ROOT, task_id) for task_id in (CLAIMED, READY, CARRIED, HELD, BLOCKED)}
    assert expected <= served
    _assert_nothing_unprojected(substrate)


def test_the_terminal_window_reads_only_projected_keys(client, tmp_path):
    with _audited_routes(client, tmp_path) as (substrate, _):
        body = _get(client, terminal=ROOT)

    window = body[f'TASKS_TERMINAL:{ROOT}']
    assert window['value'], window['reason']
    assert {task_uid(ROOT, task_id) for task_id in TERMINAL} <= {row['id'] for row in window['value']}
    _assert_nothing_unprojected(substrate)


def test_the_prd_view_and_contention_read_only_projected_keys(client, tmp_path):
    with _audited_routes(client, tmp_path) as (substrate, render_prds):
        body = render_prds(T0)

    view = body['PRD_VIEW'][ROOT]
    assert view['state'] == 'fresh', view['reason']
    contention = body['PRD_CONTENTION'][ROOT]
    assert contention['state'] == 'fresh', contention['reason']
    _assert_nothing_unprojected(substrate)


def test_every_row_read_asks_for_row_read_fields(client, tmp_path):
    with _audited_routes(client, tmp_path) as (substrate, render_prds):
        _get(client)
        _get(client, terminal=ROOT)
        render_prds(T0)

    row_reads = _row_reads(substrate)
    assert len(row_reads) >= 2, 'the active rows and the terminal window'
    assert all(args.get('fields') == list(ROW_READ_FIELDS) for args in row_reads), row_reads
