"""Tests for ``GET /api/v2/dashboard/prds``: the per-root PRD view served as a Datum.

Boundary rows 8, 9 and 11 at the endpoint (PRD dashboard-prd-lanes §4.4, §5.6),
driven over the ``prd_view`` family of ``_boundary_payloads`` and its CannedMCP
substrates. The pure view's own cases live in ``test_prd_view.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from _boundary_payloads import (
    ORPHAN,
    OVERLAY_ROW,
    PRD_C,
    T0,
    canned_prd_roots,
    prd_tree,
    prd_view_family,
)
from _canned_mcp import CannedMCP
from _prd_tree import PRD_A
from fastapi.testclient import TestClient
from shared.task_statuses import TaskStatus

from dashboard.api import prds
from dashboard.data import task_lineage
from dashboard.data.census import build_census
from dashboard.data.prd_view import Segment

ROOT = 'dark-factory'
SLOW = 'slow'
BROKEN = 'broken'


@pytest.fixture(scope='module', autouse=True)
def _module_samplers_off() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv('DASHBOARD_SAMPLERS', 'off')
        yield


@pytest.fixture(scope='module')
def bodies(_client: TestClient, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return prd_view_family(_client, tmp_path_factory.mktemp('prd_view'))


def _row(view: dict[str, Any], key: str) -> dict[str, Any]:
    [row] = [row for row in view['value']['prds'] if row['key'] == key]
    return row


def _open_ids_listed(value: dict[str, Any]) -> list[int]:
    return [
        *(task_id for row in value['prds'] for task_id in row['direct_open_ids']),
        *(followup['task_id'] for row in value['prds'] for followup in row['followup_open']),
        *(task_id for ids in value['dark_matter']['members'].values() for task_id in ids),
    ]


def test_the_body_carries_the_view_and_its_serving_instant(bodies):
    assert set(bodies['prd_view_fresh']) == {'PRD_VIEW', 'served_at'}


def test_the_view_is_keyed_by_every_configured_root(bodies):
    assert set(bodies['prd_view_fresh']['PRD_VIEW']) == {ROOT, 'reify'}


def test_a_fresh_view_counts_its_prd_exactly(bodies):
    view = bodies['prd_view_fresh']['PRD_VIEW'][ROOT]
    assert view['state'] == 'fresh'
    row = _row(view, PRD_A)
    assert row['segments'][Segment.DONE.value] == 2
    assert row['total'] == 4


def test_a_fresh_view_closes_the_census_sum_and_lists_each_open_id_once(bodies):
    value = bodies['prd_view_fresh']['PRD_VIEW'][ROOT]['value']
    status_map = prd_tree().status_map
    census = build_census(status_map)
    drawn = sum(row['total'] for row in value['prds'])
    assert (
        drawn + value['closed']['tasks'] + value['unattributed_done'] + value['dark_matter']['total']
        == census.total - census.counts[TaskStatus.CANCELLED]
    )
    open_ids = [task_id for task_id, status in status_map.items() if status not in {'done', 'cancelled'}]
    assert sorted(_open_ids_listed(value)) == sorted(open_ids)


def test_a_new_active_rows_own_prd_and_an_orphan_id_overlay_the_lineage(bodies):
    view = bodies['prd_view_overlay']['PRD_VIEW'][ROOT]
    assert _row(view, PRD_C)['direct_open_ids'] == [OVERLAY_ROW]
    assert view['value']['dark_matter']['members']['lineage_pending'] == [ORPHAN]


def test_a_failed_map_half_serves_an_unknown_view_with_the_maps_own_reason(bodies):
    view = bodies['prd_view_map_failed']['PRD_VIEW'][ROOT]
    census = bodies['prd_view_map_failed_tasks']['TASKS_SNAPSHOT'][ROOT]['census']
    assert census['state'] == 'stale'
    assert view['state'] == 'unknown'
    assert view['value'] is None
    assert view['reason'] == census['reason']


# --- a view that breaks its contract still answers 200 ---------------------


def _tamper_segments(datum):
    view = datum.value
    row, *rest = view.prds
    segments = MappingProxyType({**row.segments, Segment.DONE: row.segments[Segment.DONE] + 1})
    tampered = dataclasses.replace(view, prds=(dataclasses.replace(row, segments=segments), *rest))
    return dataclasses.replace(datum, value=tampered)


def test_a_view_breaking_its_contract_is_served_unknown(client, tmp_path, monkeypatch):
    built = prds.build_prd_view
    monkeypatch.setattr(prds, 'build_prd_view', lambda inputs, *, now: _tamper_segments(built(inputs, now=now)))
    with canned_prd_roots(client, tmp_path, {ROOT: prd_tree()}) as render:
        view = render(T0)['PRD_VIEW'][ROOT]
    assert view['state'] == 'unknown'
    assert 'value_contract' in view['reason']
    assert 'segments_sum' in view['reason']


def test_a_view_whose_build_raises_is_served_unknown(client, tmp_path, monkeypatch):
    def _raises(inputs, *, now):
        raise RuntimeError('shaping bug')

    monkeypatch.setattr(prds, 'build_prd_view', _raises)
    with canned_prd_roots(client, tmp_path, {ROOT: prd_tree()}) as render:
        view = render(T0)['PRD_VIEW'][ROOT]
    assert view['state'] == 'unknown'
    assert 'unexpected error' in view['reason']


# --- the budgeted fan-out degrades one root, never the others --------------


def _tighten(monkeypatch: pytest.MonkeyPatch, **bounds: float) -> None:
    monkeypatch.setattr(prds, 'PRDS_BUDGET', dataclasses.replace(prds.PRDS_BUDGET, **bounds))


def test_a_root_past_its_share_is_unknown_while_the_other_renders(client, tmp_path, monkeypatch):
    _tighten(monkeypatch, per_project=2.0)
    slow = prd_tree()
    slow.projection_delay = 4.0
    with canned_prd_roots(client, tmp_path, {ROOT: prd_tree(), SLOW: slow}) as render:
        views = render(T0)['PRD_VIEW']
    assert views[SLOW]['state'] == 'unknown'
    assert views[SLOW]['reason'].startswith('exceeded its 2.0s share of the 20.0s /prds budget (')
    assert views[ROOT]['state'] == 'fresh'


def test_a_root_admitted_after_the_deadline_is_skipped_unread(client, tmp_path, monkeypatch):
    _tighten(monkeypatch, total=1e-9)
    substrate = prd_tree()
    with canned_prd_roots(client, tmp_path, {ROOT: substrate}) as render:
        view = render(T0)['PRD_VIEW'][ROOT]
    assert view['state'] == 'unknown'
    assert view['reason'].startswith('skipped — the 0.0s /prds budget was already spent')
    assert substrate.calls == []


def test_a_root_whose_acquisition_raises_is_unknown_while_the_other_renders(
    client, tmp_path, monkeypatch, caplog,
):
    acquire = prds.acquire_lineage

    async def _raises_for_broken(http_client, config, root, *, now):
        if Path(root).name == BROKEN:
            raise RuntimeError('acquisition bug')
        return await acquire(http_client, config, root, now=now)

    monkeypatch.setattr(prds, 'acquire_lineage', _raises_for_broken)
    with (
        canned_prd_roots(client, tmp_path, {ROOT: prd_tree(), BROKEN: prd_tree()}) as render,
        caplog.at_level(logging.WARNING, logger=prds.__name__),
    ):
        views = render(T0)['PRD_VIEW']
    assert views[BROKEN]['state'] == 'unknown'
    assert views[BROKEN]['reason'] == (
        "unexpected error while acquiring this project's inputs (see the WARNING and its traceback)"
    )
    assert views[ROOT]['state'] == 'fresh'
    [bug] = [record for record in caplog.records if record.name == prds.__name__]
    assert bug.exc_info is not None
    assert BROKEN in bug.getMessage()


# --- row 11: a slow lineage read never holds the response ------------------

_PROJECTION_DELAY = 2.0
_SETTLE_DEADLINE = 15.0


def _projected_reads(substrate: CannedMCP) -> list[dict]:
    return [call for call in substrate.calls_to('get_tasks') if call['args'].get('fields') is not None]


def _render_until_fresh(render: Callable[[datetime], dict[str, Any]], at: datetime) -> dict[str, Any]:
    deadline = time.monotonic() + _SETTLE_DEADLINE
    view = render(at)['PRD_VIEW'][ROOT]
    while view['state'] != 'fresh' and time.monotonic() < deadline:
        time.sleep(0.05)
        view = render(at)['PRD_VIEW'][ROOT]
    return view


def test_a_slow_lineage_read_lands_for_the_next_request(client, tmp_path, monkeypatch):
    monkeypatch.setattr(task_lineage, 'LINEAGE_REQUEST_WAIT_SECONDS', 0.05)
    slow = prd_tree()
    slow.projection_delay = _PROJECTION_DELAY
    with canned_prd_roots(client, tmp_path, {ROOT: slow}) as render:
        started = time.monotonic()
        first = render(T0)['PRD_VIEW'][ROOT]
        elapsed = time.monotonic() - started
        later = _render_until_fresh(render, T0 + timedelta(seconds=20))

    assert elapsed < _PROJECTION_DELAY
    assert first['state'] == 'unknown'
    assert first['reason'].startswith('lineage: ')
    assert later['state'] == 'fresh'
    assert [state['state'] for state in later['value']['inputs']] == ['fresh', 'fresh', 'fresh']
    assert len(_projected_reads(slow)) == 1
