"""Tests for ``GET /api/v2/dashboard/prds``: the per-root PRD view and contention line.

Boundary rows 8, 9, 11 and 13–19 at the endpoint (PRD dashboard-prd-lanes §4.4,
§5.6), driven over the ``prd_view`` and ``prd_lanes`` families of
``_boundary_payloads`` and their CannedMCP substrates. The pure halves live in
``test_prd_view.py``, ``test_prd_lanes.py`` and ``test_prd_contention.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from _boundary_payloads import (
    LANES_ROOT,
    LOCK_DEPTH,
    ORPHAN,
    OVERLAY_ROW,
    PRD_C,
    QUEUELESS_ROOT,
    SCHEDULER_SNAPSHOT_AT,
    T0,
    canned_prd_roots,
    lane_title,
    prd_lanes_family,
    prd_tree,
    prd_view_family,
)
from _canned_mcp import CannedMCP, is_metadata_projection
from _prd_lanes_tree import prd_key
from _prd_tree import PRD_A
from fastapi.testclient import TestClient
from shared.task_statuses import TaskStatus

from dashboard.api import prds
from dashboard.data import escalation_corpus, served_read, task_lineage
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
    """Both families served at once, as ``build_all`` serves them: a family built later would
    read a run-ledger pool an intervening ``client`` test's lifespan already closed."""
    return {
        **prd_view_family(_client, tmp_path_factory.mktemp('prd_view')),
        **prd_lanes_family(_client, tmp_path_factory.mktemp('prd_lanes')),
    }


def _row(view: dict[str, Any], key: str) -> dict[str, Any]:
    [row] = [row for row in view['value']['prds'] if row['key'] == key]
    return row


def _open_ids_listed(value: dict[str, Any]) -> list[int]:
    return [
        *(task_id for row in value['prds'] for task_id in row['direct_open_ids']),
        *(followup['task_id'] for row in value['prds'] for followup in row['followup_open']),
        *(task_id for ids in value['dark_matter']['members'].values() for task_id in ids),
    ]


def test_the_body_carries_both_datums_and_its_serving_instant(bodies):
    assert set(bodies['prd_view_fresh']) == {'PRD_VIEW', 'PRD_CONTENTION', 'served_at'}


def test_both_datums_are_keyed_by_every_configured_root(bodies):
    assert set(bodies['prd_view_fresh']['PRD_VIEW']) == {ROOT, 'reify'}
    assert set(bodies['prd_view_fresh']['PRD_CONTENTION']) == {ROOT, 'reify'}


def test_a_quiet_root_serves_a_fresh_contention_line(bodies):
    contention = bodies['prd_view_fresh']['PRD_CONTENTION'][ROOT]
    assert contention['state'] == 'fresh', contention['reason']


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
    slow.metadata_projection_delay = 4.0
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


def _lineage_reads(substrate: CannedMCP) -> list[dict]:
    return [call for call in substrate.calls_to('get_tasks') if is_metadata_projection(call['args'])]


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
    slow.metadata_projection_delay = _PROJECTION_DELAY
    with canned_prd_roots(client, tmp_path, {ROOT: slow}) as render:
        first = render(T0)['PRD_VIEW'][ROOT]
        later = _render_until_fresh(render, T0 + timedelta(seconds=20))

    assert first['state'] == 'unknown'
    assert first['reason'].startswith('lineage: ')
    assert later['state'] == 'fresh'
    assert {state['state'] for state in later['value']['inputs']} == {'fresh'}
    assert len(_lineage_reads(slow)) == 1


# --- the request-wide reads are bounded and guarded -------------------------

_REQUEST_READS_WAIT = 0.5
"""Short, yet long enough for the run-ledger walk the same wait bounds."""


def _unreadable_walk(queues):
    raise OSError('queue directory unreadable')


def test_a_corpus_walk_that_raises_degrades_only_the_view(client, tmp_path, monkeypatch):
    monkeypatch.setattr(escalation_corpus, 'walk_corpus', _unreadable_walk)
    with canned_prd_roots(client, tmp_path, {ROOT: prd_tree()}) as render:
        body = render(T0)
    view = body['PRD_VIEW'][ROOT]
    assert view['state'] == 'unknown'
    assert 'escalations: ' in view['reason']
    assert 'OSError' in view['reason']
    assert body['PRD_CONTENTION'][ROOT]['state'] == 'fresh'


def test_a_slow_corpus_walk_never_holds_the_response_and_lands_for_the_next_request(
    client, tmp_path, monkeypatch,
):
    """The walk is released only once the first response is back, so that response cannot have waited on it."""
    monkeypatch.setattr(prds, 'REQUEST_READS_WAIT_SECONDS', _REQUEST_READS_WAIT)
    walk = escalation_corpus.walk_corpus
    answered = threading.Event()

    def _walk_after_the_first_response(queues):
        answered.wait(_SETTLE_DEADLINE)
        return walk(queues)

    monkeypatch.setattr(escalation_corpus, 'walk_corpus', _walk_after_the_first_response)
    with canned_prd_roots(client, tmp_path, {ROOT: prd_tree()}) as render:
        try:
            first = render(T0)
        finally:
            answered.set()
        later = _render_until_fresh(render, T0 + timedelta(seconds=20))

    assert first['PRD_VIEW'][ROOT]['state'] == 'unknown'
    assert 'escalations: ' in first['PRD_VIEW'][ROOT]['reason']
    assert first['PRD_CONTENTION'][ROOT]['state'] == 'fresh'
    assert later['state'] == 'fresh'
    assert {state['state'] for state in later['value']['inputs']} == {'fresh'}


def test_a_failed_corpus_walk_after_a_good_one_serves_the_last_good_walk(
    client, tmp_path, monkeypatch,
):
    monkeypatch.setattr(escalation_corpus, 'CORPUS_TTL_SECONDS', 0.0)
    with canned_prd_roots(client, tmp_path, {ROOT: prd_tree()}) as render:
        first = render(T0)['PRD_VIEW'][ROOT]
        monkeypatch.setattr(escalation_corpus, 'walk_corpus', _unreadable_walk)
        second = render(T0 + timedelta(seconds=1))['PRD_VIEW'][ROOT]

    assert first['state'] == 'fresh', first['reason']
    assert second['state'] == 'stale'
    assert _input(second, 'escalations')['state'] == 'stale'
    assert _row(second, PRD_A)


def _logged_raising(caplog: pytest.LogCaptureFixture, kind: type[BaseException]) -> logging.LogRecord | None:
    deadline = time.monotonic() + _SETTLE_DEADLINE
    while time.monotonic() < deadline:
        for record in caplog.records:
            if record.exc_info is not None and isinstance(record.exc_info[1], kind):
                return record
        time.sleep(0.05)
    return None


class _LatePoolBug(RuntimeError):
    pass


def test_a_run_ledger_walk_that_raises_after_its_wait_is_still_logged_as_a_bug(
    client, tmp_path, monkeypatch, caplog,
):
    monkeypatch.setattr(prds, 'REQUEST_READS_WAIT_SECONDS', _REQUEST_READS_WAIT)
    answered = threading.Event()

    async def _raises_after_the_response(config, pool, relative):
        await asyncio.to_thread(answered.wait, _SETTLE_DEADLINE)
        raise _LatePoolBug('pool bug')

    monkeypatch.setattr(prds, '_project_scoped_dbs_labeled', _raises_after_the_response)
    with (
        canned_prd_roots(client, tmp_path, {ROOT: prd_tree()}) as render,
        caplog.at_level(logging.WARNING, logger=served_read.__name__),
    ):
        try:
            render(T0)
        finally:
            answered.set()
        bug = _logged_raising(caplog, _LatePoolBug)

    assert bug is not None
    assert 'run-ledger walk' in bug.getMessage()
    assert 'BUG' in bug.getMessage()


def test_a_run_ledger_walk_that_raises_degrades_the_activity_input_not_the_response(
    client, tmp_path, monkeypatch,
):
    """project_dbs.py's docstring declares its helpers' underscore vestigial, so this patches a public name."""
    async def _raises(config, pool, relative):
        raise RuntimeError('pool bug')

    monkeypatch.setattr(prds, '_project_scoped_dbs_labeled', _raises)
    with canned_prd_roots(client, tmp_path, {ROOT: prd_tree()}) as render:
        body = render(T0)
    for datum in (body['PRD_VIEW'][ROOT], body['PRD_CONTENTION'][ROOT]):
        assert 'activity: ' in datum['reason']
        assert 'run-ledger walk' in datum['reason']
        assert 'no runs.db' not in datum['reason']


# --- rows 13–19 end to end: the lanes tree through every input --------------


_EXPECTED_LANES = {
    'ruling_over_tail': 'ruling',
    'gate_deps_met': 'ruling',
    'level2_escalation': 'ruling',
    'level1_escalation': 'stuck',
    'tail_over_moving': 'tail',
    'parked_over_moving': 'parked',
    'parked_caller_kinds': 'parked',
    'held_by_dead_session': 'stuck',
    'carried_by_done_carrier': 'stuck',
    'condition_stale': 'stuck',
    'invalid_record': 'stuck',
    'legacy_record': 'stuck',
    'gate_external_deps': 'stuck',
    'moving_over_stuck': 'moving',
    'started_13d': 'moving',
    'started_15d': 'stuck',
}


def _lanes_view(body: dict[str, Any]) -> dict[str, Any]:
    return body['PRD_VIEW'][LANES_ROOT]


def _lanes_row(body: dict[str, Any], name: str) -> dict[str, Any]:
    return _row(_lanes_view(body), prd_key(name))


def _input(view: dict[str, Any], name: str) -> dict[str, Any]:
    [state] = [state for state in view['value']['inputs'] if state['name'] == name]
    return state


def test_every_lanes_fixture_prd_is_served_in_its_lane(bodies):
    view = _lanes_view(bodies['prd_lanes'])
    assert view['state'] == 'fresh', view['reason']
    served = {name: _lanes_row(bodies['prd_lanes'], name)['lane'] for name in _EXPECTED_LANES}
    assert served == _EXPECTED_LANES


def test_a_deferral_whose_holder_is_dead_is_served_as_needing_a_human(bodies):
    held = _lanes_row(bodies['prd_lanes'], 'held_by_dead_session')
    assert held['state_line'].startswith('1 deferral needs a human: #311')
    assert held['detail']['needs_human_refs'] == [{'task_id': 311, 'reason': 'holder_dead'}]


def test_the_served_lanes_partition_the_served_rows(bodies):
    value = _lanes_view(bodies['prd_lanes'])['value']
    listed = Counter(key for entry in value['lanes'] for key in entry['prd_keys'])
    assert listed == Counter(row['key'] for row in value['prds'])


def test_rule_on_names_the_level2_escalation_from_the_projects_own_queue(bodies):
    assert _lanes_row(bodies['prd_lanes'], 'ruling_over_tail')['next_action'] == {
        'kind': 'rule_on',
        'target': {'task_id': 102, 'escalation_id': 'esc-102-1', 'prd': None},
        'text': 'rule on esc-102-1 (#102)',
    }


def test_the_escalations_input_is_fresh_despite_another_roots_missing_queue(bodies):
    body = bodies['prd_lanes']
    assert _input(_lanes_view(body), 'escalations')['state'] == 'fresh'
    queueless = body['PRD_VIEW'][QUEUELESS_ROOT]
    assert _input(queueless, 'escalations')['state'] == 'lower_bound'


def test_signal_marks_and_titles_read_each_prds_own_files(bodies):
    body = bodies['prd_lanes']
    marks = {
        name: _lanes_row(body, name)['signal']['mark']
        for name in ('ruling_over_tail', 'tail_over_moving', 'started_13d')
    }
    assert marks == {'ruling_over_tail': 'delivered', 'tail_over_moving': 'delivered', 'started_13d': 'unknown'}
    assert _lanes_row(body, 'started_13d')['signal']['cause'] == 'no_sidecar'
    assert _lanes_row(body, 'started_13d')['title'] == lane_title(prd_key('started_13d'))


def test_the_contention_line_serves_the_scheduler_files_own_stamp(bodies):
    contention = bodies['prd_lanes']['PRD_CONTENTION'][LANES_ROOT]
    assert contention['state'] == 'fresh', contention['reason']
    assert contention['value']['snapshot_at'] == SCHEDULER_SNAPSHOT_AT.isoformat()
    assert contention['value']['lock_depth'] == LOCK_DEPTH


def test_never_started_reads_task_started_from_the_projects_run_ledger(bodies):
    never_started = bodies['prd_lanes']['PRD_CONTENTION'][LANES_ROOT]['value']['never_started']
    assert never_started == {
        'count': 2, 'of': 2, 'ids': [113, 371],
        'window_start': (T0 - timedelta(days=20)).isoformat(),
    }


def test_an_unreachable_scheduler_degrades_only_the_contention_line(bodies):
    offline = bodies['prd_lanes_scheduler_offline']
    contention = offline['PRD_CONTENTION'][LANES_ROOT]
    assert contention['state'] == 'unknown'
    assert 'get_scheduler_state' in contention['reason']
    assert offline['PRD_VIEW'][LANES_ROOT] == bodies['prd_lanes']['PRD_VIEW'][LANES_ROOT]


def test_a_failed_lineage_read_degrades_only_the_view(bodies):
    failed = bodies['prd_lanes_lineage_failed']
    view = failed['PRD_VIEW'][LANES_ROOT]
    assert view['state'] == 'unknown'
    assert view['reason'].startswith('lineage: ')
    assert failed['PRD_CONTENTION'][LANES_ROOT] == bodies['prd_lanes']['PRD_CONTENTION'][LANES_ROOT]


def test_an_unreadable_run_ledger_makes_the_activity_input_unknown(bodies):
    body = bodies['prd_lanes_runs_db_unreadable']
    view = body['PRD_VIEW'][LANES_ROOT]
    assert view['state'] == 'unknown'
    assert 'activity: ' in view['reason']
    assert 'activity: ' in body['PRD_CONTENTION'][LANES_ROOT]['reason']
