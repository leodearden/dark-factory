"""Integration gate for the offline lane's ledger and outage paths: PRD ``plans/offline-lane-red-path-prd.md`` §6 scenarios 25-38, 40 and 41.

STRATEGY: as in ``test_offline_lane_red_path_integration.py``, every scenario drives one
REAL ``OfflineLaneWorker`` through ``on_post_merge`` and ``run`` over a real tmp repo,
a stub test runner on PATH, fused-memory's task tools as the production typed client
reaches them, a real escalation queue and an injected clock.  Here the flake ledger is
the real one in the lane's own ``runs.db``, its de-flake owners are filed by the real
drain over the same task store, and the merge gate's side is recorded through the
production recorder.  The passes a scenario reuses from scenarios 1-23 are re-driven
through the shared builders in ``_offline_lane_red_path_fixtures.py``.

This module adds tests only: no production module changes with it.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from _offline_lane_red_path_fixtures import (
    COMMIT_SPAWNS,
    NINE,
    A,
    C,
    LegacyLeg,
    RedPathLane,
    Run,
    lane_test_timeout,
    merge_gate_suppression,
)

from orchestrator.flake_ledger import UNKNOWN_TEST_ID, FlakeOccurrenceRow
from orchestrator.offline_lane_episode_state import unconfirmable_node
from orchestrator.offline_lane_ledger import NUMERIC_UNNAMED_CLEAR
from orchestrator.offline_lane_records import PassRecord

pytestmark = [pytest.mark.usefixtures('code_default_config')]

RED_A = Run(failed={A: 'AssertionError: a'})
NUMERIC_IDS = ('numeric_suite::case_1', 'numeric_suite::case_2')


def detail_of(row: FlakeOccurrenceRow) -> dict:
    return json.loads(row.detail)


def pass_directory(lane: RedPathLane, record: PassRecord) -> Path:
    return lane.project_root / 'data' / 'orchestrator' / 'offline-lane' / 'lane-fixture' / record.pass_id


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_25_scenario_4s_reproductions_write_fails_in_isolation_rows(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)

    first = await lane.advance(
        Run(failed={A: 'AssertionError: detect 1'}), Run(failed={A: 'AssertionError: confirm 1'}),
    )
    second = await lane.advance(
        Run(failed={A: 'AssertionError: detect 2'}), Run(failed={A: 'AssertionError: confirm 2'}),
    )

    rows = lane.rows(A)
    assert [(row.verdict, row.call_site, row.merge_sha, row.task_id, row.runner) for row in rows] == [
        ('fails_in_isolation', 'offline_lane', first.head, None, 'local'),
        ('fails_in_isolation', 'offline_lane', second.head, None, 'local'),
    ]
    detected = ('AssertionError: detect 1', 'AssertionError: detect 2')
    for row, record, line in zip(rows, (first, second), detected, strict=True):
        evidence = detail_of(row)['evidence']
        assert (evidence['kind'], evidence['first_line']) == ('failed', line)
        assert Path(evidence['record_dir']) == pass_directory(lane, record)


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_25_scenario_6s_clear_writes_a_passes_in_isolation_row(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    held = await lane.advance(RED_A, RED_A)

    cleared = await lane.advance(Run(failed={A: 'AssertionError: a again'}), Run(passed=(A,)))

    rows = lane.rows(A)
    assert [(row.verdict, row.merge_sha) for row in rows] == [
        ('fails_in_isolation', held.head), ('passes_in_isolation', cleared.head),
    ]
    assert detail_of(rows[1])['evidence']['first_line'] == 'AssertionError: a again'


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_25_scenario_8s_unconfirmable_passes_write_unknown_test_rows(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    confirm = Run(exit=2, complete=False, junit=False)

    first = await lane.advance(RED_A, confirm)
    second = await lane.advance(RED_A, confirm)

    rows = lane.rows()
    assert [(row.test_id, row.verdict, row.merge_sha) for row in rows] == [
        (UNKNOWN_TEST_ID, 'unconfirmable', first.head), (UNKNOWN_TEST_ID, 'unconfirmable', second.head),
    ]
    assert [detail_of(row) for row in rows] == [{'unconfirmable_reason': 'confirm_exited_nonzero_naming_no_test'}] * 2
    assert lane.rows(unconfirmable_node('lane-fixture')) == []


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
async def test_scenario_25_scenario_9s_inconclusive_writes_an_unconfirmable_row_naming_the_test(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)

    await lane.advance(RED_A, Run(errors={A: 'fixture setup failed'}))

    [row] = lane.rows()
    assert (row.test_id, row.verdict) == (A, 'unconfirmable')
    assert detail_of(row)['unconfirmable_reason'] == 'confirm_kind_differs'


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=3, extra_spawns=COMMIT_SPAWNS))
async def test_scenario_25_scenario_22as_numeric_passes_write_their_own_rows(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch, legacy=LegacyLeg.NUMERIC)
    detect = Run(exit=1, printed=('numeric suite: 2 failures',), complete=False, junit=False)

    await lane.advance(detect, Run(printed=NUMERIC_IDS, exit=100, complete=False, junit=False))
    await lane.advance(detect, Run(exit=0, complete=False, junit=False))
    await lane.advance(detect, Run(exit=64, complete=False, junit=False))

    for test_id in NUMERIC_IDS:
        assert [row.verdict for row in lane.rows(test_id)] == ['fails_in_isolation']
    unnamed = lane.rows(UNKNOWN_TEST_ID)
    assert [(row.verdict, detail_of(row)['unconfirmable_reason']) for row in unnamed] == [
        ('unconfirmable', NUMERIC_UNNAMED_CLEAR), ('unconfirmable', 'numeric_exit'),
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_25_the_lanes_rows_leave_the_merge_gates_report_counters_unchanged(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    await lane.record_merge_gate(merge_gate_suppression(C, 0), task_id='42')
    now = lane.clock() + timedelta(hours=2)
    before = lane.report(now)

    await lane.advance(RED_A, RED_A)
    await lane.advance(RED_A, Run(exit=2, complete=False, junit=False))
    after = lane.report(now)

    assert after.gate_blind_by_call_site['merge_gate'] == before.gate_blind_by_call_site['merge_gate']
    assert after.systemic_by_call_site['merge_gate'] == before.systemic_by_call_site['merge_gate']
    assert before.call_sites_written == frozenset({'merge_gate'})
    assert after.call_sites_written == frozenset({'merge_gate', 'offline_lane'})


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_26_a_first_clear_owes_debt_and_the_drain_files_its_owner_once(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)

    cleared = await lane.advance(Run(failed={A: 'AssertionError: a first'}), Run(passed=(A,)))
    [item] = lane.due()
    assert (item.test_id, item.call_site, item.head) == (A, 'offline_lane', cleared.head)
    await lane.drain()

    debt = lane.debt(A)
    assert debt is not None
    assert debt.owner_task_id is not None
    [task] = lane.store.deflake_tasks()
    assert task['id'] == debt.owner_task_id
    assert 'cleared on the offline lane (lane-fixture)' in task['title']
    assert 'cycle 1' in task['title']
    assert 'AssertionError: a first' in task['description']
    [reproduce] = [line for line in task['description'].splitlines() if line.strip().startswith('reproduce:')]
    assert A in reproduce
    metadata = task['metadata']
    assert (metadata['flake_debt_call_site'], metadata['flake_debt_trigger'], metadata['flake_debt_command']) == (
        'offline_lane', 'cleared', 'lane-fixture',
    )

    await lane.advance(Run(failed={A: 'AssertionError: a second'}), Run(passed=(A,)))
    await lane.drain()

    assert [owner['id'] for owner in lane.store.deflake_tasks()] == [task['id']]
    assert lane.due() == []


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
async def test_scenario_29_nine_clears_in_one_pass_call_no_store_and_drain_to_nine_owners(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    mark = len(lane.store.calls)

    await lane.advance(
        Run(failed={node: f'AssertionError: {index}' for index, node in enumerate(NINE)}), Run(passed=NINE),
    )

    assert lane.store.calls_since(mark) == []
    assert sorted(item.test_id for item in lane.due()) == sorted(NINE)
    await lane.drain()
    assert sorted(task['metadata']['flake_debt_test'] for task in lane.store.deflake_tasks()) == sorted(NINE)
    assert all(lane.debt(node) is not None for node in NINE)
