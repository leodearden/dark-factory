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
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _flake_golden import merge_gate_deflake_description
from _offline_lane_red_path_fixtures import (
    COMMIT_SPAWNS,
    NINE,
    NUMERIC_IDS,
    PROBED_COMMAND,
    RED_A,
    A,
    B,
    C,
    LegacyLeg,
    ProbeAnswer,
    RedPathLane,
    Run,
    StoreFault,
    effect_of,
    ended,
    lane_test_timeout,
    merge_gate_suppression,
    offered,
    probed_command,
    promote_to_level_2,
    reload_git,
    sync_storm_window,
)
from escalation.categories import EscalationCategory
from shared.lane_probe import OUTAGE_EXIT_STATUS

from orchestrator.flake_debt_drain import STALE_DUE_ESCALATION_SENTINEL
from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    DebtRow,
    DebtTrigger,
    DueRow,
    FlakeOccurrenceRow,
    due_item_context,
)
from orchestrator.offline_lane_episode_state import unconfirmable_node
from orchestrator.offline_lane_escalations import (
    LANE_ESCALATION_ROLE,
    OUTAGE_CATEGORY,
    PROBE_BROKEN_CATEGORY,
    UnlandedRed,
    file_unlanded_red,
    sentinel_task_id,
)
from orchestrator.offline_lane_ledger import NUMERIC_UNNAMED_CLEAR

pytestmark = [pytest.mark.usefixtures('code_default_config')]

GREEN_A = Run(passed=(A,))
CLEARS_A = (RED_A, GREEN_A)
STORM = EscalationCategory.MERGE_FLAKE_SUPPRESSION_STORM
REGRESSED = EscalationCategory.REGRESSED_AFTER_RESOLUTION


def detail_of(row: FlakeOccurrenceRow) -> dict:
    return json.loads(row.detail)


def trigger_of(item: DueRow) -> DebtTrigger:
    context = due_item_context(item)
    assert context is not None, f'due item {item.id} carries no context'
    return context.trigger


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
        assert Path(evidence['record_dir']) == lane.pass_directory(record)


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


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_27a_a_held_test_green_on_its_recheck_at_the_same_head_owes_recovered_debt(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    held = await lane.advance(RED_A, RED_A)

    [recheck] = await lane.tick(GREEN_A, minutes=30)

    assert recheck.head == held.head
    assert (A, 'passed') in ended(recheck)
    [item] = lane.due()
    assert (item.test_id, item.head, trigger_of(item)) == (A, held.head, DebtTrigger.recovered)
    assert [(row.verdict, row.merge_sha) for row in lane.rows(A)] == [('fails_in_isolation', held.head)]
    await lane.drain()
    [task] = lane.store.deflake_tasks()
    assert 'recovered on the offline lane (lane-fixture)' in task['title']


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=3))
async def test_scenario_27b_a_test_green_then_red_at_one_head_owes_recovered_debt_at_that_head(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    green = await lane.advance(GREEN_A)
    red = await lane.rerun(RED_A, RED_A)
    assert red.head == green.head

    advanced = await lane.advance(GREEN_A)

    assert advanced.head != green.head
    [item] = lane.due()
    assert (item.test_id, item.head, trigger_of(item)) == (A, green.head, DebtTrigger.recovered)
    assert [(row.verdict, row.merge_sha) for row in lane.rows(A)] == [('fails_in_isolation', red.head)]
    await lane.drain()
    debt = lane.debt(A)
    assert debt is not None
    assert [task['id'] for task in lane.store.deflake_tasks()] == [debt.owner_task_id]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_28_a_test_red_at_one_head_and_green_at_the_next_owes_no_debt(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    await lane.advance(RED_A, RED_A)

    passed = await lane.advance(GREEN_A)

    assert (A, 'passed') in ended(passed)
    assert lane.due() == []


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_28_an_expired_episode_owes_no_debt(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    await lane.advance(RED_A, RED_A)

    expired = await lane.advance(Run(passed=(C,)), minutes=25 * 60)

    assert (A, 'expired') in ended(expired)
    assert lane.due() == []


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=3))
async def test_scenario_28_a_test_green_after_an_outage_pass_owes_no_debt(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch, command=PROBED_COMMAND)
    held = await lane.advance(RED_A, RED_A)
    lane.stub.probe(ProbeAnswer(exit=OUTAGE_EXIT_STATUS, stdout='qdrant: down'))

    outage = await lane.rerun(RED_A)
    green = await lane.rerun(GREEN_A)

    assert outage.outcome is not None
    assert outage.outcome['arm'] == 'outage'
    assert green.head == held.head
    assert (A, 'passed') in ended(green)
    assert lane.due() == []


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=4))
async def test_scenario_33_a_clear_under_an_open_fix_task_owes_no_debt_until_the_task_is_done(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    await lane.advance(RED_A, RED_A)
    filing = await lane.advance(RED_A, RED_A)
    [fix_task_id] = effect_of(filing)['tasks_filed']

    covered = await lane.advance(*CLEARS_A)
    assert offered(covered) == []
    assert lane.due() == []
    lane.store.set_status(fix_task_id, 'done')
    owed = await lane.advance(*CLEARS_A)

    assert offered(owed) == [(A, 'cleared')]
    assert [item.test_id for item in lane.due()] == [A]
    await lane.drain()
    debt = lane.debt(A)
    assert debt is not None
    assert debt.owner_task_id is not None
    assert debt.owner_task_id != fix_task_id


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=3))
async def test_scenario_34_a_restart_between_the_ledger_write_and_the_episode_save_files_one_owner(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    held = await lane.advance(RED_A, RED_A)

    with lane.fail_episode_saves() as save_attempted:
        await lane.rerun(GREEN_A, until=save_attempted)
    [item] = lane.due()
    assert (item.test_id, item.head) == (A, held.head)
    live = lane.episode(A)
    assert live is not None
    assert live.ended_at is None

    lane.restart()
    await lane.drain()
    [owner] = lane.store.deflake_tasks()
    ending = await lane.rerun(GREEN_A)
    await lane.drain()

    assert (A, 'passed') in ended(ending)
    assert [task['id'] for task in lane.store.deflake_tasks()] == [owner['id']]
    assert lane.due() == []


def debt_of(lane: RedPathLane, test_id: str) -> DebtRow:
    debt = lane.debt(test_id)
    assert debt is not None, f'{test_id} owes no flake debt'
    return debt


async def owned_clear(lane: RedPathLane) -> str:
    """A cleared, then drained to its de-flake owner; the owner's id."""
    await lane.advance(*CLEARS_A)
    await lane.drain()
    owner = debt_of(lane, A).owner_task_id
    assert owner is not None
    return owner


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=6))
async def test_scenario_30_lane_clears_never_count_toward_the_merge_gates_storm_window(tmp_path, monkeypatch):
    await sync_storm_window(tmp_path / 'scratch')
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    storms: list[int] = []

    for index, node in enumerate(NINE[:6]):
        await lane.advance(Run(failed={node: 'AssertionError: n'}), Run(passed=(node,)))
        storms.append(len(lane.queued(category=STORM)))
        if index < 5:
            await lane.record_merge_gate(merge_gate_suppression(B, index), task_id='42')
            storms.append(len(lane.queued(category=STORM)))

    assert storms == [0] * 9 + [1, 1]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_31_a_clear_after_the_fix_landed_escalates_a_regression_and_files_cycle_2(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    first_owner = await owned_clear(lane)
    lane.store.set_status(first_owner, 'done', fix_commit=await lane.head())

    await lane.advance(*CLEARS_A)
    await lane.drain()

    [regression] = lane.queued(category=REGRESSED)
    assert 'an offline-lane clear' in regression.detail
    assert A in regression.summary
    debt = debt_of(lane, A)
    assert debt.open_count == 2
    assert debt.owner_task_id not in (None, first_owner)
    assert debt.owner_task_id is not None
    assert 'cycle 2' in lane.store.task(debt.owner_task_id)['title']
    await lane.drain()
    assert len(lane.queued(category=REGRESSED)) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=0))
async def test_scenario_32_the_merge_gates_own_suppressions_storm_and_file_its_worded_owner(tmp_path, monkeypatch):
    await sync_storm_window(tmp_path / 'scratch')
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    test_id = 'tests/test_m.py::test_merge'

    for index in range(5):
        await lane.record_merge_gate(merge_gate_suppression(test_id, index), task_id='42')
    await lane.drain()

    assert [(row.verdict, row.call_site, row.detail) for row in lane.rows(test_id)] == [
        ('passes_in_isolation', 'merge_gate', '{}'),
    ] * 5
    debt = debt_of(lane, test_id)
    assert debt.owner_task_id is not None
    [storm] = lane.queued(category=STORM)
    assert (storm.severity, storm.level) == ('critical', 2)
    [task] = lane.store.deflake_tasks()
    assert task['id'] == debt.owner_task_id
    assert task['title'] == (
        f'Flake debt: {test_id} — passes_in_isolation after a merge-gate red was suppressed, cycle 1'
    )
    assert task['description'] == merge_gate_deflake_description(debt)


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=3, extra_spawns=4))
async def test_scenario_40a_a_clear_at_a_head_without_the_fix_is_dropped_and_one_with_it_regresses(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    first_owner = await owned_clear(lane)
    fix = await lane.side_commit()
    lane.store.set_status(first_owner, 'done', fix_commit=fix)

    await lane.advance(*CLEARS_A)
    await lane.drain()
    assert len(lane.rows(A)) == 2
    assert debt_of(lane, A).open_count == 1
    assert lane.queued(category=REGRESSED) == []
    assert [task['id'] for task in lane.store.deflake_tasks()] == [first_owner]

    await lane.merge(fix)
    await lane.advance(*CLEARS_A)
    await lane.drain()

    debt = debt_of(lane, A)
    assert debt.open_count == 2
    assert len(lane.queued(category=REGRESSED)) == 1
    assert debt.owner_task_id is not None
    assert debt.owner_task_id != first_owner
    assert 'cycle 2' in lane.store.task(debt.owner_task_id)['title']


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2, extra_spawns=1))
async def test_scenario_40b_an_item_due_before_its_owner_finished_with_a_fix_it_lacks_is_dropped(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    first_owner = await owned_clear(lane)
    fix = await lane.side_commit()
    await lane.advance(*CLEARS_A)
    assert [item.test_id for item in lane.due()] == [A]

    lane.store.set_status(first_owner, 'done', fix_commit=fix)
    await lane.drain()

    assert lane.due() == []
    assert debt_of(lane, A).open_count == 1
    assert lane.queued(category=REGRESSED) == []
    assert [task['id'] for task in lane.store.deflake_tasks()] == [first_owner]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
async def test_scenario_41_a_store_outage_leaves_both_call_sites_items_due_backed_off_and_surfaced(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    lane.store.degrade(StoreFault.TRANSPORT_DOWN)
    mark = len(lane.store.calls)
    await lane.advance(*CLEARS_A)
    assert lane.store.calls_since(mark) == []
    await lane.record_merge_gate(merge_gate_suppression(B, 0), task_id='42')
    assert lane.store.calls_since(mark) == []
    t0 = datetime.now(UTC)

    attempts = {A: 0, B: 0}
    delays: dict[str, list[timedelta]] = {A: [], B: []}
    for minutes in (0, 1, 6, 7):
        at = t0 + timedelta(minutes=minutes)
        await lane.drain_at(at)
        due = lane.due()
        assert sorted(item.test_id for item in due) == [A, B]
        for item in due:
            if item.attempts > attempts[item.test_id]:
                attempts[item.test_id] = item.attempts
                delays[item.test_id].append(datetime.fromisoformat(item.next_attempt_at) - at)
    assert min(attempts.values()) >= 2
    for test_id in (A, B):
        assert delays[test_id][1] > delays[test_id][0], test_id
    assert lane.queued(task_id=STALE_DUE_ESCALATION_SENTINEL) == []

    await lane.drain_at(t0 + timedelta(minutes=31))
    [stale] = lane.queued(task_id=STALE_DUE_ESCALATION_SENTINEL)
    assert A in stale.detail
    assert B in stale.detail

    lane.store.restore()
    settle_at = max(datetime.fromisoformat(item.next_attempt_at) for item in lane.due()) + timedelta(minutes=1)
    await lane.drain_at(settle_at)
    assert all(debt_of(lane, test_id).owner_task_id is not None for test_id in (A, B))
    await lane.drain_at(settle_at + timedelta(minutes=1))
    resolved = lane.get(stale.id)
    assert resolved is not None
    assert resolved.status == 'resolved'


OUTAGE = ProbeAnswer(exit=OUTAGE_EXIT_STATUS, stdout='qdrant: down')
PROBED_SENTINEL = sentinel_task_id(PROBED_COMMAND.name)


async def in_outage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RedPathLane:
    """A probed lane holding A from a pass whose probe answered up, then two spaced passes in an outage."""
    lane = await RedPathLane.open(tmp_path, monkeypatch, command=PROBED_COMMAND)
    await lane.advance(RED_A, RED_A)
    for _ in range(2):
        lane.stub.probe(OUTAGE)
        await lane.advance(RED_A)
    return lane


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=3))
async def test_scenario_35_outage_passes_skip_the_confirm_leave_the_episode_and_keep_one_record(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch, command=PROBED_COMMAND)
    await lane.advance(RED_A, RED_A)
    held = lane.episode(A)
    assert held is not None
    rows = lane.rows()
    opened: list[str] = []

    for _ in range(2):
        invocations = len(lane.stub.invocations())
        lane.stub.probe(OUTAGE)
        record = await lane.advance(RED_A)
        assert record.outcome is not None
        assert record.outcome['arm'] == 'outage'
        assert len(lane.stub.invocations()) == invocations + 1
        [outage] = lane.pending(OUTAGE_CATEGORY)
        opened.append(outage.id)

    assert opened[0] == opened[1]
    assert (outage.agent_role, outage.level) == (LANE_ESCALATION_ROLE, 1)
    assert 'qdrant: down' in outage.detail
    assert outage.amendments == []
    episode = lane.episode(A)
    assert episode is not None
    assert episode.had_outage_pass is True
    for name in ('counted_passes', 'last_counted_started_at', 'ended_at'):
        assert getattr(episode, name) == getattr(held, name), name
    assert lane.rows() == rows
    assert lane.due() == []
    assert lane.store.lane_tasks() == []


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=6))
async def test_scenario_36_the_idle_probe_loop_follows_a_reload_and_an_up_closes_the_record_and_its_level_2(
    tmp_path, monkeypatch,
):
    lane = await in_outage(tmp_path, monkeypatch)
    [l1] = lane.pending(OUTAGE_CATEGORY)
    probes, invocations = len(lane.stub.probe_invocations()), len(lane.stub.invocations())

    lane.stub.probe(OUTAGE)
    assert await lane.tick(minutes=30) == []
    assert len(lane.stub.probe_invocations()) == probes + 1
    assert len(lane.stub.invocations()) == invocations
    l2 = await promote_to_level_2(lane.queue, PROBED_SENTINEL, l1.id)
    report = reload_git(lane, offline_lane_recheck_secs=600)
    assert 'git.offline_lane_recheck_secs' in report['applied']
    lane.stub.probe(OUTAGE)
    assert await lane.tick(minutes=10) == []
    assert len(lane.stub.probe_invocations()) == probes + 2

    [recheck] = await lane.tick(GREEN_A, minutes=10)

    assert recheck.started_at == lane.clock()
    assert len(lane.stub.probe_invocations()) == probes + 3
    for escalation_id in (l1.id, l2):
        closed = lane.get(escalation_id)
        assert closed is not None
        assert (closed.status, closed.resolved_by) == ('resolved', LANE_ESCALATION_ROLE)


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=5))
async def test_scenario_36_a_level_2_holding_another_member_is_amended_not_resolved_when_the_probe_passes(
    tmp_path, monkeypatch,
):
    lane = await in_outage(tmp_path, monkeypatch)
    [l1] = lane.pending(OUTAGE_CATEGORY)
    other = file_unlanded_red(
        lane.queue, UnlandedRed(PROBED_COMMAND.name, '9001', 'pending', (A,), lane.clock(), ()),
    )
    l2 = await promote_to_level_2(lane.queue, PROBED_SENTINEL, l1.id)
    assert await promote_to_level_2(lane.queue, PROBED_SENTINEL, other) == l2

    await lane.tick(minutes=30)
    amended = lane.get(l2)
    await lane.tick(minutes=1)

    holder = lane.get(l2)
    assert amended is not None
    assert holder is not None
    assert holder.status == 'pending'
    assert len([a for a in holder.amendments if f'closed its record {l1.id}' in a['detail']]) == 1
    assert holder.amendments == amended.amendments, 'a tick after the close amended the level-2 again'
    for escalation_id, status in ((other, 'pending'), (l1.id, 'resolved')):
        member = lane.get(escalation_id)
        assert member is not None
        assert member.status == status, escalation_id


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=5))
async def test_scenario_37_a_hand_resolved_outage_record_is_replaced_by_the_next_failing_pass_not_the_tick(
    tmp_path, monkeypatch,
):
    lane = await in_outage(tmp_path, monkeypatch)
    [old] = lane.pending(OUTAGE_CATEGORY)
    lane.resolve(old.id)
    probes = len(lane.stub.probe_invocations())

    lane.stub.probe(OUTAGE)
    await lane.tick(minutes=30)
    assert len(lane.stub.probe_invocations()) == probes + 1
    assert lane.pending(OUTAGE_CATEGORY) == []
    lane.stub.probe(OUTAGE)
    record = await lane.advance(RED_A)

    assert record.outcome is not None
    assert record.outcome['arm'] == 'outage'
    [new] = lane.pending(OUTAGE_CATEGORY)
    assert new.id != old.id


BROKEN_PROBES = {
    'killed': (ProbeAnswer(signal=signal.SIGKILL), None),
    'another exit code': (ProbeAnswer(exit=3), None),
    'the outage exit printing nothing': (ProbeAnswer(exit=OUTAGE_EXIT_STATUS, stdout=''), None),
    'past its timeout': (ProbeAnswer(sleep_secs=5), 1),
}


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
@pytest.mark.parametrize(('answer', 'timeout_secs'), BROKEN_PROBES.values(), ids=list(BROKEN_PROBES))
async def test_scenario_38_a_broken_probe_is_confirmed_as_if_none_were_declared_under_one_record(
    tmp_path, monkeypatch, answer, timeout_secs,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch, command=probed_command(timeout_secs))
    lane.stub.probe(answer)

    record = await lane.advance(RED_A, RED_A)

    assert record.outcome == {
        'arm': 'confirmed', 'mode': 'narrowed', 'reproduced': [A], 'cleared': [], 'inconclusive': [],
    }
    assert len(lane.stub.invocations()) == 2
    assert len(lane.stub.probe_invocations()) == 1
    episode = lane.episode(A)
    assert episode is not None
    assert episode.counted_passes == (record.pass_id,)
    [broken] = lane.pending(PROBE_BROKEN_CATEGORY)
    assert broken.level == 1


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
async def test_scenario_38_a_command_with_no_probe_is_confirmed_and_keeps_no_probe_record(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)

    record = await lane.advance(RED_A, RED_A)

    assert record.outcome is not None
    assert record.outcome['reproduced'] == [A]
    assert len(lane.stub.invocations()) == 2
    assert lane.stub.probe_invocations() == []
    assert lane.pending(PROBE_BROKEN_CATEGORY) == []
