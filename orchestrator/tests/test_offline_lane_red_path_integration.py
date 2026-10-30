"""Integration gate for the offline lane's red path: PRD ``plans/offline-lane-red-path-prd.md`` §6 scenarios 1-23.

STRATEGY: every scenario drives one REAL ``OfflineLaneWorker`` through its public
surface, ``on_post_merge`` and ``run``, over a real tmp repo and its real
``_offline-deep`` worktree.  The lane command is a stub ``pytest`` on PATH that plays
back one scripted sub-run per launch, so each pass's detect and confirm are real
``sh -c`` processes with real output files, junit reports, exits and signals; the
task store is fused-memory's task tools as the production typed client reaches them;
the escalation queue is real; the worker's clock is injected so passes are spaced
without sleeping.  ``_offline_lane_red_path_fixtures.py`` builds and steps the worker.

This module adds tests only: no production module changes with it.
"""

from __future__ import annotations

import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from _offline_lane_red_path_fixtures import (
    OPAQUE_COMMAND,
    A,
    B,
    C,
    RedPathLane,
    Run,
    StoreFault,
    W,
    effect_of,
    lane_test_timeout,
)

from orchestrator.offline_lane_episode_state import unconfirmable_node
from orchestrator.offline_lane_filing import HEAD_PREFIX_CHARS, SPAWN_CONTEXT

pytestmark = [pytest.mark.usefixtures('code_default_config')]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_01_a_red_and_a_green_pass_each_leave_a_record(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    red_run = Run(failed={A: 'AssertionError: a'}, errors={C: 'fixture c missing'})

    red = await lane.advance(red_run, red_run)
    red_head = await lane.head()
    green = await lane.advance(Run(passed=(A, C)))

    assert red.head == red_head
    assert green.head == await lane.head()
    for record in (red, green):
        directory = lane.project_root / 'data' / 'orchestrator' / 'offline-lane' / 'lane-fixture' / record.pass_id
        assert (directory / 'record.json').is_file()
        assert record.detect is not None
        assert record.detect.ended_by == 'exit'
        assert record.detect.started_at >= record.started_at
    assert red.detect is not None and green.detect is not None
    assert red.detect.returncode == 1
    assert green.detect.returncode == 0
    assert [(f.node, f.kind, f.first_line) for f in red.detect.failures] == [
        (A, 'failed', 'AssertionError: a'),
        (C, 'error', 'fixture c missing'),
    ]
    output = Path(red.detect.output_path)
    assert output.is_file()
    assert f'FAILED {A} - AssertionError: a' in output.read_text()
    assert green.detect.failures == ()
    assert green.outcome is None


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
@pytest.mark.parametrize('phase', ['detect', 'confirm'])
async def test_scenario_02_a_signal_ended_sub_run_is_unconfirmable(tmp_path, monkeypatch, phase):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    killed = Run(failed={A: 'AssertionError: a'}, signal=signal.SIGKILL, junit=False, complete=False)
    runs = (killed,) if phase == 'detect' else (Run(failed={A: 'AssertionError: a'}), killed)

    record = await lane.advance(*runs)

    run = record.detect if phase == 'detect' else record.confirm
    assert run is not None
    assert run.ended_by == 'signal'
    assert run.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL)
    assert record.outcome == {
        'arm': 'unconfirmable',
        'reason': f'{phase}_signal_ended',
        'detail': str(run.returncode),
    }
    if phase == 'detect':
        assert record.confirm is None


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
async def test_scenario_03_a_test_reproduced_on_pass_1_is_held_without_a_task(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    red = Run(failed={A: 'AssertionError: a'})

    record = await lane.advance(red, red)

    assert record.outcome == {
        'arm': 'confirmed', 'mode': 'narrowed', 'reproduced': [A], 'cleared': [], 'inconclusive': [],
    }
    episode = lane.episode(A)
    assert episode is not None
    assert episode.counted_passes == (record.pass_id,)
    assert lane.store.lane_tasks() == []
    confirm_argv = lane.stub.invocations()[-1]['argv']
    assert A in confirm_argv
    assert 'tests' not in confirm_argv


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_04_a_reproduction_30_minutes_later_files_exactly_one_task(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)

    first = await lane.advance(
        Run(failed={A: 'AssertionError: detect 1'}), Run(failed={A: 'AssertionError: confirm 1'}),
    )
    second = await lane.advance(
        Run(failed={A: 'AssertionError: detect 2'}), Run(failed={A: 'AssertionError: confirm 2'}),
    )

    tasks = lane.store.lane_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task['metadata']['spawn_context'] == SPAWN_CONTEXT
    assert not task['id'].startswith('tkt_')
    assert lane.store.task(task['id']) == task
    assert effect_of(first)['tasks_filed'] == []
    assert effect_of(second)['tasks_filed'] == [task['id']]
    assert task['status'] == 'pending'
    for text in (
        first.pass_id, second.pass_id,
        'AssertionError: detect 1', 'AssertionError: confirm 1',
        'AssertionError: detect 2', 'AssertionError: confirm 2',
    ):
        assert text in task['details']
    episode = lane.episode(A)
    assert episode is not None
    assert episode.covering_task_id == task['id']
    assert episode.planning_committed is True


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_05_a_reproduction_3_minutes_later_is_not_counted(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    red = Run(failed={A: 'AssertionError: a'})

    first = await lane.advance(red, red)
    second = await lane.advance(red, red, minutes=3)

    episode = lane.episode(A)
    assert episode is not None
    assert episode.counted_passes == (first.pass_id,)
    assert episode.reproduced_heads == (first.head, second.head)
    assert lane.store.lane_tasks() == []


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_06_a_red_the_narrowed_confirm_clears_ends_its_episode(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    await lane.advance(Run(failed={A: 'AssertionError: a'}), Run(failed={A: 'AssertionError: a'}))

    cleared = await lane.advance(Run(failed={A: 'AssertionError: a'}), Run(passed=(A,)))

    assert cleared.outcome == {
        'arm': 'confirmed', 'mode': 'narrowed', 'reproduced': [], 'cleared': [A], 'inconclusive': [],
    }
    assert [(ended['node'], ended['reason']) for ended in effect_of(cleared)['episodes_ended']] == [(A, 'cleared')]
    assert lane.store.lane_tasks() == []


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
async def test_scenario_07_an_opaque_command_reproduces_nothing_the_detection_did_not_name(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch, command=OPAQUE_COMMAND)

    record = await lane.advance(Run(failed={A: 'AssertionError: a'}), Run(failed={B: 'AssertionError: b'}))

    assert record.outcome is not None
    assert record.outcome['mode'] == 'intersected'
    assert record.outcome['reproduced'] == []
    assert lane.episode(B) is None
    confirm_argv = lane.stub.invocations()[-1]['argv']
    assert not any('::' in arg for arg in confirm_argv)


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=2))
async def test_scenario_08_a_confirm_exiting_non_zero_naming_no_test_files_one_command_task(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    detect = Run(failed={A: 'AssertionError: a'})
    confirm = Run(exit=2, complete=False, junit=False)

    first = await lane.advance(detect, confirm)
    second = await lane.advance(detect, confirm)

    for record in (first, second):
        assert record.outcome == {
            'arm': 'unconfirmable', 'reason': 'confirm_exited_nonzero_naming_no_test', 'detail': '2',
        }
    assert lane.episode(A) is None
    pseudo = lane.episode(unconfirmable_node('lane-fixture'))
    assert pseudo is not None
    assert pseudo.counted_passes == (first.pass_id, second.pass_id)
    tasks = lane.store.lane_tasks()
    assert [task['title'] for task in tasks] == [
        'offline-lane[lane-fixture]: exits non-zero with no reproducible failing test '
        f'since {first.head[:HEAD_PREFIX_CHARS]}',
    ]
    assert effect_of(first)['tasks_filed'] == []
    assert effect_of(second)['tasks_filed'] == [tasks[0]['id']]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
async def test_scenario_09_a_confirm_error_at_setup_leaves_a_failed_test_inconclusive(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)

    record = await lane.advance(Run(failed={A: 'AssertionError: a'}), Run(errors={A: 'fixture setup failed'}))

    assert record.outcome == {
        'arm': 'confirmed', 'mode': 'narrowed', 'reproduced': [], 'cleared': [], 'inconclusive': [A],
    }
    assert lane.episode(A) is None


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=4))
async def test_scenario_10_a_skip_and_a_truncated_session_leave_a_held_episode_unchanged(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    red = Run(failed={A: 'AssertionError: a'})
    await lane.advance(red, red)
    held = lane.episode(A)
    assert held is not None

    await lane.advance(Run(skipped=(A,)))
    await lane.advance(Run(exit=2, complete=False, junit=False))
    unchanged = lane.episode(A)
    passed = await lane.advance(Run(passed=(A,)))

    assert unchanged is not None
    for name in ('counted_passes', 'last_counted_started_at', 'reproduced_heads', 'ended_at'):
        assert getattr(unchanged, name) == getattr(held, name), name
    assert (A, 'passed') in [(ended['node'], ended['reason']) for ended in effect_of(passed)['episodes_ended']]
    ended = lane.episode(A)
    assert ended is not None
    assert ended.ended_at is not None


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=1))
async def test_scenario_11_a_failing_id_holding_whitespace_is_unconfirmable(tmp_path, monkeypatch):
    lane = await RedPathLane.open(tmp_path, monkeypatch)

    record = await lane.advance(Run(failed={W: 'AssertionError'}))

    truncated = W.split(' ')[0]
    assert record.outcome == {'arm': 'unconfirmable', 'reason': 'id_failed_round_trip', 'detail': truncated}
    assert len(lane.stub.invocations()) == 1


RED_A = Run(failed={A: 'AssertionError: a'})


async def filed_lane(tmp_path, monkeypatch) -> tuple[RedPathLane, str, datetime]:
    """A lane whose second spaced reproduction of A filed one task: the lane, the task id, the filing instant."""
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    await lane.advance(RED_A, RED_A)
    filing = await lane.advance(RED_A, RED_A)
    [task_id] = effect_of(filing)['tasks_filed']
    return lane, task_id, filing.started_at


def nodes_of(task: dict) -> list[str]:
    return task['metadata']['x_offline_lane']['nodes']


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=5))
async def test_scenario_12_tests_crossing_together_share_a_task_and_a_later_crossing_gets_its_own(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    red_ab = Run(failed={A: 'AssertionError: a', B: 'AssertionError: b'})
    red_abc = Run(failed={A: 'AssertionError: a', B: 'AssertionError: b', C: 'AssertionError: c'})

    await lane.advance(red_ab, red_ab)
    await lane.advance(red_ab, red_ab)
    assert [nodes_of(task) for task in lane.store.lane_tasks()] == [[A, B]]
    await lane.advance(red_abc, red_abc)
    await lane.advance(red_abc, red_abc)
    assert [nodes_of(task) for task in lane.store.lane_tasks()] == [[A, B], [C]]
    await lane.advance(red_abc, red_abc)

    assert len(lane.store.lane_tasks()) == 2


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=12))
async def test_scenario_13_a_persistent_red_under_an_open_task_escalates_once_at_24_hours(tmp_path, monkeypatch):
    lane, task_id, filed_at = await filed_lane(tmp_path, monkeypatch)
    raised_by: list[timedelta] = []

    for _ in range(10):
        record = await lane.advance(RED_A, RED_A, minutes=180)
        since_filing = record.started_at - filed_at
        unlanded = lane.unlanded()
        assert len(lane.store.lane_tasks()) == 1
        if since_filing < timedelta(hours=24):
            assert unlanded == [], since_filing
            continue
        assert len(unlanded) == 1, since_filing
        if not raised_by:
            raised_by.append(since_filing)
            assert f'{task_id} (pending)' in unlanded[0].summary

    assert raised_by == [timedelta(hours=24)]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=5))
async def test_scenario_14_a_covered_test_green_then_red_files_no_new_task(tmp_path, monkeypatch):
    lane, task_id, _filed_at = await filed_lane(tmp_path, monkeypatch)

    await lane.advance(Run(passed=(A,)))
    await lane.advance(RED_A, RED_A)
    await lane.advance(RED_A, RED_A)

    assert [task['id'] for task in lane.store.lane_tasks()] == [task_id]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=7))
async def test_scenario_15_a_cancelled_covering_task_escalates_once_then_releases_a_refile(tmp_path, monkeypatch):
    lane, first_id, filed_at = await filed_lane(tmp_path, monkeypatch)
    lane.store.set_status(first_id, 'cancelled')

    for hours_after_filing, raised in ((12, 0), (23, 0), (24, 1)):
        gap = filed_at + timedelta(hours=hours_after_filing) - lane.clock()
        record = await lane.advance(RED_A, RED_A, minutes=gap / timedelta(minutes=1))
        assert record.started_at - filed_at == timedelta(hours=hours_after_filing)
        assert len(lane.unlanded()) == raised
        assert [task['id'] for task in lane.store.lane_tasks()] == [first_id]
    [unlanded] = lane.unlanded()
    lane.resolve(unlanded.id)

    held = await lane.advance(RED_A, RED_A)
    episode = lane.episode(A)
    assert episode is not None
    assert episode.counted_passes == (held.pass_id,)
    assert [task['id'] for task in lane.store.lane_tasks()] == [first_id]
    await lane.advance(RED_A, RED_A)

    second = [task for task in lane.store.lane_tasks() if task['id'] != first_id]
    assert len(second) == 1
    assert second[0]['metadata']['x_offline_lane']['prior_task_ids'] == [first_id]
    assert f'{A}: {first_id}' in second[0]['details']


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=5, extra_spawns=4))
async def test_scenario_16_a_red_at_a_head_without_the_fix_has_no_effect(tmp_path, monkeypatch):
    lane, first_id, _filed_at = await filed_lane(tmp_path, monkeypatch)
    counted = lane.episode(A)
    assert counted is not None
    fix = await lane.side_commit()
    lane.store.set_status(first_id, 'done', fix_commit=fix)

    stale = await lane.advance(RED_A, RED_A)
    assert effect_of(stale)['stale_heads'] == [A]
    still = lane.episode(A)
    assert still is not None
    assert still.counted_passes == counted.counted_passes
    assert [task['id'] for task in lane.store.lane_tasks()] == [first_id]

    await lane.merge(fix)
    restarted = await lane.advance(RED_A, RED_A)
    episode = lane.episode(A)
    assert episode is not None
    assert episode.counted_passes == (restarted.pass_id,)
    await lane.advance(RED_A, RED_A)

    second = [task for task in lane.store.lane_tasks() if task['id'] != first_id]
    assert len(second) == 1
    assert second[0]['metadata']['x_offline_lane']['prior_task_ids'] == [first_id]


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=5))
async def test_scenario_17_a_done_task_without_a_fix_commit_is_contained_and_a_failed_ancestry_read_defers(
    tmp_path, monkeypatch, caplog,
):
    lane, first_id, _filed_at = await filed_lane(tmp_path, monkeypatch)
    lane.store.set_status(first_id, 'done')

    held = await lane.advance(RED_A, RED_A)
    episode = lane.episode(A)
    assert episode is not None
    assert episode.counted_passes == (held.pass_id,)
    assert episode.prior_task_ids == (first_id,)
    refiled = await lane.advance(RED_A, RED_A)
    [second_id] = effect_of(refiled)['tasks_filed']
    lane.store.set_status(second_id, 'done', fix_commit='f' * 40)

    undecided = await lane.advance(RED_A, RED_A)

    assert effect_of(undecided)['no_decision'] == [A]
    assert {task['id'] for task in lane.store.lane_tasks()} == {first_id, second_id}
    assert any(
        record.levelno == logging.ERROR and f'whether head contains the fix of task {second_id}' in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=3))
@pytest.mark.parametrize('fault', list(StoreFault))
async def test_scenario_18_a_filing_the_store_refuses_stays_due_and_lands_on_a_tick(tmp_path, monkeypatch, fault):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    await lane.advance(RED_A, RED_A)
    lane.store.degrade(fault)

    await lane.advance(RED_A, RED_A)
    due = lane.episode(A)
    assert due is not None
    assert due.filing_due is True
    assert due.filing_failure is not None
    assert due.covering_task_id is None
    assert lane.store.lane_tasks() == []
    invocations, passes = lane.stub.invocations(), lane.passes()

    lane.store.restore()
    assert await lane.tick(minutes=1) == []

    [task] = lane.store.lane_tasks()
    bound = lane.episode(A)
    assert bound is not None
    assert bound.covering_task_id == task['id']
    assert bound.planning_committed is True
    assert task['status'] == 'pending'
    assert lane.stub.invocations() == invocations
    assert lane.passes() == passes


@pytest.mark.asyncio
@pytest.mark.timeout(lane_test_timeout(drives=3))
async def test_scenario_19_a_restart_between_submit_and_commit_planning_commits_the_deferred_task(
    tmp_path, monkeypatch,
):
    lane = await RedPathLane.open(tmp_path, monkeypatch)
    await lane.advance(RED_A, RED_A)
    lane.store.park_commits()

    await lane.advance(RED_A, RED_A, until=lane.store.commit_reached)
    [task] = lane.store.lane_tasks()
    assert task['status'] == 'deferred'
    parked = lane.episode(A)
    assert parked is not None
    assert parked.covering_task_id == task['id']
    assert parked.planning_committed is False

    lane.restart()
    lane.store.release()
    assert await lane.tick(minutes=1) == []

    assert lane.store.task(task['id'])['status'] == 'pending'
    committed = lane.episode(A)
    assert committed is not None
    assert committed.planning_committed is True
    assert [name for name, _arguments in lane.store.calls].count('submit_task') == 1
