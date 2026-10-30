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

import signal
from pathlib import Path

import pytest
from _offline_lane_red_path_fixtures import (
    OPAQUE_COMMAND,
    A,
    B,
    C,
    RedPathLane,
    Run,
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
