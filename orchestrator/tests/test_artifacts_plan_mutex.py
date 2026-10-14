"""``TaskArtifacts.plan_mutex``: every plan.json read-modify-write is one critical section.

Each race runs a writer on a second thread against a holder already inside the
mutex (``_plan_mutex_race``): a writer that does not wait is clobbered by the
holder's stale write, so the final plan carrying BOTH writes is the property.
"""
from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _plan_mutex_race import (
    HOLDER_MARKER_KEY,
    PROBE_SECS,
    mark_holder,
    race_writer_against_held_mutex,
)

from orchestrator.artifacts import PLAN_SCHEMA_VERSION, TaskArtifacts

_JOIN_SECS = 10.0


def _seed_plan() -> dict:
    return {
        'files': ['a.py'],
        'prerequisites': [],
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'first',
             'status': 'pending', 'commit': None},
            {'id': 'step-2', 'type': 'impl', 'description': 'second',
             'status': 'pending', 'commit': None},
        ],
    }


@pytest.fixture
def artifacts(tmp_path: Path) -> TaskArtifacts:
    worktree = tmp_path / 'worktree'
    worktree.mkdir()
    ta = TaskArtifacts(worktree)
    ta.init('task-1', 'Test Task', 'A test task description')
    ta.write_plan(_seed_plan())
    return ta


def _step_1(plan: dict) -> dict:
    return next(step for step in plan['steps'] if step['id'] == 'step-1')


@pytest.mark.parametrize(
    ('write', 'effect', 'expected'),
    [
        pytest.param(
            lambda ta: ta.update_step_status('step-1', 'done', commit='abc'),
            lambda plan: (_step_1(plan)['status'], _step_1(plan)['commit']),
            ('done', 'abc'),
            id='update_step_status',
        ),
        pytest.param(
            lambda ta: ta.mark_step_committed('step-1', 'a' * 40),
            lambda plan: (_step_1(plan)['status'], _step_1(plan)['description']),
            ('done', '[COMMITTED aaaaaaaaaaaa] first'),
            id='mark_step_committed',
        ),
        pytest.param(
            lambda ta: ta.set_plan_files(['a.py', 'b.py'], 's1'),
            lambda plan: plan['files'],
            ['a.py', 'b.py'],
            id='set_plan_files',
        ),
        pytest.param(
            lambda ta: ta.bump_revalidation_stamp('s1'),
            lambda plan: plan.get('_revalidated_by_session'),
            's1',
            id='bump_revalidation_stamp',
        ),
        pytest.param(
            lambda ta: ta.stamp_plan_provenance('s1'),
            lambda plan: plan.get('_session_id'),
            's1',
            id='stamp_plan_provenance',
        ),
    ],
)
def test_a_plan_rmw_method_waits_for_the_mutex_and_both_writes_survive(
    artifacts: TaskArtifacts,
    write: Callable[[TaskArtifacts], object],
    effect: Callable[[dict], Any],
    expected: Any,
) -> None:
    outcome = race_writer_against_held_mutex(artifacts, lambda: write(artifacts))

    assert outcome.blocked, 'the writer did not wait for the held plan mutex'
    assert outcome.writer_error is None
    assert outcome.plan.get(HOLDER_MARKER_KEY) is True, 'the holder write was lost'
    assert effect(outcome.plan) == expected, 'the writer write was lost'


def test_a_bare_overwrite_is_ordered_after_an_in_flight_rmw(artifacts: TaskArtifacts) -> None:
    replacement = {'title': 'replacement', 'files': [], 'steps': []}

    outcome = race_writer_against_held_mutex(
        artifacts, lambda: artifacts.write_plan(dict(replacement)),
    )

    assert outcome.blocked
    assert outcome.writer_error is None
    assert outcome.plan == {**replacement, '_schema_version': PLAN_SCHEMA_VERSION}


def test_a_normalizing_read_rereads_under_the_mutex(artifacts: TaskArtifacts) -> None:
    malformed = _seed_plan()
    malformed['tdd_steps'] = malformed.pop('steps')
    (artifacts.root / 'plan.json').write_text(json.dumps(malformed))
    served: list[dict] = []

    def canonicalize(plan: dict) -> None:
        plan['steps'] = plan.pop('tdd_steps')
        mark_holder(plan)

    outcome = race_writer_against_held_mutex(
        artifacts, lambda: served.append(artifacts.read_plan()), holder_edit=canonicalize,
    )

    assert outcome.blocked, 'the normalization write-back did not wait for the mutex'
    assert outcome.writer_error is None
    assert served[0].get(HOLDER_MARKER_KEY) is True, 'read_plan served its stale snapshot'
    assert outcome.plan.get(HOLDER_MARKER_KEY) is True
    assert 'tdd_steps' not in outcome.plan


def test_the_lane_symlink_view_and_the_meta_root_view_share_one_mutex(tmp_path: Path) -> None:
    worktree_base = tmp_path / 'base'
    worktree = worktree_base / 'lane'
    worktree.mkdir(parents=True)
    meta_view = TaskArtifacts(worktree, TaskArtifacts.meta_root_for(worktree_base, 'lane'))
    meta_view.init('task-1', 'Test Task', 'A test task description')
    meta_view.write_plan(_seed_plan())
    meta_view.ensure_lane_plan_symlink()
    lane_view = TaskArtifacts(worktree)
    assert (lane_view.root / 'plan.json').is_symlink()
    entered = threading.Event()

    def enter_through_the_lane_view() -> None:
        with lane_view.plan_mutex():
            entered.set()

    thread = threading.Thread(target=enter_through_the_lane_view, daemon=True)
    with meta_view.plan_mutex():
        thread.start()
        assert not entered.wait(PROBE_SECS), 'the lane view entered a held mutex'
    thread.join(_JOIN_SECS)

    assert entered.is_set(), 'the lane view never entered after release'


def test_a_vanished_root_is_not_rematerialized(artifacts: TaskArtifacts) -> None:
    shutil.rmtree(artifacts.worktree)

    with artifacts.plan_mutex():
        pass
    artifacts.update_step_status('step-1', 'done')

    assert not artifacts.root.exists()


def test_the_mutex_is_not_the_plan_ownership_lock(artifacts: TaskArtifacts) -> None:
    artifacts.update_step_status('step-1', 'done')

    assert not artifacts.is_plan_locked()
    with artifacts.plan_mutex():
        assert artifacts.lock_plan('s') is True
