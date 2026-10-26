"""TaskArtifacts.append_step_deviation — the per-step deviation record writer.

PRD ``plans/plan-deviation-recording-prd.md`` D4: a deviation record is
appended to ``step['deviations']`` without touching the step's ``status``,
``commit``, ``type`` or ``description``, on a step in any status. A sibling of
``test_artifacts.py``, whose plan shape (``TestMarkStepCommitted._plan``) it
reuses.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from shared.plan_deviation import DeviationRecord

from orchestrator.artifacts import TaskArtifacts


@pytest.fixture
def artifacts(tmp_path: Path) -> TaskArtifacts:
    worktree = tmp_path / 'worktree'
    worktree.mkdir()
    ta = TaskArtifacts(worktree)
    ta.init('task-1', 'Test Task', 'A test task description')
    return ta


def _plan(steps_key: str = 'steps') -> dict:
    return {
        'task_id': 'task-1',
        'files': ['backend/app.py'],
        'prerequisites': [
            {'id': 'pre-1', 'type': 'test', 'description': 'Setup fixture',
             'status': 'pending', 'commit': None},
        ],
        steps_key: [
            {'id': 'step-1', 'type': 'test', 'description': 'Write test',
             'status': 'pending', 'commit': None},
            {'id': 'step-2', 'type': 'impl', 'description': 'Implement',
             'status': 'pending', 'commit': None},
        ],
    }


def _record(observed: str = 'The helper already existed.') -> DeviationRecord:
    return DeviationRecord(
        kind='never_checked_fact',
        observed=observed,
        expected='The step said to write the helper.',
        action_taken='Reused the existing helper.',
        prose_only=False,
        note='',
        recorded_at='2026-10-15T07:00:00+00:00',
        recorded_by='claude-sess-1',
    )


def _step(artifacts: TaskArtifacts, step_id: str) -> dict:
    plan = artifacts.read_plan()
    return next(
        item
        for collection in ('prerequisites', 'steps')
        for item in plan[collection]
        if item['id'] == step_id
    )


def _plan_bytes(artifacts: TaskArtifacts) -> bytes:
    return (artifacts.root / 'plan.json').read_bytes()


class TestAppendStepDeviation:
    def test_first_append_creates_the_list_and_later_ones_append_in_order(
        self, artifacts: TaskArtifacts
    ):
        artifacts.write_plan(_plan())
        first, second = _record('first'), _record('second')

        assert artifacts.append_step_deviation('step-1', first) == 1
        assert _step(artifacts, 'step-1')['deviations'] == [first.to_dict()]

        assert artifacts.append_step_deviation('step-1', second) == 2
        assert _step(artifacts, 'step-1')['deviations'] == [
            first.to_dict(),
            second.to_dict(),
        ]

    def test_leaves_the_step_fields_and_the_other_steps_untouched(
        self, artifacts: TaskArtifacts
    ):
        artifacts.write_plan(_plan())
        before = artifacts.read_plan()

        artifacts.append_step_deviation('step-1', _record())

        step = _step(artifacts, 'step-1')
        before_step = before['steps'][0]
        for key in ('status', 'commit', 'type', 'description'):
            assert step[key] == before_step[key]
        assert artifacts.read_plan()['steps'][1] == before['steps'][1]

    def test_accepts_a_step_already_marked_done(self, artifacts: TaskArtifacts):
        artifacts.write_plan(_plan())
        artifacts.update_step_status('step-1', 'done', commit='abc')

        assert artifacts.append_step_deviation('step-1', _record()) == 1

        step = _step(artifacts, 'step-1')
        assert step['status'] == 'done'
        assert step['commit'] == 'abc'
        assert step['deviations'] == [_record().to_dict()]

    def test_accepts_a_prerequisite(self, artifacts: TaskArtifacts):
        artifacts.write_plan(_plan())

        assert artifacts.append_step_deviation('pre-1', _record()) == 1

        assert _step(artifacts, 'pre-1')['deviations'] == [_record().to_dict()]

    def test_unknown_step_returns_none_and_writes_nothing(
        self, artifacts: TaskArtifacts
    ):
        artifacts.write_plan(_plan())
        before = _plan_bytes(artifacts)

        assert artifacts.append_step_deviation('nope', _record()) is None

        assert _plan_bytes(artifacts) == before


class TestDeviationsSurviveExistingPlanWriters:
    """Behaviour already on main that append_step_deviation relies on."""

    def test_update_step_status_keeps_the_deviations_list(
        self, artifacts: TaskArtifacts
    ):
        plan = _plan()
        plan['steps'][0]['deviations'] = [_record().to_dict()]
        artifacts.write_plan(plan)

        artifacts.update_step_status('step-1', 'done', commit='abc')

        assert _step(artifacts, 'step-1')['deviations'] == [_record().to_dict()]

    def test_mark_step_committed_keeps_the_deviations_list(
        self, artifacts: TaskArtifacts
    ):
        plan = _plan()
        plan['steps'][0]['deviations'] = [_record().to_dict()]
        artifacts.write_plan(plan)

        assert artifacts.mark_step_committed('step-1', 'a' * 40) is True

        assert _step(artifacts, 'step-1')['deviations'] == [_record().to_dict()]

    def test_steps_alias_normalisation_keeps_the_deviations_list(
        self, artifacts: TaskArtifacts
    ):
        plan = _plan(steps_key='tdd_steps')
        plan['tdd_steps'][0]['deviations'] = [_record().to_dict()]
        artifacts.write_plan(plan)

        returned = artifacts.read_plan()

        on_disk = json.loads(_plan_bytes(artifacts))
        for normalised in (returned, on_disk):
            assert 'tdd_steps' not in normalised
            assert normalised['steps'][0]['deviations'] == [_record().to_dict()]
