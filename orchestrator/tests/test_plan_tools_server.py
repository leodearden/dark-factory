"""Tests for the plan-tools MCP server."""

from __future__ import annotations

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.plan_tools import (
    _add_design_decision,
    _add_plan_step,
    _add_prerequisite,
    _add_reuse_item,
    _create_plan,
    _mark_step_done,
)


@pytest.fixture()
def artifacts(tmp_path):
    """TaskArtifacts pointing at a temporary worktree."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


# ---------------------------------------------------------------------------
# Architect tool tests
# ---------------------------------------------------------------------------


class TestCreatePlan:
    def test_creates_plan_with_correct_structure(self, artifacts):
        result = _create_plan(
            artifacts,
            task_id='test-1',
            title='Test task',
            analysis='Some analysis',
            modules=['mod_a'],
            files=['mod_a/foo.py'],
        )
        assert result['status'] == 'ok'

        plan = artifacts.read_plan()
        assert plan['task_id'] == 'test-1'
        assert plan['title'] == 'Test task'
        assert plan['analysis'] == 'Some analysis'
        assert plan['modules'] == ['mod_a']
        assert plan['files'] == ['mod_a/foo.py']
        assert plan['steps'] == []
        assert plan['prerequisites'] == []
        assert plan['design_decisions'] == []
        assert plan['reuse'] == []
        assert '_schema_version' in plan

    def test_defaults_files_to_empty(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        plan = artifacts.read_plan()
        assert plan['files'] == []


class TestAddPlanStep:
    def test_appends_step(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        result = _add_plan_step(artifacts, 'step-1', 'test', 'Write test for X')
        assert result['status'] == 'ok'
        assert result['total_steps'] == 1

        plan = artifacts.read_plan()
        assert len(plan['steps']) == 1
        step = plan['steps'][0]
        assert step['id'] == 'step-1'
        assert step['type'] == 'test'
        assert step['description'] == 'Write test for X'
        assert step['status'] == 'pending'
        assert step['commit'] is None

    def test_preserves_order(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        _add_plan_step(artifacts, 'step-1', 'test', 'First')
        _add_plan_step(artifacts, 'step-2', 'impl', 'Second')
        _add_plan_step(artifacts, 'step-3', 'test', 'Third')

        plan = artifacts.read_plan()
        ids = [s['id'] for s in plan['steps']]
        assert ids == ['step-1', 'step-2', 'step-3']

    def test_rejects_duplicate_id(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        _add_plan_step(artifacts, 'step-1', 'test', 'First')
        result = _add_plan_step(artifacts, 'step-1', 'impl', 'Duplicate')
        assert result['status'] == 'error'
        assert 'already exists' in result['message']

    def test_rejects_id_collision_with_prereq(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        _add_prerequisite(artifacts, 'pre-1', 'Setup')
        result = _add_plan_step(artifacts, 'pre-1', 'test', 'Collision')
        assert result['status'] == 'error'
        assert 'already exists' in result['message']

    def test_requires_existing_plan(self, artifacts):
        result = _add_plan_step(artifacts, 'step-1', 'test', 'No plan')
        assert result['status'] == 'error'
        assert 'No plan exists' in result['message']


class TestAddPrerequisite:
    def test_appends_prerequisite(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        result = _add_prerequisite(artifacts, 'pre-1', 'Setup config')
        assert result['status'] == 'ok'

        plan = artifacts.read_plan()
        assert len(plan['prerequisites']) == 1
        pre = plan['prerequisites'][0]
        assert pre['id'] == 'pre-1'
        assert pre['description'] == 'Setup config'
        assert pre['status'] == 'pending'
        assert pre['commit'] is None
        assert pre['tests'] == []

    def test_requires_existing_plan(self, artifacts):
        result = _add_prerequisite(artifacts, 'pre-1', 'No plan')
        assert result['status'] == 'error'


class TestAddDesignDecision:
    def test_appends_decision(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        result = _add_design_decision(artifacts, 'Use X over Y', 'X is simpler')
        assert result['status'] == 'ok'
        assert result['total_decisions'] == 1

        plan = artifacts.read_plan()
        dd = plan['design_decisions'][0]
        assert dd['decision'] == 'Use X over Y'
        assert dd['rationale'] == 'X is simpler'


class TestAddReuseItem:
    def test_appends_reuse(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        result = _add_reuse_item(artifacts, 'MergeResult', 'git_ops.py:14', 'Follow same pattern')
        assert result['status'] == 'ok'
        assert result['total_reuse'] == 1

        plan = artifacts.read_plan()
        r = plan['reuse'][0]
        assert r['what'] == 'MergeResult'
        assert r['where'] == 'git_ops.py:14'
        assert r['how'] == 'Follow same pattern'


# ---------------------------------------------------------------------------
# Implementer tool tests
# ---------------------------------------------------------------------------


class TestMarkStepDone:
    def _setup_plan(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        _add_plan_step(artifacts, 'step-1', 'test', 'Write test')
        _add_plan_step(artifacts, 'step-2', 'impl', 'Implement')

    def test_marks_step_done(self, artifacts):
        self._setup_plan(artifacts)
        result = _mark_step_done(artifacts, 'step-1', 'abc123')
        assert result['status'] == 'ok'
        assert result['new_status'] == 'done'
        assert result['commit'] == 'abc123'

        plan = artifacts.read_plan()
        assert plan['steps'][0]['status'] == 'done'
        assert plan['steps'][0]['commit'] == 'abc123'
        assert plan['steps'][1]['status'] == 'pending'

    def test_preserves_session_id(self, artifacts):
        self._setup_plan(artifacts)
        artifacts.stamp_plan_provenance('test-1-deadbeef')

        _mark_step_done(artifacts, 'step-1', 'abc123')

        plan = artifacts.read_plan()
        assert plan['_session_id'] == 'test-1-deadbeef'

    def test_marks_prerequisite_done(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m'])
        _add_prerequisite(artifacts, 'pre-1', 'Setup')
        _add_plan_step(artifacts, 'step-1', 'test', 'Test')

        result = _mark_step_done(artifacts, 'pre-1', 'def456')
        assert result['status'] == 'ok'

        plan = artifacts.read_plan()
        assert plan['prerequisites'][0]['status'] == 'done'
        assert plan['prerequisites'][0]['commit'] == 'def456'

    def test_unknown_step_returns_error(self, artifacts):
        self._setup_plan(artifacts)
        result = _mark_step_done(artifacts, 'nonexistent', 'abc123')
        assert result['status'] == 'error'
        assert 'not found' in result['message']
