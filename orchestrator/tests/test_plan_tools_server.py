"""Tests for the plan-tools MCP server."""

from __future__ import annotations

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.plan_tools import (
    _add_design_decision,
    _add_plan_step,
    _add_prerequisite,
    _add_reuse_item,
    _confirm_plan,
    _create_plan,
    _mark_step_done,
    _remove_plan_step,
    _replace_plan_step,
    _report_blocking_dependency,
    _report_false_premise,
    _report_task_already_done,
    _report_unactionable_task,
    _update_plan_metadata,
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
            files=['mod_a/foo.py'],
        )
        assert result['status'] == 'ok'

        plan = artifacts.read_plan()
        assert plan['task_id'] == 'test-1'
        assert plan['title'] == 'Test task'
        assert plan['analysis'] == 'Some analysis'
        assert plan['files'] == ['mod_a/foo.py']
        assert 'modules' not in plan
        assert plan['steps'] == []
        assert plan['prerequisites'] == []
        assert plan['design_decisions'] == []
        assert plan['reuse'] == []
        assert '_schema_version' in plan


class TestAddPlanStep:
    def test_appends_step(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
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
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
        _add_plan_step(artifacts, 'step-1', 'test', 'First')
        _add_plan_step(artifacts, 'step-2', 'impl', 'Second')
        _add_plan_step(artifacts, 'step-3', 'test', 'Third')

        plan = artifacts.read_plan()
        ids = [s['id'] for s in plan['steps']]
        assert ids == ['step-1', 'step-2', 'step-3']

    def test_rejects_duplicate_id(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
        _add_plan_step(artifacts, 'step-1', 'test', 'First')
        result = _add_plan_step(artifacts, 'step-1', 'impl', 'Duplicate')
        assert result['status'] == 'error'
        assert 'already exists' in result['message']

    def test_rejects_id_collision_with_prereq(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
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
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
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
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
        result = _add_design_decision(artifacts, 'Use X over Y', 'X is simpler')
        assert result['status'] == 'ok'
        assert result['total_decisions'] == 1

        plan = artifacts.read_plan()
        dd = plan['design_decisions'][0]
        assert dd['decision'] == 'Use X over Y'
        assert dd['rationale'] == 'X is simpler'


class TestAddReuseItem:
    def test_appends_reuse(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
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
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
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
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
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


# ---------------------------------------------------------------------------
# Revalidation tool tests
# ---------------------------------------------------------------------------


def _setup_full_plan(artifacts):
    """Create a plan with steps and prerequisites for revalidation tests."""
    _create_plan(
        artifacts, 'test-1', 'Test task', 'Analysis',
        ['mod_a/foo.py', 'mod_a/bar.py'],
    )
    _add_prerequisite(artifacts, 'pre-1', 'Setup config')
    _add_plan_step(artifacts, 'step-1', 'test', 'Write test for foo')
    _add_plan_step(artifacts, 'step-2', 'impl', 'Implement foo')
    _add_plan_step(artifacts, 'step-3', 'test', 'Write test for bar')


class TestUpdatePlanMetadata:
    def test_updates_files(self, artifacts):
        _setup_full_plan(artifacts)
        result = _update_plan_metadata(
            artifacts, files=['mod_a/foo.py', 'mod_a/bar.py', 'mod_a/baz.py'],
        )
        assert result['status'] == 'ok'
        assert result['files'] == 3

        plan = artifacts.read_plan()
        assert plan['files'] == ['mod_a/foo.py', 'mod_a/bar.py', 'mod_a/baz.py']
        # Analysis unchanged
        assert plan['analysis'] == 'Analysis'

    def test_updates_analysis_only(self, artifacts):
        _setup_full_plan(artifacts)
        result = _update_plan_metadata(artifacts, analysis='Updated analysis')
        assert result['status'] == 'ok'

        plan = artifacts.read_plan()
        assert plan['analysis'] == 'Updated analysis'
        # Files unchanged
        assert plan['files'] == ['mod_a/foo.py', 'mod_a/bar.py']

    def test_updates_all_fields(self, artifacts):
        _setup_full_plan(artifacts)
        result = _update_plan_metadata(
            artifacts,
            files=['mod_b/x.py'],
            analysis='New approach',
        )
        assert result['status'] == 'ok'
        plan = artifacts.read_plan()
        assert plan['files'] == ['mod_b/x.py']
        assert plan['analysis'] == 'New approach'

    def test_preserves_steps(self, artifacts):
        _setup_full_plan(artifacts)
        _update_plan_metadata(artifacts, files=['mod_a/new.py'])
        plan = artifacts.read_plan()
        assert len(plan['steps']) == 3
        assert len(plan['prerequisites']) == 1

    def test_no_plan_returns_error(self, artifacts):
        result = _update_plan_metadata(artifacts, files=['x.py'])
        assert result['status'] == 'error'

    def test_no_args_is_noop(self, artifacts):
        _setup_full_plan(artifacts)
        result = _update_plan_metadata(artifacts)
        assert result['status'] == 'ok'
        plan = artifacts.read_plan()
        assert plan['files'] == ['mod_a/foo.py', 'mod_a/bar.py']


class TestRemovePlanStep:
    def test_removes_pending_step(self, artifacts):
        _setup_full_plan(artifacts)
        result = _remove_plan_step(artifacts, 'step-2')
        assert result['status'] == 'ok'
        assert result['removed'] == 'step-2'
        assert result['collection'] == 'steps'

        plan = artifacts.read_plan()
        ids = [s['id'] for s in plan['steps']]
        assert ids == ['step-1', 'step-3']

    def test_removes_prerequisite(self, artifacts):
        _setup_full_plan(artifacts)
        result = _remove_plan_step(artifacts, 'pre-1')
        assert result['status'] == 'ok'
        assert result['collection'] == 'prerequisites'

        plan = artifacts.read_plan()
        assert plan['prerequisites'] == []

    def test_refuses_done_step(self, artifacts):
        _setup_full_plan(artifacts)
        _mark_step_done(artifacts, 'step-1', 'abc123')
        result = _remove_plan_step(artifacts, 'step-1')
        assert result['status'] == 'error'
        assert 'done' in result['message']

    def test_unknown_step_returns_error(self, artifacts):
        _setup_full_plan(artifacts)
        result = _remove_plan_step(artifacts, 'nonexistent')
        assert result['status'] == 'error'
        assert 'not found' in result['message']

    def test_no_plan_returns_error(self, artifacts):
        result = _remove_plan_step(artifacts, 'step-1')
        assert result['status'] == 'error'


class TestReplacePlanStep:
    def test_replaces_step_in_place(self, artifacts):
        _setup_full_plan(artifacts)
        result = _replace_plan_step(
            artifacts, 'step-2', 'impl', 'Implement foo differently',
        )
        assert result['status'] == 'ok'
        assert result['replaced'] == 'step-2'

        plan = artifacts.read_plan()
        step = plan['steps'][1]
        assert step['id'] == 'step-2'
        assert step['type'] == 'impl'
        assert step['description'] == 'Implement foo differently'
        assert step['status'] == 'pending'

    def test_preserves_position(self, artifacts):
        _setup_full_plan(artifacts)
        _replace_plan_step(artifacts, 'step-2', 'test', 'Changed to test')
        plan = artifacts.read_plan()
        ids = [s['id'] for s in plan['steps']]
        assert ids == ['step-1', 'step-2', 'step-3']

    def test_refuses_done_step(self, artifacts):
        _setup_full_plan(artifacts)
        _mark_step_done(artifacts, 'step-1', 'abc123')
        result = _replace_plan_step(artifacts, 'step-1', 'test', 'Nope')
        assert result['status'] == 'error'
        assert 'done' in result['message']

    def test_unknown_step_returns_error(self, artifacts):
        _setup_full_plan(artifacts)
        result = _replace_plan_step(artifacts, 'nonexistent', 'test', 'X')
        assert result['status'] == 'error'
        assert 'not found' in result['message']

    def test_no_plan_returns_error(self, artifacts):
        result = _replace_plan_step(artifacts, 'step-1', 'test', 'X')
        assert result['status'] == 'error'


class TestConfirmPlan:
    def test_stamps_revalidated_at(self, artifacts):
        _setup_full_plan(artifacts)
        result = _confirm_plan(artifacts)
        assert result['status'] == 'ok'
        assert result['steps'] == 3
        assert result['files'] == 2

        plan = artifacts.read_plan()
        assert '_revalidated_at' in plan

    def test_stamps_finalized_marker(self, artifacts):
        _setup_full_plan(artifacts)
        result = _confirm_plan(artifacts)
        assert result['status'] == 'ok'
        assert result['finalized'] is True

        plan = artifacts.read_plan()
        # _finalized_at is the durable completeness marker the workflow gates on.
        assert '_finalized_at' in plan
        assert plan['_finalized_at']

    def test_no_plan_returns_error(self, artifacts):
        result = _confirm_plan(artifacts)
        assert result['status'] == 'error'

    def test_empty_steps_returns_error(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m.py'])
        result = _confirm_plan(artifacts)
        assert result['status'] == 'error'
        assert 'no steps' in result['message'].lower()
        # A plan that cannot be confirmed must NOT receive the marker.
        assert '_finalized_at' not in artifacts.read_plan()

    def test_empty_files_returns_error(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', [])
        _add_plan_step(artifacts, 'step-1', 'test', 'Write a test')
        result = _confirm_plan(artifacts)
        assert result['status'] == 'error'
        assert 'files' in result['message'].lower()
        assert '_finalized_at' not in artifacts.read_plan()


class TestReportBlockingDependency:
    """Architect's escape hatch when a missing-dep means it cannot plan."""

    def test_writes_artifact_with_provided_main_sha(self, artifacts):
        result = _report_blocking_dependency(
            artifacts,
            depends_on_task_id='42',
            reason='task 50 references foo() introduced by task 42',
            main_sha='deadbeefcafef00d',
        )
        assert result['status'] == 'ok'
        assert result['depends_on_task_id'] == '42'

        data = artifacts.read_blocking_dependency()
        assert data is not None
        assert data['depends_on_task_id'] == '42'
        assert (
            data['reason']
            == 'task 50 references foo() introduced by task 42'
        )
        assert data['main_sha_at_report'] == 'deadbeefcafef00d'
        assert 'reported_at' in data

    def test_overwrites_prior_report(self, artifacts):
        _report_blocking_dependency(
            artifacts, '1', 'first', main_sha='aaa'
        )
        _report_blocking_dependency(
            artifacts, '2', 'second', main_sha='bbb'
        )
        data = artifacts.read_blocking_dependency()
        assert data['depends_on_task_id'] == '2'
        assert data['main_sha_at_report'] == 'bbb'

    def test_does_not_mutate_plan_json(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m/x.py'])
        _add_plan_step(artifacts, 'step-1', 'test', 'Write test')
        _report_blocking_dependency(artifacts, '5', 'r', main_sha='abc')

        plan = artifacts.read_plan()
        # plan.json must not be touched — the tool only writes the new artifact.
        assert plan['task_id'] == 'test-1'
        assert len(plan['steps']) == 1


class TestReportTaskAlreadyDone:
    """Architect's escape hatch when the task's work is already on main."""

    def test_writes_artifact(self, artifacts):
        result = _report_task_already_done(
            artifacts,
            commit='abc123def456',
            evidence='helpers.foo introduced by task 42',
        )
        assert result['status'] == 'ok'
        assert result['commit'] == 'abc123def456'

        data = artifacts.read_already_done()
        assert data is not None
        assert data['commit'] == 'abc123def456'
        assert data['evidence'] == 'helpers.foo introduced by task 42'
        assert 'reported_at' in data

    def test_overwrites_prior_report(self, artifacts):
        _report_task_already_done(artifacts, 'sha1', 'first')
        _report_task_already_done(artifacts, 'sha2', 'second')
        data = artifacts.read_already_done()
        assert data is not None
        assert data['commit'] == 'sha2'
        assert data['evidence'] == 'second'

    def test_does_not_mutate_plan_json(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m/x.py'])
        _add_plan_step(artifacts, 'step-1', 'test', 'Write test')
        _report_task_already_done(artifacts, 'sha', 'e')

        plan = artifacts.read_plan()
        assert plan['task_id'] == 'test-1'
        assert len(plan['steps']) == 1


class TestReportUnactionableTask:
    """Architect's escape hatch when the task spec is unworkable."""

    def test_writes_artifact(self, artifacts):
        result = _report_unactionable_task(
            artifacts,
            reason='spec contradicts already-merged refactor',
            evidence='task asks to add foo() but task 31 deleted it',
        )
        assert result['status'] == 'ok'
        assert result['reason'] == 'spec contradicts already-merged refactor'

        data = artifacts.read_unactionable_task()
        assert data is not None
        assert data['reason'] == 'spec contradicts already-merged refactor'
        assert 'task 31 deleted it' in data['evidence']
        assert 'reported_at' in data

    def test_overwrites_prior_report(self, artifacts):
        _report_unactionable_task(artifacts, 'first', 'e1')
        _report_unactionable_task(artifacts, 'second', 'e2')
        data = artifacts.read_unactionable_task()
        assert data is not None
        assert data['reason'] == 'second'
        assert data['evidence'] == 'e2'

    def test_does_not_mutate_plan_json(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m/x.py'])
        _add_plan_step(artifacts, 'step-1', 'test', 'Write test')
        _report_unactionable_task(artifacts, 'r', 'e')

        plan = artifacts.read_plan()
        assert plan['task_id'] == 'test-1'
        assert len(plan['steps']) == 1


class TestReportFalsePremise:
    """Architect's escape hatch when a RED-test premise is false/unreachable."""

    def test_writes_artifact(self, artifacts):
        result = _report_false_premise(
            artifacts,
            classification='exactness',
            premise='natural cubic spline reproduces a general cubic to 1e-12',
            evidence='natural BC forces M[0]=M[N]=0; impossible for a general cubic',
            proposed_resolution='change asserted end-condition or weaken bound + file follow-up',
        )
        assert result['status'] == 'ok'
        assert result['classification'] == 'exactness'

        data = artifacts.read_false_premise()
        assert data is not None
        assert data['classification'] == 'exactness'
        assert 'natural cubic spline' in data['premise']
        assert 'M[0]=M[N]=0' in data['evidence']
        assert 'change asserted end-condition' in data['proposed_resolution']
        assert 'reported_at' in data

    def test_overwrites_prior_report(self, artifacts):
        _report_false_premise(
            artifacts,
            classification='numeric_bound',
            premise='first premise',
            evidence='e1',
            proposed_resolution='r1',
        )
        _report_false_premise(
            artifacts,
            classification='dependency_capability',
            premise='second premise',
            evidence='e2',
            proposed_resolution='r2',
        )
        data = artifacts.read_false_premise()
        assert data is not None
        assert data['classification'] == 'dependency_capability'
        assert data['premise'] == 'second premise'
        assert data['evidence'] == 'e2'
        assert data['proposed_resolution'] == 'r2'

    def test_does_not_mutate_plan_json(self, artifacts):
        _create_plan(artifacts, 'test-1', 'T', 'A', ['m/x.py'])
        _add_plan_step(artifacts, 'step-1', 'test', 'Write test')
        _report_false_premise(
            artifacts,
            classification='exactness',
            premise='p',
            evidence='e',
            proposed_resolution='r',
        )

        plan = artifacts.read_plan()
        assert plan['task_id'] == 'test-1'
        assert len(plan['steps']) == 1
