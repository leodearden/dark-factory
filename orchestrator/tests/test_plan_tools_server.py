"""Tests for the plan-tools MCP server."""

from __future__ import annotations

import logging

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools
from orchestrator.mcp.plan_tools import (
    _add_design_decision,
    _add_plan_step,
    _add_prerequisite,
    _add_reuse_item,
    _artifacts_from_args,
    _coerce_files,
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


# A ~2KB "trigger payload" analysis string mirroring the documented shape
# that accompanies the harness's files-arg mis-serialization defect:
# markdown backticks, a double-dot commit range, and a bang-pathspec token
# (task 2428). NOTE: this is realistic-context padding only — _coerce_files
# / _create_plan's coercion behavior is independent of analysis content (the
# tests below pass a stringified `files` arg directly). Its purpose is to
# exercise the boundary tests with an analysis shape matching the documented
# trigger, not to causally test analysis size/content; see TestCoerceFiles
# for the coercion-behavior unit tests themselves.
_TRIGGER_ANALYSIS = (
    'Root cause analysis referencing `backtick code`, a commit range like '
    "main..HEAD, and a bang-pathspec like :!.task/ as in `git add -- . "
    "':!.task'`. "
    + ('Extra context padding sentence describing the approach in detail. ' * 30)
)


# ---------------------------------------------------------------------------
# _coerce_files helper tests (task 2428)
# ---------------------------------------------------------------------------


class TestCoerceFiles:
    """Unit tests for the ``files`` arg-recovery helper, independent of the
    MCP tool/pydantic boundary (see the TestCreatePlanTool*/TestUpdatePlanMetadataTool*
    boundary classes below for the pydantic-validation-path regression guards).
    """

    def test_none_returns_empty_list(self):
        assert _coerce_files(None) == []

    def test_list_passthrough(self):
        assert _coerce_files(['a.py', 'b.py']) == ['a.py', 'b.py']

    def test_json_array_string_recovered(self):
        assert _coerce_files('["mod/a.py","mod/b.py"]') == ['mod/a.py', 'mod/b.py']

    def test_comma_delimited_string_recovered(self):
        assert _coerce_files('mod/a.py, mod/b.py') == ['mod/a.py', 'mod/b.py']

    def test_newline_delimited_string_recovered(self):
        assert _coerce_files('mod/a.py\nmod/b.py') == ['mod/a.py', 'mod/b.py']

    def test_bare_json_string_recovered_as_single_path(self):
        assert _coerce_files('"mod/a.py"') == ['mod/a.py']

    def test_json_array_string_drops_blank_entries(self):
        assert _coerce_files('["a.py", "  ", ""]') == ['a.py']

    def test_list_entries_are_stripped(self):
        assert _coerce_files([' a.py ', ' b.py ']) == ['a.py', 'b.py']

    def test_single_quoted_python_repr_list_recovered(self):
        """A Python-repr-style stringified list (single quotes) is not
        valid JSON but is a plausible mis-serialization shape; it must be
        recovered via ast.literal_eval rather than degrading to garbage
        comma-split fragments like "['a.py'" and "'b.py']"."""
        assert _coerce_files("['mod/a.py', 'mod/b.py']") == ['mod/a.py', 'mod/b.py']

    def test_dict_shaped_json_falls_back_and_logs_warning(self, caplog):
        """A JSON value that decodes to something other than a list/str
        (e.g. a dict) is not a recognized files shape. It still falls back
        to a best-effort delimiter split, but must log a warning so the
        corruption is visible rather than silently persisting a malformed
        path."""
        with caplog.at_level(logging.WARNING, logger='orchestrator.mcp.plan_tools'):
            result = _coerce_files('{"a": 1}')

        assert result == ['{"a": 1}']
        assert any(
            'unrecognized JSON shape' in r.getMessage() for r in caplog.records
        )

    def test_residual_bracket_chars_after_split_logs_warning(self, caplog):
        """Input that is neither valid JSON nor a valid Python list literal
        (e.g. unquoted comma-separated entries wrapped in brackets) falls
        back to a delimiter split that leaves stray bracket characters in
        the result. This must be logged rather than silently accepted as
        valid file paths."""
        with caplog.at_level(logging.WARNING, logger='orchestrator.mcp.plan_tools'):
            result = _coerce_files('[mod/a.py, mod/b.py]')

        assert result == ['[mod/a.py', 'mod/b.py]']
        assert any(
            'residual bracket/quote characters' in r.getMessage() for r in caplog.records
        )


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

    def test_recovers_stringified_files_array(self, artifacts):
        """A large/complex analysis paired with a stringified files array
        (the harness mis-serialization defect, task 2428) must still recover
        the intended file list rather than storing the raw JSON text."""
        result = _create_plan(
            artifacts,
            'test-1',
            'Test task',
            _TRIGGER_ANALYSIS,
            '["mod/a.py","mod/b.py"]',
        )
        assert result['status'] == 'ok'

        plan = artifacts.read_plan()
        assert plan['files'] == ['mod/a.py', 'mod/b.py']

    def test_dropped_files_returns_actionable_error(self, artifacts):
        """When `files` arrives as None (the harness dropped it entirely —
        the 'Missing required argument' symptom, task 2428), _create_plan
        must return an actionable error naming the reliable split-call
        workaround rather than silently writing an empty-files plan."""
        result = _create_plan(artifacts, 'test-1', 'T', 'A', None)
        assert result['status'] == 'error'
        assert 'update_plan_metadata' in result['message']
        assert (
            'placeholder' in result['message'].lower()
            or 'short' in result['message'].lower()
        )

        assert artifacts.read_plan() == {}


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

    def test_recovers_stringified_files_array(self, artifacts):
        """A mis-serialized (stringified) files arg (task 2428) must be
        recovered into a clean list, same as create_plan."""
        _setup_full_plan(artifacts)
        result = _update_plan_metadata(artifacts, files='["x.py","y.py"]')
        assert result['status'] == 'ok'

        plan = artifacts.read_plan()
        assert plan['files'] == ['x.py', 'y.py']

    def test_explicit_none_files_leaves_unchanged(self, artifacts):
        """None must keep meaning 'leave files unchanged' — it must NOT be
        treated as the create_plan dropped-files error case."""
        _setup_full_plan(artifacts)
        result = _update_plan_metadata(artifacts, files=None)
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


# ---------------------------------------------------------------------------
# CLI arg parsing → TaskArtifacts (task 2258 step-5, worktree-lane-lifecycle
# W11-ε1: hands the `.task-meta` base to the agent-side plan-tools server).
# ---------------------------------------------------------------------------


class TestArtifactsFromArgs:
    """``_artifacts_from_args`` parses ``--worktree``/``--meta-root`` into a
    ``TaskArtifacts`` — the same construction shape as workflow.py's
    orchestrator-side ``self.artifacts``, so both sides resolve the
    IDENTICAL meta_root for a given worktree.
    """

    def test_meta_root_flag_produces_relocated_root(self, tmp_path):
        wt = tmp_path / 'wt'
        wt.mkdir()
        mr = tmp_path / 'base' / '.task-meta' / 'wt'
        mr.mkdir(parents=True)

        artifacts = _artifacts_from_args(
            ['--worktree', str(wt), '--meta-root', str(mr)]
        )

        assert artifacts.root == mr
        assert artifacts.worktree == wt

    def test_omitted_meta_root_falls_back_to_legacy_task_dir(self, tmp_path):
        wt = tmp_path / 'wt'
        wt.mkdir()
        (wt / '.task').mkdir()

        artifacts = _artifacts_from_args(['--worktree', str(wt)])

        assert artifacts.root == wt / '.task'
        assert artifacts.worktree == wt

    def test_worktree_attr_survives_sibling_root_relocation(self, tmp_path):
        """Documents the plan_tools.py:600 bug this task fixes: once ``root``
        is a sibling ``.task-meta`` dir, ``artifacts.root.parent`` is the
        `.task-meta` BASE, not the worktree — callers needing the worktree
        (e.g. resolving main's SHA via git) must use ``artifacts.worktree``.
        """
        wt = tmp_path / 'wt'
        wt.mkdir()
        mr = tmp_path / 'base' / '.task-meta' / 'wt'
        mr.mkdir(parents=True)

        artifacts = _artifacts_from_args(
            ['--worktree', str(wt), '--meta-root', str(mr)]
        )

        assert artifacts.root.parent != wt
        assert artifacts.worktree == wt


# ---------------------------------------------------------------------------
# Tool-level regression guard for plan_tools.py:603 (task 2258 amendment).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReportBlockingDependencyToolResolvesRealWorktree:
    """Pins the ``report_blocking_dependency`` MCP tool's git-resolution
    target directly, rather than only asserting on the ``TaskArtifacts``
    object in isolation (``TestArtifactsFromArgs`` above).

    ``TestReportBlockingDependency`` (the ``_report_blocking_dependency``
    unit tests) call the private helper with an explicit ``main_sha=``,
    bypassing the tool wrapper's own ``worktree = artifacts.worktree`` /
    ``_resolve_main_sha(worktree)`` code path entirely — a regression that
    reverted plan_tools.py:603 back to ``artifacts.root.parent`` would still
    pass every one of those tests. This test drives the actual registered
    MCP tool through ``create_server`` with a relocated (sibling)
    ``meta_root`` and monkeypatches ``_resolve_main_sha`` to capture the
    path it is invoked with, pinning that path to ``artifacts.worktree``.
    """

    async def test_tool_calls_resolve_main_sha_with_worktree_not_root_parent(
        self, tmp_path, monkeypatch
    ):
        wt = tmp_path / 'wt'
        wt.mkdir()
        mr = tmp_path / 'base' / '.task-meta' / 'wt'
        mr.mkdir(parents=True)

        artifacts = _artifacts_from_args(
            ['--worktree', str(wt), '--meta-root', str(mr)]
        )
        artifacts.init('test-1', 'Test task', 'A test')
        # Sanity check this fixture actually reproduces the bug scenario:
        # once root is a relocated sibling, root.parent is the .task-meta
        # BASE, not the worktree.
        assert artifacts.root.parent != artifacts.worktree

        captured: list[object] = []

        def fake_resolve_main_sha(worktree):
            captured.append(worktree)
            return 'deadbeefcafef00d'

        monkeypatch.setattr(
            plan_tools, '_resolve_main_sha', fake_resolve_main_sha
        )

        server = plan_tools.create_server(artifacts)
        tool = await server.get_tool('report_blocking_dependency')
        assert tool is not None
        # report_blocking_dependency is a sync def — call tool.fn() directly.
        result = tool.fn(  # type: ignore[union-attr]
            depends_on_task_id='42', reason='missing foo()'
        )

        assert result['status'] == 'ok'
        assert captured == [artifacts.worktree]

        data = artifacts.read_blocking_dependency()
        assert data is not None
        assert data['main_sha_at_report'] == 'deadbeefcafef00d'


# ---------------------------------------------------------------------------
# files-arg pydantic-boundary regression guards (task 2428).
#
# These drive the REGISTERED MCP tool via create_server(...).get_tool(...)
# and await tool.run({...}) — NOT tool.fn(...), which bypasses FastMCP's
# pydantic argument validation (FunctionTool.run's type_adapter.validate_python,
# called BEFORE the tool body executes). A corrupted `files` arg crashes at
# that boundary, so only a tool.run(...) call exercises the real defect.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreatePlanToolRecoversMisserializedFiles:
    """create_plan tolerates a mis-serialized (stringified) files arg."""

    async def test_json_array_string_recovered_via_tool_run(self, artifacts):
        server = plan_tools.create_server(artifacts)
        tool = await server.get_tool('create_plan')
        assert tool is not None

        await tool.run({
            'task_id': 'test-1',
            'title': 'Test task',
            'analysis': _TRIGGER_ANALYSIS,
            'files': '["mod/a.py","mod/b.py"]',
        })

        plan = artifacts.read_plan()
        assert plan['files'] == ['mod/a.py', 'mod/b.py']

    async def test_comma_delimited_string_recovered_via_tool_run(self, artifacts):
        server = plan_tools.create_server(artifacts)
        tool = await server.get_tool('create_plan')
        assert tool is not None

        await tool.run({
            'task_id': 'test-1',
            'title': 'Test task',
            'analysis': _TRIGGER_ANALYSIS,
            'files': 'mod/a.py,mod/b.py',
        })

        plan = artifacts.read_plan()
        assert plan['files'] == ['mod/a.py', 'mod/b.py']


@pytest.mark.asyncio
class TestCreatePlanToolDroppedFiles:
    """create_plan tolerates a fully-dropped files arg — the harness's
    'Missing required argument' symptom (task 2428) — by returning an
    actionable structured error instead of crashing pydantic validation.
    """

    async def test_missing_files_key_returns_structured_error(self, artifacts):
        server = plan_tools.create_server(artifacts)
        tool = await server.get_tool('create_plan')
        assert tool is not None

        result = await tool.run({
            'task_id': 'test-1',
            'title': 'Test task',
            'analysis': _TRIGGER_ANALYSIS,
            # 'files' deliberately omitted — the dropped-arg shape.
        })

        assert result.structured_content is not None
        assert result.structured_content['status'] == 'error'
        assert artifacts.read_plan() == {}


@pytest.mark.asyncio
class TestUpdatePlanMetadataToolRecoversMisserializedFiles:
    """update_plan_metadata tolerates a mis-serialized (stringified) files
    arg, same as create_plan (task 2428)."""

    async def test_json_array_string_recovered_via_tool_run(self, artifacts):
        _setup_full_plan(artifacts)
        server = plan_tools.create_server(artifacts)
        tool = await server.get_tool('update_plan_metadata')
        assert tool is not None

        await tool.run({'files': '["x.py","y.py"]'})

        plan = artifacts.read_plan()
        assert plan['files'] == ['x.py', 'y.py']
