"""Tests for the no-module-config verify fallback's test floor (tasks 8714, 7041).

A ``.py`` diff with no derivable test target, and any polyglot ``.py`` + ``.rs``
diff, must run the configured ``test_command`` instead of silently running no
tests. Rationale: docs/fallback-test-floor.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify_cmd import parse_config_command
from orchestrator.verify_fallback_floor import (
    apply_fallback_test_floor,
    fallback_test_floor_reason,
    has_configured_suite,
)
from orchestrator.verify_plan import PlannedRun, ScopeKind, VerifyPlan, derive_verify_plan

REIFY_TEST_COMMAND = './scripts/verify.sh test --scope branch --include-infra'

REIFY_9839_FILES = ['scripts/assert-crate-strata.py']
REIFY_9761_RS_FILE = 'crates/workspace-feature-union/src/lib.rs'
REIFY_9761_FILES = [
    'scripts/workspace-feature-union.py',
    REIFY_9761_RS_FILE,
    'Cargo.toml',
    'Cargo.lock',
    'scripts/crate-strata.json',
]

ROLES: tuple[Literal['merge', 'task'], ...] = ('task', 'merge')


def _fallback(test_command: str | None) -> ModuleConfig:
    return ModuleConfig(
        prefix='__fallback__',
        test_command=test_command,
        lint_command='ruff check scripts/x.py',
        type_check_command='pyright scripts/x.py',
    )


def _assert_only_test_slot_moved(before: ModuleConfig, after: ModuleConfig) -> None:
    assert after.prefix == before.prefix
    assert after.lint_command == before.lint_command
    assert after.type_check_command == before.type_check_command


class TestFallbackTestFloorDecision:
    @pytest.mark.parametrize('role', ROLES)
    def test_reify_9839_source_only_diff_runs_configured_command(self, role):
        before = _fallback(None)

        after, reason = apply_fallback_test_floor(before, REIFY_9839_FILES, REIFY_TEST_COMMAND, role)

        assert after.test_command == REIFY_TEST_COMMAND
        assert reason is not None
        assert reason.startswith('pytest:')
        assert '7041' in reason
        _assert_only_test_slot_moved(before, after)

    @pytest.mark.parametrize('role', ROLES)
    def test_reify_9761_polyglot_diff_runs_configured_command(self, role):
        before = _fallback(None)

        after, reason = apply_fallback_test_floor(before, REIFY_9761_FILES, REIFY_TEST_COMMAND, role)

        assert after.test_command == REIFY_TEST_COMMAND
        assert reason is not None
        assert reason.startswith('pytest:')
        assert REIFY_9761_RS_FILE in reason
        _assert_only_test_slot_moved(before, after)

    @pytest.mark.parametrize('role', ROLES)
    def test_polyglot_diff_replaces_a_narrowed_python_target(self, role):
        before = _fallback('cd tools && uv run pytest tests/test_x.py')
        files = ['tools/x.py', 'tools/tests/test_x.py', 'crates/a/src/lib.rs']

        after, reason = apply_fallback_test_floor(before, files, REIFY_TEST_COMMAND, role)

        assert after.test_command == REIFY_TEST_COMMAND
        assert reason is not None
        assert 'crates/a/src/lib.rs' in reason
        _assert_only_test_slot_moved(before, after)

    @pytest.mark.parametrize('role', ROLES)
    def test_derived_target_without_non_python_source_stands(self, role):
        before = _fallback('cd backend && uv run pytest tests/test_a.py')
        files = ['backend/a.py', 'backend/tests/test_a.py']

        after, reason = apply_fallback_test_floor(before, files, REIFY_TEST_COMMAND, role)

        assert after == before
        assert reason is None

    def test_bare_default_source_only_keeps_the_task_role_skip(self):
        before = _fallback(None)

        after, reason = apply_fallback_test_floor(before, ['pkg/helpers.py'], 'pytest', 'task')

        assert after == before
        assert reason is None

    def test_bare_default_source_only_runs_whole_at_merge_role(self):
        before = _fallback(None)

        after, reason = apply_fallback_test_floor(before, ['pkg/helpers.py'], 'pytest', 'merge')

        assert after.test_command == 'pytest'
        assert reason is not None
        assert reason.startswith('pytest:')
        _assert_only_test_slot_moved(before, after)

    def test_bare_default_polyglot_never_yields_no_test_command(self):
        before = _fallback(None)
        files = ['pkg/helpers.py', 'crates/a/src/lib.rs']

        after, reason = apply_fallback_test_floor(before, files, 'pytest', 'task')

        assert after.test_command == 'pytest'
        assert reason is not None
        assert reason.startswith('pytest:')

    @pytest.mark.parametrize('test_command', [None, ''])
    @pytest.mark.parametrize('files', [REIFY_9839_FILES, REIFY_9761_FILES])
    def test_no_configured_test_command_is_an_explicit_opt_out(self, test_command, files):
        before = _fallback(None)

        after, reason = apply_fallback_test_floor(before, files, test_command, 'merge')

        assert after == before
        assert reason is None

    @pytest.mark.parametrize('role', ROLES)
    @pytest.mark.parametrize('test_command', [REIFY_TEST_COMMAND, 'pytest'])
    def test_reason_is_none_when_a_target_was_derived_and_no_non_python_source(
        self, test_command, role,
    ):
        reason = fallback_test_floor_reason(
            ['pkg/helpers.py', 'tests/test_helpers.py'], test_command, role, derived_target=True,
        )

        assert reason is None

    @pytest.mark.parametrize('test_command', [None, '', 'pytest'])
    def test_has_configured_suite_rejects_absent_and_bare_default(self, test_command):
        assert has_configured_suite(test_command) is False

    @pytest.mark.parametrize('test_command', ['true', 'uv run pytest', REIFY_TEST_COMMAND])
    def test_has_configured_suite_accepts_any_declared_command(self, test_command):
        assert has_configured_suite(test_command) is True


def _config(test_command: str) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=Path('/fake'),
        test_command=test_command,
        lint_command='ruff check',
        type_check_command='pyright',
    )


def _fallback_plan(
    files: list[str], test_command: str, role: Literal['merge', 'task'],
) -> VerifyPlan:
    return derive_verify_plan(files, [], _config(test_command), lambda _path: None, role=role)


def _run_for(plan: VerifyPlan, leg_prefix: str) -> PlannedRun:
    (run,) = [
        r for r in plan.runs
        if r.module_prefix == '__fallback__' and r.reason.startswith(leg_prefix)
    ]
    return run


class TestFallbackPlanRecordsFloor:
    @pytest.mark.parametrize('role', ROLES)
    def test_reify_9839_records_configured_suite_floor(self, role):
        run = _run_for(_fallback_plan(REIFY_9839_FILES, REIFY_TEST_COMMAND, role), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command(REIFY_TEST_COMMAND)
        assert run.scoped_targets == ()
        assert '7041' in run.reason

    @pytest.mark.parametrize('role', ROLES)
    def test_reify_9761_polyglot_records_non_python_floor(self, role):
        run = _run_for(_fallback_plan(REIFY_9761_FILES, REIFY_TEST_COMMAND, role), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command(REIFY_TEST_COMMAND)
        assert REIFY_9761_RS_FILE in run.reason

    def test_bare_default_source_only_runs_whole_at_merge_role(self):
        run = _run_for(_fallback_plan(['pkg/helpers.py'], 'pytest', 'merge'), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command('pytest')

    def test_bare_default_source_only_keeps_the_task_role_skip(self):
        run = _run_for(_fallback_plan(['pkg/helpers.py'], 'pytest', 'task'), 'pytest:')

        assert run.scope_kind is ScopeKind.SKIPPED
        assert run.cmd is None
        assert run.reason == 'pytest: no collectable test files touched — nothing to run'

    def test_bare_default_data_module_runs_whole_at_merge_role(self):
        run = _run_for(_fallback_plan(['tests/some_data.py'], 'pytest', 'merge'), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command('pytest')

    def test_bare_default_data_module_keeps_the_task_1852_skip(self):
        run = _run_for(_fallback_plan(['tests/some_data.py'], 'pytest', 'task'), 'pytest:')

        assert run.scope_kind is ScopeKind.SKIPPED
        assert '1852' in run.reason

    @pytest.mark.parametrize('role', ROLES)
    def test_configured_suite_with_collectable_test_keeps_its_verbatim_reason(self, role):
        run = _run_for(_fallback_plan(['tests/test_widget.py'], REIFY_TEST_COMMAND, role), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command(REIFY_TEST_COMMAND)
        assert 'verbatim' in run.reason
        assert '7041' not in run.reason

    @pytest.mark.parametrize('leg_prefix', ['lint:', 'pyright:'])
    @pytest.mark.parametrize('files', [REIFY_9839_FILES, REIFY_9761_FILES])
    def test_floor_leaves_lint_and_type_runs_file_scoped_at_both_roles(self, files, leg_prefix):
        task_run = _run_for(_fallback_plan(files, REIFY_TEST_COMMAND, 'task'), leg_prefix)
        merge_run = _run_for(_fallback_plan(files, REIFY_TEST_COMMAND, 'merge'), leg_prefix)
        py_files = tuple(f for f in files if f.endswith('.py'))

        assert task_run == merge_run
        assert task_run.scope_kind is ScopeKind.FILE_SCOPED
        assert task_run.scoped_targets == py_files
