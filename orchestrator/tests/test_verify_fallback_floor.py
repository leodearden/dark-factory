"""Tests for the no-module-config verify fallback's test floor (tasks 8714, 7041).

A ``.py`` diff with no derivable test target, and any polyglot ``.py`` + ``.rs``
diff, must run the configured ``test_command`` instead of silently running no
tests. Rationale: docs/fallback-test-floor.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify_categories import FailureCategory
from orchestrator.verify_cmd import parse_config_command
from orchestrator.verify_evidence import Leg
from orchestrator.verify_fallback_floor import (
    CONFIGURED_SUITE_REASON,
    MERGE_BARE_DEFAULT_REASON,
    NO_TESTS_RAN_CATEGORY,
    apply_fallback_test_floor,
    fallback_ran_no_tests,
    fallback_test_floor_reason,
    has_configured_suite,
    non_python_reason,
)
from orchestrator.verify_plan import (
    PlannedRun,
    ScopeKind,
    VerifyPlan,
    derive_verify_plan,
    planned_run_leg,
)

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
        assert reason == CONFIGURED_SUITE_REASON
        _assert_only_test_slot_moved(before, after)

    @pytest.mark.parametrize('role', ROLES)
    def test_reify_9761_polyglot_diff_runs_configured_command(self, role):
        before = _fallback(None)

        after, reason = apply_fallback_test_floor(before, REIFY_9761_FILES, REIFY_TEST_COMMAND, role)

        assert after.test_command == REIFY_TEST_COMMAND
        assert reason == non_python_reason(REIFY_9761_RS_FILE)
        _assert_only_test_slot_moved(before, after)

    @pytest.mark.parametrize('role', ROLES)
    def test_polyglot_diff_replaces_a_narrowed_python_target(self, role):
        before = _fallback('cd tools && uv run pytest tests/test_x.py')
        files = ['tools/x.py', 'tools/tests/test_x.py', 'crates/a/src/lib.rs']

        after, reason = apply_fallback_test_floor(before, files, REIFY_TEST_COMMAND, role)

        assert after.test_command == REIFY_TEST_COMMAND
        assert reason == non_python_reason('crates/a/src/lib.rs')
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
        assert reason == MERGE_BARE_DEFAULT_REASON
        _assert_only_test_slot_moved(before, after)

    def test_bare_default_polyglot_never_yields_no_test_command(self):
        before = _fallback(None)
        files = ['pkg/helpers.py', 'crates/a/src/lib.rs']

        after, reason = apply_fallback_test_floor(before, files, 'pytest', 'task')

        assert after.test_command == 'pytest'
        assert reason == non_python_reason('crates/a/src/lib.rs')

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

    @pytest.mark.parametrize('reason', [
        CONFIGURED_SUITE_REASON, MERGE_BARE_DEFAULT_REASON, non_python_reason('crates/a/src/lib.rs'),
    ])
    def test_every_floor_reason_routes_to_the_test_leg(self, reason):
        run = PlannedRun('__fallback__', parse_config_command('pytest'), ScopeKind.FULL_SUITE, reason)

        assert planned_run_leg(run) is Leg.TEST

    @pytest.mark.parametrize('test_command', [None, '', 'pytest'])
    def test_has_configured_suite_rejects_absent_and_bare_default(self, test_command):
        assert has_configured_suite(test_command) is False

    @pytest.mark.parametrize('test_command', ['true', 'uv run pytest', REIFY_TEST_COMMAND])
    def test_has_configured_suite_accepts_any_declared_command(self, test_command):
        assert has_configured_suite(test_command) is True


class TestFallbackRanNoTests:
    @pytest.mark.parametrize('test_command', ['pytest', REIFY_TEST_COMMAND])
    def test_empty_slot_under_a_configured_command_ran_no_tests(self, test_command):
        assert fallback_ran_no_tests(None, test_command) is True

    @pytest.mark.parametrize('configured', ['', None])
    def test_no_configured_test_command_is_the_explicit_opt_out(self, configured):
        assert fallback_ran_no_tests(None, configured) is False

    @pytest.mark.parametrize(('executed', 'configured'), [
        ('pytest', 'pytest'),
        (REIFY_TEST_COMMAND, REIFY_TEST_COMMAND),
        ('pytest tests/test_x.py', 'pytest'),
    ])
    def test_a_slot_that_ran_a_command_ran_tests(self, executed, configured):
        assert fallback_ran_no_tests(executed, configured) is False

    def test_category_is_outside_the_failure_category_namespace(self):
        assert NO_TESTS_RAN_CATEGORY not in {c.value for c in FailureCategory}


def _config(test_command: str) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=Path('/fake'),
        test_command=test_command,
        lint_command='ruff check',
        type_check_command='pyright',
    )


def _fallback_plan(
    files: list[str], test_command: str, role: Literal['merge', 'task'], *, floor: bool = True,
) -> VerifyPlan:
    return derive_verify_plan(
        files, [], _config(test_command), lambda _path: None, role=role,
        fallback_test_floor=floor,
    )


def _unfloored_pytest_run(
    files: list[str], test_command: str, role: Literal['merge', 'task'],
) -> PlannedRun:
    return _run_for(_fallback_plan(files, test_command, role, floor=False), 'pytest:')


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
        assert run.reason == CONFIGURED_SUITE_REASON

    @pytest.mark.parametrize('role', ROLES)
    def test_reify_9761_polyglot_records_non_python_floor(self, role):
        run = _run_for(_fallback_plan(REIFY_9761_FILES, REIFY_TEST_COMMAND, role), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command(REIFY_TEST_COMMAND)
        assert run.reason == non_python_reason(REIFY_9761_RS_FILE)

    def test_bare_default_source_only_runs_whole_at_merge_role(self):
        run = _run_for(_fallback_plan(['pkg/helpers.py'], 'pytest', 'merge'), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command('pytest')
        assert run.reason == MERGE_BARE_DEFAULT_REASON

    def test_bare_default_source_only_keeps_the_task_role_skip(self):
        run = _run_for(_fallback_plan(['pkg/helpers.py'], 'pytest', 'task'), 'pytest:')

        assert run.scope_kind is ScopeKind.SKIPPED
        assert run == _unfloored_pytest_run(['pkg/helpers.py'], 'pytest', 'task')

    def test_bare_default_data_module_runs_whole_at_merge_role(self):
        run = _run_for(_fallback_plan(['tests/some_data.py'], 'pytest', 'merge'), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command('pytest')
        assert run.reason == MERGE_BARE_DEFAULT_REASON

    def test_bare_default_data_module_keeps_the_task_1852_skip(self):
        run = _run_for(_fallback_plan(['tests/some_data.py'], 'pytest', 'task'), 'pytest:')

        assert run.scope_kind is ScopeKind.SKIPPED
        assert run == _unfloored_pytest_run(['tests/some_data.py'], 'pytest', 'task')

    @pytest.mark.parametrize('role', ROLES)
    def test_configured_suite_with_collectable_test_keeps_its_verbatim_reason(self, role):
        run = _run_for(_fallback_plan(['tests/test_widget.py'], REIFY_TEST_COMMAND, role), 'pytest:')

        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd == parse_config_command(REIFY_TEST_COMMAND)
        assert run == _unfloored_pytest_run(['tests/test_widget.py'], REIFY_TEST_COMMAND, role)

    @pytest.mark.parametrize('leg_prefix', ['lint:', 'pyright:'])
    @pytest.mark.parametrize('files', [REIFY_9839_FILES, REIFY_9761_FILES])
    def test_floor_leaves_lint_and_type_runs_file_scoped_at_both_roles(self, files, leg_prefix):
        task_run = _run_for(_fallback_plan(files, REIFY_TEST_COMMAND, 'task'), leg_prefix)
        merge_run = _run_for(_fallback_plan(files, REIFY_TEST_COMMAND, 'merge'), leg_prefix)
        py_files = tuple(f for f in files if f.endswith('.py'))

        assert task_run == merge_run
        assert task_run.scope_kind is ScopeKind.FILE_SCOPED
        assert task_run.scoped_targets == py_files


def _write_files(root: Path, files: list[str]) -> None:
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# placeholder\n')


def _passing_result() -> verify.VerifyResult:
    return verify.VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='ok',
        failing_leg_categories=[],
    )


def _executed_configs(spy: AsyncMock) -> list[ModuleConfig]:
    return [call.args[2] for call in spy.await_args_list if len(call.args) > 2]


def _plan_pytest_run(plan: dict | None) -> dict:
    assert plan is not None
    (run,) = [r for r in plan['runs'] if r['reason'].startswith('pytest:')]
    return run


async def _run_fallback(
    root: Path,
    test_command: str,
    files: list[str],
    role: Literal['merge', 'task'],
    *,
    verdict: verify.VerifyResult | None = None,
    verify_allow_no_tests_ran: bool = False,
) -> tuple[ModuleConfig, dict, verify.VerifyResult]:
    config = OrchestratorConfig(
        project_root=root,
        test_command=test_command,
        lint_command='ruff check',
        type_check_command='pyright',
        verify_allow_no_tests_ran=verify_allow_no_tests_ran,
    )
    spy = AsyncMock(return_value=verdict if verdict is not None else _passing_result())
    with patch.object(verify, 'run_verification', new=spy):
        result = await verify.run_scoped_verification(
            root, config, [], task_files=files, role=role,
        )
    spy.assert_awaited_once()
    (executed,) = _executed_configs(spy)
    return executed, _plan_pytest_run(result.plan), result


@pytest.mark.asyncio
class TestRunScopedVerificationFallbackFloor:
    @pytest.mark.parametrize('role', ROLES)
    async def test_reify_9839_executes_configured_command(self, tmp_path, role):
        _write_files(tmp_path, REIFY_9839_FILES)

        executed, run, result = await _run_fallback(
            tmp_path, REIFY_TEST_COMMAND, REIFY_9839_FILES, role,
        )

        assert result.passed is True
        assert executed.test_command == REIFY_TEST_COMMAND
        assert run['scope_kind'] == 'full_suite'
        assert run['module_prefix'] == executed.prefix
        assert run['cmd']['raw'] == REIFY_TEST_COMMAND
        assert run['reason'] == CONFIGURED_SUITE_REASON

    @pytest.mark.parametrize('role', ROLES)
    async def test_reify_9761_polyglot_executes_configured_command(self, tmp_path, role):
        _write_files(tmp_path, REIFY_9761_FILES)

        executed, run, _ = await _run_fallback(tmp_path, REIFY_TEST_COMMAND, REIFY_9761_FILES, role)

        assert executed.test_command == REIFY_TEST_COMMAND
        assert run['scope_kind'] == 'full_suite'
        assert run['cmd']['raw'] == REIFY_TEST_COMMAND
        assert run['reason'] == non_python_reason(REIFY_9761_RS_FILE)

    async def test_pure_subproject_source_diff_executes_configured_command(self, tmp_path):
        _write_files(tmp_path, ['backend/pyproject.toml', 'backend/app/x.py'])
        test_command = 'cd backend && uv run pytest'

        executed, run, _ = await _run_fallback(tmp_path, test_command, ['backend/app/x.py'], 'task')

        assert executed.prefix == 'backend'
        assert executed.test_command == test_command
        assert executed.lint_command is not None and 'backend' in executed.lint_command
        assert executed.type_check_command is not None and 'backend' in executed.type_check_command
        assert run['scope_kind'] == 'full_suite'
        assert run['module_prefix'] == 'backend'

    @pytest.mark.parametrize('role', ROLES)
    async def test_mixed_root_and_subproject_records_each_layers_own_floor(self, tmp_path, role):
        _write_files(tmp_path, ['backend/pyproject.toml', 'tool.py', 'backend/a.py'])
        mixed_files = ['tool.py', 'backend/a.py']
        test_command = 'cd backend && uv run pytest'
        decision_floor = fallback_test_floor_reason(
            mixed_files, test_command, role, derived_target=False,
        )

        executed, run, _ = await _run_fallback(tmp_path, test_command, mixed_files, role)

        assert executed.prefix == 'backend'
        assert executed.test_command is not None
        assert executed.test_command != test_command
        assert run['cmd']['raw'] == executed.test_command
        assert run['reason'] != decision_floor

        decision = _run_for(_fallback_plan(mixed_files, test_command, role), 'pytest:')
        assert decision.scope_kind is ScopeKind.FULL_SUITE
        assert decision.reason == decision_floor

    async def test_bare_default_source_only_runs_whole_at_merge_role(self, tmp_path):
        _write_files(tmp_path, ['pkg/helpers.py'])

        executed, run, result = await _run_fallback(tmp_path, 'pytest', ['pkg/helpers.py'], 'merge')

        assert result.passed is True
        assert executed.test_command == 'pytest'
        assert run['scope_kind'] == 'full_suite'

    @pytest.mark.parametrize('role', ROLES)
    async def test_derived_configured_suite_keeps_its_verbatim_reason(self, tmp_path, role):
        _write_files(tmp_path, ['tests/test_widget.py'])

        executed, run, _ = await _run_fallback(
            tmp_path, REIFY_TEST_COMMAND, ['tests/test_widget.py'], role,
        )

        assert executed.test_command == REIFY_TEST_COMMAND
        assert run['reason'] == _unfloored_pytest_run(
            ['tests/test_widget.py'], REIFY_TEST_COMMAND, role,
        ).reason


@pytest.mark.asyncio
class TestNoTestsRanOutcome:
    async def test_bare_default_source_only_task_role_fails_as_no_tests_ran(self, tmp_path):
        _write_files(tmp_path, ['pkg/helpers.py'])

        executed, run, result = await _run_fallback(tmp_path, 'pytest', ['pkg/helpers.py'], 'task')

        assert result.passed is False
        assert result.category == NO_TESTS_RAN_CATEGORY
        assert result.trivial is False
        assert result.failing_leg_categories is None
        assert 'pkg/helpers.py' in result.cause_hint
        assert 'report_missing_plan_file' in result.cause_hint
        assert 'verify_allow_no_tests_ran' not in result.cause_hint
        assert executed.test_command is None
        assert executed.lint_command is not None
        assert executed.type_check_command is not None
        assert run['scope_kind'] == 'skipped'

    async def test_bare_default_test_tree_data_module_fails_as_no_tests_ran(self, tmp_path):
        data_module = 'shared/tests/silent_fallthrough_allowlist.py'
        _write_files(tmp_path, [data_module])

        executed, _, result = await _run_fallback(tmp_path, 'pytest', [data_module], 'task')

        assert result.category == NO_TESTS_RAN_CATEGORY
        assert executed.test_command is None

    async def test_a_lint_red_is_reported_as_itself(self, tmp_path):
        _write_files(tmp_path, ['pkg/helpers.py'])
        lint_red = verify.VerifyResult(
            passed=False, test_output='', lint_output='E1 boom', type_output='',
            summary='lint failed', category='lint_error',
        )

        _, _, result = await _run_fallback(
            tmp_path, 'pytest', ['pkg/helpers.py'], 'task', verdict=lint_red,
        )

        assert result.category == 'lint_error'
        assert result.passed is False

    async def test_opt_in_lets_the_empty_slot_pass_with_a_warning(self, tmp_path, caplog):
        _write_files(tmp_path, ['pkg/helpers.py'])

        with caplog.at_level(logging.WARNING, logger=verify.__name__):
            _, run, result = await _run_fallback(
                tmp_path, 'pytest', ['pkg/helpers.py'], 'task', verify_allow_no_tests_ran=True,
            )

        assert result.passed is True
        assert result.category != NO_TESTS_RAN_CATEGORY
        assert run['scope_kind'] == 'skipped'
        assert any(
            record.levelno == logging.WARNING
            and 'verify_allow_no_tests_ran' in record.getMessage()
            and 'pkg/helpers.py' in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.parametrize('role', ROLES)
    async def test_empty_test_command_stays_the_explicit_opt_out(self, tmp_path, role):
        _write_files(tmp_path, ['pkg/helpers.py'])

        _, _, result = await _run_fallback(tmp_path, '', ['pkg/helpers.py'], role)

        assert result.passed is True
