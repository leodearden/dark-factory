"""Tests for orchestrator.verify's reverse-dependency test widening (task 2607).

Closes the merge-verify blind spot where a task's diff scoped to
orchestrator/ SOURCE never runs escalation/'s coupled cross-package
merge_queue tests — module_configs is resolved from the TASK's own touched
modules only, so escalation's ModuleConfig is never a candidate for an
orchestrator-only diff. This has caused RED-main fix-forward 3x (1736->1761,
2173->2038, 2435->2604), each patched reactively; this task closes it
structurally.

Two layers, each with its own test class in this file:

- ``verify._reverse_dependency_module_configs`` (step-5/step-6): the impure
  wrapper that builds worktree-bound ``list_pkg_tests``/``read_content``
  callables, calls the pure ``verify_plan.reverse_dependent_test_targets``,
  and renders each coupled dependent into an executable, pytest-only
  ``ModuleConfig``.
- ``run_scoped_verification`` wiring (step-7 onward): integration tests
  asserting the widened ModuleConfig's pytest command actually executes (or
  correctly does NOT) via the ``_run_cmd`` recording-spy idiom
  (test_verify.py:3745).

A NEW file (rather than adding to test_verify.py) per the plan's design
decision: test_verify.py contains a fragile fixture
(``TestRunScopedVerificationForwardsWorktreeToFallback``) that replaces
``_build_fallback_config`` with a signature-locked fake with no ``**kwargs``
catch-all — a dedicated file avoids that hazard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig

# escalation/tests/test_server.py's real shape: a guarded, indented import of
# orchestrator.merge_queue (escalation/tests/test_server.py:31-34).
_TEST_SERVER_CONTENT = (
    'from __future__ import annotations\n'
    '\n'
    'import pytest\n'
    '\n'
    'try:\n'
    '    from orchestrator.merge_queue import SpeculativeMergeWorker\n'
    'except ImportError:\n'
    '    SpeculativeMergeWorker = None\n'
)

_TEST_UNRELATED_CONTENT = (
    'from __future__ import annotations\n'
    '\n'
    'def test_unrelated():\n'
    '    assert True\n'
)

_ESCALATION_TEST_COMMAND = (
    'uv run --project escalation --directory escalation pytest tests/ --tb=short -q'
)
_ORCHESTRATOR_TEST_COMMAND = (
    'uv run --project orchestrator --directory orchestrator pytest tests/ --tb=short -q'
)


def _build_worktree(tmp_path: Path) -> Path:
    """A tmp_path worktree: orchestrator source + escalation's importing/unrelated tests."""
    (tmp_path / 'orchestrator' / 'src' / 'orchestrator').mkdir(parents=True)
    (tmp_path / 'orchestrator' / 'src' / 'orchestrator' / 'merge_queue.py').write_text(
        'class SpeculativeMergeWorker:\n    pass\n',
    )
    (tmp_path / 'escalation' / 'tests').mkdir(parents=True)
    (tmp_path / 'escalation' / 'tests' / 'test_server.py').write_text(_TEST_SERVER_CONTENT)
    (tmp_path / 'escalation' / 'tests' / 'test_unrelated.py').write_text(_TEST_UNRELATED_CONTENT)
    return tmp_path


def _build_worktree_with_orchestrator_test(tmp_path: Path) -> Path:
    """``_build_worktree`` plus an orchestrator-side test file the diff will touch instead of source."""
    worktree = _build_worktree(tmp_path)
    (worktree / 'orchestrator' / 'tests').mkdir(parents=True)
    (worktree / 'orchestrator' / 'tests' / 'test_something.py').write_text(
        'def test_something():\n    assert True\n',
    )
    return worktree


def _build_worktree_with_deleted_source(tmp_path: Path) -> Path:
    """Escalation's tests plus an orchestrator TEST file — but no orchestrator
    SOURCE file on disk, simulating a diff that deletes it.

    ``orchestrator/tests/test_something.py`` surviving keeps 'orchestrator'
    non-empty in ``run_scoped_verification``'s ``scoped`` list (its lint
    command is always FILE_SCOPED to whatever ``.py`` files exist under the
    prefix — see ``_derive_module_runs`` — so the widening call site is
    still reached even though no orchestrator SOURCE file exists).
    """
    (tmp_path / 'escalation' / 'tests').mkdir(parents=True)
    (tmp_path / 'escalation' / 'tests' / 'test_server.py').write_text(_TEST_SERVER_CONTENT)
    (tmp_path / 'escalation' / 'tests' / 'test_unrelated.py').write_text(_TEST_UNRELATED_CONTENT)
    (tmp_path / 'orchestrator' / 'tests').mkdir(parents=True)
    (tmp_path / 'orchestrator' / 'tests' / 'test_something.py').write_text(
        'def test_something():\n    assert True\n',
    )
    # Deliberately no orchestrator/src/orchestrator/merge_queue.py on disk —
    # the diff's task_files will still name it (as a deletion).
    return tmp_path


_DASHBOARD_TEST_COMMAND = (
    'uv run --project dashboard --directory dashboard pytest tests/ --tb=short -q'
)


def _build_dashboard_worktree(tmp_path: Path) -> Path:
    """A worktree with a dashboard source file only — dashboard has no reverse-dep map entry."""
    (tmp_path / 'dashboard' / 'src' / 'dashboard').mkdir(parents=True)
    (tmp_path / 'dashboard' / 'src' / 'dashboard' / 'app.py').write_text('x = 1\n')
    return tmp_path


def _build_config(tmp_path: Path) -> OrchestratorConfig:
    """An OrchestratorConfig with escalation+orchestrator ModuleConfigs set post-construction.

    ``_module_configs`` is a Pydantic PrivateAttr (config.py:2682), not a
    constructor kwarg — set via direct attribute assignment, the established
    idiom (test_verify.py:5558 et al.).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    config._module_configs = {
        'escalation': ModuleConfig(
            prefix='escalation',
            test_command=_ESCALATION_TEST_COMMAND,
            lint_command=None,
            type_check_command=None,
        ),
        'orchestrator': ModuleConfig(
            prefix='orchestrator',
            test_command=_ORCHESTRATOR_TEST_COMMAND,
            lint_command='uv run --project orchestrator --directory orchestrator ruff check src/',
            type_check_command=(
                'uv run --project orchestrator --directory orchestrator pyright src/'
            ),
        ),
    }
    return config


class TestReverseDependencyModuleConfigs:
    """verify._reverse_dependency_module_configs(existing_files, config, worktree, already_scoped, content_cache=None).

    RED until step-6 implements the helper (AttributeError: module
    'orchestrator.verify' has no attribute '_reverse_dependency_module_configs').
    """

    def test_orchestrator_source_change_widens_to_scoped_escalation_pytest(self, tmp_path: Path):
        """An orchestrator-source-only diff widens to ONE escalation ModuleConfig.

        pytest-only (lint/type None) and scoped to the coupled test file —
        worktree-root-relative (no '--directory'), matching
        _scope_to_keyword's cwd-stripped output shape.
        """
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)

        result = verify._reverse_dependency_module_configs(
            ['orchestrator/src/orchestrator/merge_queue.py'],
            config,
            worktree,
            already_scoped={'orchestrator'},
        )

        assert len(result) == 1
        mc = result[0]
        assert mc.prefix == 'escalation'
        assert mc.lint_command is None
        assert mc.type_check_command is None
        assert mc.test_command is not None
        assert 'escalation/tests/test_server.py' in mc.test_command
        assert 'test_unrelated.py' not in mc.test_command
        assert '--directory' not in mc.test_command

    def test_already_scoped_escalation_dedupes_to_empty(self, tmp_path: Path):
        """already_scoped already containing 'escalation' -> no widening (dedup)."""
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)

        result = verify._reverse_dependency_module_configs(
            ['orchestrator/src/orchestrator/merge_queue.py'],
            config,
            worktree,
            already_scoped={'orchestrator', 'escalation'},
        )

        assert result == []

    def test_unscopable_test_command_is_skipped_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """A dependent's test_command with no literal 'pytest' keyword can't be
        narrowed by _scope_to_keyword (its own documented no-op contract:
        returns *cmd* unchanged when the keyword isn't present or the prefix
        doesn't parse). Widening it anyway would silently run the dependent's
        FULL suite instead of just the coupled file(s) -- contradicting the
        import-scoped cost/flake-bounding design. It must be skipped, with a
        warning logged rather than silently dropped.
        """
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)
        # Full-dict reassignment (not a subscript mutation) -- matches
        # _build_config's own idiom and keeps `_module_configs` a plain
        # dict[...] for pyright (the PrivateAttr's declared type is
        # dict[...] | None, so a bare subscript assignment on it triggers
        # reportOptionalSubscript outside the scope that narrowed it).
        config._module_configs = {
            **config.module_configs_or_empty,
            'escalation': ModuleConfig(
                prefix='escalation',
                test_command='./run-tests.sh',  # no literal 'pytest' -- unscopable
                lint_command=None,
                type_check_command=None,
            ),
        }

        with caplog.at_level(logging.WARNING, logger='orchestrator.verify'):
            result = verify._reverse_dependency_module_configs(
                ['orchestrator/src/orchestrator/merge_queue.py'],
                config,
                worktree,
                already_scoped={'orchestrator'},
            )

        assert result == [], (
            'an unscopable test_command must be skipped, not widened unscoped '
            f'(which would silently run the dependent\'s full suite); got: {result}'
        )
        assert any(
            'escalation' in r.getMessage() and 'could not be scoped' in r.getMessage()
            for r in caplog.records
        ), (
            f'expected a warning naming the skipped dependent; '
            f'got: {[r.getMessage() for r in caplog.records]}'
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("code_default_config")
class TestRunScopedVerificationReverseDependencyWidening:
    """Integration: run_scoped_verification's module_configs branch widens to escalation.

    RED until step-8 wires _reverse_dependency_module_configs into
    run_scoped_verification's module_configs branch.
    """

    async def test_orchestrator_only_diff_runs_escalations_coupled_test(self, tmp_path: Path):
        """An orchestrator-source-only task diff still executes escalation's coupled test.

        Uses the plan-authoritative flow (task_files given, module_configs=
        [orchestrator_mc] only — escalation is NOT among the task's own
        module_configs, mirroring a real orchestrator-only task).
        """
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)
        orchestrator_mc = config.module_configs_or_empty['orchestrator']

        recorded: list[str] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            recorded.append(cmd)
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            await verify.run_scoped_verification(
                worktree, config, [orchestrator_mc],
                task_files=['orchestrator/src/orchestrator/merge_queue.py'],
                role='merge',
            )

        escalation_pytest_cmds = [
            c for c in recorded if 'escalation' in c and 'pytest' in c
        ]
        assert any('test_server.py' in c for c in escalation_pytest_cmds), (
            f'expected a widened escalation pytest command targeting test_server.py; '
            f'recorded commands: {recorded}'
        )
        assert not any('test_unrelated.py' in c for c in escalation_pytest_cmds), (
            f'widened escalation command must not target the unrelated test file; '
            f'recorded commands: {recorded}'
        )

    async def test_widened_escalation_failure_fails_the_aggregated_result(self, tmp_path: Path):
        """A failing widened escalation pytest run must FAIL the aggregated VerifyResult.

        The point of widening is to gate the merge on the coupled test — a
        regression that appends the widened ModuleConfig to `scoped` (so it
        executes) but drops its VerifyResult from `_aggregate_results` would
        still pass every other test in this file, since they all stub
        `_run_cmd` to return rc=0 unconditionally. This pins that a real
        failure in the widened run actually surfaces and fails the gate.
        """
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)
        orchestrator_mc = config.module_configs_or_empty['orchestrator']

        recorded: list[str] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            recorded.append(cmd)
            if 'escalation' in cmd and 'pytest' in cmd:
                return 1, 'FAILED escalation/tests/test_server.py::test_x', False
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await verify.run_scoped_verification(
                worktree, config, [orchestrator_mc],
                task_files=['orchestrator/src/orchestrator/merge_queue.py'],
                role='merge',
            )

        escalation_pytest_cmds = [
            c for c in recorded if 'escalation' in c and 'pytest' in c
        ]
        assert any('test_server.py' in c for c in escalation_pytest_cmds), (
            f'expected the widened escalation pytest command to have run; recorded: {recorded}'
        )
        assert result.passed is False, (
            'a failing widened escalation test must fail the aggregated VerifyResult '
            '-- otherwise the widening runs cosmetically without gating the merge'
        )
        assert 'FAILED escalation/tests/test_server.py' in result.test_output

    async def test_deleted_orchestrator_source_named_in_task_files_still_widens(
        self, tmp_path: Path,
    ):
        """A deleted orchestrator SOURCE file still triggers widening when another
        orchestrator file (e.g. a test edited in the same diff) keeps
        'orchestrator' in `scoped` (amendment, review suggestion 2).

        `existing_files` (the `(worktree / f).exists()`-filtered list) drops
        the deleted source path entirely, hiding exactly the class of change
        most likely to break escalation's `from orchestrator.merge_queue
        import ...` contract. The trigger gate must key off the RAW
        `task_files` list instead, so a deletion/rename still counts.
        """
        worktree = _build_worktree_with_deleted_source(tmp_path)
        config = _build_config(tmp_path)
        orchestrator_mc = config.module_configs_or_empty['orchestrator']

        recorded: list[str] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            recorded.append(cmd)
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            await verify.run_scoped_verification(
                worktree, config, [orchestrator_mc],
                task_files=[
                    'orchestrator/src/orchestrator/merge_queue.py',  # deleted -- absent on disk
                    'orchestrator/tests/test_something.py',  # survives -- keeps 'orchestrator' scoped
                ],
                role='merge',
            )

        escalation_pytest_cmds = [
            c for c in recorded if 'escalation' in c and 'pytest' in c
        ]
        assert any('test_server.py' in c for c in escalation_pytest_cmds), (
            f'a deleted orchestrator source file (named in task_files but absent '
            f'on disk) must still widen to escalation; recorded: {recorded}'
        )


@pytest.mark.asyncio
class TestRunScopedVerificationReverseDependencyGuards:
    """Guard/regression tests for the step-8 wiring (task 2607 step-9/step-10).

    Same ``_run_cmd`` recording-spy idiom as
    ``TestRunScopedVerificationReverseDependencyWidening`` above. Pins three
    ways the widening must NOT fire (or must not double-fire), so the
    untriggered path stays byte-identical to pre-2607 behaviour.
    """

    @pytest.mark.usefixtures("code_default_config")
    async def test_orchestrator_test_only_diff_does_not_widen(self, tmp_path: Path):
        """A diff touching only orchestrator/tests/*.py (no orchestrator/src/) must not widen.

        Only a change to the depended-upon package's SOURCE can break the
        coupled import contract (escalation imports orchestrator SOURCE
        modules) — gating on ``<pkg>/src/`` keeps a test-only fix-forward
        from needlessly running escalation's suite.
        """
        worktree = _build_worktree_with_orchestrator_test(tmp_path)
        config = _build_config(tmp_path)
        orchestrator_mc = config.module_configs_or_empty['orchestrator']

        recorded: list[str] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            recorded.append(cmd)
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            await verify.run_scoped_verification(
                worktree, config, [orchestrator_mc],
                task_files=['orchestrator/tests/test_something.py'],
                role='merge',
            )

        assert not any('escalation' in c for c in recorded), (
            f'orchestrator test-only diff must not widen to escalation; recorded: {recorded}'
        )

    @pytest.mark.usefixtures("code_default_config")
    async def test_escalation_already_in_diff_is_not_double_widened(self, tmp_path: Path):
        """escalation already present (in diff + module_configs) -> exactly one invocation.

        already_scoped must union scoped-and-surviving prefixes with any
        dependent already present in the task's OWN module_configs, so an
        escalation-in-diff task is not double-widened.
        """
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)
        orchestrator_mc = config.module_configs_or_empty['orchestrator']
        escalation_mc = config.module_configs_or_empty['escalation']

        recorded: list[str] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            recorded.append(cmd)
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            await verify.run_scoped_verification(
                worktree, config, [orchestrator_mc, escalation_mc],
                task_files=[
                    'orchestrator/src/orchestrator/merge_queue.py',
                    'escalation/tests/test_server.py',
                ],
                role='merge',
            )

        escalation_pytest_cmds = [
            c for c in recorded if 'escalation' in c and 'pytest' in c
        ]
        assert len(escalation_pytest_cmds) == 1, (
            f'expected exactly one escalation pytest invocation (no double-widen); '
            f'recorded: {recorded}'
        )
        assert 'test_server.py' in escalation_pytest_cmds[0]

    async def test_unmapped_package_source_change_does_not_widen(self, tmp_path: Path):
        """A source-only diff to a package with no reverse-dep map entry widens to nothing."""
        worktree = _build_dashboard_worktree(tmp_path)
        config = OrchestratorConfig(project_root=worktree)
        dashboard_mc = ModuleConfig(
            prefix='dashboard',
            test_command=_DASHBOARD_TEST_COMMAND,
            lint_command=None,
            type_check_command=None,
        )
        config._module_configs = {'dashboard': dashboard_mc}

        recorded: list[str] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            recorded.append(cmd)
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            await verify.run_scoped_verification(
                worktree, config, [dashboard_mc],
                task_files=['dashboard/src/dashboard/app.py'],
                role='merge',
            )

        assert not any('escalation' in c for c in recorded), (
            f'unmapped package source change must not widen; recorded: {recorded}'
        )

    async def test_task_role_diff_does_not_widen(self, tmp_path: Path):
        """Widening is merge-path only (amendment, review suggestion 4).

        Every design decision motivating this task frames it as a
        merge-verify concern (the 3x RED-main recurrence this closes was
        always a merge-time escape) — an otherwise-identical orchestrator-
        source-only diff verified at role='task' must NOT widen to
        escalation, so per-task dev-loop verify latency for orchestrator
        tasks is unchanged from before this task.
        """
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)
        orchestrator_mc = config.module_configs_or_empty['orchestrator']

        recorded: list[str] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            recorded.append(cmd)
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            await verify.run_scoped_verification(
                worktree, config, [orchestrator_mc],
                task_files=['orchestrator/src/orchestrator/merge_queue.py'],
                role='task',
            )

        assert not any('escalation' in c for c in recorded), (
            f"role='task' diffs must not trigger reverse-dep widening; recorded: {recorded}"
        )
