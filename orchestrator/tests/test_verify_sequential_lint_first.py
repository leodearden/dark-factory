"""Tests for the sequential_lint_first knob — task 2832.

A per-project module-config bool ``sequential_lint_first`` (default False)
that, when enabled AND ``concurrent_verify`` is False AND role=='merge',
makes the sequential verify branch run LINT FIRST and short-circuit on a
lint failure (record test+type as ``CheckRun.skipped``, fail the attempt) —
so a lint-only-red merge no longer burns the ~50-min test phase before the
~4-min lint phase. Task/background roles and the concurrent branch stay
byte-unchanged.

Test coverage:
  test-config-field:      new config field schema (ModuleConfig +
                          OrchestratorConfig + _OVERRIDABLE_FIELDS membership)
  test-resolver:          verify._resolve_sequential_lint_first
                          (module-override-wins)
  test-lint-first-branch: run_verification merge-role lint-first phase order
                          + lint-red short-circuit (mocked _run_cmd)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator import verify
from orchestrator.config import (
    _OVERRIDABLE_FIELDS,
    ModuleConfig,
    OrchestratorConfig,
)


class TestSequentialLintFirstConfig:
    """test-config-field: the new config field's schema.

    RED today because ``sequential_lint_first`` does not yet exist on
    ModuleConfig / OrchestratorConfig / _OVERRIDABLE_FIELDS. Mirrors the
    ``concurrent_verify`` field triple exactly (config.py:1925, :1959, :2421).
    """

    def test_orchestrator_config_default_is_false(self, tmp_path: Path):
        """OrchestratorConfig default is opt-in False (byte-unchanged for
        existing configs — the branch never activates unless enabled)."""
        assert OrchestratorConfig(project_root=tmp_path).sequential_lint_first is False

    def test_orchestrator_config_accepts_true(self, tmp_path: Path):
        """OrchestratorConfig(sequential_lint_first=True) round-trips."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=True)
        assert config.sequential_lint_first is True

    def test_module_config_default_is_none(self):
        """ModuleConfig default is None (fall through to top-level config),
        mirroring concurrent_verify's bool|None override semantics."""
        assert ModuleConfig(prefix='x').sequential_lint_first is None

    def test_module_config_accepts_true(self):
        """ModuleConfig(sequential_lint_first=True) round-trips."""
        assert ModuleConfig(prefix='x', sequential_lint_first=True).sequential_lint_first is True

    def test_is_overridable_field(self):
        """Membership in _OVERRIDABLE_FIELDS makes per-module orchestrator.yaml
        discovery (config.py:2075 kwargs comprehension) load it with zero extra
        loader code — same as concurrent_verify."""
        assert 'sequential_lint_first' in _OVERRIDABLE_FIELDS


class TestResolveSequentialLintFirst:
    """test-resolver: verify._resolve_sequential_lint_first(config, module_config).

    RED today because the helper does not exist yet (AttributeError). Mirrors
    _resolve_concurrent_verify (verify.py:3219): module override wins over the
    top-level config value.
    """

    def test_config_only_returns_config_value_true(self, tmp_path: Path):
        """module_config=None → returns config.sequential_lint_first (True)."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=True)
        assert verify._resolve_sequential_lint_first(config, None) is True

    def test_config_only_returns_config_value_false(self, tmp_path: Path):
        """module_config=None → returns config.sequential_lint_first (False)."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=False)
        assert verify._resolve_sequential_lint_first(config, None) is False

    def test_module_override_wins_true_over_false(self, tmp_path: Path):
        """config False, module True → module override wins → True."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=False)
        mc = ModuleConfig(prefix='m', sequential_lint_first=True)
        assert verify._resolve_sequential_lint_first(config, mc) is True

    def test_module_override_wins_false_over_true(self, tmp_path: Path):
        """config True, module False → module override wins → False."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=True)
        mc = ModuleConfig(prefix='m', sequential_lint_first=False)
        assert verify._resolve_sequential_lint_first(config, mc) is False

    def test_module_none_falls_back_to_config(self, tmp_path: Path):
        """module_config.sequential_lint_first is None → fall back to config."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=True)
        mc = ModuleConfig(prefix='m')  # sequential_lint_first defaults to None
        assert mc.sequential_lint_first is None
        assert verify._resolve_sequential_lint_first(config, mc) is True


class TestSequentialLintFirstBranch:
    """test-lint-first-branch: the merge-role lint-first phase order +
    lint-red short-circuit in run_verification.

    Reuses test_verify_env_transient.py's `patch('orchestrator.verify._run_cmd',
    side_effect=fake_cmd)` seam: fake_cmd records the invoked command and returns
    (rc, out, timed_out) by token substring. The three legs use distinct tokens
    (TESTLEG/LINTLEG/TYPELEG) so we can assert exactly which phases ran.

    All configs use `echo` commands (ToolKind.OPAQUE, never PYTEST) so the
    pytest-only env-recovery retry never fires — the branch under test is the
    only thing exercised.
    """

    @staticmethod
    def _make_config(project_root: Path, *, sequential_lint_first: bool) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=project_root,
            test_command='echo TESTLEG',
            lint_command='echo LINTLEG',
            type_check_command='echo TYPELEG',
            concurrent_verify=False,
            sequential_lint_first=sequential_lint_first,
        )

    @staticmethod
    def _fake_cmd(invoked_cmds: list[str], *, lint_red: bool):
        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            invoked_cmds.append(cmd)
            if 'LINTLEG' in cmd and lint_red:
                return 1, 'file.py:10:1: E501 line too long (92 > 88 characters)', False
            return 0, '', False

        return fake_cmd

    @pytest.mark.asyncio
    async def test_lint_red_short_circuits_test_and_type(self, tmp_path: Path):
        """Case (i): knob on + role='merge' + lint-red ⇒ test/type NOT invoked
        (short-circuit) and the attempt fails. RED today — the plain-sequential
        branch runs all three legs regardless of the lint result."""
        config = self._make_config(tmp_path, sequential_lint_first=True)
        invoked: list[str] = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._fake_cmd(invoked, lint_red=True),
        ):
            result = await verify.run_verification(tmp_path, config, role='merge')

        assert any('LINTLEG' in c for c in invoked), f'lint must run: {invoked!r}'
        assert not any('TESTLEG' in c for c in invoked), (
            f'test must be short-circuited on lint-red: {invoked!r}'
        )
        assert not any('TYPELEG' in c for c in invoked), (
            f'type must be short-circuited on lint-red: {invoked!r}'
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_green_path_result_identical_to_knob_off(self, tmp_path: Path):
        """Case (ii): knob on + role='merge' + all-green ⇒ all three legs run and
        the VerifyResult is byte-identical (compared fields) to a knob-off
        all-green run. Uses isolated worktrees so warm-marker state can't leak
        between the two runs."""
        wt_on = tmp_path / 'on'
        wt_off = tmp_path / 'off'
        wt_on.mkdir()
        wt_off.mkdir()
        config_on = self._make_config(tmp_path, sequential_lint_first=True)
        config_off = self._make_config(tmp_path, sequential_lint_first=False)

        invoked_on: list[str] = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._fake_cmd(invoked_on, lint_red=False),
        ):
            result_on = await verify.run_verification(wt_on, config_on, role='merge')

        invoked_off: list[str] = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._fake_cmd(invoked_off, lint_red=False),
        ):
            result_off = await verify.run_verification(wt_off, config_off, role='merge')

        # All three legs ran on the knob-on green path.
        assert any('TESTLEG' in c for c in invoked_on), invoked_on
        assert any('LINTLEG' in c for c in invoked_on), invoked_on
        assert any('TYPELEG' in c for c in invoked_on), invoked_on
        assert result_on.passed is True

        # Result-identical across the observable fields (duration_secs excluded —
        # it is compare=False wall-clock noise).
        for field_name in (
            'passed', 'category', 'summary', 'test_output', 'lint_output',
            'type_output', 'timed_out', 'cause_hint',
        ):
            assert getattr(result_on, field_name) == getattr(result_off, field_name), (
                f'{field_name} differs: {getattr(result_on, field_name)!r} != '
                f'{getattr(result_off, field_name)!r}'
            )

    @pytest.mark.asyncio
    async def test_task_role_runs_test_even_on_lint_red(self, tmp_path: Path):
        """Case (iii): knob on + role='task' + lint-red ⇒ test IS invoked. The
        branch is merge-only, so a task-role verify keeps today's test→lint→type
        order regardless of the knob."""
        config = self._make_config(tmp_path, sequential_lint_first=True)
        invoked: list[str] = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._fake_cmd(invoked, lint_red=True),
        ):
            result = await verify.run_verification(tmp_path, config, role='task')

        assert any('TESTLEG' in c for c in invoked), (
            f'task-role test must still run (branch inactive for non-merge): {invoked!r}'
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_knob_off_merge_runs_all_phases(self, tmp_path: Path):
        """Case (iv): knob off + role='merge' + lint-red ⇒ test AND type both
        invoked (today's behavior — no short-circuit without the knob)."""
        config = self._make_config(tmp_path, sequential_lint_first=False)
        invoked: list[str] = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._fake_cmd(invoked, lint_red=True),
        ):
            result = await verify.run_verification(tmp_path, config, role='merge')

        assert any('TESTLEG' in c for c in invoked), invoked
        assert any('TYPELEG' in c for c in invoked), invoked
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_lint_red_classification_takes_lint_path(self, tmp_path: Path):
        """Case (v): knob on + role='merge' + lint-red ⇒ the failure classifies
        on the lint leg only. summary names 'lint issues' (not tests/type), and
        the category is structurally never an env/semaphore transient (those are
        PYTEST/environmental-output-gated — impossible for a skipped-test,
        lint-only attempt)."""
        config = self._make_config(tmp_path, sequential_lint_first=True)
        invoked: list[str] = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._fake_cmd(invoked, lint_red=True),
        ):
            result = await verify.run_verification(tmp_path, config, role='merge')

        assert result.passed is False
        assert 'lint issues' in result.summary, result.summary
        assert 'tests failed' not in result.summary, result.summary
        assert 'type errors' not in result.summary, result.summary
        assert result.category not in {'env_transient', 'semaphore_timeout'}, result.category
