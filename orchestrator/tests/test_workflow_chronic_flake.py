"""Tests for the chronic-flake auto-file workflow wiring (task 2358, step-17/18).

Covers:
- ``TaskWorkflow._maybe_file_chronic_flakes``: enabled-gate, delegates to
  ``chronic_flake.maybe_file_chronic_flake_tasks`` via a
  ``SchedulerChronicFlakeTaskClient`` built over ``self.scheduler``, and
  swallows any exception the entry point raises — belt-and-suspenders over
  ``chronic_flake``'s own internal catch-all (see chronic_flake.py
  step-13/14).
- ``_verify_debugfix_loop`` invokes ``_maybe_file_chronic_flakes`` right
  after the ``result is None`` guard (i.e. on every completed verify,
  green or red) without altering the DONE/BLOCKED outcome.

Both test classes build a minimal ``TaskWorkflow`` via ``object.__new__`` +
direct attribute injection rather than the full constructor (which requires
a real ``TaskAssignment``/``GitOps``/briefing/mcp rig) — only the
attributes the methods under test actually touch are set, per the task's
test-scope note.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.config import ChronicFlakeConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


def _verify_result(passed: bool = True, test_output: str = '') -> VerifyResult:
    return VerifyResult(
        passed=passed, test_output=test_output, lint_output='', type_output='',
        summary='ok',
    )


# ── Step-17 / Step-18: TaskWorkflow._maybe_file_chronic_flakes ───────────────


class TestMaybeFileChronicFlakes:
    """Unit tests for the thin wrapper method in isolation."""

    def _config(self, tmp_path, *, enabled: bool) -> OrchestratorConfig:
        config = OrchestratorConfig(project_root=tmp_path)
        config.chronic_flake = ChronicFlakeConfig(enabled=enabled)
        return config

    def _workflow(self, config: OrchestratorConfig, scheduler=None) -> TaskWorkflow:
        workflow = object.__new__(TaskWorkflow)
        workflow.config = config
        workflow.scheduler = scheduler
        workflow.task_id = '2358-test'
        return workflow

    @pytest.mark.asyncio
    async def test_noop_when_disabled(self, tmp_path, monkeypatch):
        entry_point = AsyncMock(return_value=[])
        monkeypatch.setattr(
            'orchestrator.chronic_flake.maybe_file_chronic_flake_tasks', entry_point,
        )
        workflow = self._workflow(self._config(tmp_path, enabled=False))

        await workflow._maybe_file_chronic_flakes(_verify_result())

        entry_point.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enabled_delegates_to_maybe_file_chronic_flake_tasks(self, tmp_path, monkeypatch):
        from orchestrator.chronic_flake import SchedulerChronicFlakeTaskClient

        entry_point = AsyncMock(return_value=['test_a.sh'])
        monkeypatch.setattr(
            'orchestrator.chronic_flake.maybe_file_chronic_flake_tasks', entry_point,
        )
        # Never actually dispatched to — the entry point itself is patched, so
        # the client is only ever constructed and handed through, never called.
        scheduler = object()
        workflow = self._workflow(self._config(tmp_path, enabled=True), scheduler=scheduler)
        result = _verify_result(test_output='some verify output')

        await workflow._maybe_file_chronic_flakes(result)

        entry_point.assert_awaited_once()
        call_args = entry_point.await_args
        assert call_args.args[0] == 'some verify output'
        assert call_args.args[1] is workflow.config
        assert isinstance(call_args.args[2], SchedulerChronicFlakeTaskClient)

    @pytest.mark.asyncio
    async def test_swallows_exception_from_entry_point(self, tmp_path, monkeypatch):
        entry_point = AsyncMock(side_effect=RuntimeError('boom'))
        monkeypatch.setattr(
            'orchestrator.chronic_flake.maybe_file_chronic_flake_tasks', entry_point,
        )
        workflow = self._workflow(self._config(tmp_path, enabled=True))

        result = await workflow._maybe_file_chronic_flakes(_verify_result())

        assert result is None
        entry_point.assert_awaited_once()


# ── Step-17 / Step-18: _verify_debugfix_loop invokes the hook ───────────────


class TestVerifyDebugfixLoopInvokesChronicFlakeHook:
    """Confirms the call site: the hook fires right after the ``result is
    None`` guard (on every completed verify — green or red — see
    chronic_flake.py's module docstring / task 2358's design decision), and
    never alters the DONE/BLOCKED outcome."""

    def _workflow(self, tmp_path) -> TaskWorkflow:
        config = OrchestratorConfig(project_root=tmp_path)
        config.rebase_before_verify = False
        workflow = object.__new__(TaskWorkflow)
        workflow.config = config
        workflow.worktree = tmp_path
        workflow.artifacts = object()
        workflow._last_verify_result = None
        return workflow

    @pytest.mark.asyncio
    async def test_calls_hook_after_done_result_without_changing_outcome(self, tmp_path):
        workflow = self._workflow(tmp_path)
        passed_result = _verify_result(passed=True)
        workflow._run_scoped_verification_with_infra_retry = AsyncMock(
            return_value=passed_result,
        )
        workflow._maybe_file_chronic_flakes = AsyncMock()

        outcome = await workflow._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.DONE
        workflow._maybe_file_chronic_flakes.assert_awaited_once_with(passed_result)

    @pytest.mark.asyncio
    async def test_does_not_call_hook_when_result_is_none(self, tmp_path):
        """The hook fires only once ``result`` is guaranteed non-None — the
        BLOCKED short-circuit (infra retry window exhausted) must return
        BEFORE ``_maybe_file_chronic_flakes`` is ever invoked."""
        workflow = self._workflow(tmp_path)
        workflow._run_scoped_verification_with_infra_retry = AsyncMock(return_value=None)
        workflow._maybe_file_chronic_flakes = AsyncMock()

        outcome = await workflow._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.BLOCKED
        workflow._maybe_file_chronic_flakes.assert_not_awaited()
