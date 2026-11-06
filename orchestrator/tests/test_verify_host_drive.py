"""The verify-host drive support module's own contract (``_verify_host_drive``)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from _verify_host_drive import (
    passing_verify_result,
    probe_free_orch_config,
    runner_double,
    verify_merge_request,
)

from orchestrator.config import OrchestratorConfig
from orchestrator.verify_runner import RunnerUnavailable


class TestProbeFreeOrchConfig:
    def test_switches_env_probes_off_over_a_yaml_that_declares_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / 'orch.yaml'
        path.write_text('verify_env_probes:\n  - key: uv\n    argv: ["uv", "--version"]\n')
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(path))

        assert [p.key for p in OrchestratorConfig().verify_env_probes] == ['uv']
        assert probe_free_orch_config().verify_env_probes == []


@pytest.mark.asyncio
class TestRunnerDouble:
    async def test_raises_the_injected_error_on_dispatch(self) -> None:
        runner = runner_double('h', is_local=False, error=RunnerUnavailable('down'))

        assert runner.name == 'h'
        assert runner.is_local is False
        with pytest.raises(RunnerUnavailable, match='down'):
            await runner.run_merge_verify('sha', None)

    async def test_reports_a_passing_verify_and_a_healthy_host_by_default(self) -> None:
        runner = runner_double('h', is_local=True)

        assert await runner.health() is True
        assert (await runner.run_merge_verify('sha', None)).passed is True

    async def test_reports_an_explicit_result_as_that_same_object(self) -> None:
        result = passing_verify_result(cause_hint='x')
        runner = runner_double('h', is_local=True, result=result)

        assert await runner.run_merge_verify('sha', None) is result


@pytest.mark.asyncio
class TestVerifyMergeRequest:
    async def test_binds_its_result_to_the_running_loop(self, tmp_path: Path) -> None:
        req = verify_merge_request(
            probe_free_orch_config(), task_files=['src/a.py'], worktree=tmp_path,
        )

        assert req.result.get_loop() is asyncio.get_running_loop()
        assert not req.result.done()
        assert req.task_files == ['src/a.py']
        assert req.worktree == tmp_path
