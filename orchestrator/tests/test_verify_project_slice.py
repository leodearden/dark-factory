"""Verify scopes are spawned under their project's slice (task 3394).

Everything is driven through the PUBLIC ``run_verification`` entry. The
recording ``systemd-run`` fake from ``_fake_systemd_run`` lets the verify run
while each spawn's argv is captured; one further test reads the placement
back from a real systemd scope.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _fake_systemd_run import assert_scope_carries, recorded_scope_spawns, scope_payload_command
from _fake_systemd_run import fake_systemd_run as fake_systemd_run
from _real_project_slice import RealProjectSlice
from _real_project_slice import real_project_slice as real_project_slice

from orchestrator.config import FusedMemoryConfig, ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_verification

_TEST_CMD = 'true test-leg'
_LINT_CMD = 'true lint-leg'
_TYPE_CMD = 'true type-leg'

_SLICE_ARG = '--slice=df-my_proj.slice'


def _module_config(test_command: str = _TEST_CMD) -> ModuleConfig:
    return ModuleConfig(
        prefix='pkg',
        test_command=test_command,
        lint_command=_LINT_CMD,
        type_check_command=_TYPE_CMD,
    )


def _scoped_config(**overrides: object) -> OrchestratorConfig:
    return OrchestratorConfig(
        verify_use_cgroup_scope=True,
        fused_memory=FusedMemoryConfig(project_id='my-proj'),
        **overrides,
    )


def _assert_under_project_slice(record: list[str]) -> None:
    assert _SLICE_ARG in record, f'no {_SLICE_ARG} in {record!r}'
    assert record.index(_SLICE_ARG) < record.index('/bin/bash'), (
        f'--slice= must precede the /bin/bash payload: {record!r}'
    )


@pytest.fixture
def worktree(tmp_path) -> Path:
    wt = tmp_path / 'wt'
    wt.mkdir()
    return wt


@pytest.mark.usefixtures('code_default_config')
class TestScopeSpawnCarriesProjectSlice:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('role', 'weight'), [('merge', 100), ('task', 33), ('background', 10)],
    )
    async def test_every_leg_scope_is_under_the_project_slice_with_its_role_weight(
        self, fake_systemd_run, worktree, role, weight,
    ):
        result = await run_verification(
            worktree, _scoped_config(), module_config=_module_config(),
            max_retries=0, role=role,
        )
        assert result.passed, result.summary
        records = recorded_scope_spawns(fake_systemd_run)
        assert len(records) == 3
        for record in records:
            _assert_under_project_slice(record)
            assert_scope_carries(record, f'CPUWeight={weight}')

    @pytest.mark.asyncio
    async def test_every_segment_scope_of_a_chained_test_leg_is_under_the_project_slice(
        self, fake_systemd_run, worktree,
    ):
        (worktree / 'a').mkdir()
        (worktree / 'b').mkdir()
        result = await run_verification(
            worktree, _scoped_config(),
            module_config=_module_config('cd a && true x && cd ../b && true y'),
            max_retries=0, role='task', segment_chained_test=True,
        )
        assert result.passed, result.summary
        records = recorded_scope_spawns(fake_systemd_run)
        assert sorted(scope_payload_command(record) for record in records) == sorted(
            ['true x', 'true y', _LINT_CMD, _TYPE_CMD],
        )
        for record in records:
            _assert_under_project_slice(record)

    @pytest.mark.asyncio
    async def test_scope_off_spawns_no_scope(self, fake_systemd_run, worktree):
        config = OrchestratorConfig(
            verify_use_cgroup_scope=False,
            fused_memory=FusedMemoryConfig(project_id='my-proj'),
        )
        result = await run_verification(
            worktree, config, module_config=_module_config(), max_retries=0, role='task',
        )
        assert result.passed, result.summary
        assert recorded_scope_spawns(fake_systemd_run) == []


def _scope_spawnable_in(slice_name: str) -> bool:
    try:
        probe = subprocess.run(
            [
                'systemd-run', '--user', '--scope', '--quiet', '--collect',
                f'--slice={slice_name}', '--', 'true',
            ],
            capture_output=True, timeout=15,
        )
    except Exception:
        return False
    return probe.returncode == 0


@pytest.mark.usefixtures('code_default_config')
class TestRealScopeProjectSlice:

    @pytest.mark.asyncio
    async def test_task_scope_runs_inside_the_project_slice(
        self, worktree, real_project_slice: RealProjectSlice,
    ):
        if not _scope_spawnable_in(real_project_slice.slice_name):
            pytest.skip('systemd user scopes under a slice unavailable on this host')
        config = OrchestratorConfig(
            verify_use_cgroup_scope=True,
            fused_memory=FusedMemoryConfig(project_id=real_project_slice.project_id),
        )
        module_config = ModuleConfig(
            prefix='pkg',
            test_command='cat /proc/self/cgroup',
            lint_command=None,
            type_check_command=None,
        )
        result = await run_verification(
            worktree, config, module_config=module_config, max_retries=0, role='task',
        )
        assert result.passed, result.summary
        assert f'/df.slice/{real_project_slice.slice_name}/df-verify-' in result.test_output
