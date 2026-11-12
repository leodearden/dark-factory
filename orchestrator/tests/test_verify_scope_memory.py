"""Verify scopes carry the configured cgroup memory limits (task 6106).

Everything is driven through the PUBLIC ``run_verification`` entry, with the
recording ``systemd-run`` from ``_fake_systemd_run`` first on PATH, so the real
``shutil.which`` + ``create_subprocess_exec`` resolution runs and the verify
still executes. One further test reads the limits back from a real systemd
scope.
"""

from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _fake_systemd_run import assert_scope_carries, recorded_scope_spawns
from _fake_systemd_run import fake_systemd_run as fake_systemd_run

from orchestrator.config import (
    ModuleConfig,
    OrchestratorConfig,
    VerifyCgroupMemoryConfig,
    apply_reload,
)
from orchestrator.verify import run_verification

_TEST_CMD = 'true test-leg'
_LINT_CMD = 'true lint-leg'
_TYPE_CMD = 'true type-leg'

_READ_OWN_MEMORY_LIMITS = (
    'cat'
    ' "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.high"'
    ' "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max"'
    ' "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.swap.max"'
)
_DISTINCT_LIMITS_GIB = (61, 62, 63)


def _module_config() -> ModuleConfig:
    return ModuleConfig(
        prefix='pkg',
        test_command=_TEST_CMD,
        lint_command=_LINT_CMD,
        type_check_command=_TYPE_CMD,
    )


@pytest.fixture
def worktree(tmp_path) -> Path:
    wt = tmp_path / 'wt'
    wt.mkdir()
    return wt


@pytest.mark.usefixtures('code_default_config')
class TestScopeSpawnCarriesMemoryLimits:

    @pytest.mark.asyncio
    @pytest.mark.parametrize('role', ['merge', 'task', 'background'])
    async def test_every_leg_scope_carries_configured_limits(
        self, fake_systemd_run, worktree, role,
    ):
        config = OrchestratorConfig(
            verify_use_cgroup_scope=True,
            verify_cgroup_memory=VerifyCgroupMemoryConfig(
                high_gib=24, max_gib=40, swap_max_gib=32,
            ),
        )
        result = await run_verification(
            worktree, config, module_config=_module_config(), max_retries=0, role=role,
        )
        assert result.passed, result.summary
        records = recorded_scope_spawns(fake_systemd_run)
        assert len(records) == 3
        for record in records:
            for prop in ('MemoryHigh=24G', 'MemoryMax=40G', 'MemorySwapMax=32G'):
                assert_scope_carries(record, prop)

    @pytest.mark.asyncio
    async def test_every_segment_scope_of_a_chained_test_leg_carries_limits(
        self, fake_systemd_run, worktree,
    ):
        config = OrchestratorConfig(
            verify_use_cgroup_scope=True,
            verify_cgroup_memory=VerifyCgroupMemoryConfig(
                high_gib=24, max_gib=40, swap_max_gib=32,
            ),
        )
        module_config = ModuleConfig(
            prefix='pkg',
            test_command='true segment-a && true segment-b',
            lint_command=_LINT_CMD,
            type_check_command=_TYPE_CMD,
        )
        result = await run_verification(
            worktree, config, module_config=module_config, max_retries=0, role='task',
            segment_chained_test=True,
        )
        assert result.passed, result.summary
        records = recorded_scope_spawns(fake_systemd_run)
        assert sorted(record[-1] for record in records) == sorted(
            ['true segment-a', 'true segment-b', _LINT_CMD, _TYPE_CMD],
        )
        for record in records:
            for prop in ('MemoryHigh=24G', 'MemoryMax=40G', 'MemorySwapMax=32G'):
                assert_scope_carries(record, prop)

    @pytest.mark.asyncio
    async def test_unset_limits_leave_scope_argv_unchanged(self, fake_systemd_run, worktree):
        config = OrchestratorConfig(verify_use_cgroup_scope=True)
        result = await run_verification(
            worktree, config, module_config=_module_config(), max_retries=0, role='task',
        )
        assert result.passed, result.summary
        records = recorded_scope_spawns(fake_systemd_run)
        assert len(records) == 3
        for record in records:
            assert record[:4] == ['--user', '--scope', '--quiet', '--collect'], record
            assert re.fullmatch(r'--unit=df-verify-.+\.scope', record[4]), record
            assert record[5:8] == ['-p', 'CPUWeight=33', '/bin/bash'], record

    @pytest.mark.asyncio
    async def test_reloaded_limit_applies_to_next_spawn(
        self, fake_systemd_run, worktree, monkeypatch, tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        live = OrchestratorConfig(verify_use_cgroup_scope=True)
        reloaded: dict[str, Any] = {
            'verify_use_cgroup_scope': True, 'verify_cgroup_memory': {'max_gib': 48},
        }
        apply_reload(live, OrchestratorConfig(**reloaded))
        result = await run_verification(
            worktree, live, module_config=_module_config(), max_retries=0, role='task',
        )
        assert result.passed, result.summary
        records = recorded_scope_spawns(fake_systemd_run)
        assert len(records) == 3
        for record in records:
            assert_scope_carries(record, 'MemoryMax=48G')


def _expected_limit_bytes() -> list[str]:
    return [str(gib << 30) for gib in _DISTINCT_LIMITS_GIB]


@functools.cache
def _memory_limits_settable() -> bool:
    """Whether this host can spawn a user scope with memory limits and read them back.

    Probed independently of verify.py, with distinctive limits read back
    exactly: the skip then depends only on the host, so a property verify.py
    drops shows up in the test as a wrong value (a failure), never as a skip.
    """
    high, cap, swap = _DISTINCT_LIMITS_GIB
    try:
        probe = subprocess.run(
            [
                'systemd-run', '--user', '--scope', '--quiet', '--collect',
                '-p', f'MemoryHigh={high}G',
                '-p', f'MemoryMax={cap}G',
                '-p', f'MemorySwapMax={swap}G',
                '--', 'sh', '-c', _READ_OWN_MEMORY_LIMITS,
            ],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return False
    return probe.stdout.split() == _expected_limit_bytes()


@pytest.mark.usefixtures('code_default_config')
class TestRealScopeMemoryLimits:
    """The scope tag derives from conftest's tmp project_root, so no laptop
    prefix drop-in can match the scope and override its limits.
    """

    @pytest.mark.asyncio
    async def test_task_scope_reads_back_configured_limits(self, worktree):
        if not _memory_limits_settable():
            pytest.skip('systemd user scopes with memory limits unavailable on this host')
        high, cap, swap = _DISTINCT_LIMITS_GIB
        config = OrchestratorConfig(
            verify_use_cgroup_scope=True,
            verify_cgroup_memory=VerifyCgroupMemoryConfig(
                high_gib=high, max_gib=cap, swap_max_gib=swap,
            ),
        )
        module_config = ModuleConfig(
            prefix='pkg',
            test_command=_READ_OWN_MEMORY_LIMITS,
            lint_command=None,
            type_check_command=None,
        )
        result = await run_verification(
            worktree, config, module_config=module_config, max_retries=0, role='task',
        )
        assert result.passed, result.summary
        assert result.test_output.split() == _expected_limit_bytes()
