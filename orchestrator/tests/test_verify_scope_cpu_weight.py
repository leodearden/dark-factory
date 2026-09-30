"""Verify scopes carry a per-role cgroup CPUWeight (task 5205).

Everything is driven through the PUBLIC ``run_verification`` entry. A fake
``systemd-run`` executable is prepended to PATH: it records its argv, drops its
own options and execs the payload, so the real ``shutil.which`` +
``create_subprocess_exec`` resolution runs and the verify still executes. One
further test reads the weight back from a real systemd scope.

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_verify_admission_reload.py's stated rationale).
"""

from __future__ import annotations

import functools
import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig, apply_reload
from orchestrator.verify import run_verification
from shared.verify_admission import nice_prefix

_TEST_CMD = 'true test-leg'
_LINT_CMD = 'true lint-leg'
_TYPE_CMD = 'true type-leg'

_READ_OWN_CPU_WEIGHT = 'cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cpu.weight"'


@pytest.fixture
def fake_systemd_run(tmp_path, monkeypatch) -> Path:
    """Put a recording ``systemd-run`` first on PATH; return its argv-record dir."""
    argv_dir = tmp_path / 'argv'
    argv_dir.mkdir()
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    script = fakebin / 'systemd-run'
    script.write_text(
        '#!/bin/sh\n'
        f'rec=$(mktemp {shlex.quote(str(argv_dir))}/argv.XXXXXX)\n'
        'for a in "$@"; do printf \'%s\\0\' "$a" >> "$rec"; done\n'
        'while [ $# -gt 0 ]; do\n'
        '  case "$1" in -p) shift 2;; -*) shift;; *) break;; esac\n'
        'done\n'
        'exec "$@"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{fakebin}{os.pathsep}' + os.environ['PATH'])
    return argv_dir


def _recorded(argv_dir: Path) -> list[list[str]]:
    return [
        p.read_bytes().decode().split('\0')[:-1]
        for p in sorted(argv_dir.iterdir())
    ]


def _assert_carries_weight(record: list[str], weight: int) -> None:
    """The adjacent pair ``-p CPUWeight=<weight>`` precedes the payload.

    systemd-run options must come before the command, or systemd-run hands
    them to the payload instead of applying them to the scope.
    """
    pairs = [
        i for i in range(len(record) - 1)
        if record[i] == '-p' and record[i + 1] == f'CPUWeight={weight}'
    ]
    assert pairs, f'no -p CPUWeight={weight} in {record!r}'
    assert pairs[0] < record.index('/bin/bash'), (
        f'CPUWeight must precede the /bin/bash payload: {record!r}'
    )


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
class TestScopeSpawnCarriesRoleWeight:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('role', 'expected'), [('merge', 100), ('task', 33), ('background', 10)],
    )
    async def test_every_leg_scope_carries_role_weight(
        self, fake_systemd_run, worktree, role, expected,
    ):
        config = OrchestratorConfig(verify_use_cgroup_scope=True)
        result = await run_verification(
            worktree, config, module_config=_module_config(), max_retries=0, role=role,
        )
        assert result.passed, result.summary
        records = _recorded(fake_systemd_run)
        assert len(records) == 3
        for record in records:
            _assert_carries_weight(record, expected)

    @pytest.mark.asyncio
    async def test_offline_role_spawns_scope_without_weight(self, fake_systemd_run, worktree):
        config = OrchestratorConfig(verify_use_cgroup_scope=True)
        result = await run_verification(
            worktree, config, module_config=_module_config(), max_retries=0,
            role='offline',  # type: ignore[arg-type]
        )
        assert result.passed, result.summary
        records = _recorded(fake_systemd_run)
        assert len(records) == 3
        for record in records:
            assert '-p' not in record, record
            assert not any(tok.startswith('CPUWeight=') for tok in record), record

    @pytest.mark.asyncio
    async def test_reloaded_weight_applies_to_next_spawn(
        self, fake_systemd_run, worktree, monkeypatch, tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        live = OrchestratorConfig(verify_use_cgroup_scope=True)
        apply_reload(
            live,
            OrchestratorConfig(verify_use_cgroup_scope=True, verify_cgroup_cpu_weight_task=50),
        )
        result = await run_verification(
            worktree, live, module_config=_module_config(), max_retries=0, role='task',
        )
        assert result.passed, result.summary
        records = _recorded(fake_systemd_run)
        assert len(records) == 3
        for record in records:
            _assert_carries_weight(record, 50)

    @pytest.mark.asyncio
    async def test_scope_off_spawns_no_scope(self, fake_systemd_run, worktree):
        config = OrchestratorConfig(verify_use_cgroup_scope=False)
        result = await run_verification(
            worktree, config, module_config=_module_config(), max_retries=0, role='task',
        )
        assert result.passed, result.summary
        assert _recorded(fake_systemd_run) == []

    @pytest.mark.asyncio
    @pytest.mark.real_verify_admission
    async def test_weight_and_nice_tier_are_layered(
        self, fake_systemd_run, worktree, tmp_path,
    ):
        config = OrchestratorConfig(
            verify_use_cgroup_scope=True,
            verify_admission_slots_dir=str(tmp_path / 'slots'),
        )
        result = await run_verification(
            worktree, config, module_config=_module_config(), max_retries=0, role='task',
        )
        assert result.passed, result.summary
        test_leg = [r for r in _recorded(fake_systemd_run) if 'test-leg' in r[-1]]
        assert len(test_leg) == 1
        _assert_carries_weight(test_leg[0], 33)
        assert test_leg[0][-1].startswith(shlex.join(nice_prefix('task'))), test_leg[0]


@functools.cache
def _cpu_weight_settable() -> bool:
    """Whether this host can spawn a user scope with a CPUWeight and read it back.

    Probed independently of verify.py: a scope has no cpu.weight file until a
    CPU property is set on it, so keying the skip on the file would turn a
    dropped property into a skip rather than a failure.
    """
    try:
        probe = subprocess.run(
            [
                'systemd-run', '--user', '--scope', '--quiet', '--collect',
                '-p', 'CPUWeight=7', '--', 'sh', '-c', _READ_OWN_CPU_WEIGHT,
            ],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return False
    return probe.stdout.strip() == '7'


@pytest.mark.usefixtures('code_default_config')
class TestRealScopeCpuWeight:

    @pytest.mark.asyncio
    async def test_task_scope_reads_back_weight_33(self, worktree):
        if not _cpu_weight_settable():
            pytest.skip('systemd user scopes with CPUWeight unavailable on this host')
        config = OrchestratorConfig(verify_use_cgroup_scope=True)
        module_config = ModuleConfig(
            prefix='pkg',
            test_command=_READ_OWN_CPU_WEIGHT,
            lint_command=None,
            type_check_command=None,
        )
        result = await run_verification(
            worktree, config, module_config=module_config, max_retries=0, role='task',
        )
        assert result.passed, result.summary
        assert result.test_output.strip() == '33'
