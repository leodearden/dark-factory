"""A timed-out verify reaps the scopes its command moved itself into (task 6078).

Everything is driven through the PUBLIC ``run_verification`` entry. The guards
use recording ``systemd-run`` / ``systemctl`` stand-ins first on PATH; the
confirmation runs a reify-shaped governed merge leg under real systemd, whose
governed exec relocates the workload with a nested ``systemd-run --scope
--slice``. A failing or cancelled discovery has no host trigger, so those
tests replace verify.py's ``relocated_scopes`` binding.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import random
import shlex
import signal
import stat
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from _fake_systemd_run import fake_systemd_run as fake_systemd_run
from _fake_systemd_run import recorded_scope_spawns
from shared.proc_group import read_stat_fields

from orchestrator.config import CpuGovernConfig, ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_verification

_PROC = Path('/proc')


@pytest.fixture
def worktree(tmp_path) -> Path:
    wt = tmp_path / 'wt'
    wt.mkdir()
    return wt


@pytest.fixture
def fake_systemctl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a recording ``systemctl`` first on PATH; return its argv-record dir."""
    argv_dir = tmp_path / 'systemctl-argv'
    argv_dir.mkdir()
    fakebin = tmp_path / 'fakectl'
    fakebin.mkdir()
    script = fakebin / 'systemctl'
    script.write_text(
        '#!/bin/sh\n'
        f'rec=$(mktemp {shlex.quote(str(argv_dir))}/argv.XXXXXX)\n'
        'for a in "$@"; do printf \'%s\\0\' "$a" >> "$rec"; done\n'
        'exit 0\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{fakebin}{os.pathsep}' + os.environ['PATH'])
    return argv_dir


def _killed_units(argv_dir: Path) -> set[str]:
    """The unit of every recorded ``systemctl --user kill|stop ... <unit>`` call."""
    records = [p.read_bytes().decode().split('\0')[:-1] for p in argv_dir.iterdir()]
    return {
        record[-1]
        for record in records
        if record[:1] == ['--user'] and len(record) > 2 and record[1] in ('kill', 'stop')
    }


def _launched_unit(systemd_run_argv_dir: Path) -> str:
    """The ``--unit`` of the one recorded scope spawn."""
    (spawn,) = recorded_scope_spawns(systemd_run_argv_dir)
    (launched,) = [arg.removeprefix('--unit=') for arg in spawn if arg.startswith('--unit=')]
    return launched


def _timing_out_module(test_command: str, timeout_secs: float) -> ModuleConfig:
    return ModuleConfig(
        prefix='pkg',
        test_command=test_command,
        lint_command=None,
        type_check_command=None,
        verify_command_timeout_secs=timeout_secs,
        verify_cold_command_timeout_secs=timeout_secs,
    )


@pytest.mark.usefixtures('code_default_config')
class TestReapTargetsOnlyWhatTheCommandOccupies:

    @pytest.mark.asyncio
    async def test_unscoped_timeout_kills_no_scope(self, fake_systemctl, worktree):
        config = OrchestratorConfig(verify_use_cgroup_scope=False)
        result = await run_verification(
            worktree, config, module_config=_timing_out_module('sleep 30 & wait', 1),
            max_retries=0, role='task',
        )
        assert result.timed_out, result.summary
        assert _killed_units(fake_systemctl) == set()

    @pytest.mark.asyncio
    async def test_scoped_timeout_without_relocation_kills_only_its_launch_scope(
        self, fake_systemd_run, fake_systemctl, worktree,
    ):
        config = OrchestratorConfig(verify_use_cgroup_scope=True)
        result = await run_verification(
            worktree, config, module_config=_timing_out_module('sleep 30 & wait', 1),
            max_retries=0, role='task',
        )
        assert result.timed_out, result.summary
        assert _killed_units(fake_systemctl) == {_launched_unit(fake_systemd_run)}


def _fresh_slice_name() -> str:
    """Dash-free: systemd derives a slice's parent slices from the dashes in its name."""
    return f'dftestreloc{uuid.uuid4().hex[:10]}.slice'


def _stop_unit(unit: str) -> None:
    with contextlib.suppress(Exception):
        subprocess.run(
            ['systemctl', '--user', 'stop', unit], capture_output=True, timeout=15,
        )


@functools.cache
def _nested_scope_relocates() -> bool:
    """Whether a nested ``systemd-run --scope --slice`` moves its payload out of the outer scope.

    Probed independently of verify.py, so the skip depends only on the host.
    A user manager too slow to answer raises ``subprocess.TimeoutExpired``,
    which the cache does not keep.
    """
    probe_slice = _fresh_slice_name()
    try:
        probe = subprocess.run(
            [
                'systemd-run', '--user', '--scope', '--quiet', '--collect', '--',
                '/bin/sh', '-c',
                'exec systemd-run --user --scope --quiet'
                f' --slice={probe_slice} -- cat /proc/self/cgroup',
            ],
            capture_output=True, text=True, timeout=15,
        )
    except OSError:
        return False
    finally:
        _stop_unit(probe_slice)
    cgroup = probe.stdout.strip()
    return f'/{probe_slice}/' in cgroup and cgroup.endswith('.scope')


def _argv_of(pid: str) -> list[str]:
    try:
        return os.fsdecode((_PROC / pid / 'cmdline').read_bytes()).split('\0')
    except OSError:
        return []


def _live_pids_carrying(marker: str) -> list[int]:
    pids = []
    for entry in _PROC.iterdir():
        if not entry.name.isdigit() or marker not in _argv_of(entry.name):
            continue
        fields = read_stat_fields(entry, skip_malformed=True)
        if fields is not None and not fields.terminated:
            pids.append(int(entry.name))
    return pids


def _describe(pid: int) -> str:
    try:
        cgroup = (_PROC / str(pid) / 'cgroup').read_text().strip()
    except OSError:
        cgroup = '<gone>'
    return f'pid {pid} argv={_argv_of(str(pid))!r} cgroup={cgroup!r}'


def _survivors(marker: str, *, within_secs: float = 5.0) -> list[int]:
    """The live processes still carrying *marker* once *within_secs* has passed."""
    deadline = time.monotonic() + within_secs
    survivors = _live_pids_carrying(marker)
    while survivors and time.monotonic() < deadline:
        time.sleep(0.1)
        survivors = _live_pids_carrying(marker)
    return survivors


@pytest.fixture
def marker() -> Iterator[str]:
    """A ``sleep`` duration no other process on the host carries in its argv.

    Teardown kills whatever a failing run left carrying it.
    """
    marker = f'900.{random.randrange(10**6):06d}'
    yield marker
    for pid in _live_pids_carrying(marker):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@pytest.fixture
def governed_slice(marker: str) -> Iterator[str]:
    """A fresh slice for the governed workload, stopped at teardown."""
    slice_name = _fresh_slice_name()
    yield slice_name
    _stop_unit(slice_name)


def _governed_exec(path: Path, slice_name: str) -> Path:
    """Shaped like reify's scripts/cpu-governed-exec.sh on its governed path."""
    path.write_text(
        '#!/bin/sh\n'
        '[ "$1" = --role ] && shift 2\n'
        '[ "$1" = -- ] && shift\n'
        f'exec systemd-run --user --scope --quiet --slice={slice_name} -- "$@"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.mark.usefixtures('code_default_config')
class TestRealGovernedMergeLeg:

    @pytest.mark.asyncio
    async def test_timed_out_governed_merge_leg_leaves_no_survivors(
        self, worktree, governed_slice, marker, tmp_path,
    ):
        try:
            relocates = _nested_scope_relocates()
        except subprocess.TimeoutExpired:
            pytest.skip('the systemd user manager did not answer the nested-scope probe in 15s')
        if not relocates:
            pytest.skip('nested systemd user scopes do not relocate on this host')
        governed = _governed_exec(tmp_path / 'gov.sh', governed_slice)
        config = OrchestratorConfig(
            verify_use_cgroup_scope=True,
            cpu_governance=CpuGovernConfig(enabled=True, exec_path=str(governed)),
        )
        # `& wait` keeps bash as the session leader, so GNU timeout's setpgid
        # really leaves the spawn process group.
        result = await run_verification(
            worktree, config,
            module_config=_timing_out_module(f'timeout 900 sleep {marker} & wait', 3),
            max_retries=0, role='merge',
        )
        assert result.timed_out, result.summary

        survivors = _survivors(marker)
        assert survivors == [], '\n'.join(_describe(pid) for pid in survivors)


async def _failing_discovery(*_args: object, **_kwargs: object) -> frozenset[str]:
    raise RuntimeError('cannot schedule new futures after shutdown')


async def _cancelled_discovery(*_args: object, **_kwargs: object) -> frozenset[str]:
    raise asyncio.CancelledError


@pytest.mark.usefixtures('code_default_config')
class TestDiscoveryNeverCostsTheKills:

    @pytest.mark.asyncio
    async def test_a_failed_discovery_still_kills_the_launch_scope_and_group(
        self, fake_systemd_run, fake_systemctl, worktree, marker, monkeypatch, caplog,
    ):
        monkeypatch.setattr('orchestrator.verify.relocated_scopes', _failing_discovery)
        result = await run_verification(
            worktree, OrchestratorConfig(verify_use_cgroup_scope=True),
            module_config=_timing_out_module(f'sleep {marker} & wait', 1),
            max_retries=0, role='task',
        )
        assert result.timed_out, result.summary
        assert _killed_units(fake_systemctl) == {_launched_unit(fake_systemd_run)}
        assert _survivors(marker) == []
        assert 'finding the scopes it moved into failed' in caplog.text

    @pytest.mark.asyncio
    async def test_a_cancelled_discovery_kills_the_launch_scope_and_group_then_propagates(
        self, fake_systemd_run, fake_systemctl, worktree, marker, monkeypatch,
    ):
        monkeypatch.setattr('orchestrator.verify.relocated_scopes', _cancelled_discovery)
        with pytest.raises(asyncio.CancelledError):
            await run_verification(
                worktree, OrchestratorConfig(verify_use_cgroup_scope=True),
                module_config=_timing_out_module(f'sleep {marker} & wait', 1),
                max_retries=0, role='task',
            )
        assert _killed_units(fake_systemctl) == {_launched_unit(fake_systemd_run)}
        assert _survivors(marker) == []
