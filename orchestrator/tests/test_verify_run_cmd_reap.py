"""``_run_cmd`` reaps the subtree it spawned on every abnormal exit (task 6595).

Real subprocesses throughout; no private name is patched. ``/dev/full`` as the
log path makes the first per-chunk flush raise a genuine ENOSPC. A fake
``systemd-run`` and a fake ``systemctl`` first on PATH make the cgroup-scope
teardown observable without a systemd user manager.

Fixtures are MODULE-LOCAL: a conftest.py edit trips verify.py's has_conftest
and widens the merge-time verify to the full owning-package suite.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import stat
from pathlib import Path
from typing import NamedTuple

import pytest

from orchestrator.verify import _run_cmd

_DEV_FULL = Path('/dev/full')
_LEAKER = (
    'sleep 60 >/dev/null 2>&1 & echo $! > child.pid; '
    'echo $$ > leader.pid; echo go; wait'
)


def _running(pid: int) -> bool:
    try:
        proc_stat = Path(f'/proc/{pid}/stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    state = proc_stat.rsplit(')', 1)[1].split()[0]
    return state not in ('Z', 'X')


async def _wait_all_dead(pids: list[int], timeout: float = 10.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while any(_running(pid) for pid in pids):
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.05)
    return True


def _leaker_pids(cwd: Path) -> tuple[int, int]:
    leader = int((cwd / 'leader.pid').read_text())
    child = int((cwd / 'child.pid').read_text())
    return leader, child


def _kill_leftovers(leader: int, child: int) -> None:
    if _running(leader) or _running(child):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(leader, signal.SIGKILL)


def _assert_command_failed(rc: int, out: str, timed_out: bool) -> None:
    assert rc == 1
    assert timed_out is False
    assert out.startswith('Command failed: '), out


class _ScopeRecords(NamedTuple):
    unit: Path
    systemctl: Path


def _write_executable(path: Path, body: str) -> None:
    path.write_text('#!/bin/sh\n' + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_scope_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _ScopeRecords:
    """Put recording ``systemd-run`` and ``systemctl`` first on PATH."""
    records = _ScopeRecords(
        unit=tmp_path / 'systemd-run.unit',
        systemctl=tmp_path / 'systemctl.calls',
    )
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    _write_executable(
        fakebin / 'systemd-run',
        'for a in "$@"; do\n'
        f'  case "$a" in --unit=*) printf \'%s\\n\' "${{a#--unit=}}" > {shlex.quote(str(records.unit))};; esac\n'
        'done\n'
        'while [ $# -gt 0 ]; do\n'
        '  case "$1" in -p) shift 2;; -*) shift;; *) break;; esac\n'
        'done\n'
        'exec "$@"\n',
    )
    _write_executable(
        fakebin / 'systemctl',
        f'printf \'%s\\n\' "$*" >> {shlex.quote(str(records.systemctl))}\n'
        'exit 0\n',
    )
    monkeypatch.setenv('PATH', f'{fakebin}{os.pathsep}' + os.environ['PATH'])
    return records


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_log_flush_failure_reaps_process_group(tmp_path: Path):
    rc, out, timed_out = await _run_cmd(
        _LEAKER, tmp_path, timeout=30.0, log_path=_DEV_FULL,
    )
    leader, child = _leaker_pids(tmp_path)
    try:
        _assert_command_failed(rc, out, timed_out)
        assert await _wait_all_dead([leader, child]), (
            f'leader {leader} / child {child} outlived _run_cmd'
        )
    finally:
        _kill_leftovers(leader, child)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_log_flush_failure_under_cgroup_scope_kills_scope_then_group(
    tmp_path: Path, fake_scope_tools: _ScopeRecords,
):
    rc, out, timed_out = await _run_cmd(
        _LEAKER, tmp_path, timeout=30.0, log_path=_DEV_FULL,
        use_cgroup_scope=True, scope_tag='t6595',
    )
    leader, child = _leaker_pids(tmp_path)
    try:
        _assert_command_failed(rc, out, timed_out)
        unit = fake_scope_tools.unit.read_text().strip()
        assert unit.startswith('df-verify-t6595-'), unit
        assert unit.endswith('.scope'), unit
        assert fake_scope_tools.systemctl.exists(), 'no systemctl call recorded'
        assert fake_scope_tools.systemctl.read_text().splitlines() == [
            f'--user kill --signal=SIGKILL {unit}',
            f'--user stop {unit}',
        ]
        assert await _wait_all_dead([leader, child]), (
            f'leader {leader} / child {child} outlived _run_cmd'
        )
    finally:
        _kill_leftovers(leader, child)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_clean_exit_under_cgroup_scope_runs_no_teardown(
    tmp_path: Path, fake_scope_tools: _ScopeRecords,
):
    rc, _, _ = await _run_cmd(
        'echo ok', tmp_path, timeout=30.0, log_path=tmp_path / 'ok.log',
        use_cgroup_scope=True,
    )
    assert rc == 0
    assert fake_scope_tools.unit.exists(), 'the scope launch path was not taken'
    assert not fake_scope_tools.systemctl.exists()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_spawn_failure_still_returns_command_failed(tmp_path: Path):
    rc, out, timed_out = await _run_cmd('true', tmp_path / 'absent', timeout=5.0)
    _assert_command_failed(rc, out, timed_out)
