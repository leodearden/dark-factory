"""``_run_cmd`` closes a scoped leg's launch-handshake pipe on every exit arm.

Each scoped leg opens a pipe (``scope_launch_handshake.py``) inside the
long-lived orchestrator process, so one left open per leg would be a slow
leak. Real subprocesses throughout, under the exec'ing fake ``systemd-run``
and a no-op fake ``systemctl`` for the timeout and cancel teardown.

Fixtures are MODULE-LOCAL: a conftest.py edit trips verify.py's has_conftest
and widens the merge-time verify to the full owning-package suite.
"""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from _fake_systemd_run import fake_systemd_run as fake_systemd_run
from _fake_systemd_run import recorded_scope_spawns

from orchestrator.verify import _run_cmd


@pytest.fixture
def no_op_systemctl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = tmp_path / 'systemctl-bin'
    bindir.mkdir()
    script = bindir / 'systemctl'
    script.write_text('#!/bin/sh\nexit 0\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}' + os.environ['PATH'])


def _open_fds() -> list[str]:
    return sorted(os.listdir('/proc/self/fd'))


async def _scoped_leg(cmd: str, cwd: Path, timeout: float = 30.0) -> tuple[int, str, bool]:
    return await _run_cmd(cmd, cwd, timeout=timeout, use_cgroup_scope=True, scope_tag='t7339')


async def _exits_red(cwd: Path, argv_dir: Path) -> None:
    rc, _output, timed_out = await _scoped_leg('exit 3', cwd)
    assert (rc, timed_out) == (3, False)
    assert recorded_scope_spawns(argv_dir), 'the scope launch path was not taken'


async def _times_out(cwd: Path, argv_dir: Path) -> None:
    _rc, _output, timed_out = await _scoped_leg('sleep 30', cwd, timeout=0.5)
    assert timed_out is True
    assert recorded_scope_spawns(argv_dir), 'the scope launch path was not taken'


async def _is_cancelled(cwd: Path, argv_dir: Path) -> None:
    leg = asyncio.create_task(_scoped_leg('touch started; sleep 30', cwd))
    while not (cwd / 'started').exists() and not leg.done():
        await asyncio.sleep(0.02)
    leg.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leg
    assert recorded_scope_spawns(argv_dir), 'the scope launch path was not taken'


async def _spawn_fails(cwd: Path, argv_dir: Path) -> None:
    rc, output, _timed_out = await _scoped_leg('true', cwd / 'missing')
    assert rc == 1
    assert output.startswith('Command failed: '), output


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.usefixtures('no_op_systemctl')
@pytest.mark.parametrize(
    'arm',
    [_exits_red, _times_out, _is_cancelled, _spawn_fails],
    ids=['exits-red', 'times-out', 'is-cancelled', 'spawn-fails'],
)
async def test_scoped_leg_leaves_no_fd_open(
    arm: Callable[[Path, Path], Awaitable[None]], tmp_path: Path, fake_systemd_run: Path,
):
    before = _open_fds()
    await arm(tmp_path, fake_systemd_run)
    assert _open_fds() == before
