"""Behaviour of ``orchestrator.tree_removal.remove_tree`` against real trees."""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import stat
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from orchestrator.tree_removal import TreeRemovalError, remove_tree

_LOGGER_NAME = 'orchestrator.tree_removal'


def _build_nested_tree(root: Path) -> None:
    (root / 'a' / 'b').mkdir(parents=True)
    (root / 'c').mkdir()
    (root / 'top.txt').write_text('top')
    (root / 'a' / 'one.txt').write_text('one')
    (root / 'a' / 'b' / 'two.bin').write_bytes(b'\x00' * 64)
    read_only = root / 'c' / 'read_only.txt'
    read_only.write_text('ro')
    read_only.chmod(stat.S_IRUSR)


def _make_undeletable(root: Path) -> Path:
    """Return a sub-directory the OS refuses to empty (mode 0o500, holds a file)."""
    locked = root / 'locked'
    locked.mkdir(parents=True)
    (locked / 'pinned.txt').write_text('pinned')
    locked.chmod(0o500)
    return locked


_skip_as_root = pytest.mark.skipif(
    os.geteuid() == 0, reason='root ignores directory write permission',
)


@pytest.mark.asyncio
async def test_a_nested_tree_is_removed(tmp_path: Path) -> None:
    root = tmp_path / 'tree'
    _build_nested_tree(root)

    await remove_tree(root)

    assert not os.path.lexists(root)


@pytest.mark.asyncio
async def test_a_symlink_to_an_outside_directory_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / 'outside'
    outside.mkdir()
    keep = outside / 'keep.txt'
    keep.write_text('keep')
    root = tmp_path / 'tree'
    root.mkdir()
    (root / 'link').symlink_to(outside, target_is_directory=True)

    await remove_tree(root)

    assert not os.path.lexists(root)
    assert keep.read_text() == 'keep'


@pytest.mark.asyncio
@pytest.mark.parametrize('ignore_errors', [False, True])
async def test_an_absent_path_is_already_removed(tmp_path: Path, ignore_errors: bool) -> None:
    await remove_tree(tmp_path / 'never-existed', ignore_errors=ignore_errors)


class _GatedRm:
    """An ``rm`` first on PATH that parks before and after the real one until released."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        real_rm = shutil.which('rm')
        assert real_rm is not None
        bin_dir = tmp_path / 'gated-bin'
        bin_dir.mkdir()
        self._started = tmp_path / 'rm-started'
        self._finished = tmp_path / 'rm-finished'
        gate = tmp_path / 'rm-gate'
        os.mkfifo(gate)
        script = bin_dir / 'rm'
        script.write_text(
            '#!/bin/sh\n'
            f'exec 3< {shlex.quote(str(gate))}\n'
            f': > {shlex.quote(str(self._started))}\n'
            'read -r line <&3\n'
            f'{shlex.quote(real_rm)} "$@"\n'
            'rc=$?\n'
            f': > {shlex.quote(str(self._finished))}\n'
            'read -r line <&3\n'
            'exit $rc\n'
        )
        script.chmod(0o755)
        monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
        self._writer = os.open(gate, os.O_RDWR)

    def has_started(self) -> bool:
        return self._started.exists()

    def has_finished(self) -> bool:
        return self._finished.exists()

    def release(self) -> None:
        """Let the parked ``rm`` pass its next gate."""
        os.write(self._writer, b'\n')

    def close(self) -> None:
        """Open every gate for good: a parked read sees end-of-file."""
        os.close(self._writer)


@pytest.fixture
def gated_rm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_GatedRm]:
    gate = _GatedRm(tmp_path, monkeypatch)
    try:
        yield gate
    finally:
        gate.close()


def _assert_still_running(removal: asyncio.Future[None], before: str) -> None:
    assert not removal.done(), (
        f'remove_tree finished before the gated rm on PATH {before} '
        f'(exception={removal.exception()!r})'
    )


async def _spin_until(
    reached: Callable[[], bool], removal: asyncio.Future[None], before: str,
) -> None:
    while not reached():
        _assert_still_running(removal, before)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_a_tree_is_removed(
    tmp_path: Path, gated_rm: _GatedRm,
) -> None:
    root = tmp_path / 'tree'
    _build_nested_tree(root)
    removal = asyncio.ensure_future(remove_tree(root))

    await _spin_until(gated_rm.has_started, removal, 'started')
    assert root.exists()
    gated_rm.release()

    await _spin_until(gated_rm.has_finished, removal, 'finished deleting')
    # The loop ran throughout the deletion; the rm child is alive, parked before exit.
    assert not os.path.lexists(root)
    _assert_still_running(removal, 'exited')
    gated_rm.release()
    await removal


@pytest.mark.asyncio
@_skip_as_root
async def test_a_refused_removal_raises_tree_removal_error(tmp_path: Path) -> None:
    root = tmp_path / 'tree'
    locked = _make_undeletable(root)
    try:
        with pytest.raises(TreeRemovalError) as excinfo:
            await remove_tree(root)
    finally:
        locked.chmod(0o700)

    err = excinfo.value
    assert isinstance(err, OSError)
    assert err.path == root
    assert err.returncode != 0
    assert str(root) in str(err)


@pytest.mark.asyncio
@_skip_as_root
async def test_a_refused_removal_with_ignore_errors_warns_instead(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / 'tree'
    locked = _make_undeletable(root)
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    try:
        await remove_tree(root, ignore_errors=True)
    finally:
        locked.chmod(0o700)

    warnings = [
        r for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert str(root) in warnings[0].getMessage()


def _hide_rm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make spawning ``rm`` fail, as it does when the host is out of processes."""
    empty_bin = tmp_path / 'empty-bin'
    empty_bin.mkdir()
    monkeypatch.setenv('PATH', str(empty_bin))


def _warnings_naming(caplog: pytest.LogCaptureFixture, path: Path) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
        and str(path) in r.getMessage()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('ignore_errors', [False, True])
async def test_a_removal_that_cannot_spawn_rm_still_removes_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ignore_errors: bool,
) -> None:
    root = tmp_path / 'tree'
    _build_nested_tree(root)
    _hide_rm(tmp_path, monkeypatch)
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    await remove_tree(root, ignore_errors=ignore_errors)

    assert not os.path.lexists(root)
    assert len(_warnings_naming(caplog, root)) == 1


@pytest.mark.asyncio
@_skip_as_root
async def test_a_refused_removal_without_rm_raises_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'tree'
    locked = _make_undeletable(root)
    _hide_rm(tmp_path, monkeypatch)
    try:
        with pytest.raises(OSError):
            await remove_tree(root)
    finally:
        locked.chmod(0o700)


@pytest.mark.asyncio
@_skip_as_root
async def test_a_refused_removal_without_rm_with_ignore_errors_warns_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / 'tree'
    locked = _make_undeletable(root)
    _hide_rm(tmp_path, monkeypatch)
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    try:
        await remove_tree(root, ignore_errors=True)
        assert os.path.lexists(locked)
    finally:
        locked.chmod(0o700)

    assert len(_warnings_naming(caplog, root)) == 1
