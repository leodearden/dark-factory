"""Behaviour of ``orchestrator.tree_removal.remove_tree`` against real trees."""
from __future__ import annotations

import asyncio
import logging
import os
import stat
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


def _build_wide_tree(root: Path, file_count: int) -> None:
    per_dir = 100
    for index in range(file_count):
        bucket = root / f'd{index // per_dir:03d}'
        bucket.mkdir(parents=True, exist_ok=True)
        (bucket / f'f{index:05d}').write_bytes(b'x')


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


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_a_tree_is_removed(tmp_path: Path) -> None:
    root = tmp_path / 'wide'
    _build_wide_tree(root, file_count=2000)
    sightings: list[bool] = []
    removal = asyncio.ensure_future(remove_tree(root))

    async def ticker() -> None:
        while not removal.done():
            sightings.append(root.exists())
            await asyncio.sleep(0)

    await asyncio.gather(removal, ticker())

    assert not root.exists()
    assert True in sightings, (
        'no loop tick observed the tree before remove_tree finished: the '
        'deletion ran inline on the event loop thread'
    )


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


@pytest.mark.asyncio
async def test_a_removal_that_cannot_spawn_with_ignore_errors_warns_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    root = tmp_path / 'tree'
    _build_nested_tree(root)
    empty_bin = tmp_path / 'empty-bin'
    empty_bin.mkdir()
    monkeypatch.setenv('PATH', str(empty_bin))
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)

    await remove_tree(root, ignore_errors=True)

    warnings = [
        r for r in caplog.records
        if r.name == _LOGGER_NAME and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert str(root) in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_a_removal_that_cannot_spawn_raises_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'tree'
    _build_nested_tree(root)
    empty_bin = tmp_path / 'empty-bin'
    empty_bin.mkdir()
    monkeypatch.setenv('PATH', str(empty_bin))

    with pytest.raises(OSError):
        await remove_tree(root)
