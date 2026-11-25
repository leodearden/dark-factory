"""The merge screen's warm lanes and cold fallback, against a real git repo through public GitOps seams.

The lanes live outside the ``_merge-*`` band, so the merge reapers never see
them, and they retain ``.venv/`` and ``__pycache__/`` across resets.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, _git, seed_repo
from _orch_helpers import git_env_with_ceiling

from orchestrator.config import GitConfig
from orchestrator.git_ops import (
    PROTECT_GLOB_OWNED_POOL_BANDS,
    PROTECTED_PREFIXES,
    GitOps,
    ScreenLane,
    ScreenLaneUnavailable,
    WorktreeKind,
    classify_worktree_entry,
    render_protect_glob,
)
from orchestrator.verify_cancel import lane_lock_path

_SEED = RepoSeed(files=(('pkg/mod.py', 'VALUE = 1\n'),), message='Initial')


def _head(cwd: Path) -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=cwd, env=git_env_with_ceiling(cwd),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(repo: Path, text: str) -> str:
    (repo / 'pkg' / 'mod.py').write_text(text, encoding='utf-8')
    _git(repo, 'commit', '-am', f'set {text.strip()}')
    return _head(repo)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', _SEED)


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    config = GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )
    return GitOps(config, repo)


@pytest.fixture
def shas(repo: Path) -> tuple[str, str]:
    return _commit(repo, 'VALUE = 2\n'), _commit(repo, 'VALUE = 3\n')


def _dirty(lane: Path) -> None:
    (lane / '.venv').mkdir()
    (lane / '.venv' / 'marker').write_text('venv\n', encoding='utf-8')
    (lane / 'pkg' / '__pycache__').mkdir()
    (lane / 'pkg' / '__pycache__' / 'm.cpython.pyc').write_bytes(b'\x00pyc')
    (lane / 'junk.txt').write_text('junk\n', encoding='utf-8')
    (lane / 'pkg' / 'mod.py').write_text('VALUE = "edited"\n', encoding='utf-8')


@pytest.fixture
def held_lock(git_ops: GitOps) -> Iterator[Path]:
    lock = lane_lock_path(git_ops.worktree_base / ScreenLane.MERGER.value)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield lock
    finally:
        os.close(fd)


@pytest.mark.asyncio
async def test_a_lane_is_created_then_reset_in_place_keeping_venv_and_bytecode(git_ops, shas):
    sha1, sha2 = shas

    async with git_ops.screen_lane(ScreenLane.MERGER, sha1) as lane:
        assert lane == git_ops.worktree_base / '_screen-merger'
        assert _head(lane) == sha1
        _dirty(lane)

    async with git_ops.screen_lane(ScreenLane.MERGER, sha2) as again:
        assert again == lane
        assert _head(again) == sha2
        assert (again / 'pkg' / 'mod.py').read_text(encoding='utf-8') == 'VALUE = 3\n'
        assert not (again / 'junk.txt').exists()
        assert (again / '.venv' / 'marker').read_text(encoding='utf-8') == 'venv\n'
        assert (again / 'pkg' / '__pycache__' / 'm.cpython.pyc').exists()


@pytest.mark.asyncio
async def test_a_contended_lane_flock_is_unavailable_and_leaves_the_tree_untouched(git_ops, shas, request):
    sha1, sha2 = shas
    async with git_ops.screen_lane(ScreenLane.MERGER, sha1) as lane:
        (lane / 'junk.txt').write_text('junk\n', encoding='utf-8')
    lock = request.getfixturevalue('held_lock')

    with pytest.raises(ScreenLaneUnavailable) as raised:
        async with git_ops.screen_lane(ScreenLane.MERGER, sha2, lock_wait_secs=0.2):
            pytest.fail('the body must not run without the lane flock')

    message = str(raised.value)
    assert ScreenLane.MERGER.value in message
    assert str(lock) in message
    assert '0.2' in message
    assert raised.value.lane is ScreenLane.MERGER
    assert raised.value.lock_path == lock
    assert raised.value.wait_secs == 0.2
    assert _head(lane) == sha1
    assert (lane / 'junk.txt').exists()


@pytest.mark.asyncio
async def test_the_lanes_survive_the_merge_reaper(git_ops, shas):
    sha1, _ = shas
    lanes = []
    for lane_kind in ScreenLane:
        async with git_ops.screen_lane(lane_kind, sha1) as lane:
            (lane / '.venv').mkdir()
            (lane / '.venv' / 'marker').write_text('venv\n', encoding='utf-8')
            lanes.append(lane)

    removed = await git_ops.prune_stale_merge_worktrees()

    assert removed == []
    assert all((lane / '.venv' / 'marker').exists() for lane in lanes)


@pytest.mark.parametrize('name', ['_screen-merger', '_screen-chain', '_mergescreen-abc'])
def test_screen_worktree_names_classify_as_infra(name):
    assert classify_worktree_entry(name) == 'infra'


@pytest.mark.parametrize('owned', [(), PROTECT_GLOB_OWNED_POOL_BANDS], ids=['all', 'pool-sweep'])
def test_the_protect_glob_protects_the_screen_bands(owned):
    bands = render_protect_glob(owned=owned).split(',')
    assert {'_screen-merger', '_screen-chain', '_mergescreen-*'} <= set(bands)


def test_the_screen_names_are_protected_prefix_keys():
    assert {lane.value for lane in ScreenLane} == {'_screen-merger', '_screen-chain'}
    assert {lane.value for lane in ScreenLane} <= set(PROTECTED_PREFIXES)
    assert WorktreeKind.MERGE_SCREEN.value == '_mergescreen-'
    assert WorktreeKind.MERGE_SCREEN.value in PROTECTED_PREFIXES


@pytest.mark.asyncio
async def test_the_ephemeral_fallback_mints_and_removes_a_mergescreen_worktree(git_ops, shas):
    sha1, _ = shas

    async with git_ops.ephemeral_worktree(WorktreeKind.MERGE_SCREEN, sha1) as path:
        assert path.parent == git_ops.worktree_base
        assert path.name.startswith('_mergescreen-')
        assert _head(path) == sha1

    assert not path.exists()
