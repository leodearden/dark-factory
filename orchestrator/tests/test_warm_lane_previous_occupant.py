"""A warm lane handed to a new task carries none of its previous occupant's ``.task/`` state.

Driven through the public ``GitOps.acquire_warm_lane`` / ``release_warm_lane``
against a real repo whose ``.task/`` is gitignored, as in reify. The planted
``claude-config-<id>`` dirs hold a real nested repo at
``plugins/marketplaces/m``, the shape a single-``-f`` ``git clean`` skips and
leaves behind as a husk (task 7329).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from _seed_script_stubs import (
    CURRENT_LOCKING_SEED_SCRIPT,
    commit_seed_script,
    make_seed_test_repo,
    warm_pool_git_config,
)
from shared.config_dir import CONFIG_DIR_PREFIX, TaskConfigDir
from shared.transcript_archive import durable_archive_path

from orchestrator.config import TranscriptArchiveConfig
from orchestrator.git_ops import GitOps, WorktreeInfo, _run

_ARCHIVE_ROOT = 'data/agent-transcripts'


async def _git(args: list[str], cwd: Path) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {args} failed (rc={rc}): {err}'
    return out.strip()


async def _build_repo(tmp_path: Path) -> Path:
    repo = make_seed_test_repo(tmp_path)
    (repo / '.gitignore').write_text('.task/\n')
    await _git(['add', '.gitignore'], repo)
    await _git(['commit', '-m', 'ignore .task'], repo)
    await commit_seed_script(repo, CURRENT_LOCKING_SEED_SCRIPT)
    return repo


def _pooled_git_ops(repo: Path, *, archive_root: str | None) -> GitOps:
    transcript_archive = (
        None if archive_root is None
        else TranscriptArchiveConfig(enabled=True, root=archive_root)
    )
    return GitOps(
        warm_pool_git_config(), repo, warm_lane_pool_size=1,
        transcript_archive=transcript_archive,
    )


def _plant_transcript(config_dir: Path, session_id: str) -> Path:
    transcript = config_dir / 'projects' / '-enc' / f'{session_id}.jsonl'
    transcript.parent.mkdir(parents=True)
    transcript.write_text(f'{{"session": "{session_id}"}}\n')
    return transcript


async def _plant_occupant_config_dir(task_dir: Path, task_id: str, session_id: str) -> None:
    config_dir = task_dir / f'{CONFIG_DIR_PREFIX}{task_id}'
    _plant_transcript(config_dir, session_id)
    (config_dir / '.credentials.json').write_text('{"token": "secret"}')
    marketplace = config_dir / 'plugins' / 'marketplaces' / 'm'
    marketplace.mkdir(parents=True)
    await _git(['init', '-q', str(marketplace)], marketplace)


async def _recycle_after_occupant_a(
    git_ops: GitOps, repo: Path, caplog: pytest.LogCaptureFixture,
) -> tuple[Path, WorktreeInfo]:
    start_ref = await _git(['rev-parse', 'HEAD'], repo)
    info_a = await git_ops.acquire_warm_lane('A', start_ref)
    assert isinstance(info_a, WorktreeInfo), f'seed acquire failed: {info_a!r}'
    lane = info_a.path
    task_dir = lane / '.task'
    await _plant_occupant_config_dir(task_dir, 'A', 'sA')
    _plant_transcript(task_dir / f'{CONFIG_DIR_PREFIX}A-unblock', 'u')
    (task_dir / 'verify').mkdir()
    (task_dir / 'verify' / 'out.log').write_text('A verify output')
    await git_ops.release_warm_lane(lane, 'A')

    with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
        info_b = await git_ops.acquire_warm_lane('B', start_ref)

    assert isinstance(info_b, WorktreeInfo), f'recycle acquire failed: {info_b!r}'
    assert info_b.path == lane
    assert 'route=recycle edge=' in caplog.text, caplog.text
    return lane, info_b


def _assert_no_entry_for_occupant_a(task_dir: Path) -> None:
    for name in (f'{CONFIG_DIR_PREFIX}A', f'{CONFIG_DIR_PREFIX}A-unblock'):
        assert not os.path.lexists(task_dir / name), f'{name} survived the recycle'


@pytest.mark.asyncio
async def test_recycle_acquire_evicts_released_occupants_config_dirs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    repo = await _build_repo(tmp_path)
    git_ops = _pooled_git_ops(repo, archive_root=_ARCHIVE_ROOT)

    lane, _ = await _recycle_after_occupant_a(git_ops, repo, caplog)

    task_dir = lane / '.task'
    _assert_no_entry_for_occupant_a(task_dir)
    archive_root = repo / _ARCHIVE_ROOT
    assert durable_archive_path(archive_root, 'A', 'sA') is not None
    assert durable_archive_path(archive_root, 'A-unblock', 'u') is not None
    TaskConfigDir('B', base_dir=task_dir)
    assert sorted(p.name for p in task_dir.glob(f'{CONFIG_DIR_PREFIX}*')) == [
        f'{CONFIG_DIR_PREFIX}B',
    ]


@pytest.mark.asyncio
async def test_recycle_without_archive_still_leaves_no_husk(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    repo = await _build_repo(tmp_path)
    git_ops = _pooled_git_ops(repo, archive_root=None)

    lane, _ = await _recycle_after_occupant_a(git_ops, repo, caplog)

    _assert_no_entry_for_occupant_a(lane / '.task')


@pytest.mark.asyncio
async def test_recycle_holds_a_transcript_the_archive_refused_for_the_next_boot_sweep(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    repo = await _build_repo(tmp_path)
    unwritable_archive = tmp_path / 'archive-is-a-file'
    unwritable_archive.write_text('not a directory')
    git_ops = _pooled_git_ops(repo, archive_root=str(unwritable_archive))

    lane, _ = await _recycle_after_occupant_a(git_ops, repo, caplog)

    held_dir = lane / '.task' / f'{CONFIG_DIR_PREFIX}A'
    assert (held_dir / 'projects' / '-enc' / 'sA.jsonl').is_file()
    assert not os.path.lexists(held_dir / '.credentials.json')
    assert not os.path.lexists(lane / '.task' / 'verify')


async def _orphan_branch_with_a_commit(repo: Path, tmp_path: Path, start_ref: str) -> None:
    scratch_worktree = tmp_path / 'scratch-task-B'
    await _git(['worktree', 'add', '-b', 'task/B', str(scratch_worktree), start_ref], repo)
    (scratch_worktree / 'wip.txt').write_text('B work\n')
    await _git(['add', '-A'], scratch_worktree)
    await _git(['commit', '-m', 'B wip'], scratch_worktree)
    await _git(['worktree', 'remove', '--force', str(scratch_worktree)], repo)


@pytest.mark.asyncio
async def test_reattach_acquire_keeps_incoming_tasks_own_config_and_archives_foreign(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    repo = await _build_repo(tmp_path)
    start_ref = await _git(['rev-parse', 'HEAD'], repo)
    await _orphan_branch_with_a_commit(repo, tmp_path, start_ref)
    git_ops = _pooled_git_ops(repo, archive_root=_ARCHIVE_ROOT)
    info_v = await git_ops.acquire_warm_lane('V', start_ref)
    assert isinstance(info_v, WorktreeInfo), f'seed acquire failed: {info_v!r}'
    lane = info_v.path
    task_dir = lane / '.task'
    await _plant_occupant_config_dir(task_dir, 'V', 'sV')
    own_transcript = _plant_transcript(task_dir / f'{CONFIG_DIR_PREFIX}B', 'sB')
    own_bytes = own_transcript.read_bytes()
    (task_dir / 'iterations.jsonl').write_text('{"task": "V"}\n')
    (task_dir / 'reviews').mkdir()
    (task_dir / 'reviews' / 'merge.json').write_text('{"task": "V"}')
    await git_ops.release_warm_lane(lane, 'V')

    with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
        info_b = await git_ops.acquire_warm_lane('B', start_ref)

    assert isinstance(info_b, WorktreeInfo), f'reattach acquire failed: {info_b!r}'
    assert 'route=reset_in_place_reattach edge=' in caplog.text, caplog.text
    assert own_transcript.read_bytes() == own_bytes
    assert not os.path.lexists(task_dir / f'{CONFIG_DIR_PREFIX}V')
    assert durable_archive_path(repo / _ARCHIVE_ROOT, 'V', 'sV') is not None
    assert not os.path.lexists(task_dir / 'iterations.jsonl')
    assert not os.path.lexists(task_dir / 'reviews')
