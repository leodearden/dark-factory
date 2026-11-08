"""Boundary test B1, orchestrator half: every merge-subject reader recognises what git_ops writes.

The fixture lands task ``50`` through the real writer in a repo whose naming is
``feat/`` into ``trunk``, so a reader that still assumes ``task/`` into ``main``
cannot see it. A ``Revert "<subject for feat/51>"`` commit on trunk is the
negative every reader must refuse. The fused-memory half lives in
``fused-memory/tests/test_landed_on_main.py``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from _git_fixtures import seed_repo
from _orch_helpers import git_env_with_ceiling
from shared.merge_subject import format_merge_subject

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ['git', *args], cwd=repo, env=git_env_with_ceiling(repo),
        capture_output=True, text=True, check=True,
    )
    return completed.stdout.strip()


def _subject_of(repo: Path, sha: str) -> str:
    return _git(repo, 'show', '-s', '--format=%s', sha)


@dataclass(frozen=True)
class FeatTrunk:
    git_ops: GitOps
    repo: Path
    merge_sha: str
    revert_sha: str


async def _land_through_the_writer(git_ops: GitOps, tid: str) -> str:
    worktree = await git_ops.create_worktree(tid)
    assert worktree is not None
    (worktree.path / f'b1_{tid}.txt').write_text(f'{tid}\n')
    await git_ops.commit(worktree.path, 'work for the B1 round trip')
    merged = await git_ops.merge_to_main(worktree.path, tid)
    assert merged.success and merged.merge_commit is not None
    assert merged.merge_worktree is not None
    advanced = await git_ops.advance_main(merged.merge_commit)
    assert advanced.result == 'advanced'
    await git_ops.cleanup_merge_worktree(merged.merge_worktree)
    await git_ops.cleanup_worktree(worktree.path, tid)
    return merged.merge_commit


@pytest_asyncio.fixture
async def feat_trunk(tmp_path: Path) -> FeatTrunk:
    repo = seed_repo(tmp_path / 'repo')
    _git(repo, 'branch', '-m', 'main', 'trunk')
    git_ops = GitOps(
        GitConfig(
            main_branch='trunk', branch_prefix='feat/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False,
        ),
        repo,
    )
    merge_sha = await _land_through_the_writer(git_ops, '50')
    _git(
        repo, 'commit', '--allow-empty', '--no-verify',
        '-m', 'Revert "' + format_merge_subject('feat/51', 'trunk') + '"',
    )
    return FeatTrunk(git_ops, repo, merge_sha, _git(repo, 'rev-parse', 'HEAD'))


@pytest.mark.asyncio
class TestWriterAndMergeMarker:
    async def test_the_writer_lands_the_shared_subject(self, feat_trunk: FeatTrunk) -> None:
        assert _subject_of(feat_trunk.repo, feat_trunk.merge_sha) == format_merge_subject(
            'feat/50', 'trunk',
        )

    async def test_the_marker_is_the_landing(self, feat_trunk: FeatTrunk) -> None:
        marker = await feat_trunk.git_ops.find_merge_marker('feat/50', gate_on_existing_ref=False)
        assert marker == feat_trunk.merge_sha

    async def test_a_revert_subject_is_not_a_marker(self, feat_trunk: FeatTrunk) -> None:
        git_ops = feat_trunk.git_ops
        assert await git_ops.find_merge_marker('feat/51', gate_on_existing_ref=False) is None
        assert await git_ops.find_merge_marker_checked('feat/51') is None
