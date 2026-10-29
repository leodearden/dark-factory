"""GitOps.resolve_rebased_citation: which commit on main a SHA cited by an
archived record was replayed onto, read-only and never guessed
(plans/plan-deviation-recording-prd.md D12).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from _git_fixtures import seed_repo

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run


async def _git(repo: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=repo)
    assert rc == 0, f'git {" ".join(args)} failed: {err}'
    return out.strip()


async def _commit(repo: Path, relpath: str, text: str, subject: str) -> str:
    (repo / relpath).write_text(text)
    await _git(repo, 'add', relpath)
    await _git(repo, 'commit', '-m', subject)
    return await _git(repo, 'rev-parse', 'HEAD')


@dataclass(frozen=True)
class LandedBranch:
    s1: str
    s2: str
    s1_landed: str
    s2_landed: str


async def _land_rebased_branch(repo: Path) -> LandedBranch:
    await _git(repo, 'checkout', '-b', 'task/7')
    s1 = await _commit(repo, 'feature.py', 'feature = 1\n', 'feat(7): add feature')
    s2 = await _commit(repo, 'second.py', 'second = 2\n', 'feat(7): second')
    await _git(repo, 'checkout', 'main')
    await _commit(repo, 'other.py', 'other = 3\n', 'main moves on')
    await _git(repo, 'checkout', 'task/7')
    await _git(repo, 'rebase', 'main')
    await _git(repo, 'checkout', 'main')
    await _git(repo, 'merge', '--ff-only', 'task/7')
    return LandedBranch(
        s1=s1,
        s2=s2,
        s1_landed=await _git(repo, 'rev-parse', 'HEAD~1'),
        s2_landed=await _git(repo, 'rev-parse', 'HEAD'),
    )


async def _side_commit(repo: Path, branch: str, relpath: str, text: str, subject: str) -> str:
    await _git(repo, 'checkout', '-b', branch)
    sha = await _commit(repo, relpath, text, subject)
    await _git(repo, 'checkout', 'main')
    return sha


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    config = GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )
    return GitOps(config, repo)


@pytest.mark.asyncio
class TestResolveRebasedCitation:
    async def test_rebased_commit_resolves_to_its_landed_twin(self, repo, git_ops):
        landed = await _land_rebased_branch(repo)

        assert await git_ops.resolve_rebased_citation(landed.s1) == {
            'landed': landed.s1_landed,
            'subject': 'feat(7): add feature',
            'method': 'patch-id',
        }

    async def test_abbreviated_citation_resolves(self, repo, git_ops):
        landed = await _land_rebased_branch(repo)

        result = await git_ops.resolve_rebased_citation(landed.s1[:10])

        assert result is not None
        assert result['landed'] == landed.s1_landed

    async def test_commit_already_on_main_needs_no_resolution(self, repo, git_ops):
        landed = await _land_rebased_branch(repo)

        assert await git_ops.resolve_rebased_citation(landed.s1_landed) is None

    async def test_unknown_and_non_commit_tokens_return_none(self, repo, git_ops):
        tree = await _git(repo, 'rev-parse', 'HEAD^{tree}')

        assert await git_ops.resolve_rebased_citation('deadbeef' * 5) is None
        assert await git_ops.resolve_rebased_citation(tree) is None

    async def test_side_commit_without_a_twin_returns_none(self, repo, git_ops):
        lonely = await _side_commit(repo, 'side', 'lonely.py', 'lonely = 1\n', 'side: lonely')
        await _commit(repo, 'other.py', 'other = 3\n', 'main moves on')

        assert await git_ops.resolve_rebased_citation(lonely) is None

    async def test_unique_subject_resolves_when_patch_id_misses(self, repo, git_ops):
        side = await _side_commit(
            repo, 'task/9', 'feature2.py', 'version = "side"\n', 'feat(9): only subject',
        )
        on_main = await _commit(
            repo, 'feature2.py', 'version = "rewritten on landing"\n', 'feat(9): only subject',
        )

        assert await git_ops.resolve_rebased_citation(side) == {
            'landed': on_main,
            'subject': 'feat(9): only subject',
            'method': 'subject',
        }

    async def test_ambiguous_subject_is_never_guessed(self, repo, git_ops):
        side = await _side_commit(repo, 'side', 'g.py', 'g = "side"\n', 'chore: shared subject')
        await _commit(repo, 'g.py', 'g = "first"\n', 'chore: shared subject')
        await _commit(repo, 'h.py', 'h = "second"\n', 'chore: shared subject')

        assert await git_ops.resolve_rebased_citation(side) is None

    async def test_later_commit_on_the_same_path_does_not_disturb_the_match(self, repo, git_ops):
        landed = await _land_rebased_branch(repo)
        await _commit(repo, 'feature.py', 'feature = 99\n', 'refactor: later edit')

        result = await git_ops.resolve_rebased_citation(landed.s1)

        assert result is not None
        assert result['landed'] == landed.s1_landed
        assert result['method'] == 'patch-id'
