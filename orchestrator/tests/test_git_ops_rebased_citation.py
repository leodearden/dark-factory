"""GitOps.resolve_rebased_citation: which commit on main a SHA cited by an
archived record was replayed onto, read-only and never guessed
(plans/plan-deviation-recording-prd.md D12).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from _git_fault_shim import answer_git, fail_git
from _git_fixtures import seed_repo

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, GitQueryFailed, _run


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


async def _land_by_subject_only(repo: Path) -> tuple[str, str]:
    """A side commit, and the same-subject commit on main whose diff differs."""
    side = await _side_commit(
        repo, 'task/9', 'feature2.py', 'version = "side"\n', 'feat(9): only subject',
    )
    on_main = await _commit(
        repo, 'feature2.py', 'version = "rewritten on landing"\n', 'feat(9): only subject',
    )
    return side, on_main


def _git_config(main_branch: str = 'main') -> GitConfig:
    return GitConfig(
        main_branch=main_branch, branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    return GitOps(_git_config(), repo)


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
        side, on_main = await _land_by_subject_only(repo)

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


@pytest.mark.asyncio
class TestCommitsNamed:
    async def test_maps_each_commit_token_to_its_full_sha_and_drops_the_rest(
        self, repo, git_ops,
    ):
        head = await _git(repo, 'rev-parse', 'HEAD')
        tree = await _git(repo, 'rev-parse', 'HEAD^{tree}')
        tokens = [
            head, head[:10], tree, '20261023', '3f2a9c1e', '5c8d0b7f1a33', '9c1e7b4d5a609e21',
        ]

        assert await git_ops.commits_named(tokens) == {head: head, head[:10]: head}

    async def test_a_directory_that_is_not_a_repository_raises(self, tmp_path):
        plain = tmp_path / 'plain'
        plain.mkdir()

        with pytest.raises(GitQueryFailed):
            await GitOps(_git_config(), plain).commits_named(['abc1234'])

    async def test_an_ambiguous_token_is_omitted(
        self, git_ops, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ):
        answer_git(monkeypatch, tmp_path / 'bin', 'cat-file*', 'abc1234^{commit} ambiguous\n')

        assert await git_ops.commits_named(['abc1234']) == {}

    @pytest.mark.parametrize(
        ('answer', 'raised'),
        [
            pytest.param('garbage\n', "answered 'garbage' for 'abc1234'", id='unparseable-line'),
            pytest.param('', 'answered 0 lines for 1 tokens', id='too-few-lines'),
            pytest.param(
                'abc1234^{commit} missing\nabc1234^{commit} missing\n',
                'answered 2 lines for 1 tokens',
                id='too-many-lines',
            ),
        ],
    )
    async def test_an_answer_that_is_neither_an_oid_nor_a_miss_raises(
        self, git_ops, tmp_path, monkeypatch: pytest.MonkeyPatch, answer: str, raised: str,
    ):
        answer_git(monkeypatch, tmp_path / 'bin', 'cat-file*', answer)

        with pytest.raises(GitQueryFailed, match=re.escape(raised)):
            await git_ops.commits_named(['abc1234'])


async def _cite_rebased_s1(repo: Path) -> str:
    return (await _land_rebased_branch(repo)).s1


async def _cite_subject_only_side(repo: Path) -> str:
    side, _ = await _land_by_subject_only(repo)
    return side


@pytest.mark.asyncio
class TestFaultsAreNeverReadAsFacts:
    async def test_a_root_that_is_not_a_repository_raises(self, tmp_path):
        plain = tmp_path / 'plain'
        plain.mkdir()

        with pytest.raises(GitQueryFailed):
            await GitOps(_git_config(), plain).resolve_rebased_citation('a7ceed6597')

    async def test_an_unknown_main_branch_raises(self, repo):
        landed = await _land_rebased_branch(repo)
        git_ops = GitOps(_git_config(main_branch='no-such-branch'), repo)

        with pytest.raises(GitQueryFailed):
            await git_ops.resolve_rebased_citation(landed.s1)

    @pytest.mark.parametrize(
        ('argv_glob', 'after', 'cite'),
        [
            pytest.param('diff-tree*', 0, _cite_rebased_s1, id='diff-tree'),
            pytest.param('log -p -1 *', 0, _cite_rebased_s1, id='target-log-p'),
            pytest.param('patch-id*', 0, _cite_rebased_s1, id='target-patch-id'),
            pytest.param('log -p --no-color --full-diff *', 0, _cite_rebased_s1, id='range-log-p'),
            pytest.param('patch-id*', 1, _cite_rebased_s1, id='range-patch-id'),
            pytest.param('log -1 --format=%s *', 0, _cite_rebased_s1, id='landed-subject'),
            pytest.param('log --format=%H*', 0, _cite_subject_only_side, id='range-subjects'),
        ],
    )
    async def test_a_failing_git_query_mid_lookup_raises(
        self,
        repo: Path,
        git_ops: GitOps,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        argv_glob: str,
        after: int,
        cite: Callable[[Path], Awaitable[str]],
    ):
        """Only the one call each param names fails, so a fault the code under
        test swallowed lets the lookup go on to answer and the test fails."""
        cited = await cite(repo)
        fail_git(monkeypatch, tmp_path / 'bin', argv_glob, after=after, times=1)
        failing_query = re.escape(f'git {argv_glob.rstrip("*")}')

        with pytest.raises(GitQueryFailed, match=f'^{failing_query}.* exited 128: fatal: injected'):
            await git_ops.resolve_rebased_citation(cited)
