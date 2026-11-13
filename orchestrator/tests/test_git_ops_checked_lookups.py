"""Checked landing lookups: "could not look" raises, while the fail-safe twins still answer None (task 7332)."""

import logging
from pathlib import Path

import pytest
from _git_fixtures import seed_repo

from orchestrator.config import GitConfig
from orchestrator.git_ops import CitationPatternInvalid, GitOps, GitQueryFailed


def _git_ops(repo: Path, main_branch: str = 'main') -> GitOps:
    return GitOps(GitConfig(push_after_advance=False, main_branch=main_branch), repo)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.mark.asyncio
class TestCheckedCitationLookup:
    async def test_an_unreadable_main_raises_while_the_fail_safe_twin_answers_none(self, repo: Path):
        git_ops = _git_ops(repo, main_branch='no-such-main')

        with pytest.raises(GitQueryFailed):
            await git_ops.find_task_citation_commit_checked('101')
        assert await git_ops.find_task_citation_commit('101') is None

    async def test_an_uncompilable_pattern_raises_while_the_fail_safe_twin_warns_and_answers_none(
        self, repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        git_ops = _git_ops(repo)

        with pytest.raises(CitationPatternInvalid):
            await git_ops.find_task_citation_commit_checked('101', pattern_template='(')
        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            assert await git_ops.find_task_citation_commit('101', pattern_template='(') is None

        assert any(
            'treating as no-citation (fail-safe)' in r.getMessage() for r in caplog.records
        ), [r.getMessage() for r in caplog.records]

    async def test_a_citing_subject_is_found_and_another_task_is_not(self, repo: Path):
        git_ops = _git_ops(repo)
        info = await git_ops.create_worktree('101')
        (info.path / 'a.py').write_text('a = 1\n')
        cited = await git_ops.commit(info.path, 'impl(101): work')
        result = await git_ops.merge_to_main(info.path, '101')
        assert result.success and result.merge_commit
        assert (await git_ops.advance_main(result.merge_commit)).result == 'advanced'

        assert await git_ops.find_task_citation_commit_checked('101') in {cited, result.merge_commit}
        assert await git_ops.find_task_citation_commit_checked('102') is None

    async def test_no_citation_and_a_disabled_pattern_both_answer_none(self, repo: Path):
        git_ops = _git_ops(repo)

        assert await git_ops.find_task_citation_commit_checked('101') is None
        assert await git_ops.find_task_citation_commit_checked('101', pattern_template='') is None


@pytest.mark.asyncio
class TestCheckedMergeMarkerLookup:
    async def test_an_unreadable_main_raises_while_the_fail_safe_twin_answers_none(self, repo: Path):
        git_ops = _git_ops(repo, main_branch='no-such-main')

        with pytest.raises(GitQueryFailed):
            await git_ops.find_merge_marker_checked('task/101')
        assert await git_ops.find_merge_marker('task/101') is None

    async def test_a_landed_branch_answers_its_marker(self, repo: Path):
        git_ops = _git_ops(repo)
        info = await git_ops.create_worktree('101')
        (info.path / 'a.py').write_text('a = 1\n')
        await git_ops.commit(info.path, 'work')
        result = await git_ops.merge_to_main(info.path, '101')
        assert result.success and result.merge_commit
        assert (await git_ops.advance_main(result.merge_commit)).result == 'advanced'

        assert await git_ops.find_merge_marker_checked('task/101') == result.merge_commit
