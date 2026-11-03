"""Both directions of ``DEFAULT_COMMIT_CITATION_PATTERN`` (task 4862, Defect A).

The pattern is widened to accept a trailing qualifier after the task id, as in
``impl(4369 step-8):`` and ``(task 5018 review)``.  Both directions are pinned
because they fail differently: a false negative re-dispatches work that already
landed, which is recoverable; a false positive attributes a foreign commit to a
task and corrupts its provenance, which is not (the task-4705 precedent).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.config import GitConfig
from orchestrator.git_ops import DEFAULT_COMMIT_CITATION_PATTERN, GitOps


def _cites(subject: str, tid: str) -> bool:
    """Apply the template to one SUBJECT the way ``find_task_citation_commit`` does."""
    pattern = re.compile(DEFAULT_COMMIT_CITATION_PATTERN.format(tid=re.escape(tid)))
    return pattern.search(subject) is not None


class TestWidenedFormsCite:
    @pytest.mark.parametrize(
        ('subject', 'tid'),
        [
            ('impl(4369 step-8): implement _pyright_wrapper_tokens', '4369'),
            ('amend(3393 review): tighten the claim', '3393'),
            ('test(1955 S7): cover the seam', '1955'),
            ('fix(4369, esc-4369-2): repair the guard', '4369'),
            ('feat(3043/step-16): GREEN — settle the doubles', '3043'),
            ('test: RED — resolves to main (task 5181 step-1)', '5181'),
            ('amend: name the alternative (task 5018 review)', '5018'),
            ('test(dep-direction): drop the meta-test (task 3770, step-19)', '3770'),
            ('fix(merge_queue): reclaim the worktree (task 3251/step-4)', '3251'),
        ],
    )
    def test_subject_cites_task(self, subject: str, tid: str) -> None:
        assert _cites(subject, tid)


class TestExistingFormsStillCite:
    @pytest.mark.parametrize(
        ('subject', 'tid'),
        [
            ('impl(4369): x', '4369'),
            ('test(4369: wire it)', '4369'),
            ('Merge task/4369 into main', '4369'),
            ('chore(ci): rebuild task/4369 artifacts', '4369'),
            ('refactor: tidy (#4369)', '4369'),
            ('Cleanup pass (4369)', '4369'),
            ('Handle the edge case (task 4369)', '4369'),
        ],
    )
    def test_subject_cites_task(self, subject: str, tid: str) -> None:
        assert _cites(subject, tid)


class TestNonCitationsDoNotCite:
    @pytest.mark.parametrize(
        ('subject', 'tid'),
        [
            pytest.param('impl(4370 step-8): x', '4369', id='other-task-scope'),
            pytest.param('docs: x (task 4370 step-1)', '4369', id='other-task-paren'),
            pytest.param('impl(43690 step-8): x', '4369', id='right-superstring-scope'),
            pytest.param('docs: x (task 43690 step-1)', '4369', id='right-superstring-paren'),
            pytest.param('impl(14369 step-8): x', '4369', id='left-superstring-scope'),
            pytest.param('impl(4369.2 step-1): x', '4369', id='dotted-subtask-scope'),
            pytest.param('docs: x (task 4369.2 step-1)', '4369', id='dotted-subtask-paren'),
            pytest.param(
                'fix: retry harder (4369 attempts)', '4369', id='bare-paren-stays-exact',
            ),
            pytest.param(
                'fix(dashboard): wait longer (30 s)', '30', id='bare-paren-unit-quantity',
            ),
            pytest.param('fix(2-phase commit): x', '2', id='hyphen-is-not-a-separator'),
            pytest.param(
                'docs: describe the impl(4369 step-8) convention', '4369',
                id='scope-form-is-anchored',
            ),
        ],
    )
    def test_subject_does_not_cite_task(self, subject: str, tid: str) -> None:
        assert not _cites(subject, tid)


def _git(repo: Path, *args: str) -> str:
    assert_isolated_git_repo(repo)
    proc = subprocess.run(
        ['git', *args], cwd=str(repo), capture_output=True,
        env=git_env_with_ceiling(repo), text=True, check=False,
    )
    assert proc.returncode == 0, f'git {" ".join(args)} failed: {proc.stderr.strip()}'
    return proc.stdout.strip()


def _commit(repo: Path, subject: str, *, body: str | None = None, path: str = 'f.txt') -> str:
    """Append a unique line to *path*, commit it and return the new sha."""
    target = repo / path
    existing = target.read_text() if target.exists() else ''
    target.write_text(f'{existing}{subject}\n')
    _git(repo, 'add', path)
    message = ['-m', subject] + (['-m', body] if body is not None else [])
    _git(repo, 'commit', *message)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(config, git_repo)


@pytest.mark.asyncio
class TestGitDialect:
    """The widened string is valid git ERE, and the subject-only re-test still holds."""

    async def test_trailing_qualifier_scope_subject_is_found(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        sha = _commit(git_repo, 'impl(4369 step-8): add thing')

        assert await git_ops.find_task_citation_commit('4369') == sha

    async def test_task_word_paren_with_trailing_content_is_found(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        sha = _commit(git_repo, 'docs: y (task 4369, step-2)')

        assert await git_ops.find_task_citation_commit('4369') == sha

    async def test_other_task_trailing_qualifier_is_not_found(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        _commit(git_repo, 'impl(4370 step-8): x')

        assert await git_ops.find_task_citation_commit('4369') is None

    async def test_body_only_trailing_qualifier_is_not_found(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        _commit(git_repo, 'chore: notes', body='(task 4369 step-1)')

        assert await git_ops.find_task_citation_commit('4369') is None
