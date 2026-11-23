"""Both directions of ``ALREADY_LANDED_CITATION_PATTERN`` (task 6221).

The already-landed gate's template must attribute a task's own AUTHORED
delivery in conventional-commit scope position, with or without a trailing
qualifier, and must never attribute a commit that merely mentions the id or
undoes the delivery.  Both directions matter: a false positive excuses
non-delivery; a false negative only fails closed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.config import GitConfig
from orchestrator.git_ops import DEFAULT_COMMIT_CITATION_PATTERN, GitOps, render_citation_pattern
from orchestrator.merge_lane.gates import (
    ALREADY_LANDED_CITATION_PATTERN,
    resolve_already_landed_branch,
)


def _matches(template: str, subject: str, tid: str) -> bool:
    """Apply *template* to one SUBJECT the way ``find_task_citation_commit`` does."""
    rendered = render_citation_pattern(template, tid, branch_prefix='task/', main_branch='main')
    return re.compile(rendered).search(subject) is not None


def _attributes(subject: str, tid: str) -> bool:
    return _matches(ALREADY_LANDED_CITATION_PATTERN, subject, tid)


def _default_cites(subject: str, tid: str) -> bool:
    return _matches(DEFAULT_COMMIT_CITATION_PATTERN, subject, tid)


_QUALIFIED_SCOPE_DELIVERIES = [
    ('impl(4369 step-8): implement _pyright_wrapper_tokens', '4369'),
    ('amend(3393 review): tighten the claim', '3393'),
    ('test(1910 step-1): RED — TestContentHash', '1910'),
    ('feat(3043/step-16): GREEN — settle the doubles', '3043'),
    ('fix(4369, esc-4369-2): repair the guard', '4369'),
    ('feat(4369 step-2)!: break the API', '4369'),
    ('fix(5284/#5014): GREEN — report undecodable output', '5284'),
    ('impl(15.2 step-1): x', '15.2'),
]

_SCOPE_POSITION_NON_DELIVERIES = [
    pytest.param('impl(4370 step-8): x', '4369', id='other-task-qualified'),
    pytest.param('impl(43690 step-8): x', '4369', id='right-superstring'),
    pytest.param('impl(14369 step-8): x', '4369', id='left-superstring'),
    pytest.param('impl(4369.2 step-1): x', '4369', id='dotted-subtask-of-parent'),
    pytest.param('impl(15.2 step-1): x', '15', id='dotted-parent'),
    pytest.param('impl(4369-4370): x', '4369', id='hyphen-is-not-a-separator'),
    pytest.param('fix(4200/4369): x', '4369', id='secondary-scope-id'),
    pytest.param('fix(5284/#5014): x', '5014', id='secondary-hash-scope-id'),
]

_MENTION_ONLY = ('fix(4200): back out task/4369 changes', '4369')


class TestQualifiedScopeAttributes:
    @pytest.mark.parametrize(('subject', 'tid'), _QUALIFIED_SCOPE_DELIVERIES)
    def test_subject_attributes_task(self, subject: str, tid: str) -> None:
        assert _attributes(subject, tid)


class TestExactScopeAndMergeStillAttribute:
    @pytest.mark.parametrize(
        'subject',
        ['impl(4369): x', 'fix(#4369): x', 'feat(4369)!: x', 'Merge task/4369 into main'],
    )
    def test_subject_attributes_task(self, subject: str) -> None:
        assert _attributes(subject, '4369')


class TestNonDeliveriesDoNotAttribute:
    @pytest.mark.parametrize(
        ('subject', 'tid'),
        [
            *_SCOPE_POSITION_NON_DELIVERIES,
            pytest.param(*_MENTION_ONLY, id='mention-only'),
            pytest.param('Revert "impl(4369 step-8): x"', '4369', id='revert-of-qualified'),
            pytest.param('Revert "impl(4369): x"', '4369', id='revert-of-exact'),
            pytest.param(
                'docs: describe the impl(4369 step-8): convention', '4369',
                id='unanchored-scope-form',
            ),
            pytest.param('chore: x (task 4369 step-1)', '4369', id='trailing-task-paren'),
            pytest.param('refactor: tidy (#4369)', '4369', id='hash-paren'),
            pytest.param('impl(4369 step-8) no colon', '4369', id='no-colon'),
            pytest.param('impl(4369 step-8 (a)): x', '4369', id='nested-paren'),
            pytest.param('wip(4369 step-4): x', '4369', id='type-outside-vocabulary'),
        ],
    )
    def test_subject_does_not_attribute_task(self, subject: str, tid: str) -> None:
        assert not _attributes(subject, tid)


class TestScopeArmAgreesWithDefault:
    """The scope-qualifier separator class is spelled in both templates; it must not drift."""

    @pytest.mark.parametrize(
        ('subject', 'tid'), [*_QUALIFIED_SCOPE_DELIVERIES, *_SCOPE_POSITION_NON_DELIVERIES],
    )
    def test_scope_position_subject_agrees(self, subject: str, tid: str) -> None:
        assert _attributes(subject, tid) == _default_cites(subject, tid)

    def test_the_mention_arm_is_the_deliberate_divergence(self) -> None:
        """Only the scope arm is shared: the default's ``task/{tid}`` arm is what the gate excludes."""
        assert _default_cites(*_MENTION_ONLY)
        assert not _attributes(*_MENTION_ONLY)

    def test_a_hash_led_qualified_scope_is_the_one_scope_divergence(self) -> None:
        """``(#N <qual>)`` is scope position here, yet no arm of the default cites it."""
        subject = 'fix(#4369 step-1): x'
        assert _attributes(subject, '4369')
        assert not _default_cites(subject, '4369')


def _git(repo: Path, *args: str) -> str:
    assert_isolated_git_repo(repo)
    proc = subprocess.run(
        ['git', *args], cwd=str(repo), capture_output=True,
        env=git_env_with_ceiling(repo), text=True, check=False,
    )
    assert proc.returncode == 0, f'git {" ".join(args)} failed: {proc.stderr.strip()}'
    return proc.stdout.strip()


def _commit(
    repo: Path, files: dict[str, str], subject: str, *, body: str | None = None,
) -> str:
    """Write *files* (path -> content), commit them and return the new sha."""
    for relpath, content in files.items():
        target = repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    _git(repo, 'add', *files)
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


async def _find(git_ops: GitOps, tid: str) -> str | None:
    return await git_ops.find_task_citation_commit(
        tid, pattern_template=ALREADY_LANDED_CITATION_PATTERN,
    )


@pytest.mark.asyncio
class TestGitDialect:
    """The template is valid git ERE, and the subject-only re-test still holds."""

    async def test_qualified_scope_subject_is_found(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        sha = _commit(git_repo, {'f.txt': 'one\n'}, 'impl(4369 step-8): add thing')

        assert await _find(git_ops, '4369') == sha

    async def test_other_task_qualified_scope_is_not_found(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        _commit(git_repo, {'f.txt': 'one\n'}, 'impl(4370 step-8): x')

        assert await _find(git_ops, '4369') is None

    async def test_a_newer_revert_does_not_shadow_the_delivery(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        delivery = _commit(git_repo, {'f.txt': 'one\n'}, 'impl(4369 step-8): add thing')
        _commit(git_repo, {'f.txt': ''}, 'Revert "impl(4369 step-8): add thing"')

        assert await _find(git_ops, '4369') == delivery

    async def test_body_only_qualified_scope_is_not_found(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        _commit(git_repo, {'f.txt': 'one\n'}, 'chore: notes', body='impl(4369 step-1): x')

        assert await _find(git_ops, '4369') is None


def _park(repo: Path, branch: str, sha: str) -> None:
    """Park *branch* on one of main's own commits, as a re-dispatch rebase leaves it."""
    _git(repo, 'branch', '-f', branch, sha)


@pytest.mark.asyncio
class TestGateAttributesAQualifiedTrainMember:
    PLAN_FILES = ['src/pkg/member.py']

    async def test_a_qualified_scope_train_member_is_already_landed(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        sha = _commit(
            git_repo, {'src/pkg/member.py': '# member\n'},
            'impl(923 step-2): deliver inside a coalesce train',
        )
        _park(git_repo, 'task/923', sha)

        result = await resolve_already_landed_branch(
            self.PLAN_FILES, sha, sha, git_ops, task_id='923', branch='task/923',
        )

        assert result is not None
        assert result.mechanism == 'task_citation'
        assert result.landed_sha == sha
        assert result.matched_files == self.PLAN_FILES

    async def test_a_superstring_id_is_not_its_delivery(
        self, git_ops: GitOps, git_repo: Path,
    ) -> None:
        """Paired with the positive above, which alone would pass a template matching everything.

        No revert decoy here: ``_REVERT_SUBJECT_RE`` would mask a template
        regression at this level, so revert rejection is pinned at the
        template by ``TestGitDialect`` and ``TestNonDeliveriesDoNotAttribute``.
        """
        parked = _commit(
            git_repo, {'src/pkg/member.py': '# another task\n'},
            'impl(9240 step-2): another task',
        )
        _park(git_repo, 'task/924', parked)

        assert await resolve_already_landed_branch(
            self.PLAN_FILES, parked, parked, git_ops, task_id='924', branch='task/924',
        ) is None
