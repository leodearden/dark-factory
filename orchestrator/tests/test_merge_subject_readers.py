"""Boundary test B1, orchestrator half: every merge-subject reader recognises what git_ops writes.

The fixture lands task ``50`` through the real writer in a repo whose naming is
``feat/`` into ``trunk``, so a reader that still assumes ``task/`` into ``main``
cannot see it. A ``Revert "<subject for feat/51>"`` commit on trunk is the
negative every reader must refuse. The fused-memory half lives in
``fused-memory/tests/test_landed_on_main.py``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from _git_fixtures import seed_repo
from _orch_helpers import git_env_with_ceiling
from shared.merge_subject import format_merge_subject, rendered_prefix

from orchestrator.config import GitConfig
from orchestrator.evals.reviewer_trial.mining import resolve_merge_sha_by_task_id
from orchestrator.evals.task_sampler import discover_completed_tasks
from orchestrator.git_ops import DEFAULT_COMMIT_CITATION_PATTERN, GitOps, render_citation_pattern
from orchestrator.merge_lane.gates import ALREADY_LANDED_CITATION_PATTERN

_CITATION_LIB = Path(__file__).resolve().parents[1] / 'scripts' / 'warm-lane' / 'lib_task_citation.sh'


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

    async def test_feat_50s_landing_is_not_feat_5s_marker(self, feat_trunk: FeatTrunk) -> None:
        git_ops = feat_trunk.git_ops
        assert await git_ops.find_merge_marker('feat/5', gate_on_existing_ref=False) is None
        assert await git_ops.find_merge_marker_checked('feat/5') is None


_TEMPLATES = [
    pytest.param(DEFAULT_COMMIT_CITATION_PATTERN, id='default'),
    pytest.param(ALREADY_LANDED_CITATION_PATTERN, id='already-landed'),
]


@pytest.mark.asyncio
class TestCitationTemplatesFollowTheConfiguredSubject:
    async def test_the_already_landed_template_cites_the_landing(
        self, feat_trunk: FeatTrunk,
    ) -> None:
        cited = await feat_trunk.git_ops.find_task_citation_commit(
            '50', pattern_template=ALREADY_LANDED_CITATION_PATTERN,
        )
        assert cited == feat_trunk.merge_sha

    async def test_the_default_template_cites_the_landing(self, feat_trunk: FeatTrunk) -> None:
        assert await feat_trunk.git_ops.find_task_citation_commit('50') == feat_trunk.merge_sha

    @pytest.mark.parametrize('template', _TEMPLATES)
    async def test_a_revert_subject_cites_nothing(
        self, feat_trunk: FeatTrunk, template: str,
    ) -> None:
        cited = await feat_trunk.git_ops.find_task_citation_commit('51', pattern_template=template)
        assert cited is None


def _feat_trunk_landing(tid: str = '50') -> str:
    return format_merge_subject(f'feat/{tid}', 'trunk')


class TestRenderCitationPattern:
    @pytest.mark.parametrize('template', _TEMPLATES)
    def test_the_merge_slot_cites_the_configured_landing(self, template: str) -> None:
        rendered = render_citation_pattern(template, '50', branch_prefix='feat/', main_branch='trunk')
        assert re.search(rendered, _feat_trunk_landing())

    @pytest.mark.parametrize('template', _TEMPLATES)
    @pytest.mark.parametrize('subject', [
        pytest.param('Revert "' + _feat_trunk_landing() + '"', id='revert'),
        pytest.param(format_merge_subject('feat/50', 'main'), id='other-main'),
        pytest.param(format_merge_subject('task/50', 'trunk'), id='other-prefix'),
        pytest.param(_feat_trunk_landing('500'), id='superstring-id'),
        pytest.param(_feat_trunk_landing() + ' (retry)', id='suffixed'),
    ])
    def test_the_merge_slot_refuses_anything_else(self, template: str, subject: str) -> None:
        rendered = render_citation_pattern(template, '50', branch_prefix='feat/', main_branch='trunk')
        assert re.search(rendered, subject) is None

    def test_rendering_leaves_the_conventional_commit_arms_alone(self) -> None:
        rendered = render_citation_pattern(
            ALREADY_LANDED_CITATION_PATTERN, '4369', branch_prefix='task/', main_branch='main',
        )
        assert re.search(rendered, 'impl(4369 step-8): x')
        assert re.search(rendered, 'impl(43690 step-8): x') is None


def _resolve_feat_trunk(task_id: str, repo: Path) -> str | None:
    return resolve_merge_sha_by_task_id(task_id, repo, branch_prefix='feat/', main_branch='trunk')


class TestEvalMinersFollowTheConfiguredSubject:
    @pytest.mark.asyncio
    async def test_the_sampler_discovers_only_the_landing(self, feat_trunk: FeatTrunk) -> None:
        candidates = await discover_completed_tasks(
            feat_trunk.repo, since=None, project='dark_factory',
            branch_prefix='feat/', main_branch='trunk',
        )
        first_parent = _git(feat_trunk.repo, 'rev-parse', f'{feat_trunk.merge_sha}^1')
        assert [(c.task_id, c.merge_sha, c.pre_commit) for c in candidates] == [
            ('50', feat_trunk.merge_sha, first_parent),
        ]

    def test_mining_resolves_only_the_landing(self, feat_trunk: FeatTrunk) -> None:
        assert _resolve_feat_trunk('50', feat_trunk.repo) == feat_trunk.merge_sha
        assert _resolve_feat_trunk('51', feat_trunk.repo) is None

    def test_mining_does_not_resolve_an_id_the_landing_merely_starts_with(
        self, feat_trunk: FeatTrunk,
    ) -> None:
        assert _resolve_feat_trunk('5', feat_trunk.repo) is None

    def test_mining_reads_the_subject_not_a_body_line(self, feat_trunk: FeatTrunk) -> None:
        repo = feat_trunk.repo
        quoting = _git(
            repo, 'commit-tree', 'HEAD^{tree}', '-p', 'HEAD',
            '-m', 'docs: quote', '-m', format_merge_subject('feat/52', 'trunk'),
        )
        _git(repo, 'update-ref', 'refs/heads/side', quoting)

        assert _resolve_feat_trunk('52', repo) is None

    def test_mining_answers_none_for_an_id_that_is_not_a_ref_token(
        self, feat_trunk: FeatTrunk,
    ) -> None:
        assert _resolve_feat_trunk('a b', feat_trunk.repo) is None


def _commit_wrapped_landing(repo: Path, tid: str) -> str:
    """Land ``feat/<tid>`` with its subject wrapped onto two lines, which git's %s joins."""
    wrapped = format_merge_subject(f'feat/{tid}', 'trunk').replace(' into ', '\ninto ', 1)
    _git(repo, 'commit', '--allow-empty', '--no-verify', '-m', wrapped)
    return _git(repo, 'rev-parse', 'HEAD')


class TestEveryReaderSeesAWrappedLanding:
    @pytest.mark.asyncio
    async def test_the_marker_reader_sees_it(self, feat_trunk: FeatTrunk) -> None:
        wrapped_sha = _commit_wrapped_landing(feat_trunk.repo, '53')
        marker = await feat_trunk.git_ops.find_merge_marker('feat/53', gate_on_existing_ref=False)
        assert marker == wrapped_sha

    @pytest.mark.asyncio
    async def test_the_sampler_sees_it(self, feat_trunk: FeatTrunk) -> None:
        wrapped_sha = _commit_wrapped_landing(feat_trunk.repo, '53')
        candidates = await discover_completed_tasks(
            feat_trunk.repo, since=None, project='dark_factory',
            branch_prefix='feat/', main_branch='trunk',
        )
        assert [(c.task_id, c.merge_sha) for c in candidates] == [
            ('53', wrapped_sha), ('50', feat_trunk.merge_sha),
        ]

    def test_mining_sees_it(self, feat_trunk: FeatTrunk) -> None:
        wrapped_sha = _commit_wrapped_landing(feat_trunk.repo, '53')
        assert _resolve_feat_trunk('53', feat_trunk.repo) == wrapped_sha


_LIB_CITES = (
    'source "$1"; prefix_re="$(task_citation_regex_escape "$2")"; '
    'task_citation_message_cites "$3" "$4" "$prefix_re" "$5" "$6"'
)


def _lib_cites(message: str, task_id: str, *, rendered_for: str = 'feat/') -> bool:
    """Ask lib_task_citation.sh, prefix ``feat/``, with the merge arm rendered for *rendered_for*."""
    head, into = rendered_prefix(rendered_for)
    completed = subprocess.run(
        ['bash', '-c', _LIB_CITES, '_', str(_CITATION_LIB), 'feat/', message, task_id, head, into],
        capture_output=True, text=True,
    )
    assert completed.returncode in (0, 1) and completed.stderr == '', completed
    return completed.returncode == 0


class TestLibTaskCitationFedRenderedPrefix:
    def test_the_landing_message_cites_its_task(self, feat_trunk: FeatTrunk) -> None:
        message = _git(feat_trunk.repo, 'show', '-s', '--format=%B', feat_trunk.merge_sha)
        assert _lib_cites(message, '50')

    def test_a_revert_message_cites_nothing(self, feat_trunk: FeatTrunk) -> None:
        message = _git(feat_trunk.repo, 'show', '-s', '--format=%B', feat_trunk.revert_sha)
        assert not _lib_cites(message, '51')

    def test_the_lib_uses_the_merge_arm_it_is_given(self) -> None:
        assert not _lib_cites(_feat_trunk_landing(), '50', rendered_for='wip/')
