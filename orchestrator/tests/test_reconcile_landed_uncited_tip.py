"""A landed branch whose tip subject names no task must not read as degenerate.

The warm-lane degenerate-ref script judges a landed branch by its tip subject
alone, so an uncited ``amend: …`` tip landed by ``Merge task/N into main`` reads
``degenerate`` (reify 6828, 7522, 8757).  ``branch_tip_landed`` is the
structural veto: the tip is among the commits a landing brought onto main.
Every repository here is real git; nothing in the ancestry path is mocked.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import (
    assert_isolated_git_repo,
    git_env_with_ceiling,
    wire_clear_claim_then_set_status,
    wire_live_status_from_get_task,
    wire_scheduler_liveness_mock,
)

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness
from orchestrator.merge_lane.landed_outbox import MergeProvenance
from orchestrator.merge_lane.landing_evidence import branch_tip_landed


class _Repo:
    """A synthetic repository whose every git call carries both isolation guards."""

    def __init__(self, root: Path) -> None:
        assert_isolated_git_repo(root)
        self.root = root

    def git(self, *args: str) -> str:
        assert_isolated_git_repo(self.root)
        proc = subprocess.run(
            ['git', *args], cwd=str(self.root), capture_output=True,
            env=git_env_with_ceiling(self.root), text=True, check=False,
        )
        assert proc.returncode == 0, f'git {" ".join(args)} failed: {proc.stderr.strip()}'
        return proc.stdout.strip()

    def commit(self, message: str, files: dict[str, str]) -> str:
        for rel, content in files.items():
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.sha('HEAD')

    def sha(self, ref: str) -> str:
        return self.git('rev-parse', ref)


def _git_ops(root: Path) -> GitOps:
    config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(config, root)


@pytest.fixture
def repo(tmp_path: Path) -> _Repo:
    return _Repo(seed_repo(tmp_path / 'repo'))


@pytest.fixture
def git_ops(repo: _Repo) -> GitOps:
    return _git_ops(repo.root)


def land_uncited_tip(repo: _Repo, tid: str) -> tuple[str, str, str]:
    """The 6828 shape: an uncited tip landed by a no-ff ``Merge task/<tid> into main``.

    The branch forks main, does work, merges main into itself, then ends on an
    ``amend:`` commit that names no task.  Returns (fork point, tip, landing).
    """
    branch = f'task/{tid}'
    fork_point = repo.sha('main')
    repo.git('checkout', '-b', branch, 'main')
    repo.commit(f'feat({tid}): add the note', {f'note-{tid}.txt': 'first draft\n'})
    repo.git('checkout', 'main')
    repo.commit('chore: unrelated one', {'other-1.txt': 'one\n'})
    repo.git('checkout', branch)
    repo.git('merge', '--no-ff', 'main', '-m', f'Merge main into {branch}')
    tip = repo.commit('amend: tidy the note', {f'note-{tid}.txt': 'tidied\n'})
    repo.git('checkout', 'main')
    repo.commit('chore: unrelated two', {'other-2.txt': 'two\n'})
    repo.git('merge', '--no-ff', branch, '-m', f'Merge {branch} into main')
    return fork_point, tip, repo.sha('HEAD')


def land_then_reseed_foreign_tip(repo: _Repo, tid: str) -> tuple[str, str]:
    """``task/<tid>`` landed earlier, then was reset onto a LATER main commit.

    The reseeded tip carries no work of its own.  Returns (tip, landing).
    """
    branch = f'task/{tid}'
    repo.git('checkout', '-b', branch, 'main')
    repo.commit(f'feat({tid}): real work', {f'work-{tid}.txt': 'work\n'})
    repo.git('checkout', 'main')
    repo.git('merge', '--no-ff', branch, '-m', f'Merge {branch} into main')
    landing = repo.sha('HEAD')
    later = repo.commit('chore: later main work', {'later.txt': 'later\n'})
    repo.git('branch', '-f', branch, later)
    return later, landing


@pytest.mark.asyncio
class TestBranchTipLanded:
    async def test_no_ff_landing_of_an_uncited_tip_is_landed(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        _, tip, landing = land_uncited_tip(repo, '701')

        assert await branch_tip_landed(git_ops, 'task/701', tip, landing_sha=landing) is True

    async def test_fast_forward_self_landing_is_landed(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        tip = repo.commit('chore: plain commit on main', {'plain.txt': 'plain\n'})

        assert await branch_tip_landed(git_ops, 'task/703', tip, landing_sha=tip) is True

    async def test_reseeded_foreign_tip_is_not_landed(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        tip, landing = land_then_reseed_foreign_tip(repo, '702')

        assert await branch_tip_landed(git_ops, 'task/702', tip, landing_sha=landing) is False

    async def test_tip_advanced_past_the_landing_is_not_landed(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        _, _, landing = land_uncited_tip(repo, '701')
        repo.git('checkout', 'task/701')
        tip = repo.commit('amend: one more pass', {'note-701.txt': 'again\n'})

        assert await branch_tip_landed(git_ops, 'task/701', tip, landing_sha=landing) is False

    async def test_later_citation_falls_back_to_the_branch_merge_marker(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        _, tip, _ = land_uncited_tip(repo, '701')
        follow_up = repo.commit('fix(701): follow-up', {'note-701.txt': 'fixed\n'})

        assert await branch_tip_landed(git_ops, 'task/701', tip, landing_sha=follow_up) is True

    async def test_unresolved_tip_is_not_landed(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        landing = repo.sha('main')

        assert await branch_tip_landed(git_ops, 'task/701', None, landing_sha=landing) is False

    async def test_unresolvable_landing_without_a_marker_is_not_landed(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        repo.git('checkout', '-b', 'task/704', 'main')
        tip = repo.commit('amend: never landed', {'stray.txt': 'stray\n'})
        repo.git('checkout', 'main')

        assert await branch_tip_landed(git_ops, 'task/704', tip, landing_sha='0' * 40) is False

    async def test_landing_with_no_first_parent_is_not_landed(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        root = repo.git('rev-list', '--max-parents=0', 'main')

        assert await branch_tip_landed(git_ops, 'task/705', root, landing_sha=root) is False

    async def test_no_attributed_landing_falls_back_to_the_branch_merge_marker(
        self, repo: _Repo, git_ops: GitOps,
    ) -> None:
        _, tip, _ = land_uncited_tip(repo, '701')

        assert await branch_tip_landed(git_ops, 'task/701', tip, landing_sha=None) is True


# ---------------------------------------------------------------------------
# The reconcile sweep over real git.  Only the warm-lane script is stubbed: it
# answers `degenerate`, the verdict both copies of
# warm-lane-degenerate-ref-check.sh give an uncited landed tip.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _unbound_merge_provenance() -> Iterator[None]:
    """No landed-outbox journal row leaks in: the sweep must decide from git alone."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


@pytest.fixture
def scheduler_double() -> MagicMock:
    scheduler = MagicMock()
    wire_scheduler_liveness_mock(scheduler)
    wire_clear_claim_then_set_status(scheduler)
    wire_live_status_from_get_task(scheduler)
    scheduler.get_statuses = AsyncMock(return_value=({}, None))
    scheduler.get_task = AsyncMock(return_value=None)
    scheduler.set_task_status = AsyncMock()

    async def _fake_mark_done(tid, *, kind, sha, note=None):
        provenance = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await scheduler.set_task_status(tid, 'done', done_provenance=provenance)

    scheduler.mark_done = AsyncMock(side_effect=_fake_mark_done)
    return scheduler


@pytest.fixture
def harness(
    tmp_path: Path, repo: _Repo, mock_orch_config: MagicMock, scheduler_double: MagicMock,
) -> Harness:
    mock_orch_config.project_root = repo.root
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)
    h.config.convert_to_blocked_enforce = False
    h.scheduler = scheduler_double
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    h.git_ops.warm_lane_ref_is_degenerate = AsyncMock(return_value=True)
    return h


async def run_mid_run_sweep(harness: Harness) -> None:
    """The file's one coupling to the sweep's private driver."""
    await harness._reconcile_stranded_in_progress(mid_run=True)


def stage_in_progress(scheduler: MagicMock, tid: str, metadata: dict[str, str]) -> None:
    scheduler.get_statuses = AsyncMock(return_value=({tid: 'in-progress'}, None))
    scheduler.get_task = AsyncMock(
        return_value={'status': 'in-progress', 'metadata': metadata},
    )


@pytest.mark.asyncio
class TestReconcileSweepOverUncitedLandedTip:
    async def test_landed_branch_with_uncited_tip_is_marked_done_not_reverted(
        self, repo: _Repo, harness: Harness, scheduler_double: MagicMock,
    ) -> None:
        fork_point, _, landing = land_uncited_tip(repo, '701')
        stage_in_progress(scheduler_double, '701', {'branch_base_sha': fork_point})

        await run_mid_run_sweep(harness)

        scheduler_double.mark_done.assert_awaited_once()
        assert scheduler_double.mark_done.await_args.kwargs['kind'] == 'found_on_main'
        assert scheduler_double.mark_done.await_args.kwargs['sha'] == landing
        assert call('701', 'pending') not in scheduler_double.set_task_status.await_args_list

    async def test_foreign_degenerate_tip_still_reverts(
        self, repo: _Repo, harness: Harness, scheduler_double: MagicMock,
    ) -> None:
        land_then_reseed_foreign_tip(repo, '702')
        stage_in_progress(scheduler_double, '702', {})

        await run_mid_run_sweep(harness)

        scheduler_double.set_task_status.assert_any_await('702', 'pending')
        scheduler_double.mark_done.assert_not_awaited()

    async def test_citation_disabled_reseeded_foreign_tip_still_reverts(
        self, repo: _Repo, harness: Harness, scheduler_double: MagicMock,
    ) -> None:
        harness.git_ops.config.commit_citation_pattern = ''
        land_then_reseed_foreign_tip(repo, '702')
        stage_in_progress(scheduler_double, '702', {})

        await run_mid_run_sweep(harness)

        scheduler_double.set_task_status.assert_any_await('702', 'pending')
        scheduler_double.mark_done.assert_not_awaited()

    async def test_citation_disabled_landed_uncited_tip_is_marked_done_via_its_merge_marker(
        self, repo: _Repo, harness: Harness, scheduler_double: MagicMock,
    ) -> None:
        harness.git_ops.config.commit_citation_pattern = ''
        fork_point, tip, _ = land_uncited_tip(repo, '701')
        stage_in_progress(scheduler_double, '701', {'branch_base_sha': fork_point})

        await run_mid_run_sweep(harness)

        scheduler_double.mark_done.assert_awaited_once()
        assert scheduler_double.mark_done.await_args.kwargs['kind'] == 'found_on_main'
        assert scheduler_double.mark_done.await_args.kwargs['sha'] == tip
        assert call('701', 'pending') not in scheduler_double.set_task_status.await_args_list
