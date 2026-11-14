"""What a blocked or conflicted merge outcome records for the workflow and runs.db (task 7302)."""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeVerifier,
    fails,
    healthy_verify_clock,
    lane_scene_config,
    make_lane,
    merge_through_lane,
    running_lane,
)
from escalation.categories import EscalationCategory

from orchestrator import verify_runner
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeResult
from orchestrator.merge_lane.outcome_facts import (
    MAX_RECORDED_CONFLICTED_PATHS,
    ConflictEvidence,
    MergeBlockCategory,
)
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest, QueuedBranch


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return lane_scene_config(git_repo, git_config)


async def _prepare(
    git_ops: GitOps, config: OrchestratorConfig, task_id: str, filename: str, content: str,
) -> MergeRequest:
    """Commit *content* to *filename* on a fresh branch off main, unsubmitted."""
    worktree = (await git_ops.create_worktree(task_id)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'{task_id}: edit {filename}')
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


class TestOutcomeFacts:
    @pytest.mark.parametrize('member', list(MergeBlockCategory))
    def test_every_block_category_is_a_registered_escalation_category(
        self, member: MergeBlockCategory,
    ) -> None:
        assert member.value in EscalationCategory

    def test_needs_rebase_spelling(self) -> None:
        assert MergeBlockCategory.MERGE_NEEDS_REBASE == 'merge_needs_rebase'

    def test_unattributed_evidence_omits_counterparts(self) -> None:
        evidence = ConflictEvidence(conflicted_paths=('a.py', 'b.py'))

        assert evidence.to_event_payload() == {
            'conflicted_paths': ['a.py', 'b.py'],
            'conflicted_paths_total': 2,
        }

    @pytest.mark.parametrize(
        ('counterparts', 'recorded'),
        [((), []), (('7202',), ['7202'])],
    )
    def test_attributed_evidence_records_counterparts(
        self, counterparts: tuple[str, ...], recorded: list[str],
    ) -> None:
        evidence = ConflictEvidence(
            conflicted_paths=('a.py',), counterpart_task_ids=counterparts,
        )

        assert evidence.to_event_payload()['counterpart_task_ids'] == recorded

    def test_recorded_paths_are_bounded_and_the_true_count_kept(self) -> None:
        paths = tuple(f'f{i}.py' for i in range(MAX_RECORDED_CONFLICTED_PATHS + 5))

        payload = ConflictEvidence(conflicted_paths=paths).to_event_payload()

        assert payload['conflicted_paths'] == list(paths[:MAX_RECORDED_CONFLICTED_PATHS])
        assert payload['conflicted_paths_total'] == MAX_RECORDED_CONFLICTED_PATHS + 5

    def test_evidence_is_frozen(self) -> None:
        evidence = ConflictEvidence(conflicted_paths=('a.py',))

        pytest.raises(
            dataclasses.FrozenInstanceError, setattr, evidence, 'conflicted_paths', ('b.py',),
        )

    def test_bare_blocked_outcome_defaults(self) -> None:
        outcome = MergeOutcome('blocked', reason='x')

        assert outcome.block_category is MergeBlockCategory.MERGE_ERROR
        assert outcome.conflict_evidence is None


@pytest.mark.asyncio
class TestPostMergeVerifyBlocksAreTyped:
    @pytest.mark.parametrize(
        'category', ['test_failure', verify_runner.UNSCOPED_TYPECHECK_FAILED_CATEGORY],
    )
    async def test_a_failed_post_merge_verify_is_filed_as_post_merge_verify(
        self, git_ops: GitOps, config: OrchestratorConfig, category: str,
    ) -> None:
        req = await _prepare(git_ops, config, 'vf', 'vf.py', 'x = 1\n')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        verifier = FakeVerifier(scripts={'vf': fails(category=category, summary='orchestrator')})
        lane = make_lane(git_ops, queue, verifier=verifier, clock=healthy_verify_clock())

        outcome = await merge_through_lane(lane, queue, req)

        assert outcome.status == 'blocked', outcome
        assert outcome.reason.startswith('Post-merge verification failed'), outcome.reason
        assert outcome.block_category is MergeBlockCategory.POST_MERGE_VERIFY


async def _commit_on_new_branch(git_ops: GitOps, task_id: str, filename: str, content: str) -> Path:
    worktree = (await git_ops.create_worktree(task_id)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'{task_id}: add {filename}')
    return worktree


@pytest.mark.asyncio
class TestMergeToMainReportsConflictedPaths:
    async def test_a_conflicting_merge_names_its_unmerged_paths(self, git_ops: GitOps) -> None:
        # Both branches are cut from the same main and each ADD clash.py: add/add.
        worktree_a = await _commit_on_new_branch(git_ops, 'clash-a', 'clash.py', 'x = 1\n')
        worktree_b = await _commit_on_new_branch(git_ops, 'clash-b', 'clash.py', 'x = 2\n')
        landed = await git_ops.merge_to_main(worktree_a, 'clash-a')
        assert landed.merge_commit is not None and landed.merge_worktree is not None
        await git_ops.advance_main(landed.merge_commit)
        await git_ops.cleanup_merge_worktree(landed.merge_worktree)

        result = await git_ops.merge_to_main(worktree_b, 'clash-b')

        assert result.merge_worktree is not None
        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        assert result.conflicts is True
        assert result.conflicted_paths == ('clash.py',)

    async def test_a_clean_merge_names_no_paths(self, git_ops: GitOps) -> None:
        worktree = await _commit_on_new_branch(git_ops, 'clean', 'clean.py', 'x = 1\n')

        result = await git_ops.merge_to_main(worktree, 'clean')

        assert result.merge_worktree is not None
        await git_ops.cleanup_merge_worktree(result.merge_worktree)
        assert result.success is True
        assert result.conflicted_paths == ()

    async def test_the_details_describe_the_recorded_paths(self, git_ops: GitOps) -> None:
        # A non-ASCII name is what git's default core.quotePath would quote.
        result = await _add_add_conflict(git_ops, 'clä sh.py')

        assert result.conflicted_paths == ('clä sh.py',)
        assert result.details.startswith('Conflicting files:\nclä sh.py\n'), result.details
        assert '<<<<<<<' in result.details.split('--- clä sh.py ---\n', 1)[1], result.details

    @pytest.mark.parametrize(
        'listing',
        [
            pytest.param(lambda: AsyncMock(side_effect=RuntimeError('git diff failed')), id='fails'),
            pytest.param(lambda: AsyncMock(return_value=()), id='names_nothing'),
        ],
    )
    async def test_paths_git_cannot_name_are_reported_unknown(
        self, git_ops: GitOps, listing: Callable[[], AsyncMock],
    ) -> None:
        result = await _add_add_conflict(git_ops, 'clash.py', unmerged_paths=listing())

        assert result.conflicts is True
        assert result.conflicted_paths is None
        assert 'could not be listed' in result.details, result.details


async def _add_add_conflict(
    git_ops: GitOps, filename: str, unmerged_paths: AsyncMock | None = None,
) -> MergeResult:
    """Land one ADD of *filename*, then merge a second ADD of it cut from the same main.

    *unmerged_paths*, when given, replaces the instance's listing for the
    conflicting merge only.
    """
    worktree_a = await _commit_on_new_branch(git_ops, 'add-a', filename, 'x = 1\n')
    worktree_b = await _commit_on_new_branch(git_ops, 'add-b', filename, 'x = 2\n')
    landed = await git_ops.merge_to_main(worktree_a, 'add-a')
    assert landed.merge_commit is not None and landed.merge_worktree is not None
    await git_ops.advance_main(landed.merge_commit)
    await git_ops.cleanup_merge_worktree(landed.merge_worktree)
    if unmerged_paths is not None:
        git_ops.unmerged_paths = unmerged_paths
    result = await git_ops.merge_to_main(worktree_b, 'add-b')
    assert result.merge_worktree is not None
    await git_ops.cleanup_merge_worktree(result.merge_worktree)
    return result


async def _conflict_through_lane(
    git_ops: GitOps, config: OrchestratorConfig, store: EventStore,
) -> tuple[MergeOutcome, dict[str, object]]:
    """Land conf-a, then finalize conf-b's add/add conflict; its outcome and conflict row data."""
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    lane = make_lane(git_ops, queue, event_store=store, clock=healthy_verify_clock())
    # Both cut before either lands, so each ADDS clash.py: an add/add conflict.
    first = await _prepare(git_ops, config, 'conf-a', 'clash.py', 'x = 1\n')
    second = await _prepare(git_ops, config, 'conf-b', 'clash.py', 'x = 2\n')
    # With task/conf-b resolvable, the conflict graph would probe it against
    # main and bounce it before it merged. Moving the ref away leaves the
    # graph nothing to probe, so the conflict surfaces in merge_to_main's
    # merge of the worktree HEAD (its absent-ref fallback).
    subprocess.run(
        ['git', 'branch', '-m', 'task/conf-b', 'parked/conf-b'],
        cwd=second.worktree, check=True, capture_output=True,
    )

    async with running_lane(lane) as run:
        await queue.put(first)
        assert (await run.outcome(first)).status == 'done'
        await queue.put(second)
        outcome = await run.outcome(second)

    assert outcome.status == 'conflict', outcome
    conflict_rows = [
        row for row in store.fetch_events_by_type(EventType.merge_attempt)
        if row['task_id'] == 'conf-b' and row['data']['outcome'] == 'conflict'
    ]
    assert len(conflict_rows) == 1, conflict_rows
    return outcome, conflict_rows[0]['data']


@pytest.mark.asyncio
class TestConflictFinalizationRecordsPaths:
    async def test_the_conflict_outcome_and_its_merge_attempt_row_carry_the_paths(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        store = EventStore(tmp_path / 'runs.db', run_id='run-conflict-paths')

        outcome, data = await _conflict_through_lane(git_ops, config, store)

        assert outcome.conflict_evidence == ConflictEvidence(conflicted_paths=('clash.py',))
        assert data['conflicted_paths'] == ['clash.py']
        assert data['conflicted_paths_total'] == 1
        assert 'counterpart_task_ids' not in data

    async def test_paths_git_cannot_list_are_left_unrecorded(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        # Absent keys mean "paths unknown"; an empty list would claim no path conflicted.
        store = EventStore(tmp_path / 'runs.db', run_id='run-conflict-unlisted')
        git_ops.unmerged_paths = AsyncMock(side_effect=RuntimeError('git diff failed'))

        outcome, data = await _conflict_through_lane(git_ops, config, store)

        assert outcome.conflict_evidence is None
        assert 'conflicted_paths' not in data, data
        assert 'conflicted_paths_total' not in data, data
