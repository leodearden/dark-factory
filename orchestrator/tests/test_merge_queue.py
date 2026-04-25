"""Tests for merge queue: MergeWorker, CAS update-ref, ghost-loop detection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    DropGuardResult,
    MergeOutcome,
    MergeRequest,
    MergeWorker,
    SpeculativeItem,
    SpeculativeMergeWorker,
    UnresolvedStep,
    _check_plan_targets_in_tree,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # Tests use a tmp repo with no real remote; disabling the push avoids
        # per-test subprocess noise. Push behavior is exercised explicitly in
        # test_git_ops.TestPushMain and TestPushHook below.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    pre_rebased: bool = False,
) -> MergeRequest:
    future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


def _mock_verify_pass():
    """Return a mock that makes run_scoped_verification always pass."""
    return AsyncMock(return_value=MagicMock(passed=True, summary=''))


# ---------------------------------------------------------------------------
# TestCasUpdateRef — Phase A
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCasUpdateRef:
    async def test_advance_main_with_expected(self, git_ops: GitOps):
        """CAS succeeds when expected_main matches actual main."""
        worktree = (await git_ops.create_worktree('cas-ok')).path
        (worktree / 'file.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')

        result = await git_ops.merge_to_main(worktree, 'cas-ok')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        main_sha = await git_ops.get_main_sha()
        advanced = await git_ops.advance_main(
            result.merge_commit,
            expected_main=main_sha,
        )
        assert advanced == 'advanced'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_advance_main_cas_mismatch(self, git_ops: GitOps):
        """Main moved past merge commit with no worktree for retry → not_descendant.

        Note: the CAS check (update-ref expected_main) only runs AFTER the
        descendant check passes.  Without a merge_worktree to rebase onto
        the new main, advance_main cannot make the commit a descendant, so
        it returns 'not_descendant' before reaching the CAS step.
        """
        worktree = (await git_ops.create_worktree('cas-fail')).path
        (worktree / 'file.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')

        result = await git_ops.merge_to_main(worktree, 'cas-fail')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        # Simulate external actor advancing main
        stale_sha = await git_ops.get_main_sha()
        (git_ops.project_root / 'external.py').write_text('ext = True\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'External commit'], cwd=git_ops.project_root)

        # Merge commit is no longer a descendant of (new) main and no
        # merge_worktree was passed for retry → not_descendant
        advanced = await git_ops.advance_main(
            result.merge_commit,
            expected_main=stale_sha,
        )
        assert advanced == 'not_descendant'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_advance_main_none_expected(self, git_ops: GitOps):
        """Backward compat: no expected_main → unconditional update-ref."""
        worktree = (await git_ops.create_worktree('cas-none')).path
        (worktree / 'file.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')

        result = await git_ops.merge_to_main(worktree, 'cas-none')
        assert result.success
        assert result.merge_commit is not None

        # No expected_main — should work as before
        advanced = await git_ops.advance_main(result.merge_commit)
        assert advanced == 'advanced'

        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)


# ---------------------------------------------------------------------------
# TestMergeWorker — Phase B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckPlanTargetsInTree:
    """Unit tests for the plan-target drop-guard helper."""

    async def test_all_plan_targets_present(
        self, git_ops: GitOps,
    ):
        """Plan targets that exist in the merge commit → empty missing list."""
        worktree = (await git_ops.create_worktree('plan-all-present')).path
        (worktree / 'alpha.py').write_text('alpha = 1\n')
        (worktree / 'beta.py').write_text('beta = 2\n')
        await git_ops.commit(worktree, 'Add files')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t1', 'T1', 'desc')
        artifacts.write_plan({
            'files': ['alpha.py', 'beta.py'],
            'modules': [],
            'steps': [],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-all-present')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_plan_target_never_created_not_flagged(
        self, git_ops: GitOps,
    ):
        """Plan lists a file the task never created — not a merge drop.

        The file isn't on task HEAD either, so its absence from the merge
        commit reflects the task branch's own state, not conflict loss.
        Plan-delivery gaps (listed in plan.files, never produced by the
        task) are a different class of problem and are out of scope for
        the merge-time drop-guard; catching them belongs to review/verify.
        """
        worktree = (await git_ops.create_worktree('plan-one-missing')).path
        (worktree / 'present.py').write_text('present = 1\n')
        await git_ops.commit(worktree, 'Add present only')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t2', 'T2', 'desc')
        artifacts.write_plan({
            'files': ['present.py', 'absent.py'],
            'modules': [],
            'steps': [],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-one-missing')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_plan_target_added_then_deleted_on_branch_not_flagged(
        self, git_ops: GitOps,
    ):
        """File added then deleted on the task branch → intentional, not flagged.

        Real-world case (task/982): a plan adds a scaffold, a later reviewed
        step deletes it as an anti-pattern. The file is listed in
        plan.files, is absent from the merge commit, and is absent from
        task HEAD — that matches task intent, not conflict loss.
        """
        worktree = (
            await git_ops.create_worktree('plan-added-then-deleted')
        ).path
        (worktree / 'keep.py').write_text('keep = 1\n')
        (worktree / 'scratch.py').write_text('scratch = 1\n')
        await git_ops.commit(worktree, 'Add keep + scratch')
        (worktree / 'scratch.py').unlink()
        await git_ops.commit(worktree, 'Remove scratch per review')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t2b', 'T2b', 'desc')
        artifacts.write_plan({
            'files': ['keep.py', 'scratch.py'],
            'modules': [],
            'steps': [],
        })

        merge_result = await git_ops.merge_to_main(
            worktree, 'plan-added-then-deleted',
        )
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_plan_target_on_head_dropped_by_merge_is_flagged(
        self, git_ops: GitOps,
    ):
        """File present on task HEAD but absent from merge commit → flagged.

        Simulates the real failure mode the guard was built for: conflict
        resolution accepts origin and drops a file the task branch
        produced. We synthesise the detector input by pointing the
        `merge_commit_sha` at an earlier task-branch commit that predates
        the addition of the dropped file — it has the retained file but
        not the dropped one, matching what a bad conflict resolution would
        have produced.
        """
        worktree = (await git_ops.create_worktree('plan-dropped')).path
        (worktree / 'retained.py').write_text('retained = 1\n')
        await git_ops.commit(worktree, 'Add retained')
        rc, pre_drop_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        assert rc == 0
        pre_drop_sha = pre_drop_sha.strip()

        (worktree / 'dropped.py').write_text('dropped = 1\n')
        await git_ops.commit(worktree, 'Add dropped')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t2c', 'T2c', 'desc')
        artifacts.write_plan({
            'files': ['retained.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        # pre_drop_sha has retained.py but not dropped.py, and task HEAD
        # has both — so only dropped.py should be flagged as a merge drop.
        result = await _check_plan_targets_in_tree(
            pre_drop_sha, worktree, git_ops,
        )
        missing = result.dropped
        assert missing == ['dropped.py']

    async def test_real_conflict_resolution_drop_is_flagged(
        self, git_ops: GitOps,
    ):
        """Genuine conflict-time drop (file removed during resolution) → flagged.

        Complements ``test_plan_target_on_head_dropped_by_merge_is_flagged``
        which synthesises the detector input from a pre-drop task commit.
        Here we build an actual merge commit produced by resolving a
        real conflict via `git rm`, which is the failure mode the guard
        was originally designed to catch.
        """
        # Task branch: adds contested.py + other.py
        worktree = (await git_ops.create_worktree('real-conflict-drop')).path
        full_branch = f'{git_ops.config.branch_prefix}real-conflict-drop'
        (worktree / 'contested.py').write_text('task_version = 1\n')
        (worktree / 'other.py').write_text('other = 1\n')
        await git_ops.commit(worktree, 'Task: add contested + other')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t2d', 'T2d', 'desc')
        artifacts.write_plan({
            'files': ['contested.py', 'other.py'],
            'modules': [],
            'steps': [],
        })

        # Main: independently add contested.py with different content →
        # guaranteed conflict when task branch is merged in.
        (git_ops.project_root / 'contested.py').write_text(
            'main_version = 1\n'
        )
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: add contested'],
            cwd=git_ops.project_root,
        )

        # Manually build a merge commit that resolves the conflict by
        # dropping contested.py — emulating a human (or LLM) resolving
        # the merge by removing the contested file entirely.
        merge_wt = (
            git_ops.worktree_base / '_real-merge'
        )
        await _run(
            ['git', 'worktree', 'add', '--detach', str(merge_wt), 'main'],
            cwd=git_ops.project_root,
        )
        try:
            rc, out, err = await _run(
                ['git', 'merge', '--no-ff', '--no-commit', full_branch],
                cwd=merge_wt,
            )
            # Expect a real conflict
            assert 'CONFLICT' in out or 'CONFLICT' in err, (
                f'Expected conflict; got rc={rc} out={out!r} err={err!r}'
            )
            # Resolve by dropping contested.py entirely
            await _run(['git', 'rm', '-f', 'contested.py'], cwd=merge_wt)
            rc, _, err = await _run(
                ['git', 'commit', '--no-edit'], cwd=merge_wt,
            )
            assert rc == 0, f'merge commit failed: {err}'
            _, merge_sha, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=merge_wt,
            )
            merge_sha = merge_sha.strip()

            # Detector must flag contested.py (on task HEAD, absent from merge)
            # but leave other.py (present on both) alone.
            result = await _check_plan_targets_in_tree(
                merge_sha, worktree, git_ops,
            )
            missing = result.dropped
            assert missing == ['contested.py']
        finally:
            await _run(
                ['git', 'worktree', 'remove', str(merge_wt), '--force'],
                cwd=git_ops.project_root,
            )

    async def test_no_plan_json_returns_empty(
        self, git_ops: GitOps,
    ):
        """No plan.json (architect never ran) → empty missing list."""
        worktree = (await git_ops.create_worktree('no-plan')).path
        (worktree / 'file.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')

        # Deliberately NOT creating .task/plan.json
        merge_result = await git_ops.merge_to_main(worktree, 'no-plan')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_empty_files_list_returns_empty(
        self, git_ops: GitOps,
    ):
        """Plan exists but files=[] → empty missing list."""
        worktree = (await git_ops.create_worktree('plan-empty-files')).path
        (worktree / 'some.py').write_text('some = 1\n')
        await git_ops.commit(worktree, 'Add some')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t3', 'T3', 'desc')
        artifacts.write_plan({
            'files': [],
            'modules': [],
            'steps': [],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-empty-files')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_file_deleted_by_done_step_is_not_reported_as_dropped(
        self, git_ops: GitOps,
    ):
        """File missing from merge tree but deleted in a done step → excluded from result.

        This guards against false-positive drop reports when a TDD task's later
        steps intentionally delete files that earlier steps created.  The file
        appears in plan['files'] (it was a planned target) but it was legitimately
        removed by a done step whose commit is recorded in the plan.
        """
        worktree = (await git_ops.create_worktree('plan-deleted-done')).path

        # Step 1: create the file and commit it
        (worktree / 'created_then_deleted.py').write_text('# created\n')
        create_sha = await git_ops.commit(worktree, 'Create created_then_deleted.py')
        assert create_sha is not None

        # Step 2: delete the file and commit the deletion
        (worktree / 'created_then_deleted.py').unlink()
        delete_sha = await git_ops.commit(worktree, 'Delete created_then_deleted.py')
        assert delete_sha is not None

        # Plan still lists the file (it was a planned target), but one done
        # step deleted it — so it should NOT be reported as a drop.
        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-del', 'T-del', 'desc')
        artifacts.write_plan({
            'files': ['created_then_deleted.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'create',
                    'status': 'done',
                    'commit': create_sha,
                },
                {
                    'id': 'step-2',
                    'description': 'delete',
                    'status': 'done',
                    'commit': delete_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-deleted-done')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            # File is absent from merge tree but was intentionally deleted →
            # must NOT appear in the dropped list.
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_true_drop_still_reported_when_mixed_with_expected_deletion(
        self, git_ops: GitOps,
    ):
        """True conflict-resolution drops are still returned alongside expected deletions.

        Plan has three files:
        - kept.py:         present in the merge tree (no issue)
        - deleted.py:      absent from the merge tree, but deleted by a done step → expected_absent
        - never_created.py: absent from the merge tree with no done-step deletion → real drop

        Only never_created.py should be in the returned list.
        """
        worktree = (await git_ops.create_worktree('plan-mixed-drops')).path

        # kept.py stays in the tree
        (worktree / 'kept.py').write_text('kept = 1\n')
        # deleted.py will be created and then deleted
        (worktree / 'deleted.py').write_text('deleted = 1\n')
        await git_ops.commit(worktree, 'Add kept.py and deleted.py')

        # Deletion commit — only deleted.py is removed
        (worktree / 'deleted.py').unlink()
        delete_sha = await git_ops.commit(worktree, 'Delete deleted.py')
        assert delete_sha is not None

        # never_created.py is never committed — simulates a conflict-resolution drop

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-mixed', 'T-mixed', 'desc')
        artifacts.write_plan({
            'files': ['kept.py', 'deleted.py', 'never_created.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'delete deleted.py',
                    'status': 'done',
                    'commit': delete_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-mixed-drops')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            # kept.py is present, deleted.py is expected_absent, only the
            # true drop remains.
            assert missing == ['never_created.py']
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_pending_step_deletion_is_not_trusted_as_expected_absent(
        self, git_ops: GitOps,
    ):
        """A file deleted in a *pending* step's commit is NOT treated as expected_absent.

        Only done steps' deletions are trusted.  If the step that deleted the file
        has status='pending', the file's absence is still treated as a real drop.
        """
        worktree = (await git_ops.create_worktree('plan-pending-del')).path

        (worktree / 'would_delete.py').write_text('# would be deleted\n')
        create_sha = await git_ops.commit(worktree, 'Add would_delete.py')
        assert create_sha is not None

        (worktree / 'would_delete.py').unlink()
        delete_sha = await git_ops.commit(worktree, 'Delete would_delete.py')
        assert delete_sha is not None

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-pend', 'T-pend', 'desc')
        # Step that performed the deletion has status='pending' — must NOT be trusted
        artifacts.write_plan({
            'files': ['would_delete.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'delete would_delete.py',
                    'status': 'pending',   # ← not done
                    'commit': delete_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-pending-del')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            # pending step → deletion not trusted → file is a real drop
            assert missing == ['would_delete.py']
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_done_step_with_null_commit_is_not_trusted_as_expected_absent(
        self, git_ops: GitOps,
    ):
        """A done step with commit=None is skipped — its deletions are not trusted.

        If a step is marked done but has no recorded commit SHA (e.g. it was
        marked done before the commit was recorded), the absence of its planned
        files is still treated as a real drop.
        """
        worktree = (await git_ops.create_worktree('plan-null-commit')).path

        (worktree / 'would_delete.py').write_text('# would be deleted\n')
        await git_ops.commit(worktree, 'Add would_delete.py')

        (worktree / 'would_delete.py').unlink()
        await git_ops.commit(worktree, 'Delete would_delete.py')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-null', 'T-null', 'desc')
        # Step is done but commit is None — must NOT be trusted
        artifacts.write_plan({
            'files': ['would_delete.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'delete would_delete.py',
                    'status': 'done',
                    'commit': None,   # ← no commit recorded
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-null-commit')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            # done but commit=None → deletion not trusted → file is a real drop
            assert missing == ['would_delete.py']
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_git_log_error_logs_warning_and_treats_file_as_dropped(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        """git log non-zero rc → warning logged, file remains in dropped list (fail-closed).

        A done step with a non-existent/bad commit SHA causes the git log call to fail.
        The implementation must (a) log a WARNING mentioning the step/commit, and (b)
        conservatively leave the file as a real drop (not mask it as expected_absent).
        """
        worktree = (await git_ops.create_worktree('plan-bad-sha')).path

        # gone.py is never committed to the worktree → it will be missing from
        # the merge tree (simulates a conflict-resolution drop).
        # We need at least one file so merge_to_main succeeds
        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        bad_sha = '0' * 40  # plausible SHA shape but non-existent

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-bad-sha', 'T-bad-sha', 'desc')
        artifacts.write_plan({
            'files': ['gone.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'supposedly deleted gone.py',
                    'status': 'done',
                    'commit': bad_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-bad-sha')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                result = await _check_plan_targets_in_tree(
                    merge_result.merge_commit, worktree, git_ops,
                )
                missing = result.dropped
            # (a) git log fails → file stays as dropped (fail-closed)
            assert missing == ['gone.py']
            # (b) a warning was emitted mentioning the bad commit
            warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert warning_records, 'Expected at least one WARNING log record'
            assert any(bad_sha[:12] in r.getMessage() for r in warning_records), (
                f'Expected WARNING mentioning the truncated SHA {bad_sha[:12]!r}; '
                f'got: {[r.getMessage() for r in warning_records]}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_diff_tree_failure_records_unresolved_step_with_object_missing_flag(
        self, git_ops: GitOps,
    ):
        """diff-tree rc != 0 → UnresolvedStep recorded with object_missing=True.

        When a done-step commit SHA is non-existent (e.g. pruned after amend),
        `git diff-tree` returns rc != 0.  The implementation must:
          (a) keep the file in dropped (fail-closed contract unchanged),
          (b) append exactly one UnresolvedStep to result.unresolved_steps,
          (c) set object_missing=True because the commit object is absent.

        A paired sub-case confirms that a real commit producing an empty diff
        (rc == 0) results in unresolved_steps == [] — object_missing is only
        set when diff-tree itself fails and cat-file confirms absence.
        """
        from orchestrator.merge_queue import UnresolvedStep

        worktree = (await git_ops.create_worktree('unresolved-step-flag')).path

        # anchor.py is needed so merge_to_main has something to merge
        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        bad_sha = '0' * 40  # non-existent object

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-unresolved', 'T-unresolved', 'desc')
        artifacts.write_plan({
            'files': ['gone.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'supposedly deleted gone.py',
                    'status': 'done',
                    'commit': bad_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'unresolved-step-flag')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                task_id='t-unresolved',
            )
            # (a) fail-closed: file stays in dropped
            assert result.dropped == ['gone.py'], (
                f'Expected fail-closed dropped=[gone.py], got {result.dropped!r}'
            )
            # (b) exactly one UnresolvedStep recorded
            assert len(result.unresolved_steps) == 1, (
                f'Expected 1 unresolved step, got {len(result.unresolved_steps)}: '
                f'{result.unresolved_steps!r}'
            )
            us = result.unresolved_steps[0]
            assert isinstance(us, UnresolvedStep)
            assert us.step_idx == 0, f'Expected step_idx=0, got {us.step_idx}'
            assert us.step_id == 'step-1', f'Expected step_id=step-1, got {us.step_id!r}'
            assert us.commit == bad_sha, f'Expected commit={bad_sha!r}, got {us.commit!r}'
            assert us.rc != 0, f'Expected rc != 0 (diff-tree must have failed), got {us.rc}'
            assert us.stderr, f'Expected non-empty stderr, got {us.stderr!r}'
            # (c) object_missing=True because bad_sha is absent from ODB
            assert us.object_missing is True, (
                f'Expected object_missing=True for non-existent SHA, got {us.object_missing!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_real_commit_no_diff_produces_no_unresolved_steps(
        self, git_ops: GitOps,
    ):
        """Genuinely empty-diff commit (--allow-empty) → diff-tree rc=0 with empty stdout.

        Tests the boundary case: a done-step commit that touches nothing
        (``git commit --allow-empty``).  ``git diff-tree --diff-filter=D``
        returns rc=0 with empty stdout — the ``for line in stdout.splitlines()``
        loop is a no-op, ``expected_absent`` is unchanged, and no ``UnresolvedStep``
        is created.

        This was previously untested.  The deletion case that this slot formerly
        exercised is already covered by
        ``test_check_plan_targets_returns_drop_guard_result`` and
        ``test_orphan_done_step_commit_object_in_odb_resolves_deletion_as_expected_absent``,
        so repurposing this slot gives coverage of the rc=0-with-empty-stdout
        edge case without losing anything.

        No ``merge_queue.py`` change is needed — the implementation already
        handles empty stdout correctly.
        """
        worktree = (await git_ops.create_worktree('no-unresolved-empty-diff')).path

        # present.py will be present in the merge tree → no drops
        (worktree / 'present.py').write_text('p = 1\n')
        await git_ops.commit(worktree, 'Add present.py')

        # TRUE empty commit (--allow-empty): diff-tree returns rc=0 with no output
        await _run(['git', 'commit', '--allow-empty', '-m', 'empty step'], cwd=worktree)
        rc, empty_sha_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc == 0
        empty_sha = empty_sha_out.strip()

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-empty-diff', 'T-empty-diff', 'desc')
        artifacts.write_plan({
            'files': ['present.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'empty-diff commit (allow-empty)',
                    'status': 'done',
                    'commit': empty_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'no-unresolved-empty-diff')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                task_id='t-empty-diff',
            )
            # diff-tree rc=0 with empty stdout → no unresolved steps
            assert result.unresolved_steps == [], (
                f'Expected no unresolved steps (empty-diff commit, rc=0), '
                f'got {result.unresolved_steps!r}'
            )
            # present.py is in the merge tree → not flagged as dropped
            assert result.dropped == [], (
                f'Expected no drops (present.py in merge tree), '
                f'got {result.dropped!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_merge_commit_done_step_deletions_are_detected(
        self, git_ops: GitOps,
    ):
        """Merge-commit done step: deletions are not silenced by combined-diff.

        Gap 2: git show defaults to combined-diff format for merge commits,
        which prints nothing under --diff-filter=D.  diff-tree -r <sha>^ <sha>
        forces two-way comparison against the first parent, correctly surfacing
        file deletions that happen to occur at a merge-commit boundary.
        """
        worktree = (await git_ops.create_worktree('merge-commit-del')).path

        # Add f.py on the task branch and commit
        (worktree / 'f.py').write_text('f = 1\n')
        create_sha = await git_ops.commit(worktree, 'Add f.py')
        assert create_sha

        # Create a sidekick branch from HEAD~1 (the initial commit, before f.py)
        # so that the merge produces a genuine two-parent commit.
        await _run(['git', 'branch', 'sidekick', 'HEAD~1'], cwd=worktree)
        await _run(['git', 'checkout', 'sidekick'], cwd=worktree)

        # Add g.py on the sidekick branch and commit
        (worktree / 'g.py').write_text('g = 1\n')
        await _run(['git', 'add', '-A'], cwd=worktree)
        await _run(['git', 'commit', '-m', 'Add g.py'], cwd=worktree)

        # Return to the task branch
        await _run(['git', 'checkout', '-'], cwd=worktree)

        # Start the merge without committing, then also delete f.py before
        # finalising the merge commit.
        await _run(['git', 'merge', '--no-ff', '--no-commit', 'sidekick'], cwd=worktree)
        await _run(['git', 'rm', 'f.py'], cwd=worktree)
        await _run(
            ['git', 'commit', '-m', 'Merge sidekick and delete f.py'],
            cwd=worktree,
        )

        # Capture the merge-commit SHA in the task branch
        _, merge_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-merge-del', 'T-merge-del', 'desc')
        artifacts.write_plan({
            'files': ['f.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'merge+del',
                    'status': 'done',
                    'commit': merge_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'merge-commit-del')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            # f.py was intentionally deleted inside the done step's merge commit.
            # It must NOT be reported as a drop.
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_renamed_file_in_done_step_treats_old_name_as_expected_absent(
        self, git_ops: GitOps,
    ):
        """Renamed file in done step: old name is treated as expected absent.

        Gap 3: git's default rename detection converts D+A pairs to R.  A
        planned file renamed by a done step does not appear in the D list, so
        the old name is falsely flagged as a drop.  --no-renames disables
        detection so the rename surfaces as an explicit deletion of a.py,
        placing it in expected_absent.
        """
        worktree = (await git_ops.create_worktree('rename-step')).path

        # High-similarity content (5× repeated function body) ensures git's
        # default rename detection (50% similarity threshold) fires.
        (worktree / 'a.py').write_text('def helper():\n    return 42\n' * 5)
        await git_ops.commit(worktree, 'Add a.py')

        # Rename a.py → b.py via git mv (preserves exact content → 100% similarity)
        await _run(['git', 'mv', 'a.py', 'b.py'], cwd=worktree)
        rename_sha = await git_ops.commit(worktree, 'Rename a.py to b.py')
        assert rename_sha

        # Plan lists the OLD name; the rename is recorded as a done step.
        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-rename', 'T-rename', 'desc')
        artifacts.write_plan({
            'files': ['a.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'rename',
                    'status': 'done',
                    'commit': rename_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'rename-step')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            missing = result.dropped
            # a.py was legitimately renamed to b.py in the done step.
            # It must NOT be reported as a drop.
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_done_step_diff_tree_calls_run_concurrently(
        self, git_ops: GitOps,
    ):
        """diff-tree queries for done steps run concurrently via asyncio.gather.

        Performance concern: the merge hot-path's wall-clock latency must be
        O(1) not O(N) in done-step count — total subprocess work stays O(N),
        but asyncio.gather runs them concurrently so latency tracks the
        slowest single call.

        Uses ``asyncio.Barrier(3)`` as a deterministic synchronization point:
        each intercepted coroutine waits at the barrier, which only releases
        once all three parties arrive.  Under the concurrent ``asyncio.gather``
        path all three coroutines enter and the barrier releases immediately.
        A sequential regression would block at the first call — the second
        and third parties never arrive — so the surrounding ``asyncio.wait_for``
        raises ``TimeoutError`` and fails the test with a clear signal.  No
        scheduler-timing slack (``asyncio.sleep`` + wall-clock assertion) is
        involved, so the test is immune to loaded-runner jitter.

        Each done step references a DISTINCT commit so the deduplication pass
        does not collapse the three queries into one.
        """
        worktree = (await git_ops.create_worktree('concurrent-done')).path

        # Create three distinct commits so deduplication keeps all three queries.
        (worktree / 'file_a.py').write_text('a = 1\n')
        sha_c1 = await git_ops.commit(worktree, 'Add file_a.py')
        assert sha_c1

        (worktree / 'file_b.py').write_text('b = 1\n')
        sha_c2 = await git_ops.commit(worktree, 'Add file_b.py')
        assert sha_c2

        (worktree / 'file_c.py').write_text('c = 1\n')
        sha_c3 = await git_ops.commit(worktree, 'Add file_c.py')
        assert sha_c3

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-concurrent', 'T-concurrent', 'desc')
        # forces_loop.py is a phantom file: it is absent from the merge tree,
        # so missing is non-empty and the done-step loop is entered.
        artifacts.write_plan({
            'files': ['file_a.py', 'forces_loop.py'],
            'modules': [],
            'steps': [
                {'id': 'step-1', 'description': 's1', 'status': 'done', 'commit': sha_c1},
                {'id': 'step-2', 'description': 's2', 'status': 'done', 'commit': sha_c2},
                {'id': 'step-3', 'description': 's3', 'status': 'done', 'commit': sha_c3},
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'concurrent-done')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            from orchestrator import merge_queue as mq
            original_run = mq._run
            # Barrier(3) releases only once all three intercepted coroutines
            # have arrived — i.e. only under concurrent execution.  Sequential
            # execution leaves the first caller blocked indefinitely, caught
            # below by asyncio.wait_for.
            barrier = asyncio.Barrier(3)
            call_count = 0

            async def _intercept(cmd, cwd=None, **kwargs):
                nonlocal call_count
                if '--diff-filter=D' in cmd:
                    call_count += 1
                    await barrier.wait()
                    return 0, '', ''
                return await original_run(cmd, cwd=cwd, **kwargs)

            with patch('orchestrator.merge_queue._run', new=_intercept):
                # Tight timeout: concurrent execution releases the barrier
                # effectively instantly; any sequential regression will hang
                # until this fires.  1 s is ample for CI scheduler latency
                # (barrier releases in microseconds under asyncio.gather);
                # failing fast avoids a 5 s dead wait on a real regression.
                # The except block re-raises as AssertionError so the failure
                # message is self-describing rather than a bare TimeoutError.
                try:
                    await asyncio.wait_for(
                        _check_plan_targets_in_tree(
                            merge_result.merge_commit, worktree, git_ops,
                        ),
                        timeout=1.0,
                    )
                except TimeoutError:
                    raise AssertionError(
                        'Sequential execution detected: barrier never '
                        'released within 1 s — asyncio.gather concurrency '
                        'regression in _check_plan_targets_in_tree'
                    ) from None

            assert call_count == 3, (
                f'Expected 3 diff-tree calls, got {call_count}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_malformed_step_shapes_do_not_raise_or_query_git(
        self, git_ops: GitOps,
    ):
        """Malformed step shapes are silently skipped — no exceptions, no git queries.

        The defensive ``isinstance(step, dict)`` and ``isinstance(commit, str)``
        guards in the asyncio.gather refactor had no previous test coverage.
        This test feeds a plan whose steps list contains:
          - None (non-dict)
          - a plain string (non-dict, normalised to a dict with no "status")
          - a dict whose "commit" key is absent
          - a dict whose "commit" is None (non-string)
          - a dict whose "commit" is an integer (non-string)
          - a dict with a valid string commit but status != 'done'

        Assertions:
          1. No exception is raised.
          2. No ``git diff-tree`` subprocess is fired (steps_to_query is empty).
          3. The phantom file that is absent from the merge tree is still
             reported as dropped (the guards don't suppress legitimate drops).
        """
        worktree = (await git_ops.create_worktree('malformed-steps')).path
        (worktree / 'real.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add real.py')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-malformed', 'T-malformed', 'desc')
        artifacts.write_plan({
            'files': ['phantom.py', 'real.py'],
            'modules': [],
            'steps': [
                None,                                                   # non-dict
                'plain string step',                                    # non-dict
                {'id': 'step-A', 'description': 'a', 'status': 'done'},  # no commit
                {'id': 'step-B', 'description': 'b', 'status': 'done', 'commit': None},
                {'id': 'step-C', 'description': 'c', 'status': 'done', 'commit': 123},
                {'id': 'step-D', 'description': 'd', 'status': 'pending', 'commit': 'abc'},
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'malformed-steps')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            from orchestrator import merge_queue as mq
            original_run = mq._run
            diff_tree_calls: list[list[str]] = []

            async def _intercept(cmd, cwd=None, **kwargs):
                if '--diff-filter=D' in cmd:
                    diff_tree_calls.append(list(cmd))
                return await original_run(cmd, cwd=cwd, **kwargs)

            with patch('orchestrator.merge_queue._run', new=_intercept):
                result = await _check_plan_targets_in_tree(
                    merge_result.merge_commit, worktree, git_ops,
                )
                missing = result.dropped

            # phantom.py is absent from the merge tree → correctly reported as dropped
            assert 'phantom.py' in missing
            # real.py is present in the merge tree → not reported as dropped
            assert 'real.py' not in missing
            # No diff-tree subprocess should have been launched for any of the
            # malformed / non-done steps — steps_to_query must be empty.
            assert diff_tree_calls == [], (
                f'Unexpected diff-tree calls fired: {diff_tree_calls}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_orphan_done_step_commit_object_in_odb_resolves_deletion_as_expected_absent(
        self, git_ops: GitOps,
    ):
        """Orphaned commit (amend-discarded) whose object survives in ODB → not a failure.

        Regression test for the failure mode discovered in task 1058:
        after `git commit --amend`, the pre-amend SHA is orphaned but
        remains queryable via the shared ODB until pruning.  If the plan
        records the pre-amend SHA as a done-step commit, `git diff-tree`
        can still resolve its deletion — so F.py should appear in
        expected_absent and NOT be flagged as a drop.

        The test documents and pins the contract:
            orphan-but-not-pruned == happy path; no false positive.
        """
        worktree = (await git_ops.create_worktree('orphan-odb-test')).path

        # anchor.py keeps the branch non-empty after F.py is deleted
        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        # Add F.py then delete it, capturing the SHA of the deletion commit
        (worktree / 'F.py').write_text('f = 1\n')
        await git_ops.commit(worktree, 'Add F.py')
        (worktree / 'F.py').unlink()
        await _run(['git', 'add', '-A'], cwd=worktree)
        await _run(['git', 'commit', '-m', 'Delete F.py'], cwd=worktree)
        rc, sha_del_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc == 0
        sha_del = sha_del_out.strip()

        # Amend to orphan sha_del — stage an additional file to ensure the tree
        # changes and git produces a genuinely different commit SHA.
        (worktree / 'amend_marker.py').write_text('# amend marker\n')
        await _run(['git', 'add', 'amend_marker.py'], cwd=worktree)
        await _run(['git', 'commit', '--amend', '--no-edit'], cwd=worktree)
        rc2, sha_del_prime_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc2 == 0
        sha_del_prime = sha_del_prime_out.strip()
        assert sha_del_prime != sha_del, 'amend must produce a new SHA'

        # (a) Verify sha_del is still reachable via ODB despite being orphaned
        rc_cat, cat_out, _ = await _run(
            ['git', 'cat-file', '-t', sha_del],
            cwd=git_ops.project_root,
        )
        assert rc_cat == 0, f'git cat-file failed (rc={rc_cat}): orphan left ODB?'
        assert cat_out.strip() == 'commit', (
            f'Expected "commit" type for sha_del, got {cat_out.strip()!r}'
        )

        # Write plan recording sha_del (orphaned) as the done-step commit
        artifacts = TaskArtifacts(worktree)
        artifacts.init('orphan-test', 'Orphan test', 'desc')
        artifacts.write_plan({
            'files': ['F.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'Delete F.py',
                    'status': 'done',
                    'commit': sha_del,
                },
            ],
        })

        # Merge: the merge commit will NOT contain F.py (deleted on branch)
        merge_result = await git_ops.merge_to_main(worktree, 'orphan-odb-test')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            # (b) Drop-guard must NOT flag F.py as dropped — sha_del's
            # deletion is still resolvable via the shared ODB
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                task_id='orphan-test',
            )
            missing = result.dropped
            assert missing == [], (
                f'Expected no drops (orphan sha_del deletion resolved from ODB), '
                f'got: {missing!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_pruned_orphan_commit_produces_unresolved_step_with_object_missing(
        self, git_ops: GitOps,
    ):
        """After git gc --prune=now a pruned orphan produces UnresolvedStep(object_missing=True).

        Companion to test_orphan_done_step_commit_object_in_odb_resolves_deletion_as_expected_absent
        which pins the happy path (orphan still in ODB → deletion resolved, no false positive).

        This test covers the actual failure scenario:
        - Create a commit that deletes F.py (sha_del)
        - Orphan it via ``git commit --amend``
        - Run ``git gc --prune=now`` to evict the orphan from the ODB
        - Verify ``_check_plan_targets_in_tree`` correctly records an UnresolvedStep
          with ``object_missing=True`` and non-zero ``cat_file_rc``

        If GC does not prune the orphan in the test environment (very unlikely for
        objects with no reflog protection and ``--prune=now``) the test is skipped
        so it does not produce a spurious failure on exotic git configurations.
        """
        worktree = (await git_ops.create_worktree('gc-prune-test')).path

        # anchor.py keeps the branch non-empty after F.py is deleted
        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        # Create then delete F.py — capture the deletion commit SHA (sha_del)
        (worktree / 'F.py').write_text('f = 1\n')
        await git_ops.commit(worktree, 'Add F.py')
        (worktree / 'F.py').unlink()
        await _run(['git', 'add', '-A'], cwd=worktree)
        await _run(['git', 'commit', '-m', 'Delete F.py'], cwd=worktree)
        rc, sha_del_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc == 0
        sha_del = sha_del_out.strip()

        # Amend to orphan sha_del — stage an extra file so the tree changes and
        # git produces a genuinely different SHA (not a no-op amend).
        (worktree / 'amend_marker.py').write_text('# amend marker\n')
        await _run(['git', 'add', 'amend_marker.py'], cwd=worktree)
        await _run(['git', 'commit', '--amend', '--no-edit'], cwd=worktree)
        rc2, sha_del_prime_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc2 == 0
        assert sha_del_prime_out.strip() != sha_del, 'amend must produce a new SHA'

        # Expire all reflog entries then prune — this is what makes the orphan
        # truly unreachable and eligible for GC collection.
        await _run(
            ['git', 'reflog', 'expire', '--expire=now', '--all'],
            cwd=git_ops.project_root,
        )
        await _run(['git', 'gc', '--prune=now'], cwd=git_ops.project_root)

        # Verify sha_del is now absent from the ODB; if GC kept it (unusual
        # config), skip rather than fail — the happy-path test covers that case.
        rc_cat, _, _ = await _run(
            ['git', 'cat-file', '-t', sha_del],
            cwd=git_ops.project_root,
        )
        if rc_cat == 0:
            pytest.skip(
                'git gc --prune=now did not prune the orphan; '
                'skipping (happy-path covered by test_orphan_done_step_commit_object_in_odb_resolves_deletion_as_expected_absent)'
            )

        # Plan records sha_del (now pruned) as the done-step commit
        artifacts = TaskArtifacts(worktree)
        artifacts.init('gc-prune', 'GC prune', 'desc')
        artifacts.write_plan({
            'files': ['F.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'Delete F.py (orphaned, now pruned)',
                    'status': 'done',
                    'commit': sha_del,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'gc-prune-test')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                task_id='gc-prune',
            )
            # F.py is in dropped (fail-closed — orphan pruned, deletion evidence lost)
            assert 'F.py' in result.dropped, (
                f'Expected F.py in dropped (pruned orphan), got {result.dropped!r}'
            )
            # Exactly one UnresolvedStep
            assert len(result.unresolved_steps) == 1, (
                f'Expected 1 unresolved step, got {len(result.unresolved_steps)}: '
                f'{result.unresolved_steps!r}'
            )
            us = result.unresolved_steps[0]
            assert us.step_id == 'step-1', f'Expected step_id=step-1, got {us.step_id!r}'
            assert us.commit == sha_del, f'Expected commit={sha_del!r}, got {us.commit!r}'
            assert us.rc != 0, f'Expected diff-tree rc != 0 for pruned object, got {us.rc}'
            # object_missing=True: cat-file confirmed the object is gone
            assert us.object_missing is True, (
                f'Expected object_missing=True for pruned commit, got {us.object_missing!r}'
            )
            # cat_file_rc populated: non-zero because the object is absent
            assert us.cat_file_rc != 0, (
                f'Expected cat_file_rc != 0 for pruned commit, got {us.cat_file_rc!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_check_plan_targets_returns_drop_guard_result_dataclass(
        self, git_ops: GitOps,
    ):
        """_check_plan_targets_in_tree returns a DropGuardResult dataclass.

        Asserts the structured return type introduced by task 1068:
        - return value is a DropGuardResult instance
        - has .dropped (list[str]) and .unresolved_steps (list[UnresolvedStep])
        - .unresolved_steps is empty when all done-step diff-tree queries succeed

        Will fail until step 4 introduces the DropGuardResult return type.
        """
        from orchestrator.merge_queue import DropGuardResult

        worktree = (await git_ops.create_worktree('drop-guard-result')).path
        (worktree / 'kept.py').write_text('kept = 1\n')
        await git_ops.commit(worktree, 'Add kept.py')

        # Create then delete extra.py so we have a done step with a real deletion
        (worktree / 'extra.py').write_text('extra = 1\n')
        await git_ops.commit(worktree, 'Add extra.py')
        (worktree / 'extra.py').unlink()
        await _run(['git', 'add', '-A'], cwd=worktree)
        await _run(['git', 'commit', '-m', 'Delete extra.py'], cwd=worktree)
        rc, del_sha_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc == 0
        del_sha = del_sha_out.strip()

        artifacts = TaskArtifacts(worktree)
        artifacts.init('dgr-test', 'DGR test', 'desc')
        artifacts.write_plan({
            'files': ['kept.py', 'extra.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'Delete extra.py',
                    'status': 'done',
                    'commit': del_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'drop-guard-result')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            # Must be a DropGuardResult, not a plain list
            assert isinstance(result, DropGuardResult), (
                f'Expected DropGuardResult, got {type(result).__name__}'
            )
            # extra.py was intentionally deleted by done step → not dropped
            assert result.dropped == [], f'Unexpected drops: {result.dropped}'
            # All diff-tree queries succeeded → no unresolved steps
            assert result.unresolved_steps == [], (
                f'Unexpected unresolved steps: {result.unresolved_steps}'
            )
            # Verify type hints: unresolved_steps elements would be UnresolvedStep
            assert isinstance(result.unresolved_steps, list)
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_check_plan_targets_emits_structured_warning_when_dropped_non_empty(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        """Structured WARNING is emitted when dropped is non-empty, silent when empty.

        Step 12 contract:
        - When result.dropped is non-empty, exactly one new WARNING record is emitted
          whose message includes: task_id, merge_commit_sha, the dropped file list,
          AND per-step diagnostic info for ALL queried done steps (both successful
          and failed), not just the failing ones.
        - When result.dropped is empty (all planned files present), no new WARNING
          records are emitted — the structured warning is gated on drops.

        Setup: two done steps (one real SHA that deletes file1.py, one bad SHA that
        supposedly deleted file2.py) + file3.py as a real drop (present on task HEAD
        but absent from the merge commit).  This gives:
          dropped = ['file2.py', 'file3.py'] (fail-closed)
          unresolved_steps = [UnresolvedStep for the bad SHA]
        The WARNING must mention diagnostics for BOTH done steps.

        Will fail until step 12 adds the always-on structured warning.
        """
        worktree = (await git_ops.create_worktree('struct-warn-test')).path

        # file3.py is the real drop: present on task HEAD, will be absent from merge
        # (it is never committed to the worktree — simulates a conflict-resolution drop)
        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        # good_step: real commit that deletes file1.py
        (worktree / 'file1.py').write_text('f1 = 1\n')
        await git_ops.commit(worktree, 'Add file1.py')
        (worktree / 'file1.py').unlink()
        await _run(['git', 'add', '-A'], cwd=worktree)
        await _run(['git', 'commit', '-m', 'Delete file1.py'], cwd=worktree)
        rc, good_sha_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc == 0
        good_sha = good_sha_out.strip()

        bad_sha = '0' * 40  # non-existent object

        artifacts = TaskArtifacts(worktree)
        artifacts.init('struct-warn', 'Struct warn', 'desc')
        artifacts.write_plan({
            'files': ['file1.py', 'file2.py', 'file3.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-good',
                    'description': 'deleted file1.py (real commit)',
                    'status': 'done',
                    'commit': good_sha,
                },
                {
                    'id': 'step-bad',
                    'description': 'supposedly deleted file2.py (bad SHA)',
                    'status': 'done',
                    'commit': bad_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'struct-warn-test')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            merge_sha = merge_result.merge_commit

            # ── Sub-case 1: dropped is non-empty → structured WARNING ──────────
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                result = await _check_plan_targets_in_tree(
                    merge_sha, worktree, git_ops,
                    task_id='1068-test',
                )

            # file1.py is expected_absent (good_step deleted it); file2.py and
            # file3.py are dropped (fail-closed on bad_sha, real drop on file3.py)
            assert set(result.dropped) == {'file2.py', 'file3.py'}, (
                f'Unexpected dropped: {result.dropped!r}'
            )

            warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert warn_records, 'Expected at least one WARNING when dropped is non-empty'

            # Find the structured-warning record (may coexist with per-step warnings)
            all_messages = ' '.join(r.getMessage() for r in warn_records)
            assert '1068-test' in all_messages, (
                f'Expected task_id "1068-test" in WARNING; got: {all_messages!r}'
            )
            assert merge_sha in all_messages or merge_sha[:12] in all_messages, (
                f'Expected merge_commit_sha in WARNING; got: {all_messages!r}'
            )
            # Dropped file list
            assert 'file2.py' in all_messages or 'file3.py' in all_messages, (
                f'Expected dropped files in WARNING; got: {all_messages!r}'
            )
            # Per-step diagnostics for the good step (rc=0, diff-tree succeeded)
            assert good_sha[:12] in all_messages, (
                f'Expected good_sha[:12] in WARNING step_diagnostics; '
                f'got: {all_messages!r}'
            )
            # Per-step diagnostics for the bad step (rc!=0, object_missing=True)
            assert bad_sha[:12] in all_messages, (
                f'Expected bad_sha[:12] in WARNING step_diagnostics; '
                f'got: {all_messages!r}'
            )

            # ── Sub-case 2: dropped is empty → no new WARNING ────────────────
            # Build a plan where all files are present in the merge tree.
            worktree2 = (await git_ops.create_worktree('struct-warn-empty')).path
            (worktree2 / 'present.py').write_text('p = 1\n')
            await git_ops.commit(worktree2, 'Add present.py')
            artifacts2 = TaskArtifacts(worktree2)
            artifacts2.init('sw-empty', 'SW empty', 'desc')
            artifacts2.write_plan({
                'files': ['present.py'],
                'modules': [],
                'steps': [],
            })
            merge_result2 = await git_ops.merge_to_main(worktree2, 'struct-warn-empty')
            assert merge_result2.success
            assert merge_result2.merge_commit is not None
            try:
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                    result2 = await _check_plan_targets_in_tree(
                        merge_result2.merge_commit, worktree2, git_ops,
                        task_id='1068-test-empty',
                    )
                assert result2.dropped == [], (
                    f'Expected empty dropped for present plan, got {result2.dropped!r}'
                )
                new_warn = [r for r in caplog.records if r.levelno >= logging.WARNING]
                assert not new_warn, (
                    f'Expected no WARNING when dropped is empty; '
                    f'got: {[r.getMessage() for r in new_warn]!r}'
                )
            finally:
                if merge_result2.merge_worktree:
                    await git_ops.cleanup_merge_worktree(merge_result2.merge_worktree)
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_step_diagnostics_emitted_in_plan_step_order(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        """step_diagnostics in the structured WARNING must appear in plan-step order.

        Current bug: success diagnostics are appended inline (in plan-step
        iteration order) and failure diagnostics are appended after the
        concurrent cat-file gather.  With step_idx=0 failing and step_idx=1
        succeeding, the rendered list is ``[(1, ...), (0, ...)]`` rather than
        plan order.

        Setup:
          - step_idx=0 has ``bad_sha = '0'*40`` (diff-tree rc!=0 → failure)
          - step_idx=1 has a real deletion commit for file_b.py (rc==0 → success)
          - file_a.py is never committed → absent from merge tree → real drop
          - file_b.py is intentionally deleted by step_idx=1

        After the fix, ``step_diagnostics`` is sorted by step_idx before the
        structured WARNING fires.  Assertion: ``'(0,'`` appears before ``'(1,'``
        in the rendered WARNING message.  Currently fails because the
        in-line/failures split produces the reverse order.
        """
        worktree = (await git_ops.create_worktree('issue2-diag-order')).path

        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        # step_idx=1: real deletion commit for file_b.py
        (worktree / 'file_b.py').write_text('b = 1\n')
        await git_ops.commit(worktree, 'Add file_b.py')
        (worktree / 'file_b.py').unlink()
        await _run(['git', 'add', '-A'], cwd=worktree)
        await _run(['git', 'commit', '-m', 'Delete file_b.py'], cwd=worktree)
        rc, good_sha_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        assert rc == 0
        good_sha = good_sha_out.strip()

        bad_sha = '0' * 40  # non-existent → diff-tree rc != 0

        # file_a.py never committed → real drop from merge tree
        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-issue2', 'T-issue2', 'desc')
        artifacts.write_plan({
            'files': ['file_a.py', 'file_b.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-0',
                    'description': 'supposedly deleted file_a.py (bad SHA)',
                    'status': 'done',
                    'commit': bad_sha,          # step_idx=0, failure
                },
                {
                    'id': 'step-1',
                    'description': 'deleted file_b.py (real commit)',
                    'status': 'done',
                    'commit': good_sha,         # step_idx=1, success
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'issue2-diag-order')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                result = await _check_plan_targets_in_tree(
                    merge_result.merge_commit, worktree, git_ops,
                    task_id='t-issue2',
                )

            # file_a.py is a real drop (bad_sha fail-closed); file_b.py is expected absent
            assert 'file_a.py' in result.dropped, (
                f'Expected file_a.py in dropped; got {result.dropped!r}'
            )

            # Find the structured-warning record
            warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert warn_records, 'Expected at least one WARNING when dropped is non-empty'
            all_messages = ' '.join(r.getMessage() for r in warn_records)

            # Both step indices must appear in the WARNING
            assert '(0,' in all_messages, f'"(0," not found in WARNING: {all_messages!r}'
            assert '(1,' in all_messages, f'"(1," not found in WARNING: {all_messages!r}'

            # Plan-step order: (0, ...) must precede (1, ...)
            pos_0 = all_messages.index('(0,')
            pos_1 = all_messages.index('(1,')
            assert pos_0 < pos_1, (
                f'Expected step_diagnostics in plan order (0 before 1), '
                f'but (0, appears at {pos_0}, (1, appears at {pos_1}): {all_messages!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_per_step_diff_tree_failure_log_demoted_to_debug(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        """Per-step diff-tree failure message must be at DEBUG, not WARNING.

        After Issue 1 fix, ``logger.warning('git diff-tree --diff-filter=D
        failed for step ...')`` must be demoted to ``logger.debug(...)``.
        The structured WARNING (``drop-guard: dropped_plan_targets``) already
        embeds per-step diagnostics via ``step_diagnostics``, so the per-step
        record at WARNING is redundant and noisy for operators.

        Assertions:
          (a) At least one DEBUG record contains the per-step failure substring
              'git diff-tree --diff-filter=D failed for step'.
          (b) No WARNING record contains that same per-step substring
              (only the structured drop-guard WARNING should fire at WARNING+).
          (c) The structured WARNING message contains ``bad_sha[:12]`` (confirming
              the truncated SHA is still visible via step_diagnostics).
        """
        worktree = (await git_ops.create_worktree('issue1-log-level')).path

        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        bad_sha = '0' * 40  # non-existent object → diff-tree rc != 0

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-issue1', 'T-issue1', 'desc')
        artifacts.write_plan({
            'files': ['gone.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'supposedly deleted gone.py',
                    'status': 'done',
                    'commit': bad_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'issue1-log-level')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            with caplog.at_level(logging.DEBUG, logger='orchestrator.merge_queue'):
                await _check_plan_targets_in_tree(
                    merge_result.merge_commit, worktree, git_ops,
                    task_id='t-issue1',
                )

            per_step_msg = 'git diff-tree --diff-filter=D failed for step'

            # (a) Per-step failure message must appear at DEBUG level
            debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
            assert any(per_step_msg in r.getMessage() for r in debug_records), (
                f'Expected a DEBUG record containing {per_step_msg!r}; '
                f'got DEBUG records: {[r.getMessage() for r in debug_records]!r}'
            )

            # (b) Per-step failure message must NOT appear at WARNING level
            warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert not any(per_step_msg in r.getMessage() for r in warning_records), (
                f'Expected per-step message absent from WARNING; '
                f'found in: {[r.getMessage() for r in warning_records]!r}'
            )

            # (c) Structured WARNING must carry bad_sha[:12] via step_diagnostics
            assert any(bad_sha[:12] in r.getMessage() for r in warning_records), (
                f'Expected structured WARNING to contain {bad_sha[:12]!r}; '
                f'got WARNING records: {[r.getMessage() for r in warning_records]!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_unresolved_step_step_id_is_none_when_plan_id_is_not_string(
        self, git_ops: GitOps,
    ):
        """Non-string step id (e.g. integer) must be coerced to None in UnresolvedStep.

        ``step.get('id')`` returns ``Any``.  When the plan stores an integer id
        (e.g. ``123``), the value silently flows into ``UnresolvedStep.step_id``
        which is annotated ``str | None`` — a silent type violation.

        After the fix, a non-string id is replaced with ``None`` before
        constructing the ``UnresolvedStep``.

        Setup: done step with ``id=123`` (integer) and a bad commit SHA.
        The bad SHA forces an ``UnresolvedStep`` to be created.  The plan
        file ``gone.py`` is never committed → absent from merge tree → drop.

        Assertion: ``result.unresolved_steps[0].step_id is None``.
        Currently fails because 123 flows through unmodified.
        """
        worktree = (await git_ops.create_worktree('issue5-int-step-id')).path

        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        bad_sha = '0' * 40  # non-existent → forces UnresolvedStep creation

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-issue5', 'T-issue5', 'desc')
        artifacts.write_plan({
            'files': ['gone.py'],
            'modules': [],
            'steps': [
                {
                    'id': 123,                  # integer, not str → must become None
                    'description': 'supposedly deleted gone.py',
                    'status': 'done',
                    'commit': bad_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'issue5-int-step-id')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                task_id='t-issue5',
            )
            assert result.unresolved_steps, 'Expected at least one UnresolvedStep'
            us = result.unresolved_steps[0]
            assert us.step_id is None, (
                f'Expected step_id=None for integer plan id, got {us.step_id!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_unresolved_step_stderr_truncated_at_module_constants(
        self, git_ops: GitOps,
    ):
        """Truncation of stderr/cat_file_stderr uses named module-level constants.

        Issue 6: the thresholds 500 and 200 are inline magic numbers.  After the
        fix, ``_UNRESOLVED_STDERR_MAX`` and ``_UNRESOLVED_CAT_STDERR_MAX`` are
        exported from the module and used in the truncation expression.

        This test:
          1. Imports the constants (ImportError if not yet defined → currently fails).
          2. Monkey-patches ``_run`` so diff-tree returns 1500 X's as stderr and
             cat-file commit-probe returns 600 Y's.  Merge-tree cat-file probes
             (``<sha>:<path>`` arg form) fall through to the real subprocess.
          3. Asserts that the stored stderr / cat_file_stderr lengths equal
             ``_UNRESOLVED_STDERR_MAX + len(' <truncated>')`` and
             ``_UNRESOLVED_CAT_STDERR_MAX + len(' <truncated>')``.
        """
        from orchestrator.merge_queue import (  # noqa: PLC0415
            _UNRESOLVED_CAT_STDERR_MAX,
            _UNRESOLVED_STDERR_MAX,
        )

        worktree = (await git_ops.create_worktree('issue6-trunc-const')).path

        (worktree / 'anchor.py').write_text('anchor = 1\n')
        await git_ops.commit(worktree, 'Add anchor.py')

        bad_sha = '0' * 40  # non-existent → diff-tree rc != 0

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t-issue6', 'T-issue6', 'desc')
        artifacts.write_plan({
            'files': ['gone.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'supposedly deleted gone.py',
                    'status': 'done',
                    'commit': bad_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'issue6-trunc-const')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            from orchestrator import merge_queue as mq
            original_run = mq._run

            async def _intercept(cmd, cwd=None, **kwargs):
                if '--diff-filter=D' in cmd:
                    # Inject long stderr for diff-tree failure
                    return (128, '', 'X' * 1500)
                if cmd and '^{commit}' in cmd[-1]:
                    # Inject long stderr for cat-file commit-probe failure
                    return (1, '', 'Y' * 600)
                # Merge-tree cat-file probes and all other calls fall through
                return await original_run(cmd, cwd=cwd, **kwargs)

            with patch('orchestrator.merge_queue._run', new=_intercept):
                result = await _check_plan_targets_in_tree(
                    merge_result.merge_commit, worktree, git_ops,
                    task_id='t-issue6',
                )

            assert result.unresolved_steps, 'Expected at least one UnresolvedStep'
            us = result.unresolved_steps[0]

            elision = ' <truncated>'
            assert len(us.stderr) == _UNRESOLVED_STDERR_MAX + len(elision), (
                f'Expected stderr length {_UNRESOLVED_STDERR_MAX + len(elision)}, '
                f'got {len(us.stderr)}: {us.stderr!r}'
            )
            assert len(us.cat_file_stderr) == _UNRESOLVED_CAT_STDERR_MAX + len(elision), (
                f'Expected cat_file_stderr length '
                f'{_UNRESOLVED_CAT_STDERR_MAX + len(elision)}, '
                f'got {len(us.cat_file_stderr)}: {us.cat_file_stderr!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)


@pytest.mark.asyncio
class TestMergeWorker:
    async def test_basic_merge_through_queue(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Submit a merge request → worker merges → file appears on main."""
        worktree = (await git_ops.create_worktree('queue-basic')).path
        (worktree / 'queued.py').write_text('queued = True\n')
        await git_ops.commit(worktree, 'Add queued file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('1', 'queue-basic', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        assert result.status == 'done'

        # Verify file is on main
        _, content, _ = await _run(
            ['git', 'show', 'main:queued.py'], cwd=git_ops.project_root,
        )
        assert 'queued = True' in content

        # File should also be in the working tree (working tree synced)
        assert (git_ops.project_root / 'queued.py').exists()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_blocks_when_merge_drops_plan_target(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker blocks and surfaces a clear reason on a real drop.

        Drop semantics: file was on task HEAD but is absent from the
        merge commit. Reproducing a real conflict-time drop in a unit
        test is awkward, so we mock the detector to simulate the drop
        and verify MergeWorker's handling (reason text, no advance).
        """
        worktree = (await git_ops.create_worktree('drop-guard-task')).path
        (worktree / 'kept.py').write_text('kept = True\n')
        (worktree / 'dropped.py').write_text('dropped = True\n')
        await git_ops.commit(worktree, 'Add kept + dropped')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-guard', 'Drop guard', 'desc')
        artifacts.write_plan({
            'files': ['kept.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _fake_drop_check(*_args, **_kwargs):
            return DropGuardResult(dropped=['dropped.py'])

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass(),
        ), patch(
            'orchestrator.merge_queue._check_plan_targets_in_tree',
            _fake_drop_check,
        ):
            req = _make_request('drop-guard', 'drop-guard-task', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'dropped.py' in outcome.reason
        assert 'plan target' in outcome.reason.lower()

        # Main must NOT have advanced — drop-guard fires before advance_main
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'kept.py' not in main_files

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_blocks_when_merge_drops_plan_target_real_detector(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker drives the real detector and blocks on a synthesised drop.

        Companion to ``test_blocks_when_merge_drops_plan_target``: that test
        mocks the detector to pin the reason-text contract; this one leaves
        the detector in place and mocks ``git_ops.merge_to_main`` so a
        future refactor of the wiring (argument order, return-value
        interpretation) cannot pass both tests while silently breaking the
        guard in production.
        """
        worktree = (await git_ops.create_worktree('drop-guard-real')).path
        (worktree / 'retained.py').write_text('retained = 1\n')
        await git_ops.commit(worktree, 'Add retained')
        rc, pre_drop_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        assert rc == 0
        pre_drop_sha = pre_drop_sha.strip()

        (worktree / 'dropped.py').write_text('dropped = 1\n')
        await git_ops.commit(worktree, 'Add dropped')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-guard-real', 'Drop guard real', 'desc')
        artifacts.write_plan({
            'files': ['retained.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Point the "merge commit" at a task-branch SHA that predates the
        # addition of dropped.py. The real detector sees dropped.py on task
        # HEAD but absent from that tree → flags it as a drop.
        async def _fake_merge_to_main(*_args: Any, **_kwargs: Any) -> MergeResult:
            return MergeResult(
                success=True,
                merge_commit=pre_drop_sha,
                pre_merge_sha=pre_drop_sha,
                merge_worktree=None,
            )

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass(),
        ), patch.object(git_ops, 'merge_to_main', _fake_merge_to_main):
            req = _make_request(
                'drop-guard-real', 'drop-guard-real', worktree, config,
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'dropped.py' in outcome.reason
        assert 'plan target' in outcome.reason.lower()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_merge_worker_surfaces_unresolved_steps_in_dropped_outcome_reason(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker includes unresolved-step diagnostics in the blocked outcome reason.

        When _check_plan_targets_in_tree returns a DropGuardResult with both
        dropped files AND unresolved steps (steps whose diff-tree query failed),
        the MergeOutcome.reason must mention the specific unresolved step details
        (step_id, commit SHA, object_missing flag) so the steward can investigate
        whether a real planned-file deletion was mis-flagged as a drop.

        Will fail until step 8 adds the unresolved-step suffix to the reason text.
        """
        worktree = (await git_ops.create_worktree('drop-unresolved-mw')).path
        (worktree / 'kept.py').write_text('kept = True\n')
        await git_ops.commit(worktree, 'Add kept.py')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-unresolved', 'Drop unresolved', 'desc')
        artifacts.write_plan({
            'files': ['kept.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        bad_commit = 'cf82ae6815'

        async def _fake_drop_check_with_unresolved(*_args, **_kwargs):
            return DropGuardResult(
                dropped=['dropped.py'],
                unresolved_steps=[
                    UnresolvedStep(
                        step_idx=2,
                        step_id='step-3',
                        commit=bad_commit,
                        rc=128,
                        stderr='fatal: bad object cf82ae6815',
                        object_missing=True,
                    )
                ],
            )

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass(),
        ), patch(
            'orchestrator.merge_queue._check_plan_targets_in_tree',
            _fake_drop_check_with_unresolved,
        ):
            req = _make_request('drop-unresolved', 'drop-unresolved-mw', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        # (a) Still blocked (fail-closed)
        assert outcome.status == 'blocked', f'Expected blocked, got {outcome.status!r}'
        # (b) Existing drop-guard reason contract preserved
        assert 'dropped.py' in outcome.reason, f'Missing dropped.py in reason: {outcome.reason!r}'
        assert 'plan target' in outcome.reason.lower(), (
            f'Missing "plan target" in reason: {outcome.reason!r}'
        )
        # (c) Unresolved-step diagnostics included
        assert 'step-3' in outcome.reason, (
            f'Expected step_id "step-3" in reason: {outcome.reason!r}'
        )
        assert bad_commit in outcome.reason, (
            f'Expected commit {bad_commit!r} in reason: {outcome.reason!r}'
        )
        assert 'object_missing' in outcome.reason, (
            f'Expected "object_missing" in reason: {outcome.reason!r}'
        )
        # Must indicate the diff-tree query failed (exact phrase from _format_unresolved_steps_suffix)
        reason_lower = outcome.reason.lower()
        assert 'drop-guard' in reason_lower or 'could not query' in reason_lower, (
            f'Expected drop-guard failure phrase in reason: {outcome.reason!r}'
        )

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_already_merged_returns_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Branch that's already on main returns already_merged."""
        worktree = (await git_ops.create_worktree('already-merged')).path
        (worktree / 'merged.py').write_text('merged = True\n')
        await git_ops.commit(worktree, 'Add merged file')

        # Merge manually first
        result = await git_ops.merge_to_main(worktree, 'already-merged')
        assert result.success
        assert result.merge_commit is not None
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

        # Now submit to queue — should detect already merged
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('2', 'already-merged', worktree, config)
        await queue.put(req)

        outcome = await asyncio.wait_for(req.result, timeout=10)
        assert outcome.status == 'already_merged'

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_conflict_returns_conflict(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Conflicting branch returns conflict status."""
        # Create worktree FIRST (from current main)
        worktree = (await git_ops.create_worktree('conflict-task')).path

        # THEN advance main with conflicting change to same file
        (git_ops.project_root / 'README.md').write_text('# Main version\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main change'],
            cwd=git_ops.project_root,
        )

        # Now modify same file in worktree (divergent history)
        (worktree / 'README.md').write_text('# Task version\n')
        await git_ops.commit(worktree, 'Task change')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('3', 'conflict-task', worktree, config)
        await queue.put(req)

        outcome = await asyncio.wait_for(req.result, timeout=10)
        assert outcome.status == 'conflict'
        assert outcome.conflict_details  # non-empty

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_cas_failure_reenqueues_at_front(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """CAS failure → re-enqueue at front → succeeds on retry."""
        worktree = (await git_ops.create_worktree('cas-retry')).path
        (worktree / 'retry.py').write_text('retry = True\n')
        await git_ops.commit(worktree, 'Add retry file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Monkey-patch advance_main to fail once, then succeed
        original = git_ops.advance_main
        call_count = 0

        async def _fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 'cas_failed'  # simulate CAS failure
            return await original(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_fail_then_succeed),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('4', 'cas-retry', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done'
        assert call_count == 2  # failed once, succeeded on retry

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_graceful_shutdown_drains(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """stop() resolves all pending futures as blocked."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        # Don't start worker yet — put items in queue, then stop
        worktree = (await git_ops.create_worktree('shutdown')).path
        req = _make_request('5', 'shutdown', worktree, config)
        await queue.put(req)

        # stop() should drain the queue and resolve the future
        await worker.stop()

        assert req.result.done()
        outcome = req.result.result()
        assert outcome.status == 'blocked'
        assert 'shutting down' in outcome.reason.lower()

    async def test_verify_failure_returns_blocked(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Post-merge verification failure → blocked."""
        worktree = (await git_ops.create_worktree('verify-fail')).path
        (worktree / 'bad.py').write_text('bad = True\n')
        await git_ops.commit(worktree, 'Add bad file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Mock verification to fail
        mock_verify = AsyncMock(return_value=MagicMock(passed=False, summary='tests failed'))

        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify):
            req = _make_request('6', 'verify-fail', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'verification failed' in outcome.reason.lower()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_cas_retry_limit_exhausted(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """CAS failures beyond MAX_CAS_RETRIES resolve as blocked."""
        worktree = (await git_ops.create_worktree('cas-limit')).path
        (worktree / 'limit.py').write_text('limit = True\n')
        await git_ops.commit(worktree, 'Add limit file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # advance_main always returns cas_failed
        async def _always_cas_fail(*args, **kwargs):
            return 'cas_failed'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_always_cas_fail),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('7', 'cas-limit', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'cas retry limit' in outcome.reason.lower()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_not_descendant_returns_blocked_immediately(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Permanent not_descendant failure blocks without re-enqueue."""
        worktree = (await git_ops.create_worktree('perm-fail')).path
        (worktree / 'perm.py').write_text('perm = True\n')
        await git_ops.commit(worktree, 'Add perm file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        call_count = 0

        async def _not_descendant(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return 'not_descendant'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_not_descendant),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('8', 'perm-fail', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=10)

        assert outcome.status == 'blocked'
        assert 'not_descendant' in outcome.reason
        # Should only be called once — no re-enqueue for permanent failures
        assert call_count == 1

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_merge_worker_emits_duration_ms_on_done(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """MergeWorker emits duration_ms on the 'done' outcome.

        Asserts that the merge_attempt event row for outcome='done' has a
        non-null integer duration_ms >= 0.
        """
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt = await _make_branch_with_file(
            git_ops, 'dur-done', 'dur_done.py', 'dur = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('dur-done', 'dur-done', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') AS outcome, duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        done_rows = [r for r in rows if r[0] == 'done']
        assert len(done_rows) == 1, f'Expected 1 done row, got: {rows}'
        assert done_rows[0][1] is not None, 'duration_ms should not be NULL'
        assert isinstance(done_rows[0][1], int), f'duration_ms should be int, got {type(done_rows[0][1])}'
        assert done_rows[0][1] >= 0, f'duration_ms should be >= 0, got {done_rows[0][1]}'

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_merge_worker_emits_duration_ms_on_non_done_outcomes(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Every MergeWorker emit site sets a non-null duration_ms.

        Covers already_merged, conflict, and cas_retry outcomes in addition
        to done (covered by test_merge_worker_emits_duration_ms_on_done).
        """
        # --- Scenario A: already_merged ---
        db_a = tmp_path / 'events_a.db'
        es_a = EventStore(db_path=db_a, run_id='run-a')

        wt_am = await _make_branch_with_file(
            git_ops, 'dur-am', 'dur_am.py', 'am = 1\n',
        )
        # Merge manually so it's already on main
        r = await git_ops.merge_to_main(wt_am, 'dur-am')
        assert r.success
        assert r.merge_commit is not None
        await git_ops.advance_main(r.merge_commit)
        if r.merge_worktree:
            await git_ops.cleanup_merge_worktree(r.merge_worktree)

        q_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_a = MergeWorker(git_ops, q_a, event_store=es_a)
        wt_a = asyncio.create_task(w_a.run())

        req_am = _make_request('dur-am', 'dur-am', wt_am, config)
        await q_a.put(req_am)
        out_am = await asyncio.wait_for(req_am.result, timeout=30)
        assert out_am.status == 'already_merged'
        await w_a.stop()
        wt_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wt_a

        conn = sqlite3.connect(str(db_a))
        rows_a = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_a), f'NULL duration_ms in already_merged: {rows_a}'

        # --- Scenario B: conflict ---
        db_b = tmp_path / 'events_b.db'
        es_b = EventStore(db_path=db_b, run_id='run-b')

        wt_cfl = (await git_ops.create_worktree('dur-cfl')).path
        # Advance main with a conflicting change
        (git_ops.project_root / 'README.md').write_text('# conflict-source\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Conflict source'], cwd=git_ops.project_root)
        # Make conflicting change in worktree
        (wt_cfl / 'README.md').write_text('# conflict-task\n')
        await git_ops.commit(wt_cfl, 'Conflict task change')

        q_b: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_b = MergeWorker(git_ops, q_b, event_store=es_b)
        wt_b = asyncio.create_task(w_b.run())

        req_cfl = _make_request('dur-cfl', 'dur-cfl', wt_cfl, config)
        await q_b.put(req_cfl)
        out_cfl = await asyncio.wait_for(req_cfl.result, timeout=30)
        assert out_cfl.status == 'conflict'
        await w_b.stop()
        wt_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wt_b

        conn = sqlite3.connect(str(db_b))
        rows_b = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_b), f'NULL duration_ms in conflict: {rows_b}'

        # --- Scenario C: cas_retry ---
        db_c = tmp_path / 'events_c.db'
        es_c = EventStore(db_path=db_c, run_id='run-c')

        wt_cas = await _make_branch_with_file(
            git_ops, 'dur-cas', 'dur_cas.py', 'cas = 1\n',
        )

        q_c: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_c = MergeWorker(git_ops, q_c, event_store=es_c)
        wt_c = asyncio.create_task(w_c.run())

        original_advance = git_ops.advance_main
        call_count_c = 0

        async def _fail_once(*args, **kwargs):
            nonlocal call_count_c
            call_count_c += 1
            if call_count_c == 1:
                return 'cas_failed'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_fail_once),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req_cas = _make_request('dur-cas', 'dur-cas', wt_cas, config)
            await q_c.put(req_cas)
            out_cas = await asyncio.wait_for(req_cas.result, timeout=30)

        assert out_cas.status == 'done'
        await w_c.stop()
        wt_c.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wt_c

        conn = sqlite3.connect(str(db_c))
        rows_c = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_c), f'NULL duration_ms in cas scenario: {rows_c}'

    async def test_merge_worker_success_returns_merge_sha(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker success path: MergeOutcome.merge_sha is the merge commit.

        Drives the full CAS-advance path through MergeWorker and asserts that
        the resulting MergeOutcome carries the real 40-char merge commit SHA.
        Fails initially because merge_queue.py:400 still constructs
        MergeOutcome('done') without the SHA (step-3 guard; impl in step-4).
        """
        worktree = await _make_branch_with_file(
            git_ops, 'sha-basic', 'sha_basic.py', 'sha = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('sha-task', 'sha-basic', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

        assert result.status == 'done'
        assert result.merge_sha is not None, 'merge_sha must be set on done outcome'
        assert len(result.merge_sha) == 40, f'expected 40-char SHA, got: {result.merge_sha!r}'
        assert all(c in '0123456789abcdef' for c in result.merge_sha), (
            f'merge_sha is not a hex string: {result.merge_sha!r}'
        )


# ---------------------------------------------------------------------------
# Helpers for speculative tests
# ---------------------------------------------------------------------------

async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


# ---------------------------------------------------------------------------
# TestSpeculativeMergeWorker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculativeMergeWorker:
    async def test_speculative_basic_throughput(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Submit 2 merge requests. Both complete as 'done', both files on main.

        N+1 is speculatively merged against N's merge SHA (not original main).
        Both complete without error.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'spec-n', 'file_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'spec-n1', 'file_n1.py', 'n1 = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            req_n = _make_request('spec-n', 'spec-n', wt_n, config)
            req_n1 = _make_request('spec-n1', 'spec-n1', wt_n1, config)

            # Submit both before the worker processes them
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'done', f'N failed: {outcome_n}'
        assert outcome_n1.status == 'done', f'N+1 failed: {outcome_n1}'

        # Both files must appear on main
        _, out_n, _ = await _run(
            ['git', 'show', 'main:file_n.py'], cwd=git_ops.project_root,
        )
        assert 'n = 1' in out_n

        _, out_n1, _ = await _run(
            ['git', 'show', 'main:file_n1.py'], cwd=git_ops.project_root,
        )
        assert 'n1 = 2' in out_n1

        await worker.stop()
        await worker_task

    async def test_speculative_discard_on_failure(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """When N's verification fails, N+1's speculative merge is discarded
        and re-merged against actual main.  N returns 'blocked', N+1 returns
        'done' after the fresh re-merge.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'disc-n', 'file_disc_n.py', 'disc_n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'disc-n1', 'file_disc_n1.py', 'disc_n1 = 2\n',
        )

        # Track how many times verify is called per task
        verify_calls: dict[str, int] = {}

        async def _verify_side_effect(
            merge_wt, cfg, module_configs, task_files=None, **_kwargs,
        ):
            # Determine which task by looking at which file is present
            n_file = merge_wt / 'file_disc_n.py'
            if n_file.exists():
                verify_calls['n'] = verify_calls.get('n', 0) + 1
                return MagicMock(passed=False, summary='N tests failed')
            else:
                verify_calls['n1'] = verify_calls.get('n1', 0) + 1
                return MagicMock(passed=True, summary='')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_verify_side_effect,
        ):
            req_n = _make_request('disc-n', 'disc-n', wt_n, config)
            req_n1 = _make_request('disc-n1', 'disc-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'blocked', f'N should be blocked: {outcome_n}'
        assert outcome_n1.status == 'done', f'N+1 should succeed after re-merge: {outcome_n1}'

        # N+1's file must appear on main (re-merged and advanced)
        _, out_n1, _ = await _run(
            ['git', 'show', 'main:file_disc_n1.py'], cwd=git_ops.project_root,
        )
        assert 'disc_n1 = 2' in out_n1

        # N's file must NOT be on main (N was blocked)
        rc, _, _ = await _run(
            ['git', 'cat-file', '-e', 'main:file_disc_n.py'],
            cwd=git_ops.project_root,
        )
        assert rc != 0, 'N file should not be on main'

        await worker.stop()
        await worker_task

    async def test_speculative_depth_cap(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """With depth-1 cap, N+2 is not speculatively merged until N+1's
        speculation resolves.  Submit N, N+1, N+2 — all complete as 'done'.

        Verified by tracking concurrent active merge worktrees: the count must
        never exceed 2 (N's worktree while N is being verified, plus N+1's
        speculative worktree).  N+2 is only started after N finishes.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'cap-n', 'file_cap_n.py', 'cap_n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'cap-n1', 'file_cap_n1.py', 'cap_n1 = 2\n',
        )
        wt_n2 = await _make_branch_with_file(
            git_ops, 'cap-n2', 'file_cap_n2.py', 'cap_n2 = 3\n',
        )

        # Track maximum number of merge worktrees active at the same time.
        # Each merge worktree is created in _create_merge_worktree and removed
        # in cleanup_merge_worktree.  With depth-1, the peak must be ≤ 2.
        active_worktrees: set[str] = set()
        max_concurrent = 0
        original_create = git_ops._create_merge_worktree
        original_cleanup = git_ops.cleanup_merge_worktree

        async def _tracking_create(base_sha=None):
            wt, sha = await original_create(base_sha)
            active_worktrees.add(str(wt))
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, len(active_worktrees))
            return wt, sha

        async def _tracking_cleanup(merge_wt):
            active_worktrees.discard(str(merge_wt))
            await original_cleanup(merge_wt)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, '_create_merge_worktree', side_effect=_tracking_create),
            patch.object(git_ops, 'cleanup_merge_worktree', side_effect=_tracking_cleanup),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req_n = _make_request('cap-n', 'cap-n', wt_n, config)
            req_n1 = _make_request('cap-n1', 'cap-n1', wt_n1, config)
            req_n2 = _make_request('cap-n2', 'cap-n2', wt_n2, config)
            await queue.put(req_n)
            await queue.put(req_n1)
            await queue.put(req_n2)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)
            outcome_n2 = await asyncio.wait_for(req_n2.result, timeout=60)

        assert outcome_n.status == 'done', f'N: {outcome_n}'
        assert outcome_n1.status == 'done', f'N+1: {outcome_n1}'
        assert outcome_n2.status == 'done', f'N+2: {outcome_n2}'

        # Depth-1 cap: at most 2 merge worktrees active simultaneously
        # (the item being verified + 1 speculative item)
        assert max_concurrent <= 2, (
            f'Depth-1 cap violated: {max_concurrent} concurrent merge worktrees '
            f'(expected ≤ 2)'
        )

        # All three files on main
        for fname in ('file_cap_n.py', 'file_cap_n1.py', 'file_cap_n2.py'):
            rc, _, _ = await _run(
                ['git', 'cat-file', '-e', f'main:{fname}'],
                cwd=git_ops.project_root,
            )
            assert rc == 0, f'{fname} not on main'

        await worker.stop()
        await worker_task

    async def test_speculative_single_item_degenerates(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Single merge request through SpeculativeMergeWorker completes as 'done'.

        Confirms the speculative pipeline degenerates to serial behavior when
        there is only one item in the queue (no look-ahead possible).
        """
        wt = await _make_branch_with_file(
            git_ops, 'single', 'single.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('single', 'single', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done'
        _, out, _ = await _run(['git', 'show', 'main:single.py'], cwd=git_ops.project_root)
        assert 'x = 1' in out

        await worker.stop()
        await worker_task

    async def test_speculative_verify_called_with_max_retries_zero(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Merge-queue post-merge verify must pass max_retries=0.

        A deterministic verify hang would otherwise be retried per
        ``config.verify_timeout_retries``, tripling queue-wide stall.
        This is the regression that caused the 2026-04-20 90-minute jam.
        """
        wt = await _make_branch_with_file(
            git_ops, 'retry0', 'retry0.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        captured_kwargs: list[dict] = []

        async def spy_verify(*args, **kwargs):
            captured_kwargs.append(kwargs)
            result = AsyncMock()
            result.passed = True
            result.summary = ''
            return result

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=spy_verify,
        ):
            req = _make_request('retry0', 'retry0', wt, config)
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        await worker_task

        assert captured_kwargs, 'run_scoped_verification was not invoked'
        assert captured_kwargs[0].get('max_retries') == 0, (
            f'merge-queue verify must pass max_retries=0; got {captured_kwargs[0]!r}'
        )

    async def test_speculative_shutdown_drains_both(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """stop() resolves all pending Futures as 'blocked' with shutdown reason."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        # Don't start worker — just queue items and stop
        wt_a = (await git_ops.create_worktree('shut-a')).path
        wt_b = (await git_ops.create_worktree('shut-b')).path
        req_a = _make_request('shut-a', 'shut-a', wt_a, config)
        req_b = _make_request('shut-b', 'shut-b', wt_b, config)
        await queue.put(req_a)
        await queue.put(req_b)

        await worker.stop()

        assert req_a.result.done()
        assert req_b.result.done()
        outcome_a = req_a.result.result()
        outcome_b = req_b.result.result()
        assert outcome_a.status == 'blocked'
        assert outcome_b.status == 'blocked'
        assert 'shutting down' in outcome_a.reason.lower()
        assert 'shutting down' in outcome_b.reason.lower()

    async def test_speculative_conflict_n_plus_1(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """N merges cleanly; N+1 conflicts when speculatively merged.
        N completes as 'done'; N+1 returns 'conflict'.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'cfl-n', 'file_cfl_n.py', 'cfl_n = 1\n',
        )

        # Create N+1 worktree from current main, then advance main via
        # a direct commit to cause a conflict on the same file.
        wt_n1 = (await git_ops.create_worktree('cfl-n1')).path
        # Write conflicting content to README.md in both main and wt_n1
        (git_ops.project_root / 'README.md').write_text('# Main conflict\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Main side change'], cwd=git_ops.project_root)
        (wt_n1 / 'README.md').write_text('# N+1 conflict\n')
        await git_ops.commit(wt_n1, 'N+1 conflicting change')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('cfl-n', 'cfl-n', wt_n, config)
            req_n1 = _make_request('cfl-n1', 'cfl-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'done', f'N: {outcome_n}'
        assert outcome_n1.status == 'conflict', f'N+1: {outcome_n1}'
        assert outcome_n1.conflict_details

        await worker.stop()
        await worker_task

    async def test_speculative_events_emitted(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """speculative_merge event emitted when N+1 is speculatively merged.
        speculative_discard event emitted when N fails and N+1 is discarded.
        """
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt_n = await _make_branch_with_file(
            git_ops, 'ev-n', 'file_ev_n.py', 'ev_n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'ev-n1', 'file_ev_n1.py', 'ev_n1 = 2\n',
        )

        async def _fail_n_pass_n1(merge_wt, cfg, module_configs, task_files=None):
            n_present = (merge_wt / 'file_ev_n.py').exists()
            n1_present = (merge_wt / 'file_ev_n1.py').exists()
            # Speculative verify of N: N present, N+1 not yet merged → fail.
            # Any other shape means N+1 was re-verified after N failed (which
            # contradicts the discard-on-failure contract) — fail loudly
            # rather than silently returning pass.
            if n_present and not n1_present:
                return MagicMock(passed=False, summary='N failed')
            raise AssertionError(
                f'unexpected verify call: n_present={n_present}, '
                f'n1_present={n1_present} — N+1 should have been discarded '
                f'after N failed, not re-verified'
            )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_fail_n_pass_n1,
        ):
            req_n = _make_request('ev-n', 'ev-n', wt_n, config)
            req_n1 = _make_request('ev-n1', 'ev-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)
            await asyncio.wait_for(req_n.result, timeout=60)
            await asyncio.wait_for(req_n1.result, timeout=60)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id FROM events ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [r[0] for r in rows]
        assert 'speculative_merge' in event_types, f'No speculative_merge event: {event_types}'
        assert 'speculative_discard' in event_types, f'No speculative_discard event: {event_types}'

        await worker.stop()
        await worker_task

    async def test_speculative_already_merged_n_plus_1(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """N+1 branch already on main → returns 'already_merged' without
        attempting a speculative merge.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'am-n', 'file_am_n.py', 'am_n = 1\n',
        )
        # Create N+1 as already-merged: merge it first, then submit it
        wt_n1 = (await git_ops.create_worktree('am-n1')).path
        (wt_n1 / 'file_am_n1.py').write_text('am_n1 = 2\n')
        await git_ops.commit(wt_n1, 'N+1 file')
        result = await git_ops.merge_to_main(wt_n1, 'am-n1')
        assert result.success
        assert result.merge_commit is not None
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('am-n', 'am-n', wt_n, config)
            req_n1 = _make_request('am-n1', 'am-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'done', f'N: {outcome_n}'
        assert outcome_n1.status == 'already_merged', f'N+1: {outcome_n1}'

        await worker.stop()
        await worker_task

    async def test_verifier_exception_releases_speculation_slot(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """_verify_and_advance raising must resolve N's Future and release the slot.

        Without the try/except/finally fix: the verifier loop crashes before
        calling _speculation_slot.set() and before resolving N's Future, causing
        both a deadlock (merger blocked waiting for slot) and a hung Future.

        With the fix: except clause resolves N's Future as 'blocked' with a
        'Verifier error' reason; finally clause always sets _speculation_slot.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'vex-n', 'file_vex_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'vex-n1', 'file_vex_n1.py', 'n1 = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Capture original before replacing with mock
        original_vaa = worker._verify_and_advance
        call_count = 0

        async def mock_vaa(item):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError('Unexpected verifier error')
            return await original_vaa(item)

        worker._verify_and_advance = mock_vaa  # type: ignore[method-assign]

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('vex-n', 'vex-n', wt_n, config)
            req_n1 = _make_request('vex-n1', 'vex-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            # N must resolve as 'blocked' with 'Verifier error' (not hang forever)
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'blocked', f'N: {outcome_n}'
            assert 'Verifier error' in outcome_n.reason, (
                f'Expected Verifier error in reason, got: {outcome_n.reason}'
            )
            assert 'Unexpected verifier error' in outcome_n.reason

            # _speculation_slot must be set (not stuck cleared → deadlock)
            assert worker._speculation_slot.is_set(), (
                '_speculation_slot stuck cleared — merger will deadlock on next request'
            )

            # N+1 must also complete (not hang forever)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=30)
            assert outcome_n1.status in ('done', 'blocked'), f'N+1: {outcome_n1}'

        await worker.stop()
        await worker_task

    async def test_verifier_remerge_exception_releases_slot(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """_remerge raising must resolve N+1's Future and release the speculation slot.

        Scenario: N fails verification (n_failed=True), N+1 is speculative.
        The verifier calls _remerge(N+1) which raises unexpectedly.

        Without fix: exception propagates out of loop body, N+1's Future is never
        resolved, _speculation_slot may be left cleared → downstream deadlock.

        With fix: except clause resolves N+1's Future as 'blocked' with
        'Verifier error'; finally clause always sets _speculation_slot.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'vre-n', 'file_vre_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'vre-n1', 'file_vre_n1.py', 'n1 = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # N fails verification → n_failed=True; _remerge then raises for N+1
        mock_verify = AsyncMock(return_value=MagicMock(passed=False, summary='tests failed'))

        async def raise_on_remerge(req, started_monotonic: float | None = None):  # type: ignore[no-untyped-def]
            raise RuntimeError('_remerge failed unexpectedly')

        worker._remerge = raise_on_remerge  # type: ignore[method-assign]

        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify):
            req_n = _make_request('vre-n', 'vre-n', wt_n, config)
            req_n1 = _make_request('vre-n1', 'vre-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            # N fails verification → blocked
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'blocked', f'N: {outcome_n}'

            # N+1: _remerge raised → 'blocked' with Verifier error (not hang)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=30)
            assert outcome_n1.status == 'blocked', f'N+1: {outcome_n1}'
            assert 'Verifier error' in outcome_n1.reason, (
                f'Expected Verifier error in N+1 reason, got: {outcome_n1.reason}'
            )
            assert '_remerge failed' in outcome_n1.reason

            # _speculation_slot must be released
            assert worker._speculation_slot.is_set(), (
                '_speculation_slot stuck cleared after _remerge exception'
            )

        await worker.stop()
        await worker_task

    async def test_run_cancels_subtasks_on_cancellation(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Cancelling run()'s outer task must cancel both _merger_task and _verifier_task.

        Without fix: run() only catches CancelledError and re-raises, leaving
        the subtasks running (orphaned). If one subtask raises RuntimeError,
        the other continues running forever.

        With fix: any BaseException cancels both subtasks before re-raising.
        """
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        # ── Part 1: outer task cancellation ──────────────────────────────
        worker_task = asyncio.create_task(worker.run())
        # Give merger and verifier tasks a chance to start
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert worker._merger_task is not None
        assert worker._verifier_task is not None

        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

        assert worker._merger_task.done(), 'merger_task not done after cancellation'
        assert worker._verifier_task.done(), 'verifier_task not done after cancellation'
        assert worker._merger_task.cancelled() or worker._merger_task.exception() is not None
        assert worker._verifier_task.cancelled() or worker._verifier_task.exception() is not None

        # ── Part 2: subtask RuntimeError cancels sibling ─────────────────
        worker2 = SpeculativeMergeWorker(git_ops, asyncio.Queue())

        async def crashing_merger():
            raise RuntimeError('Merger crashed unexpectedly')

        worker2._merger_loop = crashing_merger  # type: ignore[method-assign]

        worker_task2 = asyncio.create_task(worker2.run())
        # Allow merger to crash
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        with pytest.raises((RuntimeError, asyncio.CancelledError)):
            await asyncio.wait_for(worker_task2, timeout=5)

        assert worker2._verifier_task is not None
        assert worker2._verifier_task.done(), (
            'verifier_task not cancelled after merger RuntimeError'
        )

    async def test_merger_exception_sends_verifier_sentinel(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """_merger_loop must put None sentinel into verifier queue even when it crashes.

        The inner try/except (step-31) catches Exception, resolves the Future as
        'blocked', and continues the loop (rather than propagating the exception).
        The loop then exits cleanly via a shutdown sentinel. The try/finally wrapping
        the entire while-loop guarantees the verifier sentinel is always sent.

        We test _merger_loop() directly to isolate this from the run() subtask
        cancellation logic (step-24), which also terminates the verifier.
        """
        wt = await _make_branch_with_file(
            git_ops, 'mes-n', 'file_mes_n.py', 'n = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        # merge_to_main raises on the first call to simulate an unexpected crash
        async def crash_merge(worktree, branch, base_sha=None):  # type: ignore[no-untyped-def]
            raise RuntimeError('Unexpected error in merge_to_main')

        req = _make_request('mes-n', 'mes-n', wt, config)
        await queue.put(req)
        # Shutdown sentinel so the loop exits after handling the exception —
        # the inner except catches RuntimeError and continues, so without this
        # the loop would block on queue.get() forever.
        await queue.put(None)  # type: ignore[arg-type]

        with patch.object(git_ops, 'merge_to_main', new=crash_merge):
            await worker._merger_loop()

        # (1) Future must be resolved as 'blocked' by the inner exception handler.
        assert req.result.done(), (
            'Future must be resolved when merger catches an unexpected exception'
        )
        assert req.result.result().status == 'blocked'
        assert 'Merger error' in req.result.result().reason

        # (2) The verifier queue must contain the sentinel (None).
        # Without the try/finally fix, the queue would be empty here.
        assert not worker._verifier_queue.empty(), (
            'Verifier queue is empty — sentinel was never sent by dying merger'
        )
        sentinel = worker._verifier_queue.get_nowait()
        assert sentinel is None, f'Expected sentinel (None), got: {sentinel}'

    async def test_revparse_failure_produces_blocked(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """git rev-parse HEAD failure must resolve Future as blocked (not crash).

        Without fix: the return code from git rev-parse HEAD is not checked.
        A non-zero rc leaves branch_head as empty/garbage, and the subsequent
        is_ancestor() call may crash or behave incorrectly.

        With fix: rc != 0 triggers an immediate blocked outcome pushed to the
        verifier queue with reason 'rev-parse HEAD failed: <err>'. Subsequent
        requests are still processed normally.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'rp-n', 'file_rp_n.py', 'n = 1\n',
        )
        wt_ok = await _make_branch_with_file(
            git_ops, 'rp-ok', 'file_rp_ok.py', 'ok = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Mock _run to fail rev-parse for rp-n worktree only
        original_run = __import__(
            'orchestrator.merge_queue', fromlist=['_run']
        )._run
        # We patch the module-level _run used inside merge_queue
        call_log: list[tuple] = []

        async def mock_run(cmd, cwd=None, **kwargs):  # type: ignore[no-untyped-def]
            call_log.append(tuple(cmd))
            if cmd[:2] == ['git', 'rev-parse'] and cwd == wt_n:
                return (1, '', 'fatal: not a git repository')
            return await original_run(cmd, cwd=cwd, **kwargs)

        with (
            patch('orchestrator.merge_queue._run', new=mock_run),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req_n = _make_request('rp-n', 'rp-n', wt_n, config)
            req_ok = _make_request('rp-ok', 'rp-ok', wt_ok, config)
            await queue.put(req_n)
            await queue.put(req_ok)

            # rp-n must resolve as blocked with rev-parse reason
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'blocked', f'rp-n: {outcome_n}'
            assert 'rev-parse' in outcome_n.reason.lower(), (
                f'Expected rev-parse in reason: {outcome_n.reason}'
            )

            # rp-ok must still succeed (merger loop continues after the error)
            outcome_ok = await asyncio.wait_for(req_ok.result, timeout=30)
            assert outcome_ok.status == 'done', f'rp-ok: {outcome_ok}'

        await worker.stop()
        await worker_task

    async def test_merger_exception_resolves_inflight_future(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Unexpected exception in merger loop must resolve in-flight Future and continue.

        Without fix: if get_main_sha() raises after req is dequeued but before the
        SpeculativeItem is pushed to the verifier queue, the exception propagates to
        the outer try/finally which sends the sentinel but never resolves req.result.
        The caller hangs forever and the merger loop terminates.

        With fix: inner try/except Exception in the loop body resolves the in-flight
        req.result as 'blocked' (with the error message) and continues to the next
        request, keeping the merger loop alive.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'mef-n', 'file_mef_n.py', 'n = 1\n',
        )
        wt_ok = await _make_branch_with_file(
            git_ops, 'mef-ok', 'file_mef_ok.py', 'ok = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # get_main_sha raises RuntimeError on the first call, succeeds after
        original_get_main_sha = git_ops.get_main_sha
        call_count = 0

        async def failing_get_main_sha():  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError('Simulated get_main_sha failure')
            return await original_get_main_sha()

        with (
            patch.object(git_ops, 'get_main_sha', new=failing_get_main_sha),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req_n = _make_request('mef-n', 'mef-n', wt_n, config)
            req_ok = _make_request('mef-ok', 'mef-ok', wt_ok, config)
            await queue.put(req_n)
            await queue.put(req_ok)

            # mef-n must resolve as 'blocked' with reason mentioning the error
            # (not hang forever — that's the regression without the fix)
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'blocked', f'mef-n: {outcome_n}'
            assert 'Simulated get_main_sha failure' in outcome_n.reason, (
                f'Expected error message in reason, got: {outcome_n.reason}'
            )

            # mef-ok must succeed — merger loop continues after the per-request error
            outcome_ok = await asyncio.wait_for(req_ok.result, timeout=30)
            assert outcome_ok.status == 'done', f'mef-ok: {outcome_ok}'

        await worker.stop()
        await worker_task

    async def test_stop_drain_survives_cleanup_exception(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Cleanup exception during stop() verifier-queue drain must not orphan Futures.

        Without fix: if cleanup_merge_worktree raises for item1, the exception
        propagates out of the drain loop body, so item2's Future is never resolved —
        the caller hangs forever.

        With fix: cleanup is wrapped in contextlib.suppress(Exception), so the drain
        loop continues to item2 and resolves both Futures as 'blocked'.
        Covers review issue [exception_aborts_drain] at stop() ~line 367-376.
        """
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        # Do NOT start worker.run() — we test stop()'s drain logic directly.

        # Build two requests whose Futures we will check after stop().
        req1 = _make_request('drain-1', 'drain-1', git_ops.project_root, config)
        req2 = _make_request('drain-2', 'drain-2', git_ops.project_root, config)

        dummy_wt1 = git_ops.project_root / '.worktrees' / 'dummy1'
        dummy_wt2 = git_ops.project_root / '.worktrees' / 'dummy2'

        item1 = SpeculativeItem(
            request=req1, merge_result=None, merge_wt=dummy_wt1,
            base_sha='aaa', speculative=False, skip_verify=False,
        )
        item2 = SpeculativeItem(
            request=req2, merge_result=None, merge_wt=dummy_wt2,
            base_sha='bbb', speculative=False, skip_verify=False,
        )
        await worker._verifier_queue.put(item1)
        await worker._verifier_queue.put(item2)

        # First cleanup raises OSError; second succeeds.
        cleanup_calls: list[object] = []

        async def mock_cleanup(wt: object) -> None:
            cleanup_calls.append(wt)
            if len(cleanup_calls) == 1:
                raise OSError('disk full')

        with patch.object(git_ops, 'cleanup_merge_worktree', new=mock_cleanup):
            await worker.stop()

        # Both Futures must be resolved — cleanup failure must not abort the drain.
        assert req1.result.done(), 'req1 Future not resolved despite cleanup exception'
        assert req2.result.done(), 'req2 Future orphaned because drain loop aborted'
        assert req1.result.result().status == 'blocked'
        assert req2.result.result().status == 'blocked'
        # Second cleanup was still attempted despite first failure.
        assert len(cleanup_calls) == 2, (
            f'Expected 2 cleanup calls, got {len(cleanup_calls)}'
        )

    async def test_stop_race_resolves_inflight_merger_future(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """stop() must resolve Future for a request the merger is currently processing.

        Race condition: stop() drains both queues (empty), sends sentinels,
        asyncio.wait() times out while merger is still blocked inside merge_to_main.
        Verifier received its sentinel and has already exited.  When the merger
        eventually resumes and pushes a SpeculativeItem, the verifier is gone —
        the caller's Future is never resolved.

        With fix (step-35): after asyncio.wait() returns, stop() checks
        self._inflight_req.  If set and Future not done, resolves it as 'blocked'.
        The caller's Future is guaranteed to be resolved even if the merger was
        mid-operation when stop() was called.

        Covers review issue [race_condition_unresolved_future] at stop() ~line 350.
        """
        block_event = asyncio.Event()   # released after stop() returns
        merge_started = asyncio.Event() # set when merger enters merge_to_main

        original_merge = git_ops.merge_to_main

        async def blocking_merge(worktree: Path, branch: str, **kwargs: Any) -> Any:
            merge_started.set()
            await block_event.wait()  # simulates long-running merge
            return await original_merge(worktree, branch, **kwargs)

        wt = await _make_branch_with_file(
            git_ops, 'race-1', 'race_file.py', 'race = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        # Use a very short shutdown timeout so the test doesn't take 5 seconds.
        worker._shutdown_timeout = 0.1  # type: ignore[attr-defined]
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('race-1', 'race-1', wt, config)

        with (
            patch.object(git_ops, 'merge_to_main', new=blocking_merge),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            await queue.put(req)
            # Wait until the merger is definitely blocked inside merge_to_main.
            await asyncio.wait_for(merge_started.wait(), timeout=10)

            # stop() will time out (asyncio.wait) since merger is blocked.
            # Without fix: req.result is NOT done after stop() returns.
            # With fix: stop() checks _inflight_req and resolves it.
            await worker.stop()

        assert req.result.done(), (
            'Future must be resolved by stop() via _inflight_req check, '
            'even when merger was mid-operation'
        )
        assert req.result.result().status == 'blocked'

        # Release the merger so it can finish and worker_task can exit cleanly.
        block_event.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=15)

    async def test_speculative_chain_invalidation_propagates(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Chain invalidation must propagate: if N fails and N+1 is re-merged,
        N+2 (built speculatively on N+1's stale commit) must ALSO be re-merged.

        Scenario (depth-1 cap):
          - Queue has N, N+1, N+2 pre-loaded before worker starts.
          - Merger: merges N (non-spec), speculatively merges N+1 against N's
            merge commit, then awaits spec slot.
          - Verifier: N fails (has file_chain_n.py) → n_failed=True, releases slot.
          - Merger: grabs N+2, speculatively merges against N+1's STALE commit.
          - Verifier: N+1 discarded (n_failed=True), re-merged against actual main
            (no file_chain_n.py) → passes.  n_failed=False.  remerge_occurred=True.
          - Verifier: N+2 (speculative=True).
              WITHOUT FIX: n_failed=False → no discard → verification sees
              file_chain_n.py in speculative worktree → blocked.
              WITH FIX: remerge_occurred=True → discard → re-merge against actual
              main (only N+1, no N) → passes → done.

        Covers review issue [correctness_bug_in_speculative_chain_invalidation].
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'chain-n', 'file_chain_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'chain-n1', 'file_chain_n1.py', 'n1 = 2\n',
        )
        wt_n2 = await _make_branch_with_file(
            git_ops, 'chain-n2', 'file_chain_n2.py', 'n2 = 3\n',
        )

        # Pre-load all three so the Merger builds a 3-deep speculative chain:
        # N (non-spec), N+1 (spec against N's commit), N+2 (spec against N+1's
        # stale commit once the Verifier releases the slot after N fails).
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req_n = _make_request('chain-n', 'chain-n', wt_n, config)
        req_n1 = _make_request('chain-n1', 'chain-n1', wt_n1, config)
        req_n2 = _make_request('chain-n2', 'chain-n2', wt_n2, config)
        await queue.put(req_n)
        await queue.put(req_n1)
        await queue.put(req_n2)

        worker = SpeculativeMergeWorker(git_ops, queue)

        # Fail verification whenever file_chain_n.py is present (N's tainted code).
        # N's merge is non-spec → fail; N+2's speculative worktree descends from
        # N's commit → also has file_chain_n.py → would fail unless discarded first.
        async def _verify_chain(
            merge_wt, cfg, module_configs, task_files=None, **_kwargs,
        ):
            if (merge_wt / 'file_chain_n.py').exists():
                return MagicMock(passed=False, summary='N tainted: file_chain_n.py present')
            return MagicMock(passed=True, summary='')

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_verify_chain,
        ):
            worker_task = asyncio.create_task(worker.run())
            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)
            outcome_n2 = await asyncio.wait_for(req_n2.result, timeout=60)

        assert outcome_n.status == 'blocked', (
            f'N: expected blocked, got {outcome_n}'
        )
        assert outcome_n1.status == 'done', (
            f'N+1: expected done after re-merge against actual main, got {outcome_n1}'
        )
        assert outcome_n2.status == 'done', (
            f'N+2: expected done after chain-invalidation re-merge, got {outcome_n2}. '
            f'Without fix, N+2 is blocked because it was speculatively built on '
            f"N+1's stale commit (which contains file_chain_n.py from N)."
        )

        # Verify git state: N's tainted file must not be on main; N+1 and N+2 must be.
        _, ls_files, _ = await _run(
            ['git', 'ls-tree', '--name-only', 'main'], cwd=git_ops.project_root,
        )
        assert 'file_chain_n.py' not in ls_files, (
            'N (tainted) must not appear on main'
        )
        assert 'file_chain_n1.py' in ls_files, (
            'N+1 must appear on main after re-merge'
        )
        assert 'file_chain_n2.py' in ls_files, (
            'N+2 must appear on main after re-merge'
        )

        await worker.stop()
        await worker_task

    async def test_speculative_cas_failure_retries_until_advanced(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """CAS failure in SpeculativeMergeWorker retries and eventually succeeds.

        Mirrors MergeWorker.test_cas_failure_reenqueues_at_front but exercises
        the _verify_and_advance CAS-retry loop (which rebuilds SpeculativeItem
        with updated base_sha and tracks cumulative retries in _cas_retries).
        """
        wt = await _make_branch_with_file(
            git_ops, 'scas-ok', 'file_scas_ok.py', 'cas_ok = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        original_advance = git_ops.advance_main
        call_count = 0

        async def _fail_twice_then_succeed(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return 'cas_failed'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_fail_twice_then_succeed),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('scas-ok', 'scas-ok', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done', f'Expected done, got {outcome}'
        assert call_count == 3, f'Expected 3 advance_main calls (2 CAS fail + 1 success), got {call_count}'
        # _cas_retries should be cleaned up after success
        assert 'scas-ok' not in worker._cas_retries, (
            '_cas_retries not cleaned up after successful advance'
        )

        # File must appear on main
        _, content, _ = await _run(
            ['git', 'show', 'main:file_scas_ok.py'], cwd=git_ops.project_root,
        )
        assert 'cas_ok = 1' in content

        await worker.stop()
        await worker_task

    async def test_speculative_cas_retry_limit_exhausted(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """CAS failures beyond MAX_CAS_RETRIES resolve as blocked.

        Mirrors MergeWorker.test_cas_retry_limit_exhausted but exercises the
        SpeculativeMergeWorker's _verify_and_advance loop, which tracks retries
        in self._cas_retries (a per-task dict shared across calls).
        """
        wt = await _make_branch_with_file(
            git_ops, 'scas-lim', 'file_scas_lim.py', 'cas_lim = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _always_cas_fail(*args: Any, **kwargs: Any):
            return 'cas_failed'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_always_cas_fail),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('scas-lim', 'scas-lim', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked', f'Expected blocked, got {outcome}'
        assert 'cas retry limit' in outcome.reason.lower(), (
            f'Expected CAS retry limit message, got: {outcome.reason}'
        )
        # _cas_retries should be cleaned up after exhaustion
        assert 'scas-lim' not in worker._cas_retries, (
            '_cas_retries not cleaned up after retry limit exhausted'
        )

        await worker.stop()
        await worker_task

    @pytest.mark.parametrize('failure_code', ['not_descendant', 'contaminated', 'stash_failed'])
    async def test_speculative_permanent_failure_returns_blocked(
        self, git_ops: GitOps, config: OrchestratorConfig, failure_code: str,
    ):
        """Permanent advance_main failure codes block without retry.

        Mirrors MergeWorker.test_not_descendant_returns_blocked_immediately but
        exercises the SpeculativeMergeWorker's _verify_and_advance path (lines
        816-824 of merge_queue.py), which also cleans up merge worktree and
        resolves the Future.  Parameterized over all three permanent codes.
        """
        branch_name = f'sperm-{failure_code}'
        filename = f'file_sperm_{failure_code}.py'
        wt = await _make_branch_with_file(
            git_ops, branch_name, filename, f'{failure_code} = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        call_count = 0

        async def _return_failure(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            return failure_code

        with (
            patch.object(git_ops, 'advance_main', side_effect=_return_failure),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
            patch.object(git_ops, 'cleanup_merge_worktree', wraps=git_ops.cleanup_merge_worktree) as mock_cleanup,
        ):
            req = _make_request(branch_name, branch_name, wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked', f'Expected blocked for {failure_code}, got {outcome}'
        assert failure_code in outcome.reason, (
            f'Expected {failure_code} in reason, got: {outcome.reason}'
        )
        # Should only be called once — no retry for permanent failures
        assert call_count == 1, (
            f'Expected 1 advance_main call for permanent failure, got {call_count}'
        )
        # Worktree must have been cleaned up
        assert mock_cleanup.call_count >= 1, (
            f'cleanup_merge_worktree not called for {failure_code}'
        )
        # _cas_retries should be clean
        assert branch_name not in worker._cas_retries

        await worker.stop()
        await worker_task

    async def test_merger_post_merge_exception_cleans_worktree(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Post-merge exception (inside the success path) must clean up merge worktree.

        After merge_to_main succeeds, a live merge worktree exists.  An exception
        raised between lines 568-583 (assert, skip_verify calc, verifier queue put)
        is caught by the inner except Exception handler.  Without the fix the handler
        resolves the Future but never calls cleanup_merge_worktree — the worktree leaks.

        With the fix: cleanup is called (guarded by contextlib.suppress(Exception))
        before the Future is resolved.

        Scenario A: merge_commit=None triggers AssertionError at line 570.
        Scenario B: valid merge_commit but verifier-queue put raises RuntimeError.

        Covers review issue [resource_leak] at _merger_loop lines 568-614.
        """
        wt_a = await _make_branch_with_file(
            git_ops, 'pme-a', 'file_pme_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'pme-b', 'file_pme_b.py', 'b = 1\n',
        )
        wt_ok = await _make_branch_with_file(
            git_ops, 'pme-ok', 'file_pme_ok.py', 'ok = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        fake_wt_a = git_ops.project_root / '.worktrees' / '_merge-pme-a-fake'
        fake_wt_b = git_ops.project_root / '.worktrees' / '_merge-pme-b-fake'

        cleanup_calls: list[object] = []

        async def tracking_cleanup(wt: object) -> None:
            cleanup_calls.append(wt)

        # ── Scenario A: AssertionError from merge_commit=None ──────────────────
        a_result = MergeResult(success=True, merge_commit=None, merge_worktree=fake_wt_a)

        with (
            patch.object(git_ops, 'merge_to_main', AsyncMock(return_value=a_result)),
            patch.object(git_ops, 'cleanup_merge_worktree', new=tracking_cleanup),
        ):
            req_a = _make_request('pme-a', 'pme-a', wt_a, config)
            await queue.put(req_a)
            outcome_a = await asyncio.wait_for(req_a.result, timeout=30)

        assert outcome_a.status == 'blocked', f'Scenario A: expected blocked, got {outcome_a}'
        assert 'Merger error' in outcome_a.reason, (
            f'Scenario A: expected "Merger error" in reason, got: {outcome_a.reason!r}'
        )
        # The merge worktree must have been cleaned up despite the exception
        assert fake_wt_a in cleanup_calls, (
            f'Scenario A: cleanup_merge_worktree not called for fake_wt_a; '
            f'cleanup_calls={cleanup_calls}'
        )

        # ── Scenario B: RuntimeError from verifier-queue put ───────────────────
        # A valid merge_commit passes the assert; the put raises instead.
        b_merge_commit = 'ab' * 20  # 40-char fake SHA
        b_result = MergeResult(
            success=True, merge_commit=b_merge_commit, merge_worktree=fake_wt_b,
        )

        original_put = worker._verifier_queue.put
        b_put_count = 0

        async def sometimes_failing_put(item: SpeculativeItem | None) -> None:
            nonlocal b_put_count
            b_put_count += 1
            if b_put_count == 1 and isinstance(item, SpeculativeItem):
                raise RuntimeError('queue broken')
            await original_put(item)

        with (
            patch.object(git_ops, 'merge_to_main', AsyncMock(return_value=b_result)),
            patch.object(git_ops, 'cleanup_merge_worktree', new=tracking_cleanup),
            patch.object(worker._verifier_queue, 'put', new=sometimes_failing_put),
        ):
            req_b = _make_request('pme-b', 'pme-b', wt_b, config)
            await queue.put(req_b)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=30)

        assert outcome_b.status == 'blocked', f'Scenario B: expected blocked, got {outcome_b}'
        assert 'Merger error' in outcome_b.reason, (
            f'Scenario B: expected "Merger error" in reason, got: {outcome_b.reason!r}'
        )
        assert fake_wt_b in cleanup_calls, (
            f'Scenario B: cleanup_merge_worktree not called for fake_wt_b; '
            f'cleanup_calls={cleanup_calls}'
        )

        # ── Merger loop continues after both exceptions ──────────────────────
        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_ok = _make_request('pme-ok', 'pme-ok', wt_ok, config)
            await queue.put(req_ok)
            outcome_ok = await asyncio.wait_for(req_ok.result, timeout=30)

        assert outcome_ok.status == 'done', (
            f'Merger loop should continue after exceptions, got {outcome_ok}'
        )

        await worker.stop()
        await worker_task

    async def test_speculative_merge_worker_emits_duration_ms_on_done(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """SpeculativeMergeWorker emits duration_ms on the 'done' outcome.

        Exercises the verifier-phase emit at _verify_and_advance (done path).
        Asserts the merge_attempt event row has a non-null integer duration_ms.
        """
        db_path = tmp_path / 'events_spec_done.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt = await _make_branch_with_file(
            git_ops, 'sdur-done', 'sdur_done.py', 'sdur = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('sdur-done', 'sdur-done', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done', f'Expected done, got: {outcome}'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') AS outcome, duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        done_rows = [r for r in rows if r[0] == 'done']
        assert len(done_rows) == 1, f'Expected 1 done row, got: {rows}'
        assert done_rows[0][1] is not None, 'duration_ms should not be NULL'
        assert isinstance(done_rows[0][1], int), f'duration_ms should be int, got {type(done_rows[0][1])}'
        assert done_rows[0][1] >= 0, f'duration_ms should be >= 0, got {done_rows[0][1]}'

        await worker.stop()
        await worker_task

    async def test_speculative_merger_phase_emits_duration_ms(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Merger-phase emit sites set non-null duration_ms.

        Covers already_merged and conflict outcomes emitted from _merger_loop.
        """
        # --- Scenario A: already_merged (merger phase) ---
        db_a = tmp_path / 'events_sphase_a.db'
        es_a = EventStore(db_path=db_a, run_id='run-a')

        wt_n = await _make_branch_with_file(
            git_ops, 'sphase-n', 'sphase_n.py', 'sphase_n = 1\n',
        )
        # Create N+1 as already-merged: merge it first, then submit it
        wt_n1 = (await git_ops.create_worktree('sphase-n1')).path
        (wt_n1 / 'sphase_n1.py').write_text('sphase_n1 = 2\n')
        await git_ops.commit(wt_n1, 'Add sphase_n1.py')
        r = await git_ops.merge_to_main(wt_n1, 'sphase-n1')
        assert r.success
        assert r.merge_commit is not None
        await git_ops.advance_main(r.merge_commit)
        if r.merge_worktree:
            await git_ops.cleanup_merge_worktree(r.merge_worktree)

        q_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_a = SpeculativeMergeWorker(git_ops, q_a, event_store=es_a)
        wt_a = asyncio.create_task(w_a.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('sphase-n', 'sphase-n', wt_n, config)
            req_n1 = _make_request('sphase-n1', 'sphase-n1', wt_n1, config)
            await q_a.put(req_n)
            await q_a.put(req_n1)
            out_n = await asyncio.wait_for(req_n.result, timeout=30)
            out_n1 = await asyncio.wait_for(req_n1.result, timeout=30)

        assert out_n.status == 'done', f'N: {out_n}'
        assert out_n1.status == 'already_merged', f'N+1: {out_n1}'
        await w_a.stop()
        await wt_a

        conn = sqlite3.connect(str(db_a))
        rows_a = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_a), f'NULL duration_ms in already_merged scenario: {rows_a}'

        # --- Scenario B: conflict (merger phase) ---
        db_b = tmp_path / 'events_sphase_b.db'
        es_b = EventStore(db_path=db_b, run_id='run-b')

        wt_n2 = await _make_branch_with_file(
            git_ops, 'sphase-n2', 'sphase_n2.py', 'sphase_n2 = 1\n',
        )
        # Create a conflicting N+2 (README conflict)
        wt_cfl = (await git_ops.create_worktree('sphase-cfl')).path
        (git_ops.project_root / 'README.md').write_text('# conflict-src-sphase\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Conflict source sphase'], cwd=git_ops.project_root)
        (wt_cfl / 'README.md').write_text('# conflict-task-sphase\n')
        await git_ops.commit(wt_cfl, 'Conflict task sphase')

        q_b: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_b = SpeculativeMergeWorker(git_ops, q_b, event_store=es_b)
        wt_b = asyncio.create_task(w_b.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n2 = _make_request('sphase-n2', 'sphase-n2', wt_n2, config)
            req_cfl = _make_request('sphase-cfl', 'sphase-cfl', wt_cfl, config)
            await q_b.put(req_n2)
            await q_b.put(req_cfl)
            out_n2 = await asyncio.wait_for(req_n2.result, timeout=30)
            out_cfl = await asyncio.wait_for(req_cfl.result, timeout=30)

        assert out_n2.status == 'done', f'N2: {out_n2}'
        assert out_cfl.status == 'conflict', f'Conflict: {out_cfl}'
        await w_b.stop()
        await wt_b

        conn = sqlite3.connect(str(db_b))
        rows_b = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_b), f'NULL duration_ms in conflict scenario: {rows_b}'

    async def test_speculative_remerge_preserves_duration_ms(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """_remerge uses the original started_monotonic so duration_ms is realistic.

        The _remerge path is triggered when N fails verification and N+1 was
        speculatively merged. The verifier discards N+1's stale worktree and
        calls _remerge. If started_monotonic is correctly threaded through,
        the conflict emit inside _remerge yields a realistic duration (< 60s).
        If it falls back to the 0.0 default, duration would be huge (seconds
        since process start × 1000).

        Setup: N succeeds merge but fails verification, so N+1 is discarded
        and re-merged. We patch merge_to_main so the re-merge produces a
        conflict for N+1, causing a conflict emit inside _remerge.
        """
        db_path = tmp_path / 'events_remerge.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt_n = await _make_branch_with_file(
            git_ops, 'rmp-n', 'rmp_n.py', 'rmp_n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'rmp-n1', 'rmp_n1.py', 'rmp_n1 = 2\n',
        )

        # Track merge_to_main calls so we can return conflict on the re-merge
        original_merge = git_ops.merge_to_main
        merge_call_count = 0

        async def _controlled_merge(worktree, branch, **kwargs):
            nonlocal merge_call_count
            merge_call_count += 1
            # First two calls are speculative merges for N and N+1 (normal)
            # Third call is the re-merge for N+1 after N fails — return conflict
            if merge_call_count >= 3 and branch == 'rmp-n1':
                return MergeResult(
                    success=False,
                    conflicts=True,
                    details='simulated remerge conflict',
                    merge_commit=None,
                    merge_worktree=None,
                    pre_merge_sha=None,
                )
            return await original_merge(worktree, branch, **kwargs)

        async def _fail_n_pass_n1(merge_wt, cfg, module_configs, task_files=None):
            """Fail N's verification; N+1 re-merge won't reach verify (conflicts)."""
            n_present = (merge_wt / 'rmp_n.py').exists()
            if n_present:
                return MagicMock(passed=False, summary='N failed intentionally')
            return MagicMock(passed=True, summary='')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, 'merge_to_main', side_effect=_controlled_merge),
            patch('orchestrator.merge_queue.run_scoped_verification', side_effect=_fail_n_pass_n1),
        ):
            req_n = _make_request('rmp-n', 'rmp-n', wt_n, config)
            req_n1 = _make_request('rmp-n1', 'rmp-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)
            out_n = await asyncio.wait_for(req_n.result, timeout=30)
            out_n1 = await asyncio.wait_for(req_n1.result, timeout=30)

        # N should be blocked (verify failed); N+1 should be conflict (from _remerge)
        assert out_n.status == 'blocked', f'N: {out_n}'
        assert out_n1.status == 'conflict', f'N+1: {out_n1}'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        # All merge_attempt events must have non-null duration_ms
        assert all(r[1] is not None for r in rows), f'NULL duration_ms found: {rows}'

        # The conflict emit inside _remerge must have a realistic duration_ms
        # (0 to 60000 ms). If started_monotonic were 0.0 (default), the value
        # would be time-since-process-start * 1000 — many thousands of ms.
        conflict_rows = [r for r in rows if r[0] == 'conflict']
        assert len(conflict_rows) >= 1, f'Expected at least one conflict event: {rows}'
        for _outcome, dur in conflict_rows:
            assert 0 <= dur <= 60_000, (
                f'duration_ms={dur} is not realistic; '
                f'started_monotonic was likely not threaded through _remerge'
            )

        await worker.stop()
        await worker_task

    async def test_speculative_merge_worker_success_returns_merge_sha(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker success path: MergeOutcome.merge_sha is set.

        Submits one request, drives through the verify-and-advance path, and
        asserts the resulting MergeOutcome has status='done' with a 40-char
        merge commit SHA.  Fails initially because line ~1130 still constructs
        MergeOutcome('done') without merge_sha (step-5 guard; impl in step-6).
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'sspec-n', 'file_sspec_n.py', 'sspec_n = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            req_n = _make_request('sspec-task', 'sspec-n', wt_n, config)
            await queue.put(req_n)
            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)

        await worker.stop()
        await worker_task

        assert outcome_n.status == 'done', f'Expected done, got: {outcome_n}'
        assert outcome_n.merge_sha is not None, 'merge_sha must be set on done outcome'
        assert len(outcome_n.merge_sha) == 40, f'Expected 40-char SHA, got: {outcome_n.merge_sha!r}'
        assert all(c in '0123456789abcdef' for c in outcome_n.merge_sha), (
            f'merge_sha is not a hex string: {outcome_n.merge_sha!r}'
        )

    async def test_speculative_merge_worker_surfaces_unresolved_steps_in_dropped_outcome_reason(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker includes unresolved-step diagnostics in the blocked reason.

        Mirrors test_merge_worker_surfaces_unresolved_steps_in_dropped_outcome_reason
        but drives SpeculativeMergeWorker instead of MergeWorker.  Both worker
        paths must emit identical diagnostic text — downstream consumers (steward,
        dashboard) parse the same reason format regardless of which path produced it.

        Will fail until step 10 wires _format_unresolved_steps_suffix into
        SpeculativeMergeWorker._merger_loop.
        """
        wt = await _make_branch_with_file(
            git_ops, 'spec-drop-unresolved', 'kept.py', 'kept = 1\n',
        )
        artifacts = TaskArtifacts(wt)
        artifacts.init('spec-drop-unresolved', 'Spec drop unresolved', 'desc')
        artifacts.write_plan({
            'files': ['kept.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        bad_commit = 'cf82ae6815'

        async def _fake_drop_check_speculative(*_args, **_kwargs):
            return DropGuardResult(
                dropped=['dropped.py'],
                unresolved_steps=[
                    UnresolvedStep(
                        step_idx=2,
                        step_id='step-3',
                        commit=bad_commit,
                        rc=128,
                        stderr='fatal: bad object cf82ae6815',
                        object_missing=True,
                    )
                ],
            )

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass(),
        ), patch(
            'orchestrator.merge_queue._check_plan_targets_in_tree',
            _fake_drop_check_speculative,
        ):
            req = _make_request(
                'spec-drop-unresolved', 'spec-drop-unresolved', wt, config,
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)

        await worker.stop()
        await worker_task

        # (a) Still blocked (fail-closed)
        assert outcome.status == 'blocked', f'Expected blocked, got {outcome.status!r}'
        # (b) Existing drop-guard reason contract preserved
        assert 'dropped.py' in outcome.reason, f'Missing dropped.py in reason: {outcome.reason!r}'
        assert 'plan target' in outcome.reason.lower(), (
            f'Missing "plan target" in reason: {outcome.reason!r}'
        )
        # (c) Unresolved-step diagnostics included — same format as MergeWorker
        assert 'step-3' in outcome.reason, (
            f'Expected step_id "step-3" in reason: {outcome.reason!r}'
        )
        assert bad_commit in outcome.reason, (
            f'Expected commit {bad_commit!r} in reason: {outcome.reason!r}'
        )
        assert 'object_missing' in outcome.reason, (
            f'Expected "object_missing" in reason: {outcome.reason!r}'
        )
        reason_lower = outcome.reason.lower()
        assert 'drop-guard' in reason_lower or 'could not query' in reason_lower, (
            f'Expected drop-guard failure phrase in reason: {outcome.reason!r}'
        )


# ---------------------------------------------------------------------------
# TestMergeOutcomeDataclass — unit tests for MergeOutcome dataclass fields
# ---------------------------------------------------------------------------


class TestMergeOutcomeDataclass:
    def test_merge_outcome_has_merge_sha_field_default_none(self):
        """MergeOutcome.merge_sha defaults to None and can be set.

        Verifies that the field exists (step-1 / step-2 guard), that constructing
        MergeOutcome without the kwarg gives None, and that the field stores the
        value when supplied.
        """
        outcome_no_sha = MergeOutcome('done')
        assert outcome_no_sha.merge_sha is None  # type: ignore[attr-defined]

        outcome_with_sha = MergeOutcome('done', merge_sha='abc123')  # type: ignore[call-arg]
        assert outcome_with_sha.merge_sha == 'abc123'  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestSpeculativeItemDefaults — unit tests for SpeculativeItem field defaults
# ---------------------------------------------------------------------------


class TestSpeculativeItemDefaults:
    def test_started_monotonic_default_is_none(self):
        """SpeculativeItem.started_monotonic defaults to None when not passed.

        Ensures construction sites that omit started_monotonic produce NULL
        duration_ms (via _elapsed_ms) rather than a bogus time-since-process-start
        value derived from the 0.0 sentinel.

        Uses a MagicMock for request to avoid the asyncio.Future that
        MergeRequest.result requires (we only care about the dataclass default).
        """
        from unittest.mock import MagicMock

        from orchestrator.merge_queue import _elapsed_ms
        item = SpeculativeItem(
            request=MagicMock(),
            merge_result=None,
            merge_wt=None,
            base_sha='',
            speculative=False,
            skip_verify=False,
        )
        assert item.started_monotonic is None
        # Tie the default to the observability guarantee: None → NULL duration_ms
        assert _elapsed_ms(item.started_monotonic) is None


# ---------------------------------------------------------------------------
# TestEmitMergeAttemptHelper — unit tests for module-level _emit_merge_attempt
# ---------------------------------------------------------------------------


class TestEmitMergeAttemptHelper:
    def test_emit_merge_attempt_writes_row_without_attempt(
        self, tmp_path: Path,
    ):
        """Call with outcome and duration_ms — row has no 'attempt' key."""
        from orchestrator.merge_queue import _emit_merge_attempt

        db_path = tmp_path / 'eh_a.db'
        es = EventStore(db_path=db_path, run_id='eh-run')

        _emit_merge_attempt(es, 'task-1', 'conflict', duration_ms=42)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome'), "
            "       json_extract(data, '$.attempt'), "
            "       duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1, f'Expected 1 row, got: {rows}'
        outcome, attempt, dur = rows[0]
        assert outcome == 'conflict'
        assert attempt is None, f'Expected no attempt key, got {attempt!r}'
        assert dur == 42

    def test_emit_merge_attempt_writes_row_with_attempt(
        self, tmp_path: Path,
    ):
        """Call with outcome, attempt, and duration_ms — row includes 'attempt'."""
        from orchestrator.merge_queue import _emit_merge_attempt

        db_path = tmp_path / 'eh_b.db'
        es = EventStore(db_path=db_path, run_id='eh-run')

        _emit_merge_attempt(es, 'task-2', 'cas_retry', attempt=3, duration_ms=500)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome'), "
            "       json_extract(data, '$.attempt'), "
            "       duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1, f'Expected 1 row, got: {rows}'
        outcome, attempt, dur = rows[0]
        assert outcome == 'cas_retry'
        assert attempt == 3
        assert dur == 500

    def test_emit_merge_attempt_null_duration_when_none(
        self, tmp_path: Path,
    ):
        """Call with duration_ms=None — duration_ms column is NULL."""
        from orchestrator.merge_queue import _emit_merge_attempt

        db_path = tmp_path / 'eh_c.db'
        es = EventStore(db_path=db_path, run_id='eh-run')

        _emit_merge_attempt(es, 'task-3', 'done', duration_ms=None)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1, f'Expected 1 row, got: {rows}'
        outcome, dur = rows[0]
        assert outcome == 'done'
        assert dur is None, f'Expected NULL duration_ms, got {dur!r}'

    def test_emit_merge_attempt_calls_emit_when_store_provided(self):
        """Call with a real (mock) store — emit is invoked exactly once."""
        from unittest.mock import MagicMock

        from orchestrator.merge_queue import _emit_merge_attempt

        mock_es = MagicMock()
        _emit_merge_attempt(mock_es, 'task-check', 'done', duration_ms=1)
        mock_es.emit.assert_called_once()

    def test_emit_merge_attempt_noop_when_event_store_is_none(self):
        """Call with event_store=None — no exception, emit never invoked."""
        from unittest.mock import MagicMock

        from orchestrator.merge_queue import _emit_merge_attempt

        mock_es = MagicMock()
        _emit_merge_attempt(None, 'task-4', 'done', duration_ms=1)
        mock_es.emit.assert_not_called()


# ---------------------------------------------------------------------------
# TestSpeculativeBackwardCompat — step-17
# Run key MergeWorker scenarios through SpeculativeMergeWorker to confirm
# they behave identically with queue depth 1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculativeBackwardCompat:
    async def test_basic_merge(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree = (await git_ops.create_worktree('compat-basic')).path
        (worktree / 'compat.py').write_text('compat = True\n')
        await git_ops.commit(worktree, 'Add compat file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('compat-1', 'compat-basic', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        assert result.status == 'done'
        _, content, _ = await _run(
            ['git', 'show', 'main:compat.py'], cwd=git_ops.project_root,
        )
        assert 'compat = True' in content

        await worker.stop()
        await worker_task

    async def test_already_merged(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree = (await git_ops.create_worktree('compat-am')).path
        (worktree / 'am.py').write_text('am = True\n')
        await git_ops.commit(worktree, 'Add am file')

        result = await git_ops.merge_to_main(worktree, 'compat-am')
        assert result.success
        assert result.merge_commit is not None
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('compat-am', 'compat-am', worktree, config)
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=10)
        assert outcome.status == 'already_merged'

        await worker.stop()
        await worker_task

    async def test_verify_failure(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree = (await git_ops.create_worktree('compat-vf')).path
        (worktree / 'bad.py').write_text('bad = True\n')
        await git_ops.commit(worktree, 'Add bad file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        mock_verify = AsyncMock(return_value=MagicMock(passed=False, summary='tests failed'))

        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify):
            req = _make_request('compat-vf', 'compat-vf', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'verification failed' in outcome.reason.lower()

        await worker.stop()
        await worker_task


# ---------------------------------------------------------------------------
# TestMergeVerifyColdTimeout — Fix #1
# Merge worktrees are freshly created per merge (no warm cargo cache) but
# lack .task/ (only .taskmaster/), so _is_verify_cold mis-classifies them as
# warm.  The merge-queue call sites must pass is_merge_verify=True so the
# cold-track timeout applies.  These tests assert the kwarg is threaded
# through both MergeWorker and SpeculativeMergeWorker.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeVerifyColdTimeout:
    async def test_merge_worker_passes_is_merge_verify_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker's verify call must set is_merge_verify=True.

        The legacy serial worker is preserved for compat; even though
        SpeculativeMergeWorker is the default production path, this flag
        must still flow through here so tests/eval/debug harnesses that
        opt back into the serial worker also get the cold timeout.
        """
        worktree = (await git_ops.create_worktree('merge-cold-mw')).path
        (worktree / 'coldmw.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add coldmw')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        captured_kwargs: list[dict] = []

        async def spy_verify(*args, **kwargs):
            captured_kwargs.append(kwargs)
            result = AsyncMock()
            result.passed = True
            result.summary = ''
            result.timed_out = False
            return result

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=spy_verify,
        ):
            req = _make_request('cold-mw', 'merge-cold-mw', worktree, config)
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        await worker_task

        assert captured_kwargs, 'run_scoped_verification was not invoked'
        assert captured_kwargs[0].get('is_merge_verify') is True, (
            f'merge-queue verify must pass is_merge_verify=True; '
            f'got {captured_kwargs[0]!r}'
        )

    async def test_speculative_worker_passes_is_merge_verify_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker's verify call must set is_merge_verify=True.

        This is the production path — it's the call site that was
        mis-classifying merge worktrees as warm and blowing the 30-min
        timeout on each post-merge verify against reify.
        """
        wt = await _make_branch_with_file(
            git_ops, 'merge-cold-spec', 'coldspec.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        captured_kwargs: list[dict] = []

        async def spy_verify(*args, **kwargs):
            captured_kwargs.append(kwargs)
            result = AsyncMock()
            result.passed = True
            result.summary = ''
            result.timed_out = False
            return result

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=spy_verify,
        ):
            req = _make_request('cold-spec', 'merge-cold-spec', wt, config)
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        await worker_task

        assert captured_kwargs, 'run_scoped_verification was not invoked'
        assert captured_kwargs[0].get('is_merge_verify') is True, (
            f'speculative merge-queue verify must pass is_merge_verify=True; '
            f'got {captured_kwargs[0]!r}'
        )


# ---------------------------------------------------------------------------
# TestMergeVerifyTimeoutLoopBreaker — Fix #2
# After MAX_POST_MERGE_VERIFY_TIMEOUTS consecutive post-merge verify
# TIMEOUTS for the same task, the merge queue must stop running merge+verify
# and return a ``blocked`` outcome with the ABANDONED_REASON_PREFIX.  Real
# (non-timeout) verify failures must NOT feed the counter, and a successful
# merge must reset the counter.
# ---------------------------------------------------------------------------


def _mock_verify_timeout():
    """Return a mock that makes run_scoped_verification time out."""
    async def _fake(*args, **kwargs):
        result = AsyncMock()
        result.passed = False
        result.summary = 'Verification timed out'
        result.timed_out = True
        result.failure_report = lambda: '## Verify Timed Out\n\n(mock)'
        return result
    return _fake


@pytest.mark.asyncio
class TestMergeVerifyTimeoutLoopBreaker:
    async def test_merge_worker_abandons_after_threshold(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """N consecutive verify timeouts → next submission blocked without verify.

        Submits the same task_id MAX+1 times.  The first MAX submissions
        run merge+verify and surface a blocked/timeout outcome.  The next
        submission must short-circuit: no merge, no verify, blocked
        outcome with ABANDONED_REASON_PREFIX.
        """
        from orchestrator.merge_queue import ABANDONED_REASON_PREFIX

        wt = await _make_branch_with_file(
            git_ops, 'loop-break-mw', 'lb.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        assert worker.MAX_POST_MERGE_VERIFY_TIMEOUTS == 2
        worker_task = asyncio.create_task(worker.run())

        verify_call_count = 0

        async def counting_timeout_verify(*args, **kwargs):
            nonlocal verify_call_count
            verify_call_count += 1
            result = AsyncMock()
            result.passed = False
            result.summary = 'Verification timed out'
            result.timed_out = True
            result.failure_report = lambda: ''
            return result

        try:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                side_effect=counting_timeout_verify,
            ):
                # Submissions 1..MAX: run merge+verify, surface timeout.
                outcomes: list[MergeOutcome] = []
                for _ in range(worker.MAX_POST_MERGE_VERIFY_TIMEOUTS):
                    req = _make_request('lb-task', 'loop-break-mw', wt, config)
                    await queue.put(req)
                    outcomes.append(await asyncio.wait_for(req.result, timeout=30))

                # Every one of those must be blocked with the verify-failed reason.
                for o in outcomes:
                    assert o.status == 'blocked'
                    assert 'verification failed' in o.reason.lower()

                verify_calls_before_loopbreak = verify_call_count
                assert verify_calls_before_loopbreak == worker.MAX_POST_MERGE_VERIFY_TIMEOUTS

                # Submission MAX+1: must short-circuit BEFORE invoking verify.
                req_final = _make_request(
                    'lb-task', 'loop-break-mw', wt, config,
                )
                await queue.put(req_final)
                final = await asyncio.wait_for(req_final.result, timeout=10)

            assert final.status == 'blocked'
            assert final.reason.startswith(ABANDONED_REASON_PREFIX), (
                f'Expected abandoned reason prefix; got {final.reason!r}'
            )
            # Crucially: verify was NOT invoked again on the abandoned path.
            assert verify_call_count == verify_calls_before_loopbreak, (
                f'Abandoned submission must not invoke verify; '
                f'before={verify_calls_before_loopbreak} after={verify_call_count}'
            )
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

    async def test_speculative_worker_abandons_after_threshold(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker loop-breaker — same contract as MergeWorker."""
        from orchestrator.merge_queue import ABANDONED_REASON_PREFIX

        wt = await _make_branch_with_file(
            git_ops, 'loop-break-spec', 'lbs.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        assert worker.MAX_POST_MERGE_VERIFY_TIMEOUTS == 2
        worker_task = asyncio.create_task(worker.run())

        verify_call_count = 0

        async def counting_timeout_verify(*args, **kwargs):
            nonlocal verify_call_count
            verify_call_count += 1
            result = AsyncMock()
            result.passed = False
            result.summary = 'Verification timed out'
            result.timed_out = True
            result.failure_report = lambda: ''
            return result

        try:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                side_effect=counting_timeout_verify,
            ):
                for _ in range(worker.MAX_POST_MERGE_VERIFY_TIMEOUTS):
                    req = _make_request('lbs-task', 'loop-break-spec', wt, config)
                    await queue.put(req)
                    outcome = await asyncio.wait_for(req.result, timeout=30)
                    assert outcome.status == 'blocked'

                verify_calls_before_loopbreak = verify_call_count
                assert verify_calls_before_loopbreak == worker.MAX_POST_MERGE_VERIFY_TIMEOUTS

                req_final = _make_request(
                    'lbs-task', 'loop-break-spec', wt, config,
                )
                await queue.put(req_final)
                final = await asyncio.wait_for(req_final.result, timeout=10)

            assert final.status == 'blocked'
            assert final.reason.startswith(ABANDONED_REASON_PREFIX)
            assert verify_call_count == verify_calls_before_loopbreak
        finally:
            await worker.stop()
            await worker_task

    async def test_non_timeout_failure_does_not_feed_counter(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Real (non-timeout) verify failures must NOT count toward the budget.

        Submits the same task twice with a real test-failure result and
        then a third time — the third submission must still run merge+verify
        (not abandon), because the counter only advances on timed_out=True.
        """
        wt = await _make_branch_with_file(
            git_ops, 'loop-real-fail', 'lrf.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        call_count = 0

        async def real_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = AsyncMock()
            result.passed = False
            result.summary = 'tests failed'
            result.timed_out = False
            result.failure_report = lambda: ''
            return result

        try:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                side_effect=real_failure,
            ):
                # Submit 3 times — more than the abandon threshold (2).
                # Real failures must not abandon; all 3 must run verify.
                for _ in range(worker.MAX_POST_MERGE_VERIFY_TIMEOUTS + 1):
                    req = _make_request('rf-task', 'loop-real-fail', wt, config)
                    await queue.put(req)
                    outcome = await asyncio.wait_for(req.result, timeout=30)
                    assert outcome.status == 'blocked'
                    assert 'verification failed' in outcome.reason.lower()
                    # Must NOT be the abandoned-reason prefix.
                    assert not outcome.reason.startswith(
                        'Post-merge verify timed out'
                    ), (
                        f'Real failure must not produce abandoned reason; '
                        f'got {outcome.reason!r}'
                    )

            assert call_count == worker.MAX_POST_MERGE_VERIFY_TIMEOUTS + 1, (
                f'Every submission must run verify when failures are real; '
                f'got {call_count} verify calls'
            )
            # Counter must be zero (real failures never bumped it).
            assert worker._post_merge_verify_timeouts.get('rf-task', 0) == 0
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

    async def test_success_resets_counter(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """A successful merge clears the counter so future timeouts start fresh.

        Injects 1 timeout (under the threshold of 2), then a success, and
        asserts the counter is reset to zero.  A separate assert on the
        dict keeps this test decoupled from the ``_abandon_outcome`` path.
        """
        # First task: time out once.
        wt_fail = await _make_branch_with_file(
            git_ops, 'reset-fail', 'rf.py', 'x = 1\n',
        )
        # Second task: same task_id, but arrange verify to pass.
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        verify_pass_once_after_timeout = {'first_call': True}

        async def timeout_then_pass(*args, **kwargs):
            result = AsyncMock()
            if verify_pass_once_after_timeout['first_call']:
                verify_pass_once_after_timeout['first_call'] = False
                result.passed = False
                result.summary = 'Verification timed out'
                result.timed_out = True
                result.failure_report = lambda: ''
            else:
                result.passed = True
                result.summary = ''
                result.timed_out = False
            return result

        try:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                side_effect=timeout_then_pass,
            ):
                # First submission → verify times out → counter = 1.
                req1 = _make_request('reset-task', 'reset-fail', wt_fail, config)
                await queue.put(req1)
                r1 = await asyncio.wait_for(req1.result, timeout=30)
                assert r1.status == 'blocked'
                assert worker._post_merge_verify_timeouts.get('reset-task') == 1

                # Second submission for a *different* branch that merges cleanly,
                # same task_id → verify passes → counter cleared.
                wt_ok = await _make_branch_with_file(
                    git_ops, 'reset-ok', 'ro.py', 'x = 2\n',
                )
                req2 = _make_request('reset-task', 'reset-ok', wt_ok, config)
                await queue.put(req2)
                r2 = await asyncio.wait_for(req2.result, timeout=30)
                assert r2.status == 'done'

            # Counter must have been cleared by the successful merge.
            assert 'reset-task' not in worker._post_merge_verify_timeouts
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task


# ---------------------------------------------------------------------------
# TestWipHalt — WIP-safe merge queue halt mechanism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWipHaltMergeWorker:
    async def test_wip_halted_blocks_subsequent_tasks(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """wip_overlap halts queue; second request stays pending until unhalt."""
        wt1 = await _make_branch_with_file(
            git_ops, 'halt-1', 'file_halt_1.py', 'halt1 = 1\n',
        )
        wt2 = await _make_branch_with_file(
            git_ops, 'halt-2', 'file_halt_2.py', 'halt2 = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # advance_main returns wip_overlap for first request
        call_count = 0
        original_advance = git_ops.advance_main

        async def _wip_overlap_then_normal(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                git_ops._last_overlap_files = ['file_halt_1.py']
                return 'wip_overlap'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_wip_overlap_then_normal),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req1 = _make_request('halt-1', 'halt-1', wt1, config)
            await queue.put(req1)
            outcome1 = await asyncio.wait_for(req1.result, timeout=30)

        assert outcome1.status == 'wip_halted'
        assert outcome1.overlap_files == ['file_halt_1.py']
        assert worker.is_wip_halted

        # Second request: put it in queue, it should NOT resolve while halted
        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req2 = _make_request('halt-2', 'halt-2', wt2, config)
            await queue.put(req2)

            # Give the worker a chance to process (it shouldn't, it's halted)
            await asyncio.sleep(0.2)
            assert not req2.result.done(), 'Second request resolved while queue was halted'

            # Un-halt the queue
            worker.unhalt_wip()
            assert not worker.is_wip_halted

            # Now the second request should resolve
            outcome2 = await asyncio.wait_for(req2.result, timeout=30)

        assert outcome2.status == 'done'

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_done_wip_recovery_outcome(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict returns done_wip_recovery with recovery branch info."""
        wt = await _make_branch_with_file(
            git_ops, 'recov-1', 'file_recov.py', 'recov = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict(*args, **kwargs):
            git_ops._last_recovery_branch = 'wip/recovery-recov-1-20260407T120000'
            return 'pop_conflict'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('recov-1', 'recov-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done_wip_recovery'
        assert outcome.recovery_branch == 'wip/recovery-recov-1-20260407T120000'
        assert worker.is_wip_halted

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_unmerged_state_returns_unmerged_state_and_halts(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """unmerged_state: MergeWorker returns 'unmerged_state' status and halts the queue."""
        wt = await _make_branch_with_file(
            git_ops, 'uu-mw-1', 'file_uu_mw.py', 'uu_mw = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _unmerged_state(*args: Any, **kwargs: Any):
            return 'unmerged_state'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_unmerged_state),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('uu-mw-1', 'uu-mw-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'unmerged_state'
        assert 'unmerged' in outcome.reason.lower()
        assert worker.is_wip_halted

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_pop_conflict_no_advance_returns_wip_recovery_no_advance_and_halts(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict_no_advance: MergeWorker returns wip_recovery_no_advance and halts."""
        wt = await _make_branch_with_file(
            git_ops, 'pcna-mw-1', 'file_pcna_mw.py', 'pcna_mw = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict_no_advance(*args: Any, **kwargs: Any):
            git_ops._last_recovery_branch = 'wip/recovery-x-y'
            return 'pop_conflict_no_advance'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict_no_advance),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('pcna-mw-1', 'pcna-mw-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'wip_recovery_no_advance'
        assert outcome.recovery_branch == 'wip/recovery-x-y'
        assert 'did not advance' in outcome.reason.lower()
        assert worker.is_wip_halted

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task


@pytest.mark.asyncio
class TestWipHaltSpeculativeMergeWorker:
    async def test_wip_halted_blocks_subsequent_tasks(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """wip_overlap in speculative worker halts queue; unhalt resumes.

        Submit req1 alone (no speculative look-ahead) so the merger loop
        reaches _wip_halt.wait() before req2 enters the queue.
        """
        wt1 = await _make_branch_with_file(
            git_ops, 'shalt-1', 'file_shalt_1.py', 'shalt1 = 1\n',
        )
        wt2 = await _make_branch_with_file(
            git_ops, 'shalt-2', 'file_shalt_2.py', 'shalt2 = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        call_count = 0
        original_advance = git_ops.advance_main

        async def _wip_overlap_then_normal(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                git_ops._last_overlap_files = ['file_shalt_1.py']
                return 'wip_overlap'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_wip_overlap_then_normal),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            # Submit req1 alone — no req2 in queue, so no speculative look-ahead
            req1 = _make_request('shalt-1', 'shalt-1', wt1, config)
            await queue.put(req1)
            outcome1 = await asyncio.wait_for(req1.result, timeout=30)

            assert outcome1.status == 'wip_halted'
            assert outcome1.overlap_files == ['file_shalt_1.py']
            assert worker.is_wip_halted

            # Now submit req2 — merger is blocked at _wip_halt.wait()
            req2 = _make_request('shalt-2', 'shalt-2', wt2, config)
            await queue.put(req2)
            await asyncio.sleep(0.3)
            assert not req2.result.done(), 'Second request resolved while queue was halted'

            # Un-halt and wait for req2
            worker.unhalt_wip()
            outcome2 = await asyncio.wait_for(req2.result, timeout=30)

        assert outcome2.status == 'done'

        await worker.stop()
        await worker_task

    async def test_done_wip_recovery_outcome(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict in speculative worker returns done_wip_recovery and halts."""
        wt = await _make_branch_with_file(
            git_ops, 'srecov-1', 'file_srecov.py', 'srecov = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict(*args, **kwargs):
            git_ops._last_recovery_branch = 'wip/recovery-srecov-1-20260407T120000'
            return 'pop_conflict'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('srecov-1', 'srecov-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done_wip_recovery'
        assert outcome.recovery_branch == 'wip/recovery-srecov-1-20260407T120000'
        assert worker.is_wip_halted

        await worker.stop()
        await worker_task

    async def test_speculative_unmerged_state_returns_unmerged_state_and_halts(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """unmerged_state in SpeculativeMergeWorker returns 'unmerged_state' status and halts."""
        wt = await _make_branch_with_file(
            git_ops, 'uu-sw-1', 'file_uu_sw.py', 'uu_sw = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _unmerged_state(*args: Any, **kwargs: Any):
            return 'unmerged_state'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_unmerged_state),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('uu-sw-1', 'uu-sw-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'unmerged_state'
        assert 'unmerged' in outcome.reason.lower()
        assert worker.is_wip_halted

        await worker.stop()
        await worker_task

    async def test_speculative_pop_conflict_no_advance_outcome(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict_no_advance in SpeculativeMergeWorker returns wip_recovery_no_advance."""
        wt = await _make_branch_with_file(
            git_ops, 'pcna-sw-1', 'file_pcna_sw.py', 'pcna_sw = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict_no_advance(*args: Any, **kwargs: Any):
            git_ops._last_recovery_branch = 'wip/recovery-x-y'
            return 'pop_conflict_no_advance'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict_no_advance),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('pcna-sw-1', 'pcna-sw-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'wip_recovery_no_advance'
        assert outcome.recovery_branch == 'wip/recovery-x-y'
        assert 'did not advance' in outcome.reason.lower()
        assert worker.is_wip_halted

        await worker.stop()
        await worker_task


@pytest.mark.parametrize(
    'worker_cls', [MergeWorker, SpeculativeMergeWorker],
)
class TestHaltOwnerMechanics:
    """Halt-owner pointer: single source of truth for resolve-callback un-halt.

    Both MergeWorker and SpeculativeMergeWorker implement the same contract.
    These tests exercise the mechanics directly — no merge flow, just the
    halt-owner state machine. Integration is covered in test_workflow_e2e.
    """

    def test_fresh_worker_has_no_halt_owner(
        self, worker_cls, git_ops: GitOps,
    ):
        """Freshly constructed worker: not halted, owner is None, is_halt_owner is False."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        assert not worker.is_wip_halted
        assert worker.is_halt_owner('any-id') is False
        assert worker._halt_owner_esc_id is None

    def test_halt_for_wip_clears_owner(
        self, worker_cls, git_ops: GitOps,
    ):
        """halt_for_wip sets the halt flag and clears owner (workflow registers after)."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('test reason')
        assert worker.is_wip_halted
        assert worker._halt_owner_esc_id is None
        assert worker.is_halt_owner('any-id') is False

    def test_set_halt_owner_registers_id(
        self, worker_cls, git_ops: GitOps,
    ):
        """set_halt_owner records the id; is_halt_owner matches on equality only."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('test reason')
        worker.set_halt_owner('esc-42-1')

        assert worker.is_halt_owner('esc-42-1') is True
        assert worker.is_halt_owner('esc-42-2') is False
        assert worker.is_halt_owner('esc-99-1') is False

    def test_set_halt_owner_rejects_double_register(
        self, worker_cls, git_ops: GitOps,
    ):
        """set_halt_owner raises when owner is already set — catches double-halt bugs."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('test reason')
        worker.set_halt_owner('esc-42-1')

        with pytest.raises(AssertionError, match='halt owner already set'):
            worker.set_halt_owner('esc-42-2')

    def test_unhalt_wip_clears_owner(
        self, worker_cls, git_ops: GitOps,
    ):
        """unhalt_wip releases the halt and clears the owner pointer."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('test reason')
        worker.set_halt_owner('esc-42-1')
        worker.unhalt_wip()

        assert not worker.is_wip_halted
        assert worker._halt_owner_esc_id is None
        assert worker.is_halt_owner('esc-42-1') is False

    def test_halt_cycle_allows_reuse(
        self, worker_cls, git_ops: GitOps,
    ):
        """After a full halt→unhalt cycle, a new owner can be registered."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('first')
        worker.set_halt_owner('esc-1-1')
        worker.unhalt_wip()

        worker.halt_for_wip('second')
        worker.set_halt_owner('esc-2-1')
        assert worker.is_halt_owner('esc-2-1') is True
        assert worker.is_halt_owner('esc-1-1') is False


# ---------------------------------------------------------------------------
# TestEnqueueMergeRequest — step-3
# ---------------------------------------------------------------------------


class TestEnqueueMergeRequest:
    """Tests for the module-level enqueue_merge_request helper."""

    @pytest.mark.asyncio
    async def test_enqueue_helper_emits_merge_queued_and_puts_on_queue(
        self, tmp_path: Path, config: OrchestratorConfig,
    ):
        """enqueue_merge_request emits merge_queued and places req on queue."""
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs.db'
        event_store = EventStore(db_path, 'run-1')

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('42', 'task/42', wt, config)

        await enqueue_merge_request(queue, req, event_store)

        # Queue has exactly one item which is our req
        assert queue.qsize() == 1
        dequeued = queue.get_nowait()
        assert dequeued is req

        # Exactly one merge_queued row in events
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, phase, "
            "json_extract(data, '$.branch') AS branch "
            "FROM events WHERE event_type = 'merge_queued'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'merge_queued'
        assert rows[0][1] == '42'
        assert rows[0][2] == 'merge'
        assert rows[0][3] == 'task/42'

    @pytest.mark.asyncio
    async def test_enqueue_helper_with_none_event_store_still_enqueues(
        self, tmp_path: Path, config: OrchestratorConfig,
    ):
        """Passing event_store=None must still enqueue and not raise."""
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('99', 'task/99', wt, config)

        await enqueue_merge_request(queue, req, None)

        assert queue.qsize() == 1
        dequeued = queue.get_nowait()
        assert dequeued is req


# ---------------------------------------------------------------------------
# TestMergeWorkerDequeueEvent — step-5
# ---------------------------------------------------------------------------


class TestMergeWorkerDequeueEvent:
    """MergeWorker emits merge_dequeued after dequeuing a request."""

    @pytest.mark.asyncio
    async def test_merge_worker_emits_merge_dequeued_after_dequeue(
        self, tmp_path: Path, config: OrchestratorConfig, git_ops: GitOps,
    ):
        """MergeWorker emits merge_dequeued after pulling request from queue.

        Timestamp of merge_dequeued must be >= merge_queued timestamp.
        """
        from orchestrator.merge_queue import enqueue_merge_request

        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('42', 'task/42', wt, config)

        # Patch _do_merge so it immediately returns 'done' without git ops
        async def _fast_done(req):
            return MergeOutcome('done')

        worker_task = asyncio.create_task(worker.run())
        with patch.object(worker, '_do_merge', side_effect=_fast_done):
            await enqueue_merge_request(queue, req, event_store)
            outcome = await asyncio.wait_for(req.result, timeout=10)

        assert outcome.status == 'done'

        conn = sqlite3.connect(str(db_path))
        dequeued_rows = conn.execute(
            "SELECT event_type, task_id, timestamp FROM events "
            "WHERE event_type = 'merge_dequeued'"
        ).fetchall()
        queued_rows = conn.execute(
            "SELECT timestamp FROM events WHERE event_type = 'merge_queued'"
        ).fetchall()
        conn.close()

        assert len(dequeued_rows) == 1, f'Expected 1 merge_dequeued row, got: {dequeued_rows}'
        assert dequeued_rows[0][1] == '42'
        # merge_dequeued timestamp must be >= merge_queued timestamp
        assert len(queued_rows) == 1
        assert dequeued_rows[0][2] >= queued_rows[0][0]

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# TestSpeculativeMergeWorkerDequeueEvent — step-7
# ---------------------------------------------------------------------------


class TestSpeculativeMergeWorkerDequeueEvent:
    """SpeculativeMergeWorker emits merge_dequeued after dequeuing a request."""

    @pytest.mark.asyncio
    async def test_speculative_worker_emits_merge_dequeued_after_dequeue(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """SpeculativeMergeWorker emits merge_dequeued after dequeuing.

        Uses an immediate conflict path (merge_to_main returns conflicts=True)
        so the test is fast and doesn't need real git merge work.
        """
        from orchestrator.git_ops import MergeResult
        from orchestrator.merge_queue import enqueue_merge_request

        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        # Use a real worktree so rev-parse HEAD works
        wt = await _make_branch_with_file(
            git_ops, 'spec-deq', 'spec_deq.py', 'x = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker._shutdown_timeout = 2.0

        # Force an immediate conflict so the merger resolves quickly
        conflict_result = MergeResult(
            success=False, conflicts=True, details='conflict',
            merge_worktree=None, merge_commit=None, pre_merge_sha=None,
        )

        worker_task = asyncio.create_task(worker.run())
        with patch.object(git_ops, 'merge_to_main', return_value=conflict_result):
            req = _make_request('spec-deq', 'spec-deq', wt, config)
            await enqueue_merge_request(queue, req, event_store)
            outcome = await asyncio.wait_for(req.result, timeout=10)

        assert outcome.status == 'conflict'

        conn = sqlite3.connect(str(db_path))
        dequeued_rows = conn.execute(
            "SELECT event_type, task_id, timestamp FROM events "
            "WHERE event_type = 'merge_dequeued'"
        ).fetchall()
        queued_rows = conn.execute(
            "SELECT timestamp FROM events WHERE event_type = 'merge_queued'"
        ).fetchall()
        conn.close()

        assert len(dequeued_rows) == 1, f'Expected 1 merge_dequeued row, got: {dequeued_rows}'
        assert dequeued_rows[0][1] == 'spec-deq'
        # merge_dequeued timestamp must be >= merge_queued timestamp
        assert len(queued_rows) == 1
        assert dequeued_rows[0][2] >= queued_rows[0][0]

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# TestMergeWorkerCasRetryEmitsMergeQueued — step-9
# ---------------------------------------------------------------------------


class TestMergeWorkerCasRetryEmitsMergeQueued:
    """MergeWorker emits merge_queued when re-enqueuing on CAS retry."""

    @pytest.mark.asyncio
    async def test_cas_retry_reenqueue_emits_merge_queued(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """CAS retry path emits a second merge_queued, then merge_dequeued, then done.

        Event sequence expected:
          merge_queued        (initial enqueue via helper)
          merge_dequeued      (worker picks up request the first time)
          merge_attempt(cas_retry)
          merge_queued        (re-enqueue on CAS failure)
          merge_dequeued      (worker picks up from _urgent)
          merge_attempt(done)
        """
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt = await _make_branch_with_file(
            git_ops, 'cas-evt', 'cas_evt.py', 'x = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        original_advance = git_ops.advance_main
        call_count = 0

        async def _fail_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 'cas_failed'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_fail_once),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('cas-evt', 'cas-evt', wt, config)
            from orchestrator.merge_queue import enqueue_merge_request
            await enqueue_merge_request(queue, req, event_store)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done'
        assert call_count == 2

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, json_extract(data, '$.outcome') AS outcome "
            "FROM events ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [(r[0], r[1]) for r in rows]

        # Count merge_queued rows for this task — expect exactly 2
        queued_count = sum(1 for et, _ in event_types if et == 'merge_queued')
        assert queued_count == 2, f'Expected 2 merge_queued rows, got: {event_types}'

        # Count merge_dequeued rows — expect exactly 2
        dequeued_count = sum(1 for et, _ in event_types if et == 'merge_dequeued')
        assert dequeued_count == 2, f'Expected 2 merge_dequeued rows, got: {event_types}'

        # Exactly one cas_retry and one done attempt
        attempt_outcomes = [out for et, out in event_types if et == 'merge_attempt']
        assert 'cas_retry' in attempt_outcomes, f'Expected cas_retry in attempts: {attempt_outcomes}'
        assert 'done' in attempt_outcomes, f'Expected done in attempts: {attempt_outcomes}'

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# TestWorkflowSubmitUsesEnqueueHelper — step-11 test
# ---------------------------------------------------------------------------


class TestWorkflowSubmitUsesEnqueueHelper:
    """_submit_to_merge_queue delegates to enqueue_merge_request instead of put() directly."""

    @pytest.mark.asyncio
    async def test_submit_to_merge_queue_calls_enqueue_helper(self, tmp_path: Path):
        """_submit_to_merge_queue calls enqueue_merge_request with (queue, req, event_store).

        Before step-12 impl, the function calls self.merge_queue.put() directly and
        never calls enqueue_merge_request — so mock_helper.assert_called_once() fails.
        After step-12, the function calls enqueue_merge_request — assertion passes.
        """
        from orchestrator.merge_queue import MergeOutcome, MergeRequest
        from orchestrator.workflow import TaskWorkflow

        # Minimal assignment mock (mirrors test_workflow_escalation_warning pattern)
        assignment = MagicMock()
        assignment.task_id = '42'
        assignment.task = {'id': '42', 'title': 'T', 'description': 'desc'}
        assignment.modules = []

        wf_config = MagicMock()
        wf_config.fused_memory.project_id = 'test'
        wf_config.fused_memory.url = 'http://localhost'
        wf_config.max_review_cycles = 2
        wf_config.max_amendment_rounds = 1
        wf_config.lock_depth = 2
        wf_config.steward_completion_timeout = 300.0

        workflow = TaskWorkflow(
            assignment=assignment,
            config=wf_config,
            git_ops=MagicMock(),
            scheduler=MagicMock(),
            briefing=MagicMock(),
            mcp=MagicMock(),
        )

        # Wire required attributes
        merge_queue_mock: AsyncMock = AsyncMock()
        event_store_mock = MagicMock()
        workflow.merge_queue = merge_queue_mock
        workflow.event_store = event_store_mock
        workflow.worktree = tmp_path / 'wt'
        workflow.worktree.mkdir()

        # Before step-12: merge_queue.put() is called directly → resolve future
        # so _submit_to_merge_queue doesn't hang.
        async def _put_resolves_future(req):
            if isinstance(req, MergeRequest) and not req.result.done():
                req.result.set_result(MergeOutcome('done'))

        merge_queue_mock.put.side_effect = _put_resolves_future

        # After step-12: enqueue_merge_request is called → resolve future via mock.
        async def _mock_enqueue(queue, req, es):
            if not req.result.done():
                req.result.set_result(MergeOutcome('done'))

        mock_helper = AsyncMock(side_effect=_mock_enqueue)

        # Patch the source module so both local and module-level imports get the mock.
        with patch('orchestrator.merge_queue.enqueue_merge_request', mock_helper):
            await workflow._submit_to_merge_queue('task/42')

        # KEY: enqueue_merge_request must have been called exactly once
        mock_helper.assert_called_once()
        call_queue, call_req, call_es = mock_helper.call_args.args
        assert call_queue is merge_queue_mock
        assert isinstance(call_req, MergeRequest)
        assert call_req.task_id == '42'
        assert call_req.branch == 'task/42'
        assert call_es is event_store_mock


# ---------------------------------------------------------------------------
# TestEscalationServerUsesEnqueueHelper — step-13 test
# ---------------------------------------------------------------------------


class TestEscalationServerUsesEnqueueHelper:
    """escalation server merge_request tool delegates to enqueue_merge_request."""

    @pytest.mark.asyncio
    async def test_escalation_server_merge_request_uses_enqueue_helper(
        self, tmp_path: Path,
    ):
        """merge_request tool calls enqueue_merge_request(queue, req, event_store).

        Must fail until escalation/server.py accepts the event_store kwarg (step-14)
        and replaces merge_queue.put() with the helper.
        """
        from escalation.server import create_server

        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import MergeOutcome, MergeRequest

        merge_queue: asyncio.Queue = asyncio.Queue()
        event_store = EventStore(db_path=tmp_path / 'test.db', run_id='test')

        # Stub orch_config with _module_configs attribute
        stub_config = MagicMock()
        stub_config._module_configs = {}

        # Mock resolves the future so the tool doesn't hang
        async def _mock_enqueue(queue, req, es):
            if not req.result.done():
                req.result.set_result(MergeOutcome('done'))

        mock_helper = AsyncMock(side_effect=_mock_enqueue)

        # Patch the source module so local imports inside the tool get the mock
        with patch('orchestrator.merge_queue.enqueue_merge_request', mock_helper):
            # Before step-14: create_server raises TypeError (unexpected kwarg)
            mcp = create_server(
                MagicMock(),
                merge_queue=merge_queue,
                orch_config=stub_config,
                event_store=event_store,
            )
            from fastmcp.tools.function_tool import FunctionTool
            tool = await mcp.get_tool('merge_request')
            assert isinstance(tool, FunctionTool)
            await tool.fn(task_id='9', branch='task/9', worktree='/tmp/x')

        # Helper must have been called exactly once with the right args
        mock_helper.assert_called_once()
        call_queue, call_req, call_es = mock_helper.call_args.args
        assert call_queue is merge_queue
        assert isinstance(call_req, MergeRequest)
        assert call_req.task_id == '9'
        assert call_req.branch == 'task/9'
        assert call_es is event_store


# ---------------------------------------------------------------------------
# TestPushHook — main is mirrored to origin after every successful CAS advance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPushHook:
    """push_main fires once per successful CAS advance, in both worker paths.

    Push status is surfaced on MergeOutcome.push_status. A push failure must
    not change the merge outcome — local main has already been advanced.
    """

    async def test_merge_worker_invokes_push_main_on_success(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Normal MergeWorker success path calls push_main exactly once and
        propagates the result onto MergeOutcome.push_status."""
        worktree = await _make_branch_with_file(
            git_ops, 'push-hook-1', 'push_hook_1.py', 'x = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='pushed')
        with patch.object(git_ops, 'push_main', push_mock), \
             patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('push-hook-1', 'push-hook-1', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert result.status == 'done'
        assert result.push_status == 'pushed'
        assert push_mock.await_count == 1

    async def test_merge_worker_done_when_push_fails(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """A push 'error' must not change merge status — main was advanced."""
        worktree = await _make_branch_with_file(
            git_ops, 'push-hook-fail', 'push_hook_fail.py', 'y = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='error')
        with patch.object(git_ops, 'push_main', push_mock), \
             patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('push-hook-fail', 'push-hook-fail', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert result.status == 'done'
        assert result.push_status == 'error'
        assert push_mock.await_count == 1

    async def test_speculative_worker_invokes_push_main_on_success(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker success path also calls push_main once."""
        worktree = await _make_branch_with_file(
            git_ops, 'spec-push', 'spec_push.py', 'z = 3\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='pushed')
        with patch.object(git_ops, 'push_main', push_mock), \
             patch(
                 'orchestrator.merge_queue.run_scoped_verification',
                 _mock_verify_pass(),
             ):
            req = _make_request('spec-push', 'spec-push', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert result.status == 'done'
        assert result.push_status == 'pushed'
        assert push_mock.await_count == 1
