"""Tests for merge queue: MergeWorker, CAS update-ref, ghost-loop detection."""

from __future__ import annotations

import asyncio
import contextlib
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
    MergeOutcome,
    MergeRequest,
    MergeWorker,
    SpeculativeItem,
    SpeculativeMergeWorker,
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
            missing = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_one_plan_target_missing(
        self, git_ops: GitOps,
    ):
        """Plan lists a file the task never created → returned in missing."""
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
            missing = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            assert missing == ['absent.py']
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

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
            missing = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
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
            missing = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
            )
            assert missing == []
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
        """Plan lists a file absent from merge commit → MergeWorker blocks."""
        worktree = (await git_ops.create_worktree('drop-guard-task')).path
        (worktree / 'kept.py').write_text('kept = True\n')
        await git_ops.commit(worktree, 'Add kept file')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-guard', 'Drop guard', 'desc')
        # Plan claims two files but the task branch only created one.
        artifacts.write_plan({
            'files': ['kept.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass(),
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
