"""Tests for merge queue: MergeWorker, CAS update-ref, ghost-loop detection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeOutcome, MergeRequest, MergeWorker, SpeculativeMergeWorker

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
    mock = AsyncMock()
    mock.return_value.passed = True
    mock.return_value.summary = ''
    return mock


# ---------------------------------------------------------------------------
# TestCasUpdateRef — Phase A
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCasUpdateRef:
    async def test_advance_main_with_expected(self, git_ops: GitOps):
        """CAS succeeds when expected_main matches actual main."""
        worktree, _ = await git_ops.create_worktree('cas-ok')
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
        worktree, _ = await git_ops.create_worktree('cas-fail')
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
        worktree, _ = await git_ops.create_worktree('cas-none')
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
class TestMergeWorker:
    async def test_basic_merge_through_queue(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Submit a merge request → worker merges → file appears on main."""
        worktree, _ = await git_ops.create_worktree('queue-basic')
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

    async def test_already_merged_returns_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Branch that's already on main returns already_merged."""
        worktree, _ = await git_ops.create_worktree('already-merged')
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
        worktree, _ = await git_ops.create_worktree('conflict-task')

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
        worktree, _ = await git_ops.create_worktree('cas-retry')
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
        worktree, _ = await git_ops.create_worktree('shutdown')
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
        worktree, _ = await git_ops.create_worktree('verify-fail')
        (worktree / 'bad.py').write_text('bad = True\n')
        await git_ops.commit(worktree, 'Add bad file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Mock verification to fail
        mock_verify = AsyncMock()
        mock_verify.return_value.passed = False
        mock_verify.return_value.summary = 'tests failed'

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
        worktree, _ = await git_ops.create_worktree('cas-limit')
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
        worktree, _ = await git_ops.create_worktree('perm-fail')
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
    worktree, _ = await git_ops.create_worktree(branch_name)
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
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
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

        async def _verify_side_effect(merge_wt, cfg, module_configs, task_files=None):
            # Determine which task by looking at which file is present
            n_file = merge_wt / 'file_disc_n.py'
            if n_file.exists():
                verify_calls['n'] = verify_calls.get('n', 0) + 1
                result = AsyncMock()
                result.passed = False
                result.summary = 'N tests failed'
                return result
            else:
                verify_calls['n1'] = verify_calls.get('n1', 0) + 1
                result = AsyncMock()
                result.passed = True
                result.summary = ''
                return result

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
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
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
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
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
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_speculative_shutdown_drains_both(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """stop() resolves all pending Futures as 'blocked' with shutdown reason."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        # Don't start worker — just queue items and stop
        wt_a, _ = await git_ops.create_worktree('shut-a')
        wt_b, _ = await git_ops.create_worktree('shut-b')
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
        wt_n1, _ = await git_ops.create_worktree('cfl-n1')
        # Write conflicting content to README.md in both main and wt_n1
        (git_ops.project_root / 'README.md').write_text('# Main conflict\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
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
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
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
            result = AsyncMock()
            result.passed = not (merge_wt / 'file_ev_n.py').exists() or \
                            (merge_wt / 'file_ev_n1.py').exists() and \
                            not (merge_wt / 'file_ev_n.py').exists()
            # Simpler: fail when both N and N+1 are present (speculative), pass after re-merge
            n_present = (merge_wt / 'file_ev_n.py').exists()
            n1_present = (merge_wt / 'file_ev_n1.py').exists()
            if n_present and not n1_present:
                result.passed = False   # N verification → fail
                result.summary = 'N failed'
            else:
                result.passed = True    # N+1 (re-merged, no N) → pass
                result.summary = ''
            return result

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

        import sqlite3
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id FROM events ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [r[0] for r in rows]
        assert 'speculative_merge' in event_types, f'No speculative_merge event: {event_types}'
        assert 'speculative_discard' in event_types, f'No speculative_discard event: {event_types}'

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
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
        wt_n1, _ = await git_ops.create_worktree('am-n1')
        (wt_n1 / 'file_am_n1.py').write_text('am_n1 = 2\n')
        await git_ops.commit(wt_n1, 'N+1 file')
        result = await git_ops.merge_to_main(wt_n1, 'am-n1')
        assert result.success
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
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# TestSpeculativeBackwardCompat — step-17
# Run key MergeWorker scenarios through SpeculativeMergeWorker to confirm
# they behave identically with queue depth 1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculativeBackwardCompat:
    async def test_basic_merge(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree, _ = await git_ops.create_worktree('compat-basic')
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
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_already_merged(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree, _ = await git_ops.create_worktree('compat-am')
        (worktree / 'am.py').write_text('am = True\n')
        await git_ops.commit(worktree, 'Add am file')

        result = await git_ops.merge_to_main(worktree, 'compat-am')
        assert result.success
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
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_verify_failure(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree, _ = await git_ops.create_worktree('compat-vf')
        (worktree / 'bad.py').write_text('bad = True\n')
        await git_ops.commit(worktree, 'Add bad file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        mock_verify = AsyncMock()
        mock_verify.return_value.passed = False
        mock_verify.return_value.summary = 'tests failed'

        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify):
            req = _make_request('compat-vf', 'compat-vf', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'verification failed' in outcome.reason.lower()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task
