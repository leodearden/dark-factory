"""Tests for SpeculativeMergeWorker.on_merge_landed callback (task 1592).

Extended in task 1772 to cover:
  - merge_store recording: worker records an owned request and clears it on terminal.
  - Restart/recovery integration: a "crashed" worker A journals a request; fresh
    worker B rehydrates it via recover_pending_merges and drives it to 'done'.
  - Idempotency: recover_pending_merges drops already-landed branches without
    re-enqueuing them.

Verifies that:
  1. A 'done' merge invokes the on_merge_landed callback with
     (task_id, base_sha, advanced_sha) where advanced_sha == outcome.merge_sha.
  2. When on_merge_landed raises, the merge still resolves 'done' (fail-open).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeOutcome, MergeRequest, SpeculativeMergeWorker
from orchestrator.merge_queue_store import MergeQueueStore, recover_pending_merges

# ---------------------------------------------------------------------------
# Fixtures — mirror test_merge_queue.py
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


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
    return OrchestratorConfig(project_root=git_repo, git=git_config)


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


def _mock_verify_pass():
    """Return a mock that makes run_scoped_verification always pass."""
    return AsyncMock(return_value=type('VR', (), {'passed': True, 'summary': ''})())


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_merge_landed_invoked_on_done(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """on_merge_landed is awaited exactly once with (task_id, base_sha, advanced_sha)."""
    wt = await _make_branch_with_file(git_ops, 'hook-test', 'hook_file.py', 'x = 1\n')

    # Capture main SHA before the merge — this is what item.base_sha will be
    _, pre_merge_main_raw, _ = await _run(
        ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
    )
    pre_merge_main = pre_merge_main_raw.strip()

    callback = AsyncMock()
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue, on_merge_landed=callback)
    worker_task = asyncio.create_task(worker.run())

    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        _mock_verify_pass(),
    ):
        req = _make_request('hook-test', 'hook-test', wt, config)
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=60)

    assert outcome.status == 'done', f'Expected done, got: {outcome}'
    assert outcome.merge_sha is not None

    # The callback must have been awaited exactly once
    callback.assert_awaited_once()
    called_task_id, called_base_sha, called_advanced_sha = callback.call_args.args
    assert called_task_id == 'hook-test'
    assert called_base_sha == pre_merge_main, (
        f'base_sha should be pre-merge main tip; got {called_base_sha!r}'
    )
    assert called_advanced_sha == outcome.merge_sha, (
        f'advanced_sha should equal outcome.merge_sha; got {called_advanced_sha!r}'
    )

    await worker.stop()
    await worker_task


@pytest.mark.asyncio
async def test_on_merge_landed_fail_open(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """When on_merge_landed raises, the merge STILL resolves 'done' (fail-open)."""
    wt = await _make_branch_with_file(git_ops, 'hook-fail', 'hook_fail.py', 'y = 2\n')

    async def _exploding_callback(task_id: str, base_sha: str, head_sha: str) -> None:
        raise RuntimeError('Simulated callback failure')

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue, on_merge_landed=_exploding_callback)
    worker_task = asyncio.create_task(worker.run())

    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        _mock_verify_pass(),
    ):
        req = _make_request('hook-fail', 'hook-fail', wt, config)
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=60)

    # Merge must still succeed despite the callback raising
    assert outcome.status == 'done', (
        f'Merge should not fail because of the callback; got: {outcome}'
    )

    await worker.stop()
    await worker_task


# ---------------------------------------------------------------------------
# step-13 (task 1772) — worker records owned requests and clears on terminal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_merge_store_record_and_clear(
    git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
) -> None:
    """SpeculativeMergeWorker with merge_store records a request when it drains
    the queue item into a lane buffer, and removes it once the request reaches a
    terminal outcome ('done').

    RED until step-14 adds the merge_store param + _buffer_owned_request seam.
    """
    store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
    store = MergeQueueStore(store_path)

    wt = await _make_branch_with_file(git_ops, 'store-test', 'store_file.py', 'z = 3\n')
    req = _make_request('store-test', 'store-test', wt, config)

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    # Pass merge_store — fails until step-14 adds the parameter.
    worker = SpeculativeMergeWorker(git_ops, queue, merge_store=store)
    worker_task = asyncio.create_task(worker.run())

    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        _mock_verify_pass(),
    ):
        await queue.put(req)

        # Poll until the worker records the request (drains into lane buffer).
        for _ in range(200):
            if any(r.request_id == req.request_id for r in store.load()):
                break
            await asyncio.sleep(0.05)
        else:
            await worker.stop()
            await worker_task
            pytest.fail(
                f'Worker never recorded {req.request_id} in the store; '
                f'store contents: {store.load()!r}'
            )

        # Now await the merge result (should be 'done').
        outcome = await asyncio.wait_for(req.result, timeout=60)

    assert outcome.status == 'done', f'Expected done, got: {outcome}'

    # After terminal, the store entry must have been removed.
    # Allow brief async teardown by polling a short window.
    for _ in range(20):
        if not any(r.request_id == req.request_id for r in store.load()):
            break
        await asyncio.sleep(0.05)

    remaining_ids = {r.request_id for r in store.load()}
    assert req.request_id not in remaining_ids, (
        f'{req.request_id} was NOT removed from the store after terminal outcome; '
        f'store: {store.load()!r}'
    )

    await worker.stop()
    await worker_task


# ---------------------------------------------------------------------------
# step-15 (task 1772) — restart/recovery integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_recovery_integration(
    git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
) -> None:
    """Phase 1: worker A journals a request and is crash-cancelled mid-merge.
    Phase 2: fresh worker B rehydrates via recover_pending_merges and drives
    the recovered request to a terminal 'done'.

    RED until step-16 closes the gaps in reconstruct -> enqueue -> worker accept
    -> terminal remove.
    """
    store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
    store = MergeQueueStore(store_path)

    branch_name = 'restart-test'
    wt = await _make_branch_with_file(git_ops, branch_name, 'restart_file.py', 'w = 4\n')
    req = _make_request('restart-test', branch_name, wt, config)

    # --- Phase 1: start worker A, block verification, let it journal the request ---
    block_event = asyncio.Event()

    async def _blocking_verify(*args, **kwargs):  # type: ignore[no-untyped-def]
        await block_event.wait()  # block indefinitely (simulates crash before done)
        return type('VR', (), {'passed': True, 'summary': ''})()

    queue_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker_a = SpeculativeMergeWorker(git_ops, queue_a, merge_store=store)

    with patch('orchestrator.merge_queue.run_scoped_verification', _blocking_verify):
        worker_task_a = asyncio.create_task(worker_a.run())
        await queue_a.put(req)

        # Wait until worker A has journaled the request (owns it in lane buffer).
        for _ in range(200):
            if any(r.request_id == req.request_id for r in store.load()):
                break
            await asyncio.sleep(0.05)
        else:
            worker_task_a.cancel()
            pytest.fail('Worker A never recorded the request in the journal')

        # Simulate crash: cancel the worker WITHOUT calling stop().
        worker_task_a.cancel()
        try:
            await worker_task_a
        except asyncio.CancelledError:
            pass

    # The journal must still hold the request after the crash.
    persisted_ids = {r.request_id for r in store.load()}
    assert req.request_id in persisted_ids, (
        f'Journal lost the request after crash; store: {store.load()!r}'
    )

    # --- Phase 2: recover with a fresh queue and worker ---
    queue_b: asyncio.Queue[MergeRequest] = asyncio.Queue()

    report = await recover_pending_merges(
        store,
        queue_b,
        git_ops,
        config,
        event_store=None,
        main_branch=config.git.main_branch,
        branch_prefix=config.git.branch_prefix,
    )

    assert report['recovered'] == 1, f'Expected 1 recovered; got {report}'
    assert report['dropped'] == 0, f'Expected 0 dropped; got {report}'

    worker_b = SpeculativeMergeWorker(git_ops, queue_b, merge_store=store)
    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        _mock_verify_pass(),
    ):
        worker_task_b = asyncio.create_task(worker_b.run())

        # Drain the recovered item from queue_b to get the fresh request.
        recovered_req: MergeRequest = await asyncio.wait_for(
            queue_b.get(), timeout=5,
        )
        assert recovered_req.request_id == req.request_id, (
            f'Recovered wrong request; expected {req.request_id}, '
            f'got {recovered_req.request_id}'
        )

        # Await the recovered merge to complete.
        outcome = await asyncio.wait_for(recovered_req.result, timeout=60)

    assert outcome.status == 'done', f'Expected done on recovered merge; got: {outcome}'

    # Confirm the branch tip is now an ancestor of main (truly merged).
    full_branch = f'{config.git.branch_prefix}{branch_name}'
    is_on_main = await git_ops.is_ancestor(full_branch, config.git.main_branch)
    assert is_on_main, (
        f'Branch {full_branch} not an ancestor of main after recovery merge'
    )

    # Journal entry must be cleaned up.
    for _ in range(20):
        if not any(r.request_id == req.request_id for r in store.load()):
            break
        await asyncio.sleep(0.05)
    remaining = {r.request_id for r in store.load()}
    assert req.request_id not in remaining, (
        f'Journal entry not removed after successful recovery merge; store: {store.load()!r}'
    )

    await worker_b.stop()
    await worker_task_b


# ---------------------------------------------------------------------------
# step-17 (task 1772) — idempotency: already-landed branch is dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_already_landed_branch_dropped(
    git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
) -> None:
    """recover_pending_merges drops a record whose branch is already on main.

    Procedure:
      1. Create a branch, merge it into main via the worker.
      2. Seed the store with a record for that branch (simulates a journal entry
         that survived the merge but before cleanup — e.g. worker crashed between
         merge and removal).
      3. Capture main's tip SHA.
      4. Run recover_pending_merges.
      5. Assert: record dropped, queue empty, main unchanged.

    RED until step-18 confirms the is_ancestor pre-check works with real git_ops.
    """
    branch_name = 'idempotency-test'
    wt = await _make_branch_with_file(
        git_ops, branch_name, 'idempotency_file.py', 'v = 5\n',
    )

    # Merge the branch into main via the worker.
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
    store = MergeQueueStore(store_path)
    worker = SpeculativeMergeWorker(git_ops, queue, merge_store=store)
    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        _mock_verify_pass(),
    ):
        worker_task = asyncio.create_task(worker.run())
        first_req = _make_request('idempotency-test', branch_name, wt, config)
        await queue.put(first_req)
        first_outcome = await asyncio.wait_for(first_req.result, timeout=60)
    assert first_outcome.status == 'done', f'Pre-merge failed: {first_outcome}'
    await worker.stop()
    await worker_task

    # Record main's tip SHA — must be unchanged after recovery.
    _, main_sha_raw, _ = await _run(
        ['git', 'rev-parse', config.git.main_branch],
        cwd=git_ops.project_root,
    )
    main_sha_before = main_sha_raw.strip()

    # Seed the store with a stale record for the already-landed branch.
    store2 = MergeQueueStore(store_path)
    stale_req = _make_request('idempotency-stale', branch_name, wt, config)
    store2.record(stale_req)

    recovery_queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    report = await recover_pending_merges(
        store2,
        recovery_queue,
        git_ops,
        config,
        event_store=None,
        main_branch=config.git.main_branch,
        branch_prefix=config.git.branch_prefix,
    )

    # Record must be dropped (not re-enqueued).
    assert report['dropped'] >= 1, f'Expected at least 1 dropped; got {report}'
    assert recovery_queue.empty(), 'Queue should be empty — no re-enqueue for landed branch'

    remaining = {r.request_id for r in store2.load()}
    assert stale_req.request_id not in remaining, (
        'Stale record must be removed from the journal'
    )

    # main must be unchanged.
    _, main_sha_raw2, _ = await _run(
        ['git', 'rev-parse', config.git.main_branch],
        cwd=git_ops.project_root,
    )
    main_sha_after = main_sha_raw2.strip()
    assert main_sha_before == main_sha_after, (
        f'main advanced unexpectedly during recovery: {main_sha_before!r} -> {main_sha_after!r}'
    )
