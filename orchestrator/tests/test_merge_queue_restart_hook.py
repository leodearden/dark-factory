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
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    MERGE_WORKER_SHUTDOWN_REASON,
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    SpeculativeMergeWorker,
)
from orchestrator.merge_queue import (
    _journal_landed_then_advance as _real_journal_landed_then_advance,
)
from orchestrator.merge_queue_store import MergeQueueStore, recover_pending_merges
from orchestrator.merge_types import QueuedBranch

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
    return AsyncMock(return_value=type('VR', (), {'passed': True, 'summary': '', 'failing_test_ids': None})())


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
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


async def _wait_for_finalizing_head(
    worker: SpeculativeMergeWorker,
    store: MergeQueueStore,
    request_id: str,
) -> InflightEntry:
    """Poll until *request_id* is the popped-for-finalize VERIFYING entry.

    Bounded ~200 x 0.05s poll (mirrors test_restart_recovery_integration)
    for worker._finalizing_head_entry() to surface *request_id* (popped off
    _inflight, verify_task in progress) AND the request to already be
    durably journaled in *store* — the exact window this task's fix targets.
    """
    for _ in range(200):
        entry = worker._finalizing_head_entry()
        if (
            entry is not None
            and entry.item.request.request_id == request_id
            and any(r.request_id == request_id for r in store.load())
        ):
            return entry
        await asyncio.sleep(0.05)
    pytest.fail(
        f'{request_id} never reached the finalize-head verifying state; '
        f'store contents: {store.load()!r}'
    )


async def _wait_for_finalizing_head_mid_advance(
    worker: SpeculativeMergeWorker,
    request_id: str,
) -> InflightEntry:
    """Poll until *request_id* is the finalize head AND its verify_task has
    already completed -- i.e. the entry is genuinely FINALIZING (inside
    _finalize_inflight's CAS advance_main loop), not merely VERIFYING.

    Bounded ~200 x 0.05s poll, mirroring _wait_for_finalizing_head.
    """
    for _ in range(200):
        entry = worker._finalizing_head_entry()
        if (
            entry is not None
            and entry.item.request.request_id == request_id
            and entry.verify_task is not None
            and entry.verify_task.done()
        ):
            return entry
        await asyncio.sleep(0.05)
    pytest.fail(
        f'{request_id} never reached the finalize-head FINALIZING '
        f'(mid-advance) state'
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
        return type('VR', (), {'passed': True, 'summary': '', 'failing_test_ids': None})()

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
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task_a

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

    # Capture the recovered request object from the report so we can await
    # its result — the worker races for queue_b so we can't get_nowait later.
    recovered_reqs = report['requests']
    assert len(recovered_reqs) == 1, f'Expected 1 recovered request; got {recovered_reqs!r}'
    recovered_req: MergeRequest = recovered_reqs[0]
    assert recovered_req.request_id == req.request_id, (
        f'Recovered wrong request; expected {req.request_id}, '
        f'got {recovered_req.request_id}'
    )

    worker_b = SpeculativeMergeWorker(git_ops, queue_b, merge_store=store)
    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        _mock_verify_pass(),
    ):
        worker_task_b = asyncio.create_task(worker_b.run())

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


# ---------------------------------------------------------------------------
# step-21 (task 1772) — _buffer_owned_request terminal-removal discriminates
#                        by MergeOutcome.reason, NOT by status == 'blocked'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_owned_request_terminal_removal_policy(
    git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
) -> None:
    """_buffer_owned_request's done-callback must discriminate by reason, not status.

    Cases:
      (a) blocked / reason != shutdown  → REMOVED  (fails until step-22)
      (b) blocked / reason == shutdown  → KEPT
      (c) done                          → REMOVED
      (d) cancelled future              → KEPT
    """
    store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
    store = MergeQueueStore(store_path)
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue, merge_store=store)

    _SHUTDOWN_REASON = 'Merge worker shutting down'

    def _unit_req(label: str) -> MergeRequest:
        fut: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        return MergeRequest(
            task_id=label,
            branch=QueuedBranch.parse(label, config.git.branch_prefix),
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=fut,
        )

    # (a) deterministic blocked terminal (error) → REMOVED
    req_a = _unit_req('unit-a')
    worker._buffer_owned_request(req_a)
    req_a.result.set_result(MergeOutcome('blocked', reason='Merge worker error: boom'))
    await asyncio.sleep(0)
    ids = {r.request_id for r in store.load()}
    assert req_a.request_id not in ids, (
        f'Case (a): blocked/error entry should be REMOVED but found in store; ids={ids}'
    )

    # (b) graceful-shutdown blocked terminal → KEPT
    req_b = _unit_req('unit-b')
    worker._buffer_owned_request(req_b)
    req_b.result.set_result(MergeOutcome('blocked', reason=_SHUTDOWN_REASON))
    await asyncio.sleep(0)
    ids = {r.request_id for r in store.load()}
    assert req_b.request_id in ids, (
        f'Case (b): shutdown blocked entry should be KEPT but was removed; ids={ids}'
    )

    # (c) done terminal → REMOVED
    req_c = _unit_req('unit-c')
    worker._buffer_owned_request(req_c)
    req_c.result.set_result(MergeOutcome('done'))
    await asyncio.sleep(0)
    ids = {r.request_id for r in store.load()}
    assert req_c.request_id not in ids, (
        f'Case (c): done entry should be REMOVED but found in store; ids={ids}'
    )

    # (d) cancelled future → KEPT
    req_d = _unit_req('unit-d')
    worker._buffer_owned_request(req_d)
    req_d.result.cancel()
    await asyncio.sleep(0)
    ids = {r.request_id for r in store.load()}
    assert req_d.request_id in ids, (
        f'Case (d): cancelled entry should be KEPT but was removed; ids={ids}'
    )


# ---------------------------------------------------------------------------
# step-23 (task 1772) — end-to-end anti-retry: deterministic failure does NOT
#                        survive across restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anti_retry_deterministic_failure_not_requeued(
    git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
) -> None:
    """A deterministically-failing 'blocked' outcome is removed from the journal
    and is NOT re-enqueued by recover_pending_merges on the next restart.

    Procedure:
      1. Worker A processes a request; run_scoped_verification raises, yielding
         MergeOutcome('blocked', reason='Verification error: ...').
      2. Assert the journal entry is REMOVED (not kept).
      3. Simulate restart: fresh queue + recover_pending_merges finds nothing.

    RED against pre-step-22 code (where status=='blocked' kept the entry and
    recover would re-enqueue it); GREEN after step-22 adds reason discrimination.
    """
    store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
    store = MergeQueueStore(store_path)

    branch_name = 'anti-retry-test'
    wt = await _make_branch_with_file(git_ops, branch_name, 'anti_retry.py', 'x = 42\n')
    req = _make_request('anti-retry-test', branch_name, wt, config)

    async def _failing_verify(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError('Test verification failure')

    queue_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker_a = SpeculativeMergeWorker(git_ops, queue_a, merge_store=store)

    with patch('orchestrator.merge_queue.run_scoped_verification', _failing_verify):
        worker_task_a = asyncio.create_task(worker_a.run())
        await queue_a.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=60)

    assert outcome.status == 'blocked', f'Expected blocked, got: {outcome}'
    assert 'Merge worker shutting down' not in outcome.reason, (
        f'Expected a non-shutdown reason; got {outcome.reason!r}'
    )

    await worker_a.stop()
    await worker_task_a

    # Branch is still live but NOT an ancestor of main (verification failed,
    # so main was not advanced).
    full_branch = f'{config.git.branch_prefix}{branch_name}'
    branch_sha = await git_ops.resolve_branch_sha(full_branch)
    assert branch_sha is not None, 'Branch should still exist after verification failure'

    is_on_main = await git_ops.is_ancestor(full_branch, config.git.main_branch)
    assert not is_on_main, (
        'Branch should NOT be an ancestor of main after a verification failure'
    )

    # Journal entry must be REMOVED (not kept for indefinite retry).
    for _ in range(20):
        if not any(r.request_id == req.request_id for r in store.load()):
            break
        await asyncio.sleep(0.05)
    remaining = {r.request_id for r in store.load()}
    assert req.request_id not in remaining, (
        f'Blocked/error entry should be REMOVED from journal; store: {store.load()!r}'
    )

    # Simulate restart: recover_pending_merges finds nothing to re-enqueue.
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

    assert report['recovered'] == 0, (
        f'Expected 0 recovered after deterministic failure; got {report}'
    )
    assert queue_b.empty(), (
        'Queue should be empty — deterministic failures must not be retried on restart'
    )


# ---------------------------------------------------------------------------
# task 2788 — stop() must give the popped-for-finalize 'verifying' merge
# request the SHUTDOWN terminal so the durable journal retains it for
# recover_pending_merges.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_resolves_finalizing_head_verifying_to_shutdown(
    git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
) -> None:
    """stop() must give a popped-for-finalize VERIFYING request the SHUTDOWN
    terminal so _on_terminal's KEEP guard fires deterministically.

    _finalize_inflight pops its entry off self._inflight BEFORE the long
    `await entry.verify_task` — during that window the entry is invisible to
    BOTH of stop()'s other Future-resolution mechanisms (the _inflight drain,
    which only sees entries still on the deque; and
    _resolve_merging_requests, which only resolves registry-MERGING
    requests). RED until step-2: stop() never touches the finalize head
    today, so req.result stays PENDING after stop() returns.
    """
    store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
    store = MergeQueueStore(store_path)

    branch_name = 'finalize-head-shutdown'
    wt = await _make_branch_with_file(
        git_ops, branch_name, 'finalize_shutdown.py', 'a = 1\n',
    )
    req = _make_request('finalize-head-shutdown', branch_name, wt, config)
    req.request_id = 'mr-a5b5b69b'

    block_event = asyncio.Event()

    async def _blocking_verify(*args, **kwargs):  # type: ignore[no-untyped-def]
        await block_event.wait()  # never set: verify blocks for the test's lifetime
        return type(
            'VR', (), {'passed': True, 'summary': '', 'failing_test_ids': None},
        )()

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue, merge_store=store)
    worker._shutdown_timeout = 0.2  # fast shutdown for tests

    with patch('orchestrator.merge_queue.run_scoped_verification', _blocking_verify):
        worker_task = asyncio.create_task(worker.run())
        await queue.put(req)

        entry = await _wait_for_finalizing_head(worker, store, req.request_id)

        await worker.stop()

    assert req.result.done(), (
        'stop() must resolve the finalize-head request Future to a terminal '
        'outcome, not leave it pending'
    )
    outcome = req.result.result()
    assert outcome == MergeOutcome('blocked', reason=MERGE_WORKER_SHUTDOWN_REASON), (
        f'Expected the shutdown terminal; got {outcome!r}'
    )

    assert entry.verify_task is not None and entry.verify_task.done(), (
        'finalize-head verify_task must be torn down (cancelled) by stop(), '
        'mirroring the _inflight drain teardown'
    )

    ids = {r.request_id for r in store.load()}
    assert req.request_id in ids, (
        f'Durable journal must retain the finalize-head request after a '
        f'graceful stop() so recover_pending_merges can re-enqueue it on the '
        f'next boot; ids={ids}'
    )

    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task


@pytest.mark.asyncio
async def test_verifying_merge_survives_graceful_restart_and_recovers(
    git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
) -> None:
    """A verify-fail racing stop()'s graceful shutdown must not drop the
    journal record; recover_pending_merges must re-enqueue it.

    Determinism: the fix (step-2) resolves the finalize head's Future to the
    SHUTDOWN terminal synchronously, before stop()'s first `await` —
    single-threaded asyncio means that pre-empts the gated verify-fail from
    ever resolving req.result, so _on_terminal's KEEP guard fires. Without
    the fix, stop() never touches the finalize head, so it reaches its own
    first await before the finalize head is resolved and the verify-fail's
    own (non-shutdown) resolution wins the race, and _on_terminal's REMOVE
    arm deletes the journal record. RED until step-2.
    """
    store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
    store = MergeQueueStore(store_path)

    branch_name = 'finalize-head-recovers'
    wt = await _make_branch_with_file(
        git_ops, branch_name, 'finalize_recovers.py', 'b = 2\n',
    )
    req = _make_request('finalize-head-recovers', branch_name, wt, config)
    req.request_id = 'mr-a891f5fb'

    gate_event = asyncio.Event()

    async def _failing_verify_after_gate(*args, **kwargs):  # type: ignore[no-untyped-def]
        await gate_event.wait()
        return type(
            'VR', (), {
                'passed': False,
                'summary': 'induced verify failure (task 2788 regression test)',
                'failing_test_ids': None,
                'test_output': '',
            },
        )()

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue, merge_store=store)
    worker._shutdown_timeout = 0.2  # fast shutdown for tests

    with patch(
        'orchestrator.merge_queue.run_scoped_verification', _failing_verify_after_gate,
    ):
        worker_task = asyncio.create_task(worker.run())
        await queue.put(req)

        await _wait_for_finalizing_head(worker, store, req.request_id)

        # Release the gate so the verify-fail is ready to complete at the next
        # scheduling opportunity, then immediately call stop(): with the fix,
        # stop()'s synchronous shutdown-resolution (before its first await)
        # pre-empts the verify-fail; without it, stop()'s own later awaits let
        # the verify-fail land first and _on_terminal removes the record.
        gate_event.set()
        await worker.stop()
        await worker_task

    ids_after_stop = {r.request_id for r in store.load()}
    assert req.request_id in ids_after_stop, (
        f'Durable journal must retain the request even though a verify-fail '
        f'raced graceful stop(); ids={ids_after_stop}'
    )

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
    recovered_reqs = report['requests']
    assert len(recovered_reqs) == 1, f'Expected 1 recovered request; got {recovered_reqs!r}'
    assert recovered_reqs[0].request_id == req.request_id, (
        f'Recovered wrong request; expected {req.request_id}, '
        f'got {recovered_reqs[0].request_id}'
    )


@pytest.mark.asyncio
@pytest.mark.timeout(180)  # task 3927: real (unmocked) CAS advance loop with
# real git subprocesses under a gated Event -- widened from the 60s default
# to tolerate host oversubscription, same convention as test_crash_recovery.py
# (task 2376) and test_offline_lane_infra_integration.py (task 3832).
async def test_stop_does_not_preempt_finalizing_head_mid_advance(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """stop() must NOT pre-empt a finalize-head entry whose verify already
    passed and is inside _finalize_inflight's CAS advance_main loop
    (registry FINALIZING, verify_task already done) — only the still-
    VERIFYING sub-case (amendment round 1) needs stop()'s pre-emption.

    _finalizing_head_entry() matches both sub-cases (any non-terminal entry
    popped off _inflight). Pre-empting a FINALIZING entry would (a) mislabel
    a merge that actually lands as 'blocked'/SHUTDOWN, and (b) tear down its
    merge worktree via _cleanup_owned_merge_worktree concurrently with
    _finalize_inflight's own git subprocesses in that same worktree. Instead
    stop() must leave it alone and let the existing
    `asyncio.wait(tasks_to_wait, timeout)` on self._verifier_task (which
    runs _finalize_inflight) give the advance a chance to finish naturally.

    Gates _journal_landed_then_advance (called from inside the CAS advance
    loop, well past the verify_task await) on an Event so the head reaches
    FINALIZING — verify_task done() — before stop() runs.
    """
    wt = await _make_branch_with_file(
        git_ops, 'finalize-head-advancing', 'finalize_advancing.py', 'c = 3\n',
    )
    req = _make_request('finalize-head-advancing', 'finalize-head-advancing', wt, config)
    req.request_id = 'mr-fh-advancing'

    advance_gate = asyncio.Event()

    async def _gated_advance(*args, **kwargs):  # type: ignore[no-untyped-def]
        await advance_gate.wait()
        return await _real_journal_landed_then_advance(*args, **kwargs)

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue)
    # Generous shutdown timeout: stop() must let the real (gated) CAS advance
    # complete naturally rather than pre-empting it — unlike the fast
    # (0.2s) timeout used by the VERIFYING-sub-case tests above, where
    # stop() is expected to resolve the request itself.
    worker._shutdown_timeout = 5.0

    with (
        patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        patch('orchestrator.merge_queue._journal_landed_then_advance', _gated_advance),
    ):
        worker_task = asyncio.create_task(worker.run())
        await queue.put(req)

        entry = await _wait_for_finalizing_head_mid_advance(worker, req.request_id)
        assert not req.result.done(), (
            'sanity: the request must still be gated (mid-advance) before '
            'stop() runs'
        )
        pre_stop_wt = entry.merge_wt
        assert pre_stop_wt is not None and pre_stop_wt.exists(), (
            'sanity: the merge worktree must still exist while the advance '
            'is gated'
        )

        # Release the gate, then immediately call stop(): the gated
        # _finalize_inflight coroutine cannot resume until this coroutine
        # yields control, so stop()'s (synchronous, up to its own first
        # await) finalize-head check runs first — mirroring the race
        # construction in test_verifying_merge_survives_graceful_restart_
        # and_recovers above, but here proving stop() stays hands-off.
        advance_gate.set()
        await worker.stop()
        await worker_task

    assert req.result.done(), 'the gated advance must eventually resolve the request'
    outcome = req.result.result()
    assert outcome.status == 'done', (
        f"stop() must not mislabel a mid-advance FINALIZING entry as the "
        f"shutdown terminal -- expected the merge's true 'done' outcome; "
        f"got {outcome!r}"
    )
    assert outcome.reason != MERGE_WORKER_SHUTDOWN_REASON, (
        f'the true outcome must not carry the shutdown reason; got {outcome!r}'
    )
    assert outcome.merge_sha is not None

    _, post_merge_main_raw, _ = await _run(
        ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
    )
    assert post_merge_main_raw.strip() == outcome.merge_sha, (
        'main must actually have advanced to the merge commit -- proof '
        'stop() did not tear down the worktree out from under the '
        'in-flight advance'
    )
