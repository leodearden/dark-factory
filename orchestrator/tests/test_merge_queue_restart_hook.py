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
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _merge_lane_fakes import FakeVerifier, fails, hangs_until, main_health_probe_spawned
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    MERGE_WORKER_SHUTDOWN_REASON,
    MergeOutcome,
    MergeRequest,
    SpeculativeMergeWorker,
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
    return OrchestratorConfig(
        project_root=git_repo, git=git_config,
        escalate_preexisting_main_break=False,
    )


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


class _GatedAdvanceGitOps:
    """The real GitOps with ``advance_main`` gated on an Event.

    Injected through ``SpeculativeMergeWorker``'s existing ``git_ops``
    constructor argument. ``advance_main`` is what ``_journal_landed_then_advance``
    calls from inside ``_finalize_inflight``'s CAS loop, so gating it there
    holds the finalize head at FINALIZING (verify already done) without
    patching a module-level function.
    """

    def __init__(self, inner: GitOps, gate: asyncio.Event) -> None:
        self._inner = inner
        self._gate = gate

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._inner, name)

    async def advance_main(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        await self._gate.wait()
        return await self._inner.advance_main(*args, **kwargs)


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
    req: MergeRequest,
) -> None:
    """Poll until *req* is the popped-for-finalize VERIFYING entry.

    Bounded ~200 x 0.05s poll (mirrors test_restart_recovery_integration) for
    the PUBLIC snapshot to show *req* at the head of line still verifying
    (popped off _inflight, verify in progress) AND the request to already be
    durably journaled in *store* — the exact window this task's fix targets.
    """
    for _ in range(200):
        snap = worker.snapshot()
        vip = snap['verify_in_progress']
        if (
            snap['head_of_line'] == req.task_id
            and vip is not None
            and vip['phase'] == 'verifying'
            and any(r.request_id == req.request_id for r in store.load())
        ):
            return
        await asyncio.sleep(0.05)
    pytest.fail(
        f'{req.request_id} never reached the finalize-head verifying state; '
        f'store contents: {store.load()!r}'
    )


async def _wait_for_finalizing_head_mid_advance(
    worker: SpeculativeMergeWorker,
    req: MergeRequest,
) -> Path:
    """Poll until *req* is the finalize head in the FINALIZING phase -- i.e.
    inside _finalize_inflight's CAS advance_main loop, verify already done,
    not merely VERIFYING. Returns the head entry's merge worktree.

    Bounded ~200 x 0.05s poll, mirroring _wait_for_finalizing_head.
    """
    for _ in range(200):
        snap = worker.snapshot()
        vip = snap['verify_in_progress']
        if (
            snap['head_of_line'] == req.task_id
            and vip is not None
            and vip['phase'] == 'finalizing'
        ):
            return Path(snap['entries'][0]['worktree'])
        await asyncio.sleep(0.05)
    pytest.fail(
        f'{req.request_id} never reached the finalize-head FINALIZING '
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
    worker = SpeculativeMergeWorker(
        git_ops, queue, on_merge_landed=callback, verifier=FakeVerifier(),
    )
    worker_task = asyncio.create_task(worker.run())

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
    worker = SpeculativeMergeWorker(
        git_ops, queue, on_merge_landed=_exploding_callback, verifier=FakeVerifier(),
    )
    worker_task = asyncio.create_task(worker.run())

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
    worker = SpeculativeMergeWorker(
        git_ops, queue, merge_store=store, verifier=FakeVerifier(),
    )
    worker_task = asyncio.create_task(worker.run())

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
    # block_event is never set: verify blocks indefinitely (simulates a crash
    # before the request reaches 'done').
    block_event = asyncio.Event()

    queue_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker_a = SpeculativeMergeWorker(
        git_ops, queue_a, merge_store=store,
        verifier=FakeVerifier(default=hangs_until(block_event)),
    )

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

    worker_b = SpeculativeMergeWorker(
        git_ops, queue_b, merge_store=store, verifier=FakeVerifier(),
    )
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
    worker = SpeculativeMergeWorker(
        git_ops, queue, merge_store=store, verifier=FakeVerifier(),
    )
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
# step-21 (task 1772) — a CANCELLED owned request stays in the durable journal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_owned_request_is_kept_in_the_journal(
    git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
) -> None:
    """A cancelled result Future must be KEPT, so the next boot re-enqueues it.

    The owned-request done-callback discriminates by MergeOutcome.reason, not
    by status. Its other three cases are each covered end-to-end elsewhere in
    this module, so only the cancelled one is driven here:

      blocked / reason != shutdown -> REMOVED
        test_anti_retry_deterministic_failure_not_requeued
      blocked / reason == shutdown -> KEPT
        test_verifying_merge_survives_graceful_restart_and_recovers
      done                         -> REMOVED
        test_worker_merge_store_record_and_clear

    A cancellation carries no MergeOutcome at all, so it is the one case with
    no outcome-carrying end-to-end driver; it is reached here by cancelling a
    real enqueued request while its verify hangs.
    """
    store = MergeQueueStore(tmp_path / 'data' / 'orchestrator' / 'merge_queue.json')
    wt = await _make_branch_with_file(
        git_ops, 'cancelled-kept', 'cancelled_kept.py', 'e = 5\n',
    )
    req = _make_request('cancelled-kept', 'cancelled-kept', wt, config)

    block_event = asyncio.Event()  # never set: the verify hangs so the worker keeps ownership
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(
        git_ops, queue, merge_store=store,
        verifier=FakeVerifier(default=hangs_until(block_event)),
    )
    worker_task = asyncio.create_task(worker.run())
    try:
        await queue.put(req)
        await _wait_for_finalizing_head(worker, store, req)

        req.result.cancel()
        await asyncio.sleep(0)

        ids = {r.request_id for r in store.load()}
        assert req.request_id in ids, (
            f'a cancelled request must be KEPT so the next boot can re-enqueue '
            f'it; ids={ids}'
        )
    finally:
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


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
      1. Worker A processes a request; the injected verifier reports a
         deterministic test failure, yielding MergeOutcome('blocked', ...).
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

    queue_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker_a = SpeculativeMergeWorker(
        git_ops, queue_a, merge_store=store,
        verifier=FakeVerifier(default=fails(
            category='test_failure', summary='Test verification failure',
        )),
    )

    worker_task_a = asyncio.create_task(worker_a.run())
    await queue_a.put(req)
    outcome = await asyncio.wait_for(req.result, timeout=60)

    assert outcome.status == 'blocked', f'Expected blocked, got: {outcome}'
    assert not main_health_probe_spawned(outcome), outcome.reason
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

    block_event = asyncio.Event()  # never set: verify blocks for the test's lifetime

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(
        git_ops, queue, merge_store=store,
        verifier=FakeVerifier(default=hangs_until(block_event)),
    )

    worker_task = asyncio.create_task(worker.run())
    await queue.put(req)

    await _wait_for_finalizing_head(worker, store, req)

    await worker.stop()

    assert req.result.done(), (
        'stop() must resolve the finalize-head request Future to a terminal '
        'outcome, not leave it pending'
    )
    outcome = req.result.result()
    assert outcome == MergeOutcome('blocked', reason=MERGE_WORKER_SHUTDOWN_REASON), (
        f'Expected the shutdown terminal; got {outcome!r}'
    )

    assert worker.snapshot()['verify_in_progress'] is None, (
        'the finalize-head verify must be torn down (cancelled) by stop(), '
        'mirroring the _inflight drain teardown — the public projection of '
        'that teardown is the verify no longer being in progress'
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
    _failing_verify_after_gate = dataclasses.replace(
        fails(
            category='test_failure',
            summary='induced verify failure (task 2788 regression test)',
        ),
        release=gate_event,
    )

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(
        git_ops, queue, merge_store=store,
        verifier=FakeVerifier(default=_failing_verify_after_gate),
    )

    worker_task = asyncio.create_task(worker.run())
    await queue.put(req)

    await _wait_for_finalizing_head(worker, store, req)

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

    Gates git_ops.advance_main (what _journal_landed_then_advance calls from
    inside the CAS advance loop, well past the verify await) on an Event so
    the head reaches FINALIZING before stop() runs.
    """
    wt = await _make_branch_with_file(
        git_ops, 'finalize-head-advancing', 'finalize_advancing.py', 'c = 3\n',
    )
    req = _make_request('finalize-head-advancing', 'finalize-head-advancing', wt, config)
    req.request_id = 'mr-fh-advancing'

    advance_gate = asyncio.Event()

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    # The default 5.0s shutdown timeout is generous on purpose here: stop()
    # must let the real (gated) CAS advance complete naturally rather than
    # pre-empting it — unlike the VERIFYING-sub-case tests above, where stop()
    # is expected to resolve the request itself.
    worker = SpeculativeMergeWorker(
        _GatedAdvanceGitOps(git_ops, advance_gate),  # type: ignore[arg-type]
        queue,
        verifier=FakeVerifier(),
    )

    worker_task = asyncio.create_task(worker.run())
    await queue.put(req)

    pre_stop_wt = await _wait_for_finalizing_head_mid_advance(worker, req)
    assert not req.result.done(), (
        'sanity: the request must still be gated (mid-advance) before '
        'stop() runs'
    )
    assert pre_stop_wt.exists(), (
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
