"""Tests for SpeculativeMergeWorker._resolve_and_release — the single
resolve-and-release chokepoint unifying the six _verifier_loop BaseException
handlers (task MQ-refactor zeta / 1991).

Steps covered:
  step-1  RED  — _resolve_and_release contract (SpeculativeItem input,
                 InflightEntry input, release_resources=False,
                 cancel_lease=True, chain_failed=False)
  step-2  GREEN — implement _resolve_and_release
  step-3  RED  — DISPATCH path fault injection (fill + blocking-get)
  step-4  GREEN — route dispatch handlers through the chokepoint
  step-5  RED  — PASSTHROUGH-FINALIZE path fault injection (fill + blocking-get)
  step-6  GREEN — route passthrough-finalize handlers through the chokepoint
  step-7  RED  — FINALIZE-HEAD path fault injection
  step-8  GREEN — route finalize-head handler through the chokepoint
  step-9  RED  — CASCADE path fault injection + release idempotency
  step-10 GREEN — route cascade handler through the chokepoint
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Reuse the γ harness two-host fakes (established cross-test-module import
# pattern — see test_concurrent_verify_boundary.py / test_merge_speculation.py).
from test_merge_queue_concurrent_verify import (
    _gated_runner,
    _inject_two_host_allocator,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    SpeculativeItem,
    SpeculativeMergeWorker,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_concurrent_verify.py / test_merge_queue_finalize_head_visibility.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
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
    """Single-host (no verify_runners) OrchestratorConfig."""
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


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop."""
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


async def _make_real_merged_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    branch: str,
    filename: str,
    content: str,
    *,
    speculative: bool,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Build a real merged SpeculativeItem backed by a disk `_merge-*` worktree."""
    worktree = await _make_branch_with_file(git_ops, branch, filename, content)
    req = _make_request(branch, branch, worktree, config)
    merge_result = await git_ops.merge_to_main(worktree, branch)
    assert merge_result.success
    base_sha = await git_ops.get_main_sha()
    item = SpeculativeItem(
        request=req,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=base_sha,
        speculative=speculative,
        skip_verify=False,
    )
    return req, item


# ---------------------------------------------------------------------------
# step-1 RED: _resolve_and_release contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResolveAndReleaseContract:
    """Unit-level contract tests for the (not-yet-existing) chokepoint coroutine.

    RED until step-2 GREEN adds SpeculativeMergeWorker._resolve_and_release.
    Every case below fails RED with AttributeError (method missing).
    """

    async def test_speculative_item_input_defaults_release_and_resolve(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(a) SpeculativeItem input, defaults (release_resources=True,
        cancel_lease=False), chain_failed=True: resolves req.result, cleans +
        deregisters the owned merge worktree, releases the speculation slot
        exactly once, and sets _n_failed=True.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/rr-a', 'rr_a.py', 'a = 1\n', speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        assert item.merge_wt in worker._owned_merge_worktrees
        assert item.merge_wt is not None and item.merge_wt.exists()

        depth0 = worker._speculation_slot._value
        outcome = MergeOutcome('blocked', reason='x')

        await worker._resolve_and_release(item, outcome, chain_failed=True)

        assert req.result.done()
        assert req.result.result() is outcome
        assert not item.merge_wt.exists(), 'merge worktree must be removed from disk'
        assert item.merge_wt not in worker._owned_merge_worktrees, (
            'merge worktree must be deregistered from the owned ledger'
        )
        assert worker._speculation_slot._value == depth0 + 1, (
            'speculation slot must be released exactly once for a speculative item'
        )
        assert worker._n_failed is True

    async def test_inflight_entry_release_resources_false_is_a_noop_release(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(b) InflightEntry input with release_resources=False: resolves
        req.result and sets _n_failed, but performs NO release of the slot,
        worktree, or lease (they were already released by _finalize_inflight's
        finally clause at the real post-finalize call sites).
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/rr-b', 'rr_b.py', 'b = 1\n', speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        fake_lease = MagicMock()
        entry = InflightEntry(
            item=item,
            lease=fake_lease,
            verify_task=None,
            merge_wt=item.merge_wt,
            was_speculative=True,
            phase='finalizing',
        )
        depth0 = worker._speculation_slot._value
        outcome = MergeOutcome('blocked', reason='y')

        await worker._resolve_and_release(
            entry, outcome, chain_failed=True, release_resources=False,
        )

        assert req.result.done()
        assert req.result.result() is outcome
        assert worker._speculation_slot._value == depth0, 'slot must be untouched'
        assert item.merge_wt is not None and item.merge_wt.exists(), (
            'worktree must be untouched on disk'
        )
        assert item.merge_wt in worker._owned_merge_worktrees, (
            'worktree ledger entry must be untouched'
        )
        assert worker._n_failed is True

    async def test_release_resources_true_cancel_lease_true_uses_cancel_and_release(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(c) release_resources=True, cancel_lease=True on an InflightEntry
        carrying a fake lease + a stubbed allocator: cancel_and_release(lease)
        is awaited; release(lease) is NOT.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/rr-c', 'rr_c.py', 'c = 1\n', speculative=False,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        fake_lease = MagicMock()
        allocator = MagicMock()
        allocator.cancel_and_release = AsyncMock()
        allocator.release = AsyncMock()
        worker._host_allocator = allocator
        entry = InflightEntry(
            item=item,
            lease=fake_lease,
            verify_task=None,
            merge_wt=item.merge_wt,
            was_speculative=False,
            phase='finalizing',
        )
        outcome = MergeOutcome('blocked', reason='z')

        await worker._resolve_and_release(
            entry, outcome, chain_failed=True,
            release_resources=True, cancel_lease=True,
        )

        allocator.cancel_and_release.assert_awaited_once_with(fake_lease)
        allocator.release.assert_not_awaited()

    async def test_chain_failed_false_leaves_n_failed_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(d) chain_failed=False must not reset _n_failed to False — the
        helper only ever sets it True (guarded `if chain_failed: ... = True`),
        never assigns the raw flag value.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._n_failed = True  # simulate a prior chain failure
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/rr-d', 'rr_d.py', 'd = 1\n', speculative=False,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        outcome = MergeOutcome('blocked', reason='w')

        await worker._resolve_and_release(item, outcome, chain_failed=False)

        assert req.result.done()
        assert worker._n_failed is True, (
            'chain_failed=False must not reset _n_failed back to False'
        )


# ---------------------------------------------------------------------------
# Shared fault-injection driving helpers (steps 3, 5, 7, 9)
# ---------------------------------------------------------------------------


def _spy_on_resolve_and_release(
    worker: SpeculativeMergeWorker,
) -> list[dict[str, Any]]:
    """Wrap worker._resolve_and_release with a call-recording spy.

    Delegates to the real (bound) implementation so end-state assertions
    (Future resolution, releases) still hold; records (args, kwargs) per call
    so tests can assert call count and inspect kwargs (e.g. release_resources).

    Design note (reviewer_comprehensive test_coupling, task 1991 amendment):
    every test in this module that uses this spy pairs the call-count
    assertion with observable end-state assertions (Future outcome/status/
    reason, ``_speculation_slot`` value, worktree existence on disk,
    ``_n_failed``) — the call count is never the *sole* assertion.  The
    call-count check is kept deliberately, not out of habit: this module's
    entire purpose is verifying that task 1991 unified five near-identical
    inline except-handlers into ONE chokepoint, so "routes through the
    single chokepoint" is itself part of the contract under test here.  A
    regression that reintroduced an inline resolve-and-release path
    alongside the chokepoint could still leave end-state assertions green
    while silently defeating the refactor's single-chokepoint goal; only the
    call-count assertion catches that.  This is a conscious
    maintainability-vs-robustness trade-off (high private-internals coupling
    in exchange for locking in the architectural invariant) — expect churn
    here if the resolve/release call sites are restructured again.
    """
    calls: list[dict[str, Any]] = []
    original = worker._resolve_and_release

    async def _spy(*args: Any, **kwargs: Any) -> None:
        calls.append({'args': args, 'kwargs': kwargs})
        await original(*args, **kwargs)

    worker._resolve_and_release = _spy  # type: ignore[method-assign]
    return calls


async def _drive_verifier_loop_fill(
    worker: SpeculativeMergeWorker, item: SpeculativeItem,
) -> None:
    """Fill variant: pre-load the queue with [item, None] then run one pass.

    Both items are already queued before _verifier_loop starts, so its first
    DISPATCH-FILL iteration picks up *item* via the non-blocking get_nowait()
    path (fill-loop dispatch site).
    """
    worker._verifier_queue.put_nowait(item)
    worker._verifier_queue.put_nowait(None)
    await asyncio.wait_for(worker._verifier_loop(), timeout=10.0)


async def _drive_verifier_loop_blocking_get(
    worker: SpeculativeMergeWorker, item: SpeculativeItem,
) -> None:
    """Blocking-get variant: start the loop on an EMPTY queue, let it reach
    the blocking ``await self._verifier_queue.get()`` branch, THEN arrive.

    The initial ``asyncio.sleep(0)`` yield is load-bearing: without it, both
    queue.put() calls below (unbounded queue → put() never actually suspends)
    would land before the task gets any run time, so the task would see a
    non-empty queue on its very first DISPATCH-FILL iteration and take the
    get_nowait() fill-loop path instead of the blocking-get path this variant
    targets.  A single scheduler tick is sufficient because everything from
    task-start to the blocking ``get()`` call is synchronous Python (no
    intervening real await), so one `Task.__step` drives it all the way there.
    """
    task = asyncio.create_task(worker._verifier_loop())
    await asyncio.sleep(0)
    await worker._verifier_queue.put(item)
    await worker._verifier_queue.put(None)
    await asyncio.wait_for(task, timeout=10.0)


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: DISPATCH path (fill-loop ~7099 + blocking-get ~7381)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchErrorChokepoint:
    """DISPATCH path fault injection: _dispatch_item raises.

    RED until step-4 GREEN routes both dispatch except-handlers through
    _resolve_and_release. RED marker: the chokepoint spy call count is 0
    (handlers still inline the resolve+release logic); all other assertions
    already hold against the pre-refactor inline code.
    """

    @pytest.mark.parametrize(
        'driver', [_drive_verifier_loop_fill, _drive_verifier_loop_blocking_get],
        ids=['fill', 'blocking_get'],
    )
    async def test_dispatch_error_routes_through_chokepoint(
        self, git_ops: GitOps, config: OrchestratorConfig, driver: Any,
    ) -> None:
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, f'task/disp-{driver.__name__}', 'disp.py', 'x = 1\n',
            speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        calls = _spy_on_resolve_and_release(worker)

        async def _raising_dispatch(_item: Any) -> Any:
            raise Exception('boom')

        worker._dispatch_item = _raising_dispatch  # type: ignore[method-assign]

        depth0 = worker._speculation_slot._value

        await driver(worker, item)

        assert len(calls) == 1, (
            f'Expected exactly one _resolve_and_release call, got {len(calls)}'
        )
        assert req.result.done()
        outcome = req.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith('Verifier error:'), outcome.reason
        assert worker._speculation_slot._value == depth0 + 1, (
            'speculation slot must be released exactly once'
        )
        assert item.merge_wt is not None and not item.merge_wt.exists(), (
            'merge worktree must be removed from disk'
        )
        assert item.merge_wt not in worker._owned_merge_worktrees, (
            'merge worktree must be deregistered from the owned ledger'
        )
        assert worker._n_failed is True

    async def test_dispatch_cancelled_error_propagates_and_skips_chokepoint(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """CancelledError from _dispatch_item must propagate out of
        _verifier_loop untouched; the chokepoint must never see it.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _req, item = await _make_real_merged_item(
            git_ops, config, 'task/disp-cancel', 'disp_cancel.py', 'x = 1\n',
            speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        calls = _spy_on_resolve_and_release(worker)

        async def _cancelling_dispatch(_item: Any) -> Any:
            raise asyncio.CancelledError()

        worker._dispatch_item = _cancelling_dispatch  # type: ignore[method-assign]

        worker._verifier_queue.put_nowait(item)
        worker._verifier_queue.put_nowait(None)

        with pytest.raises(asyncio.CancelledError):
            await worker._verifier_loop()

        assert len(calls) == 0, 'CancelledError must never reach the chokepoint'

    async def test_dispatch_error_with_abandoned_request_drops_silently(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """An abandoned/detached request (its Future already cancelled by
        the waiter) that then hits a dispatch error must be dropped
        silently by _resolve_or_drop_abandoned — no InvalidStateError from
        calling set_result on an already-cancelled Future — while resource
        release (worktree cleanup + speculation slot) still proceeds
        normally, since release_resources is independent of the Future's
        abandonment state.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/disp-abandoned', 'disp_abandoned.py', 'x = 1\n',
            speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        calls = _spy_on_resolve_and_release(worker)

        async def _raising_dispatch(_item: Any) -> Any:
            raise Exception('boom')

        worker._dispatch_item = _raising_dispatch  # type: ignore[method-assign]

        # Simulate the waiter detaching before the dispatch error fires.
        assert req.result.cancel()
        depth0 = worker._speculation_slot._value

        await _drive_verifier_loop_fill(worker, item)

        assert len(calls) == 1, (
            f'Expected exactly one _resolve_and_release call, got {len(calls)}'
        )
        assert req.result.cancelled(), (
            'abandoned Future must remain cancelled, not overwritten '
            '(no InvalidStateError from set_result on a cancelled Future)'
        )
        assert worker._speculation_slot._value == depth0 + 1, (
            'resource release must still proceed for an abandoned waiter (no leak)'
        )
        assert item.merge_wt is not None and not item.merge_wt.exists(), (
            'merge worktree must still be cleaned up for an abandoned waiter'
        )
        assert item.merge_wt not in worker._owned_merge_worktrees
        assert worker._n_failed is True


# ---------------------------------------------------------------------------
# step-5 RED / step-6 GREEN: PASSTHROUGH-FINALIZE path (fill-loop ~7144 +
# blocking-get ~7411)
# ---------------------------------------------------------------------------


def _make_decided_item_and_entry(
    req: MergeRequest, *, was_speculative: bool,
) -> tuple[SpeculativeItem, InflightEntry]:
    """Build a DECIDED SpeculativeItem (immediate_outcome set, no real merge)
    and its passthrough InflightEntry (verify_task=None, lease=None,
    merge_wt=None) — the shape _dispatch_item returns for conflict /
    already_merged / skip_verify items.
    """
    immediate_outcome = MergeOutcome('blocked', reason='decided-elsewhere')
    item = SpeculativeItem(
        request=req,
        merge_result=None,
        merge_wt=None,
        base_sha='abc123',
        speculative=was_speculative,
        skip_verify=False,
        immediate_outcome=immediate_outcome,
    )
    entry = InflightEntry(
        item=item,
        lease=None,
        verify_task=None,
        merge_wt=None,
        was_speculative=was_speculative,
        phase='decided',
        passthrough_outcome=immediate_outcome,
    )
    return item, entry


@pytest.mark.asyncio
class TestPassthroughFinalizeErrorChokepoint:
    """PASSTHROUGH-FINALIZE path fault injection: _finalize_inflight raises
    while finalizing a passthrough (verify_task=None) entry.

    RED until step-6 GREEN routes both passthrough-finalize except-handlers
    through _resolve_and_release with release_resources=False (the real
    _finalize_inflight's finally clause is what would normally have released
    lease+slot; here it is replaced entirely by a raising stub, so NO release
    should occur at all — matching the current inline handler, which never
    touches the slot). RED marker: the chokepoint spy call count is 0.
    """

    @pytest.mark.parametrize(
        'driver', [_drive_verifier_loop_fill, _drive_verifier_loop_blocking_get],
        ids=['fill', 'blocking_get'],
    )
    async def test_passthrough_finalize_error_routes_through_chokepoint(
        self, git_ops: GitOps, git_repo: Path, config: OrchestratorConfig, driver: Any,
    ) -> None:
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req = _make_request(
            f'pt-{driver.__name__}', f'task/pt-{driver.__name__}', git_repo, config,
        )
        item, entry = _make_decided_item_and_entry(req, was_speculative=True)
        calls = _spy_on_resolve_and_release(worker)

        async def _passthrough_dispatch(_item: Any) -> Any:
            return entry

        async def _raising_finalize(_entry: Any) -> Any:
            raise Exception('boom')

        worker._dispatch_item = _passthrough_dispatch  # type: ignore[method-assign]
        worker._finalize_inflight = _raising_finalize  # type: ignore[method-assign]

        depth0 = worker._speculation_slot._value

        await driver(worker, item)

        assert len(calls) == 1, (
            f'Expected exactly one _resolve_and_release call, got {len(calls)}'
        )
        assert calls[0]['kwargs'].get('release_resources') is False, (
            f"Expected release_resources=False, got kwargs={calls[0]['kwargs']!r}"
        )
        assert req.result.done()
        outcome = req.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith('Verifier error:'), outcome.reason
        assert worker._n_failed is True
        assert worker._speculation_slot._value == depth0, (
            'the chokepoint must perform NO release when release_resources=False '
            '— the raising _finalize_inflight stub never ran its real finally'
        )

    @pytest.mark.parametrize(
        'driver', [_drive_verifier_loop_fill, _drive_verifier_loop_blocking_get],
        ids=['fill', 'blocking_get'],
    )
    async def test_passthrough_finalize_cancelled_error_propagates_and_skips_chokepoint(
        self, git_ops: GitOps, git_repo: Path, config: OrchestratorConfig, driver: Any,
    ) -> None:
        """CancelledError from _finalize_inflight (passthrough path) must
        propagate out of _verifier_loop untouched; the chokepoint must
        never see it.  Mirrors TestDispatchErrorChokepoint
        .test_dispatch_cancelled_error_propagates_and_skips_chokepoint for
        the passthrough-finalize except-clause ordering (``except
        (CancelledError, KeyboardInterrupt): raise`` ahead of ``except
        BaseException``) on both the fill-loop and blocking-get sites.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req = _make_request(
            f'pt-cancel-{driver.__name__}', f'task/pt-cancel-{driver.__name__}',
            git_repo, config,
        )
        item, entry = _make_decided_item_and_entry(req, was_speculative=True)
        calls = _spy_on_resolve_and_release(worker)

        async def _passthrough_dispatch(_item: Any) -> Any:
            return entry

        async def _cancelling_finalize(_entry: Any) -> Any:
            raise asyncio.CancelledError()

        worker._dispatch_item = _passthrough_dispatch  # type: ignore[method-assign]
        worker._finalize_inflight = _cancelling_finalize  # type: ignore[method-assign]

        with pytest.raises(asyncio.CancelledError):
            await driver(worker, item)

        assert len(calls) == 0, 'CancelledError must never reach the chokepoint'


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: FINALIZE-HEAD path (~7181)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeHeadErrorChokepoint:
    """FINALIZE-HEAD path fault injection: _finalize_inflight raises while
    finalizing the _inflight head — the FINALIZE-HEAD branch of _verifier_loop
    (distinct from the inline passthrough finalize inside DISPATCH-FILL: this
    entry carries a real verify_task, so it is appended to _inflight and
    finalized by the ``if self._inflight:`` block instead).

    RED until step-8 GREEN routes the finalize-head except-handler through
    _resolve_and_release with release_resources=False — mirrors the
    passthrough-finalize handlers: _finalize_inflight's own finally clause
    would normally have released lease+slot on every exit path including
    exceptions, but here the real _finalize_inflight is replaced by a raising
    stub so its finally never runs, and NO release should occur at all.
    RED marker: the chokepoint spy call count is 0 (the handler still inlines
    resolve logic via ``isinstance(exc, (CancelledError, KeyboardInterrupt))``
    + ``req.result.set_result``).
    """

    async def test_finalize_head_error_routes_through_chokepoint(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/fh-head', 'fh_head.py', 'x = 1\n',
            speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)

        async def _noop() -> None:
            return None

        verify_task = asyncio.ensure_future(_noop())
        await verify_task  # dummy completed/awaitable task, per the plan

        entry = InflightEntry(
            item=item,
            lease=MagicMock(),  # fake lease; must stay untouched (release_resources=False)
            verify_task=verify_task,
            merge_wt=item.merge_wt,
            was_speculative=True,
            phase='finalizing',
        )

        async def _dispatch_returns_entry(_item: Any) -> Any:
            return entry

        async def _raising_finalize(_entry: Any) -> Any:
            raise Exception('boom')

        worker._dispatch_item = _dispatch_returns_entry  # type: ignore[method-assign]
        worker._finalize_inflight = _raising_finalize  # type: ignore[method-assign]

        class _StubAllocator:
            """free_host_count()==0 so DISPATCH-FILL stops after this one item
            and proceeds straight to FINALIZE-HEAD, instead of continuing to
            fetch the queued None sentinel (which would hit the unguarded
            shutdown-drain loop's ``await self._finalize_inflight(head)``
            instead of the FINALIZE-HEAD try/except this test targets).
            """

            @staticmethod
            def free_host_count() -> int:
                return 0

        worker._ensure_host_allocator = (  # type: ignore[method-assign]
            lambda _config: _StubAllocator()
        )

        calls = _spy_on_resolve_and_release(worker)
        depth0 = worker._speculation_slot._value

        await _drive_verifier_loop_fill(worker, item)

        assert len(calls) == 1, (
            f'Expected exactly one _resolve_and_release call, got {len(calls)}'
        )
        assert calls[0]['kwargs'].get('release_resources') is False, (
            f"Expected release_resources=False, got kwargs={calls[0]['kwargs']!r}"
        )
        assert req.result.done()
        outcome = req.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith('Verifier error:'), outcome.reason
        assert worker._n_failed is True
        assert worker._speculation_slot._value == depth0, (
            'the chokepoint must perform NO release when release_resources=False '
            '— the raising _finalize_inflight stub never ran its real finally'
        )
        assert item.merge_wt is not None and item.merge_wt.exists(), (
            'worktree must be untouched on disk (release_resources=False → no cleanup)'
        )
        assert item.merge_wt in worker._owned_merge_worktrees, (
            'worktree ledger entry must be untouched'
        )
        # Loop continues past the failed head: the only _inflight entry was
        # popped as head, so the cascade guard
        # (`not _head_advanced and self._inflight`) is False (empty deque) →
        # no cascade fires → the loop reaches the queued None sentinel and
        # returns cleanly (no exception propagated out of _verifier_loop, or
        # _drive_verifier_loop_fill's asyncio.wait_for would have raised).
        assert not worker._inflight

    async def test_finalize_head_cancelled_error_propagates_and_skips_chokepoint(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """CancelledError from _finalize_inflight while finalizing the
        _inflight head must propagate out of _verifier_loop untouched; the
        chokepoint must never see it.  Mirrors TestDispatchErrorChokepoint
        .test_dispatch_cancelled_error_propagates_and_skips_chokepoint for
        the finalize-head except-clause ordering.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _req, item = await _make_real_merged_item(
            git_ops, config, 'task/fh-head-cancel', 'fh_head_cancel.py', 'x = 1\n',
            speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)

        async def _noop() -> None:
            return None

        verify_task = asyncio.ensure_future(_noop())
        await verify_task  # dummy completed/awaitable task, per the plan

        entry = InflightEntry(
            item=item,
            lease=MagicMock(),
            verify_task=verify_task,
            merge_wt=item.merge_wt,
            was_speculative=True,
            phase='finalizing',
        )

        async def _dispatch_returns_entry(_item: Any) -> Any:
            return entry

        async def _cancelling_finalize(_entry: Any) -> Any:
            raise asyncio.CancelledError()

        worker._dispatch_item = _dispatch_returns_entry  # type: ignore[method-assign]
        worker._finalize_inflight = _cancelling_finalize  # type: ignore[method-assign]

        class _StubAllocator:
            """free_host_count()==0 so DISPATCH-FILL proceeds straight to
            FINALIZE-HEAD, mirroring the sibling fault-injection test.
            """

            @staticmethod
            def free_host_count() -> int:
                return 0

        worker._ensure_host_allocator = (  # type: ignore[method-assign]
            lambda _config: _StubAllocator()
        )

        calls = _spy_on_resolve_and_release(worker)

        with pytest.raises(asyncio.CancelledError):
            await _drive_verifier_loop_fill(worker, item)

        assert len(calls) == 0, 'CancelledError must never reach the chokepoint'


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: CASCADE path (~7326)
# ---------------------------------------------------------------------------
#
# These three scenarios reproduce a real head-failure cascade via the two-host
# harness from test_merge_queue_concurrent_verify.py (γ), mirroring
# TestCascadeErrorContainment there (which locked the pre-chokepoint
# in-body-release-then-flag behaviour this step preserves byte-for-byte) and
# TestHaltAndUnavailable.test_operator_halt_aborts_all_inflight (REQUEUED
# preservation).  The cascade's per-entry try/except cannot be exercised by
# directly pre-seeding _inflight + calling _verifier_loop() the way the
# dispatch/passthrough/finalize-head tests above do: it depends on real
# overlap (a genuine background verify_task to cancel, a real lease to
# cancel_and_release, _entry_status detected from the cancelled task's
# result) that only a real SpeculativeMergeWorker.run() pipeline produces
# faithfully.


@pytest.mark.asyncio
class TestCascadeErrorChokepoint:
    """CASCADE path fault injection + release idempotency (~7326).

    RED until step-10 GREEN routes the cascade except-handler through
    _resolve_and_release(_entry, outcome, chain_failed=True,
    release_resources=not _entry_released, cancel_lease=True).  RED marker
    (scenarios A/B): the chokepoint spy call count is 0 — the handler still
    inlines the not-_entry_released release logic instead of calling
    _resolve_and_release.
    """

    async def test_cascade_remerge_error_routes_through_chokepoint(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """Scenario A: in-body release already ran (_entry_released=True)
        BEFORE _remerge raises → chokepoint invoked with
        release_resources=False (the in-body release already freed the
        slot/lease/worktree; the chokepoint must not double-release them —
        only resolve req_b and flag the chain failed).

        Adapted from TestCascadeErrorContainment
        .test_cascade_remerge_error_does_not_kill_verifier_loop
        (test_merge_queue_concurrent_verify.py), which locks the same
        end-state; this variant additionally spies on _resolve_and_release.
        """
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                # N's verify: gate and fail.
                gate_a_entered.set()
                await gate_a_release.wait()
                return MagicMock(
                    passed=False, summary='test_failure', test_output='FAILED',
                    lint_output='', type_output='', category='test_failure',
                    timed_out=False, verify_skipped=False,
                )
            # Any subsequent call (req_c's local verify) passes immediately.
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        wt_a = await _make_branch_with_file(git_ops, 'task/rrcas-a', 'rrcas_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/rrcas-b', 'rrcas_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _inject_two_host_allocator(worker, gated_remote)
        calls = _spy_on_resolve_and_release(worker)

        # Capture initial slot depth so we can verify exact-once release later.
        depth0 = worker._speculation_slot._value

        event_loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='rrcas-a', branch='task/rrcas-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=event_loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='rrcas-b', branch='task/rrcas-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=event_loop.create_future(), lane='normal',
        )

        # ── Inject killer: _remerge raises RuntimeError for req_b's task_id ─
        _original_remerge = worker._remerge

        async def _killer_remerge(req: Any, started_mono: Any) -> Any:
            if req.task_id == 'rrcas-b':
                raise RuntimeError(
                    'Failed to create merge worktree: '
                    '/repo/.worktrees/rrcas-b No space left on device'
                )
            return await _original_remerge(req, started_mono)

        worker._remerge = _killer_remerge  # type: ignore[method-assign]

        outcome_b: MergeOutcome | None = None
        outcome_c: MergeOutcome | None = None

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await q.put(req_a)
            await q.put(req_b)

            # Wait for both verifies to enter (true concurrent overlap).
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # N fails → head-failure cascade fires → _killer_remerge raises for N+1.
            gate_a_release.set()

            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail, got status={outcome_a.status!r}.'
            )

            # Unblock N+1's inner verify coroutine so it exits cleanly (the
            # cascade already cancelled the outer verify_task).
            gate_b_release.set()

            with contextlib.suppress(TimeoutError):
                outcome_b = await asyncio.wait_for(req_b.result, timeout=5.0)

            # SLOT EXACT-ONCE (measured BEFORE stop(), which over-releases for
            # safety): see the MEASUREMENT NOTE in TestCascadeErrorContainment
            # (test_merge_queue_concurrent_verify.py) — the merger perpetually
            # holds one speculative look-ahead permit, so a single correct
            # release leaves _value at depth0 - 1; a double-release (naive
            # unconditional re-release) would push it back up to depth0.
            expected_slot = depth0 - 1
            assert worker._speculation_slot._value == expected_slot, (
                f'Expected speculation slot at depth0-1={expected_slot} '
                f'(merger holds one look-ahead permit), '
                f'got {worker._speculation_slot._value!r}.'
            )

            # Loop-survival signal: a third request should dispatch on the
            # local host and resolve "done".
            wt_c = await _make_branch_with_file(
                git_ops, 'task/rrcas-c', 'rrcas_c.py', 'c = 3\n'
            )
            req_c = MergeRequest(
                task_id='rrcas-c', branch='task/rrcas-c', worktree=wt_c,
                pre_rebased=False, task_files=None, module_configs=[],
                config=config, result=event_loop.create_future(), lane='normal',
            )
            await q.put(req_c)

            with contextlib.suppress(TimeoutError):
                outcome_c = await asyncio.wait_for(req_c.result, timeout=10.0)

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # ── Loop survives + offending req blocked (pre-existing 1856 guard) ──
        assert outcome_c is not None and outcome_c.status == 'done', (
            f'Expected req_c to resolve "done" (loop survived cascade error), '
            f'got {outcome_c!r}.'
        )
        assert outcome_b is not None and outcome_b.status == 'blocked', (
            f'Expected cascade error to resolve req_b as "blocked", got {outcome_b!r}.'
        )
        assert outcome_b.reason is not None and outcome_b.reason.startswith(
            'Verifier cascade error:'
        ), outcome_b.reason
        assert gated_remote.cancel_verify.call_count >= 1, (
            'Expected at least one cancel_verify call for N+1 lease release.'
        )
        assert worker_task.done(), 'Worker task should be done after stop() + await'
        assert worker_task.exception() is None, (
            f'Expected worker task to exit cleanly, got {worker_task.exception()!r}.'
        )

        # ── Chokepoint assertions (task 1991) ────────────────────────────────
        assert len(calls) == 1, (
            f'Expected exactly one _resolve_and_release call, got {len(calls)}'
        )
        assert calls[0]['kwargs'].get('release_resources') is False, (
            f"Expected release_resources=False (in-body release already ran "
            f"before _remerge raised), got kwargs={calls[0]['kwargs']!r}"
        )
        assert calls[0]['kwargs'].get('cancel_lease') is True, (
            f"Expected cancel_lease=True, got kwargs={calls[0]['kwargs']!r}"
        )
        entry_arg = calls[0]['args'][0]
        assert entry_arg.item.request.task_id == 'rrcas-b'
        assert entry_arg.merge_wt is not None and not entry_arg.merge_wt.exists(), (
            'downstream merge worktree must be cleaned (by the in-body release; '
            'the chokepoint must not need to touch it again)'
        )
        assert entry_arg.merge_wt not in worker._owned_merge_worktrees

    async def test_cascade_cancel_and_release_raises_routes_through_chokepoint(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """Scenario B: in-body release raises on its FIRST statement
        (cancel_and_release) so _entry_released stays False → chokepoint
        invoked with release_resources=True AND cancel_lease=True; the
        chokepoint itself must perform the (best-effort, exactly-once)
        release of the lease, worktree, and speculation slot.

        Adapted from TestCascadeErrorContainment
        .test_cascade_cancel_and_release_raises_contained
        (test_merge_queue_concurrent_verify.py).
        """
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
                return MagicMock(
                    passed=False, summary='test_failure', test_output='FAILED',
                    lint_output='', type_output='', category='test_failure',
                    timed_out=False, verify_skipped=False,
                )
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        wt_a = await _make_branch_with_file(git_ops, 'task/rrcr-a', 'rrcr_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/rrcr-b', 'rrcr_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        allocator = _inject_two_host_allocator(worker, gated_remote)
        calls = _spy_on_resolve_and_release(worker)

        depth0 = worker._speculation_slot._value

        event_loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='rrcr-a', branch='task/rrcr-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=event_loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='rrcr-b', branch='task/rrcr-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=event_loop.create_future(), lane='normal',
        )

        # ── Inject: allocator.cancel_and_release raises on its first call ──
        _original_cancel_and_release = allocator.cancel_and_release
        _cancel_call_count: list[int] = [0]

        async def _raising_cancel_and_release(lease: Any, **kwargs: Any) -> Any:
            _cancel_call_count[0] += 1
            if _cancel_call_count[0] == 1:
                raise RuntimeError('Simulated cancel_and_release failure')
            return await _original_cancel_and_release(lease, **kwargs)

        allocator.cancel_and_release = _raising_cancel_and_release  # type: ignore[method-assign]

        outcome_b: MergeOutcome | None = None
        outcome_c: MergeOutcome | None = None

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await q.put(req_a)
            await q.put(req_b)

            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # N fails → head-failure cascade fires → in-body cancel_and_release raises.
            gate_a_release.set()

            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail, got status={outcome_a.status!r}.'
            )

            gate_b_release.set()

            with contextlib.suppress(TimeoutError):
                outcome_b = await asyncio.wait_for(req_b.result, timeout=5.0)

            # Loop-survival signal — see the NOTE in the sibling γ-harness
            # test: in this secondary-failure path set_result precedes the
            # slot release, so the slot assertion is placed AFTER outcome_c
            # resolves (below), not here.
            wt_c = await _make_branch_with_file(
                git_ops, 'task/rrcr-c', 'rrcr_c.py', 'c = 3\n'
            )
            req_c = MergeRequest(
                task_id='rrcr-c', branch='task/rrcr-c', worktree=wt_c,
                pre_rebased=False, task_files=None, module_configs=[],
                config=config, result=event_loop.create_future(), lane='normal',
            )
            await q.put(req_c)

            with contextlib.suppress(TimeoutError):
                outcome_c = await asyncio.wait_for(req_c.result, timeout=10.0)

            expected_slot = depth0 - 1  # one permit held by the merger look-ahead
            assert worker._speculation_slot._value == expected_slot, (
                f'Expected speculation slot at depth0-1={expected_slot} '
                f'(merger holds one look-ahead permit), '
                f'got {worker._speculation_slot._value!r}.'
            )

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        assert outcome_c is not None and outcome_c.status == 'done', (
            f'Expected req_c to resolve "done" (loop survived cascade error), '
            f'got {outcome_c!r}.'
        )
        assert outcome_b is not None and outcome_b.status == 'blocked', (
            f'Expected cascade error to resolve req_b as "blocked", got {outcome_b!r}.'
        )
        assert outcome_b.reason is not None and outcome_b.reason.startswith(
            'Verifier cascade error:'
        ), outcome_b.reason
        assert _cancel_call_count[0] >= 2, (
            f'Expected cancel_and_release called at least twice (in-body: raises; '
            f'chokepoint: suppressed-then-succeeds), got {_cancel_call_count[0]}.'
        )
        assert worker_task.done(), 'Worker task should be done after stop() + await'
        assert worker_task.exception() is None, (
            f'Expected worker task to exit cleanly, got {worker_task.exception()!r}.'
        )

        # ── Chokepoint assertions (task 1991) ────────────────────────────────
        assert len(calls) == 1, (
            f'Expected exactly one _resolve_and_release call, got {len(calls)}'
        )
        assert calls[0]['kwargs'].get('release_resources') is True, (
            f"Expected release_resources=True (in-body release never completed — "
            f"_entry_released stayed False), got kwargs={calls[0]['kwargs']!r}"
        )
        assert calls[0]['kwargs'].get('cancel_lease') is True, (
            f"Expected cancel_lease=True, got kwargs={calls[0]['kwargs']!r}"
        )
        entry_arg = calls[0]['args'][0]
        assert entry_arg.item.request.task_id == 'rrcr-b'
        assert entry_arg.merge_wt is not None and not entry_arg.merge_wt.exists(), (
            'downstream merge worktree must be cleaned BY THE CHOKEPOINT '
            '(in-body cleanup never ran — the raise happened before it)'
        )
        assert entry_arg.merge_wt not in worker._owned_merge_worktrees

    async def test_cascade_requeued_entry_skips_chokepoint(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """Scenario C: a downstream entry whose (cancelled) verify result
        status is REQUEUED (operator halt) must NOT be routed through the
        chokepoint — it `continue`s on the cascade's normal (non-exceptional)
        path; its request is resolved elsewhere (the abort-poll's own
        requeue-to-_queue, handled by stop()'s drain), never by
        _resolve_and_release.

        Adapted from TestHaltAndUnavailable.test_operator_halt_aborts_all_inflight
        (test_merge_queue_concurrent_verify.py), which locks the sibling
        _remerge-not-called invariant this scenario complements.
        """
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            gate_a_entered.set()
            await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        wt_a = await _make_branch_with_file(git_ops, 'task/rrhalt-a', 'rrhalt_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/rrhalt-b', 'rrhalt_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _inject_two_host_allocator(worker, gated_remote)
        worker.VERIFY_ABANDON_POLL_SECS = 0.01  # fast abort-poll for determinism
        calls = _spy_on_resolve_and_release(worker)

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='rrhalt-a', branch='task/rrhalt-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='rrhalt-b', branch='task/rrhalt-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        # Spy on _remerge too: the sibling γ-harness test's invariant
        # (REQUEUED entries are never re-merged) must keep holding — a
        # regression that routed a REQUEUED entry through the chokepoint
        # would also, wrongly, have called _remerge for it beforehand.
        _original_remerge = worker._remerge
        remerge_task_ids: list[str] = []

        async def _spy_remerge(req: Any, started_mono: Any) -> Any:
            remerge_task_ids.append(req.task_id)
            return await _original_remerge(req, started_mono)

        worker._remerge = _spy_remerge  # type: ignore[method-assign]

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await q.put(req_a)
            await q.put(req_b)

            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # Set operator halt — both abort-polls fire within 0.01s.
            worker._operator_halt.set()

            # Give abort-polls time to fire and requeue.
            await asyncio.sleep(0.15)

            # Release gates so the leaked inner tasks can complete (harmlessly).
            gate_a_release.set()
            gate_b_release.set()

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        assert req_a.result.done(), 'req_a.result should be resolved by stop()'
        assert req_b.result.done(), 'req_b.result should be resolved by stop()'
        assert remerge_task_ids == [], (
            f'Expected no _remerge calls after halt (each item self-requeues), '
            f'but _remerge was called for: {remerge_task_ids!r}.'
        )

        # ── Chokepoint assertion (task 1991): REQUEUED never reaches it ─────
        assert calls == [], (
            f'Expected _resolve_and_release to never be called for a REQUEUED '
            f'cascade entry (it `continue`s on the normal path), but got '
            f'{len(calls)} call(s): {calls!r}'
        )
