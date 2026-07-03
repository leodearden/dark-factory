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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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
