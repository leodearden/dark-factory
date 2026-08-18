"""DISPATCH-FILL redispatch-drain tests for task 3276.

Pins two invariants of the DISPATCH-FILL loop's early-stop guard
(``_verifier_loop``, ``orchestrator/src/orchestrator/merge_queue.py``):

* A redispatch-sourced dispatch that drains ``self._redispatch`` must not
  end the fill pass while ``self._verifier_queue`` still holds a ready
  item and a host slot is free -- whether that item was already queued
  before the dispatch, or arrives a moment after.
* Once ``self._redispatch`` and ``self._verifier_queue`` are genuinely
  both empty, the loop must still fall through to FINALIZE-HEAD rather
  than hang -- the anti-deadlock property a since-deleted redispatch-
  specific special case used to provide is now structural, coming from
  the fill loop's other fall-through paths.

See merge_queue.py's comment on the guard (immediately above
``allocator = self._ensure_host_allocator(...)`` in the DISPATCH-FILL
tail) for the current predicate and the traced fall-through argument for
why the anti-deadlock property holds without a redispatch-specific
special case.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest
from _orch_helpers import MERGE_RESULT_TIMEOUT

# Reuse the γ multi-host test harness (established cross-test-module import
# pattern -- orchestrator/tests/ has no __init__.py; precedent:
# test_concurrent_verify_boundary.py:33) -- single source of truth for the
# fixtures/fakes, zero churn to the landed γ test file.
from test_merge_queue_concurrent_verify import (
    _inject_two_host_allocator,
    _make_fake_remote,
    _make_request,
    config,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_config,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_ops,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_repo,  # noqa: F401 — pytest fixture re-exported from γ harness
)

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult
from orchestrator.merge_queue import (
    InflightEntry,
    ItemLifecycleState,
    MergeOutcome,
    RealMergeItem,
    SpeculativeMergeWorker,
)
from orchestrator.verify_runner import HostAllocator

# ---------------------------------------------------------------------------
# Shared harness: drive the REAL _verifier_loop against a REAL two-host
# HostAllocator, with _dispatch_item/_finalize_inflight replaced by small
# recording stubs that take/release real leases and hold a gated
# asyncio.Event-backed verify task.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FillDrive:
    """Recording state installed onto a worker by :func:`_drive_fill`.

    ``dispatched``       : items passed to the stubbed ``_dispatch_item``, in
                            call order.
    ``first_dispatched``: set inside the stub once the FIRST call's lease is
                            held (i.e. after ``allocator.acquire()`` returns
                            non-None) -- so a waiter observes
                            ``free_host_count()`` already reflecting the
                            dispatch, not merely "acquire() was called".
    ``second_dispatched``: same, for the SECOND call.
    ``gate``              : shared, test-controlled ``asyncio.Event``. Every
                            dispatched entry's ``verify_task`` blocks on
                            ``gate.wait()`` -- so real verify/git machinery
                            never runs, and the test decides exactly when a
                            dispatched item's verify "completes".
    """

    dispatched: list[Any] = dataclasses.field(default_factory=list)
    first_dispatched: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    second_dispatched: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    gate: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)


def _drive_fill(worker: SpeculativeMergeWorker, allocator: HostAllocator) -> _FillDrive:
    """Install recording ``_dispatch_item``/``_finalize_inflight`` stubs on *worker*.

    Keeps the fill loop's REAL control flow (item acquisition order,
    ``_note_transition``, ``_inflight_append``, the DISPATCH-FILL guard,
    FINALIZE-HEAD) and the REAL ``HostAllocator`` lease accounting under
    test, while removing real git merges and real verify runs -- so
    ``allocator.free_host_count()`` is a genuine measurement of
    simultaneously-held host leases (the task's user-observable "heartbeat
    shows 2/2" signal), not a call-count assertion on a mock.
    """
    drive = _FillDrive()

    async def _fake_dispatch_item(item: Any) -> InflightEntry | None:
        drive.dispatched.append(item)
        lease = await allocator.acquire(lambda: MagicMock())
        if lease is None:
            return None
        # Signal only after the lease is confirmed held, not merely
        # requested: a waiter on first_dispatched/second_dispatched must
        # observe free_host_count() already reflecting this dispatch. This
        # happens to hold either way today (HostAllocator.acquire has no
        # real await point), but pin the ordering explicitly rather than
        # relying on that incidental property.
        if len(drive.dispatched) == 1:
            drive.first_dispatched.set()
        elif len(drive.dispatched) == 2:
            drive.second_dispatched.set()
        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=asyncio.ensure_future(drive.gate.wait()),
            merge_wt=None,
            was_speculative=False,
        )

    async def _fake_finalize_inflight(entry: InflightEntry) -> bool:
        # _fake_dispatch_item (above) always builds entries with a real
        # verify_task -- this stub never produces a passthrough
        # (verify_task=None) entry -- so narrow loudly rather than silently
        # no-op on a None this harness should never produce (matches the
        # `if entry.verify_task is not None:` narrowing the real
        # _finalize_inflight uses for its passthrough case).
        assert entry.verify_task is not None, (
            '_fake_dispatch_item always sets a real verify_task; a None here '
            'means the harness constructed an unexpected passthrough entry'
        )
        await entry.verify_task
        if entry.lease is not None:
            await allocator.release(entry.lease)
        result = entry.item.request.result
        if not result.done():
            result.set_result(MergeOutcome('done'))
        return False

    worker._dispatch_item = _fake_dispatch_item  # type: ignore[method-assign]
    worker._finalize_inflight = _fake_finalize_inflight  # type: ignore[method-assign]
    return drive


async def _teardown_fill_drive(
    drive: _FillDrive,
    task: asyncio.Task,  # type: ignore[type-arg]
    worker: SpeculativeMergeWorker,
) -> None:
    """Unblock and shut down a ``_verifier_loop`` task driven by :func:`_drive_fill`.

    Shared by every test in this module so a RED failure and a GREEN success
    path both leave no dangling task / pending-task warning behind: release
    every gated verify task, shut the loop down, and cancel any persistent
    ``_pending_verifier_get`` getter the QueueEmpty race may have launched.

    Teardown goes through ``worker.stop()`` -- NOT a bare ``task.cancel()`` --
    mirroring how ``TestLastItemOfBurstFinalizes``
    (``test_merge_queue_concurrent_verify.py``) tears down the analogous
    queue-sourced steady state.  ``stop()``'s protocol (resolve/cancel
    ``_pending_verifier_get``, drain, then a ``None`` sentinel on
    ``_verifier_queue``) is the teardown ``_verifier_loop``'s FINALIZE-HEAD
    "Reuse the persistent getter" branch actually unwinds from, and it is
    internally bounded (its ``asyncio.wait(..., timeout=...)``), so it cannot
    hang this helper.

    A bare ``task.cancel()`` is NOT sufficient there, for a reason that is
    **pre-existing and unrelated to task 3276**: that branch's recovery clause
    (``except asyncio.CancelledError: item = await self._verifier_queue.get()``
    in ``merge_queue.py``) was written for ``stop()``'s ordering and cannot
    distinguish "only ``_pending_verifier_get`` was cancelled" from "the whole
    ``_verifier_loop`` task is being cancelled".  It absorbs the single
    cancellation request delivered through the transitively-cancelled getter
    and re-parks on a fresh, uncancelled ``get()``, so ``await task`` never
    returns.  This reproduces on the pre-task-3276 baseline for a purely
    queue-sourced dispatch (a shape the DISPATCH-FILL guard never gated), and
    is reachable in production from ``SpeculativeMergeWorker.run()``'s own raw
    ``cancel()``/``gather()`` shutdown path; it is filed as its own follow-up,
    task 4306.
    These tests fence the DISPATCH-FILL predicate, not cancellation semantics,
    so they must not depend on that path being clean -- the loop's own
    docstring already flags raw external cancellation as an accepted edge case.
    """
    drive.gate.set()
    # Exception, not BaseException: this must not swallow a CancelledError
    # aimed at the enclosing test task (or a KeyboardInterrupt) -- only
    # absorb worker.stop()'s own failures, so a genuine stop() regression
    # is never silently masked into a falsely-clean teardown.
    with contextlib.suppress(Exception):
        await worker.stop()
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    if worker._pending_verifier_get is not None:
        pending = worker._pending_verifier_get
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


def _make_real_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    task_id: str,
    base_sha: str,
) -> RealMergeItem:
    """Build a dispatch-ready RealMergeItem over a fresh MergeRequest.

    Field shape copied from TestStopDrainsInflight
    (test_merge_queue_concurrent_verify.py:3131-3144) -- the established
    direct-construction pattern for loop-driving tests.
    """
    req = _make_request(task_id, f'task/{task_id}', git_ops.project_root, config)
    wt = git_ops.project_root / '.worktrees' / task_id
    return RealMergeItem(
        request=req,
        merge_result=MergeResult(success=True, merge_commit='deadbeef', merge_worktree=wt),
        merge_wt=wt,
        base_sha=base_sha,
        speculative=False,
    )


# ---------------------------------------------------------------------------
# step-1: primary repro
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRedispatchDrainDoesNotEndFillPass:
    """A redispatch-sourced dispatch that drains ``self._redispatch`` must
    not end the DISPATCH-FILL pass while ``self._verifier_queue`` still
    holds a ready item and a host slot is free -- task 3276.

    Two shapes are pinned: the item already queued before the guard
    evaluates (below), and the item arriving a moment after (see
    ``test_late_arrival_dispatched_to_free_host_while_head_verify_runs``).
    """

    async def test_second_host_dispatched_from_verifier_queue_in_same_fill_pass(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """A redispatch-sourced dispatch that drains ``_redispatch`` must NOT
        end the fill pass while ``_verifier_queue`` still holds a ready item
        and a host slot is free.

        item_b is dispatched to the free second host in the SAME fill pass,
        immediately after item_a -- both leases held simultaneously. A
        regression here means the fill pass ends as soon as ``_redispatch``
        drains, leaving item_b stranded in ``_verifier_queue`` and the
        laptop host slot idle while the loop blocks in FINALIZE-HEAD
        awaiting item_a's verify instead.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        allocator = _inject_two_host_allocator(worker, _make_fake_remote('laptop'))
        assert allocator.free_host_count() == 2, 'precondition: two free hosts'

        drive = _drive_fill(worker, allocator)

        item_a = _make_real_item(git_ops, config, 'rd-fill-a', 'aaa')
        item_b = _make_real_item(git_ops, config, 'rd-fill-b', 'bbb')

        # Pre-register at the items' TRUE parked states so _note_transition's
        # REDISPATCH_PARKED/AWAITING_VERIFY -> DISPATCHING hop finds a
        # registered request_id and files no spurious L1
        # illegal_lifecycle_transition escalation (task 2169 registry).
        worker._register_item(item_a, initial=ItemLifecycleState.REDISPATCH_PARKED)
        worker._register_item(item_b, initial=ItemLifecycleState.AWAITING_VERIFY)

        worker._redispatch.append(item_a)
        await worker._verifier_queue.put(item_b)

        task = asyncio.ensure_future(worker._verifier_loop())
        # _assert_single_writer is live (ORCH_DEBUG_ASSERTS=1, conftest.py)
        # and self._running is True from construction -- the loop's first
        # _inflight_append would otherwise raise on the "wrong" owner task.
        worker._verifier_task = task

        try:
            await asyncio.wait_for(drive.second_dispatched.wait(), timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                'item_b (in _verifier_queue) was never dispatched within '
                f'{MERGE_RESULT_TIMEOUT}s of item_a (from _redispatch) being '
                'dispatched. This means the DISPATCH-FILL guard is ending '
                'the fill pass as soon as _redispatch drains, even though '
                '_verifier_queue still holds item_b and a host slot is '
                "free -- the loop falls through to FINALIZE-HEAD and blocks "
                "on item_a's gated verify instead of dispatching item_b to "
                'the free second host in the same fill pass.'
            )

        assert drive.dispatched == [item_a, item_b], (
            f'expected [item_a, item_b] dispatched in order, got {drive.dispatched!r}'
        )
        assert allocator.free_host_count() == 0, (
            'expected both host slots held simultaneously (local for item_a, '
            f'laptop for item_b); free_host_count() == {allocator.free_host_count()}'
        )

        await _teardown_fill_drive(drive, task, worker)

    async def test_late_arrival_dispatched_to_free_host_while_head_verify_runs(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """A late-arriving item -- one that lands in ``_verifier_queue`` a
        moment AFTER the DISPATCH-FILL guard has already evaluated -- must
        still be dispatched to a free host while the head's verify is still
        running, not left to wait out the entire head verify.

        Same shape as
        ``test_second_host_dispatched_from_verifier_queue_in_same_fill_pass``
        except ``_verifier_queue`` starts EMPTY: item_b is registered but not
        yet queued when the loop starts, and only arrives after item_a has
        already been dispatched and the guard has already run.

        Dispatching item_a falls through to the QueueEmpty multi-host
        fill-ahead race (``asyncio.wait`` over the persistent getter +
        running verify tasks), which picks up item_b the moment it is
        put() and dispatches it to the free second host. A guard that
        merely re-checks ``self._verifier_queue.empty()`` at decision time
        is NOT sufficient to pass this test: that snapshot is empty at the
        instant the guard evaluates (item_b has not arrived yet), so a
        snapshot-based guard would still end the fill pass there instead of
        letting the fall-through paths keep filling.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        allocator = _inject_two_host_allocator(worker, _make_fake_remote('laptop'))
        assert allocator.free_host_count() == 2, 'precondition: two free hosts'

        drive = _drive_fill(worker, allocator)

        item_a = _make_real_item(git_ops, config, 'rd-late-a', 'aaa')
        item_b = _make_real_item(git_ops, config, 'rd-late-b', 'bbb')

        worker._register_item(item_a, initial=ItemLifecycleState.REDISPATCH_PARKED)
        worker._register_item(item_b, initial=ItemLifecycleState.AWAITING_VERIFY)

        worker._redispatch.append(item_a)
        # _verifier_queue starts EMPTY -- item_b arrives only after the
        # guard has already run (see below), not before the loop starts.

        task = asyncio.ensure_future(worker._verifier_loop())
        worker._verifier_task = task

        try:
            await asyncio.wait_for(drive.first_dispatched.wait(), timeout=MERGE_RESULT_TIMEOUT)

            # The guard has now evaluated (deterministically -- the loop
            # yields to the event loop for the first time only after the
            # tail guard's break/continue decision has already been made).
            # item_b arrives NOW, a moment later, with item_a's verify still
            # gated (running) and a host slot still free.
            await worker._verifier_queue.put(item_b)

            await asyncio.wait_for(drive.second_dispatched.wait(), timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                'item_b was never dispatched within '
                f'{MERGE_RESULT_TIMEOUT}s of arriving in _verifier_queue '
                "while item_a's verify was still running and a host was "
                'free. This means the DISPATCH-FILL guard is stopping the '
                'fill pass on an empty-queue snapshot taken before item_b '
                'arrived, so the loop commits to FINALIZE-HEAD and blocks '
                "on item_a's gated verify -- item_b arriving moments later "
                'changes nothing. A guard that merely re-checks '
                '_verifier_queue.empty() at decision time forfeits this '
                'real, imminent work; a free host must keep falling '
                'through to the QueueEmpty fill-ahead race instead of a '
                'special-cased early exit.'
            )

        assert drive.dispatched == [item_a, item_b], (
            f'expected [item_a, item_b] dispatched in order, got {drive.dispatched!r}'
        )
        assert allocator.free_host_count() == 0, (
            'expected both host slots held simultaneously; '
            f'free_host_count() == {allocator.free_host_count()}'
        )

        await _teardown_fill_drive(drive, task, worker)


# ---------------------------------------------------------------------------
# step-3(b): anti-deadlock invariant fence (NOT a RED/GREEN pair)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCascadeAntiDeadlockPreserved:
    """Fences the anti-deadlock property the original DISPATCH-FILL guard's
    comment claimed to provide -- task 3276.

    Not a regression pin on a specific fix, but a permanent invariant
    fence: this test must stay green regardless of how the DISPATCH-FILL
    guard is implemented, proving that emptying ``_redispatch`` does not
    reintroduce the "blocking on _verifier_queue.get() ... would deadlock
    when the queue is empty after a cascade" hazard the original guard's
    comment named -- see merge_queue.py's comment on the guard for the
    traced fall-through argument this test backs up empirically.
    """

    async def test_empty_queue_after_redispatch_drain_still_finalizes_head(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """_redispatch drains to empty, _verifier_queue stays empty for the
        whole test (no cascade follow-on work ever arrives), two hosts free.

        FINALIZE-HEAD must still be reached and the merge() caller's result
        Future must still be delivered once item_a's verify completes on its
        own -- i.e. the loop must not hang forever waiting for a
        _verifier_queue arrival that never comes.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        allocator = _inject_two_host_allocator(worker, _make_fake_remote('laptop'))
        assert allocator.free_host_count() == 2, 'precondition: two free hosts'

        drive = _drive_fill(worker, allocator)

        item_a = _make_real_item(git_ops, config, 'rd-fence-a', 'aaa')
        worker._register_item(item_a, initial=ItemLifecycleState.REDISPATCH_PARKED)
        worker._redispatch.append(item_a)
        # _verifier_queue stays EMPTY for the whole test -- no follow-on
        # work ever arrives; this is the exact cascade-drain shape the
        # original guard's comment described.

        task = asyncio.ensure_future(worker._verifier_loop())
        worker._verifier_task = task

        try:
            await asyncio.wait_for(drive.first_dispatched.wait(), timeout=MERGE_RESULT_TIMEOUT)
            drive.gate.set()  # item_a's verify completes on its own now.

            outcome = await asyncio.wait_for(
                item_a.request.result, timeout=MERGE_RESULT_TIMEOUT
            )
        except TimeoutError:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                "item_a's result Future was never resolved within "
                f'{MERGE_RESULT_TIMEOUT}s of its gated verify completing, '
                'with _verifier_queue empty the whole time. FINALIZE-HEAD '
                'was never reached -- the anti-deadlock property the '
                'original guard was written to provide has been lost.'
            )

        assert outcome.status == 'done', f'expected a done outcome, got {outcome!r}'

        await _teardown_fill_drive(drive, task, worker)


# ---------------------------------------------------------------------------
# single-host coverage: the redispatch-drain shape under the surviving clause
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSingleHostRedispatchSourceDegeneracy:
    """The surviving ``free_host_count() == 0`` clause stops the fill pass
    after exactly one dispatch on a single host, whether that dispatch came
    from ``_redispatch`` or ``_verifier_queue`` -- task 3276.

    ``TestSingleHostSerialByteIdentical`` (test_merge_queue_concurrent_
    verify.py) already pins this for a queue-sourced dispatch. Since this
    task's whole premise is that redispatch-sourced and queue-sourced
    dispatch had diverged, this closes the one shape that was otherwise
    governed only by the surviving clause with no direct test: a
    redispatch-sourced dispatch on a single host.
    """

    async def test_single_host_stops_after_one_redispatch_sourced_dispatch(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """One host, one item in ``_redispatch``, a second already queued in
        ``_verifier_queue``: only the first is dispatched before the loop
        commits to FINALIZE-HEAD.

        No fake remote is injected, so the allocator's only slot is
        'local'. Once item_a's dispatch takes it, free_host_count() == 0
        makes the surviving guard clause fire immediately and
        synchronously -- there is no await point between that decision and
        ``_finalize_inflight``'s ``await entry.verify_task`` (item
        acquisition, ``_note_transition`` and ``_inflight_append`` are all
        synchronous), so by the time ``first_dispatched`` (set only after
        item_a's lease is held -- see ``_fake_dispatch_item``) resumes this
        test, the loop is already parked in FINALIZE-HEAD and item_b is
        guaranteed untouched.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        allocator = HostAllocator([], quarantine=worker._runner_quarantine)
        worker._host_allocator = allocator
        assert allocator.free_host_count() == 1, 'precondition: one free (local) host'

        drive = _drive_fill(worker, allocator)

        item_a = _make_real_item(git_ops, config, 'rd-1host-a', 'aaa')
        item_b = _make_real_item(git_ops, config, 'rd-1host-b', 'bbb')

        worker._register_item(item_a, initial=ItemLifecycleState.REDISPATCH_PARKED)
        worker._register_item(item_b, initial=ItemLifecycleState.AWAITING_VERIFY)

        worker._redispatch.append(item_a)
        await worker._verifier_queue.put(item_b)

        task = asyncio.ensure_future(worker._verifier_loop())
        worker._verifier_task = task

        try:
            await asyncio.wait_for(drive.first_dispatched.wait(), timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                'item_a (from _redispatch) was never dispatched within '
                f'{MERGE_RESULT_TIMEOUT}s on a single-host allocator.'
            )

        assert drive.dispatched == [item_a], (
            f'expected only item_a dispatched before FINALIZE-HEAD, got {drive.dispatched!r}'
        )
        assert allocator.free_host_count() == 0, (
            'expected the sole host slot held; '
            f'free_host_count() == {allocator.free_host_count()}'
        )
        assert not drive.second_dispatched.is_set(), (
            'item_b must not be dispatched in the same fill pass on a single '
            'host -- the surviving free_host_count() == 0 clause must stop '
            'the fill pass after exactly one dispatch regardless of source'
        )
        assert worker._verifier_queue.qsize() == 1, (
            'item_b must still be sitting untouched in _verifier_queue'
        )

        await _teardown_fill_drive(drive, task, worker)
