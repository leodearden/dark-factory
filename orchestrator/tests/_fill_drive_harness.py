"""The white-box DISPATCH-FILL drive harness, extracted from
``test_merge_queue_dispatch_fill_redispatch.py`` by task 5030 (PRD
``plans/merge-lane-quality-prd.md`` task γ7).

It installs fake ``_dispatch_item``/``_finalize_inflight`` methods on a worker
so a test can start ``_verifier_loop()`` itself and observe the loop's control
flow with no real git merge or verify underneath. That is exactly the coupling
γ7 removed from its own tests, which now drive a real lane through its public
surface instead.

IT LIVES ON FOR ONE CONSUMER: ``test_merge_queue_verifier_raw_cancel.py``,
which fences the raw-``task.cancel()`` termination of ``_verifier_loop``
(task 4306) and therefore has to start that loop by hand. That file belongs to
a different γ group, so γ7 did not migrate it and did not edit it -- the
harness moved here verbatim, and
``test_merge_queue_dispatch_fill_redispatch.py`` re-exports these names so the
consumer's import keeps resolving unchanged.

DELETE THIS MODULE (and that re-export) when the raw-cancel file's own γ group
migrates it or retires it. Nothing else may import it: a new white-box drive
built on this harness re-opens the seam the PRD is closing.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from typing import Any
from unittest.mock import MagicMock

from test_merge_queue_concurrent_verify import _make_request

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult
from orchestrator.merge_queue import (
    InflightEntry,
    MergeOutcome,
    RealMergeItem,
    SpeculativeMergeWorker,
)
from orchestrator.verify_runner import HostAllocator


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

    A bare ``task.cancel()`` used to be insufficient there, for a reason that
    was **pre-existing and unrelated to task 3276**: that branch's recovery
    clause (``except asyncio.CancelledError: item = await
    self._verifier_queue.get()`` in ``merge_queue.py``) was written for
    ``stop()``'s ordering and could not distinguish "only
    ``_pending_verifier_get`` was cancelled" from "the whole ``_verifier_loop``
    task is being cancelled".  It absorbed the single cancellation request
    delivered through the transitively-cancelled getter and re-parked on a
    fresh, uncancelled ``get()``, so ``await task`` never returned.  **Task
    4306 fixed that**: the clause now re-raises when
    ``asyncio.current_task().cancelling() > 0``, so a bare ``task.cancel()``
    DOES terminate the loop, and ``orchestrator/tests/
    test_merge_queue_verifier_raw_cancel.py`` fences that path.

    This helper nonetheless keeps ``stop()`` as its protocol -- a deliberate
    choice, not a workaround.  ``stop()`` is a strictly more complete teardown
    than a bare cancel: it additionally resolves in-flight request Futures,
    drains the queues, cleans merge worktrees and releases leases/permits, and
    it is internally bounded by its own ``asyncio.wait(..., timeout=...)`` so
    it cannot hang this helper.
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
