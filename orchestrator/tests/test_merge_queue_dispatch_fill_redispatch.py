"""DISPATCH-FILL redispatch-drain tests for task 3276.

The DISPATCH-FILL loop's early-stop guard (``_verifier_loop``,
``orchestrator/src/orchestrator/merge_queue.py``) tests whether
``self._redispatch`` has just drained::

    if allocator.free_host_count() == 0 or (
        not is_from_verifier_queue and not self._redispatch
    ):
        fill_done = True
        break

but the hazard the guard's own comment names is an EMPTY
``self._verifier_queue`` (blocking on ``_verifier_queue.get()`` would
deadlock after a cascade). Those are different conditions: when a
redispatch-sourced item is dispatched and ``_redispatch`` happens to drain
on that exact dispatch, the fill pass ends even though ``_verifier_queue``
still holds a ready item and a host slot is free. The loop then falls
through to FINALIZE-HEAD and blocks on the head's verify -- during which NO
further dispatch of any kind can happen, even though a free host sits idle
with ready work waiting in the queue.

Step map:
  step-1 RED  -- primary repro: a second, already-merged, ready item sitting
                 in ``_verifier_queue`` is never dispatched to the free
                 second host in the same fill pass as a redispatch-sourced
                 dispatch that drains ``_redispatch``.
  step-2 impl -- GREEN: re-predicate the guard on the verifier queue actually
                 being empty (rather than ``_redispatch`` being empty).
  step-3 RED  -- (a) the late-arrival shape the re-predicated guard still
                 forfeits (an item arriving a moment after the guard runs);
                 (b) the anti-deadlock invariant the original guard's comment
                 claimed to provide, pinned so step-4's deletion cannot
                 silently reintroduce the hang.
  step-4 impl -- GREEN: delete the second clause outright; the anti-deadlock
                 property is provided structurally by the fill loop's other
                 fall-through paths (see merge_queue.py's updated comment).
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any
from unittest.mock import MagicMock

import pytest
from _orch_helpers import MERGE_RESULT_TIMEOUT

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
    ``first_dispatched``: set synchronously inside the stub on the FIRST call.
    ``second_dispatched``: set synchronously inside the stub on the SECOND call.
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
        if len(drive.dispatched) == 1:
            drive.first_dispatched.set()
        elif len(drive.dispatched) == 2:
            drive.second_dispatched.set()
        lease = await allocator.acquire(lambda: MagicMock())
        if lease is None:
            return None
        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=asyncio.ensure_future(drive.gate.wait()),
            merge_wt=None,
            was_speculative=False,
        )

    async def _fake_finalize_inflight(entry: InflightEntry) -> bool:
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
    """Unblock and cancel a ``_verifier_loop`` task driven by :func:`_drive_fill`.

    Shared by every test in this module so a RED failure and a GREEN success
    path both leave no dangling task / pending-task warning behind: release
    every gated verify task, cancel the loop task, and cancel any persistent
    ``_pending_verifier_get`` getter the QueueEmpty race may have launched.
    """
    drive.gate.set()
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
    """DISPATCH-FILL's early-stop guard checks ``_redispatch`` draining
    instead of ``_verifier_queue`` draining -- task 3276.

    RED until step-2 GREEN re-predicates the guard (a partial fix -- see
    step-3's tests for what step-2 alone still forfeits) and step-4 GREEN
    deletes the clause outright (the complete fix).
    """

    async def test_second_host_dispatched_from_verifier_queue_in_same_fill_pass(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """A redispatch-sourced dispatch that drains ``_redispatch`` must NOT
        end the fill pass while ``_verifier_queue`` still holds a ready item
        and a host slot is free.

        RED: only item_a (from ``_redispatch``) is dispatched; the guard's
        ``not is_from_verifier_queue and not self._redispatch`` clause ends
        the fill pass right after, so item_b sits untouched in
        ``_verifier_queue`` while the laptop host slot sits idle -- the loop
        blocks in FINALIZE-HEAD awaiting item_a's gated verify instead.
        GREEN: item_b is dispatched to the free second host in the SAME fill
        pass, immediately after item_a -- both leases held simultaneously.
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
                "dispatched. RED: the DISPATCH-FILL guard's "
                '`not is_from_verifier_queue and not self._redispatch` clause '
                'ends the fill pass as soon as _redispatch drains, even '
                'though _verifier_queue still holds item_b and a host slot '
                'is free -- the loop falls through to FINALIZE-HEAD and '
                "blocks on item_a's gated verify instead. GREEN (step-2/"
                'step-4): the guard is re-predicated/deleted so item_b is '
                'dispatched to the free second host in the same fill pass.'
            )

        assert drive.dispatched == [item_a, item_b], (
            f'expected [item_a, item_b] dispatched in order, got {drive.dispatched!r}'
        )
        assert allocator.free_host_count() == 0, (
            'expected both host slots held simultaneously (local for item_a, '
            f'laptop for item_b); free_host_count() == {allocator.free_host_count()}'
        )

        await _teardown_fill_drive(drive, task, worker)
