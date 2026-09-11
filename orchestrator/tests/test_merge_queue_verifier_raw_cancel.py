"""Cancellation/shutdown semantics of ``_verifier_loop``'s persistent getter — task 4306.

Fences the FINALIZE-HEAD "Reuse the persistent getter" branch
(``_verifier_loop``, ``orchestrator/src/orchestrator/merge_queue.py``) against
its recovery clause absorbing a cancellation aimed at the loop task itself:

* A bare ``task.cancel()`` -- the exact teardown primitive
  ``SpeculativeMergeWorker.run()`` applies to ``_verifier_task`` on both its
  ``except BaseException`` and ``finally`` arms -- must terminate a
  ``_verifier_loop`` parked on a reused persistent getter, and the
  ``CancelledError`` must actually propagate (``task.cancelled()`` is True)
  rather than be converted into a normal return.
* A getter-ONLY cancel (``stop()``'s ordering-race shape, where the loop task
  itself is untouched) must still be recovered from, losing no queue item --
  the original intent of that recovery clause.

Deliberately separate from ``test_merge_queue_dispatch_fill_redispatch.py``,
whose own docstring disclaims cancellation semantics ("These tests fence the
DISPATCH-FILL predicate"); this module consumes that module's harness by
import only.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from _orch_helpers import MERGE_RESULT_TIMEOUT

# Reuse the γ multi-host test harness (established cross-test-module import
# pattern -- orchestrator/tests/ has no __init__.py; precedent:
# test_merge_queue_dispatch_fill_redispatch.py:34-46) -- single source of
# truth for the fixtures/fakes, zero churn to the landed γ test file.
from test_merge_queue_concurrent_verify import (
    _inject_two_host_allocator,
    _make_fake_remote,
    config,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_config,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_ops,  # noqa: F401 — pytest fixture re-exported from γ harness
    git_repo,  # noqa: F401 — pytest fixture re-exported from γ harness
)
from test_merge_queue_dispatch_fill_redispatch import (
    _drive_fill,
    _make_real_item,
    _teardown_fill_drive,
)

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_queue import ItemLifecycleState, SpeculativeMergeWorker

# The acceptance criterion for the raw-cancel path is "completes within a SHORT
# bound", and a hang must surface as a readable diagnostic well inside the 60s
# per-test cap -- pyproject.toml sets `timeout_method = "thread"`, which
# os._exit()s the whole xdist worker when that cap fires. Deliberately much
# tighter than MERGE_RESULT_TIMEOUT (used only for "wait for a real event").
_CANCEL_TERMINATION_TIMEOUT = 5.0

# Bound on the poll loops that wait for a purely in-process loop state
# transition (getter launched / loop parked on the reused getter). These need
# only a few event-loop turns; a generous ceiling keeps a slow CI box from
# flaking without ever approaching the per-test cap.
_STATE_POLL_TIMEOUT = 10.0
_STATE_POLL_INTERVAL = 0.01


async def _poll_until(predicate, timeout: float = _STATE_POLL_TIMEOUT) -> bool:
    """Poll *predicate* every ``_STATE_POLL_INTERVAL`` up to *timeout* seconds.

    Returns True as soon as it holds, False if the bound expires. Bounded by
    construction so a RED run reports a diagnostic instead of hanging until
    the suite's thread-method timeout ``os._exit()``s the worker.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_STATE_POLL_INTERVAL)


async def _hang_safe_teardown(
    drive,
    task: asyncio.Task,  # type: ignore[type-arg]
    worker: SpeculativeMergeWorker,
) -> None:
    """Force a possibly-hung ``_verifier_loop`` task down, for the RED path.

    Deliberately NOT ``_teardown_fill_drive``: on the RED path the loop has
    already absorbed one cancellation, so a single cancel+gather would itself
    block forever. Release every gated verify, run ``stop()``'s complete
    protocol, then cancel repeatedly -- a SECOND cancel does terminate the
    loop, because it interrupts the fresh ``get()`` the swallowed cancel
    re-parked on. Finally clear any surviving ``_pending_verifier_get`` (idiom
    copied from ``_teardown_fill_drive``) so no dangling task is left for
    ``timeout_method = "thread"`` to ``os._exit()`` on.
    """
    drive.gate.set()
    # Exception, not BaseException: never swallow a CancelledError aimed at the
    # enclosing test task -- only absorb worker.stop()'s own failures.
    with contextlib.suppress(Exception):
        await worker.stop()
    for _ in range(5):
        if task.done():
            break
        task.cancel()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(task, return_exceptions=True)),
                timeout=2.0,
            )
    if worker._pending_verifier_get is not None:
        pending = worker._pending_verifier_get
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# step-1: primary repro -- a bare task.cancel() must terminate the loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRawCancelTerminatesParkedVerifierLoop:
    """A bare ``task.cancel()`` must terminate a ``_verifier_loop`` parked on a
    reused persistent getter -- task 4306.

    This is the exact teardown primitive ``SpeculativeMergeWorker.run()``
    applies to ``_verifier_task`` (raw ``cancel()`` + ``gather()``, on both its
    ``except BaseException`` and ``finally`` arms), so the state pinned here is
    production-reachable, not merely test-reachable.
    """

    async def test_bare_cancel_of_loop_reusing_persistent_getter_terminates(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """Park the loop on the FINALIZE-HEAD getter-reuse branch, then cancel it.

        Steady state built here (the esc-1735-5 multi-host fill-ahead shape,
        normal on a 2-host configuration):

          1. one queued item is dispatched; a host stays free, so the
             DISPATCH-FILL QueueEmpty branch launches the PERSISTENT
             ``_pending_verifier_get`` and parks in the fill-ahead race;
          2. releasing the gate hands that race to the verify, so FINALIZE-HEAD
             finalizes the head and the getter survives;
          3. the next iteration finds ``_redispatch`` empty, the pending getter
             not done and ``_inflight`` now empty -- so it falls through to
             FINALIZE-HEAD's ``else:`` and awaits the REUSED getter.

        Cancelling the loop task then delivers ``CancelledError`` through that
        transitively-cancelled getter.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        allocator = _inject_two_host_allocator(worker, _make_fake_remote('laptop'))
        assert allocator.free_host_count() == 2, 'precondition: two free hosts'

        drive = _drive_fill(worker, allocator)

        item = _make_real_item(git_ops, config, 'rawcancel-a', 'aaa')
        # Pre-register at the item's TRUE parked state so _note_transition's
        # AWAITING_VERIFY -> DISPATCHING hop finds a registered request_id and
        # files no spurious L1 illegal_lifecycle_transition escalation (task
        # 2169 registry).
        worker._register_item(item, initial=ItemLifecycleState.AWAITING_VERIFY)
        # QUEUE-sourced, NOT _redispatch: the getter-reuse branch only ever
        # harvests from _verifier_queue.
        await worker._verifier_queue.put(item)

        task = asyncio.ensure_future(worker._verifier_loop())
        # _assert_single_writer is live (ORCH_DEBUG_ASSERTS=1, conftest.py)
        # and self._running is True from construction -- the loop's first
        # _inflight_append would otherwise raise on the "wrong" owner task.
        worker._verifier_task = task

        try:
            await asyncio.wait_for(drive.first_dispatched.wait(), timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            await _hang_safe_teardown(drive, task, worker)
            pytest.fail(
                'the queued item was never dispatched within '
                f'{MERGE_RESULT_TIMEOUT}s -- the parked-persistent-getter '
                'steady state this test is about was never reached.'
            )

        launched = await _poll_until(lambda: worker._pending_verifier_get is not None)
        if not launched:
            await _hang_safe_teardown(drive, task, worker)
            pytest.fail(
                '_pending_verifier_get stayed None after the first dispatch: '
                'the DISPATCH-FILL QueueEmpty fill-ahead race never launched '
                'the persistent getter, so the FINALIZE-HEAD getter-reuse '
                'branch under test is unreachable in this run.'
            )

        # Hand the fill-ahead race to the verify: it completes, the getter
        # survives, and FINALIZE-HEAD finalizes the head.
        drive.gate.set()
        try:
            await asyncio.wait_for(item.request.result, timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            await _hang_safe_teardown(drive, task, worker)
            pytest.fail(
                "the head item's result Future was never resolved within "
                f'{MERGE_RESULT_TIMEOUT}s of its gated verify completing -- '
                'FINALIZE-HEAD was never reached.'
            )

        parked = await _poll_until(
            lambda: worker._pending_verifier_get is None and not task.done()
        )
        if not parked:
            await _hang_safe_teardown(drive, task, worker)
            pytest.fail(
                'the loop never reached the parked-on-reused-getter signature '
                '(_pending_verifier_get is None AND the loop task is still '
                'running): _pending_verifier_get='
                f'{worker._pending_verifier_get!r}, task.done()={task.done()}. '
                'The FINALIZE-HEAD "Reuse the persistent getter" branch reads '
                'and nulls that attribute with no await in between, so this '
                'signature is what "parked on the reused getter" looks like.'
            )

        # run()'s exact teardown shape: raw cancel() + gather(). asyncio.shield
        # keeps wait_for's timeout from re-cancelling the gather -- a second
        # cancel WOULD terminate the loop (it interrupts the fresh get()), and
        # would therefore mask the very hang under test.
        task.cancel()
        timed_out = False
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(task, return_exceptions=True)),
                timeout=_CANCEL_TERMINATION_TIMEOUT,
            )
        except TimeoutError:
            timed_out = True

        if timed_out:
            await _hang_safe_teardown(drive, task, worker)
            pytest.fail(
                'a bare task.cancel() did not terminate _verifier_loop within '
                f'{_CANCEL_TERMINATION_TIMEOUT}s while it was parked on the '
                'reused persistent getter. The recovery clause at '
                "merge_queue.py's \"Getter was cancelled (stop() ordering "
                'race)" comment absorbed the loop task\'s OWN cancellation -- '
                'it arrived through the transitively-cancelled getter, was '
                'caught, and the clause re-parked the loop on a fresh, '
                'uncancelled _verifier_queue.get(). So run()\'s cancel() + '
                'gather() shutdown never returns and the merge worker hangs '
                'on teardown.'
            )

        assert task.cancelled(), (
            'the loop task terminated but was not CANCELLED: the '
            'CancelledError was converted into a normal return instead of '
            'propagating. run()\'s gather() depends on the cancellation '
            'actually propagating, and a task that swallows a pending cancel '
            'request must not be reported as a clean completion. '
            f'task.done()={task.done()}, exception='
            f'{task.exception() if task.done() and not task.cancelled() else None!r}'
        )

        # The gate was never released for a second dispatch and stop() was
        # never run, so clear any residue the loop left behind.
        if worker._pending_verifier_get is not None:
            pending = worker._pending_verifier_get
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# step-3: getter-only-cancel preservation fence (NOT a RED/GREEN pair)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetterOnlyCancelPreservesQueueItems:
    """A cancel aimed at ``_pending_verifier_get`` ALONE must still be recovered
    from, losing no queue item -- task 4306.

    **NOT a RED/GREEN pair.** This test is green both before and after the
    task-4306 fix, by design (same shape as the task-3276 module's
    ``TestCascadeAntiDeadlockPreserved``). It is a permanent preservation
    fence: it pins the "no queue item is lost" intent the recovery clause was
    originally written for, and is what proves the new
    ``current_task().cancelling() > 0`` discriminator does not OVER-trigger and
    turn a getter-only cancel into loop death. A reviewer seeing it pass on
    unfixed code is seeing the intended behaviour, not a broken RED.
    """

    async def test_getter_only_cancel_recovers_and_loses_no_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """Cancel ONLY the captured getter; the loop must survive and keep working.

        Same parked-on-reused-getter steady state as the raw-cancel test, but
        the getter is captured into a local BEFORE the loop nulls the
        attribute -- once the loop has parked on it, that local reference is
        the only way a test can reach it. Cancelling it is exactly the shape
        stop()'s ordering race would produce: the loop task itself is
        untouched, so its ``cancelling()`` stays 0 and the recovery clause must
        still run.

        A later-queued item_b must then still be dispatched: the recovery
        ``get()`` picks work up and nothing is dropped.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        allocator = _inject_two_host_allocator(worker, _make_fake_remote('laptop'))
        assert allocator.free_host_count() == 2, 'precondition: two free hosts'

        drive = _drive_fill(worker, allocator)

        item_a = _make_real_item(git_ops, config, 'goc-a', 'aaa')
        item_b = _make_real_item(git_ops, config, 'goc-b', 'bbb')
        worker._register_item(item_a, initial=ItemLifecycleState.AWAITING_VERIFY)
        worker._register_item(item_b, initial=ItemLifecycleState.AWAITING_VERIFY)

        # item_a is queued up front; item_b is held back until after the
        # getter-only cancel, so its dispatch proves the recovery get() works.
        await worker._verifier_queue.put(item_a)

        task = asyncio.ensure_future(worker._verifier_loop())
        worker._verifier_task = task

        try:
            await asyncio.wait_for(drive.first_dispatched.wait(), timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                'item_a was never dispatched within '
                f'{MERGE_RESULT_TIMEOUT}s -- the parked-persistent-getter '
                'steady state this test is about was never reached.'
            )

        launched = await _poll_until(lambda: worker._pending_verifier_get is not None)
        if not launched:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                '_pending_verifier_get stayed None after the first dispatch: '
                'the DISPATCH-FILL QueueEmpty fill-ahead race never launched '
                'the persistent getter, so there is no getter to cancel.'
            )

        # CAPTURE the getter now, BEFORE the FINALIZE-HEAD reuse branch reads
        # and nulls the attribute -- afterwards it is unreachable from a test.
        getter = worker._pending_verifier_get
        assert getter is not None, 'the persistent getter must be capturable here'

        drive.gate.set()
        try:
            await asyncio.wait_for(item_a.request.result, timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                "item_a's result Future was never resolved within "
                f'{MERGE_RESULT_TIMEOUT}s of its gated verify completing -- '
                'FINALIZE-HEAD was never reached.'
            )

        parked = await _poll_until(
            lambda: worker._pending_verifier_get is None and not task.done()
        )
        if not parked:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                'the loop never reached the parked-on-reused-getter signature '
                '(_pending_verifier_get is None AND the loop task is still '
                'running): _pending_verifier_get='
                f'{worker._pending_verifier_get!r}, task.done()={task.done()}.'
            )

        # Cancel ONLY the inner getter. The loop task is untouched, so its
        # cancelling() count stays 0 and the recovery clause must run.
        getter.cancel()
        await asyncio.sleep(_STATE_POLL_INTERVAL * 5)

        assert not task.done(), (
            'the loop died on a getter-only cancel: the task-4306 '
            'discriminator over-triggered and re-raised a cancellation that '
            'was never aimed at the loop task. Only _pending_verifier_get was '
            'cancelled here -- exactly the ordering race the recovery clause '
            'exists for -- so the loop must recover with a fresh get(), not '
            f'terminate. task.cancelled()={task.cancelled()}'
        )

        await worker._verifier_queue.put(item_b)
        try:
            await asyncio.wait_for(drive.second_dispatched.wait(), timeout=MERGE_RESULT_TIMEOUT)
        except TimeoutError:
            await _teardown_fill_drive(drive, task, worker)
            pytest.fail(
                'item_b was never dispatched within '
                f'{MERGE_RESULT_TIMEOUT}s of being queued after a getter-only '
                'cancel. The recovery clause left the loop unable to pick up '
                'new work -- a queue item is effectively lost, which is the '
                'exact property that clause exists to preserve.'
            )

        assert drive.dispatched == [item_a, item_b], (
            'expected [item_a, item_b] dispatched in order (nothing dropped '
            f'across the getter-only cancel), got {drive.dispatched!r}'
        )

        await _teardown_fill_drive(drive, task, worker)
