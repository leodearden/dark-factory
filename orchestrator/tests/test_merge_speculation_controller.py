"""Unit tests for orchestrator.merge_speculation_controller (task MQ-refactor
theta / 1993).

Steps covered:
  step-1  RED   — SpeculationController initial state + is_idle/take_prefetched/
                  base_for/snapshot
  step-2  GREEN — implement SpeculationController skeleton
  step-3  RED   — on_dequeue ATTACH/FALLBACK four-condition rule
  step-4  GREEN — implement on_dequeue
  step-5  RED   — look-ahead + transfer lifecycle
  step-6  GREEN — implement acquire_for_lookahead/on_lookahead_found/
                  on_lookahead_pending/on_transfer
  step-7  RED   — release paths + double-release tolerance
  step-8  GREEN — implement on_abort/on_shutdown

This module intentionally imports orchestrator.merge_speculation_controller
LOCALLY inside each test (mirrors test_merge_request_ledger.py's convention)
so a not-yet-implemented symbol never breaks collection of the rest of the
file during the RED steps. It imports only orchestrator.merge_types (NOT
orchestrator.merge_queue) at module scope — SpeculationController has no
git/GitOps dependency at all (pure data-structure unit tests), same rationale
as test_merge_request_ledger.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_types import MergeOutcome, MergeRequest

# ---------------------------------------------------------------------------
# Helpers (per-file duplication convention — see test_merge_request_ledger.py)
# ---------------------------------------------------------------------------


def _make_pending_request(task_id: str = 'pred') -> MergeRequest:
    """Build a bare MergeRequest with a fresh, still-pending result Future.

    Mirrors test_merge_request_ledger.py's _make_request helper —
    worktree/config/module_configs are irrelevant to SpeculationController,
    which only reads task_id/result off a predecessor request.
    """
    return MergeRequest(
        task_id=task_id,
        branch=f'task/{task_id}',
        worktree=Path('/tmp/unused'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=OrchestratorConfig(),
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: initial state + is_idle/take_prefetched/base_for/
# snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculationControllerInitialState:
    """SpeculationController initial state (task 1993 step-1).

    RED until step-2 GREEN adds orchestrator.merge_speculation_controller.
    """

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_initial_state_is_idle_and_empty(self, depth: int) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        controller = SpeculationController(
            ledger=PermitLedger(asyncio.Semaphore(depth), depth), depth=depth,
        )

        assert controller.spec_base is None
        assert controller.prefetched is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert controller.is_idle() is True

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_take_prefetched_returns_none_when_empty_and_stays_idle(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        controller = SpeculationController(
            ledger=PermitLedger(asyncio.Semaphore(depth), depth), depth=depth,
        )

        assert controller.take_prefetched() is None
        assert controller.is_idle() is True

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_base_for_returns_actual_main_when_spec_base_is_none(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        controller = SpeculationController(
            ledger=PermitLedger(asyncio.Semaphore(depth), depth), depth=depth,
        )

        assert controller.base_for('mainsha') == 'mainsha'

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_snapshot_has_exactly_the_expected_keys_and_initial_values(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        controller = SpeculationController(
            ledger=PermitLedger(asyncio.Semaphore(depth), depth), depth=depth,
        )

        snap = controller.snapshot()

        assert set(snap) == {
            'depth',
            'held_by_merger',
            'spec_base',
            'prefetched_task_id',
            'pending_spec_base',
            'pending_predecessor_task_id',
            'slot_available',
        }
        assert snap['depth'] == depth
        assert snap['held_by_merger'] == 0
        assert snap['spec_base'] is None
        assert snap['prefetched_task_id'] is None
        assert snap['pending_spec_base'] is None
        assert snap['pending_predecessor_task_id'] is None
        assert snap['slot_available'] == depth


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: on_dequeue ATTACH/FALLBACK four-condition rule
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOnDequeueAttachFallback:
    """SpeculationController.on_dequeue ATTACH/FALLBACK rule (task 1993 step-3).

    RED until step-4 GREEN implements on_dequeue.
    """

    async def _seeded_controller(self, depth: int = 1):
        """Build a controller seeded into the late-arrival 'retain' state.

        A permit is acquired directly through the ledger (mirrors what
        acquire_for_lookahead does) and `_permit`/`pending_spec_base`/
        `pending_predecessor` are set directly (mirrors what
        on_lookahead_pending does). Seeding this way keeps this test's only
        RED dependency on the not-yet-implemented `on_dequeue` itself.
        """
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        permit = await ledger.acquire()  # simulate the permit already held by the merger
        controller = SpeculationController(ledger=ledger, depth=depth)
        controller._permit = permit
        pred = _make_pending_request('pred')
        controller.pending_spec_base = 'PRED_SHA'
        controller.pending_predecessor = pred
        return controller, ledger, pred

    async def test_attach_when_all_four_conditions_hold(self) -> None:
        controller, ledger, _pred = await self._seeded_controller()
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result == 'PRED_SHA'
        assert controller.spec_base == 'PRED_SHA'
        assert controller.held_by_merger == 1
        assert ledger.slot_available == 0  # unchanged — permit retained/transferred
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None

    async def test_fallback_when_pending_spec_base_is_none(self) -> None:
        controller, ledger, _pred = await self._seeded_controller()
        controller.pending_spec_base = None  # break condition (a)
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result is None
        assert controller.spec_base is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert ledger.slot_available == 1  # released

    async def test_fallback_when_not_held(self) -> None:
        controller, ledger, _pred = await self._seeded_controller()
        controller._permit = None  # break condition (b)
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result is None
        assert controller.spec_base is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert ledger.slot_available == 0  # not held -> no release (avoid over-release)

    async def test_fallback_when_pending_predecessor_is_none(self) -> None:
        controller, ledger, _pred = await self._seeded_controller()
        controller.pending_predecessor = None  # break condition (c)
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result is None
        assert controller.spec_base is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert ledger.slot_available == 1  # released

    async def test_fallback_when_predecessor_already_done(self) -> None:
        controller, ledger, pred = await self._seeded_controller()
        pred.result.set_result(MergeOutcome('done'))  # break condition (d)
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result is None
        assert controller.spec_base is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert ledger.slot_available == 1  # released


# ---------------------------------------------------------------------------
# step-5 RED / step-6 GREEN: look-ahead + transfer lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLookaheadAndTransferLifecycle:
    """acquire_for_lookahead/on_lookahead_found/on_lookahead_pending/on_transfer
    (task 1993 step-5).

    RED until step-6 GREEN implements these methods. Uses depth=2 throughout
    so a post-acquire slot value of K-1==1 is distinguishable from a fully
    exhausted 0 — sharpening the "decrements by exactly one" assertions.
    """

    async def test_acquire_for_lookahead_decrements_slot_and_sets_held(self) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)

        await controller.acquire_for_lookahead()

        assert ledger.slot_available == depth - 1
        assert controller.held_by_merger == 1

    async def test_on_lookahead_found_sets_prefetched_and_spec_base_without_releasing(
        self,
    ) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')

        controller.on_lookahead_found(next_req, 'MERGE_SHA')

        assert controller.prefetched is next_req
        assert controller.spec_base == 'MERGE_SHA'
        assert controller.held_by_merger == 1
        assert ledger.slot_available == depth - 1  # unchanged — no release on found
        assert controller.is_idle() is False

    async def test_on_lookahead_pending_retains_permit_for_late_arrival(self) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        await controller.acquire_for_lookahead()
        pred = _make_pending_request('pred')

        controller.on_lookahead_pending('MERGE_SHA', pred)

        assert controller.pending_spec_base == 'MERGE_SHA'
        assert controller.pending_predecessor is pred
        assert controller.held_by_merger == 1
        assert ledger.slot_available == depth - 1  # unchanged — retained, not released
        assert controller.is_idle() is False

    async def test_on_transfer_clears_held_without_releasing_slot(self) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')

        controller.on_transfer()

        assert controller.held_by_merger == 0
        # UNCHANGED by on_transfer — the verifier now owns this permit and
        # will release it itself on drain (the zeta chokepoint).
        assert ledger.slot_available == depth - 1

    async def test_on_transfer_terminal_clears_held_and_spec_base_without_releasing_slot(
        self,
    ) -> None:
        """Amendment regression test (post-step-12 review): on_transfer_terminal()
        is the early-continue counterpart of on_transfer() — it ALSO clears
        spec_base (unlike plain on_transfer(), which leaves spec_base for the
        immediately-following look-ahead to re-derive). Used at the seven
        early-continue sites (already_merged/conflict/merge-fail/abandoned/
        revparse-fail/drop/branch-presence-guard) where the loop `continue`s
        with no subsequent look-ahead call.
        """
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        # Seed a non-None spec_base as ATTACH would (on_dequeue), simulating
        # a speculative request that is about to resolve via an
        # early-continue outcome (e.g. conflict) rather than a real merge.
        await controller.acquire_for_lookahead()
        controller.spec_base = 'PRED_SHA'

        controller.on_transfer_terminal()

        assert controller.held_by_merger == 0
        assert controller.spec_base is None
        assert controller.is_idle() is True
        # UNCHANGED by on_transfer_terminal() — the verifier now owns this
        # permit and will release it itself on drain (the zeta chokepoint).
        assert ledger.slot_available == depth - 1

    async def test_on_transfer_terminal_is_idle_immediately_unlike_on_transfer(
        self,
    ) -> None:
        """Pins the exact divergence the amendment fixes: after an
        early-continue transfer, is_idle() must be True immediately (no
        look-ahead ever runs to clean up); after the main-success transfer,
        spec_base is deliberately left for the look-ahead to handle.
        """
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2

        terminal_controller = SpeculationController(
            ledger=PermitLedger(asyncio.Semaphore(depth), depth), depth=depth,
        )
        await terminal_controller.acquire_for_lookahead()
        terminal_controller.spec_base = 'PRED_SHA'
        terminal_controller.on_transfer_terminal()
        assert terminal_controller.is_idle() is True

        main_success_controller = SpeculationController(
            ledger=PermitLedger(asyncio.Semaphore(depth), depth), depth=depth,
        )
        await main_success_controller.acquire_for_lookahead()
        main_success_controller.spec_base = 'PRED_SHA'
        main_success_controller.on_transfer()
        # Not idle yet — spec_base is still the pre-transfer value until the
        # look-ahead call (on_lookahead_found/on_lookahead_pending/
        # on_shutdown) that always immediately follows on_transfer() in
        # _merger_loop runs.
        assert main_success_controller.is_idle() is False

    async def test_acquire_found_transfer_sequence_conserves_permit(self) -> None:
        """Net effect of acquire -> found -> transfer: exactly one permit is
        handed off to the verifier. The slot stays at depth-1 throughout —
        it is never released by the merger side; only the (unexercised here)
        verifier drain would restore it to depth.
        """
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)

        await controller.acquire_for_lookahead()
        assert ledger.slot_available == depth - 1
        assert controller.held_by_merger == 1

        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')
        assert ledger.slot_available == depth - 1
        assert controller.held_by_merger == 1

        controller.on_transfer()
        assert ledger.slot_available == depth - 1  # now held by the verifier, not merger
        assert controller.held_by_merger == 0


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: release paths + double-release tolerance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReleasePathsAndDoubleReleaseTolerance:
    """on_abort/on_shutdown release paths (task 1993 step-7).

    RED until step-8 GREEN implements these methods.
    """

    async def test_on_abort_releases_when_held(self) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')

        controller.on_abort()

        assert ledger.slot_available == depth  # released back
        assert controller.spec_base is None
        assert controller.held_by_merger == 0

    async def test_on_abort_is_noop_when_not_held(self) -> None:
        """When not held, on_abort must never call release() (over-release
        guard) even if spec_base was left set by some other path — it still
        clears spec_base as a pure state-cleanup, just without touching the
        semaphore.
        """
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        controller.spec_base = 'STALE_SHA'  # simulate leftover state
        controller._permit = None

        controller.on_abort()

        assert ledger.slot_available == depth  # unchanged — never held, never released
        assert controller.spec_base is None
        assert controller.held_by_merger == 0

    async def test_on_shutdown_releases_and_clears_all_fields_when_held(self) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')
        # Seed pending_* too, so shutdown's "clears ALL five fields" claim is
        # exercised beyond what on_lookahead_found alone touches.
        pred = _make_pending_request('pred')
        controller.pending_spec_base = 'PRED_SHA'
        controller.pending_predecessor = pred

        controller.on_shutdown()

        assert ledger.slot_available == depth  # released back
        assert controller.spec_base is None
        assert controller.prefetched is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert controller.is_idle() is True

    async def test_on_shutdown_is_idempotent_when_already_idle(self) -> None:
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 2
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)

        controller.on_shutdown()
        controller.on_shutdown()  # second call — must not double-release

        assert ledger.slot_available == depth  # unchanged, no phantom release
        assert controller.is_idle() is True
        assert controller.held_by_merger == 0

    async def test_double_release_tolerance_survives_phantom_permit_after_transfer(
        self,
    ) -> None:
        """Pins the plain-Semaphore (never BoundedSemaphore) contract — now
        exercised at the ledger level.

        A pathological CancelledError-after-put race can leave the merger's
        outer finally believing it still owns a permit that has ALREADY
        transferred to (and been released by) the verifier. The release
        below simulates that verifier release via the RAW semaphore —
        bypassing the ledger's own ``live`` bookkeeping entirely, mirroring
        production (the verifier-side releases are not migrated to
        ``ledger.release()`` until task eta). Because ``live`` still
        (wrongly) considers the token outstanding, a subsequent phantom
        ``ledger.release()`` of that SAME token from ``on_shutdown()`` goes
        through as an ordinary release rather than raising — and MAY push
        the semaphore's internal counter above ``depth``. This over-release
        is documented-tolerated, not a bug in this controller (see module
        docstring's "Plain-Semaphore double-release tolerance").
        """
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        depth = 1
        slot = asyncio.Semaphore(depth)
        ledger = PermitLedger(slot, depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')
        phantom_permit = controller._permit  # captured before on_transfer clears it
        controller.on_transfer()  # permit -> None; slot stays at 0 (verifier's now)

        # Simulate the verifier's own drain release (the real release owed
        # for this transferred permit) racing independently of the
        # controller, via the RAW semaphore — mirrors production, where the
        # verifier-side releases are not yet migrated to ledger.release()
        # (task eta):
        slot.release()
        assert ledger.slot_available == depth

        # Simulate the pathological race: a phantom permit reference left on
        # the controller (e.g. the merger's outer finally not yet aware of
        # the transfer).
        controller._permit = phantom_permit
        controller.on_shutdown()  # must NOT raise

        assert ledger.slot_available == depth + 1  # over depth — tolerated


# ---------------------------------------------------------------------------
# task 2159 step-5 RED / step-6 GREEN: PermitLedger-routed construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculationControllerLedgerRouting:
    """SpeculationController constructed with a PermitLedger, not a raw
    Semaphore (task 2159 step-5, DD5).

    RED until step-6 GREEN refactors SpeculationController to take
    ``ledger=`` instead of ``slot=`` and routes acquire/transfer/release
    through it. Unlike the legacy ``slot=``-constructed tests above (task
    1993, unchanged by this task), these assert against
    ``ledger.live``/``ledger.slot_available`` rather than a raw
    ``asyncio.Semaphore``'s ``_value``.
    """

    def _make_ledger_and_controller(self, depth: int):
        from orchestrator.merge_speculation_controller import (
            PermitLedger,
            SpeculationController,
        )

        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        controller = SpeculationController(ledger=ledger, depth=depth)
        return ledger, controller

    async def test_acquire_for_lookahead_registers_a_live_token_and_sets_held(
        self,
    ) -> None:
        depth = 2
        ledger, controller = self._make_ledger_and_controller(depth)

        await controller.acquire_for_lookahead()

        assert controller.held_by_merger == 1
        assert len(ledger.live) == 1
        assert ledger.slot_available == depth - 1

    async def test_on_transfer_clears_held_but_token_remains_live(self) -> None:
        """The verifier now owns the in-flight permit — on_transfer must NOT
        release it through the ledger (task eta migrates the verifier's own
        raw release to ``ledger.release(item.permit)``); the token stays in
        ``live`` until then.
        """
        depth = 2
        ledger, controller = self._make_ledger_and_controller(depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')

        controller.on_transfer()

        assert controller.held_by_merger == 0
        assert len(ledger.live) == 1
        assert ledger.slot_available == depth - 1

    async def test_on_transfer_terminal_clears_held_and_spec_base_token_remains_live(
        self,
    ) -> None:
        depth = 2
        ledger, controller = self._make_ledger_and_controller(depth)
        await controller.acquire_for_lookahead()
        controller.spec_base = 'PRED_SHA'

        controller.on_transfer_terminal()

        assert controller.held_by_merger == 0
        assert controller.spec_base is None
        assert len(ledger.live) == 1
        assert ledger.slot_available == depth - 1

    async def test_on_transfer_returns_the_detached_permit_token(self) -> None:
        """task eta step-1: on_transfer() must RETURN the detached SpecPermit
        token (not None) so the merger can stamp it onto the enqueued item's
        `.permit` — the verifier later releases this SAME token via
        `ledger.release(item.permit)`. The token stays registered in
        `ledger.live` (only the controller's own reference is cleared).
        """
        depth = 2
        ledger, controller = self._make_ledger_and_controller(depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')
        expected = controller._permit

        returned = controller.on_transfer()

        assert returned is expected
        assert returned in ledger.live
        assert controller.held_by_merger == 0

    async def test_on_transfer_terminal_returns_the_detached_permit_token(self) -> None:
        """Same return-the-token contract as on_transfer(), plus spec_base
        clears (terminal early-continue site — no subsequent look-ahead call
        ever re-derives it for this permit).
        """
        depth = 2
        ledger, controller = self._make_ledger_and_controller(depth)
        await controller.acquire_for_lookahead()
        controller.spec_base = 'PRED_SHA'
        expected = controller._permit

        returned = controller.on_transfer_terminal()

        assert returned is expected
        assert returned in ledger.live
        assert controller.held_by_merger == 0
        assert controller.spec_base is None

    async def test_on_abort_releases_through_the_ledger(self) -> None:
        depth = 2
        ledger, controller = self._make_ledger_and_controller(depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')

        controller.on_abort()

        assert controller.held_by_merger == 0
        assert ledger.live == frozenset()
        assert ledger.slot_available == depth

    async def test_on_shutdown_releases_through_the_ledger(self) -> None:
        depth = 2
        ledger, controller = self._make_ledger_and_controller(depth)
        await controller.acquire_for_lookahead()
        next_req = _make_pending_request('next')
        controller.on_lookahead_found(next_req, 'MERGE_SHA')

        controller.on_shutdown()

        assert controller.held_by_merger == 0
        assert ledger.live == frozenset()
        assert ledger.slot_available == depth

    async def test_on_dequeue_fallback_releases_through_the_ledger(self) -> None:
        depth = 1
        ledger, controller = self._make_ledger_and_controller(depth)
        await controller.acquire_for_lookahead()
        # pending_spec_base defaults to None -> ATTACH condition (a) fails ->
        # FALLBACK releases the held permit.
        new_req = _make_pending_request('new')

        controller.on_dequeue(new_req)

        assert controller.held_by_merger == 0
        assert ledger.live == frozenset()
        assert ledger.slot_available == depth

    async def test_snapshot_slot_available_matches_ledger(self) -> None:
        depth = 2
        ledger, controller = self._make_ledger_and_controller(depth)

        await controller.acquire_for_lookahead()

        assert controller.snapshot()['slot_available'] == ledger.slot_available
