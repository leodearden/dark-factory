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
        from orchestrator.merge_speculation_controller import SpeculationController

        controller = SpeculationController(slot=asyncio.Semaphore(depth), depth=depth)

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
        from orchestrator.merge_speculation_controller import SpeculationController

        controller = SpeculationController(slot=asyncio.Semaphore(depth), depth=depth)

        assert controller.take_prefetched() is None
        assert controller.is_idle() is True

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_base_for_returns_actual_main_when_spec_base_is_none(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import SpeculationController

        controller = SpeculationController(slot=asyncio.Semaphore(depth), depth=depth)

        assert controller.base_for('mainsha') == 'mainsha'

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_snapshot_has_exactly_the_expected_keys_and_initial_values(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import SpeculationController

        controller = SpeculationController(slot=asyncio.Semaphore(depth), depth=depth)

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

        A permit is acquired directly on the real semaphore (mirrors what
        acquire_for_lookahead will do once it exists — step-6) and `_held`/
        `pending_spec_base`/`pending_predecessor` are set directly (mirrors
        what on_lookahead_pending will do once it exists — step-6). Seeding
        this way keeps this test's only RED dependency on the
        not-yet-implemented `on_dequeue` itself.
        """
        from orchestrator.merge_speculation_controller import SpeculationController

        slot = asyncio.Semaphore(depth)
        await slot.acquire()  # simulate the permit already held by the merger
        controller = SpeculationController(slot=slot, depth=depth)
        controller._held = True
        pred = _make_pending_request('pred')
        controller.pending_spec_base = 'PRED_SHA'
        controller.pending_predecessor = pred
        return controller, slot, pred

    async def test_attach_when_all_four_conditions_hold(self) -> None:
        controller, slot, _pred = await self._seeded_controller()
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result == 'PRED_SHA'
        assert controller.spec_base == 'PRED_SHA'
        assert controller.held_by_merger == 1
        assert slot._value == 0  # unchanged — permit retained/transferred
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None

    async def test_fallback_when_pending_spec_base_is_none(self) -> None:
        controller, slot, _pred = await self._seeded_controller()
        controller.pending_spec_base = None  # break condition (a)
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result is None
        assert controller.spec_base is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert slot._value == 1  # released

    async def test_fallback_when_not_held(self) -> None:
        controller, slot, _pred = await self._seeded_controller()
        controller._held = False  # break condition (b)
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result is None
        assert controller.spec_base is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert slot._value == 0  # not held -> no release (avoid over-release)

    async def test_fallback_when_pending_predecessor_is_none(self) -> None:
        controller, slot, _pred = await self._seeded_controller()
        controller.pending_predecessor = None  # break condition (c)
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result is None
        assert controller.spec_base is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert slot._value == 1  # released

    async def test_fallback_when_predecessor_already_done(self) -> None:
        controller, slot, pred = await self._seeded_controller()
        pred.result.set_result(MergeOutcome('done'))  # break condition (d)
        new_req = _make_pending_request('new')

        result = controller.on_dequeue(new_req)

        assert result is None
        assert controller.spec_base is None
        assert controller.pending_spec_base is None
        assert controller.pending_predecessor is None
        assert controller.held_by_merger == 0
        assert slot._value == 1  # released
