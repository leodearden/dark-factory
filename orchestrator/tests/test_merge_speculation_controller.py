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

import pytest

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
