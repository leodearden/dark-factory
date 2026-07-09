"""Unit tests for orchestrator.merge_types.ItemLifecycleState and
orchestrator.merge_queue.ItemLifecycle (merge-queue-reliability PRD scope-4
iota / task 2164, L-1).

Steps covered:
  step-1 RED   — ItemLifecycleState enum contract
  step-2 GREEN — add ItemLifecycleState (merge_types.py) + re-export shim
  step-3 RED   — ItemLifecycle registry storage (register/current)
  step-4 GREEN — implement the ItemLifecycle registry class
  step-5 RED   — transition() happy path + single-source
  step-6 GREEN — add _LEGAL_TRANSITIONS + implement transition()
  step-7 RED   — illegal transitions raise IllegalLifecycleTransition
  step-8 GREEN — define IllegalLifecycleTransition + harden transition()

This module imports orchestrator.merge_queue.ItemLifecycle (and friends)
LOCALLY inside each test that needs them — those symbols do not exist until
their GREEN step lands — so a not-yet-implemented symbol never breaks
collection of the rest of the file during the RED steps (mirrors
test_permit_ledger.py's convention).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: ItemLifecycleState enum contract
# ---------------------------------------------------------------------------


class TestItemLifecycleStateEnum:
    """ItemLifecycleState is a str-compatible Enum with exactly 10 members
    (task 2164 step-1).

    RED until step-2 GREEN adds ItemLifecycleState to merge_types.py (+ the
    merge_queue.py re-export shim).
    """

    _MEMBER_NAMES = frozenset({
        'QUEUED',
        'LANE_BUFFERED',
        'MERGING',
        'AWAITING_VERIFY',
        'REDISPATCH_PARKED',
        'DISPATCHING',
        'VERIFYING',
        'GATE_REVERIFY',
        'FINALIZING',
        'TERMINAL',
    })

    def test_is_a_strenum(self) -> None:
        from enum import StrEnum

        from orchestrator.merge_types import ItemLifecycleState

        assert issubclass(ItemLifecycleState, StrEnum)

    def test_members_are_exactly_the_ten_lifecycle_states(self) -> None:
        from orchestrator.merge_types import ItemLifecycleState

        assert {m.name for m in ItemLifecycleState} == self._MEMBER_NAMES
        assert len(ItemLifecycleState) == 10

    def test_members_are_str_instances_with_lowercase_wire_values(self) -> None:
        from orchestrator.merge_types import ItemLifecycleState

        assert isinstance(ItemLifecycleState.VERIFYING, str)
        assert ItemLifecycleState.VERIFYING == 'verifying'
        assert ItemLifecycleState.QUEUED == 'queued'
        assert ItemLifecycleState.GATE_REVERIFY == 'gate_reverify'

    def test_importable_from_merge_queue_reexport_shim(self) -> None:
        from orchestrator.merge_queue import ItemLifecycleState as MQState
        from orchestrator.merge_types import ItemLifecycleState as MTState

        assert MQState is MTState
