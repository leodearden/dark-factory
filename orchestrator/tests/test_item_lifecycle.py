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

import pytest

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


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: ItemLifecycle registry storage
# ---------------------------------------------------------------------------


class TestItemLifecycleRegistry:
    """ItemLifecycle() is a request_id-keyed lifecycle-state registry
    (task 2164 step-3).

    RED until step-4 GREEN implements the ItemLifecycle registry class in
    merge_queue.py.
    """

    def test_register_seeds_queued_state(self) -> None:
        from orchestrator.merge_queue import ItemLifecycle, ItemLifecycleState

        registry = ItemLifecycle()

        registry.register('mr-aaaaaaaa')

        assert registry.current('mr-aaaaaaaa') == ItemLifecycleState.QUEUED

    def test_current_of_unknown_rid_is_none(self) -> None:
        from orchestrator.merge_queue import ItemLifecycle

        registry = ItemLifecycle()

        assert registry.current('mr-unknown0') is None

    def test_two_distinct_request_ids_are_tracked_independently(self) -> None:
        from orchestrator.merge_queue import ItemLifecycle, ItemLifecycleState

        registry = ItemLifecycle()
        registry.register('mr-aaaaaaaa')
        registry.register('mr-bbbbbbbb', initial=ItemLifecycleState.MERGING)

        assert registry.current('mr-aaaaaaaa') == ItemLifecycleState.QUEUED
        assert registry.current('mr-bbbbbbbb') == ItemLifecycleState.MERGING

    def test_duplicate_register_raises_and_leaves_state_unchanged(self) -> None:
        from orchestrator.merge_queue import ItemLifecycle, ItemLifecycleState

        registry = ItemLifecycle()
        registry.register('mr-aaaaaaaa')

        with pytest.raises(ValueError):
            registry.register('mr-aaaaaaaa', initial=ItemLifecycleState.MERGING)

        assert registry.current('mr-aaaaaaaa') == ItemLifecycleState.QUEUED


# ---------------------------------------------------------------------------
# step-5 RED / step-6 GREEN: transition() happy path + single-source
# ---------------------------------------------------------------------------


class TestItemLifecycleTransitionHappyPath:
    """transition() drives the registry through the canonical legal
    sequence, and the registry is the single source of truth after each
    move (task 2164 step-5).

    RED until step-6 GREEN adds the _LEGAL_TRANSITIONS table and implements
    ItemLifecycle.transition().
    """

    def test_full_canonical_sequence_ends_at_terminal(self) -> None:
        from itertools import pairwise

        from orchestrator.merge_queue import ItemLifecycle, ItemLifecycleState

        registry = ItemLifecycle()
        rid = 'mr-aaaaaaaa'
        registry.register(rid)

        sequence = [
            ItemLifecycleState.QUEUED,
            ItemLifecycleState.LANE_BUFFERED,
            ItemLifecycleState.MERGING,
            ItemLifecycleState.AWAITING_VERIFY,
            ItemLifecycleState.DISPATCHING,
            ItemLifecycleState.VERIFYING,
            ItemLifecycleState.GATE_REVERIFY,
            ItemLifecycleState.FINALIZING,
            ItemLifecycleState.TERMINAL,
        ]

        for from_state, to_state in pairwise(sequence):
            registry.transition(rid, from_state, to_state)
            assert registry.current(rid) == to_state  # registry is the single source

        assert registry.current(rid) == ItemLifecycleState.TERMINAL

    def test_redispatch_bounce_is_a_legal_in_place_retry(self) -> None:
        """A representative in-place retry branch: DISPATCHING ->
        REDISPATCH_PARKED -> DISPATCHING (the redispatch bounce evidenced
        by ``_redispatch``)."""
        from orchestrator.merge_queue import ItemLifecycle, ItemLifecycleState

        registry = ItemLifecycle()
        rid = 'mr-bbbbbbbb'
        registry.register(rid, initial=ItemLifecycleState.DISPATCHING)

        registry.transition(
            rid, ItemLifecycleState.DISPATCHING, ItemLifecycleState.REDISPATCH_PARKED,
        )
        assert registry.current(rid) == ItemLifecycleState.REDISPATCH_PARKED

        registry.transition(
            rid, ItemLifecycleState.REDISPATCH_PARKED, ItemLifecycleState.DISPATCHING,
        )
        assert registry.current(rid) == ItemLifecycleState.DISPATCHING

    def test_cascade_remerge_from_gate_reverify_is_a_legal_in_place_retry(self) -> None:
        """The documented cascade remerge: a downstream entry parked at
        GATE_REVERIFY when a head verify fails is remerged (-> MERGING) and
        then appended to ``_redispatch`` (-> REDISPATCH_PARKED). Mirrors
        ``test_redispatch_bounce_is_a_legal_in_place_retry`` (task 2164
        step-9; fixes the GATE_REVERIFY out-set asymmetry with VERIFYING/
        FINALIZING, both of which already legally reach MERGING)."""
        from orchestrator.merge_queue import ItemLifecycle, ItemLifecycleState

        registry = ItemLifecycle()
        rid = 'mr-cccccccc'
        registry.register(rid, initial=ItemLifecycleState.GATE_REVERIFY)

        registry.transition(
            rid, ItemLifecycleState.GATE_REVERIFY, ItemLifecycleState.MERGING,
        )
        assert registry.current(rid) == ItemLifecycleState.MERGING

        registry.transition(
            rid, ItemLifecycleState.MERGING, ItemLifecycleState.REDISPATCH_PARKED,
        )
        assert registry.current(rid) == ItemLifecycleState.REDISPATCH_PARKED


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: illegal transitions raise IllegalLifecycleTransition
# ---------------------------------------------------------------------------


class TestItemLifecycleIllegalTransitions:
    """transition() rejects every illegal move with the dedicated
    IllegalLifecycleTransition, leaving the registry's stored state
    unchanged (task 2164 step-7, L-1 enforcement).

    RED until step-8 GREEN defines IllegalLifecycleTransition and hardens
    ItemLifecycle.transition() to raise it for all three guard classes.
    """

    def test_skip_stage_move_raises(self) -> None:
        from orchestrator.merge_queue import (
            IllegalLifecycleTransition,
            ItemLifecycle,
            ItemLifecycleState,
        )

        registry = ItemLifecycle()
        rid = 'mr-aaaaaaaa'
        registry.register(rid)

        with pytest.raises(IllegalLifecycleTransition):
            registry.transition(rid, ItemLifecycleState.QUEUED, ItemLifecycleState.FINALIZING)

        assert registry.current(rid) == ItemLifecycleState.QUEUED

    def test_move_out_of_terminal_raises(self) -> None:
        from orchestrator.merge_queue import (
            IllegalLifecycleTransition,
            ItemLifecycle,
            ItemLifecycleState,
        )

        registry = ItemLifecycle()
        rid = 'mr-aaaaaaaa'
        registry.register(rid, initial=ItemLifecycleState.TERMINAL)

        with pytest.raises(IllegalLifecycleTransition):
            registry.transition(rid, ItemLifecycleState.TERMINAL, ItemLifecycleState.QUEUED)

        assert registry.current(rid) == ItemLifecycleState.TERMINAL

    def test_from_state_mismatch_raises_and_leaves_state_unchanged(self) -> None:
        """Registry's actual current state is QUEUED (from register()'s
        default); the caller wrongly believes it is VERIFYING. Even though
        VERIFYING -> GATE_REVERIFY is itself a legal edge, the mismatch
        against the registry's real state must reject the call."""
        from orchestrator.merge_queue import (
            IllegalLifecycleTransition,
            ItemLifecycle,
            ItemLifecycleState,
        )

        registry = ItemLifecycle()
        rid = 'mr-aaaaaaaa'
        registry.register(rid)

        with pytest.raises(IllegalLifecycleTransition):
            registry.transition(
                rid, ItemLifecycleState.VERIFYING, ItemLifecycleState.GATE_REVERIFY,
            )

        assert registry.current(rid) == ItemLifecycleState.QUEUED

    def test_unregistered_rid_raises(self) -> None:
        from orchestrator.merge_queue import (
            IllegalLifecycleTransition,
            ItemLifecycle,
            ItemLifecycleState,
        )

        registry = ItemLifecycle()

        with pytest.raises(IllegalLifecycleTransition):
            registry.transition(
                'mr-unknown0', ItemLifecycleState.QUEUED, ItemLifecycleState.MERGING,
            )

        assert registry.current('mr-unknown0') is None
