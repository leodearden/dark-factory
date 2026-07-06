"""Tests for LaneLifecycle (W11 alpha) — LaneState, LEGAL_TRANSITIONS, durable
record I/O, illegal-transition born-at-L2 escalation, quarantine helper.

Step test-contract: RED — module orchestrator.lane_lifecycle is absent; import fails.
Step test-legal-roundtrip: RED — LaneLifecycle/transition/read absent.
Step test-illegal-escalate: RED — illegal branch not implemented (files nothing).
Step test-quarantine: RED — quarantine method absent.
"""

from __future__ import annotations

from orchestrator.lane_lifecycle import (
    ESCALATION_SENTINEL_ROLE,
    LEGAL_TRANSITIONS,
    IllegalLaneTransition,
    LaneState,
)

# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------


class TestStaticContract:
    def test_lane_state_has_exactly_six_members(self):
        assert {member.name for member in LaneState} == {
            'SEED', 'REGISTERED', 'ASSIGNED', 'IN_USE', 'RELEASED', 'QUARANTINED',
        }

    def test_lane_state_values_are_lowercase_strings(self):
        for member in LaneState:
            assert member.value == member.name.lower()

    def test_legal_transitions_contains_required_edges(self):
        required = {
            (None, LaneState.SEED),
            (LaneState.SEED, LaneState.REGISTERED),
            (LaneState.REGISTERED, LaneState.ASSIGNED),
            (LaneState.RELEASED, LaneState.ASSIGNED),
            (LaneState.ASSIGNED, LaneState.IN_USE),
            (LaneState.IN_USE, LaneState.RELEASED),
            (LaneState.ASSIGNED, LaneState.RELEASED),
        }
        assert required <= LEGAL_TRANSITIONS

    def test_legal_transitions_contains_any_to_quarantined(self):
        # "any" includes every real state AND the pre-record None origin.
        origins = [*list(LaneState), None]
        for origin in origins:
            assert (origin, LaneState.QUARANTINED) in LEGAL_TRANSITIONS

    def test_legal_transitions_excludes_illegal_edge(self):
        assert (LaneState.RELEASED, LaneState.IN_USE) not in LEGAL_TRANSITIONS

    def test_illegal_lane_transition_is_an_exception(self):
        assert issubclass(IllegalLaneTransition, Exception)

    def test_escalation_sentinel_role(self):
        assert ESCALATION_SENTINEL_ROLE == 'harness-lane-lifecycle'
        assert ESCALATION_SENTINEL_ROLE.startswith('harness-')
