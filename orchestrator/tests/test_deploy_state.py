"""Tests for the DeployState typed schema (task 2239, PRD §5.2 DS-1..4).

Step-1 covers DeployPhase's StrEnum vocabulary and DeployState's
from_metadata∘to_metadata round-trip identity (DS-1/DS-3), imported
directly from shared.deploy_state (the shared-visible home — see that
module's docstring for why). Later steps in this file add the
shared-registry registration property (observable 3), the orchestrator-only
_LEGAL transition table, and DS-2's enforce_transition.
"""

from __future__ import annotations

import enum
import json

from shared.deploy_state import DeployPhase, DeployState, VerifyBaseline


class TestDeployPhase:
    def test_is_str_enum(self) -> None:
        assert issubclass(DeployPhase, enum.StrEnum)

    def test_members_have_exact_lowercase_values(self) -> None:
        assert DeployPhase.SCHEDULED == 'scheduled'
        assert DeployPhase.RAN == 'ran'
        assert DeployPhase.VERIFIED == 'verified'
        assert DeployPhase.FAILED == 'failed'
        assert DeployPhase.ESCALATED == 'escalated'
        assert DeployPhase.DONE == 'done'


class TestDeployStateRoundTrip:
    def _make(self) -> DeployState:
        return DeployState(
            phase=DeployPhase.RAN,
            verify_baseline=VerifyBaseline(
                active_enter_timestamp_monotonic=123456789,
                main_pid=4242,
            ),
            ran_at='2026-07-07T00:00:00+00:00',
        )

    def test_from_metadata_of_to_metadata_is_identity(self) -> None:
        ds = self._make()
        assert DeployState.from_metadata(ds.to_metadata()) == ds

    def test_full_json_round_trip_preserves_int_baseline(self) -> None:
        ds = self._make()
        blob = json.loads(json.dumps(ds.to_metadata()))
        assert DeployState.from_metadata(blob) == ds

    def test_from_metadata_absent_slice_is_none(self) -> None:
        assert DeployState.from_metadata({}) is None
        assert DeployState.from_metadata({'other': 1}) is None
