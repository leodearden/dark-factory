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

import pytest
import shared.task_metadata as task_metadata_module
from shared.deploy_state import DeployPhase, DeployState, VerifyBaseline
from shared.task_metadata import parse_metadata

from orchestrator.deploy_state import (
    _LEGAL,
    IllegalDeployTransition,
    enforce_transition,
    is_legal_transition,
)


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


class TestSharedRegistryRegistration:
    """Observable 3: DeployState is registered into W3's shared _SUBMODEL_REGISTRY.

    Importing shared.deploy_state (this file's step-1 import above) is
    itself the registration trigger (§5.2 seam note) — no explicit register
    call is made here.

    NOTE — cross-suite process isolation: this module-level import
    registers the REAL DeployState into
    ``shared.task_metadata._SUBMODEL_REGISTRY`` permanently for the life of
    the interpreter (this file adds no reset fixture — the registration is
    meant to be process-durable, mirroring how the two live processes
    boot). That is safe only because the verify harness runs each workspace
    member's tests as its own `uv run --project <member> pytest` process
    (CLAUDE.md's ENV GOTCHA), so this process never also collects
    shared/tests/test_task_metadata.py. A future combined/monorepo pytest
    run (e.g. a bare `pytest` from the repo root — root pyproject.toml's
    `[tool.pytest.ini_options]` makes that collection possible) that
    collected both suites into one process would leak this real
    registration into shared/tests/test_task_metadata.py's
    TestSubmodelRegistry, whose stub-registration tests would then raise a
    conflict ValueError — that fixture's snapshot/restore only guards
    leaks *within* its own file, not a pre-existing cross-file
    registration. See shared/src/shared/deploy_state.py's docstring for the
    same note.
    """

    def test_deploy_state_registered_under_its_key(self) -> None:
        assert task_metadata_module._SUBMODEL_REGISTRY.get('deploy_state') is DeployState

    def test_parse_metadata_write_emits_no_warnings_for_deploy_state_slice(self) -> None:
        model, warnings = parse_metadata(
            {'deploy_state': {'phase': 'scheduled'}}, direction='write'
        )
        assert warnings == []
        assert isinstance(model.deploy_state, DeployState)  # type: ignore[attr-defined]
        assert model.deploy_state.phase == DeployPhase.SCHEDULED  # type: ignore[attr-defined]

    def test_parse_metadata_invalid_deploy_state_slice_yields_one_warning(self) -> None:
        _model, warnings = parse_metadata({'deploy_state': [1, 2]}, direction='read')
        assert len(warnings) == 1
        assert warnings[0].field == 'deploy_state'
        assert warnings[0].code == 'invalid_submodel'


class TestLegalTransitionTable:
    """_LEGAL + is_legal_transition (orchestrator-domain state machine).

    Pins only unambiguous edges (PRD §5.2 chart: scheduled→ran→
    {verified|failed|escalated}→done) — see design decision on why the
    exact set of middle transition edges is left to ζ coordination.
    """

    def test_legal_forward_edges_are_true(self) -> None:
        assert _LEGAL[(DeployPhase.SCHEDULED, DeployPhase.RAN)] is True
        assert _LEGAL[(DeployPhase.RAN, DeployPhase.VERIFIED)] is True
        assert _LEGAL[(DeployPhase.RAN, DeployPhase.FAILED)] is True
        assert _LEGAL[(DeployPhase.RAN, DeployPhase.ESCALATED)] is True
        assert _LEGAL[(DeployPhase.VERIFIED, DeployPhase.DONE)] is True
        assert _LEGAL[(DeployPhase.ESCALATED, DeployPhase.DONE)] is True

    def test_illegal_edges_are_false(self) -> None:
        # PRD D2: cannot write done directly from scheduled.
        assert is_legal_transition(DeployPhase.SCHEDULED, DeployPhase.DONE) is False
        # done is terminal.
        assert is_legal_transition(DeployPhase.DONE, DeployPhase.SCHEDULED) is False
        # no backward transition.
        assert is_legal_transition(DeployPhase.VERIFIED, DeployPhase.SCHEDULED) is False
        # no self-loop.
        assert is_legal_transition(DeployPhase.RAN, DeployPhase.RAN) is False

    def test_legal_table_shape_and_default_false_fallback(self) -> None:
        assert isinstance(_LEGAL, dict)
        for (old, new), legal in _LEGAL.items():
            assert isinstance(old, DeployPhase)
            assert isinstance(new, DeployPhase)
            assert isinstance(legal, bool)
        # A pair absent from the table falls back to False rather than KeyError.
        assert (DeployPhase.SCHEDULED, DeployPhase.DONE) not in _LEGAL
        assert is_legal_transition(DeployPhase.SCHEDULED, DeployPhase.DONE) is False


class TestEnforceTransition:
    """DS-2: enforce_transition raises + files a loud escalation, never silent."""

    def test_legal_edge_returns_none_and_does_not_call_sink(self) -> None:
        calls = []
        sink = lambda task_id, old, new: calls.append((task_id, old, new))  # noqa: E731

        result = enforce_transition(
            DeployPhase.SCHEDULED, DeployPhase.RAN, task_id='t1', escalation_sink=sink
        )

        assert result is None
        assert calls == []

    def test_illegal_edge_raises_and_files_escalation_before_raising(self) -> None:
        calls = []
        sink = lambda task_id, old, new: calls.append((task_id, old, new))  # noqa: E731

        with pytest.raises(IllegalDeployTransition):
            enforce_transition(
                DeployPhase.SCHEDULED, DeployPhase.DONE, task_id='t1', escalation_sink=sink
            )

        assert calls == [('t1', DeployPhase.SCHEDULED, DeployPhase.DONE)]

    def test_initial_set_from_none_returns_none_and_does_not_call_sink(self) -> None:
        calls = []
        sink = lambda task_id, old, new: calls.append((task_id, old, new))  # noqa: E731

        result = enforce_transition(None, DeployPhase.SCHEDULED, task_id='t1', escalation_sink=sink)

        assert result is None
        assert calls == []
