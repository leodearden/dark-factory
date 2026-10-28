"""What one offline-lane pass tells the flake ledger: ``offline_lane_ledger``'s pure rules.

PRD ``plans/offline-lane-red-path-prd.md`` D6 and section 5.8.  The engine that makes the
recorder calls is ``test_offline_lane_ledger_scenarios.py``'s subject.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from _lane_episode_fakes import (
    COMMAND,
    HEAD1,
    A,
    B,
    C,
    _observation,
    _red,
    _run,
)

from orchestrator.flake_ledger import (
    EvidenceKind,
    FailureEvidence,
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
)
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Outage,
    Unconfirmable,
    UnconfirmableReason,
)
from orchestrator.offline_lane_episode_state import PassObservation, unconfirmable_node
from orchestrator.offline_lane_ledger import (
    INCONCLUSIVE_UNEXPLAINED,
    NUMERIC_UNNAMED_CLEAR,
    pass_suppressions,
)
from orchestrator.offline_lane_records import (
    LEGACY_NUMERIC_COMMAND,
    FailureRecord,
    ProbeKind,
    ProbeResult,
)

RECORD_DIR = '/records/integration/p1'
PSI = 7.5


def _suppressions(observation: PassObservation, psi: float | None = PSI) -> tuple[FlakeSuppression, ...]:
    return pass_suppressions(observation, record_dir=RECORD_DIR, psi_cpu_some10=psi)


def _numeric(outcome: Confirmed | Unconfirmable) -> PassObservation:
    observation = _observation(outcome=outcome, detect=_run('detect', HEAD1, _observation().started_at, failing=(A,)))
    return replace(observation, command=LEGACY_NUMERIC_COMMAND, leg='numeric', lane_command=None)


def _evidence(node: str, kind: EvidenceKind = EvidenceKind.failed) -> FailureEvidence:
    return FailureEvidence(kind, f'AssertionError in {node}', RECORD_DIR)


class TestConfirmedVerdicts:
    def test_reproduced_and_cleared_tests_are_one_suppression_each(self) -> None:
        suppressions = _suppressions(_red(reproduced=(A,), cleared=(B,)))

        assert [(s.verdict, s.test_ids) for s in suppressions] == [
            (FlakeVerdict.fails_in_isolation, (A,)),
            (FlakeVerdict.passes_in_isolation, (B,)),
        ]

    @pytest.mark.parametrize('psi', [PSI, 0.0, None])
    def test_every_suppression_is_the_lanes_observation_at_the_pass_start(self, psi: float | None) -> None:
        observation = _red(reproduced=(A,), cleared=(B,))

        for suppression in _suppressions(observation, psi):
            assert suppression.call_site == FlakeCallSite.offline_lane
            assert suppression.runner == 'local'
            assert suppression.psi_cpu_some10 == psi
            assert suppression.observed_at == observation.started_at.isoformat()
            assert suppression.unconfirmable_reason is None

    def test_each_named_test_carries_its_detect_failure(self) -> None:
        reproduced, cleared = _suppressions(_red(reproduced=(A,), cleared=(B,)))

        assert reproduced.evidence == {A: _evidence(A)}
        assert cleared.evidence == {B: _evidence(B)}

    def test_a_test_detect_does_not_name_takes_the_confirm_failure(self) -> None:
        observation = _red(reproduced=(A,))
        detect = replace(observation.detect, failures=())
        confirm = replace(
            _run('confirm', HEAD1, observation.started_at),
            failures=(FailureRecord(A, 'error', f'AssertionError in {A}'),),
        )

        (suppression,) = _suppressions(replace(observation, detect=detect, confirm=confirm))

        assert suppression.evidence == {A: _evidence(A, EvidenceKind.error)}

    @pytest.mark.parametrize(
        ('reproduced', 'cleared', 'verdicts'),
        [
            ((A,), (), [FlakeVerdict.fails_in_isolation]),
            ((), (B,), [FlakeVerdict.passes_in_isolation]),
        ],
    )
    def test_an_empty_verdict_makes_no_suppression(self, reproduced, cleared, verdicts) -> None:
        suppressions = _suppressions(_red(reproduced=reproduced, cleared=cleared))

        assert [s.verdict for s in suppressions] == verdicts
        assert all(s.test_ids for s in suppressions)


class TestInconclusive:
    def test_one_unconfirmable_suppression_per_distinct_reason(self) -> None:
        observation = _red(reproduced=(), inconclusive=(A, B))
        confirm = replace(
            _run('confirm', HEAD1, observation.started_at),
            failures=(FailureRecord(A, 'error', f'error: {A}'),),
            returncode=1,
            outcomes=None,
            session_complete=False,
        )

        suppressions = _suppressions(replace(observation, confirm=confirm))

        assert [(s.verdict, s.test_ids, s.unconfirmable_reason) for s in suppressions] == [
            (FlakeVerdict.unconfirmable, (A,), 'confirm_kind_differs'),
            (FlakeVerdict.unconfirmable, (B,), 'confirm_session_incomplete'),
        ]

    def test_without_a_confirm_record_the_inconclusive_tests_share_one_suppression(self) -> None:
        suppressions = _suppressions(_red(reproduced=(), inconclusive=(A, B)))

        assert [(s.verdict, s.test_ids, s.unconfirmable_reason) for s in suppressions] == [
            (FlakeVerdict.unconfirmable, (A, B), INCONCLUSIVE_UNEXPLAINED),
        ]

    def test_a_confirm_that_does_not_explain_the_verdict_is_unexplained(self) -> None:
        observation = _red(reproduced=(), inconclusive=(C,))
        cleared_by_confirm = _run('confirm', HEAD1, observation.started_at)

        (suppression,) = _suppressions(replace(observation, confirm=cleared_by_confirm))

        assert (suppression.test_ids, suppression.unconfirmable_reason) == ((C,), INCONCLUSIVE_UNEXPLAINED)


class TestUnconfirmable:
    def test_an_unconfirmable_pass_is_one_id_less_suppression_with_its_reason(self) -> None:
        outcome = Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '2')
        observation = _observation(outcome=outcome, detect=_run('detect', HEAD1, _observation().started_at, failing=(A,)))

        (suppression,) = _suppressions(observation)

        assert suppression.verdict == FlakeVerdict.unconfirmable
        assert suppression.test_ids == ()
        assert suppression.unconfirmable_reason == 'confirm_exited_nonzero_naming_no_test'
        assert suppression.evidence is None

    def test_the_command_pseudo_test_is_never_named(self) -> None:
        pseudo = unconfirmable_node(COMMAND)
        outcomes = [
            Unconfirmable(UnconfirmableReason.CONFIRM_EXITED_NONZERO_NAMING_NO_TEST, '2'),
            Confirmed(ConfirmMode.NARROWED, (pseudo, A), (), ()),
        ]

        for outcome in outcomes:
            for suppression in _suppressions(_observation(outcome=outcome)):
                assert pseudo not in suppression.test_ids


class TestNumericLeg:
    def test_a_clear_naming_no_test_is_an_unnamed_unconfirmable(self) -> None:
        (suppression,) = _suppressions(_numeric(Confirmed(ConfirmMode.LEG_NATIVE, (), (), ())))

        assert (suppression.verdict, suppression.test_ids, suppression.unconfirmable_reason) == (
            FlakeVerdict.unconfirmable, (), NUMERIC_UNNAMED_CLEAR,
        )

    def test_a_named_reproduction_fails_in_isolation_and_adds_no_unnamed_clear(self) -> None:
        suppressions = _suppressions(_numeric(Confirmed(ConfirmMode.LEG_NATIVE, (A,), (), ())))

        assert [(s.verdict, s.test_ids) for s in suppressions] == [(FlakeVerdict.fails_in_isolation, (A,))]

    def test_a_numeric_exit_is_one_id_less_suppression(self) -> None:
        (suppression,) = _suppressions(_numeric(Unconfirmable(UnconfirmableReason.NUMERIC_EXIT, '64')))

        assert (suppression.test_ids, suppression.unconfirmable_reason) == ((), 'numeric_exit')


class TestNothingToRecord:
    def test_a_green_pass_records_nothing(self) -> None:
        assert _suppressions(_observation(outcome=None)) == ()

    def test_an_outage_pass_records_nothing(self) -> None:
        probe = ProbeResult(ProbeKind.OUTAGE, 75, 'qdrant: down', 'qdrant: down', 1.0)
        observation = replace(_observation(), outcome=Outage(probe), probe=probe)

        assert _suppressions(observation) == ()
