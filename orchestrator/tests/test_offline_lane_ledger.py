"""What one offline-lane pass tells the flake ledger: ``offline_lane_ledger``'s pure rules.

PRD ``plans/offline-lane-red-path-prd.md`` D6 and section 5.8.  The engine that makes the
recorder calls is ``test_offline_lane_ledger_scenarios.py``'s subject.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
from _lane_episode_fakes import (
    COMMAND,
    HEAD0,
    HEAD1,
    HEAD2,
    SPACED,
    T0,
    A,
    B,
    C,
    _apply,
    _covered,
    _episode,
    _observation,
    _record,
    _red,
    _run,
)

from orchestrator.flake_ledger import (
    DEBT_CONTEXT_EVIDENCE_LIMIT,
    UNKNOWN_TEST_ID,
    DebtContext,
    DebtTrigger,
    EvidenceKind,
    FailureEvidence,
    FlakeCallSite,
    FlakeSuppression,
    FlakeVerdict,
    RecentOccurrence,
)
from orchestrator.offline_lane_confirm import (
    Confirmed,
    ConfirmMode,
    Outage,
    Unconfirmable,
    UnconfirmableReason,
)
from orchestrator.offline_lane_episode_state import (
    CommandRecord,
    CoveringStatuses,
    CoveringVerdict,
    Episode,
    PassObservation,
    reproduce_for,
    unconfirmable_node,
)
from orchestrator.offline_lane_ledger import (
    INCONCLUSIVE_UNEXPLAINED,
    LEGACY_WORKING_DIRECTORY,
    NUMERIC_UNNAMED_CLEAR,
    DebtDecision,
    Recovery,
    debt_context,
    debt_decision,
    pass_suppressions,
    recent_occurrences,
    recoveries,
)
from orchestrator.offline_lane_records import (
    LEGACY_NUMERIC_COMMAND,
    FailureRecord,
    PassRecord,
    ProbeKind,
    ProbeResult,
    RunRecord,
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
        assert observation.detect is not None
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


def _green(head: str = HEAD1, *, at=T0 + SPACED, pass_id: str = 'p1') -> PassObservation:
    return _observation(outcome=None, head=head, at=at, pass_id=pass_id, detect=_run('detect', head, at))


def _recoveries(before: CommandRecord, observation: PassObservation, **reads) -> tuple[Recovery, ...]:
    applied = _apply(before, observation, **reads)
    return recoveries(before, applied.effect.episodes_ended, observation)


def _held(node: str = A, **fields) -> Episode:
    """An unbound episode held at one counted pass at HEAD1."""
    return _episode(node, **{'first_red_head': HEAD1, 'reproduced_heads': (HEAD1,), **fields})


class TestRecoveries:
    def test_a_green_pass_at_a_reproduced_head_recovers_at_that_head(self) -> None:
        before = _record(_held())

        assert _recoveries(before, _green(HEAD1)) == (Recovery(A, HEAD1, before.episodes[A]),)

    def test_a_green_pass_elsewhere_recovers_at_the_last_green_head_when_it_reproduced_there(self) -> None:
        before = _record(_held(last_green_head=HEAD1))

        (recovery,) = _recoveries(before, _green(HEAD2))

        assert recovery.evidence_head == HEAD1

    def test_a_test_that_never_passed_at_a_reproduced_head_does_not_recover(self) -> None:
        before = _record(_held(last_green_head=HEAD0))

        assert _recoveries(before, _green(HEAD2)) == ()

    def test_a_passed_row_in_a_red_pass_recovers(self) -> None:
        before = _record(_held())
        observation = _red(reproduced=(B,), outcomes={A: 'passed', B: 'failed'}, head=HEAD1)

        (recovery,) = _recoveries(before, observation)

        assert (recovery.node, recovery.evidence_head) == (A, HEAD1)

    @pytest.mark.parametrize(
        'episode',
        [
            _covered(A, first_red_head=HEAD1, reproduced_heads=(HEAD1,)),
            _held(had_outage_pass=True),
        ],
        ids=['fix_task_filed', 'saw_an_outage_pass'],
    )
    def test_an_episode_with_a_fix_task_or_an_outage_pass_does_not_recover(self, episode: Episode) -> None:
        assert _recoveries(_record(episode), _green(HEAD1)) == ()

    def test_an_expired_episode_does_not_recover(self) -> None:
        before = _record(_held(last_counted_started_at=T0 - timedelta(hours=25)))
        observation = _green(HEAD1)

        assert [e.reason for e in _apply(before, observation).effect.episodes_ended] == ['expired']
        assert _recoveries(before, observation) == ()

    def test_a_cleared_episode_does_not_recover(self) -> None:
        before = _record(_held())
        observation = _red(reproduced=(), cleared=(A,), head=HEAD1)

        assert [e.reason for e in _apply(before, observation).effect.episodes_ended] == ['cleared']
        assert _recoveries(before, observation) == ()

    def test_a_replaced_episode_does_not_recover(self) -> None:
        before = _record(_covered(A, first_red_head=HEAD1, reproduced_heads=(HEAD1,)))
        observation = _red((A,), head=HEAD1)
        verdicts = {A: CoveringVerdict.DONE_CONTAINED}

        assert [e.reason for e in _apply(before, observation, verdicts=verdicts).effect.episodes_ended] == ['replaced']
        assert _recoveries(before, observation, verdicts=verdicts) == ()

    def test_the_command_pseudo_test_does_not_recover(self) -> None:
        before = _record(_held(unconfirmable_node(COMMAND)))
        observation = _green(HEAD1)

        assert [e.reason for e in _apply(before, observation).effect.episodes_ended] == ['passed']
        assert _recoveries(before, observation) == ()

    def test_an_uncounted_reproduction_head_is_evidence(self) -> None:
        first = _red((A,), head=HEAD1, at=T0 + SPACED, pass_id='p1')
        held = _apply(_record(), first).record
        again = _red((A,), head=HEAD2, at=T0 + SPACED + timedelta(minutes=3), pass_id='p2')
        applied = _apply(held, again)
        before = applied.record

        assert applied.effect.tests_counted == ()
        assert before.episodes[A].reproduced_heads == (HEAD1, HEAD2)
        (recovery,) = _recoveries(before, _green(HEAD2, at=T0 + SPACED + timedelta(minutes=10), pass_id='p3'))
        assert recovery.evidence_head == HEAD2


class TestDebtDecision:
    UNREADABLE = CoveringStatuses({}, 'task store unreachable')

    @pytest.mark.parametrize('episode', [None, _held()], ids=['no_episode', 'unbound_episode'])
    def test_an_unbound_test_is_eligible_without_reading_the_covering_status(self, episode) -> None:
        assert debt_decision(A, COMMAND, episode, self.UNREADABLE) is DebtDecision.ELIGIBLE

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'blocked', 'deferred'])
    def test_an_open_or_deferred_fix_task_covers_the_test(self, status: str) -> None:
        covering = CoveringStatuses({'9001': status})

        assert debt_decision(A, COMMAND, _covered(A, '9001'), covering) is DebtDecision.COVERED

    @pytest.mark.parametrize('statuses', [{'9001': 'done'}, {'9001': 'cancelled'}, {}], ids=['done', 'cancelled', 'absent'])
    def test_a_finished_or_absent_fix_task_leaves_the_test_eligible(self, statuses: dict[str, str]) -> None:
        covering = CoveringStatuses(statuses)

        assert debt_decision(A, COMMAND, _covered(A, '9001'), covering) is DebtDecision.ELIGIBLE

    @pytest.mark.parametrize('status', ['pending', 'deferred', 'done'])
    def test_a_swallowed_binding_leaves_the_test_eligible_whatever_the_status(self, status: str) -> None:
        episode = _covered(A, '9001', binding_swallowed=True)

        assert debt_decision(A, COMMAND, episode, CoveringStatuses({'9001': status})) is DebtDecision.ELIGIBLE

    def test_an_unreadable_status_leaves_a_bound_test_undecided(self) -> None:
        assert debt_decision(A, COMMAND, _covered(A, '9001'), self.UNREADABLE) is DebtDecision.UNDECIDED

    @pytest.mark.parametrize('node', [unconfirmable_node(COMMAND), UNKNOWN_TEST_ID])
    @pytest.mark.parametrize('bound', [False, True])
    def test_the_pseudo_test_and_the_sentinel_are_never_offered(self, node: str, bound: bool) -> None:
        episode = _covered(node, '9001') if bound else None

        assert debt_decision(node, COMMAND, episode, CoveringStatuses({'9001': 'done'})) is DebtDecision.NEVER


def _occurrence(index: int, node: str = A, head: str = HEAD1) -> RecentOccurrence:
    return RecentOccurrence(f'p{index}', head, EvidenceKind.failed, f'AssertionError in {node}')


class TestDebtContext:
    def test_a_clear_with_no_episode_cites_this_pass(self) -> None:
        observation = _red(reproduced=(), cleared=(A,), head=HEAD1, pass_id='p9')
        occurrences = (_occurrence(9),)

        context = debt_context(A, DebtTrigger.cleared, observation, None, occurrences)

        assert context == DebtContext(
            call_site=FlakeCallSite.offline_lane,
            trigger=DebtTrigger.cleared,
            command=COMMAND,
            working_directory='integration',
            reproduce=reproduce_for(observation, (A,)),
            heads=(HEAD1,),
            pass_ids=('p9',),
            evidence=occurrences,
        )

    def test_the_reproduce_line_selects_the_test(self) -> None:
        observation = _red(reproduced=(), cleared=(A,))

        reproduce = debt_context(A, DebtTrigger.cleared, observation, None, ()).reproduce

        assert reproduce is not None
        assert reproduce.startswith('cd integration && pytest -m integration')
        assert A in reproduce

    def test_heads_and_pass_ids_follow_the_episode_then_the_pass_distinct_and_in_order(self) -> None:
        episode = _held(counted_passes=('p1', 'p2'), reproduced_heads=(HEAD0, HEAD1))
        observation = _green(HEAD1, pass_id='p3')

        context = debt_context(A, DebtTrigger.recovered, observation, episode, ())

        assert context.trigger is DebtTrigger.recovered
        assert context.heads == (HEAD0, HEAD1)
        assert context.pass_ids == ('p1', 'p2', 'p3')

    def test_a_pass_head_the_episode_did_not_reproduce_at_comes_last(self) -> None:
        context = debt_context(A, DebtTrigger.recovered, _green(HEAD2, pass_id='p3'), _held(), ())

        assert context.heads == (HEAD1, HEAD2)

    def test_a_legacy_leg_works_from_the_checkout_root(self) -> None:
        observation = _numeric(Confirmed(ConfirmMode.LEG_NATIVE, (A,), (), ()))

        context = debt_context(A, DebtTrigger.cleared, observation, None, ())

        assert context.working_directory == LEGACY_WORKING_DIRECTORY == '.'
        assert context.reproduce == reproduce_for(observation, (A,))

    def test_the_evidence_is_the_last_five_occurrences_most_recent_last(self) -> None:
        occurrences = tuple(_occurrence(i) for i in range(DEBT_CONTEXT_EVIDENCE_LIMIT + 2))

        context = debt_context(A, DebtTrigger.cleared, _red(reproduced=(), cleared=(A,)), None, occurrences)

        assert context.evidence == occurrences[-DEBT_CONTEXT_EVIDENCE_LIMIT:]


def _pass_record(index: int, head: str = HEAD1, *, detect: tuple[str, ...] = (), confirm: tuple[str, ...] = ()) -> PassRecord:
    at = T0 + index * SPACED
    confirm_run: RunRecord | None = None
    if confirm:
        confirm_run = replace(_run('confirm', head, at), failures=tuple(
            FailureRecord(node, 'error', f'confirm error in {node}') for node in confirm
        ), returncode=1)
    return PassRecord(
        command=COMMAND, leg='generic', head=head, pass_id=f'p{index}', started_at=at,
        detect=_run('detect', head, at, failing=detect), confirm=confirm_run, outcome=None, effect=None,
    )


class TestRecentOccurrences:
    def test_one_occurrence_per_earlier_pass_naming_the_test_oldest_first_then_this_pass(self) -> None:
        earlier = [_pass_record(1, HEAD0, detect=(A,)), _pass_record(2, detect=(B,)), _pass_record(3, detect=(A, B))]
        current = _red(reproduced=(), cleared=(A,), head=HEAD2, pass_id='p4')

        assert recent_occurrences(A, earlier, current) == (
            RecentOccurrence('p1', HEAD0, EvidenceKind.failed, f'AssertionError in {A}'),
            RecentOccurrence('p3', HEAD1, EvidenceKind.failed, f'AssertionError in {A}'),
            RecentOccurrence('p4', HEAD2, EvidenceKind.failed, f'AssertionError in {A}'),
        )

    def test_an_earlier_pass_whose_detect_does_not_name_the_test_falls_back_to_its_confirm(self) -> None:
        (occurrence,) = recent_occurrences(A, [_pass_record(1, confirm=(A,))], None)

        assert occurrence == RecentOccurrence('p1', HEAD1, EvidenceKind.error, f'confirm error in {A}')

    def test_a_current_pass_naming_no_failure_for_the_test_adds_nothing(self) -> None:
        assert recent_occurrences(A, [_pass_record(1, detect=(A,))], _green(HEAD1, pass_id='p2')) == (
            RecentOccurrence('p1', HEAD1, EvidenceKind.failed, f'AssertionError in {A}'),
        )

    def test_at_most_the_most_recent_limit_are_kept(self) -> None:
        earlier = [_pass_record(i, detect=(A,)) for i in range(DEBT_CONTEXT_EVIDENCE_LIMIT + 3)]

        found = recent_occurrences(A, earlier, None)

        assert [o.pass_id for o in found] == [r.pass_id for r in earlier[-DEBT_CONTEXT_EVIDENCE_LIMIT:]]
