"""The adjudicator's self-calibration (plans/consolidation-adjudicator-prd.md K5, E7).

Public names of ``fused_memory.reconciliation.consolidation_adjudicator_calibration``
only. No live model: the calibration is scored from synthetic outcomes here.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import pytest

from fused_memory.reconciliation.consolidation_adjudicator import (
    AdjudicationRequest,
    AdjudicatorSettings,
    Decision,
    Hazard,
    MemberInput,
    Verdict,
    VerdictReasonCode,
)
from fused_memory.reconciliation.consolidation_adjudicator_calibration import (
    BOUNDS,
    EVAL_SET_PATH,
    EVAL_SET_SHA256,
    BoundBasis,
    BoundDirection,
    BoundName,
    CaseKind,
    CaseOutcome,
    CaseSplit,
    EvalCase,
    EvalSetInvalid,
    OutcomeKind,
    case_exclusion,
    case_request,
    evaluate,
    load_eval_set,
    pool,
    score_run,
)
from fused_memory.reconciliation.consolidation_auto import AutoReasonCode, FlowReasonCode

BANNER = AutoReasonCode.member_carries_correction_banner


def _settings(**overrides) -> AdjudicatorSettings:
    fields = {
        'model': 'gpt-6.1-sol', 'reasoning_effort': 'low', 'endpoint_host': None,
        'timeout_seconds': 60.0, 'field_chars': 4000, 'member_max': 8, 'member_min': 2,
    }
    fields.update(overrides)
    return AdjudicatorSettings(**fields)


@functools.cache
def _cases() -> tuple[EvalCase, ...]:
    return tuple(load_eval_set())


def _adjudicated() -> tuple[EvalCase, ...]:
    return tuple(c for c in _cases() if case_exclusion(c, _settings()) is None)


def _by_id(case_id: str) -> EvalCase:
    return next(c for c in _cases() if c.case_id == case_id)


class TestEvalSet:
    def test_the_pinned_sha(self) -> None:
        assert EVAL_SET_SHA256 == 'df2723389d1ee696b3b5bf3e7efa4c1e558263ca08669c7e58fd8c5befc135af'

    def test_the_path_is_resolved_from_the_package(self) -> None:
        assert EVAL_SET_PATH.name == 'consolidation_adjudicator_eval_set.jsonl'
        assert EVAL_SET_PATH.is_file()

    def test_183_frozen_cases(self) -> None:
        cases = _cases()
        assert len(cases) == 183
        for field in dataclasses.fields(cases[0]):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(cases[0], field.name, None)
        assert all(isinstance(c.split, CaseSplit) and isinstance(c.kind, CaseKind) for c in cases)

    def test_the_populations(self) -> None:
        counts = collections.Counter((c.split, c.kind) for c in _cases())
        assert counts == {
            (CaseSplit.test, CaseKind.fold): 45,
            (CaseSplit.test, CaseKind.contaminated): 45,
            (CaseSplit.test, CaseKind.must_not_fold): 30,
            (CaseSplit.dev, CaseKind.fold): 29,
            (CaseSplit.dev, CaseKind.contaminated): 14,
            (CaseSplit.dev, CaseKind.must_not_fold): 13,
            (CaseSplit.dev, CaseKind.real_proposal): 7,
        }

    def test_a_case_carries_its_members_claim_hazards_and_expectation(self) -> None:
        case = _by_id('TC01')
        assert case.topic == 'recon-marker-gc-architecture'
        assert len(case.members) == 5
        assert case.claim.startswith('Reconciliation bookkeeping')
        assert case.expected.foreign_id == '82ce559f-acf9-4a40-a38c-aef4cfb2294d'
        assert case.expected.decision_set == frozenset({Decision.proceed, Decision.drop})
        assert len(case.expected.keep_ids) == 4

    def test_the_topic_comes_from_the_topic_or_the_seed_topic(self) -> None:
        assert _by_id('TF01').topic == 'scripts-tests-vs-tests-scripts'
        assert _by_id('T-M01').topic is not None
        assert _by_id('A01').topic is None

    def test_a_banner_hazard_names_its_member(self) -> None:
        hazards = [h for c in _cases() for h in c.hazards if h.code is BANNER]
        assert len(hazards) == 38
        assert all(h.member_id is not None for h in hazards)

    def test_a_modified_copy_is_refused_naming_its_sha(self, tmp_path: Path) -> None:
        tampered = tmp_path / 'eval.jsonl'
        tampered.write_bytes(EVAL_SET_PATH.read_bytes() + b'\n')
        with pytest.raises(EvalSetInvalid) as excinfo:
            load_eval_set(tampered)
        observed = hashlib.sha256(tampered.read_bytes()).hexdigest()
        assert observed in str(excinfo.value)

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(EvalSetInvalid, match='missing'):
            load_eval_set(tmp_path / 'absent.jsonl')

    def test_eval_set_invalid_is_a_value_error(self) -> None:
        assert issubclass(EvalSetInvalid, ValueError)


class TestCaseRules:
    def test_default_settings_adjudicate_176(self) -> None:
        assert len(_adjudicated()) == 176

    def test_the_mixed_category_cases_are_refused_by_code(self) -> None:
        refused = [c for c in _cases() if case_exclusion(c, _settings()) is not None]
        assert len(refused) == 7
        assert {case_exclusion(c, _settings()) for c in refused} == {AutoReasonCode.mixed_category}
        assert {c.split for c in refused} == {CaseSplit.dev}

    def test_a_member_count_outside_the_range_is_excluded(self) -> None:
        big = next(c for c in _adjudicated() if len(c.members) == 8)
        assert case_exclusion(big, _settings(member_max=7)) is (
            FlowReasonCode.too_many_members_to_adjudicate
        )
        pair = next(c for c in _adjudicated() if len(c.members) == 2)
        assert case_exclusion(pair, _settings(member_min=3)) is (
            FlowReasonCode.too_few_members_after_repair
        )

    def test_case_request_carries_the_case(self) -> None:
        case = _by_id('TC01')
        request = case_request(case)
        assert isinstance(request, AdjudicationRequest)
        assert request.topic == case.topic
        assert request.claim == case.claim
        assert request.category == case.members[0].category
        assert request.members == tuple(
            MemberInput(id=m.id, category=m.category, text=m.text) for m in case.members
        )

    def test_case_request_has_no_topic_when_the_case_has_none(self) -> None:
        assert case_request(_by_id('A01')).topic is None

    def test_case_request_carries_only_adjudicator_class_hazards(self) -> None:
        case = next(c for c in _adjudicated() if c.hazards)
        request = case_request(case)
        assert request.hazards == tuple(
            Hazard(code=h.code, subject_id=h.member_id)
            for h in case.hazards
            if h.member_id is not None
        )
        assert all(h.code is BANNER for h in request.hazards)


def _verdict(decision: Decision, keep: tuple[str, ...] = ()) -> Verdict:
    code = {
        Decision.proceed: VerdictReasonCode.same_lesson,
        Decision.drop: VerdictReasonCode.not_same_lesson,
        Decision.unresolved: VerdictReasonCode.insufficient_information,
    }[decision]
    return Verdict(decision=decision, keep_member_ids=keep, claim=None, reason_code=code, reason='r')


def _outcome(case: EvalCase, verdict: Verdict | None, kind: OutcomeKind = OutcomeKind.verdict):
    return CaseOutcome(case=case, kind=kind, verdict=verdict)


def _keep_all(case: EvalCase) -> CaseOutcome:
    return _outcome(case, _verdict(Decision.proceed, tuple(m.id for m in case.members)))


def _drop(case: EvalCase) -> CaseOutcome:
    return _outcome(case, _verdict(Decision.drop))


def _answer_key(case: EvalCase) -> CaseOutcome:
    if Decision.proceed in case.expected.decision_set:
        return _outcome(case, _verdict(Decision.proceed, case.expected.keep_ids))
    return _drop(case)


def _run(answer: Callable[[EvalCase], CaseOutcome]):
    return score_run([answer(c) for c in _adjudicated()], member_min=2)


def _figure(figures, name: BoundName):
    return next(f for f in figures.figures if f.name is name)


class TestBounds:
    def test_exactly_the_six_e7_bounds(self) -> None:
        assert len(BOUNDS) == 6
        table = {
            b.name: (b.split, b.direction, Fraction(b.numerator, b.denominator), b.basis)
            for b in BOUNDS
        }
        assert table == {
            BoundName.contaminated_false_keep: (
                CaseSplit.test, BoundDirection.at_most, Fraction(6, 45), BoundBasis.measured),
            BoundName.fold_rate: (
                CaseSplit.test, BoundDirection.at_least, Fraction(32, 45), BoundBasis.measured),
            BoundName.must_not_merge_test: (
                CaseSplit.test, BoundDirection.at_most, Fraction(15, 30), BoundBasis.sanity_floor),
            BoundName.must_not_merge_dev: (
                CaseSplit.dev, BoundDirection.at_most, Fraction(5, 13),
                BoundBasis.regression_guard),
            BoundName.real_proposal_folds: (
                CaseSplit.dev, BoundDirection.at_least, Fraction(5, 6),
                BoundBasis.regression_guard),
            BoundName.invalid_replies: (
                None, BoundDirection.at_most, Fraction(1, 20), BoundBasis.measured),
        }

    def test_the_rationals_are_kept_unreduced(self) -> None:
        by_name = {b.name: b for b in BOUNDS}
        assert (by_name[BoundName.fold_rate].numerator,
                by_name[BoundName.fold_rate].denominator) == (32, 45)
        assert (by_name[BoundName.must_not_merge_test].numerator,
                by_name[BoundName.must_not_merge_test].denominator) == (15, 30)


class TestScoring:
    def test_the_denominators_are_the_adjudicated_populations(self) -> None:
        figures = _run(_answer_key)
        denominators = {f.name: f.denominator for f in figures.figures}
        assert denominators == {
            BoundName.contaminated_false_keep: 45,
            BoundName.fold_rate: 45,
            BoundName.must_not_merge_test: 30,
            BoundName.must_not_merge_dev: 13,
            BoundName.real_proposal_folds: 6,
            BoundName.invalid_replies: 176,
        }

    def test_the_answer_key_passes_all_six(self) -> None:
        figures = _run(_answer_key)
        assert evaluate(figures) == ()
        assert all(f.holds for f in figures.figures)

    def test_always_proceed_keeping_all_fails_the_merge_bounds(self) -> None:
        assert set(evaluate(_run(_keep_all))) == {
            BoundName.contaminated_false_keep,
            BoundName.must_not_merge_test,
            BoundName.must_not_merge_dev,
        }

    def test_always_drop_fails_the_fold_bounds(self) -> None:
        assert set(evaluate(_run(_drop))) == {
            BoundName.fold_rate, BoundName.real_proposal_folds,
        }

    def test_a_proceed_failing_the_keep_rule_scores_as_a_drop(self) -> None:
        case = next(c for c in _adjudicated() if c.kind is CaseKind.fold and len(c.members) == 6)
        kept_too_few = _outcome(case, _verdict(Decision.proceed, tuple(m.id for m in case.members[:3])))
        figures = score_run([kept_too_few], member_min=2)
        assert _figure(figures, BoundName.fold_rate).numerator == 0

    def test_a_foreign_keep_failing_the_keep_rule_is_not_a_false_keep(self) -> None:
        case = next(
            c for c in _adjudicated()
            if c.kind is CaseKind.contaminated and c.split is CaseSplit.test and len(c.members) >= 4
        )
        foreign = case.expected.foreign_id
        assert foreign is not None
        other = next(m.id for m in case.members if m.id != foreign)
        figures = score_run([_outcome(case, _verdict(Decision.proceed, (foreign, other)))], member_min=2)
        assert _figure(figures, BoundName.contaminated_false_keep).numerator == 0

    @pytest.mark.parametrize('kind', [OutcomeKind.content_failure, OutcomeKind.unanswered])
    def test_a_case_without_a_verdict_is_wrong_on_its_bound(self, kind: OutcomeKind) -> None:
        by_kind = {
            CaseKind.contaminated: BoundName.contaminated_false_keep,
            CaseKind.fold: BoundName.fold_rate,
            CaseKind.must_not_fold: BoundName.must_not_merge_test,
        }
        for case_kind, name in by_kind.items():
            case = next(
                c for c in _adjudicated() if c.kind is case_kind and c.split is CaseSplit.test
            )
            figures = score_run([_outcome(case, None, kind)], member_min=2)
            figure = _figure(figures, name)
            bound = next(b for b in BOUNDS if b.name is name)
            expected = 1 if bound.direction is BoundDirection.at_most else 0
            assert (figure.numerator, figure.denominator) == (expected, 1)
            assert _figure(figures, BoundName.invalid_replies).numerator == 1

    def test_a_verdict_is_not_an_invalid_reply(self) -> None:
        figures = score_run([_answer_key(_adjudicated()[0])], member_min=2)
        assert _figure(figures, BoundName.invalid_replies).numerator == 0

    def test_an_empty_population_fails_its_bound(self) -> None:
        figures = score_run([], member_min=2)
        assert set(evaluate(figures)) == set(BoundName)

    def test_an_outcome_carries_a_verdict_exactly_when_it_is_one(self) -> None:
        case = _adjudicated()[0]
        with pytest.raises(ValueError):
            CaseOutcome(case=case, kind=OutcomeKind.verdict, verdict=None)
        with pytest.raises(ValueError):
            CaseOutcome(case=case, kind=OutcomeKind.unanswered, verdict=_verdict(Decision.drop))


def _false_keeps(count: int) -> Callable[[EvalCase], CaseOutcome]:
    contaminated = [
        c.case_id for c in _adjudicated()
        if c.kind is CaseKind.contaminated and c.split is CaseSplit.test
    ][:count]

    def answer(case: EvalCase) -> CaseOutcome:
        if case.case_id in contaminated:
            return _keep_all(case)
        return _answer_key(case)

    return answer


class TestPooling:
    def test_pooling_sums_numerators_and_denominators(self) -> None:
        first, second = _run(_false_keeps(7)), _run(_false_keeps(5))
        assert BoundName.contaminated_false_keep in evaluate(first)
        pooled = pool(first, second)
        figure = _figure(pooled, BoundName.contaminated_false_keep)
        assert (figure.numerator, figure.denominator) == (12, 90)
        assert figure.holds is True
        assert evaluate(pooled) == ()

    def test_pooling_can_fail_where_the_second_run_alone_passes(self) -> None:
        first, second = _run(_false_keeps(9)), _run(_false_keeps(4))
        assert evaluate(second) == ()
        assert evaluate(pool(first, second)) == (BoundName.contaminated_false_keep,)
