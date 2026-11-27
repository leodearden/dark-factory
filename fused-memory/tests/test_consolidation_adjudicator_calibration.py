"""The adjudicator's self-calibration (plans/consolidation-adjudicator-prd.md K5, E7).

Public names of ``fused_memory.reconciliation.consolidation_adjudicator_calibration``
only. No live model: the scoring is driven by synthetic outcomes, and a run by
the scripted double in ``tests/_adjudicator_doubles.py`` over a real ledger.
"""
from __future__ import annotations

import asyncio
import collections
import dataclasses
import functools
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest
import pytest_asyncio
from _adjudicator_doubles import (
    RESOLVED_MODEL,
    USAGE,
    EvalAnswerKey,
    ScriptedResponses,
    connection_error,
    expected_reply,
    forbidden,
    keep_all_reply,
    no_api_key,
    rate_limited,
    reply_text,
    server_error,
    unauthorized,
    verdict_reply,
)

from fused_memory.llm.list_prices import list_price
from fused_memory.reconciliation.consolidation_adjudicator import (
    AdjudicationRequest,
    AdjudicatorSettings,
    Decision,
    Hazard,
    MemberInput,
    Verdict,
    VerdictReasonCode,
    adjudicator_identity,
)
from fused_memory.reconciliation.consolidation_adjudicator_calibration import (
    BOUNDS,
    CALLS_IN_FLIGHT,
    CRASH_AGE,
    EVAL_SET_PATH,
    EVAL_SET_SHA256,
    MAX_UNANSWERED,
    SCORING_FINGERPRINT,
    STALL_BACKOFF,
    STALL_RUN_CAP,
    TRANSPORT_ATTEMPTS,
    BoundBasis,
    BoundDirection,
    BoundName,
    CalibrationFinding,
    CalibrationRecord,
    CalibrationResult,
    CalibrationState,
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
    mark_stale_if_drifted,
    pool,
    run_calibration,
    score_run,
    scoring_fingerprint,
)
from fused_memory.reconciliation.consolidation_auto import AutoReasonCode, FlowReasonCode
from fused_memory.reconciliation.consolidation_proposal import adjudicator_calibration_key
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

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


def _with_first_bound(**changes: object) -> tuple:
    return (dataclasses.replace(BOUNDS[0], **changes), *BOUNDS[1:])


class TestScoringFingerprint:
    def test_the_fingerprint_is_derived_from_the_bounds_and_the_eval_set(self) -> None:
        assert scoring_fingerprint(BOUNDS, EVAL_SET_SHA256) == SCORING_FINGERPRINT
        assert len(SCORING_FINGERPRINT) == 64
        assert set(SCORING_FINGERPRINT) <= set('0123456789abcdef')

    def test_the_order_of_the_bounds_is_not_a_rule(self) -> None:
        assert scoring_fingerprint(tuple(reversed(BOUNDS)), EVAL_SET_SHA256) == SCORING_FINGERPRINT

    @pytest.mark.parametrize(
        ('bounds', 'eval_sha256'),
        [
            (_with_first_bound(numerator=BOUNDS[0].numerator + 1), EVAL_SET_SHA256),
            (_with_first_bound(direction=BoundDirection.at_least), EVAL_SET_SHA256),
            (_with_first_bound(basis=BoundBasis.sanity_floor), EVAL_SET_SHA256),
            (BOUNDS[:-1], EVAL_SET_SHA256),
            (BOUNDS, '0' * 64),
        ],
        ids=['numerator', 'direction', 'basis', 'dropped_bound', 'eval_set'],
    )
    def test_any_rule_change_changes_the_fingerprint(self, bounds, eval_sha256: str) -> None:
        assert scoring_fingerprint(bounds, eval_sha256) != SCORING_FINGERPRINT


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


NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
RUN_SETTINGS = _settings(endpoint_host='api.openai.com')
OTHER_RESOLVED_MODEL = 'gpt-6.1-sol-2026-11-01'


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    yield store
    await store.close()


def _no_claim_problems(claim: str, topic: str | None) -> tuple[str, ...]:
    return ()


async def _calibrate(
    ledger: ReconLedgerStore, double: ScriptedResponses, *, now: datetime = NOW, **kwargs,
) -> CalibrationResult:
    return await run_calibration(
        ledger=ledger, settings=RUN_SETTINGS, model_call=double,
        claim_check=_no_claim_problems, now=now, retry_delay_s=0, **kwargs,
    )


def _key(**kwargs) -> EvalAnswerKey:
    return EvalAnswerKey(_cases(), **kwargs)


async def _stored(ledger: ReconLedgerStore) -> CalibrationRecord | None:
    row = await ledger.get_by_identity(
        **adjudicator_calibration_key(adjudicator_identity(RUN_SETTINGS)),
    )
    return None if row is None else CalibrationRecord.from_ledger_record(row)


def _test_cases(kind: CaseKind) -> list[EvalCase]:
    return [c for c in _adjudicated() if c.kind is kind and c.split is CaseSplit.test]


def _keeping_foreign(case: EvalCase):
    return keep_all_reply(case) if case.kind is CaseKind.contaminated else expected_reply(case)


def _keeping_foreign_in(case_ids: set[str]):
    def answer(case: EvalCase):
        return keep_all_reply(case) if case.case_id in case_ids else expected_reply(case)
    return answer


def _unanswered(count: int) -> dict[str, tuple[BaseException, ...]]:
    return {
        c.case_id: tuple(connection_error() for _ in range(TRANSPORT_ATTEMPTS))
        for c in _adjudicated()[:count]
    }


RETIRED_FIGURE = {'name': 'retired_bound', 'numerator': 1, 'denominator': 2, 'holds': True}
OTHER_FINGERPRINT = 'f' * 64


def _with_payload(row: ReconLedgerRecord, **changes: object) -> ReconLedgerRecord:
    payload = {**json.loads(row.payload_json), **changes}
    return dataclasses.replace(row, payload_json=json.dumps(payload))


def _under_other_rules(row: ReconLedgerRecord) -> ReconLedgerRecord:
    """*row* as a writer scoring under other bounds would have left it."""
    payload = json.loads(row.payload_json)
    return _with_payload(row, scoring_fingerprint=OTHER_FINGERPRINT, **{
        key: None if payload[key] is None else [*reversed(payload[key]), RETIRED_FIGURE]
        for key in ('figures', 'first_run')
    })


class TestCalibrationRecord:
    def _record(self) -> CalibrationRecord:
        figures = _run(_false_keeps(7))
        return CalibrationRecord(
            identity='i' * 64, state=CalibrationState.FAIL_FIRST_RUN, attempt=1,
            started_at=NOW, model='gpt-6.1-sol', endpoint_host='api.openai.com',
            first_run=figures, figures=figures, unanswered=2,
            resolved_models=(RESOLVED_MODEL,), cost_usd=0.5, wall_time_s=12.5,
        )

    def test_a_record_round_trips_through_its_ledger_row(self) -> None:
        record = self._record()
        row = record.to_ledger_record(now=NOW)
        assert row.state == 'fail_first_run'
        assert row.expires_at is None
        assert {
            'project_id': row.project_id, 'record_kind': row.record_kind, 'task_id': row.task_id,
            'flag_type': row.flag_type, 'run_id': row.run_id,
        } == adjudicator_calibration_key(record.identity)
        assert CalibrationRecord.from_ledger_record(row) == record

    def test_the_failed_bounds_are_read_from_the_figures(self) -> None:
        assert self._record().failed_bounds == (BoundName.contaminated_false_keep,)

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('attempt', 'one'),
            ('started_at', '2026-11-22T12:00:00'),
            ('figures', [{'name': 'nope', 'numerator': 1, 'denominator': 2, 'holds': True}]),
            ('cost_usd', -1),
            ('resolved_models', 'gpt'),
            ('scoring_fingerprint', 7),
        ],
    )
    def test_a_bad_payload_field_is_refused_naming_it(self, field: str, value: object) -> None:
        row = self._record().to_ledger_record(now=NOW)
        payload = json.loads(row.payload_json)
        payload[field] = value
        with pytest.raises(ValueError, match=field):
            CalibrationRecord.from_ledger_record(
                dataclasses.replace(row, payload_json=json.dumps(payload)),
            )

    def test_a_naive_timestamp_is_refused(self) -> None:
        with pytest.raises(ValueError, match='started_at'):
            dataclasses.replace(self._record(), started_at=NOW.replace(tzinfo=None))

    def test_a_record_is_scored_under_the_current_rules_by_default(self) -> None:
        record = self._record()
        assert record.scoring_fingerprint == SCORING_FINGERPRINT
        assert record.scored_under_current_rules is True

    def test_figures_stored_in_another_order_still_decode(self) -> None:
        record = self._record()
        row = record.to_ledger_record(now=NOW)
        payload = json.loads(row.payload_json)
        reordered = _with_payload(
            row, figures=payload['figures'][::-1], first_run=payload['first_run'][::-1],
        )
        decoded = CalibrationRecord.from_ledger_record(reordered)
        assert decoded == record
        assert decoded.figures is not None
        assert tuple(f.name for f in decoded.figures.figures) == tuple(b.name for b in BOUNDS)

    @pytest.mark.parametrize(
        'edit',
        [
            lambda figures: figures[:-1],
            lambda figures: [*figures, RETIRED_FIGURE],
            lambda figures: [*figures[:-1], figures[0]],
        ],
        ids=['missing', 'seventh', 'duplicate'],
    )
    def test_under_the_current_rules_the_figures_name_each_bound_once(self, edit) -> None:
        row = self._record().to_ledger_record(now=NOW)
        figures = json.loads(row.payload_json)['figures']
        with pytest.raises(ValueError, match='figures'):
            CalibrationRecord.from_ledger_record(_with_payload(row, figures=edit(figures)))

    def test_a_row_scored_under_other_rules_decodes_without_its_figures(self) -> None:
        record = self._record()

        decoded = CalibrationRecord.from_ledger_record(
            _under_other_rules(record.to_ledger_record(now=NOW)),
        )

        assert decoded.state is CalibrationState.FAIL_FIRST_RUN
        assert (decoded.figures, decoded.first_run) == (None, None)
        assert decoded.scored_under_current_rules is False
        assert (
            decoded.attempt, decoded.model, decoded.endpoint_host, decoded.resolved_models,
            decoded.cost_usd,
        ) == (
            record.attempt, record.model, record.endpoint_host, record.resolved_models,
            record.cost_usd,
        )

    def test_figures_are_carried_only_under_the_current_rules(self) -> None:
        with pytest.raises(ValueError, match='scoring_fingerprint'):
            dataclasses.replace(self._record(), scoring_fingerprint=OTHER_FINGERPRINT)
        bare = dataclasses.replace(
            self._record(), scoring_fingerprint=OTHER_FINGERPRINT, figures=None, first_run=None,
        )
        assert bare.scored_under_current_rules is False

    def test_a_pass_scored_under_other_rules_may_not_adjudicate(self) -> None:
        record = dataclasses.replace(
            self._record(), state=CalibrationState.PASS, scoring_fingerprint=OTHER_FINGERPRINT,
            figures=None, first_run=None,
        )
        assert CalibrationResult(record=record, ran=False).may_adjudicate is False


class TestCalibrationRun:
    @pytest.mark.asyncio
    async def test_a_first_run_in_bounds_passes(self, ledger) -> None:
        other = _test_cases(CaseKind.fold)[0]
        double = ScriptedResponses(_key(deviations={
            other.case_id: (
                verdict_reply(
                    Decision.proceed, other.expected.keep_ids, resolved_model=OTHER_RESOLVED_MODEL,
                ),
            ),
        }))

        result = await _calibrate(ledger, double)

        assert (result.state, result.ran, result.finding) == (CalibrationState.PASS, True, None)
        assert result.may_adjudicate is True
        assert len(double.requests) == 176
        assert double.peak_in_flight == CALLS_IN_FLIGHT
        row = await ledger.get_by_identity(
            **adjudicator_calibration_key(adjudicator_identity(RUN_SETTINGS)),
        )
        assert row is not None
        assert row.state == 'pass'
        assert row.expires_at is None
        record = CalibrationRecord.from_ledger_record(row)
        assert record == result.record
        assert record.attempt == 1
        assert record.figures is not None
        assert {f.name: (f.numerator, f.denominator) for f in record.figures.figures} == {
            BoundName.contaminated_false_keep: (0, 45),
            BoundName.fold_rate: (45, 45),
            BoundName.must_not_merge_test: (0, 30),
            BoundName.must_not_merge_dev: (0, 13),
            BoundName.real_proposal_folds: (6, 6),
            BoundName.invalid_replies: (0, 176),
        }
        assert record.failed_bounds == ()
        assert record.resolved_models == (RESOLVED_MODEL, OTHER_RESOLVED_MODEL)
        assert (record.model, record.endpoint_host) == ('gpt-6.1-sol', 'api.openai.com')
        price = list_price('gpt-6.1-sol')
        assert price is not None
        assert record.cost_usd == pytest.approx(
            176 * price.usd(USAGE.input_tokens, USAGE.output_tokens),
        )
        assert record.wall_time_s is not None
        assert record.wall_time_s >= 0
        assert record.scoring_fingerprint == SCORING_FINGERPRINT
        assert record.unanswered == 0

    @pytest.mark.asyncio
    async def test_the_row_is_running_before_the_first_reply(self, ledger) -> None:
        gate = asyncio.Event()
        double = ScriptedResponses(_key(), gate=gate)
        run = asyncio.create_task(_calibrate(ledger, double))
        while double.in_flight < CALLS_IN_FLIGHT and not run.done():
            await asyncio.sleep(0)

        stored = await _stored(ledger)

        assert stored is not None
        assert (stored.state, stored.attempt, stored.started_at) == (
            CalibrationState.RUNNING, 1, NOW,
        )
        gate.set()
        assert (await run).state is CalibrationState.PASS

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_retried_and_a_content_failure_is_not(
        self, ledger,
    ) -> None:
        recovered, lost, garbled = _test_cases(CaseKind.fold)[:3]
        key = _key(deviations={
            recovered.case_id: (rate_limited(), rate_limited()),
            lost.case_id: (rate_limited(), server_error(503), connection_error()),
            garbled.case_id: (reply_text('not json'),),
        })
        double = ScriptedResponses(key)

        result = await _calibrate(ledger, double)

        assert key.calls[recovered.case_id] == 3
        assert key.calls[lost.case_id] == TRANSPORT_ATTEMPTS
        assert key.calls[garbled.case_id] == 1
        assert len(double.requests) == 176 + 2 + (TRANSPORT_ATTEMPTS - 1)
        record = result.record
        assert record is not None
        assert record.unanswered == 1
        assert record.figures is not None
        figures = {f.name: (f.numerator, f.denominator) for f in record.figures.figures}
        assert figures[BoundName.invalid_replies] == (2, 176)
        assert figures[BoundName.fold_rate] == (43, 45)
        assert result.state is CalibrationState.PASS

    @pytest.mark.asyncio
    async def test_max_unanswered_cases_still_score(self, ledger) -> None:
        result = await _calibrate(
            ledger, ScriptedResponses(_key(deviations=_unanswered(MAX_UNANSWERED))),
        )

        assert result.state is CalibrationState.PASS
        assert result.record is not None
        assert result.record.unanswered == MAX_UNANSWERED

    @pytest.mark.asyncio
    async def test_one_more_unanswered_case_is_inconclusive(self, ledger) -> None:
        result = await _calibrate(
            ledger, ScriptedResponses(_key(deviations=_unanswered(MAX_UNANSWERED + 1))),
        )

        assert (result.state, result.ran, result.finding) == (
            CalibrationState.INCONCLUSIVE, True, None,
        )
        assert result.may_adjudicate is False
        assert result.record is not None
        assert result.record.figures is None
        assert result.record.unanswered == MAX_UNANSWERED + 1
        assert await _stored(ledger) == result.record

    @pytest.mark.asyncio
    @pytest.mark.parametrize('refusal', [unauthorized, forbidden, no_api_key])
    async def test_a_refused_key_never_fails_the_calibration(self, ledger, refusal) -> None:
        states = [
            (await _calibrate(ledger, ScriptedResponses(refusal()))).state for _ in range(3)
        ]
        assert states == [
            CalibrationState.INCONCLUSIVE, CalibrationState.INCONCLUSIVE, CalibrationState.STALLED,
        ]

    @pytest.mark.asyncio
    async def test_running_out_of_wall_time_is_inconclusive(self, ledger) -> None:
        double = ScriptedResponses(_key(), gate=asyncio.Event())

        result = await _calibrate(ledger, double, wall_time_limit_s=0.05)

        assert (result.state, result.ran, result.finding) == (
            CalibrationState.INCONCLUSIVE, True, None,
        )
        assert result.record is not None
        assert result.record.figures is None
        assert double.in_flight == 0
        stored = await _stored(ledger)
        assert stored is not None
        assert stored.state is CalibrationState.INCONCLUSIVE

    @pytest.mark.asyncio
    async def test_two_failing_runs_fail_and_the_third_call_runs_nothing(self, ledger) -> None:
        first = await _calibrate(ledger, ScriptedResponses(_key(answer=_keeping_foreign)))

        assert (first.state, first.ran, first.finding) == (
            CalibrationState.FAIL_FIRST_RUN, True, None,
        )
        assert first.may_adjudicate is False
        assert first.record is not None
        assert first.record.first_run is not None
        assert first.record.first_run == first.record.figures
        assert first.failed_bounds == (BoundName.contaminated_false_keep,)

        second_double = ScriptedResponses(_key(answer=_keeping_foreign))
        second = await _calibrate(ledger, second_double, now=NOW + timedelta(minutes=10))

        assert len(second_double.requests) == 176
        assert (second.state, second.ran) == (CalibrationState.FAIL, True)
        assert second.finding is CalibrationFinding.adjudicator_calibration_failed
        assert second.failed_bounds == (BoundName.contaminated_false_keep,)
        assert second.record is not None
        assert second.record.attempt == 2
        assert second.record.figures is not None
        pooled = _figure(second.record.figures, BoundName.contaminated_false_keep)
        assert (pooled.numerator, pooled.denominator) == (90, 90)
        assert await _stored(ledger) == second.record

        third_double = ScriptedResponses(_key())
        third = await _calibrate(ledger, third_double, now=NOW + timedelta(minutes=20))

        assert third_double.requests == []
        assert (third.state, third.ran, third.finding) == (CalibrationState.FAIL, False, None)
        assert third.may_adjudicate is False

    @pytest.mark.asyncio
    async def test_a_narrow_first_failure_and_a_clean_second_run_pool_to_pass(
        self, ledger,
    ) -> None:
        seven = {c.case_id for c in _test_cases(CaseKind.contaminated)[:7]}
        first = await _calibrate(ledger, ScriptedResponses(_key(answer=_keeping_foreign_in(seven))))
        assert first.state is CalibrationState.FAIL_FIRST_RUN

        second = await _calibrate(ledger, ScriptedResponses(_key()), now=NOW + timedelta(minutes=10))

        assert (second.state, second.ran, second.finding) == (CalibrationState.PASS, True, None)
        assert second.may_adjudicate is True
        assert second.record is not None
        assert second.record.figures is not None
        pooled = _figure(second.record.figures, BoundName.contaminated_false_keep)
        assert (pooled.numerator, pooled.denominator) == (7, 90)

    @pytest.mark.asyncio
    async def test_a_pass_row_makes_no_calls(self, ledger) -> None:
        await _calibrate(ledger, ScriptedResponses(_key()))
        later = ScriptedResponses(_key())

        result = await _calibrate(ledger, later, now=NOW + timedelta(days=30))

        assert later.requests == []
        assert (result.state, result.ran, result.finding) == (CalibrationState.PASS, False, None)
        assert result.may_adjudicate is True

    @pytest.mark.asyncio
    async def test_two_concurrent_callers_produce_one_run(self, ledger) -> None:
        gate = asyncio.Event()
        double = ScriptedResponses(_key(), gate=gate)
        calls = [asyncio.create_task(_calibrate(ledger, double)) for _ in range(2)]
        while not any(call.done() for call in calls) and double.in_flight <= CALLS_IN_FLIGHT:
            await asyncio.sleep(0)
        gate.set()

        results = await asyncio.gather(*calls)

        assert len(double.requests) == 176
        assert sorted(result.ran for result in results) == [False, True]
        winner, loser = sorted(results, key=lambda result: not result.ran)
        assert winner.state is CalibrationState.PASS
        assert (loser.state, loser.finding, loser.may_adjudicate) == (
            CalibrationState.RUNNING, None, False,
        )

    @pytest.mark.asyncio
    async def test_a_tampered_eval_set_is_refused_before_any_call_or_write(
        self, ledger, tmp_path: Path,
    ) -> None:
        tampered = tmp_path / 'eval.jsonl'
        tampered.write_bytes(EVAL_SET_PATH.read_bytes().replace(b'"split"', b'"Split"', 1))
        double = ScriptedResponses(_key())

        result = await _calibrate(ledger, double, eval_set_path=tampered)

        assert result.finding is CalibrationFinding.adjudicator_eval_set_invalid
        assert (result.state, result.record, result.ran) == (None, None, False)
        assert result.may_adjudicate is False
        assert double.requests == []
        assert await _stored(ledger) is None

    @pytest.mark.asyncio
    async def test_now_must_be_timezone_aware(self, ledger) -> None:
        with pytest.raises(ValueError, match='now'):
            await _calibrate(ledger, ScriptedResponses(_key()), now=NOW.replace(tzinfo=None))


MINUTE = timedelta(minutes=1)
SEVEN_FALSE_KEEPS = _keeping_foreign_in


def _seven() -> set[str]:
    return {c.case_id for c in _test_cases(CaseKind.contaminated)[:7]}


def _seeded(state: CalibrationState, *, started_at: datetime, **fields) -> CalibrationRecord:
    return CalibrationRecord(
        identity=adjudicator_identity(RUN_SETTINGS), state=state, started_at=started_at,
        model=RUN_SETTINGS.model, endpoint_host=RUN_SETTINGS.endpoint_host,
        **{'attempt': 1, **fields},
    )


async def _seed(ledger: ReconLedgerStore, record: CalibrationRecord) -> None:
    await ledger.upsert(record.to_ledger_record(now=record.started_at))


async def _crashed_run(ledger: ReconLedgerStore, *, started_at: datetime) -> None:
    """The ``running`` row a run leaves behind when its process dies mid-run."""
    stored = await _stored(ledger)
    assert stored is not None
    await _seed(ledger, dataclasses.replace(
        stored, state=CalibrationState.RUNNING, attempt=stored.attempt + 1,
        started_at=started_at, figures=None,
    ))


def _unanswering() -> ScriptedResponses:
    return ScriptedResponses(_key(deviations=_unanswered(MAX_UNANSWERED + 1)))


async def _row(ledger: ReconLedgerStore):
    return await ledger.get_by_identity(
        **adjudicator_calibration_key(adjudicator_identity(RUN_SETTINGS)),
    )


class TestCalibrationLifecycle:
    @pytest.mark.asyncio
    async def test_a_fresh_running_row_is_waited_for(self, ledger) -> None:
        await _seed(ledger, _seeded(
            CalibrationState.RUNNING, started_at=NOW - CRASH_AGE + MINUTE,
        ))
        double = ScriptedResponses(_key())

        result = await _calibrate(ledger, double)

        assert double.requests == []
        assert (result.state, result.ran, result.finding) == (CalibrationState.RUNNING, False, None)
        assert result.may_adjudicate is False

    @pytest.mark.asyncio
    async def test_an_old_running_row_is_marked_crashed_and_the_next_call_runs(
        self, ledger,
    ) -> None:
        first_run = _run(_false_keeps(7))
        seeded = _seeded(
            CalibrationState.RUNNING, started_at=NOW - CRASH_AGE - MINUTE, attempt=2,
            first_run=first_run, figures=first_run,
        )
        await _seed(ledger, seeded)
        double = ScriptedResponses(_key())

        marked = await _calibrate(ledger, double)

        assert double.requests == []
        assert (marked.state, marked.ran, marked.finding) == (CalibrationState.CRASHED, False, None)
        assert marked.record == dataclasses.replace(
            seeded, state=CalibrationState.CRASHED, stall_count=1,
        )
        assert await _stored(ledger) == marked.record

        rerun = await _calibrate(ledger, double, now=NOW + MINUTE)

        assert len(double.requests) == 176
        assert (rerun.state, rerun.ran) == (CalibrationState.PASS, True)
        assert rerun.record is not None
        assert (rerun.record.attempt, rerun.record.stall_count) == (3, 0)
        assert rerun.record.figures is not None
        pooled = _figure(rerun.record.figures, BoundName.contaminated_false_keep)
        assert (pooled.numerator, pooled.denominator) == (7, 90)

    @pytest.mark.asyncio
    async def test_three_runs_ending_inconclusive_or_crashed_stall_the_calibration(
        self, ledger,
    ) -> None:
        first = await _calibrate(ledger, _unanswering())
        assert first.state is CalibrationState.INCONCLUSIVE
        assert first.record is not None
        assert first.record.stall_count == 1
        await _crashed_run(ledger, started_at=NOW + MINUTE)
        crashed = await _calibrate(ledger, _unanswering(), now=NOW + 2 * MINUTE + CRASH_AGE)
        assert crashed.state is CalibrationState.CRASHED
        assert crashed.record is not None
        assert crashed.record.stall_count == 2
        assert crashed.finding is None
        stalled_at = NOW + 3 * MINUTE + CRASH_AGE

        stalled = await _calibrate(ledger, _unanswering(), now=stalled_at)

        assert STALL_RUN_CAP == 3
        assert (stalled.state, stalled.ran) == (CalibrationState.STALLED, True)
        assert stalled.finding is CalibrationFinding.adjudicator_calibration_stalled
        assert stalled.record is not None
        assert (stalled.record.stall_count, stalled.record.stalled_at) == (3, stalled_at)
        assert await _stored(ledger) == stalled.record

        waiting = ScriptedResponses(_key())
        backing_off = await _calibrate(
            ledger, waiting, now=stalled_at + STALL_BACKOFF - timedelta(hours=1),
        )

        assert waiting.requests == []
        assert (backing_off.state, backing_off.ran, backing_off.finding) == (
            CalibrationState.STALLED, False, None,
        )

        restarted = await _calibrate(ledger, _unanswering(), now=stalled_at + STALL_BACKOFF)

        assert (restarted.state, restarted.ran, restarted.finding) == (
            CalibrationState.INCONCLUSIVE, True, None,
        )
        assert restarted.record is not None
        assert (restarted.record.stall_count, restarted.record.stalled_at) == (1, None)

    @pytest.mark.asyncio
    async def test_a_crash_mark_reaching_the_cap_stalls_the_calibration(self, ledger) -> None:
        await _calibrate(ledger, _unanswering())
        await _calibrate(ledger, _unanswering(), now=NOW + MINUTE)
        await _crashed_run(ledger, started_at=NOW + 2 * MINUTE)
        marked_at = NOW + 3 * MINUTE + CRASH_AGE
        double = ScriptedResponses(_key())

        stalled = await _calibrate(ledger, double, now=marked_at)

        assert double.requests == []
        assert (stalled.state, stalled.ran) == (CalibrationState.STALLED, False)
        assert stalled.finding is CalibrationFinding.adjudicator_calibration_stalled
        assert stalled.record is not None
        assert (stalled.record.stall_count, stalled.record.stalled_at) == (
            STALL_RUN_CAP, marked_at,
        )

    @pytest.mark.parametrize(
        ('answer', 'state'),
        [
            (expected_reply, CalibrationState.PASS),
            (_keeping_foreign, CalibrationState.FAIL_FIRST_RUN),
        ],
        ids=['pass', 'fail_first_run'],
    )
    @pytest.mark.asyncio
    async def test_a_scored_run_resets_the_stall_count(
        self, ledger, answer, state: CalibrationState,
    ) -> None:
        await _calibrate(ledger, _unanswering())
        await _calibrate(ledger, _unanswering(), now=NOW + MINUTE)

        scored = await _calibrate(
            ledger, ScriptedResponses(_key(answer=answer)), now=NOW + 2 * MINUTE,
        )

        assert scored.state is state
        assert scored.record is not None
        assert scored.record.stall_count == 0

    @pytest.mark.asyncio
    async def test_an_inconclusive_second_run_keeps_the_first_run_for_pooling(
        self, ledger,
    ) -> None:
        first = await _calibrate(ledger, ScriptedResponses(_key(answer=SEVEN_FALSE_KEEPS(_seven()))))
        assert first.state is CalibrationState.FAIL_FIRST_RUN
        assert first.record is not None

        second = await _calibrate(ledger, _unanswering(), now=NOW + MINUTE)

        assert second.state is CalibrationState.INCONCLUSIVE
        assert second.record is not None
        assert second.record.first_run == first.record.first_run

        third = await _calibrate(ledger, ScriptedResponses(_key()), now=NOW + 2 * MINUTE)

        assert third.state is CalibrationState.PASS
        assert third.record is not None
        assert third.record.attempt == 3
        assert third.record.figures is not None
        pooled = _figure(third.record.figures, BoundName.contaminated_false_keep)
        assert (pooled.numerator, pooled.denominator) == (7, 90)


async def _pooled_pass(ledger: ReconLedgerStore) -> CalibrationRecord:
    """A ``pass`` reached by pooling, so it still carries the first run."""
    await _calibrate(ledger, ScriptedResponses(_key(answer=SEVEN_FALSE_KEEPS(_seven()))))
    passed = await _calibrate(ledger, ScriptedResponses(_key()), now=NOW + MINUTE)
    assert passed.state is CalibrationState.PASS
    assert passed.record is not None
    assert passed.record.first_run is not None
    return passed.record


class TestDrift:
    @pytest.mark.asyncio
    async def test_a_new_resolved_model_marks_a_pass_row_stale(self, ledger) -> None:
        record = await _pooled_pass(ledger)

        drifted = await mark_stale_if_drifted(
            ledger, record, 'gpt-6.1-sol-2027-01-01', now=NOW + timedelta(days=1),
        )

        assert drifted is True
        assert await _stored(ledger) == dataclasses.replace(record, state=CalibrationState.STALE)

    @pytest.mark.parametrize('resolved_model', [RESOLVED_MODEL, None], ids=['known', 'absent'])
    @pytest.mark.asyncio
    async def test_a_known_or_absent_model_writes_nothing(
        self, ledger, resolved_model: str | None,
    ) -> None:
        record = await _pooled_pass(ledger)
        before = await _row(ledger)

        drifted = await mark_stale_if_drifted(
            ledger, record, resolved_model, now=NOW + timedelta(days=1),
        )

        assert drifted is False
        assert await _row(ledger) == before

    @pytest.mark.asyncio
    async def test_a_record_that_is_not_a_pass_writes_nothing(self, ledger) -> None:
        first = await _calibrate(ledger, ScriptedResponses(_key(answer=_keeping_foreign)))
        assert first.record is not None
        before = await _row(ledger)

        drifted = await mark_stale_if_drifted(
            ledger, first.record, 'gpt-6.1-sol-2027-01-01', now=NOW + timedelta(days=1),
        )

        assert drifted is False
        assert await _row(ledger) == before

    @pytest.mark.parametrize(
        ('false_keeps', 'state'),
        [(0, CalibrationState.PASS), (7, CalibrationState.FAIL_FIRST_RUN)],
        ids=['clean', 'seven_false_keeps'],
    )
    @pytest.mark.asyncio
    async def test_after_stale_a_fresh_first_run_is_scored(
        self, ledger, false_keeps: int, state: CalibrationState,
    ) -> None:
        record = await _pooled_pass(ledger)
        later = NOW + timedelta(days=1)
        assert await mark_stale_if_drifted(ledger, record, 'gpt-6.1-sol-2027-01-01', now=later)
        seven = set(list(_seven())[:false_keeps])

        rerun = await _calibrate(
            ledger, ScriptedResponses(_key(answer=SEVEN_FALSE_KEEPS(seven))), now=later + MINUTE,
        )

        assert (rerun.state, rerun.ran) == (state, True)
        assert rerun.record is not None
        assert rerun.record.attempt == 3
        assert rerun.record.figures is not None
        figure = _figure(rerun.record.figures, BoundName.contaminated_false_keep)
        assert (figure.numerator, figure.denominator) == (false_keeps, 45)


async def _seed_under_other_rules(ledger: ReconLedgerStore, record: CalibrationRecord) -> None:
    await ledger.upsert(_under_other_rules(record.to_ledger_record(now=record.started_at)))


def _scored_last_month(state: CalibrationState, **fields) -> CalibrationRecord:
    return _seeded(state, started_at=NOW - timedelta(days=30), attempt=2, **fields)


class TestScoringRulesChange:
    @pytest.mark.parametrize(
        'state', [CalibrationState.PASS, CalibrationState.FAIL, CalibrationState.FAIL_FIRST_RUN],
    )
    @pytest.mark.asyncio
    async def test_a_row_scored_under_other_rules_is_scored_afresh(
        self, ledger, state: CalibrationState,
    ) -> None:
        figures = _run(_false_keeps(7))
        await _seed_under_other_rules(
            ledger, _scored_last_month(state, first_run=figures, figures=figures),
        )
        double = ScriptedResponses(_key())

        result = await _calibrate(ledger, double)

        assert (result.state, result.ran) == (CalibrationState.PASS, True)
        assert len(double.requests) == 176
        record = result.record
        assert record is not None
        assert record.attempt == 3
        assert record.scoring_fingerprint == SCORING_FINGERPRINT
        assert record.first_run is None
        assert record.figures is not None
        invalid = _figure(record.figures, BoundName.invalid_replies)
        assert (invalid.numerator, invalid.denominator) == (0, 176)
        assert await _stored(ledger) == record

    @pytest.mark.asyncio
    async def test_a_first_run_under_other_rules_is_never_pooled(self, ledger) -> None:
        figures = _run(_false_keeps(7))
        await _seed_under_other_rules(ledger, _scored_last_month(
            CalibrationState.FAIL_FIRST_RUN, first_run=figures, figures=figures,
        ))

        result = await _calibrate(ledger, ScriptedResponses(_key(answer=SEVEN_FALSE_KEEPS(_seven()))))

        assert (result.state, result.ran, result.finding) == (
            CalibrationState.FAIL_FIRST_RUN, True, None,
        )
        assert result.record is not None
        assert result.record.figures is not None
        figure = _figure(result.record.figures, BoundName.contaminated_false_keep)
        assert (figure.numerator, figure.denominator) == (7, 45)

    @pytest.mark.asyncio
    async def test_a_fresh_running_row_under_other_rules_is_waited_for(self, ledger) -> None:
        await _seed_under_other_rules(ledger, _seeded(
            CalibrationState.RUNNING, started_at=NOW - CRASH_AGE + MINUTE, attempt=2,
            first_run=_run(_false_keeps(7)),
        ))
        double = ScriptedResponses(_key())

        result = await _calibrate(ledger, double)

        assert double.requests == []
        assert (result.state, result.ran, result.may_adjudicate) == (
            CalibrationState.RUNNING, False, False,
        )
