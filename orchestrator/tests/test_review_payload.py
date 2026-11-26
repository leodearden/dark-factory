"""The review-payload normaliser: one reduction of a reviewer artifact to {verdict, issues}.

PRD plans/review-all-s18-evals-measurement-prd.md §5 C3, §6 B4 (task 6739).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from _defect_ruling_fixtures import (
    RULED_REF,
    SECOND_RULED_REF,
    contract_ruling,
    contract_ruling_without,
    parse_ref_message,
)
from escalation.disposition import ReviewerDisposition
from shared.defects import RulingSubject

from orchestrator.artifacts import TaskArtifacts
from orchestrator.evals import compare
from orchestrator.evals.metrics import EvalMetrics, compute_composite
from orchestrator.evals.results import EvalResult
from orchestrator.evals.scoring import quality_from_review_artifact
from orchestrator.mcp import verdict_tools as _verdict_tools
from orchestrator.review_payload import (
    DefectRuling,
    NoteDisposition,
    ReviewPayload,
    ReviewPayloadError,
    Verdict,
    normalize_review,
    parse_defect_rulings,
    parse_note_dispositions,
)

_REVIEWER = 'reviewer_comprehensive'

_BLOCKING = {'kind': 'defect', 'severity': 'blocking', 'description': 'b1', 'location': 'm/x.py:1'}
_SUGGESTION_1 = {'kind': 'defect', 'severity': 'suggestion', 'description': 's1', 'location': 'm/x.py:2'}
_SUGGESTION_2 = {'kind': 'defect', 'severity': 'suggestion', 'description': 's2', 'location': 'm/x.py:3'}


def _payload(verdict: str = 'ISSUES_FOUND', issues: list | None = None) -> dict:
    return {
        'reviewer': _REVIEWER,
        'verdict': verdict,
        'issues': [dict(_BLOCKING), dict(_SUGGESTION_1), dict(_SUGGESTION_2)]
        if issues is None else issues,
        'summary': 'one blocking, two suggestions',
    }


class TestNormalizeReview:
    @pytest.mark.parametrize('verdict', [Verdict.PASS, Verdict.ISSUES_FOUND])
    def test_legacy_payload_normalises_to_a_typed_payload(self, verdict: Verdict):
        payload = normalize_review(_payload(verdict.value))

        assert payload == ReviewPayload(
            verdict=verdict,
            issues=(_BLOCKING, _SUGGESTION_1, _SUGGESTION_2),
            reviewer=_REVIEWER,
            summary='one blocking, two suggestions',
        )
        assert isinstance(payload.issues, tuple)

    def test_legacy_error_payload_normalises_with_no_issues(self):
        assert normalize_review(_payload('ERROR')) == ReviewPayload(
            verdict=Verdict.ERROR,
            issues=(),
            reviewer=_REVIEWER,
            summary='one blocking, two suggestions',
        )

    @pytest.mark.parametrize('verdict', list(Verdict))
    def test_envelope_normalises_like_its_payload(self, verdict: Verdict):
        payload = _payload(verdict.value)
        envelope = _verdict_tools._envelope(_REVIEWER, 'sid', payload)

        assert normalize_review(envelope) == normalize_review(payload)

    def test_every_non_blocking_issue_is_a_suggestion(self):
        nit = {'kind': 'defect', 'severity': 'nit', 'description': 'n'}
        unrated = {'kind': 'defect', 'description': 'u'}
        payload = normalize_review(
            _payload(issues=[dict(_BLOCKING), dict(_SUGGESTION_1), nit, unrated]),
        )

        assert payload.blocking_issues == (_BLOCKING,)
        assert payload.suggestions == (_SUGGESTION_1, nit, unrated)

    def test_missing_or_null_issues_are_empty(self):
        missing = {'reviewer': _REVIEWER, 'verdict': 'PASS'}
        null = {'reviewer': _REVIEWER, 'verdict': 'PASS', 'issues': None}

        assert normalize_review(missing).issues == ()
        assert normalize_review(null).issues == ()

    def test_missing_summary_and_reviewer_take_defaults(self):
        payload = normalize_review({'verdict': 'PASS', 'issues': []})

        assert payload.summary == ''
        assert payload.reviewer is None

    def test_null_summary_is_empty(self):
        payload = normalize_review({'verdict': 'PASS', 'summary': None})

        assert payload.summary == ''

    @pytest.mark.parametrize(
        ('artifact', 'named'),
        [
            pytest.param(['PASS'], 'list', id='list-artifact'),
            pytest.param('PASS', 'str', id='string-artifact'),
            pytest.param(_payload('NEEDS_CHANGES'), 'NEEDS_CHANGES', id='off-contract-verdict'),
            pytest.param({'reviewer': _REVIEWER, 'issues': []}, 'None', id='missing-verdict'),
            pytest.param(
                {'reviewer': _REVIEWER, 'verdict': 'PASS', 'issues': 'lots'},
                'lots',
                id='non-list-issues',
            ),
            pytest.param(
                {'reviewer': _REVIEWER, 'verdict': 'ISSUES_FOUND', 'issues': ['bare string']},
                'str',
                id='non-dict-issue',
            ),
            pytest.param(
                {'reviewer': 42, 'verdict': 'PASS', 'issues': []}, '42', id='non-str-reviewer',
            ),
            pytest.param(
                {'reviewer': _REVIEWER, 'verdict': 'PASS', 'summary': ['x']},
                'list',
                id='non-str-summary',
            ),
            pytest.param(
                {'reviewer': _REVIEWER, 'verdict': 'PASS', 'summary': []},
                'list',
                id='empty-non-str-summary',
            ),
        ],
    )
    def test_malformed_artifact_raises_naming_the_offending_value(
        self, artifact: object, named: str,
    ):
        with pytest.raises(ReviewPayloadError, match=named) as raised:
            normalize_review(artifact)

        assert isinstance(raised.value, ValueError)

    def test_envelope_shape_with_a_non_dict_payload_raises(self):
        artifact = {'role': _REVIEWER, 'schema_version': 1, 'verdict': 'PASS'}

        with pytest.raises(ReviewPayloadError):
            normalize_review(artifact)

    def test_issue_dicts_are_copied(self):
        artifact = _payload()
        payload = normalize_review(artifact)

        artifact['issues'][0]['severity'] = 'suggestion'

        assert payload.issues[0] == _BLOCKING


class TestReviewPayload:
    def test_error_with_issues_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            ReviewPayload(Verdict.ERROR, issues=(dict(_BLOCKING),), reviewer='r')

    @pytest.mark.parametrize('verdict', list(Verdict))
    def test_to_record_round_trips_through_normalize_review(self, verdict: Verdict):
        payload = normalize_review(_payload(verdict.value))
        record = payload.to_record()

        assert normalize_review(record) == payload
        json.dumps(record)
        assert type(record['verdict']) is str
        assert record['verdict'] == verdict.value
        assert isinstance(record['issues'], list)

    def test_is_frozen(self):
        payload = normalize_review(_payload())

        pytest.raises(dataclasses.FrozenInstanceError, setattr, payload, 'verdict', Verdict.PASS)

    def test_is_unhashable_by_name(self):
        payload = normalize_review(_payload())

        with pytest.raises(TypeError, match="unhashable type: 'ReviewPayload'"):
            hash(payload)


_ADDRESSED_RAW = {
    'escalation_id': 'esc-42-1', 'disposition': 'addressed', 'rationale': 'fixed in foo.py',
}
_WORK_RAW = {
    'escalation_id': 'esc-42-2', 'disposition': 'work', 'rationale': 'needs its own task',
}
_ADDRESSED = NoteDisposition('esc-42-1', ReviewerDisposition.ADDRESSED, 'fixed in foo.py')
_WORK = NoteDisposition('esc-42-2', ReviewerDisposition.WORK, 'needs its own task')


def _with_notes(raw: object, verdict: str = 'PASS') -> dict:
    return {**_payload(verdict, issues=[]), 'note_dispositions': raw}


class TestNoteDispositions:
    @pytest.mark.parametrize('shape', ['flat', 'envelope'])
    def test_dispositions_normalise_to_typed_values_in_order(self, shape: str):
        payload = _with_notes([dict(_ADDRESSED_RAW), dict(_WORK_RAW)])
        artifact = (
            payload if shape == 'flat'
            else _verdict_tools._envelope(_REVIEWER, 'sid', payload)
        )

        assert normalize_review(artifact).note_dispositions == (_ADDRESSED, _WORK)

    def test_absent_or_null_dispositions_are_empty(self):
        assert normalize_review(_payload('PASS', issues=[])).note_dispositions == ()
        assert normalize_review(_with_notes(None)).note_dispositions == ()
        assert parse_note_dispositions(None) == ()

    def test_error_verdict_drops_dispositions(self):
        payload = normalize_review(_with_notes([dict(_ADDRESSED_RAW)], verdict='ERROR'))

        assert payload.note_dispositions == ()

    def test_error_payload_with_dispositions_is_rejected_at_construction(self):
        with pytest.raises(ReviewPayloadError):
            ReviewPayload(
                Verdict.ERROR, issues=(), reviewer='r', note_dispositions=(_ADDRESSED,),
            )

    def test_non_list_value_raises_naming_the_value(self):
        with pytest.raises(ReviewPayloadError, match='addressed-everything'):
            normalize_review(_with_notes('addressed-everything'))

    @pytest.mark.parametrize(
        ('entry', 'named'),
        [
            pytest.param('esc-42-2', 'str', id='entry-not-a-mapping'),
            pytest.param(
                {'disposition': 'work', 'rationale': 'r'}, 'escalation_id', id='missing-id',
            ),
            pytest.param(
                {'escalation_id': '  ', 'disposition': 'work', 'rationale': 'r'},
                "'  '",
                id='blank-id',
            ),
            pytest.param(
                {'escalation_id': 7, 'disposition': 'work', 'rationale': 'r'}, '7', id='int-id',
            ),
            pytest.param(
                {'escalation_id': 'esc-42-2', 'disposition': 'fixed', 'rationale': 'r'},
                'fixed',
                id='unknown-disposition',
            ),
            pytest.param(
                {'escalation_id': 'esc-42-2', 'disposition': 'work'}, 'rationale', id='missing-rationale',
            ),
            pytest.param(
                {'escalation_id': 'esc-42-2', 'disposition': 'work', 'rationale': ['r']},
                r"\['r'\]",
                id='non-str-rationale',
            ),
        ],
    )
    def test_malformed_entry_raises_naming_its_index_and_value(self, entry: object, named: str):
        with pytest.raises(ReviewPayloadError, match=named) as raised:
            normalize_review(_with_notes([dict(_ADDRESSED_RAW), entry]))

        assert 'entry 1' in str(raised.value)
        assert _REVIEWER in str(raised.value)

    def test_unknown_disposition_lists_the_valid_values(self):
        entry = {'escalation_id': 'esc-42-2', 'disposition': 'fixed', 'rationale': 'r'}

        with pytest.raises(ReviewPayloadError) as raised:
            parse_note_dispositions([entry], reviewer=_REVIEWER)

        for value in ReviewerDisposition:
            assert repr(value.value) in str(raised.value)

    def test_duplicated_escalation_id_names_both_indices(self):
        duplicate = {**_WORK_RAW, 'escalation_id': 'esc-42-1'}

        with pytest.raises(ReviewPayloadError, match='esc-42-1') as raised:
            parse_note_dispositions(
                [dict(_ADDRESSED_RAW), dict(_WORK_RAW), duplicate], reviewer=_REVIEWER,
            )

        assert 'entries 0 and 2' in str(raised.value)

    def test_to_record_carries_dispositions_and_round_trips(self):
        payload = normalize_review(_with_notes([dict(_ADDRESSED_RAW), dict(_WORK_RAW)]))
        record = payload.to_record()

        assert record['note_dispositions'] == [_ADDRESSED_RAW, _WORK_RAW]
        assert type(record['note_dispositions'][0]['disposition']) is str
        json.dumps(record)
        assert normalize_review(record) == payload

    def test_to_record_without_dispositions_keeps_the_four_keys(self):
        record = normalize_review(_payload('PASS', issues=[])).to_record()

        assert set(record) == {'reviewer', 'verdict', 'issues', 'summary'}

    def test_note_disposition_is_frozen(self):
        pytest.raises(
            dataclasses.FrozenInstanceError, setattr, _ADDRESSED, 'rationale', 'other',
        )


_CLOSURE_RAW = contract_ruling()
_RETIREMENT_RAW = contract_ruling(
    ref=SECOND_RULED_REF, subject=RulingSubject.RETIREMENT.value, accepted=False,
)
_CLOSURE = DefectRuling(RULED_REF, RulingSubject.CLOSURE, True)
_RETIREMENT = DefectRuling(SECOND_RULED_REF, RulingSubject.RETIREMENT, False)


def _with_rulings(raw: object, verdict: str = 'PASS') -> dict:
    return {**_payload(verdict, issues=[]), 'defect_rulings': raw}


class TestDefectRulings:
    @pytest.mark.parametrize('shape', ['flat', 'envelope'])
    def test_rulings_normalise_to_typed_values_in_order(self, shape: str):
        payload = _with_rulings([dict(_CLOSURE_RAW), dict(_RETIREMENT_RAW)])
        artifact = (
            payload if shape == 'flat'
            else _verdict_tools._envelope(_REVIEWER, 'sid', payload)
        )

        assert normalize_review(artifact).defect_rulings == (_CLOSURE, _RETIREMENT)

    def test_the_parser_reads_the_wire_shape_in_order(self):
        assert parse_defect_rulings(
            [dict(_CLOSURE_RAW), dict(_RETIREMENT_RAW)], reviewer=_REVIEWER,
        ) == (_CLOSURE, _RETIREMENT)

    def test_absent_or_null_rulings_are_empty(self):
        assert normalize_review(_payload('PASS', issues=[])).defect_rulings == ()
        assert normalize_review(_with_rulings(None)).defect_rulings == ()
        assert parse_defect_rulings(None) == ()

    def test_error_verdict_drops_rulings(self):
        payload = normalize_review(_with_rulings([dict(_CLOSURE_RAW)], verdict='ERROR'))

        assert payload.defect_rulings == ()

    def test_error_payload_with_rulings_is_rejected_at_construction(self):
        with pytest.raises(ReviewPayloadError):
            ReviewPayload(Verdict.ERROR, issues=(), reviewer='r', defect_rulings=(_CLOSURE,))

    def test_non_list_value_raises_naming_the_value(self):
        with pytest.raises(ReviewPayloadError, match='closure-everything'):
            normalize_review(_with_rulings('closure-everything'))

    @pytest.mark.parametrize(
        ('entry', 'named'),
        [
            pytest.param(RULED_REF, 'str', id='entry-not-a-mapping'),
            pytest.param(contract_ruling_without('ref'), 'None', id='missing-ref'),
            pytest.param(
                contract_ruling(ref='not-a-ref'), 'not-a-ref', id='malformed-ref',
            ),
            pytest.param(contract_ruling(subject='closed'), 'closed', id='unknown-subject'),
            pytest.param(
                contract_ruling_without('accepted'), 'accepted', id='missing-accepted',
            ),
            pytest.param(contract_ruling(accepted=1), '1', id='int-accepted'),
            pytest.param(contract_ruling(accepted='true'), "'true'", id='str-accepted'),
        ],
    )
    def test_malformed_entry_raises_naming_its_index_and_value(self, entry: object, named: str):
        with pytest.raises(ReviewPayloadError, match=named) as raised:
            normalize_review(_with_rulings([dict(_RETIREMENT_RAW), entry]))

        assert 'defect_rulings entry 1' in str(raised.value)
        assert _REVIEWER in str(raised.value)

    @pytest.mark.parametrize('ref', [None, 'not-a-ref', 'review:7048'])
    def test_a_bad_ref_is_refused_in_parse_refs_own_words(self, ref: object):
        with pytest.raises(ReviewPayloadError) as raised:
            parse_defect_rulings([contract_ruling(ref=ref)], reviewer=_REVIEWER)

        assert parse_ref_message(ref) in str(raised.value)

    def test_unknown_subject_lists_the_valid_values(self):
        with pytest.raises(ReviewPayloadError) as raised:
            parse_defect_rulings([contract_ruling(subject='closed')], reviewer=_REVIEWER)

        for subject in RulingSubject:
            assert repr(subject.value) in str(raised.value)

    def test_a_duplicated_ref_and_subject_names_both_indices(self):
        duplicate = contract_ruling(accepted=False)

        with pytest.raises(ReviewPayloadError, match=RULED_REF) as raised:
            parse_defect_rulings(
                [dict(_CLOSURE_RAW), dict(_RETIREMENT_RAW), duplicate], reviewer=_REVIEWER,
            )

        assert 'entries 0 and 2' in str(raised.value)

    def test_one_ref_may_be_ruled_under_each_subject(self):
        retirement = contract_ruling(subject=RulingSubject.RETIREMENT.value)

        assert parse_defect_rulings([dict(_CLOSURE_RAW), retirement]) == (
            _CLOSURE, DefectRuling(RULED_REF, RulingSubject.RETIREMENT, True),
        )

    def test_to_record_carries_rulings_and_round_trips(self):
        payload = normalize_review(_with_rulings([dict(_CLOSURE_RAW), dict(_RETIREMENT_RAW)]))
        record = payload.to_record()

        assert record['defect_rulings'] == [_CLOSURE_RAW, _RETIREMENT_RAW]
        assert type(record['defect_rulings'][0]['subject']) is str
        json.dumps(record)
        assert normalize_review(record) == payload

    def test_to_record_without_rulings_has_no_rulings_key(self):
        record = normalize_review(_payload('PASS', issues=[])).to_record()

        assert 'defect_rulings' not in record

    def test_defect_ruling_is_frozen(self):
        pytest.raises(dataclasses.FrozenInstanceError, setattr, _CLOSURE, 'accepted', False)


def _task_artifacts(root: Path) -> TaskArtifacts:
    artifacts = TaskArtifacts(root)
    artifacts.init('df_task', 'T', 'd')
    return artifacts


def _eval_score(worktree: Path) -> float:
    result = EvalResult('df_task', 'cfg', 'done', {'plan_steps': 5}, str(worktree))
    return compare.score_result_from_artifacts(result)


@pytest.mark.parametrize('verdict', list(Verdict))
def test_b4_envelope_and_legacy_payload_agree_in_production_and_evals(
    tmp_path: Path, verdict: Verdict,
):
    payload = _payload(verdict.value)
    envelope = _verdict_tools._envelope(_REVIEWER, 'sid-b4', payload)
    expected = (0, 0, [_REVIEWER]) if verdict is Verdict.ERROR else (1, 2, [])

    for shape, artifact in (('legacy', payload), ('envelope', envelope)):
        production = _task_artifacts(tmp_path / shape)
        production.write_review(_REVIEWER, artifact)
        aggregation = production.aggregate_reviews()
        assert (
            len(aggregation.blocking_issues),
            len(aggregation.suggestions),
            aggregation.reviewer_errors,
        ) == expected, shape

    blocking, suggestions, _ = expected
    score = compute_composite(EvalMetrics(
        tests_pass=True,
        plan_steps=5,
        review_blocking_issues=blocking,
        review_suggestions=suggestions,
    ))
    assert quality_from_review_artifact(payload, plan_steps=5) == score
    assert quality_from_review_artifact(envelope, plan_steps=5) == score

    _task_artifacts(tmp_path / 'wt_legacy').write_review(_REVIEWER, payload)
    _task_artifacts(tmp_path / 'wt_envelope').write_verdict(_REVIEWER, envelope)
    assert _eval_score(tmp_path / 'wt_legacy') == score
    assert _eval_score(tmp_path / 'wt_envelope') == score
