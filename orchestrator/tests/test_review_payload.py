"""The review-payload normaliser: one reduction of a reviewer artifact to {verdict, issues}.

PRD plans/review-all-s18-evals-measurement-prd.md §5 C3, §6 B4 (task 6739).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.evals import compare
from orchestrator.evals.metrics import EvalMetrics, compute_composite
from orchestrator.evals.results import EvalResult
from orchestrator.evals.scoring import quality_from_review_artifact
from orchestrator.mcp import verdict_tools as _verdict_tools
from orchestrator.review_payload import (
    ReviewPayload,
    ReviewPayloadError,
    Verdict,
    normalize_review,
)

_REVIEWER = 'reviewer_comprehensive'

_BLOCKING = {'severity': 'blocking', 'description': 'b1', 'location': 'm/x.py:1'}
_SUGGESTION_1 = {'severity': 'suggestion', 'description': 's1', 'location': 'm/x.py:2'}
_SUGGESTION_2 = {'severity': 'suggestion', 'description': 's2', 'location': 'm/x.py:3'}


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

    def test_error_payload_drops_its_issues_and_keeps_its_identity(self):
        payload = normalize_review(_payload('ERROR'))

        assert payload.verdict is Verdict.ERROR
        assert payload.issues == ()
        assert payload.reviewer == _REVIEWER
        assert payload.summary == 'one blocking, two suggestions'

    def test_every_non_blocking_issue_is_a_suggestion(self):
        nit = {'severity': 'nit', 'description': 'n'}
        unrated = {'description': 'u'}
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

    @pytest.mark.parametrize(
        ('artifact', 'named'),
        [
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
        ],
    )
    def test_malformed_artifact_raises_naming_the_offending_value(
        self, artifact: dict, named: str,
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
