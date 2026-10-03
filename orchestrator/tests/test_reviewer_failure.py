"""Reviewer failure classification: what killed a reviewer that left no verdict.

Pinned on real ``AgentResult`` values through the module's public API only.
"""

from __future__ import annotations

import json

import pytest
from shared.cli_invoke import AgentResult

from orchestrator.reviewer_failure import (
    ReviewerFailure,
    ReviewerFailureKind,
    classify_reviewer_exception,
    classify_reviewer_result,
    error_review,
    should_retry_reviewer,
    triage_reviewer_errors,
)
from orchestrator.scheduler import is_transient_api_requeue

ROLE = 'reviewer_comprehensive'


def _zero_work(*, transcript_turns: int | None) -> AgentResult:
    return AgentResult(
        success=False,
        output='Agent produced no output',
        subtype='error_empty_output',
        transcript_turns=transcript_turns,
    )


class TestClassifyReviewerResult:
    def test_fast_5xx_is_server_error(self):
        failure = classify_reviewer_result(
            AgentResult(success=False, output='API Error: 529', api_error_status=529)
        )
        assert failure.kind is ReviewerFailureKind.SERVER_ERROR
        assert failure.api_error_status == 529
        assert 'agent API error: HTTP 529' in failure.detail

    def test_turn_cap(self):
        failure = classify_reviewer_result(
            AgentResult(success=False, output='', subtype='error_max_turns', turns=31)
        )
        assert failure.kind is ReviewerFailureKind.TURN_CAP

    def test_budget_cap_names_the_cost(self):
        failure = classify_reviewer_result(
            AgentResult(success=False, output='', subtype='error_max_budget_usd', cost_usd=5.07)
        )
        assert failure.kind is ReviewerFailureKind.BUDGET_CAP
        assert '5.07' in failure.detail

    def test_zero_work(self):
        failure = classify_reviewer_result(_zero_work(transcript_turns=0))
        assert failure.kind is ReviewerFailureKind.ZERO_WORK

    def test_empty_output_after_productive_turns_is_not_zero_work(self):
        failure = classify_reviewer_result(_zero_work(transcript_turns=7))
        assert failure.kind is ReviewerFailureKind.AGENT_FAILURE

    def test_productive_wall_clock_kill(self):
        failure = classify_reviewer_result(
            AgentResult(success=False, output='', timed_out=True, transcript_turns=12)
        )
        assert failure.kind is ReviewerFailureKind.TIMED_OUT

    def test_completed_run_without_verdict(self):
        failure = classify_reviewer_result(AgentResult(success=True, output='looks fine'))
        assert failure.kind is ReviewerFailureKind.NO_VERDICT

    def test_other_agent_failure(self):
        failure = classify_reviewer_result(
            AgentResult(success=False, output='Connection refused')
        )
        assert failure.kind is ReviewerFailureKind.AGENT_FAILURE

    def test_api_refusal_never_rides_the_requeue_lane(self):
        failure = classify_reviewer_result(
            AgentResult(success=False, output='', stop_reason='refusal')
        )
        assert failure.kind is ReviewerFailureKind.AGENT_FAILURE
        assert not failure.requeue_eligible
        assert 'refusal' in failure.detail


class TestClassificationPrecedence:
    def test_budget_cap_beats_incidental_5xx(self):
        failure = classify_reviewer_result(
            AgentResult(
                success=False,
                output='',
                subtype='error_max_budget_usd',
                cost_usd=5.07,
                api_error_status=529,
            )
        )
        assert failure.kind is ReviewerFailureKind.BUDGET_CAP

    def test_turn_cap_beats_incidental_5xx(self):
        failure = classify_reviewer_result(
            AgentResult(
                success=False,
                output='',
                subtype='error_max_turns',
                turns=31,
                api_error_status=503,
            )
        )
        assert failure.kind is ReviewerFailureKind.TURN_CAP

    def test_productive_kill_beats_incidental_5xx(self):
        failure = classify_reviewer_result(
            AgentResult(
                success=False,
                output='',
                timed_out=True,
                transcript_turns=12,
                api_error_status=529,
            )
        )
        assert failure.kind is ReviewerFailureKind.TIMED_OUT


def test_exception_names_its_type_and_message():
    failure = classify_reviewer_exception(RuntimeError('transport closed'))
    assert failure.kind is ReviewerFailureKind.EXCEPTION
    assert 'RuntimeError' in failure.detail
    assert 'transport closed' in failure.detail


class TestKindPolicy:
    REQUEUE_ELIGIBLE = {ReviewerFailureKind.SERVER_ERROR, ReviewerFailureKind.ZERO_WORK}

    @pytest.mark.parametrize('kind', list(ReviewerFailureKind))
    def test_requeue_eligible_only_for_transient_kinds(self, kind: ReviewerFailureKind):
        failure = ReviewerFailure(kind, 'd')
        assert failure.requeue_eligible is (kind in self.REQUEUE_ELIGIBLE)

    @pytest.mark.parametrize('kind', list(ReviewerFailureKind))
    def test_only_server_errors_skip_the_in_phase_retry(self, kind: ReviewerFailureKind):
        failure = ReviewerFailure(kind, 'd')
        assert failure.retry_in_phase is (kind is not ReviewerFailureKind.SERVER_ERROR)

    @pytest.mark.parametrize('kind', list(ReviewerFailureKind))
    def test_label_names_a_cause_and_never_says_infrastructure(
        self, kind: ReviewerFailureKind
    ):
        label = ReviewerFailure(kind, 'd').label
        assert label
        assert 'infrastructure' not in label.lower()


def _one_failure_per_kind() -> list[ReviewerFailure]:
    return [
        ReviewerFailure(
            kind,
            f'detail for {kind.value}',
            api_error_status=529 if kind is ReviewerFailureKind.SERVER_ERROR else None,
        )
        for kind in ReviewerFailureKind
    ]


class TestErrorReviewRecord:
    @pytest.mark.parametrize('failure', _one_failure_per_kind(), ids=lambda f: f.kind.value)
    def test_round_trip(self, failure: ReviewerFailure):
        review = error_review(ROLE, failure, summary='s')
        assert review['verdict'] == 'ERROR'
        assert review['issues'] == []
        assert review['reviewer'] == ROLE
        assert review['summary'] == 's'
        json.dumps(review)
        assert ReviewerFailure.from_review(review) == failure

    def test_legacy_error_dict_without_record_is_unclassified(self):
        legacy = {'reviewer': ROLE, 'verdict': 'ERROR', 'issues': [], 'summary': 'old'}
        failure = ReviewerFailure.from_review(legacy)
        assert failure.kind is ReviewerFailureKind.UNCLASSIFIED
        assert 'old' in failure.detail

    def test_unknown_kind_string_is_unclassified(self):
        review = error_review(ROLE, ReviewerFailure(ReviewerFailureKind.TURN_CAP, 'd'), summary='s')
        review['failure']['kind'] = 'cosmic_ray'
        assert ReviewerFailure.from_review(review).kind is ReviewerFailureKind.UNCLASSIFIED

    @pytest.mark.parametrize('status', ['529', True, 529.0])
    def test_non_int_status_is_unclassified(self, status: object):
        review = error_review(
            ROLE,
            ReviewerFailure(ReviewerFailureKind.SERVER_ERROR, 'd', api_error_status=529),
            summary='s',
        )
        review['failure']['api_error_status'] = status
        assert ReviewerFailure.from_review(review).kind is ReviewerFailureKind.UNCLASSIFIED


class TestShouldRetryReviewer:
    def test_exception_retries(self):
        assert should_retry_reviewer(RuntimeError('boom'))

    def test_server_error_skips_the_retry_burn(self):
        failure = ReviewerFailure(ReviewerFailureKind.SERVER_ERROR, 'd', api_error_status=529)
        assert not should_retry_reviewer(error_review(ROLE, failure, summary='s'))

    @pytest.mark.parametrize(
        'kind',
        [
            ReviewerFailureKind.ZERO_WORK,
            ReviewerFailureKind.TURN_CAP,
            ReviewerFailureKind.BUDGET_CAP,
            ReviewerFailureKind.NO_VERDICT,
        ],
    )
    def test_other_errors_retry(self, kind: ReviewerFailureKind):
        assert should_retry_reviewer(error_review(ROLE, ReviewerFailure(kind, 'd'), summary='s'))

    def test_legacy_error_dict_retries(self):
        assert should_retry_reviewer({'reviewer': ROLE, 'verdict': 'ERROR', 'summary': 'old'})

    @pytest.mark.parametrize('verdict', ['PASS', 'ISSUES_FOUND'])
    def test_verdict_payload_does_not_retry(self, verdict: str):
        assert not should_retry_reviewer(
            {'reviewer': ROLE, 'verdict': verdict, 'issues': [], 'summary': 'ok'}
        )


def _errored(role: str, result: AgentResult) -> dict:
    return error_review(role, classify_reviewer_result(result), summary='s')


_SERVER_529 = AgentResult(success=False, output='API Error: 529', api_error_status=529)
_SERVER_503 = AgentResult(success=False, output='API Error: 503', api_error_status=503)
_TURN_CAP = AgentResult(success=False, output='', subtype='error_max_turns', turns=31)
_BUDGET_CAP = AgentResult(
    success=False, output='', subtype='error_max_budget_usd', turns=24, cost_usd=5.07,
)
_NO_VERDICT = AgentResult(success=True, output='looks fine')


def _triage(results: dict[str, AgentResult]):
    return triage_reviewer_errors(
        {role: _errored(role, result) for role, result in results.items()},
    )


class TestTriageRequeues:
    def test_all_5xx_rides_the_transient_lane(self):
        triage = _triage({ROLE: _SERVER_529})
        assert triage.requeue
        assert triage.api_error_status == 529
        assert 'agent API error: HTTP 529' in triage.reason
        assert is_transient_api_requeue(triage.reason, api_error_status=triage.api_error_status)

    def test_two_5xx_reviewers_carry_the_first_name_s_status(self):
        reviewer_a = _errored('reviewer_a', _SERVER_529)
        reviewer_b = _errored('reviewer_b', _SERVER_503)
        a_first = triage_reviewer_errors({'reviewer_a': reviewer_a, 'reviewer_b': reviewer_b})
        b_first = triage_reviewer_errors({'reviewer_b': reviewer_b, 'reviewer_a': reviewer_a})
        assert a_first.requeue
        assert a_first.api_error_status == 529
        assert a_first == b_first

    def test_all_zero_work_rides_the_genuine_lane(self):
        triage = _triage({ROLE: _zero_work(transcript_turns=0)})
        assert triage.requeue
        assert triage.api_error_status is None
        assert not is_transient_api_requeue(triage.reason, api_error_status=None)

    def test_5xx_plus_zero_work_carries_the_5xx_status(self):
        triage = _triage({
            'reviewer_a': _SERVER_529,
            'reviewer_b': _zero_work(transcript_turns=0),
        })
        assert triage.requeue
        assert triage.api_error_status == 529


class TestTriageBlocks:
    def test_5xx_mixed_with_a_quality_error_blocks(self):
        triage = _triage({'reviewer_a': _SERVER_529, 'reviewer_b': _NO_VERDICT})
        assert not triage.requeue

    def test_turn_cap_is_named_up_front_and_never_called_infrastructure(self):
        triage = _triage({ROLE: _TURN_CAP})
        assert not triage.requeue
        assert 'turn cap' in triage.reason
        assert 'infrastructure' not in triage.reason.lower()
        assert ReviewerFailure(ReviewerFailureKind.TURN_CAP, 'd').label in triage.reason[:200]

    def test_budget_cap_is_named_and_never_called_infrastructure(self):
        triage = _triage({ROLE: _BUDGET_CAP})
        assert not triage.requeue
        assert 'budget' in triage.reason
        assert 'infrastructure' not in triage.reason.lower()

    def test_legacy_error_dict_without_record_blocks(self):
        legacy = {'reviewer': ROLE, 'verdict': 'ERROR', 'issues': [], 'summary': 'old'}
        triage = triage_reviewer_errors({ROLE: legacy})
        assert not triage.requeue


def test_detail_names_every_reviewer_kind_and_status():
    triage = _triage({'reviewer_a': _SERVER_529, 'reviewer_b': _TURN_CAP})
    for fragment in (
        'reviewer_a', 'server_error', 'api_error_status=529',
        'reviewer_b', 'turn_cap', 'api_error_status=None',
    ):
        assert fragment in triage.detail


def test_empty_panel_is_rejected_rather_than_read_as_all_transient():
    with pytest.raises(ValueError, match='errored_reviews'):
        triage_reviewer_errors({})
