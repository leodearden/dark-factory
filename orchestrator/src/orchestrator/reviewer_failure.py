"""Why a reviewer produced no usable verdict, and what the review arm does about it.

Each errored reviewer is classified once, from its ``AgentResult`` or the
exception its invocation raised. The classification is persisted as a
structured ``failure`` record on the reviewer's ERROR review dict, so the
on-disk mirror names the cause and the panel triage reads it back without
re-parsing prose. Per-kind policy (label, requeue eligibility, in-phase retry)
lives in one table. The 5xx half follows
plans/server-side-api-error-handling-prd.md decision 8; the zero-work and cap
halves are task 3321.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.cli_invoke import (
    AgentFailureKind,
    AgentResult,
    classify_agent_failure,
    is_server_error_status,
)

from orchestrator.zero_work_failure import is_zero_work_failure

# CLI wire value for a run stopped by --max-budget-usd. Task 5521 owns
# consolidating the per-package copies of this spelling into shared.
BUDGET_CAP_SUBTYPE = 'error_max_budget_usd'


class ReviewerFailureKind(enum.StrEnum):
    SERVER_ERROR = 'server_error'
    ZERO_WORK = 'zero_work'
    TURN_CAP = 'turn_cap'
    BUDGET_CAP = 'budget_cap'
    TIMED_OUT = 'timed_out'
    NO_VERDICT = 'no_verdict'
    AGENT_FAILURE = 'agent_failure'
    EXCEPTION = 'exception'
    UNCLASSIFIED = 'unclassified'


@dataclass(frozen=True)
class _KindPolicy:
    label: str
    requeue_eligible: bool
    retry_in_phase: bool


_POLICY: Mapping[ReviewerFailureKind, _KindPolicy] = MappingProxyType({
    ReviewerFailureKind.SERVER_ERROR: _KindPolicy(
        'provider server error (5xx)', requeue_eligible=True, retry_in_phase=False,
    ),
    ReviewerFailureKind.ZERO_WORK: _KindPolicy(
        'zero-work failure (no model turn ran)', requeue_eligible=True, retry_in_phase=True,
    ),
    ReviewerFailureKind.TURN_CAP: _KindPolicy(
        'turn cap (max_turns) exhausted', requeue_eligible=False, retry_in_phase=True,
    ),
    ReviewerFailureKind.BUDGET_CAP: _KindPolicy(
        'dollar-budget cap (max_budget_usd) exhausted', requeue_eligible=False, retry_in_phase=True,
    ),
    ReviewerFailureKind.TIMED_OUT: _KindPolicy(
        'wall-clock timeout', requeue_eligible=False, retry_in_phase=True,
    ),
    ReviewerFailureKind.NO_VERDICT: _KindPolicy(
        'completed without a usable verdict', requeue_eligible=False, retry_in_phase=True,
    ),
    ReviewerFailureKind.AGENT_FAILURE: _KindPolicy(
        'agent run failed', requeue_eligible=False, retry_in_phase=True,
    ),
    ReviewerFailureKind.EXCEPTION: _KindPolicy(
        'reviewer invocation raised', requeue_eligible=False, retry_in_phase=True,
    ),
    ReviewerFailureKind.UNCLASSIFIED: _KindPolicy(
        'unclassified reviewer error', requeue_eligible=False, retry_in_phase=True,
    ),
})


def _require_complete_policy() -> None:
    missing = sorted(set(ReviewerFailureKind) - _POLICY.keys())
    if missing:
        raise RuntimeError(f'reviewer failure policy table has no row for: {missing}')


_require_complete_policy()

_NO_RECORD_DETAIL = 'no failure record on the review'


@dataclass(frozen=True)
class ReviewerFailure:
    kind: ReviewerFailureKind
    detail: str
    api_error_status: int | None = None

    @property
    def label(self) -> str:
        return _POLICY[self.kind].label

    @property
    def requeue_eligible(self) -> bool:
        return _POLICY[self.kind].requeue_eligible

    @property
    def retry_in_phase(self) -> bool:
        return _POLICY[self.kind].retry_in_phase

    def describe(self) -> str:
        return f'{self.label}: {self.detail}'

    def to_record(self) -> dict:
        return {
            'kind': self.kind.value,
            'detail': self.detail,
            'api_error_status': self.api_error_status,
        }

    @classmethod
    def from_review(cls, review: Mapping) -> ReviewerFailure:
        """Decode the ``failure`` record ``error_review`` wrote; never raises.

        A missing or malformed record decodes as UNCLASSIFIED, carrying the
        review's summary as its detail.
        """
        decoded = _decode_record(review.get('failure'))
        if decoded is not None:
            return decoded
        summary = review.get('summary')
        detail = summary if isinstance(summary, str) and summary else _NO_RECORD_DETAIL
        return cls(ReviewerFailureKind.UNCLASSIFIED, detail)


def _decode_record(record: object) -> ReviewerFailure | None:
    if not isinstance(record, Mapping):
        return None
    detail = record.get('detail')
    status = record.get('api_error_status')
    if not isinstance(detail, str):
        return None
    if status is not None and (isinstance(status, bool) or not isinstance(status, int)):
        return None
    try:
        kind = ReviewerFailureKind(record.get('kind'))
    except ValueError:
        return None
    return ReviewerFailure(kind, detail, api_error_status=status)


def classify_reviewer_result(result: AgentResult) -> ReviewerFailure:
    """Classify a reviewer run that left no usable verdict on disk.

    Cap evidence outranks an incidental 5xx: a run that exhausted its turns or
    budget did real work, so it must never ride the transient requeue lane.
    """
    if result.subtype == BUDGET_CAP_SUBTYPE:
        return ReviewerFailure(
            ReviewerFailureKind.BUDGET_CAP,
            f'subtype={result.subtype}, cost=${result.cost_usd:.2f}, turns={result.turns}',
        )
    agent_failure = classify_agent_failure(result)
    kind = _reviewer_kind(result, agent_failure.kind)
    status = result.api_error_status if kind is ReviewerFailureKind.SERVER_ERROR else None
    return ReviewerFailure(kind, agent_failure.summary, api_error_status=status)


def _reviewer_kind(result: AgentResult, agent_kind: AgentFailureKind) -> ReviewerFailureKind:
    if agent_kind is AgentFailureKind.MAX_TURNS:
        return ReviewerFailureKind.TURN_CAP
    if agent_kind is AgentFailureKind.API_ERROR and is_server_error_status(result.api_error_status):
        return ReviewerFailureKind.SERVER_ERROR
    if is_zero_work_failure(result):
        return ReviewerFailureKind.ZERO_WORK
    if agent_kind is AgentFailureKind.TIMED_OUT:
        return ReviewerFailureKind.TIMED_OUT
    if agent_kind is AgentFailureKind.SUCCESS:
        return ReviewerFailureKind.NO_VERDICT
    return ReviewerFailureKind.AGENT_FAILURE


def classify_reviewer_exception(exc: BaseException) -> ReviewerFailure:
    return ReviewerFailure(ReviewerFailureKind.EXCEPTION, f'{type(exc).__name__}: {exc}')


def error_review(role_name: str, failure: ReviewerFailure, *, summary: str) -> dict:
    """Build a reviewer's ERROR review dict; ``ReviewerFailure.from_review`` decodes it."""
    return {
        'reviewer': role_name,
        'verdict': 'ERROR',
        'issues': [],
        'summary': summary,
        'failure': failure.to_record(),
    }


def should_retry_reviewer(outcome: object) -> bool:
    """Return True when a reviewer's outcome earns an in-phase retry."""
    if isinstance(outcome, Exception):
        return True
    if isinstance(outcome, Mapping) and outcome.get('verdict') == 'ERROR':
        return ReviewerFailure.from_review(outcome).retry_in_phase
    return False


@dataclass(frozen=True)
class ReviewErrorTriage:
    requeue: bool
    reason: str
    detail: str
    api_error_status: int | None


def triage_reviewer_errors(errored_reviews: Mapping[str, Mapping]) -> ReviewErrorTriage:
    """Decide the review arm's exit for a panel's errored reviews, keyed by reviewer.

    Requeue only when EVERY errored reviewer is requeue-eligible; any cap,
    quality or unclassified failure blocks. A 5xx status, when present, is
    carried so the requeue rides the scheduler's transient lane.
    """
    if not errored_reviews:
        raise ValueError(
            'triage_reviewer_errors requires a non-empty errored_reviews; '
            'an empty panel must never read as all-transient'
        )
    failures = {
        name: ReviewerFailure.from_review(errored_reviews[name])
        for name in sorted(errored_reviews)
    }
    requeue = all(failure.requeue_eligible for failure in failures.values())
    listing = '; '.join(f'{name}: {failure.describe()}' for name, failure in failures.items())
    if requeue:
        reason = f'Review requeued: {len(failures)} reviewer(s) hit transient failures: {listing}'
    else:
        reason = f'{len(failures)} reviewer(s) failed without a usable verdict: {listing}'
    return ReviewErrorTriage(
        requeue=requeue,
        reason=reason,
        detail=_triage_detail(failures),
        api_error_status=_first_server_error_status(failures.values()),
    )


def _first_server_error_status(failures: Iterable[ReviewerFailure]) -> int | None:
    return next(
        (
            failure.api_error_status
            for failure in failures
            if failure.kind is ReviewerFailureKind.SERVER_ERROR
            and failure.api_error_status is not None
        ),
        None,
    )


def _triage_detail(failures: Mapping[str, ReviewerFailure]) -> str:
    return '\n'.join(
        f'{name} kind={failure.kind.value} '
        f'api_error_status={failure.api_error_status} detail={failure.detail}'
        for name, failure in failures.items()
    )
