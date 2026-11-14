"""The payload of one ``review_completed`` event: counts of a decided review, never issue text.

Its consumer is a client-facing feed (``plans/client-feed-prd.md`` §4 decision 16), so
descriptions, locations, fixes and reviewer names stay out of it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from shared.defects import issue_kind_of

from orchestrator.artifacts import ReviewAggregation

__all__ = ['UNSPECIFIED', 'review_completed_payload']

UNSPECIFIED = 'unspecified'


def review_completed_payload(cycle: int, reviews: ReviewAggregation) -> dict[str, object]:
    issues = (*reviews.blocking_issues, *reviews.suggestions)
    kind_readings = [issue_kind_of(issue) for issue in issues]
    return {
        'cycle': cycle,
        'has_blocking_issues': reviews.has_blocking_issues,
        'blocking_count': len(reviews.blocking_issues),
        'suggestion_count': len(reviews.suggestions),
        'by_category': _tally(_category_of(issue) for issue in issues),
        'by_kind': _tally(reading.kind.value for reading in kind_readings),
        'kind_flagged': sum(reading.flagged for reading in kind_readings),
        'reviewer_count': len(reviews.reviews),
        'reviewer_errors': len(reviews.reviewer_errors),
    }


def _tally(keys: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(keys).items()))


def _category_of(issue: dict) -> str:
    category = issue.get('category')
    return category if isinstance(category, str) and category else UNSPECIFIED
