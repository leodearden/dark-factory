"""The counts-only payload of one ``review_completed`` event."""

from __future__ import annotations

import json

from _review_fixtures import review_issue

from orchestrator.artifacts import ReviewAggregation
from orchestrator.review_outcome import UNSPECIFIED, review_completed_payload


def _issue(tag: str, severity: str = 'suggestion', **overrides: object) -> dict:
    return {**review_issue(tag, severity), **overrides}


def _without(issue: dict, key: str) -> dict:
    return {k: v for k, v in issue.items() if k != key}


def _aggregation(
    blocking: list[dict] | None = None,
    suggestions: list[dict] | None = None,
    *,
    reviews: dict[str, dict] | None = None,
    reviewer_errors: list[str] | None = None,
) -> ReviewAggregation:
    blocking = list(blocking or [])
    return ReviewAggregation(
        has_blocking_issues=bool(blocking),
        blocking_issues=blocking,
        suggestions=list(suggestions or []),
        reviews=dict(reviews or {}),
        reviewer_errors=list(reviewer_errors or []),
    )


def _stored_payload(cycle: int, reviews: ReviewAggregation):
    """The payload as EventStore.emit persists it: JSON in the events row."""
    return json.loads(json.dumps(review_completed_payload(cycle, reviews)))


def test_payload_is_exactly_the_counts_of_the_aggregation():
    reviews = _aggregation(
        [review_issue('b1', 'blocking')],
        [review_issue('s1'), review_issue('s2')],
        reviews={'reviewer_a': {}, 'reviewer_b': {}},
    )

    payload = _stored_payload(3, reviews)

    assert payload == {
        'cycle': 3,
        'has_blocking_issues': True,
        'blocking_count': 1,
        'suggestion_count': 2,
        'by_category': {'category-b1': 1, 'category-s1': 1, 'category-s2': 1},
        'by_kind': {'defect': 3},
        'kind_flagged': 3,
        'reviewer_count': 2,
        'reviewer_errors': 0,
    }
    issue_total = payload['blocking_count'] + payload['suggestion_count']
    assert sum(payload['by_category'].values()) == issue_total
    assert sum(payload['by_kind'].values()) == issue_total


def test_clean_pass_has_zero_counts_and_empty_tallies():
    payload = _stored_payload(1, _aggregation())

    assert payload == {
        'cycle': 1,
        'has_blocking_issues': False,
        'blocking_count': 0,
        'suggestion_count': 0,
        'by_category': {},
        'by_kind': {},
        'kind_flagged': 0,
        'reviewer_count': 0,
        'reviewer_errors': 0,
    }


def test_by_kind_keeps_the_accessors_defect_reading_and_counts_flagged_kinds():
    reviews = _aggregation(
        [_issue('a', 'blocking', kind='defect')],
        [
            _issue('b', kind='docs'),
            _issue('c', kind='docs'),
            review_issue('d'),
            _issue('e', kind='bug'),
        ],
    )

    payload = _stored_payload(1, reviews)

    assert payload['by_kind'] == {'defect': 3, 'docs': 2}
    assert payload['kind_flagged'] == 2
    assert list(payload['by_kind']) == sorted(payload['by_kind'])


def test_by_category_counts_shared_labels_and_folds_unusable_ones_into_unspecified():
    reviews = _aggregation(
        [_issue('a', 'blocking', category='naming')],
        [
            _issue('b', category='naming'),
            _without(review_issue('c'), 'category'),
            _issue('d', category=''),
            _issue('e', category=['x']),
        ],
    )

    payload = _stored_payload(1, reviews)

    assert payload['by_category'] == {'naming': 2, UNSPECIFIED: 3}
    assert list(payload['by_category']) == sorted(payload['by_category'])


def test_reviewer_errors_are_counted_not_named():
    payload = _stored_payload(1, _aggregation(reviewer_errors=['reviewer_x']))

    assert payload['reviewer_errors'] == 1


def test_payload_carries_no_issue_or_review_text():
    tags = ('b1', 's1')
    reviews = _aggregation(
        [review_issue('b1', 'blocking')],
        [review_issue('s1')],
        reviews={'reviewer_a': {'summary': 'SECRET-SUMMARY'}},
    )

    serialised = json.dumps(review_completed_payload(2, reviews))

    for tag in tags:
        for text in (f'description-{tag}', f'fix-{tag}', f'src/{tag}.py', f'reviewer-{tag}'):
            assert text not in serialised
    assert 'SECRET-SUMMARY' not in serialised
    assert 'reviewer_a' not in serialised
