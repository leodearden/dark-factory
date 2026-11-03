"""Review issues and their aggregation, shaped as
``orchestrator.artifacts.TaskArtifacts.aggregate_reviews`` produces them, for
tests of the code that renders, routes or escalates review feedback.
"""

from __future__ import annotations

from orchestrator.artifacts import ReviewAggregation


def review_issue(tag: str, severity: str = 'suggestion') -> dict:
    """An issue whose every field carries *tag*, so a test can find each field in rendered text."""
    return {
        'reviewer': f'reviewer-{tag}',
        'severity': severity,
        'location': f'src/{tag}.py:{len(tag)}',
        'category': f'category-{tag}',
        'description': f'description-{tag}',
        'suggested_fix': f'fix-{tag}',
    }


def review_aggregation(
    blocking_issues: list[dict], suggestions: list[dict],
) -> ReviewAggregation:
    return ReviewAggregation(
        has_blocking_issues=bool(blocking_issues),
        blocking_issues=list(blocking_issues),
        suggestions=list(suggestions),
        reviews={},
    )
