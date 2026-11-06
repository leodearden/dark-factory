"""Whole reviewer payloads, shaped as ``TaskArtifacts.write_review`` stores
them under reviews/, for tests that seed or inspect that directory. The issues
inside them come from ``_review_fixtures.review_issue``.

An ERROR payload is not built here: ``orchestrator.reviewer_failure.error_review``
is the production builder, so use that.
"""

from __future__ import annotations

from _review_fixtures import review_issue


def blocking_review(reviewer: str) -> dict:
    """An ISSUES_FOUND review from *reviewer* holding one blocking issue."""
    return {
        'reviewer': reviewer,
        'verdict': 'ISSUES_FOUND',
        'issues': [review_issue(reviewer, 'blocking')],
        'summary': f'blocking finding from {reviewer}',
    }


def pass_review(reviewer: str) -> dict:
    """A PASS review from *reviewer* with no issues."""
    return {'reviewer': reviewer, 'verdict': 'PASS', 'issues': [], 'summary': 'ok'}
