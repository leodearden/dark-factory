"""``normalize_review`` reads each issue's kind through ``shared.defects.issue_kind_of``."""

from __future__ import annotations

import pytest
from shared.defects import IssueKind, IssueKindFlag, issue_kind_of

from orchestrator.review_payload import normalize_review


def _review(*issues: dict) -> dict:
    return {'reviewer': 'r', 'verdict': 'ISSUES_FOUND', 'issues': list(issues)}


def test_a_legacy_issue_without_kind_normalises_to_a_flagged_defect():
    (issue,) = normalize_review(_review({'severity': 'blocking', 'description': 'd'})).issues

    reading = issue_kind_of(issue)

    assert reading.kind is IssueKind.DEFECT
    assert reading.flag is IssueKindFlag.MISSING
    assert issue['kind_flag'] == IssueKindFlag.MISSING.value


def test_an_unrecognised_kind_is_flagged_as_unrecognised():
    (issue,) = normalize_review(_review({'severity': 'blocking', 'kind': 'bug'})).issues

    assert issue['kind_flag'] == IssueKindFlag.UNRECOGNISED.value
    assert issue_kind_of(issue).kind is IssueKind.DEFECT


@pytest.mark.parametrize('kind', list(IssueKind))
def test_a_recognised_kind_is_left_unflagged(kind):
    (issue,) = normalize_review(_review({'severity': 'blocking', 'kind': kind.value})).issues

    assert 'kind_flag' not in issue
    assert issue_kind_of(issue).kind is kind


def test_normalising_does_not_mutate_the_received_issue():
    received = {'severity': 'blocking'}

    normalize_review(_review(received))

    assert received == {'severity': 'blocking'}
