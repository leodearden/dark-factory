"""On-contract reviewer issues for tests, keyed by the contract's own fields.

A field added to orchestrator/review_issue_contract.py::IssueField is added to
ON_CONTRACT_VALUES here once, and every test fixture built below follows.
"""

from __future__ import annotations

from orchestrator.review_issue_contract import IssueField, IssueSeverity

ON_CONTRACT_VALUES: dict[IssueField, str] = {
    IssueField.SEVERITY: IssueSeverity.BLOCKING.value,
    IssueField.LOCATION: 'pkg/mod.py:3',
    IssueField.CATEGORY: 'correctness',
    IssueField.DESCRIPTION: 'the thing is broken',
    IssueField.SUGGESTED_FIX: 'fix the thing',
}


def contract_issue(**overrides: object) -> dict[str, object]:
    """An on-contract issue, with *overrides* set over it."""
    return {**{str(field): value for field, value in ON_CONTRACT_VALUES.items()}, **overrides}


def contract_issue_without(field: IssueField, **overrides: object) -> dict[str, object]:
    """``contract_issue(**overrides)`` with *field* absent."""
    issue = contract_issue(**overrides)
    del issue[field]
    return issue
