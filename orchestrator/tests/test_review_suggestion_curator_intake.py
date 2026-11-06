"""The submit_task arguments that hand one review suggestion to the curator."""

from __future__ import annotations

import json

import pytest
from _review_fixtures import review_issue

from orchestrator.agents.triage import suggestion_hash
from orchestrator.review_suggestions.curator_intake import (
    review_suggestion_submit_arguments,
)

TASK_ID = '6201'
PROJECT_ROOT = '/srv/project'


def _arguments(suggestion: dict) -> dict:
    return review_suggestion_submit_arguments(
        suggestion, task_id=TASK_ID, project_root=PROJECT_ROOT,
    )


def test_title_description_priority_and_project_root() -> None:
    suggestion = review_issue('alpha')
    suggestion['description'] = 'x' * 100

    arguments = _arguments(suggestion)

    assert arguments['title'] == (
        f"[{suggestion['category']}] {suggestion['location']}: "
        f"{suggestion['description'][:60]}"
    )
    assert arguments['description'] == suggestion['description']
    assert arguments['priority'] == 'low'
    assert arguments['project_root'] == PROJECT_ROOT


def test_metadata_carries_provenance_and_the_per_item_idempotency_key() -> None:
    suggestion = review_issue('alpha')

    assert _arguments(suggestion)['metadata'] == {
        'spawned_from': TASK_ID,
        'spawn_context': 'review_suggestions',
        'escalation_id': f'review-suggestions-{TASK_ID}',
        'suggestion_hash': suggestion_hash(suggestion),
    }


def test_details_is_text_fastmcp_will_not_pre_parse_as_json() -> None:
    details = _arguments(review_issue('alpha'))['details']

    assert isinstance(details, str)
    with pytest.raises(ValueError):
        json.loads(details)


@pytest.mark.parametrize(
    'field', ['reviewer', 'severity', 'category', 'location', 'suggested_fix'],
)
def test_details_carries_every_field_but_the_description(field: str) -> None:
    suggestion = review_issue('alpha')

    assert suggestion[field] in _arguments(suggestion)['details']


def test_suggestion_missing_optional_keys_still_yields_arguments() -> None:
    arguments = _arguments({'description': 'only a description'})

    assert arguments['title'] == '[] : only a description'
    assert arguments['description'] == 'only a description'
    with pytest.raises(ValueError):
        json.loads(arguments['details'])
