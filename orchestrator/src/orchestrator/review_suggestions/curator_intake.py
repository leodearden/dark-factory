"""The ``submit_task`` arguments that hand one review suggestion to the curator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orchestrator.agents.triage import suggestion_hash

_DETAIL_FIELDS = ('reviewer', 'severity', 'category', 'location', 'suggested_fix')


def render_suggestion_details(suggestion: Mapping[str, Any], *, task_id: str) -> str:
    # Never a JSON document: FastMCP json-decodes such strings for non-`str` parameters.
    extra_fields = sorted(
        key for key in suggestion
        if key not in _DETAIL_FIELDS and key != 'description'
    )
    lines = [f'Review suggestion raised while reviewing task {task_id}.']
    lines.extend(
        f'{field}: {suggestion[field]}'
        for field in (*_DETAIL_FIELDS, *extra_fields)
        if field in suggestion
    )
    return '\n'.join(lines)


def review_suggestion_submit_arguments(
    suggestion: Mapping[str, Any], *, task_id: str, project_root: str,
) -> dict[str, Any]:
    category = suggestion.get('category', '')
    location = suggestion.get('location', '')
    description = suggestion.get('description', '')
    return {
        'title': f'[{category}] {location}: {description[:60]}',
        'description': description,
        'details': render_suggestion_details(suggestion, task_id=task_id),
        'priority': 'low',
        'project_root': project_root,
        'metadata': {
            'spawned_from': task_id,
            'spawn_context': 'review_suggestions',
            'escalation_id': f'review-suggestions-{task_id}',
            'suggestion_hash': suggestion_hash(dict(suggestion)),
        },
    }
