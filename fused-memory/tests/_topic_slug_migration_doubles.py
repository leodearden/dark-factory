"""Doubles shared by test_normalize_topic_slugs.py and the test_topic_slug_*.py suites."""
from __future__ import annotations

import copy
from unittest.mock import AsyncMock, MagicMock

from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY


def rec(memory_id: str, topic: object = None, **meta) -> dict:
    """One scroll-shaped record, the shape ``scroll_all_by_metadata`` yields.

    ``topic=None`` means "no ``topic`` key at all", which is a genuinely
    different case from ``topic`` present-and-``None`` — the caller spells the
    latter explicitly via ``metadata``.
    """
    metadata: dict = dict(meta)
    if topic is not None:
        metadata['topic'] = topic
    return {'id': memory_id, 'created_at': '2026-01-01T00:00:00Z', 'metadata': metadata}


def by_reason(skips: list[dict], reason: str) -> list[dict]:
    return [s for s in skips if s.get('reason') == reason]


def gate_task(
    task_id: str,
    topic: str,
    *,
    project_id: str = 'dark_factory',
    project_root: str = '/repo',
    **block_extras,
) -> dict:
    """One gate task as the MCP ``get_task`` read hands it over.

    ``block_extras`` seed the SIBLING keys inside the block (``provenance``,
    ``considered_and_kept``, ...) — the ones the patch must leave untouched.
    """
    block = {'topic': topic}
    block.update(block_extras)
    return {
        'id': task_id,
        'project_id': project_id,
        'project_root': project_root,
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': True,
            GATE_METADATA_KEY: block,
        },
    }


def task_listing(project_id: str, project_root: str, *tasks: dict) -> dict:
    """One ``get_tasks`` response in the measured live shape."""
    return {'tasks': list(tasks), 'project_id': project_id, 'project_root': project_root}


def census_client(answers: dict[str, object]) -> MagicMock:
    """A client whose ``get_tasks`` answers per root: a response, or an exception.

    Answers are deep-copied, as a JSON transport would hand them over, so the
    census can never alias the double's own state.
    """
    async def _call_tool(name, payload):
        assert name == 'get_tasks', name
        answer = answers[payload['project_root']]
        if isinstance(answer, BaseException):
            raise answer
        return copy.deepcopy(answer)

    client = MagicMock()
    client.call_tool = AsyncMock(side_effect=_call_tool)
    return client
