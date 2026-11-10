"""RebuildResult carries the tri-state read completeness (task 6498).

MemoryService.rebuild_entity_summaries reports entities_complete /
entities_incomplete_kind / edges_complete / edges_incomplete_kind; these tests
pin that RebuildResult parses them, that a missing key reads as UNKNOWN, and
that only two True verdicts prove the corpus complete.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import INCOMPLETE_SHORT_READ
from fused_memory.maintenance.rebuild_summaries import (
    RebuildResult,
    RebuildSummariesManager,
)


def _service_result(**overrides: Any) -> dict[str, Any]:
    """A MemoryService.rebuild_entity_summaries dict over a proven-complete corpus."""
    return {
        'total_entities': 3,
        'stale_entities': 1,
        'rebuilt': 1,
        'skipped': 0,
        'errors': 0,
        'details': [],
        'entities_complete': True,
        'entities_incomplete_kind': None,
        'edges_complete': True,
        'edges_incomplete_kind': None,
        **overrides,
    }


def _kind_for(verdict: bool | None) -> str | None:
    return INCOMPLETE_SHORT_READ if verdict is False else None


@pytest.mark.asyncio
async def test_manager_carries_the_four_completeness_keys():
    service = MagicMock()
    service.rebuild_entity_summaries = AsyncMock(return_value=_service_result(
        edges_complete=False, edges_incomplete_kind=INCOMPLETE_SHORT_READ,
    ))

    result = await RebuildSummariesManager(service=service, group_id='proj').run()

    assert result.entities_complete is True
    assert result.entities_incomplete_kind is None
    assert result.edges_complete is False
    assert result.edges_incomplete_kind == INCOMPLETE_SHORT_READ


def test_missing_completeness_keys_read_as_unknown_never_complete():
    result = RebuildResult.from_service_result({
        'total_entities': 1,
        'stale_entities': 0,
        'rebuilt': 0,
        'skipped': 0,
        'errors': 0,
        'details': [],
    })

    assert result.entities_complete is None
    assert result.entities_incomplete_kind is None
    assert result.edges_complete is None
    assert result.edges_incomplete_kind is None
    assert result.corpus_proven_complete is False


@pytest.mark.parametrize(
    ('entities_complete', 'edges_complete', 'expected'),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (True, None, False),
        (None, True, False),
        (None, None, False),
    ],
)
def test_corpus_proven_complete_only_when_both_reads_are_true(
    entities_complete, edges_complete, expected,
):
    result = RebuildResult(
        entities_complete=entities_complete,
        entities_incomplete_kind=_kind_for(entities_complete),
        edges_complete=edges_complete,
        edges_incomplete_kind=_kind_for(edges_complete),
    )

    assert result.corpus_proven_complete is expected
