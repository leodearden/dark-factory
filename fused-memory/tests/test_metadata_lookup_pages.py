"""``metadata_lookup_pages`` pages a metadata lookup source to the end of its matches.

The in-process lookup carries no ``pagination`` envelope, so "has more" is
``offset + len(results) < total``; this generator is that rule's one home.
"""

from __future__ import annotations

from typing import Any

import pytest
from _fake_metadata_lookup_source import FakeMetadataLookupSource

from fused_memory.backends.metadata_lookup_pages import metadata_lookup_pages
from fused_memory.backends.task_backend_types import MetadataLookupResult, MetadataLookupRow


def _row(task_id: int) -> MetadataLookupRow:
    return {
        'task_id': task_id,
        'title': f'task {task_id}',
        'status': 'pending',
        'priority': 'medium',
        'updated_at': '2026-11-02T00:00:00+00:00',
    }


async def _collect(source: FakeMetadataLookupSource, **kwargs: Any) -> list[MetadataLookupResult]:
    return [page async for page in metadata_lookup_pages(source, '/p', 'k', **kwargs)]


@pytest.mark.asyncio
async def test_pages_to_the_end_in_task_order() -> None:
    source = FakeMetadataLookupSource([_row(i) for i in range(1, 6)])
    pages = await _collect(source, page_size=2)
    assert len(pages) == 3
    assert [row['task_id'] for page in pages for row in page['results']] == [1, 2, 3, 4, 5]
    assert [call['offset'] for call in source.calls] == [0, 2, 4]


@pytest.mark.asyncio
async def test_query_is_forwarded_unchanged_on_every_call() -> None:
    source = FakeMetadataLookupSource([_row(i) for i in range(1, 4)])
    await _collect(
        source, statuses=['pending', 'deferred'], include_value=True, tag='t', page_size=2,
    )
    assert len(source.calls) == 2
    for call in source.calls:
        assert call['project_root'] == '/p'
        assert call['key'] == 'k'
        assert call['value'] is None
        assert call['statuses'] == ['pending', 'deferred']
        assert call['include_value'] is True
        assert call['tag'] == 't'
        assert call['limit'] == 2


@pytest.mark.asyncio
async def test_zero_matches_yield_one_empty_page() -> None:
    pages = await _collect(FakeMetadataLookupSource([]))
    assert len(pages) == 1
    assert pages[0]['results'] == []
    assert pages[0]['total'] == 0


@pytest.mark.asyncio
async def test_a_corpus_that_shrank_mid_read_stops() -> None:
    source = FakeMetadataLookupSource([_row(1)], total=5)
    pages = await _collect(source, page_size=1)
    assert [len(page['results']) for page in pages] == [1, 0]
    assert len(source.calls) == 2
