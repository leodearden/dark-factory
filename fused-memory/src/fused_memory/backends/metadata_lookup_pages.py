"""Page any :class:`MetadataLookupSource` to the end of its matches.

The in-process ``find_tasks_by_metadata`` carries no ``pagination`` envelope,
so "has more" is ``offset + len(results) < total``. :func:`metadata_lookup_pages`
is that rule's one home; it also stops on an empty page, so a corpus that
shrinks mid-read cannot loop forever.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from fused_memory.backends.metadata_lookup import DEFAULT_LOOKUP_LIMIT
from fused_memory.backends.task_backend_types import MetadataLookupResult

__all__ = ['LOOKUP_PAGE_SIZE', 'MetadataLookupSource', 'metadata_lookup_pages']

LOOKUP_PAGE_SIZE = 100


class MetadataLookupSource(Protocol):
    async def find_tasks_by_metadata(
        self,
        project_root: str,
        key: str,
        value: str | int | None = None,
        *,
        statuses: list[str] | None = None,
        limit: int = DEFAULT_LOOKUP_LIMIT,
        offset: int = 0,
        tag: str | None = None,
        include_value: bool = False,
    ) -> MetadataLookupResult: ...


async def metadata_lookup_pages(
    source: MetadataLookupSource,
    project_root: str,
    key: str,
    *,
    statuses: list[str] | None = None,
    include_value: bool = False,
    tag: str | None = None,
    page_size: int = LOOKUP_PAGE_SIZE,
) -> AsyncIterator[MetadataLookupResult]:
    """Every page of the tasks carrying *key*, from offset 0 to the last match."""
    offset = 0
    while True:
        page = await source.find_tasks_by_metadata(
            project_root,
            key,
            None,
            statuses=statuses,
            include_value=include_value,
            limit=page_size,
            offset=offset,
            tag=tag,
        )
        yield page
        offset += len(page['results'])
        if not page['results'] or offset >= page['total']:
            return
