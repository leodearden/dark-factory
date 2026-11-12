"""A scripted ``MetadataLookupSource`` for tests of code that pages one.

``fused_memory.backends.metadata_lookup_pages::MetadataLookupSource`` is the
protocol; this is its one test double. Import it as
``from _fake_metadata_lookup_source import FakeMetadataLookupSource``.
"""

from __future__ import annotations

from typing import Any

from fused_memory.backends.metadata_lookup import DEFAULT_LOOKUP_LIMIT
from fused_memory.backends.task_backend_types import MetadataLookupResult, MetadataLookupRow


class FakeMetadataLookupSource:
    """Serves *rows* by offset and records every call's arguments in ``calls``.

    *total* overrides the reported total (default ``len(rows)``), *page_cap*
    serves fewer rows than the requested limit, *malformed_ids* is reported on
    every page, and *error*, when set, is raised by every read.
    """

    def __init__(
        self,
        rows: list[MetadataLookupRow],
        *,
        total: int | None = None,
        page_cap: int | None = None,
        malformed_ids: list[int] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.rows = rows
        self.total = len(rows) if total is None else total
        self.page_cap = page_cap
        self.malformed_ids = malformed_ids or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

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
    ) -> MetadataLookupResult:
        self.calls.append({
            'project_root': project_root, 'key': key, 'value': value, 'statuses': statuses,
            'limit': limit, 'offset': offset, 'tag': tag, 'include_value': include_value,
        })
        if self.error is not None:
            raise self.error
        size = limit if self.page_cap is None else min(limit, self.page_cap)
        return {
            'results': self.rows[offset:offset + size],
            'total': self.total,
            'tasks_with_key': self.total,
            'malformed_metadata_ids': list(self.malformed_ids),
        }
