"""Production-shaped ``MemoryService.rebuild_entity_summaries`` results for test fakes.

The return contract lives in
fused-memory/src/fused_memory/services/memory_service.py::MemoryService.rebuild_entity_summaries:
six count/detail keys plus the four tri-state read-completeness keys. Fakes
that hand-copied it drifted when the completeness keys were added (task 6498),
quietly sending their callers down the not-proven-complete branch, so every
fake builds its result here and the next contract change touches one place.
"""
from __future__ import annotations

from typing import Any


def rebuild_service_result(**overrides: Any) -> dict[str, Any]:
    """One clean rebuild over a proven-complete corpus, with ``overrides`` applied."""
    return {
        'total_entities': 1,
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
