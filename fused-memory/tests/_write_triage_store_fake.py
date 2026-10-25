"""The store double shared by the write-triage instruments' live-edge suites.

``scripts/calibrate_write_triage.py::fetch_recall_hit`` and
``scripts/eval_write_triage_retrieval.py::prefetch_retrievals`` make the same
two reads of a ``MemoryService``: one ``search`` per record, and one
``get_memory_by_id`` probe of the record's canonical. One double serves both
suites (``test_calibrate_write_triage_recall.py``,
``test_eval_write_triage_retrieval.py``), so the two instruments are measured
against the same store shape rather than against two copies that can drift.

It lives here rather than in ``_fm_helpers``, as ``_graphiti_fake`` does: a
double shared by exactly two consumers, kept out of the suite-wide miscellany.
"""
from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import AsyncMock

from fused_memory.models.memory import MemoryResult
from fused_memory.services.memory_service import SearchResults


class FakeMemoryService:
    """Every search answers *rows*; an id in *live* resolves by id.

    ``get_memory_by_id`` keeps production's contract: the record, or ``None``
    on a miss. ``probed`` lists the ids it was asked for, in order.
    """

    def __init__(
        self,
        rows: Iterable[MemoryResult] = (),
        *,
        live: Iterable[str] = (),
        degraded: bool = False,
    ) -> None:
        self.search = AsyncMock(return_value=SearchResults(list(rows), degraded=degraded))
        self.live = frozenset(live)
        self.probed: list[str] = []

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict | None:
        self.probed.append(memory_id)
        if memory_id not in self.live:
            return None
        return {'id': memory_id, 'content': '', 'metadata': {}}
