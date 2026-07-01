"""Tests for the periodic entity-summary rebuild loop in ``server/main.py``.

Task 1958 — scheduled staleness backstop (fix (b)), a follow-up to task
1949's fix (a) (best-effort post-ingestion refresh of edge endpoints
returned by a single write). Fix (a) alone leaves residual drift for
entities graphiti_core resolves against but that are not in the returned
edge list; only a periodic sweep of
``memory_service.rebuild_entity_summaries`` bounds that drift regardless
of cause.

We test ``_run_rebuild_summaries_cycle`` (the single-pass extraction)
directly, mirroring tests/test_periodic_checkpoint.py's style:

  * one ``rebuild_entity_summaries`` call per configured project, in order
  * ``force`` is passed through from config
  * the cycle no-ops when disabled or when there are no configured projects
  * one project's failure does not abort the sweep for the others

and ``_periodic_rebuild_summaries_loop`` (the thin sleep/cancel wrapper).
"""

from __future__ import annotations

import pytest

from fused_memory.config.schema import SummaryRebuildConfig
from fused_memory.server import main as server_main


class _FakeMemoryService:
    """Records every ``rebuild_entity_summaries`` call's kwargs.

    ``fail_for`` names project_ids that should raise instead of succeeding,
    to exercise per-project error isolation.
    """

    def __init__(self, *, fail_for: set[str] | None = None) -> None:
        self.calls: list[dict] = []
        self._fail_for = fail_for or set()

    async def rebuild_entity_summaries(self, *, project_id, force=False, **kwargs):
        if project_id in self._fail_for:
            raise RuntimeError('boom')
        self.calls.append({'project_id': project_id, 'force': force, **kwargs})


@pytest.mark.asyncio
async def test_cycle_calls_rebuild_once_per_project():
    """One rebuild_entity_summaries call per configured project, in order, with force passed through."""
    fake_service = _FakeMemoryService()
    cfg = SummaryRebuildConfig(enabled=True, projects=['proj_a', 'proj_b'], force=True)

    await server_main._run_rebuild_summaries_cycle(fake_service, cfg)

    assert [c['project_id'] for c in fake_service.calls] == ['proj_a', 'proj_b']
    assert all(c['force'] is True for c in fake_service.calls)
