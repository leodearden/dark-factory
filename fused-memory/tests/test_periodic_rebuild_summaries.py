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

import asyncio
import contextlib
import logging

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
        # Mirror memory_service.rebuild_entity_summaries's real return contract
        # (errors is an int count, not a list) so callers that inspect the
        # result see production-shaped data.
        return {
            'total_entities': 1,
            'stale_entities': 1,
            'rebuilt': 1,
            'skipped': 0,
            'errors': 0,
            'details': [],
        }


@pytest.mark.asyncio
async def test_cycle_calls_rebuild_once_per_project():
    """One rebuild_entity_summaries call per configured project, in order, with force passed through."""
    fake_service = _FakeMemoryService()
    cfg = SummaryRebuildConfig(enabled=True, projects=['proj_a', 'proj_b'], force=True)

    await server_main._run_rebuild_summaries_cycle(fake_service, cfg)

    assert [c['project_id'] for c in fake_service.calls] == ['proj_a', 'proj_b']
    assert all(c['force'] is True for c in fake_service.calls)


@pytest.mark.asyncio
async def test_cycle_noop_when_disabled_or_no_projects():
    """The cycle must not call rebuild_entity_summaries when disabled or empty."""
    fake_service = _FakeMemoryService()

    # Case A: disabled, even with configured projects.
    disabled_cfg = SummaryRebuildConfig(enabled=False, projects=['proj_a'])
    await server_main._run_rebuild_summaries_cycle(fake_service, disabled_cfg)
    assert fake_service.calls == []

    # Case B: enabled, but no projects configured.
    empty_cfg = SummaryRebuildConfig(enabled=True, projects=[])
    await server_main._run_rebuild_summaries_cycle(fake_service, empty_cfg)
    assert fake_service.calls == []


@pytest.mark.asyncio
async def test_cycle_continues_past_raising_project(caplog):
    """One project's failure must not abort the sweep for the others."""
    fake_service = _FakeMemoryService(fail_for={'bad'})
    cfg = SummaryRebuildConfig(enabled=True, projects=['bad', 'good'])

    with caplog.at_level(logging.ERROR):
        await server_main._run_rebuild_summaries_cycle(fake_service, cfg)

    # 'good' was still processed despite 'bad' raising.
    assert [c['project_id'] for c in fake_service.calls] == ['good']
    # The failure for 'bad' was logged rather than silently swallowed.
    assert any(
        'bad' in rec.getMessage() for rec in caplog.records
        if rec.levelno >= logging.ERROR
    )


@pytest.mark.asyncio
async def test_cycle_logs_summary_counts_per_project(caplog):
    """Per-project rebuilt/stale/errors counts are logged (INFO clean, WARNING on errors).

    A backstop whose whole purpose is bounding staleness must give an
    operator signal about whether a sweep actually reduced it, not just
    that the call didn't raise.
    """

    class _ResultService:
        async def rebuild_entity_summaries(self, *, project_id, force=False, **kwargs):
            if project_id == 'proj_a':
                return {
                    'total_entities': 5, 'stale_entities': 2, 'rebuilt': 2,
                    'skipped': 0, 'errors': 0, 'details': [],
                }
            return {
                'total_entities': 5, 'stale_entities': 3, 'rebuilt': 2,
                'skipped': 0, 'errors': 1, 'details': [],
            }

    cfg = SummaryRebuildConfig(enabled=True, projects=['proj_a', 'proj_b'])

    with caplog.at_level(logging.INFO):
        await server_main._run_rebuild_summaries_cycle(_ResultService(), cfg)

    proj_a_records = [r for r in caplog.records if 'proj_a' in r.getMessage()]
    proj_b_records = [r for r in caplog.records if 'proj_b' in r.getMessage()]
    # Clean project (errors=0) logs at INFO with the rebuilt count visible.
    assert any(r.levelno == logging.INFO for r in proj_a_records)
    assert any('rebuilt' in r.getMessage() for r in proj_a_records)
    # Project reporting per-entity errors escalates to WARNING.
    assert any(r.levelno == logging.WARNING for r in proj_b_records)


@pytest.mark.asyncio
async def test_periodic_loop_runs_cycle_and_cancels_cleanly(monkeypatch):
    """The loop sleeps, runs one cycle per tick, and exits cleanly on cancel."""
    ran_count = 0
    first_run = asyncio.Event()

    async def _fake_cycle(memory_service, cfg):
        nonlocal ran_count
        ran_count += 1
        first_run.set()

    monkeypatch.setattr(server_main, '_run_rebuild_summaries_cycle', _fake_cycle)

    cfg = SummaryRebuildConfig(enabled=True, projects=['proj_a'], interval_seconds=0.01)
    task = asyncio.create_task(
        server_main._periodic_rebuild_summaries_loop(_FakeMemoryService(), cfg),
    )

    await asyncio.wait_for(first_run.wait(), timeout=1.0)
    assert ran_count >= 1

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.done()


@pytest.mark.asyncio
async def test_periodic_loop_logs_each_tick(monkeypatch, caplog):
    """Each tick logs a line noting the sweep ran and over how many projects.

    With a default 3600s interval and sleep-first semantics, this is the
    only signal an operator has that the loop is alive between sweeps.
    """
    ran = asyncio.Event()

    async def _fake_cycle(memory_service, cfg):
        ran.set()

    monkeypatch.setattr(server_main, '_run_rebuild_summaries_cycle', _fake_cycle)
    cfg = SummaryRebuildConfig(enabled=True, projects=['proj_a', 'proj_b'], interval_seconds=0.01)

    with caplog.at_level(logging.DEBUG):
        task = asyncio.create_task(
            server_main._periodic_rebuild_summaries_loop(_FakeMemoryService(), cfg),
        )
        await asyncio.wait_for(ran.wait(), timeout=1.0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert any(
        'sweep' in rec.getMessage() and '2' in rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.DEBUG
    )
