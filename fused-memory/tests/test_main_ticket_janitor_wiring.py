"""The janitor that ``server/main.py`` builds drains parked tickets through the interceptor.

Without this wiring nothing in production retries a ticket parked
``awaiting_curation``, so a curator-pool outage would strand every
submission it touched.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
import pytest_asyncio
from escalation.queue import EscalationQueue
from shared.project_layout import escalation_queue_dir

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig, TicketJanitorConfig
from fused_memory.middleware.curation_deferral import DeferReason
from fused_memory.middleware.curator_types import (
    CandidateTask,
    CuratorDecision,
    PreparedCandidate,
)
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_janitor import curation_backlog_anchor
from fused_memory.middleware.ticket_store import AWAITING_CURATION, TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.main import _build_ticket_janitor

PROJECT_ROOT = '/project'
PROJECT_ID = 'project'
DEADLINE_SECONDS = 5.0


class _DeferThenHoldCurator:
    """Defers its first batch; holds every later batch until the test ends."""

    def __init__(self, *, defer_first: bool) -> None:
        self._defer_next = defer_first
        self.reached = asyncio.Event()
        self._release = asyncio.Event()

    async def prepare_candidate(
        self, candidate: CandidateTask, project_id: str, project_root: str,
    ) -> PreparedCandidate:
        return PreparedCandidate(candidate=candidate, pool=[], pool_sizes={}, prompt_tokens=0)

    async def curate_batch_prepared(
        self, prepared: list[PreparedCandidate], project_id: str, project_root: str,
    ) -> list[CuratorDecision]:
        if self._defer_next:
            self._defer_next = False
            return [CuratorDecision.deferred(DeferReason.ALL_ACCOUNTS_CAPPED) for _ in prepared]
        self.reached.set()
        await self._release.wait()
        raise AssertionError('a held curation batch was released')

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _combine_audit_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path / 'data'))


@pytest.fixture
def config() -> FusedMemoryConfig:
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig(
        deferral_backoff_base_seconds=0.001,
        deferral_backoff_max_seconds=0.001,
        janitor=TicketJanitorConfig(curation_backlog_escalation_min_attempts=1),
    )
    return cfg


@pytest_asyncio.fixture
async def store(tmp_path):
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    yield ticket_store
    await ticket_store.close()


@pytest_asyncio.fixture
async def make_interceptor(config, store, tmp_path):
    buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    built: list[TaskInterceptor] = []

    def _make(curator: _DeferThenHoldCurator) -> TaskInterceptor:
        interceptor = TaskInterceptor(
            cast(Any, None), None, buffer, config=config,
            ticket_store=store, curator=cast(TaskCurator, curator),
        )
        built.append(interceptor)
        return interceptor

    yield _make
    for interceptor in built:
        await interceptor.close()
    await buffer.close()


async def _park_through(interceptor: TaskInterceptor) -> str:
    submitted = await interceptor.submit_task(PROJECT_ROOT, title='Capped work', description='d')
    ticket_id = submitted['ticket']
    answer = await asyncio.wait_for(
        interceptor.resolve_ticket(ticket_id, PROJECT_ROOT, timeout_seconds=DEADLINE_SECONDS),
        timeout=DEADLINE_SECONDS + 1,
    )
    assert answer['status'] == AWAITING_CURATION
    return ticket_id


@pytest.mark.asyncio
async def test_built_janitor_drains_through_the_interceptor(config, store, make_interceptor):
    ticket_id = await _park_through(make_interceptor(_DeferThenHoldCurator(defer_first=True)))
    curator = _DeferThenHoldCurator(defer_first=False)
    restarted = make_interceptor(curator)
    assert restarted.is_worker_alive(PROJECT_ID) is False

    janitor = _build_ticket_janitor(config, store, restarted, {})
    assert janitor is not None
    await janitor.tick()

    await curator.reached.wait()
    row = await store.get(ticket_id)
    assert row is not None
    assert (row['status'], row['attempts']) == ('pending', 1)
    assert restarted.is_worker_alive(PROJECT_ID) is True


@pytest.mark.asyncio
async def test_built_janitor_escalates_the_backlog_at_the_configured_attempts(
    config, store, make_interceptor, tmp_path,
):
    await _park_through(make_interceptor(_DeferThenHoldCurator(defer_first=True)))

    janitor = _build_ticket_janitor(config, store, None, {PROJECT_ID: str(tmp_path)})
    assert janitor is not None
    await janitor.tick()

    queue = EscalationQueue(escalation_queue_dir(str(tmp_path)))
    anchor = curation_backlog_anchor(DeferReason.ALL_ACCOUNTS_CAPPED)
    assert len(queue.get_by_task(anchor, status='pending')) == 1


@pytest.mark.asyncio
async def test_a_disabled_janitor_is_not_built(config, store, make_interceptor):
    config.curator.janitor = TicketJanitorConfig(enabled=False)

    interceptor = make_interceptor(_DeferThenHoldCurator(defer_first=False))

    assert _build_ticket_janitor(config, store, interceptor, {}) is None
