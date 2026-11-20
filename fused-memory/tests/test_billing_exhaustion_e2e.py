"""Exhausted OpenAI credits, end to end through MemoryService (task 10922).

On 2026-11-19 an exhausted OpenAI balance dead-lettered 106 writes behind
alarms that read as a transient rate limit. These tests pin what an operator
now sees instead: one born-at-L2 page naming the credits, writes PARKED
rather than dead-lettered, and both clearing by themselves once a write
lands. A plain 429 keeps today's retry-then-dead-letter path. Mem0 writes,
which already retry without limit, page the same way and resolve it when one
lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_helpers import (
    make_insufficient_quota_error,
    make_transient_rate_limit_error,
    wrap_like_graphiti,
)
from _fm_polling import poll_until
from shared.project_layout import escalation_queue_dir
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from test_e2e_durable_queue import _mock_backends

from fused_memory.middleware import _folded_escalation
from fused_memory.models.enums import SourceStore
from fused_memory.models.scope import configured_primary_root
from fused_memory.services.memory_service import MemoryService

pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.skipif(
        not _folded_escalation.HAS_ESCALATION,
        reason='escalation package unavailable (minimal env)',
    ),
]

_BILLING = 'provider_billing_exhausted'
_DEAD_LETTER = 'durable_write_dead_letter'


@pytest_asyncio.fixture
async def billing_service(mock_config, tmp_path):
    """A MemoryService over a real queue and journal, re-probing parked writes fast."""
    from fused_memory.services.write_journal import WriteJournal

    mock_config.queue.retry_max_delay_seconds = 0.5
    svc = _mock_backends(MemoryService(mock_config))
    await svc.initialize()
    journal = WriteJournal(tmp_path / 'wj_billing')
    await journal.initialize()
    svc.set_write_journal(journal)
    svc.set_known_projects({'main': str(tmp_path)})

    yield svc

    await svc.close()
    await journal.close()


def _primary_root(svc: MemoryService) -> Path:
    root = configured_primary_root(svc.config.taskmaster)
    assert root, 'the test config must pin a primary project root'
    return Path(root)


def _records(root: Path, category: str) -> list[dict]:
    """Every record of *category* on disk, the archive a resolve moves it to included."""
    records = (json.loads(p.read_text()) for p in escalation_queue_dir(root).rglob('esc-*.json'))
    return [r for r in records if r['category'] == category]


async def _counts(svc: MemoryService) -> dict:
    queue = svc.durable_queue
    assert queue is not None
    return (await queue.get_stats())['counts']


async def _add_episode(svc: MemoryService, content: str):
    return await svc.add_episode(content=content, project_id='main')


class TestExhaustedCreditsOnTheDurablePath:
    @pytest.mark.asyncio
    async def test_writes_park_behind_one_critical_page(self, billing_service, tmp_path):
        svc = billing_service
        svc.graphiti.add_episode = AsyncMock(
            side_effect=wrap_like_graphiti(make_insufficient_quota_error()),
        )

        response = await _add_episode(svc, 'an episode written into an empty balance')

        assert response.status.value == 'queued'
        root = _primary_root(svc)
        [page] = await poll_until(
            lambda: _records(root, _BILLING) or None, timeout=20.0,
            message='no provider_billing_exhausted page was filed',
        )
        assert page['severity'] == 'critical'
        assert page['level'] == 2
        assert 'credits exhausted' in page['summary'].lower()
        counts = await poll_until(
            lambda: _parked_counts(svc), timeout=20.0, message='nothing parked',
        )
        assert counts.get('dead', 0) == 0
        assert _records(root, _DEAD_LETTER) == []
        assert _records(tmp_path, _DEAD_LETTER) == []

        await _add_episode(svc, 'a second episode into the same outage')
        await poll_until(
            lambda: _parked_counts(svc, at_least=2), timeout=20.0,
            message='the second write did not park',
        )
        assert len(_records(root, _BILLING)) == 1

    @pytest.mark.asyncio
    async def test_restored_credits_drain_the_parked_writes_and_resolve_the_page(
        self, billing_service,
    ):
        svc = billing_service
        svc.graphiti.add_episode = AsyncMock(
            side_effect=wrap_like_graphiti(make_insufficient_quota_error()),
        )
        await _add_episode(svc, 'first')
        await _add_episode(svc, 'second')
        await poll_until(
            lambda: _parked_counts(svc, at_least=2), timeout=20.0, message='nothing parked',
        )

        svc.graphiti.add_episode = AsyncMock(return_value=None)

        async def _drained():
            counts = await _counts(svc)
            return counts if counts.get('completed', 0) >= 2 and not counts.get('parked') else None

        await poll_until(_drained, timeout=20.0, message='the parked writes never drained')
        root = _primary_root(svc)
        await poll_until(
            lambda: [r['status'] for r in _records(root, _BILLING)] == ['resolved'],
            timeout=20.0, message='the billing page was not resolved',
        )

    @pytest.mark.asyncio
    async def test_a_transient_429_still_dead_letters_without_a_billing_page(
        self, billing_service, tmp_path,
    ):
        svc = billing_service
        svc.graphiti.add_episode = AsyncMock(
            side_effect=wrap_like_graphiti(make_transient_rate_limit_error()),
        )

        await _add_episode(svc, 'an episode behind a transient rate limit')

        async def _dead():
            counts = await _counts(svc)
            return counts if counts.get('dead', 0) >= 1 else None

        await poll_until(_dead, timeout=20.0, message='the item never dead-lettered')
        await poll_until(
            lambda: _records(tmp_path, _DEAD_LETTER) or None, timeout=20.0,
            message='no durable_write_dead_letter record was filed',
        )
        assert _records(_primary_root(svc), _BILLING) == []


async def _add_preference(svc: MemoryService, content: str):
    return await svc.add_memory(
        content=content, category='preferences_and_norms', project_id='main',
    )


class TestExhaustedCreditsOnTheMem0Path:
    @pytest.mark.asyncio
    async def test_a_deferred_write_pages_once(self, billing_service):
        svc = billing_service
        svc.mem0.add = AsyncMock(side_effect=make_insufficient_quota_error())

        first = await _add_preference(svc, 'prefer explicit imports over star imports')
        second = await _add_preference(svc, 'prefer frozen dataclasses for values')

        assert first.stores_pending_retry == [SourceStore.mem0]
        assert second.stores_pending_retry == [SourceStore.mem0]
        [page] = _records(_primary_root(svc), _BILLING)
        assert page['severity'] == 'critical'
        assert page['level'] == 2
        assert page['status'] == 'pending'
        assert 'credits exhausted' in page['summary'].lower()

    @pytest.mark.asyncio
    async def test_a_landed_write_resolves_the_page(self, billing_service):
        svc = billing_service
        svc.mem0.add = AsyncMock(side_effect=make_insufficient_quota_error())
        await _add_preference(svc, 'prefer explicit imports over star imports')

        svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-back'}]})
        landed = await _add_preference(svc, 'prefer frozen dataclasses for values')

        assert landed.memory_ids == ['mem0-back']
        assert [r['status'] for r in _records(_primary_root(svc), _BILLING)] == ['resolved']

    @pytest.mark.asyncio
    async def test_a_transient_429_defers_without_a_billing_page(self, billing_service):
        svc = billing_service
        svc.mem0.add = AsyncMock(side_effect=make_transient_rate_limit_error())

        result = await _add_preference(svc, 'prefer explicit imports over star imports')

        assert result.stores_pending_retry == [SourceStore.mem0]
        assert _records(_primary_root(svc), _BILLING) == []


async def _parked_counts(svc: MemoryService, *, at_least: int = 1) -> dict | None:
    counts = await _counts(svc)
    return counts if counts.get('parked', 0) >= at_least else None
