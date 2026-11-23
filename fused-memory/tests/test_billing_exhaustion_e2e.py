"""Exhausted OpenAI credits, end to end through MemoryService (task 10922).

On 2026-11-19 an exhausted OpenAI balance dead-lettered 106 writes behind
alarms that read as a transient rate limit. These tests pin what an operator
now sees instead: one born-at-L2 page naming the credits, writes PARKED
rather than dead-lettered, and both clearing by themselves once a write
lands. A plain 429 keeps today's retry-then-dead-letter path. Mem0 writes,
which already retry without limit, page the same way and resolve it when one
lands. A restart neither pages again nor strands the page it inherits.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_helpers import (
    make_insufficient_quota_error,
    make_transient_rate_limit_error,
    mock_memory_backends,
    wrap_like_graphiti,
)
from _fm_polling import poll_until
from escalation.archive import archive_dir_for_date
from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths, read_escalation_for_scan
from shared.project_layout import escalation_queue_dir
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.middleware._folded_escalation import file_folded_escalation
from fused_memory.models.enums import SourceStore
from fused_memory.models.scope import configured_primary_root
from fused_memory.services.billing_exhaustion_alarm import ANCHOR_TASK_ID
from fused_memory.services.memory_service import MemoryService

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_BILLING = 'provider_billing_exhausted'
_DEAD_LETTER = 'durable_write_dead_letter'


@asynccontextmanager
async def _running_service(config, tmp_path) -> AsyncIterator[MemoryService]:
    """A MemoryService over a real queue and journal, re-probing parked writes fast.

    Every entry over one *tmp_path* opens the same queue and journal files, so
    entering again after an exit is the same server restarted.
    """
    from fused_memory.services.write_journal import WriteJournal

    config.queue.retry_max_delay_seconds = 0.5
    svc = mock_memory_backends(MemoryService(config))
    await svc.initialize()
    journal = WriteJournal(tmp_path / 'wj_billing')
    await journal.initialize()
    svc.set_write_journal(journal)
    svc.set_known_projects({'main': str(tmp_path)})
    try:
        yield svc
    finally:
        await svc.close()
        await journal.close()


@pytest_asyncio.fixture
async def billing_service(mock_config, tmp_path):
    async with _running_service(mock_config, tmp_path) as svc:
        yield svc


def _primary_root(svc: MemoryService) -> Path:
    root = configured_primary_root(svc.config.taskmaster)
    assert root, 'the test config must pin a primary project root'
    return Path(root)


def _records(root: Path, category: str) -> list[Escalation]:
    """Every *category* record on disk, archive included.

    A record a concurrent resolve archives between the listing and its read is
    skipped; an unreadable or unparsable record fails the test.
    """
    records: list[Escalation] = []
    for path in iter_all_escalation_paths(escalation_queue_dir(root)):
        esc, reason = read_escalation_for_scan(path, context='billing e2e')
        assert reason in ('ok', 'vanished'), f'{path}: {reason}'
        if esc is not None and esc.category == category:
            records.append(esc)
    return records


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
        assert page.severity == 'critical'
        assert page.level == 2
        assert 'credits exhausted' in page.summary.lower()
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
            lambda: [r.status for r in _records(root, _BILLING)] == ['resolved'],
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
        assert page.severity == 'critical'
        assert page.level == 2
        assert page.status == 'pending'
        assert 'credits exhausted' in page.summary.lower()

    @pytest.mark.asyncio
    async def test_a_landed_write_resolves_the_page(self, billing_service):
        svc = billing_service
        svc.mem0.add = AsyncMock(side_effect=make_insufficient_quota_error())
        await _add_preference(svc, 'prefer explicit imports over star imports')

        svc.mem0.add = AsyncMock(return_value={'results': [{'id': 'mem0-back'}]})
        landed = await _add_preference(svc, 'prefer frozen dataclasses for values')

        assert landed.memory_ids == ['mem0-back']
        assert [r.status for r in _records(_primary_root(svc), _BILLING)] == ['resolved']

    @pytest.mark.asyncio
    async def test_a_write_that_lands_nothing_leaves_the_page_open(self, billing_service):
        """mem0 can return no ids without calling the embedder, which proves
        nothing about the credits."""
        svc = billing_service
        svc.mem0.add = AsyncMock(side_effect=make_insufficient_quota_error())
        await _add_preference(svc, 'prefer explicit imports over star imports')

        svc.mem0.add = AsyncMock(return_value={'results': []})
        empty = await _add_preference(svc, 'prefer frozen dataclasses for values')

        assert empty.memory_ids == []
        assert [r.status for r in _records(_primary_root(svc), _BILLING)] == ['pending']

    @pytest.mark.asyncio
    async def test_a_transient_429_defers_without_a_billing_page(self, billing_service):
        svc = billing_service
        svc.mem0.add = AsyncMock(side_effect=make_transient_rate_limit_error())

        result = await _add_preference(svc, 'prefer explicit imports over star imports')

        assert result.stores_pending_retry == [SourceStore.mem0]
        assert _records(_primary_root(svc), _BILLING) == []


class TestARestartMidOutage:
    @pytest.mark.asyncio
    async def test_the_first_write_to_land_resolves_the_page_left_open(
        self, mock_config, tmp_path,
    ):
        async with _running_service(mock_config, tmp_path) as before:
            before.mem0.add = AsyncMock(side_effect=make_insufficient_quota_error())
            await _add_preference(before, 'prefer explicit imports over star imports')

        async with _running_service(mock_config, tmp_path) as after:
            await _add_preference(after, 'credits were restored during the restart')

            statuses = [r.status for r in _records(_primary_root(after), _BILLING)]
        assert statuses == ['resolved']

    @pytest.mark.asyncio
    async def test_a_park_after_the_restart_pages_nothing_new(self, mock_config, tmp_path):
        async with _running_service(mock_config, tmp_path) as before:
            before.mem0.add = AsyncMock(side_effect=make_insufficient_quota_error())
            await _add_preference(before, 'prefer explicit imports over star imports')

        async with _running_service(mock_config, tmp_path) as after:
            after.graphiti.add_episode = AsyncMock(
                side_effect=wrap_like_graphiti(make_insufficient_quota_error()),
            )
            await _add_episode(after, 'an episode into the outage the restart inherited')
            await poll_until(
                lambda: _parked_counts(after), timeout=20.0, message='nothing parked',
            )
            root = _primary_root(after)
            [page] = _records(root, _BILLING)
            assert page.status == 'pending'

            after.graphiti.add_episode = AsyncMock(return_value=None)
            await poll_until(
                lambda: [r.status for r in _records(root, _BILLING)] == ['resolved'],
                timeout=20.0, message='the inherited page was not resolved',
            )


class TestReadingPagesWhileAResolveArchivesThem:
    def test_a_page_archived_between_listing_and_read_is_found_on_the_next_read(
        self, tmp_path, monkeypatch,
    ):
        page_id = file_folded_escalation(
            str(tmp_path), anchor_task_id=ANCHOR_TASK_ID, agent_role='test-billing-e2e',
            category=_BILLING, severity='blocking', summary='credits exhausted',
            detail='a page a resolve archives mid-listing',
            suggested_action='none; this is a test page',
            logger=logging.getLogger(__name__), log_label='billing e2e race',
        )
        assert page_id
        doomed = escalation_queue_dir(tmp_path) / f'{page_id}.json'
        archived = archive_dir_for_date(
            escalation_queue_dir(tmp_path), '2026-11-23T00:00:00+00:00',
        ) / doomed.name
        original_read_text = Path.read_text

        def read_text_racing_a_resolve(self: Path, *args, **kwargs) -> str:
            if self == doomed and doomed.exists():
                archived.parent.mkdir(parents=True, exist_ok=True)
                os.replace(doomed, archived)
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, 'read_text', read_text_racing_a_resolve)

        assert [r.id for r in _records(tmp_path, _BILLING)] in ([], [page_id])
        assert archived.exists() and not doomed.exists(), 'the read never raced an archive move'
        assert [r.id for r in _records(tmp_path, _BILLING)] == [page_id]


async def _parked_counts(svc: MemoryService, *, at_least: int = 1) -> dict | None:
    counts = await _counts(svc)
    return counts if counts.get('parked', 0) >= at_least else None
