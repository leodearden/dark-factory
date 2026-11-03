"""The reconciliation escalation server (8103) is handed the in-process prefix resolver.

PRD ``plans/uuid-prefix-resolution-prd.md`` leaf δ, D7: this server runs inside
fused-memory, so its uuid-prefix guard resolves through the real IdResolver
over the harness's own memory service, with no HTTP hop. The server is built
through ``create_escalation_server``, patched here exactly as the sibling
store-identity test patches it, and the captured ``prefix_resolver`` is then
driven against fake stores.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _id_resolver_fakes import FakeGraphiti, FakeMem0, RecordingTombstones
from shared.uuid_prefix_guard import Candidate, Resolution, ResolverUnavailable

from reconciliation.harness_fixtures import (
    make_harness,
    make_harness_config,
    make_memory_service,
    open_event_buffer,
    open_journal,
)

FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
PREVIEW = 'the dead repair'


def _memory_service_over_fake_stores() -> Any:
    service = make_memory_service()
    service.mem0 = FakeMem0({FULL: {'data': PREVIEW}})
    service.graphiti = FakeGraphiti()
    service.find_mem0_deletion_tombstones = RecordingTombstones()
    service.config.mem0.collection_prefix = 'fused'
    service.config.queue.backend_read_timeout_seconds = 5.0
    return service


@asynccontextmanager
async def _captured_server_kwargs(tmp_path: Path, service: Any) -> AsyncIterator[dict[str, Any]]:
    """Start and stop the harness's escalation server; yield what it was built with."""
    async with (
        open_journal(tmp_path / 'journal') as journal,
        open_event_buffer(tmp_path / 'buffer.db') as event_buffer,
    ):
        harness = make_harness(
            journal=journal,
            event_buffer=event_buffer,
            taskmaster=AsyncMock(),
            memory_service=service,
            config=make_harness_config(explore_codebase_root=str(tmp_path), escalation_queue_dir='esc'),
        )
        with (
            patch(
                'fused_memory.reconciliation.harness.create_escalation_server',
                return_value=MagicMock(run_http_async=AsyncMock()),
            ) as mock_create,
            patch('fused_memory.reconciliation.harness._sleep', AsyncMock()),
            patch('fused_memory.reconciliation.harness.HAS_ESCALATION', True),
        ):
            try:
                await harness._start_escalation_server()
            finally:
                await harness._stop_escalation_server()
        assert mock_create.called, '_start_escalation_server did not reach create_escalation_server'
        yield dict(mock_create.call_args.kwargs)


def _prefix_resolver(kwargs: dict[str, Any]) -> Callable[[str, str], Any]:
    assert 'prefix_resolver' in kwargs, (
        f'prefix_resolver not passed to create_escalation_server; got {sorted(kwargs)}'
    )
    return kwargs['prefix_resolver']


@pytest.mark.asyncio
async def test_the_resolver_reads_the_harness_memory_service_in_process(tmp_path: Path):
    service = _memory_service_over_fake_stores()

    async with _captured_server_kwargs(tmp_path, service) as kwargs:
        resolution = await _prefix_resolver(kwargs)('reify', 'bff81530')

    assert resolution == Resolution('unique', (Candidate('mem0', FULL, PREVIEW),))


@pytest.mark.asyncio
async def test_a_store_outage_still_raises_through_the_wiring(tmp_path: Path):
    service = _memory_service_over_fake_stores()

    async with _captured_server_kwargs(tmp_path, service) as kwargs:
        service.graphiti = FakeGraphiti(error=ConnectionError('falkordb down'))
        with pytest.raises(ResolverUnavailable) as raised:
            await _prefix_resolver(kwargs)('reify', 'bff81530')

    assert raised.value.store == 'graphiti'


@pytest.mark.asyncio
async def test_the_guard_it_feeds_is_unscoped_on_the_reconciliation_store(tmp_path: Path):
    async with _captured_server_kwargs(tmp_path, _memory_service_over_fake_stores()) as kwargs:
        identity = kwargs['store_identity']

    assert identity.kind == 'reconciliation'
    assert identity.project_id is None
