"""The reconciliation escalation server (8103) is handed the in-process prefix resolver, and scoped per call.

PRD ``plans/uuid-prefix-resolution-prd.md`` leaf δ, D7: this server runs inside
fused-memory, so its uuid-prefix guard resolves through the real IdResolver
over the harness's own memory service, with no HTTP hop. The captured
``prefix_resolver`` is driven against fake stores.

Each call is scoped to one project (task 6910): on the server side through the
harness's known projects, and on the client side through the stage's
X-Escalation-Project header.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _id_resolver_fakes import FakeGraphiti, FakeMem0, RecordingTombstones
from shared.escalation_project_scope import escalation_project_headers
from shared.uuid_prefix_guard import Candidate, Resolution, ResolverUnavailable

from reconciliation.consolidator_fixtures import make_consolidator
from reconciliation.escalation_server_capture import create_escalation_server_kwargs
from reconciliation.harness_fixtures import make_harness_config, make_memory_service, open_harness

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


async def _server_kwargs(tmp_path: Path, service: Any, **harness_kwargs: Any) -> dict[str, Any]:
    async with open_harness(
        tmp_path,
        taskmaster=AsyncMock(),
        memory_service=service,
        config=make_harness_config(explore_codebase_root=str(tmp_path), escalation_queue_dir='esc'),
        **harness_kwargs,
    ) as harness:
        return await create_escalation_server_kwargs(harness)


async def _prefix_resolver(tmp_path: Path, service: Any) -> Callable[[str, str], Any]:
    kwargs = await _server_kwargs(tmp_path, service)
    assert 'prefix_resolver' in kwargs, (
        f'prefix_resolver not passed to create_escalation_server; got {sorted(kwargs)}'
    )
    return kwargs['prefix_resolver']


@pytest.mark.asyncio
async def test_the_resolver_reads_the_harness_memory_service_in_process(tmp_path: Path):
    resolve_prefix = await _prefix_resolver(tmp_path, _memory_service_over_fake_stores())

    resolution = await resolve_prefix('reify', 'bff81530')

    assert resolution == Resolution('unique', (Candidate('mem0', FULL, PREVIEW),))


@pytest.mark.asyncio
async def test_a_store_outage_still_raises_through_the_wiring(tmp_path: Path):
    service = _memory_service_over_fake_stores()
    resolve_prefix = await _prefix_resolver(tmp_path, service)

    service.graphiti = FakeGraphiti(error=ConnectionError('falkordb down'))
    with pytest.raises(ResolverUnavailable) as raised:
        await resolve_prefix('reify', 'bff81530')

    assert raised.value.store == 'graphiti'


@pytest.mark.asyncio
async def test_the_server_may_be_scoped_to_every_known_project(tmp_path: Path):
    kwargs = await _server_kwargs(
        tmp_path, make_memory_service(), known_projects={'reify': '/r', 'dark_factory': '/d'},
    )

    assert 'per_call_projects' in kwargs, (
        f'per_call_projects not passed to create_escalation_server; got {sorted(kwargs)}'
    )
    assert frozenset(kwargs['per_call_projects']) == {'reify', 'dark_factory'}


def test_a_stage_sends_its_project_on_the_escalation_connection():
    stage = make_consolidator(project_id='reify')
    stage._escalation_url = 'http://127.0.0.1:8103/mcp'

    entry = stage._build_mcp_config()['mcpServers']['escalation']

    assert entry['headers'] == escalation_project_headers('reify')
    assert entry['url'] == 'http://127.0.0.1:8103/mcp'
