"""The two-way wire contract between δ's resolver client and β's POST /resolve-prefix.

The escalation package may not import fused_memory, so the route path and
its field names exist on both sides of a process boundary. This is the one
place both sides can be imported together: escalation's
``PrefixResolverClient`` is driven over ``httpx.ASGITransport`` against the
REAL route, mounted as ``test_resolve_prefix_route.py`` mounts it, over the
real IdResolver on in-memory stores. A drift on either side fails here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from _id_resolver_fakes import FakeGraphiti, FakeMem0, make_resolver
from escalation.prefix_resolver_client import PrefixResolverClient
from shared.uuid_prefix_guard import Resolution, ResolverUnavailable

from fused_memory.server.tools import create_mcp_server

_KNOWN_PROJECTS = {'reify': '/abs/reify'}
_PREFIX = 'bff81530'
_FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
_SIBLING = 'bff81530-0000-4000-8000-000000000002'


class _Wire:
    """β's real route over fake stores, and δ's client pointed at it."""

    def __init__(self, *, mem0: FakeMem0 | None = None, graphiti: FakeGraphiti | None = None) -> None:
        self.resolver = make_resolver(mem0=mem0, graphiti=graphiti)
        server = create_mcp_server(AsyncMock(), id_resolver=self.resolver, known_projects=_KNOWN_PROJECTS)
        self.client = PrefixResolverClient(
            base_url='http://fused-memory.test',
            transport=httpx.ASGITransport(app=server.streamable_http_app()),
        )

    async def over_the_wire(self, project_id: str = 'reify') -> Resolution:
        return await self.client.resolve_prefix(project_id, _PREFIX)

    async def in_process(self) -> Resolution:
        return await self.resolver.resolve_prefix('reify', _PREFIX)


@pytest.mark.asyncio
@pytest.mark.parametrize(('mem0', 'graphiti', 'outcome'), [
    (FakeMem0({_FULL: {'data': 'a memory'}}), None, 'unique'),
    (FakeMem0({_FULL: {'data': 'a memory'}}), FakeGraphiti(nodes={_SIBLING: 'a node'}), 'ambiguous'),
    (None, None, 'none'),
], ids=['unique', 'ambiguous', 'none'])
async def test_every_outcome_round_trips_to_the_in_process_resolution(mem0, graphiti, outcome):
    wire = _Wire(mem0=mem0, graphiti=graphiti)

    expected = await wire.in_process()
    received = await wire.over_the_wire()

    assert expected.outcome == outcome
    assert received == expected


@pytest.mark.asyncio
async def test_the_failing_store_named_by_the_route_survives_the_wire():
    wire = _Wire(graphiti=FakeGraphiti(error=ConnectionError('falkordb down')))

    with pytest.raises(ResolverUnavailable) as raised:
        await wire.over_the_wire()

    assert raised.value.store == 'graphiti'


@pytest.mark.asyncio
async def test_a_rejected_request_is_unavailable_never_a_none():
    wire = _Wire(mem0=FakeMem0({_FULL: {'data': 'a memory'}}))

    with pytest.raises(ResolverUnavailable):
        await wire.over_the_wire(project_id='unknown_project')
