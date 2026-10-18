"""GraphitiBackend's timeouts propagate and name themselves (task 5699).

Each row hangs one backend call's upstream await through a public or
third-party seam, under a short budget set through the config, and expects a
``TimeoutError`` naming the store, the method and the budget, never a falsy
return that would read as an empty store or a genuine not-found.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EpisodicNode

from fused_memory.backends.graphiti_client import GraphitiBackend

_BUDGET_S = 0.05
_GROUP = 'dark_factory'


async def _hang(*args, **kwargs):
    await asyncio.Event().wait()


@dataclass(frozen=True)
class _TimedCall:
    op: str
    hang_upstream: Callable[[GraphitiBackend], AbstractContextManager]
    call: Callable[[GraphitiBackend], Awaitable]


_READS = (
    _TimedCall(
        'search',
        lambda b: patch.object(b.client, 'search', AsyncMock(side_effect=_hang)),
        lambda b: b.search('q', group_ids=[_GROUP]),
    ),
    _TimedCall(
        'search_nodes',
        lambda b: patch.object(b.client, 'search_', AsyncMock(side_effect=_hang)),
        lambda b: b.search_nodes('q', group_ids=[_GROUP]),
    ),
    _TimedCall(
        'retrieve_episodes',
        lambda b: patch.object(EpisodicNode, 'get_by_group_ids', AsyncMock(side_effect=_hang)),
        lambda b: b.retrieve_episodes([_GROUP]),
    ),
    _TimedCall(
        'get_episode_by_uuid',
        lambda b: patch.object(EpisodicNode, 'get_by_uuid', AsyncMock(side_effect=_hang)),
        lambda b: b.get_episode_by_uuid('ep-1', group_id=_GROUP),
    ),
)


@pytest.fixture
def read_budgeted_backend(mock_config, make_backend) -> GraphitiBackend:
    mock_config.queue.backend_read_timeout_seconds = _BUDGET_S
    return make_backend(mock_config)


async def _assert_named_timeout(backend: GraphitiBackend, timed: _TimedCall) -> None:
    with timed.hang_upstream(backend), pytest.raises(TimeoutError) as raised:
        await timed.call(backend)
    message = str(raised.value)
    for part in ('Graphiti', timed.op, 'timed out', str(_BUDGET_S)):
        assert part in message, f'{part!r} missing from the timeout message {message!r}'


@pytest.mark.asyncio
@pytest.mark.parametrize('timed', _READS, ids=lambda timed: timed.op)
async def test_a_read_timeout_propagates_naming_store_op_and_budget(
    read_budgeted_backend, timed,
):
    await _assert_named_timeout(read_budgeted_backend, timed)


@pytest.mark.asyncio
async def test_a_genuine_episode_miss_still_returns_none(read_budgeted_backend):
    """CONTRAST: not-found stays None, so a miss and a timeout remain distinguishable."""
    missing = AsyncMock(side_effect=NodeNotFoundError('ep-1'))
    with patch.object(EpisodicNode, 'get_by_uuid', missing):
        assert await read_budgeted_backend.get_episode_by_uuid('ep-1', group_id=_GROUP) is None
