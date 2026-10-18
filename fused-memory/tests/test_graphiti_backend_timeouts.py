"""GraphitiBackend's timeouts propagate and name themselves (task 5699).

Each row hangs one backend call's upstream await through a public or
third-party seam, under a short budget set through the config, and expects a
``TimeoutError`` naming the store, the method and the budget, never a falsy
return that would read as an empty store or a genuine not-found.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _graphiti_fake import FakeGraphitiClient
from graphiti_core.edges import EntityEdge
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
    variant: str = ''

    @property
    def id(self) -> str:
        return f'{self.op}-{self.variant}' if self.variant else self.op


@contextmanager
def _all_of(*patches: AbstractContextManager) -> Iterator[None]:
    with ExitStack() as stack:
        for each in patches:
            stack.enter_context(each)
        yield


def _graph_hanging_on(attr: str) -> Callable[[GraphitiBackend], AbstractContextManager]:
    """Serve a FalkorDB graph whose *attr* query hangs; the other answers one episode's content."""
    def _serve(b: GraphitiBackend) -> AbstractContextManager:
        graph = MagicMock()
        graph.ro_query = AsyncMock(return_value=MagicMock(result_set=[['old content']]))
        graph.query = AsyncMock()
        setattr(graph, attr, AsyncMock(side_effect=_hang))
        return patch.object(b._driver, '_get_graph', MagicMock(return_value=graph))
    return _serve


def _redact(b: GraphitiBackend) -> Awaitable:
    return b.redact_episode_content('ep-1', group_id=_GROUP, new_content='clean text')


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
    _TimedCall(
        'redact_episode_content.read', _graph_hanging_on('ro_query'), _redact,
    ),
)


def _loaded_with_hanging(attr: str) -> MagicMock:
    """A graph object whose *attr* write hangs, as ``get_by_uuid`` would load it."""
    loaded = MagicMock()
    setattr(loaded, attr, AsyncMock(side_effect=_hang))
    return loaded


_WRITES = (
    _TimedCall(
        'build_communities',
        lambda b: patch.object(b.client, 'build_communities', AsyncMock(side_effect=_hang)),
        lambda b: b.build_communities(None),
        variant='all',
    ),
    _TimedCall(
        'build_communities',
        lambda b: patch.object(b.client, 'build_communities', AsyncMock(side_effect=_hang)),
        lambda b: b.build_communities([_GROUP]),
        variant='per-group',
    ),
    _TimedCall(
        'remove_episode',
        lambda b: patch.object(
            EpisodicNode, 'get_by_uuid', AsyncMock(return_value=_loaded_with_hanging('delete')),
        ),
        lambda b: b.remove_episode('ep-1', group_id=_GROUP),
    ),
    _TimedCall(
        'remove_edge',
        lambda b: patch.object(
            EntityEdge, 'get_by_uuid', AsyncMock(return_value=_loaded_with_hanging('delete')),
        ),
        lambda b: b.remove_edge('edge-1', group_id=_GROUP),
    ),
    _TimedCall(
        'update_edge.save',
        lambda b: patch.object(
            EntityEdge, 'get_by_uuid', AsyncMock(return_value=_loaded_with_hanging('save')),
        ),
        lambda b: b.update_edge('edge-1', group_id=_GROUP, invalid_at=datetime.now(UTC)),
    ),
    _TimedCall(
        'update_edge.clear_invalid_at',
        lambda b: _all_of(
            patch.object(
                EntityEdge, 'get_by_uuid', AsyncMock(return_value=MagicMock(save=AsyncMock())),
            ),
            _graph_hanging_on('query')(b),
        ),
        lambda b: b.update_edge('edge-1', group_id=_GROUP, clear_invalid_at=True),
    ),
    _TimedCall(
        'redact_episode_content.write', _graph_hanging_on('query'), _redact,
    ),
)


class _HangingAddEpisodeGraphitiClient(FakeGraphitiClient):
    async def add_episode(self, **kwargs):
        await asyncio.Event().wait()


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
@pytest.mark.parametrize('timed', _READS, ids=lambda timed: timed.id)
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


@pytest.fixture
def write_budgeted_backend(mock_config, make_backend) -> GraphitiBackend:
    mock_config.queue.backend_write_timeout_seconds = _BUDGET_S
    return make_backend(mock_config)


@pytest.mark.asyncio
@pytest.mark.parametrize('timed', _WRITES, ids=lambda timed: timed.id)
async def test_a_write_timeout_names_store_op_and_budget(write_budgeted_backend, timed):
    await _assert_named_timeout(write_budgeted_backend, timed)


@pytest.mark.asyncio
async def test_an_add_episode_timeout_names_store_op_and_budget(
    mock_config, make_backend_over_fake_graphiti,
):
    mock_config.queue.backend_write_timeout_seconds = _BUDGET_S
    backend = make_backend_over_fake_graphiti(mock_config, _HangingAddEpisodeGraphitiClient())
    add_episode = _TimedCall(
        'add_episode',
        lambda b: nullcontext(),
        lambda b: b.add_episode('n', 'body', group_id=_GROUP),
    )
    await _assert_named_timeout(backend, add_episode)
