"""Unit tests for FalkorGraphOperations, the graphiti ``graph_operations_interface`` for FalkorDB (task 7714).

graphiti-core dispatches six graph operations to the interface with no
NotImplementedError fallback: the four bulk saves on the ``add_episode`` write
path and the two bulk embedding loads.  FalkorGraphOperations must hand each
back to graphiti's built-in behaviour, or wiring it breaks every write.

No connection is opened here: every driver is an unconnected driver from
``_query_recorder``, so query assembly and dispatch are graphiti's real code.
The stock oracle is a bare graphiti ``FalkorDriver``, which has no
``graph_operations_interface`` and so takes graphiti's built-in path.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import graphiti_core
import pytest
from _ast_guard import parse_python_module
from _query_recorder import IssuedQuery, recording_driver
from graphiti_core.driver.driver import GraphDriver, GraphDriverSession
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search import search_utils
from graphiti_core.utils import bulk_utils

from fused_memory.backends.falkor_graph_operations import FalkorGraphOperations
from fused_memory.backends.graphiti_client import _MultiTenantFalkorDriver

if TYPE_CHECKING:
    from falkordb.asyncio import FalkorDB


def hardened_driver(
    records: list[dict[str, Any]] | None = None,
) -> tuple[_MultiTenantFalkorDriver, list[IssuedQuery]]:
    driver, issued = recording_driver(_MultiTenantFalkorDriver, records or [])
    assert isinstance(driver.graph_operations_interface, FalkorGraphOperations)
    return driver, issued


def stock_driver(
    records: list[dict[str, Any]] | None = None,
) -> tuple[FalkorDriver, list[IssuedQuery]]:
    driver, issued = recording_driver(FalkorDriver, records or [])
    assert driver.graph_operations_interface is None
    return driver, issued


def declared_operations(interface: type[GraphOperationsInterface]) -> list[str]:
    """The graph operations *interface* itself defines."""
    return sorted(
        name for name, member in vars(interface).items() if inspect.iscoroutinefunction(member)
    )


OVERRIDDEN_OPERATIONS = declared_operations(FalkorGraphOperations)


class TestDriverWiring:
    """Every per-group read and write driver comes from ``clone()``, so it must carry the seam."""

    def test_driver_class_carries_falkor_graph_operations(self) -> None:
        assert isinstance(_MultiTenantFalkorDriver.graph_operations_interface, FalkorGraphOperations)

    def test_clone_carries_falkor_graph_operations(self) -> None:
        # Upstream ``__init__`` uses a supplied ``falkor_db`` directly instead of
        # dialling out; ``cast`` is for pyright only.
        stub_client = cast('FalkorDB', object())
        driver = _MultiTenantFalkorDriver(falkor_db=stub_client, database='graph_a')

        cloned = driver.clone('graph_b')

        assert isinstance(cloned, _MultiTenantFalkorDriver)
        assert isinstance(cloned.graph_operations_interface, FalkorGraphOperations)


# --- Embedding loads -----------------------------------------------------------

EmbeddingLoad = Callable[[GraphDriver], Awaitable[dict[str, list[float]]]]

# Each load through its graphiti-core caller, with one row shaped like its result.
EMBEDDING_LOADS: dict[str, tuple[EmbeddingLoad, dict[str, Any]]] = {
    'node_load_embeddings_bulk': (
        lambda driver: search_utils.get_embeddings_for_nodes(
            driver, [EntityNode(uuid='node-uuid', name='node', group_id='g')]
        ),
        {'uuid': 'node-uuid', 'name_embedding': [0.1, 0.2, 0.3]},
    ),
    'edge_load_embeddings_bulk': (
        lambda driver: search_utils.get_embeddings_for_edges(
            driver,
            [
                EntityEdge(
                    uuid='edge-uuid',
                    group_id='g',
                    source_node_uuid='source-uuid',
                    target_node_uuid='target-uuid',
                    created_at=datetime(2026, 10, 29, tzinfo=UTC),
                    name='RELATES_TO',
                    fact='alpha beta',
                )
            ],
        ),
        {'uuid': 'edge-uuid', 'fact_embedding': [0.1, 0.2, 0.3]},
    ),
}


class TestEmbeddingLoadsMatchStock:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('operation', sorted(EMBEDDING_LOADS))
    async def test_issues_graphitis_builtin_query(self, operation: str) -> None:
        """No NotImplementedError, no recursion back into the seam, stock's one query."""
        load, record = EMBEDDING_LOADS[operation]
        hardened, issued_by_hardened = hardened_driver([record])
        stock, issued_by_stock = stock_driver([record])

        loaded = await load(hardened)
        expected = await load(stock)

        assert len(issued_by_stock) == 1
        assert issued_by_hardened == issued_by_stock
        assert loaded == expected != {}


# --- Bulk saves ----------------------------------------------------------------

BULK_SAVED_AT = datetime(2026, 10, 29, tzinfo=UTC)


class RecordingTransaction(GraphDriverSession):
    """A transaction that records each ``run`` instead of executing it."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, Any]]] = []

    async def run(self, query: str, **kwargs: Any) -> None:
        self.runs.append((query, kwargs))

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass

    async def close(self) -> None:
        pass

    async def execute_write(self, func, *args, **kwargs):
        return await func(self, *args, **kwargs)


class UnusedEmbedder(EmbedderClient):
    """Every model's embedding is preset, so graphiti must never embed."""

    async def create(self, input_data: Any) -> list[float]:
        raise AssertionError('embeddings are preset; graphiti must not call the embedder')


BulkModels = tuple[list[EpisodicNode], list[EpisodicEdge], list[EntityNode], list[EntityEdge]]


def bulk_models() -> BulkModels:
    """Fresh models with fixed uuids and datetimes, so two runs record equal calls."""
    episode = EpisodicNode(
        uuid='episode-uuid',
        name='episode',
        group_id='g',
        created_at=BULK_SAVED_AT,
        source=EpisodeType.text,
        source_description='unit test',
        content='alpha beta',
        valid_at=BULK_SAVED_AT,
    )
    source = EntityNode(
        uuid='source-uuid',
        name='source',
        group_id='g',
        created_at=BULK_SAVED_AT,
        name_embedding=[0.1, 0.2, 0.3],
    )
    target = EntityNode(
        uuid='target-uuid',
        name='target',
        group_id='g',
        created_at=BULK_SAVED_AT,
        name_embedding=[0.3, 0.2, 0.1],
    )
    mention = EpisodicEdge(
        uuid='mention-uuid',
        group_id='g',
        source_node_uuid=episode.uuid,
        target_node_uuid=source.uuid,
        created_at=BULK_SAVED_AT,
    )
    fact = EntityEdge(
        uuid='edge-uuid',
        group_id='g',
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        created_at=BULK_SAVED_AT,
        name='RELATES_TO',
        fact='alpha beta',
        fact_embedding=[0.1, 0.2, 0.3],
        episodes=[episode.uuid],
        valid_at=BULK_SAVED_AT,
    )
    return [episode], [mention], [source, target], [fact]


async def bulk_write_runs(driver: GraphDriver) -> list[tuple[str, dict[str, Any]]]:
    """The transaction calls graphiti's ``add_episode`` bulk write makes through *driver*."""
    transaction = RecordingTransaction()
    episodic_nodes, episodic_edges, entity_nodes, entity_edges = bulk_models()
    await bulk_utils.add_nodes_and_edges_bulk_tx(
        transaction,
        episodic_nodes,
        episodic_edges,
        entity_nodes,
        entity_edges,
        UnusedEmbedder(),
        driver=driver,
    )
    return transaction.runs


class TestBulkWriteMatchesStock:
    @pytest.mark.asyncio
    async def test_runs_graphitis_builtin_bulk_saves(self) -> None:
        hardened, _ = hardened_driver()
        stock, _ = stock_driver()

        hardened_runs = await bulk_write_runs(hardened)
        stock_runs = await bulk_write_runs(stock)

        assert stock_runs != []
        assert hardened_runs == stock_runs


# --- Upgrade guard -------------------------------------------------------------


def _catches_not_implemented(handler: ast.ExceptHandler) -> bool:
    caught = handler.type
    names = caught.elts if isinstance(caught, ast.Tuple) else [caught]
    return any(isinstance(name, ast.Name) and name.id == 'NotImplementedError' for name in names)


def unguarded_dispatches(node: ast.AST, guarded: bool = False) -> set[str]:
    """Names of ``<x>.graph_operations_interface.<name>(...)`` calls with no NotImplementedError fallback.

    A call is guarded when it sits in the body of a ``try`` with a handler
    catching NotImplementedError; its handlers, ``else`` and ``finally`` are not.
    """
    if isinstance(node, ast.Try | ast.TryStar):
        body_guarded = guarded or any(_catches_not_implemented(h) for h in node.handlers)
        found: set[str] = set()
        for child in node.body:
            found |= unguarded_dispatches(child, body_guarded)
        for child in [*node.handlers, *node.orelse, *node.finalbody]:
            found |= unguarded_dispatches(child, guarded)
        return found

    found = set()
    if (
        not guarded
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == 'graph_operations_interface'
    ):
        found.add(node.func.attr)
    for child in ast.iter_child_nodes(node):
        found |= unguarded_dispatches(child, guarded)
    return found


def graphiti_unguarded_dispatches() -> set[str]:
    graphiti_root = pathlib.Path(graphiti_core.__file__).parent
    found: set[str] = set()
    for path in sorted(graphiti_root.rglob('*.py')):
        found |= unguarded_dispatches(parse_python_module(path))
    return found


def test_every_unguarded_dispatch_is_overridden() -> None:
    """An upstream dispatch without a fallback would raise the inherited NotImplementedError.

    graphiti-core has no upper version bound, so a new one must fail here, not in production.
    """
    unguarded = graphiti_unguarded_dispatches()

    assert unguarded, 'the scan found no dispatch at all; it can no longer see graphiti'
    assert unguarded <= set(OVERRIDDEN_OPERATIONS), sorted(unguarded - set(OVERRIDDEN_OPERATIONS))


@pytest.mark.parametrize('method', OVERRIDDEN_OPERATIONS)
def test_override_signature_matches_graph_operations_interface(method: str) -> None:
    """graphiti calls these positionally; an upstream reorder must fail here, not bind silently."""
    assert inspect.signature(getattr(FalkorGraphOperations, method)) == inspect.signature(
        getattr(GraphOperationsInterface, method)
    )
