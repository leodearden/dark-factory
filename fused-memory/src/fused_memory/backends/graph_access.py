"""The bounded Cypher egress: every raw FalkorDB graph call runs under a named budget.

A caller resolves a ``GraphHandle`` for a group id through a ``GraphAccess``
and names the operation on every call. The handle applies the read or write
budget through ``fused_memory/backends/timeout_budget.py::await_within``,
which owns the error-text contract.

It imports no graphiti_core and no falkordb at runtime, so a read-only seam
can adopt it without loading the driver.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from fused_memory.backends import paged_read
from fused_memory.backends.paged_read import (
    DEFAULT_READ_PAGE_SIZE,
    MAX_READ_PAGES,
    RESULTSET_SIZE,
    PagedRead,
)
from fused_memory.backends.timeout_budget import await_within

if TYPE_CHECKING:
    from graphiti_core.driver.falkordb_driver import FalkorDriver

__all__ = [
    'STORE',
    'BudgetedGraph',
    'FalkorGraphAccess',
    'GraphAccess',
    'GraphHandle',
    'OpBoundReader',
    'RawGraph',
    'paged_ro_query',
]

STORE = 'Graphiti'
"""The store every raw-Cypher timeout names."""

_COMPACT_READ_COMMAND = 'GRAPH.RO_QUERY'
_COMPACT_FLAG = '--compact'


class RawGraph(Protocol):
    """falkordb's ``AsyncGraph`` call shape: each call returns an awaitable, which the handle budgets."""

    def query(self, q: str, params: dict[str, Any] | None = None) -> Awaitable[Any]: ...

    def ro_query(self, q: str, params: dict[str, Any] | None = None) -> Awaitable[Any]: ...

    def execute_command(self, *args: Any) -> Awaitable[Any]: ...


class GraphHandle(Protocol):
    """One group's graph, every call budgeted and named by its ``op``."""

    __slots__ = ()

    @property
    def name(self) -> str: ...

    async def query(
        self, cypher: str, params: dict[str, Any] | None = None, *, op: str,
    ) -> Any: ...

    async def ro_query(
        self, cypher: str, params: dict[str, Any] | None = None, *, op: str,
    ) -> Any: ...

    async def ro_query_compact(self, cypher: str, *, op: str) -> Any:
        """A compact-transport read; values are inlined, because this transport binds no params."""
        ...


class GraphAccess(Protocol):
    """Resolves a group id to its budgeted graph."""

    __slots__ = ()

    def graph(self, group_id: str) -> GraphHandle: ...

    async def list_graphs(self) -> list[str]: ...


class BudgetedGraph(GraphHandle):
    """A raw graph whose writes carry the write budget and whose reads carry the read budget.

    The raw graph sees exactly the call shape it saw before budgeting:
    ``(cypher)`` without params, ``(cypher, params)`` with them. ``op`` never
    reaches it.
    """

    __slots__ = ('_name', '_raw', '_read_budget_s', '_write_budget_s')

    def __init__(
        self, raw: RawGraph, *, name: str, read_budget_s: float, write_budget_s: float,
    ) -> None:
        self._raw = raw
        self._name = name
        self._read_budget_s = read_budget_s
        self._write_budget_s = write_budget_s

    @property
    def name(self) -> str:
        return self._name

    async def query(
        self, cypher: str, params: dict[str, Any] | None = None, *, op: str,
    ) -> Any:
        return await await_within(
            self._raw.query(*_raw_args(cypher, params)),
            store=STORE, op=op, budget_s=self._write_budget_s,
        )

    async def ro_query(
        self, cypher: str, params: dict[str, Any] | None = None, *, op: str,
    ) -> Any:
        return await await_within(
            self._raw.ro_query(*_raw_args(cypher, params)),
            store=STORE, op=op, budget_s=self._read_budget_s,
        )

    async def ro_query_compact(self, cypher: str, *, op: str) -> Any:
        return await await_within(
            self._raw.execute_command(_COMPACT_READ_COMMAND, self._name, cypher, _COMPACT_FLAG),
            store=STORE, op=op, budget_s=self._read_budget_s,
        )


def _raw_args(cypher: str, params: dict[str, Any] | None) -> tuple[Any, ...]:
    return (cypher,) if params is None else (cypher, params)


class FalkorGraphAccess(GraphAccess):
    """The production provider: each group's graph from the driver's one connection.

    graphiti_core annotates ``FalkorDriver._get_graph`` as returning falkordb's
    synchronous ``Graph``, although the asyncio connection returns an
    ``AsyncGraph``; task 9189 resolves the graph through
    ``driver.client.select_graph`` instead and drops the suppression below.
    """

    __slots__ = ('_driver', '_read_budget_s', '_write_budget_s')

    def __init__(
        self, driver: FalkorDriver, *, read_budget_s: float, write_budget_s: float,
    ) -> None:
        self._driver = driver
        self._read_budget_s = read_budget_s
        self._write_budget_s = write_budget_s

    def graph(self, group_id: str) -> GraphHandle:
        return BudgetedGraph(
            self._driver._get_graph(group_id),  # pyright: ignore[reportArgumentType]  # debt: task 9189
            name=group_id,
            read_budget_s=self._read_budget_s,
            write_budget_s=self._write_budget_s,
        )

    async def list_graphs(self) -> list[str]:
        return await await_within(
            self._driver.client.list_graphs(),
            store=STORE, op='list_graphs', budget_s=self._read_budget_s,
        )


@dataclass(frozen=True, slots=True)
class OpBoundReader:
    """A raw-shaped read view of a handle with its ``op`` fixed, for graph-agnostic readers."""

    handle: GraphHandle
    op: str

    @property
    def name(self) -> str:
        return self.handle.name

    async def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        return await self.handle.ro_query(cypher, params, op=self.op)


async def paged_ro_query(
    graph: GraphHandle,
    page_template: str,
    census_cypher: str,
    *,
    op: str,
    params: dict[str, Any] | None = None,
    page_size: int = DEFAULT_READ_PAGE_SIZE,
    resultset_size: int = RESULTSET_SIZE,
    max_pages: int = MAX_READ_PAGES,
) -> PagedRead:
    """``paged_read.paged_ro_query`` over *graph*, the census and every page budgeted under *op*."""
    return await paged_read.paged_ro_query(
        OpBoundReader(graph, op),
        page_template,
        census_cypher,
        params=params,
        page_size=page_size,
        resultset_size=resultset_size,
        max_pages=max_pages,
    )
