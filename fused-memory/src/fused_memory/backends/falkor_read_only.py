"""The read-only FalkorDB seam: every read goes over ``GRAPH.RO_QUERY``, and nothing else can be issued through it.

Read-only here is SERVER-enforced, not client-promised: a ``CREATE`` sent as
``GRAPH.RO_QUERY`` is refused by a live FalkorDB and materializes nothing.

A reader on this seam must never reach the store through
``graphiti_core.driver.falkordb_driver.FalkorDriver``, whose ``__init__``
fire-and-forgets ``build_indices_and_constraints()`` — a WRITE issued before
the reader has read anything. Nor may it go through the handles of
``fused_memory/services/memory_service.py::MemoryService`` or
``fused_memory/backends/graphiti_client.py::GraphitiBackend``, which can write.

:class:`ReadOnlyGraphProxy` makes the client-side guard load-bearing. The
copies it replaces called ``assert_read_only_command(RO_COMMAND)``, a constant
compared with itself, so the guard could never fire. The proxy asserts at the
one seam where the choice of primitive is observable: every attribute a reader
resolves on the handle passes through it, and a primitive mapped to a write
command raises :class:`NonReadOnlyCommandError` before any byte reaches the
store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypedDict

from falkordb.asyncio import FalkorDB

from fused_memory.backends.graphiti_client import (
    _DEFAULT_READ_PAGE_SIZE,
    _RESULTSET_SIZE,
    PagedRead,
    _paged_ro_query,
)

logger = logging.getLogger(__name__)

RO_COMMAND = 'GRAPH.RO_QUERY'

COMMAND_BY_PRIMITIVE = MappingProxyType({
    'ro_query': 'GRAPH.RO_QUERY',
    'query': 'GRAPH.QUERY',
})


class NonReadOnlyCommandError(RuntimeError):
    def __init__(self, command: str) -> None:
        self.command = command
        super().__init__(
            f'{command!r} refused: only {RO_COMMAND} may be issued through '
            f'the read-only graph seam'
        )


def assert_read_only_command(command: str) -> None:
    if command != RO_COMMAND:
        raise NonReadOnlyCommandError(command)


class ReadOnlyGraphProxy:
    """A graph handle whose write primitive is refused and whose bookkeeping is forwarded untouched."""

    __slots__ = ('_graph',)

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def __getattr__(self, name: str) -> Any:
        command = COMMAND_BY_PRIMITIVE.get(name)
        if command is not None:
            assert_read_only_command(command)
        return getattr(self._graph, name)


@dataclass(frozen=True)
class PagedQuery:
    """One population, read in pages ordered on ``<row>.uuid`` and proven complete by ``count(<row>)``.

    Both strings are built from the one ``match``, so the page and the census
    always describe the same population, as
    ``fused_memory/backends/graphiti_client.py::_paged_ro_query`` requires.

    PRECONDITION: the MATCH yields exactly one result row per binding of
    ``row`` — a directed edge pattern with ``row`` the relationship, or a
    single node label with ``row`` the node. That is what makes ``<row>.uuid``
    a total order over the rows and ``count(<row>)`` their number.

    The sort key must be projected. A driver may refuse to order on a property
    the query does not return, and a reader or a test double can only audit
    which rows a page returned through a returned identifying column.

    The census counts ``row``, never ``*``. With the relationship variable
    unreferenced, FalkorDB's ``count(*)`` over an edge pattern counts connected
    (source, target) node pairs, not edges. Measured on dark_factory
    2026-10-01: 19,409 pairs against 20,491 edges, so a ``count(*)`` census
    would pass a read truncated by exactly the multi-edge excess. See
    docs/ruling-overreach-guard-design-2026-10-01/design.md section 2.1.
    """

    match: str
    row: str
    returns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.returns:
            raise ValueError(
                f'PagedQuery needs at least one returned column, got returns={self.returns!r}'
            )
        if not self.row.isidentifier():
            raise ValueError(
                f'PagedQuery row must be a bare Cypher variable, got row={self.row!r}'
            )
        if self.sort_key not in self.returns:
            raise ValueError(
                f'PagedQuery pages on {self.sort_key!r}, which must be projected, '
                f'but returns={self.returns!r} omits it'
            )

    @property
    def sort_key(self) -> str:
        return f'{self.row}.uuid'

    @property
    def page_template(self) -> str:
        return (
            f"{self.match.strip()} RETURN {', '.join(self.returns)} "
            f'ORDER BY {self.sort_key} SKIP {{skip}} LIMIT {{limit}}'
        )

    @property
    def census_cypher(self) -> str:
        return f'{self.match.strip()} RETURN count({self.row})'


class _HostPort(TypedDict, total=False):
    host: str
    port: int


def open_graph(
    graph_name: str,
    *,
    uri: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> Any:
    """Open a ``falkordb.asyncio`` handle on *graph_name*, addressed by a uri OR by host/port.

    Opening is not free, which is why readers open on first read rather than
    at construction. ``FalkorDB.__init__`` runs a SYNCHRONOUS ``INFO server``
    (``falkordb/cluster.py::Is_Cluster``) against its host and port, so a
    refused address raises right here. ``FalkorDB.from_url`` calls ``cls()``
    before it applies the url, so a uri open first probes localhost:6379.
    That is why the host/port form exists beside the uri one: rewriting a
    host/port address as a ``redis://`` url would add a hidden dependency on
    a localhost server.
    """
    if uri is not None and (host is not None or port is not None):
        raise ValueError(
            f'open_graph takes a uri OR a host/port, not both: '
            f'uri={uri!r} host={host!r} port={port!r}'
        )
    if uri is not None:
        client = FalkorDB.from_url(uri)
    else:
        address: _HostPort = {}
        if host is not None:
            address['host'] = host
        if port is not None:
            address['port'] = port
        client = FalkorDB(**address)
    return client.select_graph(graph_name)


class ReadOnlyGraphReader:
    """Reads one graph through the guarded handle, paged and census-proven or as one capped query.

    An injected *graph* is wrapped at construction; otherwise the handle is
    opened by :func:`open_graph` on the first read. A None paging knob means
    the default in ``fused_memory/backends/graphiti_client.py``.
    """

    def __init__(
        self,
        *,
        graph_name: str,
        graph: Any | None = None,
        uri: str | None = None,
        host: str | None = None,
        port: int | None = None,
        page_size: int | None = None,
        resultset_size: int | None = None,
    ) -> None:
        self.graph_name = graph_name
        self._uri = uri
        self._host = host
        self._port = port
        self._page_size = _DEFAULT_READ_PAGE_SIZE if page_size is None else page_size
        self._resultset_size = _RESULTSET_SIZE if resultset_size is None else resultset_size
        self._guarded = None if graph is None else ReadOnlyGraphProxy(graph)

    def guarded_graph(self) -> ReadOnlyGraphProxy:
        if self._guarded is None:
            graph = open_graph(self.graph_name, uri=self._uri, host=self._host, port=self._port)
            self._guarded = ReadOnlyGraphProxy(graph)
        return self._guarded

    async def read_paged(
        self, query: PagedQuery, *, params: dict[str, Any] | None = None
    ) -> PagedRead:
        return await _paged_ro_query(
            self.guarded_graph(),
            query.page_template,
            query.census_cypher,
            params=params,
            page_size=self._page_size,
            resultset_size=self._resultset_size,
        )

    async def read_unpaged(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[list]:
        """One query, whose answer the server truncates silently at its row cap; :meth:`read_paged` is the complete read."""
        result = await self.guarded_graph().ro_query(cypher, params)
        rows = list(getattr(result, 'result_set', None) or [])
        if len(rows) >= self._resultset_size:
            logger.warning(
                'read_unpaged on graph %r returned %d rows, at or above the assumed '
                'server cap resultset_size=%d: the server may have truncated it '
                'silently. read_paged is the complete alternative.',
                self.graph_name,
                len(rows),
                self._resultset_size,
            )
        return rows
