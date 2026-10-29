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

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

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
