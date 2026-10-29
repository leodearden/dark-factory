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
