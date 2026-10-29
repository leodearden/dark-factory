"""Tests for fused_memory/backends/falkor_read_only.py, the read-only FalkorDB seam."""
from __future__ import annotations

import pytest

from fused_memory.backends.falkor_read_only import (
    COMMAND_BY_PRIMITIVE,
    RO_COMMAND,
    NonReadOnlyCommandError,
    ReadOnlyGraphProxy,
    assert_read_only_command,
)


class _RecordingGraph:
    """A graph handle that records reads and refuses writes."""

    def __init__(self, *, resultset_cap: int = 10) -> None:
        self.resultset_cap = resultset_cap
        self.queries: list[tuple[str, dict | None]] = []
        self.write_calls: list[str] = []

    async def ro_query(self, cypher: str, params: dict | None = None) -> None:
        self.queries.append((cypher, params))

    async def query(self, cypher: str, params: dict | None = None) -> None:
        self.write_calls.append(cypher)
        raise AssertionError('the read-only seam may never issue query()')


class TestTheCommandGuard:
    def test_the_read_only_command_passes(self) -> None:
        assert RO_COMMAND == 'GRAPH.RO_QUERY'
        assert assert_read_only_command(RO_COMMAND) is None

    def test_a_writable_command_is_refused_with_a_typed_error(self) -> None:
        with pytest.raises(NonReadOnlyCommandError) as caught:
            assert_read_only_command('GRAPH.QUERY')
        error = caught.value
        assert isinstance(error, RuntimeError)
        assert error.command == 'GRAPH.QUERY'
        message = str(error)
        assert 'GRAPH.QUERY' in message
        assert 'GRAPH.RO_QUERY' in message
        assert 'read-only' in message

    def test_every_falkordb_primitive_the_proxy_knows_is_mapped(self) -> None:
        """A primitive absent from the map is forwarded UNGUARDED, so the map is what can go quietly wrong."""
        assert COMMAND_BY_PRIMITIVE['ro_query'] == RO_COMMAND
        assert COMMAND_BY_PRIMITIVE['query'] == 'GRAPH.QUERY'
        with pytest.raises(TypeError):
            COMMAND_BY_PRIMITIVE['query'] = RO_COMMAND  # type: ignore[index]


class TestReadOnlyGraphProxy:
    def test_the_read_primitive_resolves_to_the_wrapped_graph(self) -> None:
        graph = _RecordingGraph()
        assert ReadOnlyGraphProxy(graph).ro_query.__self__ is graph

    def test_the_command_guard_is_reachable_on_a_real_read(self) -> None:
        """The write primitive is refused before any byte reaches the store, naming the command rather than the method."""
        graph = _RecordingGraph()
        proxy = ReadOnlyGraphProxy(graph)
        with pytest.raises(NonReadOnlyCommandError, match='GRAPH.QUERY'):
            _ = proxy.query
        assert graph.write_calls == []

    def test_unmapped_attributes_are_forwarded_untouched(self) -> None:
        """A handle legitimately carries bookkeeping; guarding it would make the proxy brittle without making it safer."""
        graph = _RecordingGraph(resultset_cap=7)
        proxy = ReadOnlyGraphProxy(graph)
        assert proxy.queries is graph.queries
        assert proxy.resultset_cap == graph.resultset_cap
