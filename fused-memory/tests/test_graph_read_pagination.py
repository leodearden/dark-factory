"""Tests for paginated whole-graph reads in GraphitiBackend (task 4340).

FalkorDB truncates every result set at a server-wide ``RESULTSET_SIZE``
ceiling — measured 10000 on 2026-08-17 via ``GRAPH.CONFIG GET
RESULTSET_SIZE``, with nothing in this repo overriding it.  Two whole-graph
reads exceeded it on the live corpus and were therefore returning a silently
short collection:

    graph          Entity nodes   valid-edge rows   unpaginated read returned
    dark_factory       16038            24938                 10000
    reify              23589            31621                 10000

This module pins the paginated read primitive (``_paged_ro_query``), its four
fail-closed completeness paths, and the two methods routed through it.

``FakeCappedGraph`` is a purpose-built double rather than a ``make_graph_mock``
variant because it needs stateful multi-page behaviour and a query log.  It
reproduces the server cap faithfully — silently, exactly as the server does —
so a "we now get all the rows" assertion is a real before/after rather than a
tautology.
"""
from __future__ import annotations

import re

import pytest

_LOGGER_NAME = 'fused_memory.backends.graphiti_client'

# Measured live row counts (dark_factory, 2026-08-17). Used as the fixture
# corpus size so the tests exercise the real shape rather than a toy.
_LIVE_EDGE_ROWS = 24938
_LIVE_ENTITY_NODES = 16038
_LIVE_RESULTSET_CAP = 10000

_SKIP_LIMIT_RE = re.compile(r'SKIP\s+(\d+)\s+LIMIT\s+(\d+)', re.IGNORECASE)


class _FakeResult:
    """Stands in for a FalkorDB result object (the ``.result_set`` shape)."""

    def __init__(self, result_set: list[list] | None, header: list | None = None):
        self.result_set = result_set
        self.header = header if header is not None else []


class FakeCappedGraph:
    """A graph double that reproduces FalkorDB's silent server-side row cap.

    Behaviour, keyed off the cypher text:
      - contains ``count(``      -> ``[[len(corpus)]]``, a single row.  A
        single-row aggregate can never be truncated by the row cap it is
        being used to detect, which is what makes the census a proof.
      - contains ``SKIP n LIMIT m`` -> ``corpus[n : n + m]``
      - anything else            -> the whole corpus

    Every result set is then truncated to at most ``resultset_cap`` rows,
    silently — no error, no marker — exactly as the real server does.

    Every cypher seen is appended to ``self.queries`` so tests can assert on
    the emitted query shapes and on the page count.
    """

    def __init__(
        self,
        corpus: list[list],
        *,
        resultset_cap: int | None = _LIVE_RESULTSET_CAP,
        census_override: int | None = None,
        census_result_set: list[list] | None = None,
        census_result_set_set: bool = False,
    ):
        self.corpus = corpus
        self.resultset_cap = resultset_cap
        # Lets a test make the census disagree with the pages on purpose.
        self.census_override = census_override
        self._census_result_set = census_result_set
        self._census_result_set_set = census_result_set_set
        self.queries: list[str] = []
        self.params: list[dict | None] = []

    # -- query log helpers ------------------------------------------------
    @property
    def census_queries(self) -> list[str]:
        return [q for q in self.queries if 'count(' in q]

    @property
    def page_queries(self) -> list[str]:
        return [q for q in self.queries if _SKIP_LIMIT_RE.search(q)]

    # -- the graph interface under test ----------------------------------
    def _cap(self, rows: list[list]) -> list[list]:
        if self.resultset_cap is None:
            return rows
        return rows[: self.resultset_cap]

    async def ro_query(self, cypher: str, params: dict | None = None) -> _FakeResult:
        self.queries.append(cypher)
        self.params.append(params)
        if 'count(' in cypher:
            if self._census_result_set_set:
                return _FakeResult(self._census_result_set)
            count = (
                self.census_override
                if self.census_override is not None
                else len(self.corpus)
            )
            # A single-row aggregate is never truncated by the row cap.
            return _FakeResult([[count]])
        match = _SKIP_LIMIT_RE.search(cypher)
        if match:
            skip, limit = int(match.group(1)), int(match.group(2))
            return _FakeResult(self._cap(self.corpus[skip: skip + limit]))
        return _FakeResult(self._cap(list(self.corpus)))

    async def query(self, cypher: str, params: dict | None = None):  # pragma: no cover
        raise AssertionError('read paths must use ro_query, never query')


def make_edge_corpus(rows: int) -> list[list]:
    """Build ``rows`` valid-edge rows in the live shape: (n.uuid, e.uuid, fact, name).

    Each edge is double-attributed to two endpoints, matching the undirected
    ``MATCH (n:Entity)-[e:RELATES_TO]-()`` pattern: an odd total therefore
    leaves one trailing single-attributed row, which is fine — the row count
    is what the cap acts on.
    """
    corpus: list[list] = []
    for i in range(rows):
        edge_index = i // 2
        endpoint = 'a' if i % 2 == 0 else 'b'
        corpus.append([
            f'node-{edge_index}-{endpoint}',
            f'edge-{edge_index:07d}',
            f'fact-{edge_index}',
            f'name-{edge_index}',
        ])
    return corpus


# ---------------------------------------------------------------------------
# step-1: the fake really reproduces the defect, and _paged_ro_query cures it
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = (
    'MATCH (n:Entity)-[e:RELATES_TO]-() '
    'WHERE e.invalid_at IS NULL '
    'RETURN n.uuid, e.uuid, e.fact, e.name '
    'ORDER BY e.uuid, n.uuid '
    'SKIP {skip} LIMIT {limit}'
)
_CENSUS_CYPHER = (
    'MATCH (n:Entity)-[e:RELATES_TO]-() '
    'WHERE e.invalid_at IS NULL '
    'RETURN count(*)'
)


class TestPagedRoQueryHappyPath:
    """``_paged_ro_query`` recovers a corpus the server cap would truncate."""

    @pytest.mark.asyncio
    async def test_control_unpaginated_read_is_truncated_by_the_cap(self):
        """CONTROL: one unpaginated query against the live-sized corpus returns 10000.

        Without this the "we now get all the rows" assertion below proves
        nothing — it would be satisfied by a fake that simply never truncated.
        """
        graph = FakeCappedGraph(make_edge_corpus(_LIVE_EDGE_ROWS))
        result = await graph.ro_query(
            'MATCH (n:Entity)-[e:RELATES_TO]-() '
            'WHERE e.invalid_at IS NULL '
            'RETURN n.uuid, e.uuid, e.fact, e.name'
        )
        assert len(result.result_set) == _LIVE_RESULTSET_CAP
        assert len(graph.corpus) == _LIVE_EDGE_ROWS

    @pytest.mark.asyncio
    async def test_paged_read_recovers_the_whole_corpus(self):
        """``_paged_ro_query`` returns every row the server holds, and says so."""
        from fused_memory.backends.graphiti_client import _paged_ro_query

        graph = FakeCappedGraph(make_edge_corpus(_LIVE_EDGE_ROWS))
        paged = await _paged_ro_query(
            graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, page_size=5000
        )
        assert len(paged.rows) == _LIVE_EDGE_ROWS
        assert paged.complete is True
        assert paged.rows_seen == _LIVE_EDGE_ROWS
        assert paged.expected_rows == _LIVE_EDGE_ROWS
        assert paged.reason is None
        # 1 census probe + ceil(24938 / 5000) = 5 page queries.
        assert len(graph.census_queries) == 1
        assert len(graph.page_queries) == 5
        assert len(graph.queries) == 6
        # Rows arrive in corpus order, un-reshuffled and un-dropped.
        assert paged.rows[0] == graph.corpus[0]
        assert paged.rows[-1] == graph.corpus[-1]
