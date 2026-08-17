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

import logging
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


# ---------------------------------------------------------------------------
# step-3: the four independent fail-closed completeness paths
# ---------------------------------------------------------------------------
#
# The two KINDS are deliberately not redundant. ``resultset_size`` is an
# ASSUMPTION about server configuration that this repo does not set; if the
# live server is ever configured BELOW it, the structural check passes and the
# short-page break lies exactly as it does today — the identical silent
# truncation, undetected. The census is a single-row count over the identical
# MATCH/WHERE, and a single row can never be truncated by the row cap it is
# being used to detect, which is what makes ``rows_seen >= expected_rows`` a
# proof rather than one more heuristic. Conversely the structural check fails
# FAST, before any work, with an operator-actionable reason. Neither subsumes
# the other.


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ]


class TestPagedRoQueryStructuralGuards:
    """Guards that fire on the numbers alone, before any evidence is gathered."""

    @pytest.mark.parametrize('page_size', [10000, 10001, 20000])
    @pytest.mark.asyncio
    async def test_page_size_at_or_above_cap_refuses_to_enumerate(
        self, page_size, caplog
    ):
        """At or above the cap, refuse outright and return NO rows.

        The short-page break reasons "this page was not full, so the data is
        exhausted", which is sound ONLY if the server cannot be what shortened
        it. At or above the cap those two causes are indistinguishable, so
        there is nothing trustworthy to return — and a partial list would
        simply invite a caller to use it anyway, recreating the
        silently-short-collection defect one layer up.

        The comparison is ``>=`` and not ``>`` deliberately: equality is
        arithmetically safe on a server configured at exactly 10000, but since
        that constant is an assumption, equality leaves zero margin.
        """
        from fused_memory.backends.graphiti_client import _paged_ro_query

        graph = FakeCappedGraph(make_edge_corpus(100), resultset_cap=None)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            paged = await _paged_ro_query(
                graph,
                _PAGE_TEMPLATE,
                _CENSUS_CYPHER,
                page_size=page_size,
                resultset_size=_LIVE_RESULTSET_CAP,
            )
        assert paged.complete is False
        assert paged.rows == []
        assert paged.rows_seen == 0
        assert paged.expected_rows is None
        assert isinstance(paged.reason, str) and paged.reason
        assert str(page_size) in paged.reason
        assert str(_LIVE_RESULTSET_CAP) in paged.reason
        # Fails FAST: not one query was issued.
        assert graph.queries == []
        assert any(
            str(page_size) in m and str(_LIVE_RESULTSET_CAP) in m
            for m in _warnings(caplog)
        )

    @pytest.mark.asyncio
    async def test_max_pages_exhausted_on_a_full_page_reports_shortfall(self, caplog):
        """Running out of pages while the last one was still full is REPORTED.

        A page cap that truncated in silence would just be the defect this
        module exists to fix, moved one layer up. The rows fetched so far ARE
        returned here — they were really fetched — but ``complete`` is False.
        """
        from fused_memory.backends.graphiti_client import _paged_ro_query

        graph = FakeCappedGraph(make_edge_corpus(100), resultset_cap=None)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            paged = await _paged_ro_query(
                graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, page_size=10, max_pages=2
            )
        assert paged.complete is False
        assert paged.rows_seen == 20
        assert len(paged.rows) == 20
        assert isinstance(paged.reason, str) and paged.reason
        assert '2' in paged.reason and '20' in paged.reason
        assert any('20' in m for m in _warnings(caplog))


class TestPagedRoQueryCensusGuards:
    """Guards that compare what was fetched against what the server says exists."""

    @pytest.mark.parametrize(
        'census_result_set',
        [
            pytest.param([], id='empty-result-set'),
            pytest.param(None, id='null-result-set'),
            pytest.param([[]], id='row-with-no-columns'),
            pytest.param([[None]], id='null-count'),
            pytest.param([['not-a-number']], id='non-integer-count'),
        ],
    )
    @pytest.mark.asyncio
    async def test_unusable_census_is_not_a_passing_proof(
        self, census_result_set, caplog
    ):
        """An unavailable proof is not a passing proof — but the rows still come back.

        Unlike the structural refusal, the DATA here was fetched fine; only the
        PROOF is missing. Every "the store did not say" shape collapses to the
        same fail-closed verdict because the caller treats them identically and
        there is nothing to gain from distinguishing flavours of missing
        evidence.
        """
        from fused_memory.backends.graphiti_client import _paged_ro_query

        graph = FakeCappedGraph(
            make_edge_corpus(100),
            resultset_cap=None,
            census_result_set=census_result_set,
            census_result_set_set=True,
        )
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            paged = await _paged_ro_query(
                graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, page_size=10
            )
        assert paged.expected_rows is None
        assert paged.complete is False
        assert isinstance(paged.reason, str) and paged.reason
        assert paged.rows_seen == 100
        assert len(paged.rows) == 100
        assert _warnings(caplog)

    @pytest.mark.asyncio
    async def test_census_reports_more_rows_than_enumerated_is_incomplete(self, caplog):
        """A SHORTFALL is the truncation signature: report it, naming both numbers."""
        from fused_memory.backends.graphiti_client import _paged_ro_query

        graph = FakeCappedGraph(
            make_edge_corpus(100), resultset_cap=None, census_override=150
        )
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            paged = await _paged_ro_query(
                graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, page_size=10
            )
        assert paged.complete is False
        assert paged.rows_seen == 100
        assert paged.expected_rows == 150
        assert isinstance(paged.reason, str) and paged.reason
        assert '100' in paged.reason and '150' in paged.reason
        assert any('100' in m and '150' in m for m in _warnings(caplog))

    @pytest.mark.asyncio
    async def test_census_reports_fewer_rows_than_enumerated_is_complete(self, caplog):
        """THE ASYMMETRY PIN: growth between the census and the last page is not truncation.

        Completeness is ``rows_seen >= expected_rows``, not ``==``. These
        graphs are written to continuously by the live memory service, so
        strict equality would flip a healthy read to INCOMPLETE on any
        concurrent add_memory — and a warning that fires constantly is a
        warning nobody reads, which would reintroduce the exact silence this
        task exists to remove, just noisier. Only a SHORTFALL is the
        truncation signature.
        """
        from fused_memory.backends.graphiti_client import _paged_ro_query

        graph = FakeCappedGraph(
            make_edge_corpus(100), resultset_cap=None, census_override=50
        )
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            paged = await _paged_ro_query(
                graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, page_size=10
            )
        assert paged.complete is True
        assert paged.reason is None
        assert paged.rows_seen == 100
        assert paged.expected_rows == 50
        assert _warnings(caplog) == []


class TestPagedReadReasonInvariant:
    """``reason`` is None exactly when ``complete`` is True — in every case above."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'kwargs, graph_kwargs',
        [
            pytest.param({'page_size': 10}, {}, id='complete'),
            pytest.param(
                {'page_size': 10, 'resultset_size': 10}, {}, id='structural-refusal'
            ),
            pytest.param(
                {'page_size': 10, 'max_pages': 2}, {}, id='max-pages-exhausted'
            ),
            pytest.param(
                {'page_size': 10},
                {'census_result_set': [], 'census_result_set_set': True},
                id='unusable-census',
            ),
            pytest.param(
                {'page_size': 10}, {'census_override': 150}, id='census-shortfall'
            ),
        ],
    )
    async def test_reason_is_none_exactly_when_complete(self, kwargs, graph_kwargs):
        from fused_memory.backends.graphiti_client import _paged_ro_query

        graph = FakeCappedGraph(
            make_edge_corpus(100), resultset_cap=None, **graph_kwargs
        )
        paged = await _paged_ro_query(
            graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, **kwargs
        )
        if paged.complete:
            assert paged.reason is None
        else:
            assert isinstance(paged.reason, str) and paged.reason
