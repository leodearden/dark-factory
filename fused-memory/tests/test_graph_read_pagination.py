"""Tests for paginated whole-graph reads in GraphitiBackend (tasks 4340, 4869).

FalkorDB truncates every result set at a server-wide ``RESULTSET_SIZE``
ceiling, silently, and several whole-graph reads exceeded it on the live
corpus — so each was returning a short collection with no error and no
marker.  The measured corpus figures, the cap, and the per-query audit live in
ONE place: ``plans/falkordb-resultset-cap-audit.md``.  The ``_LIVE_*``
constants below are this module's local copy, used to size the fixture
corpora so the tests exercise the real shape rather than a toy; re-measure
them together.

This module pins the paginated read primitive (``_paged_ro_query``), its four
fail-closed completeness paths, the two task-4340 methods routed through it,
the three task-4869 reads routed through it (the two stale-embedding reads and
``query_edges_by_time_range``), and ``retrieve_episodes``' keyset reader
(``_read_all_group_episodes``).

``FakeCappedGraph`` is a purpose-built double rather than a ``make_graph_mock``
variant because it needs stateful multi-page behaviour and a query log.  It
reproduces the server cap faithfully — silently, exactly as the server does —
so a "we now get all the rows" assertion is a real before/after rather than a
tautology.
"""
from __future__ import annotations

import contextlib
import logging
import re
import types
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    falkor_skipif,
    unique_graph_name,
)
from falkordb.asyncio import FalkorDB

_LOGGER_NAME = 'fused_memory.backends.graphiti_client'

# Measured live row counts (dark_factory, 2026-08-17), used as fixture corpus
# sizes. Deliberately FROZEN at the measurement that motivated the fix rather
# than tracked against the live graph: the corpus grows every reconciliation
# cycle, and a test whose expectations chase it proves nothing about the cap.
# The authoritative figures are in plans/falkordb-resultset-cap-audit.md.
_LIVE_EDGE_ROWS = 24938
_LIVE_DISTINCT_EDGES = 12506
_LIVE_ENTITY_NODES = 16038
_LIVE_RESULTSET_CAP = 10000

_SKIP_LIMIT_RE = re.compile(r'SKIP\s+(\d+)\s+LIMIT\s+(\d+)', re.IGNORECASE)
# The SAME narrow census pattern conftest.make_graph_mock uses, character for
# character, so the two graph doubles in this repo cannot disagree about what
# a census probe IS. A loose `'count(' in cypher` test also captures ordinary
# queries returning a count as one column among several — e.g.
# find_duplicate_entity_nodes' `RETURN n.uuid, ..., count(e)` — and hands them
# a single-column [[n]] row, raising IndexError deep inside the method under
# test rather than anywhere near the double.
_CENSUS_RE = re.compile(r'RETURN\s+count\(\*\)\s*$', re.IGNORECASE)


class _FakeResult:
    """Stands in for a FalkorDB result object (the ``.result_set`` shape)."""

    def __init__(self, result_set: list[list] | None, header: list | None = None):
        self.result_set = result_set
        self.header = header if header is not None else []


class FakeCappedGraph:
    """A graph double that reproduces FalkorDB's silent server-side row cap.

    This is the ONLY ro_query-level double in the suite that reproduces the
    server cap; ``FakeCappedEpisodeStore`` is its graphiti-core-API
    counterpart and shares ``_LIVE_RESULTSET_CAP``.
    ``conftest.make_graph_mock`` deliberately does not, so the two cannot
    drift; it shares this module's census pattern exactly (``_CENSUS_RE``).

    Behaviour, keyed off the cypher text:
      - a bare ``RETURN count(*)`` projection -> ``[[len(corpus)]]``, a single
        row.  A single-row aggregate can never be truncated by the row cap it
        is being used to detect, which is what makes the census a proof.
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
        return [q for q in self.queries if _CENSUS_RE.search(q.strip())]

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
        if _CENSUS_RE.search(cypher.strip()):
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


def make_live_shaped_edge_corpus(
    rows: int = _LIVE_EDGE_ROWS, distinct_edges: int = _LIVE_DISTINCT_EDGES
) -> list[list]:
    """Build the MEASURED live edge corpus shape: N rows over M distinct edges.

    Measured on dark_factory 2026-08-17: 24938 valid-edge rows over 12506
    distinct edge uuids, and — importantly — 24938 distinct ``(n.uuid,
    e.uuid)`` pairs.  Every row is a distinct attribution, so a plain row
    census is an exact yardstick and the Python dedup collapses nothing here.

    Most edges are double-attributed (both endpoints); the remainder are
    single-attributed, which is what makes ``rows`` less than ``2 *
    distinct_edges``.
    """
    doubled = rows - distinct_edges
    corpus: list[list] = []
    for i in range(distinct_edges):
        edge_uuid = f'edge-{i:07d}'
        corpus.append([f'node-{i}-a', edge_uuid, f'fact-{i}', f'name-{i}'])
        if i < doubled:
            corpus.append([f'node-{i}-b', edge_uuid, f'fact-{i}', f'name-{i}'])
    assert len(corpus) == rows
    assert len({(r[0], r[1]) for r in corpus}) == rows
    assert len({r[1] for r in corpus}) == distinct_edges
    return corpus


def distinct_edge_uuids(grouped: dict[str, list[dict]]) -> set[str]:
    return {edge['uuid'] for edges in grouped.values() for edge in edges}


def total_attributions(grouped: dict[str, list[dict]]) -> int:
    return sum(len(edges) for edges in grouped.values())


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


class TestFakeCappedGraphCensusDispatch:
    """The double's census detection must match conftest.make_graph_mock's.

    Two doubles standing in for the same server are two chances to be wrong
    about it, so this suite keeps only ONE ro_query-level double that
    reproduces the cap and pins the shared behaviour here. The mirror of this test lives in
    test_conftest_fixtures.py (``test_a_count_column_among_others_is_not_a_census``);
    if either double loosens its pattern, one of the two goes red.
    """

    @pytest.mark.asyncio
    async def test_a_count_column_among_others_is_not_a_census(self):
        """Only a bare ``RETURN count(*)`` projection is a census probe.

        A loose ``'count(' in cypher`` test also captures ordinary queries
        returning a count as one column among several — find_duplicate_entity_nodes
        issues ``RETURN n.uuid, ..., count(e)`` — and handing those a
        single-column ``[[n]]`` row raises IndexError deep inside the method
        under test, nowhere near the double.
        """
        rows = [['dup-uuid-1', 200, 2]]
        graph = FakeCappedGraph(rows, resultset_cap=None)
        result = await graph.ro_query(
            'MATCH (n:Entity {name: $name})-[e:RELATES_TO]-() '
            'RETURN n.uuid, n.created_at, count(e) AS edge_count'
        )
        assert result.result_set == rows
        assert graph.census_queries == []

    @pytest.mark.asyncio
    async def test_a_bare_row_count_is_a_census(self):
        graph = FakeCappedGraph([['a'], ['b']], resultset_cap=None)
        result = await graph.ro_query('MATCH (n:Entity) RETURN count(*)')
        assert result.result_set == [[2]]
        assert len(graph.census_queries) == 1


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
        rows = result.result_set
        assert rows is not None  # the fake answered at all
        assert len(rows) == _LIVE_RESULTSET_CAP
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
        assert paged.incomplete_kind is None
        # 1 census probe + ceil(24938 / 5000) = 5 page queries.
        assert len(graph.census_queries) == 1
        assert len(graph.page_queries) == 5
        assert len(graph.queries) == 6
        # Rows arrive in corpus order, un-reshuffled and un-dropped.
        assert paged.rows[0] == graph.corpus[0]
        assert paged.rows[-1] == graph.corpus[-1]


class TestPageBoundSubstitution:
    """``_render_page_bounds``: the two placeholders, and nothing else.

    Cypher map patterns carry literal braces — ``MATCH (n:Entity {name:
    $name})``, a shape several other queries in graphiti_client.py already use
    — so a ``str.format``-based substitution would raise before any query was
    issued, from a traceback pointing at the formatter rather than at the
    template. More reads are meant to be routed through this primitive, which
    puts the next author one map literal away from that.
    """

    def test_literal_cypher_braces_survive_substitution(self):
        from fused_memory.backends.graphiti_client import _render_page_bounds

        template = (
            'MATCH (n:Entity {name: $name}) RETURN n.uuid '
            'ORDER BY n.uuid SKIP {skip} LIMIT {limit}'
        )
        rendered = _render_page_bounds(template, skip=5000, limit=5000)
        assert rendered.endswith('SKIP 5000 LIMIT 5000')
        assert '{name: $name}' in rendered   # untouched, not escaped away

    @pytest.mark.asyncio
    async def test_a_template_with_a_map_pattern_pages_normally(self):
        """End to end: the brace-carrying template really does enumerate."""
        from fused_memory.backends.graphiti_client import _paged_ro_query

        template = (
            'MATCH (n:Entity {group: $group})-[e:RELATES_TO]-() '
            'RETURN n.uuid, e.uuid, e.fact, e.name '
            'ORDER BY e.uuid, n.uuid SKIP {skip} LIMIT {limit}'
        )
        graph = FakeCappedGraph(make_edge_corpus(100), resultset_cap=None)
        paged = await _paged_ro_query(
            graph, template, _CENSUS_CYPHER, params={'group': 'g'}, page_size=40
        )
        assert paged.complete is True
        assert paged.rows_seen == 100

    @pytest.mark.parametrize('missing', ['{skip}', '{limit}'])
    def test_a_missing_placeholder_is_an_authoring_error(self, missing):
        """Not silently tolerated: without ``{skip}`` the pages never advance.

        The same offset would be re-fetched until ``max_pages``, then reported
        as a page-cap shortfall — an authoring bug wearing the costume of a
        server truncation, which is the one diagnosis this module exists to
        make trustworthy.
        """
        from fused_memory.backends.graphiti_client import _render_page_bounds

        template = (
            'MATCH (n:Entity) RETURN n.uuid ORDER BY n.uuid '
            'SKIP {skip} LIMIT {limit}'
        ).replace(missing, '0')
        with pytest.raises(ValueError, match=re.escape(missing)):
            _render_page_bounds(template, skip=0, limit=10)


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
        from fused_memory.backends.graphiti_client import (
            INCOMPLETE_STRUCTURAL_REFUSAL,
            _paged_ro_query,
        )

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
        assert paged.incomplete_kind == INCOMPLETE_STRUCTURAL_REFUSAL
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
        from fused_memory.backends.graphiti_client import (
            INCOMPLETE_PAGE_CAP,
            _paged_ro_query,
        )

        graph = FakeCappedGraph(make_edge_corpus(100), resultset_cap=None)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            paged = await _paged_ro_query(
                graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, page_size=10, max_pages=2
            )
        assert paged.complete is False
        assert paged.rows_seen == 20
        assert len(paged.rows) == 20
        assert paged.incomplete_kind == INCOMPLETE_PAGE_CAP
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
        from fused_memory.backends.graphiti_client import (
            INCOMPLETE_CENSUS_UNAVAILABLE,
            _paged_ro_query,
        )

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
        assert paged.incomplete_kind == INCOMPLETE_CENSUS_UNAVAILABLE
        assert isinstance(paged.reason, str) and paged.reason
        assert paged.rows_seen == 100
        assert len(paged.rows) == 100
        assert _warnings(caplog)

    @pytest.mark.asyncio
    async def test_census_reports_more_rows_than_enumerated_is_incomplete(self, caplog):
        """A SHORTFALL is the truncation signature: report it, naming both numbers."""
        from fused_memory.backends.graphiti_client import (
            INCOMPLETE_SHORT_READ,
            _paged_ro_query,
        )

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
        assert paged.incomplete_kind == INCOMPLETE_SHORT_READ
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
        assert paged.incomplete_kind is None
        assert paged.rows_seen == 100
        assert paged.expected_rows == 50
        assert _warnings(caplog) == []


_ALL_GUARD_CASES = [
    pytest.param({'page_size': 10}, {}, None, id='complete'),
    pytest.param(
        {'page_size': 10, 'resultset_size': 10},
        {},
        'INCOMPLETE_STRUCTURAL_REFUSAL',
        id='structural-refusal',
    ),
    pytest.param(
        {'page_size': 10, 'max_pages': 2},
        {},
        'INCOMPLETE_PAGE_CAP',
        id='max-pages-exhausted',
    ),
    pytest.param(
        {'page_size': 10},
        {'census_result_set': [], 'census_result_set_set': True},
        'INCOMPLETE_CENSUS_UNAVAILABLE',
        id='unusable-census',
    ),
    pytest.param(
        {'page_size': 10},
        {'census_override': 150},
        'INCOMPLETE_SHORT_READ',
        id='census-shortfall',
    ),
]


class TestPagedReadReasonInvariant:
    """``reason`` is None exactly when ``complete`` is True — in every case above."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('kwargs, graph_kwargs, _kind_name', _ALL_GUARD_CASES)
    async def test_reason_is_none_exactly_when_complete(
        self, kwargs, graph_kwargs, _kind_name
    ):
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


# ---------------------------------------------------------------------------
# step-16: a machine-checkable discriminator for the four incompleteness kinds
# ---------------------------------------------------------------------------
#
# Guard 1 returns a FABRICATED empty — zero queries issued, zero rows — and the
# shims collapse that into `{}` / `[]`, byte-identical to a genuinely empty
# graph. Downstream, rebuild_entity_from_edges computes
# `'\n'.join([]) == ''` and update_node_summary WRITES it. The whole fix rests
# on "refused to enumerate" being distinguishable from "enumerated an empty
# corpus", so that distinction has to be a TYPED FIELD — never a substring
# match on an English sentence that any future edit to the warning text would
# silently break. These tests assert on the constants BY IMPORT for the same
# reason: hardcoding their literal values here would re-open exactly the
# stringly-typed coupling the field exists to remove.


class TestIncompleteKindDiscriminator:
    """``incomplete_kind`` tells the four fail-closed paths apart, by type."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('kwargs, graph_kwargs, kind_name', _ALL_GUARD_CASES)
    async def test_each_guard_reports_its_own_kind(
        self, kwargs, graph_kwargs, kind_name
    ):
        """Every guard sets the kind that names it — one case per fail-closed path."""
        from fused_memory.backends import graphiti_client

        graph = FakeCappedGraph(
            make_edge_corpus(100), resultset_cap=None, **graph_kwargs
        )
        paged = await graphiti_client._paged_ro_query(
            graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, **kwargs
        )
        expected = None if kind_name is None else getattr(graphiti_client, kind_name)
        assert paged.incomplete_kind == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize('kwargs, graph_kwargs, kind_name', _ALL_GUARD_CASES)
    async def test_kind_is_none_exactly_when_complete(
        self, kwargs, graph_kwargs, kind_name
    ):
        """BOTH directions of the invariant, not just the easy one.

        A one-directional check would pass a bug that set a kind on a complete
        read (making every caller's ``if paged.incomplete_kind:`` branch fire
        on healthy data), and a bug that left the kind None on an incomplete
        one (making the raise in step-19 silently unreachable — the corrupting
        direction).
        """
        from fused_memory.backends.graphiti_client import _paged_ro_query

        graph = FakeCappedGraph(
            make_edge_corpus(100), resultset_cap=None, **graph_kwargs
        )
        paged = await _paged_ro_query(
            graph, _PAGE_TEMPLATE, _CENSUS_CYPHER, **kwargs
        )
        if paged.complete:
            assert paged.incomplete_kind is None
        else:
            assert isinstance(paged.incomplete_kind, str)
            assert paged.incomplete_kind

    def test_the_four_kinds_are_distinct(self):
        """Four distinct values — two kinds sharing one string would merge two paths."""
        from fused_memory.backends.graphiti_client import (
            INCOMPLETE_CENSUS_UNAVAILABLE,
            INCOMPLETE_PAGE_CAP,
            INCOMPLETE_SHORT_READ,
            INCOMPLETE_STRUCTURAL_REFUSAL,
        )

        kinds = [
            INCOMPLETE_STRUCTURAL_REFUSAL,
            INCOMPLETE_PAGE_CAP,
            INCOMPLETE_CENSUS_UNAVAILABLE,
            INCOMPLETE_SHORT_READ,
        ]
        assert all(isinstance(k, str) and k for k in kinds)
        assert len(set(kinds)) == 4

    def test_structural_kinds_are_exactly_the_two_deterministic_ones(self):
        """Step-19 keys the raise off the GROUP, not off one specific kind.

        The refusal and the page cap are both DETERMINISTIC: fully determined
        by configuration and code, reproduce identically on retry, and can
        never be caused by a concurrent write. The two empirical kinds are
        transient-capable — a census disagreeing by a handful of rows is the
        expected signature of a live graph being written to mid-read — which
        is why they must stay OUT of this set.
        """
        from fused_memory.backends.graphiti_client import (
            INCOMPLETE_CENSUS_UNAVAILABLE,
            INCOMPLETE_PAGE_CAP,
            INCOMPLETE_SHORT_READ,
            INCOMPLETE_STRUCTURAL_KINDS,
            INCOMPLETE_STRUCTURAL_REFUSAL,
        )

        assert isinstance(INCOMPLETE_STRUCTURAL_KINDS, frozenset)
        assert {
            INCOMPLETE_STRUCTURAL_REFUSAL,
            INCOMPLETE_PAGE_CAP,
        } == INCOMPLETE_STRUCTURAL_KINDS
        assert INCOMPLETE_CENSUS_UNAVAILABLE not in INCOMPLETE_STRUCTURAL_KINDS
        assert INCOMPLETE_SHORT_READ not in INCOMPLETE_STRUCTURAL_KINDS
        # None is not a kind: a complete read must never test as structural.
        assert None not in INCOMPLETE_STRUCTURAL_KINDS

    def test_incomplete_kind_defaults_to_none(self):
        """Back-compat: the field is last and defaulted, so existing construction holds.

        ``PagedRead`` is built positionally nowhere in the tree today, but the
        five original fields have no defaults, so the new one has to be last
        AND defaulted or every existing construction site breaks.
        """
        from fused_memory.backends.graphiti_client import PagedRead

        paged = PagedRead(
            rows=[], complete=True, rows_seen=0, expected_rows=0, reason=None
        )
        assert paged.incomplete_kind is None
        # Positional construction of the original five keeps working too.
        positional = PagedRead([], True, 0, 0, None)
        assert positional.incomplete_kind is None


# ---------------------------------------------------------------------------
# step-5: enumerate_all_valid_edges, and get_all_valid_edges as its shim
# ---------------------------------------------------------------------------


def _wire(backend, graph):
    backend._driver._get_graph = MagicMock(return_value=graph)
    return graph


class TestGetAllValidEdgesPagination:
    """THE HEADLINE REGRESSION: the bulk edge read no longer stops at the cap."""

    @pytest.mark.asyncio
    async def test_control_unpaginated_read_of_this_corpus_is_short(self):
        """CONTROL: the same fake truncates a single unpaginated read.

        Makes the assertion below a real before/after rather than a tautology.
        """
        graph = FakeCappedGraph(make_live_shaped_edge_corpus())
        result = await graph.ro_query(
            'MATCH (n:Entity)-[e:RELATES_TO]-() WHERE e.invalid_at IS NULL '
            'RETURN n.uuid, e.uuid, e.fact, e.name'
        )
        rows = result.result_set
        assert rows is not None  # the fake answered at all
        assert len(rows) == _LIVE_RESULTSET_CAP
        assert len({r[1] for r in rows}) < _LIVE_DISTINCT_EDGES

    @pytest.mark.asyncio
    async def test_all_distinct_edges_are_exposed(
        self, mock_config, make_backend
    ):
        """Every one of the 12506 distinct edge uuids reaches the caller."""
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_live_shaped_edge_corpus()))
        grouped = await backend.get_all_valid_edges(group_id='test')
        assert len(distinct_edge_uuids(grouped)) == _LIVE_DISTINCT_EDGES
        assert total_attributions(grouped) == _LIVE_EDGE_ROWS

    @pytest.mark.asyncio
    async def test_enumerate_reports_complete_on_a_full_corpus(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_live_shaped_edge_corpus()))
        grouped, paged = await backend.enumerate_all_valid_edges(group_id='test')
        assert paged.complete is True
        assert paged.reason is None
        assert paged.rows_seen == _LIVE_EDGE_ROWS
        assert paged.expected_rows == _LIVE_EDGE_ROWS
        assert len(distinct_edge_uuids(grouped)) == _LIVE_DISTINCT_EDGES

    @pytest.mark.asyncio
    async def test_enumerate_reports_incomplete_but_still_returns_the_collection(
        self, mock_config, make_backend
    ):
        """A disagreeing census flips completeness without withholding the data."""
        backend = make_backend(mock_config)
        _wire(
            backend,
            FakeCappedGraph(
                make_live_shaped_edge_corpus(),
                census_override=_LIVE_EDGE_ROWS + 5000,
            ),
        )
        grouped, paged = await backend.enumerate_all_valid_edges(group_id='test')
        assert paged.complete is False
        assert isinstance(paged.reason, str) and paged.reason
        assert grouped
        assert len(distinct_edge_uuids(grouped)) == _LIVE_DISTINCT_EDGES

    @pytest.mark.asyncio
    async def test_shim_warns_when_the_enumeration_is_incomplete(
        self, mock_config, make_backend, caplog
    ):
        """No longer a silently short dict: the shim says so, and still returns it."""
        backend = make_backend(mock_config)
        _wire(
            backend,
            FakeCappedGraph(
                make_live_shaped_edge_corpus(),
                census_override=_LIVE_EDGE_ROWS + 5000,
            ),
        )
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            grouped = await backend.get_all_valid_edges(group_id='test')
        assert grouped
        messages = _warnings(caplog)
        assert messages
        assert any('get_all_valid_edges' in m for m in messages)
        assert any(str(_LIVE_EDGE_ROWS) in m for m in messages)

    @pytest.mark.asyncio
    async def test_shim_is_silent_on_a_complete_enumeration(
        self, mock_config, make_backend, caplog
    ):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_live_shaped_edge_corpus()))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await backend.get_all_valid_edges(group_id='test')
        assert _warnings(caplog) == []


class TestGetAllValidEdgesEmittedCypher:
    """The paginated query keeps the semantics the docstring documents."""

    @pytest.fixture
    def emitted(self, mock_config, make_backend):
        async def _run():
            backend = make_backend(mock_config)
            graph = _wire(backend, FakeCappedGraph(make_live_shaped_edge_corpus()))
            await backend.get_all_valid_edges(group_id='test')
            return graph

        return _run

    @pytest.mark.asyncio
    async def test_page_query_keeps_the_match_return_shape(self, emitted):
        graph = await emitted()
        page = graph.page_queries[0]
        assert 'MATCH (n:Entity)-[e:RELATES_TO]-()' in page
        assert 'e.invalid_at IS NULL' in page
        assert 'RETURN n.uuid, e.uuid, e.fact, e.name' in page
        assert 'WITH DISTINCT' not in page
        assert 'RETURN DISTINCT' not in page

    @pytest.mark.asyncio
    async def test_page_query_orders_by_a_total_order(self, emitted):
        """ORDER BY must be TOTAL over ROWS, not just over edges.

        The undirected MATCH yields two rows per edge, so ``e.uuid`` alone
        leaves that pair free to reshuffle across a page boundary — and a
        reshuffle at a boundary silently drops rows permanently.
        """
        graph = await emitted()
        page = graph.page_queries[0]
        assert 'ORDER BY' in page
        order_by = page.split('ORDER BY', 1)[1]
        assert 'e.uuid' in order_by
        assert 'n.uuid' in order_by

    @pytest.mark.asyncio
    async def test_census_matches_the_page_population(self, emitted):
        """Same MATCH/WHERE => the two numbers describe the same population."""
        graph = await emitted()
        census = graph.census_queries[0]
        assert 'MATCH (n:Entity)-[e:RELATES_TO]-()' in census
        assert 'e.invalid_at IS NULL' in census
        assert 'count(*)' in census
        assert 'SKIP' not in census.upper()

    @pytest.mark.asyncio
    async def test_reads_use_ro_query_only(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        # FakeCappedGraph.query raises outright, so reaching it fails loudly;
        # only ro_query appends to .queries, so a non-empty log proves the
        # whole read went through the read-only path.
        graph = _wire(backend, FakeCappedGraph(make_live_shaped_edge_corpus()))
        await backend.get_all_valid_edges(group_id='test')
        assert graph.queries
        assert len(graph.queries) == len(graph.census_queries) + len(graph.page_queries)


class TestGetAllValidEdgesBehaviourPreserved:
    """Pagination must not quietly change the dedup contract (tasks 2207/2210/2213)."""

    @pytest.mark.asyncio
    async def test_double_attribution_preserved(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph([['A', 'e1', 'f', 'n'], ['B', 'e1', 'f', 'n']]))
        result = await backend.get_all_valid_edges(group_id='test')
        assert set(result.keys()) == {'A', 'B'}
        assert result['A'] == [{'uuid': 'e1', 'fact': 'f', 'name': 'n'}]
        assert result['B'] == [{'uuid': 'e1', 'fact': 'f', 'name': 'n'}]

    @pytest.mark.asyncio
    async def test_self_loop_double_match_collapsed(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph([['A', 'e1', 'f', 'n'], ['A', 'e1', 'f', 'n']]))
        result = await backend.get_all_valid_edges(group_id='test')
        assert result['A'] == [{'uuid': 'e1', 'fact': 'f', 'name': 'n'}]

    @pytest.mark.asyncio
    async def test_distinct_edges_same_entity_preserved(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        _wire(
            backend,
            FakeCappedGraph([['A', 'e1', 'f1', 'n1'], ['A', 'e2', 'f2', 'n2']]),
        )
        result = await backend.get_all_valid_edges(group_id='test')
        assert result['A'] == [
            {'uuid': 'e1', 'fact': 'f1', 'name': 'n1'},
            {'uuid': 'e2', 'fact': 'f2', 'name': 'n2'},
        ]

    @pytest.mark.asyncio
    async def test_shared_pair_differing_content_keeps_first_and_logs(
        self, mock_config, make_backend, caplog
    ):
        backend = make_backend(mock_config)
        _wire(
            backend,
            FakeCappedGraph([
                ['A', 'e1', 'first-fact', 'first-name'],
                ['A', 'e1', 'second-fact', 'second-name'],
            ]),
        )
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            result = await backend.get_all_valid_edges(group_id='test')
        assert result['A'] == [{'uuid': 'e1', 'fact': 'first-fact', 'name': 'first-name'}]
        assert any('e1' in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_null_fact_and_name_coerced(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph([['A', 'e1', None, None]]))
        result = await backend.get_all_valid_edges(group_id='test')
        assert result['A'] == [{'uuid': 'e1', 'fact': '', 'name': ''}]

    @pytest.mark.asyncio
    async def test_dedup_map_spans_pages(self, mock_config, make_backend):
        """The case pagination newly makes possible: a repeated pair across a page break.

        The ``seen`` map must be built ONCE over all pages. A per-page map
        would let the second occurrence through and double-count the edge.
        """
        backend = make_backend(mock_config)
        corpus = [
            ['A', 'e1', 'f', 'n'],
            ['B', 'e1', 'f', 'n'],
            ['A', 'e1', 'f', 'n'],  # page 2 — same pair as row 0
            ['C', 'e2', 'f2', 'n2'],
        ]
        _wire(backend, FakeCappedGraph(corpus))
        grouped, paged = await backend.enumerate_all_valid_edges(
            group_id='test', page_size=2
        )
        assert len(paged.rows) == 4  # every row really was fetched...
        assert grouped['A'] == [{'uuid': 'e1', 'fact': 'f', 'name': 'n'}]
        assert total_attributions(grouped) == 3  # ...and the repeat collapsed
        assert paged.complete is True


# ---------------------------------------------------------------------------
# step-9: enumerate_entity_nodes, and list_entity_nodes as its shim
# ---------------------------------------------------------------------------
#
# list_entity_nodes was NOT named in this task, but the re-check the task asked
# for ("any other unpaginated ro_query in graphiti_client.py") measured it
# returning 10000 of 16038 nodes on dark_factory (62%) and 10000 of 23589 on
# reify (42%).
#
# It is not merely another instance. detect_stale_with_edges calls
# list_entity_nodes and get_all_valid_edges on CONSECUTIVE lines, and the two
# truncations are INDEPENDENT: an entity that survives the node cut can still
# lose every one of its edges to the edge cut, producing a bogus "stale, zero
# valid facts" verdict that rebuild_entity_from_edges then WRITES BACK into
# n.summary. That makes the pair corrupting rather than merely
# under-reporting, which is why it is fixed here and not deferred.


def make_entity_node_corpus(rows: int = _LIVE_ENTITY_NODES) -> list[list]:
    """Build ``rows`` Entity node rows in the live shape: (uuid, name, summary)."""
    return [[f'node-{i:07d}', f'name-{i}', f'summary-{i}'] for i in range(rows)]


class TestListEntityNodesPagination:
    """The second measured truncation: 10000 of 16038 nodes, silently."""

    @pytest.mark.asyncio
    async def test_control_unpaginated_read_is_truncated_by_the_cap(self):
        graph = FakeCappedGraph(make_entity_node_corpus())
        result = await graph.ro_query('MATCH (n:Entity) RETURN n.uuid, n.name, n.summary')
        rows = result.result_set
        assert rows is not None  # the fake answered at all
        assert len(rows) == _LIVE_RESULTSET_CAP
        assert len(graph.corpus) == _LIVE_ENTITY_NODES

    @pytest.mark.asyncio
    async def test_every_entity_node_is_returned(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        graph = _wire(backend, FakeCappedGraph(make_entity_node_corpus()))
        nodes = await backend.list_entity_nodes(group_id='test')
        assert len(nodes) == _LIVE_ENTITY_NODES
        # The LAST node specifically: a silently-reordered or short final page
        # would still produce a plausible-looking count on a resumed read.
        assert nodes[-1]['uuid'] == graph.corpus[-1][0]
        assert {n['uuid'] for n in nodes} == {r[0] for r in graph.corpus}

    @pytest.mark.asyncio
    async def test_enumerate_reports_complete_on_a_full_corpus(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_entity_node_corpus()))
        nodes, paged = await backend.enumerate_entity_nodes(group_id='test')
        assert paged.complete is True
        assert paged.reason is None
        assert paged.rows_seen == _LIVE_ENTITY_NODES
        assert paged.expected_rows == _LIVE_ENTITY_NODES
        assert len(nodes) == _LIVE_ENTITY_NODES

    @pytest.mark.asyncio
    async def test_enumerate_reports_incomplete_but_still_returns_the_nodes(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        _wire(
            backend,
            FakeCappedGraph(
                make_entity_node_corpus(), census_override=_LIVE_ENTITY_NODES + 4000
            ),
        )
        nodes, paged = await backend.enumerate_entity_nodes(group_id='test')
        assert paged.complete is False
        assert isinstance(paged.reason, str) and paged.reason
        assert len(nodes) == _LIVE_ENTITY_NODES

    @pytest.mark.asyncio
    async def test_shim_warns_when_the_enumeration_is_incomplete(
        self, mock_config, make_backend, caplog
    ):
        backend = make_backend(mock_config)
        _wire(
            backend,
            FakeCappedGraph(
                make_entity_node_corpus(), census_override=_LIVE_ENTITY_NODES + 4000
            ),
        )
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            nodes = await backend.list_entity_nodes(group_id='test')
        assert len(nodes) == _LIVE_ENTITY_NODES
        messages = _warnings(caplog)
        assert any('list_entity_nodes' in m for m in messages)
        assert any(str(_LIVE_ENTITY_NODES) in m for m in messages)

    @pytest.mark.asyncio
    async def test_shim_is_silent_on_a_complete_enumeration(
        self, mock_config, make_backend, caplog
    ):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_entity_node_corpus()))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await backend.list_entity_nodes(group_id='test')
        assert _warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_null_name_and_summary_coerced(self, mock_config, make_backend):
        """Existing contract preserved: NULL properties still become ''."""
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph([['u1', None, None]]))
        nodes = await backend.list_entity_nodes(group_id='test')
        assert nodes == [{'uuid': 'u1', 'name': '', 'summary': ''}]

    @pytest.mark.asyncio
    async def test_dedup_spans_pages(self, mock_config, make_backend):
        """A uuid re-emitted across a page boundary collapses to ONE node.

        The node-path analogue of test_dedup_map_spans_pages, and a failure
        mode PAGING INTRODUCED rather than one it inherited: a single
        unpaginated query can never return the same n.uuid twice, so every
        consumer downstream is entitled to assume it cannot happen. Under
        SKIP/LIMIT paging over a graph being written to, an Entity inserted
        with a uuid sorting before the current offset shifts every later row
        up by one and the next page's SKIP re-returns the previous page's last
        row — modelled here directly as a repeated row.
        """
        backend = make_backend(mock_config)
        corpus = [
            ['u1', 'n1', 's1'],
            ['u2', 'n2', 's2'],
            ['u1', 'n1', 's1'],   # page 2 — the boundary row, re-emitted
            ['u3', 'n3', 's3'],
        ]
        _wire(backend, FakeCappedGraph(corpus, resultset_cap=None))
        nodes, paged = await backend.enumerate_entity_nodes(
            group_id='test', page_size=2
        )
        assert paged.rows_seen == 4          # every row really was fetched...
        assert [n['uuid'] for n in nodes] == ['u1', 'u2', 'u3']   # ...and deduped
        assert paged.complete is True

    @pytest.mark.asyncio
    async def test_repeated_boundary_row_does_not_inflate_the_stale_denominator(
        self, mock_config, make_backend
    ):
        """The CONSEQUENCE of the dedup above, pinned where it actually bites.

        detect_stale_with_edges reports one entry per element of the node list
        and uses ``len(entities)`` as its ``total_count`` denominator, so a
        duplicated uuid would both double-report the entity and inflate the
        denominator the pagination fix exists to make trustworthy.
        """
        backend = make_backend(mock_config)
        node_rows = [
            ['u1', 'n1', 'summary-1'],
            ['u2', 'n2', 'summary-2'],
            ['u1', 'n1', 'summary-1'],   # the re-emitted boundary row
        ]
        _wire(backend, FakeCappedGraph(node_rows, resultset_cap=None))
        nodes = await backend.list_entity_nodes(group_id='test')
        assert len(nodes) == 2
        assert len({n['uuid'] for n in nodes}) == 2


class TestListEntityNodesEmittedCypher:
    """The node page query keeps its shape and gains a total order."""

    @pytest.fixture
    def emitted(self, mock_config, make_backend):
        async def _run():
            backend = make_backend(mock_config)
            graph = _wire(backend, FakeCappedGraph(make_entity_node_corpus()))
            await backend.list_entity_nodes(group_id='test')
            return graph

        return _run

    @pytest.mark.asyncio
    async def test_page_query_keeps_the_match_return_shape(self, emitted):
        graph = await emitted()
        page = graph.page_queries[0]
        assert 'MATCH (n:Entity)' in page
        assert 'RETURN n.uuid, n.name, n.summary' in page

    @pytest.mark.asyncio
    async def test_page_query_orders_by_uuid(self, emitted):
        """``n.uuid`` alone IS a total order here — one row per node, uuids unique."""
        graph = await emitted()
        page = graph.page_queries[0]
        assert 'ORDER BY' in page
        assert 'n.uuid' in page.split('ORDER BY', 1)[1]

    @pytest.mark.asyncio
    async def test_census_matches_the_page_population(self, emitted):
        graph = await emitted()
        census = graph.census_queries[0]
        assert 'MATCH (n:Entity)' in census
        assert 'count(*)' in census
        assert 'SKIP' not in census.upper()

    @pytest.mark.asyncio
    async def test_reads_use_ro_query_only(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        graph = _wire(backend, FakeCappedGraph(make_entity_node_corpus()))
        await backend.list_entity_nodes(group_id='test')
        assert graph.queries
        assert len(graph.queries) == len(graph.census_queries) + len(graph.page_queries)


# ---------------------------------------------------------------------------
# step-18: the shims RAISE on a structural non-enumeration, WARN on an
#          empirical one
# ---------------------------------------------------------------------------
#
# The split is the whole point, and both halves are load-bearing.
#
# RAISE on structural: a refusal returns a FABRICATED empty (zero queries
# issued), and INCOMPLETE_PAGE_CAP is worse still — rows are ordered by uuid,
# so a page-capped read returns a PREFIX, and every entity whose edges sort
# into the unread tail looks edge-less rather than obviously absent. Handing
# either back as a plain collection is what let '' be written over real
# summaries. Neither kind is reachable at the shipped defaults, so this raise
# cannot fire in production today — only under the misconfiguration that would
# otherwise silently corrupt the graph.
#
# WARN on empirical: this half is what keeps the fix from regressing the
# reviewed design. A census disagreeing by a few rows is the signature of a
# live graph under concurrent write, and raising there would take down
# rebuild_entity_summaries, both reconciliation sweeps and a cleanup script
# for a transient that self-heals next cycle.


def _force_paged_kwargs(monkeypatch, **forced):
    """Re-bind the module-global ``_paged_ro_query`` to force read kwargs.

    The shims deliberately expose NO ``page_size``/``max_pages`` knob — their
    signatures must stay UNCHANGED, that is the back-compat guarantee the
    whole task rests on — and the module constants are bound as function
    defaults at def time, so ``monkeypatch.setattr`` on the constants alone
    would silently do nothing.  Wrapping the global is what reaches the guards
    through a shim while still running the REAL guard code rather than a
    stubbed verdict.
    """
    from fused_memory.backends import graphiti_client

    real = graphiti_client._paged_ro_query

    async def forced_paged(*args, **kwargs):
        return await real(*args, **{**kwargs, **forced})

    monkeypatch.setattr(graphiti_client, '_paged_ro_query', forced_paged)


def _refusal(monkeypatch):
    """Force guard 1: page_size at the assumed cap."""
    _force_paged_kwargs(
        monkeypatch, page_size=_LIVE_RESULTSET_CAP, resultset_size=_LIVE_RESULTSET_CAP
    )


def _page_cap(monkeypatch):
    """Force guard 2: a page budget far too small for the corpus."""
    _force_paged_kwargs(monkeypatch, page_size=10, max_pages=2)


class TestIncompleteEnumerationErrorType:
    """The exception type itself, before any behaviour that uses it."""

    def test_subclasses_exception_not_baseexception(self):
        """MUST be an ``Exception``, or it escapes every caller's handler.

        Both reconciliation sweeps catch ``Exception`` while deliberately
        re-raising ``CancelledError``/``KeyboardInterrupt``/``SystemExit``
        (stale_status_snapshot_edge_sweep.py, stale_priority_override_edge_sweep.py).
        Deriving from ``BaseException`` would turn a handled per-cycle error
        into an unhandled crash that escapes those handlers — converting a
        self-healing degradation into an outage.
        """
        from fused_memory.backends.graphiti_client import IncompleteEnumerationError

        assert issubclass(IncompleteEnumerationError, Exception)
        # Behavioural, not structural: `except Exception` is literally the
        # clause both sweeps are written with, so exercise THAT rather than
        # re-asserting the subclass relation a second way. (The obvious
        # `assert not issubclass(BaseException, IncompleteEnumerationError)`
        # is a TAUTOLOGY — BaseException is not a subclass of any
        # user-defined exception, so it holds for every possible definition
        # of the class and can never fail.)
        caught = False
        try:
            raise IncompleteEnumerationError('structurally incomplete')
        except Exception:  # noqa: BLE001 - the sweeps' actual handler shape
            caught = True
        assert caught, (
            'a bare `except Exception:` must catch it — that is the clause '
            'both reconciliation sweeps use'
        )


class TestShimsRaiseOnStructuralIncompleteness:
    """A structurally-incomplete read is not a read: it must not return data."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'force, kind_name',
        [
            pytest.param(_refusal, 'INCOMPLETE_STRUCTURAL_REFUSAL', id='refusal'),
            pytest.param(_page_cap, 'INCOMPLETE_PAGE_CAP', id='page-cap'),
        ],
    )
    async def test_get_all_valid_edges_raises(
        self, force, kind_name, mock_config, make_backend, monkeypatch
    ):
        from fused_memory.backends import graphiti_client

        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_live_shaped_edge_corpus()))
        force(monkeypatch)

        # The enumerate_* method still REPORTS rather than raising — the raise
        # is the shim's policy, not the primitive's.
        _, paged = await backend.enumerate_all_valid_edges(group_id='test')
        assert paged.incomplete_kind == getattr(graphiti_client, kind_name)

        with pytest.raises(graphiti_client.IncompleteEnumerationError) as exc:
            await backend.get_all_valid_edges(group_id='test')
        message = str(exc.value)
        # An operator reading only the traceback must learn what to change.
        assert 'test' in message                       # the group_id
        assert paged.incomplete_kind in message        # the kind
        assert paged.reason is not None
        assert paged.reason in message                 # the diagnostic numbers

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'force, kind_name',
        [
            pytest.param(_refusal, 'INCOMPLETE_STRUCTURAL_REFUSAL', id='refusal'),
            pytest.param(_page_cap, 'INCOMPLETE_PAGE_CAP', id='page-cap'),
        ],
    )
    async def test_list_entity_nodes_raises(
        self, force, kind_name, mock_config, make_backend, monkeypatch
    ):
        from fused_memory.backends import graphiti_client

        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_entity_node_corpus()))
        force(monkeypatch)

        _, paged = await backend.enumerate_entity_nodes(group_id='test')
        assert paged.incomplete_kind == getattr(graphiti_client, kind_name)

        with pytest.raises(graphiti_client.IncompleteEnumerationError) as exc:
            await backend.list_entity_nodes(group_id='test')
        message = str(exc.value)
        assert 'test' in message
        assert paged.incomplete_kind in message
        assert paged.reason is not None
        assert paged.reason in message

    @pytest.mark.asyncio
    async def test_page_cap_prefix_is_withheld_not_returned(
        self, mock_config, make_backend, monkeypatch
    ):
        """The page-cap case is the dangerous one: a PREFIX, not an obvious zero.

        Rows are ordered by uuid, so a truncated read hands back a complete
        picture of some entities and a zero-edge picture of the rest — the
        shape most likely to be mistaken for real data by a caller that
        checks only truthiness.
        """
        from fused_memory.backends.graphiti_client import IncompleteEnumerationError

        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_live_shaped_edge_corpus()))
        _page_cap(monkeypatch)

        grouped, paged = await backend.enumerate_all_valid_edges(group_id='test')
        assert grouped                      # non-empty: the prefix is real data
        assert paged.rows_seen == 20        # ...and nowhere near the corpus
        with pytest.raises(IncompleteEnumerationError):
            await backend.get_all_valid_edges(group_id='test')


class TestShimsStillWarnOnEmpiricalIncompleteness:
    """POLICY PRESERVATION: the reviewed warn-and-return design stays intact.

    If either of these starts raising, the fix has over-reached and taken down
    the live reconciliation loop for a transient — the exact trade-off design
    decision #1 rejected.
    """

    @staticmethod
    def _short_read_edges():
        return FakeCappedGraph(
            make_live_shaped_edge_corpus(), census_override=_LIVE_EDGE_ROWS + 5000
        )

    @staticmethod
    def _unusable_census_edges():
        return FakeCappedGraph(
            make_live_shaped_edge_corpus(),
            census_result_set=[],
            census_result_set_set=True,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'make_graph, kind_name',
        [
            pytest.param(
                _short_read_edges.__func__, 'INCOMPLETE_SHORT_READ', id='short-read'
            ),
            pytest.param(
                _unusable_census_edges.__func__,
                'INCOMPLETE_CENSUS_UNAVAILABLE',
                id='census-unavailable',
            ),
        ],
    )
    async def test_get_all_valid_edges_warns_and_returns(
        self, make_graph, kind_name, mock_config, make_backend, caplog
    ):
        from fused_memory.backends import graphiti_client

        backend = make_backend(mock_config)
        _wire(backend, make_graph())
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            grouped = await backend.get_all_valid_edges(group_id='test')

        # The data is intact — every distinct edge still reaches the caller.
        assert grouped
        assert len(distinct_edge_uuids(grouped)) == _LIVE_DISTINCT_EDGES
        messages = _warnings(caplog)
        assert any('get_all_valid_edges' in m for m in messages)

        _, paged = await backend.enumerate_all_valid_edges(group_id='test')
        assert paged.incomplete_kind == getattr(graphiti_client, kind_name)
        assert paged.incomplete_kind not in graphiti_client.INCOMPLETE_STRUCTURAL_KINDS

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'graph_kwargs, kind_name',
        [
            pytest.param(
                {'census_override': _LIVE_ENTITY_NODES + 4000},
                'INCOMPLETE_SHORT_READ',
                id='short-read',
            ),
            pytest.param(
                {'census_result_set': [], 'census_result_set_set': True},
                'INCOMPLETE_CENSUS_UNAVAILABLE',
                id='census-unavailable',
            ),
        ],
    )
    async def test_list_entity_nodes_warns_and_returns(
        self, graph_kwargs, kind_name, mock_config, make_backend, caplog
    ):
        from fused_memory.backends import graphiti_client

        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_entity_node_corpus(), **graph_kwargs))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            nodes = await backend.list_entity_nodes(group_id='test')

        assert len(nodes) == _LIVE_ENTITY_NODES
        messages = _warnings(caplog)
        assert any('list_entity_nodes' in m for m in messages)

        _, paged = await backend.enumerate_entity_nodes(group_id='test')
        assert paged.incomplete_kind == getattr(graphiti_client, kind_name)
        assert paged.incomplete_kind not in graphiti_client.INCOMPLETE_STRUCTURAL_KINDS


class TestCompleteEnumerationIsUnaffected:
    """Guard against an over-broad raise: a healthy read stays silent."""

    @pytest.mark.asyncio
    async def test_get_all_valid_edges_neither_raises_nor_warns(
        self, mock_config, make_backend, caplog
    ):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_live_shaped_edge_corpus()))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            grouped = await backend.get_all_valid_edges(group_id='test')
        assert len(distinct_edge_uuids(grouped)) == _LIVE_DISTINCT_EDGES
        assert _warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_list_entity_nodes_neither_raises_nor_warns(
        self, mock_config, make_backend, caplog
    ):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_entity_node_corpus()))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            nodes = await backend.list_entity_nodes(group_id='test')
        assert len(nodes) == _LIVE_ENTITY_NODES
        assert _warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_a_genuinely_empty_graph_still_returns_empty(
        self, mock_config, make_backend, caplog
    ):
        """The distinction the whole fix rests on, from the other side.

        An empty corpus is a COMPLETE enumeration of nothing: census says 0,
        the first page is short, ``complete is True``. It must keep returning
        an empty collection quietly — if this raised, the raise would be
        firing on the very case it exists to tell apart from a refusal.
        """
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph([]))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            grouped = await backend.get_all_valid_edges(group_id='test')
            nodes = await backend.list_entity_nodes(group_id='test')
        assert grouped == {}
        assert nodes == []
        assert _warnings(caplog) == []


# ---------------------------------------------------------------------------
# step-9: live corroboration against a REAL FalkorDB
# ---------------------------------------------------------------------------
#
# The strongest available regression test, because it exercises the ACTUAL
# truncation mechanism rather than a model of it. The fake-driven tests above
# are what keep this step RED-able in a serverless environment; this one skips
# cleanly when no FalkorDB is reachable and never blocks verify.


@pytest_asyncio.fixture
async def pagination_live_graph():
    """Provision a throwaway, uniquely-named FalkorDB graph, yield it, clean up."""
    graph_name = unique_graph_name('4340_read_pagination')
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    with contextlib.suppress(Exception):
        stale = client.select_graph(graph_name)
        await stale.delete()
    graph = client.select_graph(graph_name)
    try:
        yield graph_name, graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestListEntityNodesLiveFalkorDB:
    """Prove the truncation is real, and that pagination actually clears it."""

    LIVE_NODE_COUNT = 12000  # comfortably above the 10000 cap

    @pytest.mark.asyncio
    async def test_pagination_recovers_nodes_the_server_cap_hides(
        self, mock_config, make_backend, pagination_live_graph
    ):
        _, graph = pagination_live_graph
        await graph.query(
            f'UNWIND range(0, {self.LIVE_NODE_COUNT - 1}) AS i '
            "CREATE (:Entity {uuid: 'u' + toString(i), name: 'n' + toString(i)})"
        )

        # (a) The REAL server cap, not a fake: an unpaginated read is short.
        raw = await graph.ro_query('MATCH (n:Entity) RETURN n.uuid')
        assert len(raw.result_set) == _LIVE_RESULTSET_CAP, (
            f'expected the server to truncate at {_LIVE_RESULTSET_CAP}; got '
            f'{len(raw.result_set)}. If this server is configured with a '
            f'different RESULTSET_SIZE, _RESULTSET_SIZE needs re-measuring.'
        )

        # (b) The paginated read sees every node.
        backend = make_backend(mock_config)
        backend._driver._get_graph = MagicMock(return_value=graph)
        nodes = await backend.list_entity_nodes(group_id='test')
        assert len(nodes) == self.LIVE_NODE_COUNT
        assert {n['uuid'] for n in nodes} == {
            f'u{i}' for i in range(self.LIVE_NODE_COUNT)
        }

        _, paged = await backend.enumerate_entity_nodes(group_id='test')
        assert paged.complete is True
        assert paged.expected_rows == self.LIVE_NODE_COUNT


@falkor_skipif()
@pytest.mark.timeout(60)
@pytest.mark.integration
class TestStaleNodeEmbeddingsLiveFalkorDB:
    """Task 4869: the real cap, and the WITH-before-RETURN page syntax.

    The fake cannot check that FalkorDB accepts the page template, nor that
    a real ``vecf32`` survives the dimension parse, so this runs the vector
    read end to end on a THROWAWAY graph. Never point it at a production one.
    """

    LIVE_NODE_COUNT = 12000  # comfortably above the 10000 cap

    @pytest.mark.asyncio
    async def test_every_stale_embedding_is_found_past_the_server_cap(
        self, mock_config, make_backend, pagination_live_graph
    ):
        _, graph = pagination_live_graph
        await graph.query(
            f'UNWIND range(0, {self.LIVE_NODE_COUNT - 1}) AS i '
            "CREATE (:Entity {uuid: 'u' + toString(i), name: 'n' + toString(i), "
            'name_embedding: vecf32([0.1, 0.2, 0.3])})'
        )

        backend = make_backend(mock_config)
        backend._driver._get_graph = MagicMock(return_value=graph)
        stale = await backend.query_stale_node_embeddings(
            expected_dim=1536, group_id='test'
        )
        assert len(stale) == self.LIVE_NODE_COUNT
        assert {s[0] for s in stale} == {f'u{i}' for i in range(self.LIVE_NODE_COUNT)}
        assert {s[2] for s in stale} == {3}


# ---------------------------------------------------------------------------
# step-20: the OUTCOME, not the mechanism
# ---------------------------------------------------------------------------
#
# Steps 16-19 fix the mechanism. This pins the CONSEQUENCE the review asked
# for — "a structural refusal must not produce stale verdicts / must not blank
# summaries" — so a future refactor that re-opens the corrupting path fails
# loudly even if it keeps every mechanism test green.
#
# WHY THE NODE READ MUST STAY HEALTHY HERE, and why a simpler test would be
# worthless: both shims default to the same _DEFAULT_READ_PAGE_SIZE, so a
# refusal triggered by the page-size-vs-cap comparison fires on BOTH reads in
# lockstep — list_entity_nodes also refuses, `entities` is empty, and the
# rebuild loop never runs. A naive test would therefore pass WITHOUT the fix.
# That lockstep is an ACCIDENT of two defaults being equal, not a designed
# invariant: it breaks the moment either default is tuned separately, and it
# does not hold for INCOMPLETE_PAGE_CAP at all, where the node read can page
# to the end while the edge read returns a prefix. So these tests force the
# EDGE enumeration to be structurally incomplete while the NODE enumeration
# SUCCEEDS — the real, unmasked shape of the defect.


class DualCorpusGraph:
    """A graph double answering node reads and edge reads from separate corpora.

    ``FakeCappedGraph`` holds one corpus and cannot represent "the node read
    succeeded and the edge read did not", which is the only shape in which
    this defect is visible.
    """

    def __init__(self, node_rows: list[list], edge_rows: list[list]):
        self._nodes = FakeCappedGraph(node_rows, resultset_cap=None)
        self._edges = FakeCappedGraph(edge_rows, resultset_cap=None)
        self.queries: list[str] = []

    def _delegate(self, cypher: str):
        return self._edges if 'RELATES_TO' in cypher else self._nodes

    async def ro_query(self, cypher: str, params: dict | None = None):
        self.queries.append(cypher)
        return await self._delegate(cypher).ro_query(cypher, params)

    async def query(self, cypher: str, params: dict | None = None):  # pragma: no cover
        raise AssertionError('read paths must use ro_query, never query')


def force_edge_read_only(monkeypatch, **forced):
    """Force ``_paged_ro_query`` kwargs for the EDGE read alone.

    Dispatches on the page template so the node enumeration runs untouched.
    Without this the two reads fail in lockstep on the shared default page
    size, which masks the defect — see the section comment above.
    """
    from fused_memory.backends import graphiti_client

    real = graphiti_client._paged_ro_query

    async def dispatching(*args, **kwargs):
        page_template = args[1] if len(args) > 1 else kwargs.get('page_template', '')
        if 'RELATES_TO' in page_template:
            kwargs = {**kwargs, **forced}
        return await real(*args, **kwargs)

    monkeypatch.setattr(graphiti_client, '_paged_ro_query', dispatching)


def _healthy_entities(count: int = 5) -> list[list]:
    """Entity rows with NON-EMPTY summaries — the thing that gets blanked."""
    return [[f'u{i}', f'name-{i}', f'a real summary for {i}'] for i in range(count)]


def _healthy_edges(count: int = 5) -> list[list]:
    """One valid edge per entity, whose fact IS that entity's summary.

    So every entity is up to date and a correct read reports ZERO stale.
    """
    return [[f'u{i}', f'e{i}', f'a real summary for {i}', f'edge-{i}']
            for i in range(count)]


class TestStructuralRefusalProducesNoStaleVerdicts:
    """detect_stale_with_edges must not manufacture verdicts from a non-read."""

    @pytest.mark.asyncio
    async def test_counterfactual_an_empty_edge_read_reports_everything_stale(
        self, mock_config, make_backend
    ):
        """THE COUNTERFACTUAL, pinned rather than asserted in a comment.

        This is what an edge read returning ``{}`` does, and it is why the
        refusal had to stop being one: ``_build_stale_entry`` computes
        ``canonical = '\\n'.join([]) == ''`` for every entity, finds
        ``summary != canonical`` for every non-empty summary, and reports the
        ENTIRE graph stale.  ``rebuild_entity_from_edges`` would then write
        that ``''`` back.

        Here the empty is GENUINE (an empty edge corpus, a complete
        enumeration of nothing), so it correctly does not raise — and the
        damage it describes is exactly what a fabricated empty would have
        caused indistinguishably.  That indistinguishability is the defect.
        """
        backend = make_backend(mock_config)
        _wire(backend, DualCorpusGraph(_healthy_entities(), []))
        result = await backend.detect_stale_with_edges(group_id='test')
        assert result.total_count == 5
        assert len(result.stale) == 5           # every single one, blanked-to-be
        assert all(s['summary'] for s in result.stale)

    @pytest.mark.asyncio
    async def test_healthy_reads_report_nothing_stale(
        self, mock_config, make_backend
    ):
        """The control: with both reads intact, the same corpus is entirely fresh.

        Without this the assertion above proves nothing — it would be
        satisfied by a fixture that reports everything stale regardless.
        """
        backend = make_backend(mock_config)
        _wire(backend, DualCorpusGraph(_healthy_entities(), _healthy_edges()))
        result = await backend.detect_stale_with_edges(group_id='test')
        assert result.total_count == 5
        assert result.stale == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'forced, kind_name',
        [
            pytest.param(
                {'page_size': _LIVE_RESULTSET_CAP,
                 'resultset_size': _LIVE_RESULTSET_CAP},
                'INCOMPLETE_STRUCTURAL_REFUSAL',
                id='refusal',
            ),
            pytest.param(
                {'page_size': 2, 'max_pages': 1},
                'INCOMPLETE_PAGE_CAP',
                id='page-cap',
            ),
        ],
    )
    async def test_structural_edge_read_raises_instead_of_verdicting(
        self, forced, kind_name, mock_config, make_backend, monkeypatch
    ):
        """No StaleSummaryResult is produced at all — the raise precedes verdicts."""
        from fused_memory.backends import graphiti_client

        backend = make_backend(mock_config)
        _wire(backend, DualCorpusGraph(_healthy_entities(), _healthy_edges()))
        force_edge_read_only(monkeypatch, **forced)

        # The node read is genuinely healthy — this is the unmasked shape.
        nodes, node_paged = await backend.enumerate_entity_nodes(group_id='test')
        assert node_paged.complete is True
        assert len(nodes) == 5

        _, edge_paged = await backend.enumerate_all_valid_edges(group_id='test')
        assert edge_paged.incomplete_kind == getattr(graphiti_client, kind_name)

        sentinel = object()
        result = sentinel
        with pytest.raises(graphiti_client.IncompleteEnumerationError):
            result = await backend.detect_stale_with_edges(group_id='test')
        # Nothing was returned, so nothing with a non-empty .stale was either.
        assert result is sentinel


# ---------------------------------------------------------------------------
# task 4869: the stale-embedding reads
# ---------------------------------------------------------------------------
#
# query_stale_node_embeddings / query_stale_edge_embeddings decide staleness
# CLIENT-side (FalkorDB's size() does not work on Vectorf32), so they read
# every embedded node/edge and were therefore capped at 10000 like the 4340
# reads. The truncated shape is the worst one: a dimension migration driven by
# a short list looks FINISHED when it is not, because the operator's evidence
# of success is the very read being truncated.


def _vector_text(dim: int) -> str:
    """The raw vector text shape the stale-embedding reads parse: ``<v1, v2, ...>``."""
    return '<' + ', '.join(['0.1'] * dim) + '>'


def make_embedded_row_corpus(rows: int, dim: int) -> list[list]:
    """Build ``rows`` embedded rows in the methods' shape: (uuid, name, vector)."""
    vector = _vector_text(dim)
    return [[f'uuid-{i:07d}', f'name-{i}', vector] for i in range(rows)]


_STALE_EMBEDDING_READS = [
    pytest.param(
        'query_stale_node_embeddings',
        'n.name_embedding IS NOT NULL',
        'n.uuid',
        'n.name_embedding',
        id='nodes',
    ),
    pytest.param(
        'query_stale_edge_embeddings',
        'e.fact_embedding IS NOT NULL',
        'e.uuid',
        'e.fact_embedding',
        id='edges',
    ),
]


@pytest.mark.parametrize(
    'method, where_fragment, order_key, vector_prop', _STALE_EMBEDDING_READS
)
class TestStaleEmbeddingReadsPagination:
    """Both vector reads return every stale row past the server cap."""

    @pytest.mark.asyncio
    async def test_control_unpaginated_read_is_truncated_by_the_cap(
        self, method, where_fragment, order_key, vector_prop
    ):
        """CONTROL: keeps the headline below from being a tautology."""
        graph = FakeCappedGraph(make_embedded_row_corpus(_LIVE_ENTITY_NODES, 3))
        result = await graph.ro_query(
            f'MATCH (n) WHERE {where_fragment} RETURN {order_key}, {vector_prop}'
        )
        assert result.result_set is not None
        assert len(result.result_set) == _LIVE_RESULTSET_CAP

    @pytest.mark.asyncio
    async def test_every_stale_row_is_returned(
        self, method, where_fragment, order_key, vector_prop, mock_config, make_backend
    ):
        """HEADLINE: 16038 stale rows in, 16038 stale tuples out — not 10000."""
        backend = make_backend(mock_config)
        graph = _wire(
            backend, FakeCappedGraph(make_embedded_row_corpus(_LIVE_ENTITY_NODES, 3))
        )
        stale = await getattr(backend, method)(expected_dim=1536, group_id='test')
        assert len(stale) == _LIVE_ENTITY_NODES
        assert {s[0] for s in stale} == {r[0] for r in graph.corpus}

    @pytest.mark.asyncio
    async def test_only_mismatched_dimensions_come_back(
        self, method, where_fragment, order_key, vector_prop, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        corpus = [
            ['fresh-1', 'Fresh One', _vector_text(1536)],
            ['stale-1', 'Stale One', _vector_text(3)],
            ['fresh-2', 'Fresh Two', _vector_text(1536)],
            ['stale-2', 'Stale Two', _vector_text(3)],
        ]
        _wire(backend, FakeCappedGraph(corpus))
        stale = await getattr(backend, method)(expected_dim=1536, group_id='test')
        assert stale == [('stale-1', 'Stale One', 3), ('stale-2', 'Stale Two', 3)]

    @pytest.mark.asyncio
    async def test_emitted_cypher_pages_a_total_order_and_projects_the_vector_late(
        self, method, where_fragment, order_key, vector_prop, mock_config, make_backend
    ):
        """Pages are totally ordered, share the census population, and cut first.

        The vector is projected AFTER ``SKIP/LIMIT``, so each page sorts node
        or edge refs and materialises only ``page_size`` vectors rather than
        carrying every matched vector through every page's sort.
        """
        backend = make_backend(mock_config)
        graph = _wire(
            backend, FakeCappedGraph(make_embedded_row_corpus(_LIVE_ENTITY_NODES, 3))
        )
        await getattr(backend, method)(expected_dim=1536, group_id='test')

        assert len(graph.census_queries) == 1
        assert graph.page_queries
        assert len(graph.queries) == (
            len(graph.census_queries) + len(graph.page_queries)
        )
        census = graph.census_queries[0]
        assert where_fragment in census
        assert 'SKIP' not in census.upper()
        population = census.rsplit('RETURN count(*)', 1)[0]
        for page in graph.page_queries:
            assert page.startswith(population)
            assert order_key in page.split('ORDER BY', 1)[1]
            assert page.index('RETURN') > page.index('LIMIT')
            assert vector_prop in page.split('RETURN', 1)[1]

    @pytest.mark.asyncio
    async def test_a_uuid_re_emitted_across_pages_is_reported_once(
        self, method, where_fragment, order_key, vector_prop, mock_config, make_backend
    ):
        """SKIP re-emission under concurrent insert must not double a re-embed."""
        backend = make_backend(mock_config)
        corpus = [
            ['u1', 'n1', _vector_text(3)],
            ['u2', 'n2', _vector_text(3)],
            ['u1', 'n1', _vector_text(3)],   # the boundary row, re-emitted
            ['u3', 'n3', _vector_text(3)],
        ]
        _wire(backend, FakeCappedGraph(corpus, resultset_cap=None))
        stale = await getattr(backend, method)(expected_dim=1536, group_id='test')
        assert [s[0] for s in stale] == ['u1', 'u2', 'u3']

    @pytest.mark.asyncio
    async def test_empirical_incompleteness_warns_and_returns(
        self, method, where_fragment, order_key, vector_prop,
        mock_config, make_backend, caplog,
    ):
        backend = make_backend(mock_config)
        corpus = make_embedded_row_corpus(_LIVE_ENTITY_NODES, 3)
        _wire(
            backend,
            FakeCappedGraph(corpus, census_override=len(corpus) + 5000),
        )
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            stale = await getattr(backend, method)(expected_dim=1536, group_id='test')
        assert len(stale) == len(corpus)
        assert any(method in m for m in _warnings(caplog))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'force, kind_name',
        [
            pytest.param(_refusal, 'INCOMPLETE_STRUCTURAL_REFUSAL', id='refusal'),
            pytest.param(_page_cap, 'INCOMPLETE_PAGE_CAP', id='page-cap'),
        ],
    )
    async def test_structural_incompleteness_raises(
        self, method, where_fragment, order_key, vector_prop, force, kind_name,
        mock_config, make_backend, monkeypatch,
    ):
        """A prefix or a fabricated empty must not pass for the full stale set."""
        from fused_memory.backends import graphiti_client

        backend = make_backend(mock_config)
        _wire(
            backend, FakeCappedGraph(make_embedded_row_corpus(_LIVE_ENTITY_NODES, 3))
        )
        force(monkeypatch)
        with pytest.raises(graphiti_client.IncompleteEnumerationError) as exc:
            await getattr(backend, method)(expected_dim=1536, group_id='test')
        message = str(exc.value)
        assert "'test'" in message
        assert getattr(graphiti_client, kind_name) in message

    @pytest.mark.asyncio
    async def test_a_genuinely_empty_graph_returns_empty_quietly(
        self, method, where_fragment, order_key, vector_prop,
        mock_config, make_backend, caplog,
    ):
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph([]))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            stale = await getattr(backend, method)(expected_dim=1536, group_id='test')
        assert stale == []
        assert _warnings(caplog) == []


# ---------------------------------------------------------------------------
# task 4869: query_edges_by_time_range
# ---------------------------------------------------------------------------
#
# Bounded only by the caller's window: any window spanning more than 10000
# edges was silently truncated, and its consumer (CleanupManager.find_stale_edges)
# feeds bulk_remove_edges. FakeCappedGraph does not evaluate WHERE, so each
# corpus below IS the window's population.

_WINDOW_START = '2026-03-22T17:50:00'
_WINDOW_END = '2026-03-22T18:15:00'


def make_windowed_edge_corpus(rows: int) -> list[list]:
    """Build ``rows`` edges in the method's shape: (uuid, fact, name, valid_at, invalid_at)."""
    return [
        [f'edge-{i:07d}', f'fact-{i}', f'name-{i}', '2026-03-22T18:00:00', None]
        for i in range(rows)
    ]


async def _edges_in_window(backend):
    return await backend.query_edges_by_time_range(
        start=_WINDOW_START, end=_WINDOW_END, group_id='test'
    )


class TestQueryEdgesByTimeRangePagination:
    """A window wider than the cap returns every edge in it."""

    WINDOW_EDGES = 12000  # comfortably above the 10000 cap

    @pytest.mark.asyncio
    async def test_every_edge_in_the_window_is_returned(self, mock_config, make_backend):
        """HEADLINE: 12000 edges in the window, 12000 out — not 10000."""
        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_windowed_edge_corpus(self.WINDOW_EDGES)))
        edges = await _edges_in_window(backend)
        assert len(edges) == self.WINDOW_EDGES
        assert len({e['uuid'] for e in edges}) == self.WINDOW_EDGES

    @pytest.mark.asyncio
    async def test_census_and_every_page_describe_the_same_window(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        graph = _wire(
            backend, FakeCappedGraph(make_windowed_edge_corpus(self.WINDOW_EDGES))
        )
        await _edges_in_window(backend)
        assert len(graph.census_queries) == 1
        assert graph.page_queries
        assert graph.params
        window = {'start': _WINDOW_START, 'end': _WINDOW_END}
        assert all(params == window for params in graph.params)

    @pytest.mark.asyncio
    async def test_emitted_cypher_pages_a_total_order_over_the_window(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        graph = _wire(
            backend, FakeCappedGraph(make_windowed_edge_corpus(self.WINDOW_EDGES))
        )
        await _edges_in_window(backend)
        where = 'e.valid_at >= $start AND e.valid_at <= $end'
        assert len(graph.queries) == (
            len(graph.census_queries) + len(graph.page_queries)
        )
        census = graph.census_queries[0]
        assert where in census
        assert census.rstrip().endswith('RETURN count(*)')
        assert 'SKIP' not in census.upper()
        population = census.rsplit('RETURN count(*)', 1)[0]
        for page in graph.page_queries:
            assert page.startswith(population)
            assert where in page
            assert 'e.uuid' in page.split('ORDER BY', 1)[1]

    @pytest.mark.asyncio
    async def test_a_uuid_re_emitted_across_pages_is_returned_once(
        self, mock_config, make_backend
    ):
        """bulk_remove_edges must not be handed the same uuid twice."""
        backend = make_backend(mock_config)
        corpus = [
            ['e1', 'first fact', 'n1', '2026-03-22T18:00:00', None],
            ['e2', 'fact two', 'n2', '2026-03-22T18:00:00', None],
            ['e1', 'later fact', 'n1', '2026-03-22T18:00:00', None],
            ['e3', 'fact three', 'n3', '2026-03-22T18:00:00', None],
        ]
        _wire(backend, FakeCappedGraph(corpus, resultset_cap=None))
        edges = await _edges_in_window(backend)
        assert [e['uuid'] for e in edges] == ['e1', 'e2', 'e3']
        assert edges[0]['fact'] == 'first fact'

    @pytest.mark.asyncio
    async def test_empirical_incompleteness_warns_and_returns(
        self, mock_config, make_backend, caplog
    ):
        backend = make_backend(mock_config)
        corpus = make_windowed_edge_corpus(self.WINDOW_EDGES)
        _wire(backend, FakeCappedGraph(corpus, census_override=len(corpus) + 5000))
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            edges = await _edges_in_window(backend)
        assert len(edges) == len(corpus)
        assert any('query_edges_by_time_range' in m for m in _warnings(caplog))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'force, kind_name',
        [
            pytest.param(_refusal, 'INCOMPLETE_STRUCTURAL_REFUSAL', id='refusal'),
            pytest.param(_page_cap, 'INCOMPLETE_PAGE_CAP', id='page-cap'),
        ],
    )
    async def test_structural_incompleteness_raises(
        self, force, kind_name, mock_config, make_backend, monkeypatch
    ):
        from fused_memory.backends import graphiti_client

        backend = make_backend(mock_config)
        _wire(backend, FakeCappedGraph(make_windowed_edge_corpus(self.WINDOW_EDGES)))
        force(monkeypatch)
        with pytest.raises(graphiti_client.IncompleteEnumerationError) as exc:
            await _edges_in_window(backend)
        message = str(exc.value)
        assert "'test'" in message
        assert getattr(graphiti_client, kind_name) in message

    @pytest.mark.asyncio
    async def test_the_edge_dict_shape_is_preserved(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        _wire(
            backend,
            FakeCappedGraph([
                ['e1', 'a fact', 'a name', '2026-03-22T17:51:00', '2026-03-22T18:00:00'],
            ]),
        )
        edges = await _edges_in_window(backend)
        assert edges == [{
            'uuid': 'e1',
            'fact': 'a fact',
            'name': 'a name',
            'valid_at': '2026-03-22T17:51:00',
            'invalid_at': '2026-03-22T18:00:00',
        }]


# ---------------------------------------------------------------------------
# task 4869: retrieve_episodes, keyset-paged through graphiti-core
# ---------------------------------------------------------------------------
#
# This read reaches the server through EpisodicNode.get_by_group_ids, not
# ro_query, so the cap bites one layer further from this module. It is the
# worst-shaped truncation of all: graphiti-core orders by uuid DESC, so a capped
# read drops the LOWEST uuids, and the created_at sort then picks the
# most-recent of the SURVIVORS. The caller gets the wrong episodes, not fewer.


class FakeCappedEpisodeStore:
    """graphiti-core-API counterpart of FakeCappedGraph, for episode reads.

    ``get_by_group_ids`` models graphiti-core's contract exactly: uuid DESC,
    ``uuid < uuid_cursor`` when a cursor is given, ``limit`` when not None —
    then SILENT truncation to ``resultset_cap``, as the server does. Each call's
    ``(limit, uuid_cursor)`` is logged in ``calls`` and the uuids it returned
    in ``pages``.
    """

    def __init__(self, episodes, *, resultset_cap: int | None = _LIVE_RESULTSET_CAP):
        self.episodes = list(episodes)
        self.resultset_cap = resultset_cap
        self.calls: list[tuple[int | None, str | None]] = []
        self.pages: list[list[str]] = []

    async def get_by_group_ids(self, driver, group_ids, limit=None, uuid_cursor=None):
        self.calls.append((limit, uuid_cursor))
        page = sorted(self.episodes, key=lambda ep: ep.uuid, reverse=True)
        if uuid_cursor:
            page = [ep for ep in page if ep.uuid < uuid_cursor]
        if limit is not None:
            page = page[:limit]
        if self.resultset_cap is not None:
            page = page[: self.resultset_cap]
        self.pages.append([ep.uuid for ep in page])
        return page


_EPISODE_BASE_TIME = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def make_episode_corpus(count: int) -> list[types.SimpleNamespace]:
    """Episodes whose NEWEST members have the LOWEST uuids.

    That is exactly the tail a uuid-DESC truncation drops, so a capped read
    selects the wrong most-recent episodes rather than merely fewer.
    """
    return [
        types.SimpleNamespace(
            uuid=f'ep-{i:07d}',
            created_at=_EPISODE_BASE_TIME - timedelta(minutes=i),
            name=f'episode-{i}',
            content=f'content-{i}',
            source='message',
            group_id='dark_factory',
        )
        for i in range(count)
    ]


def _patch_episode_store(store: FakeCappedEpisodeStore):
    return patch(
        'fused_memory.backends.graphiti_client.EpisodicNode.get_by_group_ids',
        store.get_by_group_ids,
    )


class TestRetrieveEpisodesKeysetPagination:
    """Every episode is read, so the created_at sort sees the whole group."""

    CORPUS_SIZE = 12000  # comfortably above the 10000 cap

    @pytest.mark.asyncio
    async def test_control_an_unbounded_read_drops_the_newest_episodes(self):
        """CONTROL: the double reproduces the defect it stands in for."""
        store = FakeCappedEpisodeStore(make_episode_corpus(self.CORPUS_SIZE))
        episodes = await store.get_by_group_ids(MagicMock(), ['g'], limit=None)
        assert len(episodes) == _LIVE_RESULTSET_CAP
        returned = {ep.uuid for ep in episodes}
        assert not returned & {f'ep-{i:07d}' for i in range(2000)}

    @pytest.mark.asyncio
    async def test_the_most_recent_episodes_are_selected(self, mock_config, make_backend):
        """HEADLINE: the newest five, not the newest five of the survivors."""
        backend = make_backend(mock_config)
        store = FakeCappedEpisodeStore(make_episode_corpus(self.CORPUS_SIZE))
        with _patch_episode_store(store):
            result = await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=5)
        assert [ep.uuid for ep in result] == [f'ep-{i:07d}' for i in range(5)]

    @pytest.mark.asyncio
    async def test_every_episode_is_reachable(self, mock_config, make_backend):
        backend = make_backend(mock_config)
        store = FakeCappedEpisodeStore(make_episode_corpus(self.CORPUS_SIZE))
        with _patch_episode_store(store):
            result = await backend.retrieve_episodes(
                group_ids=['dark_factory'], last_n=self.CORPUS_SIZE
            )
        assert len({ep.uuid for ep in result}) == self.CORPUS_SIZE

    @pytest.mark.asyncio
    async def test_every_call_is_bounded_and_the_cursor_advances(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        store = FakeCappedEpisodeStore(make_episode_corpus(self.CORPUS_SIZE))
        with _patch_episode_store(store):
            await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=5)
        assert len(store.calls) > 1
        assert all(limit is not None for limit, _ in store.calls)
        assert store.calls[0][1] is None
        for previous_page, (_, cursor) in zip(
            store.pages[:-1], store.calls[1:], strict=True
        ):
            assert cursor == min(previous_page)

    @pytest.mark.asyncio
    async def test_a_server_cap_below_page_size_cannot_truncate_the_read(self):
        """Only an EMPTY page ends the read, so a short page is never mistaken
        for end-of-data. That is what makes a census unnecessary here."""
        from fused_memory.backends.graphiti_client import _read_all_group_episodes

        store = FakeCappedEpisodeStore(make_episode_corpus(10), resultset_cap=3)
        with _patch_episode_store(store):
            episodes = await _read_all_group_episodes(MagicMock(), ['g'], page_size=5)
        assert {ep.uuid for ep in episodes} == {f'ep-{i:07d}' for i in range(10)}

    @pytest.mark.asyncio
    async def test_page_cap_exhaustion_raises_instead_of_returning_a_prefix(self):
        from fused_memory.backends.graphiti_client import (
            INCOMPLETE_PAGE_CAP,
            IncompleteEnumerationError,
            _read_all_group_episodes,
        )

        store = FakeCappedEpisodeStore(make_episode_corpus(10))
        with (
            _patch_episode_store(store),
            pytest.raises(IncompleteEnumerationError) as exc,
        ):
            await _read_all_group_episodes(
                MagicMock(), ['g'], page_size=2, max_pages=3
            )
        message = str(exc.value)
        assert INCOMPLETE_PAGE_CAP in message
        assert 'max_pages=3' in message
        assert 'page_size=2' in message
        assert 'episodes_seen=6' in message

    @pytest.mark.asyncio
    async def test_an_empty_group_returns_empty_after_one_call(
        self, mock_config, make_backend
    ):
        backend = make_backend(mock_config)
        store = FakeCappedEpisodeStore([])
        with _patch_episode_store(store):
            result = await backend.retrieve_episodes(group_ids=['dark_factory'], last_n=5)
        assert result == []
        assert len(store.calls) == 1
