"""Tests for the live-FalkorDB scaffolding in _fm_falkor_live.py."""

import os
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Tests for shared FalkorDB test scaffolding (task 3502)
# ---------------------------------------------------------------------------
# No real FalkorDB is used here — falkordb.FalkorDB is monkeypatched
# throughout, so this suite runs in the default `-m 'not integration'` lane.

class TestFalkorHelpers:
    """Unit tests for FALKOR_HOST/FALKOR_PORT, _falkor_available(), falkor_skipif()."""

    def test_falkor_constants_are_the_scratch_endpoint_the_environment_names(self):
        """The constants come from the one scratch resolver, never a local copy of it."""
        from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT

        from fused_memory.config.falkordb_endpoints import scratch_falkordb_from_env

        scratch = scratch_falkordb_from_env(os.environ)
        assert (scratch.host, scratch.port) == (FALKOR_HOST, FALKOR_PORT)

    def test_falkor_constants_have_the_types_consumers_require(self):
        """The module constants carry the types ``FalkorDB(...)`` requires.

        Scoped deliberately to the types, which is the real hazard: ``os.environ``
        yields str, and a str port silently breaks ``FalkorDB(port=...)``. This
        holds under any environment and pins no literal default, which would
        couple this lane to the operator's environment.

        Neither test varies the environment the constants are derived from.
        That cannot be done in-process without an ``importlib.reload`` of a
        module the whole session already holds references into (it would hand
        earlier-imported tests stale dataclass identities), and a subprocess
        import costs a measured ~8.6s here. The derivation itself is covered by
        tests/test_falkordb_endpoints.py, and its agreement with these
        constants by the test above.
        """
        from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT

        assert isinstance(FALKOR_HOST, str)
        assert isinstance(FALKOR_PORT, int)

    def test_falkor_available_true_when_client_ok(self, monkeypatch):
        """_falkor_available() returns True when construct, query, and close all succeed."""
        from _fm_falkor_live import _falkor_available

        fake_graph = MagicMock()
        fake_graph.query.return_value = None
        fake_client = MagicMock()
        fake_client.select_graph.return_value = fake_graph
        fake_client.close.return_value = None
        monkeypatch.setattr('falkordb.FalkorDB', MagicMock(return_value=fake_client))

        assert _falkor_available() is True
        fake_client.close.assert_called_once()

    def test_falkor_available_false_when_query_raises(self, monkeypatch):
        """_falkor_available() returns False (never raises) when the probe query raises."""
        from _fm_falkor_live import _falkor_available

        fake_graph = MagicMock()
        fake_graph.query.side_effect = Exception('boom')
        fake_client = MagicMock()
        fake_client.select_graph.return_value = fake_graph
        monkeypatch.setattr('falkordb.FalkorDB', MagicMock(return_value=fake_client))

        assert _falkor_available() is False
        # The fork wraps close() in contextlib.suppress inside a `finally`, so a
        # probe against a half-dead server must still not leak the connection.
        fake_client.close.assert_called_once()

    def test_falkor_available_false_when_construct_raises(self, monkeypatch):
        """_falkor_available() returns False (never raises) when FalkorDB(...) itself raises."""
        from _fm_falkor_live import _falkor_available

        def _boom(*args, **kwargs):
            raise Exception('connection refused')

        monkeypatch.setattr('falkordb.FalkorDB', _boom)

        assert _falkor_available() is False

    def test_falkor_available_probe_wiring(self, monkeypatch):
        """_falkor_available() probes with the same host/port/timeout and query the forks used.

        Pins the behaviour-preserving contract: a bounded ~2s connect against
        FALKOR_HOST/FALKOR_PORT, then ``select_graph('_probe').query('RETURN 1')``.
        """
        from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, _falkor_available

        fake_graph = MagicMock()
        fake_client = MagicMock()
        fake_client.select_graph.return_value = fake_graph
        fake_ctor = MagicMock(return_value=fake_client)
        monkeypatch.setattr('falkordb.FalkorDB', fake_ctor)

        assert _falkor_available() is True

        fake_ctor.assert_called_once_with(
            host=FALKOR_HOST, port=FALKOR_PORT, socket_connect_timeout=2
        )
        fake_client.select_graph.assert_called_once_with('_probe')
        fake_graph.query.assert_called_once_with('RETURN 1')

    def test_falkor_skipif_default_reason(self):
        """falkor_skipif() returns a pytest.mark.skipif marker with a non-empty default reason.

        The exact wording is deliberately not pinned — matching
        test_fm_qdrant_live.py::TestQdrantHelpers::test_qdrant_skipif_default_reason.
        A reword is cosmetic and nothing depends on the prose.
        """
        from _fm_falkor_live import falkor_skipif

        m = falkor_skipif()

        assert m.name == 'skipif'
        assert isinstance(m.kwargs['reason'], str) and m.kwargs['reason']

    def test_falkor_skipif_custom_reason(self):
        """falkor_skipif(reason=...) propagates a caller-supplied reason string."""
        from _fm_falkor_live import falkor_skipif

        assert falkor_skipif(reason='x').kwargs['reason'] == 'x'

    def test_falkor_skipif_condition_tracks_probe(self, monkeypatch):
        """falkor_skipif()'s condition is `not _falkor_available()`, evaluated at call time.

        The load-bearing property: the factory resolves _falkor_available
        through the module global rather than capturing it, so each consuming
        module's probe fires at that module's own collection time.
        """
        import _fm_falkor_live

        monkeypatch.setattr(_fm_falkor_live, '_falkor_available', lambda: False)
        assert _fm_falkor_live.falkor_skipif().args[0] is True

        monkeypatch.setattr(_fm_falkor_live, '_falkor_available', lambda: True)
        assert _fm_falkor_live.falkor_skipif().args[0] is False

    def test_unique_graph_name_shape(self):
        """unique_graph_name(slug) is '_test_<slug>_<8 lowercase hex>'.

        Asserted as a regex on the CONTRACT rather than a fixed string, since
        the suffix is random; this pins the prefix, the slug placement and the
        8-hex width all at once.
        """
        import re

        from _fm_falkor_live import unique_graph_name

        assert re.fullmatch(r'_test_myslug_[0-9a-f]{8}', unique_graph_name('myslug'))

    def test_unique_graph_name_is_unique_per_call(self):
        """Two calls with the same slug return different names.

        This is the whole point of the idiom: concurrent runs (pytest-xdist
        -n auto, a CI matrix against one shared FalkorDB) must never share a
        graph and wipe each other's fixtures.
        """
        from _fm_falkor_live import unique_graph_name

        assert unique_graph_name('myslug') != unique_graph_name('myslug')

    def test_unique_graph_name_preserves_existing_namespace(self):
        """The helper reproduces the migrated modules' existing graph-name namespace.

        Byte-compatibility check: the migration must not silently rename live
        graphs, and the 'task id in the graph name' convention (which makes an
        orphaned graph from a killed run traceable to a module) must survive.
        """
        from _fm_falkor_live import unique_graph_name

        assert unique_graph_name('530_list_indices_integration').startswith(
            '_test_530_list_indices_integration_'
        )


# ---------------------------------------------------------------------------
# Tests for the shared await_index_operational() barrier (task 3377)
# ---------------------------------------------------------------------------
# FalkorDB builds indices asynchronously, so a test that queries an index
# immediately after creating it can silently succeed for a query the engine
# would otherwise reject (measured while characterising task 3334: 2 of 6 runs
# falsely reported success). await_index_operational is the shared barrier that
# closes that window; it was extracted out of
# test_falkor_fulltext_integration.py so both live-index modules route through
# one implementation.
#
# These tests drive the barrier against FAKE graph objects — no live FalkorDB —
# so the whole RED/GREEN cycle runs in the DEFAULT `-m 'not integration'` lane,
# on exactly the CI configuration least able to notice the regression this
# barrier exists to prevent. Following TestPollUntil's discipline, cases assert
# only on `db.indexes()` call counts and raises, never on elapsed wall-clock,
# so the suite stays load-independent.

# The measured live header (FalkorDB module v41800 / 4.18.0). `result.header`
# rows are (type, name) 2-tuples, so the column NAME is `col[1]`; `status` sits
# at index 7 of 9 today — a position the barrier must NOT hardcode.
_LIVE_INDEX_HEADER: list[tuple[int, str]] = [
    (1, 'label'),
    (1, 'properties'),
    (1, 'types'),
    (1, 'options'),
    (1, 'language'),
    (1, 'stopwords'),
    (1, 'entitytype'),
    (1, 'status'),
    (1, 'info'),
]

# The real measured not-ready value — NOT a bare 'UNDER CONSTRUCTION'. It
# carries a varying progress prefix, which is why the readiness predicate must
# match the READY side exactly rather than substring-testing the not-ready one.
_UNDER_CONSTRUCTION = '[Indexing] 3/200000: UNDER CONSTRUCTION'
_OPERATIONAL = 'OPERATIONAL'


class _FakeIndexResult:
    """A scripted ``CALL db.indexes()`` result mirroring the measured live shape."""

    def __init__(self, header: list[tuple[int, str]], result_set: list[list]):
        self.header = header
        self.result_set = result_set


class _FakeIndexGraph:
    """A fake FalkorDB graph replaying a scripted sequence of index results.

    Counts ``query()`` calls so tests can prove the barrier genuinely polled
    (rather than returning early) and that an already-ready index costs exactly
    one round-trip. Once the script is exhausted the final entry repeats
    forever, so a "never ready" case is expressed by a one-entry script.
    """

    def __init__(self, results: list[_FakeIndexResult]):
        assert results, 'scripted result sequence must be non-empty'
        self._results = results
        self.calls = 0
        self.queries: list[str] = []

    async def query(self, q: str, params=None) -> _FakeIndexResult:
        self.calls += 1
        self.queries.append(q)
        return self._results[min(self.calls - 1, len(self._results) - 1)]


def _index_row(status: str, header: list[tuple[int, str]] = _LIVE_INDEX_HEADER) -> list:
    """One ``db.indexes()`` row carrying *status* in the header's `status` column."""
    return [status if name == 'status' else f'<{name}>' for _, name in header]


def _index_result(
    *statuses: str, header: list[tuple[int, str]] = _LIVE_INDEX_HEADER
) -> _FakeIndexResult:
    """A result whose rows carry *statuses*, one row per status."""
    return _FakeIndexResult(header, [_index_row(s, header) for s in statuses])


class TestAwaitIndexOperational:
    """Unit tests for async await_index_operational(graph, timeout_s)."""

    @pytest.mark.asyncio
    async def test_already_operational_returns_after_one_query(self):
        """An already-OPERATIONAL index returns after exactly 1 db.indexes() call.

        Proves check-before-sleep: the barrier costs a single round-trip in the
        common case, so adding it to a fixture is nearly free.
        """
        from _fm_falkor_live import await_index_operational

        graph = _FakeIndexGraph([_index_result(_OPERATIONAL, _OPERATIONAL)])

        await await_index_operational(graph, timeout_s=5.0)

        assert graph.calls == 1, f'expected exactly 1 query, got {graph.calls}'
        assert graph.queries == ['CALL db.indexes()']

    @pytest.mark.asyncio
    async def test_polls_until_status_flips_to_operational(self):
        """Not-ready for the first 3 polls then OPERATIONAL: returns after exactly 4 calls.

        The call count is what proves the barrier actually blocked rather than
        returning on the first (under-construction) read.
        """
        from _fm_falkor_live import await_index_operational

        graph = _FakeIndexGraph(
            [
                _index_result(_UNDER_CONSTRUCTION),
                _index_result(_UNDER_CONSTRUCTION),
                _index_result(_UNDER_CONSTRUCTION),
                _index_result(_OPERATIONAL),
            ]
        )

        await await_index_operational(graph, timeout_s=5.0)

        assert graph.calls == 4, f'expected exactly 4 queries, got {graph.calls}'

    @pytest.mark.asyncio
    async def test_mixed_rows_do_not_satisfy_the_barrier(self):
        """One OPERATIONAL row plus one still building must NOT be treated as ready.

        EVERY index on the graph has to be ready; an `any()`-style predicate
        would let a half-built graph through and reinstate the false-green.

        Matches the TIMEOUT path's message: IndexHeaderError also subclasses
        AssertionError, so a bare `pytest.raises(AssertionError)` would stay
        green if a regression made the header check fire spuriously here.
        """
        from _fm_falkor_live import await_index_operational

        graph = _FakeIndexGraph([_index_result(_OPERATIONAL, _UNDER_CONSTRUCTION)])

        with pytest.raises(AssertionError, match='not OPERATIONAL'):
            await await_index_operational(graph, timeout_s=0.05)

    @pytest.mark.asyncio
    async def test_status_column_is_resolved_by_name_not_position(self):
        """A reordered header with a decoy 'OPERATIONAL' at index 7 must still block.

        `status` is moved to index 0 and the literal 'OPERATIONAL' is planted at
        index 7 (its position in today's live header). A barrier that read the
        column positionally would see the decoy and return success; one that
        resolves `status` by name reads the real, still-building value and
        blocks. This is the assertion that stops a FalkorDB column reorder from
        silently turning the barrier into a no-op.

        Matches the TIMEOUT path's message so the case cannot be satisfied by
        the header-shape path instead: the reordered header DOES contain a
        `status` column, so reaching IndexHeaderError here would mean the
        by-name resolution broke, not that the barrier blocked.
        """
        from _fm_falkor_live import await_index_operational

        reordered = [(1, 'status')] + [c for c in _LIVE_INDEX_HEADER if c[1] != 'status']
        assert reordered[7][1] != 'status', 'decoy must not land on the real status column'

        row = _index_row(_UNDER_CONSTRUCTION, reordered)
        row[7] = _OPERATIONAL  # decoy in the position `status` occupies live

        graph = _FakeIndexGraph([_FakeIndexResult(reordered, [row])])

        with pytest.raises(AssertionError, match='not OPERATIONAL'):
            await await_index_operational(graph, timeout_s=0.05)

    @pytest.mark.asyncio
    async def test_missing_status_column_raises_immediately_naming_the_header(self):
        """A header with no `status` column fails loudly and at once — not as a timeout.

        The two failure modes carry different operator actions: "index never
        became ready" is a slow or wedged build, while "db.indexes() has no
        status column" means FalkorDB changed its result shape and the barrier
        can no longer be trusted at all. Collapsing the second into a generic
        timeout message would hide a silent no-op behind a plausible-looking
        timeout, so this must raise on the FIRST query with the observed header
        names in the message — and must never skip.

        Asserts the EXACT type (IndexHeaderError, not the AssertionError it
        subclasses): the whole point of the distinct type is that a caller can
        tell the two paths apart, so a test that accepted the base type would
        stay green if the paths were collapsed back together.
        """
        from _fm_falkor_live import IndexHeaderError, await_index_operational

        header = [(1, 'label'), (1, 'properties'), (1, 'state')]
        graph = _FakeIndexGraph([_FakeIndexResult(header, [['Entity', ['name'], 'READY']])])

        with pytest.raises(IndexHeaderError) as excinfo:
            await await_index_operational(graph, timeout_s=5.0)

        message = str(excinfo.value)
        assert 'not OPERATIONAL' not in message, (
            f'the header-shape failure must not be dressed up as a timeout: {message!r}'
        )
        assert 'status' in message
        for name in ('label', 'properties', 'state'):
            assert name in message, f'header name {name!r} missing from {message!r}'
        assert graph.calls == 1, (
            f'expected the header-shape failure to raise on the first query, got '
            f'{graph.calls} queries — it must not be retried until the deadline and '
            f'reported as a timeout.'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('header', 'offending_entry'),
        [
            pytest.param([(1, 'label'), (7,), (1, 'status')], (7,), id='short-pair'),
            pytest.param([(1, 'label'), None, (1, 'status')], None, id='non-sequence'),
            pytest.param(['label', 'status'], 'label', id='bare-str'),
        ],
    )
    async def test_malformed_header_entry_raises_index_header_error_on_first_query(
        self, header, offending_entry
    ):
        """A header entry that is not a (type, name) pair must fail as the named
        shape error, not as a raw IndexError/TypeError or a mangled
        missing-column report.
        """
        from _fm_falkor_live import IndexHeaderError, await_index_operational

        graph = _FakeIndexGraph([_FakeIndexResult(header, [['x', _OPERATIONAL]])])

        with pytest.raises(IndexHeaderError) as excinfo:
            await await_index_operational(graph, timeout_s=5.0)

        message = str(excinfo.value)
        assert graph.calls == 1, f'expected a raise on the first query, got {graph.calls}'
        assert 'not OPERATIONAL' not in message, (
            f'the header-shape failure must not be dressed up as a timeout: {message!r}'
        )
        assert repr(offending_entry) in message, (
            f'the error must name the entry it saw, {offending_entry!r}: {message!r}'
        )

    @pytest.mark.asyncio
    async def test_never_operational_raises_with_last_seen_statuses(self):
        """An index that never becomes ready raises AssertionError naming its last statuses.

        Raising (rather than skipping or passing) is deliberate: a
        never-operational index must be a loud failure, since passing would
        reintroduce exactly the false-green this barrier removes.
        """
        from _fm_falkor_live import await_index_operational

        graph = _FakeIndexGraph([_index_result(_UNDER_CONSTRUCTION)])

        with pytest.raises(AssertionError) as excinfo:
            await await_index_operational(graph, timeout_s=0.05)

        assert _UNDER_CONSTRUCTION in str(excinfo.value)
        assert graph.calls >= 1

    @pytest.mark.asyncio
    async def test_empty_result_set_is_not_ready(self):
        """A graph reporting ZERO indices must never satisfy the barrier — fail closed.

        `all()` over an empty sequence is vacuously True, so without an explicit
        non-empty guard a graph whose index vanished (or was never created)
        would return success having gated nothing.

        Matches the TIMEOUT path's message: the header here is the well-formed
        live one, so an IndexHeaderError would mean the header check fired
        spuriously — a bare `pytest.raises(AssertionError)` could not tell.
        """
        from _fm_falkor_live import await_index_operational

        graph = _FakeIndexGraph([_FakeIndexResult(_LIVE_INDEX_HEADER, [])])

        with pytest.raises(AssertionError, match='not OPERATIONAL'):
            await await_index_operational(graph, timeout_s=0.05)
