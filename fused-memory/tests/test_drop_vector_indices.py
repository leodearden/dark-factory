"""Unit tests for the DROP path's settle barrier (task 4777).

What this module pins
---------------------
``GraphitiBackend._await_index_catalog_settled`` — the barrier
``drop_vector_indices`` puts in front of its catalog READ — and, in
``TestDropVectorIndicesInsideTheRebuildWindow``, the reported defect it fixes.

The PURE half — ``unsettled_index_statuses``, ``INDEX_STATUS_OPERATIONAL`` and
``IndexCatalogUnsettledError`` — is pinned in ``tests/test_falkor_indices.py``,
which is where it lives precisely because it performs no I/O.  This module is
the mock-driven unit home for the drop path, mirroring the existing
``test_ensure_indices.py`` / ``test_ensure_indices_integration.py`` pair for the
sibling method; the live lane is ``test_drop_vector_indices_integration.py``.

Why the defect is tested with MOCKS rather than live
----------------------------------------------------
The rebuild window is a RACE, not a narrow target: measured, the phantom opens
on only ~1 in 40-70 single-shot reads of a 1-node graph, and at 10,000 nodes
under lane conditions FalkorDB sometimes finishes the rebuild inside the DROP
round-trip so the window never opens at all.  A live test built to FAIL without
the fix would therefore be a new flake — precisely the defect task 4972 spent an
amendment pass removing from ``TestDropRebuildWindow``, which rejects the idea
for this reason, as does
``test_list_indices_integration.py::TestCallDbIndexesOverRoQuery``.  Mocking the
catalog reads makes the defect fully DETERMINISTIC: pre-fix the call raises the
verbatim measured ``ResponseError``, post-fix it returns ``[]``.

HAZARD compliance: every test here is mock-driven.  No live FalkorDB, no
``select_graph``, and no ``FalkorDriver`` / ``_MultiTenantFalkorDriver`` /
``GraphitiBackend.initialize()`` construction anywhere — ``FalkorDriver.__init__``
fire-and-forgets ``build_indices_and_constraints()`` when an event loop is
running, so merely constructing one would create indices on a real graph and
destroy esc-3375-1's protected evidence (the current absence of indices).  That
also keeps this module outside ``test_falkor_index_barrier_guard.py``'s
discovery, whose criteria are a CREATE INDEX / createNodeIndex string CONSTANT
plus a real ``select_graph`` call — do not add either.
"""

from __future__ import annotations

import itertools
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.exceptions

# ``LIVE_HEADER`` — the measured live ``CALL db.indexes()`` header — is
# IMPORTED, not restated, exactly as test_ensure_indices.py imports it: one
# definition per suite, so a FalkorDB shape change is a one-place edit rather
# than a silent disagreement between two suites about what they test (INV-5).
from test_falkor_indices import LIVE_HEADER

from fused_memory.backends.falkor_indices import IndexCatalogUnsettledError

# --- The measured row shapes -----------------------------------------------
#
# LIVE_HEADER order (measured 2026-08-06):
#   [label, properties, types, options, language, stopwords, entitytype,
#    status, info]
# so `status` is position 7.  Rows are built through _row() rather than written
# out positionally, so a header change is a one-place edit here too.


def _row(label, properties, types, *, entity_type='NODE', status='OPERATIONAL'):
    """Build one LIVE_HEADER-shaped ``CALL db.indexes()`` row."""
    return [label, properties, types, {}, 'english', [], entity_type, status, {}]


#: The STALE row FalkorDB keeps serving during the post-drop rebuild: still
#: 'OPERATIONAL', still advertising `name_embedding: ['VECTOR']`, for an index
#: that is already GONE.  Acting on this row is the reported defect.
STALE_ENTITY_ROW = _row(
    'Entity',
    ['name_embedding', 'name'],
    {'name_embedding': ['VECTOR'], 'name': ['RANGE']},
)

#: The REPLACEMENT index being built beside it, in the same result set.
REBUILDING_ENTITY_ROW = _row(
    'Entity', ['name'], {'name': ['RANGE']},
    status='[Indexing] 12/50: UNDER CONSTRUCTION',
)

#: What the catalog looks like once the rebuild finishes.
SETTLED_ENTITY_ROW = _row('Entity', ['name'], {'name': ['RANGE']})

#: A settled row that DOES carry a real, still-present VECTOR index.
SETTLED_VECTOR_ROW = _row(
    'Entity',
    ['name_embedding', 'name'],
    {'name_embedding': ['VECTOR'], 'name': ['RANGE']},
)


def _result(rows, header=LIVE_HEADER):
    """A stand-in for a FalkorDB result object: ``.header`` + ``.result_set``."""
    result = MagicMock()
    result.header = header
    result.result_set = rows
    return result


def _graph_returning(*read_results):
    """A graph mock whose successive ``ro_query`` calls return *read_results*.

    ``make_graph_mock`` answers STATICALLY per cypher and cannot express a
    SEQUENCE of differing reads, which is exactly what the poll and window tests
    need — so ``ro_query`` is an ``AsyncMock(side_effect=[...])`` here instead.
    ``query`` (the WRITE path) is a separate ``AsyncMock`` so tests can assert it
    was never awaited.
    """
    graph = MagicMock()
    graph.ro_query = AsyncMock(side_effect=list(read_results))
    graph.query = AsyncMock()
    return graph


def _graph_always_returning(read_result):
    """A graph mock whose EVERY ``ro_query`` returns *read_result*, unboundedly.

    The supply must be unbounded, not merely large, for any test that expects
    the BUDGET to end the poll loop.  The barrier is bounded by wall clock and
    nothing else, so under ``interval=0`` it issues as many reads as the machine
    can fit in ``timeout_s`` — MEASURED here, ~760 mocked reads inside a 0.01 s
    budget.  A finite list standing in for "never settles" is therefore a race
    between machine speed and list length: out-poll it and ``AsyncMock`` reports
    the exhaustion as ``StopAsyncIteration``, a mock artifact masquerading as the
    failure under test.  ``repeat`` makes "never" actually never, so the deadline
    is the only thing that can end the loop, on any host at any load.
    """
    graph = MagicMock()
    graph.ro_query = AsyncMock(side_effect=itertools.repeat(read_result))
    graph.query = AsyncMock()
    return graph


def _backend_on(make_backend, mock_config, graph):
    backend = make_backend(mock_config)
    backend._driver._get_graph = MagicMock(return_value=graph)
    return backend


class TestAwaitIndexCatalogSettled:
    """The barrier itself: block until every index is OPERATIONAL, then hand the
    CERTIFIED records back.

    Every test passes ``interval=0`` and an explicit small ``timeout_s``, so
    nothing here depends on wall-clock timing.
    """

    @pytest.mark.asyncio
    async def test_settled_catalog_returns_after_exactly_one_read(
        self, mock_config, make_backend,
    ):
        """Check-before-sleep: the barrier costs ONE round-trip, not one plus a sleep.

        This is what makes a 30s default budget headroom rather than spend — on
        an already-settled graph the whole barrier is one ``CALL db.indexes()``.
        """
        graph = _graph_returning(_result([SETTLED_ENTITY_ROW]))
        backend = _backend_on(make_backend, mock_config, graph)

        records = await backend._await_index_catalog_settled(
            'test', timeout_s=5.0, interval=0,
        )

        assert graph.ro_query.await_count == 1
        assert [r['label'] for r in records] == ['Entity']
        assert records[0]['status'] == 'OPERATIONAL'

    @pytest.mark.asyncio
    async def test_it_polls_until_settled_and_returns_the_settled_read(
        self, mock_config, make_backend,
    ):
        """Read 1 is the measured window; read 2 is settled — and read 2 is what returns.

        Returning the CERTIFIED records, rather than settling and then issuing a
        fresh ``list_indices``, is what closes the read-after-settle gap in which
        another process could open a new window between the two.
        """
        graph = _graph_returning(
            _result([STALE_ENTITY_ROW, REBUILDING_ENTITY_ROW]),
            _result([SETTLED_ENTITY_ROW]),
        )
        backend = _backend_on(make_backend, mock_config, graph)

        records = await backend._await_index_catalog_settled(
            'test', timeout_s=5.0, interval=0,
        )

        assert graph.ro_query.await_count == 2
        # The SECOND read's records, not the first: one row, no VECTOR left.
        assert len(records) == 1
        assert records[0]['field'] == ['name']
        assert records[0]['type'] == {'name': ['RANGE']}

    @pytest.mark.asyncio
    async def test_a_catalog_that_never_settles_raises_naming_what_it_saw(
        self, mock_config, make_backend,
    ):
        """Fail CLOSED.  Dropping against an index state that was never determined
        is the same silent-fail-soft class ``ensure_indices`` refuses for
        provisioning (INV-4): an under-construction read can UNDER-report vector
        indices, leaving stale ones behind while the operator believes the
        rebuild was clean.
        """
        graph = _graph_always_returning(
            _result([STALE_ENTITY_ROW, REBUILDING_ENTITY_ROW])
        )
        backend = _backend_on(make_backend, mock_config, graph)

        with pytest.raises(IndexCatalogUnsettledError) as excinfo:
            await backend._await_index_catalog_settled(
                'test', timeout_s=0.01, interval=0,
            )

        message = str(excinfo.value)
        assert 'Entity' in message
        assert '[Indexing] 12/50: UNDER CONSTRUCTION' in message
        assert 'test' in message  # the graph
        assert '0.01' in message  # the budget that expired

    @pytest.mark.asyncio
    async def test_empty_catalog_returns_immediately_without_raising(
        self, mock_config, make_backend,
    ):
        """The production divergence from ``_fm_helpers.await_index_operational``,
        pinned end-to-end at the method level.

        That helper treats an empty ``result_set`` as NOT ready — right for a
        fixture that just issued CREATE.  An index-free graph is a legitimate
        production steady state, so blocking the full budget and then raising on
        one would fail a graph that was never in trouble.
        """
        graph = _graph_returning(_result([]))
        backend = _backend_on(make_backend, mock_config, graph)

        records = await backend._await_index_catalog_settled(
            'test', timeout_s=5.0, interval=0,
        )

        assert records == []
        assert graph.ro_query.await_count == 1

    @pytest.mark.asyncio
    async def test_a_read_error_propagates_unchanged(
        self, mock_config, make_backend,
    ):
        """No try/except anywhere in the barrier.

        MEASURED: reading a never-written graph raises ``Invalid graph operation
        on empty key``.  Absorbing it would convert ``drop_vector_indices``'
        existing absent-graph behaviour into a full-budget block followed by a
        misleading ``IndexCatalogUnsettledError``.  Alpha's fail-closed shape
        errors must reach the caller untouched for the same reason.
        """
        graph = MagicMock()
        graph.ro_query = AsyncMock(
            side_effect=redis.exceptions.ResponseError(
                'Invalid graph operation on empty key'
            )
        )
        graph.query = AsyncMock()
        backend = _backend_on(make_backend, mock_config, graph)

        with pytest.raises(redis.exceptions.ResponseError) as excinfo:
            await backend._await_index_catalog_settled(
                'test', timeout_s=5.0, interval=0,
            )

        assert 'empty key' in str(excinfo.value)


class TestDropVectorIndicesInsideTheRebuildWindow:
    """THE reported defect: a second call landing in FalkorDB's post-drop
    rebuild window re-issues a DROP for an index that is already gone.

    MEASURED.  ``DROP VECTOR INDEX`` against a label whose merged index carries
    SURVIVING fields is not an in-place catalog mutation: FalkorDB builds a
    REPLACEMENT index, and until that build finishes one ``CALL db.indexes()``
    returns BOTH the new ``['name']`` row at ``'[Indexing] N/M: UNDER
    CONSTRUCTION'`` and the stale ``['name_embedding','name']`` row at
    ``'OPERATIONAL'``, still advertising the dropped VECTOR property.  A read
    landing there produces ``DROP VECTOR INDEX FOR (n:Entity) ON
    (n.name_embedding)``, and FalkorDB answers ``'Unable to drop index on
    :Entity(name_embedding): no such index.'`` — which this method deliberately
    does not absorb, so it propagates after the ``'after dropping 0 index(es)'``
    ERROR line.

    The race is mocked so the defect is DETERMINISTIC (see the module docstring
    for why a live red-first test would be a new flake).  ``graph.query`` is
    armed with the VERBATIM measured rejection, so pre-fix this class ERRORS with
    exactly the production symptom.
    """

    @pytest.mark.asyncio
    async def test_a_call_landing_in_the_window_drops_nothing_and_does_not_raise(
        self, mock_config, make_backend,
    ):
        graph = _graph_returning(
            _result([STALE_ENTITY_ROW, REBUILDING_ENTITY_ROW]),
            _result([SETTLED_ENTITY_ROW]),
        )
        graph.query = AsyncMock(
            side_effect=redis.exceptions.ResponseError(
                'Unable to drop index on :Entity(name_embedding): no such index.'
            )
        )
        backend = _backend_on(make_backend, mock_config, graph)

        assert await backend.drop_vector_indices(group_id='test') == []

    @pytest.mark.asyncio
    async def test_no_drop_statement_is_issued_at_all(
        self, mock_config, make_backend,
    ):
        """THE load-bearing assertion: it distinguishes the adopted fix from an
        error-wording absorb.

        Settling FIRST means the doomed statement is never BUILT.  A future edit
        that "fixed" this by catching ``'no such index'`` instead would still
        issue the DROP and merely swallow the response — passing the
        returns-``[]`` test above and failing this one.  Absorbing the wording is
        forbidden anyway (D2: no correctness property may rest on FalkorDB's
        error wording), and it would silently swallow the OTHER measured producer
        of that identical string — the NODE drop form issued against a
        RELATIONSHIP vector index — i.e. a drop that removes nothing while
        reporting success.
        """
        graph = _graph_returning(
            _result([STALE_ENTITY_ROW, REBUILDING_ENTITY_ROW]),
            _result([SETTLED_ENTITY_ROW]),
        )
        graph.query = AsyncMock(
            side_effect=redis.exceptions.ResponseError(
                'Unable to drop index on :Entity(name_embedding): no such index.'
            )
        )
        backend = _backend_on(make_backend, mock_config, graph)

        await backend.drop_vector_indices(group_id='test')

        assert graph.query.await_count == 0

    @pytest.mark.asyncio
    async def test_the_settles_reads_precede_any_write_and_the_certified_read_is_used(
        self, mock_config, make_backend,
    ):
        """Ordering, and no third read.

        ``ro_query`` is awaited exactly twice — the window, then the settled
        catalog — which proves the drop loop consumed the CERTIFIED read rather
        than issuing a fresh ``list_indices`` afterwards.  That re-read would
        reopen, narrowly, the very gap the barrier closes.
        """
        graph = _graph_returning(
            _result([STALE_ENTITY_ROW, REBUILDING_ENTITY_ROW]),
            _result([SETTLED_ENTITY_ROW]),
        )
        graph.query = AsyncMock(
            side_effect=redis.exceptions.ResponseError(
                'Unable to drop index on :Entity(name_embedding): no such index.'
            )
        )
        backend = _backend_on(make_backend, mock_config, graph)

        await backend.drop_vector_indices(group_id='test')

        assert graph.ro_query.await_count == 2
        assert graph.query.await_count == 0

    @pytest.mark.asyncio
    async def test_a_failing_drop_on_a_settled_catalog_still_propagates(
        self, mock_config, make_backend, caplog,
    ):
        """The UNCHANGED half of the contract.

        The barrier is on the READ, not on the DROP.  Per-statement failures are
        still NOT absorbed — the sole caller re-embeds immediately after the
        drop, so a partial drop reported as success would leave stale
        fixed-dimension indices behind while the operator believes the rebuild
        was clean — and the ERROR line still names the partial ``dropped`` list.
        """
        graph = _graph_returning(_result([SETTLED_VECTOR_ROW]))
        graph.query = AsyncMock(
            side_effect=redis.exceptions.ResponseError('some other failure')
        )
        backend = _backend_on(make_backend, mock_config, graph)

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(redis.exceptions.ResponseError),
        ):
            await backend.drop_vector_indices(group_id='test')

        assert graph.query.await_count == 1
        assert 'drop_vector_indices failed on graph' in caplog.text
