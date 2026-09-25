"""Unit tests for ``drop_vector_indices``' settle barrier (task 4777).

What this module pins
---------------------
That ``GraphitiBackend.drop_vector_indices`` settles the FalkorDB index catalog
BEFORE it reads it, driven through the public method alone — its
``settle_timeout_s`` is the budget's seam — and, in
``TestDropVectorIndicesInsideTheRebuildWindow``, the reported defect that
settling fixes.  The rebuild window itself is described in that method's
docstring.

The PURE half — ``unsettled_index_statuses``, ``INDEX_STATUS_OPERATIONAL`` and
``IndexCatalogUnsettledError`` — is pinned in ``tests/test_falkor_indices.py``,
which is where it lives precisely because it performs no I/O.  This module is
the mock-driven unit home for the drop path, mirroring the existing
``test_ensure_indices.py`` / ``test_ensure_indices_integration.py`` pair for the
sibling method; the live lane is ``test_drop_vector_indices_integration.py``.

Why the defect is tested with MOCKS rather than live
----------------------------------------------------
The window is a RACE, not a narrow target: a single live read catches it only
rarely, and sometimes FalkorDB finishes the rebuild inside the DROP round-trip
so it never opens at all (measured in
``test_drop_vector_indices_integration.py::TestDropRebuildWindow``).  A live
test built to FAIL without the fix would therefore be a new flake — the defect
task 4972 removed from that very test.  Mocking the catalog reads makes the
defect DETERMINISTIC: pre-fix the call raises the verbatim measured
``ResponseError``, post-fix it returns ``[]``.

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

from fused_memory.backends.falkor_indices import (
    IndexCatalogUnsettledError,
    vector_drop_statement,
)

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

#: A settled relationship index whose VECTOR is genuinely still there.
RELATES_TO_VECTOR_ROW = _row(
    'RELATES_TO',
    ['uuid', 'fact_embedding'],
    {'uuid': ['RANGE'], 'fact_embedding': ['VECTOR']},
    entity_type='RELATIONSHIP',
)


def _result(rows, header=LIVE_HEADER):
    """A stand-in for a FalkorDB result object: ``.header`` + ``.result_set``."""
    result = MagicMock()
    result.header = header
    result.result_set = rows
    return result


def _graph(read_side_effect, *, drop_error=None):
    """A graph mock whose ``ro_query`` is the catalog READ and ``query`` the DROP.

    ``make_graph_mock`` answers STATICALLY per cypher and cannot express a
    SEQUENCE of differing reads, which the poll and window tests need, so both
    are ``AsyncMock``s here.  The graph is their shared parent, so
    ``graph.method_calls`` records how reads and drops interleave, not merely
    how many of each happened.
    """
    graph = MagicMock()
    graph.attach_mock(AsyncMock(side_effect=read_side_effect), 'ro_query')
    graph.attach_mock(AsyncMock(side_effect=drop_error), 'query')
    return graph


def _graph_returning(*read_results, drop_error=None):
    """A graph whose successive catalog reads return *read_results*, in order."""
    return _graph(list(read_results), drop_error=drop_error)


def _graph_always_returning(read_result):
    """A graph whose EVERY catalog read returns *read_result*, unboundedly.

    Unbounded, not merely long, for any test that expects the BUDGET to end the
    poll loop: the barrier is bounded by wall clock and nothing else, so a
    finite list standing in for "never settles" races the host's speed against
    the list's length, and running it dry surfaces as ``StopAsyncIteration`` —
    a mock artifact masquerading as the failure under test.
    """
    return _graph(itertools.repeat(read_result))


def _graph_inside_the_rebuild_window():
    """Read 1 is the measured window and read 2 the settled catalog; every DROP is
    answered with FalkorDB's VERBATIM rejection of the stale row's gone VECTOR."""
    return _graph_returning(
        _result([STALE_ENTITY_ROW, REBUILDING_ENTITY_ROW]),
        _result([SETTLED_ENTITY_ROW]),
        drop_error=redis.exceptions.ResponseError(
            'Unable to drop index on :Entity(name_embedding): no such index.'
        ),
    )


def _backend_on(make_backend, mock_config, graph):
    backend = make_backend(mock_config)
    backend._driver._get_graph = MagicMock(return_value=graph)
    return backend


class TestDropVectorIndicesSettlesTheCatalogFirst:
    """The settle barrier, driven through ``drop_vector_indices`` itself."""

    @pytest.mark.asyncio
    async def test_a_settled_catalog_costs_exactly_one_read(
        self, mock_config, make_backend,
    ):
        """Check-before-sleep: on a settled graph the whole barrier is one read.

        That is what makes a generous default budget headroom rather than spend.
        """
        graph = _graph_returning(_result([SETTLED_VECTOR_ROW]))
        backend = _backend_on(make_backend, mock_config, graph)

        dropped = await backend.drop_vector_indices(group_id='test')

        assert dropped == [{'label': 'Entity', 'field': 'name_embedding'}]
        assert graph.ro_query.await_count == 1

    @pytest.mark.asyncio
    async def test_an_empty_catalog_is_settled_after_one_read(
        self, mock_config, make_backend,
    ):
        """An index-free graph is a production steady state, not a build in flight.

        The deliberate divergence from ``_fm_helpers.await_index_operational``,
        pinned at the method level; ``unsettled_index_statuses`` says why.
        """
        graph = _graph_returning(_result([]))
        backend = _backend_on(make_backend, mock_config, graph)

        assert await backend.drop_vector_indices(group_id='test') == []
        assert graph.ro_query.await_count == 1

    @pytest.mark.asyncio
    async def test_a_catalog_that_never_settles_raises_and_drops_nothing(
        self, mock_config, make_backend,
    ):
        """Fail CLOSED: no DROP against an index state that was never determined.

        The unsettled catalog still advertises a VECTOR, so a barrier that ran
        out of budget and acted anyway would issue a DROP here.
        """
        graph = _graph_always_returning(
            _result([STALE_ENTITY_ROW, REBUILDING_ENTITY_ROW])
        )
        backend = _backend_on(make_backend, mock_config, graph)

        with pytest.raises(IndexCatalogUnsettledError) as excinfo:
            await backend.drop_vector_indices(group_id='test', settle_timeout_s=0.01)

        message = str(excinfo.value)
        assert 'Entity' in message
        assert '[Indexing] 12/50: UNDER CONSTRUCTION' in message
        assert 'test' in message  # the graph
        assert '0.01' in message  # the budget that expired
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_read_error_propagates_unchanged(
        self, mock_config, make_backend,
    ):
        """No try/except anywhere in the barrier.

        MEASURED: reading a never-written graph raises ``Invalid graph operation
        on empty key``.  Absorbing it would turn the absent-graph behaviour into
        a full-budget block followed by a misleading
        ``IndexCatalogUnsettledError``.
        """
        graph = _graph(
            redis.exceptions.ResponseError('Invalid graph operation on empty key')
        )
        backend = _backend_on(make_backend, mock_config, graph)

        with pytest.raises(redis.exceptions.ResponseError) as excinfo:
            await backend.drop_vector_indices(group_id='test')

        assert 'empty key' in str(excinfo.value)
        graph.query.assert_not_awaited()


class TestDropVectorIndicesInsideTheRebuildWindow:
    """THE reported defect: a retry landing in FalkorDB's post-drop rebuild
    window used to re-issue a DROP for an index that was already gone.

    The window is mocked so the defect is DETERMINISTIC, and the doomed DROP is
    armed with FalkorDB's VERBATIM rejection, so pre-fix this class ERRORS with
    exactly the production symptom.
    """

    @pytest.mark.asyncio
    async def test_a_call_landing_in_the_window_drops_nothing_and_does_not_raise(
        self, mock_config, make_backend,
    ):
        graph = _graph_inside_the_rebuild_window()
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
        graph = _graph_inside_the_rebuild_window()
        backend = _backend_on(make_backend, mock_config, graph)

        await backend.drop_vector_indices(group_id='test')

        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_retry_after_a_partial_drop_settles_then_drops_only_what_remains(
        self, mock_config, make_backend,
    ):
        """Every settle read precedes the first DROP, and the DROPs come from the
        CERTIFIED read.

        Read 1 is what a retry sees after a run that dropped
        ``Entity.name_embedding`` and failed before ``RELATES_TO.fact_embedding``:
        the window on Entity, beside a vector index that is genuinely still
        there.  Read 2 is that catalog settled.  Acting on read 1 would re-issue
        the doomed Entity DROP; acting on read 2 drops RELATES_TO's alone.
        """
        graph = _graph_returning(
            _result([STALE_ENTITY_ROW, REBUILDING_ENTITY_ROW, RELATES_TO_VECTOR_ROW]),
            _result([SETTLED_ENTITY_ROW, RELATES_TO_VECTOR_ROW]),
        )
        backend = _backend_on(make_backend, mock_config, graph)

        dropped = await backend.drop_vector_indices(group_id='test')

        assert [name for name, _args, _kwargs in graph.method_calls] == [
            'ro_query', 'ro_query', 'query',
        ]
        graph.query.assert_awaited_once_with(
            vector_drop_statement(
                'RELATES_TO', 'fact_embedding', entity_type='RELATIONSHIP',
            )
        )
        assert dropped == [{'label': 'RELATES_TO', 'field': 'fact_embedding'}]

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
        graph = _graph_returning(
            _result([SETTLED_VECTOR_ROW]),
            drop_error=redis.exceptions.ResponseError('some other failure'),
        )
        backend = _backend_on(make_backend, mock_config, graph)

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(redis.exceptions.ResponseError),
        ):
            await backend.drop_vector_indices(group_id='test')

        assert graph.query.await_count == 1
        assert 'drop_vector_indices failed on graph' in caplog.text
