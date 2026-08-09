"""Unit tests for ``GraphitiBackend.ensure_indices`` (task 3707, β).

What this module pins
---------------------
The I/O half of β: diff the expected index set against what a graph actually
carries, issue only the statements that close the gap, absorb per-statement
failures into a structured result, and return without waiting for anything.

The PURE half — ``IndexProvisionResult``, ``range_create_statement``,
``plan_index_statements`` — is pinned in ``tests/test_falkor_indices.py``, which
is where it lives precisely because it performs no I/O.

Fixtures are DERIVED, not pinned
--------------------------------
Every mocked ``CALL db.indexes()`` row here is built by inverting
``expected_index_set()`` back into the measured live record shape (see
``_rows_for``), using ``LIVE_HEADER`` from ``test_falkor_indices``.  A hard-coded
38 would go stale the first time graphiti changes its index set, and would do so
by making these tests pass against the wrong expectation — the exact silent-drift
class the PRD exists to remove (INV-5: single home, never restate).

HAZARD compliance: every test here is mock-driven.  No live FalkorDB, no
``select_graph``, and no ``FalkorDriver`` / ``_MultiTenantFalkorDriver`` /
``GraphitiBackend.initialize()`` construction anywhere — ``FalkorDriver.__init__``
fire-and-forgets ``build_indices_and_constraints()`` when an event loop is
running, so merely constructing one would create indices on a real graph and
destroy esc-3375-1's protected evidence (the current absence of indices).  The
live lane lives in ``tests/test_ensure_indices_integration.py``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.exceptions
from test_falkor_indices import LIVE_HEADER

from fused_memory.backends.falkor_indices import (
    IndexSpec,
    expected_index_set,
    plan_index_statements,
)
from fused_memory.utils.validation import PathShapedProjectIdError

_PATH_SHAPED = '-home-leo-src-x'

#: The trap state esc-3375-1 protected as evidence: exactly two range indices.
_TRAP_PRESENT: set[IndexSpec] = {
    ('Entity', 'NODE', 'uuid', 'RANGE'),
    ('RELATES_TO', 'RELATIONSHIP', 'uuid', 'RANGE'),
}


def _rows_for(specs: set[IndexSpec]) -> list[list]:
    """Invert the normal form back into ``CALL db.indexes()``-shaped rows.

    FalkorDB merges every index on a label into ONE record, so specs are grouped
    by ``(label, entity_type)`` and each property's index types are collected into
    the ``types`` column — the same merged shape measured live 2026-08-06, where
    ``Entity`` comes back once carrying
    ``{'group_id': ['RANGE', 'FULLTEXT'], 'summary': ['FULLTEXT'], ...}``.

    Columns are emitted in ``LIVE_HEADER`` order::

        [label, properties, types, options, language, stopwords,
         entitytype, status, info]
    """
    grouped: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for label, entity_type, field, index_type in sorted(specs):
        grouped[(label, entity_type)][field].append(index_type)

    return [
        [
            label,                              # label
            list(types_by_field),               # properties
            {f: list(ts) for f, ts in types_by_field.items()},  # types
            {},                                 # options
            'english',                          # language
            [],                                 # stopwords
            entity_type,                        # entitytype
            'OPERATIONAL',                      # status
            {},                                 # info
        ]
        for (label, entity_type), types_by_field in grouped.items()
    ]


def _wire(backend, graph) -> None:
    """Point the backend's driver at *graph* — the seam ``_graph_for`` resolves through."""
    backend._driver._get_graph = MagicMock(return_value=graph)


def _issued(graph) -> list[str]:
    """The statements actually sent on the WRITE path, in order."""
    return [call.args[0] for call in graph.query.call_args_list]


def _ro_issued(graph) -> list[str]:
    """The statements actually sent on the READ-ONLY path, in order."""
    return [call.args[0] for call in graph.ro_query.call_args_list]


class TestEnsureIndicesDiff:
    """β writes only what is missing, and computes "missing" from α's normal form."""

    @pytest.mark.asyncio
    async def test_fully_provisioned_graph_is_not_written_to_at_all(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Boundary test 3, mocked: idempotence is STRUCTURAL, not a swallowed re-create.

        D2 is explicit that no correctness property may rest on FalkorDB's
        tolerance for re-creating an existing index.  An empty diff must produce
        an empty plan, so a provisioned graph receives ZERO write statements.
        """
        expected = expected_index_set()
        graph = make_graph_mock(_rows_for(expected), header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        result = await backend.ensure_indices(group_id='test')

        assert result.created == []
        assert result.failed == []
        assert result.statements == []
        assert result.already_present == result.expected_total == len(expected)
        assert result.changed is False
        assert _issued(graph) == []

    @pytest.mark.asyncio
    async def test_trap_state_provisions_exactly_the_planned_statements(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The issued statements ARE the plan — β adds nothing and drops nothing."""
        expected = expected_index_set()
        missing = expected - _TRAP_PRESENT
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        result = await backend.ensure_indices(group_id='test')

        planned = [statement for statement, _specs in plan_index_statements(missing)]
        assert _issued(graph) == planned
        assert result.statements == planned
        assert result.already_present == len(_TRAP_PRESENT)
        assert result.failed == []
        assert len(result.created) == result.expected_total - len(_TRAP_PRESENT)
        assert set(result.created) == missing
        assert result.changed is True

    @pytest.mark.asyncio
    async def test_virgin_graph_creates_the_whole_expected_set(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Boundary test 2, mocked: created is spec-granular, so this is an equality."""
        expected = expected_index_set()
        graph = make_graph_mock([], header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        result = await backend.ensure_indices(group_id='test')

        assert result.already_present == 0
        assert result.failed == []
        assert len(result.created) == result.expected_total == len(expected)
        assert set(result.created) == expected


class TestPerStatementFailureIsolation:
    """One rejected statement must not abort the rest — the core of the contract.

    This is the exact shape of the bare loop β replaces: upstream issues its
    statements without per-statement handling, so a single rejection (measured:
    ``Attribute 'uuid' is already indexed``) takes down everything after it, and
    ``falkordb_driver.py``'s ``execute_query`` swallows the error so nothing in
    the logs says so.
    """

    @staticmethod
    def _failing_graph(make_graph_mock, rows, doomed: str):
        graph = make_graph_mock(rows, header=LIVE_HEADER)
        ok_result = MagicMock()
        ok_result.result_set = []
        ok_result.header = LIVE_HEADER

        async def _query(statement, *args, **kwargs):
            if statement == doomed:
                raise RuntimeError('Attribute mock-failure is already indexed')
            return ok_result

        graph.query = AsyncMock(side_effect=_query)
        return graph

    @pytest.mark.asyncio
    async def test_a_mid_list_failure_neither_raises_nor_stops_the_rest(
        self, mock_config, make_backend, make_graph_mock,
    ):
        missing = expected_index_set() - _TRAP_PRESENT
        planned = plan_index_statements(missing)
        doomed_statement, doomed_specs = planned[len(planned) // 2]

        graph = self._failing_graph(make_graph_mock, _rows_for(_TRAP_PRESENT), doomed_statement)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        result = await backend.ensure_indices(group_id='test')

        # Every planned statement was still attempted, including all the ones
        # AFTER the failure. A loop that aborted would issue a strict prefix.
        assert _issued(graph) == [statement for statement, _ in planned]

        failed_specs = [spec for spec, _error in result.failed]
        assert set(failed_specs) == set(doomed_specs)
        for _spec, error in result.failed:
            assert 'already indexed' in error

        assert set(result.created) == missing - set(doomed_specs)

    @pytest.mark.asyncio
    async def test_the_accounting_invariant_holds_across_a_failure(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """``len(created) + len(failed) == len(missing)`` — every spec lands in exactly one."""
        missing = expected_index_set() - _TRAP_PRESENT
        planned = plan_index_statements(missing)
        doomed_statement, _specs = planned[0]

        graph = self._failing_graph(make_graph_mock, _rows_for(_TRAP_PRESENT), doomed_statement)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        result = await backend.ensure_indices(group_id='test')

        assert len(result.created) + len(result.failed) == len(missing)
        assert set(result.created).isdisjoint({spec for spec, _ in result.failed})
        assert result.already_present + len(missing) == result.expected_total
        assert result.changed is True

    @pytest.mark.asyncio
    async def test_every_statement_failing_still_returns_rather_than_raising(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The absorb is unconditional: β never re-raises a per-statement failure."""
        expected = expected_index_set()
        graph = make_graph_mock([], header=LIVE_HEADER)
        graph.query = AsyncMock(side_effect=RuntimeError('everything is broken'))
        backend = make_backend(mock_config)
        _wire(backend, graph)

        result = await backend.ensure_indices(group_id='test')

        assert result.created == []
        assert len(result.failed) == len(expected)
        assert result.changed is True


class TestNoOperationalBarrier:
    """INV-6, pinned POSITIVELY rather than by omission.

    ``CREATE`` returns in 0.5-2.0 ms while the index reaches ``OPERATIONAL`` up to
    594.5 ms later.  Adding a wait here would be an easy, plausible-looking fix
    for a flaky downstream test, and its absence is invisible to any test that
    only checks the return value.  Counting the ``db.indexes()`` calls makes the
    absence of a barrier a property the suite actively defends; establishing that
    an index is SERVING is task ε's canary.
    """

    @pytest.mark.asyncio
    async def test_exactly_one_db_indexes_call_and_no_status_poll(
        self, mock_config, make_backend, make_graph_mock,
    ):
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        await backend.ensure_indices(group_id='test')

        reads = _ro_issued(graph)
        assert reads == ['CALL db.indexes()'], (
            'ensure_indices must read db.indexes() exactly ONCE (the diff read); '
            f'a second read is a status poll in disguise. Reads: {reads!r}'
        )
        assert not any('db.indexes' in statement for statement in _issued(graph)), (
            'no db.indexes() poll may appear on the write path either'
        )


class TestGroupArgCanonicalization:
    """The ``@_canonicalize_group_args`` seam (PRD seam S4).

    Correctness, not hygiene: the CREATE statements resolve a FalkorDB graph KEY
    through ``_graph_for(group_id)``.  The inner ``list_indices`` call being
    decorated does not cover it — the diff read and the writes resolve the key
    independently.
    """

    @pytest.mark.asyncio
    async def test_path_shaped_group_id_is_rejected_before_any_db_call(
        self, mock_config, make_backend,
    ):
        backend = make_backend(mock_config)

        with pytest.raises(PathShapedProjectIdError) as excinfo:
            await backend.ensure_indices(group_id=_PATH_SHAPED)

        assert _PATH_SHAPED in str(excinfo.value)
        assert backend._driver.method_calls == []


#: MEASURED 2026-08-09: what ``CALL db.indexes()`` raises against a graph KEY that
#: does not exist yet.  Reproduced verbatim so the test exercises the real shape —
#: but note the IMPLEMENTATION must not key on this wording (D2); it decides
#: structurally, via ``list_graphs()`` membership.
_EMPTY_KEY_ERROR = redis.exceptions.ResponseError('Invalid graph operation on empty key')


def _with_graph_listing(backend, names: list[str]) -> None:
    """Point the RAW FalkorDB client's ``list_graphs()`` at *names*."""
    backend._driver.client.list_graphs = AsyncMock(return_value=names)


class TestAbsentGraph:
    """A graph that does not exist yet carries zero indices — it is not an error.

    MEASURED 2026-08-09: ``CALL db.indexes()`` against a graph KEY that has never
    been written raises ``redis.exceptions.ResponseError: Invalid graph operation
    on empty key``.  This is live, not hypothetical — PRD D6 names ``autotrade``
    and ``mission_control`` as registered projects with no graph yet, and γ's
    first-write choke point hits exactly this case.  Letting it escape would make
    ``ensure_indices`` raise on precisely the case it exists to serve.  Measured
    too: the CREATE statements auto-create the key, so provisioning from absent is
    self-healing.
    """

    @pytest.mark.asyncio
    async def test_absent_graph_provisions_the_full_set_rather_than_raising(
        self, mock_config, make_backend, make_graph_mock,
    ):
        graph = make_graph_mock([], header=LIVE_HEADER)
        graph.ro_query = AsyncMock(side_effect=_EMPTY_KEY_ERROR)
        backend = make_backend(mock_config)
        _wire(backend, graph)
        _with_graph_listing(backend, ['some_other_graph'])

        result = await backend.ensure_indices(group_id='test')

        assert result.already_present == 0
        assert result.failed == []
        assert len(result.created) == result.expected_total == len(expected_index_set())


class TestUnreachableDriverPropagates:
    """"Raises only on an unreachable driver" is a real clause, not a formality.

    An unreachable driver read as "zero indices" would issue every statement
    against a graph that may already be fully provisioned — and would do so
    silently, which is the failure class this PRD removes.  The absent-graph
    branch is therefore decided STRUCTURALLY (``list_graphs()`` membership), never
    by matching FalkorDB's error wording (D2); an unreachable driver makes
    ``list_graphs()`` raise too, so the error propagates for free.
    """

    @pytest.mark.asyncio
    async def test_read_failure_on_an_EXISTING_graph_propagates(
        self, mock_config, make_backend, make_graph_mock,
    ):
        graph = make_graph_mock([], header=LIVE_HEADER)
        graph.ro_query = AsyncMock(side_effect=redis.exceptions.ConnectionError('down'))
        backend = make_backend(mock_config)
        _wire(backend, graph)
        _with_graph_listing(backend, ['test'])  # the graph DOES exist

        with pytest.raises(redis.exceptions.ConnectionError):
            await backend.ensure_indices(group_id='test')

        assert _issued(graph) == [], 'nothing may be written after a failed diff read'

    @pytest.mark.asyncio
    async def test_a_list_graphs_failure_propagates_rather_than_reading_as_absent(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The existence probe itself failing must not be mistaken for "absent"."""
        graph = make_graph_mock([], header=LIVE_HEADER)
        graph.ro_query = AsyncMock(side_effect=_EMPTY_KEY_ERROR)
        backend = make_backend(mock_config)
        _wire(backend, graph)
        backend._driver.client.list_graphs = AsyncMock(
            side_effect=redis.exceptions.ConnectionError('down')
        )

        with pytest.raises(Exception):  # noqa: B017 - any propagation, none absorbed
            await backend.ensure_indices(group_id='test')

        assert _issued(graph) == []


class TestStructuredLogging:
    """D7: the absorb must never be silent, and a no-op must never claim success.

    The DEBUG ``Ensured indices on graph`` line this replaces fired
    unconditionally after a no-op ``pass``, and at DEBUG — so at the service's
    INFO level it produced neither a positive nor a negative signal.  There was
    no signal in the logs at all.
    """

    _LOGGER = 'fused_memory.backends.graphiti_client'

    @pytest.mark.asyncio
    async def test_warning_names_the_failing_statement_and_its_error(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        missing = expected_index_set() - _TRAP_PRESENT
        doomed_statement, _specs = plan_index_statements(missing)[0]

        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)

        async def _query(statement, *args, **kwargs):
            if statement == doomed_statement:
                raise RuntimeError('mock-rejection: already indexed')
            return MagicMock(result_set=[], header=LIVE_HEADER)

        graph.query = AsyncMock(side_effect=_query)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            await backend.ensure_indices(group_id='test')

        warnings = '\n'.join(
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        )
        assert doomed_statement in warnings
        assert 'mock-rejection: already indexed' in warnings

    @pytest.mark.asyncio
    async def test_info_reports_the_structured_counts_when_something_changed(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            result = await backend.ensure_indices(group_id='test')

        infos = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and 'ndices' in r.getMessage()
        ]
        assert infos, 'a run that provisioned something must say so at INFO'
        message = '\n'.join(infos)
        for count in (
            len(result.created), result.already_present,
            len(result.failed), result.expected_total,
        ):
            assert str(count) in message

    @pytest.mark.asyncio
    async def test_an_unchanged_run_claims_nothing(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        """INV-2: a no-op must not emit a line an operator would read as "provisioned"."""
        graph = make_graph_mock(_rows_for(expected_index_set()), header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            result = await backend.ensure_indices(group_id='test')

        assert result.changed is False
        assert [r for r in caplog.records if r.levelno >= logging.INFO] == []

    @pytest.mark.asyncio
    async def test_the_false_positive_debug_line_is_never_emitted(
        self, mock_config, make_backend, make_graph_mock, caplog,
    ):
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        with caplog.at_level(logging.DEBUG, logger=self._LOGGER):
            await backend.ensure_indices(group_id='test')

        assert not any(
            'Ensured indices on graph' in r.getMessage() for r in caplog.records
        )
