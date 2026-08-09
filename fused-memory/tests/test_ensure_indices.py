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

from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest
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
