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

``_MultiTenantFalkorDriver`` is IMPORTED here, but never INSTANTIATED: the D4
guard reads the class ``__dict__`` and invokes the unbound override with a mock
``self``.  Importing the class performs no I/O; it is ``__init__`` that
fire-and-forgets.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import redis.exceptions

# ``_TRAP_PRESENT`` — the trap state esc-3375-1 protected as evidence — is
# IMPORTED, not restated: it had one definition per suite, and a future change to
# what the trap contains would have had to be made twice or the two suites would
# silently disagree about what they were testing (INV-5: single home).
from test_falkor_indices import _TRAP_PRESENT, LIVE_HEADER

from fused_memory.backends.falkor_indices import (
    IndexHeaderShapeError,
    IndexRecordShapeError,
    IndexSpec,
    expected_index_set,
    parse_index_statement,
    plan_index_statements,
)
from fused_memory.backends.graphiti_client import _MultiTenantFalkorDriver


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

        assert result.created == ()
        assert result.failed == ()
        assert result.statements == ()
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
        assert result.statements == tuple(planned)
        assert result.already_present == len(_TRAP_PRESENT)
        assert result.failed == ()
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
        assert result.failed == ()
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

        assert result.created == ()
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


class _StatefulGraph:
    """A graph double whose ``db.indexes()`` reflects the CREATEs it has accepted.

    The plain ``make_graph_mock`` returns a FIXED row set, so a second concurrent
    call sees the same pre-write state no matter what the first one did — which
    makes the race under test invisible.  This double closes that: each accepted
    statement is folded back into the reported index set through α's own parser,
    so the read side is a real function of the write side.

    Both methods ``await asyncio.sleep(0)``, and that is LOAD-BEARING, not
    decoration: an ``AsyncMock`` resolves without ever suspending, so two gathered
    calls would run strictly one-after-the-other and the test would pass against
    an unserialized implementation.  The explicit yield is what lets the two
    interleave at all.
    """

    def __init__(self, present: set[IndexSpec]):
        self.present = set(present)
        self.issued: list[str] = []

    @staticmethod
    def _result(rows):
        result = MagicMock()
        result.result_set = rows
        result.header = LIVE_HEADER
        return result

    async def ro_query(self, statement, *args, **kwargs):
        await asyncio.sleep(0)
        return self._result(_rows_for(self.present))

    async def query(self, statement, *args, **kwargs):
        await asyncio.sleep(0)
        self.issued.append(statement)
        self.present |= set(parse_index_statement(statement))
        return self._result([])


class TestConcurrentProvisioningIsSerialized:
    """Read-diff-write is not atomic by itself — two callers for ONE graph race.

    Unserialized, both observe the same pre-write ``actual`` and both issue the
    whole plan; the loser collects ~30 ``Attribute '...' is already indexed``
    rejections into ``failed`` and logs a WARNING for each.  That is
    indistinguishable from a genuine provisioning failure, and it is precisely the
    signal task δ's drift detector keys on — so the noise would land in the one
    place it is most expensive.  γ's first-write choke point makes the concurrent
    case the NORMAL case, not a corner.
    """

    @pytest.mark.asyncio
    async def test_two_concurrent_calls_for_one_graph_do_not_both_issue_the_plan(
        self, mock_config, make_backend,
    ):
        expected = expected_index_set()
        planned = plan_index_statements(expected - _TRAP_PRESENT)
        graph = _StatefulGraph(_TRAP_PRESENT)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        first, second = await asyncio.gather(
            backend.ensure_indices(group_id='test'),
            backend.ensure_indices(group_id='test'),
        )

        # No statement is issued twice: the loser read state the winner had
        # already written, so its diff was empty.
        assert graph.issued == [statement for statement, _specs in planned]

        winner, loser = sorted((first, second), key=lambda r: -len(r.statements))
        assert list(winner.statements) == graph.issued
        assert winner.failed == () and loser.failed == (), (
            'a serialized pair must produce no "already indexed" rejections at '
            f'all: {winner.failed!r} / {loser.failed!r}'
        )
        assert loser.statements == ()
        assert loser.changed is False
        assert loser.already_present == loser.expected_total == len(expected)

    @pytest.mark.asyncio
    async def test_the_provisioning_lock_is_per_graph_and_not_the_identity_lock(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Separate registry, keyed canonically — reusing the identity lock deadlocks.

        ``asyncio.Lock`` is not reentrant, and γ calls ``ensure_indices`` from a
        write path that may already hold ``_identity_lock_for(group_id)``; sharing
        one lock between the two would hang that call rather than serialize it.
        """
        graph = make_graph_mock(_rows_for(expected_index_set()), header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        await backend.ensure_indices(group_id='My-Project')
        await backend.ensure_indices(group_id='my_project')

        assert set(backend._index_provision_locks) == {'my_project'}, (
            'the lock must be keyed on the CANONICAL id, or two spellings of one '
            'graph take two different locks and do not serialize'
        )
        assert (
            backend._index_provision_locks['my_project']
            is not backend._identity_lock_for('my_project')
        ), 'the provisioning lock must not be the write-time-identity lock'


class TestGroupArgCanonicalization:
    """The ``@_canonicalize_group_args`` seam (PRD seam S4), POSITIVE form.

    Correctness, not hygiene: the CREATE statements resolve a FalkorDB graph KEY
    through ``_graph_for(group_id)``.  The inner ``list_indices`` call being
    decorated does not cover it — the diff read and the writes resolve the key
    independently.

    The NEGATIVE form (a path-shaped group_id rejected before any DB call) is not
    restated here: ``test_graphiti_group_arg_canonicalization.py``'s sweep table
    is its authoritative home and already carries an ``ensure_indices`` entry,
    complete with the "no driver calls, no client calls" assertions.  Duplicating
    it here bought a second place to update and no coverage.  What the sweep does
    NOT cover is this — the WRITE-path key resolution.
    """

    @pytest.mark.asyncio
    async def test_the_write_path_resolves_the_CANONICAL_graph_key(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Every ``_get_graph`` call this run makes asks for the canonical key.

        Asserted over the WHOLE call list, not just the first: a regression that
        canonicalized only the read path (removing the outer decorator while
        ``list_indices`` stays decorated) would point the diff read at
        ``my_project`` and the CREATEs at ``My-Project`` — two different FalkorDB
        graphs — and every other test in this module would stay green, since they
        all pass an already-canonical id.
        """
        graph = make_graph_mock(_rows_for(_TRAP_PRESENT), header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)

        result = await backend.ensure_indices(group_id='My-Project')

        assert result.statements, 'the trap state must have produced writes to place'
        assert backend._driver._get_graph.call_args_list == [
            call('my_project')
        ] * backend._driver._get_graph.call_count, (
            'ensure_indices resolved a non-canonical graph key: '
            f'{backend._driver._get_graph.call_args_list!r}'
        )


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
        assert result.failed == ()
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

        with pytest.raises(redis.exceptions.ConnectionError):
            await backend.ensure_indices(group_id='test')

        assert _issued(graph) == []


class TestAlphaShapeErrorsFailClosed:
    """α's two fail-closed errors must never be absorbed by the absent-graph branch.

    ``IndexHeaderShapeError`` (FalkorDB changed its result columns) and
    ``IndexRecordShapeError`` (a record projected to the wrong shape) exist
    precisely so an index state that could NOT be determined is never reported as
    a determined one.  Both tests point ``list_graphs()`` at a listing WITHOUT the
    graph, i.e. straight down the absorbing branch — a handler broad enough to
    catch them there would read the graph as carrying zero indices and issue the
    entire plan against a graph whose real state was never read (INV-4), which is
    the silent-fail-soft class the PRD exists to remove.
    """

    @pytest.mark.asyncio
    async def test_an_unresolvable_header_propagates_rather_than_reading_as_absent(
        self, mock_config, make_backend, make_graph_mock,
    ):
        graph = make_graph_mock([], header=[[1, 'surprise']])
        backend = make_backend(mock_config)
        _wire(backend, graph)
        _with_graph_listing(backend, ['some_other_graph'])  # absorbing branch

        with pytest.raises(IndexHeaderShapeError):
            await backend.ensure_indices(group_id='test')

        assert _issued(graph) == [], 'nothing may be written on an undetermined state'

    @pytest.mark.asyncio
    async def test_a_malformed_record_propagates_rather_than_reading_as_absent(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The normalize step is outside the absent-graph ``try`` — pinned here."""
        malformed = [['Entity', [], {}, {}, 'english', [], 'NODE', 'OPERATIONAL', {}]]
        graph = make_graph_mock(malformed, header=LIVE_HEADER)
        backend = make_backend(mock_config)
        _wire(backend, graph)
        _with_graph_listing(backend, ['some_other_graph'])  # absorbing branch

        with pytest.raises(IndexRecordShapeError):
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
        # Asserted as FORMATTED FRAGMENTS, not as bare digit-strings: `created=36
        # ... failed=0` and a regression that transposed the two in the logger's
        # argument list emit the same four numbers, so a positionally-blind
        # `str(count) in message` check would wave the transposition through.
        for fragment in (
            f'created={len(result.created)}',
            f'already_present={result.already_present}',
            f'failed={len(result.failed)}',
            f'expected_total={result.expected_total}',
        ):
            assert fragment in message, (
                f'expected {fragment!r} in the INFO line, got: {message!r}'
            )

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
        # Filtered to THIS module's logger: caplog's handler sits on the root
        # logger, so an unrelated library emitting one INFO record during the
        # call would otherwise fail this spuriously.
        assert [
            r for r in caplog.records
            if r.name == self._LOGGER and r.levelno >= logging.INFO
        ] == []

    @pytest.mark.asyncio
    async def test_the_deliberate_no_op_claims_nothing_at_any_level(
        self, mock_config, make_backend, caplog,
    ):
        """INV-2: ``_ensure_indices`` does nothing, so it must claim nothing — at ANY level.

        ``build_indices_and_constraints`` is a ``pass`` override (D4), so this
        method provisions no index.  A log line emitted after it would be a
        false positive whatever it said, and DEBUG is the worst place to say it:
        below the service's INFO level it is neither a positive nor a negative
        signal.  So the property asserted here is the ABSENCE of any record from
        this module's logger, with NO claim about wording — a differently-phrased
        re-addition is exactly as wrong, and a substring check would wave it
        through.

        Non-vacuity matters as much as the absence: the ``_indexed_graphs`` early
        return would let this pass with the body never executing, so the awaited
        build and the memoised group_id are asserted too.

        HAZARD: the driver is a bare mock.  Constructing a real ``FalkorDriver``
        (or ``_MultiTenantFalkorDriver``) here would fire-and-forget a genuine
        index build under the running loop.
        """
        driver = MagicMock()
        driver.build_indices_and_constraints = AsyncMock(return_value=None)
        backend = make_backend(mock_config)
        backend._driver_for = MagicMock(return_value=driver)
        assert 'test' not in backend._indexed_graphs, (
            'the memoisation set must start empty or the body never runs'
        )

        with caplog.at_level(logging.DEBUG, logger=self._LOGGER):
            await backend._ensure_indices('test')

        assert [r for r in caplog.records if r.name == self._LOGGER] == []
        assert driver.build_indices_and_constraints.await_count == 1
        assert 'test' in backend._indexed_graphs


class TestProvisioningHazardGuards:
    """The D4 override β must NOT break while adding a provisioning path.

    Both guards assert on the class ``__dict__`` and on invoking the override —
    behaviour, not source text.
    """

    def test_the_build_indices_override_is_still_a_no_op_on_the_subclass(self):
        """D4: removal of the ``pass`` override was EXPLICITLY REJECTED.

        ``FalkorDriver.__init__`` fire-and-forgets ``build_indices_and_constraints()``
        when an event loop is running, and every per-group driver handed out by
        ``_driver_for()`` / ``_client_for()`` comes from ``clone()``.  The override
        suppresses that implicit build on every clone — removing it is the path
        that caused the ``723ec915c3`` connection storm (166 leaked connections).

        Asserted against the subclass ``__dict__``, not mere callability: an
        INHERITED upstream method is callable too, so a callability check would
        pass with the override deleted.
        """
        assert 'build_indices_and_constraints' in _MultiTenantFalkorDriver.__dict__, (
            'the build_indices_and_constraints override was removed from '
            '_MultiTenantFalkorDriver; D4 rejects that explicitly'
        )

    @pytest.mark.asyncio
    async def test_the_override_returns_none_without_building_anything(self):
        driver = _MultiTenantFalkorDriver.__dict__['build_indices_and_constraints']
        assert await driver(MagicMock()) is None
