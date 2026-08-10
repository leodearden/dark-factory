"""Live FalkorDB boundary tests for ``ensure_indices`` (task 3707, β).

Requires a running FalkorDB; skipped automatically when one is not reachable.

What only a live graph can show
-------------------------------
``tests/test_ensure_indices.py`` pins the algorithm against mocks, but the whole
reason this PRD exists is that the mocked belief and the real graph disagreed:
upstream's composite ``CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id,
n.name, n.created_at)`` is rejected WHOLESALE when any listed property is already
indexed, and ``falkordb_driver.py``'s ``execute_query`` swallows the rejection.
Nothing short of a real FalkorDB proves the per-property form converges where the
composite silently did not.

The three boundary tests, all MEASURED 2026-08-09 before implementation:

1. **Trap state** (an ``Entity(uuid)`` and a ``RELATES_TO(uuid)`` range index, the
   state esc-3375-1 protected as evidence) → 28 statements, 0 failures, and all
   11 previously-lost range fields present afterwards, with ZERO composite range
   statements issued.
2. **Virgin graph** → 30 statements creating all 38 specs, and the full round-trip
   ``normalize_index_records(list_indices(g)) == expected_index_set()`` EXACTLY —
   no missing, no extra.  That exactness is what makes ``already_present ==
   expected_total`` reachable on a re-run; it was not safe to assume, since this
   PRD exists because that round-trip was silently wrong.
3. **Idempotent re-run** → 0 statements, ``created == ()``,
   ``already_present == expected_total``, ``failed == ()``.

Plus the absent-graph case: ``CALL db.indexes()`` against a graph KEY that was
never written raises ``Invalid graph operation on empty key``, and the CREATE
statements auto-create the key — so provisioning from absent is self-healing.

Counts are asserted as DERIVED quantities, never as the literal 28/30/38, so a
graphiti upgrade moves them instead of leaving a stale magic number passing.

HAZARD compliance
-----------------
Every graph here is a fresh uuid-suffixed scratch graph, deleted in ``finally``;
no test ever names a real project graph.  ``GraphitiBackend.initialize()`` is
NEVER called — its startup enumeration would sweep every graph on the server,
including the six real trap graphs whose index state is protected evidence.  A
bare ``graphiti_core`` ``FalkorDriver`` is never constructed either: its
``__init__`` fire-and-forgets ``build_indices_and_constraints()`` under a running
loop.  ``_MultiTenantFalkorDriver``'s D4 ``pass`` override neuters that, and it is
injected into ``backend._driver`` directly.  No DUMP/RESTORE anywhere.
"""

from __future__ import annotations

import contextlib

import pytest
import pytest_asyncio
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    await_index_operational,
    falkor_skipif,
    unique_graph_name,
)
from falkordb.asyncio import FalkorDB

from fused_memory.backends.falkor_indices import (
    expected_index_set,
    normalize_index_records,
    parse_index_statement,
)
from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver

pytestmark = [
    falkor_skipif(),
    # Provisioning writes ~30 statements and each boundary test waits for every
    # one of them to reach OPERATIONAL, so this module opts up from the suite
    # default of 60s deliberately -- `timeout_method = "thread"` os._exit(1)s the
    # whole xdist worker, so an under-budget timeout here would read as an
    # infrastructure crash rather than a slow test.
    pytest.mark.timeout(120),
    pytest.mark.integration,
]

#: The 11 range fields the composite form loses silently against the trap state.
#: Named explicitly because "all 11 restored" is the task's user-observable signal.
_TRAP_LOST_RANGE_FIELDS = [
    ('Entity', 'NODE', field) for field in ('uuid', 'group_id', 'name', 'created_at')
] + [
    ('RELATES_TO', 'RELATIONSHIP', field)
    for field in ('uuid', 'group_id', 'name', 'created_at', 'expired_at',
                  'valid_at', 'invalid_at')
]


@pytest_asyncio.fixture
async def scratch():
    """Factory for uuid-suffixed throwaway graphs, each torn down in ``finally``.

    A SEPARATE graph per test, because the suite runs under
    ``-n auto --dist loadgroup`` and two tests sharing a graph name would
    cross-contaminate each other's index state.

    Deliberately does NOT write to the graph: ``select_graph`` alone does not
    create the KEY, which is what lets the absent-graph test observe the real
    ``Invalid graph operation on empty key`` path.
    """
    clients: list[FalkorDB] = []
    graphs: list = []

    def _make(slug: str):
        name = unique_graph_name(f'3707_ensure_indices_{slug}')
        client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        clients.append(client)
        graph = client.select_graph(name)
        graphs.append(graph)
        return name, graph

    try:
        yield _make
    finally:
        for graph in graphs:
            with contextlib.suppress(Exception):
                await graph.delete()
        for client in clients:
            with contextlib.suppress(Exception):
                await client.aclose()


@pytest_asyncio.fixture
async def live_backend(mock_config):
    """A backend wired to a REAL driver, without ``initialize()``.

    ``_MultiTenantFalkorDriver`` rather than a bare ``FalkorDriver``: the D4
    ``pass`` override is what stops ``__init__`` from fire-and-forgetting an index
    build on whatever graph it touches first.
    """
    backend = GraphitiBackend(mock_config)
    backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
    try:
        yield backend
    finally:
        await backend.close()


def _range_statements(statements: list[str]) -> list[str]:
    return [s for s in statements if parse_index_statement(s)[0][3] == 'RANGE']


class TestBoundary1TrapState:
    """The state esc-3375-1 protected: two uuid range indices and nothing else.

    MEASURED: upstream's composite ``Entity`` statement is rejected verbatim
    against this state with ``Attribute 'uuid' is already indexed`` — one
    already-present property costs all four.
    """

    @pytest.mark.asyncio
    async def test_all_eleven_lost_range_fields_are_restored(self, scratch, live_backend):
        name, graph = scratch('trap')
        await graph.query('CREATE INDEX FOR (n:Entity) ON (n.uuid)')
        await graph.query('CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid)')
        await await_index_operational(graph)

        result = await live_backend.ensure_indices(group_id=name)
        await await_index_operational(graph)

        assert result.failed == (), f'provisioning the trap state failed: {result.failed}'

        actual = normalize_index_records(await live_backend.list_indices(group_id=name))
        missing_after = [
            (label, entity_type, field)
            for label, entity_type, field in _TRAP_LOST_RANGE_FIELDS
            if (label, entity_type, field, 'RANGE') not in actual
        ]
        assert missing_after == [], (
            f'range fields still missing after provisioning: {missing_after}'
        )

    @pytest.mark.asyncio
    async def test_no_composite_range_statement_is_ever_issued(self, scratch, live_backend):
        """Boundary test 1's negative half, observed live rather than inferred.

        Each issued RANGE statement is parsed back through α's own parser and must
        fan out to exactly ONE spec.  Upstream's ``Entity`` statement fans out to
        4 and its ``RELATES_TO`` one to 7.
        """
        name, graph = scratch('no_composite')
        await graph.query('CREATE INDEX FOR (n:Entity) ON (n.uuid)')
        await graph.query('CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid)')
        await await_index_operational(graph)

        result = await live_backend.ensure_indices(group_id=name)
        await await_index_operational(graph)

        assert result.statements, 'the trap state must have produced work to do'
        for statement in _range_statements(result.statements):
            specs = parse_index_statement(statement)
            assert len(specs) == 1, f'composite RANGE statement issued: {statement}'


class TestBoundary2VirginGraph:
    """A graph that exists but carries no indices."""

    @pytest.mark.asyncio
    async def test_creates_the_whole_expected_set_and_round_trips_exactly(
        self, scratch, live_backend,
    ):
        """The EXACTNESS half was verified, not assumed.

        ``normalize_index_records(list_indices(g)) == expected_index_set()`` with
        nothing missing AND nothing extra is what makes ``already_present ==
        expected_total`` reachable on the re-run.  Assuming it would have been
        exactly the mistake this PRD exists to correct — that round-trip was
        silently wrong before task 3706.
        """
        expected = expected_index_set()
        name, graph = scratch('virgin')
        await graph.query('CREATE (:Probe {seed: 1})')  # make the KEY exist, no indices

        result = await live_backend.ensure_indices(group_id=name)
        await await_index_operational(graph)

        assert result.failed == (), f'provisioning a virgin graph failed: {result.failed}'
        assert result.already_present == 0
        assert len(result.created) == result.expected_total == len(expected)

        actual = normalize_index_records(await live_backend.list_indices(group_id=name))
        assert expected - actual == set(), f'missing after provisioning: {expected - actual}'
        assert actual - expected == set(), f'unexpected after provisioning: {actual - expected}'


class TestBoundary3IdempotentRerun:
    """Running twice writes nothing the second time."""

    @pytest.mark.asyncio
    async def test_second_run_issues_no_statements_at_all(self, scratch, live_backend):
        name, graph = scratch('idempotent')
        await graph.query('CREATE (:Probe {seed: 1})')

        first = await live_backend.ensure_indices(group_id=name)
        await await_index_operational(graph)
        assert first.failed == ()

        second = await live_backend.ensure_indices(group_id=name)

        assert second.created == ()
        assert second.failed == ()
        assert second.statements == ()
        assert second.already_present == second.expected_total
        assert second.changed is False


class TestAbsentGraphLive:
    """A registered project whose graph key was never written.

    PRD D6 names ``autotrade`` and ``mission_control`` as exactly this, and γ's
    first-write choke point hits it — so the measured ``Invalid graph operation on
    empty key`` from ``CALL db.indexes()`` must not escape.
    """

    @pytest.mark.asyncio
    async def test_provisions_from_absent_rather_than_raising(self, scratch, live_backend):
        expected = expected_index_set()
        name, graph = scratch('absent')  # deliberately never written to

        result = await live_backend.ensure_indices(group_id=name)
        await await_index_operational(graph)

        assert result.failed == (), f'provisioning an absent graph failed: {result.failed}'
        assert result.already_present == 0
        assert len(result.created) == result.expected_total == len(expected)
