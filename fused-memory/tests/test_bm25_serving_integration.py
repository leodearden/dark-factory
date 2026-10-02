"""Live FalkorDB: the PRODUCTION index set is present AND BM25 serves on it (task 3710).

PRD ``docs/prds/falkordb-index-provisioning.md`` ε: the D9 serving canary and
boundary test 6.  Requires a running FalkorDB; skipped automatically when one is
not reachable, and deselected by the default ``-m 'not integration'`` addopts.

Index METADATA saying "present" does not mean BM25 serves: a fulltext query
against an absent index returns no rows and no error, which is how BM25 returned
nothing for four months unnoticed.  The canary therefore judges service by
issuing graphiti's own BM25 query for a seeded token, never by reading
``db.indexes()``.  ``test_falkor_fulltext_integration.py`` cannot stand in for
it: that module builds its own one-field index, so a graph carrying no
production index at all passes it.

Corpora are SEEDED EPHEMERAL (PRD Open Question 2): every test seeds its own
``unique_graph_name`` scratch graph and deletes it in ``finally``.

HAZARD compliance mirrors ``test_index_provisioning_wiring_integration.py``:
scratch graphs only; ``GraphitiBackend.initialize()`` is never called; every
backend gets an EXPLICIT ``registered_graph_ids`` naming only scratch graphs;
``_MultiTenantFalkorDriver`` is injected into ``backend._driver`` and a bare
``FalkorDriver`` is never constructed.
"""

from __future__ import annotations

import pytest
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    await_index_operational,
    falkor_skipif,
    retry_until_observed,
    unique_graph_name,
)

pytestmark = [
    falkor_skipif(),
    # Provisioning and settling are bounded well under this; `timeout_method =
    # "thread"` os._exit(1)s the xdist worker, so an under-budget timeout would
    # read as an infrastructure crash.
    pytest.mark.timeout(120),
    pytest.mark.integration,
]


class TestCanaryDistinguishesPresentFromServing:
    """PRD boundary test 6: an index metadata calls present need not be serving."""

    @pytest.mark.asyncio
    async def test_not_serving_while_under_construction_then_serving_once_operational(
        self, scratch, live_backend_factory,
    ):
        name, graph = scratch('under_construction')
        backend = live_backend_factory(set())
        entity = next(t for t in production_fulltext_targets() if t.label == 'Entity')
        statement = _production_fulltext_statement(entity)
        await _seed_filler(graph, entity, group_id=name, count=_UNDER_CONSTRUCTION_CORPUS)
        await seed_canary_element(graph, entity, group_id=name)
        await graph.query(statement)

        async def observe_while_under_construction() -> CanaryReading | None:
            # The predicate is the status bracket ONLY: retrying until the
            # verdict reads NOT_SERVING would make this test a tautology.
            if not await _index_unsettled(backend, name, entity.label):
                return None
            reading = await bm25_canary(graph, entity, group_id=name)
            if not await _index_unsettled(backend, name, entity.label):
                return None
            return reading

        async def rebuild() -> None:
            await await_index_operational(graph)
            await graph.query(f"CALL db.idx.fulltext.drop('{entity.label}')")
            await graph.query(statement)

        unready = await retry_until_observed(
            observe_while_under_construction,
            reopen=rebuild,
            attempts=_OBSERVATION_ATTEMPTS,
            message=(
                f'the {entity.label} fulltext index was never listed present and '
                'not OPERATIONAL on both sides of the canary query'
            ),
        )
        assert unready.serving is False, f'served while UNDER CONSTRUCTION: {unready}'

        await await_index_operational(graph)
        ready = await bm25_canary(graph, entity, group_id=name)
        assert ready.serving is True, f'not serving once OPERATIONAL: {ready}'


class TestProductionIndexesServe:
    """BM25 serves on exactly the index set production provisions."""

    @pytest.mark.asyncio
    async def test_a_graph_without_indices_serves_nothing(
        self, scratch, live_backend_factory,
    ):
        """The four-months-ago state: tokens are there, no index is."""
        name, graph = scratch('no_indices')
        backend = live_backend_factory(set())
        targets = production_fulltext_targets()
        for target in targets:
            await seed_canary_element(graph, target, group_id=name)

        assert await backend.list_indices(group_id=name) == []
        verdicts = {
            t.label: (await bm25_canary(graph, t, group_id=name)).serving for t in targets
        }
        assert verdicts == {t.label: False for t in targets}

    @pytest.mark.asyncio
    async def test_production_startup_sweep_serves_every_fulltext_target(
        self, scratch, live_backend_factory,
    ):
        name, graph = scratch('sweep_serves')
        targets = production_fulltext_targets()
        # Seeding also creates the graph KEY, which the sweep requires.
        for target in targets:
            await seed_canary_element(graph, target, group_id=name)
        backend = live_backend_factory({name})

        await backend.provision_registered_graphs()
        await await_index_operational(graph)

        readings = [await bm25_canary(graph, t, group_id=name) for t in targets]
        assert all(r.serving for r in readings), readings
