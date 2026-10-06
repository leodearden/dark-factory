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
scratch graph, deleted on exit.  Graphs and backends come only from
``_falkor_index_live``, which enforces the live-FalkorDB HAZARD rules.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pytest
import pytest_asyncio
from _falkor_index_live import (
    injected_driver,
    live_backends,
    missing_production_indices,
    scratch_graphs,
)
from _fm_helpers import await_index_operational, falkor_skipif, retry_until_observed
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.graph_queries import (
    NEO4J_TO_FALKORDB_MAPPING,
    get_fulltext_indices,
    get_nodes_query,
    get_relationships_query,
)
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT, fulltext_query

from fused_memory.backends.falkor_indices import (
    IndexSpec,
    expected_index_set,
    parse_index_statement,
    unsettled_index_statuses,
)
from fused_memory.backends.graphiti_client import GraphitiBackend

pytestmark = [
    falkor_skipif(),
    # Provisioning and settling are bounded well under this; `timeout_method =
    # "thread"` os._exit(1)s the xdist worker, so an under-budget timeout would
    # read as an infrastructure crash.
    pytest.mark.timeout(120),
    pytest.mark.integration,
]


# --- The D9 serving canary -------------------------------------------------

# Not a graphiti stopword, alphanumeric, and a single RediSearch token.
CANARY_TOKEN = 'bm25canary'

_FULLTEXT_INDEX_NAME_BY_LABEL = {
    label: index_name for index_name, label in NEO4J_TO_FALKORDB_MAPPING.items()
}


@dataclass(frozen=True)
class FulltextTarget:
    """One production fulltext index: what BM25 queries, and which fields hold text."""

    label: str
    entity_type: str
    text_fields: tuple[str, ...]


def production_fulltext_targets() -> tuple[FulltextTarget, ...]:
    """Every fulltext index production provisions, derived from ``expected_index_set``."""
    fields_by_target: dict[tuple[str, str], set[str]] = defaultdict(set)
    for label, entity_type, field, index_type in expected_index_set():
        if index_type == 'FULLTEXT':
            fields_by_target[(label, entity_type)].add(field)
    targets = []
    for (label, entity_type), fields in sorted(fields_by_target.items()):
        text_fields = tuple(sorted(fields - {'group_id'}))
        if not text_fields:
            raise ValueError(
                f'fulltext target {label!r} ({entity_type}) indexes no field but '
                'group_id, so no canary token can be seeded into it'
            )
        targets.append(FulltextTarget(label, entity_type, text_fields))
    return tuple(targets)


async def seed_canary_element(graph, target: FulltextTarget, *, group_id: str) -> None:
    """Write one element carrying ``CANARY_TOKEN`` in every text field of *target*."""
    props = dict.fromkeys(target.text_fields, CANARY_TOKEN) | {'group_id': group_id}
    if target.entity_type == 'NODE':
        pattern = f'(x:{target.label})'
    elif target.entity_type == 'RELATIONSHIP':
        # A neutral endpoint label, so the edge never feeds a node target.
        pattern = f'(:CanaryEndpoint)-[x:{target.label}]->(:CanaryEndpoint)'
    else:
        raise ValueError(f'cannot seed entity_type {target.entity_type!r} for {target}')
    await graph.query(f'CREATE {pattern} SET x += $props', {'props': props})


@dataclass(frozen=True)
class CanaryReading:
    """How many rows BM25 returned for the canary token on one target.

    Deliberately has no ``__bool__``/``__len__``: ``retry_until_observed`` reads
    a falsy observation as a miss, so a falsy NOT_SERVING reading would be
    retried until SERVING and boundary test 6 would become a tautology.
    """

    target: FulltextTarget
    rows: int

    @property
    def serving(self) -> bool:
        return self.rows >= 1


async def bm25_canary(
    graph, target: FulltextTarget, *, group_id: str, driver: GraphDriver,
) -> CanaryReading:
    """Issue graphiti's own BM25 query for ``CANARY_TOKEN`` and count the rows.

    *driver* assembles the query exactly as it would for graphiti.  Measures
    SERVICE only: it never consults ``CALL db.indexes()``.
    """
    query = fulltext_query(CANARY_TOKEN, [group_id], driver)
    if query == '':
        raise AssertionError(
            f'fulltext_query built no query for {CANARY_TOKEN!r} in {group_id!r}; '
            'a canary that never queries proves nothing'
        )
    index_name = _FULLTEXT_INDEX_NAME_BY_LABEL[target.label]
    if target.entity_type == 'NODE':
        procedure = get_nodes_query(
            index_name, '$query', limit=RELEVANT_SCHEMA_LIMIT, provider=GraphProvider.FALKORDB,
        ) + ' YIELD node RETURN id(node)'
    elif target.entity_type == 'RELATIONSHIP':
        procedure = get_relationships_query(
            index_name, limit=RELEVANT_SCHEMA_LIMIT, provider=GraphProvider.FALKORDB,
        ) + ' YIELD relationship RETURN id(relationship)'
    else:
        raise ValueError(f'cannot query entity_type {target.entity_type!r} for {target}')
    result = await graph.query(procedure, {'query': query})
    return CanaryReading(target, len(result.result_set))


# --- Boundary test 6: holding the UNDER CONSTRUCTION window open -----------

# A one-node corpus is OPERATIONAL before the first status read.
_UNDER_CONSTRUCTION_CORPUS = 50_000
# Independent window openings before the observation is declared impossible.
_OBSERVATION_ATTEMPTS = 5

_FILLER_TEXT = 'filler'


def _production_fulltext_statement(target: FulltextTarget) -> str:
    """The one statement graphiti emits to create *target*'s fulltext index."""
    statements = [
        statement
        for statement in get_fulltext_indices(GraphProvider.FALKORDB)
        if parse_index_statement(statement)[0][:2] == (target.label, target.entity_type)
    ]
    if len(statements) != 1:
        raise ValueError(f'expected one fulltext statement for {target}, got {statements!r}')
    return statements[0]


async def _seed_filler(graph, target: FulltextTarget, *, group_id: str, count: int) -> None:
    """Write *count* token-free elements for *target*, in one query."""
    if target.entity_type != 'NODE':
        raise ValueError(f'filler is seeded for NODE targets only, got {target}')
    props = dict.fromkeys(target.text_fields, _FILLER_TEXT) | {'group_id': group_id}
    await graph.query(
        f'UNWIND range(1, $count) AS i CREATE (x:{target.label}) SET x += $props',
        {'count': count, 'props': props},
    )


async def _index_unsettled(backend: GraphitiBackend, group_id: str, label: str) -> bool:
    """Whether *label*'s index is listed AND not OPERATIONAL; an absent index is False."""
    records = await backend.list_indices(group_id=group_id)
    return any(unsettled == label for unsettled, _ in unsettled_index_statuses(records))


# --- The production expected-set check --------------------------------------

_DROP_KEYWORD_BY_INDEX_TYPE = {'RANGE': '', 'FULLTEXT': 'FULLTEXT '}
_DROP_PATTERN_BY_ENTITY_TYPE = {'NODE': '(x:{label})', 'RELATIONSHIP': '()-[x:{label}]-()'}


def _drop_statement(spec: IndexSpec) -> str:
    """The statement that drops exactly *spec*; an unknown type raises KeyError."""
    label, entity_type, field, index_type = spec
    keyword = _DROP_KEYWORD_BY_INDEX_TYPE[index_type]
    pattern = _DROP_PATTERN_BY_ENTITY_TYPE[entity_type].format(label=label)
    return f'DROP {keyword}INDEX FOR {pattern} ON (x.{field})'


# --- Fixtures ---------------------------------------------------------------


@pytest_asyncio.fixture
async def scratch():
    async with scratch_graphs('3710') as make:
        yield make


@pytest_asyncio.fixture
async def live_backend_factory(mock_config):
    async with live_backends(mock_config) as make:
        yield make


class TestCanaryDistinguishesPresentFromServing:
    """PRD boundary test 6: an index metadata calls present need not be serving."""

    @pytest.mark.asyncio
    async def test_not_serving_while_under_construction_then_serving_once_operational(
        self, scratch, live_backend_factory,
    ):
        name, graph = scratch('under_construction')
        backend = live_backend_factory(set())
        driver = injected_driver(backend)
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
            reading = await bm25_canary(graph, entity, group_id=name, driver=driver)
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
        ready = await bm25_canary(graph, entity, group_id=name, driver=driver)
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
        driver = injected_driver(backend)
        verdicts = {
            t.label: (await bm25_canary(graph, t, group_id=name, driver=driver)).serving
            for t in targets
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

        driver = injected_driver(backend)
        readings = [
            await bm25_canary(graph, t, group_id=name, driver=driver) for t in targets
        ]
        assert all(r.serving for r in readings), readings


class TestProductionExpectedSet:
    """The production-set check reports exactly what a graph lacks."""

    @pytest.mark.asyncio
    async def test_production_startup_sweep_leaves_nothing_missing(
        self, scratch, live_backend_factory,
    ):
        name, graph = scratch('sweep_complete')
        # The sweep skips graphs whose KEY does not exist yet.
        await graph.query('CREATE (:Probe {seed: 1})')
        backend = live_backend_factory({name})

        await backend.provision_registered_graphs()
        await await_index_operational(graph)

        assert await missing_production_indices(backend, name) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('spec', sorted(expected_index_set()), ids=lambda spec: '.'.join(spec))
    async def test_reports_exactly_the_one_missing_spec(
        self, scratch, live_backend_factory, spec,
    ):
        """The negative control: the check FAILS on a graph missing ANY expected index.

        A fresh graph per spec, because a partially dropped fulltext index
        cannot be re-created, so one graph cannot be repaired between cases.
        """
        name, graph = scratch('drop_one')
        await graph.query('CREATE (:Probe {seed: 1})')
        backend = live_backend_factory({name})
        await backend.provision_registered_graphs()
        await await_index_operational(graph)

        await graph.query(_drop_statement(spec))
        # A per-field drop on a merged index can open a rebuild window.
        await await_index_operational(graph)

        assert await missing_production_indices(backend, name) == [spec]
