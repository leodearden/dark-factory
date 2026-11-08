"""Tests for write-time identity primitives on GraphitiBackend (task 2198, W6-α).

Covers:
- GraphitiBackend._identity_lock_for (S2): per-group_id lazy asyncio.Lock registry,
  separate from DurableWriteQueue._group_locks.
- GraphitiBackend._resolve_or_create_entity (S1): exact-name resolve-or-collapse
  chokepoint — 0 matches is a no-op (None), 1 match resolves, >=2 collapses via
  find_duplicate_entity_nodes + merge_entities. group_id-scoped per the
  2026-07-06 amendment guarding task-2115's active cross-graph leak.
- GraphitiBackend.ensure_entity_node (task 3335): the resolve-or-MINT sibling of
  _resolve_or_create_entity — delegates resolve/collapse to it and adds only the
  mint on its documented 0-match None branch.
- GraphitiBackend._mint_entity_node (task 5003): the unconditional mint
  primitive — CREATE plus best-effort embedding, with no read, resolve or
  collapse; its 0-match precondition is the caller's to establish.
- GraphitiBackend.enumerate_ranked_entity_nodes (task 5662): the census-only
  group-scoped paged read, every row survivor-first by the shared rank rule.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import itertools
import random
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    assert_ro_query_only,
    extract_cypher,
    extract_params,
    falkor_skipif,
    unique_graph_name,
)
from falkordb.asyncio import FalkorDB

from fused_memory.backends.graphiti_client import (
    _PROVENANCE_RANK_CLAUSE,
    _PROVENANCE_RANK_ORDER,
    AmbiguousEntityError,
    GraphitiBackend,
    SurvivorRankTypeError,
    _MultiTenantFalkorDriver,
)
from fused_memory.utils.task_naming import task_node_referent
from fused_memory.utils.validation import PathShapedProjectIdError

# ---------------------------------------------------------------------------
# step-1/2: GraphitiBackend._identity_lock_for
# ---------------------------------------------------------------------------

class TestIdentityLockFor:
    """GraphitiBackend._identity_lock_for(group_id) returns a per-group_id asyncio.Lock.

    Synchronous accessor (returns the Lock, not a coroutine); lazily creates and
    caches one Lock per group_id, mirroring DurableWriteQueue._group_locks in
    fused_memory/services/durable_queue.py. Needs no initialized driver.
    """

    def test_returns_asyncio_lock(self, mock_config):
        """Returns an asyncio.Lock instance."""
        backend = GraphitiBackend(mock_config)
        lock = backend._identity_lock_for('g1')
        assert isinstance(lock, asyncio.Lock)

    def test_same_group_id_returns_same_lock(self, mock_config):
        """Two calls with the SAME group_id return the exact same object."""
        backend = GraphitiBackend(mock_config)
        a = backend._identity_lock_for('g1')
        b = backend._identity_lock_for('g1')
        assert a is b

    def test_different_group_ids_return_distinct_locks(self, mock_config):
        """Two calls with DIFFERENT group_ids return distinct objects."""
        backend = GraphitiBackend(mock_config)
        a = backend._identity_lock_for('g1')
        b = backend._identity_lock_for('g2')
        assert a is not b


# ---------------------------------------------------------------------------
# step-3/4: GraphitiBackend._resolve_or_create_entity — 0/1-match resolve path
# ---------------------------------------------------------------------------

class TestResolveOrCreateEntityResolve:
    """GraphitiBackend._resolve_or_create_entity(name, *, group_id) — 0/1-match
    resolve/no-op fast path (no collapse machinery), mirroring the
    TestMergeEntities.backend_with_mocks orchestration-mock pattern in
    test_merge_entities.py."""

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend):
        """GraphitiBackend with get_nodes_by_exact_name/find_duplicate_entity_nodes/
        merge_entities mocked as AsyncMocks for orchestration-only testing."""
        backend = make_backend(mock_config)
        backend.get_nodes_by_exact_name = AsyncMock(return_value=[])
        backend.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        backend.merge_entities = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_single_match_resolves_without_collapse(self, backend_with_mocks):
        """Exactly one match: returns its uuid; neither collapse method runs."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': 'u-1', 'name': 'Foo', 'summary': '', 'labels': []}
        ]
        result = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert result == 'u-1'
        backend.find_duplicate_entity_nodes.assert_not_awaited()
        backend.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_zero_matches_returns_none_without_minting(self, backend_with_mocks):
        """Zero matches: returns None — documented no-op; minting stays
        graphiti_core's job, this primitive only resolves/collapses."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = []
        result = await backend._resolve_or_create_entity('Ghost', group_id='test')
        assert result is None
        backend.find_duplicate_entity_nodes.assert_not_awaited()
        backend.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_get_nodes_by_exact_name_with_name_and_group_id(self, backend_with_mocks):
        """get_nodes_by_exact_name is awaited with the name and group_id."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = []
        await backend._resolve_or_create_entity('Foo', group_id='proj-x')
        backend.get_nodes_by_exact_name.assert_awaited_once_with('Foo', group_id='proj-x')


# ---------------------------------------------------------------------------
# step-5/6: GraphitiBackend._resolve_or_create_entity — >=2-match collapse path
# ---------------------------------------------------------------------------

class TestResolveOrCreateEntityCollapse:
    """GraphitiBackend._resolve_or_create_entity(name, *, group_id) — >=2-match
    collapse path: folds every non-survivor duplicate into the survivor-first
    match from find_duplicate_entity_nodes via merge_entities, mirroring the
    TestMergeEntities.backend_with_mocks orchestration-mock pattern in
    test_merge_entities.py."""

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend):
        """GraphitiBackend with get_nodes_by_exact_name/find_duplicate_entity_nodes/
        merge_entities mocked as AsyncMocks for orchestration-only testing.

        get_nodes_by_exact_name defaults to 3 same-name matches (>=2 branch).
        find_duplicate_entity_nodes returns them survivor-first (provenance_rank
        DESC, created_at ASC, uuid ASC per its own contract, where
        provenance_rank is edge_count + mentions_count -- task 4986): 'surv' has
        the richest provenance, 'dup1'/'dup2' are the duplicates to fold in.
        """
        backend = make_backend(mock_config)
        backend.get_nodes_by_exact_name = AsyncMock(return_value=[
            {'uuid': 'surv', 'name': 'Foo', 'summary': '', 'labels': []},
            {'uuid': 'dup1', 'name': 'Foo', 'summary': '', 'labels': []},
            {'uuid': 'dup2', 'name': 'Foo', 'summary': '', 'labels': []},
        ])
        backend.find_duplicate_entity_nodes = AsyncMock(return_value=[
            {'uuid': 'surv', 'created_at': 1, 'edge_count': 5,
             'mentions_count': 2, 'provenance_rank': 7},
            {'uuid': 'dup1', 'created_at': 2, 'edge_count': 1,
             'mentions_count': 1, 'provenance_rank': 2},
            {'uuid': 'dup2', 'created_at': 3, 'edge_count': 0,
             'mentions_count': 0, 'provenance_rank': 0},
        ])
        backend.merge_entities = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_merges_each_non_survivor_into_survivor(self, backend_with_mocks):
        """merge_entities is awaited once per non-survivor duplicate, with
        deprecated_uuid=that dup's uuid, surviving_uuid='surv' (the
        find_duplicate_entity_nodes[0] survivor), group_id=the caller's
        group_id. Foreign/other uuids are never passed."""
        backend = backend_with_mocks
        await backend._resolve_or_create_entity('Foo', group_id='test')
        assert backend.merge_entities.await_count == 2  # len(dups) - 1
        merged_deprecated_uuids = set()
        for call in backend.merge_entities.await_args_list:
            args, kwargs = call
            deprecated_uuid, surviving_uuid = args[0], args[1]
            merged_deprecated_uuids.add(deprecated_uuid)
            assert surviving_uuid == 'surv'
            assert kwargs.get('group_id') == 'test'
        assert merged_deprecated_uuids == {'dup1', 'dup2'}

    @pytest.mark.asyncio
    async def test_returns_survivor_uuid(self, backend_with_mocks):
        """Returns the survivor's uuid (find_duplicate_entity_nodes[0])."""
        backend = backend_with_mocks
        result = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert result == 'surv'

    @pytest.mark.asyncio
    async def test_idempotent_second_call_does_not_remerge(self, backend_with_mocks):
        """Idempotency: once collapsed to a single survivor, a subsequent call
        (reflecting post-merge reality — get_nodes_by_exact_name now returns
        only the survivor) resolves via the 1-match fast path and does NOT
        invoke merge_entities again."""
        backend = backend_with_mocks
        first = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert first == 'surv'
        assert backend.merge_entities.await_count == 2

        # Simulate the post-collapse graph state: only the survivor remains.
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': 'surv', 'name': 'Foo', 'summary': '', 'labels': []},
        ]
        second = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert second == 'surv'
        assert backend.merge_entities.await_count == 2  # unchanged — no re-merge

    @pytest.mark.asyncio
    async def test_find_duplicate_entity_nodes_returns_empty_degrades_to_none(
        self, backend_with_mocks
    ):
        """Defensive/coupling guard: get_nodes_by_exact_name and
        find_duplicate_entity_nodes are separate queries expected to filter
        identically. If they ever diverge — get_nodes_by_exact_name finds
        >=2 matches but find_duplicate_entity_nodes returns no rows — degrade
        to None instead of raising IndexError on dups[0]; merge_entities
        must never be awaited."""
        backend = backend_with_mocks
        backend.find_duplicate_entity_nodes.return_value = []
        result = await backend._resolve_or_create_entity('Foo', group_id='test')
        assert result is None
        backend.merge_entities.assert_not_awaited()


# ---------------------------------------------------------------------------
# step-7/8: group_id-scoping amendment (guards task-2115 cross-graph leak)
# ---------------------------------------------------------------------------

class TestGroupIdScopingAmendment:
    """2026-07-06 amendment: get_nodes_by_exact_name and find_duplicate_entity_nodes
    must filter by the n.group_id PROPERTY, not just the graph key selected via
    _graph_for. Task-2115's active cross-graph leak (~1 episode/day) plants
    misrouted foreign nodes (group_id property of ANOTHER project) physically
    inside a graph key; without this predicate, _resolve_or_create_entity's
    resolve-or-collapse would destructively auto-merge a foreign clone into the
    home node (permanent cross-project edge contamination)."""

    @pytest.mark.asyncio
    async def test_get_nodes_by_exact_name_filters_by_group_id(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """STRUCTURAL: Cypher gains a group_id predicate + param; the existing
        exact-name MATCH substring is preserved untouched."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_nodes_by_exact_name('Foo', group_id='home')
        cypher = extract_cypher(graph.ro_query.call_args)
        params = extract_params(graph.ro_query.call_args)
        assert 'n.group_id = $group_id' in cypher
        assert 'MATCH (n:Entity {name: $name})' in cypher  # pre-existing substring, unchanged
        assert params.get('group_id') == 'home'

    @pytest.mark.asyncio
    async def test_find_duplicate_entity_nodes_filters_by_group_id(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """STRUCTURAL: Cypher gains a group_id predicate + param; the existing
        exact-name MATCH and invalid_at substrings are preserved untouched."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.find_duplicate_entity_nodes('Foo', group_id='home')
        cypher = extract_cypher(graph.ro_query.call_args)
        params = extract_params(graph.ro_query.call_args)
        assert 'n.group_id = $group_id' in cypher
        assert '(n:Entity {name: $name})' in cypher  # pre-existing substring, unchanged
        assert 'invalid_at IS NULL' in cypher  # pre-existing substring, unchanged
        assert params.get('group_id') == 'home'

    @pytest.mark.asyncio
    async def test_foreign_clone_never_folded_into_home_survivor(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """BEHAVIORAL capstone (amendment's REQUIRED test): a foreign clone
        (group_id='foreign') physically co-located in the home graph key must
        never be returned by get_nodes_by_exact_name nor passed to
        merge_entities; only the two home nodes collapse.

        Drives the REAL get_nodes_by_exact_name + find_duplicate_entity_nodes
        (only merge_entities is mocked) against a group-aware fake ro_query, so
        the group_id predicate is exercised end-to-end rather than just
        structurally. The foreign clone is seeded with the highest edge_count
        and oldest created_at so that, pre-fix, it wins survivor-first
        ordering — the destructive scenario the amendment guards against.

        RED on current code: both queries omit the group_id param, so the fake
        returns the foreign clone too -> it is folded/returned.
        """
        seeded = [
            {'uuid': 'home-1', 'name': 'Foo', 'group_id': 'home',
             'summary': '', 'labels': ['Entity'], 'created_at': 1,
             'edge_count': 5, 'mentions_count': 1},
            {'uuid': 'home-2', 'name': 'Foo', 'group_id': 'home',
             'summary': '', 'labels': ['Entity'], 'created_at': 2,
             'edge_count': 1, 'mentions_count': 0},
            # The foreign clone keeps the highest rank AND the oldest
            # created_at, so it still wins survivor-first ordering pre-fix —
            # the destructive scenario this test exists to guard.
            {'uuid': 'foreign-1', 'name': 'Foo', 'group_id': 'foreign',
             'summary': '', 'labels': ['Entity'], 'created_at': 0,
             'edge_count': 99, 'mentions_count': 3},
        ]

        async def fake_ro_query(cypher, params):
            name = params.get('name')
            matched = [
                row for row in seeded
                if row['name'] == name
                and ('group_id' not in params or row['group_id'] == params['group_id'])
            ]
            result = MagicMock()
            if 'edge_count' in cypher:
                # find_duplicate_entity_nodes shape — emulate the DB-side
                # ORDER BY provenance_rank DESC, created_at ASC, uuid ASC,
                # where provenance_rank is edge_count + mentions_count.
                def _rank(r):
                    return r['edge_count'] + r['mentions_count']
                ordered = sorted(
                    matched, key=lambda r: (-_rank(r), r['created_at'], r['uuid'])
                )
                result.result_set = [
                    [r['uuid'], r['created_at'], r['edge_count'],
                     r['mentions_count'], _rank(r)]
                    for r in ordered
                ]
            elif 'labels(n)' in cypher:
                # get_nodes_by_exact_name shape
                result.result_set = [
                    [r['uuid'], r['name'], r['summary'], r['labels']] for r in matched
                ]
            else:
                result.result_set = []
            return result

        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        graph.ro_query = AsyncMock(side_effect=fake_ro_query)
        backend._driver._get_graph = MagicMock(return_value=graph)
        backend.merge_entities = AsyncMock()

        result = await backend._resolve_or_create_entity('Foo', group_id='home')

        assert result in ('home-1', 'home-2')
        for call in backend.merge_entities.await_args_list:
            args, kwargs = call
            assert 'foreign-1' not in args
            assert kwargs.get('group_id') != 'foreign'
        backend.merge_entities.assert_awaited_once()
        dep_uuid, sur_uuid = backend.merge_entities.await_args_list[0][0][:2]
        assert {dep_uuid, sur_uuid} == {'home-1', 'home-2'}
        assert sur_uuid == result


# ---------------------------------------------------------------------------
# task 3335 step-3/4: GraphitiBackend.ensure_entity_node — resolve-or-MINT
# ---------------------------------------------------------------------------

def _create_calls(graph: MagicMock) -> list:
    """Return the graph.query calls whose Cypher mints an Entity node.

    The mint path also issues a separate SET n.name_embedding write (via
    update_node_embedding) against the same graph mock, so tests that assert on
    the mint itself must select it rather than assume a single call.
    """
    return [c for c in graph.query.call_args_list if 'CREATE' in extract_cypher(c)]


class TestEnsureEntityNode:
    """GraphitiBackend.ensure_entity_node(name, *, group_id, summary='') — the
    resolve-or-MINT sibling of _resolve_or_create_entity (task 3335).

    _resolve_or_create_entity documents 0 matches as a no-op ("node minting
    stays graphiti_core's job"), and several callers depend on that. Splitting a
    collapsed cross-project reference needs a node that graphiti-core will never
    mint (extraction is exactly what discarded the qualifier), so this primitive
    delegates the resolve/collapse half to _resolve_or_create_entity and adds
    ONLY the mint on its None branch — leaving that contract intact.

    Inherits _resolve_or_create_entity's lock contract: callers MUST hold
    _identity_lock_for(group_id).
    """

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend, make_graph_mock):
        """Backend whose resolve inputs are mocked and whose graph is captured.

        get_nodes_by_exact_name defaults to [] (the 0-match mint branch); the
        REAL _resolve_or_create_entity runs, so resolve/collapse behaviour is
        exercised rather than stubbed.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        backend.get_nodes_by_exact_name = AsyncMock(return_value=[])
        backend.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        backend.merge_entities = AsyncMock()
        backend.update_node_embedding = AsyncMock()
        backend.client.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
        backend._test_graph = graph
        return backend

    @pytest.mark.asyncio
    async def test_single_match_resolves_without_minting(self, backend_with_mocks):
        """(a) Exactly one existing node: return its uuid, write nothing."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': 'u-1', 'name': 'dark_factory:2500', 'summary': '', 'labels': []}
        ]
        result = await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        assert result == 'u-1'
        backend._test_graph.query.assert_not_awaited()
        backend.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_matches_collapse_and_return_survivor(self, backend_with_mocks):
        """(b) >=2 existing nodes: collapse via find_duplicate_entity_nodes +
        merge_entities, return the survivor, mint nothing."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': 'surv', 'name': 'dark_factory:2500', 'summary': '', 'labels': []},
            {'uuid': 'dup1', 'name': 'dark_factory:2500', 'summary': '', 'labels': []},
        ]
        backend.find_duplicate_entity_nodes.return_value = [
            {'uuid': 'surv', 'created_at': 1, 'edge_count': 5,
             'mentions_count': 2, 'provenance_rank': 7},
            {'uuid': 'dup1', 'created_at': 2, 'edge_count': 0,
             'mentions_count': 0, 'provenance_rank': 0},
        ]
        result = await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        assert result == 'surv'
        backend.merge_entities.assert_awaited_once_with('dup1', 'surv', group_id='reify')
        assert _create_calls(backend._test_graph) == []

    @pytest.mark.asyncio
    async def test_zero_matches_mints_an_entity_node(self, backend_with_mocks):
        """(c) Zero existing nodes: mint one :Entity node and return its uuid."""
        backend = backend_with_mocks
        result = await backend.ensure_entity_node(
            'dark_factory:2500', group_id='reify', summary='cross-project ref'
        )
        creates = _create_calls(backend._test_graph)
        assert len(creates) == 1
        cypher = extract_cypher(creates[0])
        params = extract_params(creates[0])
        assert 'CREATE' in cypher
        assert ':Entity' in cypher
        assert params['name'] == 'dark_factory:2500'
        assert params['group_id'] == 'reify'
        assert params['summary'] == 'cross-project ref'
        assert params['created_at']
        # A fresh uuid4 string, returned to the caller.
        assert uuid.UUID(params['uuid'])
        assert result == params['uuid']

    @pytest.mark.asyncio
    async def test_summary_defaults_to_empty_string(self, backend_with_mocks):
        backend = backend_with_mocks
        await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        assert extract_params(_create_calls(backend._test_graph)[0])['summary'] == ''

    @pytest.mark.asyncio
    async def test_created_at_matches_graphiti_cores_wire_format(self, backend_with_mocks):
        """The minted created_at must be byte-compatible with what graphiti-core
        itself writes, or survivor selection would be poisoned when this node
        later needs collapsing.

        Confirmed round-trip (task 3335 step-4): graphiti-core hands a datetime
        to $entity_data, FalkorDriver.convert_datetimes_to_strings turns it into
        `datetime.isoformat()` (FalkorDB has no datetime type), helpers.
        parse_db_date reads it back with `datetime.fromisoformat`, and
        find_duplicate_entity_nodes orders DB-SIDE on the stored string
        (`ORDER BY ... n.created_at ASC`) — a lexicographic compare that is
        chronologically correct only for same-format, same-offset ISO strings.
        So the mint must emit exactly `datetime.now(UTC).isoformat()`.
        """
        backend = backend_with_mocks
        await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        created_at = extract_params(_create_calls(backend._test_graph)[0])['created_at']
        assert isinstance(created_at, str)
        parsed = datetime.fromisoformat(created_at)  # what parse_db_date does
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        # Lexicographic ordering agrees with chronological ordering against a
        # value written in graphiti-core's own format.
        older = (parsed - timedelta(seconds=1)).isoformat()
        newer = (parsed + timedelta(seconds=1)).isoformat()
        assert sorted([newer, created_at, older]) == [older, created_at, newer]

    @pytest.mark.asyncio
    async def test_mint_regenerates_name_embedding(self, backend_with_mocks):
        """(d) After a successful mint the name_embedding is regenerated with
        the NEW uuid, mirroring rename_entity_node."""
        backend = backend_with_mocks
        result = await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        backend.client.embedder.create.assert_awaited_once_with('dark_factory:2500')
        backend.update_node_embedding.assert_awaited_once_with(
            result, [0.1, 0.2, 0.3], group_id='reify'
        )

    @pytest.mark.asyncio
    async def test_embedder_failure_is_swallowed_and_mint_still_returns(
        self, backend_with_mocks
    ):
        """(d) Best-effort: an embedder failure is logged, not raised — the
        node itself (the primary result) already exists."""
        backend = backend_with_mocks
        backend.client.embedder.create = AsyncMock(side_effect=RuntimeError('embedder down'))
        result = await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        assert result == extract_params(_create_calls(backend._test_graph)[0])['uuid']
        backend.update_node_embedding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embedding_write_failure_is_swallowed(self, backend_with_mocks):
        backend = backend_with_mocks
        backend.update_node_embedding = AsyncMock(side_effect=RuntimeError('write failed'))
        result = await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        assert result == extract_params(_create_calls(backend._test_graph)[0])['uuid']

    @pytest.mark.asyncio
    async def test_idempotent_second_call_mints_nothing(self, backend_with_mocks):
        """(e) Idempotence: once the node exists, a second call for the same
        (name, group_id) takes the resolve path."""
        backend = backend_with_mocks
        first = await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': first, 'name': 'dark_factory:2500', 'summary': '', 'labels': []}
        ]
        second = await backend.ensure_entity_node('dark_factory:2500', group_id='reify')
        assert second == first
        assert len(_create_calls(backend._test_graph)) == 1

    @pytest.mark.asyncio
    async def test_group_id_is_canonicalized(self, backend_with_mocks):
        """(f) Decorated with @_canonicalize_group_args: a hyphenated/uppercase
        group_id resolves AND mints under the canonical key, so the graph key,
        the node's group_id property and any later $group_id filter agree."""
        backend = backend_with_mocks
        await backend.ensure_entity_node('dark_factory:2500', group_id='Dark-Factory')
        backend.get_nodes_by_exact_name.assert_awaited_once_with(
            'dark_factory:2500', group_id='dark_factory'
        )
        assert extract_params(_create_calls(backend._test_graph)[0])['group_id'] == 'dark_factory'
        backend._driver._get_graph.assert_called_with('dark_factory')


# ---------------------------------------------------------------------------
# task 4985 step-1/2: ensure_entity_node(..., merge_duplicates=False) — guard (c)
# ---------------------------------------------------------------------------

class TestEnsureEntityNodeNoMerge:
    """ensure_entity_node(..., merge_duplicates=False) — the no-merge mode.

    Mirrors task 4932 guard 2's semantics one layer down: 0 matches mint,
    1 match resolves, >=2 matches REFUSE structurally and merge NOTHING.

    The >=2 arm exists because a destructive collapse is only ever a
    deliberate act, never a side effect of a repair. The default
    (merge_duplicates=True) keeps Seam S1's episode-write dedup path
    byte-identical, which test_the_default_still_collapses pins.
    """

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend, make_graph_mock):
        """Same shape as TestEnsureEntityNode's fixture: the REAL resolve half
        runs against mocked reads, and the graph mock captures every write."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        backend.get_nodes_by_exact_name = AsyncMock(return_value=[])
        backend.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        backend.merge_entities = AsyncMock()
        backend.update_node_embedding = AsyncMock()
        backend.client.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
        backend._test_graph = graph
        return backend

    @staticmethod
    def _two_matches(backend) -> None:
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': 'u-a', 'name': 'Task 3127', 'summary': '', 'labels': []},
            {'uuid': 'u-b', 'name': 'Task 3127', 'summary': '', 'labels': []},
        ]
        backend.find_duplicate_entity_nodes.return_value = [
            {'uuid': 'u-a', 'created_at': 1, 'edge_count': 5,
             'mentions_count': 2, 'provenance_rank': 7},
            {'uuid': 'u-b', 'created_at': 2, 'edge_count': 0,
             'mentions_count': 0, 'provenance_rank': 0},
        ]

    @pytest.mark.asyncio
    async def test_two_matches_raise_ambiguous_entity_error(self, backend_with_mocks):
        """(1) >=2 matches refuse with AmbiguousEntityError instead of collapsing."""
        backend = backend_with_mocks
        self._two_matches(backend)
        with pytest.raises(AmbiguousEntityError):
            await backend.ensure_entity_node(
                'Task 3127', group_id='dark_factory', merge_duplicates=False
            )

    @pytest.mark.asyncio
    async def test_refusal_carries_structured_name_group_and_uuids(self, backend_with_mocks):
        """(1) The refusal is DATA, not a message to parse: .name/.group_id/.uuids."""
        backend = backend_with_mocks
        self._two_matches(backend)
        with pytest.raises(AmbiguousEntityError) as excinfo:
            await backend.ensure_entity_node(
                'Task 3127', group_id='dark_factory', merge_duplicates=False
            )
        exc = excinfo.value
        assert exc.name == 'Task 3127'
        assert exc.group_id == 'dark_factory'
        assert exc.uuids == ('u-a', 'u-b')
        assert isinstance(exc.uuids, tuple)

    @pytest.mark.asyncio
    async def test_refusal_message_still_lists_the_conflicting_uuids(self, backend_with_mocks):
        """(1) An operator reading a log line loses nothing to the new fields."""
        backend = backend_with_mocks
        self._two_matches(backend)
        with pytest.raises(AmbiguousEntityError) as excinfo:
            await backend.ensure_entity_node(
                'Task 3127', group_id='dark_factory', merge_duplicates=False
            )
        message = str(excinfo.value)
        assert 'Task 3127' in message
        assert 'u-a' in message
        assert 'u-b' in message

    @pytest.mark.asyncio
    async def test_refusal_merges_nothing_and_mints_nothing(self, backend_with_mocks):
        """(1) The whole point: no merge_entities, no CREATE."""
        backend = backend_with_mocks
        self._two_matches(backend)
        with pytest.raises(AmbiguousEntityError):
            await backend.ensure_entity_node(
                'Task 3127', group_id='dark_factory', merge_duplicates=False
            )
        backend.merge_entities.assert_not_awaited()
        assert _create_calls(backend._test_graph) == []

    @pytest.mark.asyncio
    async def test_single_match_resolves_without_consulting_duplicates(self, backend_with_mocks):
        """(2) Exactly one match returns that uuid; no merge, no CREATE, and
        find_duplicate_entity_nodes is never even asked."""
        backend = backend_with_mocks
        backend.get_nodes_by_exact_name.return_value = [
            {'uuid': 'u-1', 'name': 'Task 3127', 'summary': '', 'labels': []}
        ]
        result = await backend.ensure_entity_node(
            'Task 3127', group_id='dark_factory', merge_duplicates=False
        )
        assert result == 'u-1'
        backend.find_duplicate_entity_nodes.assert_not_awaited()
        backend.merge_entities.assert_not_awaited()
        assert _create_calls(backend._test_graph) == []

    @pytest.mark.asyncio
    async def test_zero_matches_mint_exactly_as_the_default_mode_does(self, backend_with_mocks):
        """(3) The mint arm is unforked: same CREATE params, same summary, same
        group_id as test_zero_matches_mints_an_entity_node pins for the default."""
        backend = backend_with_mocks
        result = await backend.ensure_entity_node(
            'dark_factory:2500',
            group_id='reify',
            summary='cross-project ref',
            merge_duplicates=False,
        )
        creates = _create_calls(backend._test_graph)
        assert len(creates) == 1
        cypher = extract_cypher(creates[0])
        params = extract_params(creates[0])
        assert 'CREATE' in cypher
        assert ':Entity' in cypher
        assert params['name'] == 'dark_factory:2500'
        assert params['group_id'] == 'reify'
        assert params['summary'] == 'cross-project ref'
        assert params['created_at']
        assert uuid.UUID(params['uuid'])
        assert result == params['uuid']

    @pytest.mark.asyncio
    async def test_the_default_still_collapses(self, backend_with_mocks):
        """(4) REGRESSION PIN, not a RED: omitting merge_duplicates on a
        2+-match fixture still collapses and returns the survivor, so Seam S1's
        episode-write dedup keeps its ratified collapse."""
        backend = backend_with_mocks
        self._two_matches(backend)
        result = await backend.ensure_entity_node('Task 3127', group_id='dark_factory')
        assert result == 'u-a'
        backend.merge_entities.assert_awaited_once_with('u-b', 'u-a', group_id='dark_factory')
        assert _create_calls(backend._test_graph) == []


# ---------------------------------------------------------------------------
# task 5003: GraphitiBackend._mint_entity_node — the unconditional mint primitive
# ---------------------------------------------------------------------------

_EXACT_NAME_READ = 'MATCH (n:Entity {name: $name})'


class TestMintEntityNode:
    """GraphitiBackend._mint_entity_node(name, *, group_id, summary='').

    CREATEs one :Entity node and best-effort embeds its name — nothing else.
    It performs no exact-name read, no resolve and no collapse: the caller
    holds _identity_lock_for(group_id) and has already seen zero exact-name
    matches under that hold, so a second read would only repeat the answer.
    """

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend, make_graph_mock):
        """Same shape as TestEnsureEntityNode's fixture. The read, collapse and
        merge seams are AsyncMocks so assert_not_awaited is meaningful."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        backend.get_nodes_by_exact_name = AsyncMock(return_value=[])
        backend.find_duplicate_entity_nodes = AsyncMock(return_value=[])
        backend.merge_entities = AsyncMock()
        backend.update_node_embedding = AsyncMock()
        backend.client.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
        backend._test_graph = graph
        return backend

    @pytest.mark.asyncio
    async def test_mints_one_entity_node_and_returns_its_uuid(self, backend_with_mocks):
        backend = backend_with_mocks
        result = await backend._mint_entity_node(
            'dark_factory:2500', group_id='reify', summary='cross-project ref'
        )
        creates = _create_calls(backend._test_graph)
        assert len(creates) == 1
        params = extract_params(creates[0])
        assert ':Entity' in extract_cypher(creates[0])
        assert params['name'] == 'dark_factory:2500'
        assert params['group_id'] == 'reify'
        assert params['summary'] == 'cross-project ref'
        assert params['created_at']
        assert uuid.UUID(params['uuid'])
        assert result == params['uuid']

    @pytest.mark.asyncio
    async def test_issues_no_exact_name_read_and_never_resolves_or_merges(
        self, backend_with_mocks
    ):
        """The whole point of the primitive: the caller already read the name."""
        backend = backend_with_mocks
        await backend._mint_entity_node('dark_factory:2500', group_id='reify')
        backend.get_nodes_by_exact_name.assert_not_awaited()
        backend.find_duplicate_entity_nodes.assert_not_awaited()
        backend.merge_entities.assert_not_awaited()
        exact_name_reads = [
            c for c in backend._test_graph.ro_query.call_args_list
            if _EXACT_NAME_READ in extract_cypher(c)
        ]
        assert exact_name_reads == []

    @pytest.mark.asyncio
    async def test_mints_even_when_a_same_name_node_exists(
        self, backend_with_mocks, make_graph_mock
    ):
        """The 0-match precondition is the CALLER's to establish; the primitive
        must not quietly re-introduce a read to check it. Both read routes, the
        method and raw Cypher on the graph, would see the existing node."""
        backend = backend_with_mocks
        existing = {'uuid': 'u-1', 'name': 'dark_factory:2500', 'summary': '', 'labels': []}
        backend.get_nodes_by_exact_name.return_value = [existing]
        graph = make_graph_mock(ro_rows=[list(existing.values())], q_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend._mint_entity_node('dark_factory:2500', group_id='reify')

        assert len(_create_calls(graph)) == 1

    @pytest.mark.asyncio
    async def test_summary_defaults_to_empty_string(self, backend_with_mocks):
        backend = backend_with_mocks
        await backend._mint_entity_node('dark_factory:2500', group_id='reify')
        assert extract_params(_create_calls(backend._test_graph)[0])['summary'] == ''

    @pytest.mark.asyncio
    async def test_created_at_is_a_utc_isoformat_string(self, backend_with_mocks):
        backend = backend_with_mocks
        await backend._mint_entity_node('dark_factory:2500', group_id='reify')
        created_at = extract_params(_create_calls(backend._test_graph)[0])['created_at']
        assert isinstance(created_at, str)
        assert datetime.fromisoformat(created_at).utcoffset() == timedelta(0)

    @pytest.mark.asyncio
    async def test_regenerates_name_embedding_for_the_new_uuid(self, backend_with_mocks):
        backend = backend_with_mocks
        result = await backend._mint_entity_node('dark_factory:2500', group_id='reify')
        backend.client.embedder.create.assert_awaited_once_with('dark_factory:2500')
        backend.update_node_embedding.assert_awaited_once_with(
            result, [0.1, 0.2, 0.3], group_id='reify'
        )

    @pytest.mark.asyncio
    async def test_embedder_failure_is_swallowed(self, backend_with_mocks):
        backend = backend_with_mocks
        backend.client.embedder.create = AsyncMock(side_effect=RuntimeError('embedder down'))
        result = await backend._mint_entity_node('dark_factory:2500', group_id='reify')
        assert result == extract_params(_create_calls(backend._test_graph)[0])['uuid']
        backend.update_node_embedding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embedding_write_failure_is_swallowed(self, backend_with_mocks):
        backend = backend_with_mocks
        backend.update_node_embedding = AsyncMock(side_effect=RuntimeError('write failed'))
        result = await backend._mint_entity_node('dark_factory:2500', group_id='reify')
        assert result == extract_params(_create_calls(backend._test_graph)[0])['uuid']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'exc_type', [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
    )
    async def test_cancellation_during_embedding_propagates(
        self, backend_with_mocks, exc_type
    ):
        backend = backend_with_mocks
        backend.client.embedder.create = AsyncMock(side_effect=exc_type())
        with pytest.raises(exc_type):
            await backend._mint_entity_node('dark_factory:2500', group_id='reify')

    @pytest.mark.asyncio
    async def test_group_id_is_canonicalized(self, backend_with_mocks):
        backend = backend_with_mocks
        await backend._mint_entity_node('dark_factory:2500', group_id='Dark-Factory')
        assert extract_params(_create_calls(backend._test_graph)[0])['group_id'] == 'dark_factory'
        backend._driver._get_graph.assert_called_with('dark_factory')

    @pytest.mark.asyncio
    async def test_path_shaped_group_id_is_rejected_before_any_db_call(
        self, backend_with_mocks
    ):
        backend = backend_with_mocks
        with pytest.raises(PathShapedProjectIdError):
            await backend._mint_entity_node('dark_factory:2500', group_id='-home-leo-src-x')
        backend._test_graph.query.assert_not_awaited()
        backend.client.embedder.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# task 5264: GraphitiBackend.find_entity_nodes_by_name_substring
# ---------------------------------------------------------------------------

class TestFindEntityNodesByNameSubstring:
    """The substring-match sibling of find_duplicate_entity_nodes.

    A task-agnostic candidate-NARROWING primitive. The family-keyed normalizer
    probes it with a task's verbatim digits to reach every spelling of one task
    in a single query — 'Task 605', 'task 605', 'tasks 605', 'task #605' — and
    then decides membership in Python via canonicalize_task_node_name. The
    method itself knows nothing about task labels, which is precisely what
    keeps the label vocabulary out of a Cypher string where it could neither be
    tested nor kept in step with utils/canonical_labels.py.
    """

    @pytest.mark.asyncio
    async def test_matches_a_bound_substring_never_an_exact_name(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """CONTAINS, not `{name: $name}` — and the substring is a BOUND param.

        Interpolating it into the query text would make any name carrying
        Cypher syntax an injection vector and defeat the planner's cache; the
        exact-equality shape it replaces is what made the old arrival-keyed
        normalizer able to see only two spellings of a family.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.find_entity_nodes_by_name_substring('605', group_id='home')

        cypher = extract_cypher(graph.ro_query.call_args)
        params = extract_params(graph.ro_query.call_args)
        assert 'n.name CONTAINS $substring' in cypher
        assert '{name: $name}' not in cypher
        assert '605' not in cypher  # bound, never interpolated
        assert params.get('substring') == '605'

    @pytest.mark.asyncio
    async def test_filters_by_the_group_id_property_and_binds_it_canonicalized(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Inherits find_duplicate_entity_nodes' 2026-07-06 scoping amendment.

        The graph KEY alone is not enough: task-2115's cross-graph leak can
        plant a node whose group_id property names ANOTHER project physically
        inside this graph key, and without the predicate the normalizer would
        happily merge that foreign node into the local family.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.find_entity_nodes_by_name_substring('605', group_id='know-live')

        cypher = extract_cypher(graph.ro_query.call_args)
        params = extract_params(graph.ro_query.call_args)
        assert 'n.group_id = $group_id' in cypher
        assert params.get('group_id') == 'know_live'

    @pytest.mark.asyncio
    async def test_returns_named_rows_ordered_survivor_first(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """rows[0] is the merge survivor — highest provenance_rank, then oldest,
        then uuid.

        Same contract as the exact-match sibling, from the same ORDER BY
        clause, so the normalizer's one survivor rule reads identically for
        both. ``name`` joins the returned columns because the caller has to
        canonicalize each candidate to decide family membership.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock([
            ['u-high', 'task 605', '2026-01-02', 13, 1, 14],
            ['u-canon', 'Task 605', '2026-01-01', 2, 0, 2],
            ['u-low', 'tasks 605', '2026-01-03', 1, 0, 1],
        ])
        backend._driver._get_graph = MagicMock(return_value=graph)

        rows = await backend.find_entity_nodes_by_name_substring('605', group_id='home')

        cypher = extract_cypher(graph.ro_query.call_args)
        assert _PROVENANCE_RANK_ORDER in cypher
        assert 'invalid_at IS NULL' in cypher  # only VALID edges are counted
        assert rows == [
            {'uuid': 'u-high', 'name': 'task 605', 'created_at': '2026-01-02',
             'edge_count': 13, 'mentions_count': 1, 'provenance_rank': 14},
            {'uuid': 'u-canon', 'name': 'Task 605', 'created_at': '2026-01-01',
             'edge_count': 2, 'mentions_count': 0, 'provenance_rank': 2},
            {'uuid': 'u-low', 'name': 'tasks 605', 'created_at': '2026-01-03',
             'edge_count': 1, 'mentions_count': 0, 'provenance_rank': 1},
        ]

    @pytest.mark.asyncio
    async def test_ranks_by_the_same_shared_clause_as_the_exact_match_sibling(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The two survivor-ranking methods cannot diverge, because they are
        built from the SAME objects — not from two clauses that happen to read
        alike.

        This method's own docstring promises it orders "exactly as
        find_duplicate_entity_nodes orders its matches". Task 5264 made that
        promise by COPYING the clause, which is how the MENTIONS-blind ranking
        reached a second, newer merge path. Asserting the shared constants
        appear verbatim in BOTH emitted queries is what turns the promise into
        something a change has to break loudly.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.find_entity_nodes_by_name_substring('605', group_id='home')
        substring_cypher = extract_cypher(graph.ro_query.call_args)
        await backend.find_duplicate_entity_nodes('Task 605', group_id='home')
        exact_cypher = extract_cypher(graph.ro_query.call_args)

        for cypher in (substring_cypher, exact_cypher):
            assert _PROVENANCE_RANK_CLAUSE in cypher
            assert _PROVENANCE_RANK_ORDER in cypher

    @pytest.mark.asyncio
    async def test_row_order_is_the_drivers_and_is_never_re_sorted_here(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Feeding rows the ORDER BY would never produce proves the method does
        not re-derive the ordering: a Python-side sort would be a second copy
        of the survivor rule, free to drift from the clause both siblings share.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock([
            ['u-low', 'tasks 605', '2026-01-03', 1, 0, 1],
            ['u-high', 'task 605', '2026-01-02', 13, 1, 14],
        ])
        backend._driver._get_graph = MagicMock(return_value=graph)

        rows = await backend.find_entity_nodes_by_name_substring('605', group_id='home')

        assert [row['uuid'] for row in rows] == ['u-low', 'u-high']

    @pytest.mark.asyncio
    async def test_no_match_yields_an_empty_list_rather_than_raising(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)

        assert await backend.find_entity_nodes_by_name_substring(
            '99999', group_id='home'
        ) == []

    @pytest.mark.asyncio
    async def test_is_read_only(self, mock_config, make_backend, make_graph_mock):
        """Structural, not asserted-about: the census workstream's whole promise
        is that nothing on its path can write, and this is its only query."""
        backend = make_backend(mock_config)

        await assert_ro_query_only(
            backend,
            make_graph_mock,
            [],
            'find_entity_nodes_by_name_substring',
            '605',
            group_id='home',
        )


# ---------------------------------------------------------------------------
# task 5662: GraphitiBackend.enumerate_ranked_entity_nodes
# ---------------------------------------------------------------------------

def _ranked_dict(row: list) -> dict:
    keys = ('uuid', 'name', 'created_at', 'edge_count', 'mentions_count', 'provenance_rank')
    return dict(zip(keys, row, strict=True))


def _null_largest(x, y) -> int:
    if x == y:
        return 0
    if x is None:
        return 1
    if y is None:
        return -1
    return -1 if x < y else 1


def _cypher_order_by(a: dict, b: dict) -> int:
    """`ORDER BY provenance_rank DESC, n.created_at ASC, n.uuid ASC` as openCypher
    reads it: one key at a time, NULL comparing larger than any value."""
    return (
        -_null_largest(a['provenance_rank'], b['provenance_rank'])
        or _null_largest(a['created_at'], b['created_at'])
        or _null_largest(a['uuid'], b['uuid'])
    )


def _page_and_census_cypher(graph: MagicMock) -> tuple[str, str]:
    cyphers = [extract_cypher(c) for c in graph.ro_query.call_args_list]
    (census,) = {c for c in cyphers if c.rstrip().endswith('RETURN count(*)')}
    (page,) = {c for c in cyphers if 'SKIP' in c}
    return page, census


class TestEnumerateRankedEntityNodes:
    """The task-family census's one read: every same-group Entity node, with the
    provenance-rank columns of the substring sibling, survivor-first.

    Membership and edge counts come from the SAME rows, so they cannot disagree
    the way a whole-graph enumeration and a per-family probe could.
    """

    @pytest.mark.asyncio
    async def test_is_group_scoped_in_both_the_census_and_the_pages_and_binds_the_canonical_id(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([['u1', 'Task 1', '2026-01-01', 1, 0, 1]])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.enumerate_ranked_entity_nodes(group_id='know-live')

        assert graph.ro_query.await_count >= 2
        for call_args in graph.ro_query.call_args_list:
            assert 'n.group_id = $group_id' in extract_cypher(call_args)
            assert extract_params(call_args).get('group_id') == 'know_live'
        backend._driver._get_graph.assert_called_with('know_live')

    @pytest.mark.asyncio
    async def test_cuts_each_page_by_uuid_before_aggregating_the_shared_rank_clause(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The paging invariant needs a total order over NON-aggregated rows: a
        rank computed per page can move between pages on a live store."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([['u1', 'Task 1', '2026-01-01', 1, 0, 1]])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.enumerate_ranked_entity_nodes(group_id='home')

        page, census = _page_and_census_cypher(graph)
        assert _PROVENANCE_RANK_CLAUSE in page
        assert page.index('WITH n ORDER BY n.uuid SKIP') < page.index(_PROVENANCE_RANK_CLAUSE)
        assert census.partition('RETURN count(*)')[0] == page.partition('WITH n')[0]

    @pytest.mark.asyncio
    async def test_returns_every_row_survivor_first_across_pages(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Highest provenance_rank, then oldest (a missing created_at is the
        largest, as in openCypher), then uuid — the order the destructive
        rankers' ORDER BY produces, applied to the whole multi-page read."""
        in_uuid_order = [
            ['u1', 'Task 1', '2026-03-01', 1, 0, 1],
            ['u2', 'Task 2', None, 5, 0, 5],
            ['u3', 'Task 3', '2026-02-01', 3, 2, 5],
            ['u4', 'Task 4', '2026-01-01', 4, 1, 5],
            ['u5', 'Task 5', '2026-03-01', 0, 1, 1],
            ['u6', 'Task 6', '2026-01-15', 7, 2, 9],
        ]
        backend = make_backend(mock_config)
        graph = make_graph_mock(in_uuid_order)
        backend._driver._get_graph = MagicMock(return_value=graph)

        rows, _ = await backend.enumerate_ranked_entity_nodes(group_id='home', page_size=2)

        by_uuid = {row[0]: row for row in in_uuid_order}
        assert rows == [
            _ranked_dict(by_uuid[u]) for u in ('u6', 'u4', 'u3', 'u2', 'u1', 'u5')
        ]

    @pytest.mark.asyncio
    async def test_a_row_re_emitted_across_a_page_boundary_is_returned_once(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([
            ['a', 'Task 1', '2026-01-01', 1, 0, 1],
            ['b', 'Task 2', '2026-01-01', 1, 0, 1],
            ['b', 'Task 2', '2026-01-01', 1, 0, 1],
            ['c', 'Task 3', '2026-01-01', 1, 0, 1],
        ])
        backend._driver._get_graph = MagicMock(return_value=graph)

        rows, paged = await backend.enumerate_ranked_entity_nodes(group_id='home', page_size=2)

        uuids = [row['uuid'] for row in rows]
        assert sorted(uuids) == ['a', 'b', 'c']
        assert paged.complete is True
        assert paged.rows_seen == 4

    @pytest.mark.asyncio
    async def test_reports_completeness_as_a_paged_read(
        self, mock_config, make_backend, make_graph_mock,
    ):
        corpus = [
            ['a', 'Task 1', '2026-01-01', 1, 0, 1],
            ['b', 'Alice', '2026-01-01', 0, 2, 2],
            ['c', 'Task 3', '2026-01-01', 1, 0, 1],
        ]
        backend = make_backend(mock_config)
        graph = make_graph_mock(corpus)
        backend._driver._get_graph = MagicMock(return_value=graph)

        _, paged = await backend.enumerate_ranked_entity_nodes(group_id='home', page_size=2)

        assert paged.complete is True
        assert paged.incomplete_kind is None
        assert paged.expected_rows == len(corpus)

    @pytest.mark.asyncio
    async def test_never_touches_the_writable_query_channel(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([['a', 'Task 1', '2026-01-01', 1, 0, 1]])
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.enumerate_ranked_entity_nodes(group_id='home')

        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_survivor_order_is_cyphers_order_by_over_null_and_tied_keys(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Every combination of tied rank, tied and NULL created_at, twice over,
        with uuids assigned independently of both — against the comparator
        spelled straight from the ORDER BY the destructive rankers emit."""
        combos = list(itertools.product(
            (0, 3, 5), (None, '2026-01-01T00:00:00Z', '2026-02-01T00:00:00Z'), (1, 2),
        ))
        random.Random(5662).shuffle(combos)
        rows = [
            {
                'uuid': f'u{i:02d}', 'name': f'Task {i}', 'created_at': created_at,
                'edge_count': rank, 'mentions_count': 0, 'provenance_rank': rank,
            }
            for i, (rank, created_at, _copy) in enumerate(combos)
        ]
        backend = make_backend(mock_config)
        graph = make_graph_mock([list(row.values()) for row in rows])
        backend._driver._get_graph = MagicMock(return_value=graph)

        ranked, _ = await backend.enumerate_ranked_entity_nodes(group_id='home', page_size=4)

        assert ranked == sorted(rows, key=functools.cmp_to_key(_cypher_order_by))

    @pytest.mark.asyncio
    async def test_a_rank_column_mixing_value_types_fails_naming_column_and_types(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([
            ['a', 'Task 1', '2026-01-01T00:00:00Z', 1, 0, 1],
            ['b', 'Task 2', 1767225600, 1, 0, 1],
        ])
        backend._driver._get_graph = MagicMock(return_value=graph)

        with pytest.raises(SurvivorRankTypeError) as raised:
            await backend.enumerate_ranked_entity_nodes(group_id='home')

        assert raised.value.column == 'created_at'
        assert raised.value.value_types == ('int', 'str')


@pytest_asyncio.fixture
async def ranked_census_live_graph():
    """Provision a throwaway, uniquely-named FalkorDB graph, yield it, then clean up."""
    graph_name = unique_graph_name('5662_ranked_census')
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
@pytest.mark.timeout(15)
@pytest.mark.integration
class TestEnumerateRankedEntityNodesLiveFalkorDB:
    """On a real server, the Python survivor sort agrees with the shared Cypher
    ORDER BY — NULL created_at included — and the read stays group-scoped."""

    @staticmethod
    async def _node(graph, uuid_, name, group_id, created_at=None):
        props = {'uuid': uuid_, 'name': name, 'gid': group_id, 'created': created_at}
        if created_at is None:
            await graph.query(
                'CREATE (:Entity {uuid: $uuid, name: $name, group_id: $gid})', props,
            )
        else:
            await graph.query(
                'CREATE (:Entity {uuid: $uuid, name: $name, group_id: $gid, '
                'created_at: $created})',
                props,
            )

    @staticmethod
    async def _relates(graph, src, edge_uuid, invalid_at=None):
        await graph.query(
            "MATCH (a:Entity {uuid: $src}), (b:Entity {uuid: 'nbr'}) "
            'CREATE (a)-[:RELATES_TO {uuid: $uuid, invalid_at: $invalid_at}]->(b)',
            {'src': src, 'uuid': edge_uuid, 'invalid_at': invalid_at},
        )

    @pytest.mark.asyncio
    async def test_python_survivor_order_matches_the_servers_order_by(
        self, mock_config, ranked_census_live_graph,
    ):
        graph_name, graph = ranked_census_live_graph
        await self._node(graph, 'nbr', 'Neighbour', graph_name, '2026-01-01T00:00:00Z')
        await self._node(graph, 'a', 'Task 605', graph_name, '2026-02-01T00:00:00Z')
        await self._node(graph, 'b', 'task 605', graph_name, '2026-01-01T00:00:00Z')
        await self._node(graph, 'c', 'tasks 605', graph_name)
        await self._node(graph, 'd', 'task #605', graph_name, '2026-03-01T00:00:00Z')
        await self._node(graph, 'e', 'TASK 605', graph_name, '2026-03-01T00:00:00Z')
        await self._node(graph, 'l', 'task 605', 'other_project', '2026-01-01T00:00:00Z')

        await self._relates(graph, 'a', 'a-e1')
        for i in range(3):
            await self._relates(graph, 'b', f'b-e{i}')
        await self._relates(graph, 'b', 'b-dead', invalid_at='2026-02-02T00:00:00Z')
        for uuid_ in ('c', 'd', 'e', 'l'):
            await self._relates(graph, uuid_, f'{uuid_}-e1')
        for i in range(4):
            await graph.query('CREATE (:Episodic {uuid: $u})', {'u': f'ep{i}'})
            await graph.query(
                "MATCH (ep:Episodic {uuid: $ep}), (n:Entity {uuid: 'a'}) "
                'CREATE (ep)-[:MENTIONS {uuid: $m}]->(n)',
                {'ep': f'ep{i}', 'm': f'm-a{i}'},
            )

        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            rows, paged = await backend.enumerate_ranked_entity_nodes(
                group_id=graph_name, page_size=2,
            )
            server_order = await backend.find_entity_nodes_by_name_substring(
                '605', group_id=graph_name,
            )
        finally:
            await backend.close()

        family = [row['uuid'] for row in rows if task_node_referent(row['name'])]
        assert family == [row['uuid'] for row in server_order] == ['a', 'b', 'd', 'e', 'c']
        assert 'l' not in {row['uuid'] for row in rows}
        assert paged.complete is True
        by_uuid = {row['uuid']: row for row in rows}
        assert by_uuid['b']['edge_count'] == 3
        assert by_uuid['a']['mentions_count'] == 4
