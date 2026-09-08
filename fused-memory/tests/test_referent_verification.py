"""The write-time verification sub-pass: does each edge hang off the node the
write is actually ABOUT? (task 3671, PRD leaf zeta of
plans/memory-referent-fidelity-prd.md).

Leaf gamma resolves WHICH referents a write is about; leaf epsilon threads that
set through the durable queue to `_execute_graphiti_write`.  This leaf is the
audit that closes the loop: after `add_episode` commits, walk the episode's
edges and ask, per endpoint, whether the node the edge landed on is one of the
things the write declared itself to be about.

zeta DETECTS and RECORDS.  It performs no writes at all: the repair
(`ensure_entity_node` -> `reassign_edge` -> `refresh_entity_summary`) and the
repair-storm streak escalation are leaf eta's, and the declaration-rate read
path is leaf iota's.  So every finding here is structured evidence handed to
eta, never an action taken.

Two checks, because resolved decision 7 says neither alone is sufficient:
SET MEMBERSHIP catches the dominant live shape (an edge landed on a task number
the write never names), and PER-EDGE PAIRING catches mode (iii), where BOTH
numbers are legitimately declared and the edge still landed on the wrong one.

The fail-closed direction is structural: a finding whose correct target cannot
be determined is RECORDED and LEFT ALONE, never guessed at.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import (
    MockAddEpisodeResult,
    MockEdge,
    MockNode,
    install_identity_mocks,
)

from fused_memory.services import memory_service as memory_service_module
from fused_memory.services.memory_service import (
    REFERENT_CHECKS,
    REFERENT_FINDING_AXES,
    MemoryService,
    ReferentFinding,
    ReferentStats,
)
from fused_memory.utils.canonical_labels import Referent


@pytest.fixture
def service(mock_config):
    """MemoryService with fully-mocked backends.

    Lean by design — zeta touches only `self.graphiti` (one read-only
    `get_nodes_by_exact_name`, and only when a finding exists).
    `install_identity_mocks` is REQUIRED, not decorative: the wiring tests drive
    `_execute_graphiti_write`, which wraps its critical section in
    `async with self.graphiti._identity_lock_for(...)`, which a bare MagicMock
    cannot satisfy.
    """
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti._require_client = MagicMock()
    svc.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])
    for name in _WRITE_PRIMITIVES:
        setattr(svc.graphiti, name, AsyncMock(return_value=None))
    install_identity_mocks(svc.graphiti)
    return svc


def _finding(**overrides) -> ReferentFinding:
    """A minimally-valid finding; overrides tune whichever field a test pins."""
    fields = {
        'edge_uuid': 'edge-1',
        'which_end': 'source',
        'check': 'set-membership',
        'old_endpoint_uuid': 'node-1',
        'old_endpoint_name': 'Task 2520',
        'endpoint_referent': Referent(number='2520'),
        'referent_set': ('Task 2519',),
    }
    fields.update(overrides)
    return ReferentFinding(**fields)


class TestReferentRecordVocabulary:
    """INV-2: findings are STRUCTURED RECORDS, not a log line."""

    def test_check_vocabulary_is_closed_and_exactly_the_two_named_checks(self):
        """The single normative site for "WHICH CHECK FIRED" (INV-5)."""
        assert REFERENT_CHECKS == ('set-membership', 'per-edge-pairing')

    def test_required_fields_construct_and_defaults_are_fail_closed(self):
        finding = _finding()

        assert finding.edge_uuid == 'edge-1'
        assert finding.which_end == 'source'
        assert finding.check == 'set-membership'
        assert finding.old_endpoint_uuid == 'node-1'
        assert finding.old_endpoint_name == 'Task 2520'
        assert finding.endpoint_referent == Referent(number='2520')
        assert finding.referent_set == ('Task 2519',)
        # The fail-closed direction: "recorded and left alone" is the DEFAULT,
        # guessing at a repair target is opt-in.
        assert finding.intended_referent is None
        assert finding.new_endpoint_uuid is None
        assert finding.resolvable is False
        assert finding.reason == ''
        # DEFAULTED, so every existing construction site — including
        # tests/test_referent_repair.py's `_finding()` and leaf eta's landed
        # repair path — keeps constructing unchanged.
        assert finding.cited == ()

    def test_is_keyword_only(self):
        """Positional construction of a record this wide is how a field
        silently lands in the wrong slot."""
        with pytest.raises(TypeError):
            ReferentFinding('edge-1', 'source', 'set-membership')  # type: ignore[misc]

    def test_is_frozen_but_replaceable(self):
        """A finding is evidence for destructive edge surgery — a consumer must
        not be able to rewrite which edge end it names."""
        finding = _finding()

        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.edge_uuid = 'edge-2'  # type: ignore[misc]

        replaced = dataclasses.replace(finding, edge_uuid='edge-2')
        assert replaced.edge_uuid == 'edge-2'
        assert finding.edge_uuid == 'edge-1'

    def test_referent_set_is_a_tuple_not_a_list(self):
        """`frozen=True` blocks REBINDING only, so a list field would leave
        `finding.referent_set.append(...)` wide open — the same reason
        `LabelScan` uses tuples."""
        assert isinstance(_finding().referent_set, tuple)

    def test_an_unregistered_check_is_rejected_by_name(self):
        """Mirrors `Referent.__post_init__`'s unregistered-kind rejection: the
        vocabulary is closed, and a third check must be REGISTERED, not
        smuggled in as a string at a call site."""
        with pytest.raises(ValueError, match='invented'):
            _finding(check='invented')

        with pytest.raises(ValueError, match='set-membership'):
            _finding(check='invented')

    def test_to_dict_is_json_safe_and_keyed_exactly_by_the_field_names(self):
        payload = _finding(
            intended_referent=Referent(number='2519'),
            new_endpoint_uuid='node-2',
            resolvable=True,
            reason='',
        ).to_dict()

        assert set(payload) == {f.name for f in dataclasses.fields(ReferentFinding)}
        assert json.loads(json.dumps(payload)) == payload
        # Referents render as their canonical node name, not as a dataclass repr.
        assert payload['endpoint_referent'] == 'Task 2520'
        assert payload['intended_referent'] == 'Task 2519'
        assert payload['referent_set'] == ['Task 2519']

    def test_to_dict_renders_an_absent_intended_referent_as_none(self):
        assert _finding().to_dict()['intended_referent'] is None


class TestReferentStatsVocabulary:
    """Counts are DERIVED over `findings`, so they cannot drift from it."""

    def test_empty_stats_are_all_zero(self):
        stats = ReferentStats()

        assert stats.edges_scanned == 0
        assert stats.endpoints_checked == 0
        assert stats.endpoints_unresolved == 0
        assert stats.findings == []
        assert stats.set_membership_findings == 0
        assert stats.pairing_findings == 0
        assert stats.unresolvable_findings == 0

    def test_counts_are_derived_from_the_findings_list(self):
        stats = ReferentStats()
        stats.findings.append(_finding(check='set-membership', resolvable=True))
        stats.findings.append(_finding(check='per-edge-pairing', resolvable=True))
        stats.findings.append(_finding(check='per-edge-pairing', resolvable=False))

        assert stats.set_membership_findings == 1
        assert stats.pairing_findings == 2
        assert stats.unresolvable_findings == 1

    def test_each_stats_instance_gets_its_own_findings_list(self):
        first, second = ReferentStats(), ReferentStats()
        first.findings.append(_finding())

        assert second.findings == []


def _episode(edges, nodes) -> MockAddEpisodeResult:
    """An add_episode result carrying *edges* and *nodes*.

    `entity_edges` is nulled so the `.edges` attribute is the one zeta reads —
    `MockAddEpisodeResult.__post_init__` mirrors entity_edges INTO edges, and a
    test that let both carry the same list could not tell which one was walked.
    """
    result = MockAddEpisodeResult(edges=edges, nodes=nodes)
    result.entity_edges = []
    return result


def _edge(uuid='e1', *, fact='', source='n-src', target='n-tgt') -> MockEdge:
    return MockEdge(
        fact=fact, uuid=uuid, source_node_uuid=source, target_node_uuid=target,
    )


#: Every GraphitiBackend primitive that MUTATES the graph. zeta detects and
#: records; the repair is leaf eta's, so none of these may ever be awaited.
_WRITE_PRIMITIVES = (
    'merge_entities',
    'rename_entity_node',
    'update_edge',
    'reassign_edge',
    'ensure_entity_node',
    'refresh_entity_summary',
    'add_episode',
)


def assert_never_repaired(service) -> None:
    for name in _WRITE_PRIMITIVES:
        getattr(service.graphiti, name).assert_not_awaited()


class TestSetMembershipCheck:
    """Did the edge land on a node this write is not about at all?

    The dominant live shape: five of the PRD's measured cases share the
    signature that the number the edge landed on is never named by the fact.
    """

    @pytest.mark.asyncio
    async def test_fires_with_every_contract_field_populated(self, service):
        result = _episode(
            edges=[_edge('e1', fact='the deploy pipeline was retried',
                         source='n-3129', target='n-x')],
            nodes=[MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        assert finding.edge_uuid == 'e1'
        assert finding.which_end == 'source'
        assert finding.old_endpoint_uuid == 'n-3129'
        assert finding.old_endpoint_name == 'Task 3129'
        assert finding.endpoint_referent == Referent(number='3129')
        assert finding.referent_set == ('Task 3127',)
        assert stats.set_membership_findings == 1
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_an_endpoint_in_the_set_produces_no_finding(self, service):
        result = _episode(
            edges=[_edge('e1', source='n-3127', target='n-x')],
            nodes=[MockNode(name='Task 3127', uuid='n-3127'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.findings == []
        assert stats.endpoints_checked == 1
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_a_non_canonical_endpoint_spelling_still_passes(self, service):
        """The check keys on `parse_node_name`'s Referent, NOT a raw-string
        compare — which is what makes it invariant across the rename/merge
        `_normalize_task_node_names` performs immediately before it."""
        result = _episode(
            edges=[_edge('e1', source='n-3127', target='n-x')],
            nodes=[MockNode(name='task #3127', uuid='n-3127'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.findings == []
        assert stats.endpoints_checked == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', ['MergeWorker', 'Task 42 orchestrator',
                                      'PRD decision D2'])
    async def test_endpoints_that_are_not_task_labels_are_never_checked(
        self, service, name,
    ):
        """The postcondition is scoped to endpoints whose NAME parses as a
        canonical task referent. 'Task 42 orchestrator' is the anchoring case:
        it MENTIONS a task and is not one."""
        result = _episode(
            edges=[_edge('e1', source='n-a', target='n-b')],
            nodes=[MockNode(name=name, uuid='n-a'),
                   MockNode(name='deploy pipeline', uuid='n-b')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.findings == []
        assert stats.endpoints_checked == 0

    @pytest.mark.asyncio
    async def test_both_ends_are_checked_and_which_end_is_recorded(self, service):
        """`which_end` is what lets eta call `reassign_edge(..., which_end=...)`."""
        result = _episode(
            edges=[_edge('e1', source='n-3127', target='n-3129')],
            nodes=[MockNode(name='Task 3127', uuid='n-3127'),
                   MockNode(name='Task 3129', uuid='n-3129')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].which_end == 'target'
        assert stats.findings[0].old_endpoint_uuid == 'n-3129'
        assert stats.endpoints_checked == 2

    @pytest.mark.asyncio
    async def test_digits_are_compared_verbatim_never_int_normalized(self, service):
        """`Referent(number='0132') != Referent(number='132')` — a referent
        never invents or reformats a task number."""
        result = _episode(
            edges=[_edge('e1', source='n-132', target='n-x')],
            nodes=[MockNode(name='Task 132', uuid='n-132'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='0132'),),
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].endpoint_referent == Referent(number='132')

    @pytest.mark.asyncio
    async def test_a_cross_project_endpoint_is_a_different_referent(self, service):
        """The qualifier is a DIFFERENT-project signal and is never normalized
        away — flattening 'reify:132' onto 'Task 132' is precisely the
        cross-project collapse utils/cross_project_refs.py exists to detect."""
        result = _episode(
            edges=[_edge('e1', source='n-r132', target='n-x')],
            nodes=[MockNode(name='reify:132', uuid='n-r132'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='132'),),
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].endpoint_referent == Referent(
            number='132', project_id='reify',
        )

    @pytest.mark.asyncio
    async def test_a_self_qualified_endpoint_is_the_local_referent(self, service):
        """A qualifier naming OUR OWN project is not a different-project signal.

        The mirror image of the test above, and the boundary between them is
        the whole point: 'reify:132' in group 'dark_factory' stays foreign,
        'dark_factory:3127' in that same group is the LOCAL Task 3127.

        `parse_node_name` alone preserves the qualifier it read, so this
        endpoint compared unequal to the very referent the write declared
        itself to be about, set-membership fired, and ZETA emitted a repair
        instruction for a correctly-attached edge. The self-qualified
        reclassification is SOURCE-INVARIANT by contract — `scan_content` and
        `resolve_referents` both answer the local referent for this spelling —
        so the endpoint path must answer it too.
        """
        result = _episode(
            edges=[_edge('e1', source='n-df3127', target='n-x')],
            nodes=[MockNode(name='dark_factory:3127', uuid='n-df3127'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.findings == []
        assert stats.endpoints_checked == 1
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_a_fact_citing_the_self_qualified_spelling_still_corroborates(
        self, service,
    ):
        """The second half of the same break: the corroboration veto.

        `endpoint in cited` compares the endpoint against `scan_content`'s
        refs, which are ALREADY locally classified. With the endpoint left
        foreign-qualified the two could never be equal, so the veto could not
        fire even though the fact cites the endpoint node's own name verbatim —
        and the finding came back `resolvable=True` with a resolved
        `new_endpoint_uuid`. Here the endpoint is undeclared (Task 2500), so
        membership DOES fire; what is pinned is that the veto reaches it.
        """
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=_rows('n-3668'),
        )
        result = _episode(
            edges=[_edge('e1',
                         fact='dark_factory:2500 was completed by the merge worker',
                         source='n-df2500', target='n-worker')],
            nodes=[MockNode(name='dark_factory:2500', uuid='n-df2500'),
                   MockNode(name='merge worker', uuid='n-worker')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3668'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        # Reclassified: the endpoint is the LOCAL Task 2500, not a foreign one.
        assert finding.endpoint_referent == Referent(number='2500')
        assert finding.resolvable is False
        assert finding.intended_referent is None
        assert 'cites' in finding.reason
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_a_none_result_is_a_no_op(self, service):
        stats = await service._verify_episode_referents(
            None, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.findings == []
        assert stats.edges_scanned == 0
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_an_edgeless_result_is_a_no_op(self, service):
        stats = await service._verify_episode_referents(
            _episode(edges=[], nodes=[MockNode(name='Task 3129', uuid='n-3129')]),
            group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings == []
        assert stats.edges_scanned == 0

    @pytest.mark.asyncio
    async def test_an_entity_edges_only_result_is_walked_like_the_sibling_sweeps(
        self, service,
    ):
        """The `result.edges or result.entity_edges` idiom, which this pass's
        docstring promises parity with.

        Every one of the six sibling post-write sweeps in `memory_service.py`
        reads `getattr(result, 'edges', None) or getattr(result,
        'entity_edges', None) or []`. A result exposing only `entity_edges` is
        a shape they all still walk, so zeta reading `edges` ALONE would make
        it a silent, total no-op there — no findings, no counters, no warning —
        rather than the degraded-but-handled behaviour it documents.

        `_episode()` deliberately nulls `entity_edges` so the other tests can
        prove `.edges` is the attribute walked; this is the mirror case, so it
        builds the result directly. Assigning after construction is what makes
        the shape reachable at all: `MockAddEpisodeResult.__post_init__`
        mirrors `entity_edges` INTO `edges`, so passing it to the constructor
        would populate both and prove nothing.
        """
        result = MockAddEpisodeResult(
            edges=[],
            nodes=[MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )
        result.entity_edges = [
            _edge('e1', fact='the deploy pipeline was retried',
                  source='n-3129', target='n-x'),
        ]
        assert result.edges == [], 'the shape under test carries no `.edges`'

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.edges_scanned == 1
        assert len(stats.findings) == 1
        assert stats.findings[0].edge_uuid == 'e1'
        assert stats.findings[0].check == 'set-membership'
        assert stats.set_membership_findings == 1
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_an_empty_referent_set_makes_the_whole_pass_a_no_op(self, service):
        """Honours epsilon's published contract ("an EMPTY `.referents` carries
        nothing to test membership against, so a downstream verifier must no-op
        on it regardless of `.source`") and the PRD's `source='none'` row."""
        result = _episode(
            edges=[_edge('e1', fact='Task 3129 was merged',
                         source='n-3129', target='n-x')],
            nodes=[MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(),
        )

        assert stats.findings == []
        assert stats.edges_scanned == 0
        assert stats.endpoints_checked == 0
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_an_endpoint_this_episode_does_not_name_is_counted(self, service):
        """The one blind spot, COUNTED rather than silently skipped: a check
        that did not run is not a check that passed."""
        result = _episode(
            edges=[_edge('e1', source='n-unknown', target='n-x')],
            nodes=[MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.findings == []
        assert stats.endpoints_unresolved == 1
        assert stats.endpoints_checked == 0

    @pytest.mark.asyncio
    async def test_edges_scanned_counts_every_edge_walked(self, service):
        result = _episode(
            edges=[_edge('e1', source='n-3127', target='n-x'),
                   _edge('e2', source='n-x', target='n-3127'),
                   _edge('e3', source='n-x', target='n-x')],
            nodes=[MockNode(name='Task 3127', uuid='n-3127'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.edges_scanned == 3
        assert stats.findings == []


class TestPerEdgePairingCheck:
    """Does THIS edge's fact talk about the node THIS edge landed on?

    Resolved decision 7: neither check alone is sufficient. Membership misses
    mode (iii) — both numbers legitimately declared, the edge still on the wrong
    one — which is exactly what pairing catches.
    """

    @pytest.mark.asyncio
    async def test_mode_iii_fires_where_set_membership_cannot(self, service):
        """The motivating live case: episode 8562760e minted Task 3074 and
        Task 3075 60us apart, and a fact about one landed on the other."""
        result = _episode(
            edges=[_edge('e1', fact='Task 3075 blocks the merge lane',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3074'), Referent(number='3075')),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'per-edge-pairing'
        assert finding.which_end == 'source'
        assert finding.endpoint_referent == Referent(number='3074')
        # 3074 IS legitimately declared, which is precisely why membership
        # alone misses this mode — the row that makes both checks required.
        assert stats.set_membership_findings == 0
        assert stats.pairing_findings == 1
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_the_membership_only_shape_of_that_episode_is_clean(self, service):
        """Complementarity as a test: the same episode shape with the fact
        citing the endpoint it landed on yields nothing. One shape, two
        verdicts, decided by the check membership cannot make."""
        result = _episode(
            edges=[_edge('e1', fact='Task 3074 blocks the merge lane',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3074'), Referent(number='3075')),
        )

        assert stats.findings == []

    @pytest.mark.asyncio
    async def test_a_binary_fact_citing_both_endpoints_is_clean(self, service):
        result = _episode(
            edges=[_edge('e1', fact='Task 3074 depends on Task 3075',
                         source='n-3074', target='n-3075')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='Task 3075', uuid='n-3075')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3074'), Referent(number='3075')),
        )

        assert stats.findings == []
        assert stats.endpoints_checked == 2

    @pytest.mark.asyncio
    async def test_an_empty_scan_is_uninformative_never_contradictory(self, service):
        """gamma's `_conflicting_referents` choice-4 rule one level down: a fact
        citing no number says NOTHING about which node its edge belongs on. A
        scanner blind spot must never manufacture evidence for destructive edge
        surgery."""
        result = _episode(
            edges=[_edge('e1', fact='The merge lane hardening work landed',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3074'),),
        )

        assert stats.findings == []
        assert stats.endpoints_checked == 1

    @pytest.mark.asyncio
    async def test_ambiguous_scan_results_are_never_promoted_into_the_cited_set(
        self, service,
    ):
        """'task 2500 tracked as reify:2500' claims 2500 BOTH bare and
        foreign-qualified, so the scan routes both to `.ambiguous` and `.refs`
        is empty. Were ambiguity promoted, the in-set endpoint below would look
        uncited and fire. Ambiguity is recorded, not guessed."""
        result = _episode(
            edges=[_edge('e1', fact='task 2500 tracked as reify:2500',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3074'),),
        )

        assert stats.findings == []

    @pytest.mark.asyncio
    async def test_a_non_canonical_fact_spelling_is_still_seen(self, service):
        """The fact is read through `scan_content` — the vocabulary leaf beta
        exists to single-source — and NOT through a second regex, which is what
        makes 'task #3075' visible at all."""
        result = _episode(
            edges=[_edge('e1', fact='task #3075 blocks the merge lane',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3074'), Referent(number='3075')),
        )

        assert stats.pairing_findings == 1

    @pytest.mark.asyncio
    async def test_an_endpoint_failing_both_checks_reports_exactly_one_finding(
        self, service,
    ):
        """Ordered, not additive. Membership is the stronger, more specific
        signal, so it wins the label — two findings naming the same
        (edge_uuid, which_end) would hand eta two repair instructions for one
        edge end and double-count in iota's rate."""
        result = _episode(
            edges=[_edge('e1', fact='Task 3127 was completed',
                         source='n-3129', target='n-lane')],
            nodes=[MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].check == 'set-membership'
        assert stats.pairing_findings == 0

    @pytest.mark.asyncio
    async def test_a_fact_citing_only_undeclared_tasks_fires_nothing(self, service):
        """Pairing is a discriminator AMONG DECLARED REFERENTS.

        Resolved decision 7 / mode (iii): the check exists to tell which of the
        write's OWN referents an edge belongs on. Here the endpoint (Task 3074)
        is itself declared, so it already satisfies the PRD's first
        postcondition — "every task-parsing endpoint is either in `referents`,
        or has been repointed to a node that is". And `Task 77` can never become
        a target, because the intersection rule in `_candidate_pool` forbids one
        from outside the declared set.

        There is therefore nothing to point at and nothing to say. This is the
        existing `if cited` guard's own rationale — UNINFORMATIVE, never
        contradictory — one step further out.
        """
        result = _episode(
            edges=[_edge('e1', fact='Task 77 supersedes this',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3074'),),
        )

        assert stats.findings == []
        # CHECKED and found clean, not skipped — a check that did not run is not
        # a check that passed.
        assert stats.endpoints_checked == 1
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_no_counter_moves_and_no_warning_is_emitted_for_that_shape(
        self, service, caplog,
    ):
        """The half a candidate-selection fix alone does not address.

        A finding unactionable BY CONSTRUCTION still inflates the
        `per-edge-pairing` process counter leaf iota reads, and still raises an
        operator WARNING, for an endpoint with no observable defect. A rate iota
        samples must not be polluted by a shape nothing can act on.
        """
        before = service.referent_finding_counts()
        result = _episode(
            edges=[_edge('e1', fact='Task 77 supersedes this',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        with caplog.at_level(logging.WARNING,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                result, group_id='dark_factory',
                referents=(Referent(number='3074'),),
            )

        assert service.referent_finding_counts() == before
        assert not [r for r in caplog.records
                    if 'Referent verification finding' in r.getMessage()]

    @pytest.mark.asyncio
    async def test_a_declared_citation_still_fires_even_when_the_fact_also_cites_undeclared_tasks(
        self, service,
    ):
        """The guard must not OVER-suppress.

        `cited & referents == {3075}` is non-empty, so the fact names a concrete
        declared alternative this edge could belong on — mode (iii), pairing's
        whole reason to exist. The regression guard proving the narrowing hits
        the uninformative shape ONLY.
        """
        result = _episode(
            edges=[_edge('e1', fact='Task 77 says Task 3075 blocks the merge lane',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3074'), Referent(number='3075')),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'per-edge-pairing'
        assert finding.resolvable is True
        assert finding.intended_referent == Referent(number='3075')
        # A target never comes from outside the declared set.
        assert finding.intended_referent != Referent(number='77')

    @pytest.mark.asyncio
    async def test_the_undeclared_only_shape_is_still_caught_when_the_endpoint_is_undeclared_too(
        self, service,
    ):
        """Precedence is unchanged: the guard constrains the PAIRING arm only.

        The endpoint sits outside the declared set, so SET MEMBERSHIP — the
        dominant live check — still fires on exactly the fact shape the guard
        silences on the other arm.
        """
        result = _episode(
            edges=[_edge('e1', fact='Task 77 supersedes this',
                         source='n-99', target='n-lane')],
            nodes=[MockNode(name='Task 99', uuid='n-99'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='10'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        assert finding.resolvable is True
        assert finding.intended_referent == Referent(number='10')


class TestFindingResolvability:
    """Recorded and left alone, never guessed at.

    One test per row of the PRD's boundary-test sketch. A finding whose correct
    target cannot be DETERMINED is still recorded — it is just recorded as
    unrepairable, with the reason, rather than dropped or guessed at.
    """

    @pytest.mark.asyncio
    async def test_mode_i_correct_node_absent_resolves_to_the_sole_referent(
        self, service,
    ):
        result = _episode(
            edges=[_edge('e1', fact='the deploy pipeline was retried',
                         source='n-3129', target='n-x')],
            nodes=[MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        finding = stats.findings[0]
        assert finding.resolvable is True
        assert finding.intended_referent == Referent(number='3127')
        assert finding.reason == ''

    @pytest.mark.asyncio
    async def test_mode_ii_llm_picked_the_sibling_resolves_to_the_cited_task(
        self, service,
    ):
        result = _episode(
            edges=[_edge('e1', fact='Task 1031 was verified',
                         source='n-1030', target='n-x')],
            nodes=[MockNode(name='Task 1030', uuid='n-1030'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='1031'),),
        )

        finding = stats.findings[0]
        assert finding.resolvable is True
        assert finding.intended_referent == Referent(number='1031')

    @pytest.mark.asyncio
    async def test_mode_iii_the_fact_cited_candidate_wins_over_the_whole_set(
        self, service,
    ):
        """Without fact-scoping, two declared referents would look ambiguous and
        a repair the fact UNAMBIGUOUSLY determines would be abandoned."""
        result = _episode(
            edges=[_edge('e1', fact='Task 3075 blocks the merge lane',
                         source='n-3074', target='n-lane')],
            nodes=[MockNode(name='Task 3074', uuid='n-3074'),
                   MockNode(name='merge lane', uuid='n-lane')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3074'), Referent(number='3075')),
        )

        finding = stats.findings[0]
        assert finding.check == 'per-edge-pairing'
        assert finding.resolvable is True
        assert finding.intended_referent == Referent(number='3075')

    @pytest.mark.asyncio
    async def test_a_unary_fact_leaves_the_other_end_recorded_but_unrepairable(
        self, service,
    ):
        """The live Task 2519/2520 row, and the PRD's explicitly unrepairable
        one. The only candidate IS the edge's other endpoint, and repointing
        onto it would form the self-loop `reassign_edge` refuses — so there is
        no correct target, and the finding is recorded rather than acted on."""
        result = _episode(
            edges=[_edge('e1', fact='Task 2519 was completed',
                         source='n-2519', target='n-2520')],
            nodes=[MockNode(name='Task 2519', uuid='n-2519'),
                   MockNode(name='Task 2520', uuid='n-2520')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='2519'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.which_end == 'target'
        assert finding.check == 'set-membership'
        assert finding.resolvable is False
        assert finding.intended_referent is None
        assert finding.reason
        assert 'Task 2519' in finding.reason
        assert stats.unresolvable_findings == 1
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_two_candidates_with_no_fact_evidence_are_never_guessed_between(
        self, service,
    ):
        result = _episode(
            edges=[_edge('e1', fact='the deploy pipeline was retried',
                         source='n-99', target='n-x')],
            nodes=[MockNode(name='Task 99', uuid='n-99'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='10'), Referent(number='11')),
        )

        finding = stats.findings[0]
        assert finding.resolvable is False
        assert finding.intended_referent is None
        assert 'Task 10' in finding.reason
        assert 'Task 11' in finding.reason

    @pytest.mark.asyncio
    async def test_a_target_never_comes_from_outside_the_declared_referent_set(
        self, service,
    ):
        """The candidate pool is fact-cited INTERSECT declared, never a union:
        'Task 77' is cited by the LLM-restated fact but the write never declared
        itself to be about it, so it must not become a repair target."""
        result = _episode(
            edges=[_edge('e1', fact='Task 77 supersedes this',
                         source='n-99', target='n-x')],
            nodes=[MockNode(name='Task 99', uuid='n-99'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='10'),),
        )

        finding = stats.findings[0]
        assert finding.intended_referent != Referent(number='77')
        # The intersection is empty, so the rule falls back to the declared set,
        # whose sole member is the only thing a repair may ever point at.
        assert finding.resolvable is True
        assert finding.intended_referent == Referent(number='10')

    @pytest.mark.asyncio
    async def test_unresolvable_findings_counts_exactly_the_unrepairable_ones(
        self, service,
    ):
        result = _episode(
            edges=[_edge('e1', fact='Task 2519 was completed',
                         source='n-2519', target='n-2520'),
                   _edge('e2', fact='the deploy pipeline was retried',
                         source='n-3129', target='n-x')],
            nodes=[MockNode(name='Task 2519', uuid='n-2519'),
                   MockNode(name='Task 2520', uuid='n-2520'),
                   MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='2519'),),
        )

        assert len(stats.findings) == 2
        assert stats.unresolvable_findings == 1
        assert sorted(f.resolvable for f in stats.findings) == [False, True]


class TestCandidateTargetSelection:
    """The pure rule, unit-tested away from the walk that drives it.

    `_candidate_targets`' own docstring advertises being "directly unit-testable
    in isolation from the walk that drives it", so the invariant that a repair
    target is never a node the edge is ALREADY attached to is pinned HERE, on
    the single site that decides targets — not on end-to-end
    `_verify_episode_referents` behaviour, which the pairing-arm narrowing
    changes independently. These assertions hold either way.
    """

    def test_the_pool_is_the_fact_cited_intersection_when_it_is_non_empty(self):
        """Mode (iii): the fact cites 3075, the edge sits on 3074."""
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='3074'), Referent(number='3075')}),
            cited=frozenset({Referent(number='3075')}),
            endpoint=Referent(number='3074'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        ) == (Referent(number='3075'),)

    def test_it_falls_back_to_the_declared_set_when_the_intersection_is_empty(self):
        """The MEMBERSHIP shape: the endpoint is outside the declared set, so
        subtracting it is a no-op and the sole declared referent survives."""
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='10')}),
            cited=frozenset({Referent(number='77')}),
            endpoint=Referent(number='99'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        ) == (Referent(number='10'),)

    def test_the_other_endpoint_is_subtracted_so_no_repair_forms_a_self_loop(self):
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='2519')}),
            cited=frozenset({Referent(number='2519')}),
            endpoint=Referent(number='2520'),
            other_endpoint=Referent(number='2519'),
            ambiguous=frozenset(), source='derived',
        ) == ()

    def test_the_order_is_deterministic_and_not_frozenset_iteration_order(self):
        """A finding must be stable across runs and diffable in eta's audit;
        frozenset iteration order is not stable across processes."""
        from fused_memory.services.memory_service import _candidate_targets

        refs = frozenset({
            Referent(number='11'), Referent(number='10'),
            Referent(number='2', project_id='reify'),
        })
        first = _candidate_targets(referents=refs, cited=frozenset(),
                                   endpoint=Referent(number='99'),
                                   other_endpoint=None,
                                   ambiguous=frozenset(), source='derived')

        assert first == tuple(sorted(first, key=lambda r: (r.kind, r.project_id,
                                                           r.number)))
        assert _candidate_targets(referents=refs, cited=frozenset(),
                                  endpoint=Referent(number='99'),
                                  other_endpoint=None,
                                  ambiguous=frozenset(),
                                  source='derived') == first

    def test_the_flagged_endpoint_is_subtracted_so_no_repair_targets_the_node_it_is_already_on(self):
        """A "repair" onto the node the edge is ALREADY attached to is not a
        repair — and is not even a harmless no-op.

        `_intended_endpoint_uuid` resolves the CANONICAL name, so with a
        non-canonical endpoint spelling ('task #3074') and a canonical
        'Task 3074' node both present, a self-targeting finding yields a
        DIFFERENT uuid and eta performs real edge surgery on an endpoint that
        was already correct.

        Zero survivors is therefore the RIGHT answer, not a defect: the caller
        records `resolvable=False` with a reason, which is the fail-closed
        "recorded and left alone" direction.
        """
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='3074')}),
            cited=frozenset({Referent(number='77')}),
            endpoint=Referent(number='3074'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        ) == ()

    def test_both_endpoints_are_subtracted_together(self):
        """Neither end of an edge can ever be its own repair target."""
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='3074'), Referent(number='3075')}),
            cited=frozenset(),
            endpoint=Referent(number='3074'),
            other_endpoint=Referent(number='3075'),
            ambiguous=frozenset(), source='derived',
        ) == ()

    def test_the_membership_arm_is_unaffected_by_the_endpoint_subtraction(self):
        """Asserted rather than merely argued: the pool is always a SUBSET of
        `referents`, and membership fires precisely when the endpoint is NOT in
        `referents`, so the subtraction provably cannot bite on that arm."""
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='10'), Referent(number='11')}),
            cited=frozenset(),
            endpoint=Referent(number='99'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        ) == (Referent(number='10'), Referent(number='11'))


    def test_a_fact_citing_the_endpoint_corroborates_it_and_nominates_nothing(self):
        """The dominant `source='metadata'` write shape, which the whole-set
        fallback used to turn into a repair instruction.

        `resolve_referents` derives `source='metadata'` from the write's ambient
        `task_id`, and its own docstring names this shape as legitimate and
        deliberately NOT a conflict: "An agent working on task 3668 legitimately
        writes memories about Task 2500". The edge's fact NAMES `Task 2500`, the
        node it landed on — the strongest possible evidence the attachment is
        CORRECT — so the declared set {3668} must not be mined for a target.
        """
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='3668')}),
            cited=frozenset({Referent(number='2500')}),
            endpoint=Referent(number='2500'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        ) == ()

    def test_corroboration_outranks_a_non_empty_intersection(self):
        """The guard is a VETO, not a fallback the intersection can outrank.

        The same legitimate ambient-task write, one sentence longer: "Task 2500
        was completed as part of task 3668 by the merge worker" cites BOTH the
        endpoint and the declared referent, so `cited & referents` is non-empty.
        Testing the intersection first would short-circuit past the guard and
        hand back `Task 3668` — a repair instruction against a fact that
        literally asserts the edge is about Task 2500.

        This is the REACHABLE shape: the endpoint (2500) is undeclared, which is
        precisely what makes the membership arm fire on it. An earlier version of
        this test pinned endpoint 3074 as both DECLARED and CITED — a shape
        `_verify_episode_referents` can never produce, since membership needs the
        endpoint undeclared and pairing needs it uncited — and so pinned the
        wrong behaviour on an input no arm can reach.
        """
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='3668')}),
            cited=frozenset({Referent(number='2500'), Referent(number='3668')}),
            endpoint=Referent(number='2500'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        ) == ()

    def test_the_guard_cannot_fire_on_the_pairing_arm_so_mode_iii_still_resolves(self):
        """Suppressing unconditionally costs mode (iii) nothing.

        The pairing arm is only reached when `endpoint_referent not in cited`, so
        no input that arm can produce ever satisfies the guard — the intersection
        below still decides, and the fact's citation of `Task 3075` remains the
        repair target.
        """
        from fused_memory.services.memory_service import _candidate_targets

        assert _candidate_targets(
            referents=frozenset({Referent(number='3074'), Referent(number='3075')}),
            cited=frozenset({Referent(number='3075')}),
            endpoint=Referent(number='3074'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        ) == (Referent(number='3075'),)


class TestUnresolvableReason:
    """"Recorded and left alone" must stay legible as a REASON, not an absence.

    A reader must be able to tell "the check had nothing to point at but the
    node it was already on" from "the only target would form a self-loop".
    """

    def test_more_than_one_candidate_names_the_ambiguity(self):
        from fused_memory.services.memory_service import _unresolvable_reason

        reason = _unresolvable_reason(
            (Referent(number='10'), Referent(number='11')),
            cited=frozenset(),
            pool=frozenset({Referent(number='10'), Referent(number='11')}),
            endpoint=Referent(number='99'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        )

        assert 'Task 10' in reason and 'Task 11' in reason
        assert 'more than one' in reason

    def test_zero_candidates_names_the_endpoint_already_attached_condition(self):
        """The pool held only the flagged endpoint itself."""
        from fused_memory.services.memory_service import _unresolvable_reason

        reason = _unresolvable_reason(
            (),
            cited=frozenset(),
            pool=frozenset({Referent(number='3074')}),
            endpoint=Referent(number='3074'),
            other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        )

        assert 'Task 3074' in reason
        assert 'already' in reason
        assert 'self-loop' not in reason

    def test_zero_candidates_still_names_the_self_loop_for_the_live_2519_row(self):
        """referents {2519}, endpoints (Task 2519, Task 2520), a unary fact."""
        from fused_memory.services.memory_service import _unresolvable_reason

        reason = _unresolvable_reason(
            (),
            cited=frozenset(),
            pool=frozenset({Referent(number='2519')}),
            endpoint=Referent(number='2520'),
            other_endpoint=Referent(number='2519'),
            ambiguous=frozenset(), source='derived',
        )

        assert 'Task 2519' in reason
        assert 'self-loop' in reason

    def test_the_two_zero_candidate_reasons_are_distinguishable(self):
        from fused_memory.services.memory_service import _unresolvable_reason

        already_attached = _unresolvable_reason(
            (), cited=frozenset(), pool=frozenset({Referent(number='3074')}),
            endpoint=Referent(number='3074'), other_endpoint=None,
            ambiguous=frozenset(), source='derived',
        )
        self_loop = _unresolvable_reason(
            (), cited=frozenset(), pool=frozenset({Referent(number='2519')}),
            endpoint=Referent(number='2520'),
            other_endpoint=Referent(number='2519'),
            ambiguous=frozenset(), source='derived',
        )

        assert already_attached != self_loop

    def test_the_endpoint_arm_wins_when_both_subtractions_apply(self):
        """Tested against the PRE-subtraction pool rather than inferred from
        `other_endpoint is None`, which is what keeps the message HONEST when
        both ends were subtracted."""
        from fused_memory.services.memory_service import _unresolvable_reason

        reason = _unresolvable_reason(
            (),
            cited=frozenset(),
            pool=frozenset({Referent(number='3074'), Referent(number='3075')}),
            endpoint=Referent(number='3074'),
            other_endpoint=Referent(number='3075'),
            ambiguous=frozenset(), source='derived',
        )

        assert 'already' in reason


    def test_a_corroborated_endpoint_says_so_rather_than_reporting_an_absence(self):
        """"The fact names the node it landed on" is the reason an operator and
        leaf eta need; it cannot be inferred from the pool, which the guard
        deliberately empties."""
        from fused_memory.services.memory_service import _unresolvable_reason

        reason = _unresolvable_reason(
            (),
            cited=frozenset({Referent(number='2500')}),
            pool=frozenset(),
            endpoint=Referent(number='2500'),
            other_endpoint=Referent(number='3668'),
            ambiguous=frozenset(), source='derived',
        )

        assert 'Task 2500' in reason
        assert 'cites' in reason
        # The self-loop branch must not win: BOTH conditions hold here, and
        # corroboration is the more specific — and the actionable — one.
        assert 'self-loop' not in reason


class TestCorroboratedEndpointIsNeverRepointed:
    """A fact that names the node its edge landed on is evidence FOR it.

    Regression for the reviewed defect: the membership arm marked such a finding
    `resolvable=True` and nominated the write's ambient task as the repair
    target, so leaf eta would have repointed a CORRECT edge — the very
    misattribution this PRD exists to prevent.
    """

    @pytest.mark.asyncio
    async def test_the_metadata_source_shape_is_recorded_but_never_resolvable(
        self, service,
    ):
        """referents={3668} from the ambient task_id; the edge is about 2500."""
        result = _episode(
            edges=[_edge('e1', fact='Task 2500 was completed by the merge worker',
                         source='n-2500', target='n-worker')],
            nodes=[MockNode(name='Task 2500', uuid='n-2500'),
                   MockNode(name='merge worker', uuid='n-worker')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3668'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        assert finding.endpoint_referent == Referent(number='2500')
        # The three fields leaf eta reads before acting. All fail-closed.
        assert finding.resolvable is False
        assert finding.intended_referent is None
        assert finding.new_endpoint_uuid is None
        assert 'Task 2500' in finding.reason
        assert stats.unresolvable_findings == 1
        assert service._referent_finding_counts['unresolvable'] == 1
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_it_holds_when_the_fact_also_names_the_ambient_task(self, service):
        """The same write one sentence longer — the shape that slipped past the
        guard while it sat behind the intersection short-circuit.

        The fact cites BOTH `Task 2500` (the endpoint) and `Task 3668` (the
        declared referent), so `cited & referents` is non-empty. An
        intersection-first order returned `Task 3668` as a sole surviving
        candidate and emitted `resolvable=True` with `new_endpoint_uuid='n-3668'`
        — a destructive-edge-surgery instruction against a fact that literally
        asserts the edge is about Task 2500. Driven end-to-end rather than
        through `_candidate_targets` alone, because the defect was invisible to
        the unit test that pinned an arm-unreachable input.
        """
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=_rows('n-3668'),
        )
        result = _episode(
            edges=[_edge('e1',
                         fact=('Task 2500 was completed as part of task 3668 '
                               'by the merge worker'),
                         source='n-2500', target='n-worker')],
            nodes=[MockNode(name='Task 2500', uuid='n-2500'),
                   MockNode(name='merge worker', uuid='n-worker')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3668'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        assert finding.endpoint_referent == Referent(number='2500')
        assert finding.resolvable is False
        assert finding.intended_referent is None
        assert finding.new_endpoint_uuid is None
        assert 'Task 2500' in finding.reason
        assert 'cites' in finding.reason
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_no_node_lookup_is_issued_for_a_corroborated_finding(self, service):
        """`intended_referent is None` short-circuits the second pass, so the
        corroborated row costs ZERO extra queries inside the identity lock."""
        result = _episode(
            edges=[_edge('e1', fact='Task 2500 was completed by the merge worker',
                         source='n-2500', target='n-worker')],
            nodes=[MockNode(name='Task 2500', uuid='n-2500'),
                   MockNode(name='merge worker', uuid='n-worker')],
        )

        await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3668'),),
        )

        service.graphiti.get_nodes_by_exact_name.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_uncorroborated_endpoint_on_the_same_shape_still_resolves(
        self, service,
    ):
        """The guard is narrow: it fires only when the fact cites the ENDPOINT.

        Same referent set, same arm — but the fact names a DIFFERENT task than
        the node the edge landed on, so on a ``'derived'`` declaration the
        declared-set fallback still supplies the repair target. This is the
        five-PRD-case signature. The ``'metadata'`` counterpart, where the
        fallback is vetoed instead, is
        :class:`TestTheMetadataSourceVetoesTheDeclaredSetFallback`.
        """
        result = _episode(
            edges=[_edge('e1', fact='Task 2500 was completed by the merge worker',
                         source='n-2501', target='n-worker')],
            nodes=[MockNode(name='Task 2501', uuid='n-2501'),
                   MockNode(name='merge worker', uuid='n-worker')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3668'),),
            referent_source='derived',
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].resolvable is True
        assert stats.findings[0].intended_referent == Referent(number='3668')


class TestTheMetadataSourceVetoesTheDeclaredSetFallback:
    """PRD: an ambient ``metadata['task_id']`` declaration is not evidence
    about which node any particular edge belongs on.

    ``resolve_referents`` ranks ``metadata['task_id']`` ABOVE the content-derived
    scan and names the resulting mismatch as deliberately NOT a conflict ("An
    agent working on task 3668 legitimately writes memories about Task 2500").
    The corroboration guard closes only the subset of that shape where the fact
    happens to NAME the endpoint; an LLM paraphrase restating no task number at
    all is the routine extraction outcome, and on it the guard cannot fire.
    """

    @pytest.mark.asyncio
    async def test_a_paraphrased_fact_on_a_metadata_write_is_recorded_not_repaired(
        self, service,
    ):
        """The dominant legitimate write shape must NOT become a repair.

        Agent dispatched on task 3668 writes about Task 2500; the extraction
        lands the edge (correctly) on the Task 2500 node with a paraphrased fact
        naming no number. Membership fires — 2500 is not declared — but there is
        nothing to repoint TO, so the finding is recorded with a reason.
        """
        result = _episode(
            edges=[_edge('e1', fact='the merge worker completed it',
                         source='n-2500', target='n-worker')],
            nodes=[MockNode(name='Task 2500', uuid='n-2500'),
                   MockNode(name='merge worker', uuid='n-worker')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3668'),),
            content='Task 2500 was completed by the merge worker',
            referent_source='metadata',
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        assert finding.resolvable is False
        assert finding.intended_referent is None
        assert 'ambient' in finding.reason and 'recorded, not repaired' in finding.reason
        # No node lookup is issued for an unresolvable finding.
        service.graphiti.get_nodes_by_exact_name.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_fact_citing_a_declared_referent_still_resolves_on_metadata(
        self, service,
    ):
        """The veto suppresses only the whole-declared-set FALLBACK.

        ``cited & referents`` is per-EDGE evidence — the fact itself names a
        declared referent — and survives on every source.
        """
        result = _episode(
            edges=[_edge('e1', fact='Task 3668 superseded the earlier plan',
                         source='n-2500', target='n-plan')],
            nodes=[MockNode(name='Task 2500', uuid='n-2500'),
                   MockNode(name='earlier plan', uuid='n-plan')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3668'),),
            referent_source='metadata',
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].resolvable is True
        assert stats.findings[0].intended_referent == Referent(number='3668')


class TestTheAmbiguousScanBoundaryRow:
    """PRD boundary row: "Ambiguous scan | ref routed to ``.ambiguous``;
    treated as undeclared; recorded, not guessed".

    Leaf epsilon drops ``.ambiguous`` from the wire on purpose, so zeta
    RE-DERIVES it from ``content``; without that, an ambiguous endpoint is
    indistinguishable from a genuine conflation and gets a destructive repair
    instruction instead of being recorded and left alone.
    """

    @pytest.mark.asyncio
    async def test_an_ambiguous_endpoint_is_recorded_but_never_resolvable(
        self, service,
    ):
        content = (
            'Task 3127 was reconciled; see reify:3127 for the mirror. '
            'Task 3200 tracks it.'
        )
        result = _episode(
            edges=[_edge('e1', fact='the mirror was reconciled',
                         source='n-3127', target='n-mirror')],
            nodes=[MockNode(name='Task 3127', uuid='n-3127'),
                   MockNode(name='mirror', uuid='n-mirror')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3200'),),
            content=content,
            referent_source='derived',
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        assert finding.endpoint_referent == Referent(number='3127')
        assert finding.resolvable is False
        assert finding.intended_referent is None
        assert 'AMBIGUOUS' in finding.reason
        assert 'recorded, not guessed at' in finding.reason
        service.graphiti.get_nodes_by_exact_name.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_self_qualified_ambiguous_spelling_still_vetoes(self, service):
        """The ambiguity set goes through `local_referent` like the endpoint
        parse does, so a self-qualified endpoint node name compares equal to the
        locally-classified ambiguous mention rather than sneaking past the veto.
        """
        content = (
            'Task 3127 was reconciled; see reify:3127 for the mirror. '
            'Task 3200 tracks it.'
        )
        result = _episode(
            edges=[_edge('e1', fact='the mirror was reconciled',
                         source='n-3127', target='n-mirror')],
            nodes=[MockNode(name='dark_factory:3127', uuid='n-3127'),
                   MockNode(name='mirror', uuid='n-mirror')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3200'),),
            content=content,
            referent_source='derived',
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].resolvable is False
        assert 'AMBIGUOUS' in stats.findings[0].reason

    @pytest.mark.asyncio
    async def test_an_unambiguous_endpoint_on_the_same_content_still_resolves(
        self, service,
    ):
        """The veto is NARROW: it fires only for the contested number.

        Same content, same declaration — but the edge landed on Task 3129, which
        the content never contested, so the row remains a repairable conflation.
        """
        content = (
            'Task 3127 was reconciled; see reify:3127 for the mirror. '
            'Task 3200 tracks it.'
        )
        result = _episode(
            edges=[_edge('e1', fact='the deploy pipeline was retried',
                         source='n-3129', target='n-x')],
            nodes=[MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory',
            referents=(Referent(number='3200'),),
            content=content,
            referent_source='derived',
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].resolvable is True
        assert stats.findings[0].intended_referent == Referent(number='3200')


def _rows(*uuids) -> list[dict]:
    return [{'uuid': u, 'name': 'Task 3127', 'summary': None, 'labels': []}
            for u in uuids]


def _one_membership_finding_episode() -> MockAddEpisodeResult:
    """One edge whose source landed on Task 3129 while the write declared 3127."""
    return _episode(
        edges=[_edge('e1', fact='the deploy pipeline was retried',
                     source='n-3129', target='n-x')],
        nodes=[MockNode(name='Task 3129', uuid='n-3129'),
               MockNode(name='deploy pipeline', uuid='n-x')],
    )


class TestNewEndpointUuidResolution:
    """Read-only, lazy, and never issued on the clean path."""

    @pytest.mark.asyncio
    async def test_an_existing_node_resolves_to_its_uuid(self, service):
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=_rows('n-3127'),
        )

        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings[0].new_endpoint_uuid == 'n-3127'
        service.graphiti.get_nodes_by_exact_name.assert_awaited_once_with(
            'Task 3127', group_id='dark_factory',
        )

    @pytest.mark.asyncio
    async def test_an_absent_node_is_none_but_still_resolvable(self, service):
        """Absence must not be confused with unrepairability — eta MINTS it via
        `ensure_entity_node`."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])

        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings[0].new_endpoint_uuid is None
        assert stats.findings[0].resolvable is True

    @pytest.mark.asyncio
    async def test_a_duplicate_name_group_is_none_but_still_resolvable(self, service):
        """The PRD measured 38 live name keys carrying more than one node. zeta
        never picks a survivor from such a group — that would pre-empt the
        identity-lock-held collapse `_resolve_or_create_entity` performs, which
        eta's `ensure_entity_node` reaches anyway."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=_rows('n-3127-a', 'n-3127-b'),
        )

        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings[0].new_endpoint_uuid is None
        assert stats.findings[0].resolvable is True

    @pytest.mark.asyncio
    async def test_the_clean_path_issues_zero_queries(self, service):
        """What keeps the common ~99.8% case free of an extra round-trip inside
        the per-group identity lock."""
        result = _episode(
            edges=[_edge('e1', source='n-3127', target='n-x')],
            nodes=[MockNode(name='Task 3127', uuid='n-3127'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.findings == []
        service.graphiti.get_nodes_by_exact_name.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_unresolvable_finding_triggers_no_lookup(self, service):
        """There is no intended referent to resolve."""
        result = _episode(
            edges=[_edge('e1', fact='Task 2519 was completed',
                         source='n-2519', target='n-2520')],
            nodes=[MockNode(name='Task 2519', uuid='n-2519'),
                   MockNode(name='Task 2520', uuid='n-2520')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='2519'),),
        )

        assert stats.findings[0].resolvable is False
        service.graphiti.get_nodes_by_exact_name.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_lookup_per_distinct_intended_referent(self, service):
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=_rows('n-3127'),
        )
        result = _episode(
            edges=[_edge('e1', source='n-3129', target='n-x'),
                   _edge('e2', source='n-3129', target='n-y'),
                   _edge('e3', source='n-y', target='n-3129')],
            nodes=[MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='deploy pipeline', uuid='n-x'),
                   MockNode(name='merge lane', uuid='n-y')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert len(stats.findings) == 3
        assert {f.new_endpoint_uuid for f in stats.findings} == {'n-3127'}
        assert service.graphiti.get_nodes_by_exact_name.await_count == 1

    @pytest.mark.asyncio
    async def test_a_lookup_failure_never_loses_the_finding(self, service):
        """Detection is the primary result; the uuid is an audit convenience."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            side_effect=RuntimeError('falkor down'),
        )

        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert len(stats.findings) == 1
        assert stats.findings[0].new_endpoint_uuid is None
        assert stats.findings[0].resolvable is True

    @pytest.mark.asyncio
    async def test_the_lookup_is_the_only_backend_call_zeta_ever_makes(self, service):
        """`get_nodes_by_exact_name` is documented `ro_query`-only — the one
        primitive that answers the question with no side effect."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=_rows('n-3127'),
        )

        await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert_never_repaired(service)
        service.graphiti._resolve_or_create_entity.assert_not_awaited()


def _mixed_findings_episode() -> MockAddEpisodeResult:
    """One membership finding (e1) and one pairing finding (e2)."""
    return _episode(
        edges=[_edge('e1', fact='the deploy pipeline was retried',
                     source='n-3129', target='n-x'),
               _edge('e2', fact='Task 3075 blocks the merge lane',
                     source='n-3074', target='n-lane')],
        nodes=[MockNode(name='Task 3129', uuid='n-3129'),
               MockNode(name='deploy pipeline', uuid='n-x'),
               MockNode(name='Task 3074', uuid='n-3074'),
               MockNode(name='merge lane', uuid='n-lane')],
    )


_MIXED_REFERENTS = (Referent(number='3074'), Referent(number='3075'))


class TestReferentFindingCounters:
    """INV-2's process-lifetime surface — leaf iota's read side.

    Mirrors `referent_source_counts()` line for line: every bucket exists from
    construction, the vocabulary is closed and keyed off the constant, and the
    accessor returns a copy of monotonic totals a reader samples and differences.
    """

    def test_every_bucket_exists_from_construction(self, service):
        """A reader never has to distinguish "zero" from "absent".

        Keyed off the CONSTANTS, not a literal, so the bucket vocabulary lives
        at ONE site (INV-5) — a fourth bucket has to be REGISTERED rather than
        spelled here and at the construction site and hoped to match.
        """
        assert service.referent_finding_counts() == dict.fromkeys(
            (*REFERENT_CHECKS, *REFERENT_FINDING_AXES), 0,
        )

    def test_the_axes_are_the_two_orthogonal_ones_not_checks(self):
        """The axis vocabulary is closed too, and its members answer questions
        ABOUT a finding rather than naming which check produced it."""
        assert REFERENT_FINDING_AXES == ('unresolvable', 'corroborated')
        assert not set(REFERENT_FINDING_AXES) & set(REFERENT_CHECKS)

    def test_the_counter_exists_even_with_no_write_journal(self, service):
        """Unconditional construction: the escape must not go dark in exactly
        the degraded configuration where a finding storm is least likely to be
        noticed any other way."""
        assert service._write_journal is None
        assert service.referent_finding_counts()

    @pytest.mark.asyncio
    async def test_the_buckets_are_monotonic_across_episodes(self, service):
        for expected in (1, 2):
            await service._verify_episode_referents(
                _mixed_findings_episode(), group_id='dark_factory',
                referents=_MIXED_REFERENTS,
            )
            counts = service.referent_finding_counts()
            assert counts['set-membership'] == expected
            assert counts['per-edge-pairing'] == expected

    @pytest.mark.asyncio
    async def test_check_and_unresolvable_are_orthogonal_axes_not_a_partition(
        self, service,
    ):
        """Which check fired and whether it can be acted on are independent, so
        the buckets intentionally do not sum to the finding total."""
        result = _episode(
            edges=[_edge('e1', fact='Task 2519 was completed',
                         source='n-2519', target='n-2520')],
            nodes=[MockNode(name='Task 2519', uuid='n-2519'),
                   MockNode(name='Task 2520', uuid='n-2520')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='2519'),),
        )

        assert len(stats.findings) == 1
        counts = service.referent_finding_counts()
        assert counts['set-membership'] == 1
        assert counts['unresolvable'] == 1

    @pytest.mark.asyncio
    async def test_a_clean_episode_leaves_every_bucket_untouched(self, service):
        result = _episode(
            edges=[_edge('e1', source='n-3127', target='n-x')],
            nodes=[MockNode(name='Task 3127', uuid='n-3127'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert set(service.referent_finding_counts().values()) == {0}

    @pytest.mark.asyncio
    async def test_a_corroborated_finding_moves_the_new_axis_too(self, service):
        """ALONGSIDE the check bucket, never instead of it.

        S1's fix is that leaf iota can SUBTRACT corroborated rows from the
        membership rate — which needs the denominator still there. Folding them
        out of `set-membership` instead would silently change what iota's
        existing rate means, and would break the "buckets are orthogonal axes,
        not a partition" contract the counter documents.
        """
        stats = await service._verify_episode_referents(
            _corroborated_membership_episode(), group_id='dark_factory',
            referents=(Referent(number='3668'),),
        )

        assert [f.corroborated for f in stats.findings] == [True]
        counts = service.referent_finding_counts()
        assert counts['corroborated'] == 1
        assert counts['set-membership'] == 1
        assert counts['unresolvable'] == 1
        assert counts['per-edge-pairing'] == 0

    @pytest.mark.asyncio
    async def test_a_genuine_finding_leaves_the_corroborated_bucket_at_zero(
        self, service,
    ):
        """The bucket covers the `endpoint in cited` veto ONLY. A misattached
        edge nothing corroborates is the defect the pass exists to catch, and
        must stay fully visible in iota's rate."""
        await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        counts = service.referent_finding_counts()
        assert counts['set-membership'] == 1
        assert counts['corroborated'] == 0

    def test_the_accessor_returns_a_copy(self, service):
        service.referent_finding_counts()['set-membership'] = 99
        service.referent_finding_counts()['corroborated'] = 99

        assert service.referent_finding_counts()['set-membership'] == 0
        assert service.referent_finding_counts()['corroborated'] == 0


class TestReferentFindingOperatorLog:
    """INV-2: a structured record, NOT the logger.debug-only shape the task
    calls out as unacceptable. The log is the OPERATOR surface — eta reads
    `ReferentStats.findings` and iota reads the counters, so nothing parses it.
    """

    @pytest.mark.asyncio
    async def test_a_finding_is_emitted_at_warning_with_its_structured_payload(
        self, service, caplog,
    ):
        with caplog.at_level(logging.WARNING,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _one_membership_finding_episode(), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        emitted = [r.getMessage() for r in caplog.records
                   if r.levelno == logging.WARNING]
        assert emitted
        payload = '\n'.join(emitted)
        assert 'e1' in payload
        assert 'n-3129' in payload
        assert 'Task 3129' in payload
        assert 'set-membership' in payload

    @pytest.mark.asyncio
    async def test_the_finding_is_reported_at_warning_and_not_at_debug(
        self, service, caplog,
    ):
        """BOTH halves of the claim, inside ONE debug-level capture.

        The shape this replaces (`test_no_finding_is_reported_at_debug_only`)
        was NEGATIVE-ONLY: capturing at DEBUG, it asserted merely that no DEBUG
        record named the check. That is trivially true of a `logger.debug` this
        pass never makes, so it passed unchanged with the `logger.warning`
        emission deleted outright — precisely the regression its name claimed to
        guard. Verified by mutation before this rewrite landed: with the
        emission removed the old test still passed and this one fails. Do not
        restore the negative-only shape; "not at DEBUG" means nothing unless it
        sits beside "and IS at WARNING".

        The fixture is deliberately NON-CORROBORATED (its fact cites nothing),
        so the INFO demotion does not apply to it, and it produces ONE finding,
        so the per-episode warning cap does not either. Both would otherwise
        change the expected level for reasons that have nothing to do with what
        this test pins.
        """
        with caplog.at_level(logging.DEBUG,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _one_membership_finding_episode(), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        emitted = _records_for(caplog, 'e1')
        assert [r.levelno for r in emitted] == [logging.WARNING]
        assert 'set-membership' in emitted[0].getMessage()

    @pytest.mark.asyncio
    async def test_a_clean_episode_logs_no_warning(self, service, caplog):
        result = _episode(
            edges=[_edge('e1', source='n-3127', target='n-x')],
            nodes=[MockNode(name='Task 3127', uuid='n-3127'),
                   MockNode(name='deploy pipeline', uuid='n-x')],
        )

        with caplog.at_level(logging.WARNING,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                result, group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


class TestPerEdgeFactScanIsLazy:
    """The per-edge fact scan runs only when an endpoint IS a task label.

    esc-3671-1's efficiency item. The scan's only consumers — the corroboration
    veto, the pairing arm's `cited_declared` guard and `_candidate_pool`'s
    intersection — are reached exclusively from the endpoint loop, and that loop
    `continue`s immediately for an endpoint whose name is not a canonical task
    label, which is the overwhelming majority of edges. Scanning every fact up
    front therefore spends a regex pass per edge to produce a value nothing
    reads, and it spends it SERIALIZED inside the per-group
    `_identity_lock_for` critical section that other same-group writes queue
    behind.

    Deferring must not cost the "scanned ONCE per edge, not once per endpoint"
    property the eager comment defends, so the two-task-endpoint case is pinned
    explicitly: laziness is memoized per edge, not recomputed per end.
    """

    @staticmethod
    def _spy(monkeypatch) -> list[str]:
        """Every text `_verify_episode_referents` hands `scan_content`.

        Patches the name as imported INTO `memory_service` (line 93), which is
        the binding the pass actually calls, and delegates to the real scanner
        so the findings under test are the production ones.
        """
        scanned: list[str] = []
        real = memory_service_module.scan_content

        def _recording(text, **kwargs):
            scanned.append(text)
            return real(text, **kwargs)

        monkeypatch.setattr(memory_service_module, 'scan_content', _recording)
        return scanned

    @pytest.mark.asyncio
    async def test_an_edge_with_no_task_endpoint_is_never_scanned(
        self, service, monkeypatch,
    ):
        """The ~99% shape: two plain entity endpoints, nothing to check.

        The fact deliberately DOES cite a task, so an eager implementation
        really does pay for the scan here. Laziness is decided by the
        ENDPOINTS, not by whether the fact happens to be scannable.
        """
        scanned = self._spy(monkeypatch)
        result = _episode(
            edges=[_edge('e1', fact='Task 3129 blocked the deploy pipeline',
                         source='n-x', target='n-y')],
            nodes=[MockNode(name='deploy pipeline', uuid='n-x'),
                   MockNode(name='merge lane', uuid='n-y')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        # The edge was WALKED — laziness bounds the scan, never the coverage.
        assert stats.edges_scanned == 1
        assert stats.findings == []
        assert scanned == []

    @pytest.mark.asyncio
    async def test_an_endpointless_edge_is_never_scanned(
        self, service, monkeypatch,
    ):
        """The other half of the same shape: this episode's result does not
        NAME either endpoint, so neither can be parsed at all."""
        scanned = self._spy(monkeypatch)
        result = _episode(
            edges=[_edge('e1', fact='Task 3129 blocked the deploy pipeline',
                         source='n-absent', target='n-also-absent')],
            nodes=[MockNode(name='Task 3129', uuid='n-3129')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.edges_scanned == 1
        assert stats.endpoints_unresolved == 2
        assert scanned == []

    @pytest.mark.asyncio
    async def test_an_edge_with_two_task_endpoints_is_scanned_exactly_once(
        self, service, monkeypatch,
    ):
        """ONCE PER EDGE, NOT ONCE PER ENDPOINT — the property the eager
        placement bought and the deferral must not give back. Both ends parse,
        so both reach the citation rules; one scan must serve them."""
        scanned = self._spy(monkeypatch)
        result = _episode(
            edges=[_edge('e1', fact='Task 3128 supersedes Task 3129',
                         source='n-3128', target='n-3129')],
            nodes=[MockNode(name='Task 3128', uuid='n-3128'),
                   MockNode(name='Task 3129', uuid='n-3129')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert stats.endpoints_checked == 2
        assert scanned == ['Task 3128 supersedes Task 3129']

    @pytest.mark.asyncio
    async def test_the_deferral_changes_no_finding(self, service, monkeypatch):
        """Byte-for-byte equality with the eager behaviour on the mixed
        membership + pairing fixture, so the deferral is provably an
        efficiency change and not a semantic one.

        The expected values were captured from the eager implementation before
        the deferral landed; `reason` is pinned in full because it is the field
        that would drift first if `cited` were ever computed from the wrong
        text or at the wrong time.
        """
        scanned = self._spy(monkeypatch)

        stats = await service._verify_episode_referents(
            _mixed_findings_episode(), group_id='dark_factory',
            referents=_MIXED_REFERENTS,
        )

        # One scan per edge: each of the two edges has a task-labelled end.
        assert scanned == ['the deploy pipeline was retried',
                           'Task 3075 blocks the merge lane']

        membership, pairing = stats.findings
        assert (membership.edge_uuid, membership.which_end, membership.check,
                membership.old_endpoint_uuid, membership.old_endpoint_name,
                membership.endpoint_referent, membership.referent_set,
                membership.intended_referent, membership.resolvable) == (
            'e1', 'source', 'set-membership', 'n-3129', 'Task 3129',
            Referent(number='3129'), ('Task 3074', 'Task 3075'), None, False)
        assert membership.reason == (
            "more than one candidate target survives (['Task 3074', "
            "'Task 3075']) and the edge fact does not discriminate between "
            'them; recorded, not guessed at')

        assert (pairing.edge_uuid, pairing.which_end, pairing.check,
                pairing.old_endpoint_uuid, pairing.old_endpoint_name,
                pairing.endpoint_referent, pairing.referent_set,
                pairing.intended_referent, pairing.resolvable,
                pairing.reason) == (
            'e2', 'source', 'per-edge-pairing', 'n-3074', 'Task 3074',
            Referent(number='3074'), ('Task 3074', 'Task 3075'),
            Referent(number='3075'), True, '')


def _corroborated_membership_episode() -> MockAddEpisodeResult:
    """esc-3671-3's reachable shape, verbatim from `_candidate_pool`'s docstring.

    The edge hangs off the `Task 2500` node and its fact names BOTH that node
    and the declared referent 3668 — "the same legitimate ambient-task write one
    sentence longer". Membership fires (2500 is not declared), and the
    corroboration veto empties the pool because the fact cites the very node the
    edge landed on.
    """
    return _episode(
        edges=[_edge('e1',
                     fact='Task 2500 was completed as part of task 3668 by '
                          'the merge worker',
                     source='n-2500', target='n-worker')],
        nodes=[MockNode(name='Task 2500', uuid='n-2500'),
               MockNode(name='merge worker', uuid='n-worker')],
    )


class TestFindingCarriesTheCitationsThatDecidedIt:
    """S3 (esc-3671-3): a finding must carry the evidence that made it fire.

    `referent_set` is the DECLARED set. On the PAIRING arm the discriminating
    input is not that set but the edge fact's own citations — the arm fires
    precisely on `cited_declared and endpoint_referent not in cited` — and
    before this field those citations were recoverable nowhere on the record.
    An operator reading a pairing warning could see which referents the write
    declared but not which ones the fact named, i.e. not the input that decided
    the verdict.
    """

    def test_the_field_defaults_to_empty_and_is_a_tuple(self):
        """A TUPLE for the reason `referent_set` is one: `frozen=True` blocks
        rebinding only, so a list would leave `finding.cited.append(...)` open
        on evidence for destructive edge surgery."""
        assert _finding().cited == ()
        assert isinstance(_finding().cited, tuple)
        assert isinstance(_finding(cited=('Task 3075',)).cited, tuple)

    def test_to_dict_renders_cited_as_a_json_safe_list(self):
        payload = _finding(cited=('Task 3075', 'Task 3668')).to_dict()

        assert payload['cited'] == ['Task 3075', 'Task 3668']
        assert json.loads(json.dumps(payload)) == payload
        # The self-maintaining key-set contract: a field the payload omits is a
        # field the operator log silently drops.
        assert set(payload) == {f.name for f in dataclasses.fields(ReferentFinding)}

    @pytest.mark.asyncio
    async def test_a_pairing_finding_carries_the_facts_citations(self, service):
        """The arm whose evidence was missing. `cited == ('Task 3075',)` is
        exactly what made `cited_declared` non-empty and fired the check."""
        stats = await service._verify_episode_referents(
            _mixed_findings_episode(), group_id='dark_factory',
            referents=_MIXED_REFERENTS,
        )

        pairing = [f for f in stats.findings if f.check == 'per-edge-pairing']
        assert [f.cited for f in pairing] == [('Task 3075',)]

    @pytest.mark.asyncio
    async def test_a_membership_finding_carries_them_too(self, service):
        """Recorded on BOTH arms, not just the one that reads them, so a reader
        never has to know which arm carries which evidence."""
        stats = await service._verify_episode_referents(
            _corroborated_membership_episode(), group_id='dark_factory',
            referents=(Referent(number='3668'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        assert finding.endpoint_referent == Referent(number='2500')
        assert finding.cited == ('Task 2500', 'Task 3668')
        # And the declared set stays what it always was — the two fields are
        # different questions, which is the whole point of recording both.
        assert finding.referent_set == ('Task 3668',)

    @pytest.mark.asyncio
    async def test_a_fact_citing_nothing_records_no_citations(self, service):
        """Empty is a real answer here, not an absence: an uninformative fact is
        why the pairing arm's `cited_declared` guard refuses to fire."""
        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings[0].cited == ()

    @pytest.mark.asyncio
    async def test_the_citations_render_in_a_deterministic_order(self, service):
        """`cited` is a FROZENSET internally, whose iteration order is not
        stable across processes under hash randomization. Sorted for the same
        reason `_candidate_targets` sorts its survivors: a finding must be
        stable across runs and diffable in eta's audit."""
        result = _episode(
            edges=[_edge('e1', fact='Task 3129 supersedes Task 3128',
                         source='n-3129', target='n-3128')],
            nodes=[MockNode(name='Task 3129', uuid='n-3129'),
                   MockNode(name='Task 3128', uuid='n-3128')],
        )

        stats = await service._verify_episode_referents(
            result, group_id='dark_factory', referents=(Referent(number='3127'),),
        )

        assert len(stats.findings) == 2
        # Both ends of one edge share ONE scan, so both carry the same rendering
        # — and it follows the fact's numeric order, not its textual one.
        assert {f.cited for f in stats.findings} == {('Task 3128', 'Task 3129')}


class TestCorroboratedIsDerivedFromTheRecordedEvidence:
    """S1 (esc-3671-3): the READ side of `_candidate_pool`'s corroboration veto.

    A fact that names the very node its edge landed on is the strongest
    available evidence the attachment is CORRECT, so `_candidate_pool` empties
    the pool for it (`if endpoint in cited: return frozenset()`). The finding is
    still recorded — "recorded and left alone, never guessed at" — but it is a
    no-observable-defect row, and the counter and the operator log both need to
    be able to tell it apart from a genuine misattachment.

    DERIVED, never stored. A stored boolean set at the construction site would
    be a second site that must agree byte-for-byte with that guard — the INV-5
    lockstep duplication whose drift produced esc-3671-3's blocking bug, where a
    guard sat one position away from where its own rationale assumed. Deriving
    it from evidence the record now carries makes disagreement unrepresentable,
    and it is the discipline `ReferentStats` already documents for its own
    counts.
    """

    def test_it_is_true_exactly_when_the_fact_names_the_endpoint(self):
        assert _finding(
            endpoint_referent=Referent(number='2500'),
            cited=('Task 2500', 'Task 3668'),
        ).corroborated is True

        assert _finding(
            endpoint_referent=Referent(number='2500'),
            cited=('Task 3668',),
        ).corroborated is False

        assert _finding(
            endpoint_referent=Referent(number='2500'), cited=(),
        ).corroborated is False

    def test_it_compares_canonical_node_names_not_spellings(self):
        """`cited` renders through `Referent.node_name`, so a foreign-qualified
        citation is a DIFFERENT referent and must not corroborate a local
        endpoint — the same discrimination `local_referent` exists to make."""
        assert _finding(
            endpoint_referent=Referent(number='2500'),
            cited=('other_project:2500',),
        ).corroborated is False

    def test_it_is_not_a_key_in_the_payload(self):
        """The key set is contractually the DATACLASS FIELD NAMES, and a derived
        property is not a field. Nothing is lost: the payload already carries
        `cited` and `endpoint_referent`, so an operator reads the corroboration
        straight off the log line."""
        payload = _finding(cited=('Task 2520',)).to_dict()

        assert 'corroborated' not in payload
        assert set(payload) == {f.name for f in dataclasses.fields(ReferentFinding)}

    @pytest.mark.asyncio
    async def test_it_agrees_with_the_pool_veto_on_the_reachable_shape(
        self, service,
    ):
        """esc-3671-3's shape, driven end-to-end rather than hand-built.

        The property and the veto must answer the same question about the same
        edge, which is the entire reason it is derived from `cited` rather than
        stored beside it.
        """
        from fused_memory.services.memory_service import _candidate_pool

        stats = await service._verify_episode_referents(
            _corroborated_membership_episode(), group_id='dark_factory',
            referents=(Referent(number='3668'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.check == 'set-membership'
        assert finding.corroborated is True
        # The veto's own verdict on the same inputs: an EMPTY pool.
        assert _candidate_pool(
            referents=frozenset({Referent(number='3668')}),
            cited=frozenset({Referent(number='2500'), Referent(number='3668')}),
            endpoint=Referent(number='2500'),
            ambiguous=frozenset(),
            source='derived',
        ) == frozenset()
        # Recorded and left alone — the finding survives, it just cannot be
        # acted on.
        assert finding.resolvable is False

    @pytest.mark.asyncio
    async def test_a_pairing_finding_can_never_be_corroborated(self, service):
        """Structural, not incidental: the pairing arm is reached only when
        `endpoint_referent not in cited`, so the property is False there by
        construction."""
        stats = await service._verify_episode_referents(
            _mixed_findings_episode(), group_id='dark_factory',
            referents=_MIXED_REFERENTS,
        )

        pairing = [f for f in stats.findings if f.check == 'per-edge-pairing']
        assert pairing
        assert [f.corroborated for f in pairing] == [False]

    @pytest.mark.asyncio
    async def test_a_genuinely_misattached_endpoint_is_not_corroborated(
        self, service,
    ):
        """The shape the whole pass exists to catch: the fact says nothing about
        the node the edge landed on, so nothing corroborates it and it resolves
        normally."""
        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings[0].corroborated is False
        assert stats.findings[0].resolvable is True


def _corroborated_and_genuine_episode() -> MockAddEpisodeResult:
    """One corroborated finding (e1) and one genuine misattachment (e2).

    Both fire SET MEMBERSHIP against the same declared set, so the only thing
    that distinguishes them is whether the edge's own fact names the node it
    landed on. That is what makes this the right fixture for a PER-FINDING
    policy: an episode-level rule cannot tell them apart.
    """
    return _episode(
        edges=[_edge('e1',
                     fact='Task 2500 was completed as part of task 3668 by '
                          'the merge worker',
                     source='n-2500', target='n-worker'),
               _edge('e2', fact='the deploy pipeline was retried',
                     source='n-3129', target='n-x')],
        nodes=[MockNode(name='Task 2500', uuid='n-2500'),
               MockNode(name='merge worker', uuid='n-worker'),
               MockNode(name='Task 3129', uuid='n-3129'),
               MockNode(name='deploy pipeline', uuid='n-x')],
    )


def _records_for(caplog, edge_uuid: str) -> list[logging.LogRecord]:
    """Every emitted finding line whose payload names *edge_uuid*."""
    return [r for r in caplog.records
            if f"'edge_uuid': '{edge_uuid}'" in r.getMessage()]


class TestCorroboratedFindingsAreNotOperatorWarnings:
    """S1 (esc-3671-3): the membership arm's counterpart of the discipline the
    pairing arm already keeps.

    `test_no_counter_moves_and_no_warning_is_emitted_for_that_shape` is the
    normative statement: a finding unactionable BY CONSTRUCTION must not raise
    an operator WARNING for an endpoint with no observable defect. The pairing
    arm achieves that by refusing to fire (`cited_declared` narrowing). The
    membership arm CANNOT refuse — a corroborated endpoint really is outside the
    declared set, and the PRD's postcondition is that it be RECORDED — so it
    keeps the same discipline the only other way available: it records the
    finding and demotes the line.

    DEMOTED, NOT DROPPED. The evidence still reaches the log, one level down;
    silently discarding it would be the fail-soft path INV-4 forbids.
    """

    @pytest.mark.asyncio
    async def test_a_corroborated_finding_raises_no_warning(self, service, caplog):
        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _corroborated_membership_episode(), group_id='dark_factory',
                referents=(Referent(number='3668'),),
            )

        assert [r for r in caplog.records
                if r.levelno >= logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_it_is_still_logged_one_level_down(self, service, caplog):
        """Demoted, never dropped: an operator who goes looking still finds the
        full structured payload."""
        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _corroborated_membership_episode(), group_id='dark_factory',
                referents=(Referent(number='3668'),),
            )

        emitted = _records_for(caplog, 'e1')
        assert [r.levelno for r in emitted] == [logging.INFO]
        message = emitted[0].getMessage()
        assert 'set-membership' in message
        assert 'Task 2500' in message

    @pytest.mark.asyncio
    async def test_the_return_value_keeps_the_finding_in_full(
        self, service, caplog,
    ):
        """The LOG is the operator surface; `ReferentStats.findings` is the
        machine surface leaf eta reads in-process. A log-level policy must never
        cost eta a finding."""
        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            stats = await service._verify_episode_referents(
                _corroborated_membership_episode(), group_id='dark_factory',
                referents=(Referent(number='3668'),),
            )

        assert len(stats.findings) == 1
        payload = stats.findings[0].to_dict()
        assert payload['edge_uuid'] == 'e1'
        assert payload['check'] == 'set-membership'
        assert payload['cited'] == ['Task 2500', 'Task 3668']
        assert set(payload) == {f.name for f in dataclasses.fields(ReferentFinding)}

    @pytest.mark.asyncio
    async def test_the_demotion_is_per_finding_not_per_episode(
        self, service, caplog,
    ):
        """A corroborated finding and a genuine misattachment in ONE episode
        must land on DIFFERENT levels. An episode-level rule — "this write looks
        like ambient-task noise, quiet it down" — would swallow the defect the
        pass exists to surface."""
        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            stats = await service._verify_episode_referents(
                _corroborated_and_genuine_episode(), group_id='dark_factory',
                referents=(Referent(number='3668'),),
            )

        assert [f.corroborated for f in stats.findings] == [True, False]
        assert [r.levelno for r in _records_for(caplog, 'e1')] == [logging.INFO]
        assert [r.levelno for r in _records_for(caplog, 'e2')] == [logging.WARNING]


def _warn_cap() -> int:
    """The per-episode warning cap, READ FROM THE CONSTANT.

    Every expectation below is computed from this rather than from a literal,
    so retuning a log-volume policy number can never red the suite. Imported
    locally, the idiom this file already uses for module-private symbols
    (`_candidate_targets`).
    """
    from fused_memory.services.memory_service import _REFERENT_FINDING_WARN_CAP

    return _REFERENT_FINDING_WARN_CAP


def _finding_storm_episode(
    warn_findings: int, corroborated_findings: int = 0,
) -> MockAddEpisodeResult:
    """*warn_findings* genuine findings plus *corroborated_findings* vetoed ones.

    The corroborated edges come FIRST, deliberately: a cap that counted every
    finding rather than every WARN-LEVEL one would spend part of its budget on
    them and visibly shortchange the genuine defects that follow.
    """
    edges = [_edge(f'c{i}',
                   fact=f'Task {8000 + i} was completed as part of task 3127 '
                        'by the merge worker',
                   source=f'n-{8000 + i}', target='n-worker')
             for i in range(corroborated_findings)]
    edges += [_edge(f'w{i}', fact='the deploy pipeline was retried',
                    source=f'n-{9000 + i}', target='n-x')
              for i in range(warn_findings)]
    nodes = [MockNode(name='deploy pipeline', uuid='n-x'),
             MockNode(name='merge worker', uuid='n-worker')]
    nodes += [MockNode(name=f'Task {8000 + i}', uuid=f'n-{8000 + i}')
              for i in range(corroborated_findings)]
    nodes += [MockNode(name=f'Task {9000 + i}', uuid=f'n-{9000 + i}')
              for i in range(warn_findings)]
    return _episode(edges=edges, nodes=nodes)


def _finding_lines(caplog, level: int) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.levelno == level
            and r.getMessage().startswith('Referent verification finding:')]


def _storm_lines(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if 'Referent verification finding storm' in r.getMessage()]


class TestTheOperatorWarningIsCappedPerEpisode:
    """A2 (esc-3671-1), the headline item: an unbounded per-finding WARNING.

    One `logger.warning` per finding, with no cap, no dedup and no warn-once. A
    single malformed episode can carry hundreds of edges, and `replay_from_store`
    re-writing a backlog multiplies that across every episode in the queue — so
    the surface that exists to make a defect VISIBLE is also the surface that
    can bury every other line an operator needs.

    THE CAP BOUNDS THE LOG, NEVER THE RECORD. `ReferentStats.findings` is what
    leaf eta reads in-process and the counters are what leaf iota samples;
    capping the operator surface must cost neither. And the truncation ANNOUNCES
    itself — a silently shortened log is a fail-soft path with nothing to hear
    it, which is what INV-4 forbids.
    """

    @pytest.mark.asyncio
    async def test_only_the_first_cap_findings_are_logged_individually(
        self, service, caplog,
    ):
        cap = _warn_cap()

        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _finding_storm_episode(cap + 5), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        assert len(_finding_lines(caplog, logging.WARNING)) == cap

    @pytest.mark.asyncio
    async def test_the_suppression_announces_itself_exactly_once(
        self, service, caplog,
    ):
        """The storm signal INV-4 asks for. The counter is the machine half of
        the escape; this line is the operator half."""
        cap, overflow = _warn_cap(), 5

        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _finding_storm_episode(cap + overflow), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        storm = _storm_lines(caplog)
        assert len(storm) == 1
        assert storm[0].levelno == logging.WARNING
        message = storm[0].getMessage()
        # How many were suppressed — matched as a PHRASE, not as a bare digit,
        # which any other number in the line would satisfy.
        assert f'{overflow} further finding' in message
        assert 'suppressed' in message
        # ...and the per-check totals for the episode, so the shape of the
        # storm is legible without the suppressed lines.
        assert f"'set-membership': {cap + overflow}" in message
        assert "'per-edge-pairing': 0" in message

    @pytest.mark.asyncio
    async def test_the_return_value_keeps_every_finding_in_full(
        self, service, caplog,
    ):
        """Leaf eta reads this in-process, inside the same critical section. A
        log-volume policy must never cost it a finding."""
        cap = _warn_cap()

        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            stats = await service._verify_episode_referents(
                _finding_storm_episode(cap + 5), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        assert len(stats.findings) == cap + 5
        field_names = {f.name for f in dataclasses.fields(ReferentFinding)}
        assert all(set(f.to_dict()) == field_names for f in stats.findings)
        assert {f.edge_uuid for f in stats.findings} == {
            f'w{i}' for i in range(cap + 5)
        }

    @pytest.mark.asyncio
    async def test_suppression_costs_no_counter_signal(self, service, caplog):
        """Leaf iota's rate must not silently deflate the moment an episode
        crosses a LOG-volume threshold."""
        cap = _warn_cap()

        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _finding_storm_episode(cap + 5), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        assert service.referent_finding_counts()['set-membership'] == cap + 5

    @pytest.mark.asyncio
    async def test_an_episode_under_the_cap_emits_no_aggregate_line(
        self, service, caplog,
    ):
        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _one_membership_finding_episode(), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        assert len(_finding_lines(caplog, logging.WARNING)) == 1
        assert _storm_lines(caplog) == []

    @pytest.mark.asyncio
    async def test_an_episode_exactly_at_the_cap_emits_no_aggregate_line(
        self, service, caplog,
    ):
        """The boundary: nothing was suppressed, so there is nothing to
        announce. An off-by-one here would page on every full-budget episode."""
        cap = _warn_cap()

        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _finding_storm_episode(cap), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        assert len(_finding_lines(caplog, logging.WARNING)) == cap
        assert _storm_lines(caplog) == []

    @pytest.mark.asyncio
    async def test_corroborated_findings_do_not_consume_the_warning_budget(
        self, service, caplog,
    ):
        """The budget is spent on findings that indicate a defect, not on the
        dominant legitimate ambient-task shape. The corroborated edges come
        first in this fixture, so a cap counting every finding would emit fewer
        than `cap` genuine warnings — and would announce a suppression that
        never happened."""
        cap = _warn_cap()

        with caplog.at_level(logging.INFO,
                             logger='fused_memory.services.memory_service'):
            await service._verify_episode_referents(
                _finding_storm_episode(cap, corroborated_findings=3),
                group_id='dark_factory', referents=(Referent(number='3127'),),
            )

        assert len(_finding_lines(caplog, logging.WARNING)) == cap
        assert len(_finding_lines(caplog, logging.INFO)) == 3
        assert _storm_lines(caplog) == []


def _count_payload_builds(monkeypatch) -> list[str]:
    """Record every `ReferentFinding.to_dict()` call, by edge uuid.

    A frozen dataclass freezes INSTANCES, not the class, so the method can be
    wrapped for the duration of one test.
    """
    built: list[str] = []
    original = ReferentFinding.to_dict

    def counting(self) -> dict:
        built.append(self.edge_uuid)
        return original(self)

    monkeypatch.setattr(ReferentFinding, 'to_dict', counting)
    return built


class TestTheFindingPayloadIsBuiltOnlyWhenTheLevelIsEnabled:
    """`%s` lazy formatting defers the string rendering, NEVER the `to_dict()`
    that produces its argument.

    So the dict is allocated per finding even where the record is thrown away,
    unless the emission is guarded by `isEnabledFor` — and that guard is what
    makes the deliberately UNCAPPED INFO half of the log (see
    `_REFERENT_FINDING_WARN_CAP`) free on a process that never asked for INFO.
    It matters here specifically because this loop runs serialized inside the
    per-group `_identity_lock_for` critical section every other same-group write
    queues behind, and because the demoted half is the DOMINANT shape.
    """

    @pytest.mark.asyncio
    async def test_a_disabled_level_builds_no_payload_at_all(
        self, service, caplog, monkeypatch,
    ):
        built = _count_payload_builds(monkeypatch)

        with caplog.at_level(logging.CRITICAL,
                             logger='fused_memory.services.memory_service'):
            stats = await service._verify_episode_referents(
                _one_membership_finding_episode(), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

        assert built == []
        assert _records_for(caplog, 'e1') == []
        # THE MACHINE SURFACES ARE UNTOUCHED by the log level — INV-2. Leaf eta
        # reads this return value in-process, and it must never depend on how a
        # handler happens to be configured.
        assert [f.edge_uuid for f in stats.findings] == ['e1']
        assert service.referent_finding_counts()['set-membership'] == 1

    @pytest.mark.asyncio
    async def test_the_demoted_half_costs_nothing_when_only_warning_is_on(
        self, service, caplog, monkeypatch,
    ):
        """The uncapped INFO half's whole defence: at the default level it does
        no per-finding work, while the warn half still emits in full."""
        built = _count_payload_builds(monkeypatch)

        with caplog.at_level(logging.WARNING,
                             logger='fused_memory.services.memory_service'):
            stats = await service._verify_episode_referents(
                _corroborated_and_genuine_episode(), group_id='dark_factory',
                referents=(Referent(number='3668'),),
            )

        corroborated = [f.edge_uuid for f in stats.findings if f.corroborated]
        assert corroborated == ['e1']
        assert built == ['e2']
        assert [r.levelno for r in _records_for(caplog, 'e2')] == [logging.WARNING]


class TestADegradedLookupIsDistinguishableFromAnAbsentNode:
    """S2 (esc-3671-3): "could not look" must not read as "not there".

    `_intended_endpoint_uuid` collapses ABSENT and DUPLICATE-NAME-GROUP to
    `new_endpoint_uuid=None` on purpose — leaf eta's `ensure_entity_node`
    resolves-or-mints and handles both identically. A transient backend error
    degrades to the same `None`, which is where the collapse stops being
    deliberate: eta reads that field at exactly one site,
    `minted=finding.new_endpoint_uuid is None`, so a FalkorDB blip books as
    minted=True telemetry — a made-up node reported where an unavailable lookup
    happened.

    Zeta's job is to PRODUCE the discriminator; acting on it is eta's, and is
    filed separately. Detection stays the primary result either way: the flag
    EXTENDS `test_a_lookup_failure_never_loses_the_finding` rather than
    replacing it, so the evidence still survives a backend that will not answer.
    """

    @pytest.mark.asyncio
    async def test_a_transient_error_is_recorded_without_losing_the_finding(
        self, service,
    ):
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            side_effect=RuntimeError('falkor down'),
        )

        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert len(stats.findings) == 1
        finding = stats.findings[0]
        assert finding.uuid_lookup_degraded is True
        # EXTENDS the existing contract, never replaces it: the uuid is still
        # None and the finding is still resolvable and still recorded in full.
        assert finding.new_endpoint_uuid is None
        assert finding.resolvable is True

    @pytest.mark.asyncio
    async def test_a_clean_lookup_finding_nothing_is_not_degraded(self, service):
        """ABSENT is a real answer the backend gave. eta mints; that is the
        documented collapse and it stays exactly as it was."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])

        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings[0].new_endpoint_uuid is None
        assert stats.findings[0].uuid_lookup_degraded is False

    @pytest.mark.asyncio
    async def test_a_duplicate_name_group_is_not_degraded_either(self, service):
        """The PRD measured 38 live name keys carrying more than one node.
        Picking a survivor is `_resolve_or_create_entity`'s job under the
        identity lock, so zeta declines — deliberately, not for lack of an
        answer."""
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=_rows('n-3127', 'n-3127-dup'),
        )

        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings[0].new_endpoint_uuid is None
        assert stats.findings[0].uuid_lookup_degraded is False

    @pytest.mark.asyncio
    async def test_a_successful_lookup_is_not_degraded(self, service):
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            return_value=_rows('n-3127'),
        )

        stats = await service._verify_episode_referents(
            _one_membership_finding_episode(), group_id='dark_factory',
            referents=(Referent(number='3127'),),
        )

        assert stats.findings[0].new_endpoint_uuid == 'n-3127'
        assert stats.findings[0].uuid_lookup_degraded is False

    def test_the_flag_defaults_false_and_the_payload_carries_it(self):
        """Fail-closed like `resolvable`: a record that never looked is not a
        record whose lookup was degraded."""
        assert _finding().uuid_lookup_degraded is False

        payload = _finding(uuid_lookup_degraded=True).to_dict()
        assert payload['uuid_lookup_degraded'] is True
        assert json.loads(json.dumps(payload)) == payload
        assert set(payload) == {f.name for f in dataclasses.fields(ReferentFinding)}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'exc_type', [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
    )
    async def test_the_lookup_never_swallows_cancellation(self, service, exc_type):
        """The `except (CancelledError, KeyboardInterrupt, SystemExit): raise`
        arm ahead of the broad handler, mirroring the sibling contract test for
        `_reconcile_episode_identity`.

        Pins EXISTING behaviour that nothing in this file pinned: a
        best-effort degradation that also swallowed cancellation would keep a
        shutting-down process inside an identity-lock critical section.
        """
        service.graphiti.get_nodes_by_exact_name = AsyncMock(
            side_effect=exc_type('interrupted'),
        )

        with pytest.raises(exc_type):
            await service._verify_episode_referents(
                _one_membership_finding_episode(), group_id='dark_factory',
                referents=(Referent(number='3127'),),
            )

