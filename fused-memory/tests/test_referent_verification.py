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

import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import (
    MockAddEpisodeResult,
    MockEdge,
    MockNode,
    install_identity_mocks,
)

from fused_memory.services.memory_service import (
    REFERENT_CHECKS,
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

    def test_is_keyword_only(self):
        """Positional construction of an eleven-field evidence record is how a
        field silently lands in the wrong slot."""
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
