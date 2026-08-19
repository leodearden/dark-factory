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
from _fm_helpers import install_identity_mocks

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
