"""The write-time referent REPAIR path (task 3672, PRD leaf eta of
plans/memory-referent-fidelity-prd.md).

Leaf zeta (task 3671) detects and records: it walks a committed episode's edges
and produces structured `ReferentFinding`s naming every edge END that landed on
a node the write was not about.  It performs no writes at all.  This leaf is the
only WRITER: it consumes zeta's `ReferentStats` inside the same identity-lock
critical section and repairs each resolvable finding with

    ensure_entity_node  ->  reassign_edge  ->  refresh_entity_summary

plus two harvested edge cases (a degenerate both-ends-on-one-node edge is
skipped WHOLE; a node this pass emptied and that parses as a canonical task
label is deleted), and the INV-4 consecutive-repair-streak escalation.

NEVER GUESS is the structural default, inherited from zeta: a finding whose
correct target could not be determined is RECORDED and LEFT ALONE.  `'failed'`
is a THIRD disposition, distinct from both — "we tried and the backend did not
cooperate" is an infrastructure signal, not a refusal to guess, and conflating
them would let a FalkorDB outage read as a scanner regression.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import install_identity_mocks
from test_referent_verification import _WRITE_PRIMITIVES, assert_never_repaired

from fused_memory.services.memory_service import (
    REFERENT_REPAIR_OUTCOMES,
    MemoryService,
    ReconcileStats,
    ReferentFinding,
    ReferentRepair,
    ReferentRepairStats,
    ReferentStats,
)
from fused_memory.utils.canonical_labels import Referent


def _repair(**overrides) -> ReferentRepair:
    """A minimally-valid repair record; overrides tune whichever field a test pins."""
    fields = {
        'edge_uuid': 'e1',
        'which_end': 'source',
        'outcome': 'repaired',
        'old_endpoint_uuid': 'n-3129',
        'check': 'set-membership',
    }
    fields.update(overrides)
    return ReferentRepair(**fields)


class TestReferentRepairRecordVocabulary:
    """INV-2: repairs are STRUCTURED RECORDS, not a log line."""

    def test_outcome_vocabulary_is_closed_and_exactly_the_four_dispositions(self):
        """The single normative site for "what happened to this finding"."""
        assert REFERENT_REPAIR_OUTCOMES == (
            'repaired', 'unrepairable', 'degenerate', 'failed',
        )

    def test_required_fields_construct_and_defaults_are_inert(self):
        record = _repair()

        assert record.edge_uuid == 'e1'
        assert record.which_end == 'source'
        assert record.outcome == 'repaired'
        assert record.old_endpoint_uuid == 'n-3129'
        assert record.check == 'set-membership'
        # Every optional field defaults to "nothing happened", so a record
        # never claims a write it did not perform.
        assert record.new_endpoint_uuid == ''
        assert record.intended_referent == ''
        assert record.minted is False
        assert record.moved is False
        assert record.summaries_refreshed == ()
        assert record.deleted_emptied_node == ''
        assert record.reason == ''

    def test_is_keyword_only(self):
        """Positional construction of a twelve-field evidence record is how a
        field silently lands in the wrong slot."""
        with pytest.raises(TypeError):
            ReferentRepair('e1', 'source', 'repaired')  # type: ignore[misc]

    def test_is_frozen(self):
        """A repair record is evidence for DESTRUCTIVE edge surgery — a
        consumer must not be able to rewrite which edge end it names."""
        record = _repair()
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.outcome = 'unrepairable'  # type: ignore[misc]

    def test_summaries_refreshed_is_a_tuple_not_a_list(self):
        """`frozen=True` blocks attribute REBINDING only — a list field would
        leave `record.summaries_refreshed.append(...)` open, letting a consumer
        quietly widen the evidence of what this pass actually refreshed."""
        record = _repair(summaries_refreshed=('n-3129', 'n-3127'))
        assert isinstance(record.summaries_refreshed, tuple)

    @pytest.mark.parametrize('outcome', list(REFERENT_REPAIR_OUTCOMES))
    def test_every_registered_outcome_constructs(self, outcome):
        assert _repair(outcome=outcome).outcome == outcome

    def test_an_unregistered_outcome_raises_naming_the_vocabulary(self):
        """Exactly as `ReferentFinding.__post_init__` does for `check`: a
        disposition no consumer can key off must not be recordable."""
        with pytest.raises(ValueError) as exc:
            _repair(outcome='partially-repaired')
        message = str(exc.value)
        assert 'partially-repaired' in message
        for registered in REFERENT_REPAIR_OUTCOMES:
            assert registered in message

    def test_to_dict_is_json_safe_and_keyed_by_field_names(self):
        record = _repair(
            new_endpoint_uuid='n-3127',
            intended_referent='Task 3127',
            minted=True,
            moved=True,
            summaries_refreshed=('n-3129', 'n-3127'),
            deleted_emptied_node='n-3129',
            reason='',
        )
        payload = record.to_dict()

        assert payload == {
            'edge_uuid': 'e1',
            'which_end': 'source',
            'outcome': 'repaired',
            'old_endpoint_uuid': 'n-3129',
            'new_endpoint_uuid': 'n-3127',
            'intended_referent': 'Task 3127',
            'check': 'set-membership',
            'minted': True,
            'moved': True,
            'summaries_refreshed': ['n-3129', 'n-3127'],
            'deleted_emptied_node': 'n-3129',
            'reason': '',
        }
        # The tuple renders as a LIST — the escalation detail carries this
        # verbatim through `json.dumps`, and a tuple is not JSON.
        assert isinstance(payload['summaries_refreshed'], list)
        json.dumps(payload)


class TestReferentRepairStats:
    """The counts are @property comprehensions, never stored fields, so they
    CANNOT drift from the list they summarize."""

    def test_an_empty_stats_reports_every_count_as_zero(self):
        stats = ReferentRepairStats()

        assert stats.repairs == []
        assert stats.repaired == 0
        assert stats.flagged_unrepairable == 0
        assert stats.degenerate_edges == 0
        assert stats.failed == 0
        assert stats.nodes_minted == 0
        assert stats.nodes_deleted == 0

    def test_repaired_counts_only_records_that_actually_moved_an_endpoint(self):
        """A `moved=False` result is `reassign_edge`'s own corroborate-before-
        acting no-op — the edge was ALREADY correct, which is the opposite of a
        repair, and must never feed the storm streak."""
        stats = ReferentRepairStats(repairs=[
            _repair(edge_uuid='e1', moved=True),
            _repair(edge_uuid='e2', moved=False),
        ])
        assert stats.repaired == 1

    def test_unrepairable_and_degenerate_share_one_flagged_bucket(self):
        """The task's NEVER GUESS rule assigns both the same disposition —
        recorded and left alone — so the operator reads one number for
        "we refused to act"."""
        stats = ReferentRepairStats(repairs=[
            _repair(edge_uuid='e1', outcome='unrepairable'),
            _repair(edge_uuid='e2', outcome='degenerate'),
            _repair(edge_uuid='e2', outcome='degenerate'),
        ])
        assert stats.flagged_unrepairable == 3

    def test_degenerate_edges_counts_EDGES_not_findings(self):
        """A degenerate edge produces one record per END; the operator's
        question is "how many edges did we skip whole"."""
        stats = ReferentRepairStats(repairs=[
            _repair(edge_uuid='e1', which_end='source', outcome='degenerate'),
            _repair(edge_uuid='e1', which_end='target', outcome='degenerate'),
            _repair(edge_uuid='e2', which_end='source', outcome='degenerate'),
            _repair(edge_uuid='e2', which_end='target', outcome='degenerate'),
        ])
        assert stats.degenerate_edges == 2
        assert stats.flagged_unrepairable == 4

    def test_failed_is_a_third_disposition_counted_separately(self):
        """"We tried and the backend did not cooperate" is an infrastructure
        signal — never folded into the NEVER-GUESS bucket."""
        stats = ReferentRepairStats(repairs=[
            _repair(edge_uuid='e1', outcome='failed', reason='falkor down'),
            _repair(edge_uuid='e2', outcome='unrepairable'),
        ])
        assert stats.failed == 1
        assert stats.flagged_unrepairable == 1
        assert stats.repaired == 0

    def test_nodes_minted_and_nodes_deleted_are_derived_from_the_records(self):
        stats = ReferentRepairStats(repairs=[
            _repair(edge_uuid='e1', minted=True, moved=True,
                    deleted_emptied_node='n-3129'),
            _repair(edge_uuid='e2', minted=False, moved=True),
        ])
        assert stats.nodes_minted == 1
        assert stats.nodes_deleted == 1

    def test_nodes_deleted_counts_DISTINCT_nodes(self):
        """Two repairs moving endpoints off the same node produce one
        deletion, stamped onto both records."""
        stats = ReferentRepairStats(repairs=[
            _repair(edge_uuid='e1', moved=True, deleted_emptied_node='n-3129'),
            _repair(edge_uuid='e2', moved=True, deleted_emptied_node='n-3129'),
        ])
        assert stats.nodes_deleted == 1


class TestReconcileStatsCarriesTheRepairRecord:
    """The aggregate the reconcile chain returns grows an eta field alongside
    zeta's, so a caller reads detection AND repair off one object."""

    def test_repair_stats_defaults_to_an_empty_instance(self):
        stats = ReconcileStats()
        assert isinstance(stats.repair_stats, ReferentRepairStats)
        assert stats.repair_stats.repairs == []

    def test_the_default_is_per_instance_not_shared(self):
        """A mutable default shared across instances would let one episode's
        repairs show up on another's audit record."""
        first = ReconcileStats()
        second = ReconcileStats()
        first.repair_stats.repairs.append(_repair())
        assert second.repair_stats.repairs == []


# ---------------------------------------------------------------------------
# The repair sequence: ensure_entity_node -> reassign_edge -> refresh_entity_summary
# ---------------------------------------------------------------------------


@pytest.fixture
def service(mock_config):
    """MemoryService with fully-mocked backends.

    `install_identity_mocks` is REQUIRED, not decorative: the end-to-end wiring
    test drives `_execute_graphiti_write`, which wraps its critical section in
    `async with self.graphiti._identity_lock_for(...)`, which a bare MagicMock
    cannot satisfy — and alpha's `ensure_entity_node` LOCK CONTRACT is the
    thing that section exists to honour.
    """
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti._require_client = MagicMock()
    svc.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])
    for name in _WRITE_PRIMITIVES:
        setattr(svc.graphiti, name, AsyncMock(return_value=None))
    svc.graphiti.ensure_entity_node = AsyncMock(return_value='n-3127')
    svc.graphiti.reassign_edge = AsyncMock(return_value=_reassigned())
    svc.graphiti.refresh_entity_summary = AsyncMock(return_value={})
    svc.graphiti.get_valid_edges_for_node = AsyncMock(return_value=[])
    svc.graphiti.delete_entity = AsyncMock(return_value={})
    install_identity_mocks(svc.graphiti)
    return svc


def _finding(**overrides) -> ReferentFinding:
    """A RESOLVABLE finding — eta's common case. Overrides tune what a test pins."""
    fields = {
        'edge_uuid': 'e1',
        'which_end': 'source',
        'check': 'set-membership',
        'old_endpoint_uuid': 'n-3129',
        'old_endpoint_name': 'Task 3129',
        'endpoint_referent': Referent(number='3129'),
        'referent_set': ('Task 3127',),
        'intended_referent': Referent(number='3127'),
        'new_endpoint_uuid': 'n-3127',
        'resolvable': True,
    }
    fields.update(overrides)
    return ReferentFinding(**fields)


def _stats(*findings, endpoints_checked: int | None = None) -> ReferentStats:
    """zeta's return value, carrying *findings*.

    `endpoints_checked` defaults to the finding count so the INV-4 "did we
    actually look" question has a truthful answer without every test spelling
    it; a test pinning the checked-nothing arm passes 0 explicitly.
    """
    stats = ReferentStats(
        edges_scanned=len({f.edge_uuid for f in findings}),
        endpoints_checked=(
            len(findings) if endpoints_checked is None else endpoints_checked
        ),
    )
    stats.findings.extend(findings)
    return stats


def _reassigned(**overrides) -> dict:
    """`reassign_edge`'s audit dict, defaulting to a real move whose internal
    summary refresh SUCCEEDED for both affected endpoints (the ~100% path)."""
    payload = {
        'uuid': 'e1',
        'which_end': 'source',
        'old_endpoint_uuid': 'n-3129',
        'new_endpoint_uuid': 'n-3127',
        'unchanged_endpoint_uuid': 'n-other',
        'moved': True,
        'refreshed_nodes': ['n-3129', 'n-3127'],
    }
    payload.update(overrides)
    return payload


class TestTheRepairSequence:
    """ensure_entity_node THEN reassign_edge, in that order, per resolvable finding."""

    @pytest.mark.asyncio
    async def test_ensure_then_reassign_in_order_with_the_contract_arguments(
        self, service,
    ):
        """The mint step graphiti_core will never do for us, then the lossless
        endpoint move — ORDER pinned, not just arguments."""
        manager = MagicMock()
        manager.attach_mock(service.graphiti.ensure_entity_node, 'ensure_entity_node')
        manager.attach_mock(service.graphiti.reassign_edge, 'reassign_edge')

        stats = await service._repair_episode_referents(
            _stats(_finding()), group_id='dark_factory',
        )

        service.graphiti.ensure_entity_node.assert_awaited_once_with(
            'Task 3127', group_id='dark_factory',
        )
        service.graphiti.reassign_edge.assert_awaited_once_with(
            'e1', 'n-3127', which_end='source', group_id='dark_factory',
        )
        assert [c[0] for c in manager.mock_calls] == [
            'ensure_entity_node', 'reassign_edge',
        ]
        assert stats.repaired == 1

    @pytest.mark.asyncio
    async def test_the_reassign_target_comes_from_ensure_entity_node_not_from_zeta(
        self, service,
    ):
        """THE assertion that keeps a stale zeta lookup from becoming the repair
        target.  `ensure_entity_node` re-reads under the lock and COLLAPSES a
        duplicate-name group; `finding.new_endpoint_uuid` is audit metadata its
        own docstring already calls "an audit convenience"."""
        service.graphiti.ensure_entity_node = AsyncMock(return_value='n-3127')
        service.graphiti.reassign_edge = AsyncMock(return_value=_reassigned())

        await service._repair_episode_referents(
            _stats(_finding(new_endpoint_uuid='n-stale')), group_id='dark_factory',
        )

        service.graphiti.reassign_edge.assert_awaited_once_with(
            'e1', 'n-3127', which_end='source', group_id='dark_factory',
        )

    @pytest.mark.asyncio
    async def test_ensure_entity_node_is_called_even_when_zeta_already_resolved_a_uuid(
        self, service,
    ):
        """UNCONDITIONALLY, not only on zeta's None branch: it is idempotent
        (the resolve path mints nothing), and branching would create a second
        site that can disagree about what the edge should point at."""
        await service._repair_episode_referents(
            _stats(_finding(new_endpoint_uuid='n-3127')), group_id='dark_factory',
        )
        service.graphiti.ensure_entity_node.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_repair_record_carries_every_contract_field(self, service):
        stats = await service._repair_episode_referents(
            _stats(_finding(check='per-edge-pairing')), group_id='dark_factory',
        )

        assert len(stats.repairs) == 1
        record = stats.repairs[0]
        assert record.edge_uuid == 'e1'
        assert record.which_end == 'source'
        assert record.outcome == 'repaired'
        assert record.moved is True
        assert record.old_endpoint_uuid == 'n-3129'
        assert record.new_endpoint_uuid == 'n-3127'
        assert record.intended_referent == 'Task 3127'
        # Carried through from the finding that justified the repair, so a
        # reader never has to join back to the ReferentStats to learn why.
        assert record.check == 'per-edge-pairing'
        assert stats.repaired == 1

    @pytest.mark.asyncio
    async def test_the_old_endpoint_is_read_from_reassign_edge_not_from_the_finding(
        self, service,
    ):
        """INV-3 corroborate-before-acting, preserved by DELEGATION:
        `reassign_edge` re-reads BOTH endpoints from topology, so its report of
        what the edge actually hung off outranks zeta's in-memory snapshot."""
        service.graphiti.reassign_edge = AsyncMock(
            return_value=_reassigned(old_endpoint_uuid='n-actually-3130'),
        )

        stats = await service._repair_episode_referents(
            _stats(_finding()), group_id='dark_factory',
        )

        assert stats.repairs[0].old_endpoint_uuid == 'n-actually-3130'

    @pytest.mark.asyncio
    async def test_a_resolvable_finding_with_no_intended_referent_is_unrepairable(
        self, service,
    ):
        """A shape zeta forbids but the TYPE permits.  Fail closed — record it,
        do not crash, and above all do not guess a target."""
        stats = await service._repair_episode_referents(
            _stats(_finding(resolvable=True, intended_referent=None)),
            group_id='dark_factory',
        )

        assert_never_repaired(service)
        assert len(stats.repairs) == 1
        assert stats.repairs[0].outcome == 'unrepairable'
        assert stats.repaired == 0

    @pytest.mark.asyncio
    async def test_no_findings_costs_nothing(self, service):
        """The ~99.8% clean path must issue ZERO backend calls inside the
        per-group identity lock."""
        stats = await service._repair_episode_referents(
            _stats(), group_id='dark_factory',
        )
        assert_never_repaired(service)
        assert stats.repairs == []


class TestTheRefreshBackstop:
    """Step 3 is a CONDITIONAL backstop, not an unconditional third call.

    `reassign_edge` already refreshes both AFFECTED endpoint summaries after a
    real move — but per-node try/except that LOGS AND SWALLOWS, reporting only
    what actually succeeded in `refreshed_nodes`.  So an unconditional third
    call doubles the summary regeneration on the ~100% happy path (inside the
    per-group identity lock, where every extra round-trip serializes same-group
    writes), while omitting step 3 leaves the PRD's stated user-observable
    signal — "the `Task N±1` summary no longer contains it" — silently
    degradable whenever that swallowed exception fires.
    """

    @pytest.mark.asyncio
    async def test_the_backstop_fires_for_both_endpoints_when_the_internal_refresh_failed(
        self, service,
    ):
        """`refreshed_nodes == []` is reassign_edge reporting that its own
        best-effort refresh was swallowed.  This is what makes the PRD's
        observable signal a GUARANTEE rather than a best effort."""
        service.graphiti.reassign_edge = AsyncMock(
            return_value=_reassigned(refreshed_nodes=[]),
        )

        stats = await service._repair_episode_referents(
            _stats(_finding()), group_id='dark_factory',
        )

        refreshed = {
            c.args[0] for c in service.graphiti.refresh_entity_summary.await_args_list
        }
        assert refreshed == {'n-3129', 'n-3127'}
        for call in service.graphiti.refresh_entity_summary.await_args_list:
            assert call.kwargs == {'group_id': 'dark_factory'}
        assert set(stats.repairs[0].summaries_refreshed) == {'n-3129', 'n-3127'}

    @pytest.mark.asyncio
    async def test_no_double_work_when_reassign_edge_already_refreshed_both(
        self, service,
    ):
        """The happy path costs ZERO extra round-trips inside the identity
        lock — and the record still reports both, because it says what is true
        of the GRAPH, not what this method happened to call."""
        service.graphiti.reassign_edge = AsyncMock(
            return_value=_reassigned(refreshed_nodes=['n-3129', 'n-3127']),
        )

        stats = await service._repair_episode_referents(
            _stats(_finding()), group_id='dark_factory',
        )

        service.graphiti.refresh_entity_summary.assert_not_awaited()
        assert set(stats.repairs[0].summaries_refreshed) == {'n-3129', 'n-3127'}

    @pytest.mark.asyncio
    async def test_a_partial_internal_refresh_re_refreshes_only_the_remainder(
        self, service,
    ):
        service.graphiti.reassign_edge = AsyncMock(
            return_value=_reassigned(refreshed_nodes=['n-3127']),
        )

        stats = await service._repair_episode_referents(
            _stats(_finding()), group_id='dark_factory',
        )

        service.graphiti.refresh_entity_summary.assert_awaited_once_with(
            'n-3129', group_id='dark_factory',
        )
        assert set(stats.repairs[0].summaries_refreshed) == {'n-3129', 'n-3127'}

    @pytest.mark.asyncio
    async def test_the_inv3_no_op_arm_refreshes_nothing_and_is_not_a_repair(
        self, service,
    ):
        """`moved=False` is `reassign_edge`'s own corroborate-before-acting
        guard: the edge had ALREADY been repointed, so no summary can have
        changed and no repair happened.  A corroborated no-op must not later
        feed the storm streak."""
        service.graphiti.reassign_edge = AsyncMock(
            return_value=_reassigned(moved=False, refreshed_nodes=[]),
        )

        stats = await service._repair_episode_referents(
            _stats(_finding()), group_id='dark_factory',
        )

        service.graphiti.refresh_entity_summary.assert_not_awaited()
        assert stats.repairs[0].outcome == 'repaired'
        assert stats.repairs[0].moved is False
        assert stats.repairs[0].summaries_refreshed == ()
        assert stats.repaired == 0

    @pytest.mark.asyncio
    async def test_a_failing_backstop_refresh_is_swallowed_and_the_repair_stands(
        self, service, caplog,
    ):
        """The topology move ALREADY COMMITTED.  Un-counting it because the
        cosmetic summary regeneration failed would report zero repairs for an
        episode that performed one."""
        service.graphiti.reassign_edge = AsyncMock(
            return_value=_reassigned(refreshed_nodes=['n-3127']),
        )
        service.graphiti.refresh_entity_summary = AsyncMock(
            side_effect=RuntimeError('falkor down'),
        )

        with caplog.at_level(logging.WARNING):
            stats = await service._repair_episode_referents(
                _stats(_finding()), group_id='dark_factory',
            )

        assert stats.repairs[0].outcome == 'repaired'
        assert stats.repairs[0].moved is True
        assert stats.repaired == 1
        # Only what actually succeeded — the record never claims a refresh
        # that raised.
        assert stats.repairs[0].summaries_refreshed == ('n-3127',)
        assert any('n-3129' in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'exc_type', [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
    )
    async def test_the_backstop_never_swallows_cancellation(self, service, exc_type):
        service.graphiti.reassign_edge = AsyncMock(
            return_value=_reassigned(refreshed_nodes=[]),
        )
        service.graphiti.refresh_entity_summary = AsyncMock(
            side_effect=exc_type('interrupted'),
        )

        with pytest.raises(exc_type):
            await service._repair_episode_referents(
                _stats(_finding()), group_id='dark_factory',
            )


class TestNeverGuess:
    """An unresolvable finding is RECORDED and LEFT ALONE.

    The PRD's live boundary row, verbatim: the unary fact "Umbrella task 2519
    was filed and then cancelled to avoid orphaning its vector" sitting on the
    `Task 2520` node.  There is no correct target — the fact names exactly one
    task and it is not the one the edge landed on — so zeta records it with
    `resolvable=False` and a reason, and eta must not invent one.
    """

    UNARY_REASON = (
        'no candidate target could be determined from the declared referents'
    )

    def _unary_finding(self, **overrides) -> ReferentFinding:
        fields = {
            'edge_uuid': 'e-2520',
            'old_endpoint_uuid': 'n-2520',
            'old_endpoint_name': 'Task 2520',
            'endpoint_referent': Referent(number='2520'),
            'referent_set': ('Task 2519',),
            'intended_referent': None,
            'new_endpoint_uuid': None,
            'resolvable': False,
            'reason': self.UNARY_REASON,
        }
        fields.update(overrides)
        return _finding(**fields)

    @pytest.mark.asyncio
    async def test_no_write_primitive_is_awaited_at_all(self, service):
        """Not "attempted and rolled back" — never attempted.  The refusal is
        decided before any backend call."""
        await service._repair_episode_referents(
            _stats(self._unary_finding()), group_id='dark_factory',
        )
        assert_never_repaired(service)

    @pytest.mark.asyncio
    async def test_the_record_carries_zetas_own_reason_verbatim(self, service):
        """The operator must see zeta's explanation, not an eta-authored
        paraphrase — a paraphrase is a second site that can drift from the rule
        that actually fired."""
        stats = await service._repair_episode_referents(
            _stats(self._unary_finding()), group_id='dark_factory',
        )

        assert len(stats.repairs) == 1
        record = stats.repairs[0]
        assert record.outcome == 'unrepairable'
        assert record.reason == self.UNARY_REASON
        assert record.edge_uuid == 'e-2520'
        assert record.which_end == 'source'
        assert record.old_endpoint_uuid == 'n-2520'
        assert record.check == 'set-membership'
        # Nothing was targeted, minted, moved or refreshed.
        assert record.new_endpoint_uuid == ''
        assert record.intended_referent == ''
        assert record.moved is False
        assert record.minted is False
        assert record.summaries_refreshed == ()
        assert record.deleted_emptied_node == ''

    @pytest.mark.asyncio
    async def test_it_lands_in_the_flagged_bucket_and_never_in_repaired(
        self, service,
    ):
        stats = await service._repair_episode_referents(
            _stats(self._unary_finding()), group_id='dark_factory',
        )
        assert stats.flagged_unrepairable == 1
        assert stats.repaired == 0
        assert stats.failed == 0

    @pytest.mark.asyncio
    async def test_a_mixed_batch_repairs_the_resolvable_one_and_records_both(
        self, service,
    ):
        """Different EDGES, so the degenerate-edge guard is not what is being
        exercised here — one finding refusing to be guessed at must not stop
        an unrelated edge's repair."""
        stats = await service._repair_episode_referents(
            _stats(_finding(edge_uuid='e1'), self._unary_finding()),
            group_id='dark_factory',
        )

        service.graphiti.ensure_entity_node.assert_awaited_once_with(
            'Task 3127', group_id='dark_factory',
        )
        service.graphiti.reassign_edge.assert_awaited_once_with(
            'e1', 'n-3127', which_end='source', group_id='dark_factory',
        )
        assert len(stats.repairs) == 2
        assert {r.outcome for r in stats.repairs} == {'repaired', 'unrepairable'}
        assert stats.repaired == 1
        assert stats.flagged_unrepairable == 1
