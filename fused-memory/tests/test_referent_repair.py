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

import dataclasses
import json

import pytest

from fused_memory.services.memory_service import (
    REFERENT_REPAIR_OUTCOMES,
    ReconcileStats,
    ReferentRepair,
    ReferentRepairStats,
)


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
