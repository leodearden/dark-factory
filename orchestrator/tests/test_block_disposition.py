"""Tests for the classify_failure -> BlockDisposition table (W9-ε task).

PRD: plans/workflow-state-machine-prd.md task ε (§10 + Contract §8.1/§8.2
BD-1/BD-2 + Resolved decision 7). Replaces ``TaskWorkflow._drive()``'s
eight-clause exception ladder (workflow.py:2175-2397) with ONE
``classify_failure(exc) -> BlockDisposition`` TABLE consulted by one
``except`` per outcome-kind, and unifies the four independent
``AllAccountsCappedException`` cap-catch sites (workflow.py, steward.py,
review_checkpoint.py, dry_run_unblock.py) through that same table.

Test coverage:
  step-01: pure-unit RequeueKind / BlockDisposition value-type tests
  step-03: classify_failure known-row value tests
  step-05: BD-2 completeness test (boundary row 11)
  step-07/09: run()-level BLOCK / REQUEUE outcome-kind tests
  step-11: _mark_blocked disposition-sourced BlockRecord / TerminalReport.category
  step-13: BD-1 four-cap-site-identity test (boundary row 10)
"""

from __future__ import annotations

import dataclasses

import pytest

# ---------------------------------------------------------------------------
# step-01: RequeueKind + BlockDisposition value types
# ---------------------------------------------------------------------------


class TestRequeueKindMemberSet:
    """RequeueKind has exactly the 3 outcome-kind members the table drives
    _drive()'s collapsed except-clauses with."""

    def test_member_names_match_expected_set_exactly(self):
        from orchestrator.workflow_types import RequeueKind
        assert {m.name for m in RequeueKind} == {'REQUEUE', 'BLOCK', 'CANCEL'}

    def test_member_count_is_three(self):
        from orchestrator.workflow_types import RequeueKind
        assert len(list(RequeueKind)) == 3

    def test_importable_via_workflow_shim(self):
        from orchestrator.workflow import RequeueKind as ShimRequeueKind
        from orchestrator.workflow_types import RequeueKind
        assert ShimRequeueKind is RequeueKind


class TestBlockDispositionShape:
    """BlockDisposition is a frozen, equatable/hashable dataclass with
    exactly the 6 PRD fields (category/escalate_to_human/requeue_kind/
    counts_against_requeue_cap/reason_prefix/block_class)."""

    def _make(self, **overrides):
        from orchestrator.unblock_types import BlockClass
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import BlockDisposition, RequeueKind
        kwargs = dict(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Workflow error',
            block_class=BlockClass.AGENT_FAILURE,
        )
        kwargs.update(overrides)
        return BlockDisposition(**kwargs)

    def test_is_dataclass(self):
        from orchestrator.workflow_types import BlockDisposition
        assert dataclasses.is_dataclass(BlockDisposition)

    def test_has_expected_fields(self):
        from orchestrator.workflow_types import BlockDisposition
        field_names = {f.name for f in dataclasses.fields(BlockDisposition)}
        assert field_names == {
            'category', 'escalate_to_human', 'requeue_kind',
            'counts_against_requeue_cap', 'reason_prefix', 'block_class',
        }

    def test_round_trips_its_fields(self):
        from orchestrator.unblock_types import BlockClass
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind
        disp = self._make(
            category=FailureCategory.NONE,
            escalate_to_human=True,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='All accounts capped',
            block_class=BlockClass.AGENT_FAILURE,
        )
        assert disp.category is FailureCategory.NONE
        assert disp.escalate_to_human is True
        assert disp.requeue_kind is RequeueKind.REQUEUE
        assert disp.counts_against_requeue_cap is False
        assert disp.reason_prefix == 'All accounts capped'
        assert disp.block_class is BlockClass.AGENT_FAILURE

    def test_is_frozen(self):
        disp = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            disp.reason_prefix = 'mutated'  # type: ignore[misc]

    def test_is_equatable(self):
        assert self._make() == self._make()

    def test_is_hashable(self):
        assert hash(self._make()) == hash(self._make())

    def test_distinct_instances_are_not_equal(self):
        assert self._make(reason_prefix='a') != self._make(reason_prefix='b')

    def test_importable_via_workflow_shim(self):
        from orchestrator.workflow import BlockDisposition as ShimBlockDisposition
        from orchestrator.workflow_types import BlockDisposition
        assert ShimBlockDisposition is BlockDisposition
