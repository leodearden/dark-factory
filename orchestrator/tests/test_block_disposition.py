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
import inspect

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


# ---------------------------------------------------------------------------
# step-03: classify_failure(exc) -> BlockDisposition known-row value tests
# ---------------------------------------------------------------------------


class TestClassifyFailureKnownRows:
    """One assertion group per exception the pre-W9-ε ladder hand-classified.

    Every row's ``.category`` is FailureCategory.NONE — none of these are
    verify-check failures (see W9-ε's design decisions).
    """

    def test_all_accounts_capped_blocks_non_escalating_agent_failure(self):
        from shared.cli_invoke import AllAccountsCappedException

        from orchestrator.unblock_types import BlockClass
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        exc = AllAccountsCappedException(retries=5, elapsed_secs=120.5, label='Task 7 [impl]')
        disp = classify_failure(exc)
        assert disp.requeue_kind is RequeueKind.BLOCK
        assert disp.escalate_to_human is False  # steward path — not an immediate L1
        assert disp.block_class is BlockClass.AGENT_FAILURE
        assert disp.reason_prefix.startswith('All accounts capped')
        assert disp.category is FailureCategory.NONE

    def test_session_budget_exhausted_blocks_non_escalating(self):
        from shared.usage_gate import SessionBudgetExhausted

        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        disp = classify_failure(SessionBudgetExhausted(cumulative_cost=42.0))
        assert disp.requeue_kind is RequeueKind.BLOCK
        assert disp.escalate_to_human is False
        assert disp.category is FailureCategory.NONE

    def test_warm_lane_pool_exhausted_requeues_and_counts_against_cap(self):
        from orchestrator.git_ops import WarmLanePoolExhausted
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        disp = classify_failure(WarmLanePoolExhausted('all lanes assigned'))
        assert disp.requeue_kind is RequeueKind.REQUEUE
        assert disp.counts_against_requeue_cap is True
        assert disp.category is FailureCategory.NONE

    def test_warm_lane_disk_pressure_requeues_without_counting_against_cap(self):
        from orchestrator.git_ops import WarmLaneDiskPressure
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        disp = classify_failure(WarmLaneDiskPressure('seed exited 75'))
        assert disp.requeue_kind is RequeueKind.REQUEUE
        assert disp.counts_against_requeue_cap is False
        assert disp.category is FailureCategory.NONE

    def test_warm_lane_pool_hard_down_requeues_without_counting_against_cap(self):
        from orchestrator.git_ops import WarmLanePoolHardDown
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        disp = classify_failure(WarmLanePoolHardDown('warm base absent'))
        assert disp.requeue_kind is RequeueKind.REQUEUE
        assert disp.counts_against_requeue_cap is False
        assert disp.category is FailureCategory.NONE

    def test_verify_infra_error_blocks_and_escalates(self):
        from orchestrator.verify import VerifyInfraError
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        disp = classify_failure(VerifyInfraError(phase='verify', errno=28))
        assert disp.requeue_kind is RequeueKind.BLOCK
        assert disp.escalate_to_human is True
        assert disp.category is FailureCategory.NONE

    def test_infra_oserror_blocks_and_escalates(self):
        import errno as errno_mod

        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        exc = OSError(errno_mod.ENOSPC, 'No space left on device')
        disp = classify_failure(exc)
        assert disp.requeue_kind is RequeueKind.BLOCK
        assert disp.escalate_to_human is True
        assert disp.category is FailureCategory.NONE

    def test_non_infra_oserror_blocks_without_escalating(self):
        import errno as errno_mod

        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        exc = OSError(errno_mod.EACCES, 'Permission denied')
        disp = classify_failure(exc)
        assert disp.requeue_kind is RequeueKind.BLOCK
        assert disp.escalate_to_human is False
        assert disp.category is FailureCategory.NONE

    def test_worktree_conflict_error_blocks_and_escalates(self):
        from pathlib import Path

        from orchestrator.git_ops import WorktreeConflictError
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        exc = WorktreeConflictError(Path('/tmp/wt'), ['a.py'])
        disp = classify_failure(exc)
        assert disp.requeue_kind is RequeueKind.BLOCK
        assert disp.escalate_to_human is True
        assert disp.category is FailureCategory.NONE

    def test_bare_exception_blocks_non_escalating_agent_failure(self):
        from orchestrator.unblock_types import BlockClass
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure
        disp = classify_failure(Exception('boom'))
        assert disp.requeue_kind is RequeueKind.BLOCK
        assert disp.escalate_to_human is False
        assert disp.block_class is BlockClass.AGENT_FAILURE
        assert disp.reason_prefix == 'Workflow error'
        assert disp.category is FailureCategory.NONE

    def test_classify_failure_is_total_for_an_unrecognized_exception(self):
        # Sanity: classify_failure never raises — an exception type with no
        # explicit row still resolves to SOME disposition (the default).
        from orchestrator.workflow_types import classify_failure

        class _SomeUnrelatedError(Exception):
            pass

        disp = classify_failure(_SomeUnrelatedError('surprise'))
        assert disp is not None


# ---------------------------------------------------------------------------
# step-05: BD-2 completeness test (boundary row 11)
# ---------------------------------------------------------------------------


def _public_exception_types(module):
    """Every public (no leading underscore) BaseException subclass in *module*."""
    return [
        obj for name, obj in vars(module).items()
        if not name.startswith('_')
        and inspect.isclass(obj)
        and issubclass(obj, BaseException)
    ]


class TestBD2Completeness:
    """Every exception exported by the four BD-2 modules has an EXPLICIT
    ``_DISPOSITION_TABLE`` row — never just the fallback default."""

    def test_every_exported_exception_has_an_explicit_row(self):
        import shared.cli_invoke as cli_invoke
        import shared.usage_gate as usage_gate

        import orchestrator.git_ops as git_ops
        import orchestrator.verify as verify
        from orchestrator.workflow_types import _lookup_disposition

        exc_types = [
            t
            for module in (git_ops, verify, cli_invoke, usage_gate)
            for t in _public_exception_types(module)
        ]
        assert exc_types, 'sanity: the 4 BD-2 modules must export at least one exception'

        missing = [t for t in exc_types if _lookup_disposition(t) is None]
        assert not missing, (
            'exported exception types with no explicit disposition row: '
            f'{[t.__qualname__ for t in missing]}'
        )

    def test_a_brand_new_exception_type_has_no_row_but_still_classifies(self):
        # A synthetic type with no table row proves the completeness check
        # above is meaningful: it FAILS for an unrecognized type rather than
        # silently matching everything.
        from orchestrator.workflow_types import _lookup_disposition, classify_failure

        class _BrandNewFailure(Exception):
            pass

        assert _lookup_disposition(_BrandNewFailure) is None
        # classify_failure is still TOTAL — it falls back to the default.
        disp = classify_failure(_BrandNewFailure('surprise'))
        assert disp is not None
