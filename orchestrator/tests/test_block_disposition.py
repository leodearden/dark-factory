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

step-09 note: the ``counts_against_requeue_cap`` policy per warm-lane
subclass (EXHAUSTED=True, DISK_PRESSURE/HARD_DOWN=False) is already pinned
by step-03's ``TestClassifyFailureKnownRows`` rows above — not repeated here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Cross-module reuse — conftest.py injects orchestrator/tests onto sys.path
# (see test_workflow_terminal_report.py for the same precedent). ``_derive_
# meta_root_like_production`` is imported (not just the plain helpers)
# because it is an autouse fixture in test_workflow_e2e.py — autouse only
# auto-applies within a module where pytest can SEE the fixture.
from test_workflow_e2e import (
    AgentStub,
    _build_workflow,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see above
    _init_repo,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify_categories import FailureCategory
from orchestrator.workflow_types import WorkflowOutcome

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
        kwargs: dict[str, Any] = dict(
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

    def test_bare_warm_lane_requeue_base_matches_old_else_branch_fallback(self):
        """Amendment (reviewer_comprehensive behavior-parity): a bare
        WarmLaneRequeue (none of the 3 named subclasses) is never raised
        anywhere in the tree today, but its table row must still mirror the
        pre-W9-ε inline triage's `else:  # WarmLaneDiskPressure` fallback —
        which mechanically classified ANY WarmLaneRequeue that wasn't
        WarmLanePoolHardDown/WarmLanePoolExhausted as disk-pressure-shaped —
        rather than a distinct 'warm_lane_requeue' literal no old code path
        ever produced. This keeps the workflow.py WarmLaneRequeue
        except-clause comment's "reproduces the old per-subclass strings
        exactly" claim true for the base class too, not just the 3 concrete
        subclasses.
        """
        from orchestrator.git_ops import WarmLaneDiskPressure, WarmLaneRequeue
        from orchestrator.workflow_types import classify_failure
        base_disp = classify_failure(WarmLaneRequeue('bare base instance'))
        disk_pressure_disp = classify_failure(WarmLaneDiskPressure('seed exited 75'))
        assert base_disp == disk_pressure_disp

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

    def test_worktree_missing_row_is_shadowed_by_the_oserror_branch(self):
        """Amendment (reviewer_comprehensive correctness): passing the
        completeness check above (``_lookup_disposition(WorktreeMissing) is
        not None``) does NOT mean ``classify_failure()`` ever returns that
        row. WorktreeMissing IS-A FileNotFoundError IS-A OSError, and
        classify_failure's ``isinstance(exc, OSError)`` branch runs BEFORE
        ``_lookup_disposition`` — see its docstring. A real WorktreeMissing
        always carries ``errno=None`` (constructed from a single message
        string, not the ``(errno, strerror)`` OSError form), which is never
        in ``verify._INFRA_ERRNOS``, so ``classify_failure(instance)`` falls
        all the way through to ``_DEFAULT_BLOCK`` — NOT the row's own
        ``escalate_to_human=True``. This test pins that gap explicitly so a
        future reader can't mistake the completeness check's success for
        "classify_failure honors this row."
        """
        from orchestrator.git_ops import WorktreeMissing
        from orchestrator.workflow_types import _disposition_table, classify_failure

        exc = WorktreeMissing('/some/worktree/path')
        assert exc.errno is None  # sanity: confirms the OSError branch decides, not the row

        row = _disposition_table()[WorktreeMissing]
        resolved = classify_failure(exc)

        assert row.escalate_to_human is True  # the (unreachable) row's own declared value
        assert resolved.escalate_to_human is False  # what classify_failure ACTUALLY returns
        assert resolved != row


# ---------------------------------------------------------------------------
# step-07: run()-level BLOCK outcome-kind tests (mirrors test_workflow_
# terminal_report.py's e2e-style block-path fixtures verbatim).
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A bare-minimum git repo with an initial commit (mirrors test_workflow_e2e.py)."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'Add farewell function',
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


@pytest.mark.asyncio
class TestRunLevelBlockOutcomeKind:
    """``run()`` collapses the 8-clause ladder to ONE ``classify_failure``-
    driven block path (boundary rows 7/8). Still RED at this step — ``_drive()``
    is not collapsed until step-8, so ``TerminalReport.category`` stays the
    hard-``None`` of the pre-refactor ladder for every case below.

    Each case forces ``_drive()`` to raise by patching ``invoke_agent`` (the
    earliest awaited coroutine in the PLAN phase) — the same "early-phase
    coroutine" technique ``test_workflow_terminal_report.py`` already uses
    for ``AllAccountsCappedException``, generalized to the other ladder
    exception types. ``_mark_blocked`` is wrapped (not replaced) with an
    ``AsyncMock`` spy so the REAL implementation still runs (producing a
    genuine ``TerminalReport``) while the call's ``escalate_to_human`` kwarg
    stays inspectable — ``TerminalReport`` itself carries no such field.
    """

    async def _run_forcing_exception(
        self, config, git_ops, task_assignment, monkeypatch, exc: BaseException,
    ):
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_exc(*args, **kwargs):
            raise exc

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )
        mark_blocked_spy = AsyncMock(wraps=workflow._mark_blocked)
        workflow._mark_blocked = mark_blocked_spy  # type: ignore[method-assign]

        report = await workflow.run()
        return report, mark_blocked_spy

    async def test_all_accounts_capped_still_blocks_with_cap_reason(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        from shared.cli_invoke import AllAccountsCappedException

        exc = AllAccountsCappedException(retries=3, elapsed_secs=120.0, label='Task 42 [architect]')
        report, mark_blocked_spy = await self._run_forcing_exception(
            config, git_ops, task_assignment, monkeypatch, exc,
        )

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason.lower().startswith('all accounts capped')
        assert report.category is FailureCategory.NONE
        assert mark_blocked_spy.await_args is not None
        assert mark_blocked_spy.await_args.kwargs.get('escalate_to_human', False) is False

    async def test_verify_infra_error_blocks_and_escalates_to_human(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        from orchestrator.verify import VerifyInfraError

        exc = VerifyInfraError(phase='verify', errno=28)
        report, mark_blocked_spy = await self._run_forcing_exception(
            config, git_ops, task_assignment, monkeypatch, exc,
        )

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.category is FailureCategory.NONE
        assert mark_blocked_spy.await_args is not None
        assert mark_blocked_spy.await_args.kwargs.get('escalate_to_human') is True

    async def test_worktree_conflict_error_blocks_and_escalates_to_human(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        from orchestrator.git_ops import WorktreeConflictError

        exc = WorktreeConflictError(Path('/tmp/wt'), ['a.py'])
        report, mark_blocked_spy = await self._run_forcing_exception(
            config, git_ops, task_assignment, monkeypatch, exc,
        )

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.category is FailureCategory.NONE
        assert mark_blocked_spy.await_args is not None
        assert mark_blocked_spy.await_args.kwargs.get('escalate_to_human') is True

    async def test_bare_runtime_error_blocks_via_steward_path(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """A bare (unclassified) failure routes to the steward path — i.e.
        ``escalate_to_human=False`` — same as the pre-refactor broad
        ``except Exception`` tail."""
        exc = RuntimeError('unexpected failure mid-plan')
        report, mark_blocked_spy = await self._run_forcing_exception(
            config, git_ops, task_assignment, monkeypatch, exc,
        )

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.category is FailureCategory.NONE
        assert mark_blocked_spy.await_args is not None
        assert mark_blocked_spy.await_args.kwargs.get('escalate_to_human', False) is False


# ---------------------------------------------------------------------------
# step-09: run()-level REQUEUE outcome-kind tests. The WarmLaneRequeue
# except-clause's block_reason must be SINGLE-SOURCED from
# classify_failure(e).reason_prefix, not _drive()'s own independent inline
# isinstance triage.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunLevelRequeueOutcomeKindSingleSourced:
    """Patches ``classify_failure`` to return a disposition carrying a
    distinctive ``reason_prefix`` and asserts ``run()``'s REQUEUED
    ``TerminalReport.reason`` reflects it.

    The un-patched table rows already carry the SAME literal strings
    ``_drive()``'s pre-W9-ε inline ``isinstance`` triage hard-codes
    ('warm_lane_pool_exhausted' / 'warm_lane_disk_pressure (transient
    infra)' / 'warm_lane_pool_hard_down' — see workflow_types._disposition_
    table), so a plain value-equality assertion against those strings would
    pass whether or not the clause actually consults the table — it would
    prove nothing. Patching ``classify_failure`` to return a SENTINEL value
    the pre-refactor triage cannot possibly produce is what makes "single-
    sourced from the table" an observable, falsifiable behavior.

    RED today: the ``except WarmLaneRequeue`` clause never calls
    ``classify_failure`` — it derives ``block_reason`` from its own
    ``isinstance(e, WarmLanePoolHardDown)/...Exhausted/...`` chain — so
    patching it has no effect and ``report.reason`` stays the hard-coded
    literal, not the sentinel. Turns GREEN in step-10.
    """

    async def _run_forcing_warm_lane_exc(
        self, tmp_path: Path, monkeypatch, exc: Exception, patched_disposition,
    ):
        from test_workflow_warm_lane_requeue import _make_workflow

        wf = _make_workflow(tmp_path=tmp_path)
        wf.git_ops.create_worktree = AsyncMock(side_effect=exc)
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]
        monkeypatch.setattr(
            'orchestrator.workflow.classify_failure',
            lambda _e: patched_disposition,
        )

        report = await wf.run()
        mark_blocked.assert_not_awaited()
        return report

    async def test_pool_exhausted_reason_is_single_sourced_from_table(
        self, tmp_path: Path, monkeypatch,
    ):
        from orchestrator.git_ops import WarmLanePoolExhausted
        from orchestrator.unblock_types import BlockClass
        from orchestrator.workflow_types import BlockDisposition, RequeueKind

        sentinel = BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=True,
            reason_prefix='SENTINEL_pool_exhausted',
            block_class=BlockClass.AGENT_FAILURE,
        )
        report = await self._run_forcing_warm_lane_exc(
            tmp_path, monkeypatch,
            WarmLanePoolExhausted('all lanes assigned'), sentinel,
        )
        assert report.outcome == WorkflowOutcome.REQUEUED
        assert report.reason == 'SENTINEL_pool_exhausted'

    async def test_disk_pressure_reason_is_single_sourced_from_table(
        self, tmp_path: Path, monkeypatch,
    ):
        from orchestrator.git_ops import WarmLaneDiskPressure
        from orchestrator.unblock_types import BlockClass
        from orchestrator.workflow_types import BlockDisposition, RequeueKind

        sentinel = BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='SENTINEL_disk_pressure',
            block_class=BlockClass.AGENT_FAILURE,
        )
        report = await self._run_forcing_warm_lane_exc(
            tmp_path, monkeypatch,
            WarmLaneDiskPressure('seed exited 75'), sentinel,
        )
        assert report.outcome == WorkflowOutcome.REQUEUED
        assert report.reason == 'SENTINEL_disk_pressure'

    async def test_pool_hard_down_reason_is_single_sourced_from_table(
        self, tmp_path: Path, monkeypatch,
    ):
        from orchestrator.git_ops import WarmLanePoolHardDown
        from orchestrator.unblock_types import BlockClass
        from orchestrator.workflow_types import BlockDisposition, RequeueKind

        sentinel = BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.REQUEUE,
            counts_against_requeue_cap=False,
            reason_prefix='SENTINEL_hard_down',
            block_class=BlockClass.AGENT_FAILURE,
        )
        report = await self._run_forcing_warm_lane_exc(
            tmp_path, monkeypatch,
            WarmLanePoolHardDown('warm base absent'), sentinel,
        )
        assert report.outcome == WorkflowOutcome.REQUEUED
        assert report.reason == 'SENTINEL_hard_down'


# ---------------------------------------------------------------------------
# step-11: _mark_blocked threads disposition.block_class into the dry-run
# investigation (replacing the classify_block_reason(reason) prose sniff)
# and disposition.category into TerminalReport.category.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMarkBlockedDispositionSourcedBlockClass:
    """``_mark_blocked``'s ``disposition=`` kwarg sources the dry-run
    investigation's ``block_class`` from ``disposition.block_class`` — not
    the ``classify_block_reason(reason)`` prose sniff — and sources
    ``TerminalReport.category`` from ``disposition.category``.

    Reuses test_workflow_dry_run_hook's ``_make_workflow``/``_Scheduler``
    harness, patching ``orchestrator.workflow.run_dry_run_unblock`` (the
    actual coroutine ``_spawn_dry_run_unblock`` schedules via
    ``asyncio.create_task``) to capture the kwargs it is invoked with — the
    most precise interception point for the ``block_class`` value that
    ultimately reaches the dry-run proposal entry (mirrors test_workflow_
    dry_run_hook.py's own ``TestMarkBlockedSpawnsFireAndForget`` technique).

    RED today: ``_spawn_dry_run_unblock`` takes no ``block_class`` param and
    always computes ``classify_block_reason(reason)`` itself, so a supplied
    disposition's ``block_class`` (POST_MERGE_RED_MAIN below) never reaches
    ``run_dry_run_unblock`` — the captured kwarg is AGENT_FAILURE (the prose
    sniff's default for a reason string matching none of the 4 canonical
    merge_gates prefixes) instead. Turns GREEN in step-12.
    """

    async def _mark_blocked_and_capture(
        self, tmp_path: Path, reason: str, **mark_blocked_kwargs,
    ):
        from test_workflow_dry_run_hook import _make_workflow

        wf, _scheduler = _make_workflow(tmp_path=tmp_path, task_id='2249', enabled=True)

        calls = []

        async def _spy_dry_run(**kwargs):
            calls.append(kwargs)

        with patch('orchestrator.workflow.run_dry_run_unblock', new=_spy_dry_run):
            await wf._mark_blocked(reason, **mark_blocked_kwargs)
            await asyncio.sleep(0)  # let the background task register its call
            pending = list(wf._background_tasks)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        return wf, calls

    async def test_disposition_block_class_reaches_dry_run_not_prose_sniff(
        self, tmp_path: Path,
    ):
        from orchestrator.unblock_types import BlockClass, classify_block_reason
        from orchestrator.workflow_types import BlockDisposition, RequeueKind

        reason = 'some reason'
        # Sanity: the prose sniff would NOT classify this as POST_MERGE_RED_MAIN
        # — proves the two are genuinely distinguishable in this test.
        assert classify_block_reason(reason) is BlockClass.AGENT_FAILURE

        disp = BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='Some disposition',
            block_class=BlockClass.POST_MERGE_RED_MAIN,
        )
        wf, calls = await self._mark_blocked_and_capture(
            tmp_path, reason, disposition=disp,
        )

        assert len(calls) == 1
        assert calls[0]['block_class'] is BlockClass.POST_MERGE_RED_MAIN
        assert wf._terminal_report is not None
        assert wf._terminal_report.category is disp.category

    async def test_no_disposition_keeps_prose_sniff_back_compat(
        self, tmp_path: Path,
    ):
        from orchestrator.unblock_types import classify_block_reason

        reason = 'some reason'
        wf, calls = await self._mark_blocked_and_capture(tmp_path, reason)

        assert len(calls) == 1
        assert calls[0]['block_class'] == classify_block_reason(reason)
        assert wf._terminal_report is not None
        assert wf._terminal_report.category is None


# ---------------------------------------------------------------------------
# step-13: BD-1 four-cap-site-identity test (boundary row 10). The SAME
# AllAccountsCappedException classification must be consulted by all four
# independent cap-catch sites: workflow.py (already covered by step-07's
# TestRunLevelBlockOutcomeKind.test_all_accounts_capped_still_blocks_with_
# cap_reason), steward.py, review_checkpoint.py, dry_run_unblock.py.
# ---------------------------------------------------------------------------


class TestBD1CanonicalCapDisposition:
    """classify_failure(AllAccountsCappedException) is canonical — the
    resolved disposition depends only on the exception's TYPE, never on the
    instance's retries/elapsed_secs/label — so every one of the four sites
    below is guaranteed to consult the identical BlockDisposition regardless
    of the retries/elapsed/label it happens to observe."""

    def test_two_distinct_cap_instances_classify_identically(self):
        from shared.cli_invoke import AllAccountsCappedException

        from orchestrator.workflow_types import classify_failure

        exc_a = AllAccountsCappedException(retries=1, elapsed_secs=10.0, label='A')
        exc_b = AllAccountsCappedException(retries=99, elapsed_secs=9999.0, label='B')
        assert classify_failure(exc_a) == classify_failure(exc_b)


@pytest.mark.asyncio
class TestBD1StewardConsultsSharedClassifier:
    """steward.py's pre-triage cap handler (steward.py:703) must derive its
    inline-triage-fallback warning from ``classify_failure(exc).reason_
    prefix`` — not a hand-written 'all accounts capped' literal.

    Patches ``orchestrator.steward.classify_failure`` to return a
    disposition with a distinctive sentinel ``reason_prefix`` and asserts
    the emitted warning reflects it — the un-patched value already contains
    'all accounts capped' today (coincidentally matching the table's own
    row), so a plain substring assertion against that literal would pass
    whether or not the site actually consults the table.

    RED today: the handler hard-codes its own prose inline, never calling
    classify_failure, so the patch has no observable effect. Turns GREEN in
    step-14.
    """

    async def test_pre_triage_warning_is_single_sourced_from_table(self, caplog):
        import json
        import logging

        from shared.cli_invoke import AllAccountsCappedException
        from test_suggestion_triage import _make_escalation, _make_steward, _make_suggestions

        from orchestrator.unblock_types import BlockClass
        from orchestrator.workflow_types import BlockDisposition, RequeueKind

        steward = _make_steward()
        suggestions = _make_suggestions(15)
        escalation = _make_escalation(detail=json.dumps(suggestions))
        cap_exc = AllAccountsCappedException(
            retries=2, elapsed_secs=30.0, label='Steward for task 42 [pre-triage]',
        )
        sentinel = BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='SENTINEL_steward_cap',
            block_class=BlockClass.AGENT_FAILURE,
        )

        with patch(
            'orchestrator.steward.invoke_with_cap_retry',
            AsyncMock(side_effect=cap_exc),
        ), patch(
            'orchestrator.steward.classify_failure', lambda _e: sentinel,
        ), caplog.at_level(logging.WARNING, logger='orchestrator.steward'):
            result = await steward._pre_triage_suggestions(escalation)

        assert result is escalation
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('sentinel_steward_cap' in t.lower() for t in warning_texts), (
            f'Expected warning single-sourced from classify_failure, got: {warning_texts}'
        )


@pytest.mark.asyncio
class TestBD1ReviewCheckpointConsultsSharedClassifier:
    """review_checkpoint.py's cap handler (review_checkpoint.py:192) must
    derive its skip-review warning from classify_failure(exc).reason_prefix.

    RED today: hard-coded 'all accounts capped' literal; patching
    orchestrator.review_checkpoint.classify_failure has no effect on the
    emitted warning. Turns GREEN in step-14.
    """

    async def test_review_warning_is_single_sourced_from_table(self, monkeypatch, caplog):
        import logging

        from shared.cli_invoke import AllAccountsCappedException
        from test_review_checkpoint_cap import _PHASE1_RESULT, _make_checkpoint

        from orchestrator.unblock_types import BlockClass
        from orchestrator.workflow_types import BlockDisposition, RequeueKind

        checkpoint = _make_checkpoint()
        cap_exc = AllAccountsCappedException(
            retries=4, elapsed_secs=300.0, label='Review checkpoint [x]',
        )
        sentinel = BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='SENTINEL_review_cap',
            block_class=BlockClass.AGENT_FAILURE,
        )

        monkeypatch.setattr(
            'orchestrator.review_checkpoint.invoke_with_cap_retry',
            AsyncMock(side_effect=cap_exc),
        )
        monkeypatch.setattr(
            'orchestrator.review_checkpoint.run_full_verification',
            AsyncMock(return_value=_PHASE1_RESULT),
        )
        monkeypatch.setattr(
            'orchestrator.review_checkpoint.classify_failure', lambda _e: sentinel,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.review_checkpoint'):
            report = await checkpoint.run_focused()

        assert report.findings_count == 0
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('sentinel_review_cap' in t.lower() for t in warning_texts), (
            f'Expected warning single-sourced from classify_failure, got: {warning_texts}'
        )


@pytest.mark.asyncio
class TestBD1DryRunUnblockConsultsSharedClassifier:
    """dry_run_unblock.py's internal cap handler + _cap_exhausted_entry
    (:358/598) must derive BOTH proposal_text and block_class from
    classify_failure(cap_exc) — not independent literals.

    RED today: proposal_text hard-codes 'all accounts capped' and
    block_class is stamped generically from the OUTER block_class param /
    classify_block_reason(reason) — unrelated to the cap exception itself
    — so patching orchestrator.dry_run_unblock.classify_failure affects
    neither. Turns GREEN in step-14.
    """

    async def test_cap_exhausted_entry_is_single_sourced_from_table(self, tmp_path):
        from shared.cli_invoke import AllAccountsCappedException
        from test_dry_run_unblock import _make_config

        from orchestrator.dry_run_unblock import run_dry_run_unblock
        from orchestrator.unblock_types import BlockClass
        from orchestrator.workflow_types import BlockDisposition, RequeueKind

        cap_exc = AllAccountsCappedException(
            retries=3, elapsed_secs=1800.0, label='Task 42 [unblock_auto]',
        )
        # block_class deliberately differs from AGENT_FAILURE (what
        # classify_block_reason('verify exhausted') would derive) so the
        # assertion below can't pass by coincidence.
        sentinel = BlockDisposition(
            category=FailureCategory.NONE,
            escalate_to_human=False,
            requeue_kind=RequeueKind.BLOCK,
            counts_against_requeue_cap=True,
            reason_prefix='SENTINEL_dry_run_cap',
            block_class=BlockClass.MERGE_VERIFY_RED,
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch(
            'orchestrator.dry_run_unblock.invoke_with_cap_retry',
            AsyncMock(side_effect=cap_exc),
        ), patch(
            'orchestrator.dry_run_unblock.classify_failure', lambda _e: sentinel,
        ):
            await run_dry_run_unblock(
                task_id='42', worktree=str(tmp_path), reason='verify exhausted',
                detail='', scheduler=scheduler, mcp=MagicMock(),
                config=_make_config(),
            )

        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert 'sentinel_dry_run_cap' in entry['proposal_text'].lower(), (
            f'Expected proposal_text single-sourced from classify_failure, got: '
            f'{entry["proposal_text"]!r}'
        )
        assert entry['block_class'] == 'merge_verify_red', (
            f'Expected block_class from classify_failure(cap_exc), got: '
            f'{entry["block_class"]!r}'
        )
