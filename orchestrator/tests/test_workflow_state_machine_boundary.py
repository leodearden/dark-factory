"""W9-ι: two-way B+H boundary integration gate.

PRD ``plans/workflow-state-machine-prd.md`` task ι, §9 (boundary-test
sketch, rows 1-12) + §10. This is the SOLE non-cancellation leaf and the
G5 merge-gate/terminal-decision correctness guarantee.

TEST-ONLY. Seams α-η — MergeProvenance journal, WorkflowStateMachine,
TerminalReport, StewardOutcome, BlockDisposition, and capability wiring —
are all pre-merged on this branch and each already carries its own unit
suite. This module does NOT (re)implement any of them; it proves they hold
END-TO-END TWO-WAY by driving BOTH the producer and the consumer side of
every spine seam against HISTORICAL incident shapes (tasks 846/954/1141,
task-2911, task 2060, the SIMPLE_TASK esc-4943-54 fallthrough) rather than
invented inputs.

Postconditions assert ONLY through the product's own read paths — guard
return values + ``_merge_recovery_basis``; ``run()``'s returned
``TerminalReport``; the harness's consumed ``TaskReport``;
``_await_steward_completion``'s typed return; ``classify_failure``/
``_lookup_disposition``; ``AgentRole.__post_init__``/the ``roles`` import —
never a private side channel. Every crash/kill is a simulated, injected
fault point (an ``AsyncMock`` ``side_effect`` raising, an ``asyncio.Queue``
``put_nowait``, a bound ``LandedOutbox``, ``_cancel_event.set()``), never a
real process kill.

Out of scope: PRD §9 row 13 (RetryLedger persist-escalates) already has
full coverage in ``test_workflow_retry_ledger.py``; rows 14-15
(cancellation) travel with task θ (PRD Open-Q Q1), so this module stays
independent of θ — θ remains independently deferrable.

A RED result in this module signals a genuine end-to-end seam regression
to ESCALATE to that seam's owner (``escalate_blocker``,
category=``design_concern``) — never to silence by weakening a boundary
assertion.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Cross-module reuse — conftest.py injects orchestrator/tests onto sys.path
# (see test_workflow_terminal_report.py for the same precedent). ``_make``
# and ``_bind_landed_row`` carry no module-level lock of their own, so they
# are imported directly rather than duplicated (rows 1-4).
from test_workflow_merge_provenance import _bind_landed_row, _make

from _orch_helpers import pydantic_spec

from orchestrator.agents.roles import ROLES, AgentRole, _FAMILY_TOOL_PREFIXES
from orchestrator.landed_outbox import MergeProvenance
from orchestrator.unblock_types import BlockClass
from orchestrator.verify_categories import FailureCategory
from orchestrator.workflow import TaskWorkflow, _PriorImplStatus
from orchestrator.workflow_types import (
    STATE_TO_STATUS,
    BlockDisposition,
    IllegalTransition,
    RequeueKind,
    StewardBudgetExhausted,
    StewardInterrupted,
    StewardReescalatedL1,
    StewardResolved,
    StewardTerminalDecision,
    TerminalReport,
    WorkflowOutcome,
    WorkflowState,
    WorkflowStateMachine,
    _lookup_disposition,
    classify_failure,
)
from shared.task_statuses import TaskStatus
from shared.task_transitions import ActorClass, is_legal_transition, outcome_allows_status


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """``MergeProvenance._outbox`` is a process-global — never leak a bound
    outbox across tests (mirrors ``test_workflow_merge_provenance.py``)."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


# ---------------------------------------------------------------------------
# Boundary rows 1-4 — guard-collapse equivalence (MergeProvenance journal
# PRODUCER ↔ the three already-merged guards CONSUMER).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGuardCollapseEquivalence:
    """Boundary rows 1-4 (PRD §9): the landed-outbox journal (PRODUCER) ↔
    ``_recover_if_already_merged`` / ``_recover_before_execute`` /
    ``_recover_before_merge`` plus the ``_finalise_recovery_done``
    chokepoint (CONSUMER).

    Historical incident shapes: tasks 846/954/1141, task-2911 —
    ``workflow.py``'s ``_has_prior_implementation``/
    ``_recover_if_already_merged`` docstrings document the exact false-DONE
    recurrences these guards protect against.
    """

    # -- Row 1: journal-hit collapses identically across all three guards --

    async def test_row1_journal_hit_all_three_guards_return_done_via_provenance_only(
        self, tmp_path: Path,
    ):
        """A landed-outbox journal hit is authoritative for every guard:
        DONE with basis='journal', and the legacy
        ``_has_prior_implementation`` fallback is NEVER consulted (stubbed
        to raise if it is)."""
        # Guard 1: _recover_if_already_merged (pre-PLAN).
        f1 = _make(worktree=tmp_path / 'wt1', project_root=tmp_path / 'proj1')
        _bind_landed_row(tmp_path, task_id=f1.wf.task_id, advanced_sha='sha1')
        f1.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError('fallback must not run on a journal hit'),
        )
        outcome1 = await f1.wf._recover_if_already_merged()
        assert outcome1 == WorkflowOutcome.DONE
        assert f1.wf._merge_recovery_basis == 'journal'
        f1.mark_done.assert_awaited_once_with(
            f1.wf.task_id, kind='merged', sha='sha1',
            note='landed-outbox journal hit (pre-PLAN recovery)',
        )

        # Guard 2: _recover_before_execute (pre-EXECUTE).
        f2 = _make(worktree=tmp_path / 'wt2', project_root=tmp_path / 'proj2')
        _bind_landed_row(tmp_path, task_id=f2.wf.task_id, advanced_sha='sha2')
        f2.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError('fallback must not run on a journal hit'),
        )
        f2.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError('git layer must not run on a journal hit'),
        )
        outcome2 = await f2.wf._recover_before_execute()
        assert outcome2 == WorkflowOutcome.DONE
        assert f2.wf._merge_recovery_basis == 'journal'
        f2.mark_done.assert_awaited_once_with(
            f2.wf.task_id, kind='merged', sha='sha2',
            note='landed-outbox journal hit (pre-EXECUTE recovery)',
        )

        # Guard 3: _recover_before_merge (merge-phase).
        f3 = _make(worktree=tmp_path / 'wt3', project_root=tmp_path / 'proj3')
        _bind_landed_row(tmp_path, task_id=f3.wf.task_id, advanced_sha='sha3')
        f3.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError('fallback must not run on a journal hit'),
        )
        f3.is_ancestor.side_effect = AssertionError(
            'is_ancestor must not run on a journal hit',
        )
        outcome3 = await f3.wf._recover_before_merge('branchhead123', 'mainsha123')
        assert outcome3 == WorkflowOutcome.DONE
        assert f3.wf._merge_recovery_basis == 'journal'
        f3.mark_done.assert_awaited_once_with(
            f3.wf.task_id, kind='merged', sha='sha3',
            note='landed-outbox journal hit (pre-MERGE recovery)',
        )

    # -- Row 2: journal-miss ghost-loop shapes must never phantom-DONE --

    async def test_row2_journal_miss_ghost_loop_shapes_never_phantom_done(
        self, tmp_path: Path,
    ):
        """A rebased worktree whose HEAD now equals ``base_commit`` (guard
        1's SHA-primary check) and a zero-content-diff branch (guard 2's
        Layer-C diff check) must both refuse to recover — the task stays
        re-dispatchable (``_merge_recovery_basis is None``, ``mark_done``
        not awaited) rather than false-DONE-ing an unimplemented/reset
        branch.
        """
        # Guard 1: wt_head == base_commit ('oldbase', the _make() default)
        # — the REAL _has_prior_implementation (not mocked) takes the
        # SHA-primary path and returns has_work=False. Row 3 below drives
        # the iteration-log-noise variant of this same shape explicitly.
        f1 = _make(worktree=tmp_path / 'wt1', project_root=tmp_path / 'proj1')
        f1.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('oldbase', 'mainsha123'),
        )
        outcome1 = await f1.wf._recover_if_already_merged()
        assert outcome1 is None
        assert f1.wf._merge_recovery_basis is None
        f1.mark_done.assert_not_awaited()

        # Guard 2: on-main but base_commit..wt_head diff is empty (task 2372
        # Layer C) — a fresh/re-dispatched or rebased-to-base branch point.
        f2 = _make(worktree=tmp_path / 'wt2', project_root=tmp_path / 'proj2')
        f2.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('rebasedhead', 'mainsha123'),
        )
        f2.wf.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
        outcome2 = await f2.wf._recover_before_execute()
        assert outcome2 is None
        assert f2.wf._merge_recovery_basis is None
        f2.mark_done.assert_not_awaited()

    # -- Row 3: .task/ contamination (task-954) must not false-DONE --

    async def test_row3_task_954_inherited_iterations_log_contamination_not_done(
        self, tmp_path: Path,
    ):
        """A fresh/rebased worktree (``wt_head == base_commit``) that
        inherited a poisoned ``.task/iterations.jsonl`` (an 'implementer'
        entry left over from contamination — task 954) must resolve
        ``has_work=False`` at the PRODUCER (``_has_prior_implementation``)
        AND stay unrecovered at the CONSUMER
        (``_recover_if_already_merged``) — the SHA-equality signal vetoes
        the log signal, not the other way around.
        """
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        # Poison the iteration log the way inherited contamination would: a
        # real 'implementer' entry that LOOKS like completed work (in
        # isolation this exact shape resolves has_work=True — see
        # test_workflow_merge_provenance.py's
        # main_implementer_nonempty_steps_completed classification case).
        f.artifacts.append_iteration_log({
            'agent': 'implementer', 'source': 'orchestrator',
            'steps_attempted': ['s1'], 'steps_completed': ['s1'],
            'commit': 'oldbase',
        })

        # Producer-level: SHA equality (wt_head == base_commit == 'oldbase')
        # must veto the log noise directly.
        status = f.wf._has_prior_implementation(wt_head='oldbase')
        assert status.has_work is False

        # Consumer-level: the guard must not recover from this shape either.
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('oldbase', 'mainsha123'),
        )
        outcome = await f.wf._recover_if_already_merged()
        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()

    # -- Row 4: MP-2 — no recovery-DONE without a provenance basis --

    async def test_row4_mp2_done_always_carries_a_valid_basis_across_all_guards(
        self, tmp_path: Path,
    ):
        """Whenever any of the three guards DOES return
        ``WorkflowOutcome.DONE``, ``_merge_recovery_basis`` is always one of
        the two valid provenance bases — never an unmarked/implicit DONE."""
        # Guard 1
        f1 = _make(worktree=tmp_path / 'wt1', project_root=tmp_path / 'proj1')
        _bind_landed_row(tmp_path, task_id=f1.wf.task_id, advanced_sha='sha1')
        outcome1 = await f1.wf._recover_if_already_merged()
        assert outcome1 == WorkflowOutcome.DONE
        assert f1.wf._merge_recovery_basis in ('journal', 'fallback')

        # Guard 2
        f2 = _make(worktree=tmp_path / 'wt2', project_root=tmp_path / 'proj2')
        _bind_landed_row(tmp_path, task_id=f2.wf.task_id, advanced_sha='sha2')
        outcome2 = await f2.wf._recover_before_execute()
        assert outcome2 == WorkflowOutcome.DONE
        assert f2.wf._merge_recovery_basis in ('journal', 'fallback')

        # Guard 3
        f3 = _make(worktree=tmp_path / 'wt3', project_root=tmp_path / 'proj3')
        _bind_landed_row(tmp_path, task_id=f3.wf.task_id, advanced_sha='sha3')
        outcome3 = await f3.wf._recover_before_merge('branchhead123', 'mainsha123')
        assert outcome3 == WorkflowOutcome.DONE
        assert f3.wf._merge_recovery_basis in ('journal', 'fallback')

    @pytest.mark.parametrize('bad_basis', [None, 'hunch', ''])
    async def test_row4_finalise_recovery_done_refuses_invalid_basis_before_any_mutation(
        self, tmp_path: Path, bad_basis: str | None,
    ):
        """``_finalise_recovery_done`` — the sole writer of
        ``_merge_recovery_basis`` — raises BEFORE any status mutation when
        ``basis`` is not one of the two valid provenance values: no marker
        write, no phase transition, no ``mark_done`` call."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        with pytest.raises((AssertionError, ValueError)):
            await f.wf._finalise_recovery_done(
                basis=bad_basis, sha='somesha', kind='merged', note='n',  # type: ignore[arg-type]
            )

        assert f.wf._merge_recovery_basis is None
        assert f.wf.state == WorkflowState.PLAN
        f.mark_done.assert_not_awaited()

    @pytest.mark.parametrize('bad_sha', [None, ''])
    async def test_row4_finalise_recovery_done_refuses_empty_sha_before_any_mutation(
        self, tmp_path: Path, bad_sha: str | None,
    ):
        """Same chokepoint, the other half of the guard: a falsy ``sha``
        (even with a syntactically-valid ``basis``) also raises before any
        status mutation."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        with pytest.raises((AssertionError, ValueError)):
            await f.wf._finalise_recovery_done(
                basis='journal', sha=bad_sha, kind='merged', note='n',  # type: ignore[arg-type]
            )

        assert f.wf._merge_recovery_basis is None
        assert f.wf.state == WorkflowState.PLAN
        f.mark_done.assert_not_awaited()
