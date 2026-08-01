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
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

# Cross-module reuse — conftest.py injects orchestrator/tests onto sys.path
# (see test_workflow_terminal_report.py for the same precedent). ``_make``
# and ``_bind_landed_row`` carry no module-level lock of their own, so they
# are imported directly rather than duplicated (rows 1-4). All of these
# factories live in _workflow_helpers.py (task 2610) — none are cross-
# imported from a sibling producer's private namespace.
#
# ``_derive_meta_root_like_production`` is imported (not just the plain
# helpers) because it is an autouse fixture — autouse only auto-applies
# within a module where pytest can SEE the fixture, and a plain
# `from _workflow_helpers import AgentStub, ...` that omits its name does
# not pull that in (see test_workflow_terminal_report.py for the same
# precedent). Without it, AgentStub's legacy-only TaskArtifacts writes are
# invisible to the workflow's relocated meta_root, so a real run() never
# reaches DONE (rows 5-6).
from _workflow_helpers import (
    AgentStub,
    FakeBriefing,
    FakeMcp,
    FakeScheduler,
    _bind_landed_row,
    _build_harness,
    _build_workflow,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see above
    _init_git_repo,
    _init_repo,
    _make,
    _make_warmlane_workflow,
)
from escalation.models import Escalation
from shared.task_statuses import TaskStatus
from shared.task_transitions import ActorClass, is_legal_transition, outcome_allows_status

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import _FAMILY_TOOL_PREFIXES, ROLES, AgentRole
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import TaskReport
from orchestrator.landed_outbox import MergeProvenance
from orchestrator.scheduler import TaskAssignment
from orchestrator.steward import TaskSteward
from orchestrator.unblock_types import BlockClass
from orchestrator.verify import VerifyResult
from orchestrator.verify_categories import FailureCategory
from orchestrator.workflow import TaskWorkflow, WorkflowMetrics
from orchestrator.workflow_types import (
    STATE_TO_STATUS,
    BlockDisposition,
    IllegalTransition,
    RequeueKind,
    StewardInterrupted,
    StewardResolved,
    StewardTerminalDecision,
    TerminalReport,
    WorkflowOutcome,
    WorkflowState,
    WorkflowStateMachine,
    _lookup_disposition,
    classify_failure,
)


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
        # Pin that f3.is_ancestor really IS the call site
        # _recover_before_merge consults (self.git_ops.is_ancestor) — not
        # merely a same-named _Fixture attribute that happens to be
        # unconnected. Without this, a future _make() shape change could
        # leave the side_effect below mutating a mock nothing actually
        # calls, turning the negative guard vacuous (false-green).
        assert f3.wf.git_ops.is_ancestor is f3.is_ancestor
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
        # Explicit confirmation independent of the side_effect ever firing
        # correctly — the guard's real contract is "never called", not
        # merely "raises if called".
        f3.is_ancestor.assert_not_called()

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


# ---------------------------------------------------------------------------
# Boundary rows 5-6 — state-machine legality + run()-exit outcome<->status
# consistency (WorkflowStateMachine/STATE_TO_STATUS PRODUCER ↔
# shared.task_transitions W2 AUTHORITY, and a REAL TaskWorkflow.run() CONSUMER).
# ---------------------------------------------------------------------------

# e2e-style fixtures for row 6's real run() drivers — duplicated verbatim
# rather than imported (established repo convention, mirrors
# test_workflow_terminal_report.py / test_repend_state_machine.py: these are
# plain, unlocked fixtures with no shared module-level state, so each
# dependent test file keeps its own copy instead of coupling to
# test_workflow_e2e.py's fixture graph).


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
class TestStateMachineLegalityAndConsistency:
    """Boundary rows 5-6 (PRD §9): ``WorkflowStateMachine``/``STATE_TO_STATUS``
    (PRODUCER) ↔ ``shared.task_transitions`` — the W2 AUTHORITY — plus a REAL
    ``TaskWorkflow.run()`` exit as CONSUMER of both.
    """

    # -- Row 5 (SM-1): terminal absorption + never-a-fourth-table delegation --

    async def test_row5_done_to_blocked_raises_and_state_unchanged(self):
        """The workflow.py:7744 'already DONE, ignoring late blocked' case,
        as a pure ``WorkflowStateMachine`` property."""
        machine = WorkflowStateMachine(WorkflowState.DONE)
        with pytest.raises(IllegalTransition):
            machine.transition(WorkflowState.BLOCKED)
        assert machine.state == WorkflowState.DONE

    @pytest.mark.parametrize('absorbing', [WorkflowState.DONE, WorkflowState.CANCELLED])
    @pytest.mark.parametrize('to', [
        WorkflowState.BLOCKED, WorkflowState.PLAN, WorkflowState.EXECUTE,
    ])
    async def test_row5_absorbing_states_reject_every_out_transition(self, absorbing, to):
        """DONE and CANCELLED are BOTH absorbing — every out-transition raises."""
        machine = WorkflowStateMachine(absorbing)
        with pytest.raises(IllegalTransition):
            machine.transition(to)
        assert machine.state == absorbing

    @pytest.mark.parametrize('absorbing', [WorkflowState.DONE, WorkflowState.CANCELLED])
    async def test_row5_absorbing_same_state_is_a_legal_noop(self, absorbing):
        machine = WorkflowStateMachine(absorbing)
        machine.transition(absorbing)
        assert machine.state == absorbing

    @pytest.mark.parametrize(('frm', 'to'), [
        # Linear phase advance / completion (same-projected-status or a
        # legal working -> DONE move).
        (WorkflowState.PLAN, WorkflowState.EXECUTE),
        (WorkflowState.MERGE, WorkflowState.DONE),
        # Block / unblock from a working phase.
        (WorkflowState.PLAN, WorkflowState.BLOCKED),
        (WorkflowState.BLOCKED, WorkflowState.CANCELLED),
        # Terminal absorption (row 5's own case, re-verified via delegation).
        (WorkflowState.DONE, WorkflowState.BLOCKED),
        (WorkflowState.CANCELLED, WorkflowState.PLAN),
        # Non-terminal but genuinely absent from the shared union.
        (WorkflowState.MERGE_DEFERRED, WorkflowState.EXECUTE),
    ])
    async def test_row5_transition_raises_iff_shared_table_says_illegal(self, frm, to):
        """Proves ``transition`` DELEGATES to ``is_legal_transition`` over the
        ``STATE_TO_STATUS``-projected pair — the machine consumes W2's table,
        never a fourth (G4 decision #1)."""
        expected_legal = is_legal_transition(
            STATE_TO_STATUS[frm], STATE_TO_STATUS[to], ActorClass.ORCHESTRATOR,
        )
        machine = WorkflowStateMachine(frm)
        if expected_legal:
            machine.transition(to)
            assert machine.state == to
        else:
            with pytest.raises(IllegalTransition):
                machine.transition(to)
            assert machine.state == frm

    # -- Row 6 (SM-2): real run()-exit outcome<->status + phase==machine.state --

    async def test_row6_done_run_is_consistent_and_phase_matches_machine(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Clean merged-DONE run (mirrors test_workflow_e2e.TestHappyPath)."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        report = await workflow.run()

        last_status = await scheduler.get_status(workflow.task_id)
        assert last_status == 'done'
        assert outcome_allows_status(report.outcome, last_status)
        assert report.phase == workflow.machine.state

    async def test_row6_allaccountscapped_blocked_run_is_consistent(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Blocked run via the AllAccountsCappedException path (e2e:5638)."""
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]',
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        report = await workflow.run()

        last_status = await scheduler.get_status(workflow.task_id)
        assert last_status == 'blocked'
        assert outcome_allows_status(report.outcome, last_status)
        assert report.phase == workflow.machine.state

    async def test_row6_warmlane_requeued_run_is_consistent(self, tmp_path: Path):
        """REQUEUED run (reuses test_workflow_warm_lane_requeue setup)."""
        from orchestrator.git_ops import WarmLanePoolHardDown

        wf = _make_warmlane_workflow(tmp_path=tmp_path)
        wf.git_ops.create_worktree = AsyncMock(  # type: ignore[method-assign]
            side_effect=WarmLanePoolHardDown(
                "warm-lane base absent (host-scoped pool hard-down) for branch "
                "'1859'; requeue",
            ),
        )
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        report = await wf.run()

        last_status = await wf.scheduler.get_status(wf.task_id)
        assert last_status == 'pending'
        assert outcome_allows_status(report.outcome, last_status)
        assert report.phase == wf.machine.state
        mark_blocked.assert_not_awaited()

    async def test_row6_outcome_status_divergence_raises_loudly_naming_both(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """NEGATIVE: DB row says 'done' while the actual exit is BLOCKED.

        The pre-empt check at the top of ``_drive()`` also calls
        ``scheduler.get_status`` — it must see a NON-terminal status there,
        or the task exits CANCELLED before ever reaching the AllAccountsCapped
        block path. Only the SECOND call (SM-2's own read, after ``_drive()``
        returns) sees the injected 'done', simulating a stale/incorrect DB
        row at the terminal boundary.
        """
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        calls = {'n': 0}

        async def fake_get_status(task_id):
            calls['n'] += 1
            return 'pending' if calls['n'] == 1 else 'done'

        monkeypatch.setattr(scheduler, 'get_status', fake_get_status)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]',
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        with pytest.raises((AssertionError, ValueError)) as exc_info:
            await workflow.run()

        message = str(exc_info.value).lower()
        assert 'blocked' in message
        assert 'done' in message

    async def test_row6_none_status_does_not_crash_normal_run(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Fail-safe guard: a None (unreadable) status must not crash run()."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )
        monkeypatch.setattr(scheduler, 'get_status', AsyncMock(return_value=None))

        report = await workflow.run()

        assert report.outcome == WorkflowOutcome.DONE

    async def test_row6_out_of_vocabulary_status_does_not_crash_normal_run(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Fail-safe guard: an out-of-vocabulary status string must not crash
        run() either — ``outcome_allows_status`` raises ``ValueError`` on it,
        which the run()-exit guard must catch rather than let escape."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )
        monkeypatch.setattr(
            scheduler, 'get_status', AsyncMock(return_value='not-a-real-status'),
        )

        report = await workflow.run()

        assert report.outcome == WorkflowOutcome.DONE


# ---------------------------------------------------------------------------
# Boundary row 7 — TerminalReport (workflow.run() PRODUCER ↔ harness
# _run_slot CONSUMER), TR-1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHarnessConsumesTerminalReport:
    """Boundary row 7 (PRD §9): ``TaskWorkflow.run()``'s returned
    ``TerminalReport`` (PRODUCER) ↔ ``Harness._run_slot``'s consumed
    ``TaskReport`` (CONSUMER), TR-1.
    """

    async def _run_stubbed_slot(
        self, tmp_path: Path, report: TerminalReport, *, task_id: str,
    ) -> TaskReport:
        """Drive ``Harness._run_slot`` with ``TaskWorkflow`` patched to
        return ``report`` directly from ``run()`` (no real agent/git work) —
        mirrors test_workflow_terminal_report.py's ``_run_stubbed_slot``.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)
        config = OrchestratorConfig(project_root=repo, max_concurrent_tasks=1)
        harness = _build_harness(config)
        # `_build_harness` leaves `Scheduler` a bare MagicMock (only the
        # liveness accessors are wired) — `is_deterministic` must be pinned
        # False or the MagicMock auto-mock (truthy) would route this through
        # `_run_deterministic_slot` instead of the TaskWorkflow path below.
        harness.scheduler.is_deterministic.return_value = False  # type: ignore[attr-defined]
        # `_apply_retry_cap` compares these against real ints from `config`
        # (`count >= self.config.requeue_cap`) whenever outcome==REQUEUED —
        # an unconfigured MagicMock return value would raise TypeError on
        # that comparison and mask the finally block's real work.
        harness.scheduler.record_requeue.return_value = 0  # type: ignore[attr-defined]
        harness.scheduler.transient_requeue_count.return_value = 0  # type: ignore[attr-defined]

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = MagicMock()
            mock_wf.run = AsyncMock(return_value=report)
            mock_wf._steward = None
            mock_wf.metrics = WorkflowMetrics()
            MockWorkflow.return_value = mock_wf

            assignment = TaskAssignment(
                task_id=task_id,
                task={
                    'id': task_id, 'title': 'Test task', 'status': 'pending',
                    'metadata': {}, 'dependencies': [],
                },
                modules=[],
            )
            sem = asyncio.Semaphore(1)
            result = await harness._run_slot(assignment, sem)

        assert result is not None
        return result

    # -- Consumer: TaskReport.block_* maps from the returned TerminalReport --

    async def test_row7_mark_blocked_exit_maps_block_phase_to_working_phase(
        self, tmp_path: Path,
    ):
        """A ``_mark_blocked``-shape exit (``phase=BLOCKED``,
        ``blocked_from_phase=VERIFY``) surfaces ``block_phase == 'verify'``
        — the PRE-block WORKING phase, NOT the terminal ``phase``."""
        report = TerminalReport(
            outcome=WorkflowOutcome.BLOCKED,
            reason='verify failed',
            phase=WorkflowState.BLOCKED,
            detail='d',
            category=None,
            blocked_from_phase=WorkflowState.VERIFY,
        )
        task_report = await self._run_stubbed_slot(tmp_path, report, task_id='701')

        assert task_report.outcome == WorkflowOutcome.BLOCKED
        assert task_report.block_reason == report.reason
        assert task_report.block_detail == report.detail
        assert task_report.block_phase == 'verify'

    async def test_row7_warm_lane_requeued_exit_keeps_block_phase_plan(
        self, tmp_path: Path,
    ):
        """A warm-lane REQUEUED exit (``blocked_from_phase=PLAN``, no BLOCKED
        transition) keeps ``block_phase == 'plan'``."""
        report = TerminalReport(
            outcome=WorkflowOutcome.REQUEUED,
            reason='warm_lane_pool_hard_down',
            phase=WorkflowState.PLAN,
            detail='d',
            category=None,
            blocked_from_phase=WorkflowState.PLAN,
        )
        task_report = await self._run_stubbed_slot(tmp_path, report, task_id='702')

        assert task_report.outcome == WorkflowOutcome.REQUEUED
        assert task_report.block_reason == report.reason
        assert task_report.block_phase == 'plan'

    async def test_row7_clean_done_exit_has_empty_block_phase(self, tmp_path: Path):
        """A clean DONE exit (``blocked_from_phase`` defaults to ``None``)
        maps to ``block_phase == ''``."""
        report = TerminalReport(
            outcome=WorkflowOutcome.DONE,
            reason='',
            phase=WorkflowState.DONE,
            detail='',
            category=None,
        )
        task_report = await self._run_stubbed_slot(tmp_path, report, task_id='703')

        assert task_report.outcome == WorkflowOutcome.DONE
        assert task_report.block_phase == ''

    # -- Producer + TR-1 behavioral pin: no _last_block_* side channel --

    async def test_row7_no_last_block_side_channel_survives_a_real_blocked_run(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """After a REAL blocked run (AllAccountsCappedException via
        ``_build_workflow``+``AgentStub``), the ``TaskWorkflow`` instance
        carries NO ``_last_block_reason``/``_last_block_detail``/
        ``_last_block_phase`` attribute — the full block context is
        reachable ONLY through the returned ``TerminalReport``."""
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]',
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        report = await workflow.run()

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert not hasattr(workflow, '_last_block_reason')
        assert not hasattr(workflow, '_last_block_detail')
        assert not hasattr(workflow, '_last_block_phase')
        # Full block context still reachable via the returned report alone.
        assert report.reason.lower().startswith('all accounts capped')
        assert report.detail
        assert report.phase == workflow.machine.state


# ---------------------------------------------------------------------------
# Boundary rows 8-9 — StewardOutcome routing (steward ``_publish_outcome``/
# ``_handle_escalation`` PRODUCER ↔ workflow ``_await_steward_completion``/
# ``_mark_blocked`` CONSUMER), SO-1 / task 2060.
# ---------------------------------------------------------------------------


def _make_workflow(
    *, tmp_path: Path, task_id: str = '2253', with_escalation_queue: bool = False,
) -> TaskWorkflow:
    """Minimal ``TaskWorkflow`` harness for rows 8-9 — mirrors
    ``test_steward_outcome.py``'s ``_make_workflow`` (this module's file-level
    lock does not cover that sibling, so the factory is duplicated per the
    established 26+-file convention rather than imported).
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    config.merge_train_former_enabled = False
    config.merge_train_max_members = 3
    # _spawn_dry_run_unblock runs unconditionally from the top of
    # _mark_blocked — keep it inert so rows 8-9 exercise only the
    # steward-outcome dispatch, not the dry-run-unblock hook.
    config.unblock_auto.enabled = False

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / '.task').mkdir(exist_ok=True)
    wf.artifacts = TaskArtifacts(worktree)
    wf.worktree = worktree
    wf.merge_queue = MagicMock()
    wf.plan = {'files': ['a.py', 'b.py', 'c.py']}
    wf._base_commit = 'base_sha'
    wf._module_configs = []
    wf.git_ops.rebind_branch_to_head = AsyncMock(return_value=True)
    wf.event_store = None

    if with_escalation_queue:
        eq = MagicMock()
        eq.has_open_l1.return_value = False
        eq.make_id.return_value = f'esc-{task_id}-1'
        wf.escalation_queue = eq

    return wf


def _make_steward(*, worktree: Path) -> TaskSteward:
    """Minimal ``TaskSteward`` harness for row 9's producer-side coverage —
    trims ``test_steward.py``'s ``steward``/``mock_config``/``mock_queue``/
    ``mock_mcp``/``mock_briefing`` fixture graph down to a single factory
    (this module's file lock does not cover that sibling either).
    """
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / '.task').mkdir(exist_ok=True)

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.steward_max_attempts = 1
    config.steward_lifetime_budget = 12.0
    config.steward_max_timeouts_per_escalation = 3
    config.steward_max_empty_outputs_per_escalation = 2

    queue = MagicMock()
    queue.get_by_task.return_value = []
    queue.get.return_value = None
    queue.make_id.return_value = 'esc-42-99'

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}

    briefing = AsyncMock()

    return TaskSteward(
        task_id='42',
        task={'id': '42', 'title': 'Test Task', 'description': 'A test'},
        worktree=worktree,
        config=config,
        mcp=mcp,
        escalation_queue=queue,
        briefing=briefing,
    )


def _make_escalation(**overrides) -> Escalation:
    defaults: dict = dict(
        id='esc-42-1',
        task_id='42',
        agent_role='orchestrator',
        severity='blocking',
        category='limit_exhausted',
        summary='execute limit exhausted',
    )
    defaults.update(overrides)
    return Escalation(**defaults)


@pytest.mark.asyncio
class TestStewardOutcomeRouting:
    """Boundary rows 8-9 (PRD §9): steward ``_publish_outcome``/
    ``_handle_escalation`` (PRODUCER) ↔ workflow ``_await_steward_completion``/
    ``_mark_blocked`` (CONSUMER), SO-1 / task 2060.
    """

    # -- Row 8: StewardResolved ---------------------------------------------

    async def test_row8_await_steward_completion_returns_channel_published_resolved(
        self, tmp_path: Path,
    ):
        wf = _make_workflow(tmp_path=tmp_path)
        wf._steward_outcome_channel = asyncio.Queue()
        wf._steward_outcome_channel.put_nowait(
            StewardResolved(resolution_text='fixed the flaky import'),
        )
        wf.scheduler.get_status = AsyncMock(return_value='blocked')

        outcome = await wf._await_steward_completion()

        assert outcome == StewardResolved(resolution_text='fixed the flaky import')

    async def test_row8_mark_blocked_routes_resolved_to_requeued_via_pending(
        self, tmp_path: Path,
    ):
        """The SINGLE isinstance dispatch (workflow.py:8685-8759):
        StewardResolved -> REQUEUED via ``_requeue()``/pending, with NO
        timestamp-window/queue-forensics read — ``escalation_queue.get_by_task``
        is never consulted for this outcome."""
        wf = _make_workflow(tmp_path=tmp_path, with_escalation_queue=True)
        wf._steward = MagicMock()
        wf._steward_outcome_channel = asyncio.Queue()
        wf._steward_outcome_channel.put_nowait(
            StewardResolved(resolution_text='fixed it'),
        )
        wf.scheduler.get_status = AsyncMock(return_value='blocked')

        outcome = await wf._mark_blocked('agent hit a transient failure')

        assert outcome == WorkflowOutcome.REQUEUED
        wf.scheduler.set_task_status.assert_any_call(  # type: ignore[attr-defined]
            wf.task_id, 'pending',
        )
        wf.escalation_queue.get_by_task.assert_not_called()  # type: ignore[union-attr]

    # -- Row 8: StewardTerminalDecision --------------------------------------

    @pytest.mark.parametrize('status', ['done', 'cancelled', 'deferred'])
    async def test_row8_await_steward_completion_synthesizes_terminal_decision_from_scheduler_status(
        self, tmp_path: Path, status: str,
    ):
        """``StewardTerminalDecision`` is NEVER published on the channel — it
        is SYNTHESIZED from a single fresh ``scheduler.get_status`` read that
        always overrides whatever (if anything) the channel produced (the
        pre-W9-delta terminal/deferred-wins ordering, preserved)."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf._steward_outcome_channel = asyncio.Queue()
        # An unrelated outcome sits on the channel; the terminal/deferred
        # status read still wins.
        wf._steward_outcome_channel.put_nowait(
            StewardResolved(resolution_text='irrelevant — status overrides'),
        )
        wf.scheduler.get_status = AsyncMock(return_value=status)

        outcome = await wf._await_steward_completion()

        assert outcome == StewardTerminalDecision(new_status=TaskStatus(status))

    async def test_row8_mark_blocked_routes_terminal_decision_done_to_done(
        self, tmp_path: Path,
    ):
        wf = _make_workflow(tmp_path=tmp_path, with_escalation_queue=True)
        wf._steward = MagicMock()
        wf._steward_outcome_channel = asyncio.Queue()
        wf._steward_outcome_channel.put_nowait(
            StewardResolved(resolution_text='irrelevant — status overrides'),
        )
        wf.scheduler.get_status = AsyncMock(return_value='done')

        outcome = await wf._mark_blocked('agent hit a transient failure')

        assert outcome == WorkflowOutcome.DONE
        assert wf.state == WorkflowState.DONE
        wf.escalation_queue.get_by_task.assert_not_called()  # type: ignore[union-attr]

    async def test_row8_mark_blocked_routes_terminal_decision_non_done_to_blocked_preserved(
        self, tmp_path: Path,
    ):
        """A steward-adjacent terminal decision that is NOT 'done'
        (cancelled/deferred) preserves the status rather than re-queueing —
        no L1, since this is a legitimate steward-observed decision, not a
        failure to resolve."""
        wf = _make_workflow(tmp_path=tmp_path, with_escalation_queue=True)
        wf._steward = MagicMock()
        wf._steward_outcome_channel = asyncio.Queue()
        wf._steward_outcome_channel.put_nowait(
            StewardResolved(resolution_text='irrelevant — status overrides'),
        )
        wf.scheduler.get_status = AsyncMock(return_value='cancelled')
        wf._ensure_l1_escalation_for_blocked = AsyncMock()

        outcome = await wf._mark_blocked('agent hit a transient failure')

        assert outcome == WorkflowOutcome.BLOCKED
        wf._ensure_l1_escalation_for_blocked.assert_not_awaited()

    # -- Row 9 (task 2060): StewardInterrupted(attempt_cap, wip=True) -------

    async def test_row9_mark_blocked_wip_interrupted_dismisses_l0_and_requeues_not_l1(
        self, tmp_path: Path,
    ):
        """workflow.py:8716-8742's resume-plan path: dismiss the still-
        pending L0 and re-pend — NOT an L1 re-escalation."""
        wf = _make_workflow(tmp_path=tmp_path, with_escalation_queue=True)
        wf._steward = MagicMock()
        wf._steward_outcome_channel = asyncio.Queue()
        wf._steward_outcome_channel.put_nowait(
            StewardInterrupted('attempt_cap', wip_commits_present=True),
        )
        wf.scheduler.get_status = AsyncMock(return_value='blocked')
        pending_l0 = MagicMock(id='esc-orig-0')
        wf.escalation_queue.get_by_task.return_value = [pending_l0]  # type: ignore[union-attr]
        wf._ensure_l1_escalation_for_blocked = AsyncMock()

        outcome = await wf._mark_blocked('agent kept hitting the retry cap')

        assert outcome == WorkflowOutcome.REQUEUED
        wf.escalation_queue.resolve.assert_called_once()  # type: ignore[union-attr]
        call = wf.escalation_queue.resolve.call_args  # type: ignore[union-attr]
        assert call.args[0] == 'esc-orig-0'
        assert call.kwargs.get('dismiss') is True
        assert call.kwargs.get('resolved_by') == 'auto-dismissed'
        wf.scheduler.set_task_status.assert_any_call(  # type: ignore[attr-defined]
            wf.task_id, 'pending',
        )
        wf._ensure_l1_escalation_for_blocked.assert_not_awaited()

    # -- Row 9 producer: _ensure_steward_started wiring ----------------------

    async def test_row9_ensure_steward_started_wires_outcome_channel_and_wip_probe(
        self, tmp_path: Path,
    ):
        """The producer wiring choke point: a freshly-built steward is
        handed THIS workflow's outcome channel and wip probe — the SAME
        primitive the workflow uses for its own grace-timeout derivation
        (:meth:`TaskWorkflow._worktree_has_wip_commits`)."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf._steward = None
        wf._steward_outcome_channel = None
        mock_steward = MagicMock()
        mock_steward.start = AsyncMock()
        wf._steward_factory = MagicMock(return_value=mock_steward)
        wf.escalation_queue = MagicMock()
        wf.escalation_queue.get_by_task.return_value = [MagicMock()]

        await wf._ensure_steward_started()

        assert wf._steward is mock_steward
        wf._steward_factory.assert_called_once_with(wf.worktree, wf._config_dir)
        assert isinstance(wf._steward_outcome_channel, asyncio.Queue)
        mock_steward.set_outcome_channel.assert_called_once_with(
            wf._steward_outcome_channel,
        )
        mock_steward.set_wip_probe.assert_called_once_with(
            wf._worktree_has_wip_commits,
        )
        mock_steward.start.assert_awaited_once()

    # -- Row 9 producer: steward attempt-cap branch, wip=True ---------------

    async def test_row9_steward_attempt_cap_wip_true_publishes_directly_skips_l1(
        self, steward_worktree: Path,
    ):
        """When the per-escalation retry cap fires with WIP present, the
        steward publishes the wip-gated ``StewardInterrupted`` DIRECTLY —
        ``_auto_escalate_to_human`` (and therefore any L1 filing) is skipped
        entirely (task-2060 fix)."""
        steward = _make_steward(worktree=steward_worktree)
        channel = asyncio.Queue()
        steward.set_outcome_channel(channel)
        steward.set_wip_probe(AsyncMock(return_value=True))
        esc = _make_escalation(id='esc-42-1')
        steward._retry_counts['esc-42-1'] = 1  # at cap (>= steward_max_attempts)

        with patch(
            'orchestrator.steward.invoke_agent', new_callable=AsyncMock,
        ) as mock_invoke:
            await steward._handle_escalation(esc)

        mock_invoke.assert_not_called()
        steward.escalation_queue.submit.assert_not_called()  # type: ignore[attr-defined]  # no L1 filed
        assert channel.get_nowait() == StewardInterrupted(
            reason='attempt_cap', wip_commits_present=True,
        )

    # -- Row 9 cancel-safety guard --------------------------------------------

    async def test_row9_cancel_event_set_forces_no_wip_so_mark_blocked_skips_resume_branch(
        self, tmp_path: Path,
    ):
        """Amendment guard, exercised through the FULL ``_mark_blocked``
        dispatch (not just ``_await_steward_completion``'s return value): a
        soft-cancel in flight must never be routed into the task-2060
        resume-plan branch, even when the worktree genuinely carries WIP —
        ``_cancel_event`` forces ``wip_commits_present=False`` at the
        synthesis choke point, so the channel-empty case falls through to
        the generic BLOCKED+L1 path instead of dismiss-and-requeue."""
        wf = _make_workflow(tmp_path=tmp_path, with_escalation_queue=True)
        wf._steward = MagicMock()
        wf._steward_outcome_channel = asyncio.Queue()  # nothing published
        wf._worktree_has_wip_commits = AsyncMock(return_value=True)
        wf.scheduler.get_status = AsyncMock(return_value='blocked')
        wf.escalation_queue.get_by_task.return_value = []  # type: ignore[union-attr]
        wf._ensure_l1_escalation_for_blocked = AsyncMock()
        wf._cancel_event.set()

        outcome = await wf._mark_blocked('agent kept hitting the retry cap')

        assert outcome == WorkflowOutcome.BLOCKED
        wf._ensure_l1_escalation_for_blocked.assert_awaited_once()
        wf.scheduler.set_task_status.assert_called_once_with(  # type: ignore[attr-defined]
            wf.task_id, 'blocked',
        )


# ---------------------------------------------------------------------------
# Boundary rows 10-11 — BlockDisposition (``classify_failure`` PRODUCER ↔ the
# four independent ``AllAccountsCappedException`` cap-catch-site CONSUMERs),
# BD-1 / BD-2.
# ---------------------------------------------------------------------------


def _public_exception_types(module):
    """Every public (no leading underscore) BaseException subclass *module*
    itself defines (``obj.__module__ == module.__name__``) — deliberately
    excludes exceptions merely imported into the module's namespace (e.g. a
    top-level ``from x import SomeError``), since those are owned/exported
    by whichever module defines them, not this one, and requiring a row for
    them here would be a spurious coupling to an unrelated module's surface.

    Duplicated verbatim from test_block_disposition.py's helper of the same
    name (established repo convention — this module re-derives the sweep
    independently rather than importing it, so row 11 stays a genuine
    boundary gate rather than a re-import of the ε seam's own test infra).
    """
    return [
        obj for name, obj in vars(module).items()
        if not name.startswith('_')
        and inspect.isclass(obj)
        and issubclass(obj, BaseException)
        and obj.__module__ == module.__name__
    ]


class TestBlockDispositionOneClassifierAndCompleteness:
    """Boundary rows 10-11 (PRD §9): ``classify_failure`` (PRODUCER) ↔ the
    four independent ``AllAccountsCappedException`` cap-catch sites —
    ``workflow.run()``, ``steward._pre_triage_suggestions``,
    ``review_checkpoint.run_focused``, ``dry_run_unblock.run_dry_run_unblock``
    (CONSUMERs) — plus the BD-2 completeness sweep, BD-1 / BD-2.
    """

    # -- Row 10 (BD-1): canonical, type-only classification ------------------

    def test_row10_two_distinct_cap_instances_classify_identically(self):
        """classify_failure is a function of the exception's TYPE alone —
        payload (retries/elapsed_secs/label) never affects the resolved
        disposition, so all four cap sites below are guaranteed to consult
        an IDENTICAL BlockDisposition regardless of instance payload."""
        from shared.cli_invoke import AllAccountsCappedException

        exc_a = AllAccountsCappedException(retries=1, elapsed_secs=10.0, label='A')
        exc_b = AllAccountsCappedException(retries=99, elapsed_secs=9999.0, label='B')
        assert classify_failure(exc_a) == classify_failure(exc_b)

    # -- Row 10 (BD-1): workflow.run() consults the REAL shared classifier ---

    @pytest.mark.asyncio
    async def test_row10_workflow_run_consults_real_classifier_for_cap_reason(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """The workflow.py cap-catch site: a real blocked run's
        ``report.reason`` and the ``_mark_blocked(escalate_to_human=...)``
        kwarg both trace back to the SAME
        ``classify_failure(AllAccountsCappedException)`` row pinned by
        test_block_disposition.py's ``TestClassifyFailureKnownRows`` — not an
        independent literal."""
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]',
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )
        mark_blocked_spy = AsyncMock(wraps=workflow._mark_blocked)
        workflow._mark_blocked = mark_blocked_spy  # type: ignore[method-assign]

        report = await workflow.run()

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason.lower().startswith('all accounts capped')
        assert report.category is FailureCategory.NONE
        assert mark_blocked_spy.await_args is not None
        assert mark_blocked_spy.await_args.kwargs.get('escalate_to_human', False) is False

    # -- Row 10 (BD-1): the other three sites consult the SAME classifier ----
    # (sentinel-patched — proves consultation rather than a coincidental
    # literal match; reuses each sibling seam's own exposed helper factories,
    # imported function-locally — a local `from X import Y` binds Y only in
    # this function's namespace, so it does not collide with this module's
    # own rows-8-9 `_make_steward`/`_make_escalation` module-level names.)

    @pytest.mark.asyncio
    async def test_row10_steward_pre_triage_consults_shared_classifier(self, caplog, steward_worktree):
        import json
        import logging

        from shared.cli_invoke import AllAccountsCappedException
        from test_suggestion_triage import _make_escalation, _make_steward, _make_suggestions

        steward = _make_steward(worktree=steward_worktree)
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
            'orchestrator.steward.invoke_with_cap_retry', AsyncMock(side_effect=cap_exc),
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
    async def test_row10_review_checkpoint_consults_shared_classifier(
        self, monkeypatch, caplog,
    ):
        import logging

        from shared.cli_invoke import AllAccountsCappedException
        from test_review_checkpoint_cap import _PHASE1_RESULT, _make_checkpoint

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
    async def test_row10_dry_run_unblock_consults_shared_classifier(self, tmp_path: Path):
        from shared.cli_invoke import AllAccountsCappedException
        from test_dry_run_unblock import _make_config

        from orchestrator.dry_run_unblock import run_dry_run_unblock

        cap_exc = AllAccountsCappedException(
            retries=3, elapsed_secs=1800.0, label='Task 42 [unblock_auto]',
        )
        # block_class deliberately differs from AGENT_FAILURE (what
        # classify_block_reason would derive from the reason string) so the
        # assertion below cannot pass by coincidence.
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
        assert entry['block_class'] == 'merge_verify_red', (
            f'Expected block_class from classify_failure(cap_exc), got: '
            f'{entry["block_class"]!r}'
        )

    # -- Row 11 (BD-2): completeness sweep across the four BD-2 modules ------

    def test_row11_every_exported_exception_across_four_modules_has_a_row(self):
        import shared.cli_invoke as cli_invoke
        import shared.usage_gate as usage_gate

        import orchestrator.git_ops as git_ops
        import orchestrator.verify as verify

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

    def test_row11_brand_new_exception_type_has_no_row_but_classify_stays_total(self):
        """A synthetic type with no table row proves the completeness check
        above is meaningful: it FAILS for an unrecognized type rather than
        silently matching everything, while classify_failure itself is still
        TOTAL — it falls back to the default disposition rather than
        raising."""

        class _BrandNewFailure(Exception):
            pass

        assert _lookup_disposition(_BrandNewFailure) is None
        disp = classify_failure(_BrandNewFailure('surprise'))
        assert disp is not None


# ---------------------------------------------------------------------------
# Boundary row 12 — capability wiring (roles.py AgentRole.__post_init__
# PRODUCER ↔ workflow._invoke gating CONSUMER), CW-1.
# ---------------------------------------------------------------------------


async def _invoke_probe(
    role: AgentRole, config: OrchestratorConfig, git_ops: GitOps,
    task_assignment: TaskAssignment,
) -> tuple[Mapping[str, Any], TaskWorkflow, Path]:
    """Build a TaskWorkflow, patch invoke_with_cap_retry, invoke ``_invoke``.

    Duplicated (not imported) from test_agent_capability_wiring.py's helper
    of the same name — that module's own η-seam unit suite already covers
    this exhaustively; this boundary module re-derives the probe
    independently so row 12 stays a genuine two-way gate (mirrors this
    module's rows 5-6 duplicated git_repo/config/git_ops/task_assignment
    fixtures — same rationale, see their docstring above).

    Returns (call_kwargs, workflow, cwd) — the kwargs invoke_with_cap_retry
    was awaited with, the workflow instance (so callers can assert against
    workflow.modules), and the worktree path (a real linked worktree, so the
    row-12 write-set assertion can check the worktree root is carved in).
    """
    wt_info = await git_ops.create_worktree(task_assignment.task_id)
    cwd = wt_info.path

    config.sandbox.enabled = True

    workflow = TaskWorkflow(
        assignment=task_assignment,
        config=config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),
    )
    workflow.artifacts = None

    with patch(
        'orchestrator.workflow.invoke_with_cap_retry',
        new_callable=AsyncMock,
        return_value=AgentResult(success=True, output=''),
    ) as mock_cap_retry:
        await workflow._invoke(role, 'PROMPT', cwd)

    assert mock_cap_retry.await_count == 1, 'invoke_with_cap_retry must be called once'
    assert mock_cap_retry.await_args is not None
    return mock_cap_retry.await_args.kwargs, workflow, cwd


class TestCapabilityWiringImportAssert:
    """Boundary row 12 (PRD §9): ``AgentRole.__post_init__``'s import-time
    capability assertion (PRODUCER) ↔ ``TaskWorkflow._invoke``'s role-derived
    sandbox/MCP gating (CONSUMER), CW-1 — the SIMPLE_TASK silent-fallthrough
    regression class (reify esc-4943-54) now fails LOUDLY at import instead
    of weeks later.
    """

    # -- Producer: import-time/construction capability assertion -------------

    def test_row12_plan_tools_tool_without_family_raises_naming_plan_tools(self):
        with pytest.raises(ValueError, match='plan_tools'):
            AgentRole(
                name='probe', system_prompt='x',
                allowed_tools=['mcp__plan-tools__create_plan'],
            )

    def test_row12_fused_memory_tool_without_family_raises_naming_orchestrator(self):
        with pytest.raises(ValueError, match='orchestrator'):
            AgentRole(
                name='probe', system_prompt='x',
                allowed_tools=['mcp__fused-memory__search'],
            )

    def test_row12_escalation_tool_without_family_raises_naming_orchestrator(self):
        with pytest.raises(ValueError, match='orchestrator'):
            AgentRole(
                name='probe', system_prompt='x',
                allowed_tools=['mcp__escalation__escalate_info'],
            )

    def test_row12_unmapped_family_drift_guard_raises_naming_the_family(self):
        """_FAMILY_TOOL_PREFIXES drift guard: a family with no prefix-mapping
        entry must fail loudly rather than silently skip validation for it."""
        with pytest.raises(ValueError, match='unmapped_family'):
            AgentRole(
                name='probe', system_prompt='x',
                mcp_families=frozenset({'unmapped_family'}),  # type: ignore[arg-type]
            )

    def test_row12_roles_module_imports_cleanly_every_shipped_role_passes(self):
        """Import-time firing: every shipped ROLES entry's __post_init__
        passes — this module's own top-of-file
        ``from orchestrator.agents.roles import ... ROLES ...`` already
        proved the module imports cleanly; this test re-derives the
        invariant explicitly against the live ROLES/_FAMILY_TOOL_PREFIXES
        objects rather than relying on import success alone."""
        assert len(ROLES) >= 9
        assert set(ROLES) >= {
            'architect', 'implementer', 'debugger', 'merger', 'steward',
            'deep_reviewer', 'reviewer_comprehensive', 'judge', 'simple_task',
        }
        for name, role in ROLES.items():
            for family, prefixes in _FAMILY_TOOL_PREFIXES.items():
                if family in role.mcp_families:
                    continue
                offending = [t for t in role.allowed_tools if t.startswith(prefixes)]
                assert not offending, (
                    f'{name!r} allows {offending!r} without declaring {family!r} '
                    'in mcp_families'
                )

    # -- Consumer: _invoke derives gating from role.mcp_families/sandboxed ---

    @pytest.mark.asyncio
    async def test_row12_invoke_derives_sandbox_modules_from_role_sandboxed(
        self, config, git_ops, task_assignment,
    ):
        role = AgentRole(
            name='probe_sbx', system_prompt='x', allowed_tools=[], sandboxed=True,
        )
        call_kwargs, _workflow, cwd = await _invoke_probe(role, config, git_ops, task_assignment)

        # Whole-worktree wiring (PRD os-sandbox D1): sandbox_modules=[] is the
        # empty-list sandbox-on gate; the write set (worktree root + carve-outs)
        # rides on sandbox_extras. Independent re-derivation of the α3 seam —
        # assert the worktree root is carved in without recomputing the full set.
        assert call_kwargs.get('sandbox_modules') == [], (
            'Expected sandbox_modules == [] (role.sandboxed=True; whole-worktree '
            f"gate) but got {call_kwargs.get('sandbox_modules')!r}. _invoke must "
            'gate sandboxing off role.sandboxed, not a role.name string check.'
        )
        sandbox_extras = call_kwargs.get('sandbox_extras')
        assert sandbox_extras is not None, (
            'Expected sandbox_extras to carry the contract write set for a '
            f'sandboxed role, but got {sandbox_extras!r}.'
        )
        assert str(cwd.resolve()) in sandbox_extras, (
            f'Expected the worktree root {str(cwd.resolve())!r} carved into '
            f'sandbox_extras, but got {sandbox_extras!r}.'
        )

    @pytest.mark.asyncio
    async def test_row12_invoke_derives_plan_tools_server_from_role_mcp_families(
        self, config, git_ops, task_assignment,
    ):
        role = AgentRole(
            name='probe_plan', system_prompt='x', allowed_tools=[],
            mcp_families=frozenset({'plan_tools'}),
        )
        call_kwargs, _workflow, _cwd = await _invoke_probe(role, config, git_ops, task_assignment)

        mcp_config = call_kwargs.get('mcp_config')
        servers = (mcp_config or {}).get('mcpServers', {})
        assert 'plan-tools' in servers, (
            "Expected mcp_config['mcpServers'] to contain 'plan-tools' "
            f"(role.mcp_families={{'plan_tools'}}) but got {mcp_config!r}. _invoke "
            "must gate the plan-tools injection off 'plan_tools' in "
            'role.mcp_families, not a role.name string check.'
        )

    @pytest.mark.asyncio
    async def test_row12_invoke_derives_mcp_config_from_role_mcp_families(
        self, config, git_ops, task_assignment,
    ):
        role = AgentRole(
            name='probe_orch', system_prompt='x', allowed_tools=[],
            mcp_families=frozenset({'orchestrator'}),
        )
        call_kwargs, _workflow, _cwd = await _invoke_probe(role, config, git_ops, task_assignment)

        assert call_kwargs.get('mcp_config') is not None, (
            "Expected mcp_config to be built (role.mcp_families={'orchestrator'}) "
            f"but got {call_kwargs.get('mcp_config')!r}. _invoke must gate the "
            "orchestrator-assembled MCP config off 'orchestrator' in "
            'role.mcp_families, not a role.name string check.'
        )

    @pytest.mark.asyncio
    async def test_row12_bare_role_gets_no_sandbox_and_no_plan_tools_negative_control(
        self, config, git_ops, task_assignment,
    ):
        """Negative control: a role declaring neither family and unsandboxed
        gets no sandbox modules and no plan-tools server wired."""
        role = AgentRole(name='probe_bare', system_prompt='x', allowed_tools=[])
        call_kwargs, _workflow, _cwd = await _invoke_probe(role, config, git_ops, task_assignment)

        assert call_kwargs.get('sandbox_modules') is None, (
            f"Expected sandbox_modules is None (role.sandboxed=False) but got "
            f"{call_kwargs.get('sandbox_modules')!r}."
        )
        mcp_config = call_kwargs.get('mcp_config')
        servers = (mcp_config or {}).get('mcpServers', {})
        assert 'plan-tools' not in servers, (
            f"Expected no 'plan-tools' server wired (role.mcp_families=frozenset()) "
            f'but found one in mcp_config={mcp_config!r}.'
        )
