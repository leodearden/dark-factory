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
from _orch_helpers import pydantic_spec
from shared.task_statuses import TaskStatus
from shared.task_transitions import ActorClass, is_legal_transition, outcome_allows_status

# Cross-module reuse — conftest.py injects orchestrator/tests onto sys.path
# (see test_workflow_terminal_report.py for the same precedent). ``_make``
# and ``_bind_landed_row`` carry no module-level lock of their own, so they
# are imported directly rather than duplicated (rows 1-4).
#
# ``_derive_meta_root_like_production`` is imported (not just the plain
# helpers) because it is an autouse fixture in test_workflow_e2e.py — autouse
# only auto-applies within a module where pytest can SEE the fixture, and a
# plain `from test_workflow_e2e import AgentStub, ...` does not pull that in
# (see test_workflow_terminal_report.py for the same precedent). Without it,
# AgentStub's legacy-only TaskArtifacts writes are invisible to the
# workflow's relocated meta_root, so a real run() never reaches DONE (rows
# 5-6).
from test_workflow_e2e import (
    AgentStub,
    _build_workflow,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see above
    _init_repo,
)
from test_workflow_merge_provenance import _bind_landed_row, _make
from test_workflow_warm_lane_requeue import _make_workflow as _make_warmlane_workflow

from orchestrator.agents.roles import _FAMILY_TOOL_PREFIXES, ROLES, AgentRole
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.landed_outbox import MergeProvenance
from orchestrator.scheduler import TaskAssignment
from orchestrator.unblock_types import BlockClass
from orchestrator.verify import VerifyResult
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
