"""Integration gate: B1–B15 boundary-test suite (task ζ).

Maps each PRD §Boundary-test row to its acceptance contract:

B1  — MERGE-entry gate: born-at-L2 pending before merge → WorkflowOutcome.ESCALATED
B2  — post-implementer gate: born-at-L2 pending → ESCALATED before next verify
B3  — stale plain L1 (blocking/level=1) → _is_gating_escalation returns False (no gate)
B4  — L2 cluster park: all members deferred; sweep quiescent (no stranded_blocked)
B5  — born-at-L2 abandon: task cancelled; sweep never touches cancelled
B6  — restart on live workflow: status-precedes-kill (C3.1); teardown stamp (C3.2)
B7  — resume memberless born-at-L2 (level=2 orphan, blocked) → pending (D7)
B8  — close_only: no set_task_status
B9  — terminate= parameter: resolve_issue returns error dict naming 5 actions
B10 — startup-sweep replay: stranded_blocked L1 filed, resolving flips to pending
B11 — re-block guard threshold trip: 4th same-signature flip withheld + L2 urgent
B12 — re-block guard signature reset: new signature resets counter to 1
B13 — chokepoint downgrade vs sentinel: implementer/critical → blocking+suffix; sentinel keeps L2
B14 — legacy in-process dismiss: resolution_action=None + dismissed → close_only → no flip
B15 — /unblock park protection: blocked task with open pending L1 → sweep files nothing
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

# Cross-module reuse — conftest.py injects orchestrator/tests onto sys.path.
from test_harness_action_dispatch import (
    _make_esc,
    _make_l1_for_queue,
    _make_l2_with_action_in_queue,
)
from test_harness_reblock_signature_guard import (
    _make_clobber_fake_scheduler,
    _make_fake_scheduler,
    _make_l1_esc,
    _make_mock_queue,
)
from test_stranded_blocked_sweep import _make_pending_l1, _make_resolved_l1
from test_workflow_e2e import (
    PLAN,
    AgentStub,
    _build_workflow_with_escalation,
    _init_repo,
)

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome, _is_gating_escalation

# ---------------------------------------------------------------------------
# Fixtures — pre-2
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Harness with mocked internals for state-machine unit testing.

    Provides the 4-patch + async-mock-scheduler pattern used across the
    action-dispatch, stranded-sweep, and reblock-guard test suites.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_status = AsyncMock(return_value='blocked')
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h.scheduler.update_task = AsyncMock(return_value=True)
    h._escalation_queue = None
    return h


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


def make_escalation_server(tmp_path: Path, *, status_lookup=None):
    """Create a real escalation server + queue for server-side boundary tests."""
    queue = EscalationQueue(tmp_path / 'esc')
    server = create_server(queue, task_status_lookup=status_lookup)
    return server, queue


# ---------------------------------------------------------------------------
# Step-1: B1/B2/B3 — workflow escalation gates (γ / C7, D4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('severity,level,expected', [
    # B1/B2 disjuncts — born-at-L2 severities at any level gate
    ('critical', 2, True),   # born-at-L2 at level 2 (B1/B2 primary)
    ('critical', 0, True),   # born-at-L2 severity disjunct at level 0
    ('urgent', 1, True),     # born-at-L2 severity disjunct at level 1
    # Level≥2 catch-all (D4) — any pending L2 is stop-the-line
    ('blocking', 2, True),   # level≥2 disjunct
    ('info', 2, True),       # even 'info' at level 2 gates
    # B3 — stale plain L1 must NOT gate a fresh run
    ('blocking', 1, False),  # [B3] blocking/L1 → False
    # Plain blocking L0 — primary gate
    ('blocking', 0, True),   # L0 blocking gates normally
    # Low-severity, low-level — not gating
    ('info', 0, False),
    ('info', 1, False),
])
def test_b1_b2_b3_predicate(severity: str, level: int, expected: bool):
    """[B1/B2/B3] _is_gating_escalation truth table (severity×level)."""
    e = Escalation(
        id=f'esc-42-{level}',
        task_id='42',
        agent_role='implementer',
        severity=severity,
        category='task_failure',
        summary=f'gate test {severity}/{level}',
        level=level,
    )
    assert _is_gating_escalation(e) is expected


@pytest.mark.asyncio
class TestWorkflowGatesB1B2:
    """Gate-in-context tests: born-at-L2 stops the workflow at MERGE-entry (B1)
    and post-implementer (B2).  Asserts WorkflowOutcome.ESCALATED and that the
    gate does not consume the pending escalation record."""

    async def test_b1_merge_entry_gate(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        """[B1] Born-at-L2 (critical/level=2) pending at MERGE entry → ESCALATED;
        escalation stays pending (gate does not consume)."""
        stub = AgentStub()
        workflow, _, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)

        # File the born-at-L2 escalation inside the stubbed verify/review phase
        # so it is pending at MERGE entry but absent during _execute_iterations.
        task_id = task_assignment.task_id

        async def _exec_and_escalate():
            queue.submit(Escalation(
                id=queue.make_id(task_id),
                task_id=task_id,
                agent_role='implementer',
                severity='critical',
                category='risk_identified',
                summary='Critical risk filed before MERGE (born at L2)',
                level=2,
            ))
            return WorkflowOutcome.DONE

        workflow._execute_verify_review_loop = _exec_and_escalate  # type: ignore[method-assign]

        outcome = await workflow.run()

        assert outcome == WorkflowOutcome.ESCALATED, (
            f'MERGE-entry gate must return ESCALATED; got {outcome}'
        )
        # Gate does not consume — escalation stays pending.
        still_pending = queue.get_by_task(task_id, status='pending', level=2)
        assert len(still_pending) == 1, (
            f'Pending L2 must survive the gate (not consumed); got {still_pending}'
        )

    async def test_b2_post_implementer_gate(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        """[B2] Born-at-L2 (critical/level=2) pending before _execute_iterations
        → ESCALATED at post-implementer gate; escalation stays pending."""
        stub = AgentStub()
        workflow, _, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
            spawn_merge_worker=False,
        )
        task_id = task_assignment.task_id

        # Submit the born-at-L2 escalation BEFORE running _execute_iterations.
        # The post-implementer gate intercepts it.
        queue.submit(Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='implementer',
            severity='critical',
            category='risk_identified',
            summary='Critical risk surfaced during implementation (born at L2)',
            level=2,
        ))

        # Wire worktree + plan so _execute_iterations can run.
        wt = tmp_path / 'wt'
        wt.mkdir()
        workflow.worktree = wt
        workflow.artifacts = TaskArtifacts(wt)
        workflow.artifacts.init('42', 'T', 'd', base_commit='deadbeef')
        plan = dict(PLAN)
        plan['steps'] = [{**s, 'status': 'pending'} for s in plan['steps']]
        workflow.artifacts.write_plan(plan)
        workflow.artifacts.stamp_plan_provenance(workflow.session_id)

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        await _init_repo(wt)
        workflow.config.inter_iteration_rebase = False

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.ESCALATED, (
            f'Post-implementer gate must return ESCALATED; got {outcome}'
        )
        still_pending = queue.get_by_task(task_id, status='pending', level=2)
        assert len(still_pending) == 1, (
            f'Pending L2 must survive the gate (not consumed); got {still_pending}'
        )


# ---------------------------------------------------------------------------
# Step-2: B9/B13 — escalation server seam (α1/α2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEscalationServerSeamB9B13:
    """Server-side boundary: terminate= migration guard (B9) and chokepoint
    downgrade vs sentinel-role preservation (B13)."""

    async def test_b9_terminate_param_returns_error_dict(self, tmp_path: Path):
        """[B9] resolve_issue with terminate=True returns {'error': ...} naming
        all five actions; the escalation record stays pending (no raise)."""
        server, queue = make_escalation_server(tmp_path)

        # Submit a pending L1 to resolve.
        esc = Escalation(
            id='esc-b9-1',
            task_id='task-b9',
            agent_role='steward',
            severity='blocking',
            category='design_concern',
            summary='B9 terminate guard test',
            level=1,
        )
        queue.submit(esc)

        # resolve_issue is a sync def — call tool.fn() directly (no await).
        tool = await server.get_tool('resolve_issue')
        assert tool is not None
        result = tool.fn(  # type: ignore[union-attr]
            escalation_id=esc.id,
            resolution='attempt to use removed parameter',
            terminate=True,
        )

        # Must return an error dict (not raise).
        assert isinstance(result, dict), 'resolve_issue must return a dict'
        assert 'error' in result, f"Missing 'error' key in result: {result}"

        error_msg = result['error']
        for action in ('resume', 'restart', 'park', 'abandon', 'close_only'):
            assert action in error_msg, (
                f"Action '{action}' not named in error message: {error_msg!r}"
            )

        # Record must remain unchanged.
        still_pending = queue.get(esc.id)
        assert still_pending is not None
        assert still_pending.status == 'pending', (
            f"Record status must stay 'pending'; got {still_pending.status!r}"
        )

    async def test_b13_chokepoint_downgrade_non_sentinel(
        self, tmp_path: Path, caplog
    ):
        """[B13] escalate_blocker with implementer role + critical severity →
        record severity='blocking', summary ends with '[downgraded:critical]',
        WARNING logged (C4/D3 downgrade)."""
        server, queue = make_escalation_server(tmp_path)
        tool = await server.get_tool('escalate_blocker')
        assert tool is not None

        with caplog.at_level(logging.WARNING, logger='escalation.server'):
            result = await tool.fn(  # type: ignore[union-attr]
                task_id='task-b13-dg',
                agent_role='implementer',
                severity='critical',
                category='risk_identified',
                summary='boom',
            )

        assert 'id' in result, f'Expected queued escalation; got {result}'

        filed = queue.get(result['id'])
        assert filed is not None
        assert filed.severity == 'blocking', (
            f'Non-sentinel critical must be downgraded to blocking; got {filed.severity!r}'
        )
        assert filed.level == 0, (
            f'Downgraded record must be level 0; got level={filed.level}'
        )
        assert filed.summary.endswith('[downgraded:critical]'), (
            f"Summary must end with '[downgraded:critical]'; got {filed.summary!r}"
        )

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_msgs, 'Expected at least one WARNING log for chokepoint downgrade'

    async def test_b13_sentinel_role_keeps_born_at_l2(self, tmp_path: Path):
        """[B13] escalate_blocker with harness-sentinel role + critical severity →
        record keeps severity='critical' and level==2 (no downgrade)."""
        server, queue = make_escalation_server(tmp_path)
        tool = await server.get_tool('escalate_blocker')
        assert tool is not None

        result = await tool.fn(  # type: ignore[union-attr]
            task_id='task-b13-sentinel',
            agent_role='harness-stranded-blocked-reaper',
            severity='critical',
            category='risk_identified',
            summary='sentinel escalation for B13',
        )

        assert 'id' in result, f'Expected queued escalation; got {result}'

        filed = queue.get(result['id'])
        assert filed is not None
        assert filed.severity == 'critical', (
            f'Sentinel role must keep severity=critical; got {filed.severity!r}'
        )
        assert filed.level == 2, (
            f'Sentinel critical must be born-at-L2 (level=2); got level={filed.level}'
        )
        assert '[downgraded' not in filed.summary, (
            f'Sentinel summary must not contain downgrade marker; got {filed.summary!r}'
        )


# ---------------------------------------------------------------------------
# Step-3: B7/B8/B14 — simple dispatch (β)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSimpleDispatchB7B8B14:
    """Direct _on_escalation_resolved dispatch for resume/close_only/legacy-dismiss."""

    async def test_b7_resume_memberless_born_at_l2(self, harness: Harness):
        """[B7] Memberless born-at-L2 (level=2) orphan with resolution_action='resume'
        and task blocked → set_task_status(tid, 'pending') (D7 level≥1 flip)."""
        task_id = 'task-b7'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='resume',
            status='resolved',
            resolved_by='steward',  # NOT l2-cascade prefix → orphan path
            level=2,
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        # Not in _escalation_events → orphan-unblock path fires.
        harness._escalation_events.clear()

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'pending',
        )

    async def test_b8_close_only_no_status_write(self, harness: Harness):
        """[B8] resolution_action='close_only' → no set_task_status call."""
        for level in (0, 1, 2):
            harness.scheduler.set_task_status = AsyncMock()
            esc = _make_esc(
                task_id=f'task-b8-l{level}',
                resolution_action='close_only',
                status='dismissed',
                resolved_by='interactive',
                level=level,
            )
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

            harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_b14_legacy_dismiss_maps_to_close_only(self, harness: Harness):
        """[B14] resolution_action=None + status='dismissed' (legacy in-process dismiss)
        → D10 mapping → close_only → no set_task_status."""
        task_id = 'task-b14'
        esc = _make_esc(
            task_id=task_id,
            resolution_action=None,
            status='dismissed',
            resolved_by='steward',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Step-4: B4/B5 — cluster park + abandon with sweep quiescence (β + ε)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestClusterActionsB4B5:
    """L2 cluster park/abandon with follow-on sweep quiescence."""

    async def test_b4_l2_park_members_blocked_open_l2_sweep_quiescent(
        self, harness: Harness, tmp_path: Path
    ):
        """[B4] L2 cluster park (version-a): both L1 member tasks set to 'blocked';
        L2 and member L1 escalations stay OPEN (pending); two sweep passes with members
        reported 'blocked' → no stranded_blocked filed (Fix #1b skip via open escalation).

        Quiescence mechanism: members are 'blocked' (inside _RECONCILE_SWEEP_STATUSES),
        but Fix #1b's gate `not get_by_task(tid, status='pending')` finds the still-pending
        member L1 escalations and skips re-filing.

        Positive control: a 'blocked' task with NO open escalation → sweep files stranded_blocked,
        proving Fix #1b ran and did not universally skip.

        Fails: queue.park() member cascade not yet implemented (step-8).
        """
        queue = EscalationQueue(tmp_path / 'esc_b4')
        l1_a = _make_l1_for_queue(queue, 'esc-b4-l1-a', 'task-b4-a')
        l1_b = _make_l1_for_queue(queue, 'esc-b4-l1-b', 'task-b4-b')
        l2 = _make_l2_with_action_in_queue(
            queue, 'esc-b4-l2', [l1_a.id, l1_b.id], resolution_action='park',
        )

        harness._escalation_queue = queue
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)
        queue.set_resolve_callback(harness._on_escalation_resolved)

        # Park the L2 cluster (new path: queue.park keeps record open).
        queue.park(l2.id, 'park the cluster')
        await asyncio.gather(*list(harness._background_tasks))

        # Both member task ids must be written to 'blocked' (version-a target).
        awaits = harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
        task_ids_written = {a.args[0]: a.args[1] for a in awaits}
        assert 'task-b4-a' in task_ids_written, (
            f'task-b4-a not written; all writes: {task_ids_written}'
        )
        assert 'task-b4-b' in task_ids_written, (
            f'task-b4-b not written; all writes: {task_ids_written}'
        )
        assert task_ids_written['task-b4-a'] == 'blocked', (
            f"task-b4-a: expected 'blocked', got {task_ids_written['task-b4-a']!r}"
        )
        assert task_ids_written['task-b4-b'] == 'blocked', (
            f"task-b4-b: expected 'blocked', got {task_ids_written['task-b4-b']!r}"
        )

        # L2 and BOTH member L1s must still be open (status='pending').
        l2_record = queue.get(l2.id)
        assert l2_record is not None and l2_record.status == 'pending', (
            f'L2 must stay open (pending); got {l2_record}'
        )
        for _l1, tid in ((l1_a, 'task-b4-a'), (l1_b, 'task-b4-b')):
            pending_l1s = queue.get_by_task(tid, status='pending')
            assert pending_l1s, (
                f'Member L1 for {tid} must stay pending (open); found: {pending_l1s}'
            )

        # Sweep quiescence: 2 passes with members reported 'blocked' → no stranded_blocked.
        harness.config.stranded_blocked_escalate_enabled = True
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)
        harness.scheduler._dispatched = set()
        harness.scheduler.lock_table = MagicMock()
        harness.scheduler.lock_table._held = set()

        # Positive control: a 'blocked' task with NO open escalation → sweep DOES file.
        # Proves the sweep actually ran and members were skipped by Fix #1b (open escalation),
        # not by an unrelated reason.
        ctrl_id = 'task-b4-ctrl'
        _make_resolved_l1(queue, 'esc-b4-ctrl-prior', ctrl_id)
        # Clear event tracking so no in-flight events gate the sweep.
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()

        # Members reported 'blocked' (now in _RECONCILE_SWEEP_STATUSES);
        # Fix #1b's open-escalation gate prevents stranded_blocked re-filing.
        # Control is 'blocked' with no open escalation → filed.
        harness.scheduler.get_statuses = AsyncMock(
            return_value=(
                {'task-b4-a': 'blocked', 'task-b4-b': 'blocked', ctrl_id: 'blocked'},
                None,
            )
        )
        harness.scheduler.get_task = AsyncMock(return_value=None)

        for _ in range(2):
            harness.scheduler.set_task_status = AsyncMock()
            await harness._reconcile_stranded_in_progress()

            # Parked members have open L1 escalations → Fix #1b skips re-filing.
            for tid in ('task-b4-a', 'task-b4-b'):
                all_pending = queue.get_by_task(tid, status='pending')
                stranded = [e for e in all_pending if e.category == 'stranded_blocked']
                assert not stranded, (
                    f'Sweep must not file stranded_blocked for {tid} (open L1 present); '
                    f'got {stranded}'
                )

        # Positive control: ctrl_id was filed in pass 1 (self-dedup suppressed pass 2),
        # proving Fix #1b ran and the open-escalation skip is the quiescence mechanism.
        ctrl_filed = queue.get_by_task(ctrl_id, status='pending')
        ctrl_stranded = [e for e in ctrl_filed if e.category == 'stranded_blocked']
        assert len(ctrl_stranded) == 1, (
            f'Sweep must file exactly 1 stranded_blocked for blocked control {ctrl_id!r}; '
            f'got {ctrl_stranded}'
        )

    async def test_b5_abandon_born_at_l2_cancelled_sweep_ignores(
        self, harness: Harness, tmp_path: Path
    ):
        """[B5] Memberless born-at-L2 (level=2) with resolution_action='abandon' →
        task set to 'cancelled'; sweep pass with task 'cancelled' → no stranded_blocked."""
        queue = EscalationQueue(tmp_path / 'esc_b5')
        task_id = 'task-b5'

        harness._escalation_queue = queue
        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.is_workflow_active = MagicMock(return_value=False)

        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='harness-reblock-guard',
            severity='urgent',
            category='task_failure',
            summary='Persistent re-block (born at L2)',
            level=2,
            resolution_action='abandon',
            status='dismissed',
            resolved_by='interactive',
        )
        # Submit and mark as already resolved (simulating a real abandon).
        # We drive the harness callback directly.
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            task_id, 'cancelled',
        )

        # Sweep: task is 'cancelled' — sweep never touches it.
        harness.config.stranded_blocked_escalate_enabled = True
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)
        harness.scheduler._dispatched = set()
        harness.scheduler.lock_table = MagicMock()
        harness.scheduler.lock_table._held = set()
        harness.scheduler.get_task = AsyncMock(return_value=None)
        harness.scheduler.set_task_status = AsyncMock()

        # Positive control: a 'blocked' sibling that the sweep DOES file for,
        # proving the sweep ran and the cancelled task was filtered by status.
        ctrl_id = 'task-b5-ctrl'
        _make_resolved_l1(queue, 'esc-b5-ctrl-prior', ctrl_id)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()

        # Cancelled task not in _RECONCILE_SWEEP_STATUSES → skipped; control is 'blocked'.
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'cancelled', ctrl_id: 'blocked'}, None)
        )

        await harness._reconcile_stranded_in_progress()

        # Cancelled task: no stranded_blocked filed.
        stranded = queue.get_by_task(task_id, status='pending')
        assert not stranded, (
            f'Sweep must not file stranded_blocked for cancelled task; got {stranded}'
        )

        # Positive control: ctrl_id filed (proves sweep ran).
        ctrl_filed = queue.get_by_task(ctrl_id, status='pending')
        ctrl_stranded = [e for e in ctrl_filed if e.category == 'stranded_blocked']
        assert len(ctrl_stranded) == 1, (
            f'Sweep must file exactly 1 stranded_blocked for blocked control {ctrl_id!r}; '
            f'got {ctrl_stranded}'
        )


# ---------------------------------------------------------------------------
# Step-5: B6 — restart kills live workflow without teardown clobber (β / C3, D9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRestartLiveWorkflowB6:
    """Restart on a live workflow: status-precedes-kill (C3.1) and teardown
    suppression stamp/clear (C3.2)."""

    def _make_kill_harness(self, harness: Harness):
        """Set up harness for kill-sequence tests: live workflow initially active."""
        call_log: list[str] = []

        async def fake_set_status(tid: str, status: str, **_) -> None:
            call_log.append(f'set_status:{tid}:{status}')

        def fake_cancel(tid: str) -> None:
            call_log.append(f'cancel:{tid}')

        def fake_is_active(tid: str) -> bool:
            return not any(e.startswith('cancel:') for e in call_log)

        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.scheduler.set_task_status = AsyncMock(side_effect=fake_set_status)
        harness.cancel_workflow = MagicMock(side_effect=fake_cancel)
        harness.hard_cancel_workflow = MagicMock(return_value=True)
        harness.is_workflow_active = MagicMock(side_effect=fake_is_active)
        harness.config.terminal_status_hard_cancel_polls = 3

        return call_log

    async def test_b6_status_precedes_kill(self, harness: Harness):
        """[B6] restart on live workflow: set_task_status('pending') BEFORE
        cancel_workflow (C3.1 status-precedes-kill ordering)."""
        call_log = self._make_kill_harness(harness)
        task_id = 'task-b6-restart'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='restart',
            status='resolved',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.cancel_workflow.assert_called_once_with(task_id)  # type: ignore[attr-defined]

        # Status write must precede the cancel call in the call_log.
        assert any('set_status' in e for e in call_log), 'set_task_status not recorded'
        assert any('cancel:' in e for e in call_log), 'cancel_workflow not recorded'

        first_set = next(i for i, e in enumerate(call_log) if 'set_status' in e)
        first_cancel = next(i for i, e in enumerate(call_log) if 'cancel:' in e)
        assert first_set < first_cancel, (
            f'set_status must precede cancel_workflow (C3.1); call_log={call_log}'
        )

    async def test_b6_teardown_stamp_present_during_cancel_cleared_after(
        self, harness: Harness
    ):
        """[B6] restart stamps task_id in _action_teardown_tasks when
        cancel_workflow fires (C3.2) and clears it after the slot clears."""
        call_log: list[str] = []
        stamped_at_cancel: dict[str, bool] = {}

        async def fake_set_status(tid: str, status: str, **_) -> None:
            call_log.append(f'set_status:{tid}:{status}')

        def fake_cancel(tid: str) -> None:
            call_log.append(f'cancel:{tid}')
            stamped_at_cancel[tid] = tid in harness._action_teardown_tasks

        def fake_is_active(tid: str) -> bool:
            return not any(e == f'cancel:{tid}' for e in call_log)

        harness.scheduler.get_status = AsyncMock(return_value='blocked')
        harness.scheduler.set_task_status = AsyncMock(side_effect=fake_set_status)
        harness.cancel_workflow = MagicMock(side_effect=fake_cancel)
        harness.hard_cancel_workflow = MagicMock()
        harness.is_workflow_active = MagicMock(side_effect=fake_is_active)
        harness.config.terminal_status_hard_cancel_polls = 3

        task_id = 'task-b6-stamp'
        esc = _make_esc(
            task_id=task_id,
            resolution_action='restart',
            status='resolved',
            resolved_by='interactive',
            level=1,
        )

        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        harness.cancel_workflow.assert_called_once_with(task_id)  # type: ignore[attr-defined]

        # Stamp must have been present when cancel_workflow was called (C3.2).
        assert stamped_at_cancel.get(task_id) is True, (
            f'task_id must be in _action_teardown_tasks when cancel_workflow fires; '
            f'stamped_at_cancel={stamped_at_cancel}'
        )
        # Stamp must be cleared after the slot clears.
        assert task_id not in harness._action_teardown_tasks, (
            '_action_teardown_tasks must be cleared after slot clears'
        )


# ---------------------------------------------------------------------------
# Step-6: B11/B12 — signature-aware re-block guard (δ / C5, C3.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReblockGuardB11B12:
    """Re-block guard: threshold trip (B11), signature reset (B12),
    metadata survival across status cycle (C3.4)."""

    async def test_b11_threshold_trip_withholds_and_files_l2(
        self, harness: Harness, caplog
    ):
        """[B11] 4th same-signature flip withheld + born-at-L2 urgent filed
        with correct fields; WARNING logged."""
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='l2-cascade:esc-100-1',
        )
        sig = Harness._reblock_signature(esc)

        q = _make_mock_queue()
        harness._escalation_queue = q

        # Seed count=3 with same signature (prev_count >= threshold).
        persisted_metadata, _ = _make_fake_scheduler(
            harness,
            initial_metadata={'reblock_guard': {'count': 3, 'signature': sig}},
        )

        set_status_calls: list = []
        original_set = harness.scheduler.set_task_status

        async def spy_set(task_id, status, **kw):  # noqa: F841
            set_status_calls.append(status)
            await original_set(task_id, status, **kw)

        harness.scheduler.set_task_status = spy_set  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            harness._on_escalation_resolved(esc)
            await asyncio.gather(*list(harness._background_tasks))

        # 4th flip must be WITHHELD.
        assert set_status_calls == [], (
            f'set_task_status must not be called at threshold; got {set_status_calls}'
        )

        # Exactly one L2 filed with correct fields.
        assert len(q._submitted) == 1, (
            f'Expected 1 L2 filed; got {len(q._submitted)}'
        )
        filed = q._submitted[0]
        assert filed.severity == 'urgent'
        assert filed.category == 'task_failure'
        assert filed.level == 2
        assert filed.agent_role == 'harness-reblock-guard'
        expected_summary = f'persistent re-block: 3 redispatches, signature {sig}'
        assert filed.summary == expected_summary, (
            f'Expected summary {expected_summary!r}; got {filed.summary!r}'
        )

        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            'Expected a WARNING log for threshold trip'
        )

    async def test_b12_signature_change_resets_count_flip_proceeds(
        self, harness: Harness
    ):
        """[B12] Different signature → counter resets to 1, flip proceeds
        (set_task_status('pending') is awaited)."""
        task_id = '42'

        old_esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='old problem',
            resolved_by='l2-cascade:esc-100-1',
        )
        old_sig = Harness._reblock_signature(old_esc)

        # New escalation with DIFFERENT category → different signature.
        new_esc = _make_l1_esc(
            task_id=task_id,
            category='task_failure',
            summary='new failure mode',
            resolved_by='l2-cascade:esc-200-1',
        )
        new_sig = Harness._reblock_signature(new_esc)
        assert new_sig != old_sig, 'Test setup error: signatures must differ'

        harness._escalation_queue = None

        # Seed count=2 with OLD signature.
        persisted_metadata, _ = _make_fake_scheduler(
            harness,
            initial_metadata={'reblock_guard': {'count': 2, 'signature': old_sig}},
        )

        set_status_calls: list = []
        original_set = harness.scheduler.set_task_status

        async def spy_set(task_id, status, **kw):  # noqa: F841
            set_status_calls.append(status)
            await original_set(task_id, status, **kw)

        harness.scheduler.set_task_status = spy_set  # type: ignore[method-assign]

        harness._on_escalation_resolved(new_esc)
        await asyncio.gather(*list(harness._background_tasks))

        # Flip must proceed (signature changed → reset to 1, not at threshold).
        assert 'pending' in set_status_calls, (
            f'set_task_status(pending) must be called on signature reset; got {set_status_calls}'
        )

        # Counter must be reset to 1 with the new signature.
        guard = persisted_metadata.get('reblock_guard')
        assert guard is not None, 'reblock_guard must be persisted'
        assert guard['count'] == 1, (
            f'Signature reset must write count=1; got count={guard["count"]}'
        )
        assert guard['signature'] == new_sig, (
            f'Persisted signature must be the new sig; got {guard["signature"]!r}'
        )

    async def test_b12_metadata_survives_full_cycle(self, harness: Harness):
        """[B12/C3.4] Counter continues 1→2 across blocked→pending→in-progress→blocked
        cycle; sibling metadata keys ('files', 'memory_hints') survive (append=True)."""
        task_id = '42'
        esc = _make_l1_esc(
            task_id=task_id,
            category='infra_issue',
            summary='disk full',
            resolved_by='l2-cascade:esc-100-1',
        )
        task_state, _ = _make_clobber_fake_scheduler(
            harness,
            initial_metadata={
                'files': ['foo.py', 'bar.py'],
                'memory_hints': ['hint1', 'hint2'],
            },
        )

        # 1st flip.
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        guard = task_state['metadata'].get('reblock_guard')
        assert guard is not None and guard['count'] == 1

        # Advance state to 'blocked' for the 2nd flip.
        task_state['status'] = 'blocked'

        # 2nd flip.
        harness._on_escalation_resolved(esc)
        await asyncio.gather(*list(harness._background_tasks))

        guard = task_state['metadata'].get('reblock_guard')
        assert guard is not None
        assert guard['count'] == 2, (
            f'Counter must continue to 2; got {guard["count"]}'
        )

        # Sibling metadata must survive (C3.4 append=True).
        meta = task_state['metadata']
        assert 'files' in meta, "Sibling 'files' was clobbered — update_task must use append=True"
        assert meta['files'] == ['foo.py', 'bar.py']
        assert 'memory_hints' in meta, "Sibling 'memory_hints' was clobbered"
        assert meta['memory_hints'] == ['hint1', 'hint2']


# ---------------------------------------------------------------------------
# Step-7: B10/B15 — stranded sweep + /unblock park protection (ε / C6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStrandedSweepB10B15:
    """Stranded-blocked sweep: B10 files and resolves; B15 open-L1 suppresses."""

    async def test_b10_startup_sweep_files_stranded_blocked_and_flips(
        self, harness: Harness, tmp_path: Path
    ):
        """[B10] Blocked task with RESOLVED prior L1 and no active workflow →
        sweep files stranded_blocked L1; resolving it (non-cascade) flips to pending
        (zero human round-trips)."""
        queue = EscalationQueue(tmp_path / 'esc_b10')
        task_id = 'task-b10'

        harness._escalation_queue = queue
        _make_resolved_l1(queue, 'esc-b10-prior', task_id)

        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()

        harness.config.stranded_blocked_escalate_enabled = True
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)
        harness.scheduler._dispatched = set()
        harness.scheduler.lock_table = MagicMock()
        harness.scheduler.lock_table._held = set()
        harness.scheduler.get_task = AsyncMock(return_value=None)
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'blocked'}, None)
        )
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        await harness._reconcile_stranded_in_progress()

        filed = queue.get_by_task(task_id, status='pending')
        assert len(filed) == 1, (
            f'Expected exactly 1 pending L1 for {task_id}; got {len(filed)}'
        )
        new_l1 = filed[0]
        assert new_l1.category == 'stranded_blocked'
        assert new_l1.agent_role == 'harness-stranded-blocked-reaper'
        assert new_l1.severity == 'blocking'
        assert new_l1.level == 1

        # Resolving the L1 (non-cascade, orphan path) → flips task to 'pending'.
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)

        queue.resolve(new_l1.id, 'resolved by auto-watcher')
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_awaited_with(  # type: ignore[attr-defined]
            task_id, 'pending',
        )

    async def test_b15_open_l1_suppresses_stranded_refiling(
        self, harness: Harness, tmp_path: Path
    ):
        """[B15] Blocked task with OPEN (pending) L1 → sweep files nothing
        (pending-escalation self-dedup); original L1 survives unchanged."""
        queue = EscalationQueue(tmp_path / 'esc_b15')
        task_id = 'task-b15'

        harness._escalation_queue = queue
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()

        open_l1 = _make_pending_l1(queue, 'esc-b15-open', task_id)

        harness.config.stranded_blocked_escalate_enabled = True
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)
        harness.scheduler._dispatched = set()
        harness.scheduler.lock_table = MagicMock()
        harness.scheduler.lock_table._held = set()
        harness.scheduler.get_task = AsyncMock(return_value=None)
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({task_id: 'blocked'}, None)
        )

        await harness._reconcile_stranded_in_progress()

        pending = queue.get_by_task(task_id, status='pending')
        assert len(pending) == 1, (
            f'Expected 1 pending L1 (the open one, not re-filed); got {len(pending)}'
        )
        assert pending[0].id == open_l1.id, (
            'Surviving L1 must be the original open one'
        )
