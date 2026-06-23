"""Tests for DeterministicRunner (β).

Step-5: RED — pure-gate path (B2/I3): escalate, stamp, block
Step-7: RED — idempotent resume + quiescence (I2/B3/B4/B11)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate_task(
    task_id: str = '99',
    title: str = 'Ship feature gate',
    description: str = 'Gate that guards the feature launch',
    deps: list | None = None,
    gate_options: list | None = None,
    gate_escalated_at: str | None = None,
) -> dict:
    """Build a deterministic pure-gate task dict."""
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'before_done': None,
    }
    if gate_options is not None:
        metadata['gate_options'] = gate_options
    if gate_escalated_at is not None:
        metadata['gate_escalated_at'] = gate_escalated_at
    task = {
        'id': task_id,
        'title': title,
        'description': description,
        'metadata': metadata,
    }
    if deps is not None:
        task['dependencies'] = deps
    return task


def _make_assignment(task: dict):
    """Build a minimal TaskAssignment-like object for the runner.

    Deterministic tasks hold an empty modules list (I4/B12: no module lock).
    """
    from orchestrator.scheduler import TaskAssignment
    return TaskAssignment(task_id=str(task['id']), task=task, modules=[])


def _mock_scheduler(task: dict):
    """Return a MagicMock scheduler with async set_task_status/update_task/get_task."""
    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_task = AsyncMock(return_value=task)
    return scheduler


def _deploy_task(
    task_id: str = '200',
    target_unit: str = 'orchestrator-reify.service',
    script: str = '/tmp/test-deploy.sh',
    args: list | None = None,
    env: dict | None = None,
    cwd: str = '/tmp',
    timeout_secs: int = 30,
    before_done_ran_at: str | None = None,
    before_done_verified_at: str | None = None,
    before_done_verified_pid: int | None = None,
) -> dict:
    """Build a deterministic deploy task dict (before_done set, always_escalates=False)."""
    before_done: dict = {
        'script': script,
        'args': args if args is not None else [],
        'env': env if env is not None else {},
        'cwd': cwd,
        'timeout_secs': timeout_secs,
        'target_unit': target_unit,
    }
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': False,
        'before_done': before_done,
    }
    if before_done_ran_at is not None:
        metadata['before_done_ran_at'] = before_done_ran_at
    if before_done_verified_at is not None:
        metadata['before_done_verified_at'] = before_done_verified_at
    if before_done_verified_pid is not None:
        metadata['before_done_verified_pid'] = before_done_verified_pid
    return {
        'id': task_id,
        'title': 'Deploy orchestrator-reify',
        'description': 'Cross-unit deploy of the reify worker',
        'metadata': metadata,
    }


# Unit states used across B6/B7 deploy tests
_BASELINE_UNIT_STATE: dict = {
    'MainPID': 100,
    'ActiveState': 'active',
    'ActiveEnterTimestamp': 'Mon 2026-06-23 10:00:00 UTC',
    'ActiveEnterTimestampMonotonic': 1_000_000,
}
_FRESH_UNIT_STATE: dict = {
    'MainPID': 200,
    'ActiveState': 'active',
    'ActiveEnterTimestamp': 'Mon 2026-06-23 10:01:00 UTC',
    'ActiveEnterTimestampMonotonic': 2_000_000,
}


# ---------------------------------------------------------------------------
# Step-5: pure-gate path (B2/I3)
# (RED until step-6 creates deterministic_runner.py)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPureGatePath:
    """DeterministicRunner — pure-gate (always_escalates=True, before_done=None)."""

    async def test_pure_gate_submits_l2_escalation(self, tmp_path: Path):
        """Pure gate files exactly ONE born-at-L2 escalation (I3)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99', deps=[10, 11])
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        # Exactly one escalation submitted
        pending = queue.get_by_task('99', status='pending')
        assert len(pending) == 1, f'Expected 1 pending escalation, got {len(pending)}'

    async def test_pure_gate_escalation_is_level_2(self, tmp_path: Path):
        """The filed escalation must have level==2 (born-at-L2, I3)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert len(escs) == 1
        assert escs[0].level == 2

    async def test_pure_gate_escalation_sentinel_role(self, tmp_path: Path):
        """agent_role must be 'orchestrator-deterministic' (sentinel keeps level=2)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert escs[0].agent_role == 'orchestrator-deterministic'

    async def test_pure_gate_escalation_category_milestone_gate(self, tmp_path: Path):
        """Category must be 'milestone_gate' for dashboard filtering."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert escs[0].category == 'milestone_gate'

    async def test_pure_gate_escalation_summary_is_title(self, tmp_path: Path):
        """Summary must equal the task title (truncated to 200 chars if needed)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99', title='My Gate')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert escs[0].summary == 'My Gate'

    async def test_pure_gate_escalation_detail_contains_description(self, tmp_path: Path):
        """Escalation detail must include the task description."""
        from orchestrator.deterministic_runner import DeterministicRunner

        desc = 'This gate guards the Q3 launch milestone'
        task = _gate_task(task_id='99', description=desc)
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert desc in escs[0].detail

    async def test_pure_gate_escalation_detail_contains_dep_ids(self, tmp_path: Path):
        """Escalation detail must include ALL dependency IDs (both 10 and 11)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99', deps=[10, 11])
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        detail = escs[0].detail
        # Both dep ids must appear — OR weakens the check (one id missing would pass)
        assert '10' in detail and '11' in detail, (
            f"Both dep ids '10' and '11' must appear in detail: {detail!r}"
        )
        # Prefer the formatted 'Landed dependencies: 10, 11' substring to confirm
        # no incidental substring match (e.g. '110' or '211').
        assert 'Landed dependencies: 10, 11' in detail, (
            f"Expected 'Landed dependencies: 10, 11' in detail: {detail!r}"
        )

    async def test_pure_gate_escalation_detail_contains_dict_dep_ids(self, tmp_path: Path):
        """Dict-shaped dependencies ({'id': N}) must have their IDs extracted into detail."""
        from orchestrator.deterministic_runner import DeterministicRunner

        # deps as list-of-dicts (the shape used by _deps_satisfied in the real scheduler)
        task = _gate_task(task_id='99', deps=[{'id': 10}, {'id': 11}])
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        detail = escs[0].detail
        assert '10' in detail and '11' in detail, (
            f"Dict dep ids '10' and '11' must appear in detail: {detail!r}"
        )
        assert 'Landed dependencies: 10, 11' in detail, (
            f"Expected 'Landed dependencies: 10, 11' from dict deps in detail: {detail!r}"
        )

    async def test_pure_gate_escalation_options_from_gate_options(self, tmp_path: Path):
        """gate_options in metadata → Escalation.options."""
        from orchestrator.deterministic_runner import DeterministicRunner

        options = ['A: Ship now', 'B: Defer to Q4']
        task = _gate_task(task_id='99', gate_options=options)
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert escs[0].options == options

    async def test_pure_gate_stamps_gate_escalated_at(self, tmp_path: Path):
        """Runner stamps metadata.gate_escalated_at after filing the escalation."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        # update_task must have been called with gate_escalated_at set (truthy)
        scheduler.update_task.assert_awaited_once()
        call_args = scheduler.update_task.call_args
        metadata_update = call_args.args[1] if call_args.args else call_args.kwargs.get('metadata', {})
        assert metadata_update.get('gate_escalated_at'), (
            'gate_escalated_at should be a truthy ISO timestamp'
        )

    async def test_pure_gate_sets_task_blocked(self, tmp_path: Path):
        """Runner sets task status to 'blocked' after stamping."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        scheduler.set_task_status.assert_awaited_once_with('99', 'blocked')

    async def test_pure_gate_returns_blocked(self, tmp_path: Path):
        """run() must return WorkflowOutcome.BLOCKED for the pure-gate path."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _gate_task(task_id='99')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

    async def test_always_escalates_false_before_done_none_raises_value_error(self, tmp_path: Path):
        """always_escalates=False with before_done=None → ValueError (unsupported in β)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99')
        task['metadata']['always_escalates'] = False  # misconfiguration
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        with pytest.raises(ValueError, match='always_escalates=False'):
            await runner.run(assignment)


# ---------------------------------------------------------------------------
# Step-7: idempotent resume + quiescence (I2/B3/B4/B11)
# (RED until step-8 adds the idempotency branch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestIdempotentResumeAndQuiescence:
    """DeterministicRunner — idempotency when gate_escalated_at is set."""

    async def test_resume_no_open_escalation_drives_to_done(self, tmp_path: Path):
        """gate_escalated_at set + no open escalation → set_task_status('done'), return DONE (I2/B4/B11)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        # gate already escalated; escalation already resolved (no pending)
        task = _gate_task(task_id='100', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty queue — no pending escalation
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        scheduler.set_task_status.assert_awaited_once_with('100', 'done')

    async def test_resume_no_new_escalation_filed(self, tmp_path: Path):
        """Resume path must NOT file a new escalation (no re-escalate)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='100', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        pending = queue.get_by_task('100', status='pending')
        assert len(pending) == 0, 'Resume path must not file a new L2'

    async def test_quiescence_open_escalation_returns_blocked(self, tmp_path: Path):
        """gate_escalated_at set + open escalation → return BLOCKED, no second escalation (B3)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _gate_task(task_id='101', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)

        # Pre-seed the queue with an open (pending) escalation for this task
        existing = Escalation(
            id=queue.make_id('101'),
            task_id='101',
            agent_role='orchestrator-deterministic',
            severity='critical',
            category='milestone_gate',
            summary='Ship feature gate',
            level=2,
        )
        queue.submit(existing)

        scheduler = _mock_scheduler(task)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

    async def test_quiescence_no_second_escalation_filed(self, tmp_path: Path):
        """Quiescence path must NOT file a second escalation (B3)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='101', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)

        # Pre-seed one pending escalation
        existing = Escalation(
            id=queue.make_id('101'),
            task_id='101',
            agent_role='orchestrator-deterministic',
            severity='critical',
            category='milestone_gate',
            summary='Ship feature gate',
            level=2,
        )
        queue.submit(existing)

        scheduler = _mock_scheduler(task)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        # Still exactly one pending escalation (no second one added)
        pending = queue.get_by_task('101', status='pending')
        assert len(pending) == 1

    async def test_quiescence_set_task_status_not_called(self, tmp_path: Path):
        """Quiescence path must NOT set task to done (B3)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='101', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)

        existing = Escalation(
            id=queue.make_id('101'),
            task_id='101',
            agent_role='orchestrator-deterministic',
            severity='critical',
            category='milestone_gate',
            summary='Ship feature gate',
            level=2,
        )
        queue.submit(existing)

        scheduler = _mock_scheduler(task)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        scheduler.set_task_status.assert_not_awaited()

    async def test_resume_act_then_ask_before_done_ran_drives_to_done(self, tmp_path: Path):
        """Act-then-ask resume: gate_escalated_at + before_done_ran_at both set, empty queue → done (γ step-10)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        # gate already escalated AND before_done already ran; gate escalation is resolved.
        task = _gate_task(task_id='100', gate_escalated_at='2026-06-23T12:00:00+00:00')
        task['metadata']['before_done'] = _deploy_task()['metadata']['before_done']
        task['metadata']['before_done_ran_at'] = '2026-06-23T10:00:00+00:00'
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty — gate escalation resolved
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        # Must NOT raise NotImplementedError — before_done already ran, drive to done
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        # Act-then-ask resume must carry deterministic-deploy provenance so the audit
        # trail matches the B6/resume paths and passes require_done_provenance.
        scheduler.set_task_status.assert_awaited_once_with(
            '100',
            'done',
            done_provenance={
                'kind': 'deterministic-deploy',
                'unit': 'orchestrator-reify.service',
                'note': 'resumed after gate resolution',
            },
        )


# ---------------------------------------------------------------------------
# Step-1: cross-unit deploy success (B6)
# (RED until step-2 adds the before_done execution path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneCrossUnitDeploy:
    """DeterministicRunner — before_done blocking cross-unit deploy success (B6)."""

    async def test_b6_script_runner_called_once_with_before_done(self, tmp_path: Path):
        """script_runner invoked exactly once and receives the full before_done dict (B6)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='200')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        before_done = task['metadata']['before_done']
        script_runner.assert_awaited_once_with(before_done)

    async def test_b6_set_task_done_with_provenance_kind(self, tmp_path: Path):
        """set_task_status awaited once with 'done' + done_provenance.kind='deterministic-deploy' (B6)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='200')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        scheduler.set_task_status.assert_awaited_once()
        call = scheduler.set_task_status.call_args
        assert call.args[0] == '200'
        assert call.args[1] == 'done'
        provenance = call.kwargs.get('done_provenance')
        assert provenance is not None, 'done_provenance must be passed as a kwarg'
        assert provenance['kind'] == 'deterministic-deploy'

    async def test_b6_done_provenance_pid_is_fresh_non_sentinel_int(self, tmp_path: Path):
        """done_provenance.pid is the post-run MainPID — a real non-sentinel int > 0 (B6)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='200')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        call = scheduler.set_task_status.call_args
        provenance = call.kwargs.get('done_provenance')
        assert isinstance(provenance['pid'], int), 'pid must be an int'
        assert provenance['pid'] > 0, 'pid must be a real (non-sentinel) PID'
        assert provenance['pid'] == _FRESH_UNIT_STATE['MainPID'], (
            'pid must be the post-run (fresh) PID, not the baseline'
        )

    async def test_b6_done_provenance_has_active_enter_timestamp(self, tmp_path: Path):
        """done_provenance.active_enter_timestamp present and from post-run inspect (B6)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='200')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        call = scheduler.set_task_status.call_args
        provenance = call.kwargs.get('done_provenance')
        assert 'active_enter_timestamp' in provenance, 'active_enter_timestamp must be in provenance'
        assert provenance['active_enter_timestamp'] == _FRESH_UNIT_STATE['ActiveEnterTimestamp']

    async def test_b6_outcome_is_done(self, tmp_path: Path):
        """Successful cross-unit deploy returns WorkflowOutcome.DONE (B6)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='200')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE

    async def test_b6_stamps_before_done_ran_at(self, tmp_path: Path):
        """update_task stamps before_done_ran_at with a truthy ISO timestamp (B6 / I1)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='200')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        # At least one update_task call must carry before_done_ran_at (truthy)
        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert stamp_calls, 'update_task must be called with a truthy before_done_ran_at stamp'

    async def test_b6_no_escalation_filed_on_success(self, tmp_path: Path):
        """No escalation is filed on a successful deploy (failures file escalations, not success)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='200')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        pending = queue.get_by_task('200', status='pending')
        assert len(pending) == 0, f'No escalation should be filed on success; got {pending}'


# ---------------------------------------------------------------------------
# Default seam: _default_run_script env merge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDefaultRunScriptEnv:
    """_default_run_script merges a partial env dict over os.environ."""

    async def test_partial_env_inherits_path(self, tmp_path: Path):
        """Passing a partial env dict must NOT drop os.environ — PATH must survive.

        Regression guard for the latent footgun where ``create_subprocess_exec``
        received the raw ``before_done['env']`` dict, which completely replaced the
        process environment (no PATH/HOME/XDG_RUNTIME_DIR → most binaries fail to
        resolve).  The fix merges over ``os.environ`` so the deploy script still runs
        in a sane environment while callers can still override individual variables.
        """
        import os
        from unittest.mock import AsyncMock as _AsyncMock
        from unittest.mock import patch

        from escalation.queue import EscalationQueue

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/bin/true',
            'args': [],
            'env': {'MY_DEPLOY_VAR': 'hello'},
            'cwd': str(tmp_path),
            'timeout_secs': 5,
            'target_unit': 'orchestrator-reify.service',
        }

        mock_proc = _AsyncMock()
        mock_proc.communicate = _AsyncMock(return_value=(b'', b''))
        mock_proc.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_run_script(before_done)

        call_kwargs = mock_exec.call_args.kwargs
        passed_env = call_kwargs['env']

        assert passed_env is not None, (
            'env must not be None when before_done.env is non-empty'
        )
        assert 'PATH' in passed_env, (
            f'child must inherit PATH from os.environ when a partial env dict is '
            f'passed; got env keys: {sorted(passed_env)}'
        )
        assert passed_env['PATH'] == os.environ['PATH'], (
            'inherited PATH must match os.environ[PATH]'
        )
        assert passed_env['MY_DEPLOY_VAR'] == 'hello', (
            'custom override from before_done.env must be present in merged env'
        )


# ---------------------------------------------------------------------------
# Step-3: B7a — script rc ≠ 0 failure (RED until step-4 implements the path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneRcNonZero:
    """DeterministicRunner — before_done deploy script fails (rc ≠ 0, B7a)."""

    async def test_b7a_files_infra_issue_escalation(self, tmp_path: Path):
        """rc ≠ 0 → exactly one pending infra_issue escalation for the task (B7a)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='300')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(1, 'boom: unit failed to start'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        pending = queue.get_by_task('300', status='pending')
        assert len(pending) == 1, f'Expected 1 pending escalation, got {len(pending)}'

    async def test_b7a_escalation_is_level_2_critical(self, tmp_path: Path):
        """Filed escalation must be level=2, severity='critical' (born-at-L2, B7a)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='300')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(1, 'boom: unit failed to start'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        escs = queue.get_by_task('300', status='pending')
        assert escs[0].level == 2
        assert escs[0].severity == 'critical'

    async def test_b7a_escalation_sentinel_role_and_category(self, tmp_path: Path):
        """agent_role='orchestrator-deterministic', category='infra_issue' (B7a)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='300')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(1, 'boom: unit failed to start'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        escs = queue.get_by_task('300', status='pending')
        assert escs[0].agent_role == 'orchestrator-deterministic'
        assert escs[0].category == 'infra_issue'

    async def test_b7a_escalation_detail_contains_output_tail_and_unit(self, tmp_path: Path):
        """Escalation detail contains the failing output tail and target_unit (B7a)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='300', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        failing_output = 'boom: unit failed to start'
        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(1, failing_output))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        escs = queue.get_by_task('300', status='pending')
        detail = escs[0].detail
        assert failing_output in detail, f'Output tail must appear in detail: {detail!r}'
        assert 'orchestrator-reify.service' in detail, (
            f'target_unit must appear in detail: {detail!r}'
        )

    async def test_b7a_sets_task_blocked_never_done(self, tmp_path: Path):
        """set_task_status called with 'blocked' and NEVER with 'done' (B7a)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='300')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(1, 'boom: unit failed to start'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        # Must have been called with 'blocked'
        blocked_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if c.args[1] == 'blocked'
        ]
        assert len(blocked_calls) == 1, 'set_task_status must be called once with blocked'
        # Must NEVER have been called with 'done'
        done_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if c.args[1] == 'done'
        ]
        assert len(done_calls) == 0, 'set_task_status must NOT be called with done on failure'

    async def test_b7a_stamps_before_done_ran_at(self, tmp_path: Path):
        """update_task stamps before_done_ran_at even on failure (I1 crash-safe stamp)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='300')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(1, 'boom: unit failed to start'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert stamp_calls, 'update_task must stamp before_done_ran_at even on rc ≠ 0'

    async def test_b7a_outcome_is_blocked(self, tmp_path: Path):
        """rc ≠ 0 → outcome is WorkflowOutcome.BLOCKED (B7a)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='300')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(1, 'boom: unit failed to start'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED


# ---------------------------------------------------------------------------
# Step-5: B7b — verify-fail: stale/missing PID or non-fresh timestamp
# (RED until step-6 implements the verify-fail path)
# ---------------------------------------------------------------------------

# Parametrize two sub-cases:
#   (a) MainPID is a sentinel (0 or '-')
#   (b) ActiveEnterTimestampMonotonic <= baseline (not strictly after)
_STALE_POST_STATES = [
    pytest.param(
        # (a) sentinel: MainPID = 0
        {'MainPID': 0, 'ActiveState': 'failed', 'ActiveEnterTimestamp': 'Mon 2026-06-23 10:01:00 UTC', 'ActiveEnterTimestampMonotonic': 2_000_000},
        id='pid-zero',
    ),
    pytest.param(
        # (a) sentinel: MainPID = '-' (systemd string for inactive)
        {'MainPID': '-', 'ActiveState': 'failed', 'ActiveEnterTimestamp': 'Mon 2026-06-23 10:00:00 UTC', 'ActiveEnterTimestampMonotonic': 2_000_000},
        id='pid-dash',
    ),
    pytest.param(
        # (b) non-fresh monotonic: equal to baseline (not strictly after)
        {'MainPID': 200, 'ActiveState': 'active', 'ActiveEnterTimestamp': 'Mon 2026-06-23 10:00:00 UTC', 'ActiveEnterTimestampMonotonic': 1_000_000},
        id='monotonic-equal-to-baseline',
    ),
    pytest.param(
        # (b) non-fresh monotonic: strictly before baseline
        {'MainPID': 200, 'ActiveState': 'active', 'ActiveEnterTimestamp': 'Mon 2026-06-23 09:59:00 UTC', 'ActiveEnterTimestampMonotonic': 500_000},
        id='monotonic-before-baseline',
    ),
]


@pytest.mark.asyncio
class TestBeforeDoneVerifyFail:
    """DeterministicRunner — before_done verify fails: stale PID / non-fresh timestamp (B7b)."""

    @pytest.mark.parametrize('post_state', _STALE_POST_STATES)
    async def test_b7b_files_exactly_one_infra_issue_escalation(self, tmp_path: Path, post_state: dict):
        """Script rc=0 but stale unit state → exactly one pending infra_issue escalation (B7b)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='400', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, post_state])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        pending = queue.get_by_task('400', status='pending')
        assert len(pending) == 1, f'Expected 1 pending escalation, got {len(pending)}'

    @pytest.mark.parametrize('post_state', _STALE_POST_STATES)
    async def test_b7b_escalation_level_role_category(self, tmp_path: Path, post_state: dict):
        """Filed escalation: level=2, severity='critical', role='orchestrator-deterministic', category='infra_issue'."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='400', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, post_state])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        escs = queue.get_by_task('400', status='pending')
        assert escs[0].level == 2
        assert escs[0].severity == 'critical'
        assert escs[0].agent_role == 'orchestrator-deterministic'
        assert escs[0].category == 'infra_issue'

    @pytest.mark.parametrize('post_state', _STALE_POST_STATES)
    async def test_b7b_escalation_detail_contains_target_unit(self, tmp_path: Path, post_state: dict):
        """Escalation detail mentions the target_unit (needed for operator triage, B7b)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='400', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, post_state])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        escs = queue.get_by_task('400', status='pending')
        assert 'orchestrator-reify.service' in escs[0].detail, (
            f'target_unit must appear in detail: {escs[0].detail!r}'
        )

    @pytest.mark.parametrize('post_state', _STALE_POST_STATES)
    async def test_b7b_sets_blocked_never_done(self, tmp_path: Path, post_state: dict):
        """set_task_status called with 'blocked' and NEVER with 'done' (B7b)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='400')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, post_state])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        blocked_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'blocked']
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(blocked_calls) == 1, 'Must set blocked once'
        assert len(done_calls) == 0, 'Must never set done on verify failure'

    @pytest.mark.parametrize('post_state', _STALE_POST_STATES)
    async def test_b7b_stamps_before_done_ran_at(self, tmp_path: Path, post_state: dict):
        """before_done_ran_at stamped even on verify-fail (I1 crash-safe)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='400')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, post_state])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert stamp_calls, 'update_task must stamp before_done_ran_at on verify-fail'

    @pytest.mark.parametrize('post_state', _STALE_POST_STATES)
    async def test_b7b_outcome_is_blocked(self, tmp_path: Path, post_state: dict):
        """Verify-fail → outcome is WorkflowOutcome.BLOCKED (B7b)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='400')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, post_state])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

# ---------------------------------------------------------------------------
# Step-7: B7 reaper / I1 once-only quiescence
# (RED until step-8 adds the before_done_ran_at idempotency branch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneOnceOnlyIdempotency:
    """DeterministicRunner — before_done_ran_at set + pending escalation → BLOCKED (B7 reaper/I1)."""

    def _pre_seed_infra_escalation(self, queue: EscalationQueue, task_id: str) -> Escalation:
        """Submit one pending infra_issue escalation (simulates a prior failed deploy)."""
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-deterministic',
            severity='critical',
            category='infra_issue',
            summary='Deploy failed: orchestrator-reify.service',
            level=2,
        )
        queue.submit(esc)
        return esc

    async def test_i1_script_runner_not_called(self, tmp_path: Path):
        """before_done_ran_at already set + pending escalation → script_runner NOT called (I1)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='500', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        self._pre_seed_infra_escalation(queue, '500')
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        script_runner.assert_not_awaited()

    async def test_i1_no_second_escalation_filed(self, tmp_path: Path):
        """before_done_ran_at set + pending escalation → queue stays at exactly ONE escalation (I1)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='500', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        self._pre_seed_infra_escalation(queue, '500')
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        pending = queue.get_by_task('500', status='pending')
        assert len(pending) == 1, f'Queue must stay at 1 escalation, got {len(pending)}'

    async def test_i1_set_task_status_never_done(self, tmp_path: Path):
        """before_done_ran_at set + pending escalation → set_task_status NEVER called with 'done'."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='500', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        self._pre_seed_infra_escalation(queue, '500')
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(done_calls) == 0, 'Must NEVER set task to done when quiescent (open escalation)'

    async def test_i1_outcome_is_blocked(self, tmp_path: Path):
        """before_done_ran_at set + pending escalation → outcome is BLOCKED (B7 reaper)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='500', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        self._pre_seed_infra_escalation(queue, '500')
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

# ---------------------------------------------------------------------------
# Step-9: resume-after-resolution (before_done_ran_at set + empty queue)
# (RED until step-10 implements the no-pending resume path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneResumeAfterResolution:
    """DeterministicRunner — before_done ran, a human resolved the failure
    escalation (archived, no longer pending) → drive to done (act-then-ask).

    The distinguishing signal vs the crash-window is that an escalation was
    *ever* filed (it now lives in the archive).  These tests seed a resolved
    escalation so ``get_by_task(status=None)`` finds it while
    ``get_by_task(status='pending')`` is empty.
    """

    def _seed_resolved_escalation(self, queue: EscalationQueue, task_id: str) -> None:
        """Submit then resolve an infra_issue escalation (archived, not pending)."""
        esc = Escalation(
            id=queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-deterministic',
            severity='critical',
            category='infra_issue',
            summary='Deploy failed: orchestrator-reify.service',
            level=2,
        )
        queue.submit(esc)
        queue.resolve(esc.id, 'human verified the unit manually')

    async def test_resume_script_runner_not_called(self, tmp_path: Path):
        """before_done_ran_at set + resolved escalation → script_runner NOT called (I1 no re-run)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='600', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        self._seed_resolved_escalation(queue, '600')  # prior escalation resolved
        scheduler = _mock_scheduler(task)

        # unit_inspector returns same state both calls (non-fresh) so re-run would → blocked
        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        script_runner.assert_not_awaited()

    async def test_resume_drives_to_done(self, tmp_path: Path):
        """before_done_ran_at set + resolved escalation → set_task_status('done') with resume provenance."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='600', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        self._seed_resolved_escalation(queue, '600')
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        # Must have been called with 'done' exactly once
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(done_calls) == 1, f'set_task_status must be called with done; got {scheduler.set_task_status.call_args_list}'
        provenance = done_calls[0].kwargs.get('done_provenance', {})
        assert provenance.get('kind') == 'deterministic-deploy'
        assert provenance.get('note') == 'resumed after human resolution'

    async def test_resume_outcome_is_done(self, tmp_path: Path):
        """before_done_ran_at set + resolved escalation → outcome is WorkflowOutcome.DONE."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='600', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        self._seed_resolved_escalation(queue, '600')
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE


# ---------------------------------------------------------------------------
# Crash-window phantom-done guard (review esc-1900-17)
# A crash between stamping before_done_ran_at and recording a terminal outcome
# (verify stamp OR failure escalation) must NOT drive the task to done.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneCrashWindow:
    """before_done_ran_at set, but neither verified nor ever escalated → re-escalate."""

    async def test_crash_window_does_not_drive_to_done(self, tmp_path: Path):
        """Stamped + empty queue + no verified marker → set_task_status NEVER 'done'."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='700', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # truly empty — no escalation ever filed
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert not done_calls, 'crash-window must NOT phantom-complete the task'

    async def test_crash_window_script_runner_not_called(self, tmp_path: Path):
        """Crash window must NOT re-run the deploy (I1 once-only)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='700', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        script_runner.assert_not_awaited()

    async def test_crash_window_files_infra_escalation_and_blocks(self, tmp_path: Path):
        """Crash window → files one infra_issue escalation, outcome BLOCKED."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='700', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('700', status='pending')
        assert len(pending) == 1, f'crash-window must re-escalate exactly once, got {len(pending)}'
        assert pending[0].category == 'infra_issue'
        blocked_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'blocked']
        assert blocked_calls, 'crash-window must leave the task blocked'


# ---------------------------------------------------------------------------
# Verified-marker resume: crash between verify stamp and the done write.
# before_done_verified_at present → safe to drive to done (no re-run).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneVerifiedResume:
    """before_done_ran_at + before_done_verified_at set, empty queue → done."""

    async def test_verified_resume_drives_to_done(self, tmp_path: Path):
        """Verified marker present → done with the recorded PID, no re-run."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='800',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_verified_at='2026-06-23T10:00:05+00:00',
            before_done_verified_pid=200,
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty — no escalation needed on this path
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        script_runner.assert_not_awaited()
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(done_calls) == 1
        provenance = done_calls[0].kwargs.get('done_provenance', {})
        assert provenance.get('kind') == 'deterministic-deploy'
        assert provenance.get('pid') == 200


# ---------------------------------------------------------------------------
# Step-1 (ε): B8 core — self-target deploy: detached restart, done=scheduled
# (RED until step-2 adds own_unit_resolver + restart_scheduler seams)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSelfRestartScheduled:
    """DeterministicRunner — self-target deploy: scheduling deferred restart, done=scheduled (B8)."""

    async def test_b8_outcome_is_done(self, tmp_path: Path):
        """Self-target deploy schedules restart and returns DONE immediately (B8)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE

    async def test_b8_restart_scheduler_called_once(self, tmp_path: Path):
        """restart_scheduler awaited exactly once (the detached systemd-run call)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        restart_scheduler.assert_awaited_once()

    async def test_b8_script_runner_not_called(self, tmp_path: Path):
        """script_runner must NOT be awaited on self-target path (no blocking cross-unit deploy)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        script_runner.assert_not_awaited()

    async def test_b8_unit_inspector_not_called(self, tmp_path: Path):
        """unit_inspector must NOT be awaited on self-target path (no baseline/verify)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        unit_inspector.assert_not_awaited()

    async def test_b8_stamps_before_done_ran_at(self, tmp_path: Path):
        """update_task must stamp before_done_ran_at with a truthy value (I1 crash-safe)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert stamp_calls, 'update_task must be called with a truthy before_done_ran_at stamp'

    async def test_b8_set_task_status_done_with_scheduled_kind(self, tmp_path: Path):
        """set_task_status awaited with ('850', 'done', done_provenance.kind='deterministic-deploy-scheduled')."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        scheduler.set_task_status.assert_awaited_once()
        call = scheduler.set_task_status.call_args
        assert call.args[0] == '850'
        assert call.args[1] == 'done'
        provenance = call.kwargs.get('done_provenance')
        assert provenance is not None, 'done_provenance must be passed as a kwarg'
        assert provenance['kind'] == 'deterministic-deploy-scheduled'

    async def test_b8_provenance_transient_unit_contains_task_id(self, tmp_path: Path):
        """done_provenance.transient_unit is a non-empty string containing the task id (step-3)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        provenance = scheduler.set_task_status.call_args.kwargs.get('done_provenance', {})
        transient_unit = provenance.get('transient_unit', '')
        assert transient_unit, 'done_provenance.transient_unit must be a non-empty string'
        assert '850' in transient_unit, (
            f"transient_unit must contain the task id '850'; got {transient_unit!r}"
        )

    async def test_b8_provenance_fire_delay_secs_positive_int(self, tmp_path: Path):
        """done_provenance.fire_delay_secs is an int > 0 (step-3)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        provenance = scheduler.set_task_status.call_args.kwargs.get('done_provenance', {})
        fire_delay = provenance.get('fire_delay_secs')
        assert isinstance(fire_delay, int), f'fire_delay_secs must be an int; got {fire_delay!r}'
        assert fire_delay > 0, f'fire_delay_secs must be > 0; got {fire_delay!r}'

    async def test_b8_provenance_unit_equals_target_unit(self, tmp_path: Path):
        """done_provenance.unit equals the target_unit from before_done (step-3)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        provenance = scheduler.set_task_status.call_args.kwargs.get('done_provenance', {})
        assert provenance.get('unit') == 'orchestrator-reify.service', (
            f"done_provenance.unit must equal target_unit; got {provenance.get('unit')!r}"
        )

    async def test_b8_restart_scheduler_called_with_transient_unit_and_delay(self, tmp_path: Path):
        """restart_scheduler awaited with transient_unit and on_active_secs kwargs (step-3)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='850', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        restart_scheduler.assert_awaited_once()
        kwargs = restart_scheduler.call_args.kwargs
        assert 'transient_unit' in kwargs, 'restart_scheduler must receive transient_unit kwarg'
        assert 'on_active_secs' in kwargs, 'restart_scheduler must receive on_active_secs kwarg'
        assert '850' in kwargs['transient_unit'], (
            f"transient_unit kwarg must contain task id '850'; got {kwargs['transient_unit']!r}"
        )
        assert isinstance(kwargs['on_active_secs'], int) and kwargs['on_active_secs'] > 0, (
            f"on_active_secs must be a positive int; got {kwargs['on_active_secs']!r}"
        )


# ---------------------------------------------------------------------------
# Step-5 (ε): B8 scheduling failure → born-at-L2 infra_issue + blocked
# (RED until step-6 implements the rc!=0 error path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSelfRestartSchedulingFailure:
    """DeterministicRunner — self-target deploy: systemd-run scheduling failure (B8/rc≠0)."""

    def _make_runner(self, tmp_path: Path, task: dict, fail_output: str = 'systemd-run: failed to start transient unit'):
        """Build a runner with self-targeting and a failing restart_scheduler."""
        from orchestrator.deterministic_runner import DeterministicRunner
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        restart_scheduler = AsyncMock(return_value=(1, fail_output))
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
        )
        return runner, queue, scheduler

    async def test_b8_failure_outcome_is_blocked(self, tmp_path: Path):
        """Scheduling failure (rc=1) → outcome is WorkflowOutcome.BLOCKED."""
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='860', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        runner, queue, scheduler = self._make_runner(tmp_path, task)

        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

    async def test_b8_failure_files_one_infra_issue_escalation(self, tmp_path: Path):
        """Scheduling failure → exactly one pending infra_issue escalation at level=2."""

        task = _deploy_task(task_id='860', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        runner, queue, scheduler = self._make_runner(tmp_path, task)

        await runner.run(assignment)

        pending = queue.get_by_task('860', status='pending')
        assert len(pending) == 1, f'Expected 1 pending escalation, got {len(pending)}'

    async def test_b8_failure_escalation_level_severity_role_category(self, tmp_path: Path):
        """Filed escalation: level=2, severity='critical', role='orchestrator-deterministic', category='infra_issue'."""

        task = _deploy_task(task_id='860', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        runner, queue, scheduler = self._make_runner(tmp_path, task)

        await runner.run(assignment)

        escs = queue.get_by_task('860', status='pending')
        assert escs[0].level == 2
        assert escs[0].severity == 'critical'
        assert escs[0].agent_role == 'orchestrator-deterministic'
        assert escs[0].category == 'infra_issue'

    async def test_b8_failure_escalation_detail_contains_target_unit(self, tmp_path: Path):
        """Escalation detail contains the target_unit."""

        task = _deploy_task(task_id='860', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        runner, queue, scheduler = self._make_runner(tmp_path, task)

        await runner.run(assignment)

        escs = queue.get_by_task('860', status='pending')
        assert 'orchestrator-reify.service' in escs[0].detail, (
            f'target_unit must appear in detail: {escs[0].detail!r}'
        )

    async def test_b8_failure_escalation_detail_contains_transient_unit(self, tmp_path: Path):
        """Escalation detail contains the transient unit name (includes task id '860')."""

        task = _deploy_task(task_id='860', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        runner, queue, scheduler = self._make_runner(tmp_path, task)

        await runner.run(assignment)

        escs = queue.get_by_task('860', status='pending')
        assert '860' in escs[0].detail, (
            f"transient unit name (containing task id '860') must appear in detail: {escs[0].detail!r}"
        )

    async def test_b8_failure_set_task_blocked_never_done(self, tmp_path: Path):
        """set_task_status called with 'blocked' and NEVER with 'done' on scheduling failure."""

        task = _deploy_task(task_id='860', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        runner, queue, scheduler = self._make_runner(tmp_path, task)

        await runner.run(assignment)

        blocked_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'blocked']
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(blocked_calls) == 1, 'set_task_status must be called once with blocked'
        assert len(done_calls) == 0, 'set_task_status must NOT be called with done on scheduling failure'

    async def test_b8_failure_stamps_before_done_ran_at(self, tmp_path: Path):
        """update_task stamps before_done_ran_at even on scheduling failure (I1 crash-safe)."""

        task = _deploy_task(task_id='860', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        runner, queue, scheduler = self._make_runner(tmp_path, task)

        await runner.run(assignment)

        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert stamp_calls, 'update_task must stamp before_done_ran_at even on scheduling failure (I1)'


# ---------------------------------------------------------------------------
# Step-7 (ε): _default_schedule_detached_restart argv shape / OnFailure wiring
# (RED until step-8 implements the real systemd-run spawn)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDefaultScheduleDetachedRestart:
    """_default_schedule_detached_restart produces correct systemd-run argv and
    OnFailure wiring to the δ escalation-submit CLI."""

    def _make_mock_proc(self, returncode: int = 0) -> object:
        """Return a mock proc with communicate() → (b'', b'') and returncode."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))
        mock_proc.returncode = returncode
        return mock_proc

    async def test_argv_contains_systemd_run_user(self, tmp_path: Path):
        """systemd-run --user must appear in the spawn argv."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-900.service',
                on_active_secs=60,
                task_id='900',
            )

        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        assert 'systemd-run' in all_argv, f'systemd-run must appear in argv: {all_argv!r}'
        assert '--user' in all_argv, f'--user must appear in argv: {all_argv!r}'

    async def test_argv_contains_on_active_and_transient_unit(self, tmp_path: Path):
        """--on-active and the transient unit name must appear in the spawn argv."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-900.service',
                on_active_secs=60,
                task_id='900',
            )

        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        assert '--on-active' in all_argv or 'on-active=60' in all_argv, (
            f'--on-active must appear in argv: {all_argv!r}'
        )
        assert 'orch-redeploy-restart-900.service' in all_argv, (
            f'transient unit name must appear in argv: {all_argv!r}'
        )

    async def test_argv_contains_collect_and_payload_script(self, tmp_path: Path):
        """--collect and the payload script path must appear in the spawn argv."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-900.service',
                on_active_secs=60,
                task_id='900',
            )

        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        assert '--collect' in all_argv, f'--collect must appear in argv: {all_argv!r}'
        assert '/usr/local/bin/restart-deploy.sh' in all_argv, (
            f'payload script must appear in argv: {all_argv!r}'
        )

    async def test_escalation_is_gated_behind_restart_failure(self, tmp_path: Path):
        """The escalation-submit must be wired into a failure branch, NOT run eagerly.

        Previously the handler was a companion ``--unit=`` transient *service*
        registered without ``--on-active`` — systemd-run starts such a unit
        immediately, filing a spurious born-at-L2 on every successful self-deploy.
        The corrected design defers the whole unit via ``--on-active`` and reaches
        the escalation only through a shell failure branch (``[ "$rc" -ne 0 ]``)
        that re-raises the restart's exit code.  Assert that gating is present
        rather than an eager second registration.
        """
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-900.service',
                on_active_secs=60,
                task_id='900',
            )

        # Exactly one systemd-run registration — no eager companion handler unit.
        assert mock_exec.call_count == 1, (
            f'expected a single systemd-run registration, got {mock_exec.call_count} '
            '(an eager OnFailure handler unit would add a second one)'
        )
        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        # The single unit is deferred (so nothing runs at registration) and the
        # escalation is reached only via a shell failure branch.
        assert '--on-active' in all_argv, (
            f'unit must be deferred via --on-active so it does not run eagerly: {all_argv!r}'
        )
        assert '-ne 0' in all_argv, (
            f'escalation must be gated behind a non-zero exit check: {all_argv!r}'
        )

    async def test_handler_does_not_execute_on_success_path(self, tmp_path: Path):
        """Higher-fidelity: simulate systemd firing the wrapped payload.

        Capture the ``/bin/sh -c`` wrapper systemd-run would defer, then execute
        it for real with a SUCCEEDING restart script.  No escalation must be
        filed (the bug was that registration itself filed one eagerly).
        """
        import asyncio as _asyncio
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path / 'q')
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        ok_script = tmp_path / 'deploy-ok.sh'
        ok_script.write_text('#!/bin/sh\nexit 0\n')
        ok_script.chmod(0o755)
        before_done = {
            'script': str(ok_script),
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }

        captured: dict = {}
        real_exec = _asyncio.create_subprocess_exec

        async def fake_exec(*argv, **kwargs):
            captured['argv'] = argv
            return self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', side_effect=fake_exec):
            rc, _ = await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-900.service',
                on_active_secs=1,
                task_id='900',
            )
        assert rc == 0

        argv = captured['argv']
        assert argv[-3] == '/bin/sh' and argv[-2] == '-c', (
            f'expected a /bin/sh -c wrapper payload, got {argv!r}'
        )
        wrapped = argv[-1]

        # Fire the wrapped payload as systemd would (real shell, real CLI path).
        proc = await real_exec(
            '/bin/sh', '-c', wrapped,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.STDOUT,
        )
        await proc.communicate()
        assert proc.returncode == 0, 'success-script wrapper must exit 0'
        assert queue.get_by_task('900') == [], (
            'no escalation may be filed on the success path'
        )

    async def test_handler_executes_on_failure_path(self, tmp_path: Path):
        """Higher-fidelity counterpart: a FAILING restart fires exactly one L2.

        Confirms the failure branch still reaches δ's escalation-submit CLI and
        preserves the non-zero exit code.
        """
        import asyncio as _asyncio
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path / 'q')
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        fail_script = tmp_path / 'deploy-fail.sh'
        fail_script.write_text('#!/bin/sh\nexit 7\n')
        fail_script.chmod(0o755)
        before_done = {
            'script': str(fail_script),
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }

        captured: dict = {}
        real_exec = _asyncio.create_subprocess_exec

        async def fake_exec(*argv, **kwargs):
            captured['argv'] = argv
            return self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', side_effect=fake_exec):
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-901.service',
                on_active_secs=1,
                task_id='901',
            )

        wrapped = captured['argv'][-1]
        proc = await real_exec(
            '/bin/sh', '-c', wrapped,
            stdout=_asyncio.subprocess.PIPE,
            stderr=_asyncio.subprocess.STDOUT,
        )
        await proc.communicate()
        assert proc.returncode == 7, 'failure-script wrapper must preserve the exit code'

        filed = queue.get_by_task('901')
        assert len(filed) == 1, f'exactly one L2 must be filed on failure, got {len(filed)}'
        assert filed[0].category == 'infra_issue'
        assert filed[0].level == 2

    async def test_argv_contains_escalation_submit_cli(self, tmp_path: Path):
        """escalation submit CLI must appear in the spawn argv for OnFailure handling."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-900.service',
                on_active_secs=60,
                task_id='900',
            )

        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        assert 'escalation' in all_argv, f"'escalation' must appear in argv: {all_argv!r}"
        assert 'submit' in all_argv, f"'submit' must appear in argv: {all_argv!r}"

    async def test_argv_contains_queue_dir(self, tmp_path: Path):
        """--queue-dir with the EscalationQueue.queue_dir path must appear in argv."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-900.service',
                on_active_secs=60,
                task_id='900',
            )

        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        assert '--queue-dir' in all_argv, f'--queue-dir must appear in argv: {all_argv!r}'
        assert str(queue.queue_dir) in all_argv, (
            f'queue_dir path {queue.queue_dir!r} must appear in argv: {all_argv!r}'
        )

    async def test_argv_contains_task_id_severity_category(self, tmp_path: Path):
        """--task, --severity critical, --category infra_issue must appear in escalation argv."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-900.service',
                on_active_secs=60,
                task_id='900',
            )

        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        assert '--task' in all_argv, f'--task must appear in argv: {all_argv!r}'
        assert '900' in all_argv, f"task id '900' must appear in argv: {all_argv!r}"
        assert '--severity' in all_argv, f'--severity must appear in argv: {all_argv!r}'
        assert 'critical' in all_argv, f"'critical' must appear in argv: {all_argv!r}"
        assert '--category' in all_argv, f'--category must appear in argv: {all_argv!r}'
        assert 'infra_issue' in all_argv, f"'infra_issue' must appear in argv: {all_argv!r}"


# ---------------------------------------------------------------------------
# Step-9 (ε): own-unit resolution from ORCH_UNIT env var + end-to-end
# self-detection without injected resolver
# (RED until step-10 finalises _default_resolve_own_unit + docstring)
# ---------------------------------------------------------------------------

class TestResolveOwnUnitSync:
    """DeterministicRunner — synchronous unit tests for _default_resolve_own_unit."""

    def test_default_resolve_own_unit_reads_env(self, tmp_path: Path, monkeypatch):
        """_default_resolve_own_unit() returns ORCH_UNIT when set."""
        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        monkeypatch.setenv('ORCH_UNIT', 'orchestrator-reify.service')
        assert runner._default_resolve_own_unit() == 'orchestrator-reify.service'

    def test_default_resolve_own_unit_returns_empty_when_unset(self, tmp_path: Path, monkeypatch):
        """_default_resolve_own_unit() returns '' when ORCH_UNIT is not set."""
        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        monkeypatch.delenv('ORCH_UNIT', raising=False)
        assert runner._default_resolve_own_unit() == ''


@pytest.mark.asyncio
class TestResolveOwnUnit:
    """DeterministicRunner — end-to-end ORCH_UNIT env self-detection without injected resolver."""

    async def test_env_self_detection_takes_self_path(self, tmp_path: Path, monkeypatch):
        """Without own_unit_resolver, ORCH_UNIT==target_unit → self path taken (restart_scheduler awaited, script_runner NOT)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        monkeypatch.setenv('ORCH_UNIT', 'orchestrator-reify.service')

        task = _deploy_task(task_id='870', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        # Construct WITHOUT own_unit_resolver — must use env var path
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            restart_scheduler=restart_scheduler,
            # own_unit_resolver intentionally omitted
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        restart_scheduler.assert_awaited_once()
        script_runner.assert_not_awaited()

        # done_provenance.kind must be 'deterministic-deploy-scheduled'
        call = scheduler.set_task_status.call_args
        provenance = call.kwargs.get('done_provenance', {})
        assert provenance.get('kind') == 'deterministic-deploy-scheduled'

    async def test_env_unset_takes_cross_unit_path(self, tmp_path: Path, monkeypatch):
        """ORCH_UNIT unset → fail-open to cross-unit path (script_runner awaited, kind='deterministic-deploy')."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        monkeypatch.delenv('ORCH_UNIT', raising=False)

        task = _deploy_task(task_id='875', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        # Construct WITHOUT own_unit_resolver — ORCH_UNIT unset → cross-unit path
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            restart_scheduler=restart_scheduler,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        script_runner.assert_awaited_once()
        restart_scheduler.assert_not_awaited()

        # done_provenance.kind must be 'deterministic-deploy' (cross-unit)
        call = scheduler.set_task_status.call_args
        provenance = call.kwargs.get('done_provenance', {})
        assert provenance.get('kind') == 'deterministic-deploy'


# ---------------------------------------------------------------------------
# Step-11 (ε): B8 robustness — self-target + always_escalates=True gates WITHOUT
# running the blocking cross-unit deploy (reviewer: robustness_self_kill).
# (RED until step-12 guards cross-unit deploy with `if not self_target:`)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSelfRestartActThenAskGate:
    """DeterministicRunner — self-target deploy with always_escalates=True: gates WITHOUT blocking deploy."""

    def _make_runner(self, scheduler, queue, unit_inspector, script_runner, restart_scheduler):
        from orchestrator.deterministic_runner import DeterministicRunner
        return DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            own_unit_resolver=lambda: 'orchestrator-reify.service',
            restart_scheduler=restart_scheduler,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )

    def _act_then_ask_task(self):
        task = _deploy_task(task_id='870', target_unit='orchestrator-reify.service')
        task['metadata']['always_escalates'] = True
        return task

    async def test_script_runner_not_awaited(self, tmp_path: Path):
        """Self-kill prevention: script_runner MUST NOT be awaited on self-target path."""
        task = self._act_then_ask_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = self._make_runner(scheduler, queue, unit_inspector, script_runner, restart_scheduler)
        await runner.run(assignment)

        script_runner.assert_not_awaited()

    async def test_unit_inspector_not_awaited(self, tmp_path: Path):
        """No baseline/verify on self-target path: unit_inspector MUST NOT be awaited."""
        task = self._act_then_ask_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = self._make_runner(scheduler, queue, unit_inspector, script_runner, restart_scheduler)
        await runner.run(assignment)

        unit_inspector.assert_not_awaited()

    async def test_restart_scheduler_called_once(self, tmp_path: Path):
        """Detached restart is scheduled exactly once (no double-deploy)."""
        task = self._act_then_ask_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = self._make_runner(scheduler, queue, unit_inspector, script_runner, restart_scheduler)
        await runner.run(assignment)

        restart_scheduler.assert_awaited_once()

    async def test_outcome_is_blocked(self, tmp_path: Path):
        """Self-target act-then-ask falls through to gate and returns BLOCKED."""
        from orchestrator.workflow import WorkflowOutcome

        task = self._act_then_ask_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = self._make_runner(scheduler, queue, unit_inspector, script_runner, restart_scheduler)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

    async def test_set_task_status_never_done(self, tmp_path: Path):
        """set_task_status must NEVER be called with 'done' on act-then-ask self-target path."""
        task = self._act_then_ask_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = self._make_runner(scheduler, queue, unit_inspector, script_runner, restart_scheduler)
        await runner.run(assignment)

        for call in scheduler.set_task_status.await_args_list:
            status = call.args[1] if len(call.args) > 1 else call.kwargs.get('status')
            assert status != 'done', (
                f'set_task_status was called with "done" but must not be on act-then-ask self-target path: {call}'
            )

    async def test_gate_escalation_category_milestone_gate(self, tmp_path: Path):
        """The escalation filed must have category=='milestone_gate' (NOT 'infra_issue')."""
        task = self._act_then_ask_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = self._make_runner(scheduler, queue, unit_inspector, script_runner, restart_scheduler)
        await runner.run(assignment)

        pending = queue.get_by_task('870', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        assert pending[0].category == 'milestone_gate', (
            f'Expected category "milestone_gate", got "{pending[0].category}"'
        )
