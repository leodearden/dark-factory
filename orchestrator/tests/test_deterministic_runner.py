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

    async def test_resume_with_before_done_set_raises_not_implemented(self, tmp_path: Path):
        """Resume path must NOT silently skip before_done — raises NotImplementedError (γ-guard)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        # gate already escalated (so we're on the resume path), but before_done is set
        task = _gate_task(task_id='100', gate_escalated_at='2026-06-23T12:00:00+00:00')
        task['metadata']['before_done'] = 'run_tests'  # γ-incompatible in β
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty — no pending escalation → resume branch
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        # Must raise, not silently skip before_done and drive to done
        with pytest.raises(NotImplementedError):
            await runner.run(assignment)

        # Must NOT have been driven to done
        scheduler.set_task_status.assert_not_awaited()


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
