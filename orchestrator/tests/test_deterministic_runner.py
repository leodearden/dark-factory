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
# Task 3286 — the real task-2902 specimen shape.
#
# `_default_run_script` merges stderr into stdout and returns `[-2000:]`, so a
# chatty predicate script's server-log noise reached `done_provenance.note`
# verbatim — and from there fused-memory's `_format_outcome_echo` wrote it into
# a Mem0 completion summary.  The live note on task 2902 is exactly 1999 chars:
# it starts MID-TOKEN (the 2000-char tail cut), carries FalkorDB identity-scan
# WARNINGs naming the unrelated project `my_solar_challenge` plus `httpx` HTTP
# request lines, and ends with the script's own pretty-printed dry-run JSON
# verdict — the ONE part actually worth keeping.
#
# Shape-faithful, not byte-identical: reproduced from the plan's analysis (the
# live task is a preserved forensic specimen and is never re-read or mutated
# by this suite).  An abridged copy lives in
# scripts/tests/test_scan_provenance_note_log_leaks.py — the two suites cannot
# share imports across orchestrator/tests/ and scripts/tests/.
# ---------------------------------------------------------------------------

POLLUTED_PREDICATE_OUTPUT = """\
_tariff_pence_per_kwh' in group 'my_solar_challenge' (exact-name identity gate should prevent this — investigate)
2026-07-30 16:39:00,523 fused_memory.backends.graphiti_client WARNING identity scan found 3 candidate nodes for 'import_tariff_pence_per_kwh' in group 'my_solar_challenge'
2026-07-30 16:39:00,584 fused_memory.backends.graphiti_client WARNING identity scan found 2 candidate nodes for 'export_tariff_pence_per_kwh' in group 'my_solar_challenge'
2026-07-30 16:39:00,625 httpx INFO HTTP Request: GET http://localhost:6333 "HTTP/1.1 200 OK"
2026-07-30 16:39:00,701 httpx INFO HTTP Request: POST http://localhost:6333/collections/mem0/points/scroll "HTTP/1.1 200 OK"
2026-07-30 16:39:00,742 fused_memory.backends.graphiti_client WARNING identity scan found 4 candidate nodes for 'battery_state_of_charge' in group 'my_solar_challenge'
{
  "dry_run": true,
  "before": {
    "total_source": 0,
    "total_with_kind": 0
  },
  "orphan_count": 0,
  "orphan_ids": [],
  "verdict": "clean"
}"""

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


def _llm_gate_task(
    task_id: str = '99',
    title: str = 'Ship feature gate',
    description: str = 'Gate that guards the feature launch',
    deps: list | None = None,
    gate_options: list | None = None,
    gate_escalated_at: str | None = None,
) -> dict:
    """Build a deterministic pure-gate task dict carrying β's llm-gate marker.

    Mirrors ``_gate_task`` verbatim, then additionally stamps
    ``metadata['x_operational_llm_gate'] = True`` and
    ``metadata['operational_mode'] = 'llm'`` — the exact shape β's
    ``inject_operational_routing`` produces for an
    ``execution_class='operational'`` + ``operational_mode='llm'`` submission
    (fused_memory.middleware.operational_routing_guard.OPERATIONAL_LLM_GATE_MARKER_KEY).
    """
    task = _gate_task(
        task_id=task_id,
        title=title,
        description=description,
        deps=deps,
        gate_options=gate_options,
        gate_escalated_at=gate_escalated_at,
    )
    task['metadata']['x_operational_llm_gate'] = True
    task['metadata']['operational_mode'] = 'llm'
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


def _seed_resolved_gate(queue: EscalationQueue, task_id: str) -> Escalation:
    """Seed a RESOLVED+ARCHIVED deterministic gate escalation for *task_id*.

    This is the true state after a human ``resume``: the L2 gate is archived
    (resolved) BEFORE the task is re-pended (task 2954).  It makes the pure-gate
    resume's archive-inclusive ``own_escalation_resolved`` proof pass, so the
    legitimate drive-to-done stays green — as distinct from a truly-empty queue
    (a LOST escalation), which the hardened resume treats as a strand.
    """
    esc = Escalation(
        id=queue.make_id(task_id), task_id=task_id,
        agent_role='orchestrator-deterministic', severity='critical',
        category='milestone_gate', summary='Ship feature gate', level=2,
    )
    queue.submit(esc)
    queue.resolve(esc.id, 'human resolved the gate', resolved_by='human')
    return esc


def _deploy_task(
    task_id: str = '200',
    target_unit: str | None = 'orchestrator-reify.service',
    script: str = '/tmp/test-deploy.sh',
    args: list | None = None,
    env: dict | None = None,
    cwd: str = '/tmp',
    timeout_secs: int | float = 30,
    before_done_ran_at: str | None = None,
    before_done_verified_at: str | None = None,
    before_done_verified_pid: int | None = None,
    before_done_scheduled_at: dict | None = None,
    description: str = 'Cross-unit deploy of the reify worker',
    phase: str | None = None,
    verify_baseline: dict | None = None,
) -> dict:
    """Build a deterministic deploy task dict (before_done set, always_escalates=False).

    ``phase`` (+ optional ``verify_baseline``), when given, seeds a ζ
    ``metadata['deploy_state']`` slice (``{'phase': phase}``, merging in
    ``verify_baseline`` when provided). Omitting ``phase`` (the default)
    keeps the pre-ζ stamp-only shape — no ``deploy_state`` key at all — so
    existing backward-compat parity tests are unaffected.
    """
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
    if before_done_scheduled_at is not None:
        metadata['before_done_scheduled_at'] = before_done_scheduled_at
    if phase is not None:
        deploy_state: dict = {'phase': phase}
        if verify_baseline is not None:
            deploy_state['verify_baseline'] = verify_baseline
        metadata['deploy_state'] = deploy_state
    return {
        'id': task_id,
        'title': 'Deploy orchestrator-reify',
        'description': description,
        'metadata': metadata,
    }


def _deploy_task_omitted_target_unit(
    task_id: str = '2632',
    script: str = '/tmp/test-deploy.sh',
    args: list | None = None,
    env: dict | None = None,
    cwd: str = '/tmp',
    timeout_secs: int | float = 30,
    description: str = 'Install trickle timer (no target_unit key)',
) -> dict:
    """Build a deterministic deploy task whose ``before_done`` OMITS the
    ``target_unit`` key entirely (task 2632 / esc-2585-1).

    Mirrors ``_deploy_task`` minus ``target_unit`` — this is the REAL shape
    of the install-trickle-timer deploy config that produced esc-2585-1:
    ``run()`` computes ``before_done.get('target_unit', '') == ''`` rather
    than an explicit ``None``.  Kept as a distinct small helper (rather than
    a ``target_unit=None`` sentinel on ``_deploy_task``) so the omitted-key
    shape is faithfully reproduced instead of merely approximated.
    """
    before_done: dict = {
        'script': script,
        'args': args if args is not None else [],
        'env': env if env is not None else {},
        'cwd': cwd,
        'timeout_secs': timeout_secs,
    }
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': False,
        'before_done': before_done,
    }
    return {
        'id': task_id,
        'title': 'Install trickle timer',
        'description': description,
        'metadata': metadata,
    }


def _predicate_task(
    task_id: str = '700',
    title: str = 'Milestone predicate check',
    description: str = 'Predicate that verifies the milestone invariant',
    script: str = '/tmp/test-predicate.sh',
    args: list | None = None,
    env: dict | None = None,
    cwd: str = '/tmp',
    timeout_secs: int | float = 30,
    gate_escalated_at: str | None = None,
) -> dict:
    """Build a deterministic PREDICATE task dict (before_done.kind='predicate', γ-predicate).

    A predicate is a READ-ONLY exit-code verdict check — never a systemd
    deploy — so ``target_unit`` is always None (mirrors ``_deploy_task``'s
    shape, minus the unit to deploy against).  ``gate_escalated_at``, when
    given, seeds ``metadata['gate_escalated_at']`` for the resume/quiescence
    tests (mirrors ``_gate_task``'s same-named parameter).
    """
    before_done: dict = {
        'script': script,
        'args': args if args is not None else [],
        'env': env if env is not None else {},
        'cwd': cwd,
        'timeout_secs': timeout_secs,
        'target_unit': None,
        'kind': 'predicate',
    }
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': False,
        'before_done': before_done,
    }
    if gate_escalated_at is not None:
        metadata['gate_escalated_at'] = gate_escalated_at
    return {
        'id': task_id,
        'title': title,
        'description': description,
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


def _seed_escalation(
    queue: EscalationQueue, task_id: str, agent_role: str, *,
    resolved: bool = False, category: str = 'infra_issue', level: int = 2,
) -> Escalation:
    """Submit (and optionally resolve/archive) an escalation for task_id/agent_role.

    Shared by the Task 2120 escalation-aliasing test classes below so each one
    doesn't re-declare its own near-identical Escalation(...) + submit()/
    resolve() seeding helper.
    """
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=agent_role,
        severity='critical',
        category=category,
        summary=f"seeded escalation ({'resolved' if resolved else 'pending'})",
        level=level,
    )
    queue.submit(esc)
    if resolved:
        queue.resolve(esc.id, 'resolved for test setup')
    return esc


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

        # gate already escalated; escalation resolved+archived (task 2954: the
        # true post-`resume` state — an empty archive would be a LOST-escalation
        # strand, which the hardened resume re-files rather than driving to done).
        task = _gate_task(task_id='100', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_resolved_gate(queue, '100')
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        # Pure-gate leg stamps deterministic-gate provenance (task 2331) so the
        # done write passes require_done_provenance instead of churning forever.
        scheduler.set_task_status.assert_awaited_once_with(
            '100',
            'done',
            done_provenance={
                'kind': 'deterministic-gate',
                'note': 'pure gate resolved',
            },
        )

    async def test_resume_no_new_escalation_filed(self, tmp_path: Path):
        """Resume path must NOT file a new escalation (no re-escalate)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='100', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_resolved_gate(queue, '100')  # resolved+archived — the true post-resume state
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

    async def test_resume_pure_gate_drives_to_done_with_gate_provenance(self, tmp_path: Path):
        """Pure gate resume (before_done=None): gate resolved → done, carrying
        done_provenance.kind='deterministic-gate' (task 2331).

        Bug trace: pre-fix, the pure-gate leg called
        ``set_task_status(task_id, 'done')`` with no done_provenance. With
        ``reconciliation.require_done_provenance=true`` the fused-memory
        done-gate rejects the write with 'done_provenance_required' and the
        scheduler re-dispatches indefinitely (infinite churn). Post-fix, the
        leg mirrors the act-then-ask leg and passes gate provenance so the
        done write is accepted on the first resumed dispatch.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _gate_task(task_id='99', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_resolved_gate(queue, '99')  # resolved+archived — the true post-resume state
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        scheduler.set_task_status.assert_awaited_once_with(
            '99',
            'done',
            done_provenance={
                'kind': 'deterministic-gate',
                'note': 'pure gate resolved',
            },
        )


# ---------------------------------------------------------------------------
# task 2954 step-7/step-8: pure-gate resume-proof hardening.
#
# Today the pure-gate section-1 resume drives a stamped gate to `done` whenever
# the PENDING queue is empty — which, if the born-at-L2 escalation was LOST
# (the reported strand), silently BYPASSES the human gate the moment an operator
# re-pends the stuck task.  Hardening requires the SAME archive-inclusive
# positive proof the deploy path already demands: drive to done only when a
# resolved/archived deterministic escalation actually exists; otherwise re-file
# the gate and stay BLOCKED.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPureGateResumeHardening:
    """DeterministicRunner — pure-gate resume requires archive-inclusive proof."""

    async def test_pure_gate_resume_no_record_refiles_and_blocks(self, tmp_path: Path):
        """gate_escalated_at set + ZERO records anywhere (no pending, no archived)
        → must NOT drive to done; re-files the born-at-L2 gate and returns BLOCKED."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _gate_task(task_id='100', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # ZERO records — no pending, no archived
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        # Must NOT silently drive to done — the human gate was never proven resolved.
        done_calls = [
            c for c in scheduler.set_task_status.await_args_list
            if len(c.args) >= 2 and c.args[1] == 'done'
        ]
        assert not done_calls, 'pure-gate resume must not drive to done without proof'
        # Instead it re-files the born-at-L2 milestone_gate to re-establish the gate.
        pending = queue.get_by_task(
            '100', status='pending', agent_role='orchestrator-deterministic',
        )
        assert len(pending) == 1
        assert pending[0].category == 'milestone_gate'
        assert pending[0].level == 2

    async def test_pure_gate_resume_with_resolved_record_drives_to_done(self, tmp_path: Path):
        """GREEN-preserving: a RESOLVED+ARCHIVED deterministic escalation (the true
        post-`resume` state) IS positive proof a human acted → drive to done with
        deterministic-gate provenance, exactly as before the hardening."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _gate_task(task_id='99', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        # A human `resume` archives the escalation before the task is re-pended.
        existing = Escalation(
            id=queue.make_id('99'), task_id='99',
            agent_role='orchestrator-deterministic', severity='critical',
            category='milestone_gate', summary='Ship feature gate', level=2,
        )
        queue.submit(existing)
        queue.resolve(existing.id, 'human approved the gate', resolved_by='human')
        # Sanity: resolved record is archived (not pending) but archive-visible.
        assert queue.get_by_task('99', status='pending') == []
        assert len(queue.get_by_task('99', agent_role='orchestrator-deterministic')) == 1

        scheduler = _mock_scheduler(task)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        scheduler.set_task_status.assert_awaited_once_with(
            '99',
            'done',
            done_provenance={
                'kind': 'deterministic-gate',
                'note': 'pure gate resolved',
            },
        )

    async def test_always_escalates_gate_stamp_implies_queryable_escalation(self, tmp_path: Path):
        """Regression guard (task 2954, the explicitly-requested end-to-end
        deliverable): a fresh always_escalates pure-gate dispatch must yield a
        DURABLY queryable L2 escalation AND stamp gate_escalated_at — pinning the
        'stamp ⟹ queryable escalation' invariant whose violation (stamp present,
        record absent) is the reported bug, and guarding the file-before-stamp
        durability contract against regression."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _gate_task(task_id='2954', gate_options=['approve', 'reject'])
        assert 'gate_escalated_at' not in task['metadata']  # fresh dispatch
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        # (d) a fresh gate blocks awaiting a human decision
        assert outcome == WorkflowOutcome.BLOCKED
        # (a) exactly one queryable born-at-L2 escalation, scoped to the runner role
        pending = queue.get_by_task(
            '2954', status='pending', agent_role='orchestrator-deterministic',
        )
        assert len(pending) == 1
        esc = pending[0]
        assert esc.level == 2
        assert esc.category == 'milestone_gate'
        # (b) the record round-trips from DISK — durable persistence, not just an
        # in-memory list — proving the invariant end-to-end.
        on_disk = queue.get(esc.id)
        assert on_disk is not None
        assert on_disk.id == esc.id
        assert on_disk.task_id == '2954'
        # (c) gate_escalated_at is stamped via a metadata merge
        stamp_calls = [
            c for c in scheduler.update_task.await_args_list
            if len(c.args) >= 2 and isinstance(c.args[1], dict)
            and 'gate_escalated_at' in c.args[1]
        ]
        assert stamp_calls, 'gate_escalated_at must be stamped on first dispatch'
        assert stamp_calls[0].kwargs.get('metadata_mode') == 'merge'


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

    async def test_b6_stamps_deploy_state_phase_ran_atomically_with_before_done_ran_at(
        self, tmp_path: Path
    ):
        """ζ DS-1: the SAME update_task call that stamps before_done_ran_at
        also carries deploy_state.phase=='ran' (one atomic merge write)."""
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

        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert stamp_calls, 'update_task must be called with a truthy before_done_ran_at stamp'
        payload = stamp_calls[0].args[1]
        assert payload.get('deploy_state', {}).get('phase') == 'ran'
        # Reviewer amendment (task 2240): deploy_state.ran_at must mirror the
        # SAME timestamp as the top-level before_done_ran_at stamp — not
        # stay null forever (it previously carried the OLD state's ran_at,
        # which was always None since nothing ever populated it).
        assert payload['deploy_state']['ran_at'] == payload['before_done_ran_at']

    async def test_b6_persists_verify_baseline_from_captured_baseline(self, tmp_path: Path):
        """ζ DS-3: deploy_state.verify_baseline is persisted from the captured
        pre-deploy baseline (monotonic + MainPID) once it has been inspected."""
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

        baseline_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('verify_baseline')
        ]
        assert baseline_calls, 'deploy_state.verify_baseline must be persisted somewhere in the call chain'
        vb = baseline_calls[0].args[1]['deploy_state']['verify_baseline']
        assert vb['main_pid'] == _BASELINE_UNIT_STATE['MainPID']
        assert vb['active_enter_timestamp_monotonic'] == (
            _BASELINE_UNIT_STATE['ActiveEnterTimestampMonotonic']
        )

    async def test_act_then_ask_advances_deploy_state_phase_ran_to_escalated_at_gate(
        self, tmp_path: Path,
    ):
        """ζ DS-1: cross-unit act-then-ask (always_escalates=True) — the
        deploy runs and verifies, then falls through to the milestone gate
        WITHOUT writing done; the gate's update_task call carries
        deploy_state.phase=='escalated'."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='250', target_unit='orchestrator-reify.service')
        task['metadata']['always_escalates'] = True
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

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('250', status='pending')
        assert len(pending) == 1
        assert pending[0].category == 'milestone_gate'

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls
        assert deploy_state_calls[-1].args[1]['deploy_state']['phase'] == 'escalated'
        # Reviewer amendment (task 2240): deploy_state.escalated_at must
        # mirror the SAME timestamp as the top-level gate_escalated_at
        # stamp written in this exact call, not stay null forever.
        assert (
            deploy_state_calls[-1].args[1]['deploy_state']['escalated_at']
            == deploy_state_calls[-1].args[1]['gate_escalated_at']
        )


# ---------------------------------------------------------------------------
# Task 2632 / esc-2585-1 residual fix: a cross-unit before_done deploy whose
# target_unit is falsy (None, or the key omitted entirely — before_done.get(
# 'target_unit', '') == '') has no specific systemd unit to baseline-inspect
# or fresh-PID-verify. Skip that machinery entirely and drive the outcome on
# the deploy script's exit code alone.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneTargetUnitlessDeploy:
    """DeterministicRunner — before_done cross-unit deploy with a falsy
    target_unit (None, or the key omitted entirely) skips the baseline/
    fresh-PID-verify machinery and drives the outcome on the script's exit
    code alone (task 2632 / esc-2585-1 residual fix)."""

    @pytest.mark.parametrize(
        'build_task',
        [
            pytest.param(
                lambda: _deploy_task(task_id='2632', target_unit=None),
                id='target_unit=None',
            ),
            pytest.param(
                lambda: _deploy_task_omitted_target_unit(task_id='2632'),
                id='target_unit-key-omitted',
            ),
        ],
    )
    async def test_targetless_deploy_rc0_drives_done_without_baseline_inspect(
        self, tmp_path: Path, build_task,
    ):
        """rc==0 with a falsy target_unit drives straight to done WITHOUT
        ever inspecting the unit (baseline or verify) — the runner must not
        inspect the empty unit name '' the way esc-2585-1's config did.

        RED today: run() unconditionally inspects target_unit (''), which
        returns a degenerate ActiveState-less dict, tripping the baseline
        gate and filing a false-block infra_issue escalation instead of
        driving to done.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = build_task()
        tid = str(task['id'])
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock()
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        unit_inspector.assert_not_awaited()
        assert queue.get_by_task(tid, status='pending') == []

        scheduler.set_task_status.assert_awaited_once()
        call = scheduler.set_task_status.call_args
        assert call.args[1] == 'done'
        assert call.kwargs['done_provenance']['kind'] == 'deterministic-deploy'

        verified_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_verified_at')
        ]
        assert verified_calls, (
            'update_task must stamp before_done_verified_at (crash-safe verified stamp)'
        )

    async def test_targetless_deploy_rc_nonzero_escalates_and_blocks(self, tmp_path: Path):
        """rc != 0 with a falsy target_unit must escalate + block — NOT
        drive to done — and must never touch unit_inspector.

        RED today: impl-targetless-happy's branch ignores rc entirely and
        always writebacks done, regardless of the script's exit code.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2632', target_unit=None)
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock()
        script_runner = AsyncMock(return_value=(1, 'boom: install failed'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('2632', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'infra_issue'
        assert esc.agent_role == 'orchestrator-deterministic'

        done_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if c.args[1] == 'done'
        ]
        assert done_calls == [], 'set_task_status must NOT be called with done on failure'
        unit_inspector.assert_not_awaited()

    async def test_targetless_deploy_outer_guard_timeout_escalates(self, tmp_path: Path):
        """A hung run_fn under a falsy target_unit must still trip the outer
        wall-clock guard and produce exactly one born-at-L2 infra_issue —
        never hang run() forever.

        RED today: the target_unit-less branch has no try/except around
        asyncio.wait_for, so the TimeoutError propagates uncaught instead of
        being translated into a BLOCKED outcome.
        """
        import asyncio

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2632', target_unit=None, timeout_secs=0)
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        async def _hang(_before_done):
            await asyncio.Event().wait()

        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=_hang,
            run_timeout_grace_secs=0,
        )

        # Hang tripwire: if the outer guard regresses, fail loudly instead of
        # stalling the suite.
        outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('2632', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'infra_issue'
        assert esc.agent_role == 'orchestrator-deterministic'
        unit_inspector.assert_not_awaited()

    async def test_targetless_deploy_run_fn_unexpected_error_escalates_and_blocks(
        self, tmp_path: Path,
    ):
        """A non-timeout run_fn error (e.g. a bug in the script runner seam,
        as opposed to a non-zero exit code or the outer guard firing) under a
        falsy target_unit must still escalate + block via exactly one
        born-at-L2 infra_issue, with a summary distinct from the rc!=0 and
        outer-guard-timeout cases, and must never touch unit_inspector.

        Covers the `except Exception as exc:` branch of
        DeterministicRunner._run_deploy_script_guarded, reached from the
        target_unit-less branch of run() (task 2632 review amendment)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2632', target_unit=None)
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock()
        script_runner = AsyncMock(side_effect=ValueError('boom'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('2632', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'infra_issue'
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.summary == 'Deploy run_fn failed (unexpected error, no target_unit)'

        done_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if c.args[1] == 'done'
        ]
        assert done_calls == [], (
            'set_task_status must NOT be called with done on an unexpected run_fn error'
        )
        unit_inspector.assert_not_awaited()

    async def test_targetless_always_escalates_true_runs_then_gates(self, tmp_path: Path):
        """always_escalates=True (act-then-ask) with a falsy target_unit:
        the script still runs, but rc==0 must fall through to the milestone
        gate instead of unconditionally writing done (task 2632).

        RED today: impl-targetless-failure's rc==0 branch always calls
        _writeback_deploy_success, ignoring always_escalates entirely.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2632', target_unit=None)
        task['metadata']['always_escalates'] = True
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock()
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        script_runner.assert_awaited_once()
        unit_inspector.assert_not_awaited()
        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('2632', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        assert pending[0].category == 'milestone_gate'

        done_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if c.args[1] == 'done'
        ]
        assert done_calls == [], 'set_task_status must NOT be called with done — must gate instead'

    async def test_named_target_wedged_baseline_still_blocks(self, tmp_path: Path):
        """Regression guard for acceptance #3: a NAMED target_unit whose
        baseline inspect is wedged (ActiveState=='') must still be caught
        by the baseline gate and block — the new falsy-target_unit branch
        must not have weakened genuine-wedge detection for a truthy unit."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2633', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        unit_inspector = AsyncMock(return_value={
            'MainPID': 0, 'ActiveState': '', 'ActiveEnterTimestamp': '',
            'ActiveEnterTimestampMonotonic': 0,
        })
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('2633', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        assert pending[0].category == 'infra_issue'
        assert 'Baseline inspect failed' in pending[0].summary
        script_runner.assert_not_called()

    async def test_named_target_happy_path_still_double_inspects_and_drives_done(
        self, tmp_path: Path,
    ):
        """Regression guard for acceptance #3: a NAMED target_unit happy
        path must still await the inspector TWICE (baseline + verify) and
        drive to done — proving the falsy-target_unit branch condition did
        not divert a truthy/named unit off the baseline/verify path."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2634', target_unit='orchestrator-reify.service')
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
        assert unit_inspector.await_count == 2, (
            'a named target_unit must still be baseline- AND verify-inspected'
        )
        scheduler.set_task_status.assert_awaited_once()
        assert scheduler.set_task_status.call_args.args[1] == 'done'
        assert queue.get_by_task('2634', status='pending') == []


# ---------------------------------------------------------------------------
# ζ D2 boundary: an illegal deploy-phase transition files a REAL born-at-L2
# escalation before raising — never a silent write.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDeployTransitionEnforcement:
    """ζ D2: DeterministicRunner._advance_deploy_phase enforces the _LEGAL
    transition table via a REAL EscalationQueue — an illegal edge (e.g.
    scheduled->done) files a born-at-L2 escalation observable via
    get_by_task, THEN raises IllegalDeployTransition."""

    async def test_illegal_transition_files_l2_escalation_before_raising(
        self, tmp_path: Path,
    ):
        """scheduled->done is pinned-illegal (D2 boundary signal): driving
        _advance_deploy_phase there must file a born-at-L2 escalation into a
        REAL EscalationQueue AND raise IllegalDeployTransition."""
        from orchestrator.deploy_state import DeployPhase, IllegalDeployTransition
        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(_deploy_task(task_id='999'))
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)

        metadata = {'deploy_state': {'phase': 'scheduled'}}

        with pytest.raises(IllegalDeployTransition):
            await runner._advance_deploy_phase('999', metadata, DeployPhase.DONE)

        escs = queue.get_by_task('999')
        assert len(escs) == 1, f'Expected exactly 1 escalation, got {len(escs)}'
        esc = escs[0]
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'illegal_deploy_transition'

    async def test_no_update_task_write_occurs_on_illegal_transition(
        self, tmp_path: Path,
    ):
        """D2 file-before-raise: the illegal transition must never reach
        update_task — no silent write of a bogus phase."""
        from orchestrator.deploy_state import DeployPhase, IllegalDeployTransition
        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(_deploy_task(task_id='999'))
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)

        metadata = {'deploy_state': {'phase': 'scheduled'}}

        with pytest.raises(IllegalDeployTransition):
            await runner._advance_deploy_phase('999', metadata, DeployPhase.DONE)

        scheduler.update_task.assert_not_awaited()


# ---------------------------------------------------------------------------
# Task 2238 (W10-δ), step-3: run()'s cross-unit `if not self_target:` branch
# delegates run+verify to proc_supervision.RestartPlan.execute() with a
# FreshPidVerify instead of an inline run-then-reinspect block.
# (RED until step-4 rewrites the cross-unit branch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCrossUnitDeployDelegatesToRestartPlan:
    """DeterministicRunner.run()'s cross-unit branch delegates run+verify to
    proc_supervision.RestartPlan.execute() (task 2238/δ)."""

    def _make_runner(
        self, tmp_path: Path, task: dict, *,
        own_unit_resolver=lambda: 'orchestrator.service',
        script_runner=None,
        unit_inspector=None,
        **extra,
    ):
        """Build a runner with a KNOWN, non-target own_unit_resolver (so the
        cross-unit branch is reached with a truthy own_unit, distinct from
        the ORCH_UNIT-unset fail-open case covered separately below)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=(
                unit_inspector if unit_inspector is not None
                else AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
            ),
            script_runner=(
                script_runner if script_runner is not None
                else AsyncMock(return_value=(0, 'ok'))
            ),
            own_unit_resolver=own_unit_resolver,
            **extra,
        )
        return runner, queue, scheduler

    async def test_execute_awaited_once_with_cross_unit_fresh_pid_verify_plan(self, tmp_path: Path):
        """RestartPlan.execute() awaited once; the constructed plan carries a
        FreshPidVerify built from the persisted baseline, no transient_unit,
        no on_failure_escalation, and a truthy own_unit != target_unit."""
        from unittest.mock import patch

        from orchestrator.proc_supervision import (
            FreshPidVerify,
            RestartDisposition,
            RestartOutcome,
            RestartPlan,
        )

        task = _deploy_task(task_id='960', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        runner, queue, scheduler = self._make_runner(tmp_path, task)

        captured_plans: list = []
        canned = RestartOutcome(disposition=RestartDisposition.DEPLOYED_AND_VERIFIED)

        async def _fake_execute(self, *, runner=None, inspector=None):
            captured_plans.append(self)
            return canned

        with patch.object(RestartPlan, 'execute', _fake_execute):
            await runner.run(assignment)

        assert len(captured_plans) == 1, 'RestartPlan.execute must be awaited exactly once'
        plan = captured_plans[0]

        assert isinstance(plan.verify, FreshPidVerify)
        assert plan.verify.baseline_main_pid == _BASELINE_UNIT_STATE['MainPID']
        assert plan.verify.baseline_active_enter_monotonic == (
            _BASELINE_UNIT_STATE['ActiveEnterTimestampMonotonic']
        )
        assert plan.verify.inspect_timeout_secs == runner._inspect_timeout_secs
        assert plan.transient_unit is None
        assert plan.on_failure_escalation is None
        assert plan.target_unit == 'orchestrator-reify.service'
        assert plan.own_unit, 'own_unit must be truthy'
        assert plan.own_unit != plan.target_unit

    async def test_deployed_and_verified_happy_path_via_real_execute(self, tmp_path: Path):
        """A REAL execute() DEPLOYED_AND_VERIFIED disposition drives done with
        fresh provenance, routes the run through the hardened seam, and
        re-inspects exactly twice (baseline + the capturing verify)."""
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='961', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))
        runner, queue, scheduler = self._make_runner(
            tmp_path, task, unit_inspector=unit_inspector, script_runner=script_runner,
        )

        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        scheduler.set_task_status.assert_awaited_once()
        call = scheduler.set_task_status.call_args
        assert call.args[1] == 'done'
        provenance = call.kwargs.get('done_provenance')
        assert provenance['kind'] == 'deterministic-deploy'
        assert provenance['pid'] == _FRESH_UNIT_STATE['MainPID']
        assert provenance['active_enter_timestamp'] == _FRESH_UNIT_STATE['ActiveEnterTimestamp']

        script_runner.assert_awaited_once_with(task['metadata']['before_done'])
        assert unit_inspector.await_count == 2

    async def test_restart_failed_disposition_files_deploy_failed_infra_issue(self, tmp_path: Path):
        """script_runner rc≠0 -> RestartDisposition.RESTART_FAILED -> blocked,
        summary 'Deploy failed: {unit}' (unchanged from pre-delegation)."""
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='962', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        script_runner = AsyncMock(return_value=(1, 'boom'))
        runner, queue, scheduler = self._make_runner(
            tmp_path, task,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=script_runner,
        )

        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('962', status='pending')
        assert len(pending) == 1
        assert pending[0].summary == 'Deploy failed: orchestrator-reify.service'
        assert 'boom' in pending[0].detail

    async def test_verify_failed_disposition_files_deploy_verify_failed_infra_issue(self, tmp_path: Path):
        """Fresh inspect returns the MainPID=0 sentinel -> RestartDisposition.
        VERIFY_FAILED -> blocked, summary 'Deploy verify failed: {unit}'
        (unchanged from pre-delegation)."""
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='963', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        stale_state = {
            'MainPID': 0, 'ActiveState': 'failed',
            'ActiveEnterTimestamp': 'Mon 2026-06-23 10:01:00 UTC',
            'ActiveEnterTimestampMonotonic': 2_000_000,
        }
        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, stale_state])
        runner, queue, scheduler = self._make_runner(
            tmp_path, task, unit_inspector=unit_inspector,
        )

        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('963', status='pending')
        assert len(pending) == 1
        assert pending[0].summary == 'Deploy verify failed: orchestrator-reify.service'

    async def test_fail_open_orch_unit_unset_still_runs_deploy_through_execute(
        self, tmp_path: Path, monkeypatch,
    ):
        """ORCH_UNIT unset (own_unit='') must still RUN the cross-unit deploy
        via execute()'s RP-2 path (through the non-self sentinel) rather than
        RP-1 REFUSED — regression lock for test_env_unset_takes_cross_unit_path."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        monkeypatch.delenv('ORCH_UNIT', raising=False)
        task = _deploy_task(task_id='964', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])

        # Construct WITHOUT own_unit_resolver — ORCH_UNIT (unset) governs.
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )

        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        script_runner.assert_awaited_once()
        pending = queue.get_by_task('964', status='pending')
        assert pending == [], 'RP-1 must never REFUSE this fail-open path'

    async def test_hanging_script_runner_trips_outer_guard_infra_issue(self, tmp_path: Path):
        """A script_runner that hangs forever still trips the outer wait_for
        guard (task 2090 Layer B) around the delegated execute() call."""
        import asyncio as _asyncio

        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='965', target_unit='orchestrator-reify.service', timeout_secs=0,
        )
        assignment = _make_assignment(task)

        async def _hang(_before_done):
            await _asyncio.Event().wait()

        runner, queue, scheduler = self._make_runner(
            tmp_path, task,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=_hang,
            run_timeout_grace_secs=0.05,
        )

        outcome = await _asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('965', status='pending')
        assert len(pending) == 1
        assert pending[0].summary.startswith('Deploy run+verify exceeded outer guard')

    async def test_deployed_and_verified_still_stamps_deploy_state_phase_ran(self, tmp_path: Path):
        """Even routed through the REAL RestartPlan.execute() delegation, the
        shared before_done_ran_at write still carries deploy_state.phase=='ran'
        (ζ DS-1 — the phase write happens before the self/cross split, not
        inside the delegated execute() call)."""
        task = _deploy_task(task_id='966', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))
        runner, queue, scheduler = self._make_runner(
            tmp_path, task, unit_inspector=unit_inspector, script_runner=script_runner,
        )

        await runner.run(assignment)

        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert stamp_calls, 'update_task must be called with a truthy before_done_ran_at stamp'
        assert stamp_calls[0].args[1]['deploy_state']['phase'] == 'ran'

    async def test_empty_baseline_install_fresh_timer_deploy_is_done_not_blocked(
        self, tmp_path: Path,
    ):
        """esc-2584-1 regression (task 2611): a deploy whose target_unit is a
        systemd ``.timer`` (installing a ``Type=oneshot`` service) ALWAYS
        reports MainPID=0 — both at the pre-deploy baseline inspect and at
        the post-deploy verify inspect — even when the install succeeded and
        the unit is genuinely active. This must converge to DONE with
        pid==0 provenance, not be falsely BLOCKED with an infra_issue
        escalation."""
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='967', target_unit='legibility-trickle@dark-factory.timer',
        )
        assignment = _make_assignment(task)
        baseline_state = {
            'MainPID': 0,
            'ActiveState': 'inactive',
            'ActiveEnterTimestamp': '',
            'ActiveEnterTimestampMonotonic': 0,
        }
        fresh_state = {
            'MainPID': 0,
            'ActiveState': 'active',
            'ActiveEnterTimestamp': 'Mon 2026-06-23 10:01:00 UTC',
            'ActiveEnterTimestampMonotonic': 5_000_000,
        }
        unit_inspector = AsyncMock(side_effect=[baseline_state, fresh_state])
        script_runner = AsyncMock(return_value=(0, 'ok'))
        runner, queue, scheduler = self._make_runner(
            tmp_path, task, unit_inspector=unit_inspector, script_runner=script_runner,
        )

        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        scheduler.set_task_status.assert_awaited_once()
        call = scheduler.set_task_status.call_args
        assert call.args[1] == 'done'
        provenance = call.kwargs.get('done_provenance')
        assert provenance['kind'] == 'deterministic-deploy'
        assert provenance['pid'] == 0
        assert provenance['unit'] == 'legibility-trickle@dark-factory.timer'

        pending = queue.get_by_task('967', status='pending')
        assert pending == [], 'a genuinely-successful install-fresh deploy must not escalate'


# ---------------------------------------------------------------------------
# Task 2066: cross-unit writeback resilience — severed-then-recovered connection
# (RED until the _writeback_deploy_success helper + constructor seams land)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCrossUnitWritebackResilience:
    """DeterministicRunner — cross-unit deploy verify+writeback survives a
    self-severed fused-memory connection (task 2066).

    The deploy script may restart the very service backing the orchestrator's
    own fused-memory/MCP connection (e.g. task 2059), severing it for the
    duration of a `--drain` restart.  These tests simulate that by making
    scheduler.update_task / scheduler.set_task_status transiently fail before
    recovering, and assert the runner patiently retries the writeback instead
    of silently stranding the task (the EVIDENCE failure on task 2059).
    """

    async def test_verified_stamp_retried_past_transient_connection_failure(self, tmp_path: Path):
        """update_task is retried past a transient severed connection until the
        verified stamp durably persists, then done is set with fresh provenance."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2066', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        # Connection severed for the verified-stamp write's first two attempts,
        # recovered on the 3rd (before_done_ran_at's stamp is unaffected — True).
        verified_call_returns: list[bool] = []

        def _update_task_side_effect(_task_id, metadata, **_kwargs):
            if 'before_done_verified_at' in metadata:
                result = len(verified_call_returns) >= 2
                verified_call_returns.append(result)
                return result
            return True

        scheduler.update_task = AsyncMock(side_effect=_update_task_side_effect)
        sleeper = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            writeback_backoffs=[0.01] * 4,
            sleeper=sleeper,
        )
        outcome = await runner.run(assignment)

        # (a) update_task retried more than once for the verified stamp, and the
        # final such call returned True — the stamp durably persisted despite
        # the transient failure.
        assert len(verified_call_returns) > 1, (
            f'Expected the verified-stamp write to be retried; got only '
            f'{len(verified_call_returns)} attempt(s): {verified_call_returns}'
        )
        assert verified_call_returns[-1] is True, (
            'Final before_done_verified_at update_task call must have returned True'
        )

        # (b) paced retry — the injected sleeper was awaited at least once.
        sleeper.assert_awaited()

        # (c) set_task_status awaited with done + fresh provenance; outcome DONE.
        scheduler.set_task_status.assert_awaited_once()
        call = scheduler.set_task_status.call_args
        assert call.args[0] == '2066'
        assert call.args[1] == 'done'
        provenance = call.kwargs.get('done_provenance')
        assert provenance is not None, 'done_provenance must be passed as a kwarg'
        assert provenance['kind'] == 'deterministic-deploy'
        assert provenance['pid'] == _FRESH_UNIT_STATE['MainPID']
        assert outcome == WorkflowOutcome.DONE

        # (d) I1 — the deploy script is NEVER re-run, even across writeback retries.
        script_runner.assert_awaited_once()

    async def test_done_write_retried_past_transient_set_task_status_error(self, tmp_path: Path):
        """A transient RuntimeError from set_task_status('done', ...) is retried
        within budget rather than propagating out of run() (task 2066)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2066', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        # before_done_verified_at stamps fine (connection alive for that write);
        # the FIRST done-write hits a transient RuntimeError (connection severed
        # mid-writeback), the second succeeds (connection recovered).
        scheduler.set_task_status = AsyncMock(
            side_effect=[RuntimeError('transient: fused-memory unavailable'), None]
        )
        sleeper = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            writeback_backoffs=[0.01] * 4,
            sleeper=sleeper,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE, (
            'A transient set_task_status RuntimeError must be retried within '
            'budget, not propagate out of run()'
        )
        done_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if len(c.args) > 1 and c.args[1] == 'done'
        ]
        assert len(done_calls) >= 2, (
            f'Expected set_task_status to be retried at least twice for the '
            f'done write; got {len(done_calls)} call(s)'
        )
        script_runner.assert_awaited_once()
        assert queue.get_by_task('2066') == [], (
            'No escalation should be filed when the writeback recovers within budget'
        )

    async def test_persistent_severed_connection_files_durable_escalation(self, tmp_path: Path):
        """A PERSISTENTLY severed connection (never recovers in-window) files a
        durable infra_issue escalation instead of silently stranding the task
        with an empty queue — the direct EVIDENCE regression guard (task 2059/2066)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2066', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        def _update_task_side_effect(_task_id, metadata, **_kwargs):
            # Connection never recovers in-window for the verified stamp.
            return 'before_done_verified_at' not in metadata

        scheduler.update_task = AsyncMock(side_effect=_update_task_side_effect)
        sleeper = AsyncMock()
        backoffs = [0.01] * 2

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            writeback_backoffs=backoffs,
            sleeper=sleeper,
        )
        outcome = await runner.run(assignment)

        # (a) durable, connection-independent escalation filed — the queue is
        # never empty, unlike the EVIDENCE failure on task 2059.
        pending = queue.get_by_task('2066')
        assert pending, 'Budget exhaustion must file a durable escalation, not leave the queue empty'
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.category == 'infra_issue'
        assert 'orchestrator-reify.service' in esc.detail
        assert 'severed' in esc.detail.lower() or 'connection' in esc.detail.lower(), (
            f'Detail must explain the self-severed-connection/writeback-stranding '
            f'cause: {esc.detail!r}'
        )

        # (b) outcome is BLOCKED, never a propagated raw exception.
        assert outcome == WorkflowOutcome.BLOCKED

        # (c) before_done_ran_at was stamped pre-deploy (I1 crash-safe stamp).
        ran_at_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert ran_at_calls, 'before_done_ran_at must be stamped before the deploy runs'

        # (d) bounded attempts — no infinite loop.
        verified_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and 'before_done_verified_at' in c.args[1]
        ]
        assert len(verified_calls) == len(backoffs) + 1, (
            f'Expected exactly len(backoffs)+1 ({len(backoffs) + 1}) verified-stamp '
            f'attempts; got {len(verified_calls)}'
        )
        assert sleeper.await_count == len(backoffs), (
            f'Expected exactly len(backoffs) ({len(backoffs)}) sleeps; got {sleeper.await_count}'
        )

        # (e) the deploy script is never re-run.
        script_runner.assert_awaited_once()

    async def test_persistent_severed_connection_with_blocked_write_failure_still_returns_blocked(
        self, tmp_path: Path,
    ):
        """Amendment regression guard (task 2066): when EVERY scheduler write
        rides the same severed connection — including the fallback
        set_task_status('blocked') issued by _file_infra_issue_and_block after
        the writeback budget is exhausted — the durable escalation must still
        be filed and run() must still return BLOCKED, never a propagated
        exception.  Unlike test_persistent_severed_connection_files_durable_escalation
        (which leaves set_task_status healthy), this simulates the connection
        being down for ALL scheduler writes, matching production where both
        writes share one connection."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2066', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        def _update_task_side_effect(_task_id, metadata, **_kwargs):
            # Connection never recovers in-window for the verified stamp, so
            # the done-write is never reached — the ONLY set_task_status call
            # in this scenario is the fallback 'blocked' write below.
            return 'before_done_verified_at' not in metadata

        scheduler.update_task = AsyncMock(side_effect=_update_task_side_effect)
        # Every set_task_status call fails too — the connection is down for
        # ALL scheduler writes, not just update_task.
        scheduler.set_task_status = AsyncMock(
            side_effect=RuntimeError('transient: fused-memory unavailable')
        )
        sleeper = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            writeback_backoffs=[0.01] * 2,
            sleeper=sleeper,
        )
        outcome = await runner.run(assignment)

        # The durable, connection-independent escalation is filed regardless.
        pending = queue.get_by_task('2066')
        assert pending, (
            'Escalation must be filed even when the fallback blocked-status '
            'write also fails'
        )
        assert pending[0].category == 'infra_issue'
        assert pending[0].level == 2

        # The outcome contract holds even when NO scheduler write succeeds:
        # BLOCKED, never a propagated exception.
        assert outcome == WorkflowOutcome.BLOCKED

        # The deploy script is never re-run.
        script_runner.assert_awaited_once()

    async def test_stamp_lands_on_final_attempt_then_done_write_fails_still_escalates(
        self, tmp_path: Path,
    ):
        """Design-edge regression guard (task 2066 amendment): the
        verified-stamp write and the done-write share ONE writeback budget and
        `attempt` counter.  If the stamp only lands on the LAST attempt, the
        done-write gets exactly one remaining try — no dedicated sub-budget.
        A single failure of that one remaining try must still degrade
        gracefully to the durable-escalation path (bounded, no propagated
        exception) rather than looping further or crashing."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2066', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        # The verified stamp fails for the first 2 attempts and lands only on
        # the 3rd/last attempt (writeback_backoffs=[0.01]*2 below -> attempts=3).
        verified_attempts: list[bool] = []

        def _update_task_side_effect(_task_id, metadata, **_kwargs):
            if 'before_done_verified_at' in metadata:
                result = len(verified_attempts) >= 2
                verified_attempts.append(result)
                return result
            return True

        scheduler.update_task = AsyncMock(side_effect=_update_task_side_effect)

        # The done-write's only remaining try (on that same last iteration)
        # fails; the fallback 'blocked' write (issued after budget exhaustion)
        # stays healthy so this test isolates the shared-budget edge case from
        # the separate blocked-write-failure scenario covered above.
        def _set_task_status_side_effect(_task_id, status, **_kwargs):
            if status == 'done':
                raise RuntimeError('transient: fused-memory unavailable')
            return None

        scheduler.set_task_status = AsyncMock(side_effect=_set_task_status_side_effect)
        sleeper = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            writeback_backoffs=[0.01] * 2,
            sleeper=sleeper,
        )
        outcome = await runner.run(assignment)

        # The stamp landed only on the 3rd/last attempt ...
        assert verified_attempts == [False, False, True]

        # ... but its single remaining try at the done-write failed, so the
        # shared budget is exhausted with no attempts left: BLOCKED + durable
        # escalation, never a propagated exception.
        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('2066')
        assert pending, (
            'Budget exhaustion after a late-landing stamp must still file a '
            'durable escalation'
        )

        done_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if len(c.args) > 1 and c.args[1] == 'done'
        ]
        assert len(done_calls) == 1, (
            f'The done-write gets exactly ONE try when the stamp lands on the '
            f'final attempt (shared budget, no dedicated sub-budget); got '
            f'{len(done_calls)}'
        )
        script_runner.assert_awaited_once()

    async def test_verified_stamp_write_advances_deploy_state_phase_to_verified(
        self, tmp_path: Path,
    ):
        """ζ DS-1: the verified-stamp update_task call — the FIRST write of
        the shared retry-budget pair — also carries
        deploy_state.phase=='verified', folded into the SAME write. No third
        write is added to the pair (the retry budget stays exactly two
        writes: the verified stamp, then the done status write)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2066', target_unit='orchestrator-reify.service')
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
        verified_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_verified_at')
        ]
        assert len(verified_calls) == 1, (
            f'phase fold must not add a THIRD write to the shared retry '
            f'budget; got {len(verified_calls)} verified-stamp write(s)'
        )
        assert verified_calls[0].args[1].get('deploy_state', {}).get('phase') == 'verified', (
            'the verified-stamp write must atomically carry deploy_state.phase==verified'
        )
        # Reviewer amendment (task 2240): deploy_state.verified_at must
        # mirror the SAME timestamp as the top-level before_done_verified_at
        # stamp written in this exact call, not stay null forever.
        assert (
            verified_calls[0].args[1]['deploy_state']['verified_at']
            == verified_calls[0].args[1]['before_done_verified_at']
        )

    async def test_illegal_source_phase_falls_back_to_stamp_only_write(
        self, tmp_path: Path,
    ):
        """Reviewer amendment (task 2240, error_handling): the up-front
        _compute_deploy_phase_advance(...VERIFIED) call sits OUTSIDE the
        retry loop and is unguarded — if metadata.deploy_state were
        corrupted/unexpected (an illegal source phase for VERIFIED, e.g.
        'escalated'), it must not raise straight out of the method. The
        deploy already succeeded by this point, so the writeback must fall
        back to a stamp-only payload (no deploy_state key) and still
        converge to done, with the illegal edge still filed loudly as an L2
        escalation (file-before-raise, DS-2)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        # 'escalated' is not a legal source phase for VERIFIED (only 'ran'
        # is) — simulates corrupted/unexpected deploy_state metadata.
        task = _deploy_task(task_id='2240', target_unit='orchestrator-reify.service', phase='escalated')
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)

        outcome = await runner._writeback_deploy_success(
            '2240',
            task['metadata'],
            _FRESH_UNIT_STATE,
            'orchestrator-reify.service',
            'Cross-unit deploy of the reify worker',
        )

        assert outcome == WorkflowOutcome.DONE, (
            'the deploy already succeeded — an illegal deploy_state advance '
            'must not strand a successful deploy'
        )
        verified_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and 'before_done_verified_at' in c.args[1]
        ]
        assert len(verified_calls) == 1
        assert 'deploy_state' not in verified_calls[0].args[1], (
            'the illegal advance must fall back to a stamp-only write, not a '
            'deploy_state payload built from a state that never persisted'
        )
        # DS-2 loudness: the illegal edge still files a born-at-L2
        # escalation (file-before-raise), even though the raise itself is
        # swallowed here.
        illegal_escs = [
            e for e in queue.get_by_task('2240')
            if e.category == 'illegal_deploy_transition'
        ]
        assert len(illegal_escs) == 1
        scheduler.set_task_status.assert_awaited_once()
        assert scheduler.set_task_status.call_args.args[1] == 'done'


# ---------------------------------------------------------------------------
# Task 2706: writeback consumes the shared orchestrator.fm_retry schedule
# (RED until step-8 replaces writeback_max_attempts/writeback_backoff_base
# with the writeback_backoffs seam)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestWritebackBackoffsSeam:
    """DeterministicRunner's writeback retry loop consumes the shared
    ``orchestrator.fm_retry`` schedule (task 2706) via a new
    ``writeback_backoffs: list[float] | None`` constructor param, replacing
    the old ``writeback_max_attempts``/``writeback_backoff_base`` pair.
    ``None`` (the default) resolves to ``fm_retry_backoffs()`` once, at
    construction time.
    """

    async def test_injected_writeback_backoffs_bounds_attempts_and_sleeps(
        self, tmp_path: Path,
    ):
        """An injected fixed-length schedule bounds both the verified-stamp
        attempt count and the sleeper await count: writeback_backoffs=[0.0]*4
        -> attempts=len+1=5, sleeps=4 (new seam — fails today: the
        constructor has no writeback_backoffs kwarg)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2706', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        # The verified-stamp write never lands (persistently severed connection).
        scheduler.update_task = AsyncMock(
            side_effect=lambda _task_id, metadata, **_kwargs:
                'before_done_verified_at' not in metadata
        )
        sleeper = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            writeback_backoffs=[0.0] * 4,
            sleeper=sleeper,
        )
        outcome = await runner.run(assignment)

        verified_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and 'before_done_verified_at' in c.args[1]
        ]
        assert len(verified_calls) == 5, (
            f'writeback_backoffs=[0.0]*4 -> attempts=len+1=5; got '
            f'{len(verified_calls)}'
        )
        assert sleeper.await_count == 4, (
            f'writeback_backoffs=[0.0]*4 -> 4 between-attempt sleeps; got '
            f'{sleeper.await_count}'
        )
        assert outcome == WorkflowOutcome.BLOCKED, (
            'a verified-stamp that never lands must still exhaust the budget '
            'and block, never hang or silently succeed'
        )

    async def test_default_construction_survives_outage_via_shared_fm_retry_schedule(
        self, tmp_path: Path, monkeypatch,
    ):
        """ACCEPTANCE — the task's user-observable signal: DEFAULT
        construction (no writeback_backoffs override) must consume the
        shared orchestrator.fm_retry schedule and survive a connection
        outage shorter than the default budget, with NO spurious
        infra_issue escalation (fails today: fm_retry_backoffs is not
        imported into deterministic_runner, so it cannot be patched)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2706', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        # Connection severed for the verified-stamp write's first K=3
        # attempts, recovered on the 4th — K is less than the patched
        # schedule's attempt count (7), so a DEFAULT-constructed runner must
        # outlast it.
        verified_call_returns: list[bool] = []

        def _update_task_side_effect(_task_id, metadata, **_kwargs):
            if 'before_done_verified_at' in metadata:
                result = len(verified_call_returns) >= 3
                verified_call_returns.append(result)
                return result
            return True

        scheduler.update_task = AsyncMock(side_effect=_update_task_side_effect)

        # Patch the shared schedule (long enough to outlast the simulated
        # outage) instead of injecting writeback_backoffs — this exercises
        # the DEFAULT `None -> fm_retry_backoffs()` seam end to end.
        monkeypatch.setattr(
            'orchestrator.deterministic_runner.fm_retry_backoffs',
            lambda *a, **kw: [0.0] * 6,
        )
        sleeper = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            sleeper=sleeper,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE, (
            'a connection outage shorter than the default writeback window '
            'must converge to done, not strand the task'
        )
        scheduler.set_task_status.assert_awaited_once()
        done_call = scheduler.set_task_status.call_args
        assert done_call.args[0] == '2706'
        assert done_call.args[1] == 'done'
        assert queue.get_by_task('2706') == [], (
            'No escalation should be filed when the writeback recovers '
            'within the shared fm_retry budget'
        )

    async def test_persistent_severed_connection_still_escalates_under_new_seam(
        self, tmp_path: Path,
    ):
        """Loud path preserved: a persistent (never-recovering) connection
        loss still files the durable infra_issue L2 escalation and returns
        BLOCKED under the new writeback_backoffs seam (fails today:
        writeback_backoffs kwarg absent)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2706', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])
        script_runner = AsyncMock(return_value=(0, 'ok'))

        def _update_task_side_effect(_task_id, metadata, **_kwargs):
            # Connection never recovers in-window for the verified stamp.
            return 'before_done_verified_at' not in metadata

        scheduler.update_task = AsyncMock(side_effect=_update_task_side_effect)
        sleeper = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
            writeback_backoffs=[0.0] * 2,
            sleeper=sleeper,
        )
        outcome = await runner.run(assignment)

        pending = queue.get_by_task('2706')
        assert pending, (
            'Budget exhaustion must file a durable escalation, not leave '
            'the queue empty'
        )
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'infra_issue'
        assert 'severed' in esc.detail.lower() or 'connection' in esc.detail.lower(), (
            f'Detail must explain the self-severed-connection/writeback-'
            f'stranding cause: {esc.detail!r}'
        )
        assert outcome == WorkflowOutcome.BLOCKED


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
# Task 2090 — Layer A: whole-process-group kill on subprocess timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneSubprocessTimeoutHardening:
    """DeterministicRunner — Layer A: kill the WHOLE process tree on timeout.

    Guards against the inherited-pipe hang behind task 2087's stranding:
    restart-fused-memory.sh --drain spawns grandchildren (systemctl restart,
    curl, journalctl, sleep, the restarted daemon) that inherit the write end
    of the merged stdout pipe.  Killing only the direct child (pre-2090
    behavior) leaves the tree alive and the pipe open forever.
    """

    async def test_timeout_kills_whole_process_group(self, tmp_path: Path):
        """On timeout, a backgrounded grandchild must be killed too, not just
        the direct child.

        RED today: current code calls ``proc.kill()`` on the direct child only
        (no ``start_new_session``, no process-group kill) — the orphaned
        grandchild survives the timeout branch.
        """
        import asyncio
        import os
        import time

        from orchestrator.deterministic_runner import DeterministicRunner

        script = tmp_path / 'hang.sh'
        pidfile = tmp_path / 'grandchild.pid'
        script.write_text(
            '#!/bin/sh\n'
            'sleep 60 &\n'
            'echo $! > "$1"\n'
            'sleep 60\n'
        )
        script.chmod(0o755)

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': str(script),
            'args': [str(pidfile)],
            'cwd': str(tmp_path),
            'timeout_secs': 1,
            'target_unit': 'orchestrator-reify.service',
        }

        # Hang tripwire: if the fix regresses into a real hang, fail loudly
        # instead of stalling the suite.
        rc, tail = await asyncio.wait_for(
            runner._default_run_script(before_done), timeout=10,
        )

        assert rc == 1, f'expected rc=1 on timeout, got {rc}'
        assert 'timed out' in tail, f'expected a timed-out marker in tail, got {tail!r}'

        grandchild_pid = int(pidfile.read_text().strip())
        deadline = time.monotonic() + 3.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                alive = False
                break
            await asyncio.sleep(0.05)
        assert not alive, (
            f'grandchild pid {grandchild_pid} must be dead after timeout — the '
            f'WHOLE process group must be killed, not just the direct child'
        )

    async def test_terminate_process_tree_swallows_process_lookup_error(
        self, tmp_path: Path,
    ):
        """``_terminate_process_tree`` must swallow ProcessLookupError from
        BOTH the killpg path and the proc.kill() fallback — an already-exited
        process must never propagate out of the timeout-cleanup helper.

        RED today: ``_terminate_process_tree`` does not exist yet
        (AttributeError).
        """
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(
            scheduler=MagicMock(),
            escalation_queue=queue,
            reap_grace_secs=0.05,
        )

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.kill = MagicMock(side_effect=ProcessLookupError('already exited'))
        mock_proc.wait = AsyncMock(return_value=None)

        with (
            patch('os.getpgid', side_effect=ProcessLookupError('no such process')),
            patch('os.killpg', side_effect=ProcessLookupError('no such process')) as mock_killpg,
        ):
            await runner._terminate_process_tree(mock_proc)

        mock_killpg.assert_not_called()
        mock_proc.kill.assert_called_once()

    async def test_terminate_process_tree_bounds_reap_when_proc_never_exits(
        self, tmp_path: Path,
    ):
        """The reap following the kill signal must be bounded by
        ``reap_grace_secs`` — a process that never exits after being signaled
        (e.g. stuck in an uninterruptible D-state) must not hang
        ``_terminate_process_tree`` forever.

        The existing ProcessLookupError-swallow test's ``mock_proc.wait``
        resolves immediately, so it never exercises the
        ``except TimeoutError: logger.warning(...)`` branch around
        ``asyncio.wait_for(proc.wait(), timeout=self._reap_grace_secs)`` —
        this test exercises that bound directly.
        """
        import asyncio
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(
            scheduler=MagicMock(),
            escalation_queue=queue,
            reap_grace_secs=0.05,
        )

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        # proc.wait() never resolves — simulates an unkillable/D-state process
        # that ignores SIGKILL.
        never_resolves = asyncio.Event()
        mock_proc.wait = AsyncMock(side_effect=never_resolves.wait)

        with (
            patch('os.getpgid', return_value=12345),
            patch('os.killpg') as mock_killpg,
        ):
            # Hang tripwire: if reap_grace_secs stops bounding the wait, fail
            # loudly instead of stalling the suite.
            await asyncio.wait_for(
                runner._terminate_process_tree(mock_proc), timeout=5,
            )

        mock_killpg.assert_called_once()


# ---------------------------------------------------------------------------
# Task 2091 — bound _default_inspect_unit's systemctl communicate() call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestInspectUnitTimeoutHardening:
    """DeterministicRunner — task 2091: a wedged unit-inspect must not hang.

    Distinct from task 2090's outer run_fn guard (esc-2090-11): this covers
    the systemctl `communicate()` call inside `_default_inspect_unit`, used
    on BOTH the baseline inspect and the post-deploy verify inspect. A wedge
    on the verify leg is caught by the existing `pid > 0` freshness check;
    a wedge on the baseline leg needs its own guard (see
    ``test_wedged_baseline_inspect_blocks_without_running_deploy``) since a
    sentinel baseline would otherwise make the freshness comparison
    trivially true instead of failing closed.
    """

    async def test_wedged_inspect_returns_mainpid_zero_sentinel(self, tmp_path: Path):
        """A wedged systemctl call must time out and return the MainPID=0
        sentinel directly, killing the stuck subprocess.

        RED today: `_default_inspect_unit` awaits `proc.communicate()` with
        no timeout — this call hangs forever instead of returning.
        """
        import asyncio
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(
            scheduler=MagicMock(),
            escalation_queue=queue,
            inspect_timeout_secs=0.05,
            reap_grace_secs=0.05,
        )

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        never_resolves = asyncio.Event()
        mock_proc.communicate = AsyncMock(side_effect=never_resolves.wait)
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=None)

        with patch('asyncio.create_subprocess_exec', AsyncMock(return_value=mock_proc)):
            # Hang tripwire: if the fix regresses, fail loudly instead of
            # stalling the suite.
            result = await asyncio.wait_for(
                runner._default_inspect_unit('orchestrator-reify.service'), timeout=5,
            )

        assert result == {
            'MainPID': 0,
            'ActiveState': '',
            'ActiveEnterTimestamp': '',
            'ActiveEnterTimestampMonotonic': 0,
        }
        mock_proc.kill.assert_called_once()

    async def test_wedged_verify_inspect_drives_verify_fail_escalation(
        self, tmp_path: Path,
    ):
        """End-to-end via run(): a wedged verify-leg inspect must produce
        BLOCKED + exactly 1 pending infra_issue escalation, not a hang —
        the MainPID=0 sentinel routes through the existing fresh-PID
        verify-fail path rather than stranding the task.
        """
        import asyncio
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='500', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        baseline_proc = MagicMock()
        baseline_proc.communicate = AsyncMock(return_value=(
            b'MainPID=100\nActiveState=active\n'
            b'ActiveEnterTimestamp=Mon 2026-06-23 10:00:00 UTC\n'
            b'ActiveEnterTimestampMonotonic=1000000\n',
            b'',
        ))

        wedged_proc = MagicMock()
        wedged_proc.pid = 12345
        never_resolves = asyncio.Event()
        wedged_proc.communicate = AsyncMock(side_effect=never_resolves.wait)
        wedged_proc.kill = MagicMock()
        wedged_proc.wait = AsyncMock(return_value=None)

        procs = iter([baseline_proc, wedged_proc])

        async def _fake_create_subprocess_exec(*args, **kwargs):
            return next(procs)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            script_runner=script_runner,
            inspect_timeout_secs=0.05,
            reap_grace_secs=0.05,
        )

        with patch(
            'asyncio.create_subprocess_exec', side_effect=_fake_create_subprocess_exec,
        ):
            # Hang tripwire: if the fix regresses, fail loudly instead of
            # stalling the suite.
            outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('500', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        assert pending[0].category == 'infra_issue'
        wedged_proc.kill.assert_called_once()

    async def test_wedged_baseline_inspect_blocks_without_running_deploy(
        self, tmp_path: Path,
    ):
        """A wedged BASELINE-leg inspect must not silently make verification
        trivially pass.

        The wedged baseline returns the MainPID=0/ActiveState=''/monotonic=0
        sentinel. The second (would-be verify-leg) subprocess is wired to a
        REAL fresh post-deploy state (MainPID=200, monotonic=2_000_000) — the
        exact input that would make the OLD, unguarded freshness comparison
        (`new_monotonic > baseline_monotonic`, with baseline_monotonic=0)
        trivially true and the `pid > 0` check also pass, silently reporting
        a false-positive verified deploy. With the fix, run() must instead
        block BEFORE the deploy script (run_fn) or the verify-leg inspect are
        ever reached — so neither script_runner nor the second subprocess
        should ever be invoked.
        """
        import asyncio
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='501', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        wedged_proc = MagicMock()
        wedged_proc.pid = 12345
        never_resolves = asyncio.Event()
        wedged_proc.communicate = AsyncMock(side_effect=never_resolves.wait)
        wedged_proc.kill = MagicMock()
        wedged_proc.wait = AsyncMock(return_value=None)

        # If the (unfixed) code proceeded past the wedged baseline, this is a
        # REAL, functioning fresh post-deploy state — not a hang — so the
        # test can observe the actual (buggy) outcome rather than timing out.
        verify_proc = MagicMock()
        verify_proc.communicate = AsyncMock(return_value=(
            b'MainPID=200\nActiveState=active\n'
            b'ActiveEnterTimestamp=Mon 2026-06-23 10:01:00 UTC\n'
            b'ActiveEnterTimestampMonotonic=2000000\n',
            b'',
        ))

        procs = iter([wedged_proc, verify_proc])

        async def _fake_create_subprocess_exec(*args, **kwargs):
            return next(procs)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            script_runner=script_runner,
            inspect_timeout_secs=0.05,
            reap_grace_secs=0.05,
        )

        with patch(
            'asyncio.create_subprocess_exec', side_effect=_fake_create_subprocess_exec,
        ):
            # Hang tripwire: if the fix regresses, fail loudly instead of
            # stalling the suite.
            outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('501', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        assert pending[0].category == 'infra_issue'
        assert 'Baseline inspect failed' in pending[0].summary
        wedged_proc.kill.assert_called_once()
        # The deploy script must NOT run against an untrusted baseline, and
        # the verify-leg subprocess must never be reached either.
        script_runner.assert_not_called()
        assert next(procs, None) is verify_proc, (
            'verify_proc must be left untouched — run() must bail out before '
            'the verify-leg inspect is ever invoked'
        )

    async def test_baseline_inspect_fail_advances_deploy_state_phase_ran_to_escalated(
        self, tmp_path: Path,
    ):
        """ζ DS-1: an untrustworthy baseline (no ActiveState) is detected
        AFTER the shared before_done_ran_at/phase=ran write (that stamp
        precedes baseline capture) — the infra_issue filing then advances
        deploy_state.phase ran->escalated, gated to the deploy path only.

        Uses a direct unit_inspector mock (rather than subprocess-level
        wedging) since only the ``ActiveState`` == '' branch matters here.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='502', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        unit_inspector = AsyncMock(return_value={
            'MainPID': 0, 'ActiveState': '', 'ActiveEnterTimestamp': '',
            'ActiveEnterTimestampMonotonic': 0,
        })

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('502', status='pending')
        assert len(pending) == 1
        assert 'Baseline inspect failed' in pending[0].summary
        script_runner.assert_not_called()

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls
        assert deploy_state_calls[-1].args[1]['deploy_state']['phase'] == 'escalated'


# ---------------------------------------------------------------------------
# Task 2119 — _default_inspect_unit delegates to systemd_inspect.inspect_systemd_unit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDefaultInspectUnitDelegatesToModule:
    """DeterministicRunner._default_inspect_unit is a thin delegate (task 2119)."""

    async def test_default_inspect_unit_delegates_to_module(self, tmp_path: Path):
        """_default_inspect_unit must forward to the hoisted
        systemd_inspect.inspect_systemd_unit, passing this instance's
        injected timeout/reap-grace seams through unchanged.

        RED today: `orchestrator.deterministic_runner` has no
        `inspect_systemd_unit` name bound in its own module namespace yet
        (it only holds a `systemd_inspect` submodule reference and calls
        through it qualified) -- patching the direct name raises AttributeError.
        """
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(
            scheduler=MagicMock(),
            escalation_queue=queue,
            inspect_timeout_secs=7.0,
            reap_grace_secs=3.0,
        )

        mock_inspect = AsyncMock(return_value={'MainPID': 999, 'ActiveState': 'active'})
        with patch('orchestrator.deterministic_runner.inspect_systemd_unit', mock_inspect):
            result = await runner._default_inspect_unit('orchestrator-reify.service')

        assert result == {'MainPID': 999, 'ActiveState': 'active'}
        mock_inspect.assert_awaited_once_with(
            'orchestrator-reify.service', timeout_secs=7.0, reap_grace_secs=3.0,
        )


# ---------------------------------------------------------------------------
# Task 2090 — Layer B: outer wall-clock guard around the cross-unit run_fn call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBeforeDoneRunFnOuterGuard:
    """DeterministicRunner — Layer B: run() must never hang on a stuck run_fn.

    Layer A hardens the DEFAULT script runner's own timeout handling. Layer B
    is the unconditional backstop around the run_fn CALL SITE in run() itself:
    even if an injected/custom script_runner seam hangs forever or raises an
    unexpected exception, run() must still reach _file_infra_issue_and_block
    and return BLOCKED — it must never strand the task with an empty
    escalation queue (the exact task-2087 evidence: before_done_ran_at
    stamped, zero escalations filed, task stuck in-progress forever).
    """

    async def test_run_fn_hang_files_infra_issue_and_blocks(self, tmp_path: Path):
        """A run_fn that hangs forever must still produce a BLOCKED outcome
        and exactly one L2 infra_issue escalation (Layer B outer guard).

        RED today: run() has no outer timeout around ``await run_fn(before_done)``.
        """
        import asyncio

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='300', target_unit='orchestrator-reify.service', timeout_secs=0,
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        async def _hang(_before_done):
            await asyncio.Event().wait()

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=_hang,
            run_timeout_grace_secs=0.05,
        )

        # Hang tripwire: if the outer guard regresses, fail loudly instead of
        # stalling the suite.
        outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('300', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.category == 'infra_issue'
        assert 'orchestrator-reify.service' in esc.detail, (
            f'target_unit must appear in detail: {esc.detail!r}'
        )
        assert any(
            phrase in esc.detail.lower()
            for phrase in ('timed out', 'timeout', 'hung', 'exceeded')
        ), f'detail must mention the timeout/hang: {esc.detail!r}'

        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert stamp_calls, (
            'before_done_ran_at must already be stamped (I1) before the outer '
            'guard fires — the deploy must never be re-run on resume'
        )

    async def test_run_fn_unexpected_exception_files_infra_issue_and_blocks(
        self, tmp_path: Path,
    ):
        """A run_fn that raises an unexpected exception must NOT propagate —
        run() must route to _file_infra_issue_and_block and return BLOCKED.

        RED today: run() propagates the RuntimeError uncaught.
        """
        import asyncio

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='301', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        async def _boom(_before_done):
            raise RuntimeError('spawn exploded')

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=_boom,
        )

        outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('301', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.category == 'infra_issue'

    async def test_run_fn_seam_internal_timeout_error_not_misattributed_to_outer_guard(
        self, tmp_path: Path,
    ):
        """A run_fn that raises its OWN TimeoutError (e.g. a custom seam with
        an inner timeout shorter than the outer guard) must be routed through
        the generic unexpected-error branch, not reported as 'the outer guard
        timeout fired' — the two are different failures with different
        operator remediation (a hung/detached subprocess vs. an application
        error), and misattributing one as the other sends an operator down
        the wrong diagnostic path.
        """
        import asyncio

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='303', target_unit='orchestrator-reify.service')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        async def _seam_internal_timeout(_before_done):
            raise TimeoutError('seam raised its own timeout, unrelated to the wall-clock guard')

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=_seam_internal_timeout,
        )

        outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('303', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.summary.startswith('Deploy run_fn failed'), (
            f'a seam-internal TimeoutError must route through the generic '
            f'unexpected-error branch, not the outer-guard-timeout branch: '
            f'{esc.summary!r}'
        )
        assert not esc.summary.startswith('Deploy run_fn timed out'), (
            f'a seam-internal TimeoutError must not be misattributed as the '
            f'outer-guard timeout branch: {esc.summary!r}'
        )
        assert 'exceeded the outer guard timeout' not in esc.detail, (
            f'a seam-internal TimeoutError must not be misattributed to the '
            f'outer wall-clock guard: {esc.detail!r}'
        )

    async def test_outer_guard_does_not_fire_for_well_behaved_runner(
        self, tmp_path: Path,
    ):
        """A run_fn that takes nearly all of before_done['timeout_secs'] but
        still finishes and returns rc=0 must take the normal success/verify
        path — the Layer-B outer guard (timeout_secs + run_timeout_grace_secs)
        must not spuriously fire just because a deploy used most of its inner
        time budget.

        Regression guard for the invariant "outer bound strictly greater than
        timeout_secs": if run_timeout_grace_secs (or the outer_timeout
        computation) ever regressed towards zero, this well-behaved seam
        would be killed and misreported as a hung subprocess instead of
        completing normally.
        """
        import asyncio

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        timeout_secs = 0.3
        task = _deploy_task(
            task_id='302', target_unit='orchestrator-reify.service',
            timeout_secs=timeout_secs,
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        async def _finishes_just_under_timeout(_before_done):
            await asyncio.sleep(timeout_secs - 0.05)
            return 0, 'deploy ok'

        unit_inspector = AsyncMock(side_effect=[_BASELINE_UNIT_STATE, _FRESH_UNIT_STATE])

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=_finishes_just_under_timeout,
        )

        outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.DONE, (
            'a run_fn finishing just under timeout_secs must take the normal '
            'success/verify path, not the outer guard'
        )
        assert queue.get_by_task('302', status='pending') == [], (
            'no escalation should be filed on the success path'
        )

    async def test_run_fn_hang_advances_deploy_state_phase_ran_to_escalated(
        self, tmp_path: Path,
    ):
        """ζ DS-1: the outer-guard timeout (Layer B) advances
        deploy_state.phase ran->escalated via _file_infra_issue_and_block,
        gated to the deploy path only."""
        import asyncio

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='300', target_unit='orchestrator-reify.service', timeout_secs=0,
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        async def _hang(_before_done):
            await asyncio.Event().wait()

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=_hang,
            run_timeout_grace_secs=0.05,
        )

        outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)
        assert outcome == WorkflowOutcome.BLOCKED

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls
        assert deploy_state_calls[-1].args[1]['deploy_state']['phase'] == 'escalated'


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

    async def test_b7a_advances_deploy_state_phase_ran_to_escalated(self, tmp_path: Path):
        """ζ DS-1: rc ≠ 0 (RESTART_FAILED) advances deploy_state.phase
        ran->escalated via _file_infra_issue_and_block, gated to the deploy
        path only (before_done with target_unit)."""
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

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls
        assert deploy_state_calls[-1].args[1]['deploy_state']['phase'] == 'escalated'


# ---------------------------------------------------------------------------
# Reviewer amendment (task 2240): _file_infra_issue_and_block's best-effort
# ran->escalated phase advance can itself fail transiently (e.g. the SAME
# severed connection that triggered the escalation). Pin the documented
# consequence: no state corruption, but a resume sees phase stuck at 'ran'
# and re-escalates instead of resuming to done — one spurious extra
# escalate/resolve round-trip that self-heals once the advance lands.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFileInfraIssueEscalatedAdvanceFailureConverges:
    """A transient failure of the best-effort ESCALATED phase advance must
    not corrupt state, but does cost one extra human round-trip before the
    task converges to done — see _file_infra_issue_and_block's docstring."""

    async def test_advance_failure_then_resolve_converges_after_extra_round_trip(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2240')
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(1, 'boom: unit failed to start'))

        # Connection severed ONLY for the ESCALATED phase-advance write
        # (deploy_state.phase=='escalated'); the RAN stamp write and every
        # set_task_status call are unaffected — the escalation itself is
        # still filed durably (EscalationQueue.submit writes to local disk).
        escalated_advance_should_fail = [True]

        def _update_task_side_effect(_task_id, metadata, **_kwargs):
            ds = metadata.get('deploy_state') or {}
            if ds.get('phase') == 'escalated' and escalated_advance_should_fail[0]:
                raise RuntimeError('connection severed (simulated)')
            return True

        scheduler.update_task = AsyncMock(side_effect=_update_task_side_effect)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )

        # --- Run 1: deploy fails (rc≠0), escalation filed, but the
        # best-effort advance to ESCALATED itself fails transiently — phase
        # stays at 'ran'.
        outcome1 = await runner.run(_make_assignment(task))
        assert outcome1 == WorkflowOutcome.BLOCKED
        assert task['metadata']['deploy_state']['phase'] == 'ran', (
            'the best-effort advance failed, so phase must stay at its '
            'pre-escalation value, not silently jump to escalated'
        )
        first_pending = queue.get_by_task('2240', status='pending')
        assert len(first_pending) == 1

        # The runner mirrors deploy_state into `metadata` in place (needed
        # for same-run multi-advance reads), but a plain top-level evidence
        # stamp like before_done_ran_at is only ever sent to the (mocked)
        # scheduler — it does not round-trip back into this dict the way a
        # real get_task() would on the next dispatch. Simulate that landed
        # write explicitly, mirroring how every other resume-path test in
        # this file seeds before_done_ran_at via _deploy_task's kwarg — only
        # the ESCALATED advance failed, not this one.
        task['metadata']['before_done_ran_at'] = '2026-07-01T00:00:00+00:00'

        # A human resolves the (only) escalation — but the phase is still
        # stuck at 'ran', not 'escalated'.
        queue.resolve(first_pending[0].id, 'human restarted the unit manually')

        # --- Run 2 (resume): connection recovers. The phase==ESCALATED
        # resolution proof is NOT satisfied (still 'ran'), so this is
        # indistinguishable from an unresolved crash-window — it
        # re-escalates instead of resuming to done (the documented spurious
        # extra round-trip). This time the advance itself succeeds, landing
        # phase=='escalated'.
        escalated_advance_should_fail[0] = False
        outcome2 = await runner.run(_make_assignment(task))
        assert outcome2 == WorkflowOutcome.BLOCKED, (
            'phase never reached ESCALATED, so resume must NOT phantom-complete'
        )
        assert task['metadata']['deploy_state']['phase'] == 'escalated', (
            'the connection recovered, so this advance must land'
        )
        second_pending = queue.get_by_task('2240', status='pending')
        assert len(second_pending) == 1
        assert second_pending[0].id != first_pending[0].id, (
            'a NEW escalation must be filed — this is the spurious extra '
            'round-trip, not a re-use of the already-resolved one'
        )

        # A human resolves the second escalation.
        queue.resolve(second_pending[0].id, 'human confirmed the unit is healthy')

        # --- Run 3 (resume): phase==ESCALATED + own_escalation_resolved now
        # both hold — converges to done, no further escalation.
        outcome3 = await runner.run(_make_assignment(task))
        assert outcome3 == WorkflowOutcome.DONE, (
            'once phase==ESCALATED is actually persisted, resolution proof '
            'holds and the task converges to done'
        )
        done_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if c.args[1] == 'done'
        ]
        assert len(done_calls) == 1


# ---------------------------------------------------------------------------
# Reviewer amendment (task 2240, robustness): _file_infra_issue_and_block's
# best-effort ESCALATED advance must not attempt a pinned-illegal self-loop
# (ESCALATED->ESCALATED / DONE->ESCALATED) when the deploy is already at
# that phase — that would file a spurious born-at-L2
# illegal_deploy_transition escalation on top of the infra_issue one.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFileInfraIssueSkipsRedundantEscalatedAdvance:
    """On the rare crash-resume edge where deploy_state.phase is already
    ESCALATED (or DONE) but resolution could not be proven, execution can
    reach the unknown-crash infra_issue path again. The best-effort advance
    must skip rather than re-attempt an illegal self-loop."""

    @pytest.mark.parametrize('seeded_phase', ['escalated', 'done'])
    async def test_no_illegal_transition_escalation_when_already_at_target(
        self, tmp_path: Path, seeded_phase: str,
    ) -> None:
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='2240', phase=seeded_phase)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)

        outcome = await runner._file_infra_issue_and_block(
            '2240',
            summary='Deploy state unknown after crash: orchestrator-reify.service',
            detail='crash detail',
            metadata=task['metadata'],
        )

        assert outcome == WorkflowOutcome.BLOCKED
        # The infra_issue itself is still filed — the skip only affects the
        # redundant phase advance, not the loud signal for this crash.
        infra_escs = [e for e in queue.get_by_task('2240') if e.category == 'infra_issue']
        assert len(infra_escs) == 1
        # No spurious illegal_deploy_transition escalation was filed.
        illegal_escs = [
            e for e in queue.get_by_task('2240')
            if e.category == 'illegal_deploy_transition'
        ]
        assert illegal_escs == []
        # No redundant deploy_state write was even attempted.
        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls == []
        assert task['metadata']['deploy_state']['phase'] == seeded_phase


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

    @pytest.mark.parametrize('post_state', _STALE_POST_STATES)
    async def test_b7b_advances_deploy_state_phase_ran_to_escalated(
        self, tmp_path: Path, post_state: dict,
    ):
        """ζ DS-1: verify-fail (VERIFY_FAILED) advances deploy_state.phase
        ran->escalated via _file_infra_issue_and_block, gated to the deploy
        path only (before_done with target_unit)."""
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

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls
        assert deploy_state_calls[-1].args[1]['deploy_state']['phase'] == 'escalated'

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


@pytest.mark.asyncio
class TestCrashWindowScheduledDoubleDispatch:
    """Task 2983 fix (a): a double-dispatched, already-completed scheduled
    self-deploy must NOT trip the crash-window false positive.

    The reported incident (task 2912 / γ3 self-unit restart deploy): the
    scheduler re-selected a deterministic self-deploy off a STALE eligibility
    snapshot that carried ONLY before_done_ran_at — the before_done_scheduled_at
    stamp and the done-writeback had not yet landed when the snapshot was read.
    By execution time the first dispatch had completed (task 'done' with
    done_provenance.kind='deterministic-deploy-scheduled'), but run() holds the
    stale snapshot, so the b-self (before_done_scheduled_at) branch is skipped
    and the second run falls through to the crash-window detector, filing a
    born-at-L2 infra_issue false positive.

    Fix (a): before re-escalating, re-read the CURRENT task via
    scheduler.get_task and, if it is an already-completed scheduled self-deploy
    (_is_scheduled_self_deploy_complete), treat it as deploy-complete — return
    DONE with NO escalation and NO status write.  A fresh read that is NOT
    scheduled-complete still re-escalates exactly as today.
    """

    def _stale_snapshot(self, task_id: str = '2912') -> dict:
        """The stale eligibility snapshot: before_done_ran_at only (no
        before_done_scheduled_at, no before_done_verified_at, no done_provenance).
        """
        return _deploy_task(
            task_id=task_id,
            target_unit='orchestrator.service',
            before_done_ran_at='2026-07-23T10:00:00+00:00',
        )

    async def test_double_dispatch_via_done_provenance_returns_done_no_side_effects(
        self, tmp_path: Path,
    ):
        """Fresh get_task shows status='done' + done_provenance.kind=
        'deterministic-deploy-scheduled' → DONE, no escalation, no status write."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        snapshot = self._stale_snapshot('2912')
        assignment = _make_assignment(snapshot)
        queue = EscalationQueue(tmp_path)  # empty — no escalation ever filed
        scheduler = _mock_scheduler(snapshot)
        # Fresh read at execution time: the first dispatch already completed.
        current_task = {
            'id': '2912',
            'status': 'done',
            'metadata': {
                'before_done_ran_at': '2026-07-23T10:00:00+00:00',
                'done_provenance': {
                    'kind': 'deterministic-deploy-scheduled',
                    'unit': 'orchestrator.service',
                },
            },
        }
        scheduler.get_task = AsyncMock(return_value=current_task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        assert queue.get_by_task('2912') == [], (
            'a double-dispatched completed scheduled self-deploy must NOT re-escalate'
        )
        scheduler.set_task_status.assert_not_awaited()

    async def test_double_dispatch_via_scheduled_stamp_returns_done_no_side_effects(
        self, tmp_path: Path,
    ):
        """Fresh get_task carries before_done_scheduled_at → DONE, no escalation,
        no status write (recognized via the stamp even without done_provenance)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        snapshot = self._stale_snapshot('2912')
        assignment = _make_assignment(snapshot)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(snapshot)
        # Fresh read carries the scheduled stamp the snapshot lacked.
        current_task = _deploy_task(
            task_id='2912',
            target_unit='orchestrator.service',
            before_done_ran_at='2026-07-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-07-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-2912.service',
                'fire_delay_secs': 60,
            },
        )
        scheduler.get_task = AsyncMock(return_value=current_task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        assert queue.get_by_task('2912') == []
        scheduler.set_task_status.assert_not_awaited()

    async def test_genuine_crash_window_still_reescalates_and_blocks(
        self, tmp_path: Path,
    ):
        """Negative / no-over-match: fresh get_task is NOT scheduled-complete
        (the stale snapshot itself) → the genuine crash-window still re-escalates
        exactly once and blocks (existing behavior preserved)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        snapshot = self._stale_snapshot('2912')
        assignment = _make_assignment(snapshot)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(snapshot)
        # Fresh read shows NO scheduled shape (still just the stale snapshot):
        # a genuine crash mid-deploy before any terminal decision.
        scheduler.get_task = AsyncMock(return_value=snapshot)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('2912', status='pending')
        assert len(pending) == 1, (
            f'genuine crash-window must re-escalate exactly once, got {len(pending)}'
        )
        assert pending[0].category == 'infra_issue'
        blocked_calls = [
            c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'blocked'
        ]
        assert blocked_calls, 'genuine crash-window must leave the task blocked'

    async def test_stale_redispatch_always_escalates_true_refiles_gate_not_done(
        self, tmp_path: Path,
    ):
        """Amendment (reviewer_comprehensive): a STALE re-dispatch of an
        always_escalates=True scheduled self-deploy must NOT be short-circuited
        to DONE by the fresh-read backstop.

        The before_done_scheduled_at stamp is written on BOTH the
        always_escalates=False path (which sets the task 'done') AND the
        act-then-ask always_escalates=True path (b-self, which re-files the
        milestone gate and BLOCKS — the gate must not be bypassed).  When the
        in-hand snapshot lacks the stamp (so b-self is skipped) but the fresh
        get_task now carries before_done_scheduled_at, the DONE short-circuit
        must apply ONLY to always_escalates=False; the act-then-ask gate must be
        re-filed (mirroring b-self), never silently bypassed with a done write.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        # Stale snapshot: before_done_ran_at only, always_escalates=True.
        snapshot = self._stale_snapshot('2913')
        snapshot['metadata']['always_escalates'] = True
        assignment = _make_assignment(snapshot)
        queue = EscalationQueue(tmp_path)  # empty — gate not yet re-observed
        scheduler = _mock_scheduler(snapshot)
        # Fresh read carries the scheduled stamp the snapshot lacked, still
        # always_escalates=True (act-then-ask; the gate is NOT resolved).
        current_task = _deploy_task(
            task_id='2913',
            target_unit='orchestrator.service',
            before_done_ran_at='2026-07-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-07-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-2913.service',
                'fire_delay_secs': 60,
            },
        )
        current_task['metadata']['always_escalates'] = True
        scheduler.get_task = AsyncMock(return_value=current_task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            restart_scheduler=AsyncMock(return_value=(0, 'scheduled')),
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED, (
            'a stale re-dispatch of an always_escalates=True scheduled self-deploy '
            'must re-file the gate and BLOCK, not return DONE'
        )
        pending = queue.get_by_task('2913', status='pending')
        assert len(pending) == 1, (
            f'the act-then-ask milestone gate must be re-filed exactly once, '
            f'got {len(pending)}'
        )
        assert pending[0].category == 'milestone_gate', (
            f'the gate re-file must be a milestone_gate, not {pending[0].category!r}'
        )
        done_calls = [
            c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done'
        ]
        assert not done_calls, (
            'the still-open act-then-ask gate must never be bypassed with a done write'
        )


@pytest.mark.asyncio
class TestCrashWindowReverify:
    """Task 2618: before re-firing 'Deploy state unknown after crash', re-run
    the read-only verify inspect against a PERSISTED verify_baseline and reuse
    the recon-sweep's own health classifier (_deterministic_deploy_health_verdict)
    — a deploy that actually succeeded but crashed inside
    _writeback_deploy_success before before_done_verified_at could be stamped
    (task 2584's shape) must recover to done instead of re-escalating forever
    on every orchestrator restart.
    """

    async def test_crash_window_reverify_persistent_baseline_drives_to_done(
        self, tmp_path: Path,
    ):
        """A persistent-PID baseline + a fresher live inspect => healthy => done."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='900',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            phase='ran',
            verify_baseline={'main_pid': 100, 'active_enter_timestamp_monotonic': 1_000_000},
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty — no prior escalation
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_FRESH_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(done_calls) == 1, 'a healthy re-verify must drive to done exactly once'
        provenance = done_calls[0].kwargs.get('done_provenance', {})
        assert provenance.get('kind') == 'deterministic-deploy'
        assert not queue.get_by_task('900', status='pending'), (
            'a healthy re-verify must NOT file a crash-window infra_issue escalation'
        )
        script_runner.assert_not_awaited()

    async def test_crash_window_reverify_empty_baseline_timer_drives_to_done(
        self, tmp_path: Path,
    ):
        """Task 2584's exact shape: an empty (.timer) baseline + a settled,
        advanced live inspect => healthy via _empty_baseline_fresh => done."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='901',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            phase='ran',
            verify_baseline={'main_pid': 0, 'active_enter_timestamp_monotonic': 1_000_000},
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        fresh_timer_state = {
            'MainPID': 0,
            'ActiveState': 'active',
            'ActiveEnterTimestamp': 'Mon 2026-06-23 10:01:00 UTC',
            'ActiveEnterTimestampMonotonic': 2_000_000,
        }
        unit_inspector = AsyncMock(return_value=fresh_timer_state)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(done_calls) == 1, (
            'a healthy empty-baseline re-verify must drive to done exactly once'
        )
        provenance = done_calls[0].kwargs.get('done_provenance', {})
        assert provenance.get('kind') == 'deterministic-deploy'
        assert not queue.get_by_task('901', status='pending'), (
            'a healthy re-verify must NOT file a crash-window infra_issue escalation'
        )
        script_runner.assert_not_awaited()

    async def test_reverify_unconfirmed_still_escalates_with_enriched_detail(
        self, tmp_path: Path,
    ):
        """A persisted baseline whose live monotonic has NOT advanced => the
        re-verify comes back 'unconfirmed' => still escalates (never
        phantom-done), but the detail records that a re-verify was attempted
        and what it observed."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='902',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            phase='ran',
            verify_baseline={'main_pid': 100, 'active_enter_timestamp_monotonic': 1_000_000},
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty — no prior escalation
        scheduler = _mock_scheduler(task)

        # Same MainPID, same (unadvanced) monotonic as the baseline => 'unconfirmed'.
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
        pending = queue.get_by_task('902', status='pending')
        assert len(pending) == 1, (
            f'an unconfirmed re-verify must still escalate exactly once, got {len(pending)}'
        )
        assert pending[0].category == 'infra_issue'
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert not done_calls, 'an unconfirmed re-verify must NOT drive to done'
        script_runner.assert_not_awaited()
        detail = pending[0].detail
        assert 'unconfirmed' in detail, (
            f'detail must record that the re-verify came back unconfirmed: {detail!r}'
        )
        assert 'MainPID' in detail, (
            f'detail must record the re-inspected live unit state: {detail!r}'
        )

    async def test_no_baseline_crash_window_does_not_reinspect(self, tmp_path: Path):
        """Pre-ζ shape (no deploy_state at all): the persisted-baseline gate
        must not re-inspect the unit and must fall straight through to the
        existing generic crash-window escalation, unchanged."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='903', before_done_ran_at='2026-06-23T10:00:00+00:00')
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
        unit_inspector.assert_not_awaited()
        pending = queue.get_by_task('903', status='pending')
        assert len(pending) == 1, (
            f'no-baseline crash-window must escalate exactly once, got {len(pending)}'
        )
        assert pending[0].category == 'infra_issue'
        assert pending[0].summary == 'Deploy state unknown after crash: orchestrator-reify.service'
        script_runner.assert_not_awaited()

    async def test_ran_phase_no_baseline_crash_window_does_not_reinspect(
        self, tmp_path: Path,
    ):
        """Test-coverage gap (task 2618 review): the re-verify gate has THREE
        conjuncts — deploy_state is not None, phase == RAN, AND
        verify_baseline is not None. test_no_baseline_crash_window_does_not_
        reinspect above exercises the first conjunct (no deploy_state at
        all, pre-ζ shape); this exercises the third conjunct in isolation: a
        genuine ζ RAN-phase strand (deploy_state IS present, phase IS 'ran')
        that never persisted a verify_baseline (the DS-3 enrichment is
        best-effort and can be skipped, e.g. the orchestrator crashed before
        the pre-deploy baseline inspect completed). Must fall through to the
        generic crash-window escalation unchanged, with no re-inspect and no
        phantom-done — pins the `verify_baseline is not None` short-circuit
        so a regression that dropped it (reverting to the classifier's
        near-constant-'healthy' no-baseline liveness fallback for an
        always-on unit — the exact scenario finding-4.0's D3 guard exists
        for) would be caught here too."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='905',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            phase='ran',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        # A live inspect that WOULD classify 'healthy' if the re-verify gate
        # were wrongly triggered — proves the assertions below are non-vacuous.
        unit_inspector = AsyncMock(return_value=_FRESH_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        unit_inspector.assert_not_awaited()
        pending = queue.get_by_task('905', status='pending')
        assert len(pending) == 1, (
            f'no-baseline RAN-phase crash-window must escalate exactly once, got {len(pending)}'
        )
        assert pending[0].category == 'infra_issue'
        assert pending[0].summary == 'Deploy state unknown after crash: orchestrator-reify.service'
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert not done_calls, 'a no-baseline RAN-phase strand must never phantom-done'
        script_runner.assert_not_awaited()

    async def test_escalated_phase_with_deleted_escalations_does_not_reinspect(
        self, tmp_path: Path,
    ):
        """Defensive-correctness (task 2618 review): the re-verify gate must be
        scoped to a genuine RAN-phase crash-window strand, not merely a
        persisted verify_baseline. If deploy_state.phase == 'escalated' but
        every escalation record for this task+role has been deleted
        (own_escalation_resolved False => resolution_proven False), control
        still reaches sub-case (c) — this must NOT be treated as a RAN-phase
        strand even though a verify_baseline is persisted, since a live unit
        that looks fresh could be masking a real prior failure. Must fall
        through to the generic crash-window escalation unchanged, with no
        re-inspect and no phantom-done."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='904',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            phase='escalated',
            verify_baseline={'main_pid': 100, 'active_enter_timestamp_monotonic': 1_000_000},
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty — no escalation record survives
        scheduler = _mock_scheduler(task)

        # A live inspect that WOULD classify 'healthy' if the re-verify gate
        # were wrongly triggered — proves the assertions below are non-vacuous.
        unit_inspector = AsyncMock(return_value=_FRESH_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        unit_inspector.assert_not_awaited()
        pending = queue.get_by_task('904', status='pending')
        assert len(pending) == 1, (
            f'escalated-phase crash-window must escalate exactly once, got {len(pending)}'
        )
        assert pending[0].category == 'infra_issue'
        assert pending[0].summary == 'Deploy state unknown after crash: orchestrator-reify.service'
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert not done_calls, 'a non-RAN-phase strand must never phantom-done'
        script_runner.assert_not_awaited()


# ---------------------------------------------------------------------------
# Task 2120: escalation-aliasing phantom-done guard.
#
# The runner proves "a human resolved the deploy/gate" via task-scoped
# escalation existence/absence. Without an agent_role filter, ANY escalation
# sharing the task_id — e.g. one filed by an unrelated starvation-watchdog —
# aliases as the runner's own dedup/quiescence/resolution-proof signal.
#
# These tests fail if run()'s section-1 gate quiescence branch, its
# before_done quiescence branch, or its ever_escalated resolution-proof check
# regress to scanning get_by_task() without scoping to
# DETERMINISTIC_AGENT_ROLE. The parity tests characterize runner-owned
# signals and must stay GREEN regardless of that scoping.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDeterministicRunnerResolutionProofAliasing:
    """An unrelated escalation sharing a task_id must never alias as this
    runner's own resolution-proof/quiescence signal."""

    _UNRELATED_ROLE = 'orchestrator-starvation-watchdog'

    # --- test a: run()'s ever_escalated resolution-proof aliasing -----------

    async def test_ever_escalated_ignores_unrelated_resolved_escalation(self, tmp_path: Path):
        """A RESOLVED unrelated (starvation-watchdog) escalation must NOT count
        as this runner's own resolution proof.

        Bug trace: pre-fix, `ever_escalated=bool(get_by_task(task_id))` finds
        the unrelated resolved escalation and takes branch (b) — phantom-done
        ('resumed after human resolution'). Post-fix, the role mismatch makes
        ever_escalated False, falling to branch (c): re-escalate (its own
        infra_issue) and BLOCK — never phantom-done, never re-run (I1).
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='800', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '800', self._UNRELATED_ROLE, resolved=True)
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

        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert not done_calls, (
            'An unrelated resolved escalation must NEVER phantom-complete the task'
        )
        pending = queue.get_by_task('800', status='pending')
        assert len(pending) == 1, (
            f'Runner must file its OWN infra_issue escalation (branch c), got {len(pending)}'
        )
        assert pending[0].category == 'infra_issue'
        assert pending[0].agent_role == 'orchestrator-deterministic'
        assert outcome == WorkflowOutcome.BLOCKED

    # --- test a2: ζ D3 — phase gate replaces bare escalation-existence ------

    async def test_phase_ran_with_resolved_runner_owned_escalation_does_not_resume_to_done(
        self, tmp_path: Path,
    ):
        """D3 negative (finding 4.0, the driving fix): a deploy stranded at
        deploy_state.phase=='ran' (NOT escalated) with a RESOLVED
        runner-owned escalation on record must NOT resume to done — bare
        escalation existence is not proof THIS deploy's own gate/failure is
        what got resolved; only phase=='escalated' (the runner's OWN
        recorded transition when it filed that escalation) is. Falls to
        the crash-window branch instead: re-escalates its own infra_issue
        and BLOCKS, exactly as if no escalation existed at all.

        RED against current code: `ever_escalated=bool(get_by_task(...,
        agent_role=...))` is True here (a resolved runner-owned record
        exists) and phantom-completes via branch (b) regardless of phase —
        this is finding 4.0.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='805', before_done_ran_at='2026-06-23T10:00:00+00:00',
            phase='ran',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '805', 'orchestrator-deterministic', resolved=True)
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

        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert not done_calls, (
            'phase==ran (not escalated) must NEVER resume to done, even with '
            'a resolved runner-owned escalation on record — bare escalation '
            'existence is not proof of resolution (finding 4.0)'
        )
        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('805', status='pending')
        assert len(pending) == 1, f'must file a fresh crash-window infra_issue, got {len(pending)}'
        assert pending[0].category == 'infra_issue'
        script_runner.assert_not_awaited()

    async def test_phase_ran_with_resolved_unrelated_escalation_does_not_resume_to_done(
        self, tmp_path: Path,
    ):
        """D3 negative: same as above but the resolved escalation is the
        UNRELATED starvation-watchdog role — doubly not proof of THIS
        runner's resolution. Must also re-escalate and BLOCK, never done."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='806', before_done_ran_at='2026-06-23T10:00:00+00:00',
            phase='ran',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '806', self._UNRELATED_ROLE, resolved=True)
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

        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert not done_calls
        assert outcome == WorkflowOutcome.BLOCKED
        pending = queue.get_by_task('806', status='pending')
        assert len(pending) == 1
        assert pending[0].category == 'infra_issue'
        assert pending[0].agent_role == 'orchestrator-deterministic'

    # --- run()'s section-1 gate quiescence aliasing -------------------------

    async def test_section1_quiescence_ignores_unrelated_pending_escalation(self, tmp_path: Path):
        """gate_escalated_at set; the runner's OWN gate is resolved but an
        UNRELATED escalation for the same task is still pending — must drive
        to done (its own gate is resolved), not alias on the unrelated one.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _gate_task(task_id='801', gate_escalated_at='2026-06-23T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        # Runner's own gate escalation: resolved.
        _seed_escalation(queue, '801', 'orchestrator-deterministic', resolved=True, category='milestone_gate')
        # Unrelated escalation for the SAME task: still pending.
        _seed_escalation(queue, '801', self._UNRELATED_ROLE)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE, (
            "An unrelated pending escalation must not alias as this runner's "
            'own open gate — the gate itself is resolved, so resume must proceed'
        )
        # Pure-gate leg stamps deterministic-gate provenance (task 2331) so the
        # done write passes require_done_provenance instead of churning forever.
        scheduler.set_task_status.assert_awaited_once_with(
            '801',
            'done',
            done_provenance={
                'kind': 'deterministic-gate',
                'note': 'pure gate resolved',
            },
        )

    # --- run()'s before_done quiescence aliasing -----------------------------

    async def test_before_done_quiescence_ignores_unrelated_pending_escalation(self, tmp_path: Path):
        """before_done_ran_at + before_done_verified_at set; an UNRELATED
        escalation is pending — must drive to done via the verified-proof
        branch (a), not the aliased quiescent BLOCKED.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='802',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_verified_at='2026-06-23T10:01:00+00:00',
            before_done_verified_pid=200,
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '802', self._UNRELATED_ROLE)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE, (
            "An unrelated pending escalation must not alias as this runner's "
            'own quiescence signal when the deploy is already verified'
        )
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(done_calls) == 1
        provenance = done_calls[0].kwargs.get('done_provenance', {})
        assert provenance.get('note') == 'resumed after verified deploy (crash before done write)'

    # --- parity: runner-owned signals must keep working ---------------------

    async def test_parity_runner_owned_resolved_escalation_still_resumes(self, tmp_path: Path):
        """Parity: a RESOLVED escalation filed by the runner itself must still
        prove resolution (branch b) — ζ D3: the new contract requires
        deploy_state.phase=='escalated' as the recorded proof the runner
        itself transitioned there when it filed the failure/gate escalation
        (see TestDeterministicRunnerResolutionProofAliasing's phase-gate
        tests for the finding-4.0 fix this narrows against).
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='803', before_done_ran_at='2026-06-23T10:00:00+00:00',
            phase='escalated',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '803', 'orchestrator-deterministic', resolved=True)
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
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert len(done_calls) == 1
        assert done_calls[0].kwargs.get('done_provenance', {}).get('note') == 'resumed after human resolution'

    async def test_parity_runner_owned_pending_escalation_still_quiescent(self, tmp_path: Path):
        """Parity: a PENDING escalation filed by the runner itself must still
        quiesce (BLOCKED, no re-escalation). Must stay GREEN regardless of
        the section-1 gate / before_done quiescence agent_role scoping.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='804', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '804', 'orchestrator-deterministic')
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
        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert not done_calls
        pending = queue.get_by_task('804', status='pending')
        assert len(pending) == 1, 'must not file a second escalation'


# ---------------------------------------------------------------------------
# Task 2120: escalation-aliasing suppress-direction guard.
#
# The dedup guards in _file_infra_issue_and_block and
# _file_milestone_gate_and_block must scope their "already pending?" check
# to the runner's own DETERMINISTIC_AGENT_ROLE — otherwise an unrelated
# PENDING escalation (e.g. a starvation-watchdog filing) falsely suppresses
# the runner's own gate/infra filing, silently swallowing a required L2.
# Two pending escalations may legitimately coexist for one task.
#
# These tests fail if either dedup guard's agent_role scoping regresses.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDeterministicRunnerDedupGuardScoping:
    """An unrelated pending escalation must never suppress the runner's own
    gate/infra filing."""

    _UNRELATED_ROLE = 'orchestrator-starvation-watchdog'

    # --- test b: _file_milestone_gate_and_block dedup-guard aliasing -------

    async def test_gate_filing_still_files_despite_unrelated_pending_escalation(self, tmp_path: Path):
        """Pure-gate task, no gate_escalated_at yet; an UNRELATED escalation is
        already pending — the runner must STILL file its own milestone_gate
        (two pending escalations legitimately coexist).
        """
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='900')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '900', self._UNRELATED_ROLE)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        pending = queue.get_by_task('900', status='pending')
        assert len(pending) == 2, (
            f"Runner's own gate filing must not be suppressed by an unrelated "
            f'pending escalation; expected 2 pending, got {len(pending)}'
        )
        own = [e for e in pending if e.agent_role == 'orchestrator-deterministic']
        assert len(own) == 1
        assert own[0].category == 'milestone_gate'

    # --- _file_infra_issue_and_block dedup-guard aliasing -------------------

    async def test_infra_filing_still_files_despite_unrelated_pending_escalation(self, tmp_path: Path):
        """before_done_ran_at set, crash window (no verify, no runner-owned
        resolution); an UNRELATED escalation is already pending — the runner
        must STILL re-escalate its own infra_issue (two pending coexist).
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(task_id='901', before_done_ran_at='2026-06-23T10:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '901', self._UNRELATED_ROLE)
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

        pending = queue.get_by_task('901', status='pending')
        assert len(pending) == 2, (
            f"Runner's own infra filing must not be suppressed by an unrelated "
            f'pending escalation; expected 2 pending, got {len(pending)}'
        )
        own = [e for e in pending if e.agent_role == 'orchestrator-deterministic']
        assert len(own) == 1
        assert own[0].category == 'infra_issue'
        assert outcome == WorkflowOutcome.BLOCKED

    # --- parity: runner-owned pending escalation still dedups ---------------

    async def test_parity_runner_owned_pending_still_suppresses_second_gate_filing(self, tmp_path: Path):
        """Parity: a runner-owned PENDING escalation (e.g. from a prior
        crash-safe re-dispatch before gate_escalated_at was stamped) must
        still suppress a second filing — dedup fires for a matching role.
        Must stay GREEN regardless of the dedup guard's agent_role scoping.
        """
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='902')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        _seed_escalation(queue, '902', 'orchestrator-deterministic', category='milestone_gate')
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        pending = queue.get_by_task('902', status='pending')
        assert len(pending) == 1, (
            f'A runner-owned pending escalation must dedup a second filing; '
            f'got {len(pending)} pending'
        )


# ---------------------------------------------------------------------------
# Scheduled-marker resume (amend: Suggestion 2): crash between
# before_done_scheduled_at stamp and the done write.
# before_done_scheduled_at set → transient unit registered → drive to done
# with scheduled provenance instead of re-escalating as a crash window.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSelfRestartScheduledCrashResume:
    """before_done_ran_at + before_done_scheduled_at set, empty queue → done=scheduled.

    Simulates a crash after the transient unit was registered (before_done_scheduled_at
    stamped) but before set_task_status('done') completed.  The resume path must
    drive to done with kind='deterministic-deploy-scheduled' rather than re-escalating
    as a generic crash window (Suggestion 2).
    """

    async def test_scheduled_resume_drives_to_done(self, tmp_path: Path):
        """Scheduled stamp present → done with scheduled provenance, no escalation."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='853',
            target_unit='orchestrator-reify.service',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-06-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-853.service',
                'fire_delay_secs': 60,
            },
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # truly empty — no prior escalation
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE

    async def test_scheduled_resume_sets_done_with_scheduled_kind(self, tmp_path: Path):
        """Resume with before_done_scheduled_at → done_provenance.kind='deterministic-deploy-scheduled'."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(
            task_id='853',
            target_unit='orchestrator-reify.service',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-06-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-853.service',
                'fire_delay_secs': 60,
            },
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
        )
        await runner.run(assignment)

        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        assert done_calls, 'set_task_status must be called with done on scheduled resume'
        provenance = done_calls[0].kwargs.get('done_provenance', {})
        assert provenance.get('kind') == 'deterministic-deploy-scheduled', (
            f"done_provenance.kind must be 'deterministic-deploy-scheduled'; "
            f"got {provenance.get('kind')!r}"
        )

    async def test_scheduled_resume_no_escalation_filed(self, tmp_path: Path):
        """Scheduled resume must NOT re-escalate as a crash window."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(
            task_id='853',
            target_unit='orchestrator-reify.service',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-06-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-853.service',
                'fire_delay_secs': 60,
            },
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
        )
        await runner.run(assignment)

        all_escs = queue.get_by_task('853')
        assert all_escs == [], (
            f'scheduled resume must NOT re-escalate as crash-window; got {all_escs}'
        )

    async def test_scheduled_resume_does_not_rerun_deploy(self, tmp_path: Path):
        """Scheduled resume must NOT re-invoke script_runner (I1 once-only)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(
            task_id='853',
            target_unit='orchestrator-reify.service',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-06-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-853.service',
                'fire_delay_secs': 60,
            },
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=script_runner,
        )
        await runner.run(assignment)

        script_runner.assert_not_awaited()

    async def test_scheduled_resume_provenance_carries_transient_unit(self, tmp_path: Path):
        """Resume provenance must carry the transient_unit from the stamp."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(
            task_id='854',
            target_unit='orchestrator-reify.service',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-06-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-854.service',
                'fire_delay_secs': 45,
            },
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
        )
        await runner.run(assignment)

        done_calls = [c for c in scheduler.set_task_status.call_args_list if c.args[1] == 'done']
        provenance = done_calls[0].kwargs.get('done_provenance', {})
        assert provenance.get('transient_unit') == 'orch-redeploy-restart-854.service', (
            f"provenance.transient_unit must match stamp; "
            f"got {provenance.get('transient_unit')!r}"
        )
        assert provenance.get('fire_delay_secs') == 45, (
            f"provenance.fire_delay_secs must match stamp; "
            f"got {provenance.get('fire_delay_secs')!r}"
        )


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

    async def test_b8_on_active_secs_clamped_to_minimum(self, tmp_path: Path):
        """on_active_delay_secs=0 in before_done is clamped to >=5 (amend: Suggestion 1).

        A zero or negative delay would produce --on-active=0, causing the transient
        unit to fire immediately — re-introducing the self-kill window.  The runner
        must enforce a floor of 5 seconds.
        """
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='851', target_unit='orchestrator-reify.service')
        # Inject zero delay into before_done to trigger the clamp
        task['metadata']['before_done']['on_active_delay_secs'] = 0
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
        on_active = restart_scheduler.call_args.kwargs.get('on_active_secs', 0)
        assert on_active >= 5, (
            f'on_active_secs must be clamped to >=5 when on_active_delay_secs=0; '
            f'got {on_active!r}'
        )

    async def test_b8_stamps_before_done_scheduled_at(self, tmp_path: Path):
        """update_task must stamp before_done_scheduled_at with transient_unit after scheduling (amend: Suggestion 2)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='852', target_unit='orchestrator-reify.service')
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

        scheduled_stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_scheduled_at')
        ]
        assert scheduled_stamp_calls, (
            'update_task must be called with a truthy before_done_scheduled_at stamp'
        )
        stamp_val = scheduled_stamp_calls[0].args[1]['before_done_scheduled_at']
        assert isinstance(stamp_val, dict), (
            f'before_done_scheduled_at must be a dict carrying transient_unit; got {stamp_val!r}'
        )
        assert '852' in stamp_val.get('transient_unit', ''), (
            f"before_done_scheduled_at.transient_unit must contain task id '852'; "
            f"got {stamp_val.get('transient_unit')!r}"
        )
        assert isinstance(stamp_val.get('fire_delay_secs'), int), (
            f'before_done_scheduled_at.fire_delay_secs must be an int; '
            f"got {stamp_val.get('fire_delay_secs')!r}"
        )

    async def test_b8_stamps_deploy_state_phase_scheduled_atomically(self, tmp_path: Path):
        """ζ DS-1: the before_done_scheduled_at write also atomically advances
        deploy_state.phase ran->scheduled (self-restart, always_escalates=False)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='852', target_unit='orchestrator-reify.service')
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

        scheduled_stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_scheduled_at')
        ]
        assert scheduled_stamp_calls
        assert scheduled_stamp_calls[0].args[1]['deploy_state']['phase'] == 'scheduled'

    async def test_b8_done_write_does_not_advance_deploy_state_past_scheduled(self, tmp_path: Path):
        """scheduled->done is pinned-illegal: the done write must not attempt a
        deploy_state phase advance — the LAST deploy_state write observed stays
        at phase=='scheduled' (task-status done != deploy-phase done)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(task_id='852', target_unit='orchestrator-reify.service')
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

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls
        assert deploy_state_calls[-1].args[1]['deploy_state']['phase'] == 'scheduled', (
            'no update_task call may advance deploy_state past scheduled on this path'
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

    async def test_argv_contains_detail_with_transient_unit(self, tmp_path: Path):
        """--detail carrying transient unit name must appear in the OnFailure escalation argv (amend: Suggestion 3).

        The in-process scheduling-failure path builds a rich detail block; the fire-time
        escalation-submit must also carry diagnostic context so a human handling a real
        fire-time restart failure has the transient unit name for journald lookup.
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

        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        assert '--detail' in all_argv, (
            f'--detail must appear in the OnFailure escalation argv: {all_argv!r}'
        )
        assert 'orch-redeploy-restart-900.service' in all_argv, (
            f'transient unit name must appear in --detail context: {all_argv!r}'
        )


# ---------------------------------------------------------------------------
# Task 2105: _default_schedule_detached_restart must consume before_done.cwd
# (systemd-run --working-directory + absolutized payload script) so relative
# deploy scripts stop failing 127 when the transient unit fires from the
# systemd user-manager's default cwd instead of the project root.
# (RED until step-2 adds --working-directory / step-4 absolutizes the payload)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDetachedRestartWorkingDirectory:
    """_default_schedule_detached_restart must consume before_done.cwd so the
    transient unit runs (and locates relative scripts) from the right directory."""

    def _make_mock_proc(self, returncode: int = 0) -> object:
        """Return a mock proc with communicate() → (b'', b'') and returncode."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))
        mock_proc.returncode = returncode
        return mock_proc

    async def test_relative_script_explicit_cwd_sets_working_directory(self, tmp_path: Path):
        """An explicit before_done['cwd'] must become a --working-directory=<cwd> option.

        Without this, the transient unit fires from the systemd user-manager's
        default cwd (~$HOME), where a repo-relative deploy script does not exist
        (exit 127 — the esc-2104-1 fire-time failure).
        """
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': 'scripts/restart-all-orchestrators.sh',
            'args': [],
            'cwd': '/home/leo/src/dark-factory',
            'target_unit': 'orchestrator-dark-factory.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-2105.service',
                on_active_secs=60,
                task_id='2105',
            )

        argv = mock_exec.call_args_list[0].args
        all_argv = ' '.join(str(a) for a in argv)
        assert '--working-directory=/home/leo/src/dark-factory' in all_argv, (
            f'--working-directory must be set from before_done["cwd"]: {all_argv!r}'
        )
        assert argv[-3] == '/bin/sh' and argv[-2] == '-c', (
            f'the /bin/sh -c wrapper tail must be preserved: {argv!r}'
        )

    async def test_relative_script_absent_cwd_falls_back_to_getcwd(self, tmp_path: Path):
        """With no before_done['cwd'], --working-directory must fall back to os.getcwd().

        The orchestrator's own systemd unit pins WorkingDirectory=project_root, so
        os.getcwd() at scheduling time is the correct fallback root.
        """
        import os
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': 'scripts/restart-all-orchestrators.sh',
            'args': [],
            'target_unit': 'orchestrator-dark-factory.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-2106.service',
                on_active_secs=60,
                task_id='2106',
            )

        all_argv = ' '.join(
            str(a) for call in mock_exec.call_args_list for a in call.args
        )
        assert f'--working-directory={os.getcwd()}' in all_argv, (
            f'--working-directory must fall back to os.getcwd(): {all_argv!r}'
        )

    async def test_relative_script_absolutized_in_payload(self, tmp_path: Path):
        """A relative script must be absolutized against cwd in the /bin/sh -c payload.

        --working-directory alone is not enough to trust for the leading command
        of the payload: absolutizing the script too is defense-in-depth so it is
        still found even if --working-directory were ever ignored.
        """
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': 'scripts/restart-all-orchestrators.sh',
            'args': [],
            'cwd': '/home/leo/src/dark-factory',
            'target_unit': 'orchestrator-dark-factory.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-2107.service',
                on_active_secs=60,
                task_id='2107',
            )

        argv = mock_exec.call_args_list[0].args
        assert argv[-3] == '/bin/sh' and argv[-2] == '-c', (
            f'expected a /bin/sh -c wrapper payload, got {argv!r}'
        )
        wrapped = argv[-1]
        expected_abs = str(Path('/home/leo/src/dark-factory') / 'scripts/restart-all-orchestrators.sh')
        assert expected_abs in wrapped, (
            f'payload must embed the absolutized script path {expected_abs!r}: {wrapped!r}'
        )
        assert not wrapped.startswith('scripts/restart-all-orchestrators.sh'), (
            f'payload must not lead with the bare relative script as the command: {wrapped!r}'
        )

    async def test_absolute_script_left_unchanged_in_payload(self, tmp_path: Path):
        """An already-absolute script must be embedded unchanged (no double-join under cwd)."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'cwd': '/home/leo/src/dark-factory',
            'target_unit': 'orchestrator-reify.service',
        }
        mock_proc = self._make_mock_proc()

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-2108.service',
                on_active_secs=60,
                task_id='2108',
            )

        argv = mock_exec.call_args_list[0].args
        wrapped = argv[-1]
        assert wrapped.startswith('/usr/local/bin/restart-deploy.sh'), (
            f'an absolute script must be embedded unchanged, leading the payload: {wrapped!r}'
        )
        assert '/home/leo/src/dark-factory/usr/local/bin/restart-deploy.sh' not in wrapped, (
            f'absolute script must not be double-joined under cwd: {wrapped!r}'
        )


# ---------------------------------------------------------------------------
# Task 2238 (W10-δ), step-1: _default_schedule_detached_restart delegates to
# proc_supervision.RestartPlan.execute() instead of building systemd-run argv
# inline (mirrors task 2237/γ's service_restart.py conversion).
# (RED until step-2 rewrites _default_schedule_detached_restart's body)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDetachedRestartDelegatesToRestartPlan:
    """_default_schedule_detached_restart delegates to
    proc_supervision.RestartPlan.execute() (task 2238/δ)."""

    async def test_execute_awaited_once_with_detached_self_target_plan(self, tmp_path: Path):
        """RestartPlan.execute() is awaited exactly once, on a DETACHED
        self-target plan (target_unit == own_unit == transient_unit,
        verify=None, transient_unit set, on_active_secs forwarded), carrying
        an EscalationSpec with the fire-time summary/detail."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.proc_supervision import (
            EscalationSpec,
            RestartDisposition,
            RestartOutcome,
            RestartPlan,
        )

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': ['--flag'],
            'cwd': '/home/leo/src/dark-factory',
            'target_unit': 'orchestrator-reify.service',
        }

        captured_plans: list = []
        canned = RestartOutcome(disposition=RestartDisposition.SCHEDULED)

        async def _fake_execute(self, *, runner=None, inspector=None):
            captured_plans.append(self)
            return canned

        with patch.object(RestartPlan, 'execute', _fake_execute):
            rc, tail = await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-950.service',
                on_active_secs=60,
                task_id='950',
            )

        assert len(captured_plans) == 1, 'RestartPlan.execute must be awaited exactly once'
        plan = captured_plans[0]

        assert plan.target_unit == 'orch-redeploy-restart-950.service'
        assert plan.own_unit == 'orch-redeploy-restart-950.service'
        assert plan.target_unit == plan.own_unit == plan.transient_unit
        assert plan.verify is None
        assert plan.on_active_secs == 60

        spec = plan.on_failure_escalation
        assert isinstance(spec, EscalationSpec)
        assert spec.task_id == '950'
        assert spec.category == 'infra_issue'
        assert spec.agent_role == 'orchestrator-deterministic'
        assert spec.summary == 'Self-restart fire-time failure: orchestrator-reify.service'
        assert 'orch-redeploy-restart-950.service' in spec.detail

        assert rc == 0
        assert tail == ''

    async def test_explicit_summary_forwarded_to_escalation_spec(self, tmp_path: Path):
        """A caller-supplied summary overrides the default fire-time summary."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.proc_supervision import RestartDisposition, RestartOutcome, RestartPlan

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        captured_plans: list = []
        canned = RestartOutcome(disposition=RestartDisposition.SCHEDULED)

        async def _fake_execute(self, *, runner=None, inspector=None):
            captured_plans.append(self)
            return canned

        with patch.object(RestartPlan, 'execute', _fake_execute):
            await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-951.service',
                on_active_secs=60,
                task_id='951',
                summary='Custom fire-time summary',
            )

        assert captured_plans[0].on_failure_escalation.summary == 'Custom fire-time summary'

    async def test_disposition_scheduled_maps_to_rc_zero(self, tmp_path: Path):
        """RestartDisposition.SCHEDULED maps to (0, '')."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.proc_supervision import RestartDisposition, RestartOutcome, RestartPlan

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        canned = RestartOutcome(disposition=RestartDisposition.SCHEDULED)

        async def _fake_execute(self, *, runner=None, inspector=None):
            return canned

        with patch.object(RestartPlan, 'execute', _fake_execute):
            rc, tail = await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-952.service',
                on_active_secs=60,
                task_id='952',
            )

        assert (rc, tail) == (0, '')

    async def test_disposition_registration_failed_maps_to_rc_one_with_detail(self, tmp_path: Path):
        """RestartDisposition.REGISTRATION_FAILED maps to (1, outcome.detail)."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.proc_supervision import RestartDisposition, RestartOutcome, RestartPlan

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'target_unit': 'orchestrator-reify.service',
        }
        canned = RestartOutcome(
            disposition=RestartDisposition.REGISTRATION_FAILED,
            detail='systemd-run registration failed: rc=1: boom',
        )

        async def _fake_execute(self, *, runner=None, inspector=None):
            return canned

        with patch.object(RestartPlan, 'execute', _fake_execute):
            rc, tail = await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-953.service',
                on_active_secs=60,
                task_id='953',
            )

        assert rc == 1
        assert tail == 'systemd-run registration failed: rc=1: boom'

    async def test_behaviour_identity_argv_and_gated_escalation_via_real_execute(self, tmp_path: Path):
        """Behaviour-identity: a REAL execute() (fake create_subprocess_exec
        runner) still produces the systemd-run argv shape and the gated
        on-failure escalation wrapper."""
        from unittest.mock import patch

        from orchestrator.deterministic_runner import DeterministicRunner

        queue = EscalationQueue(tmp_path)
        runner = DeterministicRunner(scheduler=MagicMock(), escalation_queue=queue)

        before_done = {
            'script': '/usr/local/bin/restart-deploy.sh',
            'args': [],
            'cwd': '/home/leo/src/dark-factory',
            'target_unit': 'orchestrator-reify.service',
        }

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b'', b''))
        mock_proc.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_proc) as mock_exec:
            rc, tail = await runner._default_schedule_detached_restart(
                before_done,
                transient_unit='orch-redeploy-restart-954.service',
                on_active_secs=60,
                task_id='954',
            )

        assert rc == 0
        assert tail == ''
        argv = mock_exec.call_args_list[0].args
        all_argv = ' '.join(str(a) for a in argv)
        assert 'systemd-run' in all_argv
        assert '--user' in all_argv
        assert '--on-active' in all_argv
        assert '--collect' in all_argv
        assert '--working-directory=/home/leo/src/dark-factory' in all_argv
        assert 'orch-redeploy-restart-954.service' in all_argv
        assert argv[-3] == '/bin/sh' and argv[-2] == '-c', (
            f'expected a /bin/sh -c wrapper payload, got {argv!r}'
        )
        wrapped = argv[-1]
        assert '/usr/local/bin/restart-deploy.sh' in wrapped
        assert '-ne 0' in wrapped, f'escalation must be gated behind a non-zero exit check: {wrapped!r}'
        assert 'escalation' in wrapped and 'submit' in wrapped


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

    async def test_gate_fallthrough_advances_deploy_state_phase_scheduled_to_escalated(
        self, tmp_path: Path
    ):
        """ζ: self-restart act-then-ask falls through ran->scheduled (self-restart
        stamp) -> escalated (gate filing) — the gate's update_task call carries
        deploy_state.phase=='escalated'."""
        task = self._act_then_ask_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = self._make_runner(scheduler, queue, unit_inspector, script_runner, restart_scheduler)
        await runner.run(assignment)

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls
        assert deploy_state_calls[-1].args[1]['deploy_state']['phase'] == 'escalated'

    async def test_full_phase_sequence_ran_scheduled_escalated(self, tmp_path: Path):
        """Reviewer amendment (task 2240, test_coverage): the specific reason
        the two new _LEGAL edges (ran->scheduled, scheduled->escalated) were
        added — a self-restart act-then-ask deploy traverses BOTH within a
        single dispatch. Assert the FULL ordered sequence (not just the
        final call): the shared before_done_ran_at write lands 'ran', the
        self-restart stamp lands 'scheduled', and the gate filing lands
        'escalated'."""
        task = self._act_then_ask_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        script_runner = AsyncMock(return_value=(0, 'ok'))
        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))

        runner = self._make_runner(scheduler, queue, unit_inspector, script_runner, restart_scheduler)
        await runner.run(assignment)

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        phases = [c.args[1]['deploy_state']['phase'] for c in deploy_state_calls]
        assert phases == ['ran', 'scheduled', 'escalated'], (
            f'expected the phase machine to traverse ran -> scheduled -> '
            f'escalated in order within a single dispatch, got {phases}'
        )


# ---------------------------------------------------------------------------
# Step-13 (ε): robustness_crash_resume + always_escalates=True
# Crash between before_done_scheduled_at stamp and gate filing.
# Resume must (re-)file the milestone gate and block (NOT drive to done).
# RED until step-14 guards (b-self) on always_escalates and extracts helper.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSelfRestartScheduledCrashResumeActThenAsk:
    """before_done_ran_at + before_done_scheduled_at set, always_escalates=True, empty queue.

    Simulates a crash AFTER before_done_scheduled_at was stamped but BEFORE section 3
    filed the milestone gate.  Resume must (re-)file the milestone gate and block —
    NOT drive to done — so the act-then-ask human-approval gate is never bypassed.
    """

    def _make_task(self) -> dict:
        task = _deploy_task(
            task_id='855',
            target_unit='orchestrator-reify.service',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-06-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-855.service',
                'fire_delay_secs': 60,
            },
        )
        task['metadata']['always_escalates'] = True
        return task

    async def test_outcome_is_blocked(self, tmp_path: Path):
        """always_escalates=True scheduled-resume must return BLOCKED, NOT DONE."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = self._make_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty — gate_escalated_at unset
        scheduler = _mock_scheduler(task)

        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            restart_scheduler=restart_scheduler,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

    async def test_set_task_status_never_done(self, tmp_path: Path):
        """act-then-ask scheduled-resume must NOT call set_task_status with 'done'."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = self._make_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            restart_scheduler=AsyncMock(return_value=(0, 'scheduled')),
        )
        await runner.run(assignment)

        for call in scheduler.set_task_status.await_args_list:
            status = call.args[1] if len(call.args) > 1 else call.kwargs.get('status')
            assert status != 'done', (
                f'set_task_status was called with "done" but must not be on act-then-ask '
                f'scheduled-resume: {call}'
            )

    async def test_milestone_gate_escalation_filed(self, tmp_path: Path):
        """act-then-ask scheduled-resume must file exactly one milestone_gate escalation."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = self._make_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            restart_scheduler=AsyncMock(return_value=(0, 'scheduled')),
        )
        await runner.run(assignment)

        pending = queue.get_by_task('855', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.category == 'milestone_gate', (
            f'Expected category "milestone_gate", got "{esc.category}"'
        )
        assert esc.level == 2, f'Expected level 2, got {esc.level}'
        assert esc.severity == 'critical', f'Expected severity "critical", got "{esc.severity}"'
        assert esc.agent_role == 'orchestrator-deterministic', (
            f'Expected role "orchestrator-deterministic", got "{esc.agent_role}"'
        )

    async def test_gate_escalated_at_stamped(self, tmp_path: Path):
        """act-then-ask scheduled-resume must stamp gate_escalated_at so next resume is quiescent."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = self._make_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            restart_scheduler=AsyncMock(return_value=(0, 'scheduled')),
        )
        await runner.run(assignment)

        gate_stamped = any(
            call.args[1].get('gate_escalated_at')
            for call in scheduler.update_task.await_args_list
            if len(call.args) > 1 and isinstance(call.args[1], dict)
        )
        assert gate_stamped, (
            'gate_escalated_at must be stamped via update_task so next resume is quiescent'
        )

    async def test_i1_no_reschedule(self, tmp_path: Path):
        """I1 crash-safe: restart_scheduler must NOT be awaited on scheduled-resume."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = self._make_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        restart_scheduler = AsyncMock(return_value=(0, 'scheduled'))
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            restart_scheduler=restart_scheduler,
        )
        await runner.run(assignment)

        restart_scheduler.assert_not_awaited()

    async def test_i1_no_script_runner(self, tmp_path: Path):
        """I1 crash-safe: script_runner must NOT be awaited on scheduled-resume."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = self._make_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'ok'))
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=script_runner,
            restart_scheduler=AsyncMock(return_value=(0, 'scheduled')),
        )
        await runner.run(assignment)

        script_runner.assert_not_awaited()

    async def test_i1_no_unit_inspector(self, tmp_path: Path):
        """I1 crash-safe: unit_inspector must NOT be awaited on scheduled-resume."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = self._make_task()
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock(return_value=_BASELINE_UNIT_STATE)
        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=AsyncMock(return_value=(0, 'ok')),
            restart_scheduler=AsyncMock(return_value=(0, 'scheduled')),
        )
        await runner.run(assignment)

        unit_inspector.assert_not_awaited()

    async def test_crash_resume_gate_refile_advances_scheduled_to_escalated(self, tmp_path: Path):
        """ζ: a task resuming with deploy_state.phase=='scheduled' already
        persisted (the step-6-written shape) advances to 'escalated' when the
        crash-resume re-files the milestone gate."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(
            task_id='855',
            target_unit='orchestrator-reify.service',
            before_done_ran_at='2026-06-23T10:00:00+00:00',
            before_done_scheduled_at={
                'at': '2026-06-23T10:00:01+00:00',
                'transient_unit': 'orch-redeploy-restart-855.service',
                'fire_delay_secs': 60,
            },
            phase='scheduled',
        )
        task['metadata']['always_escalates'] = True
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=AsyncMock(return_value=_BASELINE_UNIT_STATE),
            script_runner=AsyncMock(return_value=(0, 'ok')),
            restart_scheduler=AsyncMock(return_value=(0, 'scheduled')),
        )
        await runner.run(assignment)

        deploy_state_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and (c.args[1].get('deploy_state') or {}).get('phase')
        ]
        assert deploy_state_calls
        assert deploy_state_calls[-1].args[1]['deploy_state']['phase'] == 'escalated'


# ---------------------------------------------------------------------------
# TestSharedDoneProvenance (task 2167 — W3-δ: shared TaskMetadata seam, SEAM B)
# ---------------------------------------------------------------------------

class TestSharedDoneProvenance:
    """Unit tests for the ``_build_done_provenance`` helper (SEAM B).

    ``_build_done_provenance`` is the single seam every runner
    ``done_provenance`` construction routes through:
    ``DoneProvenance(kind=kind, **fields).model_dump(exclude_none=True)``.
    This shares ONE valid-kinds enum with the fused-memory validator (I2) —
    an unknown/typo kind raises pydantic ``ValidationError`` at BUILD time on
    the orchestrator side, structurally preventing the 1902/1976/1982
    permanently-blocked self-restart failure mode (a kind fused-memory
    silently rejects).  ``exclude_none`` keeps the emitted wire dict
    byte-compatible with the hand-written literals it replaces; extra fields
    (``transient_unit``/``fire_delay_secs``) survive via ``DoneProvenance``'s
    ``extra='allow'``.
    """

    def test_scheduled_kind_preserves_extra_fields(self):
        """deterministic-deploy-scheduled + unit + extras -> all preserved."""
        from orchestrator.deterministic_runner import _build_done_provenance

        result = _build_done_provenance(
            'deterministic-deploy-scheduled',
            unit='u',
            transient_unit='orch-redeploy-restart-850.service',
            fire_delay_secs=5,
        )

        assert result['kind'] == 'deterministic-deploy-scheduled'
        assert result['unit'] == 'u'
        assert result['transient_unit'] == 'orch-redeploy-restart-850.service'
        assert result['fire_delay_secs'] == 5

    def test_deploy_kind_preserves_pid_and_timestamp(self):
        """deterministic-deploy + pid + active_enter_timestamp -> preserved."""
        from orchestrator.deterministic_runner import _build_done_provenance

        result = _build_done_provenance(
            'deterministic-deploy',
            pid=200,
            unit='u',
            active_enter_timestamp='2026-07-06T00:00:00+00:00',
        )

        assert result['pid'] == 200
        assert result['active_enter_timestamp'] == '2026-07-06T00:00:00+00:00'

    def test_unknown_kind_raises_validation_error(self):
        """An unknown/typo kind raises pydantic ValidationError at build time.

        I2 structural prevention: it is impossible for the runner to emit a
        kind the shared/backend enum rejects — the exact 1902/1976/1982
        permanently-blocked self-restart failure mode.
        """
        from pydantic import ValidationError

        from orchestrator.deterministic_runner import _build_done_provenance

        with pytest.raises(ValidationError):
            _build_done_provenance('bogus')

    def test_none_valued_optional_fields_excluded(self):
        """exclude_none -> no 'commit'/'note'/'pid' keys when unset.

        Byte-compatibility with the hand-written literals it replaces: those
        never carried explicit None values either.
        """
        from orchestrator.deterministic_runner import _build_done_provenance

        result = _build_done_provenance('deterministic-deploy')

        assert 'commit' not in result
        assert 'note' not in result
        assert 'pid' not in result

    def test_emitted_dict_round_trips_through_done_provenance(self):
        """The emitted dict re-validates cleanly through DoneProvenance itself.

        'deterministic-deploy-scheduled' has no conditional commit/note
        requirement, so round-tripping the built dict back through the model
        must not raise.
        """
        from shared.task_metadata import DoneProvenance

        from orchestrator.deterministic_runner import _build_done_provenance

        built = _build_done_provenance('deterministic-deploy-scheduled', unit='u')

        DoneProvenance(**built)  # must not raise


# ---------------------------------------------------------------------------
# Task 3286: `_summarize_predicate_output` — the ALLOWLIST sanitizer standing
# between a predicate script's raw stdout tail and `done_provenance.note`.
#
# The note is not a private field: fused-memory's `_format_outcome_echo` reads
# it and appends it to a Mem0 completion-summary write, so anything that lands
# here is ingested into the knowledge graph.  Task 2902 is the specimen that
# proved raw forwarding corrupts it.
# ---------------------------------------------------------------------------

class TestSummarizePredicateOutput:
    """The sanitizer keeps the structured verdict and drops the log noise."""

    def test_specimen_starts_with_deterministic_verdict_prefix(self):
        """The verdict prefix is unconditional — a note is never empty."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        assert result.startswith('predicate check passed (rc=0)'), result

    @pytest.mark.parametrize(
        'marker',
        [
            'fused_memory.backends.graphiti_client',
            'httpx',
            'my_solar_challenge',
            'HTTP/1.1 200 OK',
            'WARNING',
        ],
    )
    def test_specimen_server_log_markers_are_dropped(self, marker: str):
        """No server-log noise survives into the note (the task-2902 leak)."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        assert marker not in result, (
            f'{marker!r} leaked into the provenance note: {result!r}'
        )

    def test_specimen_trailing_json_verdict_survives_compacted(self):
        """The script's OWN structured verdict is the part worth keeping."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        # Compact json.dumps separators — no space after ':' or ','.
        assert '"orphan_count":0' in result, result
        assert '"verdict":"clean"' in result, result

    def test_specimen_result_is_single_line(self):
        """A note flowing into a Mem0 summary must not carry newlines."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        assert '\n' not in result, result

    def test_specimen_mid_token_first_line_is_dropped(self):
        """The 2000-char tail cut starts mid-word — that fragment is noise."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)

        assert '_tariff_pence_per_kwh' not in result, result

    # -- Tier 2: one clean final line survives -----------------------------
    #
    # Each shape below is a REAL in-repo predicate script's output, and each
    # is load-bearing for a pre-existing green test.  A sanitizer that dropped
    # them would be throwing away the verdict it exists to preserve.

    @pytest.mark.parametrize(
        ('out', 'expected'),
        [
            # test_deterministic_runner.py's own predicate-mode fixtures.
            ('check ok: 0 flakes', 'check ok: 0 flakes'),
            # scripts/check_merge_flakiness.sh — drives the REAL-subprocess
            # test at test_milestone_integration_gate.py's exemplar-pass case.
            (
                'check_merge_flakiness: value=1 threshold=5 window_days=7 '
                '-- invariant holds',
                'invariant holds',
            ),
            # scripts/check_esc_analytics_perf.sh.
            (
                'measured_median_ms=12 attempts=5 threshold_ms=2000 '
                'url=http://127.0.0.1:8080/api/escalations/analytics',
                'measured_median_ms=12 attempts=5',
            ),
        ],
    )
    def test_clean_final_line_is_preserved(self, out: str, expected: str):
        """A single clean verdict line is the payload — kept verbatim."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output(out, rc=0)

        assert result.startswith('predicate check passed (rc=0)'), result
        assert expected in result, result

    def test_trailing_blank_lines_do_not_defeat_extraction(self):
        """A script ending with a newline still yields its verdict line."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output('check ok: 0 flakes\n\n', rc=0)

        assert 'check ok: 0 flakes' in result, result

    # -- Log-shaped lines are rejected -------------------------------------

    def test_pure_log_output_yields_bare_verdict(self):
        """Nothing but logger lines -> no payload at all, not a log line."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        out = (
            '2026-07-30 16:39:00,523 fused_memory.backends.graphiti_client '
            'WARNING identity scan found 3 candidate nodes\n'
            '2026-07-30 16:39:00,625 httpx INFO HTTP Request: GET '
            'http://localhost:6333 "HTTP/1.1 200 OK"'
        )

        result = _summarize_predicate_output(out, rc=0)

        assert result == 'predicate check passed (rc=0)', result

    @pytest.mark.parametrize(
        'out',
        [
            '2026-07-30 16:39:00,523 fused_memory.x INFO done',
            # No timestamp — the standalone level token alone is enough.
            'INFO: all checks passed',
        ],
    )
    def test_level_token_final_line_is_rejected(self, out: str):
        """Deliberately conservative: a level token forfeits the payload.

        Losing a payload is the safe failure direction — the verdict prefix
        always survives and the raw text is logged (step-6).
        """
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output(out, rc=0)

        assert result == 'predicate check passed (rc=0)', result

    # -- Size bound: elide wholesale, never slice mid-structure -------------

    def test_oversized_payload_is_elided_not_sliced(self):
        """Task 2054's lesson: a mid-structure cut is worse than no payload.

        A raw ``note[:N]`` slice there garbled ``8679,8680`` into
        ``8679,868``.  A sliced JSON object is worse still — unparseable, yet
        still structured-looking to a reader.
        """
        import json as _json

        from orchestrator.deterministic_runner import (
            _PREDICATE_NOTE_MAX_PAYLOAD_CHARS,
            _summarize_predicate_output,
        )

        obj = {'ids': list(range(2000))}
        compact = _json.dumps(obj, separators=(',', ':'))
        assert len(compact) > _PREDICATE_NOTE_MAX_PAYLOAD_CHARS, 'fixture too small'

        result = _summarize_predicate_output(_json.dumps(obj, indent=2), rc=0)

        # Bounded: the cap plus the verdict prefix and the marker's own text.
        assert len(result) <= _PREDICATE_NOTE_MAX_PAYLOAD_CHARS + 200, len(result)
        # The marker names the dropped size and where the full text lives.
        assert str(len(compact)) in result, result
        assert 'elided' in result, result
        # Critically: NOT a prefix-slice of the compact dump.
        assert compact[:100] not in result, result

    def test_payload_at_the_cap_is_kept(self):
        """The bound elides only what exceeds it — it is not a blanket drop."""
        from orchestrator.deterministic_runner import (
            _PREDICATE_NOTE_MAX_PAYLOAD_CHARS,
            _summarize_predicate_output,
        )

        out = 'ok ' + 'x' * (_PREDICATE_NOTE_MAX_PAYLOAD_CHARS - 10)
        result = _summarize_predicate_output(out, rc=0)

        assert out in result, result
        assert 'elided' not in result, result

    # -- Degenerate inputs -------------------------------------------------

    @pytest.mark.parametrize('out', ['', None, 42, b'check ok', ['check ok']])
    def test_degenerate_output_yields_bare_verdict(self, out: object):
        """A falsy/non-str seam return never raises and never leaks a repr."""
        from orchestrator.deterministic_runner import _summarize_predicate_output

        result = _summarize_predicate_output(out, rc=0)

        assert result == 'predicate check passed (rc=0)', result


# ---------------------------------------------------------------------------
# Task 2336 (γ-predicate): predicate deterministic mode — a read-only
# exit-code verdict check (before_done.kind == 'predicate'), NOT a systemd
# deploy.  Boundary tests B7 (pass), B8 (fail), B9 (timeout/infra), B10
# (resume).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPredicateModePassPath:
    """DeterministicRunner — predicate mode rc==0 -> done, read-only (B7)."""

    async def test_predicate_pass_outcome_is_done(self, tmp_path: Path):
        """rc==0 -> WorkflowOutcome.DONE (B7)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE

    async def test_predicate_pass_sets_done_with_milestone_provenance(self, tmp_path: Path):
        """set_task_status awaited once with 'done' + provenance.kind='deterministic-milestone' (B7)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        scheduler.set_task_status.assert_awaited_once()
        call = scheduler.set_task_status.call_args
        assert call.args[0] == '700'
        assert call.args[1] == 'done'
        provenance = call.kwargs.get('done_provenance')
        assert provenance is not None, 'done_provenance must be passed as a kwarg'
        assert provenance['kind'] == 'deterministic-milestone'

    async def test_predicate_pass_provenance_note_contains_check_verdict(self, tmp_path: Path):
        """done_provenance.note carries the check's own verdict line (B7).

        Task 3286 narrowed this from the raw stdout tail to a bounded
        structured summary; a single clean verdict line still survives intact.
        """
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        call = scheduler.set_task_status.call_args
        provenance = call.kwargs.get('done_provenance')
        assert 'check ok' in provenance.get('note', ''), (
            f"the check's verdict line must survive into the provenance "
            f'note: {provenance!r}'
        )

    async def test_predicate_pass_provenance_note_is_sanitized(self, tmp_path: Path):
        """A chatty script's log noise never reaches done_provenance.note.

        Task 3286 / specimen 2902: the note is read by fused-memory's
        `_format_outcome_echo` and appended to a Mem0 completion summary, so
        raw subprocess output landing here is ingested into memory.
        """
        from orchestrator.deterministic_runner import (
            DeterministicRunner,
            _summarize_predicate_output,
        )

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, POLLUTED_PREDICATE_OUTPUT))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        note = scheduler.set_task_status.call_args.kwargs['done_provenance']['note']

        assert note == _summarize_predicate_output(POLLUTED_PREDICATE_OUTPUT, rc=0)
        for marker in (
            'fused_memory.backends.graphiti_client',
            'httpx',
            'my_solar_challenge',
        ):
            assert marker not in note, f'{marker!r} leaked into the note: {note!r}'
        # The live specimen was 1999 chars; the note is now bounded.
        assert len(note) <= 500, f'note must stay bounded, got {len(note)}: {note!r}'

    async def test_predicate_pass_logs_raw_output_before_summarizing(
        self, tmp_path: Path, caplog,
    ):
        """The summarizer discards content — the raw text must stay recoverable.

        The orchestrator log is the right home for it: durable and
        operator-accessible, and (unlike the note) never memory-ingested.
        """
        import logging

        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, POLLUTED_PREDICATE_OUTPUT))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        with caplog.at_level(logging.INFO, logger='orchestrator.deterministic_runner'):
            await runner.run(assignment)

        logged = '\n'.join(record.getMessage() for record in caplog.records)
        assert 'my_solar_challenge' in logged, (
            f'raw predicate output must be logged, not silently dropped: {logged!r}'
        )

    async def test_predicate_pass_script_runner_called_once_with_before_done(self, tmp_path: Path):
        """script_runner invoked exactly once with the full before_done dict (B7)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        before_done = task['metadata']['before_done']
        script_runner.assert_awaited_once_with(before_done)

    async def test_predicate_pass_no_unit_inspect(self, tmp_path: Path):
        """unit_inspector is NEVER awaited on the predicate path — no systemd inspect (B7)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        unit_inspector.assert_not_awaited()

    async def test_predicate_pass_no_before_done_ran_at_stamp(self, tmp_path: Path):
        """update_task is NEVER awaited — a read-only predicate has no I1 stamp (B7)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        scheduler.update_task.assert_not_awaited()

    async def test_predicate_pass_no_escalation_filed(self, tmp_path: Path):
        """No escalation is filed on a passing predicate check (B7)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='700')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        assert queue.get_by_task('700', status='pending') == []

    async def test_predicate_pass_ignores_always_escalates_true(self, tmp_path: Path):
        """A kind='predicate' task with always_escalates=True still drives
        straight to done from the check's exit code alone — always_escalates
        is never consulted on the predicate path (documented contract;
        reviewer amendment).  No milestone_gate escalation is filed and the
        act-then-ask gate machinery is never entered."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='700')
        task['metadata']['always_escalates'] = True
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        call = scheduler.set_task_status.call_args
        assert call.args[1] == 'done'
        assert call.kwargs['done_provenance']['kind'] == 'deterministic-milestone'
        assert queue.get_by_task('700', status='pending') == [], (
            'always_escalates=True must not cause a gate escalation to be '
            'filed on the predicate path — it is simply ignored'
        )


# ---------------------------------------------------------------------------
# Step-3: RED — B8 predicate rc!=0 -> milestone_check_failed born-at-L2 +
# gate_escalated_at + blocked
# (RED until step-4 adds the rc!=0 branch + _file_milestone_check_failed_and_block)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPredicateModeFailPath:
    """DeterministicRunner — predicate mode rc!=0 -> milestone_check_failed L2 + blocked (B8)."""

    async def test_predicate_fail_outcome_is_blocked(self, tmp_path: Path):
        """rc!=0 -> WorkflowOutcome.BLOCKED (B8)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='701')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(1, 'FAIL: 3 merge flakes detected'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

    async def test_predicate_fail_files_one_milestone_check_failed_escalation(self, tmp_path: Path):
        """Exactly one pending escalation: category='milestone_check_failed',
        level==2, severity=='critical', agent_role sentinel (B8)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='701')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(1, 'FAIL: 3 merge flakes detected'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        pending = queue.get_by_task('701', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.category == 'milestone_check_failed'
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-deterministic'

    async def test_predicate_fail_escalation_detail_contains_rc_and_stdout_tail(self, tmp_path: Path):
        """Escalation detail must contain both the rc (e.g. 'rc=1') and the
        stdout tail (B8)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='701')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(1, 'FAIL: 3 merge flakes detected'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        esc = queue.get_by_task('701', status='pending')[0]
        assert 'rc=1' in esc.detail, f'rc must appear in detail: {esc.detail!r}'
        assert 'FAIL: 3 merge flakes detected' in esc.detail, (
            f'stdout tail must appear in detail: {esc.detail!r}'
        )

    async def test_predicate_fail_stamps_gate_escalated_at_not_before_done_ran_at(
        self, tmp_path: Path,
    ):
        """update_task awaited with a truthy gate_escalated_at; before_done_ran_at
        is NEVER stamped on a read-only predicate (B8)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='701')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(1, 'FAIL: 3 merge flakes detected'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        scheduler.update_task.assert_awaited_once()
        call_args = scheduler.update_task.call_args
        metadata_update = call_args.args[1] if call_args.args else call_args.kwargs.get('metadata', {})
        assert metadata_update.get('gate_escalated_at'), (
            'gate_escalated_at should be a truthy ISO timestamp'
        )
        assert 'before_done_ran_at' not in metadata_update, (
            f'a read-only predicate must never stamp before_done_ran_at: {metadata_update!r}'
        )

    async def test_predicate_fail_sets_blocked_never_done(self, tmp_path: Path):
        """set_task_status awaited with '701','blocked' and NEVER with 'done' (B8)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='701')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(1, 'FAIL: 3 merge flakes detected'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        scheduler.set_task_status.assert_awaited_once_with('701', 'blocked')

    async def test_predicate_fail_no_unit_inspect(self, tmp_path: Path):
        """unit_inspector is NEVER awaited on the predicate path — no systemd
        inspect, even on a check failure (B8)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _predicate_task(task_id='701')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(1, 'FAIL: 3 merge flakes detected'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        unit_inspector.assert_not_awaited()

    async def test_predicate_fail_dedup_guard_skips_refile_when_pending_exists(
        self, tmp_path: Path,
    ):
        """If a pending milestone_check_failed escalation already exists for
        the task (e.g. the gate_escalated_at stamp failed on a prior
        crash-safe dispatch, leaving the escalation filed but un-stamped), a
        second rc!=0 dispatch must NOT file a duplicate.  The dedup guard in
        _file_milestone_check_failed_and_block skips re-filing, but still
        (re-)stamps gate_escalated_at and blocks (test_coverage amendment —
        this crash window is otherwise unreachable via the normal quiescence
        path, which returns BLOCKED before _run_predicate ever re-runs)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='701')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)

        # Pre-seed one pending milestone_check_failed escalation, simulating a
        # prior dispatch where filing succeeded but the gate_escalated_at
        # stamp did not land.
        _seed_escalation(
            queue, '701', 'orchestrator-deterministic', category='milestone_check_failed',
        )

        scheduler = _mock_scheduler(task)
        script_runner = AsyncMock(return_value=(1, 'FAIL: 3 merge flakes detected'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('701', status='pending')
        assert len(pending) == 1, (
            f'dedup guard must skip re-filing — expected still exactly 1 '
            f'pending escalation, got {len(pending)}'
        )

        # gate_escalated_at is still (re-)stamped even though no new
        # escalation was filed.
        scheduler.update_task.assert_awaited_once()
        call_args = scheduler.update_task.call_args
        metadata_update = call_args.args[1] if call_args.args else call_args.kwargs.get('metadata', {})
        assert metadata_update.get('gate_escalated_at')

        scheduler.set_task_status.assert_awaited_once_with('701', 'blocked')

    async def test_predicate_fail_blocked_write_failure_still_returns_blocked(
        self, tmp_path: Path,
    ):
        """A severed connection on the trailing set_task_status('blocked')
        call must not mask the already-durable milestone_check_failed
        escalation — run() must still return BLOCKED, never propagate
        (mirrors _file_infra_issue_and_block's best-effort guard; reviewer
        amendment)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='701')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        scheduler.set_task_status = AsyncMock(
            side_effect=RuntimeError('transient: fused-memory unavailable')
        )

        script_runner = AsyncMock(return_value=(1, 'FAIL: invariant violated'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('701', status='pending')
        assert len(pending) == 1, (
            'the escalation must still be filed even when the fallback '
            'blocked-status write also fails'
        )
        assert pending[0].category == 'milestone_check_failed'


# ---------------------------------------------------------------------------
# Step-5: RED — B9 predicate timeout/unexpected-error -> infra_issue + blocked
# (RED until step-6 wraps _run_predicate's asyncio.wait_for call in the
# deploy path's except-TimeoutError / except-Exception outer-guard branches)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPredicateModeTimeout:
    """DeterministicRunner — predicate outer-guard timeout/error -> infra_issue + blocked (B9).

    A hung or erroring script_runner produces NO exit code, so there is no
    verdict to report — this is an INFRA fault (exactly like the deploy
    path's outer-guard handling), never a milestone_check_failed verdict.
    Routing to infra_issue (which does NOT stamp gate_escalated_at) means the
    check is re-attempted on the next dispatch rather than latched into the
    resolve-to-done path.
    """

    async def test_predicate_hang_files_infra_issue_and_blocks(self, tmp_path: Path):
        """A script_runner that hangs forever must still produce BLOCKED and
        exactly one L2 infra_issue escalation — NOT milestone_check_failed (B9).

        RED today: _run_predicate has no outer-guard try/except around its
        ``await asyncio.wait_for(_invoke_run_fn(), timeout=outer_timeout)`` —
        the TimeoutError propagates uncaught.
        """
        import asyncio

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='702', timeout_secs=0)
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        async def _hang(_before_done):
            await asyncio.Event().wait()

        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=_hang,
            run_timeout_grace_secs=0.05,
        )

        # Hang tripwire: if the outer guard regresses, fail loudly instead of
        # stalling the suite.
        outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('702', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.category == 'infra_issue', (
            f'a hung seam is an infra fault, not a verdict — must not be '
            f'milestone_check_failed: {esc.category!r}'
        )

        scheduler.set_task_status.assert_awaited_once_with('702', 'blocked')
        unit_inspector.assert_not_awaited()

    async def test_predicate_unexpected_exception_files_infra_issue_and_blocks(
        self, tmp_path: Path,
    ):
        """A script_runner that raises an unexpected exception must NOT
        propagate — _run_predicate must route to _file_infra_issue_and_block
        and return BLOCKED (B9).

        RED today: _run_predicate propagates the RuntimeError uncaught.
        """
        import asyncio

        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='703')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        async def _boom(_before_done):
            raise RuntimeError('predicate script spawn exploded')

        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=_boom,
        )

        outcome = await asyncio.wait_for(runner.run(assignment), timeout=5)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('703', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.category == 'infra_issue'

        scheduler.set_task_status.assert_awaited_once_with('703', 'blocked')
        unit_inspector.assert_not_awaited()


# ---------------------------------------------------------------------------
# Step-7: RED — B10 predicate resume: gate_escalated_at set + escalation
# resolved -> done with deterministic-milestone provenance (NOT
# deterministic-deploy). (RED until step-8 adds the predicate-aware resume
# branch in section 1, before the before_done_ran_at NotImplementedError check)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPredicateModeResume:
    """DeterministicRunner — predicate resume/quiescence via gate_escalated_at (B10).

    A predicate never stamps before_done_ran_at (read-only, I1 doesn't
    apply), so the section-1 resolved-escalation branch must short-circuit
    for a predicate BEFORE the before_done_ran_at proof-check that the deploy
    path relies on — otherwise resume would wrongly raise NotImplementedError.
    """

    async def test_predicate_resume_no_open_escalation_drives_to_done(self, tmp_path: Path):
        """gate_escalated_at set + no pending escalation + NO before_done_ran_at
        -> must NOT raise NotImplementedError; RE-RUNS the predicate check and,
        on a passing re-check (rc==0), drives to DONE with deterministic-milestone
        provenance (not deterministic-deploy) (B10; reviewer amendment: resume
        re-verifies the invariant rather than trusting the resolution blindly)."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        # gate already escalated (milestone_check_failed); escalation resolved
        # (no pending); before_done_ran_at is intentionally NEVER set for a
        # read-only predicate.
        task = _predicate_task(task_id='703', gate_escalated_at='2026-07-08T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty queue — escalation resolved
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: invariant now holds'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        # Must NOT raise NotImplementedError — a predicate never stamps
        # before_done_ran_at, so the proof-check must not apply to it.
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE

        # The check is genuinely RE-RUN on resume — not skipped/trusted blindly.
        script_runner.assert_awaited_once_with(task['metadata']['before_done'])
        unit_inspector.assert_not_awaited()

        scheduler.set_task_status.assert_awaited_once()
        call = scheduler.set_task_status.call_args
        assert call.args[0] == '703'
        assert call.args[1] == 'done'
        provenance = call.kwargs.get('done_provenance')
        assert provenance is not None, 'done_provenance must be passed as a kwarg'
        assert provenance['kind'] == 'deterministic-milestone'
        assert provenance['kind'] != 'deterministic-deploy', (
            'predicate resume must not claim a systemd deploy happened'
        )
        assert 'check ok' in provenance.get('note', ''), (
            f"the re-run's verdict line must survive into the provenance "
            f'note: {provenance!r}'
        )

    async def test_predicate_resume_recheck_still_failing_refiles_and_stays_blocked(
        self, tmp_path: Path,
    ):
        """gate_escalated_at set + no pending escalation (prior escalation was
        resolved), but the RE-CHECK still fails (rc!=0) -> must NOT drive to
        done; re-files a NEW milestone_check_failed escalation and stays
        BLOCKED (reviewer amendment: resolving the escalation is not proof the
        invariant now holds — e.g. a human resolved prematurely or in error —
        so resume re-verifies instead of latching a false 'milestone done')."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='704', gate_escalated_at='2026-07-08T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)  # empty queue — prior escalation resolved
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(1, 'still failing: 2 flakes remain'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        script_runner.assert_awaited_once_with(task['metadata']['before_done'])
        unit_inspector.assert_not_awaited()

        pending = queue.get_by_task('704', status='pending')
        assert len(pending) == 1, (
            f'Expected exactly 1 re-filed pending escalation, got {len(pending)}'
        )
        esc = pending[0]
        assert esc.category == 'milestone_check_failed'
        assert 'rc=1' in esc.detail, f'rc must appear in detail: {esc.detail!r}'
        assert 'still failing' in esc.detail, (
            f'stdout tail must appear in detail: {esc.detail!r}'
        )

        scheduler.set_task_status.assert_awaited_once_with('704', 'blocked')

    async def test_predicate_resume_quiescence_open_escalation_returns_blocked(
        self, tmp_path: Path,
    ):
        """gate_escalated_at set + still-open milestone_check_failed escalation
        -> BLOCKED, no second escalation filed, no done write (B10 quiescence).

        Already green via the unchanged section-1 pending branch — this test
        locks the contract rather than driving new behaviour.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(task_id='703', gate_escalated_at='2026-07-08T12:00:00+00:00')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)

        # Pre-seed one pending milestone_check_failed escalation (as B8 would file)
        _seed_escalation(
            queue, '703', 'orchestrator-deterministic', category='milestone_check_failed',
        )

        scheduler = _mock_scheduler(task)
        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = queue.get_by_task('703', status='pending')
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'


# ---------------------------------------------------------------------------
# Step-7 (task 2509): explicit stop-instruction guard on the FIRST dispatch of
# a non-predicate before_done (act-then-ask/deploy) task — reconciliation
# finding 0aac21b4 (task 2407 self-authorized an irreversible mutation past an
# explicit "do not apply" instruction).  (RED until step-8 adds the guard in
# run()'s section 2, after the predicate dispatch and after the
# before_done_ran_at idempotency block, BEFORE before_done_ran_at is stamped.)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStopInstructionGuard:
    """DeterministicRunner — hard-abort a non-predicate before_done deploy on
    an explicit stop instruction found in the task description, mirroring the
    task 2273 SIGTERM-kill-on-human-rehearsal-mandate precedent as a
    self-halt rather than relying on an external kill.
    """

    async def test_stop_instruction_blocks_before_running_deploy(self, tmp_path: Path):
        """Case A: description contains 'do not apply' -> BLOCKED, script_runner
        and unit_inspector are NEVER awaited, no before_done_ran_at stamp."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='400',
            description='Investigate the failure, but do not apply any fix yet.',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock()
        script_runner = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        script_runner.assert_not_awaited()
        unit_inspector.assert_not_awaited()

        stamp_calls = [
            c for c in scheduler.update_task.call_args_list
            if len(c.args) > 1 and c.args[1].get('before_done_ran_at')
        ]
        assert not stamp_calls, (
            f'before_done_ran_at must NOT be stamped when the stop-instruction '
            f'guard fires: {stamp_calls!r}'
        )

    async def test_stop_instruction_files_born_at_l2_escalation(self, tmp_path: Path):
        """Filed escalation: category='stop_instruction', level=2, severity='critical',
        agent_role='orchestrator-deterministic' (born-at-L2, task 2509)."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(
            task_id='400',
            description='Investigate the failure, but do not apply any fix yet.',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock()
        script_runner = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        pending = queue.get_by_task(
            '400', status='pending', agent_role='orchestrator-deterministic',
        )
        assert len(pending) == 1, f'Expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.category == 'stop_instruction'
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-deterministic'

    async def test_stop_instruction_sets_task_blocked(self, tmp_path: Path):
        """set_task_status is called with 'blocked' (never 'done') when the
        guard fires."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _deploy_task(
            task_id='400',
            description='Investigate the failure, but do not apply any fix yet.',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        unit_inspector = AsyncMock()
        script_runner = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        await runner.run(assignment)

        blocked_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if c.args[1] == 'blocked'
        ]
        assert len(blocked_calls) == 1, 'set_task_status must be called once with blocked'
        done_calls = [
            c for c in scheduler.set_task_status.call_args_list
            if c.args[1] == 'done'
        ]
        assert not done_calls, 'set_task_status must NEVER be called with done'

    async def test_benign_description_regression_reaches_verified_done(self, tmp_path: Path):
        """Case B (regression): identical task shape with a benign description
        runs the script and reaches the normal verified-deploy done path,
        unchanged from the pre-guard behaviour."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='401',
            description='Cross-unit deploy of the reify worker — routine rollout.',
        )
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
        script_runner.assert_awaited_once_with(task['metadata']['before_done'])

        pending = queue.get_by_task('401', status='pending')
        assert len(pending) == 0, f'No escalation should be filed on a benign description: {pending}'

    async def test_predicate_with_stop_instruction_still_runs(self, tmp_path: Path):
        """Case C (predicate exclusion): a before_done.kind=='predicate' task
        whose description contains 'do not apply' still runs the predicate —
        the guard does not fire on read-only checks."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _predicate_task(
            task_id='701',
            description='Milestone predicate check — do not apply any fix without review.',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        script_runner = AsyncMock(return_value=(0, 'check ok: 0 flakes'))
        unit_inspector = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE
        script_runner.assert_awaited_once_with(task['metadata']['before_done'])

        pending = queue.get_by_task('701', status='pending')
        assert len(pending) == 0, (
            f'A predicate must not be blocked by the stop-instruction guard: {pending}'
        )

    async def test_preexisting_unrelated_escalation_does_not_suppress_filing(
        self, tmp_path: Path,
    ):
        """Review amendment: the dedup guard is scoped to category==
        'stop_instruction' — a pre-existing PENDING escalation of a
        DIFFERENT category (e.g. infra_issue, filed by an earlier crash) on
        the same task must NOT suppress filing the stop_instruction
        escalation. A category-agnostic dedup guard would incorrectly treat
        the unrelated pending escalation as "already escalated" and skip
        filing — silently dropping the higher-authority stop-instruction
        signal.
        """
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        task = _deploy_task(
            task_id='402',
            description='Investigate the failure, but do not apply any fix yet.',
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        # Pre-seed an unrelated pending escalation for the SAME task_id and
        # agent_role (mirrors a prior _file_infra_issue_and_block filing).
        queue.submit(Escalation(
            id=queue.make_id('402'),
            task_id='402',
            agent_role='orchestrator-deterministic',
            severity='critical',
            category='infra_issue',
            summary='pre-existing unrelated infra_issue escalation',
            level=2,
        ))

        unit_inspector = AsyncMock()
        script_runner = AsyncMock()

        runner = DeterministicRunner(
            scheduler=scheduler,
            escalation_queue=queue,
            unit_inspector=unit_inspector,
            script_runner=script_runner,
        )
        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED
        script_runner.assert_not_awaited()

        pending = queue.get_by_task(
            '402', status='pending', agent_role='orchestrator-deterministic',
        )
        categories = sorted(e.category for e in pending)
        assert categories == ['infra_issue', 'stop_instruction'], (
            f'Expected the pre-existing infra_issue escalation to remain '
            f'AND a new stop_instruction escalation to be filed alongside '
            f'it, got categories={categories}: {pending}'
        )


# ---------------------------------------------------------------------------
# Task 2803 (γ, part 2): operational_llm-gate born-at-L2 escalation marker
# (PRD plans/operational-ask-routing-prd.md, task γ). β (task 2802) stamps
# metadata['x_operational_llm_gate']=True (+ preserved operational_mode='llm')
# on an execution_class='operational' + operational_mode='llm' submission,
# ALREADY merged to main. γ is the consumer: emit the stable token
# 'operational_llm_needs_lane' in the filed escalation's summary/detail so an
# llm-operational gate is machine-distinguishable from a plain gate.
# (RED until the impl step adds the token emission in
# _file_milestone_gate_and_block.)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOperationalLlmGateMarker:
    """DeterministicRunner — operational_mode='llm' pure gate (β's
    x_operational_llm_gate marker) emits the operational_llm_needs_lane token
    so its born-at-L2 escalation is machine-distinguishable from a plain
    deterministic gate."""

    async def test_llm_gate_escalation_detail_contains_token_and_preserves_invariants(
        self, tmp_path: Path,
    ):
        """An operational_mode='llm' pure gate files a born-at-L2 escalation
        whose detail contains the literal token 'operational_llm_needs_lane',
        while preserving every plain pure-gate invariant: exactly one pending
        escalation, level==2, agent_role=='orchestrator-deterministic',
        category=='milestone_gate', task set to blocked, gate_escalated_at
        stamped, and the original description still present in detail."""
        from orchestrator.deterministic_runner import DeterministicRunner
        from orchestrator.workflow import WorkflowOutcome

        desc = 'This gate guards the Q3 launch milestone'
        task = _llm_gate_task(task_id='99', description=desc)
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        outcome = await runner.run(assignment)

        pending = queue.get_by_task('99', status='pending')
        assert len(pending) == 1, f'Expected 1 pending escalation, got {len(pending)}'
        esc = pending[0]

        assert 'operational_llm_needs_lane' in esc.detail, (
            f'Expected the operational_llm_needs_lane token in detail: {esc.detail!r}'
        )
        assert desc in esc.detail, (
            f'Original description must still appear in detail: {esc.detail!r}'
        )
        assert esc.level == 2
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.category == 'milestone_gate'
        assert outcome == WorkflowOutcome.BLOCKED

        scheduler.set_task_status.assert_awaited_once_with('99', 'blocked')

        scheduler.update_task.assert_awaited_once()
        call_args = scheduler.update_task.call_args
        metadata_update = (
            call_args.args[1] if call_args.args else call_args.kwargs.get('metadata', {})
        )
        assert metadata_update.get('gate_escalated_at'), (
            'gate_escalated_at should be a truthy ISO timestamp'
        )

    async def test_llm_gate_escalation_summary_contains_token(self, tmp_path: Path):
        """The llm-gate escalation summary must also carry the token
        (dashboard/watcher visibility) — built token-first so it survives
        the summary's [:200] truncation."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _llm_gate_task(task_id='99', title='My LLM Gate')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert len(escs) == 1
        assert 'operational_llm_needs_lane' in escs[0].summary, (
            f'Expected the operational_llm_needs_lane token in summary: {escs[0].summary!r}'
        )

    async def test_plain_gate_has_no_token(self, tmp_path: Path):
        """Distinguishability guard (a): a PLAIN pure gate (no marker) must
        carry NO token in summary or detail, and its summary must still
        equal the task title verbatim — proving the token emission is
        marker-gated, not unconditional."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99', title='Plain Gate')
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert len(escs) == 1
        esc = escs[0]
        assert 'operational_llm_needs_lane' not in esc.summary, (
            f'Plain gate summary must carry no token: {esc.summary!r}'
        )
        assert 'operational_llm_needs_lane' not in esc.detail, (
            f'Plain gate detail must carry no token: {esc.detail!r}'
        )
        assert esc.summary == 'Plain Gate'

    async def test_decision_llm_mode_without_marker_has_no_token(self, tmp_path: Path):
        """Distinguishability guard (b): a decision+operational_mode='llm'
        pure gate preserves the raw operational_mode but β withholds the
        x_operational_llm_gate marker for `decision` submissions — so it
        must get NO token either, proving the marker (not the raw
        operational_mode) gates the token."""
        from orchestrator.deterministic_runner import DeterministicRunner

        task = _gate_task(task_id='99', title='Decision Gate')
        task['metadata']['operational_mode'] = 'llm'  # raw mode preserved, no marker
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)

        runner = DeterministicRunner(scheduler=scheduler, escalation_queue=queue)
        await runner.run(assignment)

        escs = queue.get_by_task('99', status='pending')
        assert len(escs) == 1
        esc = escs[0]
        assert 'operational_llm_needs_lane' not in esc.summary, (
            f'decision+llm gate (no marker) summary must carry no token: {esc.summary!r}'
        )
        assert 'operational_llm_needs_lane' not in esc.detail, (
            f'decision+llm gate (no marker) detail must carry no token: {esc.detail!r}'
        )
