"""Milestone ε: end-to-end integration gate (task 2338).

B1–B12 boundary-test suite (docs/prds/milestone-tasks.md §7) that COMPOSES
the already-landed β (scheduler.py time-gate/sweep), γ (deterministic_runner.py
predicate mode), and α (shared.task_metadata.Milestone) substrate end-to-end,
rather than re-deriving their own unit assertions.  Follows the established
test_*_integration_gate.py convention (coalesce/config_reload/warm_lane).

The one genuinely new production artifact is the exemplar predicate fixture,
scripts/check_merge_flakiness.sh — a dependency-free, executable check script
that owns the threshold and the exit-code verdict contract (PRD §5.5: the
orchestrator parses nothing).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import WorkflowOutcome

# ---------------------------------------------------------------------------
# Shared helpers — mirror the harness patterns established in
# test_scheduler.py (TestMilestoneEligibilityGate/TestMilestoneSweepDriven­
# ByAcquireNext: injected wall clock + mocked mcp_call) and
# test_deterministic_runner.py (_predicate_task/_mock_scheduler/real
# DeterministicRunner construction), stitched into ONE end-to-end thread.
# ---------------------------------------------------------------------------


def _task_response(tasks: list[dict]) -> dict:
    """Build the JSON-RPC-shaped get_tasks envelope mcp_call is mocked to return."""
    return {
        'result': {
            'content': [
                {'type': 'text', 'text': json.dumps({'tasks': tasks})}
            ]
        }
    }


def _dep_task(dep_id: str = 'X', status: str = 'pending') -> dict:
    return {
        'id': dep_id,
        'title': 'Dependency',
        'status': status,
        'dependencies': [],
        'metadata': {},
    }


def _delayed_predicate_milestone_task(
    task_id: str,
    script_path: Path,
    *,
    after_secs: int = 100,
    args: list[str] | None = None,
    timeout_secs: int | float = 30,
    milestone_deps_satisfied_at: str | None = None,
    dep_id: str = 'X',
) -> dict:
    """Build a pending, deterministic, delayed-milestone predicate task dict.

    ``before_done`` points at the REAL exemplar script (absolute path,
    kind='predicate', target_unit=None — a predicate is never a systemd
    deploy).  ``metadata.milestone`` is a 'delayed' spec whose anchor is
    stamped by the real β sweep (``Scheduler._stamp_milestone_deps_satisfied``)
    once ``dep_id`` is done.
    """
    before_done: dict = {
        'script': str(script_path),
        'args': args if args is not None else [
            '--window-days', '7', '--threshold', '0.05', '--value', '0.03',
        ],
        'env': {},
        'cwd': None,
        'timeout_secs': timeout_secs,
        'target_unit': None,
        'kind': 'predicate',
    }
    metadata: dict = {
        'task_kind': 'deterministic',
        'always_escalates': False,
        'before_done': before_done,
        'milestone': {'mode': 'delayed', 'after_secs': after_secs},
    }
    if milestone_deps_satisfied_at is not None:
        metadata['milestone_deps_satisfied_at'] = milestone_deps_satisfied_at
    return {
        'id': task_id,
        'title': 'Merge-flakiness milestone check',
        'description': 'Predicate that verifies the merge-flakiness invariant',
        'status': 'pending',
        'dependencies': [{'id': dep_id}],
        'metadata': metadata,
    }


def _build_scheduler(clock: list[datetime]) -> Scheduler:
    """A real Scheduler with an injected, mutable wall clock and a spied update_task.

    Mirrors TestMilestoneSweepDrivenByAcquireNext: update_task is replaced
    outright (not routed through mcp_call) so the sweep's stamp call can be
    asserted directly, exactly like every β sweep unit test.
    """
    config = OrchestratorConfig(max_per_module=1)
    scheduler = Scheduler(config, wall_time_source=lambda: clock[0])
    scheduler.update_task = AsyncMock(return_value=True)
    return scheduler


def _make_assignment(task: dict) -> TaskAssignment:
    """Build a TaskAssignment directly (bypassing acquire_next) for tests
    that exercise the runner half in isolation — deterministic tasks always
    hold an empty modules list (I4/B12: no module lock)."""
    return TaskAssignment(task_id=str(task['id']), task=task, modules=[])


def _mock_scheduler(task: dict) -> MagicMock:
    """A MagicMock scheduler — the runner's status SINK (mirrors every γ unit test)."""
    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_task = AsyncMock(return_value=task)
    return scheduler


def _real_runner(tmp_path: Path, task: dict, unit_inspector) -> DeterministicRunner:
    """A real DeterministicRunner: script_runner=None -> the REAL subprocess path."""
    queue = EscalationQueue(tmp_path)
    scheduler = _mock_scheduler(task)
    return DeterministicRunner(
        scheduler=scheduler,
        escalation_queue=queue,
        unit_inspector=unit_inspector,
        script_runner=None,
    )


# ---------------------------------------------------------------------------
# B... — exemplar check-script contract (self-authored: ε owns both the
# script and this test — no external numeric premise).
# ---------------------------------------------------------------------------


class TestExemplarCheckScript:
    """Contract test for the ε exemplar predicate: scripts/check_merge_flakiness.sh.

    RED until step-2 authors the script: pytest.fail on the missing-script
    sentinel (the repo_root fixture's documented contract — the .git sentinel
    exists but a required file within the repo is absent, so this must not
    silently skip).
    """

    SCRIPT_REL = 'scripts/check_merge_flakiness.sh'

    def _script_path(self, repo_root: Path) -> Path:
        script = repo_root / self.SCRIPT_REL
        if not script.exists():
            pytest.fail(
                f'{self.SCRIPT_REL} does not exist at {script} — the ε exemplar '
                f'predicate script has not been authored yet'
            )
        return script

    def test_script_exists_and_is_executable(self, repo_root: Path | None):
        if repo_root is None:
            pytest.skip('not running inside a git checkout')
        script = self._script_path(repo_root)
        assert os.access(script, os.X_OK), f'{script} is not executable (missing +x bit)'

    def test_script_exits_0_and_reports_holds_when_value_below_threshold(
        self, repo_root: Path | None,
    ):
        if repo_root is None:
            pytest.skip('not running inside a git checkout')
        script = self._script_path(repo_root)
        result = subprocess.run(
            [str(script), '--window-days', '7', '--threshold', '0.05', '--value', '0.03'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, (
            f'expected rc=0 (invariant holds); got rc={result.returncode}, '
            f'stdout={result.stdout!r}, stderr={result.stderr!r}'
        )
        tail = result.stdout.strip()
        assert 'holds' in tail, f'expected "holds" in stdout tail; got {tail!r}'

    def test_script_exits_1_and_reports_violated_when_value_at_or_above_threshold(
        self, repo_root: Path | None,
    ):
        if repo_root is None:
            pytest.skip('not running inside a git checkout')
        script = self._script_path(repo_root)
        result = subprocess.run(
            [str(script), '--value', '0.08', '--threshold', '0.05'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1, (
            f'expected rc=1 (invariant VIOLATED); got rc={result.returncode}, '
            f'stdout={result.stdout!r}, stderr={result.stderr!r}'
        )
        tail = result.stdout.strip()
        assert 'VIOLATED' in tail, f'expected "VIOLATED" in stdout tail; got {tail!r}'


# ---------------------------------------------------------------------------
# B2+B3+B7+B10 — exemplar PASS lifecycle, end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExemplarPassLifecycle:
    """A real Scheduler (injected wall clock) drives the GATE half via
    acquire_next() across multiple ticks — withhold (dep unsatisfied) ->
    stamp the deps-satisfied anchor (B2) -> withhold (timer not yet elapsed,
    B3) -> release — then hands the released TaskAssignment to a real
    DeterministicRunner (script_runner=None) running the REAL
    scripts/check_merge_flakiness.sh, asserting the done-provenance verdict
    (B7) and that no systemd unit_inspector call is ever made (B10).
    """

    AFTER_SECS = 100
    TASK_ID = '9001'

    async def test_delayed_predicate_milestone_full_pass_lifecycle(
        self, repo_root: Path | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        if repo_root is None:
            pytest.skip('not running inside a git checkout')
        script = repo_root / 'scripts' / 'check_merge_flakiness.sh'
        if not script.exists():
            pytest.fail(f'scripts/check_merge_flakiness.sh does not exist at {script}')

        anchor_base = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)
        clock = [anchor_base]

        milestone_task = _delayed_predicate_milestone_task(
            self.TASK_ID, script, after_secs=self.AFTER_SECS,
        )
        # 'in-progress', not 'pending': isolates the milestone task's gate
        # behaviour from the dep task's OWN dispatch eligibility (mirrors
        # TestMilestoneSweepDrivenByAcquireNext's dep_task convention).
        dep_task = _dep_task('X', status='in-progress')

        mock_call = AsyncMock(return_value=_task_response([dep_task, milestone_task]))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_call)

        scheduler = _build_scheduler(clock)

        # Tick 1: dep X still pending -> withheld, no anchor stamped.
        result_1 = await scheduler.acquire_next()
        assert result_1 is None, 'must withhold while dep X is unsatisfied'
        scheduler.update_task.assert_not_awaited()  # type: ignore[attr-defined]

        # Flip dep X -> done; tick 2 stamps the anchor exactly once (B2) but
        # STILL withholds this tick — the stamp is only visible on the NEXT
        # tick's get_tasks (mirrors TestMilestoneSweepDrivenByAcquireNext).
        dep_task['status'] = 'done'
        mock_call.return_value = _task_response([dep_task, milestone_task])
        result_2 = await scheduler.acquire_next()
        assert result_2 is None, 'anchor just stamped is not visible until next tick'
        scheduler.update_task.assert_awaited_once()  # type: ignore[attr-defined]
        stamp_call = scheduler.update_task.call_args  # type: ignore[attr-defined]
        assert stamp_call.args[0] == self.TASK_ID
        anchor_iso = stamp_call.args[1]['milestone_deps_satisfied_at']
        assert stamp_call.kwargs.get('metadata_mode') == 'merge'
        assert anchor_iso == clock[0].isoformat()

        # Write the stamped anchor back into the task dict for later ticks.
        milestone_task['metadata']['milestone_deps_satisfied_at'] = anchor_iso
        mock_call.return_value = _task_response([dep_task, milestone_task])

        # Tick 3: timer not yet elapsed -> withheld (B3).
        clock[0] = anchor_base + timedelta(seconds=self.AFTER_SECS - 1)
        result_3 = await scheduler.acquire_next()
        assert result_3 is None, 'must withhold before anchor + after_secs elapses'

        # Tick 4: timer elapsed, dep still done -> dispatched.
        clock[0] = anchor_base + timedelta(seconds=self.AFTER_SECS)
        assignment = await scheduler.acquire_next()
        assert assignment is not None, 'must dispatch once the delayed timer elapses'
        assert assignment.task_id == self.TASK_ID
        assert assignment.modules == [], 'deterministic tasks hold no module lock (I4/B12)'

        # Hand the released assignment to a REAL DeterministicRunner running
        # the REAL exemplar script (script_runner=None).
        unit_inspector = AsyncMock()
        runner = _real_runner(tmp_path, assignment.task, unit_inspector)

        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.DONE

        runner.scheduler.set_task_status.assert_awaited_once()
        done_call = runner.scheduler.set_task_status.call_args
        assert done_call.args[0] == self.TASK_ID
        assert done_call.args[1] == 'done'
        provenance = done_call.kwargs.get('done_provenance')
        assert provenance is not None, 'done_provenance must be passed as a kwarg'
        assert provenance['kind'] == 'deterministic-milestone'
        assert 'invariant holds' in provenance.get('note', ''), (
            f'stdout tail must appear in provenance note: {provenance!r}'
        )

        unit_inspector.assert_not_awaited()


# ---------------------------------------------------------------------------
# B8+B9 — exemplar FAIL verdict and INFRA-fault paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExemplarFailAndTimeout:
    """B8: the REAL exemplar script's rc!=0 VIOLATED verdict end-to-end.

    B9: an INFRA fault (no verdict produced) routes to infra_issue, never
    milestone_check_failed.  γ's outer ``asyncio.wait_for`` guard is
    ``before_done['timeout_secs'] + run_timeout_grace_secs`` — a STRICT
    superset of ``_default_run_script``'s own inner per-subprocess timeout,
    which always resolves first for a real, SIGKILL-able subprocess and
    returns a normal ``(rc=1, tail)`` verdict (the *intended* behaviour:
    ``run_timeout_grace_secs`` is documented as "a pure safety margin ON TOP
    of before_done['timeout_secs']", not a race to win — see
    deterministic_runner.py's ``_RUN_TIMEOUT_GRACE_SECS`` comment).  A real
    subprocess timing out therefore exercises B8 (milestone_check_failed),
    not B9 — reaching the outer guard needs a leaf that never returns at
    all, so B9 is exercised with an injected hanging ``script_runner``,
    exactly mirroring test_deterministic_runner.py's own
    ``TestPredicateModeTimeout`` unit test.  This still drives the REAL,
    unmocked ``_run_predicate``/``run()`` composition — only the leaf
    ``run_fn`` callable differs from the exemplar-script path.
    """

    TASK_ID_FAIL = '9002'
    TASK_ID_TIMEOUT = '9003'

    async def test_check_fails_files_milestone_check_failed_and_blocks(
        self, repo_root: Path | None, tmp_path: Path,
    ):
        if repo_root is None:
            pytest.skip('not running inside a git checkout')
        script = repo_root / 'scripts' / 'check_merge_flakiness.sh'
        if not script.exists():
            pytest.fail(f'scripts/check_merge_flakiness.sh does not exist at {script}')

        task = _delayed_predicate_milestone_task(
            self.TASK_ID_FAIL, script,
            args=['--threshold', '0.05', '--value', '0.08'],
        )
        assignment = _make_assignment(task)
        unit_inspector = AsyncMock()
        runner = _real_runner(tmp_path, task, unit_inspector)

        outcome = await runner.run(assignment)

        assert outcome == WorkflowOutcome.BLOCKED

        pending = runner.escalation_queue.get_by_task(self.TASK_ID_FAIL, status='pending')
        assert len(pending) == 1, f'expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.category == 'milestone_check_failed'
        assert 'rc=1' in esc.detail, f'rc must appear in detail: {esc.detail!r}'
        assert 'VIOLATED' in esc.detail, f'stdout tail must appear in detail: {esc.detail!r}'

        runner.scheduler.set_task_status.assert_awaited_once_with(self.TASK_ID_FAIL, 'blocked')

        runner.scheduler.update_task.assert_awaited_once()
        stamp_call = runner.scheduler.update_task.call_args
        metadata_update = (
            stamp_call.args[1] if stamp_call.args else stamp_call.kwargs.get('metadata', {})
        )
        assert metadata_update.get('gate_escalated_at'), (
            'gate_escalated_at should be a truthy ISO timestamp'
        )

        unit_inspector.assert_not_awaited()

    async def test_predicate_hang_files_infra_issue_not_milestone_check_failed(
        self, tmp_path: Path,
    ):
        """A hung check produces NO exit code — no verdict — so this is an
        INFRA fault: infra_issue, never milestone_check_failed, and
        gate_escalated_at is never stamped (re-attempted on the next
        dispatch rather than latched into the resolve-to-done path)."""
        task = _delayed_predicate_milestone_task(
            self.TASK_ID_TIMEOUT, Path('/nonexistent/unused.sh'), timeout_secs=0,
        )
        assignment = _make_assignment(task)
        queue = EscalationQueue(tmp_path)
        scheduler = _mock_scheduler(task)
        unit_inspector = AsyncMock()

        async def _hang(_before_done):
            await asyncio.Event().wait()

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

        pending = queue.get_by_task(self.TASK_ID_TIMEOUT, status='pending')
        assert len(pending) == 1, f'expected exactly 1 pending escalation, got {len(pending)}'
        esc = pending[0]
        assert esc.category == 'infra_issue', (
            f'a hung check is an infra fault, not a verdict — must not be '
            f'milestone_check_failed: {esc.category!r}'
        )

        scheduler.set_task_status.assert_awaited_once_with(self.TASK_ID_TIMEOUT, 'blocked')
        scheduler.update_task.assert_not_awaited()
        unit_inspector.assert_not_awaited()
