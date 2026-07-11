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
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.deterministic_runner import DeterministicRunner
from orchestrator.scheduler import Scheduler, TaskAssignment
from orchestrator.workflow import WorkflowOutcome

SCRIPT_REL = 'scripts/check_merge_flakiness.sh'

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


def _dated_milestone_task(task_id: str, at: datetime, *, task_kind: str | None = None) -> dict:
    """Build a pending, dated-milestone task dict (no dependencies — the
    'dated' mode gates purely on wall-clock, no deps-satisfied anchor)."""
    metadata: dict = {'milestone': {'mode': 'dated', 'at': at.isoformat()}}
    if task_kind is not None:
        metadata['task_kind'] = task_kind
    return {
        'id': task_id,
        'title': 'Dated milestone task',
        'status': 'pending',
        'dependencies': [],
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
    # Task 2235: acquire_next() now asserts finish_startup() has run (the
    # runtime-enforced 'startup sweeps run before the first tick' invariant).
    # Test schedulers built directly (bypassing Harness's startup sweeps)
    # must call it explicitly.
    scheduler.finish_startup()
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


@pytest.fixture
def exemplar_script(repo_root: Path | None) -> Path:
    """Resolve the real exemplar predicate script, shared by every test that
    executes the REAL scripts/check_merge_flakiness.sh subprocess
    (TestExemplarCheckScript, TestExemplarPassLifecycle,
    TestExemplarFailAndTimeout).

    Skips when not running inside a git checkout (repo_root is None); fails
    — does not skip — when the .git sentinel exists but the script itself is
    absent (the repo_root fixture's documented contract), so a genuinely
    missing file cannot silently hide behind a skip.
    """
    if repo_root is None:
        pytest.skip('not running inside a git checkout')
    script = repo_root / SCRIPT_REL
    if not script.exists():
        pytest.fail(
            f'{SCRIPT_REL} does not exist at {script} — the ε exemplar '
            f'predicate script has not been authored yet'
        )
    return script


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

    def test_script_exists_and_is_executable(self, exemplar_script: Path):
        assert os.access(exemplar_script, os.X_OK), (
            f'{exemplar_script} is not executable (missing +x bit)'
        )

    def test_script_exits_0_and_reports_holds_when_value_below_threshold(
        self, exemplar_script: Path,
    ):
        result = subprocess.run(
            [str(exemplar_script), '--window-days', '7', '--threshold', '0.05', '--value', '0.03'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, (
            f'expected rc=0 (invariant holds); got rc={result.returncode}, '
            f'stdout={result.stdout!r}, stderr={result.stderr!r}'
        )
        tail = result.stdout.strip()
        assert 'holds' in tail, f'expected "holds" in stdout tail; got {tail!r}'

    def test_script_exits_1_and_reports_violated_when_value_at_or_above_threshold(
        self, exemplar_script: Path,
    ):
        result = subprocess.run(
            [str(exemplar_script), '--value', '0.08', '--threshold', '0.05'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1, (
            f'expected rc=1 (invariant VIOLATED); got rc={result.returncode}, '
            f'stdout={result.stdout!r}, stderr={result.stderr!r}'
        )
        tail = result.stdout.strip()
        assert 'VIOLATED' in tail, f'expected "VIOLATED" in stdout tail; got {tail!r}'

    def test_script_exits_2_when_flag_value_is_missing(self, exemplar_script: Path):
        """A flag passed as the final argument with no value must be a
        deterministic usage error (rc=2) — not a `shift`-induced rc=1 that
        would be misread as the 'invariant VIOLATED' verdict."""
        result = subprocess.run(
            [str(exemplar_script), '--threshold', '0.05', '--value'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 2, (
            f'expected rc=2 (usage error) for a missing flag value; got '
            f'rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}'
        )

    def test_script_exits_2_on_unknown_argument(self, exemplar_script: Path):
        result = subprocess.run(
            [str(exemplar_script), '--bogus-flag', '1'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 2, (
            f'expected rc=2 (usage error) for an unknown argument; got '
            f'rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}'
        )

    def test_script_exits_2_on_non_numeric_value(self, exemplar_script: Path):
        """Garbage --value must be rejected explicitly, not silently
        coerced to 0 by awk (which would misreport 'invariant holds')."""
        result = subprocess.run(
            [str(exemplar_script), '--threshold', '0.05', '--value', 'garbage'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 2, (
            f'expected rc=2 (usage error) for a non-numeric --value; got '
            f'rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}'
        )

    def test_script_accepts_leading_and_trailing_dot_numeric_values(
        self, exemplar_script: Path,
    ):
        """NUMERIC_RE accepts leading-dot (.03) and trailing-dot (5.) decimal
        forms in addition to the canonical 0.NN form."""
        result = subprocess.run(
            [str(exemplar_script), '--value', '.03', '--threshold', '5.'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, (
            f'leading-dot --value and trailing-dot --threshold must both be '
            f'accepted as numeric; got rc={result.returncode}, '
            f'stdout={result.stdout!r}, stderr={result.stderr!r}'
        )
        assert 'holds' in result.stdout.strip()

    def test_script_rejects_scientific_notation_value(self, exemplar_script: Path):
        """Scientific notation (1e-3) is intentionally NOT accepted by
        NUMERIC_RE — documents the boundary rather than silently mishandling
        it, in case this exemplar is ever fed a real deployment's output."""
        result = subprocess.run(
            [str(exemplar_script), '--value', '1e-3', '--threshold', '0.05'],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 2, (
            f'scientific notation must be rejected as non-numeric (rc=2); got '
            f'rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}'
        )

    def test_script_reports_usage_error_when_awk_itself_fails(
        self, exemplar_script: Path, tmp_path: Path,
    ):
        """An awk that fails to execute at all (missing binary, crash, syntax
        error) must not be silently misread as the 'invariant VIOLATED'
        verdict (exit 1) — only a CLEAN awk exit of 0 or 1 is a real verdict;
        any other awk exit status is a usage/infra error (rc=2). Shadow the
        real awk on PATH with a fake one that exits 3 to exercise this
        deterministically, without touching the real awk binary."""
        fake_awk = tmp_path / 'awk'
        fake_awk.write_text('#!/usr/bin/env bash\nexit 3\n')
        fake_awk.chmod(0o755)

        env = dict(os.environ)
        env['PATH'] = f'{tmp_path}{os.pathsep}{env.get("PATH", "")}'

        result = subprocess.run(
            [str(exemplar_script), '--threshold', '0.05', '--value', '0.03'],
            capture_output=True, text=True, timeout=10, env=env,
        )
        assert result.returncode == 2, (
            f'an awk that exits neither 0 nor 1 must be treated as a usage/infra '
            f'error, never misread as a verdict; got rc={result.returncode}, '
            f'stdout={result.stdout!r}, stderr={result.stderr!r}'
        )

    def test_script_sleep_secs_delays_execution(self, exemplar_script: Path):
        """--sleep-secs is the knob that lets this SAME fixture drive a
        check timeout under test; assert it actually delays the verdict."""
        sleep_secs = 0.3
        start = time.monotonic()
        result = subprocess.run(
            [
                str(exemplar_script),
                '--threshold', '0.05', '--value', '0.03', '--sleep-secs', str(sleep_secs),
            ],
            capture_output=True, text=True, timeout=10,
        )
        elapsed = time.monotonic() - start
        assert result.returncode == 0
        assert elapsed >= sleep_secs * 0.9, (
            f'--sleep-secs {sleep_secs} should delay execution by roughly that '
            f'long; elapsed={elapsed:.3f}s'
        )


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
        self, exemplar_script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        anchor_base = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)
        clock = [anchor_base]

        milestone_task = _delayed_predicate_milestone_task(
            self.TASK_ID, exemplar_script, after_secs=self.AFTER_SECS,
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
        self, exemplar_script: Path, tmp_path: Path,
    ):
        task = _delayed_predicate_milestone_task(
            self.TASK_ID_FAIL, exemplar_script,
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


# ---------------------------------------------------------------------------
# B1+B4+B5 — scheduler time-gate matrix via the public acquire_next() surface.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMilestoneSchedulerGate:
    """β's ``_milestone_time_gated`` / ``_stamp_milestone_deps_satisfied`` /
    ``_eligible_for_dispatch`` are already unit-tested directly at their own
    altitude (test_scheduler.py's TestMilestoneTimeGate / TestMilestone­
    EligibilityGate / TestStampMilestoneDepsSatisfied). ε re-exercises the
    same B1/B4/B5 boundaries composed through the ONE public entry point
    production actually calls every tick — ``acquire_next()`` — with a real
    Scheduler (injected wall clock) and mocked ``mcp_call``, mirroring the
    multi-tick harness established by ``TestExemplarPassLifecycle``.
    """

    TASK_ID_DATED = '9101'
    TASK_ID_DELAYED = '9102'

    async def test_dated_milestone_withheld_then_dispatched(self, monkeypatch):
        """B1: acquire_next withholds a dated-milestone task while
        now_wall < at, and dispatches it the instant now_wall >= at."""
        at = datetime(2026, 8, 1, tzinfo=UTC)
        clock = [at - timedelta(seconds=1)]
        task = _dated_milestone_task(self.TASK_ID_DATED, at)

        mock_call = AsyncMock(return_value=_task_response([task]))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_call)
        scheduler = _build_scheduler(clock)

        result_before = await scheduler.acquire_next()
        assert result_before is None, 'must withhold while now_wall < at'

        clock[0] = at
        result_after = await scheduler.acquire_next()
        assert result_after is not None and result_after.task_id == self.TASK_ID_DATED, (
            'must dispatch once now_wall >= at'
        )

    async def test_delayed_anchor_frozen_once_not_restamped_on_second_tick(
        self, monkeypatch,
    ):
        """B4: once the deps-satisfied anchor is stamped, a SECOND
        acquire_next tick (deps still satisfied, timer still running) does
        NOT re-stamp it — update_task stays awaited exactly once across
        both ticks, and the frozen anchor value is unchanged."""
        anchor_base = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)
        clock = [anchor_base]
        after_secs = 100

        milestone_task = _delayed_predicate_milestone_task(
            self.TASK_ID_DELAYED, Path('/unused/for-this-test.sh'),
            after_secs=after_secs,
        )
        dep_task = _dep_task('X', status='done')

        mock_call = AsyncMock(return_value=_task_response([dep_task, milestone_task]))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_call)
        scheduler = _build_scheduler(clock)

        # Tick 1: deps satisfied, no anchor yet -> stamped exactly once.
        result_1 = await scheduler.acquire_next()
        assert result_1 is None, 'anchor just stamped is not visible until next tick'
        scheduler.update_task.assert_awaited_once()  # type: ignore[attr-defined]
        stamp_call = scheduler.update_task.call_args  # type: ignore[attr-defined]
        anchor_iso = stamp_call.args[1]['milestone_deps_satisfied_at']

        # Write the anchor back; advance the clock a little (still well
        # before after_secs elapses) — deps remain satisfied.
        milestone_task['metadata']['milestone_deps_satisfied_at'] = anchor_iso
        mock_call.return_value = _task_response([dep_task, milestone_task])
        clock[0] = anchor_base + timedelta(seconds=1)

        # Tick 2: must NOT re-stamp — the anchor already exists (frozen-once).
        result_2 = await scheduler.acquire_next()
        assert result_2 is None, 'timer has not elapsed yet'
        assert scheduler.update_task.await_count == 1, (  # type: ignore[attr-defined]
            'the anchor must not be re-stamped on a second tick'
        )
        unchanged_call = scheduler.update_task.call_args  # type: ignore[attr-defined]
        assert unchanged_call.args[1]['milestone_deps_satisfied_at'] == anchor_iso, (
            'the frozen anchor value must not change across ticks'
        )

    async def test_dep_regression_after_timer_elapsed_still_withholds(
        self, monkeypatch,
    ):
        """B5: even after the delayed timer has fully elapsed, a dep that
        has regressed away from 'done' in the LIVE status map still blocks
        dispatch through acquire_next — _deps_satisfied is re-checked live
        at eligibility time, not just the frozen-once anchor."""
        anchor = datetime(2026, 7, 25, tzinfo=UTC)
        after_secs = 100
        clock = [anchor + timedelta(seconds=after_secs + 1)]  # timer elapsed

        milestone_task = _delayed_predicate_milestone_task(
            self.TASK_ID_DELAYED, Path('/unused/for-this-test.sh'),
            after_secs=after_secs, milestone_deps_satisfied_at=anchor.isoformat(),
        )
        # 'in-progress', not 'pending': isolates the milestone task's gate
        # from the dep task's OWN dispatch eligibility through acquire_next
        # (mirrors TestExemplarPassLifecycle's dep_task convention) — what
        # matters to _deps_satisfied is that X is anything other than
        # 'done', not the specific non-done value.
        dep_task = _dep_task('X', status='in-progress')

        mock_call = AsyncMock(return_value=_task_response([dep_task, milestone_task]))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_call)
        scheduler = _build_scheduler(clock)

        result = await scheduler.acquire_next()

        assert result is None, 'must withhold: dep X regressed despite the elapsed timer'


# ---------------------------------------------------------------------------
# B12+B6 — normal-agent routing orthogonality and restart survival.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNormalAgentRoutingAndRestartSurvival:
    """B12: the milestone gate is orthogonal to ``task_kind`` — a plain
    ``task_kind='normal'`` milestone task is held/released identically and,
    once released, routes to the NORMAL agent path (``Scheduler.is_deterministic``
    is False), not ``DeterministicRunner``.

    B6: a persisted delayed-milestone anchor survives an orchestrator
    restart — a FRESH ``Scheduler`` instance (no prior in-memory state,
    modelling a new process) recomputes the gate purely from the task's
    persisted ``milestone_deps_satisfied_at`` + ``after_secs``, withheld
    while ``wall_now < anchor + after_secs`` and eligible once
    ``wall_now >= anchor + after_secs`` — the timer is not reset by the
    restart.
    """

    TASK_ID_NORMAL = '9201'
    TASK_ID_RESTART = '9202'

    async def test_normal_task_kind_milestone_gated_then_routes_to_normal_agent(
        self, monkeypatch,
    ):
        at = datetime(2026, 8, 1, tzinfo=UTC)
        clock = [at - timedelta(seconds=1)]
        task = _dated_milestone_task(self.TASK_ID_NORMAL, at, task_kind='normal')

        mock_call = AsyncMock(return_value=_task_response([task]))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_call)
        scheduler = _build_scheduler(clock)

        result_before = await scheduler.acquire_next()
        assert result_before is None, 'must withhold while now_wall < at, same as any milestone task'

        clock[0] = at
        assignment = await scheduler.acquire_next()
        assert assignment is not None and assignment.task_id == self.TASK_ID_NORMAL

        assert Scheduler.is_deterministic(assignment.task) is False, (
            "a task_kind='normal' milestone task must route to the normal "
            'agent path, not DeterministicRunner'
        )

    async def test_restart_survival_fresh_scheduler_recomputes_gate_from_persisted_anchor(
        self, monkeypatch,
    ):
        anchor_base = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)
        after_secs = 100
        clock_a = [anchor_base]

        milestone_task = _delayed_predicate_milestone_task(
            self.TASK_ID_RESTART, Path('/unused/for-this-test.sh'),
            after_secs=after_secs,
        )
        dep_task = _dep_task('X', status='done')

        mock_call = AsyncMock(return_value=_task_response([dep_task, milestone_task]))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_call)

        # Scheduler A stamps the anchor via its acquire_next sweep.
        scheduler_a = _build_scheduler(clock_a)
        result = await scheduler_a.acquire_next()
        assert result is None
        scheduler_a.update_task.assert_awaited_once()  # type: ignore[attr-defined]
        anchor_iso = (
            scheduler_a.update_task.call_args.args[1]['milestone_deps_satisfied_at']  # type: ignore[attr-defined]
        )

        # Persist the anchor into the task dict — the SAME mock_call is
        # reused, so this is what any orchestrator process (old or new)
        # reads back on its next get_tasks.
        milestone_task['metadata']['milestone_deps_satisfied_at'] = anchor_iso
        mock_call.return_value = _task_response([dep_task, milestone_task])

        # Simulate a restart: a BRAND-NEW Scheduler instance — no prior
        # in-memory state — with its own injected wall clock, reading the
        # SAME persisted task dict.
        clock_b = [anchor_base + timedelta(seconds=after_secs - 1)]
        scheduler_b = _build_scheduler(clock_b)

        result_before = await scheduler_b.acquire_next()
        assert result_before is None, (
            'the fresh scheduler must still withhold before anchor + after_secs — '
            'the timer is not reset by the restart'
        )

        clock_b[0] = anchor_base + timedelta(seconds=after_secs)
        result_after = await scheduler_b.acquire_next()
        assert result_after is not None and result_after.task_id == self.TASK_ID_RESTART, (
            'the fresh scheduler must dispatch once wall_now >= anchor + after_secs, '
            'computed purely from the persisted anchor'
        )


# ---------------------------------------------------------------------------
# B11 (importable slice) — shared.task_metadata.Milestone schema rejection.
# ---------------------------------------------------------------------------


class TestMilestoneSchemaRejection:
    """The importable schema-rejection slice of B11.

    ``orchestrator``'s dependency closure is escalation + dark-factory-shared
    only — ``fused_memory`` (where ``deterministic_task_guard._validate_milestone``
    lives) is not importable here.  That guard delegates milestone validation
    to this SAME shared ``Milestone`` model
    (``deterministic_task_guard._validate_milestone`` calls
    ``Milestone(**milestone)``), so asserting that this shared authority
    rejects malformed specs is the honest, importable slice of B11 available
    to ε.  The full guard-rejection path (predicate ``target_unit``/
    ``always_escalates`` prohibitions) is δ's fused-memory unit test — a
    documented boundary, not a gap.
    """

    def test_delayed_without_after_secs_raises(self):
        from pydantic import ValidationError
        from shared.task_metadata import Milestone

        with pytest.raises(ValidationError):
            Milestone(mode='delayed')

    def test_dated_with_unparseable_at_raises(self):
        from pydantic import ValidationError
        from shared.task_metadata import Milestone

        with pytest.raises(ValidationError):
            Milestone(mode='dated', at='not-a-date')
