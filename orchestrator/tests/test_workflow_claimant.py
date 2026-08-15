"""Tests for TaskWorkflow claimant_run_id/heartbeat_at dispatch stamp + heartbeat loop.

PRD plans/task-status-authority-prd.md contract C4/D4 (task 2188, omega1).

Covers:
  - step-13/14: the pending->in-progress dispatch write (inside
    _setup_worktree_and_artifacts) stamps claimant_run_id (composed from the
    process run_id, workflow session_id, and os.getpid()) and heartbeat_at
    (now, UTC ISO-8601) atomically with the status write.
  - step-15/16: a background heartbeat loop refreshes heartbeat_at on a
    bounded config cadence (claimant_run_id untouched — refresh, not
    restamp), and is stopped cleanly via _stop_claimant_heartbeat (the
    helper run()'s finally calls).
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from escalation.pins import PinReport, classify_pins
from shared.task_claimant import compose_claimant_run_id

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import WorktreeInfo
from orchestrator.task_ground_truth import ClaimantSource
from orchestrator.workflow import TaskWorkflow


@dataclasses.dataclass(frozen=True)
class _FilingRec:
    """A minimal ``escalation.pins.PinRecord`` double (task 3563).

    Mirrors ``escalation/tests/test_pins.py``'s ``_Rec`` — keeps these tests
    independent of ``Escalation``'s much wider constructor.
    """

    id: str
    level: int
    severity: str
    filing_claimant_run_id: str | None


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
    run_id: str | None = 'run-abc123',
    claimant_heartbeat_interval_secs: float = 60.0,
) -> TaskWorkflow:
    """Return a minimal TaskWorkflow with scheduler/git_ops mocked."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = project_root
    config.claimant_heartbeat_interval_secs = claimant_heartbeat_interval_secs

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock(return_value=None)
    scheduler.set_task_claimant = AsyncMock(return_value=None)
    scheduler.update_task = AsyncMock(return_value=True)
    # _setup_worktree_and_artifacts reads live status at dispatch before
    # claiming 'in-progress'; None -> non-terminal -> proceed.
    scheduler.get_status = AsyncMock(return_value=None)

    git_ops = MagicMock()
    git_ops.worktree_base = project_root / '.worktrees'

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        run_id=run_id,
    )
    return wf


async def _setup(wf: TaskWorkflow, base_sha: str = 'a' * 40) -> None:
    """Drive the create_worktree path of _setup_worktree_and_artifacts."""
    wf.git_ops.create_worktree = AsyncMock(
        return_value=WorktreeInfo(path=wf.config.project_root, base_commit=base_sha, stale_commits=0)
    )
    wf._sync_worktree_venvs = AsyncMock()
    with patch('orchestrator.workflow._run', new=AsyncMock(return_value=(0, '', ''))):
        await wf._setup_worktree_and_artifacts('task/101')


# ---------------------------------------------------------------------------
# step-13/14: dispatch stamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_stamps_claimant_run_id_and_heartbeat(tmp_path: Path):
    """The pending->in-progress write stamps claimant_run_id + heartbeat_at."""
    wf = _make_workflow(project_root=tmp_path, task_id='101', run_id='run-abc123')

    fixed_now = datetime(2026, 7, 8, 12, 0, 0, tzinfo=UTC)
    with patch('orchestrator.workflow.datetime') as mock_datetime:
        mock_datetime.now.return_value = fixed_now
        await _setup(wf)

    expected_claimant = compose_claimant_run_id('run-abc123', wf.session_id, os.getpid())
    wf.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
        '101', 'in-progress',
        claimant_run_id=expected_claimant,
        heartbeat_at=fixed_now.isoformat(),
    )

    # _setup started the background heartbeat loop; stop it so the test does
    # not leak a pending task (run()'s finally owns this in production).
    await wf._stop_claimant_heartbeat()


@pytest.mark.asyncio
async def test_dispatch_stamp_embeds_run_id_session_id_and_pid(tmp_path: Path):
    """claimant_run_id verbatim-embeds run_id, session_id, and os.getpid()."""
    wf = _make_workflow(project_root=tmp_path, task_id='202', run_id='run-xyz789')

    await _setup(wf)

    kwargs = wf.scheduler.set_task_status.call_args.kwargs  # type: ignore[attr-defined]
    claimant_run_id = kwargs['claimant_run_id']
    assert 'run-xyz789' in claimant_run_id
    assert wf.session_id in claimant_run_id
    assert f'pid={os.getpid()}' in claimant_run_id

    # _setup started the background heartbeat loop; stop it so the test does
    # not leak a pending task (run()'s finally owns this in production).
    await wf._stop_claimant_heartbeat()


# ---------------------------------------------------------------------------
# step-15/16: heartbeat loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_loop_starts_after_dispatch_stamp(tmp_path: Path):
    """A background heartbeat task is running immediately after dispatch."""
    wf = _make_workflow(project_root=tmp_path, claimant_heartbeat_interval_secs=0.01)

    await _setup(wf)

    assert wf._claimant_heartbeat_task is not None
    assert not wf._claimant_heartbeat_task.done()

    await wf._stop_claimant_heartbeat()


@pytest.mark.asyncio
async def test_heartbeat_loop_refreshes_heartbeat_only(tmp_path: Path):
    """The loop calls scheduler.set_task_claimant(task_id, heartbeat_at=...)
    without claimant_run_id (refresh, not restamp)."""
    wf = _make_workflow(project_root=tmp_path, task_id='303', claimant_heartbeat_interval_secs=0.01)

    await _setup(wf)
    # Let the loop tick at least once (interval=0.01s).
    await asyncio.sleep(0.05)
    await wf._stop_claimant_heartbeat()

    assert wf.scheduler.set_task_claimant.await_count >= 1  # type: ignore[attr-defined]
    args, kwargs = wf.scheduler.set_task_claimant.call_args  # type: ignore[attr-defined]
    assert args == ('303',)
    assert 'claimant_run_id' not in kwargs
    assert 'heartbeat_at' in kwargs


@pytest.mark.asyncio
async def test_stop_claimant_heartbeat_halts_further_refreshes(tmp_path: Path):
    """After _stop_claimant_heartbeat, no further set_task_claimant calls occur."""
    wf = _make_workflow(project_root=tmp_path, claimant_heartbeat_interval_secs=0.01)

    await _setup(wf)
    await asyncio.sleep(0.05)
    await wf._stop_claimant_heartbeat()
    count_after_stop = wf.scheduler.set_task_claimant.await_count  # type: ignore[attr-defined]

    # Give the (now-cancelled) loop plenty of time to have ticked again if it
    # were still alive.
    await asyncio.sleep(0.05)

    assert wf.scheduler.set_task_claimant.await_count == count_after_stop  # type: ignore[attr-defined]
    assert wf._claimant_heartbeat_task is None or wf._claimant_heartbeat_task.done()


@pytest.mark.asyncio
async def test_stop_claimant_heartbeat_is_idempotent_noop_when_never_started(tmp_path: Path):
    """Calling _stop_claimant_heartbeat before dispatch (no task started) must not raise."""
    wf = _make_workflow(project_root=tmp_path)
    assert wf._claimant_heartbeat_task is None

    await wf._stop_claimant_heartbeat()  # must not raise


@pytest.mark.asyncio
async def test_claimant_heartbeat_loop_exits_cleanly_on_nonnumeric_interval(tmp_path: Path):
    """A non-numeric (MagicMock) interval makes the loop exit cleanly, not crash.

    This is the fully-mocked-config case (task 2780 sub-mode a1): a MagicMock
    ``claimant_heartbeat_interval_secs`` reaches ``asyncio.sleep(<MagicMock>)``,
    whose ``if delay <= 0`` evaluates ``MagicMock() <= 0`` — and ``MagicMock``'s
    comparison dunders (``__le__``) default to ``NotImplemented``, so Python
    raises ``TypeError: '<=' not supported between instances of MagicMock and
    int``. RED before the step-4 guard: that TypeError propagates out of the
    loop, out of ``wait_for``, and fails this test. GREEN after the guard: the
    loop reads the interval once, sees it is not a positive real number, logs a
    WARNING, and returns without ever entering the sleep loop or refreshing the
    claimant.
    """
    wf = _make_workflow(project_root=tmp_path)
    wf.config.claimant_heartbeat_interval_secs = MagicMock()  # non-numeric interval

    # Must return cleanly (no TypeError, no hang) — bounded so a regression that
    # re-enters the sleep loop fails fast on timeout rather than hanging.
    await asyncio.wait_for(wf._claimant_heartbeat_loop(), timeout=1.0)

    # The guard returns before the first refresh, so the claimant is never touched.
    wf.scheduler.set_task_claimant.assert_not_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# leaked-heartbeat-loop reaper (task 2780)
#
# Mirrors the established leaked-async-task reaper family:
#   - reap_leaked_ticket_workers  (task 2737, fused-memory _fm_helpers.py)
#   - reap_leaked_aiosqlite_connections (task 2413, _orch_helpers.py)
#   - _reap_leaked_merge_workers  (task 1907, this project's conftest.py)
#
# A co-scheduled TaskWorkflow test that raises before its own inline
# _stop_claimant_heartbeat (or never starts one) orphans the background
# _claimant_heartbeat_loop task onto the shared xdist-worker loop, where it is
# destroyed-while-pending (or, under a fully-mocked config, raises an
# un-retrieved TypeError) and pollutes a later innocent test.
# ---------------------------------------------------------------------------


def _live_claimant_heartbeat_tasks() -> list[asyncio.Task]:
    """Return non-done asyncio Tasks whose coroutine is TaskWorkflow._claimant_heartbeat_loop."""
    return [
        t
        for t in asyncio.all_tasks()
        if not t.done()
        and getattr(t.get_coro(), '__qualname__', '').endswith(
            'TaskWorkflow._claimant_heartbeat_loop'
        )
    ]


@pytest.mark.asyncio
async def test_reap_leaked_claimant_heartbeats_cancels_orphaned_loop(tmp_path: Path):
    """reap_leaked_claimant_heartbeats() cancels + drains an orphaned heartbeat loop.

    Starts a leaked _claimant_heartbeat_loop task directly (the interval
    defaults to a valid 60.0, so the loop starts and blocks on its first
    asyncio.sleep) and deliberately does NOT call _stop_claimant_heartbeat —
    the point is that the reaper alone can halt the orphaned loop, exactly as
    it must for a test that crashed before its own cleanup ran.

    The import is the FIRST statement so an ImportError (RED, until the helper
    exists in step-2) fails the test before any task is created and thus cannot
    itself leak a heartbeat loop.
    """
    from _orch_helpers import reap_leaked_claimant_heartbeats

    wf = _make_workflow(project_root=tmp_path)
    task = asyncio.create_task(wf._claimant_heartbeat_loop())
    await asyncio.sleep(0)  # let the loop start and block on asyncio.sleep(60.0)

    assert _live_claimant_heartbeat_tasks(), (
        'Precondition: expected a live TaskWorkflow._claimant_heartbeat_loop '
        'task after create_task'
    )

    reaped = await reap_leaked_claimant_heartbeats()

    assert reaped >= 1, (
        f'Expected reap_leaked_claimant_heartbeats() to reap >= 1, got {reaped}'
    )
    assert _live_claimant_heartbeat_tasks() == [], (
        'No live TaskWorkflow._claimant_heartbeat_loop task should remain after reaping'
    )
    assert task.done()


@pytest.mark.asyncio
async def test_reap_leaked_claimant_heartbeats_noop_when_nothing_leaked():
    """reap_leaked_claimant_heartbeats() returns 0 when nothing was leaked.

    Pins the cheap no-op path: this test's own runner task (and any other live
    task) does not match the '...TaskWorkflow._claimant_heartbeat_loop' qualname
    filter.
    """
    from _orch_helpers import reap_leaked_claimant_heartbeats

    assert await reap_leaked_claimant_heartbeats() == 0


# ---------------------------------------------------------------------------
# task 3563: the plan.lock identity matches the DB stamp for one incarnation
# ---------------------------------------------------------------------------


def _resolve_plan_lock_claimant(wf: TaskWorkflow):
    """Resolve *wf*'s plan.lock through the REAL TaskGroundTruth resolver.

    Uses the claimant-absent task shape, which is the ONLY shape that reaches
    the plan.lock leg (a present db claimant wins outright).

    ``.task`` COMPAT SHIM: ``TaskWorkflow`` builds its artifacts with
    ``meta_root=_meta_root_for_worktree(...)``, so plan.lock lives in the
    ``.task-meta`` SIBLING, whereas ``_resolve_live_claimant`` constructs a
    bare ``TaskArtifacts(worktree_path)`` and therefore reads
    ``<worktree>/.task/plan.lock``.  The symlink below reproduces the in-repo
    ``.task`` -> ``.task-meta`` compat shape so the resolver reads the lock
    the workflow actually wrote.  That divergence is a PRE-EXISTING gap in the
    resolver, unrelated to and untouched by task 3563 (which normalises the
    identity SHAPE, not where the lock is looked up) — filed separately.
    """
    from orchestrator.task_ground_truth import TaskGroundTruth

    assert wf.artifacts is not None and wf.worktree is not None
    worktree = Path(wf.worktree)
    legacy_lock = worktree / '.task' / 'plan.lock'
    real_lock = wf.artifacts.root / 'plan.lock'
    if real_lock != legacy_lock and not legacy_lock.exists():
        legacy_lock.parent.mkdir(parents=True, exist_ok=True)
        legacy_lock.symlink_to(real_lock)

    scheduler = MagicMock()
    # Not held in memory — signal 1 short-circuits to IN_MEMORY otherwise, and
    # a bare MagicMock return is truthy.
    scheduler.is_actively_held = MagicMock(return_value=False)
    resolver = TaskGroundTruth(
        MagicMock(), scheduler, None, lambda _tid: worktree,
    )
    return resolver._resolve_live_claimant(
        '303',
        {'status': 'pending', 'claimant_run_id': None, 'heartbeat_at': None},
        worktree,
    )


async def _acquire_plan_lock_via_workflow(wf: TaskWorkflow) -> None:
    """Drive a real production plan-lock acquisition on *wf*.

    Goes through ``_apply_revalidation_skip`` — the most self-contained of the
    four ``lock_plan`` call sites — rather than calling ``lock_plan`` directly,
    so the test proves the PRODUCTION wiring passes the run id, not merely that
    ``lock_plan`` can accept one.
    """
    from orchestrator.workflow import WorkflowOutcome

    assert wf.artifacts is not None
    wf.artifacts.write_plan({'steps': [{'id': 'step-1', 'type': 'impl'}], 'files': []})
    wf._reconcile_scope_locks = AsyncMock(return_value=True)
    wf._stamp_optimistic_path = AsyncMock()

    outcome = await wf._apply_revalidation_skip({'files': []}, 'b' * 40)

    assert outcome is WorkflowOutcome.PLANNED, (
        'precondition: the revalidation-skip path must have acquired the lock'
    )
    assert wf.artifacts.is_plan_locked()


@pytest.mark.asyncio
async def test_plan_lock_identity_is_byte_identical_to_the_db_stamp(tmp_path: Path):
    """One incarnation's plan.lock and DB claimant identities are the SAME string.

    This is the payoff of task 3563.  Before it, the plan.lock claimant source
    yielded a BARE session_id while the DB source yielded a composed identity,
    so the two could never be compared — ``escalation.pins`` had to reject the
    plan.lock shape outright rather than risk converting a live filer's L0.
    """
    wf = _make_workflow(project_root=tmp_path, task_id='303', run_id='run-abc123')
    await _setup(wf)
    await wf._stop_claimant_heartbeat()

    db_identity = wf.scheduler.set_task_status.call_args.kwargs['claimant_run_id']  # type: ignore[attr-defined]
    await _acquire_plan_lock_via_workflow(wf)

    claimant = _resolve_plan_lock_claimant(wf)

    assert claimant is not None
    assert claimant.source == ClaimantSource.PLAN_LOCK
    # The single string equality that makes discrimination work at all.
    assert claimant.run_id == db_identity
    assert claimant.run_id == compose_claimant_run_id(
        'run-abc123', wf.session_id, os.getpid(),
    )


@pytest.mark.asyncio
async def test_harness_less_workflow_degrades_to_unknown_not_a_partial_identity(
    tmp_path: Path,
):
    """No process run id => run_id is None (UNKNOWN), never '/{sess}/pid={pid}'.

    A partially-composed identity would carry the ``/pid=`` marker, survive
    ``escalation.pins._norm_id``'s shape guard, and then mismatch every
    DB-composed filing identity — read downstream as "a DIFFERENT incarnation
    is live".  The fail-safe unknown is the whole point of omitting the key.
    """
    wf = _make_workflow(project_root=tmp_path, task_id='303', run_id=None)
    await _setup(wf)
    await wf._stop_claimant_heartbeat()

    await _acquire_plan_lock_via_workflow(wf)

    claimant = _resolve_plan_lock_claimant(wf)

    assert claimant is not None
    assert claimant.run_id is None
    assert claimant.session_id == wf.session_id


@pytest.mark.asyncio
async def test_resolved_plan_lock_identity_discriminates_live_from_dead_filers(
    tmp_path: Path,
):
    """The resolved identity drives REAL classify_pins discrimination.

    Previously unreachable from a plan.lock claimant: with a bare session id
    (or None) the shape guard collapsed the comparison to "unknown", so BOTH
    cases fell back to ``queue_handoff`` and a dead incarnation's L0 pinned the
    task forever.
    """
    wf = _make_workflow(project_root=tmp_path, task_id='303', run_id='run-abc123')
    await _setup(wf)
    await wf._stop_claimant_heartbeat()
    await _acquire_plan_lock_via_workflow(wf)

    claimant = _resolve_plan_lock_claimant(wf)
    assert claimant is not None
    live_id = claimant.run_id
    assert live_id is not None
    # A DIFFERENT incarnation of the same session — differing pid only.
    dead_id = compose_claimant_run_id('run-abc123', wf.session_id, os.getpid() + 1)
    assert dead_id != live_id

    def _classify(filing: str) -> PinReport:
        return classify_pins(
            '303',
            [_FilingRec(id='esc-303-1', level=0, severity='blocking',
                        filing_claimant_run_id=filing)],
            live_claimant=True,
            live_claimant_id=live_id,
        )

    # The live filer still pins — its handoff has a consumer.
    assert _classify(live_id).queue_handoff == ('esc-303-1',)
    # A dead incarnation's L0 does not — nobody is left to consume it.
    assert _classify(dead_id).dead_l0 == ('esc-303-1',)
