"""Tests for SpeculativeMergeWorker supervisor (task 1857).

The supervisor detects when _merger_loop or _verifier_loop dies unexpectedly
(unhandled exception while self._running is True), emits a loud L1 escalation,
and restarts the dead loop within a bounded cap.  On cap-exceeded it emits a
born-at-L2 terminal escalation and halts the worker via _loops_finished.

Fixture mirrors: test_merge_queue_restart_hook.py (git_repo/git_config/git_ops/config).
MagicMock escalation_queue: mirrors TestRunDriftCheck._make_fake_escalation_queue in
test_merge_queue_multihost_wiring.py.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    SpeculativeItem,
    SpeculativeMergeWorker,
)

# ---------------------------------------------------------------------------
# Fixtures — mirror test_merge_queue_restart_hook.py
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest with a pending asyncio Future."""
    future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


def _mock_verify_pass() -> AsyncMock:
    """Return a mock that makes run_scoped_verification always pass."""
    return AsyncMock(return_value=type('VR', (), {'passed': True, 'summary': ''})())


def _make_fake_escalation_queue() -> MagicMock:
    """MagicMock escalation queue (mirrors TestRunDriftCheck pattern)."""
    eq = MagicMock()
    eq.has_open_l1 = MagicMock(return_value=False)
    eq.make_id = MagicMock(side_effect=lambda key: f'esc-{key}')
    eq.submit = MagicMock()
    return eq


# ---------------------------------------------------------------------------
# Step 1 — RED: death → loud L1 escalation + new task + resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_death_escalates_loudly_and_restarts(
    git_ops: GitOps,
    config: OrchestratorConfig,
) -> None:
    """Verifier loop death must emit L1 escalation, spawn replacement task, and resume.

    Stub: first invocation crashes immediately with RuntimeError('boom').
    Subsequent invocations idle via `while worker._running: await asyncio.sleep(0.01)`.
    After the death the supervisor must:
      1. emit an Escalation (level==1, severity='blocking') with 'merge_worker_loop_died'
         + 'verifier' in summary and 'RuntimeError'/'boom' in detail, agent_role starts
         with 'orchestrator-';
      2. replace worker._verifier_task with a new, not-done Task;
      3. invoke the stub exactly twice (initial crash + 1 restart that then idles).
    """
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq = _make_fake_escalation_queue()
    worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=eq)
    worker._shutdown_timeout = 0.1

    invocation_count = 0

    async def stub_verifier_loop() -> None:
        nonlocal invocation_count
        invocation_count += 1
        if invocation_count == 1:
            raise RuntimeError('boom')
        # Second+ invocation idles until shutdown
        while worker._running:
            await asyncio.sleep(0.01)

    worker._verifier_loop = stub_verifier_loop  # type: ignore[method-assign]

    run_task = asyncio.create_task(worker.run())

    # Poll up to ~2s for the escalation to appear
    deadline = asyncio.get_event_loop().time() + 2.0
    while eq.submit.call_count == 0 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)

    # ── Assert 1: escalation was submitted once ───────────────────────────
    assert eq.submit.call_count == 1, (
        f'Expected 1 escalation submission, got {eq.submit.call_count}'
    )
    esc = eq.submit.call_args[0][0]
    assert esc.level == 1, f'Expected level 1 escalation, got level={esc.level}'
    assert esc.severity == 'blocking', f'Expected severity blocking, got {esc.severity}'
    assert 'merge_worker_loop_died' in esc.summary, (
        f'summary missing merge_worker_loop_died: {esc.summary!r}'
    )
    assert 'verifier' in esc.summary, (
        f'summary missing loop name "verifier": {esc.summary!r}'
    )
    assert 'RuntimeError' in esc.detail, (
        f'detail missing "RuntimeError": {esc.detail!r}'
    )
    assert 'boom' in esc.detail, f'detail missing "boom": {esc.detail!r}'
    assert esc.agent_role.startswith('orchestrator-'), (
        f'agent_role should start with "orchestrator-": {esc.agent_role!r}'
    )

    # ── Assert 2: replacement task created and running ────────────────────
    # Give the event loop one more pass to let _spawn_loop() fire
    await asyncio.sleep(0.05)
    assert worker._verifier_task is not None, '_verifier_task is None'
    assert not worker._verifier_task.done(), (
        '_verifier_task is done — replacement should still be running'
    )

    # ── Assert 3: stub was called exactly twice ───────────────────────────
    assert invocation_count == 2, (
        f'Expected stub invoked 2 times (crash + restart), got {invocation_count}'
    )

    # ── Cleanup ──────────────────────────────────────────────────────────
    await worker.stop()
    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(run_task, timeout=2.0)


# ---------------------------------------------------------------------------
# Step 3 — RED: bounded restarts → terminal L2 escalation + halt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_restarts_then_terminal_escalation(
    git_ops: GitOps,
    config: OrchestratorConfig,
) -> None:
    """After max_loop_restarts in-window crashes the worker must halt and emit a terminal L2.

    Stub: always raises RuntimeError (with a yield so the event loop can tick).
    Configuration: _max_loop_restarts=3, _loop_restart_window_s=1000 (all in-window).

    Assertions:
      1. run() completes within timeout (no hang — _loops_finished is set on halt).
      2. Stub invoked exactly 4 times (1 initial + 3 restarts).
      3. submit called 4 times: first 3 are level=1 'merge_worker_loop_died';
         4th (terminal) is level=2 severity='critical'.
      4. worker._supervisor_halted is True; _supervisor_halt_reason is set.
      5. run() task completed with exception() is None (clean return, not a crash).
    """
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq = _make_fake_escalation_queue()
    worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=eq)
    worker._shutdown_timeout = 0.1
    worker._max_loop_restarts = 3
    worker._loop_restart_window_s = 1000.0  # all rapid deaths fall in-window

    invocation_count = 0

    async def always_crashing_verifier() -> None:
        nonlocal invocation_count
        invocation_count += 1
        await asyncio.sleep(0)  # yield to pace restarts
        raise RuntimeError(f'verifier crash #{invocation_count}')

    worker._verifier_loop = always_crashing_verifier  # type: ignore[method-assign]

    run_task = asyncio.create_task(worker.run())

    # Wait for run() to complete (supervisor halts after cap exceeded)
    await asyncio.wait_for(run_task, timeout=10.0)

    # 1. run() returned within timeout (asserted by wait_for not raising)

    # 2. Stub invoked exactly 4 times (1 initial + 3 restarts)
    assert invocation_count == 4, (
        f'Expected 4 invocations (1 crash + 3 restarts), got {invocation_count}'
    )

    # 3. Exactly 4 submissions: 3 × L1 restart + 1 × L2 terminal
    assert eq.submit.call_count == 4, (
        f'Expected 4 escalation submissions, got {eq.submit.call_count}'
    )
    restart_escs = [eq.submit.call_args_list[i][0][0] for i in range(3)]
    terminal_esc = eq.submit.call_args_list[3][0][0]
    for esc in restart_escs:
        assert esc.level == 1, f'Restart escalation should be level 1, got {esc.level}'
        assert 'merge_worker_loop_died' in esc.summary
    assert terminal_esc.level == 2, (
        f'Terminal escalation should be level 2, got {terminal_esc.level}'
    )
    assert terminal_esc.severity == 'critical', (
        f'Terminal escalation should be critical, got {terminal_esc.severity!r}'
    )
    assert 'HALTED' in terminal_esc.summary or 'cap' in terminal_esc.summary.lower(), (
        f'Terminal summary should mention HALTED or cap: {terminal_esc.summary!r}'
    )

    # 4. Supervisor halted
    assert worker._supervisor_halted is True, 'Expected _supervisor_halted=True'
    assert worker._supervisor_halt_reason is not None, '_supervisor_halt_reason should be set'

    # 5. run() completed cleanly (no exception propagated)
    assert run_task.exception() is None, (
        f'run() should not propagate an exception; got {run_task.exception()!r}'
    )

    # 6. Healthy sibling (merger) must be cancelled/done after terminal halt so it
    #    does not leak.  The run() finally-block cancels any still-live loop task.
    assert worker._merger_task is None or worker._merger_task.done(), (
        'Healthy sibling _merger_task should be done/cancelled after terminal halt; '
        f'got done={worker._merger_task.done() if worker._merger_task else None}'
    )


# ---------------------------------------------------------------------------
# Step 5 — RED: normal shutdown / cancelled / shutdown-race do NOT escalate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_shutdown_does_not_trigger_death_path(
    git_ops: GitOps,
    config: OrchestratorConfig,
) -> None:
    """Normal stop(), cancelled loops, and shutdown-race exceptions must never escalate.

    Three sub-checks:
      (A) Clean lifecycle — real loops + stop() → no submission, not halted, clean exit.
      (B) Cancelled-task unit — _on_loop_task_done with a cancelled task → no submission.
      (C) Shutdown-race unit — _running=False + task with RuntimeError → no submission,
          no restart. (This is the check that was failing before step-6 when exc+!running
          was still treated as a death.)
    """
    # ── (A) Clean lifecycle ───────────────────────────────────────────────
    queue_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq_a = _make_fake_escalation_queue()
    worker_a = SpeculativeMergeWorker(git_ops, queue_a, escalation_queue=eq_a)
    worker_a._shutdown_timeout = 0.1

    run_task_a = asyncio.create_task(worker_a.run())
    await asyncio.sleep(0.05)  # let loops start
    await worker_a.stop()
    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(run_task_a, timeout=2.0)

    assert eq_a.submit.call_count == 0, (
        f'(A) No escalation expected on clean stop, got {eq_a.submit.call_count}'
    )
    assert worker_a._supervisor_halted is False, '(A) supervisor should not be halted'
    assert run_task_a.done(), '(A) run_task should be done after stop()'
    # run() should complete with no exception (or CancelledError from outer cancel)
    if not run_task_a.cancelled():
        exc_a = run_task_a.exception()
        assert exc_a is None, f'(A) run() should not propagate exception: {exc_a!r}'

    # ── (B) Cancelled-task unit ───────────────────────────────────────────
    queue_b: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq_b = _make_fake_escalation_queue()
    worker_b = SpeculativeMergeWorker(git_ops, queue_b, escalation_queue=eq_b)

    # Build a cancelled task
    async def _dummy() -> None:
        await asyncio.sleep(10)

    cancelled_task: asyncio.Task = asyncio.create_task(_dummy())
    cancelled_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await cancelled_task

    old_verifier_task = worker_b._verifier_task
    worker_b._on_loop_task_done('verifier', cancelled_task)

    assert eq_b.submit.call_count == 0, (
        f'(B) Cancelled task must not trigger escalation, got {eq_b.submit.call_count}'
    )
    # No new task spawned (still old value, which is None at construction time)
    assert worker_b._verifier_task is old_verifier_task, (
        '(B) _verifier_task must not change after cancelled-task callback'
    )

    # ── (C) Shutdown-race unit ────────────────────────────────────────────
    queue_c: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq_c = _make_fake_escalation_queue()
    worker_c = SpeculativeMergeWorker(git_ops, queue_c, escalation_queue=eq_c)
    # Simulate shutdown in progress
    worker_c._running = False

    # Build a finished task that carries a RuntimeError
    async def _raises() -> None:
        raise RuntimeError('shutdown-race exception')

    exc_task: asyncio.Task = asyncio.create_task(_raises())
    await asyncio.gather(exc_task, return_exceptions=True)
    assert exc_task.done() and not exc_task.cancelled()

    old_merger_task = worker_c._merger_task
    worker_c._on_loop_task_done('merger', exc_task)

    assert eq_c.submit.call_count == 0, (
        f'(C) Shutdown-race exception must NOT escalate, got {eq_c.submit.call_count}'
    )
    # No restart — _merger_task unchanged
    assert worker_c._merger_task is old_merger_task, (
        '(C) _merger_task must not change on shutdown-race death'
    )


# ---------------------------------------------------------------------------
# Step 7 — RED-as-regression-guard: verifier restart preserves state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_restart_preserves_inflight_and_redispatch(
    git_ops: GitOps,
    config: OrchestratorConfig,
) -> None:
    """Within-cap verifier restart must NOT clear _inflight or _redispatch.

    The supervisor path (_spawn_loop + _on_loop_task_done restart branch) must
    never touch self._inflight, self._redispatch, or the seeded entries' result
    Futures.  The restarted _verifier_loop naturally resumes draining them
    (_finalize_inflight's `if not req.result.done()` guard makes re-finalization
    idempotent).

    Procedure:
      1. Seed worker._redispatch with a SpeculativeItem marker.
      2. Seed worker._inflight with an InflightEntry marker (verify_task=None).
      3. Replace _verifier_loop with an idle stub.
      4. Build a dead task (RuntimeError) and invoke _on_loop_task_done directly.
      5. Assert: escalation submitted, new task spawned, _redispatch and _inflight
         unchanged, Future not resolved.
    """
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq = _make_fake_escalation_queue()
    worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=eq)

    # Build a minimal MergeRequest with a pending Future
    future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
    req = MergeRequest(
        task_id='sv7-task',
        branch='task/sv7',
        worktree=git_ops.project_root,  # doesn't need to be a real worktree
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )

    # Build a minimal SpeculativeItem
    item = SpeculativeItem(
        request=req,
        merge_result=None,
        merge_wt=None,
        base_sha='abc123dead',
        speculative=False,
        skip_verify=False,
    )

    # Build a minimal InflightEntry (passthrough: verify_task=None)
    inflight_entry = InflightEntry(
        item=item,
        lease=None,
        verify_task=None,
        merge_wt=None,
        was_speculative=False,
        phase='VERIFY',
    )

    # Seed the instance deques
    worker._redispatch.append(item)
    worker._inflight.append(inflight_entry)

    # Replace _verifier_loop with an idle stub so the spawned restart task
    # stays alive without doing real work
    async def idle_verifier() -> None:
        while worker._running:
            await asyncio.sleep(0.01)

    worker._verifier_loop = idle_verifier  # type: ignore[method-assign]

    # Build a dead task carrying RuntimeError
    async def _raises() -> None:
        raise RuntimeError('verifier died in test')

    dead_task: asyncio.Task = asyncio.create_task(_raises())
    await asyncio.gather(dead_task, return_exceptions=True)
    assert dead_task.done() and not dead_task.cancelled()

    # Set up supervisor state so the within-cap death path fires:
    # _running=True, _live_loops has both so _retire_loop doesn't set _loops_finished
    worker._running = True
    worker._live_loops = {'merger', 'verifier'}

    # Invoke the death callback directly (avoids running run() in background)
    worker._on_loop_task_done('verifier', dead_task)

    # Give the event loop a tick to register the spawned task
    await asyncio.sleep(0)

    # ── Assert 1: escalation submitted once ──────────────────────────────
    assert eq.submit.call_count == 1, (
        f'Expected 1 escalation (death escalation), got {eq.submit.call_count}'
    )

    # ── Assert 2: new verifier task spawned and running ───────────────────
    assert worker._verifier_task is not None, '_verifier_task should not be None'
    assert not worker._verifier_task.done(), (
        '_verifier_task should still be running (idle stub)'
    )

    # ── Assert 3: _redispatch and _inflight are unchanged ─────────────────
    assert len(worker._redispatch) == 1, (
        f'_redispatch should still have 1 item, got {len(worker._redispatch)}'
    )
    assert worker._redispatch[0] is item, '_redispatch[0] must be the seeded item'
    assert len(worker._inflight) == 1, (
        f'_inflight should still have 1 entry, got {len(worker._inflight)}'
    )
    assert worker._inflight[0] is inflight_entry, '_inflight[0] must be the seeded entry'

    # ── Assert 4: Future not resolved by supervisor ───────────────────────
    assert not future.done(), (
        'The seeded entry Future must not be resolved by the supervisor restart path'
    )

    # ── Cleanup ──────────────────────────────────────────────────────────
    worker._running = False
    if worker._verifier_task and not worker._verifier_task.done():
        worker._verifier_task.cancel()
        await asyncio.gather(worker._verifier_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# Amendment 4 — clock injection: rolling-window pruning keeps within-cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_clock_injection_window_pruning(git_ops: GitOps) -> None:
    """Injectable _restart_clock enables window-pruning test without real sleeps.

    Verifies that aged-out restart timestamps are pruned so a death that would
    EXCEED the cap when measured from T=0 is instead treated as WITHIN-cap once
    enough time has elapsed.

    Sequence:
      _max_loop_restarts=2, _loop_restart_window_s=100s.

      T=  0: death 1 → times=[], len=0 < 2 → within-cap, records T=0.
      T= 10: death 2 → times=[0], len=1 < 2 → within-cap, records T=10.
             times=[0, 10]; len=2. Without pruning, the NEXT death would be terminal.
      T=200: death 3 → prune: both 0 and 10 are >100s ago → times=[].
             len=0 < 2 → within-cap (NOT terminal) even though cap "appeared full".

    All three deaths should produce L1 escalations; _supervisor_halted stays False.
    Without _restart_clock injection and window pruning this would require 100+ s of
    real elapsed time.
    """
    from unittest.mock import patch

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq = _make_fake_escalation_queue()
    worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=eq)
    worker._max_loop_restarts = 2
    worker._loop_restart_window_s = 100.0

    # Inject a controllable monotonic clock.
    fake_time = 0.0
    worker._restart_clock = lambda: fake_time

    # Helper: create a task that has already failed with RuntimeError.
    async def _make_dead_task() -> asyncio.Task:  # type: ignore[type-arg]
        async def _raises() -> None:
            raise RuntimeError('test death')

        t: asyncio.Task = asyncio.create_task(_raises())  # type: ignore[type-arg]
        await asyncio.gather(t, return_exceptions=True)
        assert t.done() and not t.cancelled()
        return t

    # Replace _spawn_loop with a no-op that creates a dormant task (no done callback)
    # so invocations of _on_loop_task_done don't chain into real loop logic.
    spawned_tasks: list[asyncio.Task] = []  # type: ignore[type-arg]

    async def _dormant() -> None:
        await asyncio.sleep(3600)

    def fake_spawn(name: str) -> asyncio.Task:  # type: ignore[type-arg]
        t: asyncio.Task = asyncio.create_task(_dormant())  # type: ignore[type-arg]
        spawned_tasks.append(t)
        if name == 'verifier':
            worker._verifier_task = t
        elif name == 'merger':
            worker._merger_task = t
        return t

    # Supervisor preconditions: _running=True, both loops live (so _retire_loop
    # won't set _loops_finished on a within-cap path).
    worker._running = True
    worker._live_loops = {'merger', 'verifier'}

    with patch.object(worker, '_spawn_loop', side_effect=fake_spawn):
        # Death 1 at T=0 → within-cap (times empty, 0 < 2)
        fake_time = 0.0
        worker._on_loop_task_done('verifier', await _make_dead_task())

        assert eq.submit.call_count == 1, (
            f'Death 1: expected 1 L1 escalation, got {eq.submit.call_count}'
        )
        assert eq.submit.call_args_list[0][0][0].level == 1
        assert worker._supervisor_halted is False

        # Death 2 at T=10 → within-cap (len=1 < 2)
        fake_time = 10.0
        worker._on_loop_task_done('verifier', await _make_dead_task())

        assert eq.submit.call_count == 2, (
            f'Death 2: expected 2 L1 escalations total, got {eq.submit.call_count}'
        )
        assert eq.submit.call_args_list[1][0][0].level == 1
        assert worker._supervisor_halted is False

        # Confirm times are filled (without pruning, next death would be terminal).
        assert len(worker._loop_restart_times['verifier']) == 2, (
            'times deque should have 2 entries before advancing clock'
        )

        # Advance clock past the window: both T=0 and T=10 are now >100s old.
        fake_time = 200.0

        # Death 3 at T=200 → pruning must clear old entries → within-cap again.
        worker._on_loop_task_done('verifier', await _make_dead_task())

        assert eq.submit.call_count == 3, (
            f'Death 3: expected 3 L1 escalations total (pruning kept within-cap), '
            f'got {eq.submit.call_count}'
        )
        assert eq.submit.call_args_list[2][0][0].level == 1, (
            'Death 3 after window pruning must be L1 (within-cap), not terminal L2'
        )
        assert worker._supervisor_halted is False, (
            'Supervisor must not halt: old timestamps were pruned, death was within-cap'
        )

    # Cleanup: cancel all dormant tasks spawned by fake_spawn.
    worker._running = False
    for t in spawned_tasks:
        if not t.done():
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)


# ---------------------------------------------------------------------------
# Step 9 — RED: real _merger_loop crash must not kill the verifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_merger_crash_preserves_verifier_and_pipeline(
    git_ops: GitOps,
    config: OrchestratorConfig,
) -> None:
    """A real _merger_loop crash must NOT send a stray shutdown sentinel to the verifier.

    FAILS BEFORE step-10 FIX: the merger's outer finally (:7328) unconditionally
    does ``await self._verifier_queue.put(None)`` on the crash path.  The surviving
    verifier consumes that sentinel, drains _inflight, and returns cleanly.
    _on_loop_task_done('verifier', task) sees exc is None -> _retire_loop without
    restart.  Result: a restarted merger + a permanently-dead verifier; all newly
    merged items pile on _verifier_queue and their result Futures hang forever.

    Injection: patch _maybe_coalesce_waiting_singles to raise RuntimeError on its
    FIRST call (called OUTSIDE the inner try/except, so the raise escapes per-item
    guards and propagates into the outer try/finally).  Subsequent calls delegate to
    the real method (a harmless no-op when train_callback_factory is None).

    Assertions:
      1. outcome.status == 'done' — end-to-end liveness (post-crash request verified).
      2. worker._verifier_task IS verifier_task_before (SAME object, never retired).
      3. Escalation submitted: level==1, 'merge_worker_loop_died' in summary, 'merger'
         in summary.
      4. worker._merger_task is a fresh, not-done Task (restarted by supervisor).
    """
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq = _make_fake_escalation_queue()
    worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=eq)
    worker._shutdown_timeout = 0.1

    # Patch _maybe_coalesce_waiting_singles: crash on first call, delegate afterwards.
    real_coalesce = worker._maybe_coalesce_waiting_singles
    coalesce_call_count = 0

    async def patched_coalesce() -> bool:
        nonlocal coalesce_call_count
        coalesce_call_count += 1
        if coalesce_call_count == 1:
            raise RuntimeError('merger crash injected by test')
        return await real_coalesce()  # type: ignore[no-any-return]

    worker._maybe_coalesce_waiting_singles = patched_coalesce  # type: ignore[method-assign]

    # Build branch+request before starting run() so we can submit after the crash.
    wt = await _make_branch_with_file(git_ops, 'crash-test', 'crash_file.py', 'x = 1\n')

    with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
        run_task = asyncio.create_task(worker.run())

        # Poll up to ~2s until run() spawns both loop tasks.
        deadline = asyncio.get_event_loop().time() + 2.0
        while (
            (worker._merger_task is None or worker._verifier_task is None)
            and asyncio.get_event_loop().time() < deadline
        ):
            await asyncio.sleep(0.02)

        assert worker._merger_task is not None, 'merger_task should be set after run() starts'
        assert worker._verifier_task is not None, 'verifier_task should be set after run() starts'
        verifier_task_before = worker._verifier_task

        # Poll up to ~3s for the merger death escalation.
        deadline = asyncio.get_event_loop().time() + 3.0
        while eq.submit.call_count == 0 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)

        # ── Assert 3: escalation emitted for merger death ─────────────────────
        assert eq.submit.call_count >= 1, (
            f'Expected at least 1 escalation, got {eq.submit.call_count}'
        )
        merger_esc = eq.submit.call_args_list[0][0][0]
        assert merger_esc.level == 1, f'Expected level 1, got {merger_esc.level}'
        assert merger_esc.severity == 'blocking', (
            f'Expected severity blocking, got {merger_esc.severity!r}'
        )
        assert 'merge_worker_loop_died' in merger_esc.summary, (
            f'summary missing merge_worker_loop_died: {merger_esc.summary!r}'
        )
        assert 'merger' in merger_esc.summary, (
            f'summary missing "merger": {merger_esc.summary!r}'
        )

        # Give the event loop time for the supervisor to restart the merger.
        await asyncio.sleep(0.1)

        # Submit the post-crash request.
        req = _make_request('crash-test', 'crash-test', wt, config)
        await queue.put(req)

        # ── Assert 1: end-to-end liveness — restarted merger + surviving verifier ──
        outcome = await asyncio.wait_for(req.result, timeout=15.0)

    assert outcome.status == 'done', (
        f'Expected outcome.status="done" but got {outcome.status!r}'
    )

    # ── Assert 2: verifier is the SAME task (never retired/replaced) ─────────
    assert worker._verifier_task is verifier_task_before, (
        'Verifier task must be the SAME object — it must not have been retired/replaced'
    )
    assert not worker._verifier_task.done(), (
        'Verifier task should still be running (not retired)'
    )

    # ── Assert 4: merger was restarted (fresh, not-done task) ────────────────
    assert worker._merger_task is not None
    assert not worker._merger_task.done(), 'Restarted merger task should still be running'

    # ── Cleanup ──────────────────────────────────────────────────────────────
    await worker.stop()
    with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
        await asyncio.wait_for(run_task, timeout=2.0)


# ---------------------------------------------------------------------------
# Step 11 — RED: clean loop exit while running must be treated as death
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_loop_exit_while_running_is_treated_as_death(
    git_ops: GitOps,
    config: OrchestratorConfig,
) -> None:
    """A loop that returns cleanly while self._running is True is a synthetic death.

    FAILS BEFORE step-12: the current _on_loop_task_done retrieves exc=None for a
    clean return and the guard ``if exc is None or not self._running: _retire_loop``
    retires the loop silently with NO escalation and NO restart.

    After step-12 the SHUTDOWN GUARD (``if not self._running: _retire_loop``) is
    checked FIRST (so normal shutdown is unaffected), then any clean return while
    running is wrapped in a synthetic RuntimeError and falls through to the
    restart-cap + escalate path.

    Setup:
      - _verifier_loop replaced with an idle stub so the spawned restart stays benign.
      - _running=True, _live_loops={'merger','verifier'} (both live so _retire_loop
        won't set _loops_finished on a within-cap path).
      - A FINISHED task that returned None (not cancelled, no exception).

    Assertions:
      1. escalation_queue.submit called once with a level==1 'merge_worker_loop_died'
         escalation; summary contains 'verifier'.
      2. A NEW worker._verifier_task was spawned (a different object) and is not done
         (loop was RESTARTED, not retired).
      3. 'verifier' is still in worker._live_loops (not retired).
      4. worker._loops_finished is NOT set (not retired).
    """
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    eq = _make_fake_escalation_queue()
    worker = SpeculativeMergeWorker(git_ops, queue, escalation_queue=eq)

    # Replace _verifier_loop with a harmless idle stub so the spawned restart
    # task stays alive without doing real work.
    async def idle_verifier() -> None:
        while worker._running:
            await asyncio.sleep(0.01)

    worker._verifier_loop = idle_verifier  # type: ignore[method-assign]

    # Construct a FINISHED task that returned None cleanly (not cancelled, no exc).
    async def _clean_return() -> None:
        return None

    clean_task: asyncio.Task = asyncio.create_task(_clean_return())
    await asyncio.gather(clean_task, return_exceptions=True)
    assert clean_task.done() and not clean_task.cancelled()
    assert clean_task.exception() is None, 'task should have a clean return (exc=None)'

    # Set supervisor state: running + both loops live.
    worker._running = True
    worker._live_loops = {'merger', 'verifier'}

    # Invoke the death callback directly.
    worker._on_loop_task_done('verifier', clean_task)

    # Give the event loop a tick to register the spawned task.
    await asyncio.sleep(0)

    # ── Assert 1: escalation submitted once with level==1 ────────────────────
    assert eq.submit.call_count == 1, (
        f'Expected 1 escalation (synthetic death for clean-exit-while-running), '
        f'got {eq.submit.call_count}'
    )
    esc = eq.submit.call_args[0][0]
    assert esc.level == 1, f'Expected level 1, got {esc.level}'
    assert 'merge_worker_loop_died' in esc.summary, (
        f'summary missing merge_worker_loop_died: {esc.summary!r}'
    )
    assert 'verifier' in esc.summary, (
        f'summary missing "verifier": {esc.summary!r}'
    )

    # ── Assert 2: new verifier task spawned and not done ─────────────────────
    assert worker._verifier_task is not None, '_verifier_task should not be None'
    assert worker._verifier_task is not clean_task, (
        '_verifier_task must be a NEW task (different from the clean-return task)'
    )
    assert not worker._verifier_task.done(), (
        '_verifier_task should still be running (idle stub)'
    )

    # ── Assert 3: verifier still in live_loops ───────────────────────────────
    assert 'verifier' in worker._live_loops, (
        "'verifier' must still be in _live_loops (not retired)"
    )

    # ── Assert 4: _loops_finished NOT set ────────────────────────────────────
    assert not worker._loops_finished.is_set(), (
        '_loops_finished must NOT be set (loop was restarted, not retired)'
    )

    # ── Cleanup ──────────────────────────────────────────────────────────────
    worker._running = False
    if worker._verifier_task and not worker._verifier_task.done():
        worker._verifier_task.cancel()
        await asyncio.gather(worker._verifier_task, return_exceptions=True)
