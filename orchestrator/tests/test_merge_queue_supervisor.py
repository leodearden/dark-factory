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
import collections
import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeOutcome, MergeRequest, SpeculativeMergeWorker

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
    try:
        await asyncio.wait_for(run_task, timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


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
