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
