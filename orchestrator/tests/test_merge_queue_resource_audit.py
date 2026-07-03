"""Tests for SpeculativeMergeWorker resource-conservation audits — permits/
caps (I4) + worktree ledger (I6) (MQ-invariants iota / task 1994).

Steps covered:
  step-1  RED   — speculation_accounting_violations() unit tests (bare worker)
  step-2  GREEN — implement speculation_accounting_violations() + helpers
  step-3  RED   — worktree_ledger_violations() unit tests (bare worker)
  step-4  GREEN — implement worktree_ledger_violations()
  step-5  RED   — additive snapshot()['resource_audit'] key
  step-6  GREEN — wire resource_audit into snapshot()
  step-7  RED   — _alarm_resource_audit(...) unit tests
  step-8  GREEN — implement _alarm_resource_audit(...)
  step-9  RED   — _check_resource_audit(now) unit tests (streak + escalation)
  step-10 GREEN — implement _check_resource_audit(now)
  step-11 RED   — heartbeat wiring: resource audit runs even at depth==0
  step-12 GREEN — wire _check_resource_audit into _maybe_log_queue_heartbeat

Both audits (speculation_accounting_violations, worktree_ledger_violations)
and _check_resource_audit/_maybe_log_queue_heartbeat are pure/synchronous —
no ``await`` is needed anywhere in this module, so every test here is a
plain ``def test_...`` (never ``async def`` / ``@pytest.mark.asyncio``; this
suite's pyproject.toml turns a sync-def-inside-an-asyncio-marked-class into a
collection ERROR, so mixing would break collection).

This module reuses the bare-worker git_repo/git_config/git_ops fixtures and
_FakeEscalationQueue from test_merge_queue_request_liveness.py (per-file
duplication convention — see that module's docstring). It imports
orchestrator.merge_queue LOCALLY inside each test (mirrors
test_merge_request_ledger.py's / test_merge_queue_permit_conservation.py's
convention) so a not-yet-implemented symbol never breaks collection of the
rest of the file during the RED steps.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_request_liveness.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
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


class _FakeEscalationQueue:
    """Minimal fake escalation queue (copied from
    test_merge_queue_request_liveness.py:119 — per-file duplication
    convention).
    """

    def __init__(self, *, open_l1: bool = False):
        self._open_l1 = open_l1
        self._seq = 0
        self.submitted: list = []

    def has_open_l1(self, task_id: str) -> bool:  # noqa: ARG002
        return self._open_l1

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{self._seq}'

    def submit(self, esc) -> None:
        self.submitted.append(esc)

    def open_it(self):
        """Simulate a prior open L1 (for dedup tests)."""
        self._open_l1 = True


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: speculation_accounting_violations()
# ---------------------------------------------------------------------------


class TestSpeculationAccountingViolations:
    """Unit tests for SpeculativeMergeWorker.speculation_accounting_violations().

    Covers the I4 permit/cap conservation identities:
      (a) slot_available + held_by_merger + inflight_speculative == depth
      (b) merge_ahead_cap._value + inflight_cap_count == depth

    RED until step-2 GREEN adds the method to merge_queue.py.
    """

    @pytest.mark.parametrize('depth', [1, 2])
    def test_healthy_worker_reports_no_violations(
        self, git_ops: GitOps, depth: int,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=depth)

        assert worker.speculation_accounting_violations() == []

    def test_forced_speculation_slot_leak_yields_one_violation(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # Force a leak: a permit vanished from the shared semaphore without
        # being recorded as held-by-merger, transferred to the verifier, or
        # genuinely still available — breaks identity (a).
        worker._speculation_slot._value -= 1

        violations = worker.speculation_accounting_violations()

        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'
        assert 'speculation' in violations[0].lower()

    def test_forced_merge_ahead_cap_leak_yields_one_violation(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # Force a leak: a merge-ahead-cap permit vanished without a
        # corresponding counts_against_cap=True item in the verifier queue —
        # breaks identity (b).
        worker._merge_ahead_cap._value -= 1

        violations = worker.speculation_accounting_violations()

        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'
        assert 'cap' in violations[0].lower()

    def test_both_leaks_yield_two_violations(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        worker._speculation_slot._value -= 1
        worker._merge_ahead_cap._value -= 1

        violations = worker.speculation_accounting_violations()

        assert len(violations) == 2, f'expected exactly two violations, got: {violations!r}'

    def test_not_running_suppresses_violations(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)
        worker._speculation_slot._value -= 1
        worker._merge_ahead_cap._value -= 1
        worker._running = False

        assert worker.speculation_accounting_violations() == []
