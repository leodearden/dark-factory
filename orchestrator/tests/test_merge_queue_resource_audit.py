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
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, PERSISTENT_MERGE_WORKTREE_NAME, _run

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


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: worktree_ledger_violations()
# ---------------------------------------------------------------------------


def _mkdir_worktree(git_ops: GitOps, name: str, *, mtime: float | None = None) -> Path:
    """Create worktree_base/name (creating worktree_base itself if absent).

    When *mtime* is given, backdates/sets the directory's mtime via
    ``os.utime`` so age-based grace-window tests are deterministic without
    any real sleeping.
    """
    git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
    wt = git_ops.worktree_base / name
    wt.mkdir()
    if mtime is not None:
        os.utime(wt, (mtime, mtime))
    return wt


class TestWorktreeLedgerViolations:
    """Unit tests for SpeculativeMergeWorker.worktree_ledger_violations(now=None).

    RED until step-4 GREEN adds the method to merge_queue.py.

    ``_GRACE`` is a test-local override of ``RESOURCE_AUDIT_WORKTREE_GRACE_SECS``
    (monkeypatched per-instance), decoupled from whatever tactical production
    default step-4 chooses.
    """

    _GRACE = 100.0
    _NOW = 1_000_000.0

    def test_no_on_disk_merge_dirs_yields_no_violations(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = self._GRACE
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        assert worker.worktree_ledger_violations(now=self._NOW) == []

    def test_missing_worktree_base_yields_no_violations(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = self._GRACE

        assert not git_ops.worktree_base.exists()
        assert worker.worktree_ledger_violations(now=self._NOW) == []

    def test_unregistered_aged_dir_yields_one_violation_naming_the_path(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = self._GRACE
        wt = _mkdir_worktree(
            git_ops, '_merge-deadbeef', mtime=self._NOW - self._GRACE - 10,
        )

        violations = worker.worktree_ledger_violations(now=self._NOW)

        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'
        assert str(wt.resolve()) in violations[0]

    def test_registered_dir_yields_no_violation(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = self._GRACE
        wt = _mkdir_worktree(
            git_ops, '_merge-registered', mtime=self._NOW - self._GRACE - 10,
        )
        worker._register_owned_merge_worktree(wt)

        assert worker.worktree_ledger_violations(now=self._NOW) == []

    def test_persistent_verify_worktree_never_flagged(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = self._GRACE
        _mkdir_worktree(
            git_ops, PERSISTENT_MERGE_WORKTREE_NAME, mtime=self._NOW - self._GRACE - 10,
        )

        assert worker.worktree_ledger_violations(now=self._NOW) == []

    def test_wrong_prefix_dir_never_flagged(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = self._GRACE
        _mkdir_worktree(git_ops, '_solo-xyz', mtime=self._NOW - self._GRACE - 10)

        assert worker.worktree_ledger_violations(now=self._NOW) == []

    def test_fresh_unregistered_dir_within_grace_yields_no_violation(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = self._GRACE
        _mkdir_worktree(git_ops, '_merge-fresh', mtime=self._NOW - 1)

        assert worker.worktree_ledger_violations(now=self._NOW) == []

    def test_not_running_suppresses_even_aged_unregistered_dir(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker.RESOURCE_AUDIT_WORKTREE_GRACE_SECS = self._GRACE
        _mkdir_worktree(
            git_ops, '_merge-abandoned', mtime=self._NOW - self._GRACE - 10,
        )
        worker._running = False

        assert worker.worktree_ledger_violations(now=self._NOW) == []


# ---------------------------------------------------------------------------
# step-5 RED / step-6 GREEN: additive snapshot()['resource_audit'] key
# ---------------------------------------------------------------------------

# The full pre-existing snapshot() key set (must remain present — additive-only
# freeze), pinned as of task theta/1993 (which added 'speculation'). Mirrors
# test_merge_queue_permit_conservation.py's _PRE_EXISTING_SNAPSHOT_KEYS with
# 'speculation' folded in as pre-existing from task iota's point of view. See
# merge_queue.py's snapshot() return dict.
_PRE_EXISTING_SNAPSHOT_KEYS = {
    'entries',
    'depth',
    'head_of_line',
    'verify_in_progress',
    'is_wip_halted',
    'halt_owner_esc_id',
    'occupancy',
    'suffix_conflict_graph',
    'metrics',
    'frozen_prefix',
    'two_layer_invariants',
    'speculation',
}


class TestSnapshotResourceAuditKey:
    """Unit tests for the additive ``snapshot()['resource_audit']`` key.

    RED until step-6 GREEN adds the key to merge_queue.py's ``snapshot()``.
    """

    def test_healthy_worker_resource_audit_is_empty(self, git_ops: GitOps) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        assert worker.snapshot()['resource_audit'] == {
            'speculation_accounting': [],
            'worktree_ledger': [],
        }

    def test_forced_leak_surfaces_in_snapshot_resource_audit(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)
        worker._speculation_slot._value -= 1

        violations = worker.snapshot()['resource_audit']['speculation_accounting']

        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'

    def test_resource_audit_is_additive_and_matches_direct_audit_calls(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)
        worker._speculation_slot._value -= 1

        snap = worker.snapshot()

        # Additive-only: every pre-existing key (pinned through task theta)
        # stays present alongside the new 'resource_audit' key.
        assert set(snap) >= _PRE_EXISTING_SNAPSHOT_KEYS
        assert 'resource_audit' in snap

        # The snapshot's sub-values match calling the audit methods directly —
        # same underlying state, read synchronously with no intervening
        # mutation between the two calls.
        assert snap['resource_audit']['speculation_accounting'] == (
            worker.speculation_accounting_violations()
        )
        assert snap['resource_audit']['worktree_ledger'] == (
            worker.worktree_ledger_violations()
        )


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: _alarm_resource_audit(...)
# ---------------------------------------------------------------------------


class TestAlarmResourceAudit:
    """Unit tests for the module-level ``_alarm_resource_audit(escalation_queue,
    violations, *, event_store=None)`` helper (task 1994 step-7).

    Modeled on ``TestAlarmMergeRequestStuck`` in test_merge_request_ledger.py
    (the eta/1992 sibling). Unlike that helper's PER-REQUEST sentinel,
    ``_alarm_resource_audit`` alarms on a single FIXED worker-level sentinel
    — there is one resource audit per worker, not one per request.

    RED until step-8 GREEN adds the function (+ sentinel constant) to
    merge_queue.py.
    """

    def _call(self, eq, violations, *, event_store=None) -> None:
        from orchestrator.merge_queue import _alarm_resource_audit

        _alarm_resource_audit(eq, violations, event_store=event_store)

    def test_none_queue_is_noop(self) -> None:
        """None escalation_queue -> returns silently, no raise."""
        self._call(None, ['some violation'])  # must not raise

    def test_first_call_submits_exactly_one_escalation(self) -> None:
        eq = _FakeEscalationQueue(open_l1=False)
        self._call(eq, ['speculation-slot conservation violated: ...'])
        assert len(eq.submitted) == 1

    def test_escalation_has_level_1_and_blocking_severity(self) -> None:
        eq = _FakeEscalationQueue(open_l1=False)
        self._call(eq, ['a violation'])
        esc = eq.submitted[0]
        assert esc.level == 1
        assert esc.severity == 'blocking'

    def test_escalation_has_merge_resource_leak_category(self) -> None:
        eq = _FakeEscalationQueue(open_l1=False)
        self._call(eq, ['a violation'])
        esc = eq.submitted[0]
        assert esc.category == 'merge_resource_leak'

    def test_escalation_task_id_is_the_fixed_sentinel(self) -> None:
        from orchestrator.merge_queue import _RESOURCE_AUDIT_SENTINEL

        eq = _FakeEscalationQueue(open_l1=False)
        self._call(eq, ['a violation'])
        esc = eq.submitted[0]
        assert esc.task_id == _RESOURCE_AUDIT_SENTINEL

    def test_summary_and_detail_name_the_violations(self) -> None:
        eq = _FakeEscalationQueue(open_l1=False)
        violations = [
            'speculation-slot conservation violated: foo',
            'unregistered on-disk merge worktree /tmp/x/_merge-deadbeef',
        ]
        self._call(eq, violations)
        esc = eq.submitted[0]
        combined = esc.summary + esc.detail
        assert 'speculation-slot conservation violated: foo' in combined
        assert 'unregistered on-disk merge worktree /tmp/x/_merge-deadbeef' in combined

    def test_second_call_with_open_l1_is_deduped(self) -> None:
        eq = _FakeEscalationQueue(open_l1=True)  # alarm already open
        self._call(eq, ['a violation'])
        assert len(eq.submitted) == 0

    def test_event_store_emits_escalation_created_event(self) -> None:
        from orchestrator.event_store import EventType

        eq = _FakeEscalationQueue(open_l1=False)
        es = MagicMock()
        self._call(eq, ['a violation'], event_store=es)

        assert es.emit.called
        args, kwargs = es.emit.call_args
        emitted_type = args[0] if args else kwargs.get('event_type')
        assert emitted_type == EventType.escalation_created
