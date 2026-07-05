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
import concurrent.futures
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import PERSISTENT_MERGE_WORKTREE_NAME, GitOps, _run
from orchestrator.merge_types import MergeRequest

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
# task 2063: _redispatch-parked speculative permit accounting
#
# _inflight_speculative_count() is the single source of truth feeding both
# snapshot()['speculation']['inflight_speculative'] and
# speculation_accounting_violations(). Before the fix it scanned only
# self._inflight (was_speculative) and self._verifier_queue (.speculative),
# missing a speculative item parked on self._redispatch while it waits for a
# free verify host (DISPATCH-FILL / blocking-get paths in _verifier_loop when
# _dispatch_item returns None because verify hosts < speculation_depth). That
# gap produced the observed multi-heartbeat, self-clearing skew:
# 'slot_available(0) + held_by_merger(0) + inflight_speculative(1) == 1,
# expected depth=2' — one speculative permit countable in _verifier_queue,
# one uncounted on _redispatch.
# ---------------------------------------------------------------------------


def _make_spec_item(tmp_path: Path, *, speculative: bool):
    """Build a minimal RealMergeItem for _inflight_speculative_count() tests.

    Post task-o: SpeculativeItem is a TypeAlias = RealMergeItem | DecidedItem,
    not a constructor — build the REAL arm directly. RealMergeItem has no
    __post_init__ and the counter under test reads only ``.speculative``, so
    a near-bare MagicMock request is fine here (no real asyncio.Future needed,
    unlike test_merge_speculation.py's fuller _make_spec_item which drives
    _run_inflight_verify). ``enqueued_at`` is set to a real float (rather than
    left as an auto-attribute MagicMock) because full ``snapshot()`` calls
    build an 'entries' dict per queued item and compute
    ``max(0.0, now - req.enqueued_at)``, which raises TypeError comparing a
    bare MagicMock against a float.
    """
    from orchestrator.merge_types import RealMergeItem

    return RealMergeItem(
        request=MagicMock(enqueued_at=1_000_000.0),
        merge_result=MagicMock(merge_commit='deadbeef01234567890a'),
        merge_wt=tmp_path / '_merge-x',
        base_sha='aabbccdd00000000aaaa',
        speculative=speculative,
    )


class TestRedispatchSpeculativeAccounting:
    """Unit tests for the task-2063 _redispatch under-count fix.

    RED pre-fix: a speculative item parked on _redispatch is invisible to
    _inflight_speculative_count(), so the count comes up one short and
    speculation_accounting_violations() reports a spurious conservation
    violation. GREEN post-fix: the count includes it and the identity holds.
    """

    def test_speculative_item_parked_on_redispatch_is_counted(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # Drain both speculation permits: slot_available=0, held_by_merger=0
        # on a bare worker (nothing else touches the semaphore).
        worker._speculation_slot._value = 0

        # One speculative permit is countable via the existing
        # _verifier_queue scan...
        worker._verifier_queue.put_nowait(_make_spec_item(tmp_path, speculative=True))
        # ...and one is parked on _redispatch awaiting a free host — the gap
        # this task closes.
        worker._redispatch.append(_make_spec_item(tmp_path, speculative=True))

        assert worker._inflight_speculative_count() == 2
        assert worker.speculation_accounting_violations() == []
        assert worker.snapshot()['resource_audit']['speculation_accounting'] == []
        # Pin the OTHER consumer of _inflight_speculative_count() too: the
        # snapshot's 'speculation' key documents (merge_queue.py ~5534-5541)
        # that inflight_speculative includes _redispatch items, so assert it
        # directly rather than only exercising it transitively via the
        # resource_audit key above.
        assert worker.snapshot()['speculation']['inflight_speculative'] == 2

    def test_redispatch_undercount_produces_the_observed_violation_message(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # Same drain, but ONLY the verifier-queue item is present: a genuine
        # 1-permit-vs-2-drained imbalance that holds both pre- and post-fix —
        # pins the identity/message semantics of the exact skew being closed.
        worker._speculation_slot._value = 0
        worker._verifier_queue.put_nowait(_make_spec_item(tmp_path, speculative=True))

        violations = worker.speculation_accounting_violations()

        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'
        assert 'speculation-slot conservation violated' in violations[0]
        assert 'expected depth=2' in violations[0]

    def test_nonspeculative_redispatch_item_not_counted(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # A cascade-remerged item parked on _redispatch is non-speculative
        # (_remerge returns speculative=False; its slot was already released
        # before remerge) and must never be counted — else the fix would
        # over-count and could mask a genuine imbalance.
        worker._redispatch.append(_make_spec_item(tmp_path, speculative=False))

        assert worker._inflight_speculative_count() == 0
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


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: _check_resource_audit(now)
# ---------------------------------------------------------------------------


class TestCheckResourceAudit:
    """Unit tests for SpeculativeMergeWorker._check_resource_audit(now)
    (task 1994 step-9).

    Mirrors TestCheckRequestLiveness (test_merge_queue_request_liveness.py):
    sync, clock-injectable, WARNING-on-any-violation + streak-gated dedup'd
    L1 escalation once the violation persists across
    RESOURCE_AUDIT_ESCALATION_STREAK consecutive calls.

    RED until step-10 GREEN adds the method (+ streak field/constant) to
    merge_queue.py.
    """

    def test_healthy_worker_no_warning_no_streak_no_escalation(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(), escalation_queue=fake_eq, speculation_depth=2,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._check_resource_audit(1000.0)

        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 0
        assert worker._resource_audit_violation_streak == 0
        assert len(fake_eq.submitted) == 0

    def test_single_violating_call_warns_and_bumps_streak_without_escalating(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(), escalation_queue=fake_eq, speculation_depth=2,
        )
        worker._speculation_slot._value -= 1  # forced permit leak

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._check_resource_audit(1000.0)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
        assert 'speculation' in warnings[0].message.lower()
        assert worker._resource_audit_violation_streak == 1
        assert len(fake_eq.submitted) == 0, 'must not escalate before the streak threshold'

    def test_n_consecutive_violations_escalates_exactly_once(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(), escalation_queue=fake_eq, speculation_depth=2,
        )
        worker._speculation_slot._value -= 1  # forced, PERSISTING permit leak
        n = worker.RESOURCE_AUDIT_ESCALATION_STREAK

        for i in range(1, n + 1):
            worker._check_resource_audit(1000.0 + i)

        assert worker._resource_audit_violation_streak == n
        assert len(fake_eq.submitted) == 1, (
            f'expected exactly one escalation after {n} consecutive violating '
            f'heartbeats, got {len(fake_eq.submitted)}'
        )
        esc = fake_eq.submitted[0]
        assert esc.category == 'merge_resource_leak'

        # Simulate the real escalation queue now reporting an open L1 (as it
        # would after the submit above) so a further persisting violation
        # does not resubmit a duplicate.
        fake_eq.open_it()
        worker._check_resource_audit(1000.0 + n + 1)
        assert len(fake_eq.submitted) == 1, 'must not resubmit while the L1 is open'

    def test_clearing_the_leak_resets_streak_to_zero(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(), escalation_queue=fake_eq, speculation_depth=2,
        )
        worker._speculation_slot._value -= 1  # forced permit leak

        worker._check_resource_audit(1000.0)
        worker._check_resource_audit(1001.0)
        assert worker._resource_audit_violation_streak == 2

        worker._speculation_slot._value += 1  # clear the leak

        worker._check_resource_audit(1002.0)
        assert worker._resource_audit_violation_streak == 0
        assert len(fake_eq.submitted) == 0


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN: heartbeat wiring — _check_resource_audit runs
# UNCONDITIONALLY near the top of _maybe_log_queue_heartbeat
# ---------------------------------------------------------------------------


def _make_request(task_id: str, branch: str, worktree: Path) -> MergeRequest:
    """Build a minimal queued-only MergeRequest for a plain sync test (no
    running event loop).

    Unlike test_merge_queue_request_liveness.py's
    ``asyncio.get_running_loop().create_future()`` helper (only callable
    from ``async def`` tests), this module's tests are plain ``def
    test_...`` (see module docstring), so ``result`` is a
    ``concurrent.futures.Future`` instead — duck-type compatible with the
    only method ``snapshot()`` calls on it (``.cancelled()``) and needs no
    event loop. ``config=None`` because this request is only ever placed
    directly on ``worker._queue`` (never dispatched by a running worker), so
    ``.config`` is never read.
    """
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=None,  # type: ignore[arg-type]
        result=concurrent.futures.Future(),  # type: ignore[arg-type]
        lane='normal',
    )


class TestHeartbeatWiringRunsResourceAuditUnconditionally:
    """_maybe_log_queue_heartbeat must invoke _check_resource_audit
    UNCONDITIONALLY near the top — before the depth==0 short-circuit, and
    wrapped in its own try/except so a resource-audit bug can never
    suppress the depth heartbeat below it (mirrors the eta/1992
    _check_request_liveness wiring — see
    TestHeartbeatWiringRunsLivenessCheckFirst in
    test_merge_queue_request_liveness.py) (task 1994 step-11).

    RED until step-12 GREEN wires the call into _maybe_log_queue_heartbeat.
    """

    def test_idle_depth_zero_worker_still_warns_on_leak(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(), escalation_queue=fake_eq, speculation_depth=2,
        )
        worker._speculation_slot._value -= 1  # forced permit leak

        # Idle pipeline — nothing enqueued/inflight — is exactly the shape
        # that would otherwise short-circuit _maybe_log_queue_heartbeat
        # before any resource-audit check ran.
        assert worker.snapshot()['depth'] == 0

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._maybe_log_queue_heartbeat(1000.0)

        assert result is False, 'heartbeat must stay idle — no depth to report (depth==0)'
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one resource-audit WARNING, got: {caplog.text}'
        assert 'speculation' in warnings[0].message.lower()
        assert worker._resource_audit_violation_streak == 1

    def test_idle_depth_zero_worker_escalates_after_streak_threshold(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(), escalation_queue=fake_eq, speculation_depth=2,
        )
        worker._speculation_slot._value -= 1  # forced, PERSISTING permit leak
        n = worker.RESOURCE_AUDIT_ESCALATION_STREAK

        for i in range(n):
            result = worker._maybe_log_queue_heartbeat(1000.0 + i)
            assert result is False, 'pipeline stays idle throughout — depth never becomes > 0'

        assert worker._resource_audit_violation_streak == n
        assert len(fake_eq.submitted) == 1, (
            f'expected exactly one escalation after {n} consecutive violating heartbeats '
            f'on a depth-0 idle pipeline, got {len(fake_eq.submitted)}'
        )
        assert fake_eq.submitted[0].category == 'merge_resource_leak'

    def test_raising_resource_audit_does_not_suppress_depth_heartbeat(
        self,
        git_ops: GitOps,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('hb-task', 'hb-task', wt)
        worker._queue.put_nowait(req)  # non-zero depth so the depth heartbeat would fire

        def _boom(now: float) -> None:  # noqa: ARG001
            raise RuntimeError('boom')

        monkeypatch.setattr(worker, '_check_resource_audit', _boom)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._maybe_log_queue_heartbeat(1000.0)

        assert result is True, 'depth heartbeat must still fire despite the resource-audit bug'
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1, f'expected the swallowed exception to be logged, got: {caplog.text}'
        assert 'resource-audit' in errors[0].message.lower()

    def test_forced_leak_surfaces_in_both_snapshot_and_heartbeat_warning(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """End-to-end user-observable signal (step-11c): driving the check
        exclusively through _maybe_log_queue_heartbeat (never calling
        speculation_accounting_violations()/_check_resource_audit directly)
        still surfaces the violation both via a heartbeat WARNING log line
        and in the next snapshot()['resource_audit'].
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)
        worker._speculation_slot._value -= 1  # forced permit leak

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._maybe_log_queue_heartbeat(1000.0)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'

        violations = worker.snapshot()['resource_audit']['speculation_accounting']
        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'
