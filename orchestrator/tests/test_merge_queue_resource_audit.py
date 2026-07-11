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
no ``await`` is needed to exercise them directly, so most tests here are
plain ``def test_...``. The exception is the ledger-based census tests in
TestRedispatchSpeculativeAccounting / TestFinalizeHeadSpeculativeAccounting /
TestDispatchGapSpeculativeAccounting (task 2160/η): they acquire real
permits via ``await worker._speculation_ledger.acquire()``, so those are
``async def test_...``, each decorated individually with
``@pytest.mark.asyncio`` (this project's pyproject.toml does NOT set
``asyncio_mode = "auto"``, so pytest-asyncio's default "strict" mode
applies here — every coroutine test needs an explicit marker; mirrors the
per-function-marker convention in test_merge_speculation_controller.py /
test_merge_queue_permit_conservation.py). The marker is applied PER TEST,
never at class or module level: this suite's pyproject.toml turns a sync
``def test_...`` collected under an ``@pytest.mark.asyncio`` mark into a
collection-time ERROR (filterwarnings escalates pytest-asyncio's mismatch
warning) — exactly what a class-level marker would do to the sync
``test_nonspeculative_*``/``test_detector_not_blinded_*`` methods that share
these same classes.

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
      (a) slot_available + len(speculation_ledger.live) == depth (task
          2160/η, PRD DD6 — collapses the merger/verifier-decomposed
          slot_available + held_by_merger + inflight_speculative == depth
          form still exposed via snapshot()['speculation'])
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
        # corresponding cap_permit-bearing item in the verifier queue —
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
# task 2063 / task 2096 / task 2160 (η): speculative-permit accounting
#
# _inflight_speculative_count() used to be the single source of truth
# feeding both snapshot()['speculation']['inflight_speculative'] and
# speculation_accounting_violations() by scanning FIVE separate worker-
# internal locations (self._inflight, self._verifier_queue,
# self._redispatch, self._finalizing_head, self._dispatching_item) — task
# 2063 closed the _redispatch gap, task 2096 closed the _finalizing_head and
# _dispatching_item gaps (each an under-count window that produced a
# spurious 'speculation-slot conservation violated' finding).
#
# Task 2160 (η) threads the SpecPermit token itself onto the pipeline item
# (RealMergeItem/DecidedItem/InflightEntry.permit) and routes every release
# through the shared PermitLedger. Conservation is now enforced
# structurally: WHERE an item sits no longer matters, only whether its
# permit is still registered in ``ledger.live``. The three classes below are
# kept separate (rather than merged into one) to preserve the historical
# per-gap narrative, but their "counted" tests now pin the replacement
# ledger-only invariant directly — deliberately WITHOUT placing the
# permit-carrying item into any of the five old census locations, since a
# permit acquired through the ledger must be counted regardless of
# location. Their "not counted" tests are unchanged: an item/entry with no
# permit must never be counted, in the old census locations or anywhere
# else.
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


# ---------------------------------------------------------------------------
# task 2161 (θ) amendment pass: identity (c) merge-ahead-cap handoff
# cross-check
#
# (c) walks _verifier_queue directly and flags any RealMergeItem whose
# cap_permit is non-None but absent from merge_ahead_ledger.live — a signal
# independent of (b), which derives both its own operands from the ledger
# itself and so cannot see a handoff/threading regression. These tests pin
# (c)'s behaviour directly against a bare (non-running) worker:
#   - a foreign/never-acquired token on a queued item IS flagged (the real
#     bug class the check exists to catch);
#   - a live-acquired token that gets released while its item is STILL
#     queued (the narrow CancelledError-after-put() race documented at the
#     _merger_loop put()-failure except block in merge_queue.py) IS flagged
#     if left in place — (c) does not blindly tolerate every released token;
#   - the same released token, once the except block clears the item's
#     ``cap_permit`` back to ``None`` (the amendment-pass fix), is NOT
#     flagged — matching the state an ordinary on-drain release +
#     redispatch-clear already produces via
#     ``dataclasses.replace(item, cap_permit=None)``.
# ---------------------------------------------------------------------------


class TestMergeAheadCapHandoffCrossCheck:
    """Unit tests for speculation_accounting_violations()'s identity (c)."""

    @pytest.mark.asyncio
    async def test_healthy_queued_item_with_live_permit_no_violation(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        item = _make_spec_item(tmp_path, speculative=False)
        item.cap_permit = await worker._merge_ahead_ledger.acquire()
        worker._verifier_queue.put_nowait(item)

        assert worker.speculation_accounting_violations() == []

    def test_foreign_cap_permit_on_queued_item_flags_violation(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        """A genuine handoff bug: a token never issued by this ledger (so it
        is neither live nor released) stamped onto a queued item. (c) exists
        precisely to catch this — (b) stays green throughout because it
        never reads the item's stamped token, only the ledger's own
        bookkeeping.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker
        from orchestrator.merge_types import CapPermit

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        item = _make_spec_item(tmp_path, speculative=False)
        item.cap_permit = CapPermit()  # never acquired through the ledger
        worker._verifier_queue.put_nowait(item)

        violations = worker.speculation_accounting_violations()

        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'
        assert 'handoff' in violations[0]

    @pytest.mark.asyncio
    async def test_released_but_uncleared_cap_permit_flags_violation(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        """The false-positive scenario the amendment-pass fix targets, MINUS
        the fix: a token acquired then released while its item is still
        queued (the narrow put()-failure race). Left uncleared, (c) reports
        it — from (c)'s perspective this is indistinguishable from a genuine
        premature-release bug, which is exactly why the merger loop must
        clear the field rather than leaving (c) to blindly tolerate any
        released token (see the next test for the cleared, non-violating
        state).
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        item = _make_spec_item(tmp_path, speculative=False)
        item.cap_permit = await worker._merge_ahead_ledger.acquire()
        worker._verifier_queue.put_nowait(item)
        worker._merge_ahead_ledger.release(item.cap_permit)  # simulates the except-block release

        violations = worker.speculation_accounting_violations()

        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'
        assert 'handoff' in violations[0]

    @pytest.mark.asyncio
    async def test_released_and_cleared_cap_permit_no_violation(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        """The fix: once the merger loop's put()-failure except block clears
        the item's stamped token (``_real_item.cap_permit = None``) after
        releasing it through the ledger, (c) no longer sees a non-``None``
        cap_permit on the queued item at all — no transient false-positive.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        item = _make_spec_item(tmp_path, speculative=False)
        item.cap_permit = await worker._merge_ahead_ledger.acquire()
        worker._verifier_queue.put_nowait(item)
        worker._merge_ahead_ledger.release(item.cap_permit)
        item.cap_permit = None  # what the except-block fix now does

        assert worker.speculation_accounting_violations() == []


class TestRedispatchSpeculativeAccounting:
    """Ledger-based accounting for a speculative item's permit.

    Task 2063 originally closed the _redispatch-parked gap in the location-
    scanning census this class exercised; task 2160/η deletes that scan
    entirely, so this now pins the replacement ledger-only invariant: a
    permit acquired through the ledger is counted no matter where its item
    sits (including nowhere at all, i.e. off every worker-internal deque).

    RED (pre-η GREEN): _inflight_speculative_count() still scans only the
    five old locations, so a permit that lives solely in
    ``worker._speculation_ledger.live`` reads as 0 — see step-6 GREEN.
    """

    @pytest.mark.asyncio
    async def test_speculative_items_are_counted_via_the_ledger_alone(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # Acquire both permits THROUGH the ledger (no raw semaphore poking)
        # and attach each token to its item. Deliberately NOT placed on
        # worker._verifier_queue/_redispatch/etc — post-η the count must not
        # depend on structural location, only on ledger membership.
        item_a = _make_spec_item(tmp_path, speculative=True)
        item_a.permit = await worker._speculation_ledger.acquire()
        item_b = _make_spec_item(tmp_path, speculative=True)
        item_b.permit = await worker._speculation_ledger.acquire()

        assert worker._inflight_speculative_count() == 2
        assert worker.speculation_accounting_violations() == []
        assert worker.snapshot()['resource_audit']['speculation_accounting'] == []
        # Pin the OTHER consumer of _inflight_speculative_count() too: the
        # snapshot's 'speculation' key reads the same ledger-derived count,
        # so assert it directly rather than only exercising it transitively
        # via the resource_audit key above.
        assert worker.snapshot()['speculation']['inflight_speculative'] == 2

    def test_nonspeculative_redispatch_item_not_counted(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # A cascade-remerged item parked on _redispatch is non-speculative
        # (_remerge returns speculative=False; its permit was already
        # released before remerge, so .permit is None too) and must never be
        # counted — else the ledger-only invariant would be broken and could
        # mask a genuine imbalance.
        worker._redispatch.append(_make_spec_item(tmp_path, speculative=False))

        assert worker._inflight_speculative_count() == 0
        assert worker.speculation_accounting_violations() == []


# ---------------------------------------------------------------------------
# task 2096: _finalizing_head-held speculative permit accounting (superseded
# by task 2160/η — see the combined header comment above
# TestRedispatchSpeculativeAccounting).
#
# _finalize_inflight sets self._finalizing_head at the top of its try, awaits
# the ENTIRE `await entry.verify_task`, and only releases the speculation
# permit in its finally — so for the full verify duration (20-60 min
# observed on reify) a speculative head's permit was, pre-η, held solely by
# _finalizing_head and invisible to a location-scanning census. η replaces
# location-scanning with the ledger, so this window is now covered
# regardless of where the entry sits.
# ---------------------------------------------------------------------------


class TestFinalizeHeadSpeculativeAccounting:
    """Ledger-based accounting for a speculative permit held by a finalizing
    head — see the combined header comment above
    TestRedispatchSpeculativeAccounting for the task 2096 → task 2160/η
    context.
    """

    def _make_finalizing_entry(self, tmp_path: Path, *, speculative: bool, permit=None):
        from orchestrator.merge_queue import InflightEntry

        return InflightEntry(
            item=_make_spec_item(tmp_path, speculative=speculative),
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=speculative,
            permit=permit,
        )

    @pytest.mark.asyncio
    async def test_finalizing_head_permit_is_counted_via_the_ledger_alone(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import InflightEntry, SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # Acquire both permits THROUGH the ledger and attach each token to
        # its entry. Deliberately NOT placed on worker._inflight /
        # worker._finalizing_head — post-η the count must not depend on
        # structural location, only on ledger membership. The entries
        # themselves are never read again — leading underscores mark them as
        # intentionally unused beyond triggering the ledger acquisition.
        _inflight_entry = InflightEntry(
            item=_make_spec_item(tmp_path, speculative=True),
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=True,
            permit=await worker._speculation_ledger.acquire(),
        )
        _finalizing_entry = self._make_finalizing_entry(
            tmp_path, speculative=True, permit=await worker._speculation_ledger.acquire(),
        )

        assert worker._inflight_speculative_count() == 2
        assert worker.speculation_accounting_violations() == []
        assert worker.snapshot()['resource_audit']['speculation_accounting'] == []
        # Pin the OTHER consumer of _inflight_speculative_count() too (mirrors
        # the equivalent pin in TestRedispatchSpeculativeAccounting above).
        assert worker.snapshot()['speculation']['inflight_speculative'] == 2

    def test_nonspeculative_finalizing_head_not_counted(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        """A non-speculative finalizing head (the common case) carries no
        permit and contributes 0 — else the ledger-only invariant would be
        broken and could mask a genuine imbalance.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        worker._finalizing_head = self._make_finalizing_entry(tmp_path, speculative=False)

        assert worker._inflight_speculative_count() == 0
        assert worker.speculation_accounting_violations() == []

    def test_detector_not_blinded_by_ledger_based_census(self, git_ops: GitOps) -> None:
        """A genuinely dropped permit — the shared semaphore decremented
        directly, bypassing the ledger entirely, with the ledger left empty —
        must still trip exactly one violation: the ledger-only invariant must
        not mask the real leak class the audit exists to catch.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # Force a leak: a permit vanished from the shared semaphore without
        # being recorded as held-by-merger, acquired through the ledger, or
        # genuinely still available — breaks identity (a).
        worker._speculation_slot._value -= 1

        violations = worker.speculation_accounting_violations()

        assert len(violations) == 1, f'expected exactly one violation, got: {violations!r}'
        assert 'speculation-slot conservation violated' in violations[0]


# ---------------------------------------------------------------------------
# task 2096: dispatch-gap (in-dispatch item) speculative permit accounting
#
# _verifier_loop's DISPATCH-FILL pops an item off _redispatch/_verifier_queue
# then `entry = await self._dispatch_item(item)` — the item is off the queue
# but not yet appended to _inflight during that await (host-acquisition git
# calls, in-dispatch speculative-remerge). A speculative item's `.speculative`
# is stable across this await (Mechanism-2's chain-remerge is skipped for
# speculative items), so the gap is countable. Post-fix, self._dispatching_item
# (set immediately before the await, cleared in a bulletproof finally) closes
# this census gap the same way self._finalizing_head closed the finalize-head
# gap above.
# ---------------------------------------------------------------------------


class TestDispatchGapSpeculativeAccounting:
    """Ledger-based accounting for a speculative permit held by an in-
    dispatch item — see the combined header comment above
    TestRedispatchSpeculativeAccounting for the task 2096 → task 2160/η
    context.
    """

    @pytest.mark.asyncio
    async def test_dispatching_item_permit_is_counted_via_the_ledger_alone(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=1)

        # Acquire the single permit THROUGH the ledger and attach it to the
        # item. Deliberately NOT placed on worker._dispatching_item — post-η
        # the count must not depend on structural location, only on ledger
        # membership.
        item = _make_spec_item(tmp_path, speculative=True)
        item.permit = await worker._speculation_ledger.acquire()

        assert worker._inflight_speculative_count() == 1
        assert worker.speculation_accounting_violations() == []
        assert worker.snapshot()['speculation']['inflight_speculative'] == 1

    @pytest.mark.asyncio
    async def test_dispatching_item_plus_inflight_both_counted(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import InflightEntry, SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        # Acquire both permits THROUGH the ledger: one attached to a
        # dispatch-gap-shaped item, one to an _inflight entry. Deliberately
        # NOT placed on worker._dispatching_item / worker._inflight — post-η
        # the count must not depend on structural location, only on ledger
        # membership. The dispatch-gap item is never read again beyond
        # triggering the ledger acquisition — leading underscore marks it.
        _dispatching_shaped_item = _make_spec_item(tmp_path, speculative=True)
        _dispatching_shaped_item.permit = await worker._speculation_ledger.acquire()
        worker._inflight.append(InflightEntry(
            item=_make_spec_item(tmp_path, speculative=True),
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=True,
            permit=await worker._speculation_ledger.acquire(),
        ))

        assert worker._inflight_speculative_count() == 2
        assert worker.speculation_accounting_violations() == []
        assert worker.snapshot()['resource_audit']['speculation_accounting'] == []

    def test_nonspeculative_dispatching_item_not_counted(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        """A non-speculative in-dispatch item carries no permit and
        contributes 0 — else the ledger-only invariant would be broken and
        could mask a genuine imbalance.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        worker._dispatching_item = _make_spec_item(tmp_path, speculative=False)  # type: ignore[attr-defined]

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


# ---------------------------------------------------------------------------
# task 2159 step-5 RED / step-6 GREEN: worker's _speculation_ledger wiring
# ---------------------------------------------------------------------------


class TestSpeculationLedgerWiring:
    """SpeculativeMergeWorker exposes a PermitLedger wrapping
    ``_speculation_slot``, shared with ``_speculation_controller`` (task 2159
    step-5, DD5).

    RED until step-6 GREEN adds ``worker._speculation_ledger`` and threads it
    into the ``SpeculationController`` construction.
    """

    def test_worker_exposes_a_permit_ledger_wrapping_the_speculation_slot(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import PermitLedger, SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        assert isinstance(worker._speculation_ledger, PermitLedger)
        assert worker._speculation_ledger.slot_available == worker._speculation_slot._value

    def test_speculation_ledger_is_shared_with_the_controller(
        self, git_ops: GitOps,
    ) -> None:
        """Behavioral proof of sharing (not a private-attribute reach-in):
        an acquire driven through the controller must be visible on the
        worker's ``_speculation_ledger`` — only possible if both point at
        the same ``PermitLedger`` instance.
        """
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        asyncio.run(worker._speculation_controller.acquire_for_lookahead())

        assert worker._speculation_ledger.slot_available == 1
        assert len(worker._speculation_ledger.live) == 1

    def test_speculation_accounting_is_clean_initially_and_after_an_acquire(
        self, git_ops: GitOps,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)

        assert worker.speculation_accounting_violations() == []

        asyncio.run(worker._speculation_controller.acquire_for_lookahead())

        assert worker.speculation_accounting_violations() == []
