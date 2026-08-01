"""Tests for Harness._reconcile_stranded_in_progress and the _pid_alive helper."""

import asyncio
import json
import logging
import os
import re
import shutil
import time as _time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import mock_lock_table, pydantic_spec, wire_scheduler_liveness_mock
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.delivered_checks import DeliveredChecksBlock
from orchestrator.harness import Harness, _pid_alive
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.warm_lane_pool import WarmLanePool


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """MergeProvenance._outbox is a process-global — never leak a bound
    outbox into another test (mirrors test_workflow_merge_provenance.py's
    and test_task_ground_truth.py's identically-named fixture)."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


def _bind_landed_row(tmp_path: Path, *, task_id: str, advanced_sha: str) -> None:
    """Bind a real LandedOutbox (via MergeProvenance.bind) holding a row for
    *task_id* (mirrors test_task_ground_truth.py's identically-named helper)."""
    outbox = LandedOutbox(tmp_path / 'landed.json')
    outbox.record(LandedRow(
        task_id=task_id, branch_tip_sha='branchtip', advanced_sha=advanced_sha,
        landed_at=1.0,
    ))
    MergeProvenance.bind(outbox)


# ---------------------------------------------------------------------------
# _pid_alive helper tests
# ---------------------------------------------------------------------------

class TestPidAlive:
    def test_current_pid_is_alive(self):
        assert _pid_alive(os.getpid()) is True

    def test_impossible_pid_is_dead(self):
        # PID well beyond the Linux kernel max (2^22 on 64-bit, 2^15 on 32-bit).
        # 2**31-1 is always invalid on all Linux systems.
        assert _pid_alive(2**31 - 1) is False

    # ------------------------------------------------------------------
    # Branch-mocked tests — each covers exactly one code path in _pid_alive
    # ------------------------------------------------------------------

    def test_pid_zero_returns_false_without_calling_os_kill(self, monkeypatch):
        """pid=0 guard → returns False before os.kill is ever called."""
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(os, 'kill', lambda pid, sig: calls.append((pid, sig)))
        assert _pid_alive(0) is False
        assert calls == [], 'os.kill must not be called for pid=0'

    def test_negative_pid_returns_false_without_calling_os_kill(self, monkeypatch):
        """pid=-1 guard → returns False before os.kill is ever called."""
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(os, 'kill', lambda pid, sig: calls.append((pid, sig)))
        assert _pid_alive(-1) is False
        assert calls == [], 'os.kill must not be called for pid=-1'

    def test_process_lookup_error_returns_false(self, monkeypatch):
        """os.kill raises ProcessLookupError → process is dead → False."""
        def _raise(pid: int, sig: int) -> None:
            raise ProcessLookupError()
        monkeypatch.setattr(os, 'kill', _raise)
        assert _pid_alive(12345) is False

    def test_permission_error_returns_true(self, monkeypatch):
        """os.kill raises PermissionError → process exists (no permission to signal) → True."""
        def _raise(pid: int, sig: int) -> None:
            raise PermissionError()
        monkeypatch.setattr(os, 'kill', _raise)
        assert _pid_alive(12345) is True

    def test_generic_oserror_returns_false(self, monkeypatch):
        """os.kill raises generic OSError → treat as dead → False."""
        def _raise(pid: int, sig: int) -> None:
            raise OSError(5, 'io error')
        monkeypatch.setattr(os, 'kill', _raise)
        assert _pid_alive(12345) is False

    def test_successful_signal_returns_true(self, monkeypatch):
        """os.kill succeeds → process is alive → True."""
        monkeypatch.setattr(os, 'kill', lambda pid, sig: None)
        assert _pid_alive(12345) is True


# ---------------------------------------------------------------------------
# Harness fixture (mirrors test_crash_recovery.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Create a Harness with mocked internals for unit testing reconciliation."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    # Replace scheduler with async mocks
    h.scheduler = MagicMock()
    # Wire real (non-auto-mock) liveness-accessor behaviour (task 2235:
    # harness.py now calls scheduler.is_dispatched/.is_actively_held/
    # .workflow_cancel_recent/.note_workflow_cancelled/.clear_workflow_cancel
    # instead of reaching into _dispatched/lock_table._held/_workflow_cancel_at
    # directly) so the mid-run sweep and stranded-blocked gate tests below
    # exercise real semantics instead of an auto-mocked (always-truthy) stub.
    wire_scheduler_liveness_mock(h.scheduler)
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()

    # mark_done forwards to set_task_status so the existing assertions on
    # set_task_status.call_args_list / set_task_status.assert_awaited_once_with
    # still cover the recovery-done path after the harness was refactored to
    # use Scheduler.mark_done as a thin wrapper.
    async def _fake_mark_done(tid, *, kind, sha, note=None):
        provenance = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await h.scheduler.set_task_status(
            tid, 'done', done_provenance=provenance,
        )
    h.scheduler.mark_done = AsyncMock(side_effect=_fake_mark_done)

    # Keep worktree_base real (under tmp_path) so we can create fake worktrees
    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()

    # Mock cleanup_worktree: side_effect actually removes the directory so that
    # existing assertions like `assert not lock_path.exists()` continue to hold
    # after the impl switches from lock_path.unlink() to cleanup_worktree().
    def _fake_cleanup(worktree_path, tid):
        shutil.rmtree(worktree_path, ignore_errors=True)

    h.git_ops.cleanup_worktree = AsyncMock(side_effect=_fake_cleanup)

    # Default: is_ancestor returns False so no guard fires for existing tests.
    # Individual tests may override with AsyncMock(return_value=True).
    h.git_ops.is_ancestor = AsyncMock(return_value=False)

    # Default: resolve_branch_sha returns a fixed SHA so tests that trigger the
    # is_ancestor guard get a consistent commit in done_provenance.
    # Individual tests may override with AsyncMock(return_value=None).
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeef' + 'a' * 32)

    # Default: find_merge_marker returns None so no deleted-branch guard fires
    # for existing tests.  Individual tests may override with
    # AsyncMock(return_value='<sha>') to exercise the marker path.
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)

    # Default: find_task_citation_commit returns a SHA matching the
    # resolve_branch_sha default so existing happy-path tests pass through the
    # new citation guard unchanged.  Individual tests may override with
    # AsyncMock(return_value=None) to exercise the missing-citation path or
    # with a different SHA to assert citation precedence over branch-tip.
    h.git_ops.find_task_citation_commit = AsyncMock(
        return_value='deadbeef' + 'a' * 32,
    )

    # Default: get_task synthesizes {'status': ..., 'metadata': {}} from
    # whatever get_statuses() is currently configured to return for that tid
    # (task 2243, W10-θ2: TaskGroundTruth.derive_truth's db_status is sourced
    # from get_task(tid)['status'] — a DIFFERENT accessor than the per-tid
    # `status` string the sweep loop itself reads from get_statuses(). In
    # production both reflect the same row; the fixture keeps them
    # consistent by construction here rather than requiring every test to
    # duplicate the status in two places). Falls back to None (no row) when
    # get_statuses' return_value isn't a plain (dict, err) tuple or has no
    # entry for tid — matching the old unconditional-None default for any
    # test that doesn't care. Individual tests that need branch_base_sha (or
    # any other task-row shape) override get_task explicitly and are
    # responsible for including a matching 'status' key themselves.
    def _default_get_task(tid: str) -> dict | None:
        try:
            statuses, _err = cast(AsyncMock, h.scheduler.get_statuses).return_value
        except (TypeError, ValueError):
            return None
        status = statuses.get(tid) if isinstance(statuses, dict) else None
        return None if status is None else {'status': status, 'metadata': {}}
    h.scheduler.get_task = AsyncMock(side_effect=_default_get_task)

    # Default: warm_lane_ref_is_degenerate returns False (task 2112 angle B) so
    # the metadata-independent fallback never fires for existing tests and the
    # suite stays deterministic (no real subprocess is ever invoked in unit
    # tests). Individual tests may override with AsyncMock(return_value=True)
    # to exercise the primitive-degenerate fallback path.
    h.git_ops.warm_lane_ref_is_degenerate = AsyncMock(return_value=False)

    # Default: commit_effect_present_in_main returns True (task 2500 FIX 1)
    # so existing found_on_main mark-done tests reach the flip unchanged.
    # Individual tests may override with AsyncMock(return_value=False) to
    # exercise the post-hoc-revert blind-spot guard.
    h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)

    return h


# ---------------------------------------------------------------------------
# task 2243 (W10-θ2) step-3 — the G1+G2 MARK_DONE signal.
#
# _reconcile_one_stranded delegates the mark-done decision to
# TaskGroundTruth.recovery_for (θ1, task_ground_truth.py), which resolves
# branch state JOURNAL-FIRST (MergeProvenance.lookup) ahead of git
# archaeology (TG-1). G1 pins the journal-hit path (no git I/O at all). G2
# pins the journal-MISS path, which falls back to the exact same
# is_ancestor -> resolve_branch_sha -> find_merge_marker sequence the
# pre-migration sweep already ran, so it recovers identically — this is a
# parity/regression pin (it already holds against the pre-migration code
# too), not a novel-behavior RED case like G1.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileOneStrandedGroundTruthMarkDone:
    async def test_g1_journal_hit_marks_done_without_consulting_merge_marker(
        self, harness: Harness, tmp_path: Path,
    ):
        """A stranded in-progress task with a MergeProvenance journal row on
        main recovers to done via the journal SHA in ONE derive_truth ->
        _RECOVERY step — the resolver never falls back to git archaeology
        (TG-1: journal-first)."""
        tid = '9001'
        advanced_sha = 'a1' * 20  # distinct sentinel — never emitted by the git-archaeology mocks
        _bind_landed_row(tmp_path, task_id=tid, advanced_sha=advanced_sha)

        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )

        await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': advanced_sha,
                'note': ANY,
            },
        )
        # Journal-first: neither git primitive is ever consulted on a hit.
        harness.git_ops.find_merge_marker.assert_not_awaited()  # type: ignore[attr-defined]
        harness.git_ops.is_ancestor.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_g2_journal_miss_falls_back_to_merge_marker(
        self, harness: Harness,
    ):
        """No MergeProvenance journal row (journal miss) — the resolver's
        git fallback finds a merge marker and recovers identically: done,
        with the marker SHA as the provenance commit."""
        tid = '9002'
        marker_sha = 'b2' * 20
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )
        # is_ancestor False (fixture default) -> branch ref gone -> marker hit.
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': ANY,
            },
        )
        harness.git_ops.is_ancestor.assert_awaited_once_with(  # type: ignore[attr-defined]
            f'task/{tid}', 'main',
        )


# ---------------------------------------------------------------------------
# task 2500 FIX 1 — the found_on_main post-hoc-revert blind spot.
#
# A cited ON_MAIN commit remains an ancestor of main forever (ancestry is
# immutable history) even after a LATER commit on main reverts exactly the
# paths it touched. TaskGroundTruth's resolver only ever sees is_ancestor,
# so it still classifies MARK_DONE_WITH_PROVENANCE; the sweep-side
# effect-present refinement (mirroring the existing degenerate-branch
# refinement) is the only place this blind spot gets caught.
#
# task 2678 extends this refinement to the GONE_WITH_MERGE_MARKER sub-case
# too (routed through validate_landing_evidence in CANDIDATE mode on
# report.branch_state.sha, applied uniformly to both sub-cases): previously
# the inline check here was gated on ``on_main``, so a branch-deleted merge
# marker whose effect had been reverted at current main HEAD skipped the
# guard entirely and was stamped done unconditionally — the task-1175
# clobber shape, reproduced inside this sweep.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileOneStrandedEffectPresentGuard:
    async def test_on_main_effect_absent_reverts_instead_of_marking_done(
        self, harness: Harness,
    ):
        """Git-fallback ON_MAIN evidence (is_ancestor True) whose effect was
        reverted at current main HEAD must NOT be marked done — it reverts
        to pending like the degenerate-branch case, so the scheduler
        re-dispatches it."""
        tid = '9010'
        recovered_sha = 'deadbeef' + 'a' * 32
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )
        # Git-fallback ON_MAIN: is_ancestor(branch, main) True, resolve_branch_sha
        # returns the recovered sha (non-degenerate: no branch_base_sha in metadata).
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=recovered_sha)  # type: ignore[attr-defined]
        # The cited commit's effect was reverted at current main HEAD.
        harness.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.commit_effect_present_in_main.assert_awaited_once_with(  # type: ignore[attr-defined]
            recovered_sha,
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending',
        )
        assert result == 1

    async def test_marker_effect_present_marks_done_and_effect_check_consulted(
        self, harness: Harness,
    ):
        """Branch-deleted merge-marker evidence (GONE_WITH_MERGE_MARKER)
        whose effect IS present at current main HEAD still marks done —
        exactly like before task 2678 — but now commit_effect_present_in_main
        is actually CONSULTED for this sub-case (previously it was gated on
        ``on_main`` and never reached the marker sha at all)."""
        tid = '9011'
        marker_sha = 'c3' * 20
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )
        # Git-fallback marker sub-case: is_ancestor(branch, main) False
        # (fixture default), resolve_branch_sha None (branch ref gone),
        # find_merge_marker returns the marker sha.
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)  # type: ignore[attr-defined]
        # commit_effect_present_in_main defaults True (fixture) — a healthy
        # marker landing.

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': ANY,
            },
        )
        harness.git_ops.commit_effect_present_in_main.assert_awaited_once_with(  # type: ignore[attr-defined]
            marker_sha,
        )
        assert result == 1

    async def test_marker_effect_absent_in_progress_reverts_instead_of_marking_done(
        self, harness: Harness,
    ):
        """The task-1175 shape reproduced inside the sweep: a branch-deleted
        merge marker on main whose effect was reverted at current main HEAD
        (commit_effect_present_in_main False) must NOT be marked done. An
        in-progress task with no live claimant reverts to pending instead —
        symmetric with the ON_MAIN effect-absent case above."""
        tid = '9012'
        marker_sha = 'c4' * 20
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)  # type: ignore[attr-defined]
        harness.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.commit_effect_present_in_main.assert_awaited_once_with(  # type: ignore[attr-defined]
            marker_sha,
        )
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending',
        )
        assert result == 1

    async def test_marker_effect_absent_blocked_leaves_task_untouched(
        self, harness: Harness,
    ):
        """Same task-1175 shape but for a stranded 'blocked' task reaching
        MARK_DONE_WITH_PROVENANCE via the R4 sweep-side upgrade: blocked
        discipline forbids a silent blocked->pending revert (mirrors
        TestReconcileOneStrandedDegenerateBranchParity's blocked-degenerate
        case), so an effect-absent marker must resolve to no action at all
        rather than a phantom mark_done."""
        tid = '9013'
        marker_sha = 'c5' * 20
        harness.scheduler.get_statuses.return_value = ({tid: 'blocked'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'blocked', 'metadata': {}},
        )
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)  # type: ignore[attr-defined]
        harness.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0


# ---------------------------------------------------------------------------
# task 2243 (W10-θ2) step-5 — the REVERT_TO_PENDING signal.
#
# _reconcile_one_stranded dispatches on TaskGroundTruth.recovery_for's
# classification: a stranded in-progress task with no live claimant and no
# on-main landing evidence (_RECOVERY rows c/d — BranchStateKind
# EXISTS_OFF_MAIN or GONE_NO_MARKER) resolves to RecoveryAction.
# REVERT_TO_PENDING, dispatched straight to
# _revert_in_progress_if_no_live_claimant.  Both cases already produced this
# outcome pre-migration (the applier's own plan.lock liveness check is
# unchanged), so — like G2 above — these are parity/regression pins for the
# new explicit dispatch, not novel-behavior RED cases.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileOneStrandedGroundTruthRevert:
    async def test_c_exists_off_main_no_live_claimant_reverts(
        self, harness: Harness,
    ):
        """Branch ref still exists but is not an ancestor of main (no landing
        evidence) — recovery_for classifies REVERT_TO_PENDING; the task is
        reverted to pending and mark_done is never called."""
        tid = '9003'
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )
        # Fixture defaults already produce EXISTS_OFF_MAIN: is_ancestor=False,
        # resolve_branch_sha=<a sha> (ref exists, not an ancestor). Restated
        # explicitly here so the scenario is self-contained.
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(  # type: ignore[attr-defined]
            return_value='deadbeef' + 'a' * 32,
        )

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending',
        )
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 1

    async def test_d_gone_no_marker_no_live_claimant_reverts(
        self, harness: Harness,
    ):
        """Branch ref gone, no journal row, no merge marker on main (no
        landing evidence) — recovery_for classifies REVERT_TO_PENDING; the
        task is reverted to pending and mark_done is never called."""
        tid = '9004'
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending',
        )
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 1


# ---------------------------------------------------------------------------
# Review amendment (task 2243 W10-θ2, reviewer_comprehensive #2) —
# deploy-phase LEAVE parity.
#
# theta1's _RECOVERY table (task_ground_truth.py) requires deploy_phase is
# None for EVERY MARK_DONE_WITH_PROVENANCE / REVERT_TO_PENDING row — so a
# stranded in-progress task carrying a deploy_phase can only be classified
# LEAVE (the common case: VERIFIED / FAILED / SCHEDULED / ESCALATED / DONE,
# or RAN paired with any branch state other than GONE_NO_MARKER) or
# RE_FILE_ESCALATION (row h: GONE_NO_MARKER + RAN, the D1 crashed-mid-deploy
# shape). Both must leave the task untouched: DS-2 (the deploy-phase state
# machine) owns its own mandatory recovery path, and reverting (or
# phantom-completing) a stranded deploy could re-trigger one that already
# took effect. Before this amendment, both actions fell through the generic
# in-progress tail to _revert_in_progress_if_no_live_claimant, silently
# overriding the table's LEAVE/defer-to-DS-2 intent.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileOneStrandedDeployPhaseLeaveParity:
    async def test_failed_deploy_phase_off_main_left_alone_not_reverted(
        self, harness: Harness,
    ):
        """A stranded in-progress task in a FAILED deploy phase, off-main, no
        live claimant, no open escalation — the table's deploy_phase gap
        classifies this LEAVE (not REVERT_TO_PENDING, which requires
        deploy_phase is None). Must not be reverted: DS-2 owns FAILED's own
        mandatory recovery path (FAILED -> ESCALATED)."""
        tid = '9008'
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                'status': 'in-progress',
                'metadata': {'deploy_state': {'phase': 'failed'}},
            },
        )
        # EXISTS_OFF_MAIN: ref present, not an ancestor of main.
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(  # type: ignore[attr-defined]
            return_value='deadbeef' + 'a' * 32,
        )

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0

    async def test_verified_deploy_phase_on_main_left_alone_not_marked_done(
        self, harness: Harness,
    ):
        """A stranded in-progress task in a VERIFIED (terminal-success)
        deploy phase, even WITH on-main branch evidence, is left alone —
        deploy_phase != None excludes it from every MARK_DONE row too, so
        this is a deliberate no-op, not a missed done-flip."""
        tid = '9009'
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                'status': 'in-progress',
                'metadata': {'deploy_state': {'phase': 'verified'}},
            },
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(  # type: ignore[attr-defined]
            return_value='deadbeef' + 'a' * 32,
        )

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0

    async def test_ran_deploy_phase_gone_no_marker_left_alone_not_reverted(
        self, harness: Harness,
    ):
        """D1's crashed-mid-deploy shape (row h: RAN + GONE_NO_MARKER) also
        must not be reverted by this generic reaper — the table routes it to
        RE_FILE_ESCALATION (owned elsewhere, not this in-progress path),
        never a silent revert that could re-run an in-flight deploy."""
        tid = '9010'
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                'status': 'in-progress',
                'metadata': {'deploy_state': {'phase': 'ran'}},
            },
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0


# ---------------------------------------------------------------------------
# task 2243 (W10-θ2) step-7 — degenerate-branch parity (RESOLVER GAP).
#
# θ1's _RECOVERY table has no degenerate-branch guard: a stranded task whose
# branch is an ancestor of main but carries ZERO commits beyond its creation
# point (tip == branch_base_sha) classifies ON_MAIN -> MARK_DONE, same as a
# genuinely merged branch.  The migrated sweep keeps a thin degenerate
# refinement (design decision, task 2243) that downgrades this to a revert
# — pre-migration behavior, preserved.  Covers both the metadata-based
# signal (_branch_is_degenerate) and the metadata-independent primitive
# fallback (warm_lane_ref_is_degenerate).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileOneStrandedDegenerateBranchParity:
    async def test_degenerate_branch_via_metadata_reverts_instead_of_done(
        self, harness: Harness,
    ):
        """ON_MAIN evidence (is_ancestor=True) but the branch tip equals the
        recorded branch_base_sha (zero commits pushed) — the metadata-based
        degenerate signal downgrades MARK_DONE_WITH_PROVENANCE to a revert."""
        tid = '9005'
        degenerate_sha = 'aabbccdd' + 'e' * 32
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                'status': 'in-progress',
                'metadata': {'branch_base_sha': degenerate_sha},
            },
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=degenerate_sha)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending',
        )
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 1

    async def test_degenerate_branch_via_primitive_reverts_instead_of_done(
        self, harness: Harness,
    ):
        """ON_MAIN evidence but no branch_base_sha in metadata (the
        metadata-based signal can't fire) — the metadata-independent
        warm_lane_ref_is_degenerate primitive still downgrades to a revert."""
        tid = '9006'
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': {}},
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(  # type: ignore[attr-defined]
            return_value='deadbeef' + 'a' * 32,
        )
        harness.git_ops.warm_lane_ref_is_degenerate = AsyncMock(return_value=True)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending',
        )
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 1

    async def test_blocked_degenerate_branch_leaves_task_untouched(
        self, harness: Harness,
    ):
        """R4 upgrades a stranded 'blocked' task with ON_MAIN evidence to
        MARK_DONE_WITH_PROVENANCE — but a degenerate branch carries no real
        work, and blocked discipline forbids a silent blocked->pending
        revert, so this must resolve to no action at all (matching
        pre-migration behavior): no mark_done, no status change."""
        tid = '9007'
        degenerate_sha = 'aabbccdd' + 'e' * 32
        harness.scheduler.get_statuses.return_value = ({tid: 'blocked'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                'status': 'blocked',
                'metadata': {'branch_base_sha': degenerate_sha},
            },
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=degenerate_sha)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0


# ---------------------------------------------------------------------------
# task 2787 — the 2729/2753 foreign-branch-tip recurrence.
#
# A stranded in-progress task on a ZERO-COMMIT branch whose tip == the main
# HEAD captured at branch-create time (an UNRELATED task's merge commit).
# is_ancestor(branch, main) is trivially True, so the resolver's ON_MAIN
# git-archaeology fast-path returns that FOREIGN commit as the "landing" sha.
# _branch_is_degenerate no-ops (no branch_base_sha in metadata) and
# warm_lane_ref_is_degenerate is False, so neither degeneracy signal catches
# it; the foreign (genuinely-landed) commit's effect IS present at main, so the
# CANDIDATE-mode effect-present guard passes too. Pre-fix the task
# phantom-completes stamped with the foreign SHA. The resolver's
# positive-citation attribution guard (task 2787) rejects the un-attributable
# trivial-ancestor branch -> EXISTS_OFF_MAIN -> REVERT_TO_PENDING
# (re-dispatch), never a found_on_main stamp of the foreign commit.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileOneStrandedForeignBranchTipGuard:
    async def test_foreign_branch_tip_no_citation_reverts_not_marked_done(
        self, harness: Harness,
    ):
        """The 2729/2753 recurrence: a zero-commit branch trivially an ancestor
        of main (tip == a foreign task's merge commit) with NO commit on main
        citing this task must NOT be stamped found_on_main with that foreign
        SHA — it reverts to pending (re-dispatch) instead.

        RED on current main: the resolver classifies ON_MAIN(foreign tip), no
        degeneracy signal fires, the foreign effect is present, so the sweep
        marks done with the foreign SHA."""
        tid = '2729'
        # The branch tip is an UNRELATED task's merge commit (the main HEAD at
        # branch-create time), NOT any commit this task produced.
        foreign_sha = 'fac4c813' + 'd' * 32
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        # No branch_base_sha in metadata (fixture synthesizes
        # {'status': 'in-progress', 'metadata': {}} from get_statuses) ->
        # _branch_is_degenerate no-ops.
        # Zero-commit branch is trivially an ancestor of main; its tip is the
        # foreign merge commit.
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=foreign_sha)  # type: ignore[attr-defined]
        # NO commit on main cites 2729 — it never did any work.
        harness.git_ops.find_task_citation_commit = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        # Neither degeneracy signal fires (no branch_base_sha; primitive False),
        # and the foreign commit's effect genuinely IS present at main — exactly
        # the gap that let the foreign SHA through pre-fix.
        harness.git_ops.warm_lane_ref_is_degenerate = AsyncMock(return_value=False)  # type: ignore[attr-defined]
        harness.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)  # type: ignore[attr-defined]

        result = await harness._reconcile_stranded_in_progress()

        # The foreign commit is NEVER stamped as this task's landing.
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending',
        )
        # Belt-and-suspenders: prove no set_task_status call carried a
        # found_on_main provenance (which would cite the foreign SHA).
        for call in harness.scheduler.set_task_status.call_args_list:  # type: ignore[attr-defined]
            provenance = call.kwargs.get('done_provenance')
            assert provenance is None, (
                f'foreign SHA must never be stamped as a landing; got {provenance!r}'
            )
        assert result == 1


# ---------------------------------------------------------------------------
# _reconcile_stranded_in_progress tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileStrandedInProgress:
    async def test_orphan_without_worktree_reverted(self, harness: Harness):
        """In-progress task with no worktree dir → reverted to pending (no-lock)."""
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'5': 'in-progress', '6': 'pending'}, None
        )
        # No worktree directory for task 5 exists (worktree_base not even created)

        await harness._reconcile_stranded_in_progress()

        calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
        assert len(calls) == 1
        assert calls[0].args[0] == '5'
        assert calls[0].args[1] == 'pending'

    async def test_in_progress_with_live_owner_pid_left_alone(
        self, harness: Harness, tmp_path: Path, caplog
    ):
        """In-progress task with plan.lock pointing to live PID → untouched, no revert logged."""
        harness.scheduler.get_statuses.return_value = ({'7': 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create worktree with a plan.lock containing our own (live) PID
        lock_dir = harness.git_ops.worktree_base / '7' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '7-abcd1234',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': os.getpid(),
        }))

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # Must NOT revert
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        # Lock file must still exist
        assert lock_path.exists()
        # No revert must have been logged ('reverted task' matches the stable log format)
        assert not any('reverted task' in r.message for r in caplog.records)

    async def test_stale_plan_lock_cleared_and_reverted(
        self, harness: Harness, monkeypatch
    ):
        """In-progress task with stale plan.lock (dead PID) → lock cleared and task reverted."""
        harness.scheduler.get_statuses.return_value = ({'8': 'in-progress'}, None)  # type: ignore[attr-defined]
        # Use a synthetic owner_pid — _pid_alive is mocked to always return False,
        # so no real PID is needed and there is no kernel-recycle race.
        owner_pid = 99999
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create worktree with plan.lock referencing the synthetic dead PID
        lock_dir = harness.git_ops.worktree_base / '8' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '8-dead0001',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': owner_pid,
        }))

        await harness._reconcile_stranded_in_progress()

        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with('8', 'pending')  # type: ignore[attr-defined]
        # Stale lock must be deleted
        assert not lock_path.exists()

    async def test_stale_plan_lock_but_live_db_claimant_left_alone(
        self, harness: Harness, monkeypatch, caplog
    ):
        """Task-2588 regression pin.

        plan.lock/owner_pid forensics alone would call this task unclaimed
        (dead owner_pid — same staging as test_stale_plan_lock_cleared_and_
        reverted above) — the pre-2243 sweep's actual root cause for the
        2588 un-claim (a live task reverted out from under its own live
        claimant). But get_task returns a FRESH db claimant
        (claimant_run_id + heartbeat within _RECONCILE_HEARTBEAT_TTL), so
        recovery_for's live_claimant resolution defers to the db signal and
        classifies LEAVE: _revert_in_progress_if_no_live_claimant must
        never be dispatched — no revert, no un-claim, lock preserved.
        """
        harness.scheduler.get_statuses.return_value = ({'62': 'in-progress'}, None)  # type: ignore[attr-defined]
        # Same dead-PID staging as test_stale_plan_lock_cleared_and_reverted:
        # plan.lock/owner_pid forensics alone would read "stale".
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)
        lock_dir = harness.git_ops.worktree_base / '62' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '62-dead0001',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': 99999,
        }))

        # But the DB row carries a fresh claimant — the live cross-process
        # signal task 2243 made primary over plan.lock/owner_pid forensics.
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                'status': 'in-progress',
                'metadata': {},
                'claimant_run_id': 'run-x/session-x/pid=123',
                'heartbeat_at': datetime.now(UTC).isoformat(),
            },
        )

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # Must NOT revert or un-claim.
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        # plan.lock must be preserved.
        assert lock_path.exists()
        # Worktree must not be cleaned up.
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        # No revert must have been logged ('reverted task' matches the
        # stable log format).
        assert not any('reverted task' in r.message for r in caplog.records)

    async def test_in_progress_with_open_l1_left_intact(
        self, harness: Harness, monkeypatch
    ):
        """In-progress task with an open L1 escalation → worktree NOT reaped.

        Reproduces the /unblock-session reap: an escalated task sits at
        'in-progress' with an open L1 while a human edits its worktree.  The
        worktree's plan.lock is stale (the agent exited), so WITHOUT the L1
        guard the stranded sweep would clear the lock, force-delete the
        worktree, and revert to pending.  The open L1 must veto all of that —
        return None, no cleanup_worktree, no status change, lock preserved.
        """
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        harness.scheduler.get_statuses.return_value = ({'60': 'in-progress'}, None)  # type: ignore[attr-defined]

        # Stale lock (dead PID) — the would-be-reaped shape absent the guard.
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)
        lock_dir = harness.git_ops.worktree_base / '60' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '60-human',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': 99999,
        }))

        # Real EscalationQueue with an open L1 for task 60 (the human handoff).
        queue_dir = harness.git_ops.worktree_base.parent / 'escalations'
        harness._escalation_queue = EscalationQueue(queue_dir)
        harness._escalation_queue.submit(Escalation(
            id=harness._escalation_queue.make_id('60'),
            task_id='60',
            agent_role='task-steward',
            severity='blocking',
            category='task_failure',
            summary='Escalated — human is unblocking in the worktree',
            level=1,
            status='pending',
        ))

        await harness._reconcile_stranded_in_progress()

        # The worktree must be left intact: no cleanup, no revert, lock present.
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        assert lock_path.exists()

    async def test_in_progress_without_l1_still_reaped(
        self, harness: Harness, monkeypatch
    ):
        """Inverse of the L1 guard: a stale-lock in-progress task with NO open
        L1 for *that* task is still reaped — pins the guard to the specific tid.

        An L1 exists for a *different* task (999); the target (61) has none, so
        has_open_l1('61') is False and the stale-lock recovery proceeds.
        """
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        harness.scheduler.get_statuses.return_value = ({'61': 'in-progress'}, None)  # type: ignore[attr-defined]

        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)
        lock_dir = harness.git_ops.worktree_base / '61' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '61-dead',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': 99999,
        }))

        queue_dir = harness.git_ops.worktree_base.parent / 'escalations'
        harness._escalation_queue = EscalationQueue(queue_dir)
        # L1 belongs to an unrelated task — must NOT shield task 61.
        harness._escalation_queue.submit(Escalation(
            id=harness._escalation_queue.make_id('999'),
            task_id='999',
            agent_role='task-steward',
            severity='blocking',
            category='task_failure',
            summary='Unrelated escalation',
            level=1,
            status='pending',
        ))

        await harness._reconcile_stranded_in_progress()

        # No L1 for 61 → normal stale-lock recovery: reaped + reverted.
        harness.git_ops.cleanup_worktree.assert_awaited_once()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_called_once_with('61', 'pending')  # type: ignore[attr-defined]
        assert not lock_path.exists()

    @pytest.mark.parametrize(
        'lock_contents,task_id,expect_reverted,expect_lock_exists,warn_pattern',
        [
            # (a) Corrupt JSON → task 2763: the applier no longer READS the
            #     worktree lock for liveness, so a corrupt lock no longer fails
            #     closed. recovery_for already ruled out a live claimant →
            #     REVERT: cleanup_worktree called, task reverted, lock gone,
            #     no ERROR.
            pytest.param(
                'not-valid-json', 9, True, False,
                None,
                id='corrupt-json',
            ),
            # (b) Missing owner_pid key → reverted. Task 2763: the applier no
            #     longer reads the lock, so the old 'no owner_pid; treating as
            #     stale' observability WARNING is gone (warn_pattern=None); the
            #     task still reverts (cleanup_worktree called, lock gone).
            pytest.param(
                json.dumps({'session_id': 'test-10', 'locked_at': '2026-01-01T00:00:00+00:00'}),
                '10', True, False, None,
                id='missing-owner-pid',
            ),
            # (b2) Explicit null owner_pid → reverted, same as (b): no warning
            #      (applier no longer reads the lock), cleanup called, lock gone.
            pytest.param(
                json.dumps({'session_id': 'test-16', 'locked_at': '2026-01-01T00:00:00+00:00', 'owner_pid': None}),
                16, True, False, None,
                id='null-owner-pid',
            ),
            # (b3) Non-numeric owner_pid → int('abc') raises ValueError
            #      → except (TypeError, ValueError) catches it → owner_alive=False
            #      → stale-lock path: cleanup_worktree called, task reverted, lock gone
            pytest.param(
                json.dumps({'session_id': 'test-42', 'locked_at': '2026-01-01T00:00:00+00:00', 'owner_pid': 'abc'}),
                42, True, False, None,
                id='non-numeric-owner-pid',
            ),
            # (e) Non-dict JSON (list) → task 2763: same as (a). The applier no
            #     longer reads the lock, so a non-dict lock no longer fails
            #     closed → REVERT: cleanup_worktree called, task reverted, lock
            #     gone, no ERROR.
            pytest.param(
                '["not", "an", "object"]', 14, True, False,
                None,
                id='non-dict-json',
            ),
            # (c) Numeric-string owner_pid of a live process → task IS reverted.
            #     Task 2243, W10-θ2 step-16: the applier no longer re-derives
            #     plan.lock owner_pid liveness (formerly this case's live pid
            #     short-circuited the revert here) — recovery_for's
            #     live_claimant already ruled out a live claimant (this
            #     lock's locked_at is far outside heartbeat_ttl, so the
            #     resolver treats it as stale regardless of pid liveness)
            #     before REVERT_TO_PENDING reaches the applier.
            pytest.param(
                'LIVE_PID', 11, True, False, None,
                id='live-pid-as-string',
            ),
            # (d1) No lock file, id as int → reverted via no-lock branch
            pytest.param(None, 12, True, False, None, id='no-lock-int-id'),
            # (d2) No lock file, id as str → reverted via no-lock branch
            pytest.param(None, '13', True, False, None, id='no-lock-str-id'),
        ],
    )
    async def test_reconcile_lock_format_variants(
        self,
        harness: Harness,
        caplog,
        lock_contents,
        task_id,
        expect_reverted: bool,
        expect_lock_exists: bool,
        warn_pattern,
    ):
        """Parametrized coverage of plan.lock format edge cases."""
        import logging

        harness.scheduler.get_statuses.return_value = ({str(task_id): 'in-progress'}, None)  # type: ignore[attr-defined]

        tid_str = str(task_id)
        lock_dir = harness.git_ops.worktree_base / tid_str / '.task'
        lock_path = lock_dir / 'plan.lock'

        if lock_contents is not None:
            # Resolve sentinel for live-PID case
            if lock_contents == 'LIVE_PID':
                lock_contents = json.dumps({
                    'session_id': f'{tid_str}-live',
                    'locked_at': '2026-01-01T00:00:00+00:00',
                    'owner_pid': str(os.getpid()),
                })
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(lock_contents)

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
        if expect_reverted:
            assert len(calls) == 1, f'Expected 1 revert call, got: {calls}'
            assert calls[0].args[0] == tid_str, (
                f'Expected set_task_status called with id={tid_str!r}, got {calls[0].args[0]!r}'
            )
            assert calls[0].args[1] == 'pending'
        else:
            assert len(calls) == 0, f'Expected no calls (task untouched), got: {calls}'

        assert lock_path.exists() == expect_lock_exists, (
            f'Lock file existence mismatch: expected {expect_lock_exists}, '
            f'got {lock_path.exists()}'
        )

        if warn_pattern is not None:
            matching = [
                r for r in caplog.records
                if re.search(warn_pattern, r.message, re.IGNORECASE)
            ]
            assert len(matching) >= 1, (
                f'Expected log record matching {warn_pattern!r} in orchestrator.harness logs, '
                f'got: {[r.message for r in caplog.records]}'
            )
            # Corrupt/unreadable lock on startup → ERROR (task stranded indefinitely,
            # operator action required). Other warn_pattern cases (missing/null owner_pid)
            # remain at WARNING level.
            expected_level = (
                logging.ERROR
                if re.search(r'(unreadable|corrupt).*leaving worktree intact', warn_pattern)
                else logging.WARNING
            )
            assert matching[0].levelno == expected_level, (
                f'Expected {logging.getLevelName(expected_level)} level, '
                f'got {logging.getLevelName(matching[0].levelno)}'
            )

        # Verify cleanup_worktree call behavior.
        # When a worktree was created on disk (lock_contents is not None) and the
        # task was reverted, cleanup_worktree must have been called with the correct
        # args.  When no worktree exists (no-lock-*-id cases, lock_contents=None) or
        # the task was left alone (live-pid-as-string), cleanup_worktree must not fire.
        worktree_path = harness.git_ops.worktree_base / tid_str
        if expect_reverted and lock_contents is not None:
            harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
                worktree_path, tid_str
            )
        else:
            harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_corrupt_plan_lock_reverted_on_startup(
        self, harness: Harness, caplog
    ):
        """Corrupt worktree plan.lock on a startup sweep (mid_run=False) is now
        REVERTED, not fail-closed.

        Task 2763: the applier no longer READS <worktree>/.task/plan.lock for
        liveness (the durable lock lives at the meta-root; recovery_for's
        DB-claimant resolution already ruled out a live claimant before
        dispatching REVERT_TO_PENDING). A corrupt worktree lock therefore no
        longer strands the task — it reverts unconditionally, cleaning the
        worktree and defensively unlinking the vestigial lock, with NO
        fail-closed ERROR.

        CRITICAL: get_statuses must return a NON-EMPTY dict so the resolver_failed
        guard in _reconcile_stranded_in_progress does not abort the sweep before
        the revert.
        """
        tid = '200'
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]

        lock_dir = harness.git_ops.worktree_base / tid / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text('not-valid-json')
        worktree_path = harness.git_ops.worktree_base / tid

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # Now reverted — a corrupt worktree lock no longer fails closed.
        harness.scheduler.set_task_status.assert_called_once_with(tid, 'pending')  # type: ignore[attr-defined]
        # The worktree is cleaned up (not in _recovered_plans/_preserved_worktrees).
        harness.git_ops.cleanup_worktree.assert_called_once_with(worktree_path, tid)  # type: ignore[attr-defined]
        # The vestigial worktree lock is defensively cleared.
        assert not lock_path.exists()
        # NO fail-closed ERROR — the read that produced it is gone (task 2763).
        fail_closed = [
            r for r in caplog.records
            if r.levelno == logging.ERROR
            and re.search(r'leaving worktree intact', r.message, re.IGNORECASE)
        ]
        assert not fail_closed, (
            f'fail-closed ERROR must be gone; got: {[r.message for r in caplog.records]}'
        )

    async def test_no_lock_worktree_cleaned_when_not_recovered(
        self, harness: Harness, tmp_path: Path
    ):
        """Worktree dir exists but has no plan.lock and task is NOT in _recovered_plans
        → cleanup_worktree is called and task is reverted to pending."""
        tid = 30
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create the worktree directory (no .task/plan.lock inside)
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)
        # _recovered_plans is empty (default)

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must have been called with the worktree path and tid
        harness.git_ops.cleanup_worktree.assert_called_once_with(worktree_path, str(tid))  # type: ignore[attr-defined]
        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_no_lock_worktree_preserved_when_recovered(
        self, harness: Harness, tmp_path: Path
    ):
        """Worktree dir exists but has no plan.lock and task IS in _recovered_plans
        → cleanup_worktree is NOT called (worktree preserved), task still reverted."""
        tid = 31
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create the worktree directory (no .task/plan.lock inside)
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)
        # Mark task as recovered — worktree must be preserved for resumption
        harness._recovered_plans[str(tid)] = {'task_id': str(tid), 'steps': []}

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must NOT have been called
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        # Worktree directory must still exist
        assert worktree_path.exists()
        # Task must still be reverted to pending (recovery runs separately)
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_no_lock_worktree_retained_when_branch_has_wip_commits(
        self, harness: Harness, tmp_path: Path
    ):
        """Worktree dir exists but has no plan.lock and task is NOT in
        _recovered_plans/_preserved_worktrees, but the leftover branch still
        carries WIP commits beyond main (the shape left behind when the
        stale-lock reap retained a non-degenerate branch) → cleanup_worktree
        is NOT called: the worktree is retained so the next dispatch can
        resume it via create_worktree's cold-path γ reattach, instead of
        reaping the dir and forcing _cleanup_leftover_branch to raise later.
        Task is still reverted to pending.

        Mirrors test_no_lock_worktree_preserved_when_recovered, but the
        retention signal here is branch WIP (_orphan_has_commits), not
        _recovered_plans/_preserved_worktrees membership.
        """
        tid = 36
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create the worktree directory (no .task/plan.lock inside)
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)
        # tid is NOT in _recovered_plans / _preserved_worktrees (defaults empty)
        # Simulate a re-attached-eligible WIP branch: the leftover branch
        # still carries commits beyond main.
        harness.git_ops._orphan_has_commits = AsyncMock(return_value=True)

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must NOT have been called — WIP is retained.
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        # Worktree directory must still exist.
        assert worktree_path.exists()
        # Task must still be reverted to pending (next dispatch resumes it).
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_no_lock_worktree_preserved_when_in_preserved_set(
        self, harness: Harness, tmp_path: Path
    ):
        """Worktree dir exists but has no plan.lock and task IS in
        _preserved_worktrees → cleanup_worktree is NOT called (worktree
        preserved for revalidation), task still reverted to pending.

        Mirrors test_no_lock_worktree_preserved_when_recovered for the
        crash-recovery-stamped-but-not-pre-loaded case (the architect filed
        a stamped plan that was rejected by blast-radius lock conflict).
        """
        tid = '34'
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)
        # Mark task as preserved (architect ran, plan stamped, no done steps).
        harness._preserved_worktrees.add(str(tid))

        await harness._reconcile_stranded_in_progress()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert worktree_path.exists()
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_stale_lock_worktree_preserved_when_in_preserved_set(
        self, harness: Harness, monkeypatch
    ):
        """Stale plan.lock + task IS in _preserved_worktrees →
        cleanup_worktree NOT called, worktree kept, stale lock unlinked,
        task reverted.  Defensive: _recover_crashed_tasks already unlinks
        plan.lock when adding to _preserved_worktrees, so this combined
        state shouldn't appear in practice — locks the invariant against
        future drift, mirroring test_stale_lock_worktree_preserved_when_recovered.
        """
        tid = '35'
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        lock_dir = harness.git_ops.worktree_base / str(tid) / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))
        worktree_path = harness.git_ops.worktree_base / str(tid)
        harness._preserved_worktrees.add(str(tid))

        await harness._reconcile_stranded_in_progress()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert worktree_path.exists()
        assert not lock_path.exists()
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_stale_lock_worktree_cleaned_when_not_recovered(
        self, harness: Harness, monkeypatch
    ):
        """In-progress task with stale plan.lock (dead PID), not in _recovered_plans
        → cleanup_worktree called (removing entire worktree dir), task reverted."""
        tid = 32
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create worktree with a plan.lock referencing a synthetic dead PID
        lock_dir = harness.git_ops.worktree_base / str(tid) / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))
        worktree_path = harness.git_ops.worktree_base / str(tid)
        # _recovered_plans is empty (default)

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must have been called (rmtree removes entire worktree)
        harness.git_ops.cleanup_worktree.assert_called_once_with(worktree_path, str(tid))  # type: ignore[attr-defined]
        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]
        # The entire worktree dir is gone (side_effect rmtree'd it)
        assert not worktree_path.exists()

    async def test_stale_lock_worktree_preserved_when_recovered(
        self, harness: Harness, monkeypatch
    ):
        """In-progress task with stale plan.lock (dead PID), task IS in _recovered_plans
        → cleanup_worktree NOT called, worktree preserved, stale lock unlinked, task reverted.

        NOTE — defensive branch only: this combined state (recovered plan + stale lock still
        present) is unreachable in the normal startup flow.  _recover_crashed_tasks always
        unlinks plan.lock before adding a task to _recovered_plans (harness.py:864-868), so
        in practice a recovered task arrives at the no-lock branch, not the stale-lock branch.
        This test exists to lock the invariant against future drift in the recovery path.
        """
        tid = 33
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create worktree with a plan.lock referencing a synthetic dead PID
        lock_dir = harness.git_ops.worktree_base / str(tid) / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))
        worktree_path = harness.git_ops.worktree_base / str(tid)
        # Mark task as recovered — worktree must be preserved for resumption
        harness._recovered_plans[str(tid)] = {'task_id': str(tid), 'steps': []}

        await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must NOT have been called
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        # Worktree directory must still exist
        assert worktree_path.exists()
        # Stale lock must be removed (so the resumed session doesn't immediately requeue)
        assert not lock_path.exists()
        # Task must be reverted to pending
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]

    async def test_no_lock_branch_cleanup_worktree_raises_still_reverts(
        self, harness: Harness, caplog
    ):
        """Regression lockdown: task is reverted even when cleanup_worktree raises
        in the no-lock branch.  Covers the uncovered except Exception at harness.py:908-913.

        The no-lock branch has no lock to unlink; after cleanup failure it must
        still call set_task_status so the task escapes in-progress.
        """
        tid = 41
        harness.scheduler.get_statuses.return_value = ({str(tid): 'in-progress'}, None)  # type: ignore[attr-defined]
        # Create worktree dir with NO plan.lock inside
        worktree_path = harness.git_ops.worktree_base / str(tid)
        worktree_path.mkdir(parents=True)

        # Make cleanup_worktree raise so we exercise the except-branch
        harness.git_ops.cleanup_worktree = AsyncMock(side_effect=OSError('boom'))  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # cleanup_worktree must have been attempted
        harness.git_ops.cleanup_worktree.assert_called_once_with(worktree_path, str(tid))  # type: ignore[attr-defined]
        # Task must still be reverted to pending despite cleanup failure
        harness.scheduler.set_task_status.assert_called_once_with(str(tid), 'pending')  # type: ignore[attr-defined]
        # Cleanup-failure WARNING must be present in logs
        matching = [
            r for r in caplog.records
            if re.search(r'cleanup_worktree failed.*41.*no-lock', r.message, re.IGNORECASE)
        ]
        assert len(matching) >= 1, (
            f'Expected cleanup-failure WARNING in harness logs, got: {[r.message for r in caplog.records]}'
        )

    async def test_reconcile_uses_get_statuses_not_get_tasks(self, harness: Harness):
        """_reconcile_stranded_in_progress must use get_statuses, not get_tasks.

        RED against current code: harness still calls get_tasks.
        After the migration (step-14 impl), get_statuses is called and
        get_tasks is never called for the reconcile sweep.
        """
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'5': 'in-progress', '6': 'pending'}, None
        )
        # No worktree for task 5 (orphan → will be reverted to pending)
        # Task 6 is pending → not touched

        await harness._reconcile_stranded_in_progress()

        harness.scheduler.get_statuses.assert_called_once()  # type: ignore[attr-defined]
        harness.scheduler.get_tasks.assert_not_called()  # type: ignore[attr-defined]
        calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
        assert len(calls) == 1
        assert calls[0].args[0] == '5'
        assert calls[0].args[1] == 'pending'

    async def test_unexpected_exception_propagates_out_of_reconcile(
        self, harness: Harness
    ):
        """TypeError from json.loads must propagate — not be silently swallowed.

        RED against current code: `except Exception:` catches TypeError and
        treats the lock as stale (task reverted, lock deleted, no exception).
        After the fix (narrow to OSError/JSONDecodeError/ValueError), TypeError
        propagates out, set_task_status is never called, and the lock survives.
        """
        from unittest.mock import patch as _patch

        harness.scheduler.get_statuses.return_value = ({'15': 'in-progress'}, None)  # type: ignore[attr-defined]
        lock_dir = harness.git_ops.worktree_base / '15' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text('{"session_id": "15-xyz", "owner_pid": 1}')  # valid-looking

        with _patch('orchestrator.harness.json.loads', side_effect=TypeError('unexpected')), pytest.raises(TypeError, match='unexpected'):
            await harness._reconcile_stranded_in_progress()

        # No revert must have happened
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        # Lock file must not have been deleted
        assert lock_path.exists(), 'Lock file must survive when an unexpected exception propagates'

    async def test_already_merged_branch_marked_done_with_provenance(
        self, harness: Harness
    ):
        """Stranded in-progress task whose branch is already merged to main →
        marked done with provenance; no pending revert; no cleanup_worktree.

        RED state: the guard doesn't exist yet; reconcile takes the no-lock
        branch and calls set_task_status('50', 'pending'), never calling
        is_ancestor.
        """
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'50': 'in-progress'}, None
        )
        # No worktree dir or plan.lock for task 50 — guard must fire before
        # any worktree analysis.

        await harness._reconcile_stranded_in_progress()

        # is_ancestor must have been invoked with the configured branch +
        # main_branch (task 2787: now also awaited a second time for the
        # citation lineage check, so assert the branch/main await specifically).
        harness.git_ops.is_ancestor.assert_any_await('task/50', 'main')  # type: ignore[attr-defined]

        # set_task_status must be called exactly once: ('50', 'done') with kind/commit/note
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '50', 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': 'deadbeef' + 'a' * 32,
                'note': 'reconcile: branch already on main when stranded in-progress',
            },
        )
        # task 2787 re-introduced a POSITIVE-citation attribution guard on the
        # resolver's ON_MAIN git-archaeology fast-path (mirroring
        # validate_landing_evidence DISCOVERY mode): find_task_citation_commit
        # IS consulted, and the ON_MAIN sha is now the citation itself — here
        # == the fixture's resolve_branch_sha default, so the provenance commit
        # asserted above is unchanged.
        harness.git_ops.find_task_citation_commit.assert_awaited()  # type: ignore[attr-defined]

        # cleanup_worktree must NOT have been called
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_already_merged_skipped_when_l1_escalation_open(
        self, harness: Harness, tmp_path: Path
    ):
        """Open L1 escalation for the task → reconciler must NOT flip to done.

        Reproduces the reify-3399 false-positive: branch tip equals main HEAD
        because the architect was never given a chance to write any commits
        (the task was declared unactionable and _mark_blocked(escalate_to_human=True)
        fired before any commit landed).  The deliberate human-escalation
        disposition must take precedence over the degenerate is_ancestor==True
        observation.
        """
        from escalation.models import Escalation
        from escalation.queue import EscalationQueue

        # Wire up a real EscalationQueue and submit an L1 record for task 50.
        queue_dir = tmp_path / 'escalations'
        harness._escalation_queue = EscalationQueue(queue_dir)
        harness._escalation_queue.submit(Escalation(
            id=harness._escalation_queue.make_id('50'),
            task_id='50',
            agent_role='task-steward',
            severity='blocking',
            category='task_failure',
            summary='Task declared unactionable',
            level=1,
            status='pending',
        ))

        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'50': 'blocked'}, None
        )

        await harness._reconcile_stranded_in_progress()

        # is_ancestor was invoked (proves we got into the branch). task 2787:
        # the resolver's ON_MAIN fast-path now awaits is_ancestor a second time
        # for the citation lineage check, so assert the branch/main await
        # specifically rather than an exact once.
        harness.git_ops.is_ancestor.assert_any_await('task/50', 'main')  # type: ignore[attr-defined]

        # But neither mark_done nor set_task_status('done', ...) fired —
        # the L1 guard bailed before the flip.
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    # test_already_merged_skipped_when_main_lacks_task_citation and
    # test_already_merged_uses_citation_commit_when_available were removed
    # (task 2243, W10-θ2 step-4): both asserted the citation-guard's
    # SHA-preference/skip behaviour (find_task_citation_commit), which
    # step-4's plan explicitly retires in favour of TaskGroundTruth's
    # coarser ON_MAIN/GONE_WITH_MERGE_MARKER classification — see the
    # design decision recorded on task 2243's plan and esc-2243-4.

    async def test_already_merged_drops_recovered_plan_and_cleans_worktree(
        self, harness: Harness
    ):
        """Regression: when is_ancestor=True and the task has a recovered plan,
        the stale _recovered_plans entry must be dropped and the orphaned
        worktree must be cleaned up — no entry should linger after the task
        transitions to a terminal 'done' state where resumption is impossible.
        """
        tid = '52'
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        # Seed a recovered plan — simulates _recover_crashed_tasks having run
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}

        # Create the worktree dir on disk so the cleanup branch is reachable
        worktree_path = harness.git_ops.worktree_base / tid
        worktree_path.mkdir(parents=True)

        await harness._reconcile_stranded_in_progress()

        # (1) Stale recovered-plan entry must be dropped
        assert tid not in harness._recovered_plans, (
            '_recovered_plans entry must be popped when branch is already on main'
        )

        # (2) cleanup_worktree must be called exactly once (unconditional cleanup)
        harness.git_ops.cleanup_worktree.assert_awaited_once_with(  # type: ignore[attr-defined]
            worktree_path, tid
        )

        # (3) Task must be marked done with the expected provenance (kind + commit + note)
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': 'deadbeef' + 'a' * 32,
                'note': 'reconcile: branch already on main when stranded in-progress',
            },
        )

        # (4) Worktree dir is gone — proves cleanup_worktree's rmtree side_effect ran
        assert not worktree_path.exists(), (
            'worktree dir must be removed by cleanup_worktree'
        )

    async def test_already_merged_takes_precedence_over_stale_lock(
        self, harness: Harness, monkeypatch
    ):
        """Placement-precedence regression lock: is_ancestor guard fires BEFORE
        the stale-lock analysis.

        A task with a stale plan.lock AND is_ancestor=True must take the done
        path (no pending revert, stale-lock analysis bypassed). The guard also
        cleans up the stale worktree dir (amendment: prevents worktree cruft
        accumulation when orchestrator crashed after merge but before cleanup).
        This test would fail if a future refactor moved the guard below the
        lock analysis (set_task_status would be called with 'pending').
        """
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'51': 'in-progress'}, None
        )
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create a worktree with a stale plan.lock (dead PID)
        worktree_path = harness.git_ops.worktree_base / '51'
        lock_dir = worktree_path / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '51-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))

        await harness._reconcile_stranded_in_progress()

        # Must be marked done, NOT reverted to pending
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            '51', 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': 'deadbeef' + 'a' * 32,
                'note': 'reconcile: branch already on main when stranded in-progress',
            },
        )

        # cleanup_worktree IS called — the guard cleans up stale worktrees for
        # already-merged tasks to prevent worktree cruft from accumulating
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            worktree_path, '51'
        )

        # plan.lock is gone — cleanup_worktree's side_effect rmtree'd the dir,
        # proving the stale-lock analysis branch was bypassed (which would have
        # also called set_task_status('51', 'pending') if it had run)
        assert not lock_path.exists(), (
            'plan.lock should be removed by cleanup_worktree in the is_ancestor guard'
        )

    # NB: an earlier test (``test_already_merged_skipped_when_branch_unresolved``)
    # covered the ``resolve_branch_sha returns None`` race during the
    # is_ancestor fast-path.  After the post-fix flow uses
    # ``find_task_citation_commit`` to source the done_provenance SHA, that
    # race is structurally impossible (the citation grep does not depend on
    # a live branch ref).  Skip-without-flip semantics on missing evidence
    # are covered by ``test_already_merged_skipped_when_main_lacks_task_citation``.

    # ------------------------------------------------------------------
    # Guard 3 — branch-advanced check (is_ancestor fast-path)
    # Guards 1 (open L1) and 2 (citation grep) already passed;
    # Guard 3 structurally rejects zero-commit branches sitting on main.
    # ------------------------------------------------------------------

    async def test_already_merged_skipped_when_branch_never_advanced(
        self, harness: Harness
    ):
        """Guard 3: branch tip == branch_base_sha → never advanced; veto flip.

        The branch base equals the current tip, proving no commits were ever
        pushed on this incarnation.  The reconciler must NOT flip to done.
        """
        branch_tip = 'deadbeef' + 'a' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)
        # Branch tip equals base → branch never advanced past creation point
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=branch_tip)
        harness.scheduler.get_task = AsyncMock(
            return_value={'id': '50', 'status': 'blocked', 'metadata': {'branch_base_sha': branch_tip}}
        )
        harness.scheduler.get_statuses.return_value = ({'50': 'blocked'}, None)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        # Guard 3 must veto
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    # test_already_merged_flips_when_branch_advanced and
    # test_already_merged_falls_through_when_branch_base_sha_missing_or_malformed
    # were removed (task 2243, W10-θ2 step-4): both asserted the retired
    # citation-guard's SHA-preference behaviour (done_provenance.commit ==
    # the find_task_citation_commit result rather than the branch tip) —
    # see the design decision on task 2243's plan and esc-2243-4. The
    # "Guard 3 does not spuriously veto a non-degenerate branch" property
    # they also covered is subsumed by
    # test_already_merged_branch_marked_done_with_provenance (an
    # already-merged, non-degenerate branch flips to done).

    # ------------------------------------------------------------------
    # Degenerate-branch IN-PROGRESS recovery (#1823 follow-up / task-2992).
    # is_ancestor==True + tip == branch_base_sha means the branch did ZERO
    # work; an in-progress incarnation with no live claimant must be REVERTED
    # to pending (re-dispatch), not left to sit in-progress forever.  'blocked'
    # keeps the leave-alone discipline.
    # ------------------------------------------------------------------

    async def test_degenerate_in_progress_no_citation_reverted(
        self, harness: Harness
    ):
        """The task-2992 strand: in-progress on a degenerate provisioning
        branch (tip == branch_base_sha), no live claimant → reverted to
        pending; NOT marked done."""
        branch_tip = 'deadbeef' + 'a' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=branch_tip)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'id': '50', 'metadata': {'branch_base_sha': branch_tip}}
        )
        harness.scheduler.get_statuses.return_value = ({'50': 'in-progress'}, None)  # type: ignore[attr-defined]
        # No worktree dir for task 50 → no-lock orphan revert path.

        await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with('50', 'pending')  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]

    async def test_degenerate_in_progress_no_citation_stale_lock_reverted(
        self, harness: Harness, monkeypatch
    ):
        """Guard 2 path with a stale plan.lock (dead owner_pid): lock cleared and
        in-progress task reverted to pending."""
        branch_tip = 'deadbeef' + 'a' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.find_task_citation_commit = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=branch_tip)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'id': '8', 'metadata': {'branch_base_sha': branch_tip}}
        )
        harness.scheduler.get_statuses.return_value = ({'8': 'in-progress'}, None)  # type: ignore[attr-defined]
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)
        lock_dir = harness.git_ops.worktree_base / '8' / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': '8-dead0001',
            'locked_at': datetime.now(UTC).isoformat(),
            'owner_pid': 99999,
        }))

        await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_called_once_with('8', 'pending')  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert not lock_path.exists()

    async def test_degenerate_in_progress_with_citation_reverted(
        self, harness: Harness
    ):
        """Guard 3 path: in-progress + a commit on main cites the task id BUT the
        branch is degenerate (tip == branch_base_sha).  The citation belongs to a
        prior incarnation; the current incarnation did no work → reverted to
        pending, NOT marked done."""
        branch_tip = 'deadbeef' + 'a' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.find_task_citation_commit = AsyncMock(  # type: ignore[attr-defined]
            return_value='cafefeed' + 'b' * 32
        )
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=branch_tip)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'id': '50', 'metadata': {'branch_base_sha': branch_tip}}
        )
        harness.scheduler.get_statuses.return_value = ({'50': 'in-progress'}, None)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with('50', 'pending')  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]

    async def test_degenerate_blocked_not_reverted(self, harness: Harness):
        """Blocked discipline regression: a degenerate branch on a BLOCKED task is
        left intact — never blocked→pending, never marked done.  Pins the revert
        to in-progress only (the in-progress-vs-blocked branch in both guards)."""
        branch_tip = 'deadbeef' + 'a' * 32
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=branch_tip)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'id': '50', 'metadata': {'branch_base_sha': branch_tip}}
        )
        harness.scheduler.get_statuses.return_value = ({'50': 'blocked'}, None)  # type: ignore[attr-defined]

        await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # find_merge_marker guard tests (deleted-branch fast-path)
    # ------------------------------------------------------------------

    async def test_deleted_branch_with_merge_marker_marked_done(
        self, harness: Harness
    ):
        """Stranded in-progress task whose branch was deleted but whose merge
        marker is found on main → marked done with {commit, note} provenance.

        is_ancestor=False (branch doesn't exist, so is_ancestor can't resolve it),
        find_merge_marker returns a SHA → task must be marked done with the marker
        SHA in done_provenance['commit'] and cleanup_worktree must NOT be called
        (no worktree dir was created in this test).
        """
        tid = '70'
        marker_sha = 'abc123def' + 'a' * 31
        harness.git_ops.find_merge_marker = AsyncMock(  # type: ignore[attr-defined]
            return_value=marker_sha
        )
        # TaskGroundTruth._resolve_branch_state treats a non-None
        # resolve_branch_sha as "branch still exists" (EXISTS_OFF_MAIN) and
        # never reaches find_merge_marker at all — must resolve None here to
        # reflect the deleted-branch premise of this test (task 2243, W10-θ2).
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        # No worktree dir for task 70

        await harness._reconcile_stranded_in_progress()

        # find_merge_marker must have been invoked with the full branch name
        harness.git_ops.find_merge_marker.assert_awaited_once_with(  # type: ignore[attr-defined]
            f'task/{tid}'
        )

        # Task must be marked done with kind + commit + note provenance
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': 'reconcile: branch deleted but merge marker found on main',
            },
        )

        # No worktree to clean up
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_deleted_branch_no_merge_marker_falls_through_to_revert(
        self, harness: Harness
    ):
        """Stranded in-progress task whose branch is deleted and whose marker is
        absent → falls through to the existing revert-to-pending path.

        Proves the marker guard does NOT swallow the no-lock / no-marker case:
        the task must still be reverted to pending so it can be re-queued.
        """
        tid = '71'
        # Default: find_merge_marker returns None (already in fixture)
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        # No worktree, no lock

        await harness._reconcile_stranded_in_progress()

        # Must fall through to the revert path
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending'
        )

    async def test_marker_takes_precedence_over_stale_lock(
        self, harness: Harness, monkeypatch
    ):
        """Placement-precedence: find_merge_marker guard fires BEFORE the
        stale-lock analysis.

        A task with a stale plan.lock AND a merge marker must take the done
        path with marker provenance.  cleanup_worktree is called once (worktree
        dir existed), and the stale-lock branch is bypassed entirely.
        """
        tid = '72'
        marker_sha = 'deadc0de' + 'b' * 32
        harness.git_ops.find_merge_marker = AsyncMock(  # type: ignore[attr-defined]
            return_value=marker_sha
        )
        # TaskGroundTruth._resolve_branch_state treats a non-None
        # resolve_branch_sha as "branch still exists" (EXISTS_OFF_MAIN) and
        # never reaches find_merge_marker at all — must resolve None here to
        # reflect the deleted-branch premise of this test (task 2243, W10-θ2).
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)

        # Create a worktree with a stale plan.lock (dead PID)
        worktree_path = harness.git_ops.worktree_base / tid
        lock_dir = worktree_path / '.task'
        lock_dir.mkdir(parents=True)
        lock_path = lock_dir / 'plan.lock'
        lock_path.write_text(json.dumps({
            'session_id': f'{tid}-dead',
            'locked_at': '2026-01-01T00:00:00+00:00',
            'owner_pid': 99999,
        }))

        await harness._reconcile_stranded_in_progress()

        # Must be marked done with marker provenance, NOT reverted to pending
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': 'reconcile: branch deleted but merge marker found on main',
            },
        )

        # cleanup_worktree IS called — worktree dir existed
        harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
            worktree_path, tid
        )

        # plan.lock is gone — cleanup_worktree's rmtree side_effect ran,
        # proving the stale-lock branch was bypassed
        assert not lock_path.exists(), (
            'plan.lock should be removed by cleanup_worktree in the marker guard'
        )

    async def test_marker_drops_recovered_plan_and_cleans_worktree(
        self, harness: Harness
    ):
        """Regression: when find_merge_marker returns a SHA and the task has a
        recovered plan, the stale _recovered_plans entry must be dropped and the
        orphaned worktree must be cleaned up.

        Analog of test_already_merged_drops_recovered_plan_and_cleans_worktree
        for the marker path.
        """
        tid = '73'
        marker_sha = 'cafe1234' + 'c' * 32
        harness.git_ops.find_merge_marker = AsyncMock(  # type: ignore[attr-defined]
            return_value=marker_sha
        )
        # TaskGroundTruth._resolve_branch_state treats a non-None
        # resolve_branch_sha as "branch still exists" (EXISTS_OFF_MAIN) and
        # never reaches find_merge_marker at all — must resolve None here to
        # reflect the deleted-branch premise of this test (task 2243, W10-θ2).
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {tid: 'in-progress'}, None
        )
        # Seed a recovered plan — simulates _recover_crashed_tasks having run
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}

        # Create the worktree dir on disk so the cleanup branch is reachable
        worktree_path = harness.git_ops.worktree_base / tid
        worktree_path.mkdir(parents=True)

        await harness._reconcile_stranded_in_progress()

        # (1) Stale recovered-plan entry must be dropped
        assert tid not in harness._recovered_plans, (
            '_recovered_plans entry must be popped when marker is found on main'
        )

        # (2) cleanup_worktree must be called exactly once
        harness.git_ops.cleanup_worktree.assert_awaited_once_with(  # type: ignore[attr-defined]
            worktree_path, tid
        )

        # (3) Task must be marked done with marker provenance
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main',
                'commit': marker_sha,
                'note': 'reconcile: branch deleted but merge marker found on main',
            },
        )

        # (4) Worktree dir is gone — cleanup_worktree's rmtree side_effect ran
        assert not worktree_path.exists(), (
            'worktree dir must be removed by cleanup_worktree'
        )

    # ------------------------------------------------------------------
    # Stale-marker check tests (find_merge_marker path)
    # A marker that pre-dates the current branch incarnation must be
    # rejected; a marker from this incarnation must flip the task.
    # ------------------------------------------------------------------

    # test_marker_skipped_when_marker_predates_branch_base_sha_parametrized,
    # test_marker_accepted_when_marker_postdates_branch_base_sha_parametrized,
    # test_marker_falls_through_when_branch_base_sha_missing, and
    # test_stale_marker_veto_leaves_inprogress_without_lock_revert were
    # removed (task 2243, W10-θ2 step-4): all four asserted the retired
    # stale-marker/prior-incarnation veto (is_ancestor(marker_sha,
    # branch_base_sha)) that lived inside the now-deleted inline
    # find_merge_marker archaeology block — TaskGroundTruth's
    # GONE_WITH_MERGE_MARKER resolution carries no equivalent check. See the
    # design decision on task 2243's plan and esc-2243-4.


    async def test_find_merge_marker_not_invoked_when_is_ancestor_true(
        self, harness: Harness
    ):
        """Efficiency lock: find_merge_marker is never called when is_ancestor
        returns True.

        The is_ancestor branch short-circuits via `continue` before the marker
        guard is reached.
        """
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {'74': 'in-progress'}, None
        )

        await harness._reconcile_stranded_in_progress()

        # is_ancestor fired → should NOT call find_merge_marker
        harness.git_ops.find_merge_marker.assert_not_called()  # type: ignore[attr-defined]
        # But the task must be marked done via the is_ancestor path
        harness.scheduler.set_task_status.assert_awaited_once()  # type: ignore[attr-defined]

    async def test_is_ancestor_not_invoked_for_terminal_or_pending_tasks(
        self, harness: Harness
    ):
        """Placement-efficiency regression lock: is_ancestor is never called
        for tasks the sweep does not consider stranded — i.e. anything outside
        ``{in-progress, blocked}``.

        Note: 'blocked' is now included in the sweep (R4) — out-of-band-merged
        blocked tasks need recovery — so the input set here is restricted to
        statuses the sweep does NOT touch.
        """
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {
                '60': 'pending',
                '61': 'done',
                '63': 'cancelled',
                '64': 'review',
                '65': 'deferred',
            },
            None,
        )

        await harness._reconcile_stranded_in_progress()

        # is_ancestor must never be called (no sweep-eligible tasks)
        harness.git_ops.is_ancestor.assert_not_called()  # type: ignore[attr-defined]
        # No status changes either
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    async def test_find_merge_marker_not_invoked_for_terminal_or_pending_tasks(
        self, harness: Harness
    ):
        """Placement-efficiency regression lock: find_merge_marker is never
        called for tasks the sweep does not consider stranded.

        Same reasoning as ``test_is_ancestor_not_invoked_for_terminal_or_pending_tasks``:
        only ``{in-progress, blocked}`` are swept after R4.
        """
        harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
            {
                '80': 'pending',
                '81': 'done',
                '83': 'cancelled',
                '84': 'review',
                '85': 'deferred',
            },
            None,
        )

        await harness._reconcile_stranded_in_progress()

        # find_merge_marker must never be called (no sweep-eligible tasks)
        harness.git_ops.find_merge_marker.assert_not_called()  # type: ignore[attr-defined]
        # No status changes either
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # Done-branch side-effect suite (symmetry + cleanup-failure + INFO log)
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'scenario,is_ancestor_val,marker_sha_val,reason,expected_provenance,cleanup_raises',
        [
            pytest.param(
                'is_ancestor',
                True,
                None,
                'branch-already-on-main',
                {
                    'kind': 'found_on_main',
                    'commit': 'deadbeef' + 'a' * 32,
                    'note': 'reconcile: branch already on main when stranded in-progress',
                },
                False,
                id='is_ancestor-branch-success',
            ),
            pytest.param(
                'marker',
                False,
                'cafebabe' + 'd' * 32,
                'branch-deleted-marker-found',
                {
                    'kind': 'found_on_main',
                    'commit': 'cafebabe' + 'd' * 32,
                    'note': 'reconcile: branch deleted but merge marker found on main',
                },
                False,
                id='marker-branch-success',
            ),
            pytest.param(
                'is_ancestor',
                True,
                None,
                'branch-already-on-main',
                {
                    'kind': 'found_on_main',
                    'commit': 'deadbeef' + 'a' * 32,
                    'note': 'reconcile: branch already on main when stranded in-progress',
                },
                True,
                id='is_ancestor-branch-cleanup-fails',
            ),
            pytest.param(
                'marker',
                False,
                'cafebabe' + 'd' * 32,
                'branch-deleted-marker-found',
                {
                    'kind': 'found_on_main',
                    'commit': 'cafebabe' + 'd' * 32,
                    'note': 'reconcile: branch deleted but merge marker found on main',
                },
                True,
                id='marker-branch-cleanup-fails',
            ),
        ],
    )
    async def test_done_branch_side_effects(
        self,
        harness: Harness,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        scenario: str,
        is_ancestor_val: bool,
        marker_sha_val: str | None,
        reason: str,
        expected_provenance: dict,
        cleanup_raises: bool,
    ):
        """Symmetry regression + cleanup-failure contract for both done-branches.

        cleanup_raises=False: verifies pop / cleanup / set_task_status / worktree
        removal / INFO log are all produced identically for both branches.
        cleanup_raises=True: verifies that a cleanup failure is swallowed and a
        WARNING is emitted — set_task_status still fires for both branches.
        """
        tid = '95'
        harness.git_ops.is_ancestor = AsyncMock(return_value=is_ancestor_val)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha_val)  # type: ignore[attr-defined]
        if not is_ancestor_val:
            # TaskGroundTruth._resolve_branch_state treats a non-None
            # resolve_branch_sha as "branch still exists" (EXISTS_OFF_MAIN)
            # and never reaches find_merge_marker at all — must resolve None
            # on the marker-branch scenario to reach the marker guard
            # (task 2243, W10-θ2).
            harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]

        # Seed a recovered plan entry — helper must pop it.
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}  # type: ignore[attr-defined]

        # Create worktree dir — helper must attempt cleanup.
        worktree_path = harness.git_ops.worktree_base / tid
        worktree_path.mkdir(parents=True)

        if cleanup_raises:
            harness.git_ops.cleanup_worktree = AsyncMock(  # type: ignore[attr-defined]
                side_effect=OSError('boom')
            )

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            await harness._reconcile_stranded_in_progress()

        # Recovered plan must be gone regardless of cleanup outcome.
        assert tid not in harness._recovered_plans  # type: ignore[attr-defined]

        # set_task_status must always fire with 'done' and the expected provenance.
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done', done_provenance=expected_provenance
        )

        if cleanup_raises:
            # cleanup_worktree must have been called (before the OSError was swallowed).
            harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
                worktree_path, tid
            )
            # Exception swallowed; WARNING log must contain tid and reason.
            warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert any(
                tid in r.getMessage() and reason in r.getMessage()
                for r in warning_logs
            ), (
                f'Expected WARNING containing tid={tid!r} and reason={reason!r}; '
                f'got: {[r.getMessage() for r in warning_logs]}'
            )
        else:
            # cleanup_worktree must have been called exactly once.
            harness.git_ops.cleanup_worktree.assert_called_once_with(  # type: ignore[attr-defined]
                worktree_path, tid
            )
            # Worktree dir must have been removed by the cleanup side-effect.
            assert not worktree_path.exists()
            # INFO log must mention tid and reason.
            info_logs = [r for r in caplog.records if r.levelno == logging.INFO]
            assert any(
                tid in r.getMessage() and reason in r.getMessage()
                for r in info_logs
            ), (
                f'Expected INFO containing tid={tid!r} and reason={reason!r}; '
                f'got: {[r.getMessage() for r in info_logs]}'
            )

    @pytest.mark.parametrize(
        'scenario,is_ancestor_val,marker_sha_val,expected_commit',
        [
            pytest.param(
                'is_ancestor',
                True,
                None,
                'deadbeef' + 'a' * 32,
                id='is_ancestor-branch',
            ),
            pytest.param(
                'marker',
                False,
                'cafebabe' + 'd' * 32,
                'cafebabe' + 'd' * 32,
                id='marker-branch',
            ),
        ],
    )
    async def test_absent_worktree_dir_skips_cleanup(
        self,
        harness: Harness,
        scenario: str,
        is_ancestor_val: bool,
        marker_sha_val: str | None,
        expected_commit: str,
    ):
        """When the worktree directory does not exist, cleanup_worktree must
        NOT be called — the existence guard must hold for both done-branches.
        scheduler.mark_done (harness.py:1464) still fires with kind='found_on_main'
        and the expected SHA, and _recovered_plans is popped regardless of
        worktree absence.
        """
        tid = '97'
        harness.git_ops.is_ancestor = AsyncMock(return_value=is_ancestor_val)  # type: ignore[attr-defined]
        harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha_val)  # type: ignore[attr-defined]
        if not is_ancestor_val:
            # TaskGroundTruth._resolve_branch_state treats a non-None
            # resolve_branch_sha as "branch still exists" (EXISTS_OFF_MAIN)
            # and never reaches find_merge_marker at all — must resolve None
            # on the marker-branch scenario to reach the marker guard
            # (task 2243, W10-θ2).
            harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]

        # Seed a recovered plan entry — helper must pop it even without a worktree dir.
        harness._recovered_plans[tid] = {'task_id': tid, 'steps': []}  # type: ignore[attr-defined]

        # Deliberately do NOT mkdir — the worktree dir is absent.

        await harness._reconcile_stranded_in_progress()

        # Recovered plan must be gone regardless of worktree absence.
        assert tid not in harness._recovered_plans  # type: ignore[attr-defined]

        # cleanup_worktree must NOT have been called.
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        # Task must still be marked done at the production boundary (harness.py:1464).
        # note=ANY: pinning the literal prose adds no regression-detection value
        # beyond assert_awaited_once + kind + sha.
        harness.scheduler.mark_done.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, kind='found_on_main', sha=expected_commit, note=ANY
        )

    # test_get_task_fetched_exactly_once_regardless_of_path and
    # test_hoisted_metadata_is_consumed_by_each_guard were removed (task
    # 2243, W10-θ2 step-4): both pinned implementation details of the
    # now-retired inline archaeology — a single hoisted get_task fetch
    # shared by every fast-path guard, and Guard 3's
    # metadata.get('branch_base_sha')-driven stale-marker/branch-advanced
    # checks. TaskGroundTruth.derive_truth fetches the task row itself
    # (internally, once) to resolve db_status/live_claimant/deploy_phase,
    # independent of the sweep's own hoisted fetch for the downstream
    # blocked/revert paths — so get_task is now awaited twice per stranded
    # task (an accepted trade-off; see the comment above the hoisted fetch
    # in _reconcile_one_stranded). Guard 3's specific checks have no
    # TaskGroundTruth equivalent (see the design decision on task 2243's
    # plan and esc-2243-4); the disposition properties both tests also
    # pinned are covered by test_already_merged_branch_marked_done_with_provenance
    # and the marker-path tests in this class.


# ---------------------------------------------------------------------------
# Harness.run() call-order test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_harness_run_invokes_reconcile_before_scheduler_loop(
    tmp_path: Path,
):
    """run() must call _recover_crashed_tasks → _reconcile_stranded_in_progress
    → scheduler.acquire_next in that order.
    """
    call_order: list[str] = []

    git_cfg = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.git = git_cfg
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.review.enabled = False
    config.sandbox.backend = 'auto'
    config.max_concurrent_tasks = 2
    config.fused_memory.project_id = 'test'
    config.sandbox.backend = 'auto'
    # Real Path so OverrideStore.from_config(config) can call .parent.mkdir()
    # and sqlite3.connect(str(...)) without leaking MagicMock-named files —
    # Harness.__init__ wires OverrideStore unconditionally (task 1313).
    config.overrides_db_path = tmp_path / 'overrides.db'
    # No-landings breaker (task 1918): ints are read at construction (deque
    # maxlen); enabled=False keeps run() from spinning up the breaker loop.
    config.no_landings_breaker_enabled = False
    config.no_landings_breaker_window_samples = 30
    config.no_landings_breaker_disk_free_floor_bytes = 50 * 1024 * 1024 * 1024
    # Sweeps not under test here (task 2241, W10-η): disabled at the config
    # gate so _build_lifecycle_registry() never registers their
    # BackgroundService — this config is a bare spec_set MagicMock, so an
    # unset interval would be a MagicMock and crash asyncio.sleep().
    config.orphan_l0_reaper_enabled = False
    config.terminal_status_watcher_enabled = False
    config.stranded_reconcile_enabled = False
    config.main_tip_sweep_enabled = False

    with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.OverrideStore'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(config)

    # --- mock infrastructure methods so run() doesn't fail early ---
    h.git_ops = MagicMock()
    h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
    h.git_ops.worktree_base = tmp_path / '.worktrees'

    mock_mcp = mock_mcp_cls.return_value
    mock_mcp.start = AsyncMock()
    mock_mcp.stop = AsyncMock()

    h._start_escalation_server = AsyncMock()
    h._start_merge_worker = AsyncMock()
    h._dismiss_stale_escalations = AsyncMock()
    h._tag_task_modules = AsyncMock()

    # Provide one pending task so the "no pending tasks" check passes.
    # get_statuses is used by the startup block (post-step-16);
    # get_tasks is retained for methods not yet migrated (e.g. _tag_prd_metadata).
    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[
        {'id': 1, 'status': 'pending', 'title': 'A task'},
    ])
    h.scheduler.get_statuses = AsyncMock(return_value=({'1': 'pending'}, None))
    h.scheduler.set_task_status = AsyncMock()

    # Track ordering: _recover_crashed_tasks
    async def _fake_recover():
        call_order.append('recover')
    h._recover_crashed_tasks = _fake_recover

    # Runs between recover and reconcile-stranded; not under test here.
    h._reconcile_lane_checkouts = AsyncMock()

    # Track ordering: _reconcile_stranded_in_progress
    async def _fake_reconcile(*, mid_run: bool = False) -> int:
        call_order.append('reconcile')
        return 0
    h._reconcile_stranded_in_progress = _fake_reconcile

    # Track ordering: acquire_next — append then raise to break the loop
    async def _fake_acquire():
        call_order.append('acquire')
        raise RuntimeError('stop the loop')
    h.scheduler.acquire_next = _fake_acquire

    with pytest.raises(RuntimeError, match='stop the loop'):
        await h.run(prd_path=None)

    # _recover_crashed_tasks then _reconcile_stranded_in_progress then acquire_next
    assert 'recover' in call_order, "_recover_crashed_tasks was not called"
    assert 'reconcile' in call_order, "_reconcile_stranded_in_progress was not called"
    assert 'acquire' in call_order, "scheduler.acquire_next was not called"
    recover_idx = call_order.index('recover')
    reconcile_idx = call_order.index('reconcile')
    acquire_idx = call_order.index('acquire')
    assert recover_idx < reconcile_idx, "_recover_crashed_tasks must precede _reconcile_stranded_in_progress"
    assert reconcile_idx < acquire_idx, "_reconcile_stranded_in_progress must precede scheduler.acquire_next"

    # prd_path=None means _tag_prd_metadata is never called, so get_tasks
    # (which _tag_prd_metadata uses for full task data) must not be called.
    # This assertion locks in the migration boundary: all startup-block status
    # checks have moved to get_statuses; get_tasks is only retained for the
    # prd_path code paths that need full task metadata.
    h.scheduler.get_tasks.assert_not_called()


# ---------------------------------------------------------------------------
# Sweep only touches {in-progress, blocked} (regression guard for non-goal)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_and_pending_statuses_ignored(harness: Harness):
    """The sweep only touches {in-progress, blocked} tasks.

    Blocked tasks are checked (R4: out-of-band-merge recovery) but are only
    flipped to done when on-main evidence is observed; without is_ancestor /
    find_merge_marker hits, they stay blocked. ``done`` / ``cancelled`` /
    ``deferred`` / ``pending`` / ``review`` are never written.
    """
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {
            '20': 'pending',
            '21': 'done',
            '22': 'blocked',  # checked but no on-main evidence → untouched
            '23': 'cancelled',
            '24': 'review',
            '25': 'in-progress',  # orphan-revert candidate
            '26': 'deferred',
        },
        None,
    )
    # No worktree for task 25 (orphan)

    await harness._reconcile_stranded_in_progress()

    calls = harness.scheduler.set_task_status.call_args_list  # type: ignore[attr-defined]
    assert len(calls) == 1, f"Expected exactly 1 call, got: {calls}"
    assert calls[0].args[0] == '25'
    assert calls[0].args[1] == 'pending'


# ---------------------------------------------------------------------------
# Mid-run filter (Fix 4)
#
# When the sweep runs *during* a live orchestrator run (mid_run=True), tasks
# that the scheduler is actively dispatching (in ``_dispatched``) or holding
# locks for (``lock_table._held``) must be skipped — they are not stranded,
# they're being worked on right now, and reverting their status would race
# the running workflow.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mid_run_skips_dispatched_tasks(harness: Harness):
    """Task in scheduler._dispatched is not stranded — left untouched."""
    # Replace the scheduler MagicMock attrs with real containers so the
    # mid-run guard can inspect membership.
    harness.scheduler._dispatched = {'40'}  # type: ignore[attr-defined]
    harness.scheduler.lock_table = mock_lock_table()  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'40': 'in-progress'}, None,
    )
    # No worktree for task 40 — would normally trigger no-lock revert.

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mid_run_skips_lock_held_tasks(harness: Harness):
    """Task with active lock_table membership is not stranded — left untouched."""
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = mock_lock_table({'41': {'mod_a'}})  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'41': 'in-progress'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mid_run_reverts_genuine_strand(harness: Harness):
    """Task NOT in _dispatched / _held but in-progress → genuinely stranded."""
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = mock_lock_table()  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'42': 'in-progress'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    assert changed == 1
    harness.scheduler.set_task_status.assert_called_once_with('42', 'pending')  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_returns_change_count(harness: Harness):
    """Reconcile returns int count (revert + marked-done) for main-loop hook."""
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = mock_lock_table()  # type: ignore[attr-defined]

    # No in-progress tasks → nothing to do.
    harness.scheduler.get_statuses.return_value = ({'10': 'pending'}, None)  # type: ignore[attr-defined]
    assert await harness._reconcile_stranded_in_progress() == 0

    # One stranded task → returns 1.
    harness.scheduler.get_statuses.return_value = ({'11': 'in-progress'}, None)  # type: ignore[attr-defined]
    assert await harness._reconcile_stranded_in_progress() == 1


# ---------------------------------------------------------------------------
# Per-tid try/except + N-strikes escalation (Stage 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_one_task_rejection_does_not_kill_sweep(harness: Harness, caplog):
    """A SetTaskStatusRejected on one task must NOT abort the iteration over others."""
    from orchestrator.scheduler import DoneGateRejection

    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'100': 'in-progress', '101': 'in-progress'}, None,
    )

    call_count = 0

    async def _flaky_mark_done(tid, *, kind, sha, note=None):
        nonlocal call_count
        call_count += 1
        if tid == '100':
            raise DoneGateRejection(
                task_id=tid, missing_files=['x.py'],
                raw='done_gate_missing_files — metadata.files lists missing paths',
            )

    harness.scheduler.mark_done = AsyncMock(side_effect=_flaky_mark_done)  # type: ignore[attr-defined]

    with caplog.at_level(logging.ERROR, logger='orchestrator.harness'):
        result = await harness._reconcile_stranded_in_progress()

    # Both tasks attempted; only one succeeded.
    assert call_count == 2
    assert result == 1  # only task 101 marked done
    # Per-tid failure counter retired (task 2243, W10-θ2 step-13/14): a
    # rejection now escalates directly instead of being tallied toward a
    # threshold — see test_rejection_escalates_immediately_no_counter_threshold.
    assert not hasattr(harness, '_reconcile_failure_counts')
    # Honest log mentions the error_code, not "marked done".
    assert any(
        'failed to mark task 100 done' in r.getMessage()
        and 'done_gate_missing_files' in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_rejection_escalates_immediately_no_counter_threshold(harness: Harness):
    """A SetTaskStatusRejected escalates on the FIRST occurrence.

    Task 2243, W10-θ2 step-13/14: the per-tid ``_reconcile_failure_counts``
    dedup counter and its ``MAX_RECONCILE_FAILURES``-strikes threshold gate
    are retired — recovery_for's caller now escalates directly on every
    persistent rejection instead of silently swallowing it behind a
    multi-sweep tally.
    """
    from orchestrator.scheduler import DoneGateRejection

    # Inject a stub escalation queue to capture submissions.
    submissions = []

    class _StubEscalationQueue:
        def make_id(self, task_id):
            return f'esc-{task_id}-{len(submissions)}'
        def submit(self, esc):
            submissions.append(esc)
        def has_open_l1(self, task_id):  # noqa: ARG002
            return False
        def get_by_task(self, task_id, status=None):  # noqa: ARG002
            # TaskGroundTruth._resolve_open_escalations (task 2243, W10-θ2)
            # consults this to fold open escalations into the recovery
            # shape — no escalation is open for this test's task.
            return []

    harness._escalation_queue = _StubEscalationQueue()  # type: ignore[assignment]

    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'200': 'in-progress'}, None,
    )

    async def _always_reject(tid, *, kind, sha, note=None):
        raise DoneGateRejection(
            task_id=tid, missing_files=['x.py'],
            raw='done_gate_missing_files — metadata.files lists missing paths',
        )

    harness.scheduler.mark_done = AsyncMock(side_effect=_always_reject)  # type: ignore[attr-defined]

    # A single sweep is enough — there is no strikes-threshold to climb.
    await harness._reconcile_stranded_in_progress()

    assert len(submissions) == 1, (
        f'expected exactly one L1 submission after a single rejection, '
        f'got {len(submissions)}'
    )
    esc = submissions[0]
    assert esc.task_id == '200'
    assert esc.severity == 'blocking'
    assert esc.category == 'reconcile_persistent_rejection'
    assert not hasattr(harness, '_reconcile_failure_counts')

    # A second consecutive rejection escalates AGAIN — there is no per-tid
    # dedup counter left to suppress it.
    await harness._reconcile_stranded_in_progress()
    assert len(submissions) == 2, (
        f'expected a second L1 submission on the next rejection (no dedup '
        f'counter remains), got {len(submissions)}'
    )


# test_reconcile_persistent_citation_miss_escalates_l1,
# test_degenerate_zero_commit_branch_suppresses_citation_missing,
# test_citation_missing_still_escalates_when_branch_advanced, and
# test_citation_missing_skipped_for_degenerate_ref_via_primitive were
# removed (task 2243, W10-θ2 step-4): all four asserted the retired Guard-2
# citation-miss skip/escalate mechanism (find_task_citation_commit,
# harness._reconcile_skip_counts, _escalate_reconcile_skip) that lived
# inside the now-deleted inline is_ancestor archaeology block —
# TaskGroundTruth's ON_MAIN/GONE_WITH_MERGE_MARKER resolution carries no
# citation check or skip-counter equivalent (a citation-less on-main branch
# now resolves straight to MARK_DONE_WITH_PROVENANCE). The degenerate-branch
# recovery property they also covered (revert instead of phantom-done) is
# picked back up by the degenerate-branch parity tests deferred to plan
# step-7/8. See the design decision on task 2243's plan and esc-2243-4.


# test_failure_counter_resets_on_success was removed (task 2243, W10-θ2
# step-13/14): it asserted that a successful mark_done cleared the per-tid
# `_reconcile_failure_counts` counter — that counter (and its
# MAX_RECONCILE_FAILURES threshold gate) is retired entirely, so there is no
# longer any counter state to reset. See
# test_rejection_escalates_immediately_no_counter_threshold above for the
# replacement parity coverage (a rejection now escalates directly, with no
# per-tid tally to clear on the next success).


# ---------------------------------------------------------------------------
# task 2677 step-5/step-6 — done_evidence_stale must never be treated as
# marked_done, must not file the generic L1 reconcile_persistent_rejection
# (that path is for OTHER SetTaskStatusRejected subclasses), must not
# release the warm lane (the task is NOT done), and must file exactly one
# born-at-L2 provenance_conflict escalation via the shared
# ProvenanceConflictSink — folding (not duplicating) on a same-reopen_at
# repeat.
# ---------------------------------------------------------------------------

def _wire_stale_evidence_mark_done(harness: Harness, *, evidence_commit: str = 'deadbeef'):
    """Replace scheduler.mark_done with one that always rejects as stale.

    Mirrors the existing ``_always_reject`` / ``_flaky_mark_done`` helpers
    above but raises the new ``StaleEvidenceRejection`` subclass.
    """
    from orchestrator.scheduler import StaleEvidenceRejection

    async def _stale_reject(tid, *, kind, sha, note=None):  # noqa: ARG001
        raise StaleEvidenceRejection(
            task_id=tid,
            evidence_commit=evidence_commit,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at='2026-07-15T00:00:00+00:00',
            agent_id='claude-recon-x',
            raw="success=False payload={'error': 'done_evidence_stale'}",
        )
    harness.scheduler.mark_done = AsyncMock(side_effect=_stale_reject)  # type: ignore[attr-defined]


@pytest.mark.asyncio
class TestReconcileOneStrandedStaleEvidenceConflict:
    async def test_returns_stale_conflict_not_marked_done(
        self, harness: Harness, tmp_path: Path,
    ):
        """_reconcile_one_stranded must report the honest 'stale_conflict'
        disposition — not the misleading 'marked_done' — and must not
        release the warm lane, since the task is not actually done."""
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        tid = '9101'
        advanced_sha = 'c3' * 20
        _bind_landed_row(tmp_path, task_id=tid, advanced_sha=advanced_sha)
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                'status': 'in-progress',
                'metadata': {'reopen_at': '2026-07-15T00:00:00+00:00'},
            },
        )
        harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
        harness._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=harness._escalation_queue,
        )
        harness.git_ops.release_lane_for_terminal_task = AsyncMock(  # type: ignore[attr-defined]
            return_value=False,
        )
        _wire_stale_evidence_mark_done(harness, evidence_commit=advanced_sha)

        outcome = await harness._reconcile_one_stranded(tid, 'in-progress', mid_run=False)

        assert outcome == 'stale_conflict', f'expected stale_conflict, got {outcome!r}'
        harness.git_ops.release_lane_for_terminal_task.assert_not_called()  # type: ignore[attr-defined]

    async def test_sweep_files_one_l2_not_an_l1_and_dedupes_on_repeat(
        self, harness: Harness, tmp_path: Path,
    ):
        """The full sweep (_reconcile_stranded_in_progress) must route a
        stale-evidence rejection to exactly one pending L2
        provenance_conflict — never the generic L1
        reconcile_persistent_rejection — and a second sweep at the same
        reopen_at must not create a second pending record (should_skip
        short-circuit or dedupe_count fold — either mechanism is
        acceptable; only the outcome is pinned)."""
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        tid = '9102'
        advanced_sha = 'd4' * 20
        _bind_landed_row(tmp_path, task_id=tid, advanced_sha=advanced_sha)
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                'status': 'in-progress',
                'metadata': {'reopen_at': '2026-07-15T00:00:00+00:00'},
            },
        )
        harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
        harness._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=harness._escalation_queue,
        )
        harness.git_ops.release_lane_for_terminal_task = AsyncMock(  # type: ignore[attr-defined]
            return_value=False,
        )
        _wire_stale_evidence_mark_done(harness, evidence_commit=advanced_sha)

        await harness._reconcile_stranded_in_progress()

        pending = harness._escalation_queue.get_by_task(tid, status='pending')
        assert not any(e.category == 'reconcile_persistent_rejection' for e in pending), (
            'a stale-evidence rejection must not be escalated as a generic '
            'persistence-layer rejection (wrong escalation category)'
        )
        conflicts = [e for e in pending if e.category == 'provenance_conflict']
        assert len(conflicts) == 1, f'expected exactly one pending L2, got {len(conflicts)}'
        assert conflicts[0].level == 2
        assert conflicts[0].severity == 'urgent'

        # Second full sweep, unchanged reopen_at.
        await harness._reconcile_stranded_in_progress()
        pending_after = harness._escalation_queue.get_by_task(tid, status='pending')
        conflicts_after = [e for e in pending_after if e.category == 'provenance_conflict']
        assert len(conflicts_after) == 1, (
            f'expected still exactly one pending L2 after a second sweep, '
            f'got {len(conflicts_after)}'
        )


# ---------------------------------------------------------------------------
# R3: mid_run alive owner_pid (this run's harness PID) → fall through (Stage 3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mid_run_alive_owner_pid_not_in_dispatch_recovers(
    harness: Harness, tmp_path: Path,
):
    """R3 fix: in mid_run sweep, an alive owner_pid that isn't in the
    dispatch table represents a workflow that exited without releasing the
    lock — the sweep must fall through to recovery rather than skip.

    Pre-fix: ``if owner_alive: continue`` skipped these tasks forever
    because harness.pid (this PID) is alive throughout the run.
    """
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = mock_lock_table()  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'400': 'in-progress'}, None,
    )

    # Create a worktree with plan.lock pointing to OUR pid (harness.pid).
    lock_dir = harness.git_ops.worktree_base / '400' / '.task'
    lock_dir.mkdir(parents=True)
    (lock_dir / 'plan.lock').write_text(json.dumps({
        'session_id': '400-x', 'locked_at': datetime.now(UTC).isoformat(),
        'owner_pid': os.getpid(),
    }))

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    # Stage 3: must fall through to revert (was skipped pre-fix).
    assert changed == 1
    harness.scheduler.set_task_status.assert_called_once_with('400', 'pending')  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_startup_alive_owner_pid_left_alone(harness: Harness):
    """At startup (mid_run=False), an alive owner_pid still skips recovery.

    R3 only changes mid_run behaviour — the startup path retains the
    historical skip-on-alive contract.
    """
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'401': 'in-progress'}, None,
    )

    lock_dir = harness.git_ops.worktree_base / '401' / '.task'
    lock_dir.mkdir(parents=True)
    (lock_dir / 'plan.lock').write_text(json.dumps({
        'session_id': '401-x', 'locked_at': datetime.now(UTC).isoformat(),
        'owner_pid': os.getpid(),
    }))

    changed = await harness._reconcile_stranded_in_progress(mid_run=False)

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mid_run_cancel_window_grace(harness: Harness, tmp_path: Path):
    """R3 race guard: a workflow whose cancel_event was set within the grace
    window is skipped — its finally: block may still be writing state.

    Outside the window the sweep proceeds.
    """
    harness.scheduler._dispatched = set()  # type: ignore[attr-defined]
    harness.scheduler.lock_table = mock_lock_table()  # type: ignore[attr-defined]

    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'500': 'in-progress'}, None,
    )

    # Stamp cancel-time NOW — within grace window → skip.
    import time as _time
    harness._workflow_cancel_at['500'] = _time.monotonic()

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)
    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    # Stamp cancel-time well in the past → proceed.
    harness._workflow_cancel_at['500'] = (
        _time.monotonic() - harness.scheduler._RECONCILE_CANCEL_GRACE_S - 1
    )
    changed = await harness._reconcile_stranded_in_progress(mid_run=True)
    # Task has no worktree → orphan revert.
    assert changed == 1
    harness.scheduler.set_task_status.assert_called_once_with('500', 'pending')  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# task 2243 (W10-θ2) step-11 — LEAVE parity + race-guard removal.
#
# _reconcile_stranded_in_progress's own `if mid_run and
# self.scheduler.is_actively_held(tid): continue` guard duplicates what
# recovery_for already derives (is_actively_held folds into
# report.live_claimant, and ANY live claimant collapses every _RECOVERY row
# to the LEAVE default). Step-12 deletes that driver-level guard together
# with _reconcile_one_stranded's own has_open_l1(tid) veto (L1-only; the
# resolver's row (f) / LEAVE default already folds an open escalation at
# ANY level). Both tests below are RED under the pre-step-12 code: the
# live-claimant case is currently protected ONLY by the driver's own
# continue (so _reconcile_one_stranded is never even called), and the
# open-L2-escalation case is currently NOT protected at all (has_open_l1
# misses it and the sweep falls through to an incorrect revert).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mid_run_live_claimant_not_shortcut_by_driver_guard(harness: Harness):
    """A live-claimant (is_actively_held) in-progress task is still left
    untouched once the driver's own continue-guard is removed — the DRIVER
    still calls _reconcile_one_stranded unconditionally for every candidate
    (no driver-level continue reintroduced; task 2243 step-12 parity).

    RED under the pre-step-12 code: the driver's `if mid_run and
    self.scheduler.is_actively_held(tid): continue` guard 'continue's before
    _reconcile_one_stranded is ever called, so the spy below is never
    invoked and the assertion fails.

    Review amendment (reviewer_comprehensive #1): _reconcile_one_stranded
    itself now short-circuits on this same is_actively_held signal BEFORE
    calling recovery_for (cheap, in-memory — avoids paying for
    derive_truth's git archaeology on every normally-running task), so
    recovery_for is no longer actually reached for this specific scenario.
    This test still pins the DRIVER-level contract (called unconditionally
    for every candidate); see
    test_mid_run_actively_held_short_circuits_before_recovery_for below for
    a direct pin of the new function-level short-circuit.
    """
    harness.scheduler._dispatched = {'70'}  # type: ignore[attr-defined]
    harness.scheduler.lock_table = mock_lock_table()  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = ({'70': 'in-progress'}, None)  # type: ignore[attr-defined]
    # No worktree/plan.lock for '70' — if the sweep ever fell through to the
    # revert applier despite the live claimant, it would revert.

    spy = AsyncMock(wraps=harness._reconcile_one_stranded)
    harness._reconcile_one_stranded = spy  # type: ignore[method-assign]

    changed = await harness._reconcile_stranded_in_progress(mid_run=True)

    spy.assert_awaited_once_with('70', 'in-progress', mid_run=True)
    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mid_run_actively_held_short_circuits_before_recovery_for(
    harness: Harness,
):
    """Review amendment (task 2243 W10-θ2, reviewer_comprehensive #1): a
    mid-run actively-held task must not pay for recovery_for's derive_truth
    (MergeProvenance lookup + git archaeology fallback + get_task) — the
    outcome is a guaranteed LEAVE (is_actively_held folds into
    report.live_claimant, and no _RECOVERY row matches a live claimant), so
    _reconcile_one_stranded returns None before ever building the resolver
    or fetching the task row.
    """
    tid = '72'
    harness.scheduler._dispatched = {tid}  # type: ignore[attr-defined]
    harness.scheduler.lock_table = mock_lock_table()  # type: ignore[attr-defined]

    harness._get_ground_truth = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError(
            '_get_ground_truth must not be called for an actively-held '
            'mid-run candidate — the short-circuit must fire first',
        ),
    )
    harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
        side_effect=AssertionError(
            'get_task must not be called for an actively-held mid-run '
            'candidate — the short-circuit must fire first',
        ),
    )

    result = await harness._reconcile_one_stranded(tid, 'in-progress', mid_run=True)

    assert result is None
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
    harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_in_progress_on_main_with_open_l2_escalation_leaves_untouched(
    harness: Harness, tmp_path: Path,
):
    """An in-progress task whose branch is already on main but carries an
    OPEN escalation at level 2 (not level 1) must be left alone —
    recovery_for's row (f) folds an open escalation at ANY level, not just
    level 1, into the LEAVE veto.

    RED under current code: the in-function guard checks
    self._escalation_queue.has_open_l1(tid) (level=1 ONLY), which misses an
    L2-only escalation and falls through to the revert applier.
    """
    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = ({'71': 'in-progress'}, None)  # type: ignore[attr-defined]
    # No plan.lock/worktree for '71' — if the sweep fell through to the
    # revert applier, it would revert (no-lock orphan path).

    harness._escalation_queue = EscalationQueue(tmp_path / 'esc_l2_on_main')
    harness._escalation_queue.submit(Escalation(
        id=harness._escalation_queue.make_id('71'),
        task_id='71', agent_role='steward', severity='critical',
        category='infra_issue', summary='open L2, on-main evidence present',
        level=2, status='pending',
    ))

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# R4: blocked-task pass (Stage 4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocked_with_branch_on_main_marked_done(harness: Harness):
    """Blocked task whose branch is already on main → marked done.

    Out-of-band-merged-while-blocked recovery: a human merged the branch
    manually, leaving the row in 'blocked'. Next sweep should mark it done.
    """
    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'600': 'blocked'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 1
    # Provenance note distinguishes blocked-merge from in-progress-merge.
    harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
        '600', 'done',
        done_provenance={
            'kind': 'found_on_main',
            'commit': 'deadbeef' + 'a' * 32,
            'note': 'reconcile: branch on main while task was blocked (out-of-band merge)',
        },
    )


@pytest.mark.asyncio
async def test_blocked_without_on_main_evidence_left_alone(harness: Harness):
    """Blocked task with no on-main evidence → untouched.

    'blocked' is a deliberate state; we only flip it on observed evidence.
    """
    # is_ancestor=False, find_merge_marker=None (defaults from fixture)
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'601': 'blocked'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fix #1b: backstop stranded-`blocked` sweep (re-file L1, never re-pend)
# ---------------------------------------------------------------------------

def _pending(queue: EscalationQueue, tid: str) -> list[Escalation]:
    return queue.get_by_task(tid, status='pending')


@pytest.mark.asyncio
async def test_stranded_blocked_refiles_single_l1_without_status_change(
    harness: Harness, tmp_path: Path,
):
    """blocked + no on-main evidence + no open escalation + no active workflow
    → exactly one L1 re-filed, status UNCHANGED (never re-pended)."""
    harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
    harness.config.stranded_blocked_escalate_enabled = True
    # is_ancestor=False, find_merge_marker=None (fixture defaults) → no on-main.
    # recovery_for's db_status (task 2243, W10-θ2) is sourced from get_task(),
    # which the fixture derives from get_statuses() — set explicitly since this
    # test calls _reconcile_one_stranded directly, bypassing get_statuses().
    harness.scheduler.get_statuses.return_value = ({'601': 'blocked'}, None)  # type: ignore[attr-defined]

    result = await harness._reconcile_one_stranded('601', 'blocked', mid_run=False)

    assert result is None
    # Status must NOT change — re-file, never re-pend.
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
    pending = _pending(harness._escalation_queue, '601')
    assert len(pending) == 1, f'expected exactly one re-filed escalation, got {pending}'
    assert pending[0].level == 1
    assert pending[0].category == 'stranded_blocked'


@pytest.mark.asyncio
async def test_stranded_blocked_second_sweep_does_not_duplicate(
    harness: Harness, tmp_path: Path,
):
    """Once the backstop L1 is pending, a subsequent sweep must NOT re-file
    (the pending-escalation check self-dedupes)."""
    harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
    harness.config.stranded_blocked_escalate_enabled = True
    harness.scheduler.get_statuses.return_value = ({'601': 'blocked'}, None)  # type: ignore[attr-defined]

    await harness._reconcile_one_stranded('601', 'blocked', mid_run=False)
    assert len(_pending(harness._escalation_queue, '601')) == 1

    # Second sweep — the L1 from the first pass is still pending.
    await harness._reconcile_one_stranded('601', 'blocked', mid_run=False)

    assert len(_pending(harness._escalation_queue, '601')) == 1, (
        'second sweep must not file a duplicate L1'
    )


@pytest.mark.asyncio
async def test_stranded_blocked_skips_when_escalation_already_open(
    harness: Harness, tmp_path: Path,
):
    """An already-open escalation (any level) suppresses the re-file."""
    harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
    harness.config.stranded_blocked_escalate_enabled = True
    harness.scheduler.get_statuses.return_value = ({'601': 'blocked'}, None)  # type: ignore[attr-defined]
    harness._escalation_queue.submit(Escalation(
        id=harness._escalation_queue.make_id('601'),
        task_id='601', agent_role='steward', severity='blocking',
        category='design_concern', summary='pre-existing', level=1,
    ))

    await harness._reconcile_one_stranded('601', 'blocked', mid_run=False)

    pending = _pending(harness._escalation_queue, '601')
    assert len(pending) == 1, 'must not add a second escalation'
    assert pending[0].category == 'design_concern', 'pre-existing one untouched'


@pytest.mark.asyncio
async def test_stranded_blocked_skips_when_active_workflow(
    harness: Harness, tmp_path: Path,
):
    """An active workflow (task_id in _escalation_events) owns its own re-pend
    — the backstop must not re-file."""
    harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
    harness.config.stranded_blocked_escalate_enabled = True
    harness.scheduler.get_statuses.return_value = ({'601': 'blocked'}, None)  # type: ignore[attr-defined]
    harness._escalation_events['601'] = asyncio.Event()
    # Production dispatch ordering (task 2243, W10-θ2 design decision):
    # acquire_next() adds the tid to scheduler._dispatched before
    # _register_escalation_event runs, so is_actively_held is already True
    # whenever _escalation_events holds an entry — mirror that here, since
    # recovery_for's live_claimant (via is_actively_held) is now the signal
    # that suppresses re-filing, not _escalation_events membership directly.
    harness.scheduler._dispatched.add('601')  # type: ignore[attr-defined]

    await harness._reconcile_one_stranded('601', 'blocked', mid_run=False)

    assert _pending(harness._escalation_queue, '601') == []


@pytest.mark.asyncio
async def test_stranded_blocked_skips_when_recently_cancelled(
    harness: Harness, tmp_path: Path,
):
    """A recent release_workflow/cancel park (task_id in _workflow_cancel_at)
    must not be re-filed — the human is mid-handling."""
    harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
    harness.config.stranded_blocked_escalate_enabled = True
    harness.scheduler.get_statuses.return_value = ({'601': 'blocked'}, None)  # type: ignore[attr-defined]
    harness._workflow_cancel_at['601'] = _time.monotonic()

    await harness._reconcile_one_stranded('601', 'blocked', mid_run=False)

    assert _pending(harness._escalation_queue, '601') == []


@pytest.mark.asyncio
async def test_stranded_blocked_skips_when_flag_disabled(
    harness: Harness, tmp_path: Path,
):
    """Config flag off → no re-file (clean disable)."""
    harness._escalation_queue = EscalationQueue(tmp_path / 'esc')
    harness.config.stranded_blocked_escalate_enabled = False
    harness.scheduler.get_statuses.return_value = ({'601': 'blocked'}, None)  # type: ignore[attr-defined]

    await harness._reconcile_one_stranded('601', 'blocked', mid_run=False)

    assert _pending(harness._escalation_queue, '601') == []


@pytest.mark.asyncio
async def test_stranded_blocked_skips_when_no_queue(harness: Harness):
    """No escalation queue wired → no crash, no action."""
    harness._escalation_queue = None
    harness.config.stranded_blocked_escalate_enabled = True
    harness.scheduler.get_statuses.return_value = ({'601': 'blocked'}, None)  # type: ignore[attr-defined]

    result = await harness._reconcile_one_stranded('601', 'blocked', mid_run=False)

    assert result is None
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancelled_and_deferred_never_swept(harness: Harness):
    """'cancelled' (terminal-by-decision) and 'deferred' (human-deferred) are
    never touched by the sweep, even when their branch is on main."""
    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'700': 'cancelled', '701': 'deferred'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 0
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
    harness.git_ops.is_ancestor.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_blocked_with_merge_marker_marked_done(harness: Harness):
    """Blocked task whose branch was deleted but a merge marker is on main."""
    marker_sha = 'cafe' + 'b' * 36
    harness.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)  # type: ignore[attr-defined]
    # TaskGroundTruth._resolve_branch_state treats a non-None
    # resolve_branch_sha as "branch still exists" (EXISTS_OFF_MAIN) and
    # never reaches find_merge_marker at all — must resolve None here to
    # reflect the deleted-branch premise of this test (task 2243, W10-θ2).
    harness.git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    harness.scheduler.get_statuses.return_value = (  # type: ignore[attr-defined]
        {'602': 'blocked'}, None,
    )

    changed = await harness._reconcile_stranded_in_progress()

    assert changed == 1
    harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
        '602', 'done',
        done_provenance={
            'kind': 'found_on_main',
            'commit': marker_sha,
            'note': 'reconcile: merge marker found on main while task was blocked',
        },
    )


# ---------------------------------------------------------------------------
# merge-deferred guard tests (step-3 / step-4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_skips_merge_deferred_status(harness: Harness):
    """_reconcile_one_stranded('77', 'merge-deferred') returns None without
    touching the worktree, the status, or any git operations.

    Pins the explicit early-return guard that mirrors the open-L1 /unblock
    veto (harness.py:1598-1607): merge-deferred tasks are train-parked and
    must not be reaped or reverted by the stranded-reconciler.
    """
    result = await harness._reconcile_one_stranded('77', 'merge-deferred', mid_run=False)

    assert result is None, f'Expected None for merge-deferred, got {result!r}'
    # Worktree must be left intact: no cleanup, no status change, no git I/O.
    harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
    harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
    harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
    harness.git_ops.is_ancestor.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_sweep_excludes_merge_deferred(harness: Harness):
    """_reconcile_stranded_in_progress with a merge-deferred task in the
    status map must NOT dispatch _reconcile_one_stranded for that task.

    The implicit sweep_statuses filter (only 'in-progress'/'blocked') already
    prevents it, but this test pins the observable contract so a future
    expansion of sweep_statuses can never accidentally include merge-deferred.
    NOTE: Scheduler.get_statuses returns a (dict, error) tuple.
    """
    harness.scheduler.get_statuses.return_value = ({'77': 'merge-deferred'}, None)  # type: ignore[attr-defined]

    # Spy on _reconcile_one_stranded so we can assert it was never called for '77'.
    spy = AsyncMock(return_value=None)
    harness._reconcile_one_stranded = spy  # type: ignore[method-assign]

    await harness._reconcile_stranded_in_progress()

    # The spy must never have been invoked for task '77'.
    for call in spy.call_args_list:
        tid_arg = call.args[0] if call.args else call.kwargs.get('tid')
        assert tid_arg != '77', (
            f'_reconcile_one_stranded was called for merge-deferred task 77: {call}'
        )


# ===========================================================================
# Step-9 RED: warm-lane resolution in _reconcile_one_stranded /
#             _mark_in_progress_done
# ===========================================================================


def _attach_reconcile_pool(harness: Harness, size: int = 2) -> WarmLanePool:
    """Attach a WarmLanePool to harness.git_ops.warm_lane_pool.

    Must be constructed against the same worktree_base as the fixture assigns
    (h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()) so that
    is_lane/assignment_for path comparisons match.
    """
    base = harness.git_ops.worktree_base
    base.mkdir(parents=True, exist_ok=True)
    pool = WarmLanePool(worktree_base=base, size=size)
    harness.git_ops.warm_lane_pool = pool
    return pool


@pytest.mark.asyncio
async def test_reconcile_stale_lock_uses_lane_path(harness: Harness, monkeypatch):
    """_reconcile_one_stranded resolves the worktree via the pool assignment.

    When a WarmLanePool is attached and task '42' is assigned to '_lane-0',
    the lock-state classification must inspect base/'_lane-0'/.task/plan.lock
    (not the non-existent base/'42'/.task/plan.lock).  After clearing the stale
    lock, cleanup_worktree must be called with (base/'_lane-0', '42').

    Fails today because worktree_path = worktree_base / tid computes base/'42'.
    """
    pool = _attach_reconcile_pool(harness, size=2)
    base = harness.git_ops.worktree_base
    lane_path = base / '_lane-0'
    cold_path = base / '42'

    # Create the lane dir with a stale plan.lock (dead PID)
    lock_dir = lane_path / '.task'
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / 'plan.lock'
    lock_path.write_text(json.dumps({
        'session_id': '42-dead',
        'locked_at': datetime.now(UTC).isoformat(),
        'owner_pid': 99999,
    }))

    # Restore the assignment so the pool knows '42' lives in '_lane-0'
    pool.restore_assignment('42', lane_path)

    monkeypatch.setattr('orchestrator.harness._pid_alive', lambda pid: False)
    # is_ancestor False → skip the found-on-main path, go to lock-state branch
    harness.git_ops.is_ancestor = AsyncMock(return_value=False)  # type: ignore[attr-defined]

    await harness._reconcile_one_stranded('42', 'in-progress', mid_run=False)

    # Task must be reverted to pending
    harness.scheduler.set_task_status.assert_called_once_with('42', 'pending')  # type: ignore[attr-defined]
    # cleanup_worktree must have been called with the LANE path, not base/'42'
    cleanup_calls = harness.git_ops.cleanup_worktree.call_args_list  # type: ignore[attr-defined]
    cleanup_paths = [c.args[0] for c in cleanup_calls]
    assert lane_path in cleanup_paths, (
        f'cleanup_worktree must be called with lane path {lane_path}; '
        f'actual calls: {cleanup_paths}'
    )
    assert cold_path not in cleanup_paths, (
        f'cleanup_worktree must NOT be called with cold path {cold_path}'
    )


@pytest.mark.asyncio
async def test_mark_in_progress_done_uses_lane_path(harness: Harness):
    """_mark_in_progress_done resolves the worktree via the pool assignment.

    When task '42' is assigned to '_lane-0', the done path must call
    cleanup_worktree with (base/'_lane-0', '42'), not (base/'42', '42').

    Setup: force the found-on-main flip (is_ancestor→True, valid citation,
    no branch_base_sha guard), and create the lane dir so cleanup fires.

    Fails today because worktree_path = worktree_base / tid computes base/'42'.
    """
    pool = _attach_reconcile_pool(harness, size=2)
    base = harness.git_ops.worktree_base
    lane_path = base / '_lane-0'
    cold_path = base / '42'

    # Create the lane dir on disk so cleanup_worktree has something to act on
    lane_path.mkdir(parents=True, exist_ok=True)

    # Restore the assignment so the pool maps '42' → '_lane-0'
    pool.restore_assignment('42', lane_path)

    # Force the found-on-main path: is_ancestor→True. get_task must still
    # carry status='in-progress' — TaskGroundTruth.derive_truth sources
    # db_status from this same get_task(tid) call (task 2243, W10-θ2), so a
    # bare None row (the pre-migration "no metadata" stand-in) would resolve
    # db_status='' and never match the _RECOVERY table's IN_PROGRESS row.
    harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
        return_value={'status': 'in-progress', 'metadata': {}},
    )

    result = await harness._reconcile_one_stranded('42', 'in-progress', mid_run=False)

    assert result == 'marked_done', f'Expected marked_done, got {result!r}'

    # cleanup_worktree must have been called with the LANE path
    cleanup_calls = harness.git_ops.cleanup_worktree.call_args_list  # type: ignore[attr-defined]
    cleanup_paths = [c.args[0] for c in cleanup_calls]
    assert lane_path in cleanup_paths, (
        f'cleanup_worktree must be called with lane path {lane_path}; '
        f'actual calls: {cleanup_paths}'
    )
    assert cold_path not in cleanup_paths, (
        f'cleanup_worktree must NOT be called with cold path {cold_path}'
    )

# ---------------------------------------------------------------------------
# Step-3 RED: _reconcile_stranded_in_progress err/empty guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconcileStrandedResolverGuard:
    """RED tests: _reconcile_stranded_in_progress must guard on resolver errors."""

    async def test_case_a_transient_error_with_data_aborts_sweep_warning(
        self, harness: Harness, caplog
    ):
        """CASE A: get_statuses returns error tuple → sweep aborted, WARNING emitted.

        Pre-fix: err is discarded (statuses, _ = ...), loop iterates {'5': 'in-progress'}
        and calls _reconcile_one_stranded('5', ...).  After fix: returns 0 without
        calling _reconcile_one_stranded, and emits WARNING naming 'error'.
        """
        harness.scheduler.get_statuses = AsyncMock(  # type: ignore[attr-defined]
            return_value=({'5': 'in-progress'}, RuntimeError('mcp down'))
        )

        with (
            patch.object(harness, '_reconcile_one_stranded', new_callable=AsyncMock) as mock_one,
            caplog.at_level(logging.WARNING, logger='orchestrator.harness'),
        ):
            result = await harness._reconcile_stranded_in_progress()

        assert result == 0, f'Expected 0, got {result!r}'
        mock_one.assert_not_awaited()  # pre-fix fails here (loop iterates '5')
        assert any(
            r.levelno >= logging.WARNING and 'error' in r.message.lower()
            for r in caplog.records
        ), f'Expected WARNING naming error; got: {[r.message for r in caplog.records]!r}'

    async def test_case_b_genuine_empty_returns_zero_with_debug(
        self, harness: Harness, caplog
    ):
        """CASE B: get_statuses returns ({}, None) → returns 0 + DEBUG naming 'empty'.

        An empty task tree is a normal idle state (no tasks yet), so we log at
        DEBUG (not WARNING) to avoid recurring noise.  This is different from
        CASE A where an error forces a WARNING.
        """
        harness.scheduler.get_statuses = AsyncMock(  # type: ignore[attr-defined]
            return_value=({}, None)
        )

        with caplog.at_level(logging.DEBUG, logger='orchestrator.harness'):
            result = await harness._reconcile_stranded_in_progress()

        assert result == 0, f'Expected 0, got {result!r}'
        assert any(
            r.levelno == logging.DEBUG and 'empty' in r.message.lower()
            for r in caplog.records
        ), f'Expected DEBUG naming empty; got: {[r.message for r in caplog.records]!r}'


# ---------------------------------------------------------------------------
# task 2794 — the delivered-capability ground-truth guard.
#
# Structurally mirrors TestReconcileOneStrandedEffectPresentGuard (above):
# the same ON_MAIN git-fallback setup (is_ancestor True, resolve_branch_sha
# <sha>, commit_effect_present_in_main default True so control reaches the
# NEW guard AFTER the effect-present downgrade), but with a truthy
# metadata.delivered_checks and the guard patched to return each row of the
# acceptance matrix. The capability guard asks the orthogonal question git
# attribution + effect-present cannot: is the task's OWN declared capability
# actually present on main? A hollow-done (attribution/effect present, but
# the declared capability absent) must re-dispatch (in-progress) or be left
# alone (blocked), never stamped found_on_main.
#
# RETARGETED by task 3057 step-19: the patch target moved DOWN a layer, from
# the low-level `verify_delivered_checks_on_main` runner to the shared
# `gate_mark_done_on_delivered_checks` DECISION that all eleven
# attribution-shaped stamp seams now route through. Every behavioural
# assertion below is preserved verbatim — only the staged mock return values
# change (DeliveredChecksVerdict -> DeliveredChecksBlock | None). That is
# what proves the generalization is behaviour-preserving on the one arm that
# already had coverage.
#
# Consequence of single-sourcing: the harness no longer decides ANYTHING
# locally. Inertness (no delivered_checks), the kill switch, and main-SHA
# resolution all moved INTO the helper, so rows that used to assert
# "helper not awaited" now assert "delegated, with the right inputs". The
# coverage those rows lost is not gone — it is pinned AT SOURCE in
# test_delivered_check_gate.py::TestGateMarkDoneOnDeliveredChecks.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestReconcileOneStrandedDeliveredChecksGuard:
    _PATCH_TARGET = 'orchestrator.harness.gate_mark_done_on_delivered_checks'

    @staticmethod
    def _gate(block: DeliveredChecksBlock | None) -> AsyncMock:
        """Stand-in for the shared helper that returns *block*.

        Faithful to the real helper in the one respect these rows assert on:
        a ``reason='failed'`` decision emits its WARNING — naming the task,
        the failed check, its pattern and the main SHA — on the CALLER-SUPPLIED
        ``log``, never on the delivered_checks module logger. Task 2794's
        caplog assertions therefore keep testing something real after the
        retarget: they pass only if the harness hands its own module logger
        through.
        """
        async def _impl(task_id, metadata, *, log, **kwargs):
            if block is not None and block.reason == 'failed':
                failed = block.failed_check or {}
                log.warning(
                    'Task %s: delivered check %r (pattern %r) is NOT present '
                    'on main %s — withholding the done stamp',
                    task_id, failed.get('name'), failed.get('pattern'),
                    block.main_sha,
                )
            return block

        return AsyncMock(side_effect=_impl)

    def _on_main_in_progress(self, harness: Harness, tid: str, *, sha: str,
                             main_sha: str, metadata: dict) -> None:
        """Common ON_MAIN git-fallback setup for an in-progress stranded task
        whose effect IS present (so the effect-present guard passes and
        control reaches the new delivered-checks guard).

        With an active commit-citation pattern (dark-factory's default), the
        ON_MAIN landing sha is the CITATION commit, not the raw branch tip
        (task_ground_truth._resolve_branch_state) — so find_task_citation_commit
        is what determines report.branch_state.sha (and thus the found_on_main
        provenance commit), pinned here to *sha*.
        """
        harness.scheduler.get_statuses.return_value = ({tid: 'in-progress'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'in-progress', 'metadata': metadata},
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(return_value=sha)  # type: ignore[attr-defined]
        harness.git_ops.find_task_citation_commit = AsyncMock(return_value=sha)  # type: ignore[attr-defined]
        harness.git_ops.get_main_sha = AsyncMock(return_value=main_sha)  # type: ignore[attr-defined]

    # --- row 2: FAILED, in-progress -> re-dispatch (revert to pending) ------

    async def test_failed_in_progress_reverts_and_warns(
        self, harness: Harness, caplog,
    ):
        """A FAILED delivered check on an in-progress task means the declared
        capability is provably absent from main: NOT marked done, reverted to
        pending for re-dispatch, and a WARNING naming the task, the failed
        check, its pattern, and the main SHA."""
        tid = '2794001'
        sha = 'deadbeef' + 'a' * 32
        main_sha = 'ma' * 20
        failing = {'name': 'cap-x', 'kind': 'grep', 'pattern': 'SomePattern'}
        self._on_main_in_progress(
            harness, tid, sha=sha, main_sha=main_sha,
            metadata={'delivered_checks': [failing]},
        )
        helper = self._gate(DeliveredChecksBlock(
            reason='failed', main_sha=main_sha, failed_check=failing,
        ))

        with patch(self._PATCH_TARGET, helper), \
                caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'pending',
        )
        assert result == 1
        helper.assert_awaited_once()
        # The WARNING now comes FROM the helper, but must still be addressable
        # as 'orchestrator.harness' — that holds only because the harness
        # passes its own module logger through as log=. Pinned directly...
        assert helper.await_args is not None
        assert helper.await_args.kwargs['log'] is logging.getLogger(
            'orchestrator.harness',
        )
        # ...and end-to-end: _gate emits on whatever `log` it was handed, so
        # these caplog-content assertions (task 2794's, verbatim) pass ONLY
        # if that logger is the harness one.
        assert tid in caplog.text
        assert 'cap-x' in caplog.text
        assert 'SomePattern' in caplog.text
        assert main_sha in caplog.text

    # --- row 3: all_delivered -> mark done (unchanged) ----------------------

    async def test_all_delivered_marks_done(self, harness: Harness):
        """All checks DELIVERED on main -> the capability is present, so the
        stranded in-progress task is stamped found_on_main exactly as today."""
        tid = '2794002'
        sha = 'deadbeef' + 'b' * 32
        main_sha = 'mb' * 20
        check = {'name': 'cap-y', 'kind': 'grep', 'pattern': 'Pat', 'expect': 'present'}
        self._on_main_in_progress(
            harness, tid, sha=sha, main_sha=main_sha,
            metadata={'delivered_checks': [check]},
        )
        helper = self._gate(None)

        with patch(self._PATCH_TARGET, helper):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={'kind': 'found_on_main', 'commit': sha, 'note': ANY},
        )
        assert result == 1
        helper.assert_awaited_once()

    # --- row 4: no delivered_checks -> DELEGATED, and inertly marks done ----

    async def test_no_delivered_checks_marks_done_and_delegates(
        self, harness: Harness,
    ):
        """A check-less task must not gain a new requirement: it is stamped
        found_on_main exactly as today.

        Changed by 3057 step-19: the harness now DELEGATES unconditionally
        rather than short-circuiting on `metadata.delivered_checks` itself —
        the inertness lives in the shared helper (one implementation for all
        eleven seams), pinned at source by
        test_delivered_check_gate.py::TestGateMarkDoneOnDeliveredChecks row 4,
        which additionally asserts the zero-I/O property (no get_main_sha, no
        check run) that cannot be observed from here.
        """
        tid = '2794003'
        sha = 'deadbeef' + 'c' * 32
        self._on_main_in_progress(
            harness, tid, sha=sha, main_sha='mc' * 20, metadata={},
        )
        helper = self._gate(None)

        with patch(self._PATCH_TARGET, helper):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={'kind': 'found_on_main', 'commit': sha, 'note': ANY},
        )
        assert result == 1
        helper.assert_awaited_once()
        assert helper.await_args is not None
        assert 'delivered_checks' not in helper.await_args.args[1]

    # --- row 5: ERRORED/timeout -> fail-safe (no mark, no revert) ------------

    async def test_errored_is_fail_safe_no_action(self, harness: Harness):
        """An ERRORED verdict (a check could not be evaluated) is fail-safe:
        make no claim either way — do not mark done, do not revert, leave the
        task in-progress to be retried next sweep (result uncounted)."""
        tid = '2794004'
        main_sha = 'md' * 20
        check = {'name': 'cap-z', 'kind': 'grep', 'pattern': 'Pat'}
        self._on_main_in_progress(
            harness, tid, sha='deadbeef' + 'd' * 32, main_sha=main_sha,
            metadata={'delivered_checks': [check]},
        )
        helper = self._gate(DeliveredChecksBlock(
            reason='errored', main_sha=main_sha,
        ))

        with patch(self._PATCH_TARGET, helper):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0
        helper.assert_awaited_once()

    # --- main-sha unresolved -> fail-safe (sibling of ERRORED) --------------

    async def test_main_sha_unresolved_is_fail_safe(self, harness: Harness):
        """An unresolvable main SHA means the checks have no ref to run
        against: fail-safe no-op (no mark, no revert), retried next sweep.

        Changed by 3057 step-19: main-SHA resolution moved INTO the shared
        helper, so this row now stages the resulting decision rather than the
        git failure that produces it. The raises-vs-empty-string distinction
        this test used to own is not lost — it is pinned AT SOURCE, as two
        separate rows, in
        test_delivered_check_gate.py::TestGateMarkDoneOnDeliveredChecks
        (get_main_sha raising, and get_main_sha returning ''), where the
        helper's own fail-safe arms live.
        """
        tid = '2794005'
        check = {'name': 'cap-z', 'kind': 'grep', 'pattern': 'Pat'}
        self._on_main_in_progress(
            harness, tid, sha='deadbeef' + 'e' * 32, main_sha='unused',
            metadata={'delivered_checks': [check]},
        )
        harness.git_ops.get_main_sha = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError('git rev-parse failed'),
        )
        helper = self._gate(DeliveredChecksBlock(reason='main_sha_unresolved'))

        with patch(self._PATCH_TARGET, helper):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0
        helper.assert_awaited_once()

    # --- main_sha_unresolved must NOT revert, unlike 'failed' ---------------

    async def test_main_sha_unresolved_does_not_revert_in_progress(
        self, harness: Harness,
    ):
        """The fail-safe reasons are DISTINCT from 'failed' at this seam.

        'failed' is a definitive absence and drives revert-to-pending;
        'main_sha_unresolved' makes no claim either way, so an in-progress
        task must be left exactly as it is. Pinned separately so a refactor
        that collapsed the two reasons into one branch — losing the
        distinction 2794 established — fails here.
        """
        tid = '2794010'
        check = {'name': 'cap-z', 'kind': 'grep', 'pattern': 'Pat'}
        self._on_main_in_progress(
            harness, tid, sha='deadbeef' + '0' * 32, main_sha='m0' * 20,
            metadata={'delivered_checks': [check]},
        )
        helper = self._gate(DeliveredChecksBlock(reason='main_sha_unresolved'))

        with patch(self._PATCH_TARGET, helper):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0

    # --- row 6: FAILED, blocked -> leave untouched (blocked discipline) ------

    async def test_failed_blocked_leaves_task_untouched(self, harness: Harness):
        """A FAILED delivered check on a stranded 'blocked' task must NOT
        silently flip it: blocked discipline forbids a blocked->pending
        revert (mirrors the effect-present blocked case), so the result is no
        action at all — no mark_done, no set_task_status."""
        tid = '2794006'
        main_sha = 'mf' * 20
        failing = {'name': 'cap-b', 'kind': 'grep', 'pattern': 'BPat'}
        harness.scheduler.get_statuses.return_value = ({tid: 'blocked'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'blocked',
                          'metadata': {'delivered_checks': [failing]}},
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(  # type: ignore[attr-defined]
            return_value='deadbeef' + 'f' * 32,
        )
        harness.git_ops.get_main_sha = AsyncMock(return_value=main_sha)  # type: ignore[attr-defined]
        helper = self._gate(DeliveredChecksBlock(
            reason='failed', main_sha=main_sha, failed_check=failing,
        ))

        with patch(self._PATCH_TARGET, helper):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0
        helper.assert_awaited_once()

    # --- blocked + ERRORED -> leave untouched (blocked discipline + fail-safe) -

    async def test_errored_blocked_leaves_task_untouched(self, harness: Harness):
        """An ERRORED delivered-check verdict on a stranded 'blocked' task
        takes the same no-action ``return None`` path as the blocked+FAILED
        case: the ERRORED arm is fail-safe (make no claim either way) AND
        blocked discipline forbids a blocked->pending flip — so no mark_done,
        no set_task_status. Parallel to test_failed_blocked_leaves_task_untouched;
        together they pin blocked discipline across BOTH non-satisfied
        outcomes (the helper IS consulted here, unlike the main-sha-unresolved
        fail-safe which short-circuits before it)."""
        tid = '2794009'
        main_sha = 'mg' * 20
        check = {'name': 'cap-be', 'kind': 'grep', 'pattern': 'BEPat'}
        harness.scheduler.get_statuses.return_value = ({tid: 'blocked'}, None)  # type: ignore[attr-defined]
        harness.scheduler.get_task = AsyncMock(  # type: ignore[attr-defined]
            return_value={'status': 'blocked',
                          'metadata': {'delivered_checks': [check]}},
        )
        harness.git_ops.is_ancestor = AsyncMock(return_value=True)  # type: ignore[attr-defined]
        harness.git_ops.resolve_branch_sha = AsyncMock(  # type: ignore[attr-defined]
            return_value='deadbeef' + '9' * 32,
        )
        harness.git_ops.get_main_sha = AsyncMock(return_value=main_sha)  # type: ignore[attr-defined]
        helper = self._gate(DeliveredChecksBlock(
            reason='errored', main_sha=main_sha,
        ))

        with patch(self._PATCH_TARGET, helper):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        assert result == 0
        helper.assert_awaited_once()

    # --- kill switch: enabled=False -> FORWARDED, and inertly marks done ----

    async def test_kill_switch_disabled_is_forwarded_and_marks_done(
        self, harness: Harness,
    ):
        """config.delivered_checks.enabled=False turns the whole guard off,
        and the task is STILL marked done.

        Changed by 3057 step-19: the harness FORWARDS `enabled=False` rather
        than short-circuiting on it. That is the point of single-sourcing —
        the kill switch has exactly one implementation, so one hot reload
        disarms all eleven attribution-shaped seams together instead of
        eleven local checks drifting apart. The inert-when-disabled behaviour
        itself is pinned at source in
        test_delivered_check_gate.py::TestGateMarkDoneOnDeliveredChecks row 6,
        which also asserts that neither get_main_sha nor the check runner is
        awaited.
        """
        tid = '2794007'
        sha = 'deadbeef' + '7' * 32
        failing = {'name': 'cap-k', 'kind': 'grep', 'pattern': 'KPat'}
        harness.config.delivered_checks.enabled = False
        self._on_main_in_progress(
            harness, tid, sha=sha, main_sha='mk' * 20,
            metadata={'delivered_checks': [failing]},
        )
        # The REAL helper is inert when disabled, so stage its inert answer.
        helper = self._gate(None)

        with patch(self._PATCH_TARGET, helper):
            result = await harness._reconcile_stranded_in_progress()

        harness.scheduler.set_task_status.assert_awaited_once_with(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={'kind': 'found_on_main', 'commit': sha, 'note': ANY},
        )
        assert result == 1
        helper.assert_awaited_once()
        assert helper.await_args is not None
        assert helper.await_args.kwargs['enabled'] is False


# ===========================================================================
# Task 3057 step-23 (RED) — the withheld train member's recovery edge.
#
# All four delivered-checks train seams (harness `mark_member_done` /
# `redrive_member`, workflow's inline `_mark_member_done` /
# `_attribute_train_failure` solo-passer) originally withheld the done-flip by
# RETURNING with the member's status untouched at 'merge-deferred', on the
# documented promise that the stranded sweep would re-evaluate it.  That loop
# does not exist.  'merge-deferred' is a status with NO recovery edge at all:
#
#   1. `_RECONCILE_SWEEP_STATUSES` excludes it, so the sweep never iterates it;
#   2. `_reconcile_one_stranded` early-returns on it even if the sweep did;
#   3. the train already advanced main, so the merge worker will not re-drive;
#   4. `_WARM_LANE_RECLAIM_PROTECTED_STATUSES` contains it, so even its warm
#      lane leaks (`release_lane_for_terminal_task` is on the skipped
#      fall-through, and the member is not terminal).
#
# This class pins (1), (2) and (4) against the REAL imported constants — not
# literals — so a future edit to either filter surfaces here, and pins that
# the status the withhold path reverts to actually satisfies the
# reclaim-eligibility PREDICATE rather than a hardcoded 'pending'.
# ===========================================================================

_WEDGE_TRAIN = 'train-wedge'
_WEDGE_MID = '7001'
_WEDGE_CHECK = {
    'name': 'cap-x', 'kind': 'grep', 'pattern': 'SomePattern', 'expect': 'present',
}


class _RecordingScheduler:
    """Scheduler double that HOLDS the member's status like the real one.

    Deliberately not a MagicMock: the property under test is what the member's
    status actually BECOMES, which an auto-mock cannot express.
    """

    def __init__(self, mid: str = _WEDGE_MID, status: str = 'merge-deferred') -> None:
        self.mid = mid
        self.status = status
        self.marked_done: list[str] = []
        self.clear_requeue_count = MagicMock()

    async def get_statuses(self, ids):  # noqa: ANN001, ANN202
        return ({i: self.status for i in ids if i == self.mid}, None)

    async def get_task(self, tid):  # noqa: ANN001, ANN202
        return {'id': tid, 'metadata': {'delivered_checks': [_WEDGE_CHECK]}}

    async def mark_done(self, tid, **kwargs):  # noqa: ANN001, ANN003, ANN202
        self.marked_done.append(tid)
        self.status = 'done'

    async def set_task_status(self, tid, status, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if tid == self.mid:
            self.status = status


def _wedge_config(tmp_path: Path) -> MagicMock:
    from orchestrator.config import DeliveredChecksConfig

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = tmp_path
    config.delivered_checks = DeliveredChecksConfig(
        enabled=True, check_timeout_secs=7.5,
    )
    return config


async def _drive_withheld_flip(tmp_path: Path) -> _RecordingScheduler:
    """Drive `mark_member_done` with a FAILED block; return the scheduler."""
    from orchestrator.harness import build_train_callback_factory

    sched = _RecordingScheduler()
    git_ops = MagicMock()
    git_ops.release_lane_for_terminal_task = AsyncMock()
    cbs = build_train_callback_factory(
        sched, git_ops, _wedge_config(tmp_path),
    )(_WEDGE_TRAIN)

    block = DeliveredChecksBlock(
        reason='failed', main_sha='m' * 40, failed_check=_WEDGE_CHECK,
    )
    with patch(
        'orchestrator.harness.gate_mark_done_on_delivered_checks',
        AsyncMock(return_value=block),
    ), patch('orchestrator.harness.MergeProvenance'):
        await cbs.mark_member_done(_WEDGE_MID, 'deadbeefcafe')

    assert sched.marked_done == [], 'the withholding must not stamp'
    return sched


class TestWithheldTrainMemberHasRecoveryEdge:
    """'merge-deferred' is a permanent wedge — the withhold path must leave it."""

    def test_merge_deferred_is_excluded_from_the_stranded_sweep(self) -> None:
        """Filter (1): the sweep never even iterates a merge-deferred task."""
        from orchestrator import harness as harness_mod

        assert 'merge-deferred' not in harness_mod._RECONCILE_SWEEP_STATUSES, (
            'if merge-deferred is ever swept, revisit the withhold recovery '
            'edge — the sweep would then be a real second recovery path'
        )

    def test_merge_deferred_lane_is_protected_from_reclaim(self) -> None:
        """Filter (4): the leaked warm lane cannot be reclaimed either."""
        from orchestrator import harness as harness_mod

        assert 'merge-deferred' in harness_mod._WARM_LANE_RECLAIM_PROTECTED_STATUSES

    @pytest.mark.asyncio
    async def test_reconcile_one_stranded_is_not_the_recovery_edge(
        self, harness: Harness,
    ) -> None:
        """Filter (2): the early return is real, so the sweep cannot recover it.

        Pinned HERE (not only in ``test_skips_merge_deferred_status``) because
        the four train seams' recovery design depends on this specific fact.
        """
        result = await harness._reconcile_one_stranded(
            _WEDGE_MID, 'merge-deferred', mid_run=False,
        )

        assert result is None
        harness.scheduler.mark_done.assert_not_called()  # type: ignore[attr-defined]
        harness.scheduler.set_task_status.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_withheld_member_leaves_merge_deferred(
        self, tmp_path: Path,
    ) -> None:
        sched = await _drive_withheld_flip(tmp_path)

        assert sched.status != 'merge-deferred', (
            'a member left at merge-deferred has no recovery edge at all'
        )

    @pytest.mark.asyncio
    async def test_withheld_status_satisfies_the_reclaim_predicate(
        self, tmp_path: Path,
    ) -> None:
        """The PREDICATE, never the literal 'pending'.

        ``_warm_lane_reclaim_candidates`` admits any status that is
        non-terminal and not protected; asserting the predicate means the
        lane-leak fix cannot regress if either status set is edited later.
        """
        from orchestrator import harness as harness_mod
        from orchestrator.task_status import TERMINAL_STATUSES

        sched = await _drive_withheld_flip(tmp_path)

        assert sched.status not in TERMINAL_STATUSES, (
            f'{sched.status!r} would make the member terminal-by-status'
        )
        assert sched.status not in harness_mod._WARM_LANE_RECLAIM_PROTECTED_STATUSES, (
            f'{sched.status!r} would keep the warm lane leaked'
        )

    @pytest.mark.asyncio
    async def test_withheld_members_lane_becomes_a_reclaim_candidate(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """End-to-end through the REAL reclaim-candidate provider."""
        sched = await _drive_withheld_flip(tmp_path)
        branch = f'task/{_WEDGE_MID}'
        harness.scheduler.get_statuses = AsyncMock(  # type: ignore[attr-defined]
            return_value=({branch: sched.status}, None),
        )

        admitted = await harness._warm_lane_reclaim_candidates([branch])

        assert admitted == {branch}, (
            f'status {sched.status!r} must un-leak the lane, got {admitted!r}'
        )

    @pytest.mark.asyncio
    async def test_withheld_status_matches_the_harness_hand_back_edge(
        self, tmp_path: Path,
    ) -> None:
        """Same status the factory's OWN re-dispatch edge already uses.

        ``redrive_member``'s not-on-main arm is the harness's established
        "hand this member back to the scheduler" transition, and the scheduler
        dispatches from exactly that status.  Comparing the two production
        paths (rather than a literal) keeps them from drifting apart.
        """
        from orchestrator.harness import build_train_callback_factory

        withheld = await _drive_withheld_flip(tmp_path)

        hand_back = _RecordingScheduler()
        cbs = build_train_callback_factory(
            hand_back, MagicMock(), _wedge_config(tmp_path),
        )(_WEDGE_TRAIN)
        assert cbs.redrive_member is not None
        await cbs.redrive_member(_WEDGE_MID, False, None)

        assert withheld.status == hand_back.status, (
            f'withhold reverts to {withheld.status!r} but the factory hands '
            f'members back at {hand_back.status!r}'
        )
