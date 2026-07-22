"""Tests for the verified-green stranded remediation detector.

Covers ``orchestrator.stranded_verified_green`` — the pure helpers and the
async ``detect_verified_green`` shape check that the stranded-blocked reaper
consults before submitting a lane branch directly to the merge queue
(stranding-remediation-scheduler-ergonomics-prd.md leaf α, §2.1).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import wire_scheduler_liveness_mock
from escalation.queue import EscalationQueue

from orchestrator.artifacts import TaskArtifacts
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneRecord, LaneState
from orchestrator.stranded_verified_green import VerifiedGreenMatch


def _emit_verify(
    store: EventStore, task_id: str, *, passed: bool, tip_sha: str | None,
) -> None:
    """Emit a single workflow_verify row (mirrors workflow.py:2084 shape)."""
    data: dict = {'passed': passed}
    if tip_sha is not None:
        data['tip_sha'] = tip_sha
    store.emit(EventType.workflow_verify, task_id=task_id, data=data)


class TestLastVerifiedGreenTip:
    """Unit tests for ``last_verified_green_tip(event_store, task_id)``."""

    def test_latest_passed_with_tip_wins(self, tmp_path: Path) -> None:
        """The LATEST passed row carrying a tip_sha wins over an earlier one."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, '7', passed=True, tip_sha='aaa111')
        _emit_verify(store, '7', passed=True, tip_sha='bbb222')

        assert last_verified_green_tip(store, '7') == 'bbb222'

    def test_later_failed_row_does_not_erase_earlier_green(
        self, tmp_path: Path,
    ) -> None:
        """A later FAILED re-verify does not erase the latest passed-with-tip."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, '7', passed=True, tip_sha='green9')
        _emit_verify(store, '7', passed=False, tip_sha='red9')

        assert last_verified_green_tip(store, '7') == 'green9'

    def test_later_passed_without_tip_falls_back_to_earlier_with_tip(
        self, tmp_path: Path,
    ) -> None:
        """A later passed row lacking a tip_sha does not shadow an earlier
        passed-WITH-tip: the latest passed-WITH-tip wins."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, '7', passed=True, tip_sha='withtip')
        _emit_verify(store, '7', passed=True, tip_sha=None)

        assert last_verified_green_tip(store, '7') == 'withtip'

    def test_none_when_no_passed_row_carries_a_tip(self, tmp_path: Path) -> None:
        """passed rows without a tip_sha (and empty-string tips) yield None."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, '7', passed=True, tip_sha=None)
        _emit_verify(store, '7', passed=True, tip_sha='')
        _emit_verify(store, '7', passed=False, tip_sha='red')

        assert last_verified_green_tip(store, '7') is None

    def test_none_when_no_rows(self, tmp_path: Path) -> None:
        """No workflow_verify rows for the task → None."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, 'other', passed=True, tip_sha='x')

        assert last_verified_green_tip(store, '7') is None

    def test_none_when_event_store_none(self) -> None:
        """event_store=None → None (fail-safe)."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        assert last_verified_green_tip(None, '7') is None

    def test_none_when_task_id_none(self, tmp_path: Path) -> None:
        """task_id=None → None (fail-safe)."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        assert last_verified_green_tip(store, None) is None

    def test_reads_cross_run(self, tmp_path: Path) -> None:
        """A prior-run green is visible via fetch_events_by_type_all_runs.

        The strand can span an orchestrator restart, so the green may live
        under a PRIOR run_id — a run-scoped read would miss it.
        """
        from orchestrator.stranded_verified_green import last_verified_green_tip

        db_path = tmp_path / 'runs.db'
        prior = EventStore(db_path, 'run-1')
        _emit_verify(prior, '7', passed=True, tip_sha='priorgreen')

        current = EventStore(db_path, 'run-2')
        assert last_verified_green_tip(current, '7') == 'priorgreen'


# ---------------------------------------------------------------------------
# detect_verified_green — light fakes for git_ops collaborators
# ---------------------------------------------------------------------------


class _FakeLaneLifecycle:
    def __init__(self, records: dict[str, LaneRecord]) -> None:
        self._records = records

    def all_records(self) -> dict[str, LaneRecord]:
        return self._records


class _FakeGitOps:
    """Minimal stand-in exposing exactly the surface detect_verified_green uses."""

    def __init__(
        self,
        worktree_base: Path,
        records: dict[str, LaneRecord],
        branch_shas: dict[str, str],
        *,
        raise_on_resolve: bool = False,
    ) -> None:
        self.worktree_base = worktree_base
        self._lane_lifecycle = _FakeLaneLifecycle(records)
        self._branch_shas = branch_shas
        self._raise_on_resolve = raise_on_resolve
        # git_ops.config is a GitConfig (branch_prefix lives directly on it),
        # NOT the full OrchestratorConfig — mirror that shape exactly.
        self.config = SimpleNamespace(branch_prefix='task/')

    async def resolve_branch_sha(self, branch_name: str) -> str | None:
        if self._raise_on_resolve:
            raise RuntimeError('boom (fail-safe probe)')
        return self._branch_shas.get(branch_name)


_TID = '7'
_LANE = '_lane-0'
_TIP = 'deadbeef0123456789abcdef0123456789abcdef'


def _write_plan(worktree_base: Path, lane: str, steps: list[dict]) -> None:
    meta_root = TaskArtifacts.meta_root_for(worktree_base, lane)
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / 'plan.json').write_text(json.dumps({'task_id': _TID, 'steps': steps}))


def _all_done_steps() -> list[dict]:
    return [
        {'id': 'step-1', 'status': 'done'},
        {'id': 'step-2', 'status': 'done'},
    ]


def _build_env(
    tmp_path: Path,
    *,
    state: LaneState = LaneState.ASSIGNED,
    record_task_id: str | None = _TID,
    record_branch: str | None = 'task/7',
    branch_sha: str | None = _TIP,
    resolve_key: str = 'task/7',
    verified_tip: str | None = _TIP,
    steps: list[dict] | None = None,
    empty_records: bool = False,
    raise_on_resolve: bool = False,
) -> tuple[_FakeGitOps, EventStore, Callable[[str], Path], Path]:
    """Build (git_ops, event_store, worktree_resolver, worktree) for the detector.

    Defaults describe the POSITIVE verified-green shape; override one knob per
    negative case.
    """
    worktree_base = tmp_path / 'wtbase'
    worktree_base.mkdir(parents=True, exist_ok=True)
    worktree = worktree_base / _LANE

    records: dict[str, LaneRecord] = {}
    if not empty_records:
        records[_LANE] = LaneRecord(
            state=state, task_id=record_task_id, branch=record_branch,
        )

    branch_shas: dict[str, str] = {}
    if branch_sha is not None:
        branch_shas[resolve_key] = branch_sha

    git_ops = _FakeGitOps(
        worktree_base, records, branch_shas, raise_on_resolve=raise_on_resolve,
    )

    event_store = EventStore(tmp_path / 'runs.db', 'run-1')
    if verified_tip is not None:
        _emit_verify(event_store, _TID, passed=True, tip_sha=verified_tip)

    _write_plan(worktree_base, _LANE, steps if steps is not None else _all_done_steps())

    def _resolver(tid: str) -> Path:
        return worktree

    return git_ops, event_store, _resolver, worktree


class TestDetectVerifiedGreen:
    """Async shape check for detect_verified_green (PRD §2.1)."""

    pytestmark = pytest.mark.asyncio

    async def test_positive_match(self, tmp_path: Path) -> None:
        from orchestrator.stranded_verified_green import (
            VerifiedGreenMatch,
            detect_verified_green,
        )

        git_ops, event_store, resolver, worktree = _build_env(tmp_path)
        match = await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        )
        assert isinstance(match, VerifiedGreenMatch)
        assert match.lane == _LANE
        assert match.branch == 'task/7'
        assert match.tip_sha == _TIP
        assert match.worktree == worktree

    async def test_positive_match_branch_fallback_when_record_branch_none(
        self, tmp_path: Path,
    ) -> None:
        """record.branch=None → branch derives from config prefix + task id."""
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(
            tmp_path, record_branch=None, resolve_key='task/7',
        )
        match = await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        )
        assert match is not None
        assert match.branch == 'task/7'

    async def test_none_when_no_lane_record(self, tmp_path: Path) -> None:
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(tmp_path, empty_records=True)
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_when_record_task_id_mismatch(self, tmp_path: Path) -> None:
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(
            tmp_path, record_task_id='999',
        )
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_when_record_in_use(self, tmp_path: Path) -> None:
        """IN_USE means a live workflow holds the lane — not the stranded shape."""
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(
            tmp_path, state=LaneState.IN_USE,
        )
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_when_record_released(self, tmp_path: Path) -> None:
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(
            tmp_path, state=LaneState.RELEASED,
        )
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_when_branch_tip_unresolved(self, tmp_path: Path) -> None:
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(tmp_path, branch_sha=None)
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_when_tip_mismatch(self, tmp_path: Path) -> None:
        """Lane tip advanced/stale vs the verified-green tip → no match."""
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(
            tmp_path, branch_sha='a' * 40, verified_tip='b' * 40,
        )
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_when_no_verified_green(self, tmp_path: Path) -> None:
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(tmp_path, verified_tip=None)
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_when_plan_phase_execute(self, tmp_path: Path) -> None:
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(
            tmp_path,
            steps=[{'id': 's1', 'status': 'done'}, {'id': 's2', 'status': 'pending'}],
        )
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_when_plan_phase_plan(self, tmp_path: Path) -> None:
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(tmp_path, steps=[])
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None

    async def test_none_fail_safe_on_collaborator_exception(
        self, tmp_path: Path,
    ) -> None:
        """A collaborator raising → None (never a false match, PRD §2.2)."""
        from orchestrator.stranded_verified_green import detect_verified_green

        git_ops, event_store, resolver, _ = _build_env(
            tmp_path, raise_on_resolve=True,
        )
        assert await detect_verified_green(
            _TID, git_ops=git_ops, event_store=event_store, worktree_resolver=resolver,
        ) is None


# ---------------------------------------------------------------------------
# Harness wiring — _maybe_submit_stranded_verified_green (PRD §2.1 ON MATCH)
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Harness with mocked internals for the verified-green submit wiring.

    Mirrors test_reconcile_stranded.py / test_stranded_blocked_sweep.py: mocked
    scheduler, real ``self._merge_queue`` (an asyncio.Queue built in __init__),
    and a real EventStore so the merge_queued row is queryable.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.update_task = AsyncMock(return_value=True)
    h.scheduler.get_status = AsyncMock(return_value='blocked')

    # Derive get_task's row from get_statuses() so TaskGroundTruth.derive_truth's
    # db_status reflects the same status the sweep loop reads (mirrors
    # test_stranded_blocked_sweep.py's identically-purposed default).
    def _default_get_task(tid: str) -> dict | None:
        ret = h.scheduler.get_statuses.return_value
        try:
            statuses, _err = ret
        except (TypeError, ValueError):
            return None
        status = statuses.get(tid) if isinstance(statuses, dict) else None
        return None if status is None else {'status': status, 'metadata': {}}
    h.scheduler.get_task = AsyncMock(side_effect=_default_get_task)
    wire_scheduler_liveness_mock(h.scheduler)

    # Real event store so merge_queued rows are queryable via the event API.
    h.event_store = EventStore(tmp_path / 'harness-runs.db', 'run-1')
    # module_configs_or_empty is a @property (exposed by pydantic_spec) — force
    # an empty mapping so list(...values()) yields [] for the MergeRequest.
    h.config.module_configs_or_empty = {}
    h.config.stranded_verified_green_merge_enabled = True
    return h


def _match(tmp_path: Path, *, tip: str = _TIP) -> VerifiedGreenMatch:
    """A VerifiedGreenMatch describing the happy verified-green shape for _TID."""
    return VerifiedGreenMatch(
        lane=_LANE, branch='task/7', tip_sha=tip, worktree=tmp_path / 'wt',
    )


@pytest.mark.asyncio
class TestMaybeSubmitStrandedVerifiedGreen:
    """Harness._maybe_submit_stranded_verified_green — the happy submit (step-7)."""

    async def test_happy_submit_enqueues_one_merge_request(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """On a match: exactly one MergeRequest (branch.bare_id==tid,
        snapshot_tip==match.tip_sha) is enqueued and the method returns True."""
        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            result = await harness._maybe_submit_stranded_verified_green(_TID, {})

        assert result is True
        assert harness._merge_queue.qsize() == 1
        req = harness._merge_queue.get_nowait()
        assert req.branch.bare_id == _TID
        assert req.snapshot_tip == _TIP

    async def test_submit_tags_merge_queued_event_source(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """The submission's merge_queued event carries data.source='stranded-reaper'
        (PRD §6α acceptance signal)."""
        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            await harness._maybe_submit_stranded_verified_green(_TID, {})

        rows = harness.event_store.fetch_events_by_type_all_runs(
            EventType.merge_queued, task_id=_TID,
        )
        assert len(rows) == 1
        assert (rows[0].get('data') or {}).get('source') == 'stranded-reaper'

    async def test_submit_files_no_stranded_blocked_l1_itself(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """The submit method itself files NO pending stranded_blocked L1 — the
        record escalation is a separate step (step-10)."""
        queue = EscalationQueue(tmp_path / 'esc')
        harness._escalation_queue = queue
        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            await harness._maybe_submit_stranded_verified_green(_TID, {})

        assert queue.get_by_task(_TID, status='pending') == []

    async def test_no_match_returns_false_and_enqueues_nothing(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """detect_verified_green → None ⇒ return False, enqueue nothing."""
        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=None),
        ):
            result = await harness._maybe_submit_stranded_verified_green(_TID, {})

        assert result is False
        assert harness._merge_queue.qsize() == 0

    async def test_kill_switch_off_returns_false_without_detecting(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """config.stranded_verified_green_merge_enabled=False short-circuits
        BEFORE detection: returns False, enqueues nothing, never calls detect."""
        harness.config.stranded_verified_green_merge_enabled = False
        detect = AsyncMock(return_value=_match(tmp_path))
        with patch('orchestrator.harness.detect_verified_green', detect):
            result = await harness._maybe_submit_stranded_verified_green(_TID, {})

        assert result is False
        assert harness._merge_queue.qsize() == 0
        detect.assert_not_awaited()


@pytest.mark.asyncio
class TestStrandedVerifiedGreenRecordEscalation:
    """The action-recording (step-9): a submit files a stranded_blocked
    escalation and IMMEDIATELY dismisses it (close_only) so the task stays
    blocked — NOT re-pended (PRD §2.1)."""

    async def test_records_action_via_dismissed_escalation_without_repend(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            result = await harness._maybe_submit_stranded_verified_green(_TID, {})
        # Drain any coros scheduled by the resolve callback (there should be
        # none for the WORKFLOW_NONE/close_only path — mirrors the flip tests).
        await asyncio.gather(*list(harness._background_tasks))

        assert result is True

        # No pending stranded_blocked L1 remains — it was immediately dismissed.
        assert queue.get_by_task(_TID, status='pending') == []

        # The record IS archived (dismissed) as an audit trail, noting the
        # merge request_id + source.
        dismissed = queue.get_by_task(_TID, status='dismissed')
        assert len(dismissed) == 1
        rec = dismissed[0]
        assert rec.category == 'stranded_blocked'
        assert rec.agent_role == 'harness-stranded-blocked-reaper'
        assert 'source=stranded-reaper' in rec.resolution
        assert 'request_id=mr-' in rec.resolution

        # CRITICALLY: the record must NOT re-pend the task (close_only →
        # WORKFLOW_NONE, not a resume-resolution).
        for call in harness.scheduler.set_task_status.await_args_list:
            assert tuple(call.args[:2]) != (_TID, 'pending'), (
                'verified-green record must NOT re-pend the task'
            )


@pytest.mark.asyncio
class TestReconcileStrandedDriverVerifiedGreen:
    """End-to-end through Harness._reconcile_stranded_in_progress() (step-11).

    The driver classifies a blocked task with no open escalation and no live
    claimant as RE_FILE_ESCALATION; the verified-green gate runs BEFORE the
    re-file.  A match submits merge-queue-direct; a non-match preserves today's
    re-file path exactly.
    """

    async def test_a_verified_green_submits_and_no_pending_l1_no_repend(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        tid = _TID
        queue = EscalationQueue(tmp_path / 'esc_a')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
        # git_ops mocks so recovery_for classifies the blocked task as
        # GONE_NO_MARKER → RE_FILE_ESCALATION (mirrors test_stranded_blocked_sweep).
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({tid: 'blocked'}, None),
        )

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            await harness._reconcile_stranded_in_progress()
        await asyncio.gather(*list(harness._background_tasks))

        # A MergeRequest was enqueued (merge-queue-direct submission).
        assert harness._merge_queue.qsize() == 1
        # No pending stranded_blocked L1 remains — the record was auto-dismissed.
        assert queue.get_by_task(tid, status='pending') == []
        # The task was NOT re-pended.
        for call in harness.scheduler.set_task_status.await_args_list:
            assert tuple(call.args[:2]) != (tid, 'pending')

    async def test_b_non_matching_preserves_today_refile_path(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        tid = '8'
        queue = EscalationQueue(tmp_path / 'esc_b')
        harness._escalation_queue = queue
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({tid: 'blocked'}, None),
        )

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=None),
        ):
            await harness._reconcile_stranded_in_progress()

        # Today's behavior preserved: exactly one pending stranded_blocked L1
        # and NO MergeRequest enqueued.
        filed = queue.get_by_task(tid, status='pending')
        assert len(filed) == 1
        assert filed[0].category == 'stranded_blocked'
        assert filed[0].agent_role == 'harness-stranded-blocked-reaper'
        assert harness._merge_queue.qsize() == 0


_DURABLE_FAILURE_STATUSES = [
    'conflict', 'blocked', 'error', 'unknown_branch',
    'unmerged_state', 'stash_failed', 'wip_recovery_no_advance',
]
_SUCCESS_TRANSIENT_STATUSES = [
    'done', 'already_merged', 'done_wip_recovery', 'superseded', 'wip_halted',
]


@pytest.mark.asyncio
class TestStrandedMergeFailedL2:
    """Durable merge/verify FAILURE → born-at-L2 stranded_merge_failed (step-13).

    The MergeRequest's done-callback files a born-at-L2 escalation on a durable
    failure and is a strict no-op on success/transient outcomes; the task stays
    blocked and the branch + lane are preserved (by omission — the callback
    never touches status/lane/branch).
    """

    async def _submit_and_get_req(self, harness: Harness, tmp_path: Path, queue):
        harness._escalation_queue = queue
        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            await harness._maybe_submit_stranded_verified_green(_TID, {})
        return harness._merge_queue.get_nowait()

    async def _drive_callback(self, harness: Harness) -> None:
        # add_done_callback fires via call_soon; sleep(0) lets it run and
        # schedule _file_stranded_merge_failed onto _background_tasks.
        await asyncio.sleep(0)
        await asyncio.gather(*list(harness._background_tasks))

    @pytest.mark.parametrize('status', _DURABLE_FAILURE_STATUSES)
    async def test_durable_failure_files_born_at_l2(
        self, harness: Harness, tmp_path: Path, status: str,
    ) -> None:
        from orchestrator.merge_types import MergeOutcome

        queue = EscalationQueue(tmp_path / 'esc')
        req = await self._submit_and_get_req(harness, tmp_path, queue)
        req.result.set_result(MergeOutcome(status=status, reason='durable-fail'))
        await self._drive_callback(harness)

        l2s = queue.get_by_task(
            _TID, status='pending', level=2,
            agent_role='harness-stranded-blocked-reaper',
        )
        assert len(l2s) == 1
        rec = l2s[0]
        assert rec.category == 'stranded_merge_failed'
        assert rec.severity == 'critical'
        assert rec.level == 2
        # Task NOT flipped (stays blocked; branch + lane preserved by omission).
        for call in harness.scheduler.set_task_status.await_args_list:
            assert tuple(call.args[:2]) != (_TID, 'pending')

    @pytest.mark.parametrize('status', _SUCCESS_TRANSIENT_STATUSES)
    async def test_success_transient_files_no_l2(
        self, harness: Harness, tmp_path: Path, status: str,
    ) -> None:
        from orchestrator.merge_types import MergeOutcome

        queue = EscalationQueue(tmp_path / 'esc')
        req = await self._submit_and_get_req(harness, tmp_path, queue)
        req.result.set_result(MergeOutcome(status=status))
        await self._drive_callback(harness)

        assert queue.get_by_task(
            _TID, status='pending', level=2,
            agent_role='harness-stranded-blocked-reaper',
        ) == []

    async def test_second_durable_failure_no_duplicate_l2(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_types import MergeOutcome

        queue = EscalationQueue(tmp_path / 'esc')
        req1 = await self._submit_and_get_req(harness, tmp_path, queue)
        req1.result.set_result(MergeOutcome(status='conflict'))
        await self._drive_callback(harness)

        req2 = await self._submit_and_get_req(harness, tmp_path, queue)
        req2.result.set_result(MergeOutcome(status='error'))
        await self._drive_callback(harness)

        l2s = queue.get_by_task(
            _TID, status='pending', level=2,
            agent_role='harness-stranded-blocked-reaper',
        )
        assert len(l2s) == 1  # scoped dedup — no duplicate born-at-L2
