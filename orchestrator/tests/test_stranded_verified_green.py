"""Tests for the verified-green stranded remediation detector.

Covers ``orchestrator.stranded_verified_green`` — the pure helpers and the
async ``detect_verified_green`` shape check that the stranded-blocked reaper
consults before submitting a lane branch directly to the merge queue
(stranding-remediation-scheduler-ergonomics-prd.md leaf α, §2.1).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import wire_scheduler_liveness_mock
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.artifacts import TaskArtifacts
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.lane_lifecycle import LaneRecord, LaneState
from orchestrator.stranded_verified_green import (
    VerifiedGreenMatch,
    merge_request_marker_is_fresh,
    submit_verified_green_merge_request,
)
from orchestrator.task_ground_truth import EscalationRef


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """MergeProvenance._outbox is a process-global — never leak a bound outbox
    into another test (mirrors test_reconcile_stranded.py's identically-named
    fixture).  A None outbox is the correct default for the non-landing tests
    (MergeProvenance.lookup misses → git-archaeology fallback, already mocked)."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


def _bind_landed_row(tmp_path: Path, *, task_id: str, advanced_sha: str) -> None:
    """Bind a real LandedOutbox holding a row for *task_id* so its branch
    resolves ON_MAIN via MergeProvenance.lookup (mirrors
    test_reconcile_stranded.py's identically-named helper)."""
    outbox = LandedOutbox(tmp_path / 'landed.json')
    outbox.record(LandedRow(
        task_id=task_id, branch_tip_sha='branchtip', advanced_sha=advanced_sha,
        landed_at=1.0,
    ))
    MergeProvenance.bind(outbox)


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
) -> tuple[GitOps, EventStore, Callable[[str], Path], Path]:
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

    return cast(GitOps, git_ops), event_store, _resolver, worktree


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
        ret = cast(AsyncMock, h.scheduler.get_statuses).return_value
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
    cast(Any, h.config).module_configs_or_empty = {}
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

        assert harness.event_store is not None
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
        assert rec.resolution is not None
        assert 'source=stranded-reaper' in rec.resolution
        assert 'request_id=mr-' in rec.resolution

        # CRITICALLY: the record must NOT re-pend the task (close_only →
        # WORKFLOW_NONE, not a resume-resolution).
        for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
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
        for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
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


_OFF_MAIN_SHA = 'off' + 'a' * 37  # 40-char unmerged branch tip


def _wire_exists_off_main(harness: Harness, tid: str) -> None:
    """Wire *harness* so *tid*'s blocked branch resolves EXISTS_OFF_MAIN.

    Branch ref present but NOT an ancestor of main, no merge marker, and no
    bound MergeProvenance journal row (the autouse ``_reset_merge_provenance``
    fixture guarantees the miss) — the shape a verified-green task that never
    got merged is actually in.
    """
    harness.git_ops.is_ancestor = AsyncMock(return_value=False)
    harness.git_ops.resolve_branch_sha = AsyncMock(return_value=_OFF_MAIN_SHA)
    harness.git_ops.find_merge_marker = AsyncMock(return_value=None)
    harness.scheduler.get_statuses = AsyncMock(  # type: ignore[attr-defined]
        return_value=({tid: 'blocked'}, None),
    )


def _seed_pending(queue: EscalationQueue, tid: str, category: str, *, level: int = 1) -> str:
    """Seed ONE pending escalation of *category* for *tid* and return its id.

    The id is deliberately DISTINCT from ``queue.make_id(tid)`` so it cannot
    collide with (or be mistaken for) a record the reaper files itself.
    """
    esc_id = f'esc-seed-{category}-{tid}'
    queue.submit(Escalation(
        id=esc_id, task_id=tid, agent_role='steward', severity='blocking',
        category=category, summary=f'pre-existing {category}', level=level,
    ))
    return esc_id


def _asserted_no_repend(harness: Harness, tid: str) -> None:
    """The task was never flipped back to 'pending'."""
    for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
        assert tuple(call.args[:2]) != (tid, 'pending')


@pytest.mark.asyncio
class TestVerifiedGreenVetoRelaxExistsOffMain:
    """δ: a merge-remediable open escalation no longer vetoes the auto-merge.

    Boundary pair on the EXISTS_OFF_MAIN sweep-side RE_FILE_ESCALATION upgrade
    — the branch shape a verified-green-but-never-merged task is in.  D-1: the
    reaper's OWN ``stranded_blocked`` request no longer blocks the merge it
    asked for.  D-2: a human-concern escalation still vetoes, unchanged.
    """

    async def test_d1_merge_remediable_esc_no_longer_vetoes_submit(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        tid = _TID
        queue = EscalationQueue(tmp_path / 'esc_d1')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
        _wire_exists_off_main(harness, tid)
        _seed_pending(queue, tid, 'stranded_blocked')

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            await harness._reconcile_stranded_in_progress()
        await asyncio.gather(*list(harness._background_tasks))

        # The verified-green branch was submitted merge-queue-direct despite
        # the open stranded_blocked — the anti-synergy δ closes.
        assert harness._merge_queue.qsize() == 1
        assert harness.event_store is not None
        rows = harness.event_store.fetch_events_by_type_all_runs(
            EventType.merge_queued, task_id=tid,
        )
        assert len(rows) == 1
        assert (rows[0].get('data') or {}).get('source') == 'stranded-reaper'
        # Still left blocked for the merge to land — never re-pended.
        _asserted_no_repend(harness, tid)

    async def test_d2_human_concern_esc_still_vetoes(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """A design_concern is NOT merge-remediable → LEAVE, unchanged."""
        tid = _TID
        queue = EscalationQueue(tmp_path / 'esc_d2')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
        _wire_exists_off_main(harness, tid)
        seeded = _seed_pending(queue, tid, 'design_concern')

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            await harness._reconcile_stranded_in_progress()
        await asyncio.gather(*list(harness._background_tasks))

        # No auto-submit: the human-concern escalation holds the task.
        assert harness._merge_queue.qsize() == 0
        _asserted_no_repend(harness, tid)
        pending = queue.get_by_task(tid, status='pending')
        assert [e.id for e in pending] == [seeded]
        assert pending[0].category == 'design_concern'

    async def test_non_match_with_open_remediable_esc_files_no_duplicate(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """Relaxing the veto must not turn the re-file path into a duplicator.

        The relax means RE_FILE_ESCALATION is now reachable with a
        stranded_blocked ALREADY pending.  On a verified-green NON-match the
        submit declines, and the fallback re-file would stack a SECOND
        stranded_blocked L1 for the same task — so the call-site must leave the
        already-open escalation for its handler instead.
        """
        tid = _TID
        queue = EscalationQueue(tmp_path / 'esc_dedup')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
        _wire_exists_off_main(harness, tid)
        seeded = _seed_pending(queue, tid, 'stranded_blocked')

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=None),
        ):
            await harness._reconcile_stranded_in_progress()
        await asyncio.gather(*list(harness._background_tasks))

        assert harness._merge_queue.qsize() == 0
        _asserted_no_repend(harness, tid)
        pending = queue.get_by_task(tid, status='pending')
        assert [e.id for e in pending] == [seeded], (
            'must not stack a second stranded_blocked L1'
        )


_ADVANCED_SHA = 'ad' * 20  # 40-hex on-main landing SHA


def _wire_on_main_mark_done(harness: Harness, tid: str, tmp_path: Path) -> None:
    """Wire *harness* so *tid*'s blocked branch resolves ON_MAIN and the
    found_on_main mark-done collaborators are satisfied.

    Mirrors ``test_submit_then_land_marks_done_via_found_on_main``'s sweep-2
    wiring: a bound MergeProvenance journal row is the authoritative ON_MAIN
    signal, and mark_done forwards to set_task_status so the provenance is
    assertable.
    """
    _bind_landed_row(tmp_path, task_id=tid, advanced_sha=_ADVANCED_SHA)

    async def _fake_mark_done(t, *, kind, sha, note=None):
        prov = {'kind': kind, 'commit': sha}
        if note is not None:
            prov['note'] = note
        await harness.scheduler.set_task_status(t, 'done', done_provenance=prov)

    harness.scheduler.mark_done = AsyncMock(side_effect=_fake_mark_done)  # type: ignore[attr-defined]
    harness.git_ops.is_ancestor = AsyncMock(return_value=False)
    harness.git_ops.find_merge_marker = AsyncMock(return_value=None)
    harness.git_ops.cleanup_worktree = AsyncMock()
    harness.git_ops.release_lane_for_terminal_task = AsyncMock()
    harness.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)
    harness.git_ops.warm_lane_ref_is_degenerate = AsyncMock(return_value=False)
    harness.scheduler.get_statuses = AsyncMock(  # type: ignore[attr-defined]
        return_value=({tid: 'blocked'}, None),
    )


def _done_provenance(harness: Harness, tid: str) -> dict | None:
    """The done_provenance the task was marked done with, or None if it wasn't."""
    for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
        if tuple(call.args[:2]) == (tid, 'done'):
            return call.kwargs.get('done_provenance')
    return None


@pytest.mark.asyncio
class TestVerifiedGreenVetoRelaxOnMain:
    """δ: a merge-remediable open escalation no longer vetoes the self-heal.

    Boundary pair on the ON_MAIN sweep-side MARK_DONE_WITH_PROVENANCE upgrade —
    the shape of a task whose branch LANDED while it was still blocked.  Case A:
    the reaper's own ``stranded_blocked`` no longer holds the task blocked
    forever after its branch landed.  Case B: a human-concern escalation still
    vetoes, unchanged.
    """

    async def test_case_a_merge_remediable_esc_no_longer_vetoes_mark_done(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        tid = _TID
        queue = EscalationQueue(tmp_path / 'esc_ma')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
        _wire_on_main_mark_done(harness, tid, tmp_path)
        _seed_pending(queue, tid, 'stranded_blocked')

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=None),
        ):
            await harness._reconcile_stranded_in_progress()
        await asyncio.gather(*list(harness._background_tasks))

        prov = _done_provenance(harness, tid)
        assert prov is not None, 'landed branch must self-heal to done'
        assert prov['kind'] == 'found_on_main'

    async def test_case_b_human_concern_esc_still_vetoes_mark_done(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """A design_concern is NOT merge-remediable → the task stays blocked."""
        tid = _TID
        queue = EscalationQueue(tmp_path / 'esc_mb')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
        _wire_on_main_mark_done(harness, tid, tmp_path)
        seeded = _seed_pending(queue, tid, 'design_concern')

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=None),
        ):
            await harness._reconcile_stranded_in_progress()
        await asyncio.gather(*list(harness._background_tasks))

        assert _done_provenance(harness, tid) is None, (
            'a human-concern escalation must still hold the task'
        )
        pending = queue.get_by_task(tid, status='pending')
        assert [e.id for e in pending] == [seeded]


def _marker(*, tip: str = _TIP, request_id: str = 'mr-prev1234', age_s: float = 0.0) -> dict:
    """A ``stranded_merge_request`` metadata marker (step-16 race-guard).

    ``age_s`` sets ``submitted_at`` that many seconds in the past — 0 is fresh,
    a large value (beyond the recency grace) is stale.
    """
    return {
        'tip_sha': tip,
        'request_id': request_id,
        'submitted_at': (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat(),
    }


@pytest.mark.asyncio
class TestStrandedVerifiedGreenIdempotency:
    """Metadata-marker dedup — no double-submit across sweeps (step-15).

    A durable ``metadata.stranded_merge_request={tip_sha,request_id,submitted_at}``
    marker guards against the periodic stranded sweep re-enqueuing the same
    branch every tick while the merge is in-flight.  A fresh marker whose
    ``tip_sha`` matches the current lane tip short-circuits the re-submit
    (task stays blocked); a STALE marker or an ADVANCED lane tip re-drives a
    fresh submit (self-healing, restart-safe — PRD leaf α §7).
    """

    async def test_submit_stamps_metadata_marker(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """A fresh submit stamps the durable marker via scheduler.update_task
        (merge mode — only the ``stranded_merge_request`` key is written)."""
        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            result = await harness._maybe_submit_stranded_verified_green(_TID, {})

        assert result is True
        assert harness._merge_queue.qsize() == 1
        harness.scheduler.update_task.assert_awaited()  # type: ignore[attr-defined]
        call = harness.scheduler.update_task.await_args  # type: ignore[attr-defined]
        assert call.args[0] == _TID
        # The single supplied key preserves siblings under merge mode.
        payload = call.args[1]
        assert list(payload.keys()) == ['stranded_merge_request']
        marker = payload['stranded_merge_request']
        assert marker['tip_sha'] == _TIP
        assert marker['request_id'].startswith('mr-')
        assert isinstance(marker['submitted_at'], str) and marker['submitted_at']

    async def test_fresh_marker_same_tip_skips_resubmit(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """Marker present, fresh, tip == current lane tip → return True WITHOUT
        re-enqueuing, re-stamping, or filing any escalation (already in flight;
        task stays blocked)."""
        queue = EscalationQueue(tmp_path / 'esc')
        harness._escalation_queue = queue
        metadata = {'stranded_merge_request': _marker(tip=_TIP)}

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path, tip=_TIP)),
        ):
            result = await harness._maybe_submit_stranded_verified_green(
                _TID, metadata,
            )

        assert result is True
        assert harness._merge_queue.qsize() == 0  # NO additional MergeRequest
        harness.scheduler.update_task.assert_not_awaited()  # type: ignore[attr-defined]  # no re-stamp
        # No new escalation filed (neither a pending L1 nor a dismissed record).
        assert queue.get_by_task(_TID, status='pending') == []
        assert queue.get_by_task(_TID, status='dismissed') == []

    async def test_stale_marker_resubmits(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """Marker tip matches but submitted_at is older than the recency grace
        → a fresh submit IS made (self-healing after a lost/hung in-flight)."""
        metadata = {'stranded_merge_request': _marker(tip=_TIP, age_s=7200.0)}

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path, tip=_TIP)),
        ):
            result = await harness._maybe_submit_stranded_verified_green(
                _TID, metadata,
            )

        assert result is True
        assert harness._merge_queue.qsize() == 1  # re-submit
        harness.scheduler.update_task.assert_awaited()  # type: ignore[attr-defined]  # marker re-stamped

    async def test_advanced_lane_tip_resubmits(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """Marker is fresh but its tip_sha != the current lane tip (lane
        advanced past the marker) → a fresh submit IS made for the new tip."""
        new_tip = 'f' * 40
        metadata = {'stranded_merge_request': _marker(tip=_TIP)}  # stale-vs-tip

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path, tip=new_tip)),
        ):
            result = await harness._maybe_submit_stranded_verified_green(
                _TID, metadata,
            )

        assert result is True
        assert harness._merge_queue.qsize() == 1  # re-submit for the new tip
        req = harness._merge_queue.get_nowait()
        assert req.snapshot_tip == new_tip

    async def test_malformed_marker_resubmits(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """A non-dict / unparseable marker fails safe → treat as absent and
        submit (a benign re-submit, never a wedged skip)."""
        metadata = {'stranded_merge_request': 'not-a-dict'}

        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path, tip=_TIP)),
        ):
            result = await harness._maybe_submit_stranded_verified_green(
                _TID, metadata,
            )

        assert result is True
        assert harness._merge_queue.qsize() == 1

    async def test_first_sweep_stamps_marker_via_driver(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        """End-to-end: the first _reconcile_stranded_in_progress() sweep submits
        AND stamps the marker via scheduler.update_task (honours the two-sweep
        idempotency framing — the stamp is what a second sweep reads)."""
        tid = _TID
        queue = EscalationQueue(tmp_path / 'esc_drv')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
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

        assert harness._merge_queue.qsize() == 1
        harness.scheduler.update_task.assert_awaited()  # type: ignore[attr-defined]
        marker = harness.scheduler.update_task.await_args.args[1][  # type: ignore[attr-defined]
            'stranded_merge_request'
        ]
        assert marker['tip_sha'] == _TIP
        assert marker['request_id'].startswith('mr-')


# Parametrization derives from the PRODUCTION classification sets so a new
# MergeOutcome.status can never drift out of test coverage — and
# test_status_sets_exhaust_merge_outcome_vocabulary (below) fails CI if the two
# sets ever stop partitioning the full MergeOutcome.status vocabulary.
_DURABLE_FAILURE_STATUSES = sorted(Harness._DURABLE_MERGE_FAILURE_STATUSES)
_SUCCESS_TRANSIENT_STATUSES = sorted(Harness._SUCCESS_TRANSIENT_MERGE_STATUSES)


def test_status_sets_exhaust_merge_outcome_vocabulary() -> None:
    """The two classification frozensets MUST partition MergeOutcome.status.

    Guards the loud-over-silent norm (PRD leaf α §2.2): if a new value is added
    to ``MergeOutcome.status`` without being sorted into exactly one of
    ``_DURABLE_MERGE_FAILURE_STATUSES`` / ``_SUCCESS_TRANSIENT_MERGE_STATUSES``,
    the runtime ``_on_stranded_merge_done`` backstop has to guess — so fail CI
    here and force the author to classify it deliberately.  Also asserts the two
    sets are disjoint (no status is both a durable failure and a success) and
    that neither carries a phantom status ``MergeOutcome`` no longer declares.
    """
    from typing import get_args, get_type_hints

    from orchestrator.merge_types import MergeOutcome

    vocabulary = set(get_args(get_type_hints(MergeOutcome)['status']))
    assert vocabulary, 'MergeOutcome.status is not a Literal — cannot derive vocabulary'

    durable = Harness._DURABLE_MERGE_FAILURE_STATUSES
    transient = Harness._SUCCESS_TRANSIENT_MERGE_STATUSES

    assert not (durable & transient), (
        'statuses classified as BOTH durable and success/transient: '
        f'{sorted(durable & transient)}'
    )
    unclassified = vocabulary - durable - transient
    assert not unclassified, (
        f'MergeOutcome.status value(s) {sorted(unclassified)} are unclassified '
        '— add each to Harness._DURABLE_MERGE_FAILURE_STATUSES or '
        'Harness._SUCCESS_TRANSIENT_MERGE_STATUSES'
    )
    phantom = (durable | transient) - vocabulary
    assert not phantom, (
        f'classified status(es) {sorted(phantom)} are not in MergeOutcome.status'
    )


def _ref(category: str, *, level: int = 1) -> EscalationRef:
    """An open-escalation ref carrying *category* (δ predicate input)."""
    return EscalationRef(id=f'esc-{category}', level=level, category=category)


class TestOnlyMergeRemediable:
    """Harness.MERGE_REMEDIABLE_ESC_CATEGORIES + _only_merge_remediable (δ).

    The single predicate authority for the relaxed verified-green auto-merge
    veto: an escalation that exists to REQUEST the merge must not itself veto
    the merge, while every human-concern class still vetoes unchanged.
    """

    def test_set_contains_the_merge_request_class(self) -> None:
        assert 'stranded_blocked' in Harness.MERGE_REMEDIABLE_ESC_CATEGORIES

    @pytest.mark.parametrize('category', [
        'design_concern',
        'task_failure',
        'review_issues',
        'stranded_merge_failed',
        'infra_issue',
    ])
    def test_set_excludes_human_concern_classes(self, category: str) -> None:
        """A human-concern esc is NOT remediable by re-merging.

        ``stranded_merge_failed`` in particular is the DURABLE
        merge/verify-failure born-at-L2 signal — re-merging cannot remediate
        it, so it MUST keep vetoing.
        """
        assert category not in Harness.MERGE_REMEDIABLE_ESC_CATEGORIES

    def test_empty_is_vacuously_true(self) -> None:
        """No open escalation → True, byte-identical to today's
        ``not report.open_escalations``."""
        assert Harness._only_merge_remediable([]) is True

    def test_all_remediable_is_true(self) -> None:
        assert Harness._only_merge_remediable([_ref('stranded_blocked')]) is True

    def test_mixed_is_false(self) -> None:
        """One non-remediable esc among remediable ones still vetoes."""
        refs = [_ref('stranded_blocked'), _ref('design_concern')]
        assert Harness._only_merge_remediable(refs) is False

    @pytest.mark.parametrize('category', ['design_concern', 'stranded_merge_failed'])
    def test_single_non_remediable_is_false(self, category: str) -> None:
        assert Harness._only_merge_remediable([_ref(category)]) is False


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
        req.result.set_result(MergeOutcome(status=cast(Any, status), reason='durable-fail'))
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
        for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            assert tuple(call.args[:2]) != (_TID, 'pending')

    @pytest.mark.parametrize('status', _SUCCESS_TRANSIENT_STATUSES)
    async def test_success_transient_files_no_l2(
        self, harness: Harness, tmp_path: Path, status: str,
    ) -> None:
        from orchestrator.merge_types import MergeOutcome

        queue = EscalationQueue(tmp_path / 'esc')
        req = await self._submit_and_get_req(harness, tmp_path, queue)
        req.result.set_result(MergeOutcome(status=cast(Any, status)))
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

    async def test_unclassified_status_files_l2_and_logs_loud(
        self, harness: Harness, tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A status in NEITHER classification set is treated LOUDLY as a failure.

        Runtime backstop for the exhaustiveness test: if a new MergeOutcome
        status ever reaches the callback unclassified, it must still file the
        born-at-L2 (never a silent strand) AND log at ERROR — honoring the
        loud-over-silent norm rather than defaulting to a silent no-op.
        """
        import logging

        from orchestrator.merge_types import MergeOutcome

        queue = EscalationQueue(tmp_path / 'esc')
        req = await self._submit_and_get_req(harness, tmp_path, queue)
        # A status that is in neither _DURABLE_MERGE_FAILURE_STATUSES nor
        # _SUCCESS_TRANSIENT_MERGE_STATUSES (MergeOutcome is a plain dataclass —
        # no runtime Literal validation, so an arbitrary string is accepted).
        unknown = 'brand_new_unclassified_status'
        assert unknown not in Harness._DURABLE_MERGE_FAILURE_STATUSES
        assert unknown not in Harness._SUCCESS_TRANSIENT_MERGE_STATUSES
        with caplog.at_level(logging.ERROR, logger='orchestrator.harness'):
            req.result.set_result(MergeOutcome(status=cast(Any, unknown)))
            await self._drive_callback(harness)

        l2s = queue.get_by_task(
            _TID, status='pending', level=2,
            agent_role='harness-stranded-blocked-reaper',
        )
        assert len(l2s) == 1  # unknown outcome → born-at-L2, not a silent strand
        assert l2s[0].category == 'stranded_merge_failed'
        assert any(
            'UNCLASSIFIED' in r.getMessage() and unknown in r.getMessage()
            for r in caplog.records
        ), 'expected a loud ERROR log naming the unclassified status'


@pytest.mark.asyncio
class TestStrandedVerifiedGreenHappyPathIntegration:
    """PRD §6α happy-path integration + non-interference (step-17).

    Sweep 1 submits the verified-green branch merge-queue-direct and records
    the action via an auto-dismissed escalation (NO pending L1, marker
    stamped).  The merge then LANDS (a MergeProvenance journal row makes the
    branch resolve ON_MAIN).  Sweep 2 marks the blocked task done via the
    EXISTING found_on_main MARK_DONE_WITH_PROVENANCE path — reachable because
    (a) the record escalation is DISMISSED, not pending, so the upgrade's
    open-escalation guard is vacuously satisfied, and (b) the
    ``stranded_merge_request`` marker lives in metadata, ignored by the
    mark-done path.  No ``stranded_merge_failed`` L2 is ever filed.

    Since PRD leaf δ that guard is ``_only_merge_remediable``, so a pending
    ``stranded_blocked`` record would not block the flip either — this test
    still pins the DISMISSED shape the submit path actually produces, and
    TestVerifiedGreenVetoRelaxOnMain covers the still-pending case.
    """

    async def test_submit_then_land_marks_done_via_found_on_main(
        self, harness: Harness, tmp_path: Path,
    ) -> None:
        tid = _TID
        advanced_sha = 'ad' * 20  # 40-hex on-main landing SHA
        queue = EscalationQueue(tmp_path / 'esc_hp')
        harness._escalation_queue = queue
        harness._loop = asyncio.get_running_loop()
        queue.set_resolve_callback(harness._on_escalation_resolved)
        harness._escalation_events.clear()
        harness._workflow_cancel_at.clear()
        harness.git_ops.is_ancestor = AsyncMock(return_value=False)
        harness.git_ops.find_merge_marker = AsyncMock(return_value=None)

        # --- Sweep 1: verified-green blocked task → merge-queue-direct submit.
        harness.scheduler.get_statuses = AsyncMock(
            return_value=({tid: 'blocked'}, None),
        )
        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=_match(tmp_path)),
        ):
            await harness._reconcile_stranded_in_progress()
        await asyncio.gather(*list(harness._background_tasks))

        assert harness._merge_queue.qsize() == 1
        # Record auto-dismissed → NO pending escalation left to block the flip.
        assert queue.get_by_task(tid, status='pending') == []
        harness.scheduler.update_task.assert_awaited()  # type: ignore[attr-defined]
        marker = harness.scheduler.update_task.await_args.args[1][  # type: ignore[attr-defined]
            'stranded_merge_request'
        ]

        # --- The merge LANDS: bind a journal row so the branch resolves ON_MAIN.
        _bind_landed_row(tmp_path, task_id=tid, advanced_sha=advanced_sha)

        # Wire the existing found_on_main mark-done collaborators (mirror
        # test_reconcile_stranded.py: mark_done forwards to set_task_status).
        async def _fake_mark_done(t, *, kind, sha, note=None):
            prov = {'kind': kind, 'commit': sha}
            if note is not None:
                prov['note'] = note
            await harness.scheduler.set_task_status(t, 'done', done_provenance=prov)
        harness.scheduler.mark_done = AsyncMock(side_effect=_fake_mark_done)
        harness.git_ops.cleanup_worktree = AsyncMock()
        harness.git_ops.release_lane_for_terminal_task = AsyncMock()
        harness.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)
        harness.git_ops.warm_lane_ref_is_degenerate = AsyncMock(return_value=False)
        # Sweep 2's task row carries the marker → prove it does NOT interfere
        # with the found_on_main mark-done path.
        harness.scheduler.get_task = AsyncMock(
            return_value={
                'status': 'blocked',
                'metadata': {'stranded_merge_request': marker},
            },
        )

        # --- Sweep 2: the EXISTING found_on_main path flips blocked → done.
        with patch(
            'orchestrator.harness.detect_verified_green',
            AsyncMock(return_value=None),
        ):
            await harness._reconcile_stranded_in_progress()
        await asyncio.gather(*list(harness._background_tasks))

        harness.scheduler.set_task_status.assert_any_await(  # type: ignore[attr-defined]
            tid, 'done',
            done_provenance={
                'kind': 'found_on_main', 'commit': advanced_sha, 'note': ANY,
            },
        )
        # The task was NOT re-pended anywhere along the way.
        for call in harness.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            assert tuple(call.args[:2]) != (tid, 'pending')
        # No stranded_merge_failed L2 was ever filed (success path is a no-op).
        l2s = [
            e for e in queue.get_by_task(
                tid, status='pending', level=2,
                agent_role='harness-stranded-blocked-reaper',
            )
            if e.category == 'stranded_merge_failed'
        ]
        assert l2s == []


# ---------------------------------------------------------------------------
# Shared merge-submit primitives (task 3031 β, INV-5 extraction).
#
# The stranded reaper and the architect's `report_ready_to_merge` desync exit
# both need the same build→register-callback→enqueue→stamp-marker plumbing.
# It lives here as two free functions so neither caller copies it and
# workflow.py needn't import harness.py (circular).  The two callers differ
# ONLY in `source=` tag, marker key, and terminal done-callback.
# ---------------------------------------------------------------------------


class TestMergeRequestMarkerIsFresh:
    """The shared freshness predicate behind BOTH submit-dedup guards."""

    def test_fresh_matching_marker_is_fresh(self) -> None:
        assert merge_request_marker_is_fresh(_marker(tip=_TIP), _TIP) is True

    def test_non_dict_marker_is_not_fresh(self) -> None:
        """Fail-safe: a malformed marker must never wedge a submit."""
        assert merge_request_marker_is_fresh('not-a-dict', _TIP) is False
        assert merge_request_marker_is_fresh(None, _TIP) is False
        assert merge_request_marker_is_fresh(['a'], _TIP) is False

    def test_mismatched_tip_is_not_fresh(self) -> None:
        assert merge_request_marker_is_fresh(_marker(tip='f' * 40), _TIP) is False

    def test_missing_or_unparseable_timestamp_is_not_fresh(self) -> None:
        assert merge_request_marker_is_fresh({'tip_sha': _TIP}, _TIP) is False
        assert merge_request_marker_is_fresh(
            {'tip_sha': _TIP, 'submitted_at': ''}, _TIP,
        ) is False
        assert merge_request_marker_is_fresh(
            {'tip_sha': _TIP, 'submitted_at': 'not-a-timestamp'}, _TIP,
        ) is False
        assert merge_request_marker_is_fresh(
            {'tip_sha': _TIP, 'submitted_at': 12345}, _TIP,
        ) is False

    def test_stale_timestamp_is_not_fresh(self) -> None:
        assert merge_request_marker_is_fresh(
            _marker(tip=_TIP, age_s=7200.0), _TIP,
        ) is False

    def test_grace_is_configurable(self) -> None:
        """The recency window is a parameter, not a hard-coded constant — the
        reaper passes its own grace; the default is the shared window."""
        marker = _marker(tip=_TIP, age_s=120.0)
        assert merge_request_marker_is_fresh(marker, _TIP, grace_s=60.0) is False
        assert merge_request_marker_is_fresh(marker, _TIP, grace_s=600.0) is True

    def test_naive_timestamp_is_read_as_utc(self) -> None:
        """A tz-naive submitted_at is interpreted as UTC (not local), so a
        just-stamped naive marker still reads fresh."""
        naive = datetime.now(UTC).replace(tzinfo=None).isoformat()
        assert merge_request_marker_is_fresh(
            {'tip_sha': _TIP, 'submitted_at': naive}, _TIP,
        ) is True

    def test_future_timestamp_is_fresh(self) -> None:
        """Clock skew lands on the safe side: a future stamp reads fresh
        (skip) rather than triggering a duplicate submit."""
        assert merge_request_marker_is_fresh(
            _marker(tip=_TIP, age_s=-300.0), _TIP,
        ) is True


@pytest.mark.asyncio
class TestSubmitVerifiedGreenMergeRequest:
    """The shared build→callback→enqueue→stamp submit helper."""

    async def _submit(
        self,
        tmp_path: Path,
        *,
        source: str = 'architect-desync',
        marker_key: str = 'architect_merge_request',
        done_callback: Callable[[asyncio.Future], None] | None = None,
        update_task: Any = None,
        event_store: EventStore | None = None,
        merge_queue: asyncio.Queue | None = None,
        config: Any = None,
    ):
        from orchestrator.merge_types import QueuedBranch

        queue = merge_queue if merge_queue is not None else asyncio.Queue()
        cfg = config
        if cfg is None:
            cfg = MagicMock()
            cfg.module_configs_or_empty = {}
        req = await submit_verified_green_merge_request(
            task_id=_TID,
            branch=QueuedBranch.parse(_TID, 'task/'),
            worktree=tmp_path / 'wt',
            tip_sha=_TIP,
            config=cfg,
            module_configs=[],
            merge_queue=queue,
            event_store=event_store,
            source=source,
            marker_key=marker_key,
            done_callback=done_callback or (lambda fut: None),
            update_task=update_task or AsyncMock(return_value=True),
        )
        return req, queue

    async def test_builds_and_enqueues_one_merge_request(
        self, tmp_path: Path,
    ) -> None:
        req, queue = await self._submit(tmp_path)

        assert queue.qsize() == 1
        queued = queue.get_nowait()
        assert queued is req
        assert queued.task_id == _TID
        assert queued.branch.bare_id == _TID
        assert queued.snapshot_tip == _TIP
        assert queued.pre_rebased is False
        assert queued.task_files is None
        assert queued.worktree == tmp_path / 'wt'

    async def test_registers_done_callback_on_the_future(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_types import MergeOutcome

        seen: list[Any] = []
        req, _queue = await self._submit(
            tmp_path, done_callback=lambda fut: seen.append(fut),
        )
        req.result.set_result(MergeOutcome(status='done', merge_sha='abc'))
        await asyncio.sleep(0)

        assert len(seen) == 1
        assert seen[0] is req.result

    async def test_threads_source_to_merge_queued_event(
        self, tmp_path: Path,
    ) -> None:
        """The submission's merge_queued event carries data.source=<source>,
        so the two callers stay distinguishable in the event stream."""
        store = EventStore(tmp_path / 'submit-runs.db', 'run-1')
        await self._submit(tmp_path, source='architect-desync', event_store=store)

        rows = store.fetch_events_by_type_all_runs(
            EventType.merge_queued, task_id=_TID,
        )
        assert len(rows) == 1
        assert (rows[0].get('data') or {}).get('source') == 'architect-desync'

    async def test_stamps_marker_under_the_given_key(
        self, tmp_path: Path,
    ) -> None:
        update_task = AsyncMock(return_value=True)
        req, _queue = await self._submit(
            tmp_path, marker_key='architect_merge_request', update_task=update_task,
        )

        update_task.assert_awaited_once()
        call = update_task.await_args
        assert call.args[0] == _TID
        payload = call.args[1]
        # A single key keeps merge-mode writes from clobbering siblings — and
        # keeps the two submit paths' markers from clobbering each other.
        assert list(payload.keys()) == ['architect_merge_request']
        marker = payload['architect_merge_request']
        assert marker['tip_sha'] == _TIP
        assert marker['request_id'] == req.request_id
        datetime.fromisoformat(marker['submitted_at'])

    async def test_stamped_marker_reads_back_as_fresh(
        self, tmp_path: Path,
    ) -> None:
        """The stamp and the freshness predicate agree — a just-stamped marker
        suppresses an immediate re-submit by the same caller."""
        update_task = AsyncMock(return_value=True)
        await self._submit(tmp_path, update_task=update_task)
        marker = update_task.await_args.args[1]['architect_merge_request']

        assert merge_request_marker_is_fresh(marker, _TIP) is True

    async def test_failing_update_task_is_swallowed(
        self, tmp_path: Path,
    ) -> None:
        """Best-effort marker stamp: a lost marker only risks a benign
        re-submit, so a write failure must never abort the remediation (the
        request is already enqueued)."""
        update_task = AsyncMock(side_effect=RuntimeError('scheduler down'))
        req, queue = await self._submit(tmp_path, update_task=update_task)

        assert queue.qsize() == 1
        assert req.request_id.startswith('mr-')
