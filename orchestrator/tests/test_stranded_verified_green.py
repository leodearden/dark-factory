"""Tests for the verified-green stranded remediation detector.

Covers ``orchestrator.stranded_verified_green`` — the pure helpers and the
async ``detect_verified_green`` shape check that the stranded-blocked reaper
consults before submitting a lane branch directly to the merge queue
(stranding-remediation-scheduler-ergonomics-prd.md leaf α, §2.1).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.event_store import EventStore, EventType
from orchestrator.lane_lifecycle import LaneRecord, LaneState


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
        self.config = SimpleNamespace(git=SimpleNamespace(branch_prefix='task/'))

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
