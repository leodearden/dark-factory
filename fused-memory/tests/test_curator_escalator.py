"""Tests for the curator → orchestrator escalation router (R2).

``CuratorEscalator`` decides what happens when the curator's LLM call
fails. The policy is:

* If the project's orchestrator holds its exclusive lock → queue a
  level-1 escalation; the watcher will run ``/unblock``.
* If no orchestrator is running → raise :class:`CuratorFailureError` so
  the interactive MCP caller sees the outage loudly.

These tests exercise both branches plus the 1-hour per-project cooldown
that suppresses spam from a stuck curator.
"""

from __future__ import annotations

import fcntl
from typing import IO, Literal, overload

import pytest

from fused_memory.middleware.curator_escalator import CuratorEscalator
from fused_memory.middleware.task_curator import CuratorFailureError


@overload
def _make_orchestrator_layout(root, *, hold_lock: Literal[True]) -> IO[bytes]: ...
@overload
def _make_orchestrator_layout(root, *, hold_lock: Literal[False]) -> None: ...
def _make_orchestrator_layout(root, *, hold_lock: bool) -> IO[bytes] | None:
    """Create the orchestrator.lock file; optionally hold LOCK_EX on it.

    Returns the open file handle when ``hold_lock=True`` so the caller
    can keep the lock alive for the duration of the test. When
    ``hold_lock=False`` the file exists but nothing holds an exclusive
    lock — matching the "orchestrator not running" case.
    """
    lock_dir = root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / 'orchestrator.lock'
    lock_path.write_text('')
    if not hold_lock:
        return None
    handle = lock_path.open('r+b')
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


class TestOrchestratorLivenessProbe:
    def test_missing_lock_file_reports_not_running(self, tmp_path):
        escalator = CuratorEscalator()
        # No lock file created — treat as "no orchestrator".
        assert escalator._orchestrator_running(str(tmp_path)) is False

    def test_unlocked_file_reports_not_running(self, tmp_path):
        _make_orchestrator_layout(tmp_path, hold_lock=False)
        escalator = CuratorEscalator()
        assert escalator._orchestrator_running(str(tmp_path)) is False

    def test_held_exclusive_lock_reports_running(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator()
            assert escalator._orchestrator_running(str(tmp_path)) is True
        finally:
            handle.close()


class TestReportFailure:
    @pytest.mark.asyncio
    async def test_no_orchestrator_raises(self, tmp_path):
        escalator = CuratorEscalator()
        with pytest.raises(CuratorFailureError):
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='boom',
                candidate_title='some candidate',
            )

    @pytest.mark.asyncio
    async def test_orchestrator_running_queues_escalation(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator()
            await escalator.report_failure(
                project_root=str(tmp_path),
                project_id='proj-x',
                justification='max_turns exhausted',
                candidate_title='Add Type::Error arm',
            )

            queue_dir = tmp_path / 'data' / 'escalations'
            files = sorted(queue_dir.glob('esc-*.json'))
            assert len(files) == 1
            body = files[0].read_text()
            assert 'curator_failure' in body
            assert '"level": 1' in body
            assert 'max_turns exhausted' in body
            assert 'Add Type::Error arm' in body
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_duplicate_within_window(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            # 1h cooldown is the default; give a wide window explicitly.
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            for _ in range(3):
                await escalator.report_failure(
                    project_root=str(tmp_path),
                    project_id='proj-x',
                    justification='repeat',
                    candidate_title='T',
                )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            # Only the first call should produce a queue file.
            assert len(files) == 1
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_zero_cooldown_queues_each_call(self, tmp_path):
        handle = _make_orchestrator_layout(tmp_path, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=0.0)
            for _ in range(2):
                await escalator.report_failure(
                    project_root=str(tmp_path),
                    project_id='proj-x',
                    justification='repeat',
                    candidate_title='T',
                )
            files = sorted((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
            assert len(files) == 2
        finally:
            handle.close()

    @pytest.mark.asyncio
    async def test_separate_projects_dont_share_cooldown(self, tmp_path):
        """Cooldown is per-project. A noisy project A must not silence project B."""
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'
        handle_a = _make_orchestrator_layout(root_a, hold_lock=True)
        handle_b = _make_orchestrator_layout(root_b, hold_lock=True)
        try:
            escalator = CuratorEscalator(cooldown_secs=3600.0)
            await escalator.report_failure(
                project_root=str(root_a), project_id='proj-a',
                justification='boom', candidate_title='T',
            )
            await escalator.report_failure(
                project_root=str(root_b), project_id='proj-b',
                justification='boom', candidate_title='T',
            )
            assert len(list((root_a / 'data' / 'escalations').glob('esc-*.json'))) == 1
            assert len(list((root_b / 'data' / 'escalations').glob('esc-*.json'))) == 1
        finally:
            handle_a.close()
            handle_b.close()
