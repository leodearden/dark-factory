"""Tests for the server-side ReconWritePolicy gate (W5-ζ, task 2224).

Rejects recon-stage task writes at the interceptor boundary so the post-hoc
reconciliation guards (μ's job to delete) become redundant. Three independent
early-return gates in :func:`recon_write_policy.check`:

1. ``op == 'update_task'`` AND ``live_status`` is terminal ->
   ``ReconTerminalWriteRejected``.
2. ``op == 'set_task_status'`` AND a live workflow is detected for the task ->
   ``ReconLiveWorkflowWriteRejected``.
3. ``snapshot_token is not None`` AND it disagrees with ``live_status`` (any
   op) -> ``ReconStaleSnapshotRejected``.

This module unit-tests the pure ``Verdict``/``check()``/``extract_snapshot_token``
building blocks in isolation. Interceptor boundary tests (P1/P2/P3) reuse the
fixtures from ``test_task_write_agent_id.py`` and live further down this file.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.middleware import recon_write_policy
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

# Recon-stage agent_id used by the interceptor boundary tests (P1/P2/P3) —
# matches test_task_write_agent_id.py's AGENT_ID so the recon-stage
# task-write setup is identical to the epsilon task's.
AGENT_ID = 'recon-stage-task_knowledge_sync'

# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_ok_verdict_is_not_rejection(self):
        verdict = recon_write_policy.Verdict(outcome='ok')
        assert verdict.is_rejection is False

    def test_ok_verdict_to_error_dict_is_empty(self):
        verdict = recon_write_policy.Verdict(outcome='ok')
        assert verdict.to_error_dict() == {}

    def test_rejection_verdict_is_rejection(self):
        verdict = recon_write_policy.Verdict(
            outcome='rejection',
            op='update_task',
            task_id='1',
            agent_id='recon-stage-x',
            error_type='ReconTerminalWriteRejected',
            reason='task is done',
            live_status='done',
        )
        assert verdict.is_rejection is True

    def test_rejection_verdict_to_error_dict_shape(self):
        verdict = recon_write_policy.Verdict(
            outcome='rejection',
            op='update_task',
            task_id='1',
            agent_id='recon-stage-x',
            error_type='ReconTerminalWriteRejected',
            reason='task is done',
            live_status='done',
        )
        error_dict = verdict.to_error_dict()
        assert isinstance(error_dict['error'], str)
        assert error_dict['error_type'] == 'ReconTerminalWriteRejected'
        assert error_dict['agent_id'] == 'recon-stage-x'
        assert error_dict['task_id'] == '1'
        assert error_dict['op'] == 'update_task'
        assert error_dict['live_status'] == 'done'


# ---------------------------------------------------------------------------
# check() helper
# ---------------------------------------------------------------------------


def _check(op: str, **overrides) -> recon_write_policy.Verdict:
    """Call check() with sensible defaults; override any kwarg via overrides."""
    kwargs = {
        'task_id': '1',
        'project_root': '/p',
        'agent_id': 'recon-stage-x',
        'target_status': None,
        'live_status': 'pending',
        'snapshot_token': None,
    }
    kwargs.update(overrides)
    return recon_write_policy.check(op, **kwargs)


# ---------------------------------------------------------------------------
# check() gate 1 — terminal (update_task only)
# ---------------------------------------------------------------------------


class TestCheckGate1Terminal:
    def test_update_task_on_done_task_rejects(self):
        verdict = _check('update_task', live_status='done')
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconTerminalWriteRejected'

    def test_update_task_on_cancelled_task_rejects(self):
        verdict = _check('update_task', live_status='cancelled')
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconTerminalWriteRejected'

    def test_update_task_on_in_progress_task_is_ok(self):
        verdict = _check('update_task', live_status='in-progress')
        assert verdict.is_rejection is False

    def test_set_task_status_op_is_not_scoped_by_gate_1(self):
        """Gate 1 is update_task-only: a set_task_status call against a done
        task must never surface ReconTerminalWriteRejected."""
        verdict = _check('set_task_status', target_status='pending', live_status='done')
        assert verdict.error_type != 'ReconTerminalWriteRejected'


# ---------------------------------------------------------------------------
# check() gate 2 — live workflow (set_task_status only)
# ---------------------------------------------------------------------------


class TestCheckGate2LiveWorkflow:
    def test_set_task_status_with_live_workflow_rejects(self, monkeypatch):
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )
        verdict = _check('set_task_status', live_status='in-progress')
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconLiveWorkflowWriteRejected'

    def test_set_task_status_without_live_workflow_is_ok(self, monkeypatch):
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: False,
        )
        verdict = _check('set_task_status', live_status='in-progress')
        assert verdict.is_rejection is False

    def test_update_task_op_is_not_scoped_by_gate_2(self, monkeypatch):
        """Gate 2 is set_task_status-only: update_task must not surface
        ReconLiveWorkflowWriteRejected even when the detector reports live."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )
        verdict = _check('update_task', live_status='in-progress')
        assert verdict.error_type != 'ReconLiveWorkflowWriteRejected'

    def test_gate_2_forwards_live_status_as_status_kwarg(self, monkeypatch):
        """is_workflow_live_for_task must receive the caller's live_status as
        its `status` kwarg so it can suppress the project-wide
        orchestrator_live signal for done/cancelled/deferred tasks (see
        live_workflow_detector.ORCH_LIVE_INELIGIBLE_STATUSES) — otherwise a
        live orchestrator elsewhere in the project would falsely flag a
        terminal/deferred task's set_task_status write as gate-2-live."""
        captured = {}

        def _spy(*args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs
            return False

        monkeypatch.setattr(recon_write_policy, 'is_workflow_live_for_task', _spy)
        _check('set_task_status', task_id='7', live_status='deferred')

        assert captured['kwargs'].get('status') == 'deferred'


# ---------------------------------------------------------------------------
# check() gate 3 — stale snapshot (op-agnostic) + precedence
# ---------------------------------------------------------------------------


class TestCheckGate3StaleSnapshot:
    def test_stale_snapshot_rejects(self):
        verdict = _check(
            'update_task', live_status='in-progress', snapshot_token='pending',
        )
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconStaleSnapshotRejected'

    def test_fresh_snapshot_matching_live_status_is_ok(self):
        verdict = _check(
            'update_task', live_status='in-progress', snapshot_token='in-progress',
        )
        assert verdict.is_rejection is False

    def test_no_snapshot_token_skips_gate(self):
        verdict = _check(
            'update_task', live_status='in-progress', snapshot_token=None,
        )
        assert verdict.is_rejection is False

    def test_terminal_gate_takes_precedence_over_stale_snapshot(self):
        """Gate 1 (terminal) is checked before gate 3 (stale snapshot): a
        done task with a stale snapshot reports the more fundamental
        ReconTerminalWriteRejected, not ReconStaleSnapshotRejected."""
        verdict = _check(
            'update_task', live_status='done', snapshot_token='pending',
        )
        assert verdict.error_type == 'ReconTerminalWriteRejected'


# ---------------------------------------------------------------------------
# SNAPSHOT_TOKEN_KEYS / extract_snapshot_token
# ---------------------------------------------------------------------------


class TestExtractSnapshotToken:
    def test_snapshot_token_keys(self):
        assert recon_write_policy.SNAPSHOT_TOKEN_KEYS == ('snapshot_status', 'observed_status')

    def test_extract_from_dict_snapshot_status(self):
        assert recon_write_policy.extract_snapshot_token(
            {'snapshot_status': 'pending'},
        ) == 'pending'

    def test_extract_from_dict_observed_status_alias(self):
        assert recon_write_policy.extract_snapshot_token(
            {'observed_status': 'done'},
        ) == 'done'

    def test_extract_from_json_string(self):
        assert recon_write_policy.extract_snapshot_token(
            '{"snapshot_status":"pending"}',
        ) == 'pending'

    def test_extract_from_none_is_none(self):
        assert recon_write_policy.extract_snapshot_token(None) is None

    def test_extract_from_non_dict_is_none(self):
        assert recon_write_policy.extract_snapshot_token(42) is None

    def test_extract_from_dict_without_either_key_is_none(self):
        assert recon_write_policy.extract_snapshot_token({'other': 'x'}) is None

    def test_extract_from_dict_with_none_value_is_none(self):
        """A None snapshot value must not be coerced to the string 'None',
        which could never equal a real live_status and would spuriously
        trigger ReconStaleSnapshotRejected."""
        assert recon_write_policy.extract_snapshot_token({'snapshot_status': None}) is None

    def test_extract_from_dict_with_int_value_is_none(self):
        assert recon_write_policy.extract_snapshot_token({'snapshot_status': 42}) is None

    def test_extract_from_dict_with_bool_value_is_none(self):
        assert recon_write_policy.extract_snapshot_token({'snapshot_status': True}) is None

    def test_extract_from_dict_with_empty_string_value_is_none(self):
        assert recon_write_policy.extract_snapshot_token({'snapshot_status': ''}) is None


# ---------------------------------------------------------------------------
# Interceptor boundary fixtures (mirrors test_task_write_agent_id.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.remove_tasks = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': [{'type': 'knowledge_captured'}]})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'interceptor_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


# ---------------------------------------------------------------------------
# P1 — interceptor.update_task terminal-write boundary
# ---------------------------------------------------------------------------


class TestInterceptorUpdateTaskTerminalBoundary:
    @pytest.mark.asyncio
    async def test_recon_stage_update_task_on_done_task_rejects(self, interceptor, taskmaster):
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project', title='x', agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_recon_agent_id_is_not_gated(self, interceptor, taskmaster):
        """Recon-scoping negative: a non-recon-stage agent_id on the same
        done task is never gated — the write proceeds normally."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        await interceptor.update_task('1', '/project', title='x', agent_id=None)

        taskmaster.update_task.assert_awaited_once()


# ---------------------------------------------------------------------------
# P2 — interceptor.set_task_status live-workflow boundary
# ---------------------------------------------------------------------------


class TestInterceptorSetTaskStatusLiveWorkflowBoundary:
    @pytest.mark.asyncio
    async def test_recon_stage_set_task_status_with_live_workflow_rejects(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """Default taskmaster.get_task fixture returns status='pending'."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )

        result = await interceptor.set_task_status(
            '1', 'in-progress', '/project', agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_recon_agent_id_is_not_gated(self, interceptor, taskmaster, monkeypatch):
        """Recon-scoping negative: a non-recon-stage agent_id is never gated
        even when the detector reports a live workflow."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )

        await interceptor.set_task_status('1', 'in-progress', '/project', agent_id=None)

        taskmaster.set_task_status.assert_awaited_once()


# ---------------------------------------------------------------------------
# P3 — interceptor.update_task stale-snapshot boundary
# ---------------------------------------------------------------------------


class TestInterceptorUpdateTaskStaleSnapshotBoundary:
    @pytest.mark.asyncio
    async def test_recon_stage_update_task_with_stale_snapshot_rejects(
        self, interceptor, taskmaster,
    ):
        """live_status='in-progress' is non-terminal, so gate 1 does not fire."""
        taskmaster.get_task = AsyncMock(
            return_value={'id': '1', 'status': 'in-progress', 'title': 'T'},
        )

        result = await interceptor.update_task(
            '1', '/project', metadata={'snapshot_status': 'pending'}, agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconStaleSnapshotRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fresh_snapshot_matching_live_status_proceeds(self, interceptor, taskmaster):
        taskmaster.get_task = AsyncMock(
            return_value={'id': '1', 'status': 'in-progress', 'title': 'T'},
        )

        await interceptor.update_task(
            '1', '/project', metadata={'snapshot_status': 'in-progress'}, agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
