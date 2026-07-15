"""Tests for the code-enforced before/after live-task-write self-check (task 2624).

Incident: dark_factory task 2588 un-claim (mem0 5bde6829) — the "read
status/claimant_run_id/heartbeat_at via get_task immediately BEFORE and
AFTER a write to a live in-progress task, and self-file a finding on
unexpected divergence" rule was only an LLM prompt convention, skippable by
an inattentive/budget-constrained Stage-2 recon run. This module makes it
code-enforced.

Layout mirrors ``test_recon_write_policy.py``: pure-unit tests for the
detector core first, then the async ``guarded_recon_task_write`` wrapper
tests, then interceptor-boundary tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.middleware import live_task_write_guard as guard_mod
from fused_memory.middleware import recon_write_policy
from fused_memory.middleware.live_task_write_guard import (
    LIFECYCLE_RESET_FLAG_TYPE,
    LifecycleResetFinding,
    LiveTaskSnapshot,
    detect_lifecycle_reset,
    extract_live_snapshot,
    has_live_claimant,
)
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

LIVE_CLAIMANT = 'run-1/session-1/pid=123'

# Recon-stage agent_id used by the interceptor boundary tests — matches
# test_recon_write_policy.py's AGENT_ID so the recon-stage task-write setup
# is identical to that module's.
RECON_AGENT_ID = 'recon-stage-task_knowledge_sync'


def _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT, heartbeat_at='2026-07-15T12:00:00+00:00', **extra):
    payload = {
        'id': '1',
        'status': status,
        'claimant_run_id': claimant_run_id,
        'heartbeat_at': heartbeat_at,
    }
    payload.update(extra)
    return payload


def _nested(status='in-progress', claimant_run_id=LIVE_CLAIMANT, heartbeat_at='2026-07-15T12:00:00+00:00', **extra):
    return {'data': _flat(status, claimant_run_id, heartbeat_at, **extra)}


# ---------------------------------------------------------------------------
# (a) extract_live_snapshot — flat / nested shape handling
# ---------------------------------------------------------------------------


class TestExtractLiveSnapshot:
    def test_flat_shape(self):
        snapshot = extract_live_snapshot(_flat())
        assert snapshot == LiveTaskSnapshot(
            status='in-progress',
            claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )

    def test_nested_data_shape(self):
        snapshot = extract_live_snapshot(_nested())
        assert snapshot == LiveTaskSnapshot(
            status='in-progress',
            claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )

    def test_flat_shape_wins_when_status_key_present(self):
        """Mirrors _extract_status's discriminator: presence of a top-level
        'status' key means flat, even if a 'data' key also happens to be
        present alongside it."""
        task_data = _flat()
        task_data['data'] = {'status': 'done', 'claimant_run_id': None, 'heartbeat_at': None}
        snapshot = extract_live_snapshot(task_data)
        assert snapshot.status == 'in-progress'
        assert snapshot.claimant_run_id == LIVE_CLAIMANT

    def test_missing_fields_default_safely(self):
        snapshot = extract_live_snapshot({'id': '1'})
        assert snapshot.status == 'unknown'
        assert snapshot.claimant_run_id is None
        assert snapshot.heartbeat_at is None

    def test_non_mapping_input_defaults_safely(self):
        snapshot = extract_live_snapshot(None)
        assert snapshot.status == 'unknown'
        assert snapshot.claimant_run_id is None
        assert snapshot.heartbeat_at is None


# ---------------------------------------------------------------------------
# (b) has_live_claimant
# ---------------------------------------------------------------------------


class TestHasLiveClaimant:
    def test_true_for_in_progress_with_claimant(self):
        assert has_live_claimant(_flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)) is True

    def test_true_for_in_progress_with_claimant_nested(self):
        assert has_live_claimant(_nested(status='in-progress', claimant_run_id=LIVE_CLAIMANT)) is True

    def test_false_when_not_in_progress(self):
        assert has_live_claimant(_flat(status='pending', claimant_run_id=LIVE_CLAIMANT)) is False
        assert has_live_claimant(_flat(status='done', claimant_run_id=LIVE_CLAIMANT)) is False

    def test_false_when_claimant_none(self):
        assert has_live_claimant(_flat(status='in-progress', claimant_run_id=None)) is False

    def test_false_when_claimant_blank(self):
        assert has_live_claimant(_flat(status='in-progress', claimant_run_id='   ')) is False

    def test_false_when_claimant_empty_string(self):
        assert has_live_claimant(_flat(status='in-progress', claimant_run_id='')) is False


# ---------------------------------------------------------------------------
# (c)-(f) detect_lifecycle_reset
# ---------------------------------------------------------------------------


def _detect(before, after, *, op='update_task', task_id='1', project_id='dark_factory',
            requested_status=None, requested_claimant_write=False, requested_heartbeat_write=False):
    return detect_lifecycle_reset(
        before,
        after,
        op=op,
        task_id=task_id,
        project_id=project_id,
        requested_status=requested_status,
        requested_claimant_write=requested_claimant_write,
        requested_heartbeat_write=requested_heartbeat_write,
    )


class TestDetectLifecycleReset:
    def test_unrequested_claimant_reset_fires(self):
        """(c) claimant_run_id resets to None un-requested during update_task."""
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='in-progress', claimant_run_id=None)

        finding = _detect(
            before, after, op='update_task', task_id='42', project_id='dark_factory',
            requested_status=None, requested_claimant_write=False,
        )

        assert finding is not None
        assert isinstance(finding, LifecycleResetFinding)
        assert finding.flag_type == LIFECYCLE_RESET_FLAG_TYPE
        assert finding.flag_type == 'task_lifecycle_reset_detected'
        assert finding.task_id == '42'
        assert finding.project_id == 'dark_factory'
        assert finding.op == 'update_task'
        assert 'claimant_run_id' in finding.diverged_fields

    def test_claimant_unchanged_is_benign(self):
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)

        assert _detect(before, after) is None

    def test_requested_claimant_write_clearing_is_benign(self):
        """The write itself explicitly cleared the claimant — not unexpected."""
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='in-progress', claimant_run_id=None)

        assert _detect(before, after, requested_claimant_write=True) is None

    def test_before_not_in_progress_is_dormant(self):
        before = _flat(status='pending', claimant_run_id=None)
        after = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)

        assert _detect(before, after) is None

    def test_before_in_progress_without_live_claimant_is_dormant(self):
        before = _flat(status='in-progress', claimant_run_id=None)
        after = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)

        assert _detect(before, after) is None

    def test_status_changing_to_exactly_requested_status_is_benign(self):
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='done', claimant_run_id=LIVE_CLAIMANT)

        finding = _detect(
            before, after, op='set_task_status', requested_status='done',
            requested_claimant_write=False,
        )

        assert finding is None

    def test_unrequested_status_revert_fires(self):
        """(e) requested_status='done' but after status=='pending'."""
        before = _flat(status='in-progress', claimant_run_id=LIVE_CLAIMANT)
        after = _flat(status='pending', claimant_run_id=LIVE_CLAIMANT)

        finding = _detect(
            before, after, op='set_task_status', requested_status='done',
            requested_claimant_write=False,
        )

        assert finding is not None
        assert 'status' in finding.diverged_fields

    def test_heartbeat_only_change_does_not_trigger(self):
        """(f) claimant + status unchanged, only heartbeat_at differs (a
        legitimate concurrent heartbeat refresh) — must not false-positive."""
        before = _flat(
            status='in-progress', claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )
        after = _flat(
            status='in-progress', claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:05:00+00:00',
        )

        assert _detect(before, after) is None

    def test_heartbeat_recorded_as_corroboration_when_finding_fires(self):
        """When a real finding fires, an un-requested heartbeat divergence is
        included in diverged_fields as corroborating evidence."""
        before = _flat(
            status='in-progress', claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )
        after = _flat(
            status='in-progress', claimant_run_id=None,
            heartbeat_at=None,
        )

        finding = _detect(before, after, op='update_task', requested_claimant_write=False)

        assert finding is not None
        assert 'claimant_run_id' in finding.diverged_fields
        assert 'heartbeat_at' in finding.diverged_fields

    def test_requested_heartbeat_write_suppresses_heartbeat_corroboration(self):
        before = _flat(
            status='in-progress', claimant_run_id=LIVE_CLAIMANT,
            heartbeat_at='2026-07-15T12:00:00+00:00',
        )
        after = _flat(
            status='in-progress', claimant_run_id=None,
            heartbeat_at='2026-07-15T12:05:00+00:00',
        )

        finding = _detect(
            before, after, op='update_task',
            requested_claimant_write=False, requested_heartbeat_write=True,
        )

        assert finding is not None
        assert 'heartbeat_at' not in finding.diverged_fields


# ---------------------------------------------------------------------------
# guarded_recon_task_write — async wrapper (acceptance regression)
# ---------------------------------------------------------------------------

SENTINEL_RESULT = {'success': True, 'sentinel': 'do-write-result'}


def _live_claimant_before(claimant_run_id=LIVE_CLAIMANT, status='in-progress'):
    return _flat(status=status, claimant_run_id=claimant_run_id)


class TestGuardedReconTaskWrite:
    @pytest.mark.asyncio
    async def test_claimant_reset_files_finding(self):
        """(1) SIMULATED CLAIMANT RESET: get_task returns a live-claimant
        before-state then an un-claimed after-state; do_write's result is
        returned and file_finding is awaited once with the finding."""
        before = _live_claimant_before()
        after = _flat(status='in-progress', claimant_run_id=None)
        get_task = AsyncMock(side_effect=[before, after])
        do_write = AsyncMock(return_value=SENTINEL_RESULT)
        file_finding = AsyncMock()

        result = await guard_mod.guarded_recon_task_write(
            task_id='42',
            project_id='dark_factory',
            op='update_task',
            get_task=get_task,
            do_write=do_write,
            file_finding=file_finding,
            project_root='/project',
        )

        assert result == SENTINEL_RESULT
        do_write.assert_awaited_once()
        file_finding.assert_awaited_once()
        finding = file_finding.await_args.args[0]
        assert isinstance(finding, LifecycleResetFinding)
        assert finding.flag_type == 'task_lifecycle_reset_detected'
        assert finding.task_id == '42'

    @pytest.mark.asyncio
    async def test_benign_write_does_not_file(self):
        """(2) benign write (claimant identical before/after) -> not filed."""
        before = _live_claimant_before()
        after = _live_claimant_before()
        get_task = AsyncMock(side_effect=[before, after])
        do_write = AsyncMock(return_value=SENTINEL_RESULT)
        file_finding = AsyncMock()

        result = await guard_mod.guarded_recon_task_write(
            task_id='42', project_id='dark_factory', op='update_task',
            get_task=get_task, do_write=do_write, file_finding=file_finding,
            project_root='/project',
        )

        assert result == SENTINEL_RESULT
        do_write.assert_awaited_once()
        file_finding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dormant_when_before_not_live_claimant(self):
        """(3) dormant: before has no live claimant -> do_write still runs,
        no get_task-after (get_task called exactly once), not filed."""
        before = _flat(status='pending', claimant_run_id=None)
        get_task = AsyncMock(return_value=before)
        do_write = AsyncMock(return_value=SENTINEL_RESULT)
        file_finding = AsyncMock()

        result = await guard_mod.guarded_recon_task_write(
            task_id='42', project_id='dark_factory', op='update_task',
            get_task=get_task, do_write=do_write, file_finding=file_finding,
            project_root='/project',
        )

        assert result == SENTINEL_RESULT
        do_write.assert_awaited_once()
        get_task.assert_awaited_once()
        file_finding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_requested_claimant_write_clear_not_filed(self):
        """(4) requested_claimant_write=True with claimant->None -> not filed."""
        before = _live_claimant_before()
        after = _flat(status='in-progress', claimant_run_id=None)
        get_task = AsyncMock(side_effect=[before, after])
        do_write = AsyncMock(return_value=SENTINEL_RESULT)
        file_finding = AsyncMock()

        result = await guard_mod.guarded_recon_task_write(
            task_id='42', project_id='dark_factory', op='set_task_status',
            get_task=get_task, do_write=do_write, file_finding=file_finding,
            project_root='/project', requested_claimant_write=True,
        )

        assert result == SENTINEL_RESULT
        file_finding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_do_write_raising_propagates(self):
        """(5) do_write raising propagates; file_finding is never invoked."""
        before = _live_claimant_before()
        get_task = AsyncMock(return_value=before)
        do_write = AsyncMock(side_effect=RuntimeError('write blew up'))
        file_finding = AsyncMock()

        with pytest.raises(RuntimeError, match='write blew up'):
            await guard_mod.guarded_recon_task_write(
                task_id='42', project_id='dark_factory', op='update_task',
                get_task=get_task, do_write=do_write, file_finding=file_finding,
                project_root='/project',
            )

        file_finding.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_file_finding_raising_is_swallowed(self):
        """(6) file_finding raising is swallowed; do_write's result still returns."""
        before = _live_claimant_before()
        after = _flat(status='in-progress', claimant_run_id=None)
        get_task = AsyncMock(side_effect=[before, after])
        do_write = AsyncMock(return_value=SENTINEL_RESULT)
        file_finding = AsyncMock(side_effect=RuntimeError('filer blew up'))

        result = await guard_mod.guarded_recon_task_write(
            task_id='42', project_id='dark_factory', op='update_task',
            get_task=get_task, do_write=do_write, file_finding=file_finding,
            project_root='/project',
        )

        assert result == SENTINEL_RESULT
        file_finding.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_after_read_raising_is_swallowed(self):
        """(7) after-read get_task raising is swallowed; write result still returns."""
        before = _live_claimant_before()
        get_task = AsyncMock(side_effect=[before, RuntimeError('get_task blew up')])
        do_write = AsyncMock(return_value=SENTINEL_RESULT)
        file_finding = AsyncMock()

        result = await guard_mod.guarded_recon_task_write(
            task_id='42', project_id='dark_factory', op='update_task',
            get_task=get_task, do_write=do_write, file_finding=file_finding,
            project_root='/project',
        )

        assert result == SENTINEL_RESULT
        file_finding.assert_not_awaited()


# ---------------------------------------------------------------------------
# Interceptor boundary fixtures (mirrors test_recon_write_policy.py:376-407)
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
    buf = EventBuffer(db_path=tmp_path / 'interceptor_eb_guard.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


# ---------------------------------------------------------------------------
# Interceptor boundary — update_task recon-stage lifecycle guard
# ---------------------------------------------------------------------------


class TestInterceptorUpdateTaskLifecycleGuard:
    @pytest.mark.asyncio
    async def test_claimant_reset_invokes_filer(self, interceptor, taskmaster):
        """(a) update_task recon-stage write to the in-progress live-claimant
        task invokes the filer once with flag_type='task_lifecycle_reset_detected'."""
        before = _live_claimant_before()
        after = _flat(status='in-progress', claimant_run_id=None)
        taskmaster.get_task = AsyncMock(side_effect=[before, after])
        filer = AsyncMock()
        interceptor.set_lifecycle_reset_filer(filer)

        await interceptor.update_task('1', '/project', title='x', agent_id=RECON_AGENT_ID)

        filer.assert_awaited_once()
        finding = filer.await_args.args[0]
        assert finding.flag_type == 'task_lifecycle_reset_detected'
        assert finding.task_id == '1'

    @pytest.mark.asyncio
    async def test_benign_after_state_does_not_invoke_filer(self, interceptor, taskmaster):
        """(c) a benign after-state (claimant unchanged) does NOT invoke the filer."""
        before = _live_claimant_before()
        after = _live_claimant_before()
        taskmaster.get_task = AsyncMock(side_effect=[before, after])
        filer = AsyncMock()
        interceptor.set_lifecycle_reset_filer(filer)

        await interceptor.update_task('1', '/project', title='x', agent_id=RECON_AGENT_ID)

        filer.assert_not_awaited()
        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_recon_agent_id_does_not_invoke_filer(self, interceptor, taskmaster):
        """(d) a non-recon agent_id does NOT invoke the filer and the write proceeds."""
        before = _live_claimant_before()
        taskmaster.get_task = AsyncMock(return_value=before)
        filer = AsyncMock()
        interceptor.set_lifecycle_reset_filer(filer)

        await interceptor.update_task('1', '/project', title='x', agent_id=None)

        filer.assert_not_awaited()
        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_filer_set_write_proceeds_unchanged(self, interceptor, taskmaster):
        """(e) with NO filer set (default), the write proceeds unchanged and
        nothing is filed — guards existing behavior."""
        before = _live_claimant_before()
        after = _flat(status='in-progress', claimant_run_id=None)
        taskmaster.get_task = AsyncMock(side_effect=[before, after])
        # No set_lifecycle_reset_filer call — filer stays at its None default.

        result = await interceptor.update_task('1', '/project', title='x', agent_id=RECON_AGENT_ID)

        taskmaster.update_task.assert_awaited_once()
        assert taskmaster.get_task.await_count == 1
        assert result.get('success') is not False


# ---------------------------------------------------------------------------
# Interceptor boundary — set_task_status recon-stage lifecycle guard
# ---------------------------------------------------------------------------


class TestInterceptorSetTaskStatusLifecycleGuard:
    @pytest.mark.asyncio
    async def test_claimant_reset_invokes_filer(self, interceptor, taskmaster, monkeypatch):
        """(b) set_task_status recon-stage transition invokes the filer when
        the claimant resets un-requested."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: False,
        )
        before = _live_claimant_before()
        after = _flat(status='review', claimant_run_id=None)
        taskmaster.get_task = AsyncMock(side_effect=[before, after])
        filer = AsyncMock()
        interceptor.set_lifecycle_reset_filer(filer)

        await interceptor.set_task_status('1', 'review', '/project', agent_id=RECON_AGENT_ID)

        filer.assert_awaited_once()
        finding = filer.await_args.args[0]
        assert finding.flag_type == 'task_lifecycle_reset_detected'
        assert finding.op == 'set_task_status'

    @pytest.mark.asyncio
    async def test_benign_after_state_does_not_invoke_filer(self, interceptor, taskmaster, monkeypatch):
        """(c) a benign after-state (claimant unchanged) does NOT invoke the filer."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: False,
        )
        before = _live_claimant_before()
        after = _flat(status='review', claimant_run_id=LIVE_CLAIMANT)
        taskmaster.get_task = AsyncMock(side_effect=[before, after])
        filer = AsyncMock()
        interceptor.set_lifecycle_reset_filer(filer)

        await interceptor.set_task_status('1', 'review', '/project', agent_id=RECON_AGENT_ID)

        filer.assert_not_awaited()
        taskmaster.set_task_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_recon_agent_id_does_not_invoke_filer(self, interceptor, taskmaster, monkeypatch):
        """(d) a non-recon agent_id does NOT invoke the filer and the write proceeds."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )
        before = _live_claimant_before()
        taskmaster.get_task = AsyncMock(return_value=before)
        filer = AsyncMock()
        interceptor.set_lifecycle_reset_filer(filer)

        await interceptor.set_task_status('1', 'review', '/project', agent_id=None)

        filer.assert_not_awaited()
        taskmaster.set_task_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_filer_set_write_proceeds_unchanged(self, interceptor, taskmaster, monkeypatch):
        """(e) with NO filer set (default), the write proceeds unchanged and
        nothing is filed — guards existing behavior."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: False,
        )
        before = _live_claimant_before()
        after = _flat(status='review', claimant_run_id=None)
        taskmaster.get_task = AsyncMock(side_effect=[before, after])
        # No set_lifecycle_reset_filer call — filer stays at its None default.

        await interceptor.set_task_status('1', 'review', '/project', agent_id=RECON_AGENT_ID)

        taskmaster.set_task_status.assert_awaited_once()
        assert taskmaster.get_task.await_count == 1
