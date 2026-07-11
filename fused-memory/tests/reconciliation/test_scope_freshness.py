"""Tests for fused_memory.reconciliation.scope_freshness — task 2417.

Reconciliation freshness pre-check: skip re-derivation of unchanged
cross-project scope-correction threads.  Grown step-by-step per plan.json:

- TestIsCrossProjectScopeCorrection   (step-1/2)
- TestScopeSignature                  (step-3/4)
- TestBuildScopeSnapshotMetadata      (step-5/6)
- TestSnapshotFreshness               (step-7/8)
- TestPrecheckBootstrap               (step-9/10)
- TestPrecheckFreshSkip               (step-11/12)
- TestPrecheckChangedAndFailOpen      (step-13/14)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestIsCrossProjectScopeCorrection:
    """Tests for is_cross_project_scope_correction(finding, project_id) -> bool."""

    def test_true_for_cross_project_flag_type_with_foreign_cited_task(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is True

    def test_true_for_cross_project_routing_category_with_foreign_cited_task(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'category': 'cross_project_routing',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is True

    def test_false_for_same_project_only_cited_tasks(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'autopilot_video', 'task_id': '540', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_for_non_cross_project_flag_type(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'flag_type': 'task_memory_mismatch',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_for_memory_stale_category(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'category': 'memory_stale',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_when_no_cited_tasks(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {'flag_type': 'cross_project'}
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_when_cited_tasks_empty_list(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {'flag_type': 'cross_project', 'cited_tasks': []}
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_for_empty_finding_dict(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        assert is_cross_project_scope_correction({}, 'autopilot_video') is False

    def test_false_for_malformed_finding_none(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        assert is_cross_project_scope_correction(None, 'autopilot_video') is False

    def test_false_when_cited_tasks_entries_not_dicts(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {'flag_type': 'cross_project', 'cited_tasks': ['not-a-dict']}
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_false_when_cited_tasks_not_a_list(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {'flag_type': 'cross_project', 'cited_tasks': 'dark_factory:2405'}
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is False

    def test_true_when_one_of_several_cited_tasks_is_foreign(self):
        from fused_memory.reconciliation.scope_freshness import (
            is_cross_project_scope_correction,
        )

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'autopilot_video', 'task_id': '540', 'title': 'x'},
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'y'},
            ],
        }
        assert is_cross_project_scope_correction(finding, 'autopilot_video') is True


class TestScopeSignature:
    """Tests for select_primary_subject(finding, project_id) -> tuple[str, str] | None
    and compute_scope_signature(finding, project_id) -> tuple[str, str] | None."""

    def test_select_primary_subject_returns_first_foreign_entry(self):
        from fused_memory.reconciliation.scope_freshness import select_primary_subject

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'autopilot_video', 'task_id': '540', 'title': 'x'},
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'y'},
                {'project_id': 'dark_factory', 'task_id': '1097', 'title': 'z'},
            ],
        }
        assert select_primary_subject(finding, 'autopilot_video') == ('dark_factory', '2405')

    def test_select_primary_subject_falls_back_to_first_entry_when_none_foreign(self):
        from fused_memory.reconciliation.scope_freshness import select_primary_subject

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'autopilot_video', 'task_id': '540', 'title': 'x'},
                {'project_id': 'autopilot_video', 'task_id': '544', 'title': 'y'},
            ],
        }
        assert select_primary_subject(finding, 'autopilot_video') == ('autopilot_video', '540')

    def test_select_primary_subject_none_when_cited_tasks_empty(self):
        from fused_memory.reconciliation.scope_freshness import select_primary_subject

        assert select_primary_subject({'flag_type': 'cross_project'}, 'autopilot_video') is None
        assert (
            select_primary_subject(
                {'flag_type': 'cross_project', 'cited_tasks': []}, 'autopilot_video',
            )
            is None
        )

    def test_select_primary_subject_coerces_task_id_to_str(self):
        from fused_memory.reconciliation.scope_freshness import select_primary_subject

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': 2405, 'title': 'x'}],
        }
        assert select_primary_subject(finding, 'autopilot_video') == ('dark_factory', '2405')

    def test_compute_scope_signature_uses_flag_type(self):
        from fused_memory.reconciliation.scope_freshness import compute_scope_signature

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }
        assert compute_scope_signature(finding, 'autopilot_video') == (
            'dark_factory:2405', 'cross_project',
        )

    def test_compute_scope_signature_falls_back_to_category(self):
        from fused_memory.reconciliation.scope_freshness import compute_scope_signature

        finding = {
            'category': 'cross_project_routing',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }
        assert compute_scope_signature(finding, 'autopilot_video') == (
            'dark_factory:2405', 'cross_project_routing',
        )

    def test_compute_scope_signature_none_when_no_flag_type_or_category(self):
        from fused_memory.reconciliation.scope_freshness import compute_scope_signature

        finding = {
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }
        assert compute_scope_signature(finding, 'autopilot_video') == ('dark_factory:2405', '')

    def test_compute_scope_signature_none_when_no_primary_subject(self):
        from fused_memory.reconciliation.scope_freshness import compute_scope_signature

        assert compute_scope_signature({'flag_type': 'cross_project'}, 'autopilot_video') is None


class TestBuildScopeSnapshotMetadata:
    """Tests for build_scope_snapshot_metadata(...) -> dict."""

    def _build(self, **overrides):
        from fused_memory.reconciliation.scope_freshness import (
            build_scope_snapshot_metadata,
        )

        kwargs = {
            'task_ref': 'dark_factory:2405',
            'flag_key': 'cross_project',
            'subject_project_id': 'dark_factory',
            'subject_task_id': '2405',
            'status': 'pending',
            'updated_at': '2026-07-10T10:00:00Z',
            'description': 'Some description text.',
            'run_id': 'run-1',
            'snapshot_at': '2026-07-10T14:29:33Z',
        }
        kwargs.update(overrides)
        return build_scope_snapshot_metadata(**kwargs)

    def test_returns_canonical_payload(self):
        from fused_memory.reconciliation.scope_freshness import (
            CONSOLIDATED_SCOPE_KIND,
            SCOPE_FRESHNESS_SOURCE,
            _content_fingerprint,
        )

        metadata = self._build()

        assert metadata['kind'] == CONSOLIDATED_SCOPE_KIND
        assert metadata['source'] == SCOPE_FRESHNESS_SOURCE
        assert metadata['task_id'] == 'dark_factory:2405'
        assert metadata['flag_type'] == 'cross_project'
        assert metadata['subject_project_id'] == 'dark_factory'
        assert metadata['subject_task_id'] == '2405'
        assert metadata['subject_status'] == 'pending'
        assert metadata['subject_updated_at'] == '2026-07-10T10:00:00Z'
        assert metadata['subject_description_fingerprint'] == _content_fingerprint(
            'Some description text.',
        )
        assert metadata['run_id'] == 'run-1'
        assert metadata['snapshot_at']

    def test_no_change_flag_defaults_absent(self):
        metadata = self._build()
        assert 'no_change' not in metadata

    def test_no_change_flag_set_when_requested(self):
        metadata = self._build(no_change=True)
        assert metadata['no_change'] is True

    def test_description_fingerprint_is_deterministic(self):
        first = self._build(description='Identical description.')
        second = self._build(description='Identical description.')
        assert (
            first['subject_description_fingerprint']
            == second['subject_description_fingerprint']
        )


class TestSnapshotFreshness:
    """Tests for _extract_task_fields(resp) and
    snapshot_is_fresh(snapshot_metadata, live_task) -> bool."""

    def test_extract_task_fields_handles_flat_sqlite_shape(self):
        from fused_memory.reconciliation.scope_freshness import _extract_task_fields

        task = {
            'id': 2405,
            'status': 'pending',
            'updatedAt': '2026-07-10T10:00:00Z',
            'description': 'd',
            'metadata': {'foo': 'bar'},
        }
        assert _extract_task_fields(task) == (
            'pending', '2026-07-10T10:00:00Z', 'd', {'foo': 'bar'},
        )

    def test_extract_task_fields_handles_data_envelope(self):
        from fused_memory.reconciliation.scope_freshness import _extract_task_fields

        resp = {
            'data': {
                'status': 'pending',
                'updatedAt': '2026-07-10T10:00:00Z',
                'description': 'd',
                'metadata': {'foo': 'bar'},
            },
        }
        assert _extract_task_fields(resp) == (
            'pending', '2026-07-10T10:00:00Z', 'd', {'foo': 'bar'},
        )

    def test_extract_task_fields_tolerates_missing_metadata(self):
        from fused_memory.reconciliation.scope_freshness import _extract_task_fields

        status, updated_at, description, metadata = _extract_task_fields(
            {'status': 'pending', 'updatedAt': 't0', 'description': 'd'},
        )
        assert (status, updated_at, description) == ('pending', 't0', 'd')
        assert metadata == {}

    def _snapshot_meta(self, **overrides):
        from fused_memory.reconciliation.scope_freshness import (
            build_scope_snapshot_metadata,
        )

        kwargs = {
            'task_ref': 'dark_factory:2405',
            'flag_key': 'cross_project',
            'subject_project_id': 'dark_factory',
            'subject_task_id': '2405',
            'status': 'pending',
            'updated_at': '2026-07-10T10:00:00Z',
            'description': 'd',
            'run_id': 'run-0',
            'snapshot_at': '2026-07-10T14:29:33Z',
        }
        kwargs.update(overrides)
        return build_scope_snapshot_metadata(**kwargs)

    def test_fresh_when_status_updated_at_and_description_all_match(self):
        from fused_memory.reconciliation.scope_freshness import snapshot_is_fresh

        snapshot_meta = self._snapshot_meta()
        live_task = {
            'status': 'pending', 'updatedAt': '2026-07-10T10:00:00Z', 'description': 'd',
        }
        assert snapshot_is_fresh(snapshot_meta, live_task) is True

    def test_not_fresh_when_updated_at_advanced(self):
        from fused_memory.reconciliation.scope_freshness import snapshot_is_fresh

        snapshot_meta = self._snapshot_meta()
        live_task = {
            'status': 'pending', 'updatedAt': '2026-07-11T00:00:00Z', 'description': 'd',
        }
        assert snapshot_is_fresh(snapshot_meta, live_task) is False

    def test_not_fresh_when_status_differs(self):
        from fused_memory.reconciliation.scope_freshness import snapshot_is_fresh

        snapshot_meta = self._snapshot_meta()
        live_task = {
            'status': 'in-progress', 'updatedAt': '2026-07-10T10:00:00Z', 'description': 'd',
        }
        assert snapshot_is_fresh(snapshot_meta, live_task) is False

    def test_not_fresh_when_description_differs(self):
        from fused_memory.reconciliation.scope_freshness import snapshot_is_fresh

        snapshot_meta = self._snapshot_meta()
        live_task = {
            'status': 'pending', 'updatedAt': '2026-07-10T10:00:00Z',
            'description': 'a materially different description',
        }
        assert snapshot_is_fresh(snapshot_meta, live_task) is False

    def test_not_fresh_when_snapshot_missing_subject_fields(self):
        from fused_memory.reconciliation.scope_freshness import snapshot_is_fresh

        live_task = {
            'status': 'pending', 'updatedAt': '2026-07-10T10:00:00Z', 'description': 'd',
        }
        assert snapshot_is_fresh({}, live_task) is False
        assert snapshot_is_fresh(
            {'subject_status': 'pending'}, live_task,
        ) is False


class TestPrecheckBootstrap:
    """Tests for precheck_scope_correction_freshness — bootstrap path (no prior snapshot)."""

    @pytest.mark.asyncio
    async def test_bootstrap_keeps_everything_and_writes_snapshot(self):
        from fused_memory.reconciliation.scope_freshness import (
            CONSOLIDATED_SCOPE_KIND,
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []
        taskmaster = AsyncMock()
        taskmaster.get_task.return_value = {
            'id': 2405,
            'status': 'pending',
            'updatedAt': '2026-07-10T10:00:00Z',
            'description': 'd',
            'metadata': {},
        }

        cross_project_finding = {
            'flag_type': 'cross_project',
            'description': 'scope correction thread',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }
        non_scope_finding = {
            'flag_type': 'task_memory_mismatch',
            'description': 'unrelated finding',
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-1',
            findings=[cross_project_finding, non_scope_finding],
        )

        # Bootstrap: no prior snapshot exists, so both findings are kept.
        assert cross_project_finding in result.to_reinvestigate
        assert non_scope_finding in result.to_reinvestigate
        assert result.skipped == []

        # Exactly one get_task call, for the cross-project finding's subject.
        taskmaster.get_task.assert_awaited_once_with(
            task_id='2405', project_root='/roots/dark_factory',
        )

        # A fresh snapshot was written for the cross-project finding.
        memory_service.add_memory.assert_awaited_once()
        _, kwargs = memory_service.add_memory.await_args
        assert kwargs['metadata']['kind'] == CONSOLIDATED_SCOPE_KIND
        assert kwargs['metadata']['task_id'] == 'dark_factory:2405'


class TestPrecheckFreshSkip:
    """Tests for precheck_scope_correction_freshness — fresh-skip path
    (a prior snapshot exists and the subject is unchanged)."""

    @pytest.mark.asyncio
    async def test_unchanged_subject_is_skipped_and_no_change_marker_written(self):
        from fused_memory.reconciliation.scope_freshness import (
            CONSOLIDATED_SCOPE_KIND,
            build_scope_snapshot_metadata,
            precheck_scope_correction_freshness,
        )

        prior_metadata = build_scope_snapshot_metadata(
            task_ref='dark_factory:2405',
            flag_key='cross_project',
            subject_project_id='dark_factory',
            subject_task_id='2405',
            status='pending',
            updated_at='2026-07-10T10:00:00Z',
            description='d',
            run_id='run-0',
            snapshot_at='2026-07-10T14:29:33Z',
        )

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'a4ed9cad',
                'created_at': '2026-07-10T14:29:33Z',
                'metadata': prior_metadata,
            },
        ]
        taskmaster = AsyncMock()
        # SAME status/updatedAt/description as the prior snapshot — unchanged.
        taskmaster.get_task.return_value = {
            'id': 2405,
            'status': 'pending',
            'updatedAt': '2026-07-10T10:00:00Z',
            'description': 'd',
            'metadata': {},
        }

        cross_project_finding = {
            'flag_type': 'cross_project',
            'description': 'scope correction thread',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-2',
            findings=[cross_project_finding],
        )

        # Unchanged subject: skipped, not sent back for re-investigation.
        assert cross_project_finding in result.skipped
        assert cross_project_finding not in result.to_reinvestigate

        # Exactly one get_task call, for the cross-project finding's subject.
        taskmaster.get_task.assert_awaited_once_with(
            task_id='2405', project_root='/roots/dark_factory',
        )

        # A lightweight 'still blocked, no change' marker was written.
        memory_service.add_memory.assert_awaited_once()
        _, add_kwargs = memory_service.add_memory.await_args
        assert add_kwargs['metadata']['kind'] == CONSOLIDATED_SCOPE_KIND
        assert add_kwargs['metadata']['task_id'] == 'dark_factory:2405'
        assert add_kwargs['metadata']['no_change'] is True

        # Prior snapshot pool-capped: deleted after the new marker was added.
        memory_service.delete_memory.assert_awaited_once_with(
            memory_id='a4ed9cad', store='mem0', project_id='autopilot_video',
        )

        assert result.stats['scope_freshness_skipped'] == 1


class TestPrecheckChangedAndFailOpen:
    """Tests for precheck_scope_correction_freshness — a changed subject, and
    fail-open behaviour on every kind of per-finding uncertainty."""

    @pytest.mark.asyncio
    async def test_changed_subject_is_reinvestigated_and_snapshot_rewritten(self):
        from fused_memory.reconciliation.scope_freshness import (
            build_scope_snapshot_metadata,
            precheck_scope_correction_freshness,
        )

        prior_metadata = build_scope_snapshot_metadata(
            task_ref='dark_factory:2405',
            flag_key='cross_project',
            subject_project_id='dark_factory',
            subject_task_id='2405',
            status='pending',
            updated_at='2026-07-10T10:00:00Z',
            description='d',
            run_id='run-0',
            snapshot_at='2026-07-10T14:29:33Z',
        )

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'a4ed9cad',
                'created_at': '2026-07-10T14:29:33Z',
                'metadata': prior_metadata,
            },
        ]
        taskmaster = AsyncMock()
        # ADVANCED updatedAt since the snapshot was taken — the subject moved.
        taskmaster.get_task.return_value = {
            'id': 2405,
            'status': 'pending',
            'updatedAt': '2026-07-11T00:00:00Z',
            'description': 'd',
            'metadata': {},
        }

        cross_project_finding = {
            'flag_type': 'cross_project',
            'description': 'scope correction thread',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-3',
            findings=[cross_project_finding],
        )

        assert cross_project_finding in result.to_reinvestigate
        assert cross_project_finding not in result.skipped

        memory_service.add_memory.assert_awaited_once()
        _, kwargs = memory_service.add_memory.await_args
        assert kwargs['metadata']['subject_updated_at'] == '2026-07-11T00:00:00Z'
        assert 'no_change' not in kwargs['metadata']

        # Stale prior snapshot pool-capped after the fresh one was written.
        memory_service.delete_memory.assert_awaited_once_with(
            memory_id='a4ed9cad', store='mem0', project_id='autopilot_video',
        )

    @pytest.mark.asyncio
    async def test_get_task_failure_keeps_finding_and_never_raises(self):
        from fused_memory.backends.task_backend_errors import TaskmasterError
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = []
        taskmaster = AsyncMock()
        taskmaster.get_task.side_effect = TaskmasterError(
            'TASKMASTER_TOOL_ERROR', 'No tasks found',
        )

        cross_project_finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-4',
            findings=[cross_project_finding],
        )

        assert result.to_reinvestigate == [cross_project_finding]
        assert result.skipped == []
        memory_service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_foreign_project_keeps_finding_without_calling_get_task(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        taskmaster = AsyncMock()

        cross_project_finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: None,
            run_id='run-5',
            findings=[cross_project_finding],
        )

        assert result.to_reinvestigate == [cross_project_finding]
        assert result.skipped == []
        taskmaster.get_task.assert_not_awaited()
        memory_service.get_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_memory_read_failure_keeps_finding_and_never_raises(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.side_effect = RuntimeError(
            'qdrant unavailable',
        )
        taskmaster = AsyncMock()

        cross_project_finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [
                {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
            ],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-6',
            findings=[cross_project_finding],
        )

        assert result.to_reinvestigate == [cross_project_finding]
        assert result.skipped == []

    @pytest.mark.asyncio
    async def test_non_cross_project_finding_passes_through_untouched(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        taskmaster = AsyncMock()

        non_scope_finding = {'flag_type': 'task_memory_mismatch', 'description': 'unrelated'}

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-7',
            findings=[non_scope_finding],
        )

        assert result.to_reinvestigate == [non_scope_finding]
        assert result.skipped == []
        taskmaster.get_task.assert_not_awaited()
        memory_service.get_memories_by_metadata.assert_not_awaited()
        memory_service.add_memory.assert_not_awaited()
