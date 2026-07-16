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

Amendment round (post-plan reviewer findings, all in-scope for this task's
locked modules):

- TestPoolCapScopeSnapshots           (_pool_cap_scope_snapshots direct coverage)
- TestPrecheckCatastrophicFailure     (outer, whole-batch fail-open guard)
- TestPrecheckStatsReconcilable       (candidates/reinvestigated/skipped counters)
- TestPrecheckConsecutiveSkipCap      (skip_streak + forced re-investigation)

Second amendment round (task 2417 amendment pass #2 — reviewer findings on
the FIRST amendment round's own fixes):

- TestPrecheckCreatedAtTypeSafety     (max() sort-key non-str created_at)
- TestPrecheckReinvestigatedExactCount (reinvestigated == len(to_reinvestigate))
"""

from __future__ import annotations

from typing import Any
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

        kwargs: dict[str, Any] = {
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

        kwargs: dict[str, Any] = {
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


# ── Amendment round: post-plan reviewer findings ──────────────────────────
#
# All four classes below were added in the amendment pass (not part of the
# original step-1..step-18 TDD plan) to address reviewer findings:
#   - TestPoolCapScopeSnapshots / TestPrecheckCatastrophicFailure:
#     test_coverage finding — _pool_cap_scope_snapshots and the outer
#     (whole-batch) fail-open guard had no direct test.
#   - TestPrecheckStatsReconcilable: observability_stats_inconsistency
#     finding — scope_freshness_candidates must only count findings with a
#     resolvable root, and every per-finding failure path must still
#     increment scope_freshness_reinvestigated.
#   - TestPrecheckConsecutiveSkipCap: robustness_silent_degradation finding
#     — a (task_ref, flag_key) pair must not be skipped forever; see
#     scope_freshness.py's "Consecutive-skip cap" module docstring section.


class TestPoolCapScopeSnapshots:
    """Tests for _pool_cap_scope_snapshots(memory_service, prior_memories, project_id, task_ref)."""

    @pytest.mark.asyncio
    async def test_deletes_every_prior_memory(self):
        from fused_memory.reconciliation.scope_freshness import _pool_cap_scope_snapshots

        memory_service = AsyncMock()
        prior_memories = [
            {'id': 'a1', 'created_at': 't0', 'metadata': {}},
            {'id': 'a2', 'created_at': 't1', 'metadata': {}},
        ]

        await _pool_cap_scope_snapshots(
            memory_service, prior_memories, 'autopilot_video', 'dark_factory:2405',
        )

        assert memory_service.delete_memory.await_count == 2
        memory_service.delete_memory.assert_any_await(
            memory_id='a1', store='mem0', project_id='autopilot_video',
        )
        memory_service.delete_memory.assert_any_await(
            memory_id='a2', store='mem0', project_id='autopilot_video',
        )

    @pytest.mark.asyncio
    async def test_noop_when_no_prior_memories(self):
        from fused_memory.reconciliation.scope_freshness import _pool_cap_scope_snapshots

        memory_service = AsyncMock()

        await _pool_cap_scope_snapshots(
            memory_service, [], 'autopilot_video', 'dark_factory:2405',
        )

        memory_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_swallows_per_item_delete_error_and_continues(self):
        from fused_memory.reconciliation.scope_freshness import _pool_cap_scope_snapshots

        memory_service = AsyncMock()
        memory_service.delete_memory.side_effect = [RuntimeError('boom'), None]
        prior_memories = [
            {'id': 'a1', 'created_at': 't0', 'metadata': {}},
            {'id': 'a2', 'created_at': 't1', 'metadata': {}},
        ]

        # Must not raise even though the first delete fails; both ids are
        # still attempted.
        await _pool_cap_scope_snapshots(
            memory_service, prior_memories, 'autopilot_video', 'dark_factory:2405',
        )

        assert memory_service.delete_memory.await_count == 2


class TestPrecheckCatastrophicFailure:
    """Tests for precheck_scope_correction_freshness's OUTER fail-open guard —
    a failure in the pure per-finding triage itself (is_cross_project_scope_correction
    / compute_scope_signature / select_primary_subject, called OUTSIDE the
    inner per-finding try/except), not merely a get_task/Mem0 I/O failure."""

    @pytest.mark.asyncio
    async def test_catastrophic_failure_returns_all_findings_unfiltered(self):
        from unittest.mock import patch

        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        taskmaster = AsyncMock()

        findings = [
            {
                'flag_type': 'cross_project',
                'cited_tasks': [
                    {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'},
                ],
            },
            {'flag_type': 'task_memory_mismatch', 'description': 'unrelated'},
        ]

        with patch(
            'fused_memory.reconciliation.scope_freshness.is_cross_project_scope_correction',
            side_effect=RuntimeError('pure helper exploded'),
        ):
            result = await precheck_scope_correction_freshness(
                memory_service=memory_service,
                taskmaster=taskmaster,
                project_id='autopilot_video',
                resolve_project_root=lambda pid: f'/roots/{pid}',
                run_id='run-8',
                findings=findings,
            )

        assert result.to_reinvestigate == findings
        assert result.skipped == []
        assert result.stats == {
            'scope_freshness_candidates': 0,
            'scope_freshness_reinvestigated': 0,
            'scope_freshness_skipped': 0,
            'scope_freshness_forced_reinvestigation': 0,
        }
        taskmaster.get_task.assert_not_awaited()
        memory_service.get_memories_by_metadata.assert_not_awaited()


class TestPrecheckStatsReconcilable:
    """Tests locking in the stats-counter fix (reviewer finding
    observability_stats_inconsistency): scope_freshness_candidates only
    counts findings with a resolvable root, and every per-finding failure
    path still increments scope_freshness_reinvestigated."""

    @pytest.mark.asyncio
    async def test_unresolvable_root_not_counted_as_candidate(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        taskmaster = AsyncMock()
        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: None,
            run_id='run-9',
            findings=[finding],
        )

        assert result.stats['scope_freshness_candidates'] == 0
        assert result.stats['scope_freshness_reinvestigated'] == 1

    @pytest.mark.asyncio
    async def test_per_finding_exception_still_increments_reinvestigated(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.side_effect = RuntimeError('qdrant down')
        taskmaster = AsyncMock()
        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-10',
            findings=[finding],
        )

        # candidates was incremented (root resolved, so a live comparison
        # was actually attempted) and the exception path still increments
        # reinvestigated — the two stay reconcilable even on this failure
        # path.
        assert result.stats['scope_freshness_candidates'] == 1
        assert result.stats['scope_freshness_reinvestigated'] == 1
        assert (
            result.stats['scope_freshness_candidates']
            <= result.stats['scope_freshness_reinvestigated']
            + result.stats['scope_freshness_skipped']
        )


class TestPrecheckConsecutiveSkipCap:
    """Tests for the consecutive-skip cap (reviewer finding
    robustness_silent_degradation): a (task_ref, flag_key) pair may be
    skipped at most `max_consecutive_skips - 1` cycles in a row before
    precheck_scope_correction_freshness forces a real re-investigation, so a
    genuinely-stranded thread can never be silently skipped past Stage 3's
    persistence-escalation window forever."""

    def _prior_snapshot(self, **overrides):
        from fused_memory.reconciliation.scope_freshness import (
            build_scope_snapshot_metadata,
        )

        kwargs: dict[str, Any] = {
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

    def _unchanged_finding_and_backends(self, prior_metadata):
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {'id': 'a1', 'created_at': 't0', 'metadata': prior_metadata},
        ]
        taskmaster = AsyncMock()
        taskmaster.get_task.return_value = {
            'id': 2405, 'status': 'pending',
            'updatedAt': '2026-07-10T10:00:00Z', 'description': 'd', 'metadata': {},
        }
        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }
        return memory_service, taskmaster, finding

    @pytest.mark.asyncio
    async def test_skip_streak_increments_while_under_cap(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        prior_metadata = self._prior_snapshot(skip_streak=1)
        memory_service, taskmaster, finding = self._unchanged_finding_and_backends(prior_metadata)

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-11',
            findings=[finding],
            max_consecutive_skips=4,
        )

        assert finding in result.skipped
        assert finding not in result.to_reinvestigate
        _, add_kwargs = memory_service.add_memory.await_args
        assert add_kwargs['metadata']['skip_streak'] == 2
        assert add_kwargs['metadata']['no_change'] is True
        assert result.stats['scope_freshness_forced_reinvestigation'] == 0

    @pytest.mark.asyncio
    async def test_forces_reinvestigation_when_cap_reached(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        # Already skipped 3 times in a row; max_consecutive_skips=4 means
        # this 4th consecutive fresh cycle must be forced back for real
        # re-investigation rather than skipped a 4th time.
        prior_metadata = self._prior_snapshot(skip_streak=3)
        memory_service, taskmaster, finding = self._unchanged_finding_and_backends(prior_metadata)

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-12',
            findings=[finding],
            max_consecutive_skips=4,
        )

        assert finding in result.to_reinvestigate
        assert finding not in result.skipped
        assert result.stats['scope_freshness_forced_reinvestigation'] == 1
        assert result.stats['scope_freshness_reinvestigated'] == 1
        assert result.stats['scope_freshness_skipped'] == 0

        # The rewritten snapshot resets the streak and is NOT a no_change marker.
        _, add_kwargs = memory_service.add_memory.await_args
        assert add_kwargs['metadata']['skip_streak'] == 0
        assert 'no_change' not in add_kwargs['metadata']

    @pytest.mark.asyncio
    async def test_default_max_consecutive_skips_applies_when_not_passed(self):
        from fused_memory.reconciliation.scope_freshness import (
            DEFAULT_MAX_CONSECUTIVE_SKIPS,
            precheck_scope_correction_freshness,
        )

        prior_metadata = self._prior_snapshot(skip_streak=DEFAULT_MAX_CONSECUTIVE_SKIPS - 1)
        memory_service, taskmaster, finding = self._unchanged_finding_and_backends(prior_metadata)

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-13',
            findings=[finding],
        )

        assert finding in result.to_reinvestigate
        assert result.stats['scope_freshness_forced_reinvestigation'] == 1

    @pytest.mark.asyncio
    async def test_non_integer_prior_skip_streak_treated_as_zero(self):
        """A pre-amendment snapshot (or a corrupted one) with a malformed
        skip_streak must not crash the comparison — fail safe to streak=0
        (first skip under the new regime), not to an immediate force."""
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        prior_metadata = self._prior_snapshot()
        prior_metadata['skip_streak'] = 'not-a-number'
        memory_service, taskmaster, finding = self._unchanged_finding_and_backends(prior_metadata)

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-14',
            findings=[finding],
            max_consecutive_skips=4,
        )

        assert finding in result.skipped
        _, add_kwargs = memory_service.add_memory.await_args
        assert add_kwargs['metadata']['skip_streak'] == 1


class TestPrecheckCreatedAtTypeSafety:
    """Tests for precheck_scope_correction_freshness — reviewer finding
    robustness (task 2417 amendment pass #2): the latest-prior-snapshot
    selection assumed every record's created_at was a string. A non-str
    created_at (e.g. a Mem0 backend returning a datetime or numeric
    timestamp) must not raise, must still pick the correct latest snapshot,
    and must log loudly rather than silently and permanently falling back
    to full re-investigation every cycle."""

    @pytest.mark.asyncio
    async def test_non_str_created_at_does_not_raise_and_picks_latest(self):
        from fused_memory.reconciliation.scope_freshness import (
            build_scope_snapshot_metadata,
            precheck_scope_correction_freshness,
        )

        older_metadata = build_scope_snapshot_metadata(
            task_ref='dark_factory:2405', flag_key='cross_project',
            subject_project_id='dark_factory', subject_task_id='2405',
            status='pending', updated_at='2026-07-09T10:00:00Z',
            description='old', run_id='run-0', snapshot_at='2026-07-09T10:00:00Z',
        )
        newer_metadata = build_scope_snapshot_metadata(
            task_ref='dark_factory:2405', flag_key='cross_project',
            subject_project_id='dark_factory', subject_task_id='2405',
            status='pending', updated_at='2026-07-10T10:00:00Z',
            description='new', run_id='run-1', snapshot_at='2026-07-10T10:00:00Z',
        )

        from datetime import UTC, datetime

        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'old-id',
                'created_at': datetime(2026, 7, 9, 10, 0, tzinfo=UTC),
                'metadata': older_metadata,
            },
            {
                'id': 'new-id',
                'created_at': datetime(2026, 7, 10, 10, 0, tzinfo=UTC),
                'metadata': newer_metadata,
            },
        ]
        taskmaster = AsyncMock()
        # Matches the NEWER snapshot only — proves max() picked 'new-id'
        # despite neither created_at being a str.
        taskmaster.get_task.return_value = {
            'id': 2405, 'status': 'pending',
            'updatedAt': '2026-07-10T10:00:00Z', 'description': 'new', 'metadata': {},
        }

        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-15',
            findings=[finding],
        )

        # Did not raise (no catastrophic fallback) and correctly compared
        # against the newer of the two snapshots.
        assert result.stats['scope_freshness_candidates'] == 1
        assert finding in result.skipped
        assert finding not in result.to_reinvestigate

    @pytest.mark.asyncio
    async def test_non_str_created_at_logs_loudly(self, caplog):
        import logging
        from datetime import UTC, datetime

        from fused_memory.reconciliation.scope_freshness import (
            build_scope_snapshot_metadata,
            precheck_scope_correction_freshness,
        )

        prior_metadata = build_scope_snapshot_metadata(
            task_ref='dark_factory:2405', flag_key='cross_project',
            subject_project_id='dark_factory', subject_task_id='2405',
            status='pending', updated_at='2026-07-10T10:00:00Z',
            description='d', run_id='run-0', snapshot_at='2026-07-10T10:00:00Z',
        )
        memory_service = AsyncMock()
        memory_service.get_memories_by_metadata.return_value = [
            {
                'id': 'a1',
                'created_at': datetime(2026, 7, 10, 10, 0, tzinfo=UTC),
                'metadata': prior_metadata,
            },
        ]
        taskmaster = AsyncMock()
        taskmaster.get_task.return_value = {
            'id': 2405, 'status': 'pending',
            'updatedAt': '2026-07-10T10:00:00Z', 'description': 'd', 'metadata': {},
        }
        finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }

        with caplog.at_level(logging.WARNING):
            await precheck_scope_correction_freshness(
                memory_service=memory_service,
                taskmaster=taskmaster,
                project_id='autopilot_video',
                resolve_project_root=lambda pid: f'/roots/{pid}',
                run_id='run-16',
                findings=[finding],
            )

        drift_logs = [
            r for r in caplog.records
            if r.message == 'reconciliation.scope_freshness_created_at_type_drift'
        ]
        assert drift_logs, (
            f'Expected a loud warning on created_at type drift, got: '
            f'{[r.message for r in caplog.records]}'
        )


class TestPrecheckReinvestigatedExactCount:
    """Tests for precheck_scope_correction_freshness — reviewer finding
    observability (task 2417 amendment pass #2): scope_freshness_reinvestigated
    must equal len(to_reinvestigate) EXACTLY, including for findings that
    never become scope-correction "candidates" at all (plain pass-through,
    and scope-correction findings with no usable subject/signature) — a
    prior fix only covered the unresolvable-root and per-finding-error
    kept-branches."""

    @pytest.mark.asyncio
    async def test_non_scope_correction_finding_increments_reinvestigated(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        taskmaster = AsyncMock()
        plain_finding = {
            'flag_type': 'task_memory_mismatch',
            'category': 'memory_stale',
            'description': 'unrelated finding',
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-17',
            findings=[plain_finding],
        )

        assert result.to_reinvestigate == [plain_finding]
        assert result.stats['scope_freshness_reinvestigated'] == len(result.to_reinvestigate)
        taskmaster.get_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_usable_subject_finding_increments_reinvestigated(self):
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        taskmaster = AsyncMock()
        # Cross-project-flagged but no cited_tasks at all -> no usable subject.
        no_subject_finding = {
            'flag_type': 'cross_project',
            'description': 'malformed scope-correction finding',
            'cited_tasks': [],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: f'/roots/{pid}',
            run_id='run-18',
            findings=[no_subject_finding],
        )

        assert result.to_reinvestigate == [no_subject_finding]
        assert result.stats['scope_freshness_reinvestigated'] == len(result.to_reinvestigate)
        taskmaster.get_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mixed_batch_reinvestigated_equals_to_reinvestigate_length(self):
        """A batch mixing a plain pass-through finding, a no-subject finding,
        and an unresolvable-root candidate must still satisfy the exact
        identity — not merely the older inequality."""
        from fused_memory.reconciliation.scope_freshness import (
            precheck_scope_correction_freshness,
        )

        memory_service = AsyncMock()
        taskmaster = AsyncMock()
        plain_finding = {'flag_type': 'task_memory_mismatch', 'category': 'memory_stale'}
        no_subject_finding = {'flag_type': 'cross_project', 'cited_tasks': []}
        unresolvable_root_finding = {
            'flag_type': 'cross_project',
            'cited_tasks': [{'project_id': 'dark_factory', 'task_id': '2405', 'title': 'x'}],
        }

        result = await precheck_scope_correction_freshness(
            memory_service=memory_service,
            taskmaster=taskmaster,
            project_id='autopilot_video',
            resolve_project_root=lambda pid: None,
            run_id='run-19',
            findings=[plain_finding, no_subject_finding, unresolvable_root_finding],
        )

        assert len(result.to_reinvestigate) == 3
        assert result.stats['scope_freshness_reinvestigated'] == 3
        assert result.stats['scope_freshness_skipped'] == len(result.skipped) == 0
