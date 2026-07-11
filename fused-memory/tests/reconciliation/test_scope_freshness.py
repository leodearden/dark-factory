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
