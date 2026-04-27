"""Tests for ScopeViolationEscalator — writes scope_violation escalations on path-guard rejection."""

from __future__ import annotations

import json

import pytest

from fused_memory.middleware import scope_violation_escalator as sve_mod
from fused_memory.middleware.scope_violation_escalator import ScopeViolationEscalator


@pytest.mark.skipif(
    not sve_mod.HAS_ESCALATION,
    reason='escalation package not installed in this environment',
)
class TestEscalationEnabled:
    def test_writes_escalation_under_project_root(self, tmp_path):
        esc = ScopeViolationEscalator()
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='Edit fused-memory/X',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
            suggested_root='/home/leo/src/dark-factory',
        )
        assert esc_id is not None
        # Escalation lands under {project_root}/data/escalations/{id}.json.
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('*.json'))
        assert len(files) == 1, f'expected one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == esc_id
        assert payload['category'] == 'scope_violation'
        assert payload['severity'] == 'info'
        assert payload['agent_role'] == 'fused-memory/path-guard'
        assert 'fused-memory/' in payload['summary']
        assert 'dark_factory' in payload['summary']
        # detail carries the routing context for the operator.
        assert 'reify' in payload['detail']
        assert 'fused-memory/' in payload['detail']
        assert 'dark_factory' in payload['detail']
        assert payload['suggested_action'] == 'resubmit_to_dark_factory'

    def test_caches_queue_per_project_root(self, tmp_path):
        """Repeated rejections for the same project_root reuse one EscalationQueue."""
        esc = ScopeViolationEscalator()
        esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='one',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='two',
            matched_paths=('orchestrator/',),
            suggested_project='dark_factory',
        )
        queues = esc._queues  # private but we explicitly assert caching here
        assert list(queues.keys()) == [str(tmp_path)]
        # Both escalations should be on disk with distinct IDs.
        files = sorted((tmp_path / 'data' / 'escalations').glob('*.json'))
        assert len(files) == 2

    def test_no_suggested_project_uses_manual_route_action(self, tmp_path):
        esc = ScopeViolationEscalator()
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='ambiguous task',
            matched_paths=('fused-memory/', 'crates_other/'),
            suggested_project=None,
        )
        assert esc_id is not None
        files = list((tmp_path / 'data' / 'escalations').glob('*.json'))
        payload = json.loads(files[0].read_text())
        assert payload['suggested_action'] == 'manual_route'

    def test_queue_failure_returns_none_does_not_raise(self, tmp_path, monkeypatch):
        """A queue submit failure must be swallowed — escalation is additive."""
        esc = ScopeViolationEscalator()

        # Force the underlying queue.submit to raise.
        from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

        def boom(self, _esc):
            raise RuntimeError('disk full')

        monkeypatch.setattr(EscalationQueue, 'submit', boom)
        result = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='will fail',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        assert result is None  # no exception propagates


class TestEscalationDisabled:
    def test_no_op_when_escalation_pkg_unavailable(self, tmp_path, monkeypatch):
        """When HAS_ESCALATION is False the escalator silently no-ops."""
        monkeypatch.setattr(sve_mod, 'HAS_ESCALATION', False)
        esc = ScopeViolationEscalator()
        result = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='reify',
            candidate_title='whatever',
            matched_paths=('fused-memory/',),
            suggested_project='dark_factory',
        )
        assert result is None
        # No file is written.
        queue_dir = tmp_path / 'data' / 'escalations'
        assert not queue_dir.exists() or not list(queue_dir.glob('*.json'))
