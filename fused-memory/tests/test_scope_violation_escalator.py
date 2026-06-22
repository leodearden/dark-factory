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


    def test_llm_reason_appended_to_detail(self, tmp_path):
        """report_rejection with llm_reason= includes the reason in the escalation detail."""
        esc = ScopeViolationEscalator()
        llm_reason = 'LLM: genuine misroute — task edits orchestrator/harness.py'
        esc_id = esc.report_rejection(
            project_root=str(tmp_path),
            project_id='dark_factory',
            candidate_title='Fix orchestrator bug',
            matched_paths=('orchestrator/',),
            suggested_project='orchestrator',
            llm_reason=llm_reason,
        )
        assert esc_id is not None
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('*.json'))
        assert len(files) == 1
        import json
        payload = json.loads(files[0].read_text())
        assert 'llm_adjudicator_reason' in payload['detail']
        assert llm_reason in payload['detail']

    def test_no_llm_reason_keeps_detail_clean(self, tmp_path):
        """Back-compat: without llm_reason the detail does NOT contain llm_adjudicator_reason."""
        esc = ScopeViolationEscalator()
        esc.report_rejection(
            project_root=str(tmp_path),
            project_id='dark_factory',
            candidate_title='Fix orchestrator bug',
            matched_paths=('orchestrator/',),
            suggested_project='orchestrator',
        )
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('*.json'))
        import json
        payload = json.loads(files[0].read_text())
        assert 'llm_adjudicator_reason' not in payload['detail']


@pytest.mark.skipif(
    not sve_mod.HAS_ESCALATION,
    reason='escalation package not installed in this environment',
)
class TestBudgetMisconfigEscalation:
    """ScopeViolationEscalator.report_budget_misconfig() — loud config-defect signal."""

    def test_writes_distinct_escalation_file(self, tmp_path):
        """report_budget_misconfig writes one file with category='adjudicator_config_defect'."""
        esc = ScopeViolationEscalator()
        esc_id = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.11,
            turns=2,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert esc_id is not None
        queue_dir = tmp_path / 'data' / 'escalations'
        files = list(queue_dir.glob('*.json'))
        assert len(files) == 1, f'expected one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == esc_id
        # DISTINCT category — not scope_violation.
        assert payload['category'] == 'adjudicator_config_defect'
        assert payload['severity'] == 'blocking'
        assert payload['agent_role'] == 'fused-memory/path-scope-adjudicator'
        # Summary and detail name the budget misconfiguration.
        assert 'budget' in payload['summary'].lower() or 'misconfig' in payload['summary'].lower()
        assert str(0.11) in payload['detail'] or '0.11' in payload['detail']
        assert str(0.30) in payload['detail'] or '0.30' in payload['detail'] or '0.3' in payload['detail']
        assert 'sonnet' in payload['detail']
        # Fix hint present.
        assert 'max_budget_usd' in payload['detail']

    def test_dedup_second_call_within_window_writes_no_file(self, tmp_path):
        """Two consecutive calls for same project_id write exactly ONE file (burst-suppress)."""
        esc = ScopeViolationEscalator(budget_misconfig_dedup_window_secs=9999.0)
        first = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.11,
            turns=2,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert first is not None
        second = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.12,
            turns=3,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert second is None
        files = list((tmp_path / 'data' / 'escalations').glob('*.json'))
        assert len(files) == 1, 'second call within dedup window must not write a new file'

    def test_queue_submit_failure_returns_none_does_not_raise(self, tmp_path, monkeypatch):
        """A queue.submit failure must be swallowed — escalation is additive."""
        esc = ScopeViolationEscalator()
        from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

        def boom(self, _esc):
            raise RuntimeError('disk full')

        monkeypatch.setattr(EscalationQueue, 'submit', boom)
        result = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.11,
            turns=2,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert result is None  # no exception propagates

    def test_escalation_detail_includes_turns_and_fix_hint(self, tmp_path):
        """Detail must carry turns= and a concrete fix hint."""
        esc = ScopeViolationEscalator()
        esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.05,
            turns=4,
            max_budget_usd=0.10,
            model='claude-3-5-haiku-20241022',
        )
        files = list((tmp_path / 'data' / 'escalations').glob('*.json'))
        payload = json.loads(files[0].read_text())
        assert 'turns' in payload['detail']
        # Fix hint names the config key.
        assert 'max_budget_usd' in payload['detail']

    def test_dedup_window_expired_second_call_files_new_escalation(self, tmp_path):
        """Once the dedup window has elapsed a second call MUST file a second file.

        A regression that made dedup permanent (e.g. comparing against a sentinel
        that never ages out) would pass a suppression-only test but break the
        'one reminder per window' contract.  Using window=0.0 means the timestamp
        is always stale by definition, so the second call always re-files.
        """
        esc = ScopeViolationEscalator(budget_misconfig_dedup_window_secs=0.0)
        first = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.11,
            turns=2,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert first is not None
        second = esc.report_budget_misconfig(
            project_root=str(tmp_path),
            project_id='dark_factory',
            cost_usd=0.12,
            turns=3,
            max_budget_usd=0.30,
            model='sonnet',
        )
        assert second is not None, (
            'second call after dedup window expired must file a new escalation'
        )
        assert second != first, 'second escalation must have a distinct id'
        files = list((tmp_path / 'data' / 'escalations').glob('*.json'))
        assert len(files) == 2, (
            f'expected two escalation files after window expired, found: {files}'
        )


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
