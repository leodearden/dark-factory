"""Unit tests for fused_memory.middleware.operational_suggestion_guard.

Step 5 (RED -> step-6 GREEN): detection matrix for operational_suggestion_finding
plus the operational_suggestion_warning payload shape.

Modeled on test_routing_intent_guard.py, but this guard has no enforce/reject
path (WARN-ONLY, task mandate) — operational_suggestion_finding is imported at
module top; operational_suggestion_warning is imported locally in its own
tests since it is exercised alongside the finding fixture below.
"""

from __future__ import annotations

import logging

import pytest

from fused_memory.middleware.operational_suggestion_guard import (
    operational_suggestion_finding,
)


class TestOperationalSuggestionFindingDetectionMatrix:
    """Implicit operational-phrasing lint matrix (task_kind='normal' only)."""

    def test_operational_phrasing_in_description_with_no_files_finds_and_names_markers(
        self,
    ):
        """A normal task's DESCRIPTION declaring restart+confirm phrasing with
        no metadata.files -> finding carries both matched marker labels."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers
        assert 'confirm' in finding.markers

    def test_non_normal_task_kind_is_none(self):
        """task_kind='deterministic' carrying the same phrasing -> None; only
        task_kind='normal' submissions are suggested."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='deterministic',
            metadata=None,
        )
        assert finding is None

    def test_execution_class_operational_is_exempt(self):
        """metadata.execution_class='operational' + markers -> None (an
        honest non-code declaration is not a mismatch to flag)."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'operational'},
        )
        assert finding is None

    def test_execution_class_decision_is_exempt(self):
        """metadata.execution_class='decision' + markers -> None."""
        finding = operational_suggestion_finding(
            title=None,
            description='Redeploy the orchestrator fleet and confirm health.',
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'decision'},
        )
        assert finding is None

    def test_metadata_files_present_is_exempt(self):
        """metadata.files is non-empty (a code deliverable) -> None even
        though the description carries operational phrasing."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata={'files': ['fused-memory/src/fused_memory/server/tools.py']},
        )
        assert finding is None

    @pytest.mark.parametrize('signal_word', ['fix', 'bug', 'crash', 'implement'])
    def test_code_change_signal_suppresses_finding(self, signal_word):
        """A code-change signal (fix/bug/crash/implement) anywhere in the
        combined text suppresses the finding -- a genuine code task that
        merely mentions 'restart' is not nudged."""
        finding = operational_suggestion_finding(
            title=f'{signal_word} the ingestion pipeline',
            description='Restart the service once the change lands.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    def test_clean_normal_task_is_none(self):
        """A clean normal task with no operational phrasing anywhere -> None."""
        finding = operational_suggestion_finding(
            title='Add a retry helper to the sync client',
            description='Wrap the sync client call in a bounded retry loop.',
            details='Use the existing backoff helper in shared.retry.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    @pytest.mark.parametrize(
        ('field_text', 'expected_marker'),
        [
            ('Restart the fused-memory service.', 'restart'),
            ('Redeploy the orchestrator fleet.', 'redeploy'),
            ('Reload the configuration file.', 'reload'),
            ('Confirm the service is healthy.', 'confirm'),
            ('Deploy the latest build to production.', 'deploy'),
            ('Run systemctl status to verify.', 'systemctl'),
        ],
    )
    def test_each_marker_fires_independently_in_description(
        self, field_text, expected_marker
    ):
        """Each of the six operational markers independently produces a
        finding naming itself when present in an otherwise-clean description
        with no metadata.files."""
        finding = operational_suggestion_finding(
            title=None,
            description=field_text,
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert expected_marker in finding.markers

    def test_marker_in_details_only_finds_independently(self):
        """A marker present ONLY in details (title/description clean) ->
        finding, and the finding records 'details' as a matched field."""
        finding = operational_suggestion_finding(
            title='',
            description='',
            details='Restart the service once the escalation resolves.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers
        assert 'details' in finding.fields

    @pytest.mark.parametrize(
        'description',
        [
            'Confirm the retry logic handles timeouts.',
            'Reload the config module after edit.',
            'Deploy config schema field.',
        ],
    )
    def test_weak_marker_suppressed_by_code_artifact_in_same_field(self, description):
        """The generic ('weak') markers -- confirm/reload/deploy -- are
        suppressed when the SAME field also names a code-level artifact
        (logic/module/schema/field/...) rather than a running system: these
        are genuine code-task sentences, not operational asks (task 2679
        amendment pass, reviewer_comprehensive robustness finding)."""
        finding = operational_suggestion_finding(
            title=None,
            description=description,
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    def test_strong_marker_still_fires_alongside_code_artifact_noun(self):
        """Unlike the weak markers, 'restart' (a specific/strong marker) is
        NOT gated by a co-occurring code-artifact noun in the same field --
        it still fires standalone."""
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the service; the auth module needs a bounce.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'restart' in finding.markers

    def test_cross_field_markers_aggregate_distinct_labels_and_fields(self):
        """A marker in TITLE and a different marker in DETAILS -> both
        marker labels appear (deduped) and both field names appear in
        `fields` (deduped), in first-matched order."""
        finding = operational_suggestion_finding(
            title='Restart the ingestion worker',
            description='',
            details='Confirm the worker is healthy afterwards.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert finding.markers == ('restart', 'confirm')
        assert finding.fields == ('title', 'details')

    def test_same_marker_in_two_fields_dedupes_to_one_label(self):
        """The same marker present in both TITLE and DESCRIPTION dedupes to
        a single entry in `markers`, but BOTH field names appear in
        `fields`."""
        finding = operational_suggestion_finding(
            title='Restart the ingestion worker',
            description='Restart the ingestion worker again to be sure.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert finding.markers == ('restart',)
        assert finding.fields == ('title', 'description')

    def test_code_change_signal_in_details_only_suppresses_finding(self):
        """A code-change signal located ONLY in `details` (not title/
        description) still suppresses the finding -- the code-change check
        scans the COMBINED title+description+details text, not just
        title."""
        finding = operational_suggestion_finding(
            title='Ops task',
            description='Restart the fused-memory service.',
            details='This closes out the bug found during rollout.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is None


# ---------------------------------------------------------------------------
# Payload shape for the warn outcome (WARN-ONLY -- no reject/enforce path)
#
# operational_suggestion_warning does not exist until step-6 -- imported
# locally (inside the test body) rather than at module level, so only these
# tests fail at RED while TestOperationalSuggestionFindingDetectionMatrix
# above keeps collecting (its own import fails too until step-6, since the
# whole module does not exist yet -- both classes are RED simultaneously
# pre-step-6, GREEN together after).
# ---------------------------------------------------------------------------


class TestOperationalSuggestionWarningPayload:
    """Payload shape for the non-blocking warn outcome."""

    def _finding(self):
        finding = operational_suggestion_finding(
            title=None,
            description='Restart the fused-memory service and confirm it is back up.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        return finding

    def test_operational_suggestion_warning_is_non_blocking_structured_payload(
        self, caplog
    ):
        """operational_suggestion_warning(finding) ->
        {'operational_suggestion_warning': {...}} with markers/fields/detail/
        hint, and NO top-level 'error'/'error_type' (it must be
        non-blocking). Also emits a greppable
        'operational_task_suggestion.flagged' census WARNING."""
        from fused_memory.middleware.operational_suggestion_guard import (
            operational_suggestion_warning,
        )

        finding = self._finding()
        with caplog.at_level(logging.WARNING):
            result = operational_suggestion_warning(finding)

        assert 'error' not in result
        assert 'error_type' not in result
        assert 'operational_suggestion_warning' in result
        payload = result['operational_suggestion_warning']
        assert list(payload['markers']) == list(finding.markers)
        assert list(payload['fields']) == list(finding.fields)
        assert 'detail' in payload
        assert 'hint' in payload
        assert any(
            'operational_task_suggestion.flagged' in rec.message
            for rec in caplog.records
        )

    def test_operational_suggestion_warning_hint_names_deterministic_or_execution_class(
        self,
    ):
        """The hint must suggest either task_kind='deterministic' or
        metadata.execution_class as the fix -- non-blocking guidance, never
        a coercion."""
        from fused_memory.middleware.operational_suggestion_guard import (
            operational_suggestion_warning,
        )

        finding = self._finding()
        result = operational_suggestion_warning(finding)
        hint = result['operational_suggestion_warning']['hint']
        assert 'deterministic' in hint
        assert 'execution_class' in hint
