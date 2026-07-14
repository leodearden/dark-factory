"""Unit tests for fused_memory.middleware.routing_intent_guard.

Step 1 (RED -> step-2 GREEN): detection matrix for routing_intent_finding.
Step 3 (RED -> step-4 GREEN): routing_intent_reject / routing_intent_warning
payload shapes and the routing_intent_enforced() env flag.
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.routing_intent_guard import routing_intent_finding


class TestRoutingIntentFindingDetectionMatrix:
    """Declaration-only routing-intent lint matrix (task_kind='normal' only)."""

    def test_declarative_marker_in_description_finds_and_names_markers(self):
        """A normal task's DESCRIPTION declaring 'DO NOT IMPLEMENT ...
        escalate to a human instead of implementing' -> finding carries
        both matched marker labels."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'do_not_implement' in finding.markers
        assert 'escalate_instead_of_implementing' in finding.markers

    def test_mined_task_2332_shape_do_not_author_plan(self):
        """The mined task-2332 shape ('do not attempt to author a TDD plan
        for this task') -> finding."""
        finding = routing_intent_finding(
            title=None,
            description='Do not attempt to author a TDD plan for this task.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'do_not_author_plan' in finding.markers

    def test_marker_in_title_only_finds_independently(self):
        """A marker present ONLY in title (description/details clean) ->
        finding, and the finding records 'title' as a matched field."""
        finding = routing_intent_finding(
            title='DO NOT IMPLEMENT this task',
            description='',
            details='',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'do_not_implement' in finding.markers
        assert 'title' in finding.fields

    def test_marker_in_details_only_finds_independently(self):
        """A marker present ONLY in details (title/description clean) ->
        finding, and the finding records 'details' as a matched field."""
        finding = routing_intent_finding(
            title='',
            description='',
            details='Escalate to a human instead of implementing this work.',
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert 'escalate_instead_of_implementing' in finding.markers
        assert 'details' in finding.fields

    def test_task_2408_passing_mention_is_not_a_false_positive(self):
        """A genuine code task whose prose merely MENTIONS 'deterministic
        pure-gate' in passing (task-2408 shape) -> None, not a finding."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'This normal code task refactors the deterministic '
                'pure-gate helper; implement the fix in X.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    def test_code_change_signal_suppresses_a_marker_match_across_fields(self):
        """A code-change signal ('Fix') in title suppresses a marker match
        found in description -- proves _CODE_CHANGE_SIGNALS is wired in as
        a blanket, cross-field guard (mirrors operational_ask_registry's
        title-signal-suppresses-any-match precedent), not dead code."""
        finding = routing_intent_finding(
            title='Fix the ingestion pipeline',
            description='DO NOT IMPLEMENT until design review completes.',
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is None

    def test_deterministic_task_kind_is_never_linted(self):
        """task_kind='deterministic' carrying the same markers -> None; only
        task_kind='normal' submissions are linted."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='deterministic',
            metadata=None,
        )
        assert finding is None

    def test_execution_class_operational_is_exempt(self):
        """metadata.execution_class='operational' + markers -> None (an
        honest non-code declaration is not a mismatch to flag)."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'operational'},
        )
        assert finding is None

    def test_execution_class_decision_is_exempt(self):
        """metadata.execution_class='decision' + markers -> None."""
        finding = routing_intent_finding(
            title=None,
            description='Do not attempt to author a TDD plan for this task.',
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'decision'},
        )
        assert finding is None

    def test_execution_class_code_tdd_is_not_exempt(self):
        """metadata.execution_class='code_tdd' + markers -> STILL a finding
        (code_tdd is the normal-code-task class; it does not exempt a
        conflicting routing declaration)."""
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='normal',
            metadata={'execution_class': 'code_tdd'},
        )
        assert finding is not None

    def test_clean_normal_task_is_none(self):
        """A clean normal task with no markers anywhere -> None."""
        finding = routing_intent_finding(
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
            ('DO NOT IMPLEMENT this task.', 'do_not_implement'),
            (
                'Do not attempt to author a TDD plan for this task.',
                'do_not_author_plan',
            ),
            ('This is a no-code task.', 'no_code_label'),
            ('This is deterministic; no worktree required.', 'no_worktree'),
            (
                'Escalate to a human instead of implementing.',
                'escalate_instead_of_implementing',
            ),
            (
                'Run --apply against the live store once reviewed.',
                'apply_live_store',
            ),
            ('DO NOT COMPLETE until sign-off.', 'do_not_complete'),
        ],
    )
    def test_each_marker_fires_independently_in_description(
        self, field_text, expected_marker
    ):
        """Each of the seven declarative markers independently produces a
        finding naming itself when present in an otherwise-clean description."""
        finding = routing_intent_finding(
            title=None,
            description=field_text,
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        assert expected_marker in finding.markers


# ---------------------------------------------------------------------------
# Step-3: reject/warning payload shapes + the routing_intent_enforced() flag
#
# routing_intent_reject, routing_intent_warning, and routing_intent_enforced
# do not exist until step-4 -- each test below imports locally (inside the
# test body) rather than at module level, so only these new tests fail at
# RED while TestRoutingIntentFindingDetectionMatrix above keeps collecting
# and passing.
# ---------------------------------------------------------------------------


class TestRoutingIntentPayloadsAndFlag:
    """Payload shapes for the warn/reject outcomes, plus the enforce flag."""

    def _finding(self):
        finding = routing_intent_finding(
            title=None,
            description=(
                'DO NOT IMPLEMENT this; escalate to a human instead of '
                'implementing.'
            ),
            details=None,
            task_kind='normal',
            metadata=None,
        )
        assert finding is not None
        return finding

    def test_routing_intent_reject_returns_validation_error_naming_markers(self):
        """routing_intent_reject(finding) -> a structured ValidationError
        dict naming the matched marker(s), with a hint key."""
        from fused_memory.middleware.routing_intent_guard import routing_intent_reject

        finding = self._finding()
        result = routing_intent_reject(finding)
        assert result.get('error_type') == 'ValidationError'
        assert 'hint' in result
        for marker in finding.markers:
            assert marker in result['error']

    def test_routing_intent_warning_is_non_blocking_structured_payload(self, caplog):
        """routing_intent_warning(finding) -> {'routing_intent_warning': {...}}
        with markers + hint, and NO top-level 'error'/'error_type' (it must
        be non-blocking). Also emits a greppable 'routing_intent_lint.flagged'
        census WARNING -- the observe-rate-before-flipping-enforcement
        precedent."""
        import logging

        from fused_memory.middleware.routing_intent_guard import routing_intent_warning

        finding = self._finding()
        with caplog.at_level(logging.WARNING):
            result = routing_intent_warning(finding)

        assert 'error' not in result
        assert 'error_type' not in result
        assert 'routing_intent_warning' in result
        payload = result['routing_intent_warning']
        assert list(payload['markers']) == list(finding.markers)
        assert 'hint' in payload
        assert any(
            'routing_intent_lint.flagged' in rec.message for rec in caplog.records
        )

    def test_routing_intent_enforced_defaults_false_when_unset(self, monkeypatch):
        """routing_intent_enforced() -> False when FUSED_ROUTING_INTENT_ENFORCE
        is unset (the default, warn-only behavior)."""
        from fused_memory.middleware.routing_intent_guard import routing_intent_enforced

        monkeypatch.delenv('FUSED_ROUTING_INTENT_ENFORCE', raising=False)
        assert routing_intent_enforced() is False

    def test_routing_intent_enforced_true_when_env_set_truthy(self, monkeypatch):
        """routing_intent_enforced() -> True when FUSED_ROUTING_INTENT_ENFORCE
        is set to a truthy value ('1')."""
        from fused_memory.middleware.routing_intent_guard import routing_intent_enforced

        monkeypatch.setenv('FUSED_ROUTING_INTENT_ENFORCE', '1')
        assert routing_intent_enforced() is True
