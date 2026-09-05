"""Unit tests for fused_memory.middleware.recurring_gate_guard (task 3588).

Step 1 (RED → step-2 GREEN): the SUBMISSION-READER layer — the canonical
``gate_subject`` metadata key, its read-side aliases, and the
``is_gate_submission`` predicate that decides whether a ``submit_task``
payload is a human-gate carrier at all.

The module does not exist until step-2, so every test imports the names it
needs LOCALLY inside its body — the RED-collection idiom documented at
tests/test_operational_routing_guard.py:7-12 (same style in
test_execution_class_guard.py's TestInjectExecutionClass). That keeps this
file COLLECTABLE at RED: each test fails on its own in-body import, so the
count of pending assertions stays visible instead of the whole module
erroring out at collection time.
"""

from __future__ import annotations

import json


class TestGateSubjectKeyConstants:
    """The canonical subject key and its ordered read-side alias tuple."""

    def test_canonical_key_is_gate_subject(self):
        from fused_memory.middleware.recurring_gate_guard import GATE_SUBJECT_KEY

        assert GATE_SUBJECT_KEY == 'gate_subject'

    def test_aliases_are_ordered_with_canonical_first(self):
        from fused_memory.middleware.recurring_gate_guard import (
            GATE_SUBJECT_ALIASES,
            GATE_SUBJECT_KEY,
        )

        # Ordered sequence (not a set): resolution order is load-bearing.
        assert isinstance(GATE_SUBJECT_ALIASES, tuple)
        assert GATE_SUBJECT_ALIASES[0] == GATE_SUBJECT_KEY

    def test_aliases_cover_the_already_filed_carrier_spellings(self):
        from fused_memory.middleware.recurring_gate_guard import GATE_SUBJECT_ALIASES

        # 5902/5916/5929 key their subject via stranded_task_id; related_task_id
        # is the singular sibling spelling. History is never rewritten, so both
        # must be readable.
        assert 'stranded_task_id' in GATE_SUBJECT_ALIASES
        assert 'related_task_id' in GATE_SUBJECT_ALIASES


class TestExtractGateSubject:
    """extract_gate_subject(metadata) -> str | None."""

    def test_canonical_key_returns_str(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        result = extract_gate_subject({'gate_subject': '5879'})
        assert result == '5879'
        assert isinstance(result, str)

    def test_falls_back_to_stranded_task_id(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'stranded_task_id': '5879'}) == '5879'

    def test_falls_back_to_related_task_id(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'related_task_id': '5879'}) == '5879'

    def test_alias_order_stranded_beats_related(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert (
            extract_gate_subject(
                {'stranded_task_id': '5879', 'related_task_id': '5858'}
            )
            == '5879'
        )

    def test_canonical_wins_over_alias(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert (
            extract_gate_subject(
                {
                    'gate_subject': '5879',
                    'stranded_task_id': '5858',
                    'related_task_id': '5801',
                }
            )
            == '5879'
        )

    def test_int_scalar_is_coerced_to_str(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        result = extract_gate_subject({'gate_subject': 5879})
        assert result == '5879'
        assert isinstance(result, str)

    def test_surrounding_whitespace_is_stripped(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': '  5879\n'}) == '5879'

    def test_absent_key_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'execution_class': 'operational'}) is None

    def test_empty_string_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': ''}) is None

    def test_whitespace_only_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': '   '}) is None

    def test_none_value_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': None}) is None

    def test_non_scalar_values_return_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': {'id': '5879'}}) is None
        assert extract_gate_subject({'gate_subject': ['5879']}) is None

    def test_bool_is_not_accepted_as_a_scalar(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        # bool is an int subclass — True must not become the subject 'True'.
        assert extract_gate_subject({'gate_subject': True}) is None

    def test_json_string_metadata_blob_is_read(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject(json.dumps({'gate_subject': '5879'})) == '5879'

    def test_unparseable_string_metadata_returns_none_without_raising(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject('not json at all {{{') is None

    def test_none_metadata_returns_none_without_raising(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject(None) is None


class TestIsGateSubmission:
    """is_gate_submission(metadata) -> bool.

    | execution_class | operational_mode | is_gate_submission |
    |-----------------|------------------|--------------------|
    | operational     | 'gate'           | True               |
    | operational     | absent           | True  (default)    |
    | operational     | 'llm'            | False              |
    | code_tdd        | (any)            | False              |
    | decision        | (any)            | False              |
    | absent          | (any)            | False              |
    """

    def test_operational_with_explicit_gate_is_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(
                {'execution_class': 'operational', 'operational_mode': 'gate'}
            )
            is True
        )

    def test_operational_with_absent_mode_is_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        # TaskMetadata.operational_mode is Literal['gate','llm'] = 'gate', and
        # inject_operational_routing coerces operational+absent to a pure gate.
        assert is_gate_submission({'execution_class': 'operational'}) is True

    def test_operational_llm_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(
                {'execution_class': 'operational', 'operational_mode': 'llm'}
            )
            is False
        )

    def test_code_tdd_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(
                {'execution_class': 'code_tdd', 'operational_mode': 'gate'}
            )
            is False
        )

    def test_decision_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(
                {'execution_class': 'decision', 'operational_mode': 'gate'}
            )
            is False
        )

    def test_absent_execution_class_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert is_gate_submission({'operational_mode': 'gate'}) is False

    def test_truthy_non_matching_operational_mode_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        # Value-sensitive, mirroring TaskInterceptor._is_gate_metadata: a
        # truthy non-'gate' value must not satisfy the predicate.
        assert (
            is_gate_submission(
                {'execution_class': 'operational', 'operational_mode': 1}
            )
            is False
        )
        assert (
            is_gate_submission(
                {'execution_class': 'operational', 'operational_mode': 'false'}
            )
            is False
        )

    def test_none_metadata_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert is_gate_submission(None) is False

    def test_unparseable_metadata_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert is_gate_submission('not json at all {{{') is False

    def test_json_string_metadata_blob_is_read(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(json.dumps({'execution_class': 'operational'}))
            is True
        )
