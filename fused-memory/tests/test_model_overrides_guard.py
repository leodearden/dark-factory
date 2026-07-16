"""Unit tests for fused_memory.middleware.model_overrides_guard.

Step 5 (RED -> step-6 GREEN): model_overrides_error shape-validation matrix
(PRD adaptive-model-routing task ζ, decision 9). Mirrors
test_deterministic_task_guard.py's style.
"""

from __future__ import annotations

import json

from fused_memory.middleware.model_overrides_guard import model_overrides_error

# ---------------------------------------------------------------------------
# Acceptance: absent / empty / well-formed model_overrides
# ---------------------------------------------------------------------------


class TestModelOverridesErrorAcceptance:
    """model_overrides_error returns None when there is nothing to reject."""

    def test_none_metadata_accepts(self):
        """metadata=None -> accept (no model_overrides to validate)."""
        assert model_overrides_error(None) is None

    def test_empty_dict_metadata_accepts(self):
        """metadata={} -> accept."""
        assert model_overrides_error({}) is None

    def test_metadata_without_model_overrides_key_accepts(self):
        """metadata with unrelated keys and no model_overrides -> accept."""
        assert model_overrides_error({'task_kind': 'normal'}) is None

    def test_empty_model_overrides_accepts(self):
        """metadata.model_overrides={} (empty) -> accept."""
        assert model_overrides_error({'model_overrides': {}}) is None

    def test_well_formed_model_overrides_accepts(self):
        """A single well-formed role -> model pin -> accept."""
        result = model_overrides_error({'model_overrides': {'implementer': 'haiku'}})
        assert result is None

    def test_well_formed_multi_role_model_overrides_accepts(self):
        """Multiple well-formed role -> model pins -> accept."""
        result = model_overrides_error(
            {'model_overrides': {'implementer': 'haiku', 'reviewer_comprehensive': 'opus'}}
        )
        assert result is None

    def test_empty_string_metadata_accepts(self):
        """metadata='' (empty string, benign-absent) -> accept."""
        assert model_overrides_error('') is None

    def test_well_formed_model_overrides_as_json_string_accepts(self):
        """metadata supplied as a JSON string with a well-formed override -> accept."""
        meta_str = json.dumps({'model_overrides': {'implementer': 'haiku'}})
        assert model_overrides_error(meta_str) is None


# ---------------------------------------------------------------------------
# Rejection: malformed model_overrides
# ---------------------------------------------------------------------------


class TestModelOverridesErrorRejection:
    """model_overrides_error returns a structured ValidationError dict."""

    def test_unknown_role_rejects(self):
        """An unrecognised role name is rejected; the message names it."""
        result = model_overrides_error({'model_overrides': {'bogus_role': 'haiku'}})
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'bogus_role' in result['error']

    def test_non_string_value_rejects(self):
        """A non-string model value (e.g. int) is rejected."""
        result = model_overrides_error({'model_overrides': {'implementer': 123}})
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_non_dict_model_overrides_list_rejects(self):
        """model_overrides as a list (not a dict) is rejected."""
        result = model_overrides_error({'model_overrides': ['implementer']})
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_non_dict_model_overrides_bare_string_rejects(self):
        """model_overrides as a bare string (not a dict) is rejected."""
        result = model_overrides_error({'model_overrides': 'implementer'})
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_malformed_model_overrides_via_json_string_metadata_rejects(self):
        """A malformed override supplied via a JSON-string metadata is also rejected."""
        meta_str = json.dumps({'model_overrides': {'bogus_role': 'haiku'}})
        result = model_overrides_error(meta_str)
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_rejection_includes_hint(self):
        """Rejected results always include an actionable 'hint' string."""
        result = model_overrides_error({'model_overrides': {'bogus_role': 'haiku'}})
        assert result is not None
        assert 'hint' in result
        assert isinstance(result['hint'], str)

    def test_one_bad_role_among_many_still_rejects(self):
        """A mix of valid and invalid roles is rejected (fail on the first bad one)."""
        result = model_overrides_error(
            {'model_overrides': {'implementer': 'haiku', 'bogus_role': 'sonnet'}}
        )
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
