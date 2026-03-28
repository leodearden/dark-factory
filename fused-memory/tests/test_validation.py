"""Unit tests for fused_memory.utils.validation helpers."""

import pytest

from fused_memory.utils.validation import (
    require_project_root,
    validate_project_id,
    validate_project_root,
    validate_run_id,
)


class TestValidateRunId:
    """validate_run_id returns None for safe identifiers, error dict for unsafe ones."""

    def test_empty_string_returns_error(self):
        result = validate_run_id('')
        assert result is not None
        assert 'error' in result
        assert 'error_type' in result
        assert result['error_type'] == 'ValidationError'

    def test_whitespace_only_returns_error(self):
        result = validate_run_id('   ')
        assert result is not None
        assert 'error' in result
        assert result['error_type'] == 'ValidationError'

    def test_newline_returns_error(self):
        result = validate_run_id('run\nid')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_backtick_returns_error(self):
        result = validate_run_id('run`id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_double_quote_returns_error(self):
        result = validate_run_id('run"id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_single_quote_returns_error(self):
        result = validate_run_id("run'id")
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_curly_brace_returns_error(self):
        result = validate_run_id('run{id}')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_semicolon_returns_error(self):
        result = validate_run_id('run;id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_dollar_sign_returns_error(self):
        result = validate_run_id('run$id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_space_in_middle_returns_error(self):
        result = validate_run_id('run id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_valid_uuid_format_returns_none(self):
        """Standard uuid4 hex-and-hyphens format is accepted."""
        result = validate_run_id('550e8400-e29b-41d4-a716-446655440000')
        assert result is None

    def test_plain_alphanumeric_returns_none(self):
        result = validate_run_id('testrun1')
        assert result is None

    def test_underscores_accepted(self):
        result = validate_run_id('test_run_1')
        assert result is None

    def test_hyphens_accepted(self):
        result = validate_run_id('test-run-1')
        assert result is None

    def test_mixed_alphanumeric_hyphen_underscore_returns_none(self):
        result = validate_run_id('Run_ID-42')
        assert result is None

    def test_returns_error_dict_not_raises(self):
        """validate_run_id must not raise — it returns an error dict."""
        try:
            result = validate_run_id('\x00malicious')
            assert result is not None
        except Exception as exc:
            pytest.fail(f'validate_run_id raised unexpectedly: {exc}')

    def test_trailing_newline_returns_error(self):
        """Trailing newline must be rejected — not silently accepted by $ anchor bypass.

        Python's re.match(r'^[a-zA-Z0-9_-]+$', 'valid-id\\n') returns a truthy match
        because $ matches just before a trailing newline. This test exposes that bypass;
        re.fullmatch() is required to catch it.
        """
        result = validate_run_id('valid-id\n')
        assert result is not None, (
            'validate_run_id accepted trailing newline — likely using .match() instead '
            'of .fullmatch(). Switch to re.fullmatch() to fix.'
        )
        assert result['error_type'] == 'ValidationError'

    def test_trailing_newline_after_uuid_returns_error(self):
        """A UUID followed by a trailing newline must be rejected."""
        result = validate_run_id('550e8400-e29b-41d4-a716-446655440000\n')
        assert result is not None, (
            'validate_run_id accepted UUID with trailing newline — likely using .match() '
            'instead of .fullmatch(). Switch to re.fullmatch() to fix.'
        )
        assert result['error_type'] == 'ValidationError'


class TestValidateProjectRoot:
    """validate_project_root returns None for valid absolute paths, error dict otherwise."""

    def test_rejects_whitespace_only(self):
        result = validate_project_root('   ')
        assert result is not None
        assert 'error' in result
        assert 'error_type' in result
        assert result['error_type'] == 'ValidationError'

    def test_rejects_tab_only(self):
        result = validate_project_root('\t')
        assert result is not None
        assert 'error' in result
        assert 'error_type' in result
        assert result['error_type'] == 'ValidationError'


class TestRequireProjectRoot:
    """require_project_root raises ValueError for invalid paths, returns None for valid ones."""

    def test_valid_absolute_path_raises_nothing(self):
        require_project_root('/some/path')  # must not raise

    def test_invalid_path_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_project_root('relative/path')

    def test_error_message_matches_validate_error_field(self):
        invalid = 'relative/path'
        err_dict = validate_project_root(invalid)
        assert err_dict is not None
        with pytest.raises(ValueError) as exc_info:
            require_project_root(invalid)
        assert str(exc_info.value) == err_dict['error']
