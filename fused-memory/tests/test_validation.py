"""Unit tests for fused_memory.utils.validation."""

from __future__ import annotations

from fused_memory.utils.validation import validate_project_id, validate_project_root


class TestValidateProjectId:
    """Unit tests for validate_project_id character-set validation."""

    # --- Invalid: characters that could inject into prompts ---

    def test_rejects_newline_in_project_id(self):
        """Newline in project_id would allow arbitrary prompt injection."""
        result = validate_project_id('bad\nproject')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'project_id' in result.get('error', '').lower()

    def test_rejects_space_in_project_id(self):
        """Space in project_id breaks key: value parsing and may confuse the LLM."""
        result = validate_project_id('my project')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_rejects_backtick_in_project_id(self):
        """Backtick would inject markdown code-span syntax into the prompt."""
        result = validate_project_id('back`tick')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_rejects_unicode_in_project_id(self):
        """Non-ASCII characters are outside the allowed character set."""
        result = validate_project_id('unicode_日本語')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_rejects_double_quote_in_project_id(self):
        """Double quotes could break out of quoted contexts in prompts."""
        result = validate_project_id('bad"project')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_rejects_carriage_return_in_project_id(self):
        """Carriage return is a control character that can corrupt prompts."""
        result = validate_project_id('bad\rproject')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    # --- Valid: alphanumeric + underscore + hyphen ---

    def test_accepts_simple_alphanumeric(self):
        """Plain alphanumeric project IDs should pass."""
        assert validate_project_id('darkfactory') is None

    def test_accepts_underscore_separated(self):
        """Underscore-separated names are valid (resolve_project_id produces these)."""
        assert validate_project_id('dark_factory') is None

    def test_accepts_hyphen_separated(self):
        """Hyphens are allowed for raw project names that haven't been normalised."""
        assert validate_project_id('my-project') is None

    def test_accepts_mixed_case(self):
        """Mixed-case project IDs are valid."""
        assert validate_project_id('MyProject123') is None

    def test_accepts_digits_only(self):
        """Purely numeric IDs (unlikely but valid per character set)."""
        assert validate_project_id('12345') is None

    def test_accepts_underscore_and_hyphen_combined(self):
        """Combining underscores and hyphens is valid."""
        assert validate_project_id('my_project-v2') is None

    # --- Existing empty/whitespace checks still work ---

    def test_rejects_empty_string(self):
        """Empty string must still return an error with distinct message."""
        result = validate_project_id('')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'non-empty' in result.get('error', '')

    def test_rejects_whitespace_only(self):
        """Whitespace-only strings must still return an error."""
        result = validate_project_id('   ')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'non-empty' in result.get('error', '')

    def test_rejects_tab_only(self):
        """Tab-only string is effectively empty."""
        result = validate_project_id('\t')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'


class TestValidateProjectRoot:
    """Unit tests for validate_project_root path validation."""

    # --- Valid: non-empty absolute paths ---

    def test_accepts_root_slash(self):
        """The filesystem root is a valid absolute path."""
        assert validate_project_root('/') is None

    def test_accepts_absolute_path(self):
        """A typical absolute path should pass without error."""
        assert validate_project_root('/home/user/myproject') is None

    def test_accepts_deep_absolute_path(self):
        """Deep nested absolute paths are valid."""
        assert validate_project_root('/opt/workspace/dark-factory/fused-memory') is None

    # --- Invalid: empty or relative paths ---

    def test_rejects_empty_string(self):
        """Empty string must return an error dict."""
        result = validate_project_root('')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'project_root' in result.get('error', '').lower()

    def test_rejects_relative_path(self):
        """Relative paths are not valid project roots."""
        result = validate_project_root('relative/path')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'
        assert 'project_root' in result.get('error', '').lower()

    def test_rejects_dot_relative_path(self):
        """Dot-prefixed relative paths are not valid project roots."""
        result = validate_project_root('./local/path')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    def test_rejects_parent_relative_path(self):
        """Double-dot relative paths are not valid project roots."""
        result = validate_project_root('../parent/path')
        assert result is not None
        assert result.get('error_type') == 'ValidationError'

    # --- Error dict shape ---

    def test_error_dict_has_required_keys(self):
        """Error dict must contain both 'error' and 'error_type' keys."""
        result = validate_project_root('not/absolute')
        assert result is not None
        assert 'error' in result
        assert 'error_type' in result
        assert result['error_type'] == 'ValidationError'

    def test_error_message_includes_bad_value(self):
        """Error message should include the bad value for diagnostics."""
        bad_value = 'my/relative/path'
        result = validate_project_root(bad_value)
        assert result is not None
        assert bad_value in result.get('error', '')


class TestReturnContractCompatibility:
    """Integration tests confirming both validators share the same return contract.

    tools.py callers use the pattern ``if err := _validate_...: return err``.
    Both canonical validators must return the exact same dict shape so that
    tools.py handlers can pass the result directly to the MCP client.
    """

    def test_validate_project_id_error_dict_shape(self):
        """validate_project_id must return {'error': str, 'error_type': 'ValidationError'}."""
        result = validate_project_id('bad project id!')
        assert isinstance(result, dict)
        assert set(result.keys()) == {'error', 'error_type'}
        assert result['error_type'] == 'ValidationError'
        assert isinstance(result['error'], str)
        assert len(result['error']) > 0

    def test_validate_project_root_error_dict_shape(self):
        """validate_project_root must return {'error': str, 'error_type': 'ValidationError'}."""
        result = validate_project_root('relative/path')
        assert isinstance(result, dict)
        assert set(result.keys()) == {'error', 'error_type'}
        assert result['error_type'] == 'ValidationError'
        assert isinstance(result['error'], str)
        assert len(result['error']) > 0

    def test_both_return_none_for_valid_inputs(self):
        """Both validators return None (not empty dict or False) for valid inputs."""
        assert validate_project_id('dark_factory') is None
        assert validate_project_root('/home/user/project') is None

    def test_error_type_value_is_exactly_validation_error(self):
        """error_type must be exactly 'ValidationError' — tools.py callers check this key."""
        pid_err = validate_project_id('bad id')
        root_err = validate_project_root('relative')
        assert pid_err is not None
        assert root_err is not None
        assert pid_err['error_type'] == 'ValidationError'
        assert root_err['error_type'] == 'ValidationError'
