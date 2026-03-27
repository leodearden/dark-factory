"""Unit tests for fused_memory.utils.validation."""

from __future__ import annotations

from fused_memory.utils.validation import validate_project_id


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
