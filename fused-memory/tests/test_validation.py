"""Tests for shared validation utilities in fused_memory.utils.validation."""

import pytest

from fused_memory.utils.validation import (
    require_project_root,
    validate_project_id,
    validate_project_root,
)


class TestValidateProjectRoot:
    """Tests for validate_project_root (returns dict | None)."""

    def test_empty_string_returns_error_dict(self):
        result = validate_project_root('')
        assert result is not None
        assert 'error' in result
        assert result['error_type'] == 'ValidationError'

    def test_relative_path_returns_error_dict(self):
        result = validate_project_root('relative/path')
        assert result is not None
        assert 'error' in result
        assert result['error_type'] == 'ValidationError'

    def test_bare_filename_returns_error_dict(self):
        result = validate_project_root('myproject')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_valid_absolute_path_returns_none(self):
        result = validate_project_root('/home/user/project')
        assert result is None

    def test_valid_absolute_path_with_trailing_slash_returns_none(self):
        result = validate_project_root('/home/user/project/')
        assert result is None

    def test_tmp_path_returns_none(self):
        result = validate_project_root('/tmp/test-project')
        assert result is None

    def test_error_dict_contains_the_bad_value(self):
        result = validate_project_root('bad/path')
        assert result is not None
        assert 'bad/path' in result['error']


class TestValidateProjectId:
    """Tests for validate_project_id (returns dict | None)."""

    def test_empty_string_returns_error_dict(self):
        result = validate_project_id('')
        assert result is not None
        assert 'error' in result
        assert result['error_type'] == 'ValidationError'

    def test_valid_id_returns_none(self):
        result = validate_project_id('dark_factory')
        assert result is None

    def test_any_nonempty_string_returns_none(self):
        result = validate_project_id('x')
        assert result is None


class TestRequireProjectRoot:
    """Tests for require_project_root (raises ValueError)."""

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            require_project_root('')

    def test_relative_path_raises_value_error(self):
        with pytest.raises(ValueError):
            require_project_root('relative/path')

    def test_bare_filename_raises_value_error(self):
        with pytest.raises(ValueError):
            require_project_root('myproject')

    def test_valid_absolute_path_returns_none(self):
        result = require_project_root('/home/user/project')
        assert result is None

    def test_tmp_path_returns_none(self):
        result = require_project_root('/tmp/test-project')
        assert result is None

    def test_error_message_mentions_bad_value(self):
        with pytest.raises(ValueError, match='bad/path'):
            require_project_root('bad/path')
