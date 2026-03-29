"""Unit tests for fused_memory.utils.validation helpers."""

import pytest

from fused_memory.utils.validation import (
    _validate_identifier,
    require_project_id,
    require_project_root,
    require_run_id,
    validate_project_id,
    validate_project_root,
    validate_run_id,
)


class TestValidateIdentifierHelper:
    """_validate_identifier(value, field_name) — private shared helper."""

    def test_empty_string_returns_error_with_field_name(self):
        result = _validate_identifier('', 'my_field')
        assert result is not None
        assert 'error' in result
        assert 'error_type' in result
        assert 'my_field' in result['error']

    def test_whitespace_only_returns_error(self):
        result = _validate_identifier('   ', 'my_field')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_injection_newline_returns_error_with_field_name(self):
        result = _validate_identifier('bad\nvalue', 'some_field')
        assert result is not None
        assert result['error_type'] == 'ValidationError'
        assert 'some_field' in result['error']

    def test_valid_identifier_returns_none(self):
        result = _validate_identifier('valid-id_123', 'project_id')
        assert result is None

    def test_error_dict_has_exactly_two_keys(self):
        result = _validate_identifier('', 'field')
        assert result is not None
        assert set(result.keys()) == {'error', 'error_type'}

    def test_character_rejection_error_mentions_field_name(self):
        result = _validate_identifier('bad`value', 'run_id')
        assert result is not None
        assert 'run_id' in result['error']

    def test_trailing_newline_rejected_fullmatch(self):
        """fullmatch guarantee: trailing newline must be rejected."""
        result = _validate_identifier('valid-id\n', 'project_id')
        assert result is not None, (
            '_validate_identifier accepted trailing newline — must use fullmatch'
        )
        assert result['error_type'] == 'ValidationError'


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

    def test_valid_absolute_path_returns_none(self):
        result = validate_project_root('/some/path')
        assert result is None

    def test_rejects_relative_path(self):
        result = validate_project_root('relative/path')
        assert result is not None
        assert 'error' in result
        assert 'error_type' in result
        assert result['error_type'] == 'ValidationError'

    def test_empty_string_returns_error(self):
        result = validate_project_root('')
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

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_project_root('')

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_project_root('   ')


class TestRequireProjectId:
    """require_project_id raises ValueError for invalid ids, returns None for valid ones."""

    def test_valid_id_raises_nothing(self):
        require_project_id('dark_factory')  # must not raise

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_project_id('')

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_project_id('   ')

    def test_error_message_matches_validate_error_field(self):
        invalid = ''
        err_dict = validate_project_id(invalid)
        assert err_dict is not None
        with pytest.raises(ValueError) as exc_info:
            require_project_id(invalid)
        assert str(exc_info.value) == err_dict['error']

    def test_injection_vector_newline_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_project_id('proj\nid')


class TestRequireRunId:
    """require_run_id raises ValueError for invalid run ids, returns None for valid ones."""

    def test_valid_uuid_raises_nothing(self):
        require_run_id('550e8400-e29b-41d4-a716-446655440000')  # must not raise

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_run_id('')

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_run_id('   ')

    def test_injection_vector_newline_raises_valueerror(self):
        with pytest.raises(ValueError):
            require_run_id('run\nid')

    def test_error_message_matches_validate_error_field(self):
        invalid = ''
        err_dict = validate_run_id(invalid)
        assert err_dict is not None
        with pytest.raises(ValueError) as exc_info:
            require_run_id(invalid)
        assert str(exc_info.value) == err_dict['error']


class TestValidateProjectId:
    """validate_project_id returns None for valid ids, error dict for empty/whitespace."""

    def test_empty_string_returns_error(self):
        result = validate_project_id('')
        assert result is not None
        assert 'error' in result
        assert 'error_type' in result
        assert result['error_type'] == 'ValidationError'

    def test_whitespace_only_returns_error(self):
        """Whitespace-only project_id must be rejected — the canonical contract includes .strip()."""
        result = validate_project_id('   ')
        assert result is not None
        assert 'error' in result
        assert result['error_type'] == 'ValidationError'

    def test_tab_only_returns_error(self):
        result = validate_project_id('\t')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_newline_only_returns_error(self):
        result = validate_project_id('\n')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_valid_id_returns_none(self):
        result = validate_project_id('dark_factory')
        assert result is None

    def test_valid_id_with_hyphens_returns_none(self):
        result = validate_project_id('my-project-1')
        assert result is None

    def test_single_character_returns_none(self):
        result = validate_project_id('x')
        assert result is None

    def test_newline_in_middle_returns_error(self):
        result = validate_project_id('proj\nid')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_backtick_returns_error(self):
        result = validate_project_id('proj`id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_double_quote_returns_error(self):
        result = validate_project_id('proj"id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_single_quote_returns_error(self):
        result = validate_project_id("proj'id")
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_curly_brace_returns_error(self):
        result = validate_project_id('proj{id}')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_semicolon_returns_error(self):
        result = validate_project_id('proj;id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_dollar_sign_returns_error(self):
        result = validate_project_id('proj$id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_space_in_middle_returns_error(self):
        result = validate_project_id('proj id')
        assert result is not None
        assert result['error_type'] == 'ValidationError'

    def test_trailing_newline_returns_error(self):
        """Trailing newline must be rejected — not silently accepted by $ anchor bypass.

        Python's re.match(r'^[a-zA-Z0-9_-]+$', 'dark_factory\\n') returns a truthy match
        because $ matches just before a trailing newline. This test exposes that bypass;
        re.fullmatch() is required to catch it.
        """
        result = validate_project_id('dark_factory\n')
        assert result is not None, (
            'validate_project_id accepted trailing newline — likely using .match() instead '
            'of .fullmatch(). Switch to re.fullmatch() to fix.'
        )
        assert result['error_type'] == 'ValidationError'

    def test_character_rejection_error_mentions_allowed_chars(self):
        result = validate_project_id('proj`id')
        assert result is not None
        assert 'invalid characters' in result['error']
        assert 'ASCII letters' in result['error']

    def test_returns_error_dict_not_raises(self):
        """validate_project_id must not raise — it returns an error dict."""
        try:
            result = validate_project_id('\x00bad')
            assert result is not None
        except Exception as exc:
            pytest.fail(f'validate_project_id raised unexpectedly: {exc}')

    def test_numeric_only_returns_none(self):
        result = validate_project_id('42')
        assert result is None

    def test_uppercase_returns_none(self):
        result = validate_project_id('DARK_FACTORY')
        assert result is None

    def test_mixed_case_hyphen_underscore_returns_none(self):
        result = validate_project_id('My-Project_1')
        assert result is None


class TestSymmetricValidatorBehavior:
    """validate_project_id and validate_run_id must enforce identical character-set rules.

    Both validators share the same allowlist (ASCII letters, digits, hyphens, underscores).
    This class makes that symmetric contract explicit and guards against future divergence
    where one pattern gets tightened or loosened independently of the other.
    """

    INJECTION_VECTORS = [
        'bad\nvalue',        # embedded newline
        'bad`value',         # backtick
        'bad"value',         # double quote
        "bad'value",         # single quote
        'bad{value}',        # curly braces
        'bad;value',         # semicolon
        'bad$value',         # dollar sign
        'bad value',         # space
        'valid\n',           # trailing newline (fullmatch bypass)
    ]

    VALID_IDENTIFIERS = [
        'simple',
        'with-hyphens',
        'with_underscores',
        'MixedCase',
        'num42',
        '42num',
        'a',
        'UPPER',
        'My-Project_1',
    ]

    @pytest.mark.parametrize('vector', INJECTION_VECTORS)
    def test_both_reject_injection_vector(self, vector):
        """Both validators must reject the same injection vectors."""
        pid_result = validate_project_id(vector)
        rid_result = validate_run_id(vector)
        assert pid_result is not None, (
            f'validate_project_id accepted injection vector {vector!r} — '
            'character-set allowlists have diverged'
        )
        assert rid_result is not None, (
            f'validate_run_id accepted injection vector {vector!r} — '
            'character-set allowlists have diverged'
        )

    @pytest.mark.parametrize('identifier', VALID_IDENTIFIERS)
    def test_both_accept_valid_identifier(self, identifier):
        """Both validators must accept the same safe identifiers."""
        pid_result = validate_project_id(identifier)
        rid_result = validate_run_id(identifier)
        assert pid_result is None, (
            f'validate_project_id rejected safe identifier {identifier!r} — '
            'character-set allowlists have diverged'
        )
        assert rid_result is None, (
            f'validate_run_id rejected safe identifier {identifier!r} — '
            'character-set allowlists have diverged'
        )


class TestValidateIdentifierDelegation:
    """validate_project_id and validate_run_id must return identical results to _validate_identifier."""

    def test_validate_project_id_matches_helper_rejection(self):
        bad = 'proj`id'
        assert validate_project_id(bad) == _validate_identifier(bad, 'project_id')

    def test_validate_project_id_matches_helper_acceptance(self):
        good = 'dark_factory'
        assert validate_project_id(good) == _validate_identifier(good, 'project_id')

    def test_validate_run_id_matches_helper_rejection(self):
        bad = 'run\nid'
        assert validate_run_id(bad) == _validate_identifier(bad, 'run_id')

    def test_validate_run_id_matches_helper_acceptance(self):
        good = '550e8400-e29b-41d4-a716-446655440000'
        assert validate_run_id(good) == _validate_identifier(good, 'run_id')


class TestValidatorErrorDictShape:
    """All validate_* functions return dicts with exactly 'error' and 'error_type' keys."""

    def test_validate_project_root_error_has_required_keys(self):
        result = validate_project_root('relative/path')
        assert result is not None
        assert set(result.keys()) == {'error', 'error_type'}

    def test_validate_project_id_error_has_required_keys(self):
        result = validate_project_id('')
        assert result is not None
        assert set(result.keys()) == {'error', 'error_type'}

    def test_validate_run_id_error_has_required_keys(self):
        result = validate_run_id('')
        assert result is not None
        assert set(result.keys()) == {'error', 'error_type'}
