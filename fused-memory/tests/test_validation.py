"""Unit tests for fused_memory.utils.validation helpers."""

import pytest

from fused_memory.utils.validation import (
    InputValidationError,
    _safe_repr,
    _validate_identifier,
    require_project_id,
    require_project_root,
    require_run_id,
    validate_project_id,
    validate_project_root,
    validate_run_id,
)


class TestInputValidationError:
    """InputValidationError — custom exception for input parameter validation failures."""

    def test_is_valueerror_subclass(self):
        """InputValidationError must be a ValueError subclass for backward compat."""
        assert issubclass(InputValidationError, ValueError)

    def test_is_exception_subclass(self):
        """InputValidationError must ultimately be an Exception."""
        assert issubclass(InputValidationError, Exception)

    def test_preserves_message(self):
        """Constructor message is preserved and accessible via str()."""
        msg = 'project_root must be a non-empty absolute path'
        exc = InputValidationError(msg)
        assert str(exc) == msg

    def test_catchable_as_valueerror(self):
        """Raising InputValidationError can be caught with except ValueError (backward compat)."""
        caught = False
        try:
            raise InputValidationError('test message')
        except ValueError:
            caught = True
        assert caught, 'InputValidationError must be catchable as ValueError'

    def test_catchable_as_input_validation_error(self):
        """Raising InputValidationError can be caught with except InputValidationError."""
        caught = False
        try:
            raise InputValidationError('test message')
        except InputValidationError:
            caught = True
        assert caught

    def test_plain_valueerror_not_caught_as_input_validation_error(self):
        """A plain ValueError is NOT caught as InputValidationError (it's a subclass, not the other way)."""
        caught_as_input = False
        try:
            raise ValueError('plain error')
        except InputValidationError:
            caught_as_input = True
        except ValueError:
            pass
        assert not caught_as_input, (
            'Plain ValueError must not match except InputValidationError — '
            'InputValidationError is a subclass OF ValueError, not a superclass'
        )

    def test_empty_message(self):
        """InputValidationError with no message is still constructable."""
        exc = InputValidationError()
        assert isinstance(exc, ValueError)


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

    def test_invalid_path_raises_input_validation_error(self):
        """require_project_root must raise InputValidationError, not a plain ValueError."""
        with pytest.raises(InputValidationError):
            require_project_root('relative/path')

    def test_empty_string_raises_input_validation_error(self):
        with pytest.raises(InputValidationError):
            require_project_root('')


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

    def test_empty_string_raises_input_validation_error(self):
        """require_project_id must raise InputValidationError, not a plain ValueError."""
        with pytest.raises(InputValidationError):
            require_project_id('')

    def test_injection_vector_raises_input_validation_error(self):
        with pytest.raises(InputValidationError):
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

    def test_empty_string_raises_input_validation_error(self):
        """require_run_id must raise InputValidationError, not a plain ValueError."""
        with pytest.raises(InputValidationError):
            require_run_id('')

    def test_injection_vector_raises_input_validation_error(self):
        with pytest.raises(InputValidationError):
            require_run_id('run\nid')


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


class TestSafeRepr:
    """_safe_repr(value, max_len) — truncates repr() output to max_len characters."""

    def test_short_string_returns_full_repr_unchanged(self):
        """A string whose repr is shorter than max_len is returned as-is."""
        result = _safe_repr('hello', max_len=200)
        assert result == repr('hello')

    def test_empty_string_returns_full_repr(self):
        """Empty string repr is two quote chars — well within any reasonable limit."""
        result = _safe_repr('', max_len=200)
        assert result == repr('')

    def test_string_at_exact_limit_is_not_truncated(self):
        """When repr length equals max_len exactly, no truncation occurs."""
        # Build a string whose repr is exactly max_len chars.
        # repr of a plain ASCII string of n chars is n+2 (two quote chars).
        max_len = 20
        # 'x' * 18 -> repr is "'xxxxxxxxxxxxxxxxxx'" = 20 chars
        value = 'x' * (max_len - 2)
        r = repr(value)
        assert len(r) == max_len, 'Test setup error: repr length must equal max_len'
        result = _safe_repr(value, max_len=max_len)
        assert result == r
        assert '...(truncated)' not in result

    def test_string_exceeding_limit_is_truncated(self):
        """A string whose repr exceeds max_len is sliced and '...(truncated)' appended."""
        max_len = 20
        value = 'a' * 100  # repr is 102 chars, well over 20
        result = _safe_repr(value, max_len=max_len)
        assert result.endswith('...(truncated)')
        # The non-marker portion must be exactly max_len characters.
        prefix = result[: -len('...(truncated)')]
        assert len(prefix) == max_len

    def test_truncated_prefix_matches_repr_slice(self):
        """The prefix before the truncation marker must match repr(value)[:max_len]."""
        max_len = 30
        value = 'b' * 200
        result = _safe_repr(value, max_len=max_len)
        expected_prefix = repr(value)[:max_len]
        assert result == expected_prefix + '...(truncated)'

    def test_non_ascii_chars_repr_expansion_handled(self):
        """Non-ASCII chars expand in repr(); truncation is applied after expansion."""
        # Each non-ASCII byte may expand to \\xNN (4 chars) in repr.
        # A 60-char string of non-ASCII may have repr of 242 chars (60*4 + 2 quotes).
        value = '\xff' * 60  # each char -> \\xff (4 chars) in repr
        max_len = 50
        result = _safe_repr(value, max_len=max_len)
        assert result.endswith('...(truncated)')
        prefix = result[: -len('...(truncated)')]
        assert len(prefix) == max_len

    def test_control_chars_repr_expansion_handled(self):
        """Control chars like \\n expand in repr(); truncation is applied after."""
        value = '\n' * 100  # each -> \\n (2 chars) in repr
        max_len = 40
        result = _safe_repr(value, max_len=max_len)
        assert result.endswith('...(truncated)')

    def test_default_max_len_is_200(self):
        """Default max_len is 200: a 201-char repr triggers truncation without explicit arg."""
        # repr of n ASCII chars = n+2; need n+2 > 200 -> n >= 199
        value = 'z' * 199  # repr = 201 chars
        result = _safe_repr(value)
        assert result.endswith('...(truncated)')

    def test_default_max_len_is_200_no_truncation_for_199(self):
        """repr of 198 ASCII chars = 200 chars: must NOT be truncated with default max_len."""
        value = 'z' * 198  # repr = 200 chars == max_len, no truncation
        result = _safe_repr(value)
        assert '...(truncated)' not in result
        assert result == repr(value)


class TestTruncationInErrorMessages:
    """Truncation is applied in error messages for oversized inputs."""

    def test_validate_identifier_1mb_invalid_string_message_is_short(self):
        """A 1 MB invalid string must produce an error message shorter than 400 chars.

        Regression: without _safe_repr, repr() of a 1 MB string embeds the full
        million-character repr in the error dict, bloating logs and MCP responses.
        With _safe_repr(max_len=200), the repr is capped to 200 + 14 ('...(truncated)')
        = 214 chars, plus ~107 chars of static message overhead = at most ~321 chars total.
        """
        big_value = 'x' * 100 + '`' + 'y' * (1024 * 1024 - 101)  # invalid due to backtick
        result = _validate_identifier(big_value, 'project_id')
        assert result is not None
        assert len(result['error']) < 400, (
            f'Error message length {len(result["error"])} exceeds 400 — '
            '_validate_identifier must use _safe_repr to cap the embedded value'
        )

    def test_validate_identifier_1mb_invalid_string_contains_truncation_marker(self):
        """The truncation marker must appear in the error message for an oversized input."""
        big_value = 'bad`' + 'z' * (1024 * 1024)
        result = _validate_identifier(big_value, 'run_id')
        assert result is not None
        assert '...(truncated)' in result['error'], (
            "Error message must contain '...(truncated)' to indicate value was capped"
        )

    def test_validate_identifier_1mb_error_dict_shape_preserved(self):
        """The error dict shape (exactly 'error' and 'error_type' keys) is preserved."""
        big_value = 'bad\n' + 'a' * (1024 * 1024)
        result = _validate_identifier(big_value, 'project_id')
        assert result is not None
        assert set(result.keys()) == {'error', 'error_type'}


class TestValidateProjectRootTruncation:
    """validate_project_root must cap the repr() of oversized project_root values."""

    def test_validate_project_root_1mb_relative_path_message_is_short(self):
        """A 1 MB relative path must produce an error message shorter than 400 chars.

        Regression: without _safe_repr, repr() of a 1 MB string embeds the full
        million-character repr in the error dict, bloating logs and MCP responses.
        With _safe_repr(max_len=200), the repr is capped to 200 + 14 ('...(truncated)')
        = 214 chars, plus ~58 chars of static message overhead = at most ~272 chars total.
        """
        big_path = 'x' * (1024 * 1024)  # relative path (no leading '/') — triggers error
        result = validate_project_root(big_path)
        assert result is not None
        assert len(result['error']) < 400, (
            f'Error message length {len(result["error"])} exceeds 400 — '
            'validate_project_root must use _safe_repr to cap the embedded value'
        )

    def test_validate_project_root_1mb_relative_path_contains_truncation_marker(self):
        """The truncation marker must appear in the error message for an oversized relative path."""
        big_path = 'y' * (1024 * 1024)
        result = validate_project_root(big_path)
        assert result is not None
        assert '...(truncated)' in result['error'], (
            "Error message must contain '...(truncated)' to indicate value was capped"
        )

    def test_validate_project_root_1mb_error_dict_shape_preserved(self):
        """The error dict shape (exactly 'error' and 'error_type' keys) is preserved."""
        big_path = 'z' * (1024 * 1024)
        result = validate_project_root(big_path)
        assert result is not None
        assert set(result.keys()) == {'error', 'error_type'}


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


class TestRequireFunctionsTruncation:
    """require_* functions delegate to validate_* which uses _safe_repr — truncation is inherited."""

    def test_require_project_id_1mb_invalid_raises_with_short_message(self):
        """require_project_id raises InputValidationError with a truncated message for a 1MB input.

        Regression: require_project_id delegates to validate_project_id → _validate_identifier.
        Since _validate_identifier now uses _safe_repr, the raised exception message must
        also be short, not a million-character string.
        """
        big_id = 'bad`' + 'x' * (1024 * 1024)
        with pytest.raises(InputValidationError) as exc_info:
            require_project_id(big_id)
        msg = str(exc_info.value)
        assert len(msg) < 400, (
            f'Exception message length {len(msg)} exceeds 400 — '
            'require_project_id must inherit truncation from _validate_identifier'
        )
        assert '...(truncated)' in msg

    def test_require_run_id_1mb_invalid_raises_with_short_message(self):
        """require_run_id raises InputValidationError with a truncated message for a 1MB input."""
        big_run = 'bad\n' + 'y' * (1024 * 1024)
        with pytest.raises(InputValidationError) as exc_info:
            require_run_id(big_run)
        msg = str(exc_info.value)
        assert len(msg) < 400, (
            f'Exception message length {len(msg)} exceeds 400 — '
            'require_run_id must inherit truncation from _validate_identifier'
        )
        assert '...(truncated)' in msg

    def test_require_project_root_1mb_relative_raises_with_short_message(self):
        """require_project_root raises InputValidationError with a truncated message for a 1MB relative path."""
        big_path = 'z' * (1024 * 1024)
        with pytest.raises(InputValidationError) as exc_info:
            require_project_root(big_path)
        msg = str(exc_info.value)
        assert len(msg) < 400, (
            f'Exception message length {len(msg)} exceeds 400 — '
            'require_project_root must inherit truncation from validate_project_root'
        )
        assert '...(truncated)' in msg

    def test_require_functions_raise_input_validation_error_not_base_value_error(self):
        """require_* functions raise InputValidationError (not plain ValueError) for oversized inputs."""
        big_id = 'bad`' + 'a' * (1024 * 1024)
        with pytest.raises(InputValidationError):
            require_project_id(big_id)
        big_root = 'b' * (1024 * 1024)
        with pytest.raises(InputValidationError):
            require_project_root(big_root)


class TestNormalLengthInputsDiagnosticQuality:
    """Normal-length invalid inputs still carry full repr value — truncation must not activate."""

    def test_validate_project_id_short_invalid_contains_full_repr(self):
        """A short invalid project_id embeds its full repr in the error message.

        Regression guard: truncation must only activate on oversized inputs.
        A 7-char string with a backtick must still appear in full in the error message
        so operators can diagnose exactly what was received.
        """
        bad_id = 'proj`id'
        result = validate_project_id(bad_id)
        assert result is not None
        assert repr(bad_id) in result['error'], (
            f'Short invalid value repr must appear untruncated in error message; '
            f'got: {result["error"]!r}'
        )
        assert '...(truncated)' not in result['error']

    def test_validate_run_id_short_invalid_contains_full_repr(self):
        """A short invalid run_id embeds its full repr in the error message."""
        bad_run = 'run\nid'
        result = validate_run_id(bad_run)
        assert result is not None
        assert repr(bad_run) in result['error']
        assert '...(truncated)' not in result['error']

    def test_validate_project_root_short_relative_path_contains_full_repr(self):
        """A short relative path embeds its full repr in the validate_project_root error message."""
        rel_path = 'relative/path'
        result = validate_project_root(rel_path)
        assert result is not None
        assert repr(rel_path) in result['error'], (
            f'Short relative path repr must appear untruncated in error message; '
            f'got: {result["error"]!r}'
        )
        assert '...(truncated)' not in result['error']

    def test_validate_project_id_199_char_invalid_not_truncated(self):
        """An invalid value whose repr is exactly 199 chars is not truncated.

        repr of 197 ASCII chars = 199 chars (less than default max_len=200).
        """
        # 197 plain ASCII chars + a backtick to make it invalid; repr = 'xxx...`' (199 chars incl. quotes)
        value = 'x' * 96 + '`' + 'y' * 100  # 198 chars total, repr = 200 chars
        r = repr(value)
        # We want repr length strictly < 200 to be safe from edge cases; use shorter value
        value_short = 'x' * 50 + '`' + 'y' * 50  # 102 chars, repr = 104 chars
        result = validate_project_id(value_short)
        assert result is not None
        assert '...(truncated)' not in result['error']
        assert repr(value_short) in result['error']

    def test_require_project_id_short_invalid_raises_with_full_repr(self):
        """require_project_id exception message contains the full repr for short inputs."""
        bad_id = 'bad;input'
        with pytest.raises(InputValidationError) as exc_info:
            require_project_id(bad_id)
        msg = str(exc_info.value)
        assert repr(bad_id) in msg
        assert '...(truncated)' not in msg
