"""Sanity checks on orchestrator/pyproject.toml filterwarnings entries.

Guards against silent regressions where the regex in a filterwarnings entry
stops matching the actual upstream message text (e.g., after a pytest /
pytest-asyncio / CPython upgrade, or when edited by hand). These tests are
pure-Python string checks — no pytest subprocess, no async machinery — so
they run in milliseconds and give a clear failure message naming the exact
regex vs. actual-text mismatch.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / 'pyproject.toml'


def _load_filterwarnings() -> list[str]:
    cfg = tomllib.loads(PYPROJECT.read_text())
    return cfg['tool']['pytest']['ini_options']['filterwarnings']


def _extract_message_regex(entry: str, *, expected_category: str) -> str:
    """Return the `message` field (index 1) of a pytest filterwarnings entry.

    The filterwarnings mini-language is `action:message:category:module:lineno`
    with ':' as field separator.  Message regexes therefore must not contain
    a literal ':' or the field splitter misparses them.
    """
    fields = entry.split(':')
    assert len(fields) >= 3, f'malformed filterwarnings entry: {entry!r}'
    action, message, category = fields[0], fields[1], fields[2]
    assert action == 'error', f'expected error action in {entry!r}, got {action!r}'
    assert category == expected_category, (
        f'expected category {expected_category!r} in {entry!r}, got {category!r}'
    )
    return message


def test_unraisable_coroutine_filter_matches_pytest_wrapper_text():
    """The PytestUnraisableExceptionWarning filter must match the wrapper text
    pytest emits via sys.unraisablehook across CPython versions.

    Pytest formats the message as `f"{err_msg}: {unraisable.object!r}"` where
    err_msg comes from `unraisable.err_msg` (see _pytest/unraisableexception.py).
    The exact prefix varies by CPython version:
    - Python <3.14: err_msg is None → prefix is "Exception ignored in"
      → full text: "Exception ignored in: <coroutine object ...>"
    - Python 3.14+: err_msg is set by CPython to include the coroutine repr
      → full text: "Exception ignored while finalizing coroutine
                    <coroutine object ...>: None"

    The filter regex must match both forms.
    """
    filters = _load_filterwarnings()
    unraisable_entries = [
        f for f in filters if f.endswith('pytest.PytestUnraisableExceptionWarning')
    ]
    assert len(unraisable_entries) == 1, (
        f'expected exactly one PytestUnraisableExceptionWarning filter, got '
        f'{unraisable_entries!r}'
    )
    message_regex = _extract_message_regex(
        unraisable_entries[0], expected_category='pytest.PytestUnraisableExceptionWarning'
    )
    # Python <3.14 format: err_msg is None → pytest uses "Exception ignored in"
    text_pre314 = (
        'Exception ignored in: <coroutine object '
        'AsyncMockMixin._execute_mock_call at 0x7f0000000000>'
    )
    # Python 3.14+ format: CPython passes custom err_msg with coroutine repr
    text_314 = (
        'Exception ignored while finalizing coroutine '
        '<coroutine object AsyncMockMixin._execute_mock_call at 0x3741a326500>: None'
    )
    for actual_text in (text_pre314, text_314):
        assert re.search(message_regex, actual_text), (
            f'filterwarnings regex {message_regex!r} does not match the '
            f'pytest-emitted unraisable-warning text {actual_text!r}. '
            f'This means the guardrail is dead code — a late-GC AsyncMock '
            f'coroutine leak will print a warning but will NOT fail CI.'
        )


def test_unraisable_coroutine_filter_has_no_literal_colon_in_message():
    """The message regex must use `.*` (or similar) instead of a literal ':'
    to match the `: ` separator pytest inserts between err_msg and repr.

    A literal ':' inside the message field would be consumed by pytest's
    filterwarnings field splitter and silently reparsed as a new field,
    breaking the filter.  This test pins that convention.
    """
    filters = _load_filterwarnings()
    unraisable_entries = [
        f for f in filters if f.endswith('pytest.PytestUnraisableExceptionWarning')
    ]
    assert len(unraisable_entries) == 1
    message_regex = _extract_message_regex(
        unraisable_entries[0], expected_category='pytest.PytestUnraisableExceptionWarning'
    )
    assert ':' not in message_regex, (
        f'message regex {message_regex!r} contains a literal colon; filterwarnings '
        f"uses ':' as field separator so the regex will be truncated."
    )


def test_runtime_warning_filter_still_matches_inline_gc_message():
    """The RuntimeWarning filter must continue to match the inline-GC case
    (CPython's native `coroutine 'X' was never awaited` message)."""
    filters = _load_filterwarnings()
    rt_entries = [f for f in filters if f.endswith(':RuntimeWarning')]
    assert len(rt_entries) == 1
    message_regex = _extract_message_regex(
        rt_entries[0], expected_category='RuntimeWarning'
    )
    actual_runtime_text = (
        "coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    assert re.search(message_regex, actual_runtime_text), (
        f'RuntimeWarning regex {message_regex!r} does not match inline-GC text '
        f'{actual_runtime_text!r}'
    )
