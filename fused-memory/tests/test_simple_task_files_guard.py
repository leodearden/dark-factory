"""Unit tests for fused_memory.middleware.simple_task_files_guard.

The guard takes a mapping, so every case here is a plain dict; JSON-string
metadata and both creation paths are covered end to end by the MCP-boundary
tests in test_task_tools.py.
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.simple_task_files_guard import simple_task_files_error
from fused_memory.middleware.task_interceptor import interceptor_write_succeeded

# ---------------------------------------------------------------------------
# Rejection: a simple declaration with no declared files
# ---------------------------------------------------------------------------


def _assert_simple_task_files_rejection(result):
    assert result is not None
    assert result['error_type'] == 'SimpleTaskFilesRequired'
    assert result['complexity'] == 'simple'
    assert result['required_key'] == 'files'
    assert result['error']
    assert result['hint']
    assert interceptor_write_succeeded(result) is False


class TestSimpleTaskFilesErrorRejects:
    """A simple task plans and edits under its own lock, so it must name files."""

    @pytest.mark.parametrize(
        'metadata',
        [
            {'complexity': 'simple'},
            {'complexity': 'simple', 'files': [123, None]},
            {'complexity': 'simple', 'files': 'a.py'},
            {'complexity': ' Simple '},
        ],
        ids=['files-absent', 'no-str-entries', 'non-list-files', 'non-canonical-spelling'],
    )
    def test_simple_without_files_rejects(self, metadata):
        _assert_simple_task_files_rejection(simple_task_files_error(metadata))

    def test_empty_files_list_rejects(self):
        """``files=[]`` defers scope to the architect, and a simple task never
        dispatches one, so the empty list does not satisfy the requirement."""
        _assert_simple_task_files_rejection(
            simple_task_files_error({'complexity': 'simple', 'files': []}),
        )

    def test_hint_names_the_fix_and_the_opt_out(self):
        result = simple_task_files_error({'complexity': 'simple'})

        assert result is not None
        assert 'metadata.files' in result['hint']
        assert 'complexity' in result['hint']


# ---------------------------------------------------------------------------
# Acceptance: declared files, or no simple declaration at all
# ---------------------------------------------------------------------------


class TestSimpleTaskFilesErrorAccepts:
    """Only simple-declared submissions are gated; a fileless full-path task
    stays legal because its architect derives scope from the plan."""

    @pytest.mark.parametrize(
        'metadata',
        [
            {'complexity': 'simple', 'files': ['widget/src/widget/app.py']},
            {},
            {'complexity': 'full'},
            {'complexity': None},
            {'complexity': ['simple']},
            {'task_kind': 'normal'},
        ],
        ids=['simple-with-files', 'empty', 'full', 'null', 'list-of-simple', 'no-complexity'],
    )
    def test_accepts(self, metadata):
        assert simple_task_files_error(metadata) is None
