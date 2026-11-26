"""Unit tests for fused_memory.middleware.retired_key_modules_guard.

The guard takes a mapping, so every case here is a plain dict; JSON-string
metadata and both creation paths are covered end to end by the MCP-boundary
tests in test_task_tools.py.
"""

from __future__ import annotations

import pytest

from fused_memory.middleware.retired_key_modules_guard import retired_key_modules_error

# ---------------------------------------------------------------------------
# Rejection: PRESENCE of the top-level key is the violation
# ---------------------------------------------------------------------------


class TestRetiredKeyModulesErrorRejects:
    """Any value under top-level ``modules`` mints a carrier, so all reject."""

    @pytest.mark.parametrize(
        'value',
        [['fused-memory/src'], 'dashboard', [], None],
        ids=['list', 'scalar', 'empty-list', 'null'],
    )
    def test_presence_rejects_whatever_the_value(self, value):
        result = retired_key_modules_error({'modules': value})

        assert result is not None
        assert result['error_type'] == 'RetiredMetadataKey'
        assert result['retired_key'] == 'modules'
        assert result['replacement_key'] == 'files'
        assert result['error']
        assert result['hint']

    def test_valid_files_alongside_does_not_excuse_the_retired_key(self):
        result = retired_key_modules_error(
            {'modules': ['dashboard'], 'files': ['dashboard/src/dashboard/app.py']},
        )

        assert result is not None
        assert result['error_type'] == 'RetiredMetadataKey'


# ---------------------------------------------------------------------------
# Acceptance: only the top-level metadata key trips the guard
# ---------------------------------------------------------------------------


class TestRetiredKeyModulesErrorAccepts:
    """No top-level ``modules`` key, no rejection — which is why no bypass
    flag is needed (PRD open question 2): archaeology quotes the key in prose
    or nests it, and neither trips the guard."""

    @pytest.mark.parametrize(
        'metadata',
        [
            {},
            {'files': ['src/a.py'], 'task_kind': 'normal'},
            {'files': []},
            {'x_note': 'metadata.modules is retired'},
            {'x_legacy': {'modules': ['src']}},
        ],
        ids=['empty', 'files-and-kind', 'files-deferred', 'prose-mention', 'nested-key'],
    )
    def test_accepts(self, metadata):
        assert retired_key_modules_error(metadata) is None
