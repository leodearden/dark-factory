"""Tests for orchestrator.service_restart — StaleServiceRestartCoordinator and helpers."""

from __future__ import annotations

import pytest

from orchestrator.service_restart import diff_touches_watched_paths


DEFAULT_PREFIXES = ['fused-memory/src/']


# ---------------------------------------------------------------------------
# diff_touches_watched_paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'changed_files,prefixes,expected',
    [
        # Basic match: a file under the watched prefix
        (
            ['fused-memory/src/server/main.py'],
            DEFAULT_PREFIXES,
            True,
        ),
        # Exact prefix boundary: file at the root of the prefix dir
        (
            ['fused-memory/src/'],
            DEFAULT_PREFIXES,
            True,
        ),
        # Multiple files, one match is enough
        (
            ['fused-memory/docs/overview.md', 'fused-memory/src/reconciliation/harness.py'],
            DEFAULT_PREFIXES,
            True,
        ),
        # docs-only: should NOT trigger
        (
            ['fused-memory/docs/x.md'],
            DEFAULT_PREFIXES,
            False,
        ),
        # tests-only: should NOT trigger
        (
            ['fused-memory/tests/test_x.py'],
            DEFAULT_PREFIXES,
            False,
        ),
        # Unrelated package: should NOT trigger
        (
            ['orchestrator/src/orchestrator/harness.py'],
            DEFAULT_PREFIXES,
            False,
        ),
        # Empty changed-files list: nothing to match
        (
            [],
            DEFAULT_PREFIXES,
            False,
        ),
        # Near-miss: fused-memory/srcfoo does NOT match fused-memory/src/
        # (boundary-safe: no false positive for a path that starts with the
        # prefix string but is NOT inside the directory)
        (
            ['fused-memory/srcfoo/file.py'],
            DEFAULT_PREFIXES,
            False,
        ),
        # Empty prefix list: never matches
        (
            ['fused-memory/src/server/main.py'],
            [],
            False,
        ),
        # Multiple prefixes, second one matches
        (
            ['orchestrator/src/orchestrator/merge_queue.py'],
            ['fused-memory/src/', 'orchestrator/src/'],
            True,
        ),
        # Nested directory match
        (
            ['fused-memory/src/memory/stores/graphiti_client.py'],
            DEFAULT_PREFIXES,
            True,
        ),
    ],
)
def test_diff_touches_watched_paths(
    changed_files: list[str],
    prefixes: list[str],
    expected: bool,
) -> None:
    result = diff_touches_watched_paths(changed_files, prefixes)
    assert result is expected, (
        f'diff_touches_watched_paths({changed_files!r}, {prefixes!r}) '
        f'returned {result!r}, expected {expected!r}'
    )
