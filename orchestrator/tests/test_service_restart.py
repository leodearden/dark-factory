"""Tests for orchestrator.service_restart — StaleServiceRestartCoordinator and helpers."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from orchestrator.service_restart import (
    StaleServiceRestartCoordinator,
    diff_touches_watched_paths,
)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coordinator(
    diff_files: list[str],
    *,
    watch_prefixes: list[str] | None = None,
    debounce_secs: float = 120.0,
    enabled: bool = True,
    clock_values: list[float] | None = None,
    restart_executor: object = None,
) -> tuple[StaleServiceRestartCoordinator, AsyncMock]:
    """Build a coordinator with a mock git_ops and controllable clock."""
    git_ops = MagicMock()
    git_ops.get_merge_diff_files = AsyncMock(return_value=diff_files)
    event_store = MagicMock()
    # Simple counter-based clock
    _clock_vals = list(clock_values or [0.0])
    _idx = [0]

    def _clock() -> float:
        val = _clock_vals[min(_idx[0], len(_clock_vals) - 1)]
        _idx[0] += 1
        return val

    coord = StaleServiceRestartCoordinator(
        git_ops=git_ops,
        event_store=event_store,
        watch_prefixes=watch_prefixes if watch_prefixes is not None else ['fused-memory/src/'],
        debounce_secs=debounce_secs,
        enabled=enabled,
        restart_executor=restart_executor or AsyncMock(),
        clock=_clock,
    )
    return coord, git_ops.get_merge_diff_files


# ---------------------------------------------------------------------------
# note_merge tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_note_merge_arms_pending_on_fused_memory_src_file() -> None:
    """A diff containing a fused-memory/src file: note_merge returns True and arms pending."""
    coord, mock_diff = _make_coordinator(
        ['fused-memory/src/server/main.py', 'fused-memory/docs/overview.md'],
        clock_values=[1000.0],
    )
    result = await coord.note_merge('task-42', 'base_sha_abc', 'head_sha_xyz')

    assert result is True
    assert coord.is_pending is True
    mock_diff.assert_awaited_once_with('base_sha_abc', 'head_sha_xyz')


@pytest.mark.asyncio
async def test_note_merge_does_not_arm_on_docs_only_diff() -> None:
    """A diff with only docs files does NOT arm pending."""
    coord, mock_diff = _make_coordinator(
        ['fused-memory/docs/x.md', 'fused-memory/docs/design.md'],
        clock_values=[1000.0],
    )
    result = await coord.note_merge('task-99', 'base_sha', 'head_sha')

    assert result is False
    assert coord.is_pending is False


@pytest.mark.asyncio
async def test_note_merge_does_not_arm_on_tests_only_diff() -> None:
    """A diff with only test files does NOT arm pending."""
    coord, mock_diff = _make_coordinator(
        ['fused-memory/tests/test_something.py'],
        clock_values=[1000.0],
    )
    result = await coord.note_merge('task-99', 'base_sha', 'head_sha')

    assert result is False
    assert coord.is_pending is False


@pytest.mark.asyncio
async def test_note_merge_does_not_arm_on_unrelated_package_diff() -> None:
    """A diff with only orchestrator files does NOT arm pending."""
    coord, mock_diff = _make_coordinator(
        ['orchestrator/src/orchestrator/harness.py'],
        clock_values=[1000.0],
    )
    result = await coord.note_merge('task-99', 'base_sha', 'head_sha')

    assert result is False
    assert coord.is_pending is False


@pytest.mark.asyncio
async def test_note_merge_disabled_short_circuits() -> None:
    """When enabled=False, note_merge returns False and does NOT call get_merge_diff_files."""
    coord, mock_diff = _make_coordinator(
        ['fused-memory/src/server/main.py'],
        enabled=False,
    )
    result = await coord.note_merge('task-42', 'base_sha', 'head_sha')

    assert result is False
    assert coord.is_pending is False
    mock_diff.assert_not_awaited()
