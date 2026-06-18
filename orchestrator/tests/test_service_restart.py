"""Tests for orchestrator.service_restart — StaleServiceRestartCoordinator and helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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


# ---------------------------------------------------------------------------
# maybe_restart tests
# ---------------------------------------------------------------------------


def _make_coordinator_with_mutable_clock(
    diff_files: list[str],
    *,
    debounce_secs: float = 120.0,
    enabled: bool = True,
    restart_executor: AsyncMock | None = None,
) -> tuple[StaleServiceRestartCoordinator, list[float], AsyncMock, MagicMock]:
    """Build a coordinator with a mutable-list clock and injectable executor."""
    git_ops = MagicMock()
    git_ops.get_merge_diff_files = AsyncMock(return_value=diff_files)
    event_store = MagicMock()
    # Mutable current time — tests advance by writing current_time[0]
    current_time: list[float] = [0.0]
    executor = restart_executor or AsyncMock()

    coord = StaleServiceRestartCoordinator(
        git_ops=git_ops,
        event_store=event_store,
        watch_prefixes=['fused-memory/src/'],
        debounce_secs=debounce_secs,
        enabled=enabled,
        restart_executor=executor,
        clock=lambda: current_time[0],
    )
    return coord, current_time, executor, event_store


@pytest.mark.asyncio
async def test_maybe_restart_not_pending_returns_false() -> None:
    """(a) not pending → no executor call, returns False."""
    coord, current_time, executor, _ = _make_coordinator_with_mutable_clock([])

    result = await coord.maybe_restart(agents_idle=True)

    assert result is False
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_restart_pending_but_not_idle_defers() -> None:
    """(b) pending but agents_idle=False → no executor call, returns False, still pending."""
    coord, current_time, executor, _ = _make_coordinator_with_mutable_clock(
        ['fused-memory/src/server/main.py'], debounce_secs=0.0
    )
    current_time[0] = 1000.0
    await coord.note_merge('task-1', 'base', 'head')

    # Should be pending now; advance time well past debounce
    current_time[0] = 2000.0
    result = await coord.maybe_restart(agents_idle=False)

    assert result is False
    assert coord.is_pending is True
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_restart_pending_idle_debounce_not_elapsed() -> None:
    """(c) pending, agents_idle=True, debounce NOT elapsed → no call, returns False."""
    coord, current_time, executor, _ = _make_coordinator_with_mutable_clock(
        ['fused-memory/src/server/main.py'], debounce_secs=120.0
    )
    current_time[0] = 1000.0
    await coord.note_merge('task-1', 'base', 'head')

    # Advance only 60 s (< 120 s debounce)
    current_time[0] = 1060.0
    result = await coord.maybe_restart(agents_idle=True)

    assert result is False
    assert coord.is_pending is True
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_restart_fires_when_all_conditions_met(caplog: pytest.LogCaptureFixture) -> None:
    """(d) pending, agents_idle=True, debounce elapsed → fires once, event emitted, pending cleared."""
    import logging
    coord, current_time, executor, event_store_mock = _make_coordinator_with_mutable_clock(
        ['fused-memory/src/server/main.py'], debounce_secs=120.0
    )
    current_time[0] = 1000.0
    await coord.note_merge('task-42', 'base_sha_abc', 'head_sha_xyz')

    # Advance past debounce
    current_time[0] = 1200.0
    with caplog.at_level(logging.WARNING, logger='orchestrator.service_restart'):
        result = await coord.maybe_restart(agents_idle=True)

    assert result is True
    assert coord.is_pending is False
    executor.assert_awaited_once()

    # Check event emitted via event_store
    event_store_mock.emit.assert_called_once()
    call_kwargs = event_store_mock.emit.call_args
    data = call_kwargs.kwargs['data']
    assert data['service'] == 'fused-memory'
    assert 'task-42' in data['trigger_task_ids']
    assert 'head_sha_xyz' in data['merge_shas']
    assert data['reason'] == 'post_merge_fused_memory_code_change'

    # WARNING must have been logged
    assert any('Restarting fused-memory' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_maybe_restart_idempotent_no_double_fire() -> None:
    """(e) calling maybe_restart again with no new note_merge → executor NOT called again."""
    coord, current_time, executor, _ = _make_coordinator_with_mutable_clock(
        ['fused-memory/src/server/main.py'], debounce_secs=120.0
    )
    current_time[0] = 1000.0
    await coord.note_merge('task-42', 'base', 'head')

    current_time[0] = 1200.0
    first = await coord.maybe_restart(agents_idle=True)
    second = await coord.maybe_restart(agents_idle=True)

    assert first is True
    assert second is False
    executor.assert_awaited_once()  # exactly once total


# ---------------------------------------------------------------------------
# Executor-raises: fail-open robustness (amendment)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_restart_executor_raises_does_not_propagate(caplog: pytest.LogCaptureFixture) -> None:
    """When the restart executor raises, maybe_restart returns False (fail-open).

    The exception must NOT propagate — a missing/non-executable script must never
    crash the orchestrator's run-forever loop.  Pending is also cleared so that
    subsequent idle ticks don't retry and crash repeatedly.
    """
    import logging

    raising_executor = AsyncMock(side_effect=FileNotFoundError('script not found'))
    coord, current_time, _, _ = _make_coordinator_with_mutable_clock(
        ['fused-memory/src/server/main.py'],
        debounce_secs=120.0,
        restart_executor=raising_executor,
    )
    current_time[0] = 1000.0
    await coord.note_merge('task-42', 'base_sha', 'head_sha')
    assert coord.is_pending

    # Advance past debounce — executor would fire, but raises
    current_time[0] = 1200.0
    with caplog.at_level(logging.WARNING, logger='orchestrator.service_restart'):
        result = await coord.maybe_restart(agents_idle=True)

    # Fail-open: no propagation
    assert result is False
    # Pending cleared — subsequent ticks must not retry
    assert coord.is_pending is False
    # A warning with exc_info should have been logged
    assert any('executor failed' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_maybe_restart_executor_raises_subsequent_tick_does_not_retry() -> None:
    """After an executor failure, is_pending is False — subsequent idle ticks are no-ops."""
    raising_executor = AsyncMock(side_effect=OSError('permission denied'))
    coord, current_time, _, _ = _make_coordinator_with_mutable_clock(
        ['fused-memory/src/server/main.py'],
        debounce_secs=0.0,
        restart_executor=raising_executor,
    )
    current_time[0] = 0.0
    await coord.note_merge('task-1', 'base', 'head')

    # First call: executor raises, fail-open
    result1 = await coord.maybe_restart(agents_idle=True)
    assert result1 is False
    assert not coord.is_pending

    # Second call: no pending, must be a no-op (executor called exactly once total)
    result2 = await coord.maybe_restart(agents_idle=True)
    assert result2 is False
    raising_executor.assert_awaited_once()


# ---------------------------------------------------------------------------
# Burst-coalescing (step-7) + default executor tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_burst_coalescing_fires_exactly_once_after_last_merge() -> None:
    """A burst of fused-memory merges debounces into exactly ONE restart.

    Timeline:
    - t=0: note_merge #1 — arms pending, last_request=0
    - t=30: maybe_restart → debounce not elapsed (need 120s from last note_merge)
    - t=60: note_merge #2 — re-arms, last_request=60
    - t=90: maybe_restart → only 30s since last note_merge → still deferred
    - t=120: note_merge #3 — re-arms, last_request=120
    - t=150: maybe_restart → only 30s since last note_merge → still deferred
    - t=240 (120s after last): maybe_restart → fires exactly once
    """
    git_ops = MagicMock()
    git_ops.get_merge_diff_files = AsyncMock(return_value=['fused-memory/src/server/main.py'])
    event_store = MagicMock()
    executor = AsyncMock()
    current_time: list[float] = [0.0]

    coord = StaleServiceRestartCoordinator(
        git_ops=git_ops,
        event_store=event_store,
        watch_prefixes=['fused-memory/src/'],
        debounce_secs=120.0,
        enabled=True,
        restart_executor=executor,
        clock=lambda: current_time[0],
    )

    current_time[0] = 0.0
    await coord.note_merge('task-1', 'base1', 'head1')
    assert coord.is_pending

    # 30s — debounce not elapsed (0s since last note_merge at t=0 → 30s elapsed < 120)
    current_time[0] = 30.0
    r = await coord.maybe_restart(agents_idle=True)
    assert r is False
    executor.assert_not_awaited()

    # t=60: second merge re-arms, resets last_request to 60
    current_time[0] = 60.0
    await coord.note_merge('task-2', 'base2', 'head2')

    # t=90: only 30s since last note_merge (t=60) — still deferred
    current_time[0] = 90.0
    r = await coord.maybe_restart(agents_idle=True)
    assert r is False
    executor.assert_not_awaited()

    # t=120: third merge re-arms, resets last_request to 120
    current_time[0] = 120.0
    await coord.note_merge('task-3', 'base3', 'head3')

    # t=150: only 30s since last note_merge (t=120) — still deferred
    current_time[0] = 150.0
    r = await coord.maybe_restart(agents_idle=True)
    assert r is False
    executor.assert_not_awaited()

    # t=240: 120s since last note_merge (t=120) — fires!
    current_time[0] = 240.0
    r = await coord.maybe_restart(agents_idle=True)
    assert r is True
    executor.assert_awaited_once()
    assert not coord.is_pending

    # Calling again without a new note_merge → no second fire
    r = await coord.maybe_restart(agents_idle=True)
    assert r is False
    executor.assert_awaited_once()  # still exactly one call total


@pytest.mark.asyncio
async def test_default_executor_spawns_script_detached() -> None:
    """Default executor spawns scripts/restart-fused-memory.sh --drain detached."""
    from unittest.mock import MagicMock as MM
    from unittest.mock import patch

    git_ops = MagicMock()
    git_ops.get_merge_diff_files = AsyncMock(return_value=['fused-memory/src/server/main.py'])
    event_store = MagicMock()
    current_time: list[float] = [0.0]

    # Construct WITHOUT injecting restart_executor to exercise the default path
    coord = StaleServiceRestartCoordinator(
        git_ops=git_ops,
        event_store=event_store,
        watch_prefixes=['fused-memory/src/'],
        debounce_secs=0.0,  # no debounce — fire immediately
        enabled=True,
        project_root='/fake/project',
        script_path='scripts/restart-fused-memory.sh',
        clock=lambda: current_time[0],
    )

    current_time[0] = 0.0
    await coord.note_merge('task-1', 'base1', 'head1')

    # Patch asyncio.create_subprocess_exec inside the service_restart module
    fake_proc = MM()
    with patch(
        'orchestrator.service_restart.asyncio.create_subprocess_exec',
        new_callable=AsyncMock,
        return_value=fake_proc,
    ) as mock_exec:
        r = await coord.maybe_restart(agents_idle=True)

    assert r is True
    mock_exec.assert_awaited_once()
    call_args = mock_exec.call_args
    # Positional args: script path and --drain flag
    pos_args = call_args.args if call_args.args else call_args[0]
    assert str(pos_args[0]).endswith('scripts/restart-fused-memory.sh')
    assert pos_args[1] == '--drain'
    # Must be spawned detached (fire-and-forget, survives MCP reconnect)
    assert call_args.kwargs.get('start_new_session') is True
    # The process must NOT be awaited — fake_proc.wait/communicate not called
    fake_proc.wait.assert_not_called()
    fake_proc.communicate.assert_not_called()


# ---------------------------------------------------------------------------
# service_name parameterization tests (step-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_name_dashboard_emits_correct_event_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(a) Coordinator built with service_name='dashboard' emits correct event data and log."""
    import logging

    coord, current_time, executor, event_store_mock = _make_coordinator_with_mutable_clock(
        ['dashboard/src/app.py'],
    )
    # Rebuild with dashboard prefixes and service_name
    git_ops = MagicMock()
    git_ops.get_merge_diff_files = AsyncMock(return_value=['dashboard/src/app.py'])
    event_store = MagicMock()
    current_time2: list[float] = [0.0]
    exec2 = AsyncMock()

    coord2 = StaleServiceRestartCoordinator(
        git_ops=git_ops,
        event_store=event_store,
        watch_prefixes=['dashboard/src/'],
        debounce_secs=0.0,
        enabled=True,
        restart_executor=exec2,
        clock=lambda: current_time2[0],
        service_name='dashboard',
    )

    await coord2.note_merge('task-99', 'base_sha', 'head_sha')

    current_time2[0] = 1.0  # past debounce (0)
    with caplog.at_level(logging.WARNING, logger='orchestrator.service_restart'):
        result = await coord2.maybe_restart(agents_idle=True)

    assert result is True
    exec2.assert_awaited_once()

    # Event data must use service_name
    event_store.emit.assert_called_once()
    data = event_store.emit.call_args.kwargs['data']
    assert data['service'] == 'dashboard'
    assert data['reason'] == 'post_merge_dashboard_code_change'

    # Log must mention 'dashboard'
    assert any('Restarting dashboard' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_service_name_default_fused_memory_byte_identical(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(b) Default service_name keeps data['service']=='fused-memory' and legacy reason — regression guard."""
    import logging

    coord, current_time, executor, event_store_mock = _make_coordinator_with_mutable_clock(
        ['fused-memory/src/server/main.py'], debounce_secs=0.0
    )
    current_time[0] = 1000.0
    await coord.note_merge('task-42', 'base_sha_abc', 'head_sha_xyz')
    current_time[0] = 1001.0

    with caplog.at_level(logging.WARNING, logger='orchestrator.service_restart'):
        result = await coord.maybe_restart(agents_idle=True)

    assert result is True
    event_store_mock.emit.assert_called_once()
    data = event_store_mock.emit.call_args.kwargs['data']
    assert data['service'] == 'fused-memory'
    assert data['reason'] == 'post_merge_fused_memory_code_change'
    assert any('Restarting fused-memory' in r.message for r in caplog.records)
