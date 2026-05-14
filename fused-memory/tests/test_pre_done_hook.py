"""Tests for fused_memory.middleware.pre_done_hook."""

from unittest.mock import AsyncMock

import pytest

from fused_memory.middleware.pre_done_hook import resolve_hook_command, run_hook
from fused_memory.models.scope import resolve_project_id

# ── resolve_hook_command ───────────────────────────────────────────────────────


def test_resolve_hook_command_returns_none_when_env_var_unset(monkeypatch):
    """When no env var is set for the project, resolve_hook_command returns None."""
    # Clear the env var that would correspond to project_root '/home/leo/src/reify'
    monkeypatch.delenv('FUSED_MEMORY_PREDONE_HOOK_REIFY', raising=False)
    result = resolve_hook_command('/home/leo/src/reify')
    assert result is None


def test_resolve_hook_command_uses_project_id_upper_case(monkeypatch):
    """Env var name uses the uppercased project_id derived from the project_root basename."""
    # Simple project name
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_REIFY', 'foo {id}')
    assert resolve_hook_command('/home/leo/src/reify') == 'foo {id}'

    # Hyphenated project name: dark-factory → DARK_FACTORY
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_DARK_FACTORY', 'bar --task {id}')
    # Unset the simple one to avoid accidental cross-contamination
    monkeypatch.delenv('FUSED_MEMORY_PREDONE_HOOK_REIFY', raising=False)
    assert resolve_hook_command('/home/leo/src/dark-factory') == 'bar --task {id}'


def test_resolve_hook_command_returns_none_for_empty_or_whitespace_value(monkeypatch):
    """An empty or whitespace-only env var value is treated as unset (returns None)."""
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_REIFY', '')
    assert resolve_hook_command('/home/leo/src/reify') is None

    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_REIFY', '   ')
    assert resolve_hook_command('/home/leo/src/reify') is None


# ── run_hook ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_hook_returns_none_on_zero_exit(monkeypatch):
    """run_hook returns None (success) when the hook subprocess exits with code 0.

    Uses '/tmp' as project_root so project_id='tmp' and the env var is
    FUSED_MEMORY_PREDONE_HOOK_TMP — a deterministic name for the test.
    """
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_TMP', '/bin/true')
    result = await run_hook('42', '/tmp')
    assert result is None


@pytest.mark.asyncio
async def test_run_hook_returns_none_when_unconfigured(monkeypatch):
    """run_hook returns None immediately when no env var is configured (no subprocess)."""
    monkeypatch.delenv('FUSED_MEMORY_PREDONE_HOOK_TMP', raising=False)
    result = await run_hook('42', '/tmp')
    assert result is None


@pytest.mark.asyncio
async def test_run_hook_returns_rejected_error_on_nonzero_exit(monkeypatch):
    """run_hook returns a structured rejection error when the hook exits non-zero."""
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_TMP', '/bin/false')
    result = await run_hook('1', '/tmp')
    assert result is not None
    assert result['success'] is False
    assert result['error'] == 'pre_done_hook_rejected'
    assert result['task_id'] == '1'
    assert result['exit_code'] != 0
    assert 'hint' in result


@pytest.mark.asyncio
async def test_run_hook_includes_stderr_in_error(monkeypatch):
    """run_hook captures and includes stderr in the rejection error dict."""
    monkeypatch.setenv(
        'FUSED_MEMORY_PREDONE_HOOK_TMP',
        "sh -c 'echo my-failure-reason >&2; exit 2'",
    )
    result = await run_hook('99', '/tmp')
    assert result is not None
    assert 'my-failure-reason' in result['stderr']
    assert result['exit_code'] == 2


@pytest.mark.asyncio
async def test_run_hook_substitutes_task_id_in_argv(monkeypatch, tmp_path):
    """run_hook substitutes {id} with the actual task_id in each argv token."""
    marker = tmp_path / 'got_id.txt'
    # Script writes the value of $1 (the substituted task id) to a marker file
    cmd = f"sh -c 'echo $1 > {marker}; exit 0' -- {{id}}"
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_TMP', cmd)
    result = await run_hook('4242', '/tmp')
    # Hook should succeed
    assert result is None
    # Marker file should contain the task_id
    assert marker.exists(), 'Script did not write marker file'
    assert marker.read_text().strip() == '4242'


@pytest.mark.asyncio
async def test_run_hook_fails_closed_on_timeout(monkeypatch, tmp_path):
    """run_hook returns a timeout error and kills the process when it exceeds timeout."""
    # Script that would sleep forever (or long enough) but we give it 0.5s
    marker = tmp_path / 'completed.txt'
    cmd = f"sh -c 'sleep 5; touch {marker}; exit 0'"
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_TMP', cmd)
    result = await run_hook('1', '/tmp', timeout=0.5)
    assert result is not None
    assert result['success'] is False
    assert result['error'] == 'pre_done_hook_timeout'
    assert result['timeout_seconds'] == 0.5
    assert result['task_id'] == '1'
    # The subprocess should have been killed — the marker file should NOT exist
    assert not marker.exists(), 'Subprocess was not killed (marker file exists)'


@pytest.mark.asyncio
async def test_run_hook_fails_closed_on_misconfiguration_unterminated_quote(monkeypatch):
    """run_hook returns misconfigured error when the command has an unterminated quote."""
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_TMP', 'foo "unterminated')
    result = await run_hook('1', '/tmp')
    assert result is not None
    assert result['success'] is False
    assert result['error'] == 'pre_done_hook_misconfigured'
    assert result['task_id'] == '1'
    assert 'reason' in result


@pytest.mark.asyncio
async def test_run_hook_empty_env_var_is_noop(monkeypatch):
    """Whitespace-only env var is treated as unset — resolve_hook_command returns None,
    so run_hook is a no-op (not a misconfigured error)."""
    monkeypatch.setenv('FUSED_MEMORY_PREDONE_HOOK_TMP', '   ')
    result = await run_hook('1', '/tmp')
    # Whitespace-only is treated as unset → no-op → None
    assert result is None


# ── Three-mode integration test ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'hook_cmd,expect_success',
    [
        (None, True),           # env-unset → behaves exactly as before the feature
        ('/bin/true', True),    # passing hook → transition completes normally
        ('/bin/false', False),  # failing hook → done-flip refused, taskmaster untouched
    ],
    ids=['env-unset', 'hook-passes', 'hook-rejects'],
)
async def test_predone_hook_three_mode_integration(
    hook_cmd, expect_success, monkeypatch, tmp_path
):
    """Integration: three hook modes produce the user-observable signals from the task spec.

    Mode 1 — env-unset: behaves exactly as before the feature (no subprocess overhead).
    Mode 2 — /bin/true: hook passes, transition completes; taskmaster.set_task_status called.
    Mode 3 — /bin/false: hook rejects, done-flip refused; taskmaster.set_task_status NOT called.

    Uses a real TaskInterceptor wired with AsyncMock taskmaster/reconciler so the full
    gate-chain path is exercised end-to-end.
    """
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.reconciliation.event_buffer import EventBuffer

    taskmaster = AsyncMock()
    taskmaster.get_task = AsyncMock(
        return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'}
    )
    taskmaster.set_task_status = AsyncMock(return_value={'success': True})

    reconciler = AsyncMock()
    reconciler.reconcile_task = AsyncMock(return_value={'actions': []})

    event_buf = EventBuffer(
        db_path=tmp_path / 'three_mode_eb.db', buffer_size_threshold=100
    )
    await event_buf.initialize()
    try:
        interceptor = TaskInterceptor(taskmaster, reconciler, event_buf)

        env_key = f'FUSED_MEMORY_PREDONE_HOOK_{resolve_project_id(str(tmp_path)).upper()}'
        if hook_cmd is None:
            monkeypatch.delenv(env_key, raising=False)
        else:
            monkeypatch.setenv(env_key, hook_cmd)

        result = await interceptor.set_task_status('1', 'done', str(tmp_path))

        if expect_success:
            # Transition should succeed: no error key, taskmaster was invoked
            assert 'error' not in result, f'Expected success but got: {result}'
            taskmaster.set_task_status.assert_called_once()
        else:
            # Hook rejected: structured error returned, taskmaster untouched
            assert result.get('success') is False
            assert result.get('error') == 'pre_done_hook_rejected'
            assert result.get('task_id') == '1'
            taskmaster.set_task_status.assert_not_called()
    finally:
        await event_buf.close()
