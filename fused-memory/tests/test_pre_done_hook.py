"""Tests for fused_memory.middleware.pre_done_hook."""

import asyncio
import os

import pytest

from fused_memory.middleware.pre_done_hook import resolve_hook_command, run_hook


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
