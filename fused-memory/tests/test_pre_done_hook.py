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
