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
