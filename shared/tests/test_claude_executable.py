"""Contract for shared.claude_executable: which claude executable each launch surface runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from shared.claude_executable import (
    DEFAULT_CLAUDE_EXECUTABLE,
    ClaudeLaunchSurface,
    claude_executable,
)

# The LOCAL tree's module, run by path exactly as skills/spawn/spawn-claude.sh runs it.
_MODULE_PATH = Path(__file__).resolve().parents[1] / 'src' / 'shared' / 'claude_executable.py'

_OWN_VAR = {
    ClaudeLaunchSurface.FLEET: 'CLAUDE_BINARY',
    ClaudeLaunchSurface.SPAWN: 'CLAUDE_SPAWN_CLAUDE_BIN',
}


def test_spawn_surface_takes_its_override():
    environ = {'CLAUDE_SPAWN_CLAUDE_BIN': '/opt/pool/claude-pool'}
    assert claude_executable(ClaudeLaunchSurface.SPAWN, environ) == '/opt/pool/claude-pool'


def test_fleet_surface_takes_its_override():
    environ = {'CLAUDE_BINARY': '/opt/claude/bin/claude'}
    assert claude_executable(ClaudeLaunchSurface.FLEET, environ) == '/opt/claude/bin/claude'


@pytest.mark.parametrize('surface', list(ClaudeLaunchSurface))
def test_unset_override_means_bare_claude(surface):
    assert claude_executable(surface, {}) == DEFAULT_CLAUDE_EXECUTABLE == 'claude'


@pytest.mark.parametrize('surface', list(ClaudeLaunchSurface))
def test_empty_override_counts_as_unset(surface):
    assert claude_executable(surface, {_OWN_VAR[surface]: ''}) == 'claude'


@pytest.mark.parametrize(
    ('surface', 'other_surface'),
    [
        (ClaudeLaunchSurface.SPAWN, ClaudeLaunchSurface.FLEET),
        (ClaudeLaunchSurface.FLEET, ClaudeLaunchSurface.SPAWN),
    ],
)
def test_a_surface_never_reads_another_surfaces_override(surface, other_surface):
    environ = {_OWN_VAR[other_surface]: '/opt/elsewhere/claude-wrapper'}
    assert claude_executable(surface, environ) == 'claude'


def test_fleet_binding_ignores_the_spawn_override(monkeypatch):
    from shared.cli_invoke import claude_binary_spec

    monkeypatch.setenv('CLAUDE_SPAWN_CLAUDE_BIN', '/opt/pool/claude-pool')
    monkeypatch.delenv('CLAUDE_BINARY', raising=False)
    assert claude_binary_spec() == 'claude'


def _run_by_path(*argv: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-I', str(_MODULE_PATH), *argv],
        env={'PATH': os.environ['PATH'], **overrides},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_prints_the_spawn_override():
    proc = _run_by_path('spawn', CLAUDE_SPAWN_CLAUDE_BIN='/opt/pool dir/claude-pool')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == '/opt/pool dir/claude-pool\n'


def test_cli_prints_the_fleet_override():
    proc = _run_by_path('fleet', CLAUDE_BINARY='/opt/claude/bin/claude')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == '/opt/claude/bin/claude\n'


def test_cli_prints_bare_claude_without_an_override():
    proc = _run_by_path('spawn')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == 'claude\n'


def test_cli_rejects_an_unknown_surface():
    proc = _run_by_path('tmux', CLAUDE_SPAWN_CLAUDE_BIN='/opt/pool/claude-pool')
    assert proc.returncode == 2
    assert proc.stdout == ''
