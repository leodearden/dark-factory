"""Tests for scripts/watcher-rearm.sh -- the canonical bounded-wait +
re-arm wrapper both escalation-watcher SKILL.md files shell out to.

Two kinds of coverage:
  - Guard/usage tests run the script with a deliberately minimal env -- no
    ambient DARK_FACTORY_ROOT/PROJECT_ROOT leaks in, so the "missing env"
    case is exercised regardless of the host running these tests.
  - Live smoke tests (see _live_env) drive the REAL escalation.watcher
    subprocess against a real temp queue dir, via a test-injected
    WATCHER_REARM_PYTHON=sys.executable + PYTHONPATH -- the watcher reads
    the queue directly via inotify, never through the MCP server, so this
    is the faithful hermetic "live" end-to-end path.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / 'scripts' / 'watcher-rearm.sh'


def _run(*args, env=None, timeout=30):
    """Run watcher-rearm.sh via subprocess. When env is None, uses a bare
    minimal environment (PATH only) so ambient DARK_FACTORY_ROOT/
    PROJECT_ROOT vars on the host never leak into a guard test."""
    full_env = env if env is not None else {'PATH': os.environ.get('PATH', '')}
    return subprocess.run(
        ['bash', str(SCRIPT), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_script_is_executable():
    """The working-tree script must carry the executable bit (mode 100755)."""
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); it is not. '
        f'Run: chmod +x {SCRIPT}'
    )


def test_missing_env_fails_loudly():
    """No DARK_FACTORY_ROOT, no --queue-dir, no PROJECT_ROOT -> loud
    stderr diagnostic + non-zero exit, never a silent no-op."""
    result = _run('--check', '--level', '2')

    assert result.returncode != 0, (
        f'Expected a non-zero exit with no env configured; got 0\n'
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    assert result.stderr.strip(), (
        f'Expected a non-empty stderr diagnostic (never a silent no-op); '
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
