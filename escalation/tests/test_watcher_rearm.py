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
import sys
from pathlib import Path

from escalation.models import Escalation

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'watcher-rearm.sh'


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


def _live_env(root):
    """Env for driving the REAL escalation.watcher subprocess: injects
    WATCHER_REARM_PYTHON=sys.executable + PYTHONPATH so the wrapper's
    default `uv run` interpreter is bypassed and the subprocess imports
    escalation.watcher (and shared.timestamps) directly from source."""
    env = dict(os.environ)
    env['DARK_FACTORY_ROOT'] = str(root)
    env['WATCHER_REARM_PYTHON'] = sys.executable
    env['PYTHONPATH'] = f'{root}/escalation/src:{root}/shared/src'
    return env


def _write_pending(queue_dir, esc_id, task_id='1', level=2):
    """Write a valid pending Escalation JSON file into queue_dir, named by
    its id -- mirrors test_watcher.py's _write_esc helper."""
    esc = Escalation(
        id=esc_id, task_id=task_id, agent_role='orchestrator',
        severity='blocking', category='task_failure', summary='pending',
        level=level,
    )
    (queue_dir / f'{esc_id}.json').write_text(esc.to_json())
    return esc


def test_fired_smoke(tmp_path):
    """A pending L2 escalation already in the queue dir -> the wrapper
    exits 0, passes the escalation JSON through to stdout untouched, and
    emits the FIRED outcome line on stderr."""
    queue_dir = tmp_path / 'queue'
    queue_dir.mkdir()
    esc = _write_pending(queue_dir, 'esc-50-1', task_id='50', level=2)

    result = _run(
        '--queue-dir', str(queue_dir), '--level', '2', '--timeout', '5',
        env=_live_env(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f'Expected exit 0; got {result.returncode}\n'
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    assert esc.id in result.stdout, (
        f'Expected the escalation JSON on stdout; got stdout={result.stdout!r} '
        f'stderr={result.stderr!r}'
    )
    assert 'WATCHER_REARM_OUTCOME: FIRED exit=0' in result.stderr, (
        f'Expected the FIRED outcome line on stderr; got {result.stderr!r}'
    )


def test_ceiling_smoke(tmp_path):
    """An empty queue dir -> the bounded wait expires; the wrapper exits
    124, emits the CEILING outcome line on stderr, and prints no
    escalation JSON to stdout."""
    queue_dir = tmp_path / 'queue'
    queue_dir.mkdir()

    result = _run(
        '--queue-dir', str(queue_dir), '--level', '2', '--timeout', '1',
        env=_live_env(REPO_ROOT),
    )

    assert result.returncode == 124, (
        f'Expected exit 124; got {result.returncode}\n'
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    assert 'WATCHER_REARM_OUTCOME: CEILING exit=124' in result.stderr, (
        f'Expected the CEILING outcome line on stderr; got {result.stderr!r}'
    )
    assert 'esc-' not in result.stdout, (
        f'Expected no escalation JSON on stdout; got {result.stdout!r}'
    )


def test_exclude_file_ownership_suppresses(tmp_path):
    """A pending escalation whose id is already listed in the wrapper's
    OWNED default exclude-file (<queue_dir>/.watcher-rearm-exclude-l<level>)
    must be suppressed even with no --exclude-file override on the command
    line -- proving the wrapper defaults its own exclude-file path AND
    wires it into the watcher invocation, not just computes it for --check
    output."""
    queue_dir = tmp_path / 'queue'
    queue_dir.mkdir()
    esc = _write_pending(queue_dir, 'esc-51-1', task_id='51', level=2)

    default_exclude_file = queue_dir / '.watcher-rearm-exclude-l2'
    default_exclude_file.write_text(f'{esc.id}\n')

    result = _run(
        '--queue-dir', str(queue_dir), '--level', '2', '--timeout', '1',
        env=_live_env(REPO_ROOT),
    )

    assert result.returncode == 124, (
        f'Expected exit 124 (pending item suppressed via the wrapper-owned '
        f'exclude-file); got {result.returncode}\n'
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    assert 'WATCHER_REARM_OUTCOME: CEILING exit=124' in result.stderr, (
        f'Expected the CEILING outcome line on stderr; got {result.stderr!r}'
    )
    assert 'esc-' not in result.stdout, (
        f'Expected no escalation JSON on stdout; got {result.stdout!r}'
    )
