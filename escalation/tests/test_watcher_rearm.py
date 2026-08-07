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
from typing import NamedTuple

import pytest

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


class _Mode(NamedTuple):
    """One of the wrapper's two filtering modes, as a caller sees it.

    The live smoke tests below come in levelled/level-less pairs that differ
    ONLY in these four values, so the pair is expressed once here rather than
    copied per test (a third mode costs one row, not a third copy).

    level_args    -- the --level argv the caller passes (empty = omitted).
    seeded_level  -- the level of the escalation seeded into the queue. The
                     level-less case deliberately seeds 0, maximally distant
                     from the levelled callers' 2, so a level-less test also
                     fails if an implementation quietly DEFAULTS LEVEL rather
                     than genuinely omitting the filter.
    level_token   -- how the wrapper must render the resolved level, both in
                     the --check dry-run line and in the live stderr preamble.
                     `<all>` is deliberate: an empty `level=` would read as an
                     unset-variable bug rather than the match-all mode.
    exclude_name  -- the wrapper-OWNED default exclude-file basename for this
                     mode. Both stay dotfiles that do not end in '.json',
                     which is what keeps them invisible to the watcher's own
                     esc-*.json glob (watcher.py:85) and to the dashboard's
                     broader *.json recon glob (dashboard/.../escalations.py:89).
    """

    level_args: list[str]
    seeded_level: int
    level_token: str
    exclude_name: str


_MODES = [
    pytest.param(
        _Mode(['--level', '2'], 2, 'level=2', '.watcher-rearm-exclude-l2'),
        id='levelled',
    ),
    pytest.param(
        _Mode([], 0, 'level=<all>', '.watcher-rearm-exclude'),
        id='level-less',
    ),
]


@pytest.mark.parametrize('mode', _MODES)
def test_pending_escalation_fires(tmp_path, mode):
    """A pending escalation already in the queue dir -> the wrapper exits 0,
    passes the escalation JSON through to stdout untouched, and emits the
    FIRED outcome line on stderr.

    Run for both modes: the level-less case pins that the wrapper drives
    escalation.watcher's already-supported match-all mode (--level declared
    type=int default=None at watcher.py:192; _matches treats None as
    match-all at watcher.py:67), so the flat, level-less reconciliation
    queue can reuse this wrapper."""
    queue_dir = tmp_path / 'queue'
    queue_dir.mkdir()
    esc = _write_pending(queue_dir, 'esc-50-1', task_id='50', level=mode.seeded_level)

    result = _run(
        '--queue-dir', str(queue_dir), *mode.level_args, '--timeout', '5',
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
    assert mode.level_token in result.stderr, (
        f'Expected the LIVE run to declare its resolved level as '
        f'{mode.level_token!r} on stderr -- since --level is optional, a '
        f'caller that omitted it by accident must see match-all mode declared '
        f'on every run, not only under --check; got {result.stderr!r}'
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


@pytest.mark.parametrize('mode', _MODES)
def test_check_dry_run_reports_resolved_mode(tmp_path, mode):
    """`--check` must reach the dry-run branch in BOTH modes (level-less
    must not exit 2 on a required-flag guard) and report what it resolved:
    the queue dir, the level token, and the exclude-file path.

    Both renderings of the level token are pinned here -- `level=2` and the
    deliberate `level=<all>` -- so a regression to an empty `level=`, or to
    a quietly-defaulted `level=2` on a level-less run, fails loudly. Same
    for the two default exclude-file spellings: the level-less one must be
    exactly `.watcher-rearm-exclude`, with the levelled `-l<level>` suffix
    DROPPED rather than left dangling as a bare `-l`."""
    queue_dir = tmp_path / 'queue'
    queue_dir.mkdir()

    result = _run(
        '--check', '--queue-dir', str(queue_dir), *mode.level_args,
        env=_live_env(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f'Expected exit 0 from --check; got {result.returncode}\n'
        f'stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    assert str(queue_dir) in result.stdout, (
        f'Expected the resolved queue-dir on stdout; got {result.stdout!r}'
    )
    assert mode.level_token in result.stdout, (
        f'Expected the dry-run to render the resolved level as '
        f'{mode.level_token!r}; got {result.stdout!r}'
    )
    resolved = result.stdout.split('exclude-file=')[-1].strip()
    assert resolved == str(queue_dir / mode.exclude_name), (
        f'Expected --check to resolve exclude-file to '
        f'{str(queue_dir / mode.exclude_name)!r}; got {resolved!r}'
    )


def test_explicit_level_still_filters(tmp_path):
    """Regression guard for the two existing sibling-skill callers
    (skills/escalation-watcher and skills/escalation-watcher-auto, which
    pass --level 2 / --level 1): making --level OPTIONAL must not make it
    INERT. A pending level=0 item must not fire a --level 2 run."""
    queue_dir = tmp_path / 'queue'
    queue_dir.mkdir()
    _write_pending(queue_dir, 'esc-53-1', task_id='53', level=0)

    result = _run(
        '--queue-dir', str(queue_dir), '--level', '2', '--timeout', '1',
        env=_live_env(REPO_ROOT),
    )

    assert result.returncode == 124, (
        f'Expected exit 124 (level=0 item filtered out by --level 2); got '
        f'{result.returncode}\nstdout={result.stdout!r} stderr={result.stderr!r}'
    )
    assert 'WATCHER_REARM_OUTCOME: CEILING exit=124' in result.stderr, (
        f'Expected the CEILING outcome line on stderr; got {result.stderr!r}'
    )
    assert 'esc-' not in result.stdout, (
        f'Expected no escalation JSON on stdout; got {result.stdout!r}'
    )


@pytest.mark.parametrize('mode', _MODES)
def test_exclude_file_ownership_suppresses(tmp_path, mode):
    """A pending escalation whose id is already listed in the wrapper's
    OWNED default exclude-file must be suppressed even with no
    --exclude-file override on the command line -- proving the wrapper
    defaults its own exclude-file path AND wires it into the watcher
    invocation, not just computes it for --check output.

    Run for both modes, so the level-less default
    (<queue_dir>/.watcher-rearm-exclude) is proven WIRED IN, not merely
    printed -- a far stronger claim than the --check assertion in
    test_check_dry_run_reports_resolved_mode can make on its own."""
    queue_dir = tmp_path / 'queue'
    queue_dir.mkdir()
    esc = _write_pending(queue_dir, 'esc-51-1', task_id='51', level=mode.seeded_level)

    default_exclude_file = queue_dir / mode.exclude_name
    default_exclude_file.write_text(f'{esc.id}\n')

    result = _run(
        '--queue-dir', str(queue_dir), *mode.level_args, '--timeout', '1',
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
