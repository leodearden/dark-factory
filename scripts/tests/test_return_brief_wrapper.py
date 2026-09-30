"""Tests for scripts/return-brief.sh — the 05:30 wrapper: Fable prepare, then the deterministic render (task 5376).

Drives the wrapper via subprocess with its ``PREPARE_CMD`` / ``RETURN_BRIEF_CMD``
seams pointed at a fake recorder, ``REPO`` at a tmp fake repo, and a recording
``git`` shim first on PATH. No real uv, claude or store is ever reached.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[1] / 'return-brief.sh'
PREFIX = 'return-brief: '
DONE_LINE = re.compile(r'^return-brief: done \(prepare=(\S+) brief=(\S+)\)$')
SEAM_VARS = ('PREPARE_CMD', 'RETURN_BRIEF_CMD', 'RETURN_BRIEF_SKIP_PREPARE')

_FAKE_RECORDER_SRC = '''#!/usr/bin/env python3
"""Fake step recorder for return-brief.sh, mirroring the one in
scripts/tests/test_flag_marker_sweep_wrapper.py (house duplicate-with-docstring
convention). Appends argv[1:] as one JSON line to $FAKE_STEP_STATE, then exits
$FAKE_PREPARE_EXIT when argv names nightly_prepare.py, else $FAKE_BRIEF_EXIT.
Silent, so every line the wrapper run prints is the wrapper's own.
"""
import json
import os
import sys

with open(os.environ["FAKE_STEP_STATE"], "a") as sink:
    sink.write(json.dumps(sys.argv[1:]) + "\\n")
is_prepare = any(arg.endswith("nightly_prepare.py") for arg in sys.argv[1:])
sys.exit(int(os.environ["FAKE_PREPARE_EXIT" if is_prepare else "FAKE_BRIEF_EXIT"]))
'''

_FAKE_GIT_SRC = '''#!/usr/bin/env bash
# Recording git shim: the wrapper must never call git (the artifact is gitignored).
printf '%s\\n' "$*" >> "$FAKE_GIT_LOG"
exit 0
'''


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


def _recorded(state: Path) -> list[list[str]]:
    if not state.exists():
        return []
    return [json.loads(line) for line in state.read_text().splitlines()]


def _run_wrapper(tmp_path: Path, *, prepare_exit: int = 0, brief_exit: int = 0,
                 seams: bool = True, repo_name: str = 'fake-repo',
                 extra_env: dict[str, str] | None = None):
    """Run the wrapper; returns ``(result, repo, state, git_log)``.

    With ``seams=False`` both command seams are left unset, so the default
    ``uv run`` prefix runs, reaching the fake recorder installed as ``uv``.
    """
    bin_dir = tmp_path / 'bin'
    _write_executable(bin_dir / 'fake-step-recorder', _FAKE_RECORDER_SRC)
    _write_executable(bin_dir / 'uv', _FAKE_RECORDER_SRC)
    _write_executable(bin_dir / 'git', _FAKE_GIT_SRC)
    repo = tmp_path / repo_name
    repo.mkdir(exist_ok=True)
    state = tmp_path / 'steps.jsonl'
    git_log = tmp_path / 'git.log'

    env = {k: v for k, v in os.environ.items() if k not in SEAM_VARS}
    env.update({
        'PATH': f'{bin_dir}{os.pathsep}{env["PATH"]}',
        'REPO': str(repo),
        'FAKE_STEP_STATE': str(state),
        'FAKE_PREPARE_EXIT': str(prepare_exit),
        'FAKE_BRIEF_EXIT': str(brief_exit),
        'FAKE_GIT_LOG': str(git_log),
    })
    if seams:
        env['PREPARE_CMD'] = 'fake-step-recorder'
        env['RETURN_BRIEF_CMD'] = 'fake-step-recorder'
    env.update(extra_env or {})

    result = subprocess.run(['bash', str(WRAPPER)], env=env, capture_output=True, text=True, timeout=30)
    return result, repo, state, git_log


def _done(result: subprocess.CompletedProcess[str]) -> tuple[str, str]:
    last = result.stdout.splitlines()[-1]
    match = DONE_LINE.match(last)
    assert match, f'last stdout line is not the summary: {last!r}'
    return match.group(1), match.group(2)


def test_wrapper_is_executable():
    assert os.access(WRAPPER, os.X_OK), f'chmod +x {WRAPPER}'


def test_prepare_runs_then_render_each_once_inside_repo(tmp_path):
    result, repo, state, _ = _run_wrapper(tmp_path)

    assert result.returncode == 0, result.stderr
    calls = _recorded(state)
    assert [call[0] for call in calls] == [
        f'{repo}/scripts/sitting/nightly_prepare.py',
        f'{repo}/scripts/sitting/return_brief.py',
    ]
    render = calls[1]
    output = Path(render[render.index('--output') + 1])
    assert output == repo / 'data' / 'return-brief.md'
    absolute = [arg for call in calls for arg in call if arg.startswith('/')]
    assert absolute and all(arg.startswith(f'{repo}/') for arg in absolute), absolute


@pytest.mark.parametrize('prepare_exit', [1, 2, 124])
def test_render_runs_even_when_prepare_fails(tmp_path, prepare_exit):
    result, _, state, _ = _run_wrapper(tmp_path, prepare_exit=prepare_exit)

    assert result.returncode == 0
    assert [Path(call[0]).name for call in _recorded(state)] == ['nightly_prepare.py', 'return_brief.py']
    assert _done(result) == (str(prepare_exit), '0')


def test_skip_prepare_renders_only_and_says_so(tmp_path):
    result, _, state, _ = _run_wrapper(tmp_path, extra_env={'RETURN_BRIEF_SKIP_PREPARE': '1'})

    assert result.returncode == 0
    assert [Path(call[0]).name for call in _recorded(state)] == ['return_brief.py']
    assert _done(result) == ('skipped', '0')


@pytest.mark.parametrize('brief_exit', [1, 2])
def test_always_exits_zero_and_reports_the_real_codes(tmp_path, brief_exit):
    result, _, _, _ = _run_wrapper(tmp_path, prepare_exit=1, brief_exit=brief_exit)

    assert result.returncode == 0
    assert _done(result) == ('1', str(brief_exit))


def test_every_line_is_prefixed_and_failures_go_to_stderr(tmp_path):
    result, _, _, _ = _run_wrapper(tmp_path, prepare_exit=1, brief_exit=2)

    lines = result.stdout.splitlines() + result.stderr.splitlines()
    assert lines and all(line.startswith(PREFIX) for line in lines), lines
    assert result.stderr.strip(), 'a failed step is narrated on stderr'
    assert all(line.startswith(PREFIX) for line in result.stderr.splitlines())


def test_default_seams_are_uv_run_frozen_under_the_shared_project(tmp_path):
    """A space in REPO makes a dropped quote around ``$REPO/shared`` word-split the argv."""
    result, repo, state, _ = _run_wrapper(tmp_path, seams=False, repo_name='fake repo with spaces')

    assert result.returncode == 0, result.stderr
    calls = _recorded(state)
    assert len(calls) == 2
    for call in calls:
        assert call[:5] == ['run', '--frozen', '--project', f'{repo}/shared', 'python'], call
    assert [Path(call[5]).name for call in calls] == ['nightly_prepare.py', 'return_brief.py']


def test_git_is_never_invoked(tmp_path):
    result, _, _, git_log = _run_wrapper(tmp_path)

    assert result.returncode == 0
    assert not git_log.exists(), git_log.read_text()
