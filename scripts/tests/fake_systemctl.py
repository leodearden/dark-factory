"""A recording ``systemctl`` first on PATH that prints a configured ActiveState per unit.

A plain module rather than a conftest fixture, the same convention as
``committed_git_repo``: conftest appends this directory to ``sys.path``, so
consumers import it by bare name. Code under test spawns the fake exactly as
it would the real binary, so no test reads this host's systemd.
"""
from __future__ import annotations

import os
import shlex
import stat
from collections.abc import Mapping
from pathlib import Path

import pytest

_UNIT_SEP = '\x1f'
_RECORD_SEP = '\x1e'


def install_fake_systemctl(
    bin_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    states: Mapping[str, str],
    *,
    default: str = 'inactive',
    exit_code: int = 0,
) -> Path:
    """Write the fake into *bin_dir*, prepend it to PATH, and return its calls-log path.

    The fake prints the state *states* maps its last argument (the unit) to,
    or *default*, then exits *exit_code*.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    calls = bin_dir / 'systemctl.calls'
    cases = ''.join(
        f'  {shlex.quote(unit)}) echo {shlex.quote(state)} ;;\n' for unit, state in states.items()
    )
    script = bin_dir / 'systemctl'
    script.write_text(
        '#!/bin/sh\n'
        f'{{ for a in "$@"; do printf \'%s\\037\' "$a"; done; printf \'\\036\'; }} >> {shlex.quote(str(calls))}\n'
        'for last in "$@"; do :; done\n'
        'case "$last" in\n'
        f'{cases}'
        f'  *) echo {shlex.quote(default)} ;;\n'
        'esac\n'
        f'exit {int(exit_code)}\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}' + os.environ['PATH'])
    return calls


def systemctl_calls(log: Path) -> list[list[str]]:
    """Each call's arguments, without the program name; no log means no call."""
    if not log.exists():
        return []
    return [record.split(_UNIT_SEP)[:-1] for record in log.read_text().split(_RECORD_SEP)[:-1]]
