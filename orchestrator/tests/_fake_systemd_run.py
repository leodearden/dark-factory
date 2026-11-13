"""A recording ``systemd-run`` stand-in for tests that spawn verify scopes.

The fake is an executable put first on PATH, so the code under test resolves
and spawns it exactly as it would the real one. It writes each spawn's
arguments to its own record file, drops its own options and execs the
payload, so the verify still runs.

Imported by bare module name, like ``_proc_supervision_doubles``. A module
using the fixture re-exports it by redundant alias
(``from _fake_systemd_run import fake_systemd_run as fake_systemd_run``): the
import binding is what registers it with pytest. A module rather than
``conftest.py`` because a conftest.py edit trips verify.py's has_conftest and
widens the merge-time verify to the whole owning-package suite.
"""

from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

import pytest


@pytest.fixture
def fake_systemd_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a recording ``systemd-run`` first on PATH; return its argv-record dir."""
    argv_dir = tmp_path / 'argv'
    argv_dir.mkdir()
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    script = fakebin / 'systemd-run'
    script.write_text(
        '#!/bin/sh\n'
        f'rec=$(mktemp {shlex.quote(str(argv_dir))}/argv.XXXXXX)\n'
        'for a in "$@"; do printf \'%s\\0\' "$a" >> "$rec"; done\n'
        'while [ $# -gt 0 ]; do\n'
        '  case "$1" in -p) shift 2;; -*) shift;; *) break;; esac\n'
        'done\n'
        'exec "$@"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{fakebin}{os.pathsep}' + os.environ['PATH'])
    return argv_dir


def recorded_scope_spawns(argv_dir: Path) -> list[list[str]]:
    """Each spawn's systemd-run arguments, without the program name itself."""
    return [
        p.read_bytes().decode().split('\0')[:-1]
        for p in sorted(argv_dir.iterdir())
    ]


def assert_scope_carries(record: list[str], prop: str) -> None:
    """The adjacent pair ``-p <prop>`` precedes the payload.

    systemd-run options must come before the command, or systemd-run hands
    them to the payload instead of applying them to the scope.
    """
    pairs = [
        i for i in range(len(record) - 1)
        if record[i] == '-p' and record[i + 1] == prop
    ]
    assert pairs, f'no -p {prop} in {record!r}'
    assert pairs[0] < record.index('/bin/bash'), (
        f'{prop} must precede the /bin/bash payload: {record!r}'
    )
