"""A recording ``systemctl`` stand-in for tests whose code under test calls it.

The fake is an executable put first on PATH, so the code under test resolves
and spawns it exactly as it would the real one. It writes each call's
arguments to its own record file, then exits with ``FAKE_SYSTEMCTL_RC``
(default 0). When ``FAKE_SYSTEMCTL_SLEEP_SECS`` is set it instead becomes
``sleep`` for that long, so a caller's timeout branch can be driven.

Imported by bare module name, like ``_fake_systemd_run``. A module using the
fixture re-exports it by redundant alias
(``from _fake_systemctl import fake_systemctl as fake_systemctl``): the import
binding is what registers it with pytest. A module rather than
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
def fake_systemctl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a recording ``systemctl`` first on PATH; return its argv-record dir."""
    argv_dir = tmp_path / 'systemctl-argv'
    argv_dir.mkdir()
    fakebin = tmp_path / 'fakectl'
    fakebin.mkdir()
    script = fakebin / 'systemctl'
    script.write_text(
        '#!/bin/sh\n'
        f'rec=$(mktemp {shlex.quote(str(argv_dir))}/argv.XXXXXX)\n'
        'for a in "$@"; do printf \'%s\\0\' "$a" >> "$rec"; done\n'
        'if [ -n "${FAKE_SYSTEMCTL_SLEEP_SECS:-}" ]; then exec sleep "$FAKE_SYSTEMCTL_SLEEP_SECS"; fi\n'
        'exit "${FAKE_SYSTEMCTL_RC:-0}"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{fakebin}{os.pathsep}' + os.environ['PATH'])
    monkeypatch.delenv('FAKE_SYSTEMCTL_RC', raising=False)
    monkeypatch.delenv('FAKE_SYSTEMCTL_SLEEP_SECS', raising=False)
    return argv_dir


def recorded_calls(argv_dir: Path) -> list[list[str]]:
    """Each call's systemctl arguments, without the program name itself."""
    return [
        p.read_bytes().decode().split('\0')[:-1]
        for p in sorted(argv_dir.iterdir())
    ]
