"""Plant a FAKE reify sibling under ``tmp_path`` for the lock-charter drift suites.

The off-gate cross-repo checker (``scripts/check_lock_charter_drift.py``) and
the router that files its outcome (``scripts/lock_charter_drift_heal.py``) both
answer questions about a reify checkout, and neither suite may touch the real
one: reify's live vectors may legitimately differ from dark-factory's at any
instant, which is exactly what those scripts exist to report. So both suites
build their reify from these two pieces: a fake ``scripts/lock-charter-guard.sh``
emitter, and a real ``git init`` checkout whose index tracks chosen paths.
``tests/scripts/conftest.py`` puts this directory on ``sys.path``, so a suite
imports it bare.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def plant_reify_guard(
    repo: Path,
    *,
    extensions: list[str] | None = None,
    extensionless: list[str] | None = None,
    rc: int = 0,
    sleep_secs: float | None = None,
    warning: str | None = None,
) -> Path:
    """Write a FAKE reify ``scripts/lock-charter-guard.sh`` under *repo* and return it.

    Mimics the real script's flag surface — ``--list-extensions`` and
    ``--list-extensionless`` each print their vector one entry per line, and an
    unrecognised flag prints usage to stderr and exits 2 (measured 2026-08-09 on
    the live script).  *rc* and *sleep_secs* exist to drive the BROKEN-emitter
    arms: a script that is present and then fails, and one that hangs.
    *warning* is a ``printf`` format written to stderr before answering, so an
    octal escape in it reaches the stream as a raw byte.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'lock-charter-guard.sh'

    def _emit(values: list[str] | None) -> str:
        if not values:
            # `printf '%s\n'` with no operands still prints one empty line, so
            # an empty vector must emit nothing at all rather than a blank.
            return ':'
        joined = ' '.join(f"'{v}'" for v in values)
        return f"printf '%s\\n' {joined}"

    body = ['#!/usr/bin/env bash']
    if warning is not None:
        body.append(f"printf '{warning}\\n' >&2")
    if sleep_secs is not None:
        body.append(f'sleep {sleep_secs}')
    if rc != 0:
        body.append(f'echo "boom" >&2; exit {rc}')
    body += [
        'case "$1" in',
        f'  --list-extensions) {_emit(extensions)} ;;',
        f'  --list-extensionless) {_emit(extensionless)} ;;',
        '  *) echo "usage: lock-charter-guard.sh --list-extensions" >&2; exit 2 ;;',
        'esac',
    ]
    script.write_text('\n'.join(body) + '\n')
    script.chmod(0o755)
    return script


def plant_tracked_repo(root: Path, paths: list[str], *, gitlinks: list[str] | None = None) -> Path:
    """`git init` a checkout at *root* with *paths* tracked, plus any *gitlinks*.

    Gitlinks are planted with ``git update-index --add --cacheinfo 160000,...``
    rather than a real submodule: that is what actually produces the mode column
    the sweep filters on, without needing a second repo on disk or network.
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
    for rel in paths:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('x\n')
    if paths:
        subprocess.run(
            ['git', '-C', str(root), 'add', '--', *paths], check=True, capture_output=True
        )
    for rel in gitlinks or []:
        subprocess.run(
            [
                'git', '-C', str(root), 'update-index', '--add',
                '--cacheinfo', f'160000,{"0" * 39}1,{rel}',
            ],
            check=True,
            capture_output=True,
        )
    return root
