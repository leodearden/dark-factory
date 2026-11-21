"""Run a child process that cannot write any regular file, standing in for a sandbox whose
SQLite temp directory is unwritable (task 7251).

:func:`require_sort_to_spill` asserts the fixture's premise first, so a test of a
no-temp-file reader can never pass on a sort that would not have spilled anyway.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    'SPILL_PAD',
    'SPILL_ROWS',
    'require_sort_to_spill',
    'run_unable_to_write_files',
]

# About 8 MB of payload, roughly 4x SQLite's default in-memory sorter budget.
SPILL_ROWS = 2000
SPILL_PAD = 'x' * 4000

_PLAIN_READ_ONLY_QUERY = (
    'import sqlite3, sys\n'
    'from pathlib import Path\n'
    "uri = f'{Path(sys.argv[1]).resolve().as_uri()}?mode=ro'\n"
    'sqlite3.connect(uri, uri=True).execute(sys.argv[2]).fetchall()\n'
)

_STDERR_TAIL_CHARS = 2000


def run_unable_to_write_files(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run *argv* under ``RLIMIT_FSIZE=0``, with its output captured on pipes.

    RLIMIT_FSIZE limits regular files only, so a pipe still carries output; a
    child whose stdout is redirected to a file exits 120 instead. The limit is
    set by ``/bin/sh`` rather than ``preexec_fn``, which is unsafe in a parent
    that has threads, as pytest-timeout's thread method does.
    """
    return subprocess.run(
        ['/bin/sh', '-c', 'ulimit -f 0 && exec "$@"', 'sh', *argv],
        capture_output=True,
        text=True,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
    )


def require_sort_to_spill(db: Path, sql: str) -> None:
    """Assert that *sql* fails on a plain ``mode=ro`` connection to *db* when no file can be written."""
    result = run_unable_to_write_files([sys.executable, '-c', _PLAIN_READ_ONLY_QUERY, str(db), sql])
    if result.returncode != 0 and 'sqlite3.OperationalError' in result.stderr:
        return
    raise AssertionError(
        f'premise failed: {sql!r} against {db} did not spill to a temp file on a plain '
        f'mode=ro connection (returncode {result.returncode}; stderr tail: '
        f'{result.stderr[-_STDERR_TAIL_CHARS:]!r}). A fixture too small to spill, or a '
        'query that does not sort, would let a no-temp-file test pass vacuously.'
    )
