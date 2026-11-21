"""Run a child process that cannot write any regular file, standing in for a sandbox whose
SQLite temp directory is unwritable (task 7251).

:func:`require_sort_to_spill` asserts the fixture's premise first: the query succeeds
on a plain ``mode=ro`` connection with a writable temp dir and no limit, and fails
once no file can be written. The failure is therefore the write limit's, so a test of a no-temp-file
reader cannot pass on a query that never needed a temp file, nor on one that
fails for any other reason.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
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
    'import json, sqlite3, sys\n'
    'from pathlib import Path\n'
    "uri = f'{Path(sys.argv[1]).resolve().as_uri()}?mode=ro'\n"
    'sqlite3.connect(uri, uri=True).execute(sys.argv[2], json.loads(sys.argv[3])).fetchall()\n'
)

_WRITE_LIMIT_PREFIX = ('/bin/sh', '-c', 'ulimit -f 0 && exec "$@"', 'sh')

_STDERR_TAIL_CHARS = 2000


def run_unable_to_write_files(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run *argv* under ``RLIMIT_FSIZE=0``, with its output captured on pipes.

    RLIMIT_FSIZE limits regular files only, so a pipe still carries output; a
    child whose stdout is redirected to a file exits 120 instead. The limit is
    set by ``/bin/sh`` rather than ``preexec_fn``, which is unsafe in a parent
    that has threads, as pytest-timeout's thread method does.
    """
    return _run_captured([*_WRITE_LIMIT_PREFIX, *argv], _child_env())


def require_sort_to_spill(db: Path, sql: str, params: Sequence[object] = ()) -> None:
    """Assert that *sql*, bound to *params*, runs on a plain ``mode=ro`` connection to *db*
    and fails with ``sqlite3.OperationalError`` only once no file can be written.

    Both runs keep SQLite's temp files beside *db*, so the control still has a
    writable temp dir inside a sandbox whose own one is not.
    """
    argv = [sys.executable, '-c', _PLAIN_READ_ONLY_QUERY, str(db), sql, json.dumps(list(params))]
    env = _child_env(SQLITE_TMPDIR=str(Path(db).resolve().parent))
    unlimited = _run_captured(argv, env)
    limited = _run_captured([*_WRITE_LIMIT_PREFIX, *argv], env)
    if (
        unlimited.returncode == 0
        and limited.returncode != 0
        and 'sqlite3.OperationalError' in limited.stderr
    ):
        return
    raise AssertionError(
        f'premise failed: {sql!r} with {list(params)!r} against {db} did not fail only for '
        f'want of a temp file on a plain mode=ro connection. Control run with no write '
        f'limit: {_describe(unlimited)}. Run unable to write files: {_describe(limited)}. '
        'The control must exit 0 and the limited run must raise sqlite3.OperationalError; '
        'otherwise the fixture is too small to spill, the query does not sort, or it fails '
        'for another reason, and a no-temp-file test would pass vacuously.'
    )


def _child_env(**overrides: str) -> dict[str, str]:
    return {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **overrides}


def _run_captured(argv: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, env=env)


def _describe(result: subprocess.CompletedProcess[str]) -> str:
    return f'returncode {result.returncode}, stderr tail {result.stderr[-_STDERR_TAIL_CHARS:]!r}'
