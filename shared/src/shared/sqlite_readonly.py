"""Open any SQLite file strictly read-only, with SQLite's temporary storage held in memory.

A read-only reader may have no writable temp directory (an agent sandbox), and a
sort that outgrows memory would then fail mid-query with ``unable to open
database file``, which sweep scripts misreport as an unreadable store (task 7251).
The task-store refusals live one layer up, in
shared/src/shared/tasks_db_readonly.py::connect_ro. It imports only the standard
library.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    """Open *path* with ``mode=ro`` and ``PRAGMA temp_store=MEMORY``.

    The URI is built from the RESOLVED absolute path, so a relative one cannot
    be re-resolved against a different cwd by a subprocess or a later chdir,
    which would surface as ``unable to open database file``. It is
    percent-encoded, so a ``#``, ``?`` or ``%`` in the path cannot end the
    path early and drop ``mode=ro``.
    """
    resolved = Path(path).resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn
