"""Open a task store strictly read-only, or refuse naming the mistake (task 5330).

:func:`connect_ro` opens a tasks.db read-only or raises a
:class:`TaskDbUnreadable` whose :class:`TaskDbProblem` says which mistake the
reader made. What to do with a refusal (surface it, skip that store, degrade
and log) is each caller's policy, not this module's. It imports only the
standard library.
"""
from __future__ import annotations

import sqlite3
from enum import Enum
from pathlib import Path


class TaskDbProblem(Enum):
    """Why :func:`connect_ro` refused a path — the discriminator to branch on.

    An enum member rather than a substring of the message, so callers and
    tests never grow an ad-hoc parser of prose that is free to improve.
    """

    ABSENT = "absent"
    IS_A_DIRECTORY = "is_a_directory"
    EMPTY_STUB = "empty_stub"
    NO_TABLES = "no_tables"
    NOT_A_DATABASE = "not_a_database"


_REFUSAL_REMEDY = {
    TaskDbProblem.ABSENT: (
        "no such file. The live task store is the MAIN checkout's "
        ".taskmaster/tasks/tasks.db; .taskmaster/ is not tracked in git, so it "
        "never exists inside a worktree. `git worktree list --porcelain` names "
        "the main checkout on its first line."
    ),
    TaskDbProblem.IS_A_DIRECTORY: (
        "a directory, not a file. Two spellings land here: naming the "
        "CONTAINING directory (.taskmaster/tasks) instead of the store inside "
        "it (.taskmaster/tasks/tasks.db), and an EMPTY path string, which "
        "resolves to the current working directory. sqlite opens neither — it "
        "reports `disk I/O error`, which names no path and suggests a failing "
        "disk rather than a mistyped argument."
    ),
    TaskDbProblem.EMPTY_STUB: (
        "0 bytes — an empty stub, not a task store. Opening it read-only would "
        "succeed and then answer `no such table: tasks`. Two things produce "
        "one: the decoy .taskmaster/tasks.db that sits one directory ABOVE the "
        "live .taskmaster/tasks/tasks.db, and a read-write sqlite3.connect of a "
        "path that did not exist."
    ),
    TaskDbProblem.NO_TABLES: (
        "a readable sqlite database with NO tables in it, so every query "
        "against it answers `no such table: tasks`. This is what pointing at "
        "the wrong .db file looks like from the inside — and it is also what "
        "the 0-byte decoy above the live store becomes the moment any process "
        "opens it read-write, which is why size alone cannot catch it."
    ),
    TaskDbProblem.NOT_A_DATABASE: (
        "not a sqlite database at all — its bytes carry no database header, "
        "so no query of any shape will run against it. A reader who lands "
        "here is pointing at something other than a task store; the live one "
        "is the MAIN checkout's .taskmaster/tasks/tasks.db."
    ),
}


TABLE_NAMES_SQL = (
    "SELECT name FROM sqlite_master "
    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
)


class TaskDbUnreadable(Exception):
    """*path* is not a readable task store, for the structured *reason*.

    Carries the resolved :attr:`path` and a :class:`TaskDbProblem`
    :attr:`reason` as fields; the formatted message is for the human only.

    Deliberately NOT a ``sqlite3.Error`` subclass, so a broad
    ``except sqlite3.Error`` cannot swallow the refusal: a caller that means to
    tolerate it must name it.
    """

    def __init__(self, path: Path, reason: TaskDbProblem) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {_REFUSAL_REMEDY[reason]}")


def connect_ro(path: str | Path) -> sqlite3.Connection:
    """Open *path* strictly read-only, or refuse with :class:`TaskDbUnreadable`.

    Raises rather than returning a connection whose first query answers
    ``no such table: tasks`` — an error that reads as "this store is empty"
    when it in fact means "you are looking at the wrong file".

    A store that cannot be OPENED for some other reason — permissions, a lock,
    a file that vanished mid-call — propagates as the original
    :class:`sqlite3.Error`. Only sqlite's own ``SQLITE_NOTADB`` becomes a
    refusal, because only that code means the bytes are not a database;
    ``SQLITE_IOERR`` for a directory is caught EARLIER, by ``is_dir()``, since
    that code is indistinguishable from a genuinely failing disk;
    :class:`sqlite3.OperationalError` is a SUBCLASS of
    :class:`sqlite3.DatabaseError`, so branching on the exception class would
    report every unreadable store as "not a database".

    The URI is built from the RESOLVED absolute path, so a relative one cannot
    be re-resolved against a different cwd by a subprocess or a later chdir —
    the ``unable to open database file`` shape of the same confusion. It is
    percent-encoded, so a ``#``, ``?`` or ``%`` in the path cannot end the
    path early and drop ``mode=ro``.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise TaskDbUnreadable(resolved, TaskDbProblem.ABSENT)
    if resolved.is_dir():
        raise TaskDbUnreadable(resolved, TaskDbProblem.IS_A_DIRECTORY)
    if resolved.stat().st_size == 0:
        raise TaskDbUnreadable(resolved, TaskDbProblem.EMPTY_STUB)

    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    try:
        first_table = conn.execute(f"{TABLE_NAMES_SQL} LIMIT 1").fetchone()
    except sqlite3.DatabaseError as exc:
        conn.close()
        if exc.sqlite_errorcode != sqlite3.SQLITE_NOTADB:
            raise
        raise TaskDbUnreadable(resolved, TaskDbProblem.NOT_A_DATABASE) from exc
    if first_table is None:
        conn.close()
        raise TaskDbUnreadable(resolved, TaskDbProblem.NO_TABLES)
    return conn
