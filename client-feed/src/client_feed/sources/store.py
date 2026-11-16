"""The one read-only opener for a project's sqlite stores; a store that cannot be read raises, never reads as empty."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from enum import StrEnum
from pathlib import Path

from client_feed.contract import Source


class StoreProblem(StrEnum):
    MISSING = 'missing'
    UNREADABLE = 'unreadable'


class StoreReadError(Exception):
    """A store that cannot be read; its str names the store relative to the root."""

    def __init__(self, *, source: Source, relpath: Path, problem: StoreProblem, detail: str) -> None:
        self.source = source
        self.relpath = relpath
        self.problem = problem
        self.detail = detail
        super().__init__(f'{source} store {relpath} is {problem}: {detail}')


@contextmanager
def read_only(root: Path, db_path: Path, *, source: Source) -> Iterator[sqlite3.Connection]:
    """A query-only connection to `db_path` under `root`, every query in the block reading one snapshot."""
    relpath = db_path.relative_to(root)
    if not db_path.is_file():
        raise StoreReadError(
            source=source, relpath=relpath, problem=StoreProblem.MISSING, detail='no such file'
        )
    try:
        conn = sqlite3.connect(f'{db_path.absolute().as_uri()}?mode=ro', uri=True, timeout=5)
    except sqlite3.Error as exc:
        raise StoreReadError(
            source=source, relpath=relpath, problem=StoreProblem.UNREADABLE, detail=str(exc)
        ) from exc
    try:
        conn.execute('PRAGMA query_only = ON')
        conn.execute('BEGIN')
        yield conn
    except sqlite3.Error as exc:
        raise StoreReadError(
            source=source, relpath=relpath, problem=StoreProblem.UNREADABLE, detail=str(exc)
        ) from exc
    finally:
        with suppress(sqlite3.Error):
            conn.rollback()
        conn.close()
