"""Tests for shared/src/shared/sqlite_readonly.py::connect_readonly (task 7251).

The contract under test: any SQLite file opens strictly read-only, and a
query whose sort outgrows memory still runs when no temp file can be written.
"""
from __future__ import annotations

import contextlib
import sqlite3
import sys
from pathlib import Path

import pytest

from shared.sqlite_readonly import connect_readonly
from shared.testing_sqlite_spill import (
    SPILL_PAD,
    SPILL_ROWS,
    require_sort_to_spill,
    run_unable_to_write_files,
)

_SORT_ON_AN_UNINDEXED_COLUMN = "SELECT pad FROM t ORDER BY k"


def _make_db(path: Path, rows: int = 1, pad: str = "p") -> Path:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT, pad TEXT)")
        conn.executemany(
            "INSERT INTO t (k, pad) VALUES (?, ?)",
            [(f"k{i}", pad) for i in range(rows)],
        )
        conn.commit()
    return path


def test_a_sort_that_outgrows_memory_needs_no_temp_file(tmp_path):
    db = _make_db(tmp_path / "spill.db", rows=SPILL_ROWS, pad=SPILL_PAD)
    require_sort_to_spill(db, _SORT_ON_AN_UNINDEXED_COLUMN)

    result = run_unable_to_write_files([
        sys.executable,
        "-c",
        "import sys; from shared.sqlite_readonly import connect_readonly; "
        "print(len(connect_readonly(sys.argv[1]).execute(sys.argv[2]).fetchall()))",
        str(db),
        _SORT_ON_AN_UNINDEXED_COLUMN,
    ])

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(SPILL_ROWS)


def test_connect_readonly_connection_cannot_write(tmp_path):
    db = _make_db(tmp_path / "ro.db")

    with (
        contextlib.closing(connect_readonly(db)) as conn,
        pytest.raises(sqlite3.OperationalError) as refusal,
    ):
        conn.execute("UPDATE t SET pad = 'changed'")

    assert "readonly" in str(refusal.value)


def test_connect_readonly_never_creates_an_absent_store(tmp_path):
    absent = tmp_path / "absent.db"

    with pytest.raises(sqlite3.OperationalError):
        connect_readonly(absent)

    assert not absent.exists()


@pytest.mark.parametrize("uri_syntax", ["#", "?", "%41"])
def test_connect_readonly_opens_a_file_whose_path_holds_uri_syntax(tmp_path, uri_syntax):
    """A ``#`` or ``?`` spliced raw into the URI would drop ``mode=ro`` and open a
    truncated prefix read-write, creating a sibling; a ``%41`` names another file."""
    db_dir = tmp_path / f"a{uri_syntax}b"
    db_dir.mkdir()
    db = _make_db(db_dir / "x.db")

    with contextlib.closing(connect_readonly(db)) as conn:
        rows = conn.execute("SELECT k, pad FROM t").fetchall()

    assert rows == [("k0", "p")]
    assert list(tmp_path.iterdir()) == [db_dir]


def test_connect_readonly_returns_a_plain_connection_not_a_wrapper(tmp_path):
    db = _make_db(tmp_path / "plain.db")

    with contextlib.closing(connect_readonly(db)) as conn:
        assert type(conn) is sqlite3.Connection
