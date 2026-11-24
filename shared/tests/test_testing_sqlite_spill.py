"""Tests for shared/src/shared/testing_sqlite_spill.py::require_sort_to_spill (task 7251).

The premise check must accept only a query that runs with no write limit and
fails under it, so a mistyped or stale query cannot satisfy it.
"""
from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest

from shared.testing_sqlite_spill import SPILL_PAD, SPILL_ROWS, require_sort_to_spill

_BOUND_SORT = "SELECT pad FROM t WHERE k NOT IN (?, ?) ORDER BY k"
_BOUND_PARAMS = ("k0", "k1")


def _make_db(path: Path, rows: int) -> Path:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT, pad TEXT)")
        conn.executemany(
            "INSERT INTO t (k, pad) VALUES (?, ?)",
            [(f"k{i}", SPILL_PAD) for i in range(rows)],
        )
        conn.commit()
    return path


def test_accepts_a_bound_sort_that_spills(tmp_path):
    require_sort_to_spill(_make_db(tmp_path / "big.db", SPILL_ROWS), _BOUND_SORT, _BOUND_PARAMS)


@pytest.mark.parametrize(
    ("rows", "sql", "params"),
    [
        pytest.param(1, _BOUND_SORT, _BOUND_PARAMS, id="sort-fits-in-memory"),
        pytest.param(SPILL_ROWS, "SELECT nosuchcol FROM t ORDER BY k", (), id="bad-column"),
        pytest.param(SPILL_ROWS, "SELECT k FROM nosuchtable ORDER BY k", (), id="bad-table"),
        pytest.param(SPILL_ROWS, _BOUND_SORT, (), id="unbound-parameters"),
    ],
)
def test_rejects_a_query_whose_failure_is_not_the_write_limit(tmp_path, rows, sql, params):
    db = _make_db(tmp_path / "t.db", rows)

    with pytest.raises(AssertionError, match="premise failed"):
        require_sort_to_spill(db, sql, params)


def test_rejects_an_absent_database(tmp_path):
    with pytest.raises(AssertionError, match="Control run with no write limit: returncode 1"):
        require_sort_to_spill(tmp_path / "absent.db", _BOUND_SORT, _BOUND_PARAMS)
