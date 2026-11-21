"""The SQLite durability policy leaf: its declared values, and the WAL invariant."""

from __future__ import annotations

import sqlite3

import pytest

from shared.sqlite_durability import DURABILITY_TRIAD, SqlitePragma, require_wal_journal_mode


def test_the_triad_declares_the_phase3_values():
    assert {p.name: p.value for p in DURABILITY_TRIAD} == {
        'synchronous': 2,
        'wal_autocheckpoint': 100,
        'journal_size_limit': 67108864,
    }


@pytest.mark.parametrize('pragma', DURABILITY_TRIAD, ids=lambda p: p.name)
def test_each_triad_pragma_reads_back_the_value_it_declares(tmp_path, pragma: SqlitePragma):
    conn = sqlite3.connect(tmp_path / 'spec.db')
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute(f'PRAGMA {pragma.name}=0')
        assert conn.execute(f'PRAGMA {pragma.name}').fetchone()[0] != pragma.value
        conn.execute(pragma.statement)
        assert conn.execute(f'PRAGMA {pragma.name}').fetchone()[0] == pragma.value
    finally:
        conn.close()


def test_require_wal_journal_mode_accepts_wal():
    assert require_wal_journal_mode(('wal',)) is None


@pytest.mark.parametrize(
    ('row', 'got'),
    [(('delete',), "'delete'"), (None, 'None')],
)
def test_require_wal_journal_mode_names_the_mode_it_got(row, got):
    with pytest.raises(RuntimeError, match='WAL') as excinfo:
        require_wal_journal_mode(row)
    assert got in str(excinfo.value)
