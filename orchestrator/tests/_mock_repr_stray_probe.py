"""The tests that test_mock_repr_stray_guard.py runs in a nested pytest session.

Not collected by the suite (no ``test_`` prefix): two of them deliberately
leave a ``<MagicMock ...>`` repr-named entry in the CWD, one a SQLite file and
one a directory tree.
"""
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock


def test_leaks_a_mock_repr_sqlite_file() -> None:
    sqlite3.connect(str(MagicMock().config.project_root / 'data' / 'runs.db')).close()


def test_leaks_a_mock_repr_directory_tree() -> None:
    tree = Path(str(MagicMock().config.project_root))
    tree.mkdir()
    (tree / 'runs.db').touch()


def test_leaves_the_cwd_alone() -> None:
    pass
