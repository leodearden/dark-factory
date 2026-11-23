"""Tests for mem0_history_at.py — read-only access to mem0's history, and a memory's text at an instant.

A synthetic history.db in tmp only. The module is loaded by path
(``scripts/`` is not a package), lazily through ``_mod()``.
"""
from __future__ import annotations

import functools
import sqlite3
import types
from datetime import datetime
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from _mem0_history_db import mem0_history_db

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'mem0_history_at.py', mod_name='mem0_history_at')


@pytest.fixture
def history(tmp_path: Path) -> Path:
    """p1 edited twice, p2 once, and p3 once by an UPDATE that kept no old text."""
    return mem0_history_db(tmp_path / 'history.db', [
        ('p1', None, 'first', 'ADD', '2026-10-01T00:00:00+00:00'),
        ('p1', 'first', 'second', 'UPDATE', '2026-10-05T00:00:00+00:00'),
        ('p1', 'second', 'third', 'UPDATE', '2026-10-09T00:00:00+00:00'),
        ('p2', 'other', 'other v2', 'UPDATE', '2026-10-03T00:00:00+00:00'),
        ('p3', None, 'lost', 'UPDATE', '2026-10-03T00:00:00+00:00'),
    ])


class TestReadOnly:
    def test_a_read_only_connection_refuses_a_write(self, history: Path) -> None:
        with _mod().read_only(history) as connection, pytest.raises(sqlite3.OperationalError):
            connection.execute(
                "INSERT INTO history VALUES ('h9', 'p9', NULL, 'new', 'ADD', NULL, NULL, 0, NULL, NULL)"
            )


class TestTextAt:

    @pytest.mark.parametrize(('instant', 'expected'), [
        ('2026-10-02T12:00:00.000001+00:00', 'first'),
        ('2026-10-07T00:00:00+00:00', 'second'),
        ('2026-10-10T00:00:00+00:00', 'third, current'),
    ])
    def test_is_the_text_before_the_first_edit_after_the_instant(
        self, history: Path, instant: str, expected: str,
    ) -> None:
        at = datetime.fromisoformat(instant)
        assert _mod().text_at(history, 'p1', at, 'third, current') == expected

    def test_an_edit_without_its_old_text_is_unrecoverable(self, history: Path) -> None:
        at = datetime.fromisoformat('2026-10-02T00:00:00+00:00')
        assert _mod().text_at(history, 'p3', at, 'lost') is None


class TestMemoryHistory:
    """MemoryHistory reads the UPDATE events once and answers as text_at does."""

    @pytest.mark.parametrize(('instant', 'expected'), [
        ('2026-10-02T12:00:00.000001+00:00', 'first'),
        ('2026-10-07T00:00:00+00:00', 'second'),
        ('2026-10-10T00:00:00+00:00', 'third, current'),
    ])
    def test_answers_as_text_at_does(self, history: Path, instant: str, expected: str) -> None:
        at = datetime.fromisoformat(instant)
        history_at = _mod()
        assert history_at.MemoryHistory.read(history).text_at('p1', at, 'third, current') == expected
        assert history_at.text_at(history, 'p1', at, 'third, current') == expected

    def test_an_edit_without_its_old_text_is_unrecoverable(self, history: Path) -> None:
        at = datetime.fromisoformat('2026-10-02T00:00:00+00:00')
        assert _mod().MemoryHistory.read(history).text_at('p3', at, 'lost') is None

    @pytest.mark.parametrize(('memory_id', 'instant', 'edited'), [
        ('p1', '2026-10-07T00:00:00+00:00', True),
        ('p1', '2026-10-10T00:00:00+00:00', False),
        ('p3', '2026-10-02T00:00:00+00:00', True),
        ('never-edited', '2026-10-01T00:00:00+00:00', False),
    ])
    def test_edited_after_is_whether_an_update_follows_the_instant(
        self, history: Path, memory_id: str, instant: str, edited: bool,
    ) -> None:
        at = datetime.fromisoformat(instant)
        assert _mod().MemoryHistory.read(history).edited_after(memory_id, at) is edited

    def test_a_history_read_for_some_memories_refuses_any_other(self, history: Path) -> None:
        partial = _mod().MemoryHistory.read(history, memory_ids=['p2'])
        at = datetime.fromisoformat('2026-10-02T00:00:00+00:00')
        assert partial.text_at('p2', at, 'other v2') == 'other'
        with pytest.raises(ValueError, match='p1'):
            partial.text_at('p1', at, 'third, current')
        with pytest.raises(ValueError, match='p1'):
            partial.edited_after('p1', at)
