#!/usr/bin/env python3
"""Read-only access to mem0's ``history.db``, and a memory's text at an instant.

mem0 records every edit of a memory as an UPDATE event keeping the old text,
so the text a memory read at an instant is the old text of its earliest
UPDATE after that instant, or its current text when no edit followed. Every
connection is opened read-only (:func:`read_only`), so nothing here can
change the store it reads.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path


def read_only(database: Path) -> closing[sqlite3.Connection]:
    """A read-only connection to *database*, with temp storage kept in memory.

    Any sqlite file opens this way, not only mem0's history.db; the soak
    builder opens the write journal with it too.
    """
    connection = sqlite3.connect(f'{Path(database).resolve().as_uri()}?mode=ro', uri=True)
    connection.execute('PRAGMA temp_store=MEMORY')
    return closing(connection)


class MemoryHistory:
    """The UPDATE events of mem0's history table, read once from the read-only database."""

    def __init__(
        self, updates: Mapping[str, Sequence[tuple[datetime, str | None]]],
        covers: frozenset[str] | None,
    ) -> None:
        self._updates = updates
        self._covers = covers

    @classmethod
    def read(cls, history_db: Path, memory_ids: Iterable[str] | None = None) -> MemoryHistory:
        """Every memory's UPDATE events, or only those of *memory_ids*, which it then alone answers for."""
        covers = None if memory_ids is None else frozenset(memory_ids)
        updates: defaultdict[str, list[tuple[datetime, str | None]]] = defaultdict(list)
        with read_only(history_db) as connection:
            rows = connection.execute(
                "SELECT memory_id, old_memory, updated_at FROM history WHERE event = 'UPDATE'"
            ).fetchall()
        for memory_id, old, at in rows:
            if at and (covers is None or memory_id in covers):
                updates[memory_id].append((datetime.fromisoformat(at), old))
        return cls(updates, covers)

    def text_at(self, memory_id: str, instant: datetime, current_text: str) -> str | None:
        """A memory's text at *instant*.

        The old text of the earliest UPDATE after *instant*; *current_text* when
        the memory was not edited after it; None when that UPDATE kept no old text.
        """
        later = self._edits_after(memory_id, instant)
        if not later:
            return current_text
        return min(later, key=lambda edit: edit[0])[1]

    def edited_after(self, memory_id: str, instant: datetime) -> bool:
        """Whether an UPDATE of the memory follows *instant*."""
        return bool(self._edits_after(memory_id, instant))

    def _edits_after(self, memory_id: str, instant: datetime) -> list[tuple[datetime, str | None]]:
        if self._covers is not None and memory_id not in self._covers:
            raise ValueError(f'this history was read for {len(self._covers)} memories, not {memory_id!r}')
        return [(at, old) for at, old in self._updates.get(memory_id, ()) if at > instant]


def text_at(history_db: Path, memory_id: str, instant: datetime, current_text: str) -> str | None:
    """:meth:`MemoryHistory.text_at` for one memory, read from *history_db*."""
    return MemoryHistory.read(history_db, [memory_id]).text_at(memory_id, instant, current_text)
