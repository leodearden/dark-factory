"""The one home of the SQLite durability policy both ``shared`` pragma helpers apply.

``shared.async_sqlite_base`` and ``shared.sqlite_sync_base`` each loop over
:data:`DURABILITY_TRIAD` and call :func:`require_wal_journal_mode`.  This leaf
is stdlib-only so ``shared.sqlite_sync_base`` loads without aiosqlite.  See
``docs/task-recovery-2026-05-13/`` for the production incident behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

__all__ = ['DURABILITY_TRIAD', 'SqlitePragma', 'require_wal_journal_mode']


class SqlitePragma(NamedTuple):
    """One pragma setting; ``value`` is the value ``PRAGMA <name>`` reads back."""

    name: str
    value: int

    @property
    def statement(self) -> str:
        return f'PRAGMA {self.name}={self.value}'


DURABILITY_TRIAD: tuple[SqlitePragma, ...] = (
    SqlitePragma('synchronous', 2),  # FULL: fsync per commit, ~1-5ms each, no reliance on checkpoint timing
    SqlitePragma('wal_autocheckpoint', 100),  # pages
    SqlitePragma('journal_size_limit', 64 * 1024 * 1024),  # bytes
)


def require_wal_journal_mode(row: Sequence[object] | None) -> None:
    """Raise unless ``row``, the result of ``PRAGMA journal_mode=WAL``, reads back 'wal'."""
    if row is None or row[0] != 'wal':
        got = row[0] if row is not None else None
        raise RuntimeError(f'Failed to enable WAL journal mode (got {got!r})')
