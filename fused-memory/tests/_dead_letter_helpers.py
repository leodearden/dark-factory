"""Dead-letter fixtures shared by the writer, EventQueue and MCP-tool tests.

``seed_dead_letters`` writes records in the on-disk format that reading,
counting and replay consume. ``hold_fifo`` makes a dead-letter path whose
appends cannot finish until the test releases it, which is how a test proves
an append ran off the caller's thread. ``parked_queue`` is its drainer-side
counterpart: a running EventQueue whose drainer cannot commit anything until
the block exits, so ``journaled_events`` and ``journaled_ids`` (what a fresh
journal reader still lists) show the settled-at-enqueue state without racing
the drain. ``overflowed_queue`` is an unstarted EventQueue of capacity one
whose second event has already overflowed to the dead-letter writer.
``record_dead_letter_writes`` spies on ``DeadLetterWriter.append`` and hands
the test every write's future, so it can wait until a write has resolved.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from fused_memory.reconciliation.dead_letter_writer import DeadLetterWriter
from fused_memory.reconciliation.event_journal import EventJournal
from fused_memory.reconciliation.event_queue import EventQueue

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from concurrent.futures import Future
    from typing import Any

    import pytest

    from fused_memory.models.reconciliation import ReconciliationEvent


def seed_dead_letters(path: Path, events: list[ReconciliationEvent]) -> None:
    """Append ``events`` to ``path`` as ``non_retriable`` dead-letter records."""
    with path.open('a', encoding='utf-8') as fh:
        for event in events:
            record = {
                'event': event.model_dump(mode='json'),
                'reason': 'non_retriable',
                'attempts': 1,
                'failed_at': datetime.now(UTC).isoformat(),
            }
            fh.write(json.dumps(record) + '\n')


@dataclass
class HeldFifo:
    """A FIFO whose only reader stays away until ``release`` is set.

    ``open('a')`` on a FIFO blocks until a reader opens it, so an append to
    ``path`` cannot finish while the FIFO is held. The reader also gives up
    waiting after 5s, so an append that blocks its caller fails the
    ``reader_opened`` assertion instead of hanging the test.
    """

    path: Path
    release: threading.Event = field(default_factory=threading.Event)
    reader_opened: threading.Event = field(default_factory=threading.Event)
    received: bytearray = field(default_factory=bytearray)
    reader: threading.Thread | None = None

    def start(self) -> HeldFifo:
        os.mkfifo(self.path)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        return self

    def _read(self) -> None:
        self.release.wait(timeout=5.0)
        self.reader_opened.set()
        with self.path.open('rb') as fh:
            self.received.extend(fh.read())

    def records(self) -> list[dict]:
        """Wait for the reader to reach EOF, then parse every line it received."""
        assert self.reader is not None
        self.reader.join(timeout=5.0)
        assert not self.reader.is_alive(), 'FIFO reader never saw EOF'
        return [json.loads(line) for line in self.received.decode('utf-8').splitlines()]


def hold_fifo(path: Path) -> HeldFifo:
    return HeldFifo(path).start()


def record_dead_letter_writes(monkeypatch: pytest.MonkeyPatch) -> list[Future[bool]]:
    """Every future ``DeadLetterWriter.append`` returns from now on, in call order."""
    writes: list[Future[bool]] = []
    real_append = DeadLetterWriter.append

    def append(
        self: DeadLetterWriter,
        record: Mapping[str, Any],
        *,
        on_settled: Callable[[bool], None] | None = None,
    ) -> Future[bool]:
        written = real_append(self, record, on_settled=on_settled)
        writes.append(written)
        return written

    monkeypatch.setattr(DeadLetterWriter, 'append', append)
    return writes


@asynccontextmanager
async def overflowed_queue(
    dead_letter_path: Path,
    accepted: ReconciliationEvent,
    overflowed: ReconciliationEvent,
    *,
    journal_path: Path | None = None,
) -> AsyncIterator[EventQueue]:
    """An unstarted EventQueue of capacity one that holds ``accepted`` and has
    handed ``overflowed`` to its dead-letter writer. Leaving the block closes it.
    """
    queue = EventQueue(
        AsyncMock(), dead_letter_path=dead_letter_path, journal_path=journal_path, maxsize=1,
    )
    try:
        assert queue.enqueue(accepted) is True
        assert queue.enqueue(overflowed) is False
        yield queue
    finally:
        await queue.close()


@asynccontextmanager
async def parked_queue(dead_letter_path: Path, journal_path: Path) -> AsyncIterator[EventQueue]:
    """A started EventQueue whose drainer parks in ``push`` until the block exits.

    Leaving the block releases the drainer before closing the queue, so
    close() drains rather than timing out and dead-lettering what is left.
    """
    released = asyncio.Event()

    async def park(_event: ReconciliationEvent) -> None:
        await released.wait()

    buffer = AsyncMock()
    buffer.push = AsyncMock(side_effect=park)
    queue = EventQueue(buffer, dead_letter_path=dead_letter_path, journal_path=journal_path)
    await queue.start()
    try:
        yield queue
    finally:
        released.set()
        await queue.close()


def journaled_events(journal_path: Path) -> dict[str, ReconciliationEvent]:
    """Events a FRESH journal over ``journal_path`` still lists as unprocessed, by id."""
    journal = EventJournal(journal_path)
    try:
        return {event.id: event for event in journal.load_unprocessed()}
    finally:
        journal.close()


def journaled_ids(journal_path: Path) -> set[str]:
    return set(journaled_events(journal_path))
