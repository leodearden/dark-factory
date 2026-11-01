"""Dead-letter fixtures shared by the writer, EventQueue and MCP-tool tests.

``seed_dead_letters`` writes records in the on-disk format that reading,
counting and replay consume. ``hold_fifo`` makes a dead-letter path whose
appends cannot finish until the test releases it, which is how a test proves
an append ran off the caller's thread.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
