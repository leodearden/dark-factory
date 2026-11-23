"""Parking fixtures for tests that need a DurableWriteQueue holding parked rows.

Imported by name, like ``_fm_polling``. ``Exhausted`` is a failure only a human
clears, ``RecordingParkPolicy`` parks exactly that failure and records each
hook call, and ``queue_holding_parked_rows`` yields a running queue whose every
write has parked.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock

from _fm_polling import poll_until
from _queued_write_fixtures import queued

from fused_memory.services.durable_queue import DurableWriteQueue, ParkedWriteEvent


class Exhausted(Exception):
    """Stands in for a refusal that only a human can clear."""


@dataclass
class RecordingParkPolicy:
    parked: list[ParkedWriteEvent] = field(default_factory=list)
    cleared: int = 0
    raise_on_parked: bool = False
    raise_on_cleared: bool = False

    def park_reason(self, exc: BaseException) -> str | None:
        return 'test-park' if isinstance(exc, Exhausted) else None

    async def on_parked(self, event: ParkedWriteEvent) -> None:
        self.parked.append(event)
        if self.raise_on_parked:
            raise RuntimeError('on_parked hook failed')

    async def on_cleared(self) -> None:
        self.cleared += 1
        if self.raise_on_cleared:
            raise RuntimeError('on_cleared hook failed')


@asynccontextmanager
async def queue_holding_parked_rows(
    tmp_path: Path, *, rows: int, group_id: str = 'proj1',
) -> AsyncIterator[DurableWriteQueue]:
    """A running queue in which *rows* writes to *group_id* have all parked.

    The re-probe cadence outlasts any test, so the rows stay parked until the
    queue closes on exit.
    """
    q = DurableWriteQueue(
        data_dir=tmp_path / 'parked_queue',
        execute_write=AsyncMock(side_effect=Exhausted('no credits remaining')),
        workers_per_group=1,
        retry_max_delay_seconds=60.0,
        park_policy=RecordingParkPolicy(),
    )
    await q.initialize()
    try:
        for n in range(rows):
            await q.enqueue(queued(
                'add_episode',
                {'content': f'parks {n}', 'group_id': group_id, 'name': 'ep'},
                group_id=group_id,
            ))

        async def all_parked() -> bool:
            return (await q.get_stats())['counts'].get('parked', 0) == rows

        await poll_until(all_parked, message=f'{rows} write(s) did not all park')
        yield q
    finally:
        await q.close()
