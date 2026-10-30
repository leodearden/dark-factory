"""HeldWrites: a DurableWriteQueue ``execute_write`` fake that holds writes open.

Imported by name (``from _fm_held_writes import HeldWrites``), like
``_fm_helpers``, so test files that drive a real queue with writes still
executing share one fake of the execute_write contract.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime


class HeldWrites:
    """An execute_write fake that holds each write, keyed by its payload ``name``, open until released.

    Entry is stamped with ``datetime.now(UTC)``, standing in for the
    ``utc_now()`` that graphiti_core's add_episode takes as the episode's
    created_at. With ``fail_on_release`` a released write raises instead of
    returning, so every retried attempt of it raises at once.
    """

    def __init__(self, *, fail_on_release: bool = False) -> None:
        self.entered_at: dict[str, datetime] = {}
        self._releases: dict[str, asyncio.Event] = {}
        self._fail_on_release = fail_on_release

    def _release_event(self, name: str) -> asyncio.Event:
        return self._releases.setdefault(name, asyncio.Event())

    async def execute(self, operation: str, payload: dict) -> dict:
        name = payload['name']
        self.entered_at[name] = datetime.now(UTC)
        await self._release_event(name).wait()
        if self._fail_on_release:
            raise RuntimeError(f'write {name} failed')
        return {'episode_uuid': f'ep-{name}'}

    def release(self, name: str) -> None:
        self._release_event(name).set()

    def release_all(self) -> None:
        for event in self._releases.values():
            event.set()
