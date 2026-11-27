"""A task body that survives every cancel, for tests of a wait that must stay bounded."""

from __future__ import annotations

import asyncio


async def ignore_cancels_until(release: asyncio.Event) -> None:
    """Swallow every cancel until *release* is set, then return."""
    while not release.is_set():
        try:
            await release.wait()
        except asyncio.CancelledError:
            continue
