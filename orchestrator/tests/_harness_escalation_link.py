"""Link a test Harness to a real EscalationQueue, as Harness startup does.

``Harness`` exposes no public seam for this: startup assigns its own queue
and registers ``_on_escalation_resolved`` as the queue's resolve callback,
which schedules the resolution's effect (a restart teardown, say) as a
harness background task. A test resolving through the real ``resolve_issue``
tool links the two here, then drains those background tasks before asserting.
"""

from __future__ import annotations

import asyncio
from typing import Any

from escalation.queue import EscalationQueue


def link_escalation_queue(harness: Any, queue: EscalationQueue) -> None:
    harness._escalation_queue = queue
    queue.set_resolve_callback(harness._on_escalation_resolved)


async def drain_background_tasks(harness: Any) -> None:
    """Await every background task, including those scheduled while draining."""
    background = harness._background_tasks
    while background:
        await asyncio.gather(*list(background))
