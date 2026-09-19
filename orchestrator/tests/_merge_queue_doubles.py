"""The merge-queue double that stands in for the merge WORKER.

Imported by bare module name (``from _merge_queue_doubles import ...``), like
``_orch_helpers`` -- ``orchestrator/tests/`` has no ``__init__.py``.

Distinct in purpose from ``_merge_lane_fakes``, which fakes the lane's PORTS
(verifier, clock) for tests that drive the worker itself.  This module serves
the tests on the other side of the queue: a workflow submits a merge request
and needs an outcome to come back with no worker running.
"""
from __future__ import annotations

import asyncio
from typing import Any

from orchestrator.merge_queue import MergeOutcome


class ResolvingMergeQueue(asyncio.Queue):
    """A real ``asyncio.Queue`` that also plays the part of the merge WORKER.

    Each request put on it is genuinely enqueued -- ``qsize``/``get_nowait``
    assertions still read the real item -- and its ``result`` future is then
    resolved with *outcome*, exactly as the worker would.  That is what lets a
    workflow reach its outcome through the real
    ``register_and_enqueue_merge_request`` -> ``enqueue_merge_request`` ->
    ``_await_cancellable`` chain, rather than through a monkeypatch that
    replaces a link of it (task 5027 gamma4).

    ``outcome=None`` leaves the future pending, which is what a test needs when
    something OTHER than the worker must win the ``_await_cancellable`` race --
    a preset soft-cancel event, say.
    """

    def __init__(self, outcome: MergeOutcome | None = None) -> None:
        super().__init__()
        self.outcome = outcome

    async def put(self, item: Any) -> None:
        await super().put(item)
        if self.outcome is not None and not item.result.done():
            item.result.set_result(self.outcome)
