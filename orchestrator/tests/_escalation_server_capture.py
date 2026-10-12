"""Read the kwargs ``Harness._start_escalation_server`` hands ``create_server``.

Lives outside conftest.py for the same reason as ``_orch_helpers.py``: a
root-level pytest run that loads several subprojects' conftests collides on
``sys.modules['conftest']``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from orchestrator.harness import Harness


async def create_server_kwargs(h: Harness) -> dict[str, Any]:
    """Drive ``_start_escalation_server`` with ``create_server`` patched, never serving.

    The server task is made to look alive, or the method's did-it-crash check
    re-raises from a ``MagicMock`` exception. Patching ``asyncio.create_task``
    leaves the serve coroutine built but never scheduled, so it is captured and
    closed: a coroutine GC'd un-awaited raises an unraisable exception that
    pytest pins on whichever test the xdist worker is running when the GC
    fires, a spurious failure in an unrelated file.
    """
    serve_task = MagicMock()
    serve_task.done.return_value = False
    created_coros: list = []

    def _capture_task(coro, **kwargs):
        created_coros.append(coro)
        return serve_task

    with patch('orchestrator.harness.create_server') as mock_create, \
         patch('asyncio.create_task', side_effect=_capture_task), \
         patch('asyncio.sleep', new=AsyncMock()):
        await h._start_escalation_server()

    for coro in created_coros:
        coro.close()

    assert mock_create.called, '_start_escalation_server did not reach create_server'
    return dict(mock_create.call_args.kwargs)
