"""Drive ``ReconciliationHarness``'s escalation server: read the kwargs it hands ``create_escalation_server``, or serve it for a block.

The fused-memory twin of ``orchestrator/tests/_escalation_server_capture.py``:
the harness-private seams patched or called to reach that server are named
here only.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

from _embedded_uvicorn_fixtures import ReadySignallingServer

if TYPE_CHECKING:
    from starlette.types import ASGIApp

    from fused_memory.reconciliation.harness import ReconciliationHarness

_HARNESS_MODULE = 'fused_memory.reconciliation.harness'


async def create_escalation_server_kwargs(harness: ReconciliationHarness) -> dict[str, Any]:
    """Start and stop *harness*'s escalation server with the factory patched, never serving."""
    with (
        patch(
            f'{_HARNESS_MODULE}.create_escalation_server',
            return_value=MagicMock(),
        ) as mock_create,
        patch(f'{_HARNESS_MODULE}._sleep', AsyncMock()),
        patch(f'{_HARNESS_MODULE}.HAS_ESCALATION', True),
    ):
        try:
            await harness._start_escalation_server()
        finally:
            await harness._stop_escalation_server()
    assert mock_create.called, '_start_escalation_server did not reach create_escalation_server'
    return dict(mock_create.call_args.kwargs)


@contextlib.asynccontextmanager
async def serving_escalation_server(
    harness: ReconciliationHarness,
    *,
    app: ASGIApp | None = None,
    stop_bound_secs: float | None = None,
) -> AsyncIterator[None]:
    """Serve *harness*'s escalation server for the block, then stop it the way the harness does.

    The block begins once uvicorn's startup has completed, not after the
    harness's settle sleep. *app*, if given, is served in place of the
    escalation MCP app; *stop_bound_secs*, if given, replaces the bound on
    the graceful stop.
    """
    stop_overrides = {} if stop_bound_secs is None else {'stop_bound_secs': stop_bound_secs}
    try:
        with contextlib.ExitStack() as patches:
            patches.enter_context(patch(f'{_HARNESS_MODULE}.EmbeddedUvicornServer', ReadySignallingServer))
            patches.enter_context(patch(f'{_HARNESS_MODULE}._sleep', AsyncMock()))
            if app is not None:
                patches.enter_context(
                    patch(
                        f'{_HARNESS_MODULE}.create_escalation_server',
                        return_value=MagicMock(http_app=MagicMock(return_value=app)),
                    )
                )
            await harness._start_escalation_server()
        server = harness._escalation_server
        assert isinstance(server, ReadySignallingServer), '_start_escalation_server built no escalation server'
        await server.until_started()
        yield
    finally:
        await harness._stop_escalation_server(**stop_overrides)
