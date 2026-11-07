"""Read the kwargs ``ReconciliationHarness._start_escalation_server`` hands ``create_escalation_server``.

The fused-memory twin of ``orchestrator/tests/_escalation_server_capture.py``:
the harness-private seams patched to reach that call are named here only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from fused_memory.reconciliation.harness import ReconciliationHarness

_HARNESS_MODULE = 'fused_memory.reconciliation.harness'


async def create_escalation_server_kwargs(harness: ReconciliationHarness) -> dict[str, Any]:
    """Start and stop *harness*'s escalation server with the factory patched, never serving."""
    with (
        patch(
            f'{_HARNESS_MODULE}.create_escalation_server',
            return_value=MagicMock(run_http_async=AsyncMock()),
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
