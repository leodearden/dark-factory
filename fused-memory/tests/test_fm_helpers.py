"""Tests for the submit_and_resolve helper in _fm_helpers.py."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import submit_and_resolve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_interceptor(
    *,
    submit_result: dict,
    resolve_result: dict,
    ticket_store_row,
) -> MagicMock:
    """Return a minimal MagicMock interceptor wired for submit_and_resolve tests."""
    interceptor = MagicMock()
    interceptor.submit_task = AsyncMock(return_value=submit_result)
    interceptor.resolve_ticket = AsyncMock(return_value=resolve_result)
    interceptor._ticket_store = MagicMock()
    interceptor._ticket_store.get = AsyncMock(return_value=ticket_store_row)
    return interceptor


# ---------------------------------------------------------------------------
# Step-1: guard test — success-dict carrying a non-fatal 'error' field
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_and_resolve_proceeds_when_submit_result_carries_nonfatal_error_with_ticket():
    """submit_and_resolve must not bail out on a success-with-warning submit_result.

    A future evolution of submit_task may return {'ticket': ..., 'error': 'non-fatal warning'}.
    The helper should use the presence of 'ticket' (not the absence of 'error') to decide
    whether to proceed.  Currently the guard `if 'error' in submit_result` bails out early,
    which is why this test is RED until step 2 fixes the guard.
    """
    expected = {'id': '99', 'title': 'Done'}
    interceptor = _make_stub_interceptor(
        submit_result={'ticket': 'tkt_x', 'error': 'non-fatal warning'},
        resolve_result={'status': 'created', 'task_id': '99'},
        ticket_store_row={'result_json': json.dumps(expected)},
    )

    result = await submit_and_resolve(interceptor, '/project', title='T')

    # Must have proceeded past the guard and returned the parsed result_json
    assert result == expected, f'expected {expected!r}, got {result!r}'
    interceptor.resolve_ticket.assert_awaited_once()


# ---------------------------------------------------------------------------
# Step-3: fallback test — helper must raise AssertionError when no result_json
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    'row_value',
    [None, {}, {'result_json': ''}],
    ids=['row-none', 'row-empty', 'result-json-empty'],
)
async def test_submit_and_resolve_raises_when_no_result_json(row_value):
    """submit_and_resolve must raise AssertionError (not silently return resolve_result) when
    the worker never wrote a result_json.

    The AssertionError message must name the ticket id and include enough of resolve_result
    to diagnose the root cause.  Currently fails because line 178 silently returns resolve_result.
    """
    resolve_result = {'status': 'failed', 'reason': 'timeout', 'task_id': None}
    interceptor = _make_stub_interceptor(
        submit_result={'ticket': 'tkt_y'},
        resolve_result=resolve_result,
        ticket_store_row=row_value,
    )

    with pytest.raises(AssertionError) as excinfo:
        await submit_and_resolve(interceptor, '/project', title='T')

    message = str(excinfo.value)
    assert 'tkt_y' in message, f"ticket id 'tkt_y' not in error message: {message!r}"
    # resolve_result must be surfaced so the diagnostician can see what the worker returned
    assert 'timeout' in message, f"resolve_result evidence not in error message: {message!r}"
