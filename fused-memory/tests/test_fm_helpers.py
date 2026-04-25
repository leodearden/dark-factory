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
# Guard tests — submit_task return-shape routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_and_resolve_proceeds_when_ticket_present_regardless_of_other_keys():
    """submit_and_resolve must route on 'ticket' presence, not 'error' absence.

    Forward-compatibility test: a future evolution of submit_task may return a success dict
    that also carries a non-fatal advisory field (e.g. {'ticket': ..., 'error': 'non-fatal
    warning'}).  The helper must use the *presence of 'ticket'* — not the *absence of 'error'*
    — to decide whether to proceed to resolve_ticket.  An 'error'-absent guard would silently
    return the success-with-warning dict verbatim and never resolve the ticket.
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'non_dict_value',
    [None, [], 'err', 42],
    ids=['none', 'list', 'str', 'int'],
)
async def test_submit_and_resolve_asserts_when_submit_result_is_non_dict(non_dict_value):
    """submit_and_resolve must raise AssertionError (not silently return) when submit_task returns a non-dict.

    submit_task is contract-bound to always return a dict; a non-dict indicates a regression in
    the production code.  The helper is test-only, so a loud AssertionError is preferred over
    silent pass-through that would mask the underlying bug.

    The error message must:
    - contain the structural marker 'submit_task returned non-dict' for easy grepping, and
    - contain repr(submit_result) so the diagnostician can see exactly what was returned.

    Execution must blow up BEFORE any resolve_ticket call.
    """
    interceptor = _make_stub_interceptor(
        submit_result=non_dict_value,  # type: ignore[arg-type]
        resolve_result={},
        ticket_store_row=None,
    )

    with pytest.raises(AssertionError) as excinfo:
        await submit_and_resolve(interceptor, '/project', title='T')

    message = str(excinfo.value)
    assert 'submit_task returned non-dict' in message, (
        f"structural marker not in error message: {message!r}"
    )
    assert repr(non_dict_value) in message, (
        f"repr of non-dict value {non_dict_value!r} not in error message: {message!r}"
    )
    interceptor.resolve_ticket.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_and_resolve_returns_submit_error_when_no_ticket():
    """submit_and_resolve must return the submit-error dict verbatim and NOT call resolve_ticket.

    This is the current-production behavior: submit_task returns an error dict with no 'ticket'
    key (e.g. backlog gate rejection or closed-server error) and the helper passes it through
    immediately.  All callers in test_task_interceptor.py and test_backlog_policy.py assert on
    result.get('error') / result.get('error_type') in this path.
    """
    submit_result = {'error': 'closed', 'error_type': 'ClosedError'}
    interceptor = _make_stub_interceptor(
        submit_result=submit_result,
        resolve_result={},
        ticket_store_row=None,
    )

    result = await submit_and_resolve(interceptor, '/project', title='T')

    assert result == submit_result, f'expected submit-error dict {submit_result!r}, got {result!r}'
    interceptor.resolve_ticket.assert_not_awaited()


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
    # resolve_result must be dumped verbatim so the diagnostician can see what the worker returned.
    # Check for the structural marker 'resolve_result=' (from the format string) AND 'failed'
    # (from resolve_result['status']), rather than a coincidental inner-value substring.
    assert 'resolve_result=' in message and 'failed' in message, (
        f"resolve_result not surfaced in error message: {message!r}"
    )
