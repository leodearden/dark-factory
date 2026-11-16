"""Awaiting a backend call under a time budget — the one timeout policy for every store.

A backend timeout PROPAGATES: it is raised as ``TimeoutError`` and never
degraded into a falsy return.  A timed-out read that returns ``[]`` or ``None``
is indistinguishable from a genuinely empty store or a genuine not-found
(INV-11 ``no-silent-fail-soft``).

It also NAMES the store, the operation and the budget it overran.
``asyncio.wait_for``'s own ``TimeoutError`` stringifies empty, and this text is
what reaches an operator's log and
``services/memory_service.py::_store_failure_diagnostics``'s ``error`` field —
so the message is a contract, and this module is its only home.

It is also the home of the socket-deadline rule: a client's own socket timeout
must outlast every budget its calls run under (``socket_timeout_outlasting``).
"""

import asyncio
import math
from collections.abc import Awaitable
from typing import TypeVar

_T = TypeVar('_T')


async def await_within(awaitable: Awaitable[_T], *, store: str, op: str, budget_s: float) -> _T:
    """Await *awaitable* within *budget_s* seconds, or raise a ``TimeoutError`` naming it.

    *op* is the operation as a reader grepping a log would look for it.
    *budget_s* is echoed into the message, so it can never name a budget other
    than the one applied.  Any other exception passes through untouched.
    """
    try:
        return await asyncio.wait_for(awaitable, timeout=budget_s)
    except TimeoutError as exc:
        raise TimeoutError(f'{store} {op} timed out after {budget_s}s') from exc


def socket_timeout_outlasting(*budgets_s: float) -> int:
    """The smallest whole second strictly above every one of *budgets_s*.

    A client's socket timeout must outlast every ``await_within`` budget its
    calls run under, so a stalled call fails on its NAMED budget and never on an
    anonymous socket error first; it is whole seconds because qdrant-client's
    ``timeout`` is an ``int``.
    """
    return math.floor(max(budgets_s)) + 1
