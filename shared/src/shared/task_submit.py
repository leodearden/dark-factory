"""The typed outcome of one fused-memory ``submit_task`` call.

Every machine filer in the orchestrator (auto-eval, the flake ledger's de-flake
filing, chronic-flake filing, the offline lane's fix task) calls the same tool
and used to read a bare string out of its response with its own parser.  Those
parsers disagreed about the envelope (the production JSON-RPC body is two layers
deep) and about what counts as an id (a ``tkt_`` ticket is not a task).  This
module parses the response ONCE into a closed set of outcomes, so a caller
branches on a type instead of on the shape of a string:

* :class:`SubmittedTask` — ``planning_mode=True`` created a task synchronously,
  or (``combined=True``) the candidate-key index folded the submission into an
  existing task, whose id and status are what the server returned.
* :class:`SubmittedTicket` — the two-phase path accepted the submission and the
  curator will resolve it later.  No task exists yet, so no caller may store
  this id as a task id (PRD plans/offline-lane-red-path-prd.md D9).
* :class:`SubmitFailed` — the tool reported an error, or the response could not
  be read; ``reason`` carries the server's own text when it gave one.

The module also owns the predicates that tell those two id kinds apart,
:func:`is_ticket_id` and :func:`looks_like_task_id`, so ``resolve_ticket`` can
recognise a task id handed back where a ticket id was expected.

Lives in ``shared`` because ``orchestrator/src/orchestrator/flake_ledger.py``
depends on ``shared`` alone (see its ``FlakeLedgerTaskClient`` docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.mcp_envelope import parse_tool_result

__all__ = [
    'SubmitFailed',
    'SubmitOutcome',
    'SubmittedTask',
    'SubmittedTicket',
    'TICKET_ID_PREFIX',
    'is_ticket_id',
    'looks_like_task_id',
    'parse_submit_task_result',
]

TICKET_ID_PREFIX = 'tkt_'


def is_ticket_id(value: object) -> bool:
    """Return True when *value* is a two-phase ticket id (``tkt_…``), not a task id."""
    return isinstance(value, str) and value.startswith(TICKET_ID_PREFIX)


def looks_like_task_id(value: object) -> bool:
    """Return True when *value* parses as a non-negative integer task id.

    Accepts ``int`` directly, plus strings of ASCII digits (with surrounding
    whitespace trimmed). Rejects empty strings, negatives, non-numeric strings,
    and non-ASCII digits such as ``'²'``, which ``str.isdigit`` would accept.

    Used by ``resolve_ticket`` to short-circuit when a caller passes a numeric
    task id that came back from ``submit_task(planning_mode=True)`` — the
    planning-mode path returns the id synchronously, so the two-step
    submit/resolve dance does not apply, but agents may still call it blindly.
    """
    if isinstance(value, bool):
        # bool is a subclass of int in Python; reject explicitly.
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, str):
        stripped = value.strip()
        return stripped.isascii() and stripped.isdigit()
    return False


@dataclass(frozen=True)
class SubmittedTicket:
    ticket_id: str


@dataclass(frozen=True)
class SubmittedTask:
    task_id: str
    status: str | None
    combined: bool = False


@dataclass(frozen=True)
class SubmitFailed:
    reason: str


SubmitOutcome = SubmittedTicket | SubmittedTask | SubmitFailed


def parse_submit_task_result(result: Any) -> SubmitOutcome:
    """Classify a ``dispatch_tool('submit_task', ...)`` response.  Never raises.

    *result* is the JSON-RPC body the transport returns; the payload is read
    through ``shared/src/shared/mcp_envelope.py::parse_tool_result`` in whole-inner-dict mode,
    so a bare, already-unwrapped dict is a failure here, not a payload.
    """
    inner, err = parse_tool_result(result, None, dict)
    if err is not None:
        return SubmitFailed(f'unreadable submit_task response ({err.shape}): {err.payload_prefix}')
    assert inner is not None  # parse_tool_result invariant: (None, err) | (value, None)
    if 'error' in inner:
        return SubmitFailed(
            f'submit_task error={inner.get("error")!r}, error_type={inner.get("error_type")!r}'
        )
    ticket = inner.get('ticket')
    if is_ticket_id(ticket):
        return SubmittedTicket(ticket)
    task_id = inner.get('task_id')
    combined = bool(inner.get('combined', False))
    if isinstance(task_id, str) and task_id:
        status = inner.get('status')
        return SubmittedTask(task_id, status if isinstance(status, str) else None, combined)
    if combined:
        return SubmitFailed('submit_task combined the submission into a task it did not name')
    return SubmitFailed(f'submit_task response names no task id (keys={sorted(map(str, inner))!r})')
