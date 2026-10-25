"""Submit a batch of fused-memory ``submit_task`` calls and report what each one became."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shared.mcp_envelope import write_failure_reason
from shared.mcp_post import call_mcp_tool, open_mcp_client

logger = logging.getLogger(__name__)

_NO_REPLY = 'no tool reply; see the shared.mcp_post / shared.mcp_envelope warning'
_CLIP_CHARS = 200


@dataclass(frozen=True)
class SubmitOutcome:
    """One submission's result: the curator ticket it holds, or why it holds none."""

    title: str
    ticket: str | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        if (self.ticket is None) == (self.failure is None):
            raise ValueError(
                'a SubmitOutcome holds exactly one of ticket and failure: '
                f'title={self.title!r} ticket={self.ticket!r} failure={self.failure!r}'
            )


@dataclass(frozen=True)
class SubmitBatchResult:
    outcomes: tuple[SubmitOutcome, ...]

    @property
    def tickets(self) -> tuple[str, ...]:
        return tuple(o.ticket for o in self.outcomes if o.ticket is not None)

    @property
    def failures(self) -> tuple[SubmitOutcome, ...]:
        return tuple(o for o in self.outcomes if o.failure is not None)

    @property
    def all_delivered(self) -> bool:
        return not self.failures


async def submit_task_batch(
    base_url: str,
    arguments_list: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> SubmitBatchResult:
    """Send every ``submit_task`` over one client; never raises, warns once per failure."""
    outcomes: list[SubmitOutcome] = []
    try:
        async with open_mcp_client() as client:
            for arguments in arguments_list:
                outcomes.append(
                    await _submit_one(client, base_url, arguments, context=context),
                )
    except Exception as exc:
        reason = f'{type(exc).__name__}: {exc}'
        outcomes.extend(
            SubmitOutcome(title=_title_of(arguments), failure=reason)
            for arguments in arguments_list[len(outcomes):]
        )
    for outcome in outcomes:
        if outcome.failure is not None:
            logger.warning(
                '%s: submit_task for %r holds no curator ticket: %s',
                context, outcome.title, outcome.failure,
            )
    return SubmitBatchResult(outcomes=tuple(outcomes))


async def _submit_one(
    client: Any, base_url: str, arguments: Mapping[str, Any], *, context: str,
) -> SubmitOutcome:
    title = _title_of(arguments)
    try:
        reply = await call_mcp_tool(
            client, base_url, 'submit_task', dict(arguments), context=context,
        )
    except Exception as exc:
        return SubmitOutcome(title=title, failure=f'{type(exc).__name__}: {exc}')
    return _outcome_of_reply(title, reply)


def _outcome_of_reply(title: str, reply: dict | None) -> SubmitOutcome:
    if reply is None:
        return SubmitOutcome(title=title, failure=_NO_REPLY)
    refusal = write_failure_reason(reply)
    if refusal is not None:
        return SubmitOutcome(title=title, failure=refusal)
    ticket = reply.get('ticket')
    if isinstance(ticket, str):
        return SubmitOutcome(title=title, ticket=ticket)
    return SubmitOutcome(
        title=title, failure=f'reply carried no ticket: {repr(reply)[:_CLIP_CHARS]}',
    )


def _title_of(arguments: Mapping[str, Any]) -> str:
    return str(arguments.get('title', ''))
