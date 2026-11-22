"""One call to the OpenAI Responses API, the token-usage reader and the endpoint-host rule.

The callers (``server/write_triage_judge.py`` and
``reconciliation/consolidation_adjudicator.py``) own every policy: what an
incomplete answer means, and what an SDK exception means. This module raises
nothing of its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, TypeGuard
from urllib.parse import urlsplit


class TokenUsage(NamedTuple):
    """What the provider reported for one call. Output includes reasoning."""

    input_tokens: int
    output_tokens: int
    #: ``None`` when the provider does not report reasoning separately.
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ResponsesReply:
    """One Responses answer, as data.

    An ``incomplete`` answer is still returned: each caller decides what it means.
    """

    text: str | None
    incomplete: bool
    incomplete_reason: str | None
    #: The model the provider says answered, which may be a dated snapshot.
    resolved_model: str | None
    usage: TokenUsage | None


class ResponsesCall(Protocol):
    """:func:`call_openai_responses` with its client kwargs already bound."""

    async def __call__(
        self,
        *,
        model: str,
        instructions: str,
        user_content: str,
        text_format: Mapping[str, Any],
        max_output_tokens: int,
        timeout: float,
        reasoning_effort: str | None,
    ) -> ResponsesReply: ...


def _is_token_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def read_token_usage(
    usage: object, *, input_field: str, output_field: str, details_field: str | None,
) -> TokenUsage | None:
    """Read a provider's *usage* object as a :class:`TokenUsage`, or ``None``.

    Counts are taken only when they are real ``int``s — never coerced, because
    ``int()`` of an unspecced Mock is 1 and a ``bool`` is an ``int``. Never
    raises: usage is metadata about a call that already succeeded.
    """
    input_tokens = getattr(usage, input_field, None)
    output_tokens = getattr(usage, output_field, None)
    if not (_is_token_count(input_tokens) and _is_token_count(output_tokens)):
        return None
    reasoning_tokens = None
    if details_field is not None:
        details = getattr(usage, details_field, None)
        reasoning_tokens = getattr(details, 'reasoning_tokens', None)
    return TokenUsage(
        input_tokens,
        output_tokens,
        reasoning_tokens if _is_token_count(reasoning_tokens) else None,
    )


def endpoint_host(base_url: str | None) -> str | None:
    """``host[:port]`` of *base_url* without credentials, or None for the SDK default."""
    if not base_url:
        return None
    return urlsplit(base_url).netloc.rpartition('@')[2]


async def call_openai_responses(
    client_kwargs: Mapping[str, Any],
    *,
    model: str,
    instructions: str,
    user_content: str,
    text_format: Mapping[str, Any],
    max_output_tokens: int,
    timeout: float,
    reasoning_effort: str | None,
    max_retries: int = 0,
) -> ResponsesReply:
    """One Responses call on a client built and closed for this call alone.

    Exactly one sampling control is sent: a set *reasoning_effort* goes out as
    ``reasoning`` with no ``temperature``, which reasoning models reject;
    unset means a non-reasoning model pinned to ``temperature=0.0``.

    A per-call client keeps a hot-reloaded key or endpoint from being pinned
    by a cached one, and ``async with`` around ``asyncio.wait_for`` closes the
    connection pool even when the timeout cancels the request. Every
    exception propagates unchanged; a call past *timeout* raises TimeoutError.
    ``openai.AsyncOpenAI`` is looked up at call time.
    """
    import openai

    sampling: dict[str, Any] = (
        {'temperature': 0.0}
        if reasoning_effort is None
        else {'reasoning': {'effort': reasoning_effort}}
    )
    async with openai.AsyncOpenAI(**client_kwargs, max_retries=max_retries) as client:
        response = await asyncio.wait_for(
            client.responses.create(
                model=model,
                instructions=instructions,
                input=user_content,
                max_output_tokens=max_output_tokens,
                text={'format': text_format},
                **sampling,
            ),
            timeout=timeout,
        )
    resolved_model = getattr(response, 'model', None)
    return ResponsesReply(
        text=response.output_text,
        incomplete=getattr(response, 'status', None) == 'incomplete',
        incomplete_reason=getattr(
            getattr(response, 'incomplete_details', None), 'reason', None,
        ),
        resolved_model=resolved_model if isinstance(resolved_model, str) else None,
        usage=read_token_usage(
            getattr(response, 'usage', None),
            input_field='input_tokens',
            output_field='output_tokens',
            details_field='output_tokens_details',
        ),
    )
