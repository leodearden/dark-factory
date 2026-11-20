"""Tell an exhausted provider credit balance apart from a transient rate limit.

Both arrive as an HTTP 429 ``RateLimitError``. A transient rate limit clears
by itself within seconds; an exhausted balance (OpenAI's
``insufficient_quota``) clears only when a human pays, so retrying into it
until the retry budget runs out turns every write into a dead letter.

graphiti_core re-raises the provider's error as its own unstructured
``RateLimitError`` (``raise RateLimitError from e``), so the classifier walks
the cause chain. A structured verdict (``code``, ``type``, ``body`` or
``status_code``) anywhere in the chain is trusted over a message substring.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class BillingSource(StrEnum):
    """Which paid allowance ran out."""

    API_KEY_CREDITS = 'api_key_credits'


@dataclass(frozen=True)
class _SourceFacts:
    provider: str
    key_scope: str


_FACTS_BY_SOURCE: Mapping[BillingSource, _SourceFacts] = MappingProxyType({
    BillingSource.API_KEY_CREDITS: _SourceFacts(
        provider='openai',
        key_scope=(
            'OPENAI_API_KEY (fused-memory/config/config.yaml '
            'llm.providers.openai.api_key and embedder.providers.openai.api_key)'
        ),
    ),
})

_SOURCE_BY_EXHAUSTION_CODE: Mapping[str, BillingSource] = MappingProxyType({
    'insufficient_quota': BillingSource.API_KEY_CREDITS,
})

_MAX_CHAIN_DEPTH = 8
_PROVIDER_MESSAGE_CHARS = 300


@dataclass(frozen=True)
class BillingExhaustion:
    """A provider refused a call because a paid allowance is used up."""

    source: BillingSource
    provider: str
    key_scope: str
    error_type: str
    provider_message: str


def classify_billing_exhaustion(exc: BaseException) -> BillingExhaustion | None:
    """Return the exhaustion *exc* reports, or None for any other failure.

    Never matches a bare BaseException such as ``asyncio.CancelledError``.
    """
    if not isinstance(exc, Exception):
        return None
    for link in _causal_chain(exc):
        markers = _structured_markers(link)
        if not markers and not _carries_status(link):
            continue
        for marker in markers:
            source = _SOURCE_BY_EXHAUSTION_CODE.get(marker)
            if source is not None:
                return _exhaustion(source, link)
        return None
    message = str(exc).lower()
    for code, source in _SOURCE_BY_EXHAUSTION_CODE.items():
        if code in message:
            return _exhaustion(source, exc)
    return None


def _causal_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen and len(seen) < _MAX_CHAIN_DEPTH:
        seen.add(id(link))
        yield link
        if link.__cause__ is not None:
            link = link.__cause__
        elif link.__suppress_context__:
            link = None
        else:
            link = link.__context__


def _structured_markers(link: BaseException) -> list[str]:
    candidates = [getattr(link, 'code', None), getattr(link, 'type', None)]
    body = getattr(link, 'body', None)
    if isinstance(body, dict):
        candidates += [body.get('code'), body.get('type')]
        nested = body.get('error')
        if isinstance(nested, dict):
            candidates += [nested.get('code'), nested.get('type')]
    return [
        str(value).lower()
        for value in candidates
        if isinstance(value, str | int) and not isinstance(value, bool) and str(value)
    ]


def _carries_status(link: BaseException) -> bool:
    return getattr(link, 'status_code', None) is not None


def _exhaustion(source: BillingSource, link: BaseException) -> BillingExhaustion:
    return exhaustion_of(source, error_type=type(link).__name__, provider_message=str(link))


def exhaustion_of(
    source: BillingSource, *, error_type: str, provider_message: str,
) -> BillingExhaustion:
    """Build the exhaustion of *source*, its provider and key scope filled in."""
    facts = _FACTS_BY_SOURCE[source]
    return BillingExhaustion(
        source=source,
        provider=facts.provider,
        key_scope=facts.key_scope,
        error_type=error_type,
        provider_message=provider_message[:_PROVIDER_MESSAGE_CHARS],
    )
