"""The routing layer's one LLM call: its endpoint (routing_endpoint), its request parameters (routing_chat_params) and the AsyncOpenAI each owner holds (LazyOpenAIClient)."""

import logging
from typing import Any

from openai import AsyncOpenAI

from fused_memory.config.env_precedence import warn_if_ambient_base_url_is_overridden
from fused_memory.config.schema import FusedMemoryConfig, blank_means_inherit

logger = logging.getLogger(__name__)

ROUTING_MAX_TOKENS = 256


class RoutingEndpointUnset(ValueError):
    """routing.model is pinned but an endpoint leaf it needs is unset, so no endpoint is inherited."""


def routing_chat_params(config: FusedMemoryConfig) -> dict[str, Any]:
    """The routing chat.completions call's model, sampling control and token cap; the messages are the caller's."""
    routing = config.routing
    model = routing.model or config.llm.model
    if routing.reasoning_effort is None:
        return {'model': model, 'temperature': 0.0, 'max_tokens': ROUTING_MAX_TOKENS}
    return {
        'model': model, 'reasoning_effort': routing.reasoning_effort,
        'max_completion_tokens': ROUTING_MAX_TOKENS,
    }


def routing_endpoint(config: FusedMemoryConfig) -> tuple[str | None, str | None]:
    """(api_key, base_url) for the routing call.

    Unpinned, each routing leaf, else llm.providers.openai's, else None for the
    SDK env fallback. A pinned routing.model takes exactly its own leaves and
    raises RoutingEndpointUnset when either is unset, so it never runs on the
    llm provider's key.
    """
    routing = config.routing
    api_key = blank_means_inherit(routing.openai_api_key)
    base_url = blank_means_inherit(routing.openai_api_url)
    if routing.model is not None:
        unset = [
            f'routing.{leaf}' for leaf, value in
            (('openai_api_url', base_url), ('openai_api_key', api_key)) if value is None
        ]
        if unset:
            raise RoutingEndpointUnset(
                f'routing.model {routing.model} is pinned but {" and ".join(unset)} is unset; '
                'a pinned model never inherits llm.providers.openai',
            )
        return api_key, base_url
    provider = config.llm.providers.openai
    return (
        api_key or (provider.api_key if provider else None),
        base_url or (provider.api_url if provider else None),
    )


class LazyOpenAIClient:
    """Lazily builds, caches and releases one AsyncOpenAI for its owner; an injected client is owned the same way.

    The endpoint is routing_endpoint's; config wins over OPENAI_BASE_URL. Building logs one INFO line naming the
    model, effort and endpoint, never the key.
    """

    def __init__(self, config: FusedMemoryConfig, client: AsyncOpenAI | None = None) -> None:
        self._config = config
        self._client = client

    def get(self) -> AsyncOpenAI:
        if self._client is None:
            api_key, base_url = routing_endpoint(self._config)
            warn_if_ambient_base_url_is_overridden(base_url, context='routing LLM')
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            logger.info(
                'Routing LLM: model=%s reasoning_effort=%s base_url=%s',
                routing_chat_params(self._config)['model'],
                self._config.routing.reasoning_effort or 'default', self._client.base_url,
            )
        return self._client

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()
