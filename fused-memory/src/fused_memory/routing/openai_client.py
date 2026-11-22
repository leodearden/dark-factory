"""The routing layer's AsyncOpenAI ownership policy; ReadRouter and WriteClassifier each hold their own LazyOpenAIClient."""

from openai import AsyncOpenAI

from fused_memory.config.env_precedence import warn_if_ambient_base_url_is_overridden
from fused_memory.config.schema import FusedMemoryConfig, blank_means_inherit


def _routing_endpoint(config: FusedMemoryConfig) -> tuple[str | None, str | None]:
    """(api_key, base_url): each routing leaf, else llm.providers.openai's, else None for the SDK env fallback."""
    provider = config.llm.providers.openai
    api_key = blank_means_inherit(config.routing.openai_api_key) or (
        provider.api_key if provider else None
    )
    base_url = blank_means_inherit(config.routing.openai_api_url) or (
        provider.api_url if provider else None
    )
    return api_key, base_url


class LazyOpenAIClient:
    """Lazily builds, caches and releases one AsyncOpenAI for its owner; an injected client is owned the same way.

    Endpoint and key are routing.openai_api_url / routing.openai_api_key, each inheriting llm.providers.openai's when unset; config wins over OPENAI_BASE_URL.
    """

    def __init__(self, config: FusedMemoryConfig, client: AsyncOpenAI | None = None) -> None:
        self._config = config
        self._client = client

    def get(self) -> AsyncOpenAI:
        if self._client is None:
            api_key, base_url = _routing_endpoint(self._config)
            warn_if_ambient_base_url_is_overridden(base_url, context='routing LLM')
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._client

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()
