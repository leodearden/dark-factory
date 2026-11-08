"""The routing layer's AsyncOpenAI ownership policy; ReadRouter and WriteClassifier each hold their own LazyOpenAIClient."""

from openai import AsyncOpenAI

from fused_memory.config.schema import FusedMemoryConfig


class LazyOpenAIClient:
    """Lazily builds, caches and releases one AsyncOpenAI for its owner; an injected client is owned the same way.

    Built from llm.providers.openai.api_key with no base_url, so the SDK's OPENAI_BASE_URL fallback picks the endpoint.
    """

    def __init__(self, config: FusedMemoryConfig, client: AsyncOpenAI | None = None) -> None:
        self._config = config
        self._client = client

    def get(self) -> AsyncOpenAI:
        if self._client is None:
            provider = self._config.llm.providers.openai
            self._client = AsyncOpenAI(api_key=provider.api_key if provider else None)
        return self._client

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()
