"""Per-project Mem0 AsyncMemory instance manager."""

import asyncio
import logging
from typing import Any

from mem0 import AsyncMemory

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.scope import Scope

logger = logging.getLogger(__name__)


class Mem0Backend:
    """Lazily creates AsyncMemory instances keyed by project_id."""

    def __init__(self, config: FusedMemoryConfig):
        self.config = config
        self._instances: dict[str, AsyncMemory] = {}
        self._read_timeout: float = config.queue.backend_read_timeout_seconds
        self._write_timeout: float = config.queue.backend_write_timeout_seconds
        self._async_qdrant_client = None  # Lazy async client for count/list ops

    def _build_config_dict(self, collection_name: str) -> dict[str, Any]:
        """Build a Mem0 config dict from the unified config."""
        cfg = self.config
        config_dict: dict[str, Any] = {
            'version': 'v1.1',
            'vector_store': {
                'provider': 'qdrant',
                'config': {
                    'url': cfg.mem0.qdrant_url,
                    'collection_name': collection_name,
                },
            },
        }

        # LLM
        if cfg.llm.provider == 'openai' and cfg.llm.providers.openai:
            config_dict['llm'] = {
                'provider': 'openai',
                'config': {
                    'model': cfg.llm.model,
                    'temperature': cfg.llm.temperature or 0.1,
                    'max_tokens': cfg.llm.max_tokens,
                    'api_key': cfg.llm.providers.openai.api_key,
                },
            }
        elif cfg.llm.provider == 'anthropic' and cfg.llm.providers.anthropic:
            config_dict['llm'] = {
                'provider': 'anthropic',
                'config': {
                    'model': cfg.llm.model,
                    'temperature': cfg.llm.temperature or 0.1,
                    'max_tokens': cfg.llm.max_tokens,
                    'api_key': cfg.llm.providers.anthropic.api_key,
                },
            }

        # Embedder
        if cfg.embedder.provider == 'openai' and cfg.embedder.providers.openai:
            config_dict['embedder'] = {
                'provider': 'openai',
                'config': {
                    'model': cfg.embedder.model,
                    'api_key': cfg.embedder.providers.openai.api_key,
                },
            }

        return config_dict

    async def _get_instance(self, scope: Scope) -> AsyncMemory:
        """Lazily create and cache an AsyncMemory instance for a project."""
        project_id = scope.project_id
        if project_id not in self._instances:
            collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
            config_dict = self._build_config_dict(collection_name)
            instance = AsyncMemory.from_config(config_dict)
            self._instances[project_id] = instance
            logger.info(f'Mem0 instance created for project {project_id} (collection: {collection_name})')
        return self._instances[project_id]

    async def add(
        self,
        content: str,
        scope: Scope,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a memory to Mem0.

        Callers into this backend (add_memory, classify_and_add, queued
        mem0_add dispatch) all pass already-distilled content per the
        CLAUDE.md contract. We pin ``infer=False`` so Mem0 stores the
        content verbatim and returns the assigned ID, rather than running
        its LLM fact-extractor over the input — which silently drops any
        content the extractor does not classify as a declarative fact
        (normative/procedural/behavioral text) and returns
        ``{'results': []}`` with no error.
        """
        instance = await self._get_instance(scope)
        return await asyncio.wait_for(
            instance.add(
                messages=content,
                user_id=scope.mem0_user_id,
                agent_id=scope.agent_id,
                run_id=scope.session_id,
                metadata=metadata,
                infer=False,
            ),
            timeout=self._write_timeout,
        )

    async def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search memories in Mem0."""
        instance = await self._get_instance(scope)
        try:
            return await asyncio.wait_for(
                instance.search(
                    query=query,
                    user_id=scope.mem0_user_id,
                    agent_id=None,
                    run_id=None,
                    limit=limit,
                ),
                timeout=self._read_timeout,
            )
        except TimeoutError:
            logger.warning(f'Mem0 search timed out after {self._read_timeout}s')
            return {}

    async def get_all(
        self,
        scope: Scope,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get all memories for a scope."""
        instance = await self._get_instance(scope)
        try:
            return await asyncio.wait_for(
                instance.get_all(
                    user_id=scope.mem0_user_id,
                    agent_id=scope.agent_id,
                    run_id=scope.session_id,
                    limit=limit,
                ),
                timeout=self._read_timeout,
            )
        except TimeoutError:
            logger.warning(f'Mem0 get_all timed out after {self._read_timeout}s')
            return {}

    async def get(self, memory_id: str, scope: Scope) -> dict[str, Any] | None:
        """Get a single memory by ID."""
        instance = await self._get_instance(scope)
        try:
            return await asyncio.wait_for(
                instance.get(memory_id),
                timeout=self._read_timeout,
            )
        except TimeoutError:
            logger.warning(f'Mem0 get timed out after {self._read_timeout}s')
            return None

    async def update(self, memory_id: str, data: str, scope: Scope) -> dict[str, Any]:
        """Update a memory."""
        instance = await self._get_instance(scope)
        return await asyncio.wait_for(
            instance.update(memory_id, data),
            timeout=self._write_timeout,
        )

    async def delete(self, memory_id: str, scope: Scope) -> dict[str, Any]:
        """Delete a memory."""
        instance = await self._get_instance(scope)
        return await asyncio.wait_for(
            instance.delete(memory_id),
            timeout=self._write_timeout,
        )

    async def _get_async_qdrant(self):
        """Get or create a shared async Qdrant client for lightweight ops."""
        if self._async_qdrant_client is None:
            from qdrant_client import AsyncQdrantClient

            self._async_qdrant_client = AsyncQdrantClient(
                url=self.config.mem0.qdrant_url,
                timeout=int(self._read_timeout),
            )
        return self._async_qdrant_client

    async def count(self, scope: Scope) -> int:
        """Count memories using native async Qdrant count API."""
        collection_name = scope.mem0_collection_name(self.config.mem0.collection_prefix)
        client = await self._get_async_qdrant()
        result = await asyncio.wait_for(
            client.count(
                collection_name=collection_name,
                exact=True,
            ),
            timeout=self._read_timeout,
        )
        return result.count

    async def close(self) -> None:
        """Close all cached AsyncMemory instances and release their connections."""
        import contextlib
        for instance in self._instances.values():
            with contextlib.suppress(Exception):
                client = getattr(getattr(instance, 'vector_store', None), 'client', None)
                if client is not None and hasattr(client, 'close'):
                    await client.close()
        self._instances.clear()
        if self._async_qdrant_client is not None:
            with contextlib.suppress(Exception):
                await self._async_qdrant_client.close()
            self._async_qdrant_client = None

    async def list_projects(self) -> list[tuple[str, str]]:
        """Enumerate projects by scanning Qdrant collections matching the prefix.

        Returns list of (project_id, collection_name) tuples.
        """
        client = await self._get_async_qdrant()
        prefix = f'{self.config.mem0.collection_prefix}_'
        result = []
        collections = await client.get_collections()
        for c in collections.collections:
            if c.name.startswith(prefix):
                project_id = c.name[len(prefix):]
                if project_id:
                    result.append((project_id, c.name))
        return result
