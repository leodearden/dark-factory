"""Per-project Mem0 AsyncMemory instance manager."""

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
            instance = AsyncMemory.from_config(config_dict)  # type: ignore[assignment]
            self._instances[project_id] = instance
            logger.info(f'Mem0 instance created for project {project_id} (collection: {collection_name})')
        return self._instances[project_id]

    async def add(
        self,
        content: str,
        scope: Scope,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a memory to Mem0."""
        instance = await self._get_instance(scope)
        result = await instance.add(
            messages=content,
            user_id=scope.mem0_user_id,
            agent_id=scope.agent_id,
            run_id=scope.session_id,
            metadata=metadata,
        )
        return result

    async def search(
        self,
        query: str,
        scope: Scope,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search memories in Mem0."""
        instance = await self._get_instance(scope)
        result = await instance.search(
            query=query,
            user_id=scope.mem0_user_id,
            agent_id=scope.agent_id,
            run_id=scope.session_id,
            limit=limit,
        )
        return result

    async def get_all(
        self,
        scope: Scope,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get all memories for a scope."""
        instance = await self._get_instance(scope)
        result = await instance.get_all(
            user_id=scope.mem0_user_id,
            agent_id=scope.agent_id,
            run_id=scope.session_id,
            limit=limit,
        )
        return result

    async def get(self, memory_id: str, scope: Scope) -> dict[str, Any] | None:
        """Get a single memory by ID."""
        instance = await self._get_instance(scope)
        return await instance.get(memory_id)

    async def update(self, memory_id: str, data: str, scope: Scope) -> dict[str, Any]:
        """Update a memory."""
        instance = await self._get_instance(scope)
        return await instance.update(memory_id, data)

    async def delete(self, memory_id: str, scope: Scope) -> dict[str, Any]:
        """Delete a memory."""
        instance = await self._get_instance(scope)
        return await instance.delete(memory_id)
