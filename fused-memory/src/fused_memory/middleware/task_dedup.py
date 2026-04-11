"""Vector-similarity task deduplication via Qdrant.

Maintains a dedicated Qdrant collection of task title embeddings.  Before
creating a new task, the interceptor calls ``find_duplicate`` to check whether
a semantically similar task already exists.  After successful creation,
``record_task`` stores the new embedding for future checks.

The collection is separate from Mem0's main memory collection — its name is
``task_dedup_{project_id}`` so it cannot collide with ``fused_{project_id}``.
"""

from __future__ import annotations

import logging
import uuid as uuid_mod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fused_memory.config.schema import FusedMemoryConfig

logger = logging.getLogger(__name__)

# Default cosine-similarity threshold.  0.92 is aggressive enough to catch
# near-identical titles while avoiding false positives on unrelated tasks.
DEFAULT_SIMILARITY_THRESHOLD = 0.92


class TaskDeduplicator:
    """Qdrant-backed vector similarity checker for task titles."""

    def __init__(self, config: FusedMemoryConfig) -> None:
        self._config = config
        self._client = None  # AsyncQdrantClient, created lazily
        self._embedder = None  # OpenAIEmbedder, created lazily
        self._initialized_collections: set[str] = set()

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    async def _get_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(
                url=self._config.mem0.qdrant_url,
                timeout=30,
            )
        return self._client

    async def _get_embedder(self):
        if self._embedder is None:
            cfg = self._config.embedder
            if cfg.provider == 'openai' and cfg.providers.openai:
                api_key = cfg.providers.openai.api_key
                if api_key:
                    from graphiti_core.embedder import OpenAIEmbedder
                    from graphiti_core.embedder.openai import OpenAIEmbedderConfig

                    self._embedder = OpenAIEmbedder(
                        config=OpenAIEmbedderConfig(
                            api_key=api_key,
                            embedding_model=cfg.model,
                            base_url=cfg.providers.openai.api_url,
                            embedding_dim=cfg.dimensions,
                        ),
                    )
            if self._embedder is None:
                raise RuntimeError(
                    'TaskDeduplicator requires an OpenAI embedder — check config.embedder'
                )
        return self._embedder

    def _collection_name(self, project_id: str) -> str:
        return f'task_dedup_{project_id}'

    async def _ensure_collection(self, project_id: str) -> str:
        """Create the dedup collection if it doesn't exist yet."""
        name = self._collection_name(project_id)
        if name in self._initialized_collections:
            return name

        client = await self._get_client()
        if not await client.collection_exists(name):
            from qdrant_client.models import Distance, VectorParams

            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._config.embedder.dimensions,
                    distance=Distance.COSINE,
                ),
            )
            logger.info('Created task dedup collection: %s', name)

        self._initialized_collections.add(name)
        return name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def find_duplicate(
        self,
        title: str,
        project_id: str,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> dict[str, Any] | None:
        """Return the best-matching existing task if similarity >= threshold.

        Returns a dict with ``task_id``, ``task_title``, and ``score``,
        or ``None`` if no match exceeds the threshold.
        """
        try:
            collection = await self._ensure_collection(project_id)
            embedder = await self._get_embedder()
            embedding = await embedder.create(title)

            client = await self._get_client()
            results = await client.query_points(
                collection_name=collection,
                query=embedding,
                limit=1,
                score_threshold=threshold,
                with_payload=True,
            )

            if results.points:
                point = results.points[0]
                payload = point.payload or {}
                return {
                    'task_id': payload.get('task_id', ''),
                    'task_title': payload.get('task_title', ''),
                    'score': point.score,
                }
        except Exception:
            # Dedup is best-effort — never block task creation
            logger.warning('Task dedup similarity check failed', exc_info=True)

        return None

    async def record_task(
        self,
        task_id: str,
        title: str,
        project_id: str,
    ) -> None:
        """Store a task title embedding for future dedup checks."""
        try:
            collection = await self._ensure_collection(project_id)
            embedder = await self._get_embedder()
            embedding = await embedder.create(title)

            from qdrant_client.models import PointStruct

            # Use a UUID derived from (project_id, task_id) for deterministic IDs
            point_id = str(
                uuid_mod.uuid5(uuid_mod.NAMESPACE_URL, f'{project_id}/{task_id}')
            )

            client = await self._get_client()
            await client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            'task_id': task_id,
                            'task_title': title,
                            'project_id': project_id,
                            'created_at': datetime.now(UTC).isoformat(),
                        },
                    ),
                ],
            )
        except Exception:
            # Best-effort — don't break task creation if recording fails
            logger.warning('Task dedup record failed for task %s', task_id, exc_info=True)

    async def close(self) -> None:
        """Release the Qdrant client connection."""
        if self._client is not None:
            import contextlib

            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None
