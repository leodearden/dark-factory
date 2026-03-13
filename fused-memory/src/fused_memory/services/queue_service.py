"""Queue service for managing sequential episode processing.

Adapted from graphiti/mcp_server/src/services/queue_service.py with the addition
of a post_process_callback parameter for dual-write routing after extraction.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from fused_memory.backends.graphiti_client import GraphitiBackend

logger = logging.getLogger(__name__)


class QueueService:
    """Manages sequential episode processing queues by group_id."""

    def __init__(self):
        self._episode_queues: dict[str, asyncio.Queue] = {}
        self._queue_workers: dict[str, bool] = {}
        self._graphiti_backend: GraphitiBackend | None = None

    async def add_episode_task(
        self, group_id: str, process_func: Callable[[], Awaitable[None]]
    ) -> int:
        """Add an episode processing task to the queue.

        Returns the position in the queue.
        """
        if group_id not in self._episode_queues:
            self._episode_queues[group_id] = asyncio.Queue()

        await self._episode_queues[group_id].put(process_func)

        if not self._queue_workers.get(group_id, False):
            asyncio.create_task(self._process_episode_queue(group_id))

        return self._episode_queues[group_id].qsize()

    async def _process_episode_queue(self, group_id: str) -> None:
        """Process episodes for a specific group_id sequentially."""
        logger.info(f'Starting episode queue worker for group_id: {group_id}')
        self._queue_workers[group_id] = True

        try:
            while True:
                process_func = await self._episode_queues[group_id].get()
                try:
                    await process_func()
                except Exception as e:
                    logger.error(
                        f'Error processing queued episode for group_id {group_id}: {e}'
                    )
                finally:
                    self._episode_queues[group_id].task_done()
        except asyncio.CancelledError:
            logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
        except Exception as e:
            logger.error(f'Unexpected error in queue worker for group_id {group_id}: {e}')
        finally:
            self._queue_workers[group_id] = False
            logger.info(f'Stopped episode queue worker for group_id: {group_id}')

    def get_queue_size(self, group_id: str) -> int:
        if group_id not in self._episode_queues:
            return 0
        return self._episode_queues[group_id].qsize()

    def is_worker_running(self, group_id: str) -> bool:
        return self._queue_workers.get(group_id, False)

    async def initialize(self, graphiti_backend: GraphitiBackend) -> None:
        """Initialize the queue service with a Graphiti backend."""
        self._graphiti_backend = graphiti_backend
        logger.info('Queue service initialized')

    async def add_episode(
        self,
        group_id: str,
        name: str,
        content: str,
        source_description: str,
        source: Any,
        uuid: str | None = None,
        post_process_callback: Callable[[Any], Awaitable[None]] | None = None,
    ) -> int:
        """Add an episode for processing.

        Args:
            post_process_callback: Called with the add_episode result after Graphiti
                finishes processing. Used by MemoryService to hook in post-extraction
                classification and Mem0 dual-write.
        """
        if self._graphiti_backend is None:
            raise RuntimeError('Queue service not initialized — call initialize() first')

        backend = self._graphiti_backend

        async def process_episode():
            try:
                logger.info(f'Processing episode {uuid} for group {group_id}')

                result = await backend.add_episode(
                    name=name,
                    content=content,
                    source=source,
                    group_id=group_id,
                    source_description=source_description,
                    reference_time=datetime.now(timezone.utc),
                    uuid=uuid,
                )

                logger.info(f'Successfully processed episode {uuid} for group {group_id}')

                # Run post-process callback (classification + dual-write)
                if post_process_callback is not None:
                    try:
                        await post_process_callback(result)
                    except Exception as cb_err:
                        logger.error(
                            f'Post-process callback failed for episode {uuid}: {cb_err}'
                        )
            except Exception as e:
                logger.error(f'Failed to process episode {uuid} for group {group_id}: {e}')
                raise

        return await self.add_episode_task(group_id, process_episode)
