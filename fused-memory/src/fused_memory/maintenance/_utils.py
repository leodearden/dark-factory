"""Private maintenance utilities."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncGenerator, Generator

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def override_config_path(config_path: str | None) -> Generator[None, None, None]:
    """Temporarily set CONFIG_PATH in the environment, then restore it.

    When *config_path* is ``None`` the environment is left completely
    untouched — the context manager is a no-op.

    When *config_path* is provided:
    - ``os.environ['CONFIG_PATH']`` is set to *config_path* on entry.
    - On exit (normal **or** exceptional) the original state is restored:
      if CONFIG_PATH was absent before entry it is popped; if it had a
      value that value is reinstated.

    .. warning:: **Not thread-safe.**  This context manager mutates
       ``os.environ`` without any locking.  Concurrent calls within the
       same process will race on the ``CONFIG_PATH`` key.  Use only from
       a single thread (or a single ``asyncio`` event loop with no
       concurrent tasks that also call this function).
    """
    if config_path is None:
        yield
        return

    old_value = os.environ.get('CONFIG_PATH')
    os.environ['CONFIG_PATH'] = config_path
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop('CONFIG_PATH', None)
        else:
            os.environ['CONFIG_PATH'] = old_value


@contextlib.asynccontextmanager
async def maintenance_service(
    config_path: str | None,
) -> AsyncGenerator[tuple[FusedMemoryConfig, MemoryService], None]:
    """Async context manager that manages the full maintenance service lifecycle.

    Sets up CONFIG_PATH, loads FusedMemoryConfig, creates and initialises
    a MemoryService, yields ``(config, service)``, then closes the service on
    exit — suppressing any close() errors so that caller exceptions propagate
    cleanly.

    Args:
        config_path: Optional path to the YAML config file.  When given it is
                     set as CONFIG_PATH before constructing FusedMemoryConfig.
                     When ``None`` the environment is left untouched.

    Yields:
        Tuple of (FusedMemoryConfig, MemoryService) — service is already
        initialized when yielded.

    Example::

        async with maintenance_service(config_path) as (config, service):
            manager = MyManager(backend=service.graphiti)
            result = await manager.run()
    """
    service: MemoryService | None = None
    with override_config_path(config_path):
        try:
            config = FusedMemoryConfig()
            service = MemoryService(config)
            await service.initialize()
            yield config, service
        finally:
            if service is not None:
                try:
                    await service.close()
                except Exception:
                    logger.warning('Error closing service in maintenance_service', exc_info=True)
