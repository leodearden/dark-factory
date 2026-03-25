"""Shared helpers for maintenance entrypoints."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fused_memory.services.memory_service import MemoryService


async def _safe_close(
    service: MemoryService,
    logger: logging.Logger,
    context_name: str,
) -> None:
    """Close a MemoryService, logging but not raising on failure.

    Ensures CONFIG_PATH restoration (or other finally-block cleanup)
    always runs even if close() raises.
    """
    try:
        await service.close()
    except Exception:
        logger.warning(
            'Error closing service during %s cleanup',
            context_name,
            exc_info=True,
        )
