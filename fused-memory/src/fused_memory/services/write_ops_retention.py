"""Run ``WriteJournal.prune_write_ops`` from ``WriteJournalConfig``, at startup and every ``prune_interval_seconds``."""

from __future__ import annotations

import asyncio
import logging

from fused_memory.config.schema import WriteJournalConfig
from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)


async def prune_write_ops_pass(journal: WriteJournal, config: WriteJournalConfig) -> int:
    """One bounded prune pass; returns the rows deleted."""
    return await journal.prune_write_ops(
        read_older_than_days=config.read_retention_days,
        search_older_than_days=config.search_retention_days,
        write_older_than_days=config.write_retention_days,
        batch_size=config.prune_batch_size,
        max_rows=config.prune_max_rows_per_run,
        max_seconds=config.prune_max_seconds,
    )


async def run_periodic_write_ops_prune(journal: WriteJournal, config: WriteJournalConfig) -> None:
    """Run a pass every ``config.prune_interval_seconds``, sleeping first, until cancelled."""
    while True:
        await asyncio.sleep(config.prune_interval_seconds)
        try:
            await prune_write_ops_pass(journal, config)
        except Exception:
            logger.exception('write_ops periodic prune pass failed; the loop continues')
