"""Maintenance script to remediate dead-letter queue buildup.

Workflow:
  1. Print baseline stats (dead count by group/error pattern)
  2. Call replay_dead() to reset dead → pending for retry
  3. Poll get_stats() until dead count stabilises or drain_timeout elapses
  4. Snapshot remaining dead items
  5. Purge non-recoverable items (default: NodeNotFoundError% pattern)
  6. Print final stats

Run via:
    python -m fused_memory.maintenance.remediate_dlq
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from fused_memory.maintenance._utils import maintenance_service
from fused_memory.services.durable_queue import DurableWriteQueue

logger = logging.getLogger(__name__)


@dataclass
class RemediationResult:
    """Result of a DLQ remediation run."""

    baseline_dead: int = 0
    replayed: int = 0
    post_replay_dead: int = 0
    purged_node_not_found: int = 0
    final_dead: int = 0
    remaining_dead_details: list[dict[str, Any]] = field(default_factory=list)


class RemediationPlan:
    """Orchestrates the full DLQ remediation workflow.

    Args:
        queue: An initialized DurableWriteQueue instance.
    """

    def __init__(self, queue: DurableWriteQueue) -> None:
        self.queue = queue

    async def execute(
        self,
        drain_timeout: float = 60.0,
        drain_poll_interval: float = 2.0,
        purge_patterns: tuple[str, ...] = ('NodeNotFoundError%',),
    ) -> RemediationResult:
        """Run the full remediation workflow.

        Args:
            drain_timeout: Max seconds to wait for queue to drain after replay.
            drain_poll_interval: Seconds between drain-poll stat checks.
            purge_patterns: SQL LIKE patterns to purge after drain.
                Defaults to ('NodeNotFoundError%',) — purges orphan-node items.

        Returns:
            RemediationResult with counts and remaining item details.
        """
        result = RemediationResult()

        # 1. Baseline dead count
        stats = await self.queue.get_stats()
        result.baseline_dead = stats['counts'].get('dead', 0)
        logger.info('DLQ baseline: %d dead items', result.baseline_dead)

        # 2. Replay dead → pending
        replayed = await self.queue.replay_dead()
        result.replayed = replayed
        logger.info('Replayed %d dead items to pending', replayed)

        # 3. Poll until dead count stabilises or drain_timeout elapses.
        #    We break when the count is the same as the previous poll (stabilised)
        #    OR when drain_timeout elapses.  prev_dead=None means "not yet polled".
        deadline = time.monotonic() + drain_timeout
        prev_dead: int | None = None
        current_dead = result.baseline_dead
        while time.monotonic() < deadline:
            await asyncio.sleep(drain_poll_interval)
            stats = await self.queue.get_stats()
            current_dead = stats['counts'].get('dead', 0)
            logger.info('Drain poll: %d dead remaining', current_dead)
            if prev_dead is not None and current_dead >= prev_dead:
                # No improvement since last poll — stabilised
                break
            prev_dead = current_dead

        result.post_replay_dead = current_dead

        # 4. Snapshot remaining dead items
        remaining = await self.queue.get_dead_items()
        result.remaining_dead_details = remaining

        # 5. Purge non-recoverable items by pattern
        total_purged = 0
        for pattern in purge_patterns:
            count = await self.queue.purge_dead(error_pattern=pattern)
            logger.info('Purged %d items matching error_pattern=%r', count, pattern)
            total_purged += count
        result.purged_node_not_found = total_purged

        # 6. Final dead count
        final_stats = await self.queue.get_stats()
        result.final_dead = final_stats['counts'].get('dead', 0)
        logger.info(
            'DLQ remediation complete: baseline=%d replayed=%d post_drain=%d '
            'purged=%d final=%d',
            result.baseline_dead,
            result.replayed,
            result.post_replay_dead,
            result.purged_node_not_found,
            result.final_dead,
        )
        return result


def _print_summary(result: RemediationResult) -> None:
    """Print a human-readable remediation summary."""
    print('\n=== DLQ Remediation Summary ===')
    print(f'  Baseline dead:         {result.baseline_dead}')
    print(f'  Replayed to pending:   {result.replayed}')
    print(f'  Post-drain dead:       {result.post_replay_dead}')
    print(f'  Purged (non-recov.):   {result.purged_node_not_found}')
    print(f'  Final dead:            {result.final_dead}')
    if result.remaining_dead_details:
        print(f'\n  Remaining dead items ({len(result.remaining_dead_details)}):')
        for item in result.remaining_dead_details:
            print(
                f'    id={item["id"]} group={item["group_id"]} '
                f'op={item["operation"]} error={item["error"]!r}'
            )
    print('================================\n')


async def main(config_path: str | None = None) -> RemediationResult:
    """Load config, initialize service, run remediation, return result.

    Args:
        config_path: Optional path to YAML config file. When None,
                     CONFIG_PATH env var is used.

    Returns:
        RemediationResult with full remediation outcome.
    """
    async with maintenance_service(config_path) as (config, service):
        if service.durable_queue is None:
            logger.error('DurableWriteQueue is not configured — cannot remediate DLQ')
            return RemediationResult()
        plan = RemediationPlan(queue=service.durable_queue)
        result = await plan.execute()
        _print_summary(result)
        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
