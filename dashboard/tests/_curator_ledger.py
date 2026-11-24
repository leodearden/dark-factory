"""The fused-memory curator's invocation ledger, written by tests through its production writer.

The ledger is a ``shared/src/shared/cost_store.py::CostStore`` database: it holds
``invocations`` and ``account_events`` only, with no ``task_results`` and no ``events``
table. Writing it through ``CostStore`` keeps its schema and migrations the production
ones, so a test cannot pass against a shape the curator never writes.

Every row bills project ``dark_factory``, model ``opus`` and role ``task_curator``,
with no task id, as the curator's own calls do.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from shared.cost_store import CapReason, CostStore

_CALL_DURATION = timedelta(minutes=1)


@dataclass(frozen=True)
class CuratorInvocation:
    """One curator call: what it cost, when it finished, and the ceiling that ended it, if any."""

    cost_usd: float
    completed_at: datetime
    capped_reason: CapReason | None = None


async def write_curator_invocations(db_path: Path, invocations: Iterable[CuratorInvocation]) -> None:
    """Append *invocations* to the ledger at *db_path*, creating it if absent."""
    store = CostStore(db_path)
    await store.open()
    try:
        for invocation in invocations:
            await store.save_invocation(
                run_id='fused-memory-x',
                task_id=None,
                project_id='dark_factory',
                account_name='a',
                model='opus',
                role='task_curator',
                cost_usd=invocation.cost_usd,
                input_tokens=None,
                output_tokens=None,
                cache_read_tokens=None,
                cache_create_tokens=None,
                duration_ms=int(_CALL_DURATION.total_seconds() * 1000),
                capped=invocation.capped_reason is not None,
                capped_reason=invocation.capped_reason,
                started_at=(invocation.completed_at - _CALL_DURATION).isoformat(),
                completed_at=invocation.completed_at.isoformat(),
            )
    finally:
        await store.close()
