"""Test helpers shared by the landing-evidence call-site suites (task 4496)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from orchestrator.delivered_checks import DeliveredCheckResult
from orchestrator.event_store import EventStore, EventType


def landing_reject_rows(store: EventStore | None) -> list[dict[str, Any]]:
    """Every ``landing_evidence_rejected`` row in *store*, oldest first, ``data`` decoded."""
    assert store is not None, 'no event store is bound'
    conn = sqlite3.connect(str(store.db_path))
    try:
        return [
            {'task_id': tid, 'data': json.loads(raw or '{}')}
            for tid, raw in conn.execute(
                'SELECT task_id, data FROM events WHERE event_type = ? ORDER BY id',
                (str(EventType.landing_evidence_rejected),),
            )
        ]
    finally:
        conn.close()


async def differential_confirms(
    check: dict[str, Any], *, project_root: str | Path, ref: str = 'main', runner: Any = None,
) -> DeliveredCheckResult:
    """A ``run_delivered_check`` fake: FALSE before the landing (``<sha>^1``), TRUE at it and at main."""
    return DeliveredCheckResult.FAILED if ref.endswith('^1') else DeliveredCheckResult.DELIVERED
