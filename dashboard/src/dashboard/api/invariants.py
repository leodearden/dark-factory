"""`/api/v2/dashboard/invariants` — per-root claimant-invariant census (contract E-1).

``docs/prds/claimant-invariant-detection.md`` E-1/D-2: one record per known
project root, read directly and read-only from that root's ``tasks.db``. The
MCP task fetch is not used because its 400-row terminal window misses old
residue, and residue on terminal rows is exactly what this gauge counts. One
``now`` is captured per request; it is every record's ``measured_at`` and the
response's ``served_at``.

Each record also carries the recurrence chains section
(``docs/prds/recurring-deterministic-tasks.md`` R-D5/C-4). A paused chain's
pinning escalations are read through the escalation corpus' one cache; a
corpus walk that raises leaves them unknown and costs no root its record.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, unknown_datum
from dashboard.data.db import DbPool
from dashboard.data.escalation_corpus import (
    CORPUS_FRESHNESS_BOUND_SECONDS,
    EscalationCorpus,
    acquire_corpus,
    corpus_queues,
    pending_escalation_ids,
)
from dashboard.data.root_invariants import scan_root_invariants
from dashboard.project_dbs import TASKS_DB_REL_PATH

logger = logging.getLogger(__name__)

router = APIRouter()


async def _escalation_corpus(config: DashboardConfig, *, now: datetime) -> Datum[EscalationCorpus]:
    try:
        return await acquire_corpus(corpus_queues(config), now=now)
    except Exception as exc:
        logger.exception('invariants: escalation corpus walk failed; pinning escalations unknown')
        return unknown_datum(
            f'the escalation corpus walk failed: {type(exc).__name__}: {exc}',
            CORPUS_FRESHNESS_BOUND_SECONDS,
        )


@router.get('/api/v2/dashboard/invariants')
async def api_invariants(request: Request) -> JSONResponse:
    """INVARIANTS + served_at — one E-1 record per known project root, in roster order."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    now = datetime.now(UTC)  # clock-exempt: single-capture route
    corpus = await _escalation_corpus(config, now=now)
    roots = config.all_project_roots
    db_paths = [root / TASKS_DB_REL_PATH for root in roots]
    conns = [await pool.get(db_path) for db_path in db_paths]
    records = await asyncio.gather(
        *(
            scan_root_invariants(
                root,
                db_path,
                conn,
                now=now,
                pending_escalations=pending_escalation_ids(corpus, str(root)),
            )
            for root, db_path, conn in zip(roots, db_paths, conns, strict=True)
        )
    )
    return JSONResponse(
        {'INVARIANTS': [record.to_wire() for record in records], 'served_at': now.isoformat()}
    )
