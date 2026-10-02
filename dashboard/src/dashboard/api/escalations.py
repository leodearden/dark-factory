"""`/api/v2/dashboard/escalations` and `/escalation-analytics` — two surfaces over one corpus.

Both routes read the escalation corpus through
``dashboard.data.escalation_corpus.acquire_corpus`` and its one cache
(``plans/dashboard-one-datum-one-path-prd.md``, decision 13), so the
Escalations tab's live queue and the analytics tab's history are counted over
the same walk, at the same instant.

/escalations builds the live-queue table from that walk, attributes each
reconciliation row to the root its task is ACTIVE in (the snapshot units'
active rows), and reads each row's task card by id through
``dashboard.data.task_lookup.lookup_tasks`` — never a whole task tree
(decision 12).

/escalation-analytics derives its aggregates from the same walk. The
derivation — the archive-wide aggregation and the pins_recovery fan-out — is
memoised per corpus GENERATION, its queues and its walk's instant, so it is
paid once per walk and cannot be served over a different walk than the one
/escalations reads. The memo's TTL is the corpus TTL by reference: it holds a
derivation of one cache entry, and is not a second freshness authority.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data import escalation_corpus, redux_api
from dashboard.data.datum import Datum
from dashboard.data.escalation_analytics import build_escalation_analytics
from dashboard.data.escalation_corpus import (
    EscalationCorpus,
    QueueKind,
    QueueRef,
    acquire_corpus,
    corpus_queues,
)
from dashboard.data.escalations import (
    build_escalation_queues,
    card_datums,
    card_task_refs,
    fetch_pins_recovery,
)
from dashboard.data.mcp_fanout import TTLCache
from dashboard.data.task_lookup import lookup_tasks
from dashboard.data.task_snapshot import acquire_snapshot
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

router = APIRouter()


def _orchestrator_roots(queues: Sequence[QueueRef]) -> list[str]:
    return [queue.id for queue in queues if queue.kind is QueueKind.ORCHESTRATOR]


async def _active_rows(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    queues: Sequence[QueueRef],
    *,
    now: datetime,
) -> dict[str, Sequence[dict]]:
    """Each orchestrator root's ACTIVE rows: the reconciliation owner probe's population."""
    roots = _orchestrator_roots(queues)
    snapshots = await asyncio.gather(
        *(acquire_snapshot(client, config, root, now=now) for root in roots),
    )
    return {
        root: snapshot.rows.value or ()
        for root, snapshot in zip(roots, snapshots, strict=True)
    }


@router.get('/api/v2/dashboard/escalations')
async def api_escalations(request: Request) -> JSONResponse:
    """ESCALATIONS — the live queues, their views over the corpus, and each row's task card."""
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client
    render_at = resolve_now(None)

    queues = corpus_queues(config)
    corpus = await acquire_corpus(queues, now=render_at)
    active_rows = await _active_rows(http_client, config, queues, now=render_at)
    built = build_escalation_queues(corpus, active_rows=active_rows)
    lookup = await lookup_tasks(http_client, config, card_task_refs(built), now=render_at)

    # Resolved AFTER the fan-out, as api/merge_queue.py::api_merge_queue does:
    # every as_of above was stamped at or before this instant, never after it.
    served_at = resolve_now(None)
    return JSONResponse(
        redux_api.shape_escalations(built, card_datums(built, lookup), served_at=served_at)
    )


# ---------------------------------------------------------------------------
# Escalation analytics, memoised per corpus generation
# ---------------------------------------------------------------------------

_AnalyticsKey = tuple[tuple[QueueRef, ...], datetime | None]

_analytics_memo: TTLCache[dict, _AnalyticsKey] = TTLCache(
    ttl_seconds=lambda: escalation_corpus.CORPUS_TTL_SECONDS
)


def _analytics_memo_clear() -> None:
    """Clear the analytics memo (test/admin hook)."""
    _analytics_memo.clear()


def _runs_dbs(config: DashboardConfig, queues: Sequence[QueueRef]) -> dict[str, Path]:
    """Each orchestrator queue's ``runs.db``: the configured one for the primary root."""
    primary = str(config.project_root)
    return {
        root: config.runs_db if root == primary
        else Path(root) / 'data' / 'orchestrator' / 'runs.db'
        for root in _orchestrator_roots(queues)
    }


async def _derive_analytics(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    queues: Sequence[QueueRef],
    corpus: Datum[EscalationCorpus],
) -> dict:
    """The analytics payload over *corpus*, annotated with each project's pins.

    The pins fan-out is async and stays on this side of the ``to_thread``
    boundary: ``build_escalation_analytics`` is pure-sync, and an MCP round
    trip inside it would block a worker thread on the network.
    ``fetch_pins_recovery`` already isolates per-project failures (an
    unreachable orchestrator maps to ``None``, unknown); this guard covers
    only the unexpected, so the tab still renders one annotation short.
    """
    try:
        pins = await fetch_pins_recovery(client, config.escalation_urls)
    except Exception as exc:  # noqa: BLE001 — the tab must survive this
        logger.warning('pins_recovery fan-out failed (analytics served unannotated): %s', exc)
        pins = None
    return await asyncio.to_thread(
        build_escalation_analytics, corpus, _runs_dbs(config, queues), pins_by_project=pins,
    )


@router.get('/api/v2/dashboard/escalation-analytics')
async def api_escalation_analytics(request: Request) -> JSONResponse:
    """ESCALATION_ANALYTICS — origin/lifespan/workflow aggregates over the corpus' history."""
    config: DashboardConfig = request.app.state.config
    http_client: httpx.AsyncClient = request.app.state.http_client

    queues = corpus_queues(config)
    corpus = await acquire_corpus(queues, now=resolve_now(None))

    async def _refresh() -> dict:
        return await _derive_analytics(http_client, config, queues, corpus)

    payload = await _analytics_memo.get_or_refresh((queues, corpus.as_of), _refresh)
    return JSONResponse(
        redux_api.shape_escalation_analytics(payload, served_at=resolve_now(None))
    )
