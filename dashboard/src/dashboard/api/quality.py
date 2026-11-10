"""`/api/v2/dashboard/quality` — one project's per-landing quality series.

The ``?window=`` chip counts landings, not days: ``_QUALITY_WINDOWS`` is this
route's vocabulary, parsed by ``dashboard.api.window.parse_landing_window``.
``?project=`` takes the root's label or its project id in any spelling
``shared.project_id.fold_project_token`` folds. The marker and snapshot datums
are unknown until their producers land (plans/quality-dashboard-prd.md leaves δ
and ι).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from shared.project_id import fold_project_token, project_id_for_root

from dashboard.api.clock import app_clock
from dashboard.api.window import parse_landing_window, with_window
from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, DatumContractError, unknown_datum, validate_datum
from dashboard.data.db import DbPool
from dashboard.data.quality_series import QUALITY_FRESHNESS_BOUND_SECONDS, read_quality_series

logger = logging.getLogger(__name__)

router = APIRouter()

_QUALITY_WINDOWS: dict[str, int | None] = {
    '100': 100,
    '300': 300,
    'all': None,
}

_NO_MARKERS = (
    'no producer has written here: the quality marker stream (derived routing and config '
    'markers, curated regime markers) has no reader yet (quality-dashboard PRD leaf δ; '
    'prompt markers also need routing_decision.prompt_digest, leaf ζ)'
)
_NO_SNAPSHOTS = (
    'no producer has written here: the nightly quality-metrics snapshot series has no '
    'writer yet (quality-dashboard PRD leaf ι)'
)


def _project_root(config: DashboardConfig, token: str) -> Path | None:
    """The configured root labelled *token*, else the one whose project id folds as *token* does."""
    labelled = config.project_labels.root_for(token)
    if labelled is not None:
        return labelled
    folded = fold_project_token(token)
    return next(
        (root for root in config.all_project_roots
         if fold_project_token(project_id_for_root(root)) == folded),
        None,
    )


async def _quality(
    config: DashboardConfig, pool: DbPool, token: str | None, window_landings: int | None,
    now: datetime,
) -> Datum[Any]:
    if not token:
        return unknown_datum('no project was named: pass ?project=<label or project id>',
                             QUALITY_FRESHNESS_BOUND_SECONDS)
    root = _project_root(config, token)
    if root is None:
        return unknown_datum(f'no configured project root is labelled or identified as {token!r}',
                             QUALITY_FRESHNESS_BOUND_SECONDS)
    try:
        return await read_quality_series(await pool.get(config.quality_db), root,
                                         window_landings, now=now)
    except sqlite3.Error as exc:
        logger.warning('quality.db unreadable for %s', root, exc_info=True)
        return unknown_datum(f'quality.db unreadable: {type(exc).__name__}: {exc}',
                             QUALITY_FRESHNESS_BOUND_SECONDS)
    except DatumContractError as broken:
        logger.warning('the quality series for %s breaks the %s invariant and is served as '
                       'UNKNOWN; this is a BUG, not an outage', root, broken.invariant.value,
                       exc_info=broken)
        return unknown_datum(f'the quality series broke the {broken.invariant.value} '
                             f'invariant: {broken}', QUALITY_FRESHNESS_BOUND_SECONDS)
    except Exception as exc:
        logger.warning('unexpected error while reading the quality series for %s, which is served '
                       'as UNKNOWN for this render; this is a BUG, not an outage', root,
                       exc_info=True)
        return unknown_datum(f'unexpected error while reading the quality series: '
                             f'{type(exc).__name__}: {exc} (see the WARNING and its traceback)',
                             QUALITY_FRESHNESS_BOUND_SECONDS)


@router.get('/api/v2/dashboard/quality')
async def api_quality(request: Request) -> JSONResponse:
    """QUALITY + QUALITY_MARKERS + QUALITY_SNAPSHOTS + WINDOW + served_at for one project."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    window = parse_landing_window(request.query_params, vocabulary=_QUALITY_WINDOWS,
                                   default='100')
    now = app_clock(request.app)()
    datums: dict[str, Datum[Any]] = {
        'QUALITY': await _quality(config, pool, request.query_params.get('project'),
                                  window.landings, now),
        'QUALITY_MARKERS': unknown_datum(_NO_MARKERS, QUALITY_FRESHNESS_BOUND_SECONDS),
        'QUALITY_SNAPSHOTS': unknown_datum(_NO_SNAPSHOTS, QUALITY_FRESHNESS_BOUND_SECONDS),
    }
    for datum in datums.values():
        validate_datum(datum, now)
    payload = {key: datum.to_wire() for key, datum in datums.items()}
    return JSONResponse(with_window({**payload, 'served_at': now.isoformat()}, window))
