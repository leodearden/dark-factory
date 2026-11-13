"""`/api/v2/dashboard/quality` — one project's per-landing quality series.

The ``?window=`` chip counts landings, not days: ``_QUALITY_WINDOWS`` is this
route's vocabulary, parsed by ``dashboard.api.window.parse_landing_window``.
``?project=`` takes the root's label or its project id in any spelling
``shared.project_id.fold_project_token`` folds. ``?module=`` drills down: QUALITY
serves only the landings whose lines touch that module, its freshness still read
over the whole project (blank or absent serves every landing). QUALITY_MARKERS is the project's
stored marker derivation plus the curated markers in its scope, placed on its
stored landing axis; its regime events are what the QUALITY statistics split at
and exclude. Before the project's first derivation, the curated markers in its
scope still are. The snapshot datum is unknown until its producer lands
(plans/quality-dashboard-prd.md leaf ι).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from shared.project_id import fold_project_token, project_id_for_root
from shared.quality_series.stats import RegimeEvent

from dashboard.api.clock import app_clock
from dashboard.api.window import parse_landing_window, with_window
from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, DatumContractError, unknown_datum, validate_datum
from dashboard.data.db import DbPool
from dashboard.data.quality_markers import (
    CuratedMarkers,
    MarkerStream,
    load_curated_markers,
    marker_datum,
    regime_events,
)
from dashboard.data.quality_series import QUALITY_FRESHNESS_BOUND_SECONDS, read_quality_series
from dashboard.data.quality_store import landing_axis, read_marker_state

logger = logging.getLogger(__name__)

router = APIRouter()

_QUALITY_WINDOWS: dict[str, int | None] = {
    '100': 100,
    '300': 300,
    'all': None,
}

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


def _unknown(reason: str) -> Datum[Any]:
    return unknown_datum(reason, QUALITY_FRESHNESS_BOUND_SECONDS)


def _unresolved_reason(token: str | None) -> str:
    if not token:
        return 'no project was named: pass ?project=<label or project id>'
    return f'no configured project root is labelled or identified as {token!r}'


_T = TypeVar('_T')


async def _served(what: str, root: Path, read: Callable[[], Awaitable[Datum[_T]]]) -> Datum[_T]:
    """*read*'s datum, or an unknown one naming why *what* could not be read; never raises."""
    try:
        return await read()
    except sqlite3.Error as exc:
        logger.warning('quality.db unreadable for %s', root, exc_info=True)
        return _unknown(f'quality.db unreadable: {type(exc).__name__}: {exc}')
    except DatumContractError as broken:
        logger.warning('%s for %s breaks the %s invariant and is served as UNKNOWN; this is a '
                       'BUG, not an outage', what, root, broken.invariant.value, exc_info=broken)
        return _unknown(f'{what} broke the {broken.invariant.value} invariant: {broken}')
    except Exception as exc:
        logger.warning('unexpected error while reading %s for %s, which is served as UNKNOWN for '
                       'this render; this is a BUG, not an outage', what, root, exc_info=True)
        return _unknown(f'unexpected error while reading {what}: {type(exc).__name__}: {exc} '
                        f'(see the WARNING and its traceback)')


async def _marker_stream(
    config: DashboardConfig, pool: DbPool, project_id: str, curated: CuratedMarkers, now: datetime,
) -> Datum[MarkerStream]:
    db = await pool.get(config.quality_db)
    stored = None if db is None else await read_marker_state(db, project_id)
    axis = [] if db is None or stored is None else await landing_axis(db, project_id)
    return marker_datum(project_id, stored, curated, axis, now=now,
                        freshness_bound_seconds=QUALITY_FRESHNESS_BOUND_SECONDS)


def _regime_events(
    markers: Datum[MarkerStream], curated: CuratedMarkers, project_id: str,
) -> Sequence[RegimeEvent]:
    """The served stream's regime events; with no stream served, the curated ones in the
    project's scope, which need no derivation."""
    if markers.value is not None:
        return markers.value.regime_events()
    return regime_events(curated.for_project(project_id))


async def _quality_series(
    config: DashboardConfig, pool: DbPool, root: Path, window_landings: int | None,
    now: datetime, events: Sequence[RegimeEvent], module: str | None,
) -> Datum[Any]:
    return await read_quality_series(await pool.get(config.quality_db), root, window_landings,
                                     now=now, events=events, module=module)


@router.get('/api/v2/dashboard/quality')
async def api_quality(request: Request) -> JSONResponse:
    """QUALITY + QUALITY_MARKERS + QUALITY_SNAPSHOTS + WINDOW + served_at for one project."""
    config: DashboardConfig = request.app.state.config
    pool: DbPool = request.app.state.db
    window = parse_landing_window(request.query_params, vocabulary=_QUALITY_WINDOWS,
                                   default='100')
    now = app_clock(request.app)()
    token = request.query_params.get('project')
    root = _project_root(config, token) if token else None
    if root is None:
        quality = markers = _unknown(_unresolved_reason(token))
    else:
        project_id = project_id_for_root(root)
        curated = await asyncio.to_thread(load_curated_markers)
        markers = await _served('the quality marker stream', root,
                                lambda: _marker_stream(config, pool, project_id, curated, now))
        events = _regime_events(markers, curated, project_id)
        module = request.query_params.get('module') or None
        quality = await _served('the quality series', root, lambda: _quality_series(
            config, pool, root, window.landings, now, events, module))
    datums: dict[str, Datum[Any]] = {
        'QUALITY': quality,
        'QUALITY_MARKERS': markers,
        'QUALITY_SNAPSHOTS': _unknown(_NO_SNAPSHOTS),
    }
    for datum in datums.values():
        validate_datum(datum, now)
    payload = {key: datum.to_wire() for key, datum in datums.items()}
    return JSONResponse(with_window({**payload, 'served_at': now.isoformat()}, window))
