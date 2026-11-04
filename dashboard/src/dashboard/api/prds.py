"""`/api/v2/dashboard/prds` — one PRD view per project root, as a ``Datum``.

Serves ``{"PRD_VIEW": {label: Datum[PrdView]}, "served_at": iso}`` (PRD
``plans/dashboard-prd-lanes-prd.md`` §4.4, §5.6). This is the γ1 subset of
§5.5's ``PrdView``: per-PRD segments, totals, open members and ready age;
closed PRDs; unattributed done; dark matter; the multi-link count; each
input's provenance; and every figure's rule text. γ2 adds lanes, signals,
activity and a second per-project datum beside this one.

Each root ACQUIRES its inputs inside a budgeted fan-out: its snapshot unit,
the same cache ``/tasks`` reads, and its lineage index. Each view is then
built at ``served_at`` and validated again here, and any failure degrades
that one root to an explained ``unknown``, so the endpoint always answers 200.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.config import DashboardConfig
from dashboard.data.datum import Datum, DatumContractError, unknown_datum, validate_datum
from dashboard.data.prd_view import PrdView, PrdViewInputs, build_prd_view, validate_prd_view
from dashboard.data.task_lineage import LINEAGE_FRESHNESS_BOUND_SECONDS, acquire_lineage
from dashboard.data.task_snapshot import acquire_snapshot
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass(frozen=True, slots=True)
class PrdsBudget:
    """The three bounds the /prds per-root fan-out runs under.

    The figures are /tasks', by the same arithmetic. ``per_project`` (14.0 s)
    holds the snapshot unit's roster, ``task_snapshot.PER_CALL_TIMEOUT`` (4.4)
    x its three bounded operations = 13.2 s; the lineage index waits only
    ``task_lineage.LINEAGE_REQUEST_WAIT_SECONDS`` (5.0) inside it. ``total``
    (20.0 s) is 10 s under ``data.js``'s 30 s fetch abort, so a degraded payload
    still renders. ``root_concurrency`` (4) keeps the walk parallel but
    bounded. ``dashboard/tests/test_prds_budget.py`` checks the constraints.
    """

    per_project: float
    total: float
    root_concurrency: int

    def __post_init__(self) -> None:
        for name, seconds in (('per_project', self.per_project), ('total', self.total)):
            if not seconds > 0:
                raise ValueError(f'PrdsBudget.{name} must be positive, got {seconds!r}')
        if self.root_concurrency < 1:
            raise ValueError(
                f'PrdsBudget.root_concurrency must be at least 1, got {self.root_concurrency!r}'
            )


PRDS_BUDGET = PrdsBudget(per_project=14.0, total=20.0, root_concurrency=4)


@dataclass(frozen=True, slots=True)
class RootUnserved:
    """A root whose inputs this render could not acquire, and why."""

    reason: str


async def _acquire_inputs(
    client: httpx.AsyncClient, config: DashboardConfig, root: Path, *, now: datetime,
) -> PrdViewInputs:
    snapshot, lineage = await asyncio.gather(
        acquire_snapshot(client, config, root, now=now),
        acquire_lineage(client, config, root, now=now),
    )
    return PrdViewInputs(
        project_root=str(root),
        status_map=snapshot.status_map,
        census=snapshot.census,
        rows=snapshot.rows,
        lineage=lineage,
    )


async def _acquire_within_budget(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    *,
    now: datetime,
    budget: PrdsBudget,
) -> dict[Path, PrdViewInputs | RootUnserved]:
    """Every configured root's inputs, or why a root has none, in canonical root order.

    Mirrors ``dashboard/src/dashboard/data/active_tasks.py::_serve_roots_within_budget``
    without its admission rotation: one deadline, a bounded semaphore, and a
    per-root ``wait_for`` sized after admission, so one slow or broken root
    never blanks the others.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget.total
    slots = asyncio.Semaphore(budget.root_concurrency)
    labels = config.project_labels

    async def _one(root: Path) -> PrdViewInputs | RootUnserved:
        label = labels.of(root)
        async with slots:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return _unserved(
                    label,
                    f'skipped — the {budget.total:.1f}s /prds budget was already spent '
                    'before this project was reached',
                )
            try:
                return await asyncio.wait_for(
                    _acquire_inputs(client, config, root, now=now),
                    timeout=min(remaining, budget.per_project),
                )
            except TimeoutError:
                return _unserved(
                    label,
                    f'exceeded its {budget.per_project:.1f}s share of the {budget.total:.1f}s '
                    f'/prds budget ({remaining:.1f}s remained)',
                )
            except Exception:
                logger.warning(
                    'project %s: unexpected error while acquiring its PRD view inputs; it is '
                    'served UNKNOWN for this render so the other roots still render; this is '
                    'a BUG, not an outage',
                    label, exc_info=True,
                )
                return RootUnserved(
                    'unexpected error while acquiring this project\'s inputs '
                    '(see the WARNING and its traceback)'
                )

    roots = config.all_project_roots
    acquired = await asyncio.gather(*(_one(root) for root in roots))
    return dict(zip(roots, acquired, strict=True))


def _unserved(label: str, reason: str) -> RootUnserved:
    logger.warning('project %s: %s; its PRD view is UNKNOWN for this render', label, reason)
    return RootUnserved(reason)


def _served_view(
    label: str, inputs: PrdViewInputs | RootUnserved, served_at: datetime,
) -> Datum[PrdView]:
    """*label*'s view at *served_at*, validated here as well as where it was built."""
    if isinstance(inputs, RootUnserved):
        return unknown_datum(inputs.reason, LINEAGE_FRESHNESS_BOUND_SECONDS)
    what = f"project {label}'s PRD view"
    try:
        datum = build_prd_view(inputs, now=served_at)
        if datum.value is not None:
            validate_prd_view(datum.value, inputs.status_map)
        validate_datum(datum, served_at)
    except DatumContractError as broken:
        logger.warning(
            '%s breaks the %s invariant and is served as UNKNOWN for this render; this is '
            'a BUG, not an outage',
            what, broken.invariant.value, exc_info=broken,
        )
        return unknown_datum(
            f'{what} broke the {broken.invariant.value} invariant: {broken}',
            LINEAGE_FRESHNESS_BOUND_SECONDS,
        )
    except Exception:
        logger.warning(
            'unexpected error while building %s, which is served as UNKNOWN for this '
            'render; this is a BUG, not an outage',
            what, exc_info=True,
        )
        return unknown_datum(
            f'unexpected error while building {what} (see the WARNING and its traceback)',
            LINEAGE_FRESHNESS_BOUND_SECONDS,
        )
    return datum


@router.get('/api/v2/dashboard/prds')
async def api_prds(request: Request) -> JSONResponse:
    """One PRD view per configured root, keyed by its label, plus ``served_at``.

    ``render_at`` stamps what this render measures; ``served_at`` is resolved
    after the fan-out, and every view is built and validated at it, so the
    composite's aged inputs and the freshness check read the same instant.
    """
    config: DashboardConfig = request.app.state.config
    client: httpx.AsyncClient = request.app.state.http_client
    render_at = resolve_now(None)
    acquired = await _acquire_within_budget(client, config, now=render_at, budget=PRDS_BUDGET)
    served_at = resolve_now(None)
    labels = config.project_labels
    views = {
        labels.of(root): _served_view(labels.of(root), inputs, served_at).to_wire()
        for root, inputs in acquired.items()
    }
    return JSONResponse({'PRD_VIEW': views, 'served_at': served_at.isoformat()})
