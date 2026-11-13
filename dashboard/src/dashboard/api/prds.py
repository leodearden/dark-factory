"""`/api/v2/dashboard/prds` — one PRD view and one contention line per project root.

Serves ``{"PRD_VIEW": {label: Datum[PrdView]}, "PRD_CONTENTION": {label:
Datum[Contention]}, "served_at": iso}`` (PRD ``plans/dashboard-prd-lanes-prd.md``
§4.4, §5.5, §5.6). The view carries each PRD's segments, lane, state line,
signal mark, next action and detail; the contention line the lock keys the
ready set wants and who holds them.

Each root ACQUIRES its inputs inside a budgeted fan-out: its snapshot unit,
the same cache ``/tasks`` reads, its lineage index, its scheduler state and its
run ledger's activity together, then the documents of the PRDs it draws. The
escalation corpus and the run-ledger connections are acquired once per
request, each waited on at most ``REQUEST_READS_WAIT_SECONDS`` before the
fan-out, and a read that fails or misses that wait degrades its own input,
never the response. Each datum is then built
at ``served_at`` and validated again here. The two degrade apart: a scheduler
failure degrades only ``PRD_CONTENTION``, a lineage failure only
``PRD_VIEW``, and any failure degrades that one root's datum to an explained
``unknown``, so the endpoint always answers 200.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import aiosqlite
import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dashboard.api.clock import app_clock
from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import TasksBudget
from dashboard.data.datum import (
    Datum,
    DatumContractError,
    map_datum,
    unknown_datum,
    validate_datum,
)
from dashboard.data.db import DbPool
from dashboard.data.escalation_corpus import (
    EscalationCorpus,
    corpus_queues,
    pending_level2_by_task,
    serve_corpus,
)
from dashboard.data.prd_contention import (
    Contention,
    ContentionInputs,
    build_contention,
    validate_contention,
)
from dashboard.data.prd_signal import PRD_DOCUMENTS_REQUEST_WAIT_SECONDS, read_prd_documents
from dashboard.data.prd_view import (
    PrdView,
    PrdViewInputs,
    build_prd_view,
    open_work_keys,
    validate_prd_view,
)
from dashboard.data.scheduler import SCHEDULER_STATE_REQUEST_WAIT_SECONDS, acquire_scheduler_state
from dashboard.data.served_read import wait_shielded
from dashboard.data.task_activity import (
    ACTIVITY_REQUEST_WAIT_SECONDS,
    RUNS_DB,
    acquire_activity,
)
from dashboard.data.task_lineage import acquire_lineage
from dashboard.data.task_snapshot import acquire_snapshot
from dashboard.project_dbs import _project_scoped_dbs_labeled

logger = logging.getLogger(__name__)

router = APIRouter()

T = TypeVar('T')


PRDS_BUDGET = TasksBudget(per_project=18.0, total=20.0, root_concurrency=4)
"""The three bounds the /prds per-root fan-out runs under, read at each request.

``per_project`` (18.0 s) holds a root's two sequential stages. The first is
its snapshot unit's roster, ``task_snapshot.PER_CALL_TIMEOUT`` (4.4) x its
three bounded operations = 13.2 s; the lineage index, scheduler state and
activity run beside it and wait at most 5.0, 4.0 and 4.0 s. The second is its
PRD documents, waiting at most ``prd_signal.PRD_DOCUMENTS_REQUEST_WAIT_SECONDS``
(4.0): 13.2 + 4.0 = 17.2 s. ``total`` (20.0 s) is the request stage's
``REQUEST_READS_WAIT_SECONDS`` (2.0) plus one root's share, and 10 s under
``data.js``'s 30 s fetch abort, so a degraded payload still renders.
``root_concurrency`` (4) keeps the walk parallel but bounded.
``dashboard/tests/test_prds_budget.py`` checks the constraints.
"""

REQUEST_READS_WAIT_SECONDS = 2.0
"""The longest a request waits on the escalation corpus and on the run-ledger roots
walk before the per-root fan-out.

2.0 + ``PRDS_BUDGET.per_project`` (18.0) = ``PRDS_BUDGET.total`` (20.0), so the
stage never shortens a root's share. A read still in flight lands for the next poll.
"""

STAND_IN_VIEW_BOUND_SECONDS = 900
"""The freshness bound of the unknown served in place of a view or contention
line this render could not acquire, build or validate.

An unknown has no ``as_of``, so its bound only states the contract a served
datum keeps. A built datum takes its oldest input's bound, so a stand-in
states the loosest any input carries, the lineage index's;
``dashboard/tests/test_prds_budget.py`` pins that equality.
"""


@dataclass(frozen=True, slots=True)
class RootUnserved:
    """A root whose inputs this render could not acquire, and why."""

    reason: str


@dataclass(frozen=True, slots=True)
class RootInputs:
    """What one root's two datums are built from."""

    view: PrdViewInputs
    contention: ContentionInputs


@dataclass(frozen=True, slots=True)
class _RequestReads:
    """The reads one request makes once for every root: the corpus and each root's runs.db.

    ``runs_db_absent_reason`` says why the request has no run-ledger
    connections; it is None when the walk finished.
    """

    corpus: Datum[EscalationCorpus]
    runs_dbs: Mapping[str, aiosqlite.Connection | None]
    runs_db_absent_reason: str | None


async def _open_run_ledgers(
    config: DashboardConfig, pool: DbPool,
) -> tuple[Mapping[str, aiosqlite.Connection | None], str | None]:
    """Each root's runs.db connection by root, or none and why the walk gave none."""
    wait = REQUEST_READS_WAIT_SECONDS
    try:
        labeled = await wait_shielded(
            _project_scoped_dbs_labeled(config, pool, RUNS_DB), wait, what='the run-ledger walk',
        )
    except TimeoutError:
        logger.warning(
            'the run-ledger walk did not finish within its %.1fs request wait; every '
            "root's activity is served its last good value or UNKNOWN for this render",
            wait,
        )
        return {}, f'the run-ledger walk did not finish within {wait:.1f}s'
    except Exception as exc:
        logger.warning(
            "the run-ledger walk raised; every root's activity is served its last good "
            'value or UNKNOWN for this render; this is a BUG, not an outage',
            exc_info=True,
        )
        return {}, f'the run-ledger walk raised {type(exc).__name__}: {exc}'
    return dict(labeled), None


async def _acquire_request_reads(
    config: DashboardConfig, pool: DbPool, *, now: datetime,
) -> _RequestReads:
    """The corpus and the run-ledger connections, each waited on at most REQUEST_READS_WAIT_SECONDS."""
    corpus, (runs_dbs, absent_reason) = await asyncio.gather(
        serve_corpus(corpus_queues(config), now=now, wait_seconds=REQUEST_READS_WAIT_SECONDS),
        _open_run_ledgers(config, pool),
    )
    return _RequestReads(corpus=corpus, runs_dbs=runs_dbs, runs_db_absent_reason=absent_reason)


async def _acquire_inputs(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    root: Path,
    reads: _RequestReads,
    *,
    now: datetime,
) -> RootInputs:
    snapshot, lineage, scheduler_state, activity = await asyncio.gather(
        acquire_snapshot(client, config, root, now=now),
        acquire_lineage(client, config, root, now=now),
        acquire_scheduler_state(
            client, config, root, now=now, wait_seconds=SCHEDULER_STATE_REQUEST_WAIT_SECONDS,
        ),
        acquire_activity(
            reads.runs_dbs.get(str(root)), root, now=now,
            wait_seconds=ACTIVITY_REQUEST_WAIT_SECONDS,
            absent_reason=reads.runs_db_absent_reason,
        ),
    )
    keys = open_work_keys(
        str(root), snapshot.status_map, snapshot.rows.value or [], lineage.value or {},
    )
    documents = await read_prd_documents(
        root, keys, now=now, wait_seconds=PRD_DOCUMENTS_REQUEST_WAIT_SECONDS,
    )
    view = PrdViewInputs(
        project_root=str(root),
        status_map=snapshot.status_map,
        census=snapshot.census,
        rows=snapshot.rows,
        lineage=lineage,
        escalations=pending_level2_by_task(reads.corpus, str(root)),
        activity=activity,
        documents=documents,
        liveness=map_datum(snapshot.rows, lambda _rows: snapshot.holder_liveness),
    )
    contention = ContentionInputs(
        project_root=str(root),
        status_map=snapshot.status_map,
        census=snapshot.census,
        rows=snapshot.rows,
        activity=activity,
        scheduler=scheduler_state,
    )
    return RootInputs(view=view, contention=contention)


async def _acquire_within_budget(
    client: httpx.AsyncClient,
    config: DashboardConfig,
    pool: DbPool,
    *,
    now: datetime,
    budget: TasksBudget,
) -> dict[Path, RootInputs | RootUnserved]:
    """Every configured root's inputs, or why a root has none, in canonical root order.

    Mirrors ``dashboard/src/dashboard/data/active_tasks.py::_serve_roots_within_budget``
    without its admission rotation: one deadline, a bounded semaphore, and a
    per-root ``wait_for`` sized after admission, so one slow or broken root
    never blanks the others. The request-wide reads come first, each waited on
    at most ``REQUEST_READS_WAIT_SECONDS``, and one that fails or misses that
    wait degrades its own input, never the response.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget.total
    slots = asyncio.Semaphore(budget.root_concurrency)
    labels = config.project_labels
    reads = await _acquire_request_reads(config, pool, now=now)

    async def _one(root: Path) -> RootInputs | RootUnserved:
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
                    _acquire_inputs(client, config, root, reads, now=now),
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
    logger.warning(
        'project %s: %s; its PRD view and contention line are UNKNOWN for this render',
        label, reason,
    )
    return RootUnserved(reason)


def _served(
    what: str,
    build: Callable[[], Datum[T]],
    validate: Callable[[T], None],
    served_at: datetime,
) -> Datum[T]:
    """*build*'s datum, validated here as well as where it was built; unknown if either breaks."""
    try:
        datum = build()
        if datum.value is not None:
            validate(datum.value)
        validate_datum(datum, served_at)
    except DatumContractError as broken:
        logger.warning(
            '%s breaks the %s invariant and is served as UNKNOWN for this render; this is '
            'a BUG, not an outage',
            what, broken.invariant.value, exc_info=broken,
        )
        return unknown_datum(
            f'{what} broke the {broken.invariant.value} invariant: {broken}',
            STAND_IN_VIEW_BOUND_SECONDS,
        )
    except Exception:
        logger.warning(
            'unexpected error while building %s, which is served as UNKNOWN for this '
            'render; this is a BUG, not an outage',
            what, exc_info=True,
        )
        return unknown_datum(
            f'unexpected error while building {what} (see the WARNING and its traceback)',
            STAND_IN_VIEW_BOUND_SECONDS,
        )
    return datum


def _served_view(
    label: str, inputs: RootInputs | RootUnserved, served_at: datetime,
) -> Datum[PrdView]:
    """*label*'s view at *served_at*."""
    if isinstance(inputs, RootUnserved):
        return unknown_datum(inputs.reason, STAND_IN_VIEW_BOUND_SECONDS)
    view = inputs.view
    return _served(
        f"project {label}'s PRD view",
        lambda: build_prd_view(view, now=served_at),
        lambda value: validate_prd_view(value, view.status_map),
        served_at,
    )


def _served_contention(
    label: str, inputs: RootInputs | RootUnserved, served_at: datetime,
) -> Datum[Contention]:
    """*label*'s contention line at *served_at*."""
    if isinstance(inputs, RootUnserved):
        return unknown_datum(inputs.reason, STAND_IN_VIEW_BOUND_SECONDS)
    contention = inputs.contention
    return _served(
        f"project {label}'s contention line",
        lambda: build_contention(contention, now=served_at),
        validate_contention,
        served_at,
    )


@router.get('/api/v2/dashboard/prds')
async def api_prds(request: Request) -> JSONResponse:
    """One PRD view and one contention line per configured root, keyed by its label.

    ``render_at`` stamps what this render measures; ``served_at`` is resolved
    after the fan-out, and every datum is built and validated at it, so the
    composite's aged inputs and the freshness check read the same instant.
    """
    config: DashboardConfig = request.app.state.config
    client: httpx.AsyncClient = request.app.state.http_client
    pool: DbPool = request.app.state.db
    clock = app_clock(request.app)
    render_at = clock()
    acquired = await _acquire_within_budget(
        client, config, pool, now=render_at, budget=PRDS_BUDGET,
    )
    served_at = clock()
    labels = config.project_labels
    views: dict[str, object] = {}
    contentions: dict[str, object] = {}
    for root, inputs in acquired.items():
        label = labels.of(root)
        views[label] = _served_view(label, inputs, served_at).to_wire()
        contentions[label] = _served_contention(label, inputs, served_at).to_wire()
    return JSONResponse({
        'PRD_VIEW': views,
        'PRD_CONTENTION': contentions,
        'served_at': served_at.isoformat(),
    })
