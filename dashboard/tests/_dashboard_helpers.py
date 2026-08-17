"""Non-fixture test helpers for dashboard tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from dashboard.config import DashboardConfig


def live_aiosqlite_worker_threads() -> list[threading.Thread]:
    """Return every currently-alive aiosqlite worker thread in this process.

    ``aiosqlite.Connection.__init__`` builds its worker as
    ``Thread(target=_connection_worker_thread, args=(self._tx,))``, so a live
    worker is identified by ``thread._target.__name__ ==
    '_connection_worker_thread'``.  The thread's *name* is not usable: it is the
    generic auto-assigned ``Thread-N (_connection_worker_thread)`` on CPython
    3.10+ but plain ``Thread-N`` on older/alternate runtimes, and nothing in
    aiosqlite pins it.

    PRIVATE-ATTRIBUTE PIN.  ``Thread._target`` is a CPython implementation
    detail and ``_connection_worker_thread`` is an aiosqlite private module
    function.  Both are verified against **aiosqlite >=0.22.x** — bump and
    re-verify this pin if either moves, exactly as the ``_connection`` /
    ``_running`` / ``_thread`` pin in ``test_db.py`` documents.

    The ``getattr(..., None)`` guards below mean a moved attribute degrades to
    "no workers found" rather than an ``AttributeError`` mid-assertion — which
    would make every leak assertion built on this pass VACUOUSLY.  What makes
    such a move LOUD is
    ``test_db.py::TestLiveAiosqliteWorkerThreads::test_detects_and_then_stops_detecting_a_real_connection``:
    it opens one real connection and asserts this function finds exactly that
    thread, then stops finding it after ``close()``.  A positive round-trip is
    used rather than a ``hasattr`` guard on ONE of the two names because
    ``Thread._target`` is equally load-bearing and equally movable (``Thread.run``
    already deletes it in its ``finally``), and because a bare module-level
    ``assert`` is stripped under ``python -O``.
    """
    live: list[threading.Thread] = []
    for thread in threading.enumerate():
        target = getattr(thread, '_target', None)
        if getattr(target, '__name__', None) == '_connection_worker_thread':
            live.append(thread)
    return live


def apply_isolated_env(mp: pytest.MonkeyPatch, root: Path) -> None:
    """Point every DashboardConfig-derived path at *root* instead of the live checkout.

    Sets ``DASHBOARD_PROJECT_ROOT``, which redirects ``burndown_db``,
    ``metrics_db``, ``runs_db``, ``escalations_dir``, ``memory_evals_dir`` and
    ``load_samples_db`` (task 3503).  The dashboard app's ``lifespan()`` opens
    ``burndown_db`` and ``metrics_db`` as **writable WAL** stores, so without
    this every ``TestClient(app)`` in the suite wrote into the operator's live
    ``data/burndown/``.

    Also DELETES ``DASHBOARD_KNOWN_PROJECT_ROOTS``.  ``project_root`` is not the
    only root the app fans out over: ``from_env()`` reads that var into
    ``known_project_roots``, and ``_project_scoped_dbs`` / ``_cost_dbs`` /
    ``_performance_resources`` (dashboard/src/dashboard/app.py) plus
    ``data/burndown.py`` then ``DbPool.get(root / 'data/orchestrator/runs.db')``
    for EVERY entry.  Left ambient, that reopens live WAL databases in whatever
    checkouts the operator registered — the same task-3466
    ``SQLITE_READONLY_RECOVERY`` class this helper exists to close.  Ambient
    presence is live, not hypothetical: it is a shared registry var read across
    fused-memory, and the installed dashboard systemd unit sets it.

    Also DELETES ``RECONCILIATION_DATA_DIR`` and ``QUEUE_DATA_DIR``.  Those are
    read straight from ``os.environ`` by ``DashboardConfig._runtime_data_dir``
    and WIN over ``project_root`` for ``reconciliation_db``, ``tickets_db``,
    ``write_queue_db``, ``write_journal_db`` and
    ``reconciliation_escalations_dir`` — so setting only
    ``DASHBOARD_PROJECT_ROOT`` would leave them pointed wherever the ambient
    environment says.  ``reconciliation_db`` and ``tickets_db`` are exactly the
    two read-only ``DbPool.get()`` opens in ``_metrics_loop`` that produced the
    task-3466 ``SQLITE_READONLY_RECOVERY``, so that gap would leave the
    incident's own trigger path un-isolated.  Ambient presence is live, not
    hypothetical: the orchestrator's managed fused-memory spawn
    (``orchestrator/src/orchestrator/mcp_lifecycle.py``) injects both.

    DELETE rather than redirect, deliberately.  For the two runtime dirs,
    deleting makes the config fall back to ``project_root``-relative paths,
    which keeps ``test_scaffold.py``'s
    ``TestConfigDefaults.test_config_derived_paths`` assertions valid — and in
    fact makes them hermetic, since today they silently depend on the ambient
    environment happening to have these unset.  Redirecting under *root* would
    instead break them.  For ``DASHBOARD_KNOWN_PROJECT_ROOTS``, deleting yields
    the empty list, i.e. exactly one root to fan out over; there is no temp
    path to redirect it to that would be more isolated than none.

    A plain function rather than a fixture so the env contract is directly
    unit-testable against a simulated operator environment — a session-scoped
    autouse fixture cannot be re-run from inside a test.
    """
    mp.setenv('DASHBOARD_PROJECT_ROOT', str(root))
    mp.delenv('DASHBOARD_KNOWN_PROJECT_ROOTS', raising=False)
    mp.delenv('RECONCILIATION_DATA_DIR', raising=False)
    mp.delenv('QUEUE_DATA_DIR', raising=False)


# ---------------------------------------------------------------------------
# Dual-payload memory-eval / escalation artifact tree (task 3471)
# ---------------------------------------------------------------------------

_DUAL_EVAL_ID = 'e1-retrieval-health'
_DUAL_RUN_STAMPS = ('20260703T031500Z', '20260704T031500Z', '20260705T031500Z')
_DUAL_LINKED_FINGERPRINT = f'eval:{_DUAL_EVAL_ID}|metric:dangling-pointers|item:node-7'
_DUAL_STORM_FINGERPRINT = f'storm:{_DUAL_RUN_STAMPS[-1]}'
# Deliberately names an eval and metric the tree does not contain, so no
# verdict can ever claim it — that is what makes it `no_matching_verdict`.
_DUAL_ORPHAN_FINGERPRINT = 'eval:e9-absent|metric:never-emitted'


@dataclass(frozen=True)
class DualEscalationTree:
    """What :func:`build_dual_escalation_tree` wrote, and where.

    ``config`` is the single :class:`DashboardConfig` BOTH payload producers
    read — that shared root is the whole point of the fixture, since the
    coverage gap it closes is that no test had ever built ``MEMORY_EVALS`` and
    ``ESCALATIONS`` from one directory.

    The id fields name the records by the MEMORY_EVALS reach path each one is
    written to exercise, so a test can assert about a specific path without
    re-deriving which record landed where.
    """

    config: DashboardConfig
    linked_id: str
    """Reaches ``evals[i].metrics[j].escalation`` — its fingerprint matches a verdict.

    In ``storm=True`` mode it reaches ``unmatched_escalations`` instead, with
    reason ``storm_suppressed``: a triggered storm collapses per-metric links
    into the aggregate, so the row renders none.  Same record, different reach
    path — which is exactly the behaviour the two modes exist to cover.
    """

    unmatched_id: str
    """Reaches ``unmatched_escalations`` with reason ``no_matching_verdict``."""

    unfingerprinted_id: str
    """Reaches ``unmatched_escalations`` with reason ``no_fingerprint``.

    Carries no ``dedupe_fingerprint``, so it never enters the index at all —
    ``_index_escalations`` carries it out in its second return value instead.
    """

    storm_id: str | None
    """Reaches ``storm_escape.escalation``; ``None`` unless built with ``storm=True``."""

    resolved_id: str
    """Reaches ESCALATIONS ONLY — closed, so ``_index_escalations`` drops it."""

    other_category_id: str
    """Reaches ESCALATIONS ONLY — not ``eval_regression``, so the join skips it."""


def dump_artifact(path: Path, body: Any) -> Path:
    """Write *body* in the producers' canonical artifact serialization.

    ``indent=2, sort_keys=True, ensure_ascii=False`` plus a trailing newline —
    the form ``shared/tests/fixtures/memory_eval/README.md`` pins for the
    committed exemplars, mirroring ``test_memory_evals_data._dump`` so the
    tmp_path trees are byte-shaped like the real ones.

    PUBLIC so that every caller building one of these trees goes through ONE
    serialization.  A caller that hand-rolls ``json.dumps(..., indent=2)``
    instead silently drops ``sort_keys``/``ensure_ascii`` and stops being
    byte-shaped like the real artifacts, which is the claim this docstring
    makes on behalf of every tree built here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    return path


def write_escalation_record(
    esc_dir: Path,
    esc_id: Any,
    *,
    category: str = 'eval_regression',
    status: str = 'pending',
    dedupe_fingerprint: str | None = None,
    summary: str = 'memory-eval regression',
    severity: str = 'blocking',
    level: int = 1,
    timestamp: str = '2026-07-30T03:15:00+00:00',
    omit_id: bool = False,
    **extra: Any,
) -> Path:
    """Write one escalation queue record at ``<esc_dir>/<esc_id>.json``.

    Field names mirror ``escalation.models.Escalation`` (``timestamp``, not
    ``created_at``) since that is what the queue serialises and what
    ``load_queue_escalations`` passes through unchanged.  The
    ``category='eval_regression'`` / ``status='pending'`` defaults are exactly
    the pair ``memory_evals._index_escalations`` accepts; every other
    combination is dropped by that reader and can only ever appear on the
    ESCALATIONS side.

    *esc_id* is deliberately untyped: it is written into the ``id`` field
    VERBATIM, so a caller can hand it a JSON number and get an ``id`` that
    survives to both payloads un-coerced — which is what makes a str/int drift
    testable rather than hypothetical.  ``omit_id=True`` leaves the ``id`` key
    out of the record entirely, the shape that makes a queue file address
    NOTHING; ``**extra`` overrides or adds any other field.

    KNOWN DUPLICATION, deliberately left: ``test_memory_evals_data.py`` carries
    its own ``_write_escalation``/``_dump`` with this same field set.  Folding
    those into this module is the right consolidation but edits a file outside
    task 3471's lock scope, so it is not done here.
    """
    body: dict[str, Any] = {
        'id': esc_id,
        'task_id': 'memory-eval-e1',
        'agent_role': 'memory-eval-runner',
        'severity': severity,
        'category': category,
        'summary': summary,
        'detail': '',
        'timestamp': timestamp,
        'status': status,
        'level': level,
        'dedupe_fingerprint': dedupe_fingerprint,
    }
    if omit_id:
        del body['id']
    body.update(extra)
    return dump_artifact(esc_dir / f'{esc_id}.json', body)




def build_dual_escalation_tree(tmp_path: Path, *, storm: bool = False) -> DualEscalationTree:
    """One artifact tree feeding BOTH dashboard payloads that carry escalation ids.

    ``build_memory_evals`` and ``build_escalation_queues`` read the same queue
    directory but had no shared fixture: the set of test modules importing each
    was disjoint, so nothing ever checked that an id one payload emits is one
    the other can resolve.  This builds the tree both of them read.

    Every path is resolved from *config properties*, never hand-spelled.
    ``reconciliation_escalations_dir`` honours ``RECONCILIATION_DATA_DIR``
    ahead of ``project_root`` (config.py ``_runtime_data_dir``), so a spelled
    path would write where only one of the two readers looks — and the test
    would then be checking a directory the producers do not share, which is
    the one thing this fixture exists to avoid.

    That same env-var precedence is also the hazard, so it is ASSERTED and not
    merely documented: with ``RECONCILIATION_DATA_DIR`` set, the resolved queue
    dir is wherever that var points and this function would ``mkdir`` and write
    six escalation records into a LIVE reconciliation queue, outside
    *tmp_path*.  The suite's session-scoped autouse fixture deletes the var,
    but conftest.py states outright that this sets a default and not a lock — a
    function-scoped ``monkeypatch.setenv`` overrides it, and a caller reaching
    this helper from outside that fixture gets no protection at all.  The
    containment check below turns that from a silent write into a failure.

    Args:
        storm: write a TRIGGERED ``storm_escape`` block into the root verdicts
            artifact, with an aggregate escalation to match.

    WHY ``storm`` IS A MODE AND NOT JUST ANOTHER RECORD.  The two states are
    mutually exclusive in the producer, measured not assumed: ``_build_eval``
    consults the escalation index only when the program-wide storm block is
    ``None``, so a triggered storm empties EVERY per-metric link and moves
    those fingerprints to ``unmatched_escalations`` with reason
    ``storm_suppressed``.  ``evals[i].metrics[j].escalation`` and
    ``storm_escape.escalation`` therefore cannot both be populated in one
    payload — the storm exists precisely to collapse the per-metric links into
    one aggregate filing.  Covering both reach paths takes two trees; a caller
    wanting full coverage builds one of each and takes the union.

    Returns a :class:`DualEscalationTree` naming each record by the reach path
    it exercises.
    """
    config = DashboardConfig(project_root=tmp_path, known_project_roots=[])
    root = config.memory_evals_dir
    esc_dir = config.reconciliation_escalations_dir
    assert esc_dir.is_relative_to(tmp_path), (
        f'the resolved escalations queue dir {str(esc_dir)!r} is NOT under the '
        f'tmp_path {str(tmp_path)!r} this tree was asked to build in. '
        '`DashboardConfig.reconciliation_escalations_dir` reads '
        'RECONCILIATION_DATA_DIR ahead of project_root, so this environment is '
        'un-isolated and building here would write escalation records into a '
        'live reconciliation queue. Delete RECONCILIATION_DATA_DIR (see '
        'apply_isolated_env) rather than relaxing this check.'
    )
    esc_dir.mkdir(parents=True, exist_ok=True)

    # M1 metrics series — one file per run, ordered by filename stamp.
    for stamp in _DUAL_RUN_STAMPS:
        dump_artifact(root / _DUAL_EVAL_ID / f'metrics-{stamp}.json', {
            'schema_version': 1,
            'eval_id': _DUAL_EVAL_ID,
            'run_stamp': stamp,
            'corpus': {
                'project_id': 'dark_factory',
                'counts': {'entities_and_relations': 1204, 'temporal_facts': 588},
            },
            'metrics': [
                {
                    'metric_id': 'canonical-in-top-5', 'kind': 'proportion',
                    'value': 0.94, 'n': 50, 'denominator': 50,
                },
                {'metric_id': 'dangling-pointers', 'kind': 'count', 'value': 4.0, 'n': 4},
            ],
        })

    # Per-eval limits (provenance only — no dashboard verdict comes from here).
    dump_artifact(root / _DUAL_EVAL_ID / 'limits-current.json', {
        'schema_version': 1,
        'eval_id': _DUAL_EVAL_ID,
        'run_stamp': _DUAL_RUN_STAMPS[-1],
        'alpha': 0.002777777777777778,
        'false_alarm_budget': 1.0,
        'runs_per_quarter': 90,
        'min_samples': 10,
        'baseline_window': 3,
        'baseline_run_stamps': list(_DUAL_RUN_STAMPS[:2]),
        'grandfather_set_hash': 'f8c4' * 16,
        'generator': 'shared.memory_eval_limits',
        'verdicts': [
            {'metric_id': 'canonical-in-top-5', 'rule_kind': 'proportion', 'status': 'ok', 'alarms': []},
            {'metric_id': 'dangling-pointers', 'rule_kind': 'count', 'status': 'ok', 'alarms': []},
        ],
    })

    # Root-scoped M2 verdicts.  The alarming metric's fingerprint is what the
    # escalation below joins on, so the metric-row reach path yields a link.
    verdicts: dict[str, Any] = {
        'schema_version': 1,
        'run_stamp': _DUAL_RUN_STAMPS[-1],
        'entries': [
            {
                'eval_id': _DUAL_EVAL_ID, 'metric_id': 'dangling-pointers',
                'verdict': 'alarm', 'fingerprint': _DUAL_LINKED_FINGERPRINT,
                'value': 4.0, 'limit_ref': 'count>=3',
                'run_stamp': _DUAL_RUN_STAMPS[-1], 'item': 'node-7',
            },
            {
                'eval_id': _DUAL_EVAL_ID, 'metric_id': 'canonical-in-top-5',
                'verdict': 'no_alarm',
                'fingerprint': f'eval:{_DUAL_EVAL_ID}|metric:canonical-in-top-5',
                'value': 0.94, 'limit_ref': None,
                'run_stamp': _DUAL_RUN_STAMPS[-1],
            },
        ],
    }
    storm_id: str | None = None
    if storm:
        # Run-scoped and only surfaced while TRIGGERED — an untriggered block
        # is dropped by `_load_verdicts` and would render as no storm at all.
        verdicts['storm_escape'] = {
            'triggered': True,
            'alarm_count': 9,
            'aggregate_fingerprint': _DUAL_STORM_FINGERPRINT,
        }
        storm_id = 'esc-eval-storm'
        write_escalation_record(
            esc_dir, storm_id, dedupe_fingerprint=_DUAL_STORM_FINGERPRINT,
            summary='memory-eval storm escape',
        )
    dump_artifact(root / 'verdicts-current.json', verdicts)

    # (1) the linked record — a metric-row link, or a storm_suppressed
    #     unmatched entry when the storm collapsed the per-metric links.
    write_escalation_record(esc_dir, 'esc-eval-linked',
                            dedupe_fingerprint=_DUAL_LINKED_FINGERPRINT)
    # (2) open, fingerprinted, and claimed by no verdict — `no_matching_verdict`.
    write_escalation_record(esc_dir, 'esc-eval-unmatched',
                            dedupe_fingerprint=_DUAL_ORPHAN_FINGERPRINT,
                            summary='memory-eval regression nothing claims')
    # (3) open with NO usable fingerprint — `no_fingerprint`.  Reaches the
    #     payload through `_index_escalations`' second return value rather than
    #     the index, so it is a distinct code path, not a second instance of (2).
    write_escalation_record(esc_dir, 'esc-eval-unfingerprinted', dedupe_fingerprint=None,
                            summary='memory-eval regression with no fingerprint')

    # (4)+(5) NOT reachable from MEMORY_EVALS at all — one closed, one of
    # another category.  `shape_escalations` ships both (it filters nothing);
    # `_index_escalations` drops both.  They are what makes the containment
    # STRICT and therefore observable: without them the two id spaces would be
    # equal in this fixture and the asymmetry test would have nothing to see.
    #
    # NOT gated behind a keyword: they cannot disturb the reach-path
    # preconditions, because being dropped by that reader is the entire reason
    # they are here.  Distinct fingerprints rather than a collision with (1),
    # so their exclusion is unambiguously the status/category filter rather
    # than the duplicate-fingerprint one.
    write_escalation_record(esc_dir, 'esc-eval-resolved', status='resolved',
                            dedupe_fingerprint='eval:e1-retrieval-health|metric:closed-alarm',
                            summary='memory-eval regression an operator closed')
    write_escalation_record(esc_dir, 'esc-other-category', category='reconciliation_drift',
                            dedupe_fingerprint='recon:drift|entity:node-7',
                            summary='not a memory-eval regression at all')

    return DualEscalationTree(
        config=config,
        linked_id='esc-eval-linked',
        unmatched_id='esc-eval-unmatched',
        unfingerprinted_id='esc-eval-unfingerprinted',
        storm_id=storm_id,
        resolved_id='esc-eval-resolved',
        other_category_id='esc-other-category',
    )


RECONCILIATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS watermarks (
    project_id TEXT PRIMARY KEY,
    last_full_run_id TEXT,
    last_full_run_completed TEXT,
    last_episode_timestamp TEXT,
    last_memory_timestamp TEXT,
    last_task_change_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_type TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    events_processed INTEGER DEFAULT 0,
    stage_reports TEXT DEFAULT '{}',
    status TEXT DEFAULT 'running'
);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS journal_entries (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage TEXT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    target_system TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT,
    reasoning TEXT DEFAULT '',
    evidence TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_journal_run ON journal_entries(run_id);

CREATE TABLE IF NOT EXISTS judge_verdicts (
    run_id TEXT PRIMARY KEY,
    reviewed_at TEXT NOT NULL,
    severity TEXT NOT NULL,
    findings TEXT DEFAULT '[]',
    action_taken TEXT DEFAULT 'none'
);
CREATE INDEX IF NOT EXISTS idx_verdicts_reviewed ON judge_verdicts(reviewed_at);

CREATE TABLE IF NOT EXISTS event_buffer (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_source TEXT NOT NULL,
    agent_id TEXT,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'buffered'
);
CREATE INDEX IF NOT EXISTS idx_eb_project_status ON event_buffer(project_id, status);
CREATE INDEX IF NOT EXISTS idx_eb_agent_timestamp ON event_buffer(agent_id, timestamp)
    WHERE agent_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS reconciliation_locks (
    project_id TEXT PRIMARY KEY,
    instance_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS burst_state (
    agent_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'idle',
    last_write_at TEXT NOT NULL,
    burst_started_at TEXT
);

CREATE TABLE IF NOT EXISTS chunk_boundaries (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT,
    events_count INTEGER,
    status TEXT DEFAULT 'processing',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_project ON chunk_boundaries(project_id);

CREATE TABLE IF NOT EXISTS run_actions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target TEXT NOT NULL,
    operation TEXT NOT NULL,
    detail TEXT DEFAULT '{}',
    causation_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ra_run ON run_actions(run_id);
"""

assert 'started_at TEXT NOT NULL' in RECONCILIATION_SCHEMA, (
    "RECONCILIATION_SCHEMA must contain 'started_at TEXT NOT NULL'; "
    "update RELAXED_RECONCILIATION_SCHEMA derivation if the schema changes."
)

RELAXED_RECONCILIATION_SCHEMA = RECONCILIATION_SCHEMA.replace(
    'started_at TEXT NOT NULL', 'started_at TEXT'
)


@asynccontextmanager
async def make_recon_db(
    tmp_path: Path,
    inserts: Sequence[str | tuple[str, Any]],
    *,
    name: str = 'test.db',
    schema: str | None = None,
):
    """Async context manager that creates a temporary SQLite reconciliation DB.

    Creates a DB at ``tmp_path / name``, applies ``schema`` (defaults to
    ``RECONCILIATION_SCHEMA``), executes each statement in ``inserts``, then
    yields an :class:`aiosqlite.Connection` with ``row_factory`` set to
    :class:`aiosqlite.Row`.  The connection is closed on context exit.
    """
    if schema is None:
        schema = RECONCILIATION_SCHEMA

    db_path = tmp_path / name
    sync_conn = sqlite3.connect(str(db_path))
    sync_conn.executescript(schema)
    for stmt in inserts:
        if isinstance(stmt, str):
            sync_conn.execute(stmt)
        else:
            sql, params = stmt
            sync_conn.execute(sql, params)
    sync_conn.commit()
    sync_conn.close()

    async with aiosqlite.connect(str(db_path)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn
