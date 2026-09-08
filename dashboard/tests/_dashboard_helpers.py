"""Non-fixture test helpers for dashboard tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.
"""

from __future__ import annotations

import html.parser
import json
import re
import sqlite3
import threading
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import aiosqlite
import httpx
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
# Mocked MCP wire envelopes (task 3952)
#
# ONE definition of each of the three envelopes a mocked MCP server returns.
# Both consumers build from these: the cold-session response LIST below, and
# every test module's ``_PerPortHandler`` / ``_SessionAwareHandler``, which
# import these rather than redefining them.  That single source is the point —
# these envelopes encode what ``dashboard.data.memory.McpSession`` accepts
# (the negotiated ``protocolVersion``, the ``mcp-session-id`` header, the
# ``result.content[0].text`` JSON-in-text nesting), so were they duplicated
# per module a change on the McpSession side could be applied to some copies
# and not others, leaving half the suite green against a stale envelope.
# ---------------------------------------------------------------------------

MCP_SESSION_ID = 'test-session-id'
"""The ``mcp-session-id`` every mocked response carries.

McpSession reads this off the initialize response and echoes it on subsequent
posts; tests that assert on session reuse match against this exact value.
"""


def mcp_init_response(request_id: int = 1) -> httpx.Response:
    """The ``initialize`` result a mocked MCP server returns."""
    return httpx.Response(
        200,
        json={
            'jsonrpc': '2.0',
            'id': request_id,
            'result': {
                'protocolVersion': '2025-03-26',
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'test', 'version': '0.1'},
            },
        },
        headers={'mcp-session-id': MCP_SESSION_ID},
    )


def mcp_notify_response() -> httpx.Response:
    """The 202 Accepted a mocked MCP server returns for ``notifications/*``.

    Bodiless by protocol — a notification has no id and takes no result.
    """
    return httpx.Response(202, headers={'mcp-session-id': MCP_SESSION_ID})


def mcp_tool_response(inner: dict, request_id: int = 1) -> httpx.Response:
    """A ``tools/call`` result carrying *inner* as JSON text content.

    MCP nests the tool's own payload as a JSON *string* inside
    ``result.content[0].text``, so *inner* is serialized, not embedded — which
    is exactly the double-encoding the dashboard readers have to undo.
    """
    return httpx.Response(
        200,
        json={
            'jsonrpc': '2.0',
            'id': request_id,
            'result': {
                'content': [{'type': 'text', 'text': json.dumps(inner)}],
            },
        },
        headers={'mcp-session-id': MCP_SESSION_ID},
    )


def cold_session_responses(
    inner: dict, url: str = 'http://localhost:8000',
) -> list[httpx.Response]:
    """The three responses a COLD McpSession consumes, in post order.

    ``mcp_tool_call`` against a cold session issues ``initialize``, then
    ``notifications/initialized``, then ``tools/call`` — three HTTP posts.
    An AsyncMock client bypasses MockTransport, which is what normally
    attaches ``.request``, so each response needs it set by hand or
    ``raise_for_status()`` raises RuntimeError even on a 200.

    The envelopes come from the three builders above — the same ones the mock
    handlers serve — so an AsyncMock-driven test and a MockTransport-driven
    test can never be asserting against different wire shapes.  What this
    function adds over calling them directly is the ordering and the
    ``resp.request`` attachment, which is the easy-to-drop part that was
    copy-pasted byte-for-byte across four test modules.
    """
    responses = [
        mcp_init_response(),
        mcp_notify_response(),
        mcp_tool_response(inner),
    ]
    for resp in responses:
        resp.request = httpx.Request('POST', f'{url.rstrip("/")}/mcp')
    return responses


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


# ---------------------------------------------------------------------------
# JSX source slicing.
#
# There is no JS runtime in this project, so the dashboard suite asserts
# structural contracts against the *served* .jsx text.  Nearly every such
# assertion must first scope itself to one function's body — otherwise a token
# appearing anywhere else in the file satisfies it and the test proves nothing.
#
# These helpers used to be private copies in nine test modules (task 3549),
# under two names covering FOUR distinct implementations, so a fix had to be
# applied nine times or not at all.  Their contract lives in
# test_jsx_source_helpers.py.
#
# There are three of them now.  `find_function_params` is the paren-depth walk
# that locates a declaration's parameter list; `extract_function_body` resumes
# from where it stops to take the body, and test_charts_axis_labels.py's
# `_extract_signature` takes the params themselves.  Those two return DISJOINT,
# adjacent slices of the same declaration, so neither can be built on the
# other — but the walk that finds the boundary between them is one walk, and it
# lives here rather than in each of them.
#
# All three are built on ONE quote-aware scanner, `_scan_js`.  Giving them a
# scanner each would re-create in miniature exactly the duplication this
# consolidation removed — and they need the same answer to the same question:
# which stretches of this text are NOT code?
# ---------------------------------------------------------------------------


class _JsSpan(NamedTuple):
    """One stretch of JS source that is not code: a string literal or a comment.

    ``end`` is exclusive.  ``opener`` is the quote character for a string span
    and ``'//'`` / ``'/*'`` for a comment.  ``closed`` is False when the source
    ran out before the span was terminated.
    """

    start: int
    end: int
    kind: str  # 'string' | 'comment'
    opener: str
    closed: bool


def _scan_js(source: str) -> list[_JsSpan]:
    """Return the string-literal and comment spans of *source*, in order.

    Deliberately not a full JS lexer: a regex literal containing ``//`` or a
    quote (``str.replace(/'/g, '')``), or a quote nested inside a template's
    ``${...}``, would confuse it.  Neither occurs in the assets these helpers
    are used on, and `_assert_js_lexable` turns the failure mode that WOULD
    reach them — an apostrophe in JSX prose — into a loud error.
    """
    spans: list[_JsSpan] = []
    i, n = 0, len(source)

    while i < n:
        ch = source[i]

        if ch in '\'"`':
            start = i
            i += 1
            closed = False
            while i < n:
                if source[i] == '\\':  # an escaped char cannot close the literal
                    i += 2
                    continue
                if source[i] == ch:
                    i += 1
                    closed = True
                    break
                i += 1
            spans.append(_JsSpan(start, min(i, n), 'string', ch, closed))
            continue

        if source[i : i + 2] == '//':
            end = source.find('\n', i)
            end = n if end == -1 else end
            spans.append(_JsSpan(i, end, 'comment', '//', True))
            i = end
            continue

        if source[i : i + 2] == '/*':
            end = source.find('*/', i + 2)
            spans.append(_JsSpan(i, n if end == -1 else end + 2, 'comment', '/*', end != -1))
            i = n if end == -1 else end + 2
            continue

        i += 1

    return spans


def _assert_js_lexable(source: str, spans: Sequence[_JsSpan]) -> None:
    """Raise if the scan hit a state that means it almost certainly misparsed.

    THE FAILURE THIS EXISTS FOR is an apostrophe in JSX *text* — ``don't`` in a
    label — which the scanner reads as an opening quote.  Everything up to the
    next ``'`` anywhere later in the file is then treated as string contents,
    so real comments inside it are left unstripped (prose leaks into what the
    consumers call "code") and real braces inside it are not counted (a body is
    mis-scoped).  Both are silent: the caller gets a plausible string back.

    Two states betray it, and neither can occur in well-formed source that this
    scanner actually understands:

    * a literal still open at end-of-input;
    * a non-template literal spanning a newline — JS forbids a raw newline in a
      ``'``/``"`` literal, so this is a misparse, not a long string.

    A template literal (backtick) legitimately spans lines and is exempt; there
    is one in tabs.jsx.
    """
    def _line(index: int) -> int:
        return source.count('\n', 0, index) + 1

    _APOSTROPHE_HINT = (
        'The likeliest cause is an apostrophe in JSX text (a label reading '
        '`don\'t`), which this scanner reads as an opening quote — everything '
        'after it is then treated as string contents, so comments inside it are '
        'left unstripped and braces inside it are not counted, and the caller '
        'gets a plausible-looking wrong answer. Reword the prose, or write the '
        'apostrophe as `&apos;`. (A regex literal containing a quote, e.g. '
        '`/\'/`, would also do it — see `_scan_js`.)'
    )

    for span in spans:
        if not span.closed:
            what = 'string literal' if span.kind == 'string' else 'block comment'
            raise AssertionError(
                f'Cannot scan this JS source: a {span.opener} {what} opened at '
                f'line {_line(span.start)} is never closed. {_APOSTROPHE_HINT}'
            )
        if span.kind == 'string' and span.opener != '`' and '\n' in source[span.start : span.end]:
            raise AssertionError(
                f'Cannot scan this JS source: a {span.opener} literal opened at '
                f'line {_line(span.start)} spans a newline (it appears to close at '
                f'line {_line(span.end - 1)}). A raw newline is illegal inside a '
                f'{span.opener} literal in JS, so this is a misparse rather than a '
                f'long string. {_APOSTROPHE_HINT}'
            )


def _mask_js(source: str, spans: Sequence[_JsSpan]) -> str:
    """Return *source* with every non-code span blanked, LENGTH PRESERVED.

    Equal length is the whole point: the caller searches and walks the mask but
    slices the ORIGINAL with the indices it finds, so the returned text is the
    real source rather than a blanked copy.  Newlines survive so a line number
    computed from the mask still means something.
    """
    chars = list(source)
    for span in spans:
        for k in range(span.start, span.end):
            if chars[k] != '\n':
                chars[k] = ' '
    masked = ''.join(chars)
    assert len(masked) == len(source), 'the mask must be index-aligned with the source'
    return masked


def find_function_params(
    source: str,
    func_name: str,
    miss: Callable[[str], BaseException] | None = None,
) -> tuple[str, int, int]:
    """Locate ``function <func_name>(``'s parameter list.

    Returns ``(masked, params_start, params_end)`` where *masked* is the
    `_mask_js` copy the caller keeps walking, ``source[params_start:params_end]``
    is the parameter-list text with the parens EXCLUDED, and *params_end* is the
    index OF the matching ``)``.  A caller wanting the body resumes with
    ``masked.find('{', params_end + 1)``.

    The search and the depth walk run over the mask, so a ``(``, ``)`` or the
    word ``function`` inside a STRING LITERAL OR A COMMENT is not counted.  The
    mask is length-preserving and index-aligned, so both indices address the
    ORIGINAL source and the slice a caller takes is the real text.

    Paren-DEPTH rather than "find the next ``)``": a destructured parameter
    (``function Foo({ a, b }) {``) carries its own ``{``/``}`` pair inside the
    parameter list, so a caller that took the first ``{`` after the opening
    paren would get the destructuring pattern instead of the body.

    The search regex is deliberately NOT line-anchored, so a declaration NESTED
    inside another function is found (the real instance is
    ``function statusMatches(s) {`` indented inside ``TasksTab`` in
    tab_tasks.jsx).  Its trailing ``\\s*\\(`` is equally load-bearing in the
    other direction: without it a prefix sibling declared earlier would shadow
    the target — ``function TaskGraphEdges(`` at tab_tasks.jsx:33 precedes
    ``function TaskGraph(`` at :151.

    RAISES on a miss rather than returning a sentinel, because every consumer
    slices the source by the returned indices and a sentinel would hand them a
    silently wrong — or empty — slice, over which absence assertions pass
    vacuously.  *miss* lets a caller supply the exception: `extract_function_body`
    threads its own four-way wording through it, and test_charts_axis_labels.py
    keeps a file-specific message.  The default raises ``AssertionError``.
    """
    def _default_miss(what: str) -> BaseException:
        return AssertionError(
            f'Could not locate the `function {func_name}(` parameter list: '
            f'{what}. Either the function was removed or renamed, or it was '
            f'rewritten as an arrow function or a class method — neither is '
            f'matched, only a named `function` declaration is.'
        )

    _miss = miss if miss is not None else _default_miss

    spans = _scan_js(source)
    _assert_js_lexable(source, spans)
    masked = _mask_js(source, spans)

    match = re.search(rf'\bfunction\s+{re.escape(func_name)}\s*\(', masked)
    if match is None:
        raise _miss('no such declaration in this source')

    paren_depth = 1
    i = match.end()
    while i < len(masked) and paren_depth > 0:
        if masked[i] == '(':
            paren_depth += 1
        elif masked[i] == ')':
            paren_depth -= 1
        i += 1
    if paren_depth != 0:
        raise _miss('its parameter list is never closed')

    return masked, match.end(), i - 1


def extract_function_body(source: str, func_name: str) -> str:
    """Return the brace-delimited body of a ``function <func_name>(`` declaration.

    The returned slice starts at the body's opening ``{`` and ends at its
    matching ``}``, both included — the SIGNATURE AND PARAMETER LIST ARE
    EXCLUDED.  Only named ``function`` declarations are matched: an arrow
    function bound to a const, and a class method spelled ``Foo(a) {``, carry
    no ``function`` keyword and are misses.

    The search and both depth walks run over a `_mask_js` copy of the source, so
    a brace, paren or ``function`` keyword inside a STRING LITERAL OR A COMMENT
    is not counted.  Without that, ``const s = '}'`` inside a body ends the
    brace walk early and the caller gets a truncated, unbalanced slice — every
    absence assertion over which then passes vacuously, which is the same
    permanent false GREEN the raise-on-miss rule below exists to prevent.  The
    mask is index-aligned with the source, so the slice returned is the real
    text, comments and literals intact.

    Paren-depth walks past the parameter list before looking for the body's
    opening ``{`` — a destructured parameter (``function Foo({ a, b }) {``)
    contains its own ``{``/``}`` pair *inside* the parameter list, so naively
    taking the first ``{`` after the opening ``(`` would return just the
    destructuring pattern (e.g. ``{ a, b }``) instead of the function body.

    The search regex is deliberately NOT line-anchored, so a declaration
    NESTED inside another function is found and scoped to its own body (the
    real instance is ``function statusMatches(s) {`` indented inside
    ``TasksTab`` in tab_tasks.jsx).  Its trailing ``\\s*\\(`` is equally
    load-bearing in the other direction: without it a prefix sibling declared
    earlier would shadow the target — ``function TaskGraphEdges(`` at
    tab_tasks.jsx:33 precedes ``function TaskGraph(`` at :151.

    RAISES ``AssertionError`` on any miss rather than returning ``''``.  An
    empty body makes every downstream ABSENCE assertion pass vacuously, which
    is a permanent false GREEN that no amount of care at the call site can
    detect; a loud failure naming the function is strictly better.
    """
    def _miss(what: str) -> AssertionError:
        return AssertionError(
            f'Could not locate the `function {func_name}(` body: {what}. Either '
            f'the function was removed or renamed, or it was rewritten as an '
            f'arrow function or a class method — neither is matched, only a '
            f'named `function` declaration is. This cannot silently return an '
            f'empty body: an absence assertion over one would pass vacuously.'
        )

    masked, _params_start, params_end = find_function_params(
        source, func_name, miss=_miss,
    )

    start = masked.find('{', params_end + 1)
    if start == -1:
        raise _miss('no opening brace follows its parameter list')

    depth = 0
    for j in range(start, len(masked)):
        char = masked[j]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return source[start : j + 1]
    raise _miss('its body brace is never closed')


def strip_js_comments(source: str) -> str:
    """Return *source* with every JS/JSX comment blanked, string literals intact.

    The probes built on these helpers are plain substring/regex searches, so
    without this they also match PROSE.  That coupling deformed the production
    source once already: charts.jsx carried a comment whose content was an
    apology for what it could not say, because naming the very expression the
    component had just stopped using would fail CI with a message claiming the
    component "still contains hole-blind scale/path arithmetic" — pointing at a
    comment.  A comment is exactly where that expression SHOULD be quotable.

    Quote-aware rather than a bare regex: blanking from a ``//`` inside a string
    literal (a URL, say) to end-of-line would delete real CODE, and an absence
    assertion over deleted code is a permanent false GREEN.  All three quote
    styles (``'``, ``"``, backtick) are tracked, and an escaped character inside
    a literal cannot close it.

    Each comment is replaced by a SINGLE SPACE rather than removed outright, so
    two previously separated tokens can never be spliced into a new match.

    Deliberately not a full JS lexer (see `_scan_js` for the exact blind spots).
    The one it is most likely to meet — an apostrophe in JSX text, which reads
    as an opening quote and leaves every comment up to the next ``'`` in the
    file unstripped — is not silently tolerated: `_assert_js_lexable` RAISES on
    it.  A missed comment is a false RED for an absence probe and a false GREEN
    for a presence one, and neither is visible at the call site.
    """
    spans = _scan_js(source)
    _assert_js_lexable(source, spans)

    out: list[str] = []
    prev = 0
    for span in spans:
        if span.kind != 'comment':
            continue  # a string literal is passed through verbatim
        out.append(source[prev : span.start])
        out.append(' ')
        prev = span.end
    out.append(source[prev:])

    return ''.join(out)


# ---------------------------------------------------------------------------
# window.DF_CHARTS namespace destructure/export parsing.
#
# charts.jsx publishes its components as `window.DF_CHARTS = { ... }` and each
# consumer picks them up with `const { ... } = window.DF_CHARTS`.  Several
# suites parse those two lines rather than hardcoding a component list, so the
# list they check against can never drift from what the files actually say.
#
# `destructure_bindings` returns (canonical, local) PAIRS, and each caller
# projects the half it needs.  This is not fussiness — the three consumers it
# replaces answer OPPOSITE questions over the identical line:
#   test_charts_consumer_bindings wants the LOCAL/alias name  — what the file
#       must actually reference, since the alias is what it renders by;
#   test_charts_axis_labels wants the CANONICAL/source name   — what must
#       actually exist on the namespace object;
#   test_tab_burndown wants BOTH, as an alias -> canonical map.
# On tabs.jsx's real `HistBar: HB` those are 'HB' and 'HistBar'.  A primitive
# that picked one side would silently INVERT one of the two suites, which is
# precisely the canonical-vs-alias slip test_charts_consumer_bindings.py
# freezes a negative-control fixture against.
#
# The list shape is load-bearing for the same reason: order is preserved and
# duplicates are NOT collapsed, because the callers' own collection shapes
# differ (dict last-wins / ordered list keeping duplicates / deduped set).
# Share the parser, not the policy — each consumer also keeps its own
# search-vs-finditer choice and its own miss behaviour.
#
# The `[^{}]*` class in both patterns is brace-HOSTILE ON PURPOSE and must NOT
# be widened.  Two independent reasons, from the two suites that documented it:
#   - Widening to swallow the other DF_CHARTS access shapes is actively wrong,
#     not merely extra work: a namespace binding's own name IS used, so a naive
#     extension flags it as a false positive; and member reads off a namespace
#     object are not statically enumerable the way a destructure list is.  The
#     defect these suites exist to catch can only exist in the destructure
#     shape anyway.
#   - A NESTED brace must fail loudly at the call site that names the coupling,
#     rather than yield a half-read binding list that turns a downstream
#     assertion red with an unrelated-looking message.  Three call sites turn
#     the miss into a self-naming assertion for exactly that reason.
#
# Contract: test_jsx_source_helpers.py::TestDfChartsDestructure.
# ---------------------------------------------------------------------------

DF_CHARTS_DESTRUCTURE_RE = re.compile(r'const\s*\{([^{}]*)\}\s*=\s*window\.DF_CHARTS')
"""The CONSUMER shape: `const { Foo, Bar: B } = window.DF_CHARTS`."""

DF_CHARTS_EXPORT_RE = re.compile(r'window\.DF_CHARTS\s*=\s*\{([^{}]*)\}')
"""The PROVIDER shape: `window.DF_CHARTS = { Foo, Bar }` in charts.jsx."""


def destructure_bindings(brace_body: str) -> list[tuple[str, str]]:
    """Split a destructure/object-literal brace body into (canonical, local) pairs.

    *brace_body* is the inside of the braces — typically ``m.group(1)`` from one
    of the two patterns above, though the surrounding braces are harmless.

    ``{ StackedAreaChart, HistBar: HB }`` yields
    ``[('StackedAreaChart', 'StackedAreaChart'), ('HistBar', 'HB')]``: a bare
    name is BOTH canonical and local; an aliased one splits on the FIRST colon,
    canonical left and local right.  Both halves are whitespace-stripped, empty
    parts (a trailing comma, say) produce no entry, and source ORDER is
    preserved with DUPLICATES INTACT so each caller can impose its own
    collection shape.
    """
    pairs: list[tuple[str, str]] = []
    for part in brace_body.strip().strip('{}').split(','):
        part = part.strip()
        if not part:
            continue
        canonical, _, alias = part.partition(':')
        canonical = canonical.strip()
        pairs.append((canonical, alias.strip() or canonical))
    return pairs


# ---------------------------------------------------------------------------
# Balanced-delimiter walking, and window.DF_DATA seed-block extraction.
#
# The dashboard's served data.js carries its fixture payload as a
# `window.DF_DATA = { KEY: { ... }, ... }` literal, and several suites need to
# scope an assertion to ONE key's object so a token elsewhere in the file
# cannot satisfy it.
#
# This used to be a private copy in three test modules (test_tab_escalations,
# test_tab_memory_evals, test_tab_escalation_analytics) whose code was
# byte-identical.  Its contract lives in
# test_jsx_source_helpers.py::TestExtractDfDataBlock.
#
# The WALK is separated from the ANCHOR for the same reason `extract_function_body`
# was rebuilt on `find_function_params`: two helpers needed the identical
# depth loop with DIFFERENT anchors, so keeping the loop in both meant the
# string-literal blind spot below was documented — and would have to be
# fixed — in two places.  `walk_balanced` is the loop;
# `extract_df_data_block` anchors it on data.js's `key: {` seed form and
# test_tab_memory_evals.py's `_extract_const_object` anchors it on a
# module-scope `const NAME = {`/`[` declaration.  Its own contract lives in
# test_jsx_source_helpers.py::TestWalkBalanced.
#
# Two behaviours differ from the sibling `extract_function_body` and are
# deliberately kept as they were rather than changed in the move: these return
# `''` SILENTLY on a miss where the other RAISES (every call site already
# asserts on the returned value, so raising would only relocate their
# failures), and the depth walk is NOT quote-aware where the other is.  Both
# are pinned as current behaviour, so upgrading either later is a visible
# contract edit rather than silent drift.
# ---------------------------------------------------------------------------


def walk_balanced(
    src: str, start: int, open_char: str = '{', close_char: str = '}'
) -> str:
    """Return ``src`` from ``start`` through the delimiter matching ``src[start]``.

    ``start`` must be the index OF the opening delimiter; both callers get it
    from a regex whose pattern ends on that delimiter (``m.end() - 1``).  The
    walk counts ``open_char``/``close_char`` so a NESTED pair does not
    terminate it early — which is the whole reason a ``[^}]*`` regex was
    rejected for this job.

    Returns the delimited text INCLUDING both delimiters, or the empty string
    if the opening delimiter is never closed.  Returning ``''`` rather than
    raising is the deliberate policy of this family (see the banner above).

    Note: the depth walk does not skip delimiters inside JS string literals,
    so a quoted ``{`` or ``}`` miscounts.  This is the single place that
    limitation now lives; the callers document what makes it acceptable for
    the sources they read.
    """
    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    return ''


def extract_df_data_block(src: str, key: str) -> str:
    """Return the body of the ``<key>: { ... }`` seed object, braces included.

    Locates ``<key>:`` followed by ``{`` (allowing arbitrary whitespace), then
    hands off to ``walk_balanced`` to find the matching close brace.
    This is brace-aware: a simple regex ``[^}]*`` would stop at the first
    nested ``}`` and miss later keys.
    Returns the empty string if no matching block is found.

    Note: ``walk_balanced`` does not skip ``{``/``}`` inside JS string
    literals.  This is acceptable because the data.js seed block uses simple
    numeric/array values and does not embed brace characters inside quoted
    strings.
    """
    m = re.search(rf'{re.escape(key)}\s*:\s*\{{', src)
    if m is None:
        return ''
    return walk_balanced(src, m.end() - 1)  # m.end() - 1 is the opening `{`


# ---------------------------------------------------------------------------
# Served-HTML script order.
#
# index.html loads its scripts as classic synchronous tags, so DOCUMENT order
# is EXECUTION order — which is what lets a text-level test assert that a
# provider script loads before the consumer that dereferences it at module
# scope.  That equivalence is fragile: `defer`, `async` or `type="module"` on
# either tag breaks it, and a position comparison over a deferred pair is a
# false pass, not a failure.  Hence the guard below runs BEFORE the comparison.
#
# These three used to be private copies in FIVE test modules
# (test_esc_flow_diagram, test_index_html, test_tab_escalation_analytics,
# test_tab_escalations, test_tab_memory_evals), so a fix to the false-pass
# guard had to be applied five times or not at all.  Their contract lives in
# test_jsx_source_helpers.py::TestScriptOrderHelpers.
#
# The five copies agreed byte-for-byte except in one place, resolved here in
# favour of the canonical 4-of-5 form: the ORDERING failure message ends with
# the caller's `consumer_note`, where test_index_html.py alone substituted a
# fixed two-sentence string.  Nothing pins that message — the only tests that
# match this helper's failure text (test_index_html.py:303 and :338) pin the
# GUARD phrases, which are identical across all five — so both variants were
# green, and this one strictly carries more information: it surfaces the note
# at index_html's 13 note-passing call sites instead of discarding the notes
# at the other four modules' 14 sites.  The guard messages themselves are
# carried over verbatim and MUST stay that way; index_html's
# `_DEFERRED_CDN_CASES` / `_DEFERRED_TAB_TASKS_CASES` turn them into
# `pytest.raises(match=...)` patterns.
# ---------------------------------------------------------------------------


class ScriptTagCollector(html.parser.HTMLParser):
    """Collects the attribute dicts for every <script> start-tag encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.script_attrs: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == 'script':
            self.script_attrs.append(dict(attrs))


def find_script_position(
    body: str, src_prefix: str
) -> tuple[int, dict[str, str | None]] | None:
    """Return ``(index, attrs)`` for the first <script> tag whose ``src``
    starts with ``src_prefix``, or ``None`` if no such tag exists.

    ``index`` is the tag's 0-based position in ``ScriptTagCollector.script_attrs``
    (document order, since the list preserves insertion order).  Returning attrs
    alongside the position avoids a second parse when the caller also needs the
    src or other attributes.
    """
    collector = ScriptTagCollector()
    collector.feed(body)
    for i, attrs in enumerate(collector.script_attrs):
        if (attrs.get('src') or '').startswith(src_prefix):
            return i, attrs
    return None


def assert_script_loads_before(
    body: str,
    before_src_prefix: str,
    after_src_prefix: str,
    before_label: str,
    after_label: str,
    consumer_note: str = '',
) -> None:
    """Assert that the script for ``before_src_prefix`` loads BEFORE the
    script for ``after_src_prefix`` in ``body``.  Combines a
    defer/async/type=module false-pass guard with the document-order
    position comparison.
    """
    before_result = find_script_position(body, before_src_prefix)
    assert before_result is not None, (
        f'No <script src="{before_src_prefix}..."> tag found in index.html. '
        f'{consumer_note}'
    )
    before_pos, before_attrs = before_result
    before_src = before_attrs.get('src')

    after_result = find_script_position(body, after_src_prefix)
    assert after_result is not None, (
        f'<script src="{after_src_prefix}..."> not found in index.html — '
        f'cannot verify load-order invariant for {before_label}.'
    )
    after_pos, after_attrs = after_result

    # Both tags must be classic synchronous scripts — otherwise document order
    # diverges from execution order and the position comparison below is moot.
    for _label, _attrs in [
        (before_label, before_attrs),
        (after_label, after_attrs),
    ]:
        assert 'defer' not in _attrs, (
            f'{_label} has a defer attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert 'async' not in _attrs, (
            f'{_label} has an async attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert (_attrs.get('type') or '').lower() != 'module', (
            f'{_label} has type="module"; ES modules are deferred by default, '
            f'so document order no longer implies execution order.'
        )

    assert before_pos < after_pos, (
        f'{before_label} (position {before_pos}, src={before_src!r}) must load '
        f'BEFORE {after_label} (position {after_pos}). '
        f'{consumer_note}'
    )
