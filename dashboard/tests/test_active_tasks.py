"""Tests for the ACTIVE_TASKS aggregator that joins task tree + worktrees."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot
from shared.task_statuses import ACTIVE, TaskStatus

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import (
    _build_task_row,
    _minutes_since,
    collect_active_tasks,
    collect_tasks_with_counts,
    shape_terminal_rows,
)
from dashboard.data.census import TaskCensus
from dashboard.data.datum import DatumState
from dashboard.data.task_snapshot import SnapshotHealth, TaskSnapshot, classify

# ---------------------------------------------------------------------------
# helpers used inside the aggregator
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_root_rotation_between_tests():
    """Start every test from rotation offset 0.

    REQUIRED, not tidy. ``active_tasks._root_rotation_offset`` is module state
    that ``collect_tasks_with_counts`` advances on every call (task 4884), so
    without this any test that asserts WHICH roots were served — or in what
    order they were admitted — silently depends on how many times an EARLIER
    test in the session happened to call the collector. That is a test that
    passes or fails by file order, which is worse than one that fails.
    """
    import dashboard.data.active_tasks as at_mod

    at_mod._reset_root_rotation()
    yield
    at_mod._reset_root_rotation()


@pytest.fixture(autouse=True)
def _clear_the_snapshot_unit_between_tests():
    """No test may be served the previous test's snapshot.

    REQUIRED for the same reason as the rotation reset above. The unit holds
    a 15 s TTL cache AND a never-expiring last-good store, both keyed by
    project-root STRING, and tests here reuse root names across ``tmp_path``
    directories. Without this a test's first acquisition can be answered from
    an earlier test's cache — issuing no MCP call at all — or, worse, served
    an earlier test's rows as ``stale``.
    """
    import dashboard.data.task_snapshot as snapshot_mod

    snapshot_mod._snapshot_cache_clear()
    yield
    snapshot_mod._snapshot_cache_clear()


def test_minutes_since_handles_z_suffix_and_naive_iso():
    one_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace('+00:00', 'Z')
    minutes = _minutes_since(one_hour_ago)
    assert minutes is not None  # a parseable start time is never the unknown-start None
    assert 59 <= minutes <= 61


def test_minutes_since_returns_none_on_missing_and_on_bad(caplog):
    """A MISSING start time and a present-but-unparseable one are both None.

    ``None``/``''`` is the per-task artifact-read-failure signal on
    ``TaskRuntimeEntry.started`` (see ``shared/src/shared/task_runtime_state.py``
    — "never a fabricated 0"), so the helper must propagate the unknown rather
    than render it as '0m running'. A present-but-unparseable timestamp is a
    different failure (upstream data damage, no known producer), but renders
    identically misleadingly as '0m running' if faked to 0, so it is also
    surfaced as None rather than fabricated (task 4365; task 4055 scoped its
    fix to the missing/empty case only and left this branch for follow-up).
    The unparseable case must also be LOUD (loud-over-silent-degradation): a
    WARNING naming the offending value, not just a silently-swapped return —
    mirroring ``test_queue.py::test_unparseable_timestamp_logs_a_warning``.
    """
    assert _minutes_since(None) is None
    assert _minutes_since('') is None

    with caplog.at_level(logging.WARNING):
        assert _minutes_since('not-a-date') is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any('not-a-date' in m for m in warnings), warnings


def test_minutes_since_uses_provided_now():
    """_minutes_since(iso, now=fixed) derives its result from the passed now, not the clock."""
    fixed = datetime(2026, 4, 11, 12, 30, 0, tzinfo=UTC)
    ts = fixed - timedelta(minutes=37, seconds=10)
    expected = int((fixed - ts).total_seconds() // 60)
    assert _minutes_since(ts.isoformat(), now=fixed) == expected


def test_minutes_since_no_now_resolves_via_clock():
    """Without now, _minutes_since still resolves via resolve_now (the live clock).

    Brackets the real clock read with before/after captures rather than
    patching a module-level ``datetime`` symbol, mirroring
    ``Test_Cutoff.test_cutoff_no_now_uses_current_time`` in test_costs_data.py.
    """
    ts = datetime.now(UTC) - timedelta(minutes=10)
    before = datetime.now(UTC)
    result = _minutes_since(ts.isoformat())
    after = datetime.now(UTC)

    lower = int((before - ts).total_seconds() // 60)
    upper = int((after - ts).total_seconds() // 60)
    assert result is not None  # a parseable start time is never the unknown-start None
    assert lower <= result <= upper


# ---------------------------------------------------------------------------
# collect_active_tasks against in-memory MCP-shaped fixture
# ---------------------------------------------------------------------------


def _shape_task(task: dict) -> dict:
    """Build a row in the dashboard's per-task wire shape.

    Mirrors tasks.py::_shape_task — must include ``updated_at`` so that
    done-task ordering tests work correctly.
    """
    return {
        'id': int(task['id']),
        'title': task.get('title') or '',
        'description': task.get('description') or '',
        'details': task.get('details') or '',
        'status': task.get('status'),
        'priority': task.get('priority'),
        'dependencies': list(task.get('dependencies') or []),
        'metadata': task.get('metadata', {}),
        'updated_at': task.get('updated_at'),
    }


def _make_project(root, *, project_dir, tasks):
    """Create a project root dir and return ``(project_root, shaped_tasks)``.

    The tasks themselves no longer live on disk — fused-memory MCP owns task
    state — so we return them in their dashboard-shaped form for the caller
    to register against ``fetch_tasks`` via monkeypatch. Per-task runtime
    state (loops/attempts/agent/lane/phase/lane_state) is likewise sourced
    over MCP now, not read from a ``.worktrees/.task`` artifact tree — see
    ``_runtime_entry``/``_register_runtime`` below.
    """
    project_root = root / project_dir
    project_root.mkdir(parents=True, exist_ok=True)
    return project_root, [_shape_task(t) for t in tasks]


def _runtime_entry(task_id: int, **overrides) -> TaskRuntimeEntry:
    """A ``TaskRuntimeEntry`` with sane defaults; ``overrides`` replace fields."""
    base: dict[str, Any] = dict(
        task_id=task_id,
        has_worktree=True,
        loops=0,
        attempts=0,
        started=None,
        lane=None,
        phase=None,
        lane_state=None,
        error=None,
    )
    base.update(overrides)
    return TaskRuntimeEntry(**base)


def _register_runtime(monkeypatch, mapping: dict[str, list[TaskRuntimeEntry]]) -> None:
    """Monkeypatch ``fetch_task_runtime`` to return a fixed ``{label: TaskRuntimeSnapshot}``.

    *mapping* maps project label -> list of ``TaskRuntimeEntry``. A label
    absent from *mapping* is simply absent from the returned dict (mirroring
    a real fan-out that only covers configured ``escalation_urls``) — tests
    that need an explicit "online but empty" snapshot for a label must
    include it with an empty list.
    """
    snapshots = {label: TaskRuntimeSnapshot(tasks=entries) for label, entries in mapping.items()}

    async def _fake_fetch_task_runtime(client, escalation_urls):
        return dict(snapshots)

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', _fake_fetch_task_runtime)


def _register_fetch_tasks(monkeypatch, fetch) -> None:
    """Register a full-tree *fetch* as the snapshot unit's two reads.

    ``active_tasks`` issues no read of its own since task 5587: it consumes a
    ``TaskSnapshot``, and the unit that acquires one is what reads.  So the
    fakes are registered against ``dashboard.data.task_snapshot``; a test that
    patched ``active_tasks.fetch_tasks`` would be patching a name that is no
    longer there.

    TWO fakes, because the unit reads through two functions with different
    answers: ``fetch_tasks`` returns the COMPLETE set matching its ``statuses``
    filter, ``fetch_statuses`` returns the compact ``{int id: status}`` map
    over the WHOLE tree.  Each emulates its own substrate contract — the
    ``statuses`` filter really is applied server-side — because a fake laxer
    than the real signature is how a call-site regression passes its tests.

    Tests here are about SHAPING; the wire contract itself is asserted against
    a canned ``mcp_tool_call`` in ``TestShapeOneProjectNarrowing``.

    *fetch* keeps its original ``(client, config, project_root)`` signature and
    may still return an offline marker dict, which is propagated unchanged.
    """

    # ``timeout``/``cached`` are accepted-and-ignored: the unit threads
    # task_snapshot.PER_CALL_TIMEOUT into both reads and asks the row read for
    # an UNCACHED answer, so a stub missing either keyword raises TypeError
    # instead of shaping rows.
    async def _narrowed(
        client, config, project_root, *,
        statuses=None, chunk_size=None, timeout=None, cached=True,
    ):
        rows = await fetch(client, config, project_root)
        if not isinstance(rows, list):
            return rows
        if statuses is not None:
            rows = [r for r in rows if r.get('status') in statuses]
        return sorted(rows, key=lambda r: r.get('id') or 0)  # ORDER BY id ASC

    async def _statuses(client, config, project_root, *, timeout=None):
        rows = await fetch(client, config, project_root)
        if not isinstance(rows, list):
            return {'offline': True, 'error': 'task fetch offline'}
        return {
            r['id']: r.get('status') for r in rows if isinstance(r.get('id'), int)
        }

    monkeypatch.setattr('dashboard.data.task_snapshot.fetch_tasks', _narrowed)
    monkeypatch.setattr('dashboard.data.task_snapshot.fetch_statuses', _statuses)


_STUB_AS_OF = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
"""The instant a canned snapshot claims it was measured at."""


def _stub_snapshot(*, done: int = 0, unreachable: bool = False, rows=()):
    """A ``TaskSnapshot`` as ``acquire_snapshot`` would have returned one.

    For the budget / concurrency / fairness harnesses, which replace the whole
    per-root share and so must hand back the unit it would have acquired.

    *done* is how many done tasks the census reports, built by tallying a real
    status map through ``build_census`` rather than by constructing a census
    field by field — a hand-built one could violate the census's own
    ``sum(counts) == total`` invariant and make a passing test meaningless.
    """
    from types import MappingProxyType

    from dashboard.data.census import build_census
    from dashboard.data.datum import Datum, DatumState
    from dashboard.data.task_snapshot import (
        FRESHNESS_BOUND_SECONDS,
        SnapshotFailure,
        TaskSnapshot,
    )

    if unreachable:
        unknown = Datum(
            None, None, DatumState.UNKNOWN, 'canned unreachable root',
            FRESHNESS_BOUND_SECONDS,
        )
        return TaskSnapshot(
            census=unknown, rows=unknown,
            in_progress_live=None, in_progress_stranded=None, skew_seconds=None,
            status_map=MappingProxyType({}), failure=SnapshotFailure.UNREACHABLE,
        )

    status_map = {task_id: 'done' for task_id in range(1, done + 1)}
    return TaskSnapshot(
        census=Datum(
            build_census(status_map), _STUB_AS_OF, DatumState.FRESH, None,
            FRESHNESS_BOUND_SECONDS,
        ),
        rows=Datum(
            list(rows), _STUB_AS_OF, DatumState.FRESH, None,
            FRESHNESS_BOUND_SECONDS,
        ),
        in_progress_live=0, in_progress_stranded=0, skew_seconds=0,
        status_map=MappingProxyType(status_map), failure=SnapshotFailure.NONE,
    )


def _labelled(snapshots, health) -> list[str]:
    """The labels whose entry classifies as *health*, in the collector's order.

    The three project lists ``/api/v2/dashboard/tasks`` emits are DERIVED from
    the entries rather than returned beside them, so reading them back here is
    what the wire does — through ``task_snapshot.classify``, the one place the
    ``(state, failure kind)`` routing is decided, so a test can never assert a
    partition the handler does not draw.
    """
    return [label for label, snap in snapshots.items() if classify(snap) is health]


def _offline(snapshots) -> list[str]:
    return _labelled(snapshots, SnapshotHealth.OFFLINE)


def _degraded(snapshots) -> list[str]:
    return _labelled(snapshots, SnapshotHealth.DEGRADED)


def _count_unknown(snapshots) -> list[str]:
    return _labelled(snapshots, SnapshotHealth.COUNT_UNKNOWN)


def _measured_census(snapshot: TaskSnapshot) -> TaskCensus:
    """The census of a snapshot that is asserted to have measured one.

    ``Datum.value`` is optional by design — an UNKNOWN or BUDGET snapshot
    carries none — so every read of it has to say which case it expects. This
    says "measured", and fails naming the state it actually found rather than
    raising ``AttributeError`` on ``None`` three frames later.
    """
    census = snapshot.census.value
    assert census is not None, (
        f'expected a measured census, got state {snapshot.census.state} '
        f'({snapshot.census.reason})'
    )
    return census


def _measured_rows(snapshot: TaskSnapshot) -> list[dict]:
    """The rows of a snapshot asserted to have measured them; see ``_measured_census``."""
    rows = snapshot.rows.value
    assert rows is not None, (
        f'expected measured rows, got state {snapshot.rows.state} '
        f'({snapshot.rows.reason})'
    )
    return rows


def _done_counts(snapshots) -> dict[str, int]:
    """``{label: done count}`` for every root whose census was MEASURED.

    A root with a non-fresh census is absent rather than zero — the same
    "no count was measured, so none is fabricated" rule the old
    ``done_counts`` return value carried, now read off the census Datum's own
    state instead of a parallel dict.
    """
    return {
        label: snap.census.value.counts['done']
        for label, snap in snapshots.items()
        if snap.census.state is DatumState.FRESH
    }


@pytest.fixture()
def two_project_config(tmp_path, monkeypatch):
    """Two-project layout with shaped task lists registered against fetch_tasks."""
    started = (datetime.now(UTC) - timedelta(minutes=14)).isoformat()
    df_root, df_tasks = _make_project(
        tmp_path,
        project_dir='dark-factory',
        tasks=[
            {'id': 19, 'title': 'consolidation retry', 'status': 'in-progress',
             'dependencies': [15, 17],
             'metadata': {'files': ['src/agents/consolidation.py', 'src/store/graphiti_adapter.py']}},
            {'id': 17, 'title': 'pre-filter', 'status': 'done', 'dependencies': []},
            {'id': 15, 'title': 'partitioning', 'status': 'done', 'dependencies': []},
            {'id': 23, 'title': 'collision', 'status': 'pending', 'dependencies': [21]},
            {'id': 21, 'title': 'dedup index', 'status': 'in-progress',
             'dependencies': []},
        ],
    )
    reify_root, reify_tasks = _make_project(
        tmp_path,
        project_dir='reify',
        tasks=[{'id': 8, 'title': 'parser recovery', 'status': 'blocked',
                'dependencies': []}],
    )

    by_root = {df_root.resolve(): df_tasks, reify_root.resolve(): reify_tasks}

    async def _fake_fetch_tasks(client, config, project_root):
        return list(by_root.get(project_root.resolve(), []))

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'dark-factory': [
            # 1/3 reviews passed -> attempts == 3 (total review count, not pass count)
            _runtime_entry(19, loops=2, attempts=3, started=started),
            _runtime_entry(21, loops=1, attempts=1, started=started),
        ],
        'reify': [
            _runtime_entry(8, loops=0, attempts=0, started=started),
        ],
    })

    return DashboardConfig(project_root=df_root, known_project_roots=[reify_root])


@pytest.mark.asyncio
async def test_collect_active_tasks_filters_to_active_statuses(two_project_config, dummy_client):
    active, _ = await collect_active_tasks(client=dummy_client, config=two_project_config)
    statuses = {t['status'] for t in active}
    assert statuses <= {'in-progress', 'blocked', 'pending'}
    # Done tasks (17, 15) should not appear.
    ids = {t['id'] for t in active}
    assert 'dark-factory/T-17' not in ids
    assert 'dark-factory/T-15' not in ids


@pytest.mark.asyncio
async def test_collect_active_tasks_resolves_deps_with_done_flags(two_project_config, dummy_client):
    active, _ = await collect_active_tasks(client=dummy_client, config=two_project_config)
    by_id = {t['id']: t for t in active}
    t19 = by_id['dark-factory/T-19']
    assert {d['id']: d['done'] for d in t19['deps']} == {
        'dark-factory/T-15': True,
        'dark-factory/T-17': True,
    }
    t23 = by_id['dark-factory/T-23']
    # T-21 is in-progress, not done
    assert t23['deps'] == [{'id': 'dark-factory/T-21', 'title': 'dedup index', 'done': False}]


@pytest.mark.asyncio
async def test_collect_active_tasks_pulls_metadata_and_loops(two_project_config, dummy_client):
    active, _ = await collect_active_tasks(client=dummy_client, config=two_project_config)
    t19 = next(t for t in active if t['id'] == 'dark-factory/T-19')
    assert t19['agent'] == 'claude-task-19'
    assert t19['loops'] == 2
    assert t19['attempts'] == 3
    # `started` is the minutes-since difference, allow a small slack vs 14.
    assert 13 <= t19['started'] <= 15
    # meta_files is the module lock source used by the scheduler pipeline.
    assert 'src/agents/consolidation.py' in t19['meta_files']


@pytest.mark.asyncio
async def test_collect_active_tasks_started_uses_provided_now(tmp_path, monkeypatch, dummy_client):
    """collect_active_tasks(now=fixed) computes every row's `started` against that one instant."""
    fixed = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
    created_at = (fixed - timedelta(minutes=42)).isoformat()
    root, shaped = _make_project(
        tmp_path,
        project_dir='fixedclock',
        tasks=[{'id': 1, 'title': 'a', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    _register_runtime(monkeypatch, {'fixedclock': [_runtime_entry(1, started=created_at)]})
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg, now=fixed)
    assert len(active) == 1
    assert active[0]['started'] == 42


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_started_uses_provided_now_across_projects(
    tmp_path, monkeypatch, dummy_client,
):
    """collect_tasks_with_counts(now=fixed) shares ONE now across every project's rows.

    Two projects, each with a task started at a different known offset from the
    same fixed instant — proves `now` is resolved once at the aggregation
    boundary and threaded down, not re-read per project.
    """
    fixed = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
    df_created = (fixed - timedelta(minutes=10)).isoformat()
    reify_created = (fixed - timedelta(minutes=25)).isoformat()
    df_root, df_tasks = _make_project(
        tmp_path,
        project_dir='df',
        tasks=[{'id': 1, 'title': 'a', 'status': 'in-progress', 'dependencies': []}],
    )
    reify_root, reify_tasks = _make_project(
        tmp_path,
        project_dir='reify',
        tasks=[{'id': 2, 'title': 'b', 'status': 'pending', 'dependencies': []}],
    )
    by_root = {df_root.resolve(): df_tasks, reify_root.resolve(): reify_tasks}

    async def _fake_fetch_tasks(client, config, project_root):
        return list(by_root.get(project_root.resolve(), []))

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'df': [_runtime_entry(1, started=df_created)],
        'reify': [_runtime_entry(2, started=reify_created)],
    })
    cfg = DashboardConfig(project_root=df_root, known_project_roots=[reify_root])

    active, _snapshots = await collect_tasks_with_counts(client=dummy_client, config=cfg, now=fixed)
    started_by_id = {t['id']: t['started'] for t in active}
    assert started_by_id == {'df/T-1': 10, 'reify/T-2': 25}


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_stamps_every_snapshot_at_the_provided_now(
    tmp_path, monkeypatch, dummy_client,
):
    """Every returned unit's two halves are stamped at exactly the caller's *now*.

    The collector-side half of the ``served_at`` contract. ``api_tasks``
    validates every datum against its own ``served_at``, and ``validate_datum``
    refuses a negative age. So the handler's instant has to BE the instant the
    producer stamps. This pins the threading here as well as through the
    endpoint (``test_app.py``), because a fix confined to one call site would
    leave the other free to regress.

    Driven through the real ``acquire_snapshot`` over a canned
    ``mcp_tool_call``, so the stamp is the one the unit really applies.
    ``test_app.py::_snapshot`` cannot catch this: it stamps ``as_of`` at the
    live clock BEFORE the request, which inverts production's ordering. That
    inversion is how 350 green dashboard tests sat over a handler that routed
    every project to ``unknown`` on every cache-miss render.
    ``dashboard/src/dashboard/data/scheduler.py::collect_scheduler_state`` and
    ``collect_active_tasks`` already forwarded ``now``; ``api_tasks`` was the
    only call site that did not.
    """
    fixed = datetime(2026, 4, 11, 12, 0, 0, tzinfo=UTC)
    df_root = tmp_path / 'df'
    reify_root = tmp_path / 'reify'
    for root in (df_root, reify_root):
        root.mkdir()
    pairs = ((1, 'in-progress'), (2, 'pending'), (3, 'done'))
    mcp, _calls = _canned_mcp(
        [_raw_row(task_id, status) for task_id, status in pairs], dict(pairs),
    )
    monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)
    cfg = DashboardConfig(project_root=df_root, known_project_roots=[reify_root])

    _active, snapshots = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, now=fixed,
    )

    assert set(snapshots) == {'df', 'reify'}
    for label, snapshot in snapshots.items():
        assert snapshot.census.state is DatumState.FRESH, (label, snapshot.census)
        assert snapshot.rows.state is DatumState.FRESH, (label, snapshot.rows)
        assert snapshot.census.as_of == fixed, (label, snapshot.census.as_of)
        assert snapshot.rows.as_of == fixed, (label, snapshot.rows.as_of)


@pytest.mark.asyncio
async def test_collect_active_tasks_handles_missing_worktree_metadata(tmp_path, monkeypatch, dummy_client):
    """A pending task absent from an ONLINE runtime map still appears, with honest zeros."""
    root, shaped = _make_project(
        tmp_path, project_dir='solo',
        tasks=[{'id': 1, 'title': 'lonely', 'status': 'pending', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    # Project is online (a snapshot is registered for its label) but the
    # snapshot carries no entry for task 1 — the honest-zero case, distinct
    # from an offline project (see the runtime-offline tests below).
    _register_runtime(monkeypatch, {'solo': []})
    cfg = DashboardConfig(project_root=root)
    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert active == [{
        'id': 'solo/T-1', 'project': 'solo', 'title': 'lonely',
        'status': 'pending', 'agent': None,
        'started': 0, 'loops': 0, 'attempts': 0, 'deps': [],
        'meta_files': [], 'train': None, 'external_deps': [], 'prd': None,
        'lane': None, 'phase': None, 'lane_state': None, 'runtime_offline': False,
        'runtime_status': 'ok',
        # Claim projection (task 3543): carried on every row. A 'pending' task
        # is never stranded — the shared predicate gates on 'in-progress'.
        'claimant_run_id': None, 'heartbeat_at': None, 'stranded': False,
    }]


@pytest.mark.asyncio
async def test_no_task_row_carries_description_or_details(tmp_path, monkeypatch, dummy_client):
    """ACCEPTANCE (task 5815): no row either builder shapes ships a task's prose.

    The Task Detail pane fetches description/details for the ONE selected task
    (dashboard/src/dashboard/api/task_prose.py::api_task_prose); shipping them
    on every row was ~84% of the /tasks payload. Every raw row here carries
    NON-EMPTY prose, so a missing key is a real omission, not an empty field.

    Since task 5587 there are TWO row builders: the collector's active rows,
    which are the same objects ``TASKS_SNAPSHOT[p].rows`` carries, and
    ``shape_terminal_rows`` for the on-demand ``?terminal=`` window.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='prose',
        tasks=[
            {'id': task_id, 'title': status, 'status': status, 'dependencies': [],
             'description': f'why {task_id}', 'details': f'how {task_id}',
             'updated_at': '2026-09-24T00:00:00+00:00'}
            for task_id, status in enumerate(
                ['in-progress', 'pending', 'blocked', 'done', 'cancelled'], start=1,
            )
        ],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {'prose': []})
    cfg = DashboardConfig(project_root=root)

    active, snapshots = await collect_tasks_with_counts(client=dummy_client, config=cfg)
    terminal = shape_terminal_rows(
        root,
        [task for task in shaped if task['status'] in ('done', 'cancelled')],
        now=datetime.now(UTC),
    )

    # NON-VACUITY: both builders must have shaped rows.
    assert {row['status'] for row in active} == {'in-progress', 'pending', 'blocked'}
    assert {row['status'] for row in terminal} == {'done', 'cancelled'}
    rows = [*active, *(snapshots['prose'].rows.value or []), *terminal]
    carrying = [row['id'] for row in rows if 'description' in row or 'details' in row]
    assert carrying == [], f'these task rows still ship prose: {carrying}'


@pytest.mark.asyncio
async def test_collect_active_tasks_surfaces_offline_projects(tmp_path, monkeypatch, dummy_client):
    """A project whose MCP fetch returns an offline marker is reported."""
    root = tmp_path / 'offline-project'
    root.mkdir()

    async def _fake_fetch_tasks(client, config, project_root):
        return {'offline': True, 'error': 'connection refused'}

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    cfg = DashboardConfig(project_root=root)
    active, offline_projects = await collect_active_tasks(client=dummy_client, config=cfg)
    assert active == []
    assert offline_projects == ['offline-project']


# ---------------------------------------------------------------------------
# get_task_runtime_state MCP join (task 2636 step-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_active_tasks_runtime_join_populates_lane_phase_lane_state(
    tmp_path, monkeypatch, dummy_client,
):
    """An online runtime entry's loops/attempts/lane/phase/lane_state/agent all
    join onto the row; started is computed via _minutes_since against `now`.
    """
    fixed = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
    entry_started = (fixed - timedelta(minutes=7)).isoformat()
    root, shaped = _make_project(
        tmp_path, project_dir='warmlane',
        tasks=[{'id': 42, 'title': 'warm task', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'warmlane': [_runtime_entry(
            42, loops=3, attempts=1, started=entry_started,
            lane='_lane-7', phase='EXECUTE', lane_state='assigned',
        )],
    })
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg, now=fixed)
    assert len(active) == 1
    row = active[0]
    assert row['loops'] == 3
    assert row['attempts'] == 1
    assert row['agent'] == 'claude-task-42'
    assert row['started'] == 7
    assert row['lane'] == '_lane-7'
    assert row['phase'] == 'EXECUTE'
    assert row['lane_state'] == 'assigned'
    assert row['runtime_offline'] is False


@pytest.mark.asyncio
async def test_collect_active_tasks_runtime_unparseable_started_yields_none_row(
    tmp_path, monkeypatch, dummy_client,
):
    """An ONLINE entry whose ``started`` is present-but-unparseable renders as
    ``None`` on the row, not a fabricated ``0`` — the row-level counterpart to
    ``test_minutes_since_returns_none_on_missing_and_on_bad`` (task 4365),
    mirroring ``test_task_runtime_boundary.py``'s
    ``test_b6_online_per_task_read_failure_yields_none_started`` shape for the
    unparseable-rather-than-missing case. ``runtime_offline`` stays False: this
    is a damaged field on an otherwise-online snapshot, not an outage.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='damagedlane',
        tasks=[{'id': 9, 'title': 'damaged task', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'damagedlane': [_runtime_entry(9, started='not-a-date')],
    })
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    assert row['started'] is None
    assert row['runtime_offline'] is False


@pytest.mark.asyncio
async def test_collect_active_tasks_runtime_offline_snapshot_yields_all_none(
    tmp_path, monkeypatch, dummy_client,
):
    """A project whose runtime snapshot reports offline=True gets honest None
    fields (never a fabricated 0), with runtime_offline=True.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='downlane',
        tasks=[{'id': 5, 'title': 'stuck task', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    async def _offline_fetch_task_runtime(client, escalation_urls):
        return {'downlane': TaskRuntimeSnapshot(offline=True)}

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_task_runtime', _offline_fetch_task_runtime)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    for key in ('agent', 'loops', 'attempts', 'started', 'lane', 'phase', 'lane_state'):
        assert row[key] is None, f'expected {key}=None when runtime offline, got {row[key]!r}'
    assert row['runtime_offline'] is True
    # offline=True with NO reason is out-of-contract for a dashboard-synthesized
    # snapshot. Report it as an honest 'unknown' — never guess 'unreachable',
    # which is precisely the fabricated diagnosis task 3517 exists to prevent.
    assert row['runtime_status'] == 'unknown'


@pytest.mark.asyncio
async def test_collect_active_tasks_no_escalation_url_treated_as_offline(
    tmp_path, monkeypatch, dummy_client,
):
    """A project absent from the runtime map entirely (no escalation URL
    configured for it) is treated identically to an explicit offline=True
    snapshot — we genuinely have no runtime source for it either way.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='nourl',
        tasks=[{'id': 6, 'title': 'no url task', 'status': 'pending', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {})  # no label registered for 'nourl' at all
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    for key in ('agent', 'loops', 'attempts', 'started', 'lane', 'phase', 'lane_state'):
        assert row[key] is None, f'expected {key}=None when no escalation URL, got {row[key]!r}'
    assert row['runtime_offline'] is True
    # ...but the CAUSE is separable now: nothing was ever probed here, so this
    # is the expected/permanent case, not an orchestrator fault.
    assert row['runtime_status'] == 'not_configured'


@pytest.mark.asyncio
async def test_collect_active_tasks_runtime_per_task_read_failure_stays_online(
    tmp_path, monkeypatch, dummy_client,
):
    """A per-task artifact read failure (loops/attempts/started/phase=None,
    error set) is honest but distinct from project-offline: runtime_offline
    stays False.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='flaky',
        tasks=[{'id': 9, 'title': 'flaky task', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    _register_runtime(monkeypatch, {
        'flaky': [_runtime_entry(
            9, loops=None, attempts=None, started=None, phase=None,
            lane_state=None, error='wire-contract violation: bad enum value',
        )],
    })
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    assert row['loops'] is None
    assert row['attempts'] is None
    assert row['phase'] is None
    assert row['started'] is None, (
        "a per-task read failure must not fabricate '0m running'"
    )
    assert row['runtime_offline'] is False, (
        'a per-task read failure is an honest error, not an offline project'
    )
    assert row['runtime_status'] == 'ok', (
        'the PROBE succeeded — the failure is per-task, not a probe fault domain'
    )


# ---------------------------------------------------------------------------
# runtime_status probe discriminator (task 3517)
# ---------------------------------------------------------------------------


async def _one_task_row_with_snapshot(
    tmp_path, monkeypatch, dummy_client, snapshot: TaskRuntimeSnapshot | None,
) -> dict:
    """Collect a single-task project whose runtime map holds *snapshot*.

    ``None`` means the label is absent from the map entirely (no escalation
    URL configured for it) — the never-probed case.
    """
    root, shaped = _make_project(
        tmp_path, project_dir='probe',
        tasks=[{'id': 3, 'title': 'probed task', 'status': 'in-progress', 'dependencies': []}],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    async def _fake_fetch_task_runtime(client, escalation_urls):
        return {} if snapshot is None else {'probe': snapshot}

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    monkeypatch.setattr(
        'dashboard.data.active_tasks.fetch_task_runtime', _fake_fetch_task_runtime,
    )
    active, _ = await collect_active_tasks(
        client=dummy_client, config=DashboardConfig(project_root=root),
    )
    assert len(active) == 1
    return active[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['deadline_exceeded', 'unreachable'])
async def test_collect_active_tasks_row_carries_probe_reason(
    tmp_path, monkeypatch, dummy_client, reason,
):
    """The probe's fault-domain discriminator reaches the task row intact.

    Without this an operator sees identical blank cells whether the
    orchestrator is down or the dashboard was too starved to ask — the
    2026-07-30 misdiagnosis.
    """
    row = await _one_task_row_with_snapshot(
        tmp_path, monkeypatch, dummy_client,
        TaskRuntimeSnapshot(offline=True, offline_reason=reason),
    )
    assert row['runtime_status'] == reason
    # Back-compat: runtime_offline keeps its exact prior meaning, and degraded
    # rows still carry honest Nones rather than fabricated zeros.
    assert row['runtime_offline'] is True
    for key in ('agent', 'loops', 'attempts', 'started', 'lane', 'phase', 'lane_state'):
        assert row[key] is None, f'expected {key}=None when probe failed, got {row[key]!r}'


@pytest.mark.asyncio
async def test_collect_active_tasks_online_snapshot_with_entry_is_ok(
    tmp_path, monkeypatch, dummy_client,
):
    row = await _one_task_row_with_snapshot(
        tmp_path, monkeypatch, dummy_client,
        TaskRuntimeSnapshot(tasks=[_runtime_entry(3, loops=4)]),
    )
    assert row['runtime_status'] == 'ok'
    assert row['runtime_offline'] is False
    assert row['loops'] == 4


@pytest.mark.asyncio
async def test_collect_active_tasks_includes_merge_deferred_and_train_field(
    tmp_path, monkeypatch, dummy_client
):
    """merge-deferred tasks survive the active filter and carry the `train` field.

    Task 101 has metadata.train set; the output dict must have train={'id', 'order'}
    (members[] is intentionally omitted from the projected wire shape).
    Task 102 has no train metadata; the output dict must have train=None.
    """
    root, shaped = _make_project(
        tmp_path,
        project_dir='trainyard',
        tasks=[
            {
                'id': 101,
                'title': 'train task with metadata',
                'status': 'merge-deferred',
                'dependencies': [],
                'metadata': {
                    'train': {'id': 'demo', 'order': 0, 'members': ['T-101', 'T-102']},
                    'files': [],
                },
            },
            {
                'id': 102,
                'title': 'merge-deferred without train metadata',
                'status': 'merge-deferred',
                'dependencies': [],
                'metadata': {'files': []},
            },
        ],
    )

    async def _fake_fetch_tasks(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch_tasks)
    cfg = DashboardConfig(project_root=root)
    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    ids = {t['id'] for t in active}
    assert 'trainyard/T-101' in ids, (
        "merge-deferred task 101 was dropped by the active-status filter — "
        "add 'merge-deferred' to _ACTIVE_STATUSES"
    )
    assert 'trainyard/T-102' in ids, (
        "merge-deferred task 102 was dropped by the active-status filter — "
        "add 'merge-deferred' to _ACTIVE_STATUSES"
    )

    by_id = {t['id']: t for t in active}
    assert by_id['trainyard/T-101']['train'] == {'id': 'demo', 'order': 0}, (
        "task 101 with train metadata should have train={'id': 'demo', 'order': 0} "
        "(members[] is intentionally omitted from the projected wire shape)"
    )
    assert by_id['trainyard/T-102']['train'] is None, (
        "task 102 without train metadata should have train=None"
    )


# ---------------------------------------------------------------------------
# Terminal rows never reach the default render (task 5587 step-7)
#
# The per-bucket caps, their ordering rules and the live-PRD terminal-member
# exemption retired with the default-render terminal fetch: the collector no
# longer asks for a terminal row at all, and the bounded window moved behind
# ``?terminal=<project>`` as its own ``lower_bound`` Datum (PRD decision 5).
# What remains here is the fact that survived them — the collector emits
# ACTIVE rows and nothing else.
# ---------------------------------------------------------------------------


def _make_done_project(root, *, project_dir, active_tasks, done_tasks):
    """Layout a project with active + done tasks carrying updated_at.

    ``active_tasks`` and ``done_tasks`` are raw dicts; done tasks MUST include
    ``updated_at`` so the ordering / completed-field assertions work.
    """
    project_root = root / project_dir
    project_root.mkdir(parents=True, exist_ok=True)
    shaped = [_shape_task(t) for t in (active_tasks + done_tasks)]
    return project_root, shaped


@pytest.mark.asyncio
async def test_collect_active_tasks_excludes_done_rows(tmp_path, monkeypatch, dummy_client):
    """The collector must return NO done row — for the scheduler or anyone else.

    This used to be the DEFAULT behaviour, switchable on by a cap. There is no
    switch any more: the row read is narrowed to ``shared.task_statuses.ACTIVE``
    server-side, so a done row cannot reach this list however it is called.
    """
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[
            {'id': 1, 'title': 'active', 'status': 'pending', 'dependencies': []},
        ],
        done_tasks=[
            {'id': 2, 'title': 'done', 'status': 'done', 'dependencies': [],
             'updated_at': '2026-05-29T10:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    # Call with NO max_done_per_project — default behaviour
    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    done_rows = [t for t in active if t['status'] == 'done']
    assert done_rows == [], (
        'Default collect_active_tasks must NOT return done rows — '
        'scheduler.py must not receive done tasks'
    )


# ---------------------------------------------------------------------------
# external_deps field on task rows (step-1 / step-2)
# ---------------------------------------------------------------------------


def test_build_task_row_external_deps_from_metadata():
    """_build_task_row carries external_deps as [{'id','status':'unknown'}] per entry."""
    task = {
        'id': 42,
        'title': 'cross-project waiter',
        'description': '',
        'details': '',
        'status': 'pending',
        'metadata': {'external_deps': ['dark_factory:13', 'reify:8']},
    }
    row = _build_task_row('myproject', task, 42, {}, 'myproject/T-42')
    assert row['external_deps'] == [
        {'id': 'dark_factory:13', 'status': 'unknown'},
        {'id': 'reify:8', 'status': 'unknown'},
    ]


def test_build_task_row_external_deps_empty_when_absent():
    """_build_task_row yields external_deps=[] when metadata.external_deps is absent."""
    task = {'id': 1, 'title': 'no ext', 'status': 'pending', 'metadata': {}}
    row = _build_task_row('myproject', task, 1, {}, 'myproject/T-1')
    assert row['external_deps'] == []


def test_build_task_row_external_deps_empty_when_non_list():
    """_build_task_row yields external_deps=[] when metadata.external_deps is not a list."""
    for bad_value in [None, 'foo:1', 123, {'a': 'b'}]:
        task = {'id': 1, 'title': 'bad', 'status': 'pending',
                'metadata': {'external_deps': bad_value}}
        row = _build_task_row('p', task, 1, {}, 'p/T-1')
        assert row['external_deps'] == [], (
            f'expected [] for external_deps={bad_value!r}'
        )


def test_build_task_row_external_deps_ignores_non_str_and_empty():
    """_build_task_row ignores empty strings and non-str items in external_deps."""
    task = {'id': 1, 'title': 'x', 'status': 'pending',
            'metadata': {'external_deps': ['', 'dark_factory:13', '', None, 42]}}
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['external_deps'] == [{'id': 'dark_factory:13', 'status': 'unknown'}]


# ---------------------------------------------------------------------------
# prd field coalescing on task rows (step-1 / step-2)
# ---------------------------------------------------------------------------


def test_build_task_row_prd_field_from_prd_path():
    """_build_task_row coalesces metadata.prd_path into row['prd'] verbatim."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 'plans/dashboard-taskgraph-legibility-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/dashboard-taskgraph-legibility-prd.md'


def test_build_task_row_prd_field_strips_anchor_suffix():
    """A trailing '#anchor' fragment on prd_path is stripped."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 'plans/foo-prd.md#implementation-notes'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/foo-prd.md'


def test_build_task_row_prd_field_strips_section_suffix():
    """A trailing '§section' fragment on prd_path is stripped."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 'plans/foo-prd.md§Contract'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/foo-prd.md'


def test_build_task_row_prd_field_trims_whitespace():
    """Surrounding whitespace on prd_path is trimmed."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': '  plans/foo-prd.md  '},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/foo-prd.md'


def test_build_task_row_prd_field_legacy_prd_key():
    """Legacy 'prd' key is coalesced when prd_path is absent."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd': 'docs/legacy-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'docs/legacy-prd.md'


def test_build_task_row_prd_field_legacy_prd_ref_key():
    """Legacy 'prd_ref' key is coalesced when prd_path and prd are absent."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_ref': 'docs/legacy-ref-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'docs/legacy-ref-prd.md'


def test_build_task_row_prd_field_precedence_prd_path_over_prd_and_ref():
    """When multiple provenance keys are present, prd_path wins over prd and prd_ref."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {
            'prd_path': 'plans/winner-prd.md',
            'prd': 'plans/loser-prd.md',
            'prd_ref': 'plans/loser-ref-prd.md',
        },
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/winner-prd.md'


def test_build_task_row_prd_field_empty_prd_path_falls_through_to_prd():
    """An empty-string prd_path is skipped in favor of a non-empty prd."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': '', 'prd': 'plans/fallback-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/fallback-prd.md'


def test_build_task_row_prd_field_suffix_only_prd_path_falls_through():
    """A prd_path that is ONLY a suffix (cleans to '') falls through to the next key."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': '#just-an-anchor', 'prd': 'plans/fallback-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] == 'plans/fallback-prd.md'


def test_build_task_row_prd_field_none_when_no_provenance_keys():
    """row['prd'] is None when no prd_path/prd/prd_ref keys are present."""
    task = {'id': 1, 'title': 'x', 'status': 'pending', 'metadata': {}}
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] is None


def test_build_task_row_prd_field_non_string_values_skipped():
    """Non-string prd_path values (int, None) are skipped, yielding None."""
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 123},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1')
    assert row['prd'] is None

    task_none = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': None},
    }
    row_none = _build_task_row('p', task_none, 1, {}, 'p/T-1')
    assert row_none['prd'] is None


def test_build_task_row_prd_kwarg_overrides_metadata_coalescing():
    """An explicit prd= kwarg wins verbatim over metadata coalescing.

    Callers that already ran _coalesce_prd (e.g. the terminal-bucket loop, to
    decide live-PRD membership) pass the result through instead of paying for
    the split/strip work a second time; this proves the passthrough is used
    rather than silently re-deriving from metadata.
    """
    task = {
        'id': 1, 'title': 'x', 'status': 'pending',
        'metadata': {'prd_path': 'plans/metadata-derived-prd.md'},
    }
    row = _build_task_row('p', task, 1, {}, 'p/T-1', prd='plans/explicit-prd.md')
    assert row['prd'] == 'plans/explicit-prd.md', (
        'an explicit prd kwarg must win over metadata coalescing'
    )


@pytest.mark.asyncio
async def test_collect_active_tasks_includes_external_deps_with_unknown_sentinel(
    tmp_path, monkeypatch, dummy_client,
):
    """A task carrying metadata.external_deps surfaces external_deps with 'unknown' sentinels."""
    root, shaped = _make_project(
        tmp_path,
        project_dir='xdeps',
        tasks=[
            {
                'id': 5,
                'title': 'waits on upstream',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['dark_factory:13', 'reify:8']},
            },
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    row = active[0]
    assert row['external_deps'] == [
        {'id': 'dark_factory:13', 'status': 'unknown'},
        {'id': 'reify:8', 'status': 'unknown'},
    ]


@pytest.mark.asyncio
async def test_collect_active_tasks_external_deps_empty_when_absent(
    tmp_path, monkeypatch, dummy_client,
):
    """Tasks without external_deps carry external_deps=[] (no KeyError)."""
    root, shaped = _make_project(
        tmp_path,
        project_dir='nodeps',
        tasks=[
            {'id': 1, 'title': 'plain task', 'status': 'pending',
             'dependencies': [], 'metadata': {}},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)
    assert len(active) == 1
    assert active[0]['external_deps'] == []


# ---------------------------------------------------------------------------
# collect_tasks_with_counts resolve_external (step-5 / step-6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_overwrites_status(
    tmp_path, monkeypatch, dummy_client,
):
    """With resolve_external=True, statuses from fetch_external_statuses overwrite 'unknown'.

    'dark_factory:13' resolves to 'done'; 'reify:8' is absent from the map
    and keeps the honest 'unknown' sentinel (no fabricated status).
    """
    root, shaped = _make_project(
        tmp_path,
        project_dir='xdeps',
        tasks=[
            {
                'id': 5,
                'title': 'waits on upstream',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['dark_factory:13', 'reify:8']},
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    async def _fake_ext_statuses(client, config, deps):
        # Returns only 'dark_factory:13'; 'reify:8' is absent (simulates partial map).
        return {'dark_factory:13': 'done'}

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _fake_ext_statuses)

    cfg = DashboardConfig(project_root=root)
    active, _snapshots = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, resolve_external=True,
    )
    assert len(active) == 1
    row = active[0]
    assert row['external_deps'] == [
        {'id': 'dark_factory:13', 'status': 'done'},      # resolved
        {'id': 'reify:8', 'status': 'unknown'},            # absent from map → stays 'unknown'
    ]


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_false_skips_mcp(
    tmp_path, monkeypatch, dummy_client,
):
    """With resolve_external=False (default), fetch_external_statuses is NOT called."""
    root, shaped = _make_project(
        tmp_path,
        project_dir='xdeps',
        tasks=[
            {
                'id': 5,
                'title': 'waits on upstream',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['dark_factory:13']},
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    async def _must_not_be_called(*args, **kwargs):
        raise AssertionError('fetch_external_statuses must NOT be called when resolve_external=False')

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _must_not_be_called)

    cfg = DashboardConfig(project_root=root)
    # Default resolve_external=False — must NOT call fetch_external_statuses.
    active, _snapshots = await collect_tasks_with_counts(client=dummy_client, config=cfg)
    # Rows keep 'unknown' sentinel (unresolved).
    assert active[0]['external_deps'] == [{'id': 'dark_factory:13', 'status': 'unknown'}]


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_single_batched_call(
    tmp_path, monkeypatch, dummy_client,
):
    """resolve_external=True issues exactly ONE batched fetch_external_statuses call
    covering the deduped union of ALL rows' external dep ids.
    """
    root, shaped = _make_project(
        tmp_path,
        project_dir='multi',
        tasks=[
            {
                'id': 1, 'title': 'A', 'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['proj:10', 'proj:20']},
            },
            {
                'id': 2, 'title': 'B', 'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['proj:20', 'proj:30']},  # 'proj:20' deduped
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    calls = []

    async def _record_call(client, config, deps):
        calls.append(sorted(deps))
        return {}

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _record_call)

    cfg = DashboardConfig(project_root=root)
    await collect_tasks_with_counts(client=dummy_client, config=cfg, resolve_external=True)

    assert len(calls) == 1, f'expected 1 batched call, got {len(calls)}: {calls}'
    # Deduped union: proj:10, proj:20, proj:30
    assert calls[0] == ['proj:10', 'proj:20', 'proj:30']


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_skips_call_when_no_deps(
    tmp_path, monkeypatch, dummy_client,
):
    """resolve_external=True skips the fetch call when no rows have external deps."""
    root, shaped = _make_project(
        tmp_path,
        project_dir='nodeps',
        tasks=[
            {'id': 1, 'title': 'plain', 'status': 'pending',
             'dependencies': [], 'metadata': {}},
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    async def _must_not_be_called(*args, **kwargs):
        raise AssertionError('fetch_external_statuses must NOT be called when union is empty')

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _must_not_be_called)

    cfg = DashboardConfig(project_root=root)
    active, _snapshots = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, resolve_external=True,
    )
    assert active[0]['external_deps'] == []


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_skips_completed_rows(
    tmp_path, monkeypatch, dummy_client,
):
    """resolve_external=True must NOT include a COMPLETED row's external dep ids.

    A completed task's external deps are no longer actionable: their ids must
    not bloat the batched MCP request, and their entries must keep the honest
    ``'unknown'`` sentinel rather than being re-stamped.

    The guard keys on the ``completed`` field, which is what a terminal row
    carries — so the row is injected at the per-root seam rather than fetched.
    The collector's own reads are narrowed to ``shared.task_statuses.ACTIVE``
    since task 5587 and can no longer produce one, which is exactly why this
    test must not try to make them.
    """
    root = tmp_path / 'xdeps'
    root.mkdir(parents=True, exist_ok=True)

    def _row(uid, *, dep, completed=None):
        row = {
            'id': f'xdeps/T-{uid}', 'project': 'xdeps', 'status': 'pending',
            'external_deps': [{'id': dep, 'status': 'unknown'}],
        }
        if completed is not None:
            row['completed'] = completed
        return row

    async def _stub(client, config, project_root, *, now=None, runtime=None):
        rows = [
            _row(5, dep='proj:10'),
            _row(6, dep='proj:99', completed='2026-05-29T10:00:00+00:00'),
        ]
        return rows, _stub_snapshot()

    monkeypatch.setattr('dashboard.data.active_tasks._acquire_and_shape', _stub)

    calls: list[list[str]] = []

    async def _record_call(client, config, deps):
        calls.append(sorted(deps))
        return {'proj:10': 'done'}

    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _record_call)
    _register_runtime(monkeypatch, {})

    cfg = DashboardConfig(project_root=root)
    active, _snapshots = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, resolve_external=True,
    )

    assert calls == [['proj:10']], (
        f"a completed row's deps must not appear in the batched call; got {calls}"
    )

    by_id = {r['id']: r for r in active}
    assert by_id['xdeps/T-5']['external_deps'] == [{'id': 'proj:10', 'status': 'done'}]
    assert by_id['xdeps/T-6']['external_deps'] == [{'id': 'proj:99', 'status': 'unknown'}]


# ---------------------------------------------------------------------------
# ...and the same for cancelled rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_active_tasks_excludes_cancelled(
    tmp_path, monkeypatch, dummy_client,
):
    """The collector must return ZERO cancelled rows, for the same reason."""
    root, shaped = _make_done_project(
        tmp_path,
        project_dir='df',
        active_tasks=[
            {'id': 1, 'title': 'active', 'status': 'pending', 'dependencies': []},
        ],
        done_tasks=[
            {'id': 60, 'title': 'cancelled', 'status': 'cancelled', 'dependencies': [],
             'updated_at': '2026-05-29T10:00:00+00:00'},
        ],
    )

    async def _fake(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake)
    from dashboard.config import DashboardConfig
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    cancelled_rows = [t for t in active if t['status'] == 'cancelled']
    assert cancelled_rows == [], (
        'collect_active_tasks must NOT return cancelled rows'
    )


# ---------------------------------------------------------------------------
# deferred via active path (step-1 / step-2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_active_tasks_includes_deferred_via_active_path(
    tmp_path, monkeypatch, dummy_client,
):
    """deferred tasks must flow through the ACTIVE path (with resolved deps + started).

    A deferred task (id 30, deps=[31]) plus a done dep (id 31).
    Expected:
    (a) 'proj/T-30' appears in the returned rows — deferred survives the active filter.
    (b) The deferred row's deps == [{'id': 'proj/T-31', 'title': ..., 'done': True}] —
        proves it flows through the active path with resolved deps, NOT the stripped
        bounded-bucket path.
    (c) The row has a 'started' key and does NOT have a 'completed' key.

    RED today: 'deferred' is not in _ACTIVE_STATUSES, so the task is dropped.
    """
    root, shaped = _make_project(
        tmp_path,
        project_dir='proj',
        tasks=[
            {
                'id': 30,
                'title': 'parked work',
                'status': 'deferred',
                'dependencies': [31],
                'metadata': {},
            },
            {
                'id': 31,
                'title': 'finished dep',
                'status': 'done',
                'dependencies': [],
                'metadata': {},
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    cfg = DashboardConfig(project_root=root)

    active, _ = await collect_active_tasks(client=dummy_client, config=cfg)

    ids = {t['id'] for t in active}
    assert 'proj/T-30' in ids, (
        "deferred task T-30 was dropped by the active-status filter — "
        "add 'deferred' to _ACTIVE_STATUSES"
    )

    by_id = {t['id']: t for t in active}
    row = by_id['proj/T-30']

    # (b) resolved deps via active path — done flag on the dep.
    #
    # Task 3857: this is the scheduler path (both terminal caps 0), which by
    # design now fetches NO terminal rows at all, so the done dep's full row
    # is not available and its title degrades to ''. The done flag — the
    # load-bearing half of the chip — still resolves, via the compact status
    # map. Resolving the title would cost an extra whole-tree read, which is
    # the unbounded fetch this design removed.
    assert row['deps'] == [{'id': 'proj/T-31', 'title': '', 'done': True}], (
        f"expected deferred row deps with done=True, got: {row.get('deps')}"
    )

    # (c) active-path fields present / absent
    assert 'started' in row, "deferred row must have 'started' key (active path)"
    assert 'completed' not in row, (
        "deferred row must NOT have 'completed' key (that is the bounded-bucket sentinel)"
    )


# ---------------------------------------------------------------------------
# resolve_external + offline marker (step-11 / step-12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_tasks_with_counts_resolve_external_offline_marker(
    tmp_path, monkeypatch, dummy_client,
):
    """When fetch_external_statuses returns the offline marker, each dep entry gets
    status=='offline' (distinct from 'unknown'=task-not-found).

    Fails today because status_map.get(id, 'unknown') yields 'unknown' for all deps
    when the map is the offline marker {'offline':True,'error':...}.
    """
    from dashboard.config import DashboardConfig
    from dashboard.data.active_tasks import collect_tasks_with_counts

    root, shaped = _make_project(
        tmp_path,
        project_dir='offline_ext',
        tasks=[
            {
                'id': 99,
                'title': 'blocked on upstream',
                'status': 'pending',
                'dependencies': [],
                'metadata': {'external_deps': ['dark_factory:42', 'reify:7']},
            },
        ],
    )

    async def _fake_fetch(client, config, project_root):
        return list(shaped)

    async def _offline_ext_statuses(client, config, deps):
        return {'offline': True, 'error': 'down'}

    _register_fetch_tasks(monkeypatch, _fake_fetch)
    monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _offline_ext_statuses)

    cfg = DashboardConfig(project_root=root)
    active, _snapshots = await collect_tasks_with_counts(
        client=dummy_client, config=cfg, resolve_external=True,
    )

    assert len(active) == 1
    row = active[0]
    for entry in row['external_deps']:
        assert entry['status'] == 'offline', (
            f"expected status='offline' for dep {entry['id']!r} when MCP is offline, "
            f"got: {entry['status']!r}"
        )


# ---------------------------------------------------------------------------
# TestShapeOneProjectNarrowing — the default render must ask for only what it
# renders, and take its counts from the compact seam (task 3857 step-7,
# retargeted onto the snapshot unit by task 5587)
# ---------------------------------------------------------------------------


def _canned_mcp(rows, status_map):
    """Return ``(mcp_tool_call_fake, calls)`` emulating the fused-memory substrate.

    Faithful to what was traced for task 3857, because the whole point of the
    narrowing work is that the SERVER does the filtering:

    * ``get_tasks`` applies ``statuses`` as a row filter (SQL ``status IN``),
      then slices ``page_size``/``offset`` over a list ordered by ASCENDING
      ``id`` — so reaching the high-id end requires a computed offset.
    * ``get_statuses`` returns the compact ``{id: status}`` map. No
      ``pagination`` envelope: its absence is the substrate's own spelling of
      "this answer is COMPLETE", which is what these small fixtures are.
      ``test_task_snapshot.py::CannedMCP`` is the paging-aware emulation, and
      the walk itself is asserted there rather than re-asserted here.

    *rows* are raw MCP rows (string ids); *status_map* is ``{int id: status}``.
    """
    calls: list[dict] = []

    async def _mcp(client, url, tool, args, **_kw):
        # ``kwargs`` is recorded too so the per-request budget (``timeout=``,
        # which rides as a keyword and never inside ``args``) is assertable at
        # the wire — see test_every_per_project_call_carries_the_per_request_budget.
        calls.append({'tool': tool, 'args': dict(args), 'kwargs': dict(_kw)})
        if tool == 'get_statuses':
            return {'statuses': {str(k): v for k, v in status_map.items()}}
        if tool == 'get_tasks':
            statuses = args.get('statuses')
            selected = [
                r for r in rows
                if statuses is None or r.get('status') in statuses
            ]
            selected.sort(key=lambda r: int(r['id']))  # ORDER BY id ASC
            page_size = args.get('page_size')
            if page_size is not None:
                start = args.get('offset', 0)
                selected = selected[start:start + page_size]
            return {'tasks': selected}
        raise AssertionError(f'unexpected tool {tool!r}')

    return _mcp, calls


def _raw_row(task_id, status, *, title=None, updated_at=None):
    return {
        'id': str(task_id),
        'title': title or f'task {task_id}',
        'status': status,
        'dependencies': [],
        'metadata': {},
        'updatedAt': updated_at or f'2026-01-01T00:00:{task_id % 60:02d}+00:00',
    }


class TestShapeOneProjectNarrowing:
    """The Tasks-tab fetch must have a ceiling that does not grow with the tree.

    Asserted at the MCP wire, through the real ``acquire_snapshot`` /
    ``fetch_tasks`` / ``fetch_statuses`` path, because "the dashboard discards
    the done rows afterwards" is precisely the defect — what matters is which
    rows the server was asked for.

    Since task 5587 there is no terminal read on this path at all: the bounded
    window moved behind ``?terminal=<project>`` and is asserted at that seam.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_tasks_cache_clear()
        _register_runtime(monkeypatch, {})
        yield
        tasks_mod._fetch_tasks_cache_clear()

    @staticmethod
    def _one_project_config(tmp_path):
        root = tmp_path / 'dark-factory'
        root.mkdir(parents=True, exist_ok=True)
        return DashboardConfig(project_root=root)

    @staticmethod
    def _get_tasks_calls(calls):
        return [c for c in calls if c['tool'] == 'get_tasks']

    @staticmethod
    async def _share(config, client, *, now=None):
        """One root's whole share of a render: acquire the unit, shape its rows."""
        from dashboard.data.active_tasks import _acquire_and_shape

        return await _acquire_and_shape(
            client, config, config.project_root,
            now=now or _STUB_AS_OF, runtime=None,
        )

    async def test_the_default_path_never_asks_for_a_terminal_row(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(a) EVERY path is now the scheduler path: two calls, no 'done'.

        This used to be a claim about the caps-0/0 CALLER. There is no other
        caller any more — the terminal window is a separate request — so the
        claim is about the render, and the ``for call in calls`` sweep below
        is exhaustive rather than a sample.
        """
        from dashboard.data.task_snapshot import PER_PROJECT_MCP_CALLS

        rows = [_raw_row(1, 'in-progress'), _raw_row(2, 'pending')]
        rows += [_raw_row(i, 'done') for i in range(10, 60)]
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        active, snapshot = await self._share(config, dummy_client)

        assert classify(snapshot) is SnapshotHealth.OK
        assert len(calls) == 2, f'expected exactly 2 MCP calls, got {calls}'
        assert len(calls) < len(PER_PROJECT_MCP_CALLS), (
            'the default render must spend only two of the three bounded '
            f'operations in the roster {PER_PROJECT_MCP_CALLS} — the third is '
            'reserved for the ?terminal= request, which is the only one that '
            'asks for terminal rows'
        )
        tools = sorted(c['tool'] for c in calls)
        assert tools == ['get_statuses', 'get_tasks']

        get_tasks_call = self._get_tasks_calls(calls)[0]
        assert get_tasks_call['args'].get('statuses') == sorted(ACTIVE), (
            'the row read must ask for every ACTIVE status — including the '
            "'review' and 'infra-hold' members the tab's own five-status copy "
            'omitted, which rendered as tasks silently missing from the table'
        )
        for call in calls:
            requested = call['args'].get('statuses') or []
            assert 'done' not in requested and 'cancelled' not in requested, (
                f'the default render must never request terminal rows: {call}'
            )
        assert {r['title'] for r in active} == {'task 1', 'task 2'}
        assert _measured_census(snapshot).counts[TaskStatus.DONE] == 50

    async def test_every_per_project_call_carries_the_per_request_budget(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """The budget ROSTER must describe the shipped operations, not merely count them.

        ``test_tasks_budget.py`` machine-checks
        ``PER_CALL_TIMEOUT * len(PER_PROJECT_MCP_CALLS) <=
        _TASKS_PER_PROJECT_BUDGET``.  That arithmetic is only a true statement
        ABOUT THIS SYSTEM if every enumerated operation actually threads the
        term.  ``fetch_statuses`` shipped without it, so one of the three ran
        on ``mcp_tool_call``'s 10 s default and could alone overrun the
        per-project budget the roster claims to bound — a constants-only test
        cannot see that, which is why this one asserts at the WIRE.

        The roster enumerates bounded OPERATIONS, not HTTP requests: the
        status-map walk is ``ceil(N / 2000)`` requests bounded as ONE by the
        ``wait_for`` around it.  So the assertion is that the operations
        OBSERVED are a subset of the roster and each request carries the term
        — never that the request count equals the roster length, which would
        be false the moment a tree needs a second page.

        The term is the Tasks-tab-LOCAL ``PER_CALL_TIMEOUT`` (task 4884), NOT
        ``tasks.DEFAULT_PER_CALL_TIMEOUT``.  Asserting the shared default here
        would be actively wrong in a way this test exists to catch: three
        route budgets bind the shared default by reference, so pinning the
        Tasks tab to it re-couples exactly what the local constant was
        introduced to decouple.  If this assertion fails because the two
        values converged, delete the local constant — do not edit this test to
        follow it.
        """
        from dashboard.data.task_snapshot import (
            PER_CALL_TIMEOUT,
            PER_PROJECT_MCP_CALLS,
        )

        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(10, 20)]
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        await self._share(config, dummy_client)

        observed = {
            'get_tasks': 'get_tasks[active]',
            'get_statuses': 'get_statuses[walk]',
        }
        spent = {observed[c['tool']] for c in calls}
        assert spent <= set(PER_PROJECT_MCP_CALLS), (
            f'the render spent {sorted(spent)}, which the roster '
            f'{PER_PROJECT_MCP_CALLS} does not enumerate — an operation '
            'outside the roster is unbudgeted, and test_tasks_budget.py '
            'cannot see it'
        )
        assert spent == {'get_tasks[active]', 'get_statuses[walk]'}
        for call in calls:
            assert call['kwargs'].get('timeout') == PER_CALL_TIMEOUT, (
                f"{call['tool']} was issued with timeout="
                f"{call['kwargs'].get('timeout')!r}, not the Tasks tab's own "
                f'PER_CALL_TIMEOUT ({PER_CALL_TIMEOUT}s) — with '
                "no keyword it falls back to mcp_tool_call's 10s default, and "
                'with the SHARED default it silently under-budgets the '
                '5 000-task trees this tab reads, so the per-project budget '
                'arithmetic in test_tasks_budget.py does not describe it'
            )

    async def test_the_census_comes_from_the_map_not_the_rows(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(c) The status map has terminal tasks the ROWS can never contain.

        A count derived from the returned rows is provably wrong here — the
        rows are narrowed to ACTIVE server-side, so every one of the 20 done
        tasks is invisible to them. That is the whole reason the count comes
        from the compact seam.
        """
        rows = [_raw_row(1, 'in-progress')]
        rows += [_raw_row(i, 'done') for i in range(100, 120)]  # 20 done rows
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, _calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._one_project_config(tmp_path)
        active, snapshot = await self._share(config, dummy_client)

        assert classify(snapshot) is SnapshotHealth.OK
        emitted_done = [r for r in active if r.get('status') == 'done']
        assert emitted_done == [], 'sanity: the rows really are ACTIVE-only'
        assert _measured_census(snapshot).counts[TaskStatus.DONE] == 20, (
            'the done count must come from the compact status map (20), not '
            f'from the {len(emitted_done)} done rows the render fetched'
        )
        assert _measured_census(snapshot).total == 21

    async def test_offline_active_fetch_still_reports_the_project_offline(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(f) The existing offline contract is preserved by the new call shape."""
        import httpx

        async def _refuse(client, url, tool, args, **_kw):
            raise httpx.ConnectError('refused')

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _refuse)

        config = self._one_project_config(tmp_path)
        active, snapshot = await self._share(config, dummy_client)

        assert classify(snapshot) is SnapshotHealth.OFFLINE
        assert active == []
        assert snapshot.census.value is None, (
            'a failed read must never produce a census — a zero one reads as '
            'a measured "this project has no tasks"'
        )
        assert snapshot.census.reason, 'an unknown datum must say why'

    async def test_a_failed_map_leaves_the_census_non_fresh_and_the_rows_fresh(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """A failed compact-map read must not declare a healthy project offline.

        The two halves degrade INDEPENDENTLY: the rows are still good and
        still fresh, and only the census loses its source. That split is what
        the count-unknown banner is drawn from, and merging it into offline
        would tell an operator fused-memory is down while its rows render.
        """
        import httpx

        rows = [_raw_row(1, 'in-progress'), _raw_row(100, 'done')]

        async def _statuses_fail(client, url, tool, args, **_kw):
            if tool == 'get_statuses':
                raise httpx.ConnectError('refused')
            statuses = args.get('statuses')
            selected = [
                r for r in rows
                if statuses is None or r.get('status') in statuses
            ]
            return {'tasks': selected}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _statuses_fail)

        config = self._one_project_config(tmp_path)
        active, snapshot = await self._share(config, dummy_client)

        assert classify(snapshot) is SnapshotHealth.COUNT_UNKNOWN, (
            'the row read succeeded — the project is neither offline nor '
            'degraded, and its count is simply not measured'
        )
        assert [r['title'] for r in active if r.get('status') == 'in-progress'] == ['task 1']
        assert snapshot.rows.state is DatumState.FRESH
        assert snapshot.census.state is not DatumState.FRESH
        assert snapshot.census.value is None, (
            'the census must be UNKNOWN, not a fabricated zero: the rows are '
            'ACTIVE-only, so counting them would report zero done tasks for a '
            'project that has them'
        )
        assert 'ConnectError' in (snapshot.census.reason or ''), (
            'the reason must carry the producer\'s own failure text verbatim, '
            f'got {snapshot.census.reason!r}'
        )

    async def test_the_collector_keeps_that_split_per_root(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """...and it survives the walk, so the wire can name the root.

        The front end has no authoritative count for such a root and falls
        back to counting the rows it received — which is why the entry must
        say UNKNOWN rather than simply omit a number, and why the collector
        must not quietly reclassify the root as offline or degraded.
        """
        import httpx

        rows = [_raw_row(1, 'in-progress'), _raw_row(100, 'done')]

        async def _statuses_fail(client, url, tool, args, **_kw):
            if tool == 'get_statuses':
                raise httpx.ConnectError('refused')
            statuses = args.get('statuses')
            return {'tasks': [
                r for r in rows
                if statuses is None or r.get('status') in statuses
            ]}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _statuses_fail)

        config = self._one_project_config(tmp_path)
        _active, snapshots = await collect_tasks_with_counts(dummy_client, config)

        assert _offline(snapshots) == [], 'the row read succeeded — not offline'
        assert _degraded(snapshots) == [], 'the budget was not exceeded — not degraded'
        assert _count_unknown(snapshots) == ['dark-factory']
        assert 'dark-factory' not in _done_counts(snapshots), (
            'a project whose count is UNKNOWN must contribute no count at '
            f'all, not a fabricated value; got {_done_counts(snapshots)!r}'
        )


# ---------------------------------------------------------------------------
# TestDepsOutsideTheFetchedRows — dependency chips must not silently vanish
# (task 3857 step-9)
# ---------------------------------------------------------------------------


class TestDepsOutsideTheFetchedRows:
    """A narrowed row fetch must not silently delete dependency chips.

    ``_resolve_deps`` used to read a ``by_id`` built over the WHOLE tree, so
    every dep id resolved.  Now the rows are ACTIVE-only, so a done dependency
    is never among them — and the ``continue`` would drop a chip that renders
    today.  The unit's raw status map is the bounded source that keeps the
    load-bearing half of the chip (the ``done`` flag) honest, which is why the
    map stays on the in-process record even though it is not part of the wire
    shape.
    """

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        import dashboard.data.tasks as tasks_mod
        tasks_mod._fetch_tasks_cache_clear()
        _register_runtime(monkeypatch, {})
        yield
        tasks_mod._fetch_tasks_cache_clear()

    @staticmethod
    def _config(tmp_path):
        root = tmp_path / 'proj'
        root.mkdir(parents=True, exist_ok=True)
        return DashboardConfig(project_root=root)

    async def _shape(self, monkeypatch, tmp_path, dummy_client):
        """One active task depending on ids inside, outside and beyond the tree."""
        rows = [
            _raw_row(1, 'in-progress', title='the active one'),
            _raw_row(5, 'pending', title='an active dep'),
            # A done dep: present in the status map, and — since the row read
            # is narrowed to ACTIVE — never among the fetched rows.
            _raw_row(10, 'done', title='long-parked dep'),
            _raw_row(90, 'done', title='recent dep'),
        ]
        rows[0]['dependencies'] = ['5', '10', '90', '999']
        status_map = {int(r['id']): r['status'] for r in rows}
        mcp, _calls = _canned_mcp(rows, status_map)
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        config = self._config(tmp_path)
        active, _snapshot = await TestShapeOneProjectNarrowing._share(
            config, dummy_client,
        )
        row = next(r for r in active if r.get('status') == 'in-progress')
        return {int(d['id'].rsplit('T-', 1)[-1]): d for d in row['deps']}

    async def test_done_dep_outside_the_rows_is_still_emitted(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(a) An honest partial entry beats a dropped chip."""
        deps = await self._shape(monkeypatch, tmp_path, dummy_client)

        assert 10 in deps, (
            'a done dependency outside the fetched rows must still render a '
            f'chip — got only {sorted(deps)}'
        )
        assert deps[10]['done'] is True, 'the done flag comes from the status map'
        assert deps[10]['title'] == '', (
            'the title is unresolvable without an extra whole-tree read, so it '
            'degrades to the empty string the shape already allows'
        )

    async def test_dep_present_in_the_rows_keeps_its_real_title(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(b) No regression for deps whose full row was actually fetched."""
        deps = await self._shape(monkeypatch, tmp_path, dummy_client)

        assert deps[5]['title'] == 'an active dep'

    async def test_dep_absent_from_rows_and_map_is_still_dropped(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(c) The id does not exist — fabricating a chip is worse than omitting it."""
        deps = await self._shape(monkeypatch, tmp_path, dummy_client)

        assert 999 not in deps, (
            'an id absent from both the rows and the status map must be dropped'
        )

    async def test_active_dep_resolves_with_done_false(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(d) A non-done dep is emitted with done False, however it resolved."""
        deps = await self._shape(monkeypatch, tmp_path, dummy_client)

        assert deps[5]['done'] is False

    async def test_active_dep_outside_the_rows_yields_done_false(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(d) The status-map fallback must not assume 'not fetched' means done."""
        rows = [
            _raw_row(1, 'in-progress', title='the active one'),
            _raw_row(7, 'blocked', title='a blocked dep'),
            _raw_row(80, 'done'),
        ]
        rows[0]['dependencies'] = ['7', '80']
        status_map = {int(r['id']): r['status'] for r in rows}

        async def _mcp(client, url, tool, args, **_kw):
            if tool == 'get_statuses':
                return {'statuses': {str(k): v for k, v in status_map.items()}}
            statuses = args.get('statuses')
            # Deliberately omit the blocked row from the ACTIVE rows so its
            # only source is the compact map.
            selected = [
                r for r in rows
                if (statuses is None or r.get('status') in statuses)
                and r['id'] != '7'
            ]
            selected.sort(key=lambda r: int(r['id']))
            return {'tasks': selected}

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _mcp)

        config = self._config(tmp_path)
        active, _snapshot = await TestShapeOneProjectNarrowing._share(
            config, dummy_client,
        )
        row = next(r for r in active if r.get('status') == 'in-progress')
        deps = {int(d['id'].rsplit('T-', 1)[-1]): d for d in row['deps']}

        assert deps[7]['done'] is False, 'a blocked dep must never render as done'
        assert deps[7]['title'] == ''


# ---------------------------------------------------------------------------
# The unit's wire rows are the shaped rows (task 5587 step-19)
# ---------------------------------------------------------------------------


class TestTheReturnedUnitsCarryTheShapedRows:
    """``snapshots[label].rows.value`` is the row list ``ACTIVE_TASKS`` is joined from.

    The CACHED unit holds the raw MCP rows, because the next render shapes
    from them. The unit the collector RETURNS must carry the shaped rows, in
    the same ``TaskRow`` shape as ``ACTIVE_TASKS``, because that is what goes
    on the wire. These tests pin that the two exposures cannot disagree, and
    that putting shaped rows on the wire does not use up the raw rows in the
    cache.

    Every read goes through ``dashboard.data.tasks.mcp_tool_call``, so the real
    ``acquire_snapshot`` and its cache run underneath.
    """

    @staticmethod
    def _roots(tmp_path, *names):
        roots = [tmp_path / name for name in names]
        for root in roots:
            root.mkdir()
        return roots

    @staticmethod
    def _tree():
        rows = [_raw_row(1, 'in-progress'), _raw_row(2, 'pending'), _raw_row(3, 'blocked')]
        return rows, {int(row['id']): row['status'] for row in rows}

    async def test_active_tasks_is_the_units_rows_joined_in_root_order(
        self, tmp_path, monkeypatch, dummy_client,
    ):
        """ACTIVE_TASKS is every root's ``rows.value``, concatenated in canonical root order.

        Then the two exposures cannot disagree, and γ3 can retire
        ``ACTIVE_TASKS`` by deleting it.
        """
        df, reify = self._roots(tmp_path, 'df', 'reify')
        mcp, _calls = _canned_mcp(*self._tree())
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)
        config = DashboardConfig(project_root=df, known_project_roots=[reify])

        active, snapshots = await collect_tasks_with_counts(dummy_client, config)

        joined = [row for label in ('df', 'reify') for row in _measured_rows(snapshots[label])]
        assert active == joined
        assert [row['project'] for row in active] == ['df'] * 3 + ['reify'] * 3

    async def test_the_external_dep_overwrite_shows_through_both_exposures(
        self, tmp_path, monkeypatch, dummy_client,
    ):
        """The unit carries the SAME row objects, not a copy that would drift.

        The batched external-dep tail overwrites ``entry['status']`` in place,
        after the per-root walk. It reaches the wire through
        ``TASKS_SNAPSHOT`` only if both exposures hold one set of dicts.
        """
        (root,) = self._roots(tmp_path, 'xdeps')
        mcp, _calls = _canned_mcp(
            [_raw_row(5, 'pending') | {'metadata': {'external_deps': ['dark_factory:13']}}],
            {5: 'pending'},
        )
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)

        async def _resolved(client, config, deps):
            return {'dark_factory:13': 'done'}

        monkeypatch.setattr('dashboard.data.active_tasks.fetch_external_statuses', _resolved)

        active, snapshots = await collect_tasks_with_counts(
            dummy_client, DashboardConfig(project_root=root), resolve_external=True,
        )

        (wire_row,) = _measured_rows(snapshots['xdeps'])
        assert wire_row is active[0]
        assert wire_row['external_deps'] == [{'id': 'dark_factory:13', 'status': 'done'}]

    async def test_an_unmeasured_root_keeps_no_value_rather_than_an_empty_list(
        self, tmp_path, monkeypatch, dummy_client,
    ):
        """``value is None`` is how the envelope says "never measured".

        ``[]`` would claim a measured zero and break the
        ``unknown <=> value is None <=> as_of is None`` triad.
        """
        live, dead = self._roots(tmp_path, 'live', 'dead')
        healthy, _calls = _canned_mcp(*self._tree())

        async def _mcp(client, url, tool, args, **kwargs):
            if args.get('project_root') == str(dead.resolve()):
                raise httpx.ReadTimeout('canned read timeout')
            return await healthy(client, url, tool, args, **kwargs)

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _mcp)
        config = DashboardConfig(project_root=live, known_project_roots=[dead])

        active, snapshots = await collect_tasks_with_counts(dummy_client, config)

        assert snapshots['dead'].rows.state is DatumState.UNKNOWN
        assert snapshots['dead'].rows.value is None
        assert snapshots['live'].rows.value == active

    async def test_a_root_the_budget_cut_off_holds_no_rows_even_with_a_last_good(
        self, tmp_path, monkeypatch, dummy_client,
    ):
        """A cut-off root's unit keeps ``value is None``, even when it was measured before.

        The render shaped no rows for it, so ``ACTIVE_TASKS`` gets none, and
        its unit must agree. The unit could otherwise only carry the last
        good RAW list, which is unshaped and heavy on the wire, or the empty
        shaped list, which claims a measured zero at the last good's instant.
        Its census still ages, because a count needs no shaping.

        The budget is squeezed through the module constant, the same idiom
        ``TestCollectTasksBudget._tighten`` uses. That is the only way to cut
        off a real acquisition without waiting out the shipped 14 s budget.
        """
        import dashboard.data.task_snapshot as snapshot_mod

        (root,) = self._roots(tmp_path, 'df')
        healthy, _calls = _canned_mcp(*self._tree())
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', healthy)
        config = DashboardConfig(project_root=root)
        measured, _ = await collect_tasks_with_counts(dummy_client, config, now=_STUB_AS_OF)
        assert measured, 'the root must have been measured once, leaving a last good'

        async def _hangs(client, url, tool, args, **kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _hangs)
        monkeypatch.setattr(snapshot_mod, 'SNAPSHOT_TTL_SECONDS', 0.0)
        monkeypatch.setattr('dashboard.data.active_tasks._TASKS_PER_PROJECT_BUDGET', 0.05)
        active, snapshots = await collect_tasks_with_counts(
            dummy_client, config, now=_STUB_AS_OF + timedelta(seconds=20),
        )

        unit = snapshots['df']
        assert classify(unit) is SnapshotHealth.DEGRADED
        assert active == []
        assert unit.rows.state is DatumState.UNKNOWN
        assert unit.rows.value is None, unit.rows.value
        assert unit.census.state is DatumState.STALE
        assert unit.census.as_of == _STUB_AS_OF

    async def test_the_cached_unit_keeps_the_raw_rows_the_next_render_shapes(
        self, tmp_path, monkeypatch, dummy_client,
    ):
        """Putting shaped rows on the wire must not use up what the cache holds.

        A second render inside the 15 s TTL is served from the cached unit and
        shapes from its rows. Had the first render replaced them with shaped
        rows, the second would find no integer ids and render an empty table.
        """
        from dashboard.data.task_snapshot import acquire_snapshot

        (root,) = self._roots(tmp_path, 'df')
        mcp, calls = _canned_mcp(*self._tree())
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', mcp)
        config = DashboardConfig(project_root=root)

        first, _ = await collect_tasks_with_counts(dummy_client, config, now=_STUB_AS_OF)
        reads = len(calls)
        second, snapshots = await collect_tasks_with_counts(dummy_client, config, now=_STUB_AS_OF)
        cached = await acquire_snapshot(dummy_client, config, config.project_root, now=_STUB_AS_OF)

        assert len(calls) == reads, 'both later reads must be served from the cached unit'
        assert second == first
        assert snapshots['df'].rows.value == second
        assert cached.rows.value is not None
        assert all('metadata' in row for row in cached.rows.value), (
            'the cached unit must still hold the RAW rows'
        )

    async def test_a_stale_rows_half_is_shaped_and_keeps_its_provenance(
        self, tmp_path, monkeypatch, dummy_client,
    ):
        """Last-good raw rows served as STALE reach the wire shaped, still aged and explained."""
        import dashboard.data.task_snapshot as snapshot_mod

        (root,) = self._roots(tmp_path, 'df')
        healthy, _calls = _canned_mcp(*self._tree())
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', healthy)
        config = DashboardConfig(project_root=root)
        fresh, _ = await collect_tasks_with_counts(dummy_client, config, now=_STUB_AS_OF)

        async def _rows_read_fails(client, url, tool, args, **kwargs):
            if tool == 'get_tasks':
                raise httpx.ReadTimeout('canned read timeout')
            return await healthy(client, url, tool, args, **kwargs)

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _rows_read_fails)
        monkeypatch.setattr(snapshot_mod, 'SNAPSHOT_TTL_SECONDS', 0.0)
        later = _STUB_AS_OF + timedelta(seconds=20)
        active, snapshots = await collect_tasks_with_counts(dummy_client, config, now=later)

        rows = snapshots['df'].rows
        assert rows.state is DatumState.STALE
        assert rows.as_of == _STUB_AS_OF, 'a stale half keeps the instant it was measured'
        assert 'ReadTimeout' in (rows.reason or ''), rows.reason
        assert rows.value == active == fresh

    async def test_a_stale_rows_half_is_strand_judged_when_it_was_measured(
        self, tmp_path, monkeypatch, dummy_client,
    ):
        """Row badges on a STALE half agree with the unit's own ``in_progress_stranded``.

        Both go through ``task_is_stranded`` at ``rows.as_of``, the instant the
        rows were measured. Judging the badge at the render instant instead
        would, once the substrate has been unreachable longer than
        ``STRANDED_HEARTBEAT_TTL``, badge every claim as abandoned while the
        split (judged at ``as_of``) reports none — the same rows disagreeing
        with themselves on one wire payload.
        """
        import dashboard.data.task_snapshot as snapshot_mod
        from dashboard.data.tasks import STRANDED_HEARTBEAT_TTL

        (root,) = self._roots(tmp_path, 'df')
        claimed = _raw_row(1, 'in-progress')
        claimed['claimant_run_id'] = 'run-live'
        claimed['heartbeat_at'] = _STUB_AS_OF.isoformat()
        rows_in = [claimed, _raw_row(2, 'pending')]
        healthy, _calls = _canned_mcp(rows_in, {int(r['id']): r['status'] for r in rows_in})
        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', healthy)
        config = DashboardConfig(project_root=root)
        fresh, fresh_units = await collect_tasks_with_counts(dummy_client, config, now=_STUB_AS_OF)
        assert [row['stranded'] for row in fresh if row['status'] == 'in-progress'] == [False]
        assert fresh_units['df'].in_progress_stranded == 0

        async def _rows_read_fails(client, url, tool, args, **kwargs):
            if tool == 'get_tasks':
                raise httpx.ReadTimeout('canned read timeout')
            return await healthy(client, url, tool, args, **kwargs)

        monkeypatch.setattr('dashboard.data.tasks.mcp_tool_call', _rows_read_fails)
        monkeypatch.setattr(snapshot_mod, 'SNAPSHOT_TTL_SECONDS', 0.0)
        much_later = _STUB_AS_OF + STRANDED_HEARTBEAT_TTL + timedelta(minutes=1)
        active, snapshots = await collect_tasks_with_counts(dummy_client, config, now=much_later)

        unit = snapshots['df']
        assert unit.rows.state is DatumState.STALE
        assert unit.rows.as_of == _STUB_AS_OF
        assert (unit.in_progress_live, unit.in_progress_stranded) == (1, 0)
        badged = [row['stranded'] for row in active if row['status'] == 'in-progress']
        assert badged == [False], 'the badge must be judged at as_of, like the split'
        assert unit.rows.value == active == fresh


# ---------------------------------------------------------------------------
# collect_tasks_with_counts whole-handler budget (task 3857 steps 13/14)
# ---------------------------------------------------------------------------


# Far beyond any budget exercised below, so a project that sleeps this long can
# only ever end by being CUT OFF. Picking a number near the budget instead would
# make "did the deadline fire?" a race rather than a fact.
_BUDGET_SLOW = 5.0


def _register_shaper(monkeypatch, delays, *, offline=(), done_counts=None,
                     external_deps=None, raises=None):
    """Patch ``_acquire_and_shape`` with a per-project coroutine that sleeps.

    ``_acquire_and_shape`` is ONE root's whole share of the render — acquire
    its snapshot, shape its rows — and so is the unit the budget bounds and
    the semaphore admits. It is the seam these tests replace because they are
    about the WALK, not about either half of the share.

    *delays* maps project label -> seconds to sleep before returning; a label
    absent from it returns immediately. Labels in *offline* come back with an
    UNREACHABLE snapshot — a read that demonstrably FAILED, which must stay
    distinguishable from a project the budget never reached.

    *raises* maps project label -> an exception INSTANCE to raise instead of
    returning. Models the unexpected-failure path (a shape bug, a decode
    error, an ``httpx`` transport error escaping the fan-out) as distinct from
    both the timeout and the offline marker.

    *external_deps* is an optional list of external dep ids stamped onto every
    emitted row (in the ``_build_task_row`` shape, each on the ``'unknown'``
    sentinel), so the batched ``fetch_external_statuses`` leg of the handler is
    reachable from this harness — without it ``dep_ids`` is empty and that leg
    short-circuits.

    Returns the list of labels the share was actually INVOKED with, in order.
    That record is what makes "never got its turn" a checkable fact rather
    than an inference from an absence in the output.
    """
    invoked: list[str] = []
    counts = done_counts or {}
    explode = raises or {}

    async def _fake_share(client, config, project_root, *, now=None, runtime=None):
        label = project_root.name
        invoked.append(label)
        delay = delays.get(label, 0.0)
        if delay:
            await asyncio.sleep(delay)
        if label in explode:
            raise explode[label]
        if label in offline:
            return [], _stub_snapshot(unreachable=True)
        row = {'id': f'{label}/T-1', 'project': label, 'status': 'in-progress'}
        if external_deps:
            row['external_deps'] = [
                {'id': dep, 'status': 'unknown'} for dep in external_deps
            ]
        return [row], _stub_snapshot(done=counts.get(label, 0))

    monkeypatch.setattr('dashboard.data.active_tasks._acquire_and_shape', _fake_share)
    return invoked


def _budget_config(tmp_path, labels):
    """A DashboardConfig whose project roots are *labels*, primary first."""
    roots = []
    for label in labels:
        root = tmp_path / label
        root.mkdir(parents=True, exist_ok=True)
        roots.append(root)
    return DashboardConfig(project_root=roots[0], known_project_roots=roots[1:])


class TestCollectTasksBudget:
    """The Tasks-tab aggregation must be bounded as a WHOLE, and degrade honestly.

    ``collect_tasks_with_counts`` walks every configured project root
    sequentially with no deadline anywhere, so its worst case is the SUM of
    every project's worst case — unbounded in the number of roots. The fix is
    the ``/healthz`` shape: one ``loop.time()`` deadline for the handler, one
    ``asyncio.wait_for`` per project, and — the part that is easy to get wrong
    — an explicit marker for every project the budget did not reach.

    That last part is the real contract here. A truncated-but-confident payload
    (rows for the projects that finished, silence for the rest) renders as "no
    active work" on those projects, which is the same invisible-failure class
    the fan-out logging policy was raised to WARNING to close. *degraded*
    (budget expired — state UNKNOWN) is a strictly different fact from
    *offline* (fetch demonstrably failed), and the two must never be merged.
    """

    @pytest.fixture(autouse=True)
    def _no_runtime_fanout(self, monkeypatch):
        """Runtime fan-out returns instantly, so every measured second is the loop's."""
        _register_runtime(monkeypatch, {})

    @staticmethod
    def _tighten(monkeypatch, *, total, per_project):
        monkeypatch.setattr('dashboard.data.active_tasks._TASKS_TOTAL_BUDGET', total)
        monkeypatch.setattr(
            'dashboard.data.active_tasks._TASKS_PER_PROJECT_BUDGET', per_project
        )

    async def test_returns_rows_and_one_validated_snapshot_per_root(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(a) the return shape is ``(rows, {label: TaskSnapshot})``.

        One entry for EVERY configured root, never a row list beside three
        parallel lists of labels. The three lists are DERIVED from the entries
        downstream, so they cannot disagree with the snapshot they describe —
        which is precisely what a fourth and fifth parallel list could.
        """
        _register_shaper(monkeypatch, {}, done_counts={'alpha': 3, 'beta': 5})
        config = _budget_config(tmp_path, ['alpha', 'beta'])

        result = await collect_tasks_with_counts(client=dummy_client, config=config)

        assert len(result) == 2, (
            'expected (active_rows, snapshots_by_label), got '
            f'{len(result)} elements'
        )
        _active, snapshots = result
        assert list(snapshots) == ['alpha', 'beta'], (
            'every configured root must carry an entry, in canonical root '
            f'order; got {list(snapshots)}'
        )
        assert _degraded(snapshots) == []
        assert _offline(snapshots) == []
        assert _count_unknown(snapshots) == []
        assert _done_counts(snapshots) == {'alpha': 3, 'beta': 5}

    async def test_deadline_expiry_marks_unreached_projects_degraded(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(b) projects the handler never reached are named, not silently dropped.

        Budgets are chosen so the arithmetic is one-directional rather than a
        race: ``alpha`` returns instantly, then ``beta`` and ``gamma`` each
        sleep far past their per-project budget and so consume 0.2s + 0.1s =
        the entire 0.3s handler budget. Timers overshoot and never undershoot,
        so ``delta`` and ``epsilon`` are guaranteed to find a non-positive
        remaining budget — they can only be reached by the deadline branch.

        ``_TASKS_ROOT_CONCURRENCY`` is pinned to 1 for exactly that reason,
        and the pin is what makes the derivation above true rather than a
        coincidence. This test is about the DEADLINE branch, not about the
        width: at the shipped width the fast roots slot into the first wave
        and the never-reached branch is simply not the one under test. That
        the branch still fires at the shipped width is asserted separately by
        ``TestCollectTasksWithCountsConcurrency::
        test_degraded_is_preserved_under_concurrency`` (task 4884) — so
        isolating the two here costs no coverage and buys a deterministic
        arithmetic that does not have to be re-derived every time the width
        moves.
        """
        invoked = _register_shaper(
            monkeypatch,
            {'beta': _BUDGET_SLOW, 'gamma': _BUDGET_SLOW},
            done_counts={'alpha': 7},
        )
        self._tighten(monkeypatch, total=0.3, per_project=0.2)
        monkeypatch.setattr(
            'dashboard.data.active_tasks._TASKS_ROOT_CONCURRENCY', 1
        )
        config = _budget_config(
            tmp_path, ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        )

        active, snapshots = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )
        offline, counts, degraded = (
            _offline(snapshots), _done_counts(snapshots), _degraded(snapshots)
        )

        # The project that completed still contributes its rows and its count.
        assert [row['project'] for row in active] == ['alpha']
        assert counts == {'alpha': 7}

        # Everything the budget did not deliver is NAMED.
        assert set(degraded) == {'beta', 'gamma', 'delta', 'epsilon'}

        # ...and the two never-reached projects are provably never-reached:
        # they were not invoked at all, so their degraded marker cannot have
        # come from a per-project timeout.
        assert 'delta' not in invoked and 'epsilon' not in invoked, (
            f'expected the handler deadline to skip delta/epsilon, but it '
            f'invoked {invoked}'
        )

        # Never proven unreachable -> never reported offline.
        assert offline == []
        # No count was measured -> none is fabricated (not even a 0, which
        # would render as a real "this project has zero done tasks").
        assert 'delta' not in counts and 'epsilon' not in counts

    async def test_slow_project_is_cut_off_and_the_next_one_still_runs(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(c) one slow project must not starve its neighbours of their turn.

        The handler budget is left generous here so the ONLY thing that can
        cut ``beta`` short is its own per-project budget — which is what makes
        ``gamma`` completing a fact about per-project containment rather than
        a coincidence of the total.
        """
        invoked = _register_shaper(
            monkeypatch,
            {'beta': _BUDGET_SLOW},
            done_counts={'alpha': 1, 'gamma': 2},
        )
        self._tighten(monkeypatch, total=10.0, per_project=0.2)
        config = _budget_config(tmp_path, ['alpha', 'beta', 'gamma'])

        started = time.monotonic()
        active, snapshots = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )
        elapsed = time.monotonic() - started
        offline, counts, degraded = (
            _offline(snapshots), _done_counts(snapshots), _degraded(snapshots)
        )

        assert degraded == ['beta']
        assert offline == []
        assert 'gamma' in invoked, 'the project after the slow one never got its turn'
        assert {row['project'] for row in active} == {'alpha', 'gamma'}
        assert counts == {'alpha': 1, 'gamma': 2}
        assert elapsed < 1.0, (
            f'elapsed {elapsed:.3f}s — beta sleeps {_BUDGET_SLOW}s, so anything '
            'near that means the per-project budget did not fire'
        )

    async def test_degraded_and_offline_are_disjoint(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(d) a project is either demonstrably offline or unknown — never both.

        Merging the lists would let an operator read "the budget ran out" as
        "fused-memory is down", which sends them to restart a healthy service.
        """
        _register_shaper(
            monkeypatch,
            {'gamma': _BUDGET_SLOW},
            offline=('beta',),
            done_counts={'alpha': 4},
        )
        self._tighten(monkeypatch, total=10.0, per_project=0.2)
        config = _budget_config(tmp_path, ['alpha', 'beta', 'gamma'])

        _active, snapshots = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )
        offline, counts, degraded = (
            _offline(snapshots), _done_counts(snapshots), _degraded(snapshots)
        )

        assert offline == ['beta']
        assert degraded == ['gamma']
        assert set(offline).isdisjoint(degraded)
        # An offline project already had no count; a degraded one must not
        # acquire a fabricated one either.
        assert counts == {'alpha': 4}

    async def test_one_projects_unexpected_error_cannot_blank_the_whole_tab(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(e) an UNEXPECTED exception from one root must not 500 the handler.

        The per-project ``try`` caught ``TimeoutError`` only, so any other
        exception escaping ``_shape_one_project`` — a decode error, a shape
        bug, an ``httpx`` transport error not converted to an offline marker by
        the fan-out — propagated out of the whole aggregation and 500'd
        ``/api/v2/dashboard/tasks``, discarding every HEALTHY project's rows
        that had already been collected.

        That is the same "one bad root blanks the whole tab" failure the
        ``TASKS_OFFLINE`` fix exists to close, relocated from the banner to the
        handler.  The fan-out normally converts failures into offline markers,
        so this is defense-in-depth rather than a demonstrated crash — which is
        exactly why it needs a test: nothing else exercises the path.

        The failing root is marked OFFLINE, not degraded: the read demonstrably
        failed, which is what *offline* means.  *degraded* is reserved for
        "the budget never let us find out".
        """
        invoked = _register_shaper(
            monkeypatch,
            {},
            done_counts={'alpha': 4, 'gamma': 7},
            raises={'beta': ValueError('malformed get_tasks payload')},
        )
        config = _budget_config(tmp_path, ['alpha', 'beta', 'gamma'])

        active, snapshots = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )
        offline, counts, degraded = (
            _offline(snapshots), _done_counts(snapshots), _degraded(snapshots)
        )

        # The walk CONTINUED past the exploding root rather than unwinding.
        assert invoked == ['alpha', 'beta', 'gamma'], (
            f'invoked {invoked} — gamma never got its turn, so the exception '
            'aborted the aggregation instead of being contained to beta'
        )
        assert offline == ['beta']
        assert degraded == [], 'the budget did not expire — nothing is UNKNOWN'
        # Every healthy project still renders, and beta contributes no
        # fabricated count.
        assert {row['project'] for row in active} == {'alpha', 'gamma'}
        assert counts == {'alpha': 4, 'gamma': 7}

    async def test_unexpected_error_is_logged_at_warning_with_the_project(
        self, monkeypatch, tmp_path, dummy_client, caplog
    ):
        """(f) ...and the swallowed exception must not be silent.

        Containing the failure is only half the fix: an exception absorbed into
        an offline marker with no log is a bug that renders as a routine
        outage forever.  The record must name the project and carry the
        traceback, so the next reader can tell "fused-memory is down" from
        "our shaping code raised".
        """
        _register_shaper(
            monkeypatch, {}, raises={'beta': ValueError('malformed get_tasks payload')},
        )
        config = _budget_config(tmp_path, ['alpha', 'beta'])

        with caplog.at_level(logging.WARNING, logger='dashboard.data.active_tasks'):
            await collect_tasks_with_counts(client=dummy_client, config=config)

        records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'beta' in r.getMessage()
        ]
        assert records, (
            'an unexpected per-project exception was swallowed with no WARNING'
        )
        assert any(r.exc_info for r in records), (
            'the WARNING carries no traceback — the exception type and origin '
            'are exactly what distinguishes this from a routine outage'
        )

    async def test_total_wall_time_is_bounded_by_the_handler_budget(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(e) the whole call is bounded, not merely each project within it."""
        import dashboard.data.active_tasks as active_tasks_mod

        delays = {label: _BUDGET_SLOW for label in ('beta', 'gamma', 'delta')}
        _register_shaper(monkeypatch, delays)
        self._tighten(monkeypatch, total=0.5, per_project=0.2)
        config = _budget_config(tmp_path, ['alpha', 'beta', 'gamma', 'delta'])

        started = time.monotonic()
        await collect_tasks_with_counts(client=dummy_client, config=config)
        elapsed = time.monotonic() - started

        sum_of_sleeps = sum(delays.values())
        assert elapsed < sum_of_sleeps / 2, (
            f'elapsed {elapsed:.3f}s is not well under the {sum_of_sleeps}s sum '
            'of per-project sleeps — the walk is still additive in the number '
            'of roots'
        )
        # +0.5s of tolerance for event-loop scheduling, the same convention as
        # test_healthz_deadline.py's elapsed assertions.
        assert elapsed < active_tasks_mod._TASKS_TOTAL_BUDGET + 0.5, (
            f'elapsed {elapsed:.3f}s exceeded the whole-handler budget of '
            f'{active_tasks_mod._TASKS_TOTAL_BUDGET}s'
        )

    async def test_external_status_fetch_cannot_overrun_the_handler_budget(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(e2) the batched external-dep call is BOUNDED, not merely deadline-CHECKED.

        It runs AFTER the per-project walk, guarded only by an
        ``ext_remaining <= 0`` skip.  With a small POSITIVE remainder the call
        still proceeded unbounded on ``mcp_tool_call``'s 10 s default — and a
        cold MCP session performs three posts, per fan-out URL — so the handler
        could exceed ``_TASKS_TOTAL_BUDGET`` by ~30 s and blow past ``data.js``'s
        30 000 ms fetch abort.  That is precisely the "the degraded payload is
        aborted before it can be rendered" failure ``test_tasks_budget.py``
        exists to prevent and structurally cannot see: it checks constants, and
        this leg simply did not honour them.

        Expiry leaves every entry on its honest ``'unknown'`` sentinel — the
        same treatment the ``ext_remaining <= 0`` skip and the per-project
        ``TimeoutError`` branch already give.
        """
        import dashboard.data.active_tasks as active_tasks_mod

        _register_shaper(
            monkeypatch, {}, external_deps=['dark_factory:13', 'reify:8'],
        )

        async def _slow_ext(client, config, deps):
            await asyncio.sleep(_BUDGET_SLOW)
            return {'dark_factory:13': 'done'}

        monkeypatch.setattr(
            'dashboard.data.active_tasks.fetch_external_statuses', _slow_ext
        )
        # A positive remainder when the external leg is reached: the
        # ext_remaining <= 0 skip must NOT be what saves us here, or the test
        # would pass against the unbounded code.
        self._tighten(monkeypatch, total=1.0, per_project=0.5)
        config = _budget_config(tmp_path, ['alpha'])

        started = time.monotonic()
        active, snapshots = await collect_tasks_with_counts(
            client=dummy_client, config=config, resolve_external=True,
        )
        elapsed = time.monotonic() - started
        offline = _offline(snapshots)

        assert elapsed < _BUDGET_SLOW / 2, (
            f'elapsed {elapsed:.3f}s is not well under the {_BUDGET_SLOW}s '
            'external-status sleep — the batched external-dep call is still '
            'unbounded, so the handler budget does not bound the handler'
        )
        assert elapsed < active_tasks_mod._TASKS_TOTAL_BUDGET + 0.5, (
            f'elapsed {elapsed:.3f}s exceeded the whole-handler budget of '
            f'{active_tasks_mod._TASKS_TOTAL_BUDGET}s'
        )
        # Degrade honestly: no status was read, so none is fabricated, and the
        # project is NOT declared offline (its rows loaded fine).
        assert offline == []
        assert active[0]['external_deps'] == [
            {'id': 'dark_factory:13', 'status': 'unknown'},
            {'id': 'reify:8', 'status': 'unknown'},
        ]

    async def test_happy_path_is_unchanged_by_the_budget(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(f) with fast projects, nothing degrades and the payload is identical."""
        _register_shaper(monkeypatch, {}, done_counts={'alpha': 11, 'beta': 22})
        # Shipped constants deliberately NOT tightened here: the happy path
        # must hold under the values that actually ship.
        config = _budget_config(tmp_path, ['alpha', 'beta'])

        active, snapshots = await collect_tasks_with_counts(
            client=dummy_client, config=config,
        )

        assert _degraded(snapshots) == []
        assert _offline(snapshots) == []
        assert _done_counts(snapshots) == {'alpha': 11, 'beta': 22}
        assert [row['id'] for row in active] == ['alpha/T-1', 'beta/T-1']

    async def test_collect_active_tasks_still_returns_two_elements(
        self, monkeypatch, tmp_path, dummy_client
    ):
        """(g) the scheduler's caller keeps its two-element contract.

        ``data/scheduler.py`` unpacks ``(active, offline)``; the snapshots are
        absorbed by ``collect_active_tasks``, not leaked to it, and the offline
        labels it does return are read off their kind rather than carried in a
        parallel list.
        """
        _register_shaper(monkeypatch, {'beta': _BUDGET_SLOW}, done_counts={'alpha': 1})
        self._tighten(monkeypatch, total=10.0, per_project=0.2)
        config = _budget_config(tmp_path, ['alpha', 'beta'])

        result = await collect_active_tasks(client=dummy_client, config=config)

        assert len(result) == 2, (
            f'collect_active_tasks must keep its (active, offline) shape, got '
            f'{len(result)} elements'
        )
        active, offline = result
        assert [row['project'] for row in active] == ['alpha']
        # A degraded project is NOT offline here either — the marker is
        # dropped by this narrower contract, not silently reclassified.
        assert offline == []


# ---------------------------------------------------------------------------
# workstream C cause 2 (task 4884, #4795): the per-root walk is CONCURRENT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCollectTasksWithCountsConcurrency:
    """The per-root walk must be parallel-but-bounded, and still deterministic.

    The starvation this closes is arithmetic. The walk used to be SEQUENTIAL,
    so with N roots the wall clock was the SUM of the per-root costs against a
    single ``_TASKS_TOTAL_BUDGET`` — and the incident's 9 roots could not fit,
    so the same trailing roots were reported degraded on every render
    (``project pump-web-ui: skipped — the 20.0s Tasks budget was already
    spent``). Concurrency at width W turns the worst case into roughly
    ``ceil(N / W) * _TASKS_PER_PROJECT_BUDGET``.

    (a) and (b) are the fix. (c)-(f) are the REGRESSION FENCE around it: the
    offline/degraded/order semantics are load-bearing product facts and
    concurrency is exactly the kind of change that quietly breaks them, so
    they are asserted here rather than assumed to be covered elsewhere.
    """

    def _n_root_config(self, tmp_path, n: int) -> DashboardConfig:
        """A config with *n* roots: the primary plus ``n - 1`` known roots."""
        roots = []
        for i in range(n):
            root = tmp_path / f'proj-{i:02d}'
            root.mkdir()
            roots.append(root)
        return DashboardConfig(
            project_root=roots[0], known_project_roots=roots[1:],
        )

    def _tracking_stub(self, monkeypatch, *, dwell: float = 0.05, rows=None,
                       offline_for=None, raise_for=None, serve_first=None):
        """Patch ``_acquire_and_shape`` with a stub that records enter/exit.

        Returns the shared ``events`` list of ``(label, 'enter'|'exit', t)``.

        With *serve_first* set to N the first N ADMISSIONS return without
        awaiting at all and every admission after them hangs forever, so which
        roots a render serves is a property of the admission order rather than
        a race between a dwell and a budget. Same idiom, and the same reason,
        as ``TestCollectTasksWithCountsFairness._admission_recorder``.
        """
        import dashboard.data.active_tasks as at_mod

        events: list[tuple[str, str, float]] = []
        offline_for = set(offline_for or ())
        raise_for = set(raise_for or ())
        admitted = 0

        async def _stub(client, config, project_root, **kwargs):
            nonlocal admitted
            label = project_root.name
            loop = asyncio.get_running_loop()
            events.append((label, 'enter', loop.time()))
            admitted += 1
            try:
                if serve_first is not None:
                    if admitted > serve_first:
                        # A wedged MCP leg: never returns, so the caller's own
                        # budget is what ends it — on every host alike.
                        await asyncio.Event().wait()
                else:
                    await asyncio.sleep(dwell)
                if label in raise_for:
                    raise RuntimeError(f'shaping blew up for {label}')
                if label in offline_for:
                    return [], _stub_snapshot(unreachable=True)
                row = (
                    [dict(rows_for_label) for rows_for_label in rows(label)]
                    if rows is not None
                    else [{'_task_uid': f'{label}/T-1', 'project': label}]
                )
                return row, _stub_snapshot(done=7)
            finally:
                events.append((label, 'exit', asyncio.get_running_loop().time()))

        monkeypatch.setattr(at_mod, '_acquire_and_shape', _stub)
        return events

    @staticmethod
    def _max_simultaneous(events) -> int:
        """Peak number of roots inside the stub at once, from the event log."""
        live = peak = 0
        # Tie-break ENTER before EXIT at an identical loop.time(): the loop
        # clock is coarse enough for a fast stub's exit and the next root's
        # entry to share a timestamp, and ordering the exit first would
        # under-count live occupancy and fail the strict `peak == width`
        # assertion below for a reason that is not about the semaphore.
        for _label, kind, _t in sorted(events, key=lambda e: (e[2], e[1] == 'exit')):
            if kind == 'enter':
                live += 1
                peak = max(peak, live)
            else:
                live -= 1
        return peak

    async def test_walk_saturates_the_semaphore_and_never_exceeds_it(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(a) The peak in-flight root count is EXACTLY the configured width.

        Asserting equality rather than ``> 1`` is deliberate and is the whole
        value of this test: a semaphore that is present but never saturated
        proves nothing (the walk could still be effectively serial), and an
        unbounded ``gather`` would show all 9 — the ``httpx.PoolTimeout``
        hazard ``burndown.py`` records against the shared client.
        """
        import dashboard.data.active_tasks as at_mod

        config = self._n_root_config(tmp_path, 9)
        _register_runtime(monkeypatch, {})
        events = self._tracking_stub(monkeypatch)

        await collect_tasks_with_counts(client=dummy_client, config=config)

        peak = self._max_simultaneous(events)
        assert peak == at_mod._TASKS_ROOT_CONCURRENCY, (
            f'peak in-flight roots was {peak}, expected exactly '
            f'_TASKS_ROOT_CONCURRENCY={at_mod._TASKS_ROOT_CONCURRENCY} over 9 '
            'roots — 1 means the sequential walk that starved the tail is '
            'still in place, 9 means the semaphore is missing (an unbounded '
            'fan-out against the single fused-memory server on the httpx '
            'client the 3s render polls share), and anything strictly between '
            '1 and the width means the semaphore is never saturated so the '
            'concurrency it claims to provide is not actually delivered'
        )

    async def test_admission_proceeds_in_ceil_n_over_w_waves(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(b) The 9 roots are admitted in ``ceil(N/W)`` waves, not 9 turns.

        This is the assertion that actually pins the user-visible symptom —
        "a cold render costs the entire 20 s budget and the tail degrades" —
        because wall clock is ``waves * per-root cost``.

        DERIVED FROM THE EVENT LOG, NOT FROM A CLOCK. The earlier form ran 9
        real ``asyncio.sleep(0.05)``s and asserted ``elapsed < 0.30s``, which
        leaves ~0.15 s of headroom for scheduling — the same class of
        host-speed race commit a83febb5bc removed three of elsewhere on this
        branch, and one that fails LOUDEST under the xdist contention CI
        actually runs with. Here each wave is released by an explicit gate, so
        the wave COUNT is observed directly and no host can be too slow.
        """
        import math

        import dashboard.data.active_tasks as at_mod

        width = at_mod._TASKS_ROOT_CONCURRENCY
        config = self._n_root_config(tmp_path, 9)
        _register_runtime(monkeypatch, {})

        entered: list[str] = []
        gate = asyncio.Event()

        async def _stub(client, config_, project_root, **kwargs):
            entered.append(project_root.name)
            # Reads `gate` at CALL time, so a root admitted in wave k waits on
            # wave k's gate object and is unaffected by the rebinding below.
            await gate.wait()
            return (
                [{'_task_uid': f'{project_root.name}/T-1', 'project': project_root.name}],
                _stub_snapshot(done=1),
            )

        monkeypatch.setattr(at_mod, '_acquire_and_shape', _stub)

        async def _settle() -> None:
            """Yield until the walk stops admitting. Bounded, and clock-free."""
            stable = 0
            while stable < 5:
                before = len(entered)
                await asyncio.sleep(0)
                stable = stable + 1 if len(entered) == before else 0

        walk = asyncio.create_task(
            collect_tasks_with_counts(client=dummy_client, config=config),
        )

        admitted = 0
        wave_sizes: list[int] = []
        while admitted < 9:
            await _settle()
            newly = len(entered) - admitted
            assert 0 < newly <= width, (
                f'wave {len(wave_sizes) + 1} admitted {newly} roots against a '
                f'_TASKS_ROOT_CONCURRENCY of {width} — 0 means the walk '
                'stalled with slots free, more than the width means the '
                'semaphore is not bounding anything (an unbounded fan-out '
                'against the single fused-memory server on the httpx client '
                'the 3 s render polls share)'
            )
            wave_sizes.append(newly)
            admitted += newly
            opening, gate = gate, asyncio.Event()
            opening.set()  # let this wave finish, freeing its slots

        active, snapshots = await walk
        degraded = _degraded(snapshots)

        assert len(wave_sizes) == math.ceil(9 / width), (
            f'the 9-root walk took {len(wave_sizes)} waves '
            f'({wave_sizes}) at width {width}, not the expected '
            f'{math.ceil(9 / width)} — 9 means the roots are still walked one '
            'at a time, which is the cost model that made the cold render '
            'exhaust _TASKS_TOTAL_BUDGET and starve the tail'
        )
        assert wave_sizes[0] == width, (
            f'the first wave admitted {wave_sizes[0]} of {width} slots; a '
            'semaphore that is never saturated delivers none of the '
            'concurrency it claims'
        )
        assert len(active) == 9 and degraded == [], (
            'every root was released, so every root must have been served'
        )

    async def test_row_order_is_root_order_not_completion_order(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(c) Concurrency must not be allowed to reorder the payload.

        The Tasks tab renders ``all_active`` directly, so completion order
        leaking into the payload would reshuffle the table on every 3 s poll.
        The stub finishes roots in REVERSE root order to force the issue.
        """
        config = self._n_root_config(tmp_path, 6)
        _register_runtime(monkeypatch, {})

        async def _stub(client, config_, project_root, **kwargs):
            label = project_root.name
            # Earlier roots dwell LONGER, so completion order is the reverse
            # of root order and an append-as-you-finish implementation would
            # emit proj-05 first.
            await asyncio.sleep(0.01 * (6 - int(label.split('-')[1])))
            return [{'_task_uid': f'{label}/T-1', 'project': label}], _stub_snapshot()

        monkeypatch.setattr(
            'dashboard.data.active_tasks._acquire_and_shape', _stub,
        )

        active, _snapshots = (
            await collect_tasks_with_counts(client=dummy_client, config=config)
        )

        got = [row['project'] for row in active]
        assert got == [f'proj-{i:02d}' for i in range(6)], (
            f'rows came back in {got}, not primary-first ROOT order — the '
            'concurrent walk is appending rows from inside the per-root '
            'coroutines, so completion order (here deliberately the reverse) '
            'leaks into the rendered table'
        )

    async def test_two_roots_sharing_a_basename_both_render(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(g) Per-root results are paired by ROOT, never by display label.

        ``_project_label`` is the directory BASENAME, so two configured roots
        can share one (``/a/proj`` and ``/b/proj``). Keying the gathered
        results by label collapses them: the survivor's rows are extended into
        ``all_active`` TWICE — duplicate ``task_uid``s, which the React tab
        uses as its map key — and the other root's rows vanish with no
        offline or degraded marker naming them. That is silent DATA LOSS, and
        it is the invisible-failure class this whole task exists to close.
        """
        a = tmp_path / 'a' / 'proj'
        b = tmp_path / 'b' / 'proj'
        for root in (a, b):
            root.mkdir(parents=True)
        config = DashboardConfig(project_root=a, known_project_roots=[b])
        _register_runtime(monkeypatch, {})

        async def _stub(client, config_, project_root, **kwargs):
            uid = f'{project_root.parent.name}/{project_root.name}/T-1'
            return [{'_task_uid': uid, 'project': project_root.name}], _stub_snapshot()

        monkeypatch.setattr(
            'dashboard.data.active_tasks._acquire_and_shape', _stub,
        )

        active, snapshots = (
            await collect_tasks_with_counts(client=dummy_client, config=config)
        )
        offline, degraded = _offline(snapshots), _degraded(snapshots)

        uids = [row['_task_uid'] for row in active]
        assert sorted(uids) == ['a/proj/T-1', 'b/proj/T-1'], (
            f'the two same-named roots rendered {uids}; each root must '
            'contribute its OWN rows exactly once. A duplicated uid means one '
            "root's rows were emitted twice under the other's identity, and a "
            'missing one means a root was dropped with nothing naming it'
        )
        assert offline == [] and degraded == [], (
            'both roots answered — neither may be marked offline or degraded'
        )

    async def test_degraded_is_preserved_under_concurrency(
        self, monkeypatch, tmp_path, dummy_client, caplog,
    ):
        """(d) A root the budget never served is degraded, not offline, and has NO count.

        A MIXED partition is the whole point, and the earlier form of this
        test never produced one: it dwelled 0.05 s against a 0.03 s
        per-project budget, so all 9 roots blew their budget, `active` was
        empty and `counts` was `{}` — the three `label not in ...` assertions
        below iterated 9 labels against three EMPTY collections and could not
        fail. The `serve_first` cut makes the partition deterministic: the
        first 3 admissions return without awaiting, the rest hang until their
        own per-project budget ends them.
        """
        import dashboard.data.active_tasks as at_mod

        config = self._n_root_config(tmp_path, 9)
        _register_runtime(monkeypatch, {})
        self._tracking_stub(monkeypatch, serve_first=3)
        monkeypatch.setattr(at_mod, '_TASKS_PER_PROJECT_BUDGET', 0.05)
        # Generous, deliberately: the TOTAL budget expiring is a DIFFERENT
        # branch (tested by the fairness class). What must be exercised here
        # is the per-project expiry.
        monkeypatch.setattr(at_mod, '_TASKS_TOTAL_BUDGET', 5.0)

        with caplog.at_level(logging.WARNING):
            active, snapshots = (
                await collect_tasks_with_counts(client=dummy_client, config=config)
            )
        offline, counts, degraded = (
            _offline(snapshots), _done_counts(snapshots), _degraded(snapshots)
        )

        assert degraded, (
            'no root was reported degraded even though 6 of the 9 roots never '
            'returned — a root the handler never served must be NAMED, or it '
            'renders as "no active work"'
        )
        served = {row['project'] for row in active}
        # VACUITY GUARD, mirroring the fairness class's: the three assertions
        # below compare the degraded labels against `served`/`counts`/`offline`,
        # so all three pass trivially if those are empty.
        assert 0 < len(served) < 9, (
            f'{len(served)} of 9 roots were served — this test asserts a MIXED '
            'partition, and is vacuous unless both halves are non-empty'
        )
        assert counts, 'the served roots must carry done counts, or the count assertions are vacuous'
        assert len(degraded) == 9 - len(served)
        for label in degraded:
            assert label not in served, f'{label} is both degraded and served'
            assert label not in counts, (
                f'{label} timed out but carries a done_count of '
                f'{counts.get(label)!r} — no count was measured, so none may '
                'be fabricated (not even a 0, which renders as a confident '
                '"this project has zero done tasks")'
            )
            assert label not in offline, (
                f'{label} is reported BOTH degraded and offline — the two are '
                'distinct facts: offline means the read demonstrably failed, '
                'degraded means the budget expired so the state is UNKNOWN. '
                'Merging them tells an operator to restart a healthy service.'
            )

    async def test_offline_is_preserved_under_concurrency(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(e) #4795 acceptance 3: a genuinely unreachable root still reports OFFLINE.

        Acceptance 1 (a clean cold render) may not be bought by widening
        budgets until nothing can fail — so the offline marker must survive
        the concurrency change under a budget generous enough that nothing
        degrades.
        """
        config = self._n_root_config(tmp_path, 4)
        _register_runtime(monkeypatch, {})
        self._tracking_stub(monkeypatch, dwell=0.0, offline_for={'proj-02'})

        active, snapshots = (
            await collect_tasks_with_counts(client=dummy_client, config=config)
        )
        offline, counts, degraded = (
            _offline(snapshots), _done_counts(snapshots), _degraded(snapshots)
        )

        assert offline == ['proj-02'], (
            f'expected proj-02 offline, got offline={offline!r} — a fetch that '
            'demonstrably failed must still be reported offline under the '
            'concurrent walk'
        )
        assert 'proj-02' not in degraded, (
            'proj-02 is offline (the read failed), not degraded (the budget '
            'expired) — the concurrent walk must not collapse the two'
        )
        assert 'proj-02' not in counts, (
            'an offline project must contribute no done_count'
        )
        assert {row['project'] for row in active} == {
            'proj-00', 'proj-01', 'proj-03',
        }, 'one offline root must not cost the other three their rows'

    async def test_broad_exception_path_is_preserved_under_concurrency(
        self, monkeypatch, tmp_path, dummy_client, caplog,
    ):
        """(f) A raising root is marked offline and the other eight still render.

        The broad ``except Exception`` must stay INSIDE the per-root unit. If
        it moved out to the gather, one shaping bug would unwind the whole
        walk and 500 the handler — the "one bad root blanks the whole tab"
        failure relocated from the banner to the aggregator.
        """
        config = self._n_root_config(tmp_path, 9)
        _register_runtime(monkeypatch, {})
        self._tracking_stub(monkeypatch, dwell=0.0, raise_for={'proj-04'})

        with caplog.at_level(logging.WARNING):
            active, snapshots = (
                await collect_tasks_with_counts(client=dummy_client, config=config)
            )
        offline, degraded = _offline(snapshots), _degraded(snapshots)

        assert offline == ['proj-04'], (
            f'expected proj-04 offline after it raised, got {offline!r}'
        )
        assert 'proj-04' not in degraded, (
            'a root that RAISED is offline (the read demonstrably failed), '
            'never degraded (which means the budget expired first)'
        )
        assert len(active) == 8, (
            f'only {len(active)} of the 8 healthy roots rendered — one root '
            'raising must not unwind the concurrent gather and take the '
            'others with it; the broad except must stay inside the per-root '
            'coroutine'
        )
        assert any(
            'proj-04' in rec.message or 'proj-04' in str(rec.args)
            for rec in caplog.records
        ), (
            'the raising root was absorbed into an offline marker with no '
            'WARNING — an exception logged as a routine outage is a bug that '
            'renders as an outage forever'
        )


# ---------------------------------------------------------------------------
# workstream C cause 3 (task 4884, #4795): the walk ORDER rotates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCollectTasksWithCountsFairness:
    """A budget that cannot serve every root must not starve the SAME ones.

    Concurrency (cause 2) shrinks the wall clock but does not make the walk
    fair: at 9 roots and a 20 s total, some render will still run out, and
    with a FIXED order the roots that lose are always the last ones. That is
    what the journal shows — ``project solar-challenge-platform: skipped — the
    20.0s Tasks budget was already spent before this project was reached``
    and ``project pump-web-ui: skipped ...``, the same trailing pair, render
    after render. Those two projects were effectively invisible on the Tasks
    tab while the dashboard reported itself healthy.

    Rotation makes the starvation FAIR, not absent. These tests assert the
    fairness, not the absence.
    """

    def _n_root_config(self, tmp_path, n: int) -> DashboardConfig:
        roots = []
        for i in range(n):
            root = tmp_path / f'proj-{i:02d}'
            root.mkdir()
            roots.append(root)
        return DashboardConfig(
            project_root=roots[0], known_project_roots=roots[1:],
        )

    def _admission_recorder(self, monkeypatch, *, dwell: float, serve_first=None):
        """Patch ``_acquire_and_shape`` to record ADMISSION order per call.

        With *serve_first* set to N, the first N admissions recorded in
        ``admissions`` return WITHOUT awaiting at all and every admission
        after them hangs forever.  That makes "which roots this render
        served" a property of the ADMISSION ORDER alone — the thing these
        tests are about — instead of a race between a dwell and a budget.
        Callers using it must clear ``admissions`` between renders.
        """
        admissions: list[str] = []

        async def _stub(client, config, project_root, **kwargs):
            label = project_root.name
            admissions.append(label)
            if serve_first is not None and len(admissions) > serve_first:
                # A wedged MCP leg: never returns, so the caller's own budget
                # is what ends it — and ends it whatever the host's speed.
                await asyncio.Event().wait()
            elif dwell:
                await asyncio.sleep(dwell)
            return [{'_task_uid': f'{label}/T-1', 'project': label}], _stub_snapshot(done=1)

        monkeypatch.setattr(
            'dashboard.data.active_tasks._acquire_and_shape', _stub,
        )
        return admissions

    async def test_no_root_is_systematically_starved(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(a) Across enough consecutive renders, EVERY root gets served.

        This is stronger than "two runs differ": a two-cycle alternation would
        satisfy that and still leave roots 5-9 permanently invisible. The
        assertion is on the UNION over consecutive calls covering all nine.

        The number of calls is DERIVED from the observed rotation stride
        rather than hard-coded. With ``s`` roots served per render and the
        offset advancing by ONE slot per render, the served window is
        contiguous and slides by one, so covering ``N`` roots takes
        ``N - s + 1`` renders — not ``ceil(N / s)``, which would be the count
        for a stride of ``s``. Stride one is the finer-grained rotation and
        the one ``_rotated_project_roots`` implements; deriving the count here
        keeps this test correct if that choice is ever revisited.
        """
        import dashboard.data.active_tasks as at_mod

        config = self._n_root_config(tmp_path, 9)
        _register_runtime(monkeypatch, {})
        # DETERMINISTIC cut, not a wall-clock one. The served count must be a
        # strict subset on EVERY host, so the first three admissions of each
        # render return without awaiting at all and the rest hang until the
        # budget kills them. The earlier form derived the cut from
        # `asyncio.sleep(0.05)` against a 0.16s budget and read 0 of 9 served
        # under xdist contention, tripping the vacuousness guard below.
        admissions = self._admission_recorder(
            monkeypatch, dwell=0.0, serve_first=3,
        )
        # Width 1 so the served subset is a contiguous window of the admission
        # order and the arithmetic above is exact rather than probabilistic.
        monkeypatch.setattr(at_mod, '_TASKS_ROOT_CONCURRENCY', 1)
        # Small enough that the six hung roots cost ~0.3s per render, large
        # enough that no scheduling delay can starve a root that the stub
        # above serves without awaiting. The TOTAL budget is deliberately NOT
        # the constraint here: it is measured from before the runtime fan-out,
        # so tightening it would reintroduce exactly the host-speed dependence
        # this test just removed.
        monkeypatch.setattr(at_mod, '_TASKS_PER_PROJECT_BUDGET', 0.05)
        monkeypatch.setattr(at_mod, '_TASKS_TOTAL_BUDGET', 5.0)
        at_mod._reset_root_rotation()

        async def _one_render() -> set[str]:
            admissions.clear()  # serve_first counts within ONE render
            active, _snapshots = (
                await collect_tasks_with_counts(client=dummy_client, config=config)
            )
            return {row['project'] for row in active}

        first = await _one_render()
        served_per_render = len(first)
        assert 0 < served_per_render < 9, (
            f'the budget served {served_per_render} of 9 roots — this test is '
            'vacuous unless a STRICT subset is served (0 means nothing ran, 9 '
            'means the budget was never the constraint and starvation cannot '
            'be observed at all). Adjust the tightened budgets, not the claim.'
        )

        union = set(first)
        for _ in range(9 - served_per_render):
            union |= await _one_render()

        missing = {f'proj-{i:02d}' for i in range(9)} - union
        assert not missing, (
            f'{sorted(missing)} were never served across '
            f'{9 - served_per_render + 1} consecutive renders while '
            f'{served_per_render} roots were served each time — the walk '
            'order is FIXED, so the same trailing roots starve on every '
            'render. That is the incident behaviour: solar-challenge-platform '
            'and pump-web-ui were skipped render after render while the '
            'dashboard reported itself healthy.'
        )

    async def test_rotation_advances_by_exactly_one_slot_per_call(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(b) The rotation is a DETERMINISTIC round-robin, not randomised.

        Determinism is the point: an operator reading two consecutive renders
        can predict which roots were served, and this test can assert it.
        A random shuffle would also spread the starvation but would make both
        of those impossible.
        """
        import dashboard.data.active_tasks as at_mod

        config = self._n_root_config(tmp_path, 5)
        _register_runtime(monkeypatch, {})
        admissions = self._admission_recorder(monkeypatch, dwell=0.0)
        # Width 1 so admission order is unambiguous rather than a race between
        # concurrently-admitted roots.
        monkeypatch.setattr(at_mod, '_TASKS_ROOT_CONCURRENCY', 1)
        at_mod._reset_root_rotation()

        await collect_tasks_with_counts(client=dummy_client, config=config)
        first = list(admissions)
        admissions.clear()
        await collect_tasks_with_counts(client=dummy_client, config=config)
        second = list(admissions)

        assert len(first) == 5 and len(second) == 5, (
            f'expected all 5 roots admitted in each render, got {first!r} then '
            f'{second!r} — the budget must not be the constraint in this test'
        )
        assert second == first[1:] + first[:1], (
            f'render 2 admitted {second!r}; expected {first[1:] + first[:1]!r} '
            f'— render 1 admitted {first!r} and the offset must advance by '
            'exactly ONE slot, so the order is a predictable round-robin an '
            'operator can reason about across two consecutive renders'
        )

    async def test_all_project_roots_stays_primary_first(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(c) The rotation must NOT leak into the shared root helper.

        ``_all_project_roots`` has other callers that depend on primary-first
        ordering (``app.py``, ``scheduler.py``, ``api/tasks.py``'s root count,
        and ``test_app.py``'s patch
        point). Rotating it in place would silently repoint every one of them
        at a different project — a far larger blast radius than the Tasks tab.
        """
        import dashboard.data.active_tasks as at_mod

        config = self._n_root_config(tmp_path, 5)
        _register_runtime(monkeypatch, {})
        self._admission_recorder(monkeypatch, dwell=0.0)
        at_mod._reset_root_rotation()

        for render in range(4):
            await collect_tasks_with_counts(client=dummy_client, config=config)
            roots = at_mod._all_project_roots(config)
            assert roots[0] == config.project_root, (
                f'after {render + 1} render(s) _all_project_roots returned '
                f'{[r.name for r in roots]} — the primary root is no longer '
                'first, so the rotation has leaked into the shared helper and '
                'every other caller now reads a different project'
            )

    async def test_output_order_is_unaffected_by_rotation(
        self, monkeypatch, tmp_path, dummy_client,
    ):
        """(d) Rotation changes ADMISSION order only, never rendered order.

        If the rotated order reached the payload, the Tasks table would
        reshuffle on every 3 s poll — a fix for invisibility that trades it
        for unreadability.
        """
        import dashboard.data.active_tasks as at_mod

        config = self._n_root_config(tmp_path, 5)
        _register_runtime(monkeypatch, {})
        self._admission_recorder(monkeypatch, dwell=0.0)
        at_mod._reset_root_rotation()

        expected = [f'proj-{i:02d}' for i in range(5)]
        for render in range(4):
            active, _snapshots = (
                await collect_tasks_with_counts(client=dummy_client, config=config)
            )
            got = [row['project'] for row in active]
            assert got == expected, (
                f'render {render + 1} returned rows in {got}, expected '
                f'{expected} — the rendered order must stay canonical '
                'primary-first root order at every rotation offset'
            )
