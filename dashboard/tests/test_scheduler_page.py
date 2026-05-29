"""Tests for dashboard /scheduler page — data layer and HTTP routes.

Step-by-step TDD (matching plan.json):
  step-1   test_module_contention_counts_sorted_desc
  step-3   test_compose_rows_joins_active_tasks_with_snapshot
  step-5   test_skip_event_sparkline_bins / test_skip_event_sparkline_empty
  step-7   test_collect_scheduler_state_happy_path
  step-9   test_collect_scheduler_state_surfaces_offline
  step-11  test_shape_scheduler_envelope
  step-13  test_scheduler_endpoint_returns_envelope_shape
  step-15  test_override_endpoint_rejects_invalid_body
  step-17  test_override_endpoint_proxies_verbatim / _returns_502
  step-19  test_clear_override_endpoint_validation_and_proxy
  step-21  test_reorder_pin_queue_endpoint_validation_and_proxy
  step-23  test_index_html_references_new_jsx_files
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# task-1454: helper — centralises the 6-field scheduler-state snapshot literal
# ---------------------------------------------------------------------------


def _scheduler_snapshot(**overrides):
    """Return a base scheduler-state snapshot dict with the 6 standard fields.

    Canonical base fields (all empty/default):
      skip_counts, parks, effective_priorities, pin_queue, overrides,
      current_holders

    Any kwargs override a matching base key or add an extra key verbatim
    (e.g. ``is_paused``, ``pause_reason``, ``snapshot_at``).

    Each call returns a fresh dict — no cross-test aliasing.
    """
    base = {
        'skip_counts': {},
        'parks': {},
        'effective_priorities': {},
        'pin_queue': [],
        'overrides': {},
        'current_holders': {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# task-1454 step-1: aliasing safety check for _scheduler_snapshot helper
# ---------------------------------------------------------------------------


def test_scheduler_snapshot_returns_independent_dicts():
    """Successive calls return independent dicts; mutating one doesn't affect another."""
    snap1 = _scheduler_snapshot()
    snap2 = _scheduler_snapshot()
    snap1['skip_counts']['1'] = 99
    assert snap2['skip_counts'] == {}, (
        'snap2 was mutated by modifying snap1 — base dict is being aliased'
    )
    snap1['pin_queue'].append('X')
    assert snap2['pin_queue'] == [], (
        'snap2 pin_queue was mutated by modifying snap1 — base list is being aliased'
    )


# ---------------------------------------------------------------------------
# step-1: pure-function test for _module_contention_counts
# ---------------------------------------------------------------------------


def test_module_contention_counts_sorted_desc():
    """_module_contention_counts returns list sorted by contention desc, ties alpha.

    Three tasks with overlapping lock sets {a,b}, {b,c}, {b,d}.
    'b' appears in all three → contention 3.
    'a', 'c', 'd' appear once each → contention 1, ties broken alphabetically.
    Expected order: b(3), a(1), c(1), d(1).
    """
    from dashboard.data.scheduler import _module_contention_counts

    rows = [
        {'task_id': 'T1', 'lock_set': ['a', 'b']},
        {'task_id': 'T2', 'lock_set': ['b', 'c']},
        {'task_id': 'T3', 'lock_set': ['b', 'd']},
    ]
    current_holders = {'b': 'T1'}  # T1 holds 'b'

    result = _module_contention_counts(rows, current_holders)

    assert isinstance(result, list)
    assert len(result) == 4

    # b must be first with contention 3 and the correct holder
    assert result[0]['path'] == 'b'
    assert result[0]['contention'] == 3
    assert result[0]['holder'] == 'T1'

    # a, c, d follow with contention 1 (alphabetical for ties), no holder
    assert [r['path'] for r in result[1:]] == ['a', 'c', 'd']
    assert all(r['contention'] == 1 for r in result[1:])
    assert all(r['holder'] is None for r in result[1:])


# ---------------------------------------------------------------------------
# step-3: _compose_rows joins active tasks with snapshot
# ---------------------------------------------------------------------------


def test_compose_rows_joins_active_tasks_with_snapshot():
    """_compose_rows produces correctly joined rows with priority_differs flag.

    Feed a synthetic active-tasks list (with ``meta_files`` and ``priority``) and a
    synthetic snapshot dict; assert each row has all expected fields, and that
    ``priority_differs=True`` when effective_priority != declared priority.
    """
    from dashboard.data.scheduler import _compose_rows

    now_iso = '2026-01-01T12:00:00+00:00'

    active_tasks = [
        {
            'id': 'proj/T-1',
            'task_id': '1',
            'title': 'Task One',
            'priority': 'high',
            'status': 'in-progress',
            'started': 10,  # minutes
            'meta_files': ['src/a.py', 'src/b.py'],
        },
        {
            'id': 'proj/T-2',
            'task_id': '2',
            'title': 'Task Two',
            'priority': 'medium',
            'status': 'pending',
            'started': 5,
            'meta_files': ['src/c.py'],
        },
    ]

    snapshot = _scheduler_snapshot(
        skip_counts={'1': 3, '2': 0},
        parks={
            '1': {
                'modules': ['src/a.py'],
                'installed_at': now_iso,
            }
        },
        effective_priorities={
            '1': 'critical',  # differs from 'high'
            '2': 'medium',    # same as declared
        },
        overrides={
            '1': {
                'boost_tier': 'critical',
                'pinned': False,
                'reserve_now': True,
                'ttl_until': None,
            }
        },
        current_holders={'src/a.py': '1'},
    )

    rows = _compose_rows(active_tasks, snapshot)

    assert len(rows) == 2

    # Required fields on every row
    required_fields = {
        'task_id', 'title', 'declared_priority', 'effective_priority',
        'priority_differs', 'skip_count', 'park_state', 'age_seconds',
        'lock_set', 'pinned', 'reserve_now', 'boost_tier', 'ttl_until',
    }
    for row in rows:
        assert required_fields <= set(row.keys()), (
            f'Row missing fields: {required_fields - set(row.keys())}'
        )

    # Row 1: priority_differs=True (critical vs high), parked, skip_count=3
    r1 = rows[0]
    assert r1['task_id'] == '1'
    assert r1['title'] == 'Task One'
    assert r1['declared_priority'] == 'high'
    assert r1['effective_priority'] == 'critical'
    assert r1['priority_differs'] is True
    assert r1['skip_count'] == 3
    assert r1['park_state'] is not None
    assert r1['age_seconds'] >= 0        # installed_at = now, so ~0
    assert r1['lock_set'] == ['src/a.py', 'src/b.py']
    assert r1['reserve_now'] is True
    assert r1['boost_tier'] == 'critical'
    assert r1['pinned'] is False

    # Row 2: priority_differs=False (medium == medium), not parked, age from started
    r2 = rows[1]
    assert r2['task_id'] == '2'
    assert r2['declared_priority'] == 'medium'
    assert r2['effective_priority'] == 'medium'
    assert r2['priority_differs'] is False
    assert r2['skip_count'] == 0
    assert r2['park_state'] is None
    assert r2['age_seconds'] == 5 * 60  # 5 minutes → 300 seconds
    assert r2['lock_set'] == ['src/c.py']
    assert r2['pinned'] is False
    assert r2['reserve_now'] is False
    assert r2['boost_tier'] is None


# ---------------------------------------------------------------------------
# step-5: _skip_event_sparkline — binning and empty-history guard
# ---------------------------------------------------------------------------


def test_skip_event_sparkline_bins_into_five_minute_buckets():
    """_skip_event_sparkline bins task_skipped events into 5-minute buckets.

    Setup: 60-minute window starting at ``since``, three task_skipped events
    for T1 placed at minutes 5, 15, and 35.  Expect exactly 12 buckets (one
    per 5-minute slot), the three occupied buckets have count > 0, and a
    different event_type is ignored.
    """
    from dashboard.data.scheduler import _skip_event_sparkline

    since = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    until = since + timedelta(hours=1)  # fixed window → exactly 12 buckets

    def make_event(event_type: str, minutes_offset: float) -> dict:
        ts = since + timedelta(minutes=minutes_offset)
        return {'event_type': event_type, 'task_id': 'T1', 'timestamp': ts.isoformat()}

    events = [
        make_event('task_skipped', 5),    # bucket 1
        make_event('task_skipped', 15),   # bucket 3
        make_event('task_scheduled', 20), # should be ignored (wrong type)
        make_event('task_skipped', 35),   # bucket 7
    ]

    result = _skip_event_sparkline(events, since=since, until=until, bin_seconds=300)

    assert 'labels' in result
    assert 'values' in result
    assert len(result['labels']) == len(result['values'])
    # 60-minute window / 5-minute buckets = exactly 12 buckets
    assert len(result['labels']) == 12

    # Buckets at minutes 5, 15, 35 → bucket indices 1, 3, 7
    assert result['values'][1] == 1   # 5-min mark falls in bucket 1
    assert result['values'][3] == 1   # 15-min mark falls in bucket 3
    assert result['values'][7] == 1   # 35-min mark falls in bucket 7

    # Other buckets are zero
    other_buckets = [i for i in range(12) if i not in (1, 3, 7)]
    assert all(result['values'][i] == 0 for i in other_buckets)

    # Total skipped events counted
    assert sum(result['values']) == 3


def test_skip_event_sparkline_returns_empty_when_no_history():
    """Empty events list → {labels: [], values: []} (no PRNG fallback)."""
    from dashboard.data.scheduler import _skip_event_sparkline

    since = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    result = _skip_event_sparkline([], since=since)

    assert result == {'labels': [], 'values': []}


# ---------------------------------------------------------------------------
# step-7: collect_scheduler_state happy path
# ---------------------------------------------------------------------------


async def test_collect_scheduler_state_happy_path(dummy_client, dummy_config):
    """collect_scheduler_state returns correct shapes and joined fields.

    Patches:
      - mcp_tool_call → returns snapshot then events list
      - collect_active_tasks → returns one synthetic in-progress task
    Asserts the 5-tuple (rows, modules, pin_queue, events_by_task, offline_projects)
    has the right shapes and a couple of joined field values.
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.data.scheduler import collect_scheduler_state

    project = dummy_config.project_root.name  # _project_label result

    snapshot = _scheduler_snapshot(
        skip_counts={'1': 2},
        effective_priorities={'1': 'high'},
        pin_queue=[{'task_id': '1', 'order': 0}],
        overrides={
            '1': {
                'boost_tier': 'high',
                'pinned': True,
                'reserve_now': False,
                'ttl_until': None,
            }
        },
        current_holders={'src/a.py': '1'},
        snapshot_at='2026-01-01T00:00:00+00:00',
    )
    now_iso = datetime.now(UTC).isoformat()
    events = [
        {'event_type': 'task_skipped', 'task_id': '1', 'timestamp': now_iso},
    ]

    active_tasks = [
        {
            'id': f'{project}/T-1',
            'project': project,
            'title': 'Task One',
            'priority': 'medium',
            'status': 'in-progress',
            'started': 5,
            'meta_files': ['src/a.py', 'src/b.py'],
        }
    ]

    mock_mcp = AsyncMock(side_effect=[snapshot, events])
    mock_active = AsyncMock(return_value=(active_tasks, []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        rows, modules, pin_queue, events_by_task, offline_projects, paused_projects = \
            await collect_scheduler_state(dummy_client, dummy_config)

    assert offline_projects == []
    assert len(rows) == 1

    # Joined fields
    assert rows[0]['lock_set'] == ['src/a.py', 'src/b.py']
    assert rows[0]['task_id'] == '1'
    assert rows[0]['pinned'] is True
    assert rows[0]['skip_count'] == 2

    # Module contention: both src/a.py and src/b.py appear once;
    # src/a.py sorts first (alpha tie-break) and carries the holder
    assert len(modules) == 2
    assert modules[0]['holder'] == '1'   # src/a.py is held by task '1'

    assert len(pin_queue) == 1
    assert pin_queue[0]['task_id'] == '1'

    # events_by_task is keyed by composite '{project}/{task_id}' to avoid
    # silent overwrite when the same numeric task_id appears in two project roots.
    composite_key = f'{project}/1'
    assert composite_key in events_by_task
    assert 'labels' in events_by_task[composite_key]
    assert 'values' in events_by_task[composite_key]


# ---------------------------------------------------------------------------
# step-9: collect_scheduler_state surfaces offline when MCP unreachable
# ---------------------------------------------------------------------------


async def test_collect_scheduler_state_surfaces_offline_when_mcp_unreachable(
    dummy_client, dummy_config
):
    """When every fused-memory URL is unreachable, offline_projects is non-empty.

    Mirrors test_collect_active_tasks_surfaces_offline_projects.
    All other tuple members must be empty.
    """
    from unittest.mock import AsyncMock, patch

    import httpx

    from dashboard.data.scheduler import collect_scheduler_state

    project = dummy_config.project_root.name

    active_tasks = [
        {
            'id': f'{project}/T-1',
            'project': project,
            'title': 'Task One',
            'priority': 'medium',
            'status': 'in-progress',
            'started': 5,
            'locks': ['src/a.py'],
        }
    ]

    mock_mcp = AsyncMock(side_effect=httpx.ConnectError('refused'))
    mock_active = AsyncMock(return_value=(active_tasks, []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        rows, modules, pin_queue, events_by_task, offline_projects, paused_projects = \
            await collect_scheduler_state(dummy_client, dummy_config)

    assert rows == []
    assert modules == []
    assert pin_queue == []
    assert events_by_task == {}
    assert offline_projects == [project]


# ---------------------------------------------------------------------------
# step-1370-5: collect_scheduler_state surfaces paused_projects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_scheduler_state_surfaces_paused_projects(
    dummy_client, dummy_config
):
    """collect_scheduler_state returns paused_projects in the 6th tuple element.

    Mirrors test_collect_scheduler_state_surfaces_offline_when_mcp_unreachable.
    A snapshot with is_paused=True must yield paused_projects=[{project, reason}].
    A snapshot with is_paused absent or False must yield paused_projects=[].
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.data.scheduler import collect_scheduler_state

    project = dummy_config.project_root.name
    pause_reason = 'park-stop: 5 tasks parked in 1h'

    # --- Snapshot with is_paused=True ---
    paused_snapshot = _scheduler_snapshot(
        is_paused=True,
        pause_reason=pause_reason,
        snapshot_at='2026-05-15T00:00:00+00:00',
    )

    mock_mcp_paused = AsyncMock(side_effect=[paused_snapshot, []])
    mock_active = AsyncMock(return_value=([], []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp_paused),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        _, _, _, _, _, paused_projects = \
            await collect_scheduler_state(dummy_client, dummy_config)

    assert paused_projects == [{'project': project, 'reason': pause_reason}], (
        f'Expected paused_projects with one entry; got {paused_projects!r}'
    )

    # --- Snapshot with is_paused=False (or absent) must yield empty list ---
    not_paused_snapshot = _scheduler_snapshot(
        is_paused=False,
        pause_reason=None,
        snapshot_at='2026-05-15T00:00:00+00:00',
    )

    mock_mcp_not_paused = AsyncMock(side_effect=[not_paused_snapshot, []])
    mock_active2 = AsyncMock(return_value=([], []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp_not_paused),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active2),
    ):
        _, _, _, _, _, paused_projects_empty = \
            await collect_scheduler_state(dummy_client, dummy_config)

    assert paused_projects_empty == [], (
        f'Expected empty paused_projects when not paused; got {paused_projects_empty!r}'
    )

    # --- Snapshot with is_paused absent (pre-upgrade on-disk) must yield empty list ---
    legacy_snapshot = _scheduler_snapshot(
        snapshot_at='2026-05-15T00:00:00+00:00',
    )

    mock_mcp_legacy = AsyncMock(side_effect=[legacy_snapshot, []])
    mock_active3 = AsyncMock(return_value=([], []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp_legacy),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active3),
    ):
        _, _, _, _, _, paused_projects_legacy = \
            await collect_scheduler_state(dummy_client, dummy_config)

    assert paused_projects_legacy == [], (
        f'Expected empty paused_projects for legacy snapshot; got {paused_projects_legacy!r}'
    )


async def test_collect_scheduler_state_isolates_paused_across_projects(
    dummy_client, tmp_path,
):
    """Mixed two-project fleet: only the paused project appears in paused_projects.

    Project A (p1) is paused; project B (p2) is not.  A single
    collect_scheduler_state call must include A in paused_projects and must NOT
    include B — exercising the per-project conditional append at
    scheduler.py:391-395 with a mixed fleet.
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.config import DashboardConfig
    from dashboard.data.scheduler import collect_scheduler_state

    p1 = tmp_path / 'p1'
    p2 = tmp_path / 'p2'
    p1.mkdir()
    p2.mkdir()

    config = DashboardConfig(project_root=p1, known_project_roots=[p2])

    pause_reason = 'park-stop: 5 tasks parked in 1h'
    snap_A = _scheduler_snapshot(
        is_paused=True,
        pause_reason=pause_reason,
        snapshot_at='2026-05-15T00:00:00+00:00',
    )
    snap_B = _scheduler_snapshot(
        is_paused=False,
        pause_reason=None,
        snapshot_at='2026-05-15T00:00:00+00:00',
    )
    snapshots = {str(p1.resolve()): snap_A, str(p2.resolve()): snap_B}

    async def mock_mcp_call(client, url, tool, args):
        if tool == 'get_scheduler_state':
            return snapshots[args.get('project_root')]
        return []

    mock_active = AsyncMock(return_value=([], []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', side_effect=mock_mcp_call),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        _rows, _modules, _pins, _events, _offline, paused_projects = \
            await collect_scheduler_state(dummy_client, config)

    assert paused_projects == [{'project': p1.name, 'reason': pause_reason}], (
        f'Expected paused_projects with only project A; got {paused_projects!r}'
    )


# ---------------------------------------------------------------------------
# step-11: shape_scheduler envelope
# ---------------------------------------------------------------------------


def test_shape_scheduler_envelope():
    """shape_scheduler wraps inputs in SCHEDULER key with offline and paused flags."""
    from dashboard.data.redux_api import shape_scheduler

    rows = [{'task_id': '1', 'title': 'T1'}]
    modules = [{'path': 'src/a.py', 'holder': '1', 'contention': 1}]
    pin_queue = [{'task_id': '1', 'order': 0}]
    events_by_task = {'1': {'labels': [], 'values': []}}
    snapshot_at = '2026-01-01T00:00:00+00:00'
    paused_entry = {'project': 'proj-a', 'reason': 'park-stop: 3 tasks'}

    # Non-empty offline_projects + non-empty paused_projects
    result_offline = shape_scheduler(
        rows=rows,
        modules=modules,
        pin_queue=pin_queue,
        events_by_task=events_by_task,
        offline_projects=['proj-a'],
        paused_projects=[paused_entry],
        snapshot_at=snapshot_at,
    )
    assert 'SCHEDULER' in result_offline
    inner = result_offline['SCHEDULER']
    assert set(inner.keys()) == {
        'rows', 'modules', 'pin_queue', 'events_by_task',
        'snapshot_at', 'offline', 'offline_projects',
        'paused', 'paused_projects',
    }
    assert inner['offline'] is True
    assert inner['offline_projects'] == ['proj-a']
    assert inner['paused'] is True
    assert inner['paused_projects'] == [paused_entry]
    assert inner['rows'] == rows
    assert inner['modules'] == modules
    assert inner['pin_queue'] == pin_queue
    assert inner['events_by_task'] == events_by_task
    assert inner['snapshot_at'] == snapshot_at

    # Empty offline_projects + empty paused_projects → both False/[]
    result_online = shape_scheduler(
        rows=rows,
        modules=modules,
        pin_queue=pin_queue,
        events_by_task=events_by_task,
        offline_projects=[],
        paused_projects=[],
        snapshot_at=snapshot_at,
    )
    assert result_online['SCHEDULER']['offline'] is False
    assert result_online['SCHEDULER']['paused'] is False
    assert result_online['SCHEDULER']['paused_projects'] == []


def test_shape_scheduler_top_level_lists_are_shallow_copies():
    """shape_scheduler shallow-copies top-level containers so callers can mutate.

    Pins the docstring claim that `list(rows)` / `dict(events_by_task)` are
    fresh containers — appending to the result must not affect the caller's
    inputs.  Inner dicts and sparkline lists remain aliased (caveat in the
    docstring), but that's by design and not under test here.
    """
    from dashboard.data.redux_api import shape_scheduler

    rows = [{'task_id': '1'}]
    modules = [{'path': 'src/a.py'}]
    pin_queue = [{'task_id': '1', 'order': 0}]
    events_by_task = {'1': {'labels': [], 'values': []}}

    paused_projects = [{'project': 'p1', 'reason': 'park-stop: x'}]
    result = shape_scheduler(
        rows=rows,
        modules=modules,
        pin_queue=pin_queue,
        events_by_task=events_by_task,
        offline_projects=[],
        paused_projects=paused_projects,
        snapshot_at=None,
    )
    inner = result['SCHEDULER']

    # Top-level lists/dicts are fresh containers — mutating the result
    # must not alter the caller's originals.
    inner['rows'].append({'task_id': 'sentinel'})
    inner['modules'].append({'path': 'sentinel'})
    inner['pin_queue'].append({'task_id': 'sentinel'})
    inner['events_by_task']['sentinel'] = {'labels': [], 'values': []}
    inner['offline_projects'].append('sentinel')
    inner['paused_projects'].append({'project': 'sentinel', 'reason': None})

    assert rows == [{'task_id': '1'}], 'rows must not be mutated by caller'
    assert modules == [{'path': 'src/a.py'}], 'modules must not be mutated'
    assert pin_queue == [{'task_id': '1', 'order': 0}], 'pin_queue must not be mutated'
    assert 'sentinel' not in events_by_task, 'events_by_task must be a copy'
    assert len(paused_projects) == 1, 'paused_projects must not be mutated by caller'


# ---------------------------------------------------------------------------
# step-13: GET /api/v2/dashboard/scheduler envelope shape
# ---------------------------------------------------------------------------


def test_scheduler_endpoint_returns_envelope_shape(client):
    """GET /scheduler returns 200 with SCHEDULER key containing the inner keys.

    Also pins the empty-state contract: when the collector returns empty tuples,
    the shaped response must have offline=False, rows=[], etc.
    """
    from unittest.mock import AsyncMock, patch

    empty_6tuple = ([], [], [], {}, [], [])
    with patch('dashboard.app.collect_scheduler_state', new=AsyncMock(return_value=empty_6tuple)):
        resp = client.get('/api/v2/dashboard/scheduler')

    assert resp.status_code == 200
    data = resp.json()
    assert 'SCHEDULER' in data
    inner = data['SCHEDULER']
    assert 'rows' in inner
    assert 'modules' in inner
    assert 'pin_queue' in inner
    assert 'events_by_task' in inner
    assert 'snapshot_at' in inner
    assert 'offline' in inner
    assert 'offline_projects' in inner
    assert 'paused' in inner
    assert 'paused_projects' in inner
    # Pin empty-state values so a future shape change doesn't silently regress
    assert inner['offline'] is False
    assert inner['rows'] == []
    assert inner['modules'] == []
    assert inner['pin_queue'] == []
    assert inner['events_by_task'] == {}
    assert inner['snapshot_at'] is None
    assert inner['offline_projects'] == []
    assert inner['paused'] is False
    assert inner['paused_projects'] == []


# ---------------------------------------------------------------------------
# task-1569 step-1: get_scheduler_snapshot — TTL cache and expiry
# ---------------------------------------------------------------------------


async def test_get_scheduler_snapshot_caches_within_ttl_and_refetches_after_expiry(
    dummy_client, dummy_config, monkeypatch
):
    """get_scheduler_snapshot returns cached result within TTL and re-fetches after expiry.

    Part (a) Within TTL: two calls should invoke the inner collector exactly once;
    both calls must return identical (six_tuple, snapshot_at), the six_tuple must
    equal the empty 6-tuple, and snapshot_at must be a non-None ISO-8601 string.

    Part (b) Expiry: after _scheduler_cache_clear() + TTL set to 0.0, two calls
    should each invoke the inner collector (counter == 2).
    """
    from unittest.mock import AsyncMock, patch

    import dashboard.data.scheduler as sched

    empty_6tuple = ([], [], [], {}, [], [])

    # Part (a): within TTL
    sched._scheduler_cache_clear()
    mock_collector = AsyncMock(return_value=empty_6tuple)
    with patch('dashboard.data.scheduler.collect_scheduler_state', new=mock_collector):
        result1 = await sched.get_scheduler_snapshot(dummy_client, dummy_config)
        result2 = await sched.get_scheduler_snapshot(dummy_client, dummy_config)

    assert mock_collector.call_count == 1, (
        f'expected 1 call within TTL, got {mock_collector.call_count}'
    )
    six_tuple1, snapshot_at1 = result1
    six_tuple2, snapshot_at2 = result2
    assert six_tuple1 == empty_6tuple
    assert six_tuple2 == empty_6tuple
    assert result1 == result2, 'both calls must return identical results within TTL'
    assert snapshot_at1 is not None, 'snapshot_at must be non-None'
    # Verify it is a valid ISO-8601 string
    from datetime import datetime as _dt
    _dt.fromisoformat(snapshot_at1)  # raises ValueError if not valid ISO

    # Part (b): expiry — reset cache and TTL
    sched._scheduler_cache_clear()
    monkeypatch.setattr(sched, '_SCHEDULER_TTL_SECONDS', 0.0)
    mock_collector2 = AsyncMock(return_value=empty_6tuple)
    with patch('dashboard.data.scheduler.collect_scheduler_state', new=mock_collector2):
        await sched.get_scheduler_snapshot(dummy_client, dummy_config)
        await sched.get_scheduler_snapshot(dummy_client, dummy_config)

    assert mock_collector2.call_count == 2, (
        f'expected 2 calls after TTL=0.0, got {mock_collector2.call_count}'
    )


# ---------------------------------------------------------------------------
# task-1569 step-3: get_scheduler_snapshot — single-flight concurrency
# ---------------------------------------------------------------------------


async def test_get_scheduler_snapshot_single_flight_collapses_concurrent_misses(
    dummy_client, dummy_config, monkeypatch
):
    """Concurrent cache misses collapse to a single underlying collection.

    Three tasks call get_scheduler_snapshot simultaneously on a cold cache.
    Only one should invoke collect_scheduler_state; the other two should
    await the lock, then return the freshly-filled cache.  All three must
    receive the same snapshot_at.
    """
    import asyncio
    from unittest.mock import patch

    import dashboard.data.scheduler as sched

    sched._scheduler_cache_clear()
    # Disable TTL expiry so the test doesn't race against monotonic time
    monkeypatch.setattr(sched, '_SCHEDULER_TTL_SECONDS', 9999.0)

    empty_6tuple = ([], [], [], {}, [], [])
    counter = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_collector(_client, _config):
        nonlocal counter
        counter += 1
        started.set()
        await release.wait()
        return empty_6tuple

    with patch('dashboard.data.scheduler.collect_scheduler_state', side_effect=slow_collector):
        tasks = [
            asyncio.create_task(sched.get_scheduler_snapshot(dummy_client, dummy_config))
            for _ in range(3)
        ]
        # Wait for the first coroutine to enter the collector, then let the
        # other two reach the lock before releasing.
        await started.wait()
        await asyncio.sleep(0)  # yield to let the other two tasks queue on the lock
        release.set()
        results = await asyncio.gather(*tasks)

    assert counter == 1, (
        f'expected single underlying collection, got counter={counter}'
    )
    snapshot_ats = [r[1] for r in results]
    assert len(set(snapshot_ats)) == 1, (
        f'all callers must receive the same snapshot_at, got {snapshot_ats}'
    )


# ---------------------------------------------------------------------------
# task-1569 step-5: endpoint threads snapshot_at through the envelope
# ---------------------------------------------------------------------------


def test_scheduler_endpoint_threads_snapshot_at_through_envelope(client):
    """GET /scheduler passes the real snapshot_at from get_scheduler_snapshot into the envelope.

    When get_scheduler_snapshot returns a known snapshot_at string, the
    SCHEDULER.snapshot_at field in the response must equal it.  The envelope
    must still contain all expected keys with empty-state values.
    """
    from unittest.mock import AsyncMock, patch

    known_snapshot_at = '2026-05-29T12:00:00+00:00'
    empty_6tuple = ([], [], [], {}, [], [])

    with patch(
        'dashboard.app.get_scheduler_snapshot',
        new=AsyncMock(return_value=(empty_6tuple, known_snapshot_at)),
    ):
        resp = client.get('/api/v2/dashboard/scheduler')

    assert resp.status_code == 200
    data = resp.json()
    assert 'SCHEDULER' in data
    inner = data['SCHEDULER']
    assert inner['snapshot_at'] == known_snapshot_at
    # Verify envelope structure is still intact
    for key in ('rows', 'modules', 'pin_queue', 'events_by_task', 'offline',
                'offline_projects', 'paused', 'paused_projects'):
        assert key in inner, f'missing key {key!r}'
    assert inner['rows'] == []
    assert inner['modules'] == []
    assert inner['offline'] is False
    assert inner['paused'] is False


# ---------------------------------------------------------------------------
# step-15: POST /scheduler/override — validation → 400
# ---------------------------------------------------------------------------

_PATCH_TARGET = 'dashboard.data.memory.mcp_tool_call'

_OVERRIDE_INVALID_BODIES = [
    pytest.param(None, None, id='non-dict-body'),
    pytest.param({}, None, id='missing-task_id'),
    pytest.param({'task_id': 123}, None, id='task_id-not-string'),
    pytest.param({'task_id': ''}, None, id='task_id-empty'),
    # project_root must be present so validation reaches the named branches below
    pytest.param({'task_id': 'T1', 'project_root': '/proj', 'boost_tier': 'invalid_tier'}, 'invalid_boost_tier', id='bad-boost_tier'),
    pytest.param({'task_id': 'T1', 'project_root': '/proj', 'pin_order': 5}, 'invalid_pin_order', id='pin_order-without-pinned'),
    # Type checks: forwarding non-bool/non-int verbatim is inconsistent with strict validation
    pytest.param({'task_id': 'T1', 'project_root': '/proj', 'pinned': 'yes'}, 'invalid_pinned', id='pinned-not-bool'),
    pytest.param({'task_id': 'T1', 'project_root': '/proj', 'reserve_now': 1}, 'invalid_reserve_now', id='reserve_now-not-bool'),
    pytest.param({'task_id': 'T1', 'project_root': '/proj', 'pinned': True, 'pin_order': '5'}, 'invalid_pin_order', id='pin_order-not-int'),
]


@pytest.mark.parametrize('body,expected_error', _OVERRIDE_INVALID_BODIES)
def test_override_endpoint_rejects_invalid_body(client, body, expected_error):
    """Invalid body → 400 with no MCP call.

    For cases that reach named validation branches (boost_tier, pin_order),
    also assert the specific error code so the branch is actually exercised.
    """
    from unittest.mock import AsyncMock, patch

    with patch(_PATCH_TARGET, new=AsyncMock()) as mock_mcp:
        if body is None:
            resp = client.post(
                '/api/v2/dashboard/scheduler/override',
                content=b'not json',
                headers={'Content-Type': 'application/json'},
            )
        else:
            resp = client.post('/api/v2/dashboard/scheduler/override', json=body)

    assert resp.status_code == 400
    # Background monitoring tasks call mcp_tool_call via get_memory_status /
    # get_queue_stats (both defined in memory.py, so they resolve through the
    # patched module attribute).  Build a robust tool-name extractor that handles
    # both positional and kwarg-style invocations, then assert an explicit
    # allowlist: only the two known background tools are permitted; anything else
    # (including 'set_task_priority_override') signals a regression where the
    # override endpoint is invoking MCP before validation completes.
    _BACKGROUND_MCP_TOOLS = frozenset({'get_status', 'get_queue_stats'})

    def _tool_name(c):
        return c.kwargs.get('tool_name') or (c.args[2] if len(c.args) >= 3 else None)

    unexpected = {_tool_name(c) for c in mock_mcp.call_args_list} - _BACKGROUND_MCP_TOOLS - {None}
    assert not unexpected, (
        f'Unexpected MCP tool call(s) on 400 path: {unexpected!r}. '
        f'Full call list: {mock_mcp.call_args_list}'
    )
    if expected_error is not None:
        assert resp.json().get('error') == expected_error


# ---------------------------------------------------------------------------
# step-17: POST /scheduler/override — happy path and 502
# ---------------------------------------------------------------------------


def test_override_endpoint_proxies_verbatim_on_success(client):
    """Valid body → 200 with MCP result forwarded verbatim."""
    from unittest.mock import AsyncMock, patch

    mcp_result = {'status': 'ok', 'task_id': 'T1'}
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/scheduler/override',
            json={'task_id': 'T1', 'project_root': '/proj', 'boost_tier': 'high'},
        )

    assert resp.status_code == 200
    assert resp.json() == mcp_result
    mock_mcp.assert_called_once()
    _client, _url, tool_arg, _args = mock_mcp.call_args.args
    assert tool_arg == 'set_task_priority_override'


def test_override_endpoint_returns_502_when_all_unreachable(client):
    """All URLs unreachable → 502."""
    from unittest.mock import AsyncMock, patch

    import httpx

    with patch(_PATCH_TARGET, new=AsyncMock(side_effect=httpx.ConnectError('refused'))):
        resp = client.post(
            '/api/v2/dashboard/scheduler/override',
            json={'task_id': 'T1', 'project_root': '/proj'},
        )

    assert resp.status_code == 502
    assert resp.json().get('error') == 'fused_memory_unreachable'


# ---------------------------------------------------------------------------
# step-19: POST /scheduler/clear-override — validation, proxy, 404, 502
# ---------------------------------------------------------------------------

_CLEAR_INVALID_BODIES = [
    pytest.param(None, id='non-dict-body'),
    pytest.param({}, id='missing-task_id'),
    pytest.param({'task_id': ''}, id='empty-task_id'),
    pytest.param({'task_id': 'T1', 'fields': 'not-a-list'}, id='fields-not-list'),
    pytest.param({'task_id': 'T1', 'fields': ['bad_field']}, id='invalid-field'),
]


@pytest.mark.parametrize('body', _CLEAR_INVALID_BODIES)
def test_clear_override_rejects_invalid_body(client, body):
    """Invalid body → 400 with no MCP call."""
    from unittest.mock import AsyncMock, patch

    with patch(_PATCH_TARGET, new=AsyncMock()) as mock_mcp:
        if body is None:
            resp = client.post(
                '/api/v2/dashboard/scheduler/clear-override',
                content=b'not json',
                headers={'Content-Type': 'application/json'},
            )
        else:
            resp = client.post('/api/v2/dashboard/scheduler/clear-override', json=body)

    assert resp.status_code == 400
    assert mock_mcp.call_count == 0


def test_clear_override_proxies_verbatim_on_success(client):
    """Valid body → 200, tool name is clear_task_priority_override."""
    from unittest.mock import AsyncMock, patch

    mcp_result = {'status': 'ok', 'task_id': 'T1'}
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/scheduler/clear-override',
            json={'task_id': 'T1', 'project_root': '/proj', 'fields': ['boost_tier']},
        )

    assert resp.status_code == 200
    assert resp.json() == mcp_result
    _client, _url, tool_arg, _args = mock_mcp.call_args.args
    assert tool_arg == 'clear_task_priority_override'


def test_clear_override_returns_404_on_not_found(client):
    """MCP returns not_found → 404 verbatim."""
    from unittest.mock import AsyncMock, patch

    mcp_result = {'error': 'not_found', 'task_id': 'T1'}
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)):
        resp = client.post(
            '/api/v2/dashboard/scheduler/clear-override',
            json={'task_id': 'T1', 'project_root': '/proj'},
        )

    assert resp.status_code == 404
    assert resp.json() == mcp_result


def test_clear_override_returns_502_when_all_unreachable(client):
    """All URLs unreachable → 502."""
    from unittest.mock import AsyncMock, patch

    import httpx

    with patch(_PATCH_TARGET, new=AsyncMock(side_effect=httpx.ConnectError('refused'))):
        resp = client.post(
            '/api/v2/dashboard/scheduler/clear-override',
            json={'task_id': 'T1', 'project_root': '/proj'},
        )

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# step-21: POST /scheduler/reorder-pin-queue — validation, proxy, 502
# ---------------------------------------------------------------------------

_REORDER_INVALID_BODIES = [
    pytest.param(None, id='non-dict-body'),
    pytest.param({}, id='missing-task_ids'),
    pytest.param({'task_ids': 'not-a-list'}, id='task_ids-not-list'),
    pytest.param({'task_ids': ['T1', 'T1'], 'project_root': '/proj'}, id='duplicate-task_ids'),
    # Element-type validation — match the strict checking on the override
    # endpoint so junk values fail fast at the boundary rather than at MCP.
    pytest.param({'task_ids': [1, '1'], 'project_root': '/proj'}, id='task_ids-non-string-element'),
    pytest.param({'task_ids': [None], 'project_root': '/proj'}, id='task_ids-null-element'),
    pytest.param({'task_ids': [''], 'project_root': '/proj'}, id='task_ids-empty-element'),
    pytest.param({'task_ids': [{}, []], 'project_root': '/proj'}, id='task_ids-junk-elements'),
]


@pytest.mark.parametrize('body', _REORDER_INVALID_BODIES)
def test_reorder_pin_queue_rejects_invalid_body(client, body):
    """Invalid body → 400 with no MCP call."""
    from unittest.mock import AsyncMock, patch

    # Patch collect_metrics_snapshot to prevent the background _metrics_loop
    # from calling mcp_tool_call during the request and inflating call_count.
    with patch('dashboard.app.collect_metrics_snapshot', new=AsyncMock()), \
         patch(_PATCH_TARGET, new=AsyncMock()) as mock_mcp:
        if body is None:
            resp = client.post(
                '/api/v2/dashboard/scheduler/reorder-pin-queue',
                content=b'not json',
                headers={'Content-Type': 'application/json'},
            )
        else:
            resp = client.post('/api/v2/dashboard/scheduler/reorder-pin-queue', json=body)

    assert resp.status_code == 400
    assert mock_mcp.call_count == 0


def test_reorder_pin_queue_proxies_verbatim_on_success(client):
    """Valid body → 200, tool name is reorder_pin_queue."""
    from unittest.mock import AsyncMock, patch

    mcp_result = {'status': 'ok'}
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/scheduler/reorder-pin-queue',
            json={'task_ids': ['T1', 'T2'], 'project_root': '/proj'},
        )

    assert resp.status_code == 200
    assert resp.json() == mcp_result
    _client, _url, tool_arg, args = mock_mcp.call_args.args
    assert tool_arg == 'reorder_pin_queue'
    assert args['task_ids'] == ['T1', 'T2']
    assert args['project_root'] == '/proj'


def test_reorder_pin_queue_returns_502_when_all_unreachable(client):
    """All URLs unreachable → 502."""
    from unittest.mock import AsyncMock, patch

    import httpx

    with patch(_PATCH_TARGET, new=AsyncMock(side_effect=httpx.ConnectError('refused'))):
        resp = client.post(
            '/api/v2/dashboard/scheduler/reorder-pin-queue',
            json={'task_ids': ['T1'], 'project_root': '/proj'},
        )

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# step-32: _compose_rows propagates project and project_root fields
# ---------------------------------------------------------------------------


def test_compose_rows_propagates_project_and_project_root():
    """_compose_rows copies project and project_root from input tasks verbatim.

    Without these fields the React drawer's ActionsPanel.submit and
    ActivePinsStrip unpin handlers send empty project_root, causing every
    override/unpin call to 400 with invalid_project_root.
    """
    from dashboard.data.scheduler import _compose_rows

    active_tasks = [
        {
            'task_id': '1',
            'title': 'Task One',
            'priority': 'high',
            'started': 5,
            'locks': ['src/a.py'],
            'project': 'my-project',
            'project_root': '/home/user/projects/my-project',
        },
        {
            'task_id': '2',
            'title': 'Task Two',
            'priority': 'medium',
            'started': 3,
            'locks': [],
            'project': 'other-project',
            'project_root': '/home/user/projects/other-project',
        },
    ]

    snapshot = _scheduler_snapshot()

    rows = _compose_rows(active_tasks, snapshot)

    assert len(rows) == 2
    assert rows[0]['project'] == 'my-project'
    assert rows[0]['project_root'] == '/home/user/projects/my-project'
    assert rows[1]['project'] == 'other-project'
    assert rows[1]['project_root'] == '/home/user/projects/other-project'


# ---------------------------------------------------------------------------
# step-34: collect_scheduler_state enriches active tasks with project_root
# ---------------------------------------------------------------------------


async def test_collect_scheduler_state_enriches_active_tasks_with_project_root(
    dummy_client, tmp_path
):
    """collect_scheduler_state must populate project_root and project on every row.

    Regression: the enrichment loop only added task_id; rows were missing
    project_root and project, causing override/unpin calls to 400 with
    invalid_project_root.
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.config import DashboardConfig
    from dashboard.data.scheduler import collect_scheduler_state

    # Two project roots
    p1 = tmp_path / 'p1'
    p2 = tmp_path / 'p2'
    p1.mkdir()
    p2.mkdir()

    config = DashboardConfig(
        project_root=p1,
        known_project_roots=[p2],
    )

    snapshot = _scheduler_snapshot(
        snapshot_at='2026-01-01T00:00:00+00:00',
    )

    # Track which project_root each get_scheduler_state call received
    recorded_roots: list[str] = []

    async def mock_mcp_call(client, url, tool, args):
        if tool == 'get_scheduler_state':
            recorded_roots.append(args.get('project_root'))
            return snapshot
        # get_scheduler_events returns empty list
        return []

    active_tasks = [
        {
            'id': f'{p1.name}/T-1',
            'project': p1.name,
            'title': 'Task in P1',
            'priority': 'high',
            'status': 'in-progress',
            'started': 5,
            'locks': ['src/a.py'],
        },
        {
            'id': f'{p2.name}/T-2',
            'project': p2.name,
            'title': 'Task in P2',
            'priority': 'medium',
            'status': 'pending',
            'started': 3,
            'locks': [],
        },
    ]

    mock_active = AsyncMock(return_value=(active_tasks, []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', side_effect=mock_mcp_call),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        rows, _, _, _, offline_projects, _ = await collect_scheduler_state(dummy_client, config)

    assert offline_projects == []
    assert len(rows) == 2

    # Each row must carry project and project_root matching the iteration root
    p1_rows = [r for r in rows if r.get('project') == p1.name]
    p2_rows = [r for r in rows if r.get('project') == p2.name]

    assert len(p1_rows) == 1, f'Expected 1 row for p1, got {p1_rows}'
    assert p1_rows[0]['project_root'] == str(p1), (
        f"Expected project_root={str(p1)!r}, got {p1_rows[0].get('project_root')!r}"
    )

    assert len(p2_rows) == 1, f'Expected 1 row for p2, got {p2_rows}'
    assert p2_rows[0]['project_root'] == str(p2), (
        f"Expected project_root={str(p2)!r}, got {p2_rows[0].get('project_root')!r}"
    )


# ---------------------------------------------------------------------------
# step-36: rows support project-filter isolation (mirrors JSX visibleRows)
# ---------------------------------------------------------------------------


def test_compose_rows_supports_project_filter_isolation():
    """Rows returned by _compose_rows can be filtered by project field.

    Pins the data-shape contract supporting the JSX visibleRows filter:
        visibleRows = rows.filter(r => selectedProjects.has(r.project))
    Guards against the silent no-op regression where project is absent.
    """
    from dashboard.data.scheduler import _compose_rows

    active_tasks = [
        {
            'task_id': '1',
            'title': 'A-1',
            'priority': 'high',
            'started': 1,
            'locks': [],
            'project': 'proj-a',
            'project_root': '/a',
        },
        {
            'task_id': '2',
            'title': 'A-2',
            'priority': 'medium',
            'started': 2,
            'locks': [],
            'project': 'proj-a',
            'project_root': '/a',
        },
        {
            'task_id': '3',
            'title': 'B-1',
            'priority': 'low',
            'started': 3,
            'locks': [],
            'project': 'proj-b',
            'project_root': '/b',
        },
    ]

    snapshot = _scheduler_snapshot()

    rows = _compose_rows(active_tasks, snapshot)
    assert len(rows) == 3

    # Python equivalent of JSX visibleRows filter
    proj_a_rows = [r for r in rows if r.get('project') in {'proj-a'}]
    assert len(proj_a_rows) == 2
    assert all(r['project'] == 'proj-a' for r in proj_a_rows)

    proj_b_rows = [r for r in rows if r.get('project') in {'proj-b'}]
    assert len(proj_b_rows) == 1
    assert proj_b_rows[0]['project'] == 'proj-b'
    assert proj_b_rows[0]['task_id'] == '3'


# ---------------------------------------------------------------------------
# step-37: clear-override boundary uses ttl_until (not raw MCP ttl wire name)
# ---------------------------------------------------------------------------


def test_clear_override_whitelist_uses_ttl_until_not_ttl(client):
    """clear-override: fields=['ttl_until'] valid (200); fields=['ttl'] invalid (400).

    The dashboard boundary speaks 'ttl_until' (matching the row shape);
    the MCP wire name 'ttl' must NOT be accepted at the dashboard layer.
    """
    from unittest.mock import AsyncMock, patch

    mcp_result = {'status': 'ok', 'task_id': 'T1'}

    # ttl_until is the canonical dashboard name → accepted
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)):
        resp = client.post(
            '/api/v2/dashboard/scheduler/clear-override',
            json={'task_id': 'T1', 'project_root': '/p', 'fields': ['ttl_until']},
        )
    assert resp.status_code == 200, (
        f"Expected 200 for fields=['ttl_until'], got {resp.status_code}: {resp.text}"
    )

    # 'ttl' is the MCP wire name — it must be rejected at the dashboard boundary
    with patch(_PATCH_TARGET, new=AsyncMock()) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/scheduler/clear-override',
            json={'task_id': 'T1', 'project_root': '/p', 'fields': ['ttl']},
        )
    assert resp.status_code == 400, (
        f"Expected 400 for fields=['ttl'], got {resp.status_code}: {resp.text}"
    )
    assert mock_mcp.call_count == 0


def test_clear_override_translates_ttl_until_to_mcp_ttl(client):
    """Dashboard translates fields=['ttl_until'] → fields=['ttl'] before calling MCP.

    Keeps dashboard boundary speaking 'ttl_until' while the MCP tool
    receives its expected 'ttl' key. Add a comment in the impl so future
    maintainers don't re-invert the mapping.
    """
    from unittest.mock import AsyncMock, patch

    mcp_result = {'status': 'ok', 'task_id': 'T1'}
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/scheduler/clear-override',
            json={'task_id': 'T1', 'project_root': '/p', 'fields': ['ttl_until']},
        )

    assert resp.status_code == 200
    _client, _url, tool_arg, args = mock_mcp.call_args.args
    assert tool_arg == 'clear_task_priority_override'
    # Dashboard must translate 'ttl_until' → 'ttl' before forwarding to MCP
    assert args.get('fields') == ['ttl'], (
        f"Expected MCP fields=['ttl'], got {args.get('fields')!r}"
    )


# ---------------------------------------------------------------------------
# step-39: override endpoint accepts ttl_minutes and translates to ttl_secs
# ---------------------------------------------------------------------------


def test_override_endpoint_accepts_ttl_minutes_and_translates_to_ttl_secs(client):
    """POST /override with ttl_minutes=60 → MCP receives ttl_secs=3600.

    Client sends ttl_minutes (matching the drawer UI input field); dashboard
    translates to MCP's ttl_secs. The user-facing name must NOT leak to MCP.
    Verified against fused-memory tool signature (tools.py:2380).
    """
    from unittest.mock import AsyncMock, patch

    mcp_result = {'status': 'ok', 'task_id': 'T1'}
    with patch(_PATCH_TARGET, new=AsyncMock(return_value=mcp_result)) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/scheduler/override',
            json={
                'task_id': 'T1',
                'project_root': '/proj',
                'pinned': True,
                'ttl_minutes': 60,
            },
        )

    assert resp.status_code == 200
    _client, _url, tool_arg, args = mock_mcp.call_args.args
    assert 'ttl_secs' in args, f'Expected ttl_secs in MCP args, got: {args}'
    assert args['ttl_secs'] == 3600, f'Expected 3600, got {args["ttl_secs"]}'
    assert 'ttl_minutes' not in args, (
        f'ttl_minutes must not leak to MCP, got args={args}'
    )


@pytest.mark.parametrize('bad_ttl', [
    pytest.param(0, id='ttl_minutes-zero'),
    pytest.param(-5, id='ttl_minutes-negative'),
    pytest.param(1441, id='ttl_minutes-too-large'),
    pytest.param('one-hour', id='ttl_minutes-string'),
    pytest.param(None, id='ttl_minutes-null'),
])
def test_override_endpoint_rejects_invalid_ttl_minutes(client, bad_ttl):
    """POST /override with invalid ttl_minutes → 400 invalid_ttl_minutes, no MCP call."""
    from unittest.mock import AsyncMock, patch

    # Patch collect_metrics_snapshot to prevent the background _metrics_loop
    # from calling mcp_tool_call during the request and inflating call_count.
    with patch('dashboard.app.collect_metrics_snapshot', new=AsyncMock()), \
         patch(_PATCH_TARGET, new=AsyncMock()) as mock_mcp:
        resp = client.post(
            '/api/v2/dashboard/scheduler/override',
            json={
                'task_id': 'T1',
                'project_root': '/proj',
                'ttl_minutes': bad_ttl,
            },
        )

    assert resp.status_code == 400, (
        f'Expected 400 for ttl_minutes={bad_ttl!r}, got {resp.status_code}: {resp.text}'
    )
    assert mock_mcp.call_count == 0


# ---------------------------------------------------------------------------
# step-23: index.html references new scheduler JSX files + cache-buster bumped
# ---------------------------------------------------------------------------


def test_index_html_references_new_scheduler_jsx_files(client):
    """index.html must include script tags for the three new scheduler JSX files.

    Also asserts all ?v= cache-busters on /static/redux/* assets share the same
    version number and that number is ≥ 10 (the bump from the previous max of 11).
    """
    import re

    resp = client.get('/static/redux/index.html')
    assert resp.status_code == 200
    body = resp.text

    for jsx in ('tab_scheduler.jsx', 'scheduler_heatmap.jsx', 'scheduler_drawer.jsx'):
        assert jsx in body, f'index.html missing script tag for {jsx}'

    # Extract all ?v=NNN suffixes from /static/redux/ asset URLs
    versions = re.findall(r'/static/redux/[^"\']+\?v=(\d+)', body)
    assert versions, 'No ?v= version strings found in index.html'
    unique = set(versions)
    assert len(unique) == 1, (
        f'Multiple different ?v= versions in index.html: {unique!r}. '
        'All assets must share one version string.'
    )
    version = int(unique.pop())
    # Pin: version must be at least 10 (bumped from previous max of 11)
    assert version >= 10, f'Expected ?v= ≥ 10, got {version}'


# ---------------------------------------------------------------------------
# step-41: multi-project isolation for modules, pins, and event keys
# ---------------------------------------------------------------------------
#
# Regression pins for the review-cycle blockers (esc-1231-85):
#   * _merge_module_lists must key by (project, path) so two projects sharing
#     the same file path don't conflate contention.
#   * pin_queue entries must be tagged with project + project_root so React
#     can deduplicate task_id collisions across projects.
#   * Module entries must carry `holder_project` so the front-end can build
#     the composite '${holder_project}/${holder}' key used to look up
#     events_by_task (otherwise the p50 hint silently never renders).


def test_module_contention_counts_tags_project_and_holder_project():
    """_module_contention_counts attaches project/holder_project when given a project."""
    from dashboard.data.scheduler import _module_contention_counts

    rows = [{'task_id': '1', 'lock_set': ['src/a.py']}]
    result = _module_contention_counts(
        rows, {'src/a.py': '1'}, project='proj-a',
    )

    assert len(result) == 1
    assert result[0]['project'] == 'proj-a'
    # holder_project tracks the project that owns the holder task_id
    assert result[0]['holder_project'] == 'proj-a'

    # When no holder is present, holder_project must be None
    result2 = _module_contention_counts(
        rows, {}, project='proj-a',
    )
    assert result2[0]['project'] == 'proj-a'
    assert result2[0]['holder_project'] is None


def test_holder_project_set_iff_holder_set_jsx_contract():
    """holder_project is non-None iff holder is non-None — JSX filter contract.

    SchedulerTab.visibleModules (tab_scheduler.jsx) filters on:
        m.holder_project && projectFilter.includes(m.holder_project)
    If holder_project is None when a holder IS set, the module would be
    silently hidden when the selected project owns the holder task.  Pin
    the bi-conditional here so the data layer cannot silently break the
    front-end assumption.
    """
    from dashboard.data.scheduler import _module_contention_counts

    rows = [{'task_id': '1', 'lock_set': ['src/a.py']}]
    with_holder = _module_contention_counts(rows, {'src/a.py': '1'}, project='proj-a')
    without_holder = _module_contention_counts(rows, {}, project='proj-a')

    # Bi-conditional: set together or not at all
    assert (with_holder[0]['holder'] is not None) == (with_holder[0]['holder_project'] is not None), (
        "holder_project must be non-None when holder is set"
    )
    assert (without_holder[0]['holder'] is not None) == (without_holder[0]['holder_project'] is not None), (
        "holder_project must be None when no holder is set"
    )
    # Exact values
    assert with_holder[0]['holder_project'] == 'proj-a'
    assert without_holder[0]['holder_project'] is None


def test_merge_module_lists_keys_by_project_and_path():
    """_merge_module_lists must NOT conflate two projects' identically-named modules.

    Regression: same file path under two project roots used to collide into
    one row with summed contention and an arbitrary holder, breaking the
    heatmap (holder task_id matched no row in the other project).
    """
    from dashboard.data.scheduler import _merge_module_lists

    a = [{
        'path': 'src/utils.py',
        'holder': '1',
        'contention': 2,
        'project': 'proj-a',
        'holder_project': 'proj-a',
    }]
    b = [{
        'path': 'src/utils.py',
        'holder': '5',
        'contention': 3,
        'project': 'proj-b',
        'holder_project': 'proj-b',
    }]

    merged = _merge_module_lists(a, b)

    # Two distinct entries — one per project — not a single conflated row
    assert len(merged) == 2

    by_project = {m['project']: m for m in merged}
    assert by_project['proj-a']['contention'] == 2
    assert by_project['proj-a']['holder'] == '1'
    assert by_project['proj-a']['holder_project'] == 'proj-a'
    assert by_project['proj-b']['contention'] == 3
    assert by_project['proj-b']['holder'] == '5'
    assert by_project['proj-b']['holder_project'] == 'proj-b'


async def test_collect_scheduler_state_tags_pins_with_project(dummy_client, tmp_path):
    """pin_queue entries must carry project + project_root after aggregation.

    Without these fields, two projects pinning their own task_id='1' would
    produce duplicate React keys and the reorder handler would send all
    task_ids under a single (arbitrary) project_root.
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.config import DashboardConfig
    from dashboard.data.scheduler import collect_scheduler_state

    p1 = tmp_path / 'p1'
    p2 = tmp_path / 'p2'
    p1.mkdir()
    p2.mkdir()

    config = DashboardConfig(project_root=p1, known_project_roots=[p2])

    snap_p1 = _scheduler_snapshot(
        # Both projects have a pin with task_id='1' — the collision case
        pin_queue=[{'task_id': '1', 'order': 0}],
        snapshot_at='2026-01-01T00:00:00+00:00',
    )
    snap_p2 = _scheduler_snapshot(
        pin_queue=[{'task_id': '1', 'order': 0}],
        snapshot_at='2026-01-01T00:00:00+00:00',
    )

    snapshots = {str(p1): snap_p1, str(p2): snap_p2}

    async def mock_mcp_call(client, url, tool, args):
        if tool == 'get_scheduler_state':
            return snapshots[args.get('project_root')]
        return []

    mock_active = AsyncMock(return_value=([], []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', side_effect=mock_mcp_call),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        _rows, _modules, pin_queue, _events, _offline, _paused = \
            await collect_scheduler_state(dummy_client, config)

    assert len(pin_queue) == 2
    # Each pin carries its owning project + project_root so React can build
    # a composite key and the reorder handler can dispatch per-project.
    projects = sorted(p['project'] for p in pin_queue)
    assert projects == sorted([p1.name, p2.name])
    roots = sorted(p['project_root'] for p in pin_queue)
    assert roots == sorted([str(p1), str(p2)])


async def test_collect_scheduler_state_keeps_module_contention_per_project(
    dummy_client, tmp_path,
):
    """Same module path under two projects must stay as two distinct module rows.

    Regression: _merge_module_lists used to key by path alone, so a file
    named 'src/utils.py' contended in both project A and project B would
    collapse to one row with summed contention and a holder task_id that
    referenced a phantom row from the wrong project.
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.config import DashboardConfig
    from dashboard.data.scheduler import collect_scheduler_state

    p1 = tmp_path / 'p1'
    p2 = tmp_path / 'p2'
    p1.mkdir()
    p2.mkdir()

    config = DashboardConfig(project_root=p1, known_project_roots=[p2])

    snap_p1 = _scheduler_snapshot(
        current_holders={'src/utils.py': '1'},
        snapshot_at='2026-01-01T00:00:00+00:00',
    )
    snap_p2 = _scheduler_snapshot(
        current_holders={'src/utils.py': '5'},
        snapshot_at='2026-01-01T00:00:00+00:00',
    )
    snapshots = {str(p1): snap_p1, str(p2): snap_p2}

    async def mock_mcp_call(client, url, tool, args):
        if tool == 'get_scheduler_state':
            return snapshots[args.get('project_root')]
        return []

    active_tasks = [
        {
            'id': f'{p1.name}/T-1',
            'project': p1.name,
            'title': 'P1 task',
            'priority': 'high',
            'status': 'in-progress',
            'started': 1,
            'meta_files': ['src/utils.py'],
        },
        {
            'id': f'{p2.name}/T-5',
            'project': p2.name,
            'title': 'P2 task',
            'priority': 'high',
            'status': 'in-progress',
            'started': 1,
            'meta_files': ['src/utils.py'],
        },
    ]
    mock_active = AsyncMock(return_value=(active_tasks, []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', side_effect=mock_mcp_call),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        _rows, modules, _pins, _events, _offline, _paused = \
            await collect_scheduler_state(dummy_client, config)

    # Two project-scoped entries, NOT one conflated row
    assert len(modules) == 2
    by_project = {m['project']: m for m in modules}
    # Each project's contention count stays at 1 (only that project's task
    # has the module in its lock_set) — not the conflated 2.
    assert by_project[p1.name]['contention'] == 1
    assert by_project[p1.name]['holder'] == '1'
    assert by_project[p1.name]['holder_project'] == p1.name
    assert by_project[p2.name]['contention'] == 1
    assert by_project[p2.name]['holder'] == '5'
    assert by_project[p2.name]['holder_project'] == p2.name


# ---------------------------------------------------------------------------
# step-42: defensive boundary fixes (esc-1231-86 triage)
# ---------------------------------------------------------------------------


def test_scheduler_proxy_handles_non_dict_mcp_result(client):
    """_scheduler_proxy: a non-dict MCP result must not 500 on `.get('error')`.

    Older/buggy MCP tools could return a list or None.  The not_found→404
    mapping must guard with isinstance(result, dict) so the unexpected type
    is forwarded verbatim as 200 rather than escaping as AttributeError/500.
    """
    from unittest.mock import AsyncMock, patch

    with patch(_PATCH_TARGET, new=AsyncMock(return_value=['unexpected', 'list'])):
        resp = client.post(
            '/api/v2/dashboard/scheduler/clear-override',
            json={'task_id': 'T1', 'project_root': '/proj'},
        )

    # Forwarded verbatim — the list reaches JSONResponse without crashing.
    assert resp.status_code == 200
    assert resp.json() == ['unexpected', 'list']


def test_compose_rows_tolerates_non_string_installed_at():
    """park_state.installed_at coerced to str so non-string values don't AttributeError.

    Regression guard: if a buggy producer writes installed_at as a number or
    dict, age_seconds must fall back to 0 rather than raising AttributeError
    on `.replace(...)`.
    """
    from dashboard.data.scheduler import _compose_rows

    active_tasks = [{
        'task_id': '1',
        'title': 'T',
        'priority': 'high',
        'started': 0,
        'locks': [],
    }]

    # Number for installed_at — used to crash on .replace(...)
    snapshot_num = _scheduler_snapshot(
        parks={'1': {'installed_at': 12345}},
    )
    rows_num = _compose_rows(active_tasks, snapshot_num)
    assert rows_num[0]['age_seconds'] == 0

    # Dict for installed_at — same crash path
    snapshot_dict = _scheduler_snapshot(
        parks={'1': {'installed_at': {'nested': 'oops'}}},
    )
    rows_dict = _compose_rows(active_tasks, snapshot_dict)
    assert rows_dict[0]['age_seconds'] == 0


async def test_collect_scheduler_state_normalises_non_dict_snapshot(
    dummy_client, dummy_config,
):
    """A non-dict MCP get_scheduler_state result must be normalised to {} (online-but-empty).

    Without normalisation, downstream `.get('current_holders')` would
    AttributeError and propagate as exception → caller would never see the
    project.  Treating as online-but-empty matches the documented contract:
    snapshot={} means quiescent, snapshot=None means offline.
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.data.scheduler import collect_scheduler_state

    project = dummy_config.project_root.name

    # MCP returns a list (buggy/older server) instead of a dict
    mock_mcp = AsyncMock(side_effect=[['unexpected'], []])
    mock_active = AsyncMock(return_value=([], []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        rows, modules, pin_queue, events, offline, _paused = \
            await collect_scheduler_state(dummy_client, dummy_config)

    # The project is treated as online-but-empty, NOT offline.
    assert offline == [], f'expected online treatment, got offline={offline}'
    assert rows == []
    assert modules == []
    assert pin_queue == []
    # No project should be misclassified as offline either
    assert project not in offline


# ---------------------------------------------------------------------------
# task-dashboard-lock-alignment: lock_set is normalized to the scheduler's
# lock_depth from meta_files, and holders resolve via the shared prefix rule.
# These deep-path cases would FAIL on main, where lock_set was the raw
# plan.json file paths and the holder lookup was a dict-equality miss.
# ---------------------------------------------------------------------------


def test_compose_rows_normalizes_meta_files_to_module_locks():
    """_compose_rows derives lock_set from meta_files normalized to lock_depth.

    A deep file path (orchestrator/src/orchestrator/scheduler.py) must collapse
    to the depth-2 module key 'orchestrator/src' — matching the snapshot's
    current_holders keying.  On main, lock_set kept the raw file path and never
    matched a normalized holder key.
    """
    from dashboard.data.scheduler import _compose_rows

    active_tasks = [
        {
            'task_id': '1',
            'title': 'Deep task',
            'priority': 'high',
            'started': 1,
            # Raw plan.json paths (file granularity) — must be ignored in favour
            # of meta_files, the scheduler's footprint source.
            'locks': ['orchestrator/src/orchestrator/scheduler.py'],
            'meta_files': [
                'orchestrator/src/orchestrator/scheduler.py',
                'orchestrator/src/orchestrator/workflow.py',
            ],
        }
    ]

    rows = _compose_rows(active_tasks, _scheduler_snapshot(), depth=2)

    assert rows[0]['lock_set'] == ['orchestrator/src'], (
        f"Expected normalized module lock, got {rows[0]['lock_set']!r}"
    )


def test_compose_rows_with_no_meta_files_produces_empty_lock_set():
    """Tasks without meta_files produce an empty lock_set.

    The scheduler derives module locks exclusively from meta_files (taskmaster
    ``metadata.files``).  A task whose taskmaster metadata carries no file
    footprint holds no scheduler locks and correctly produces lock_set=[].
    Such tasks are invisible to the contention view, which is correct — the
    scheduler would not block on them.
    """
    from dashboard.data.scheduler import _compose_rows

    active_tasks = [
        {
            'task_id': '1',
            'title': 'No metadata',
            'priority': 'high',
            'started': 1,
            'meta_files': [],
        }
    ]

    rows = _compose_rows(active_tasks, _scheduler_snapshot(), depth=2)
    assert rows[0]['lock_set'] == []


def test_compose_rows_deep_path_holder_resolves_in_module_contention():
    """Deep-path footprint + holder keyed at lock_depth → module shows the holder.

    End-to-end of the alignment fix: a task touching
    orchestrator/src/orchestrator/scheduler.py normalizes to 'orchestrator/src',
    and a snapshot holding 'orchestrator/src' resolves as that module's holder.
    On main this returned None (raw path vs normalized key mismatch).
    """
    from dashboard.data.scheduler import _compose_rows, _module_contention_counts

    active_tasks = [
        {
            'task_id': '7',
            'title': 'Deep holder',
            'priority': 'high',
            'started': 1,
            'meta_files': ['orchestrator/src/orchestrator/scheduler.py'],
        }
    ]
    snapshot = _scheduler_snapshot(current_holders={'orchestrator/src': '7'})

    rows = _compose_rows(active_tasks, snapshot, depth=2)
    modules = _module_contention_counts(rows, snapshot['current_holders'], project='orch')

    assert len(modules) == 1
    assert modules[0]['path'] == 'orchestrator/src'
    assert modules[0]['holder'] == '7', (
        f"Expected holder '7' via normalized key match, got {modules[0]['holder']!r}"
    )


def test_module_contention_counts_resolves_sub_lock_depth_parent_holder():
    """A holder keyed at a sub-lock_depth parent matches a deeper child module.

    The scheduler's hierarchical lock means a holder on 'foo' (one component)
    conflicts with a child module 'foo/bar'.  The dashboard must surface that
    holder on the child module's row via the shared prefix rule, not equality.
    """
    from dashboard.data.scheduler import _module_contention_counts

    rows = [{'task_id': '1', 'lock_set': ['foo/bar']}]
    # Holder is keyed at the parent 'foo' (fewer than lock_depth components).
    result = _module_contention_counts(rows, {'foo': 'T-9'}, project='p')

    assert len(result) == 1
    assert result[0]['path'] == 'foo/bar'
    assert result[0]['holder'] == 'T-9', (
        'sub-lock_depth parent holder must match child module via prefix rule'
    )


async def test_collect_scheduler_state_uses_meta_files_for_deep_path(
    dummy_client, dummy_config,
):
    """End-to-end: collect_scheduler_state normalizes meta_files using snapshot lock_depth.

    A snapshot carrying lock_depth=2 + a holder at 'orchestrator/src', joined
    with an active task whose meta_files is a deep path, must produce a row
    lock_set of ['orchestrator/src'] and a module whose holder resolves.
    """
    from unittest.mock import AsyncMock, patch

    from dashboard.data.scheduler import collect_scheduler_state

    project = dummy_config.project_root.name

    snapshot = _scheduler_snapshot(
        current_holders={'orchestrator/src': '1'},
        lock_depth=2,
        snapshot_at='2026-01-01T00:00:00+00:00',
    )

    active_tasks = [
        {
            'id': f'{project}/T-1',
            'project': project,
            'title': 'Deep task',
            'priority': 'high',
            'status': 'in-progress',
            'started': 5,
            'locks': ['orchestrator/src/orchestrator/scheduler.py'],
            'meta_files': ['orchestrator/src/orchestrator/scheduler.py'],
        }
    ]

    mock_mcp = AsyncMock(side_effect=[snapshot, []])
    mock_active = AsyncMock(return_value=(active_tasks, []))

    with (
        patch('dashboard.data.scheduler.mcp_tool_call', mock_mcp),
        patch('dashboard.data.scheduler.collect_active_tasks', mock_active),
    ):
        rows, modules, _pins, _events, offline, _paused = \
            await collect_scheduler_state(dummy_client, dummy_config)

    assert offline == []
    assert len(rows) == 1
    assert rows[0]['lock_set'] == ['orchestrator/src']
    assert len(modules) == 1
    assert modules[0]['path'] == 'orchestrator/src'
    assert modules[0]['holder'] == '1'
