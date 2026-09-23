"""Integration tests for the redux dashboard app.

These tests focus on the route-level contract: the SPA index, the static
asset mount, and the JSON API under ``/api/v2/dashboard/*``.  Most endpoints
read from per-project DBs that don't exist in the temp fixture, so they are
expected to return empty-but-well-formed JSON — the assertion is on the
response *shape*, not the contents, which exercises the shape adapters.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dashboard.api.window import _parse_window
from dashboard.data import redux_api

# ---------------------------------------------------------------------------
# _parse_window helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('value, expected', [
    ('24h', 1),
    ('7d', 7),
    ('30d', 30),
    ('all', 3650),
    ('weird', 30),  # default
    (None, 30),
])
def test_parse_window_known_and_unknown(value, expected):
    query_params: dict[str, str] = {}
    if value is not None:
        query_params['window'] = value
    assert _parse_window(query_params) == expected


# ---------------------------------------------------------------------------
# SPA + health
# ---------------------------------------------------------------------------


def test_index_serves_redux_html(client):
    resp = client.get('/')
    assert resp.status_code == 200
    body = resp.text
    assert '<div id="root">' in body
    assert '/static/redux/app.jsx' in body


def test_static_redux_assets_load(client):
    for name in ('app.jsx', 'data.js', 'shell.jsx', 'tabs.jsx', 'styles.css'):
        resp = client.get(f'/static/redux/{name}')
        assert resp.status_code == 200, f'expected 200 for /static/redux/{name}'


def test_health_endpoint(client):
    resp = client.get('/api/health')
    assert resp.status_code == 200
    assert resp.json() == {'status': 'ok'}


# ---------------------------------------------------------------------------
# JSON API: shape contracts
# ---------------------------------------------------------------------------


def test_orchestrators_returns_orchestrators_and_projects(client):
    """Even with no running orchestrators the response carries every key."""
    with patch(
        'dashboard.api.orchestrators.discover_orchestrators',
        new=AsyncMock(return_value=[]),
    ):
        resp = client.get('/api/v2/dashboard/orchestrators')
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        'ORCHESTRATORS', 'PROJECTS', 'ORCHESTRATORS_SPARK', 'served_at',
    }
    assert isinstance(body['ORCHESTRATORS'], list)
    assert isinstance(body['PROJECTS'], list)
    assert isinstance(body['ORCHESTRATORS_SPARK'], dict)
    assert 'labels' in body['ORCHESTRATORS_SPARK']
    assert 'values' in body['ORCHESTRATORS_SPARK']


def test_orchestrators_payload_is_stamped_and_carries_no_task_summary(client):
    """The endpoint says when it looked, and claims no task count at all.

    ``summary`` used to be the whole reason discovery fetched every root's
    task tree. The tree is gone, so the key must be gone too — an all-zero
    summary shaped from an absent one would read as a measured "no tasks",
    which is the fabricated zero this PRD exists to remove.
    """
    from datetime import datetime

    raw = [{
        'pids': [4321],
        'prd': '/proj/dark-factory/prd.md',
        'label': '/proj/dark-factory/prd.md',
        'project_root': '/proj/dark-factory',
        'running': True,
        'started': 'Mar18',
    }]
    with patch(
        'dashboard.api.orchestrators.discover_orchestrators',
        new=AsyncMock(return_value=raw),
    ):
        resp = client.get('/api/v2/dashboard/orchestrators')

    assert resp.status_code == 200
    body = resp.json()
    served_at = datetime.fromisoformat(body['served_at'])
    assert served_at.utcoffset() is not None, (
        'served_at must name one moment, not a local-clock reading'
    )
    [orch] = body['ORCHESTRATORS']
    assert 'summary' not in orch, f'no task count is measured here: {orch}'
    assert orch['pids'] == [4321]
    assert orch['project'] == 'dark-factory'


def _snapshot(*, health='ok', done=0):
    """A ``TaskSnapshot`` in the state *health* names, as the collector emits one.

    Built through the real ``Datum``/``build_census`` types rather than a mock,
    because the handler validates every emitted datum against the payload's own
    ``served_at`` — a mock would sail past exactly the contract these tests are
    about. Stamped at the live clock so a FRESH half is inside its own
    freshness bound at the instant the handler serves it.
    """
    from datetime import UTC, datetime
    from types import MappingProxyType

    from dashboard.data.census import build_census
    from dashboard.data.datum import Datum, DatumState
    from dashboard.data.task_snapshot import (
        FRESHNESS_BOUND_SECONDS,
        SnapshotFailure,
        TaskSnapshot,
    )

    as_of = datetime.now(UTC)
    status_map = {task_id: 'done' for task_id in range(1, done + 1)}
    fresh_rows = Datum([], as_of, DatumState.FRESH, None, FRESHNESS_BOUND_SECONDS)
    fresh_census = Datum(
        build_census(status_map), as_of, DatumState.FRESH, None,
        FRESHNESS_BOUND_SECONDS,
    )
    unknown = Datum(
        None, None, DatumState.UNKNOWN, f'canned {health} root',
        FRESHNESS_BOUND_SECONDS,
    )
    failure, rows, census = {
        'ok': (SnapshotFailure.NONE, fresh_rows, fresh_census),
        'offline': (SnapshotFailure.UNREACHABLE, unknown, unknown),
        'degraded': (SnapshotFailure.BUDGET, unknown, unknown),
        'count_unknown': (SnapshotFailure.NONE, fresh_rows, unknown),
    }[health]
    return TaskSnapshot(
        census=census, rows=rows,
        in_progress_live=None if rows is unknown else 0,
        in_progress_stranded=None if rows is unknown else 0,
        skew_seconds=None if unknown in (rows, census) else 0,
        status_map=MappingProxyType(status_map), failure=failure,
    )


def _snapshots(labels, *, offline=(), degraded=(), count_unknown=(), done=None):
    """``{label: TaskSnapshot}`` for *labels*, in order, routed by membership."""
    counts = done or {}
    health = dict.fromkeys(offline, 'offline')
    health.update(dict.fromkeys(degraded, 'degraded'))
    health.update(dict.fromkeys(count_unknown, 'count_unknown'))
    return {
        label: _snapshot(health=health.get(label, 'ok'), done=counts.get(label, 0))
        for label in labels
    }


_TASKS_KEYS = {
    'ACTIVE_TASKS', 'TASKS_SNAPSHOT', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS',
    'TASKS_DEGRADED_PROJECTS', 'TASKS_COUNT_UNKNOWN_PROJECTS',
    'TASKS_PROJECT_COUNT', 'served_at',
}
"""The default render's whole payload. DONE_COUNTS is gone, and its ABSENCE is
asserted rather than an empty dict: ``data.js::applyKey`` returns early on a
missing key, so the client keeps its seeded default, while ``{}`` would read as
a measured "no project has any done tasks"."""


def test_tasks_endpoint_omits_file_locks_and_returns_active_only(client):
    with patch(
        'dashboard.api.tasks.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], {})),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == _TASKS_KEYS
    assert 'FILE_LOCKS' not in body
    assert 'DONE_COUNTS' not in body
    assert isinstance(body['ACTIVE_TASKS'], list)
    assert body['TASKS_OFFLINE'] is False
    assert body['TASKS_OFFLINE_PROJECTS'] == []
    assert body['TASKS_SNAPSHOT'] == {}


def test_tasks_endpoint_carries_a_validated_census_per_root(client):
    """The server, not the client, owns the count — and says when it measured it.

    Replaces ``test_tasks_endpoint_includes_done_counts``: the fact that test
    protected (a per-project done count comes from the SERVER) is unchanged,
    but it now travels inside a ``Datum`` that says how fresh it is, so a
    client can tell a measured zero from a failed read.
    """
    from datetime import datetime

    from dashboard.data.datum import Datum, DatumState, validate_datum

    with patch(
        'dashboard.api.tasks.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], _snapshots(['dark-factory'], done={'dark-factory': 7}))),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()

    served_at = datetime.fromisoformat(body['served_at'])
    assert served_at.utcoffset() is not None, (
        'served_at must name one moment, not a local-clock reading'
    )

    entry = body['TASKS_SNAPSHOT']['dark-factory']
    assert set(entry) == {
        'census', 'rows', 'in_progress_live', 'in_progress_stranded',
        'skew_seconds',
    }
    census = entry['census']
    assert census['state'] == 'fresh'
    assert census['value']['counts']['done'] == 7
    assert census['value']['total'] == 7

    # Every emitted datum must satisfy the envelope's contract against the
    # payload's OWN served_at — the user-observable signal this leaf delivers.
    for key in ('census', 'rows'):
        half = entry[key]
        as_of = datetime.fromisoformat(half['as_of'])
        assert as_of <= served_at, f'{key} claims to be measured after it was served'
        validate_datum(
            Datum(
                half['value'], as_of, DatumState(half['state']), half['reason'],
                half['freshness_bound_seconds'],
            ),
            served_at,
        )


def test_tasks_default_render_issues_no_terminal_fetch(client):
    """No ``?terminal=``, no terminal row — asserted at the MCP wire.

    The default render's whole point is that its cost stops growing with the
    terminal tree, so "the handler discards the done rows afterwards" would be
    precisely the defect.
    """
    calls: list[dict] = []

    async def _mcp(http_client, url, tool, args, **_kw):
        calls.append({'tool': tool, 'args': dict(args)})
        if tool == 'get_statuses':
            return {'statuses': {}}
        return {'tasks': []}

    with patch('dashboard.data.tasks.mcp_tool_call', new=_mcp):
        resp = client.get('/api/v2/dashboard/tasks')

    assert resp.status_code == 200
    for call in calls:
        requested = call['args'].get('statuses') or []
        assert 'done' not in requested and 'cancelled' not in requested, (
            f'the default render must ask for no terminal row: {call}'
        )


def _render_through_the_real_collector(client, canned):
    """GET /api/v2/dashboard/tasks with only ``mcp_tool_call`` canned, on a cold unit.

    Everything above the substrate is real: the handler, the collector,
    ``acquire_snapshot`` and its cache. The unit cache starts EMPTY because a
    cache miss is the render that measures during the request, which is the
    render production serves every 15 s.
    """
    import dashboard.data.task_snapshot as snapshot_mod
    import dashboard.data.tasks as tasks_mod

    snapshot_mod._snapshot_cache_clear()
    tasks_mod._fetch_tasks_cache_clear()
    try:
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            return client.get('/api/v2/dashboard/tasks')
    finally:
        snapshot_mod._snapshot_cache_clear()
        tasks_mod._fetch_tasks_cache_clear()


def test_a_healthy_root_is_fresh_when_the_real_collector_measures_it(client, caplog):
    """``served_at`` must be the instant the COLLECTOR stamps, not merely the one it is checked against.

    Drives the REAL ``collect_tasks_with_counts``: only
    ``dashboard.data.tasks.mcp_tool_call`` is replaced, by the paging-aware
    ``CannedMCP``, so every ``Datum`` on this payload is measured DURING the
    request, the way production measures it. The root list is the app's own
    configured one, which is what the collector fans out over. Patching only
    the handler's root enumerator would give the handler and the collector two
    different populations.

    WHY THE REST OF THIS FILE CANNOT CATCH THIS. ``_snapshot`` stamps ``as_of``
    at ``datetime.now(UTC)`` BEFORE the request, so every fixture datum is a
    little OLDER than the handler's ``served_at``. That inverts production's
    ordering. In production the handler resolved ``served_at``, and the
    collector then read the clock again and stamped every fresh half
    microseconds AFTER it. ``validate_datum`` refuses a negative age, so
    ``_validated`` routed every project to ``unknown`` on every cache-miss
    render, and 350 green dashboard tests sat over that handler.
    ``dashboard/src/dashboard/data/scheduler.py::collect_scheduler_state`` and
    ``collect_active_tasks`` already forward ``now``; ``api_tasks`` was the only
    call site that did not.
    """
    import logging
    from datetime import datetime

    from test_task_snapshot import CannedMCP, _raw_row

    label = client.app.state.config.project_root.name
    pairs = ((1, 'in-progress'), (2, 'pending'), (3, 'done'))
    canned = CannedMCP(
        rows=[_raw_row(task_id, status) for task_id, status in pairs],
        status_map=dict(pairs),
        status_page_size=2000,
    )
    with caplog.at_level(logging.WARNING):
        resp = _render_through_the_real_collector(client, canned)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    entry = body['TASKS_SNAPSHOT'][label]
    # (a) Both halves were measured, and the payload says so.
    assert entry['census']['state'] == 'fresh', entry['census']
    assert entry['rows']['state'] == 'fresh', entry['rows']
    # (b) A healthy tree is named in no banner.
    assert body['TASKS_COUNT_UNKNOWN_PROJECTS'] == []
    assert body['TASKS_OFFLINE_PROJECTS'] == []
    # (c) Nothing on the payload claims to have been measured after it was served.
    served_at = datetime.fromisoformat(body['served_at'])
    for project, wire in body['TASKS_SNAPSHOT'].items():
        for half in ('census', 'rows'):
            as_of = wire[half]['as_of']
            assert as_of is not None, f'{project}.{half} was never measured: {wire[half]}'
            assert datetime.fromisoformat(as_of) <= served_at, (
                f'{project}.{half} claims as_of={as_of}, after served_at={body["served_at"]}'
            )
    # (d) The fallback that reports a contract break never fired.
    broken = [
        record.getMessage() for record in caplog.records
        if 'this is a BUG, not an outage' in record.getMessage()
    ]
    assert broken == []


def test_a_unit_another_render_refreshed_later_is_not_a_contract_break(client, caplog):
    """Two renders share one unit cache, and the later-resolved one can refresh first.

    Render B resolves its instant, then render A resolves a later one and
    refreshes the root before B reaches it. B is then served A's unit, whose
    ``as_of`` is AFTER the instant B resolved. B's ``served_at`` must still
    postdate it: a negative age is a contract break, and it used to route a
    healthy root to TASKS_OFFLINE_PROJECTS.
    """
    import logging
    from datetime import datetime, timedelta

    from test_task_snapshot import CannedMCP, _raw_row

    import dashboard.data.task_snapshot as snapshot_mod
    import dashboard.data.tasks as tasks_mod
    from dashboard.data.utils import resolve_now

    config = client.app.state.config
    root = config.project_root
    pairs = ((1, 'in-progress'), (2, 'pending'), (3, 'done'))
    canned = CannedMCP(
        rows=[_raw_row(task_id, status) for task_id, status in pairs],
        status_map=dict(pairs),
        status_page_size=2000,
    )
    instants = iter(())

    def b_clock(now):
        return next(instants, None) or resolve_now(now)

    snapshot_mod._snapshot_cache_clear()
    tasks_mod._fetch_tasks_cache_clear()
    try:
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            render_a = client.get('/api/v2/dashboard/tasks')
            assert render_a.status_code == 200, render_a.text
            a_as_of = datetime.fromisoformat(
                render_a.json()['TASKS_SNAPSHOT'][root.name]['rows']['as_of']
            )
            render_b = a_as_of - timedelta(seconds=1)
            instants = iter([render_b])
            with (
                patch('dashboard.api.tasks.resolve_now', new=b_clock),
                caplog.at_level(logging.WARNING),
            ):
                resp = client.get('/api/v2/dashboard/tasks')
    finally:
        snapshot_mod._snapshot_cache_clear()
        tasks_mod._fetch_tasks_cache_clear()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    label = root.name
    assert body['TASKS_OFFLINE_PROJECTS'] == []
    entry = body['TASKS_SNAPSHOT'][label]
    assert entry['rows']['state'] == 'fresh', entry['rows']
    assert entry['census']['state'] == 'fresh', entry['census']
    assert datetime.fromisoformat(entry['rows']['as_of']) == a_as_of
    assert datetime.fromisoformat(body['served_at']) >= a_as_of
    assert not [
        record for record in caplog.records
        if 'this is a BUG, not an outage' in record.getMessage()
    ]


def _bug_warnings(caplog):
    return [
        record.getMessage() for record in caplog.records
        if 'this is a BUG, not an outage' in record.getMessage()
    ]


def test_a_snapshot_that_breaks_the_envelope_borrows_no_last_good(client, caplog):
    """A root whose unit breaks the contract is unknown on BOTH halves, by construction.

    The stand-in used to come from ``unmeasured_snapshot(label, ...)``, whose
    first argument keys the ROOT-keyed last-good store. A label never matched a
    real root's key, so the stand-in came out unknown only because the two key
    spaces happened not to meet. Here they meet: a last good sits under the
    very key the label spells. The stand-in must still borrow nothing, because
    no half of a unit that broke its own envelope can be trusted.
    """
    import asyncio
    import logging
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    import httpx
    from test_task_snapshot import CannedMCP, _raw_row

    import dashboard.data.task_snapshot as snapshot_mod
    from dashboard.data.datum import DatumInvariant

    label = 'p0'
    config = client.app.state.config
    canned = CannedMCP(
        rows=[_raw_row(1, 'in-progress')], status_map={1: 'in-progress', 2: 'done'},
        status_page_size=2000,
    )

    async def _measure_earlier():
        async with httpx.AsyncClient() as http_client:
            await snapshot_mod.acquire_snapshot(
                http_client, config, label, now=datetime.now(UTC) - timedelta(seconds=10),
            )

    healthy = _snapshot()
    broken = replace(
        healthy,
        census=replace(healthy.census, as_of=datetime.now(UTC) + timedelta(hours=1)),
    )
    snapshot_mod._snapshot_cache_clear()
    try:
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            asyncio.run(_measure_earlier())
        with patch(
            'dashboard.api.tasks.collect_tasks_with_counts',
            new=AsyncMock(return_value=([], {label: broken})),
        ), patch(
            'dashboard.api.tasks._all_project_roots',
            new=lambda config: [Path('/proj') / label],
        ), caplog.at_level(logging.WARNING):
            resp = client.get('/api/v2/dashboard/tasks')
    finally:
        snapshot_mod._snapshot_cache_clear()

    assert resp.status_code == 200, resp.text
    entry = resp.json()['TASKS_SNAPSHOT'][label]
    for half in ('census', 'rows'):
        assert entry[half]['state'] == 'unknown', entry[half]
        assert entry[half]['value'] is None and entry[half]['as_of'] is None
    assert DatumInvariant.FRESHNESS_BOUND.value in entry['census']['reason']
    assert (entry['in_progress_live'], entry['in_progress_stranded']) == (None, None)
    assert entry['skew_seconds'] is None
    assert len(_bug_warnings(caplog)) == 1, _bug_warnings(caplog)


def test_a_terminal_window_that_breaks_the_envelope_is_unknown_not_a_500(client, caplog):
    """The window degrades to unknown by the same rule as a root's snapshot.

    A ``lower_bound`` datum with no reason breaks the envelope. The key still
    answers, as an explained unknown, and the rest of the payload survives.
    """
    import logging
    from datetime import UTC, datetime
    from pathlib import Path

    from dashboard.data.datum import Datum, DatumInvariant, DatumState
    from dashboard.data.task_snapshot import FRESHNESS_BOUND_SECONDS

    label = 'p0'
    unexplained = Datum(
        [], datetime.now(UTC), DatumState.LOWER_BOUND, None, FRESHNESS_BOUND_SECONDS,
    )
    with patch(
        'dashboard.api.tasks.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], _snapshots([label]))),
    ), patch(
        'dashboard.api.tasks._all_project_roots',
        new=lambda config: [Path('/proj') / label],
    ), patch(
        'dashboard.api.tasks.acquire_terminal_window',
        new=AsyncMock(return_value=unexplained),
    ), caplog.at_level(logging.WARNING):
        resp = client.get(f'/api/v2/dashboard/tasks?terminal={label}')

    assert resp.status_code == 200, resp.text
    body = resp.json()
    window = body[f'TASKS_TERMINAL:{label}']
    assert window['state'] == 'unknown', window
    assert window['value'] is None and window['as_of'] is None
    assert DatumInvariant.REASON_REQUIRED.value in window['reason']
    assert body['TASKS_SNAPSHOT'][label]['census']['state'] == 'fresh'
    assert len(_bug_warnings(caplog)) == 1, _bug_warnings(caplog)


_TASK_ROW_KEYS = frozenset({
    'id', 'project', 'title', 'description', 'details', 'status', 'agent',
    'loops', 'attempts', 'lane', 'phase', 'lane_state', 'runtime_offline',
    'runtime_status', 'claimant_run_id', 'heartbeat_at', 'stranded',
    'meta_files', 'train', 'external_deps', 'prd', 'started', 'deps',
})
"""One active ``TaskRow``'s keys: ``_build_task_row``'s plus ``started`` and ``deps``.

A literal rather than a derivation, so that adding a field to the wire is a
deliberate edit here.
"""

_RAW_ONLY_KEYS = frozenset({'priority', 'dependencies', 'metadata', 'updated_at'})
"""The raw ``_shape_task`` fields the shaper narrows away, ``metadata`` the heaviest."""


def test_the_snapshot_rows_on_the_wire_are_the_shaped_task_rows(client):
    """``TASKS_SNAPSHOT[p].rows`` carries the rows ``ACTIVE_TASKS`` carries, not raw MCP rows.

    ``Datum.to_wire()`` renders ``value`` verbatim. So the unit's RAW rows used
    to ship beside the shaped ones: every active row twice per render, the
    second copy with the whole ``metadata`` blob, on the endpoint this leaf
    exists to shrink. That also broke the PRD's declared
    ``Datum[list[TaskRow]]``. With one configured root, ``ACTIVE_TASKS`` is
    exactly that root's rows, so the two exposures must be equal.
    """
    from test_task_snapshot import CannedMCP, _raw_row

    label = client.app.state.config.project_root.name
    pairs = ((1, 'in-progress'), (2, 'pending'), (3, 'blocked'))
    canned = CannedMCP(
        rows=[
            _raw_row(task_id, status, metadata={'files': ['a.py'], 'prd': 'plans/x.md'})
            for task_id, status in pairs
        ],
        status_map=dict(pairs),
        status_page_size=2000,
    )

    resp = _render_through_the_real_collector(client, canned)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    wire_rows = body['TASKS_SNAPSHOT'][label]['rows']['value']
    assert len(wire_rows) == len(pairs)
    for row in wire_rows:
        assert set(row) == _TASK_ROW_KEYS, sorted(set(row) ^ _TASK_ROW_KEYS)
        assert not set(row) & _RAW_ONLY_KEYS
    assert body['ACTIVE_TASKS'] == wire_rows


def test_tasks_surfaces_offline_marker_when_mcp_unreachable(client):
    """When every root's read demonstrably failed, the payload sets ``offline=True``."""
    with patch(
        'dashboard.api.tasks.collect_tasks_with_counts',
        new=AsyncMock(return_value=([], _snapshots(['dark-factory'], offline=['dark-factory']))),
    ), patch(
        'dashboard.api.tasks._all_project_roots',
        new=lambda config: _fake_roots(1),
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    body = resp.json()
    assert body['TASKS_OFFLINE'] is True
    assert body['TASKS_OFFLINE_PROJECTS'] == ['dark-factory']


def test_tasks_endpoint_passes_resolve_external_true_and_forwards_external_deps(client):
    """api_tasks must call collect_tasks_with_counts with resolve_external=True
    and forward the external_deps field in ACTIVE_TASKS rows unchanged.

    Asserts:
    (a) collect_tasks_with_counts is called with resolve_external=True
    (b) ACTIVE_TASKS[0]['external_deps'] contains the resolved dep
    (c) Top-level key set is unchanged (non-breaking)
    """
    mock_row = {
        'id': 'dark-factory/T-5',
        'project': 'dark-factory',
        'title': 'waits on upstream',
        'status': 'pending',
        'external_deps': [{'id': 'dark_factory:13', 'status': 'done'}],
    }
    mock = AsyncMock(return_value=([mock_row], {}))

    with patch('dashboard.api.tasks.collect_tasks_with_counts', new=mock):
        resp = client.get('/api/v2/dashboard/tasks')

    assert resp.status_code == 200
    body = resp.json()

    # (a) resolve_external=True was passed
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs.get('resolve_external') is True, (
        f'expected resolve_external=True in call_args.kwargs, got: {call_kwargs}'
    )

    # (b) external_deps passes through unmodified
    assert body['ACTIVE_TASKS'][0]['external_deps'] == [
        {'id': 'dark_factory:13', 'status': 'done'}
    ]

    # (c) top-level key set unchanged
    assert set(body) == _TASKS_KEYS


def test_tasks_endpoint_asks_for_no_terminal_cap(client):
    """The handler passes no render cap, because there is no cap left to pass.

    Replaces ``test_tasks_endpoint_passes_max_cancelled_per_project``: the
    per-bucket caps retired with the default-render terminal fetch, so the
    contract to protect is now the ABSENCE of those arguments — passing one
    would be the handler re-imposing a server-side render policy on a request
    that no longer asks for terminal rows at all.
    """
    mock = AsyncMock(return_value=([], {}))

    with patch('dashboard.api.tasks.collect_tasks_with_counts', new=mock):
        resp = client.get('/api/v2/dashboard/tasks')

    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert 'max_done_per_project' not in call_kwargs
    assert 'max_cancelled_per_project' not in call_kwargs
    assert call_kwargs.get('resolve_external') is True
    assert set(resp.json()) == _TASKS_KEYS


# ---------------------------------------------------------------------------
# /api/v2/dashboard/tasks — honest offline signal (task 3857 steps 15/16)
# ---------------------------------------------------------------------------


def _fake_roots(n):
    """N distinct project roots, for patching ``app._all_project_roots``.

    The route-level ``client`` fixture runs against a single isolated temp
    root, so "k of N failed with k < N" is not expressible against the real
    config at all. Patching the root enumerator is the seam ``api_tasks``
    itself uses to learn N, so a test that patches it is asserting on the same
    fact the handler reads — not on a parallel fixture that could drift.
    """
    from pathlib import Path

    return [Path(f'/proj/p{i}') for i in range(n)]


def _tasks_body(
    client, *, offline_projects, degraded_projects=(),
    count_unknown_projects=(), total_roots,
):
    """GET /api/v2/dashboard/tasks with a canned collector result and N roots.

    The three lists are no longer RETURNED by the collector — they are derived
    from the per-root units — so the fixture builds the units that must yield
    them. A test that could only state the lists directly would be asserting
    the handler copies a list; this one asserts it reads a state machine.
    """
    labels = sorted(
        {*offline_projects, *degraded_projects, *count_unknown_projects}
        | {f'p{i}' for i in range(total_roots)}
    )
    collector = AsyncMock(
        return_value=(
            [],
            _snapshots(
                labels,
                offline=offline_projects,
                degraded=degraded_projects,
                count_unknown=count_unknown_projects,
            ),
        )
    )
    with patch('dashboard.api.tasks.collect_tasks_with_counts', new=collector), patch(
        'dashboard.api.tasks._all_project_roots', new=lambda config: _fake_roots(total_roots)
    ):
        resp = client.get('/api/v2/dashboard/tasks')
    assert resp.status_code == 200
    return resp.json()


def test_tasks_partial_outage_does_not_set_the_global_offline_flag(client):
    """One failing root out of nine must not claim a total outage.

    THE headline defect: ``TASKS_OFFLINE`` is ``bool(offline_projects)``, so a
    single unreachable root raises a global "fused-memory offline — task data
    unavailable" banner directly above eight other projects' rows, which are
    on the wire and fine. The per-project list already carries the honest
    fact; the boolean's job is the DIFFERENT fact of a total outage.
    """
    body = _tasks_body(client, offline_projects=['p3'], total_roots=9)

    assert body['TASKS_OFFLINE'] is False, (
        '1 of 9 roots failing is not a fused-memory outage — the other 8 '
        "projects' rows are in this very payload"
    )
    assert body['TASKS_OFFLINE_PROJECTS'] == ['p3']


def test_tasks_all_roots_offline_sets_the_global_flag(client):
    """Every configured root failing IS the outage signal.

    One fused-memory URL serves every root, so all-roots-failed is the
    observable proxy for "every configured fused-memory URL is unreachable" —
    the state the global banner's copy actually describes.
    """
    body = _tasks_body(
        client, offline_projects=['p0', 'p1', 'p2'], total_roots=3,
    )

    assert body['TASKS_OFFLINE'] is True
    assert body['TASKS_OFFLINE_PROJECTS'] == ['p0', 'p1', 'p2']


def test_tasks_no_offline_roots_leaves_the_global_flag_false(client):
    body = _tasks_body(client, offline_projects=[], total_roots=4)

    assert body['TASKS_OFFLINE'] is False
    assert body['TASKS_OFFLINE_PROJECTS'] == []
    assert body['TASKS_DEGRADED_PROJECTS'] == []


def test_tasks_degraded_projects_surface_without_claiming_offline(client):
    """Budget expiry is its own fact on the wire, and never an outage claim.

    A project the handler ran out of budget for was never proven unreachable —
    its state is UNKNOWN. Folding it into ``TASKS_OFFLINE`` (or into
    ``TASKS_OFFLINE_PROJECTS``) would report a healthy fused-memory as down.
    """
    body = _tasks_body(
        client,
        offline_projects=[],
        degraded_projects=['p2', 'p3'],
        total_roots=4,
    )

    assert body['TASKS_DEGRADED_PROJECTS'] == ['p2', 'p3']
    assert body['TASKS_OFFLINE'] is False, (
        'a budget expiry is not a demonstrated outage'
    )
    assert body['TASKS_OFFLINE_PROJECTS'] == []


def test_tasks_all_roots_degraded_is_still_not_an_outage(client):
    """Even ALL roots degrading is not an outage — nothing was proven down.

    Guards the fix against over-correcting into "any total failure sets the
    flag": the global banner's copy says fused-memory is unreachable, a claim
    a timeout does not license.
    """
    body = _tasks_body(
        client,
        offline_projects=[],
        degraded_projects=['p0', 'p1'],
        total_roots=2,
    )

    assert body['TASKS_OFFLINE'] is False
    assert body['TASKS_DEGRADED_PROJECTS'] == ['p0', 'p1']


def test_tasks_payload_carries_the_root_count_the_banner_denominates_with(client):
    """The "k of N" denominator must come from the SAME population as its k.

    The partial-outage notice reads "k of N projects". ``k`` counts entries in
    ``TASKS_OFFLINE_PROJECTS`` — task project roots, from
    ``active_tasks._all_project_roots``. ``N`` was
    ``(DF_T.PROJECTS || []).length``, the ORCHESTRATOR-derived project list
    from /api/v2/dashboard/orchestrators — a different population. A root with
    no orchestrator, or an orchestrator with no task root, makes them diverge,
    and the notice then prints "3 of 5" with nine roots actually configured.
    ``countPhrase``'s ``n >= k`` guard prevents an outright "1 of 0" but
    licenses every understatement below it.

    The handler already computes exactly the right N — it needs it to decide
    ``TASKS_OFFLINE`` at all — and then threw it away. Putting it on the wire
    means numerator and denominator come from one enumerator, by construction.
    """
    body = _tasks_body(client, offline_projects=['p3'], total_roots=9)

    assert body['TASKS_PROJECT_COUNT'] == 9, (
        'the banner denominator must be the root count the handler itself '
        f'fanned out over, got {body.get("TASKS_PROJECT_COUNT")!r}'
    )
    # The fact it denominates, in the same payload — so a future change that
    # decouples them fails here rather than in a screenshot.
    assert len(body['TASKS_OFFLINE_PROJECTS']) <= body['TASKS_PROJECT_COUNT']
    assert body['TASKS_OFFLINE'] is False


def test_tasks_project_count_is_zero_for_a_degenerate_no_roots_config(client):
    """No configured roots is 0, not a crash and not a fabricated 1."""
    body = _tasks_body(client, offline_projects=[], total_roots=0)

    assert body['TASKS_PROJECT_COUNT'] == 0
    assert body['TASKS_OFFLINE'] is False, (
        'nothing configured to fail is not an outage'
    )


def test_tasks_payload_keeps_file_locks_out_and_carries_the_banner_facts(client):
    """The payload carries every banner fact, and FILE_LOCKS stays gone."""
    body = _tasks_body(client, offline_projects=[], total_roots=1)

    assert set(body) == _TASKS_KEYS
    assert 'FILE_LOCKS' not in body
    assert 'DONE_COUNTS' not in body, (
        'an absent key leaves the client on its seeded default; an empty dict '
        'would read as a measured "no project has any done tasks"'
    )


# ---------------------------------------------------------------------------
# /api/v2/dashboard/tasks?terminal=<project> — the on-demand window (task 5587)
# ---------------------------------------------------------------------------


def _terminal_row(task_id, status='done'):
    """One raw MCP ``get_tasks`` row in a terminal status, distinguishable by id."""
    return {
        'id': task_id,
        'title': f'task {task_id}',
        'status': status,
        'dependencies': [],
        'metadata': {},
        'updatedAt': f'2026-01-0{task_id % 9 + 1}T00:00:00+00:00',
    }


def _terminal_render(
    client, monkeypatch, *, terminal, labels=('dark-factory',),
    terminal_rows=(), done=None, window=None, count_unknown=(),
):
    """``GET /tasks?terminal=<terminal>`` over a canned substrate; ``(body, calls)``.

    The collector and the root enumerator are canned so the CENSUS the window
    is positioned from is known exactly — the window's offset is a function of
    the terminal population, so a test that cannot state that population
    cannot check the offset. ``mcp_tool_call`` is left REAL underneath, so the
    terminal read is asserted at the wire: "the handler discards the rows
    afterwards" is precisely the defect, and which rows the server was asked
    for is observable nowhere else.
    """
    from pathlib import Path

    import dashboard.data.task_snapshot as snapshot_mod
    import dashboard.data.tasks as tasks_mod

    if window is not None:
        monkeypatch.setattr(snapshot_mod, '_TERMINAL_FETCH_WINDOW', window)
    tasks_mod._fetch_tasks_cache_clear()

    roots = [Path('/proj') / label for label in labels]
    rendered = {str(root) for root in roots}
    calls: list[dict] = []

    async def _mcp(http_client, url, tool, args, **kwargs):
        # Only THIS render's roots are logged. The lifespan's burndown loop
        # snapshots the real configured roots through this same patched seam
        # as the app starts, and its reads landed here whenever the two
        # overlapped (measured: 5 of 10 runs of this class failed on them).
        # Nothing but the handler can reach a /proj root: only the patched
        # enumerator below hands one out.
        if args.get('project_root') in rendered:
            # kwargs is kept: the per-request budget rides as ``timeout=`` and
            # is assertable at the wire nowhere else.
            calls.append({'tool': tool, 'args': dict(args), 'kwargs': dict(kwargs)})
        if tool == 'get_statuses':
            return {'statuses': {}}
        rows = sorted(terminal_rows, key=lambda row: int(row['id']))  # ORDER BY id ASC
        statuses = args.get('statuses')
        if statuses is not None:
            rows = [row for row in rows if row.get('status') in statuses]
        page_size = args.get('page_size')
        if page_size is not None:
            start = args.get('offset') or 0
            rows = rows[start:start + page_size]
        return {'tasks': rows}

    snapshots = _snapshots(labels, count_unknown=count_unknown, done=done)
    try:
        with patch(
            'dashboard.api.tasks.collect_tasks_with_counts',
            new=AsyncMock(return_value=([], snapshots)),
        ), patch(
            'dashboard.api.tasks._all_project_roots',
            new=lambda config: list(roots),
        ), patch('dashboard.data.tasks.mcp_tool_call', new=_mcp):
            resp = client.get(f'/api/v2/dashboard/tasks?terminal={terminal}')
    finally:
        tasks_mod._fetch_tasks_cache_clear()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # data.js::ON_DEMAND_KEYS.terminal.key names the body key AND the DF_DATA
    # key, and refreshOne applies body[key] verbatim, so the window must answer
    # under exactly one flat per-project key. A nested TASKS_TERMINAL map would
    # pass every read below and never reach the browser.
    assert _terminal_keys(body) == [f'TASKS_TERMINAL:{terminal}'], (
        'the window must answer under the ONE flat key the client reads, and '
        f'under no other terminal key; got {_terminal_keys(body)}'
    )
    return body, calls


def _terminal_keys(body):
    """Every key of *body* naming a terminal window, bare or per-project."""
    return sorted(key for key in body if key.startswith('TASKS_TERMINAL'))


def _terminal_get_tasks(calls):
    return [call for call in calls if call['tool'] == 'get_tasks']


def _node():
    """The ``node`` binary: a skip locally, a FAILURE under CI.

    The ``test_chip_label_disambiguation.py::_node`` idiom — CI carries node,
    so its absence there is a toolchain regression, not an optional extra.
    """
    import os
    import shutil

    path = shutil.which('node')
    if not path:
        if os.environ.get('CI'):
            pytest.fail('node is required in CI but not found on PATH')
        pytest.skip('node not available')
    return path


# The REAL client's on-demand path, run over a body handed in on stdin. The
# three scripts load in index.html's order because each destructures the one
# before it at module scope, and `document` stays undefined on purpose: data.js
# starts polling only in a real browser document, so without one it loads
# inert (data_poll.test.mjs::loadDataJs records why).
_ON_DEMAND_CLIENT = r"""
const fs = require('fs');
const path = require('path');
const [redux, project] = process.argv.slice(1);
const body = JSON.parse(fs.readFileSync(0, 'utf8'));
globalThis.window = { dispatchEvent() {} };
require(path.join(redux, 'endpoint_staleness.js'));
require(path.join(redux, 'datum.js'));
const api = require(path.join(redux, 'data.js'));
api.requestOnDemand('terminal', project, {
  state: api.createPollState(),
  deps: { fetchImpl: () => Promise.resolve({ ok: true, json: async () => body }), now: () => 1 },
}).then(outcome => {
  const datum = api.datumFor(`TASKS_TERMINAL:${project}`);
  process.stdout.write(JSON.stringify({ outcome, datum }));
});
"""


def _request_on_demand(body, project):
    """``(outcome, datum)`` from data.js's REAL ``requestOnDemand`` over *body*."""
    import json
    import subprocess
    from pathlib import Path

    redux = Path(__file__).parent.parent / 'src' / 'dashboard' / 'static' / 'redux'
    result = subprocess.run(
        [_node(), '-e', _ON_DEMAND_CLIENT, str(redux), project],
        input=json.dumps(body), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f'the client driver exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    applied = json.loads(result.stdout)
    return applied['outcome'], applied['datum']


class TestTerminalWindow:
    """``?terminal=<project>`` — the only render that pays for terminal rows.

    The default render stopped fetching them, so every fact the retired
    ``TestShapeOneProjectNarrowing`` terminal cases protected — the window
    reaches the HIGH-id end, it goes through ``fetch_task_page``, truncation
    WARNS rather than capping silently, and an unpositionable window is not
    fetched at all — has to hold HERE or it holds nowhere.
    """

    def test_the_window_is_a_lower_bound_datum_naming_the_bound(
        self, client, monkeypatch
    ):
        """(a) The disclosure IS the state: a windowed read under-reports, and says so.

        The window size is parsed OUT of ``reason`` and compared to the module
        constant rather than matched against a fixed string, so a reworded
        message cannot silently change the bound the payload discloses.
        """
        import re
        from datetime import datetime

        from dashboard.data.task_snapshot import _TERMINAL_FETCH_WINDOW

        body, _calls = _terminal_render(
            client, monkeypatch, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
            done={'dark-factory': 10},
        )

        entry = body['TASKS_TERMINAL:dark-factory']
        assert entry['state'] == 'lower_bound', (
            'a windowed read is a measured value known to under-report — the '
            f'state is the disclosure, got {entry["state"]!r}'
        )
        assert len(entry['value']) == 10
        assert all(row['started'] == 0 for row in entry['value']), (
            'a finished task has no elapsed runtime to report'
        )
        assert all(row['completed'] for row in entry['value']), (
            'a terminal row carries the completion instant the substrate offers'
        )
        as_of = datetime.fromisoformat(entry['as_of'])
        assert as_of.utcoffset() is not None, (
            'as_of must name one moment, not a local-clock reading'
        )
        assert as_of <= datetime.fromisoformat(body['served_at'])
        assert _TERMINAL_FETCH_WINDOW in [
            int(number) for number in re.findall(r'\d+', entry['reason'] or '')
        ], (
            'the reason must disclose the window the rows were read under, so '
            'a consumer can tell "these are all of them" from "these are the '
            f'newest {_TERMINAL_FETCH_WINDOW}"; got {entry["reason"]!r}'
        )

    def test_the_window_reaches_the_high_id_end(self, client, monkeypatch):
        """(b) ``offset`` must select the NEWEST terminal rows, not the oldest.

        Relocated from ``TestShapeOneProjectNarrowing``. ``page_size``/
        ``offset`` slice an ASCENDING-id list, so a naive ``offset=0`` returns
        the oldest terminal rows — the opposite of what the tab renders.
        """
        body, calls = _terminal_render(
            client, monkeypatch, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
            done={'dark-factory': 10}, window=3,
        )

        page = _terminal_get_tasks(calls)
        assert len(page) == 1, f'exactly one windowed read, got {page}'
        assert page[0]['args'].get('statuses') == ['cancelled', 'done']
        assert page[0]['args'].get('page_size') == 3
        assert page[0]['args'].get('offset') == 7, 'max(0, n_terminal - window)'

        emitted = sorted(
            int(row['id'].rsplit('T-', 1)[-1])
            for row in body['TASKS_TERMINAL:dark-factory']['value']
        )
        assert emitted == [107, 108, 109], (
            f'the window must reach the high-id end, got {emitted}'
        )

    def test_the_terminal_window_goes_through_fetch_task_page(
        self, client, monkeypatch
    ):
        """(b) The read that wants a PARTIAL answer names itself one.

        Relocated from ``TestShapeOneProjectNarrowing``. After task 5018 the
        partial-answer intent lives in the function name rather than in an
        argument combination, so a reader cannot mistake this for a whole-tree
        read — and ``fetch_tasks``, the whole-set read, must not be reached at
        all on this path.
        """
        from shared.task_statuses import TERMINAL

        import dashboard.data.task_snapshot as snapshot_mod

        whole: list[dict] = []
        paged: list[dict] = []

        async def _whole_set(client_, config, project_root, **kwargs):
            whole.append(kwargs)
            return []

        async def _one_page(client_, config, project_root, **kwargs):
            paged.append(kwargs)
            return []

        monkeypatch.setattr(snapshot_mod, 'fetch_tasks', _whole_set)
        monkeypatch.setattr(snapshot_mod, 'fetch_task_page', _one_page)

        _terminal_render(
            client, monkeypatch, terminal='dark-factory',
            done={'dark-factory': 10},
        )

        assert len(paged) == 1, f'exactly one windowed read, got {paged}'
        assert paged[0]['statuses'] == sorted(TERMINAL)
        assert paged[0]['page_size'] == snapshot_mod._TERMINAL_FETCH_WINDOW
        assert paged[0]['offset'] == max(
            0, 10 - snapshot_mod._TERMINAL_FETCH_WINDOW
        )
        assert whole == [], (
            'the terminal window is a PARTIAL read and must not borrow the '
            f'whole-set read to make it, got {whole}'
        )

    def test_no_truncation_warning_when_the_population_fits(
        self, client, monkeypatch, caplog
    ):
        """(c) n_terminal <= window → offset 0, every terminal row, no WARNING."""
        import logging

        body, calls = _terminal_render(
            client, monkeypatch, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 106)],
            done={'dark-factory': 6}, window=10,
        )

        assert _terminal_get_tasks(calls)[0]['args'].get('offset') == 0
        assert len(body['TASKS_TERMINAL:dark-factory']['value']) == 6
        assert not [
            record for record in caplog.records
            if record.name == 'dashboard.data.task_snapshot'
            and record.levelno >= logging.WARNING
        ], 'no truncation WARNING may fire when the population fits the window'

    def test_truncation_warns_naming_project_and_counts(
        self, client, monkeypatch, caplog
    ):
        """(c) n_terminal > window → a WARNING naming project, count and window.

        Relocated from ``TestShapeOneProjectNarrowing``. No silent cap: the
        window selects by descending id while a reader expects recency, so
        when the gap can bite it has to be visible in the log.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger='dashboard.data.task_snapshot'):
            _terminal_render(
                client, monkeypatch, terminal='dark-factory',
                terminal_rows=[_terminal_row(i) for i in range(100, 112)],
                done={'dark-factory': 12}, window=3,
            )

        messages = [
            record.getMessage() for record in caplog.records
            if record.name == 'dashboard.data.task_snapshot'
            and record.levelno >= logging.WARNING
        ]
        assert any(
            'dark-factory' in message and '12' in message and '3' in message
            for message in messages
        ), f'expected a truncation WARNING naming project/count/window, got {messages}'

    def test_an_unpositionable_window_is_unknown_not_the_oldest_rows(
        self, client, monkeypatch
    ):
        """(d) No census, no offset — and offset 0 would serve the OLDEST rows.

        Relocated from ``test_offline_status_map_never_emits_the_oldest_
        terminal_rows``. The offset is ``n_terminal - window`` and
        ``n_terminal`` comes from the census; with the census unmeasured the
        offset collapses to ``max(0, 0 - window) == 0``, which slices the
        ASCENDING-id list at its OLDEST end — months-old rows presented as the
        newest. Emitting nothing and saying why is the honest failure.
        """
        body, calls = _terminal_render(
            client, monkeypatch, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
            count_unknown=['dark-factory'], window=3,
        )

        entry = body['TASKS_TERMINAL:dark-factory']
        assert entry['state'] == 'unknown'
        assert entry['value'] is None, (
            'omitting the rows is honest; showing the OLDEST as the newest is not'
        )
        assert entry['as_of'] is None
        assert (entry['reason'] or '').strip(), (
            'an unknown datum must say why it is unknown'
        )
        assert _terminal_get_tasks(calls) == [], (
            'a window that cannot be positioned must not be fetched at all, '
            f'got {_terminal_get_tasks(calls)}'
        )
        # The rest of the payload is unaffected — this is a partial
        # degradation of one key, not a failed render.
        assert body['TASKS_COUNT_UNKNOWN_PROJECTS'] == ['dark-factory']

    def test_an_unconfigured_project_is_unknown_not_a_500(
        self, client, monkeypatch
    ):
        """(e) A name that resolves to no root is answered, not crashed on.

        An empty success would be the worse failure of the two: it reads as
        "this project has no terminal tasks", which is a measurement nobody
        made.
        """
        body, calls = _terminal_render(
            client, monkeypatch, terminal='no-such-project',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
            done={'dark-factory': 10},
        )

        entry = body['TASKS_TERMINAL:no-such-project']
        assert entry['state'] == 'unknown'
        assert entry['value'] is None
        assert 'no-such-project' in (entry['reason'] or ''), (
            f'the reason must name what could not be resolved, got {entry["reason"]!r}'
        )
        assert _terminal_get_tasks(calls) == []

    def test_no_terminal_query_carries_no_terminal_key(self, client):
        """(f) No terminal key at all, bare or per-project — an empty one is a claim."""
        with patch(
            'dashboard.api.tasks.collect_tasks_with_counts',
            new=AsyncMock(return_value=([], _snapshots(['dark-factory']))),
        ):
            resp = client.get('/api/v2/dashboard/tasks')

        assert resp.status_code == 200
        body = resp.json()
        assert _terminal_keys(body) == []
        assert set(body) == _TASKS_KEYS

    def test_the_terminal_fetch_spends_the_third_roster_slot(
        self, client, monkeypatch
    ):
        """(g) The window is bounded by the budget the roster reserved for it.

        The roster names three bounded OPERATIONS and the default render spends
        two; this render spends the third. The arithmetic
        ``PER_CALL_TIMEOUT * len(PER_PROJECT_MCP_CALLS) <=
        _TASKS_PER_PROJECT_BUDGET`` is what makes that affordable, and the
        ``timeout=`` at the wire is what makes it true rather than aspirational.
        """
        from dashboard.data.active_tasks import _TASKS_PER_PROJECT_BUDGET
        from dashboard.data.task_snapshot import (
            PER_CALL_TIMEOUT,
            PER_PROJECT_MCP_CALLS,
        )

        _body, calls = _terminal_render(
            client, monkeypatch, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
            done={'dark-factory': 10},
        )

        page = _terminal_get_tasks(calls)
        assert len(page) == 1
        assert page[0]['kwargs'].get('timeout') == PER_CALL_TIMEOUT, (
            'the terminal read must carry the per-operation budget its roster '
            f'slot reserves, got {page[0]["kwargs"]}'
        )
        assert any('terminal' in slot for slot in PER_PROJECT_MCP_CALLS), (
            'the roster must name the slot this read spends, or the budget '
            f'arithmetic is not about it: {PER_PROJECT_MCP_CALLS}'
        )
        assert PER_CALL_TIMEOUT * len(PER_PROJECT_MCP_CALLS) <= _TASKS_PER_PROJECT_BUDGET

    def test_the_real_client_applies_the_window_this_endpoint_serves(
        self, client, monkeypatch
    ):
        """(h) Both halves of the key contract, run against each other.

        ``data.js::ON_DEMAND_KEYS.terminal.key`` names the body key AND the
        DF_DATA key, and ``refreshOne`` applies ``body[key]`` verbatim. Each
        side once pinned only its own literal, both suites were green, and the
        pair still disagreed: this endpoint nested the window under
        ``TASKS_TERMINAL`` while the client read ``TASKS_TERMINAL:<project>``,
        so the window would never have applied — and nothing would have said
        so, because ``applied`` is reported for a body that merely LACKS the
        key. Feeding this endpoint's own JSON through the REAL
        ``requestOnDemand`` fails on a rename on either side.
        """
        body, _calls = _terminal_render(
            client, monkeypatch, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
            done={'dark-factory': 10},
        )

        outcome, stored = _request_on_demand(body, 'dark-factory')

        assert outcome == 'applied'
        assert stored['state'] == 'lower_bound', (
            'the client found no window under the key it reads, so it still '
            f'holds its pre-request placeholder: {stored}'
        )
        assert stored['_served_at'] == body['served_at'], (
            'the receipt must come from the body the window arrived in'
        )
        assert stored['value'] == body['TASKS_TERMINAL:dark-factory']['value'], (
            'the client must store the row list the server served, unchanged'
        )


def test_memory_returns_memory_status(client):
    """memory endpoint composes status + queue stats into a MEMORY_STATUS block."""
    with patch(
        'dashboard.data.memory.get_memory_status',
        new=AsyncMock(return_value={'offline': True, 'error': 'no fused-memory'}),
    ), patch(
        'dashboard.data.memory.get_queue_stats',
        new=AsyncMock(return_value={'counts': {}, 'oldest_pending_age_seconds': None}),
    ):
        resp = client.get('/api/v2/dashboard/memory')
    assert resp.status_code == 200
    body = resp.json()
    assert 'MEMORY_STATUS' in body
    ms = body['MEMORY_STATUS']
    for key in ('graphiti', 'mem0', 'taskmaster', 'queue'):
        assert key in ms


def test_memory_graphs_returns_timeseries_and_breakdown(client):
    resp = client.get('/api/v2/dashboard/memory-graphs')
    assert resp.status_code == 200
    body = resp.json()
    assert {'MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN'} <= set(body)
    ts = body['MEMORY_TIMESERIES']
    assert {'labels', 'reads', 'writes'} <= set(ts)
    assert isinstance(body['MEMORY_OPS_BREAKDOWN'], list)


def test_recon_returns_recon_state_and_agents(client):
    resp = client.get('/api/v2/dashboard/recon')
    assert resp.status_code == 200
    body = resp.json()
    assert {'RECON_STATE', 'AGENTS'} <= set(body)
    rs = body['RECON_STATE']
    for key in ('buffer', 'burst_state', 'watermarks', 'verdict', 'runs'):
        assert key in rs


def test_merge_queue_returns_merge_queue(client):
    resp = client.get('/api/v2/dashboard/merge-queue')
    assert resp.status_code == 200
    body = resp.json()
    assert 'MERGE_QUEUE' in body
    assert isinstance(body['MERGE_QUEUE'], dict)


def test_merge_queue_uses_24h_recent_window(client):
    """The /api/v2/dashboard/merge-queue endpoint must pass recent_window_minutes=1440
    to build_per_project_merge_queue.  Asserted via call-site kwargs because the
    test fixture carries no per-project DBs, so the window is not observable in
    the JSON payload (it returns an empty MERGE_QUEUE map regardless of window)."""
    mock_build = AsyncMock(return_value={})
    with (
        patch('dashboard.api.merge_queue.build_per_project_merge_queue', new=mock_build),
        patch('dashboard.api.merge_queue.get_merge_halt_status', new=AsyncMock(return_value=None)),
    ):
        resp = client.get('/api/v2/dashboard/merge-queue')
    assert resp.status_code == 200
    assert mock_build.await_args is not None, "build_per_project_merge_queue was never awaited"
    assert mock_build.await_args.kwargs['recent_window_minutes'] == 1440, (
        f"expected recent_window_minutes=1440, got: {mock_build.await_args.kwargs}"
    )


def test_costs_returns_full_costs_block(client):
    resp = client.get('/api/v2/dashboard/costs?window=7d')
    assert resp.status_code == 200
    body = resp.json()
    assert 'COSTS' in body
    costs = body['COSTS']
    for key in ('summary', 'by_project', 'by_account', 'by_role', 'trend', 'events', 'by_model_role'):
        assert key in costs, f'COSTS missing {key}'
    # Shape, not content, per this module's docstring — the `client` fixture
    # resolves against a real project root, which may have live runs.db data
    # (task 2534 step-13).
    assert isinstance(costs['by_model_role'], dict)
    assert isinstance(costs['by_model_role']['rows'], list)
    assert isinstance(costs['by_model_role']['turn_cap_saturation'], dict)


def test_costs_route_threads_shared_now_to_all_aggregates(client):
    """api_costs must capture one `now` and pass the SAME now to all seven
    route aggregates (task 2534 step-13 added aggregate_model_role_rollup as
    the seventh). Asserted via call-site kwargs (mirrors
    test_merge_queue_uses_24h_recent_window) because the test fixture
    carries no per-project DBs, so a shared reference timestamp is not
    observable in the JSON payload. Closes the per-DB clock-skew race at
    the route layer, on top of the aggregator-level guarantee from task
    2170 step-6/step-8."""
    mock_names = (
        'aggregate_cost_summary',
        'aggregate_cost_by_project',
        'aggregate_cost_by_account',
        'aggregate_cost_by_role',
        'aggregate_cost_trend',
        'aggregate_model_role_rollup',
    )
    mocks = {name: AsyncMock(return_value={}) for name in mock_names}
    mocks['aggregate_account_events'] = AsyncMock(return_value=[])

    with (
        patch('dashboard.app.aggregate_cost_summary', new=mocks['aggregate_cost_summary']),
        patch('dashboard.app.aggregate_cost_by_project', new=mocks['aggregate_cost_by_project']),
        patch('dashboard.app.aggregate_cost_by_account', new=mocks['aggregate_cost_by_account']),
        patch('dashboard.app.aggregate_cost_by_role', new=mocks['aggregate_cost_by_role']),
        patch('dashboard.app.aggregate_cost_trend', new=mocks['aggregate_cost_trend']),
        patch('dashboard.app.aggregate_account_events', new=mocks['aggregate_account_events']),
        patch(
            'dashboard.app.aggregate_model_role_rollup',
            new=mocks['aggregate_model_role_rollup'],
        ),
    ):
        resp = client.get('/api/v2/dashboard/costs?window=7d')

    assert resp.status_code == 200

    nows = []
    for name, mock in mocks.items():
        assert mock.await_args is not None, f'{name} was never awaited'
        assert 'now' in mock.await_args.kwargs, f'{name} was not called with now='
        now = mock.await_args.kwargs['now']
        assert now is not None, f'{name} received now=None'
        assert now.tzinfo is not None, f'{name} received a naive (non-UTC-aware) datetime: {now!r}'
        nows.append(now)

    assert all(n == nows[0] for n in nows), (
        f"expected all seven aggregates to share one reference now, got {nows!r}"
    )


def test_costs_route_includes_by_model_role(client):
    """The rollup computed by aggregate_model_role_rollup (rows +
    turn_cap_saturation) flows through shape_costs into COSTS.by_model_role
    (task 2534 step-13)."""
    fake_rollup = {
        'rows': [
            {'model': 'sonnet', 'role': 'implementer', 'invocation_count': 4,
             'done_count': 2, 'blocked_count': 1, 'done_rate': 0.5, 'blocked_rate': 0.25,
             'capped_count': 1, 'cap_hit_rate': 0.25, 'total_cost_usd': 8.0, 'cost_per_done': 4.0},
        ],
        'turn_cap_saturation': {'simple_task': 0.5, 'architect': None},
    }
    with patch(
        'dashboard.app.aggregate_model_role_rollup',
        new=AsyncMock(return_value=fake_rollup),
    ):
        resp = client.get('/api/v2/dashboard/costs?window=7d')

    assert resp.status_code == 200
    by_model_role = resp.json()['COSTS']['by_model_role']
    assert by_model_role['rows'] == fake_rollup['rows']
    assert by_model_role['turn_cap_saturation'] == fake_rollup['turn_cap_saturation']


def test_shape_costs_places_model_role_rollup_under_by_model_role():
    """shape_costs(..., by_model_role=<aggregate_model_role_rollup output>)
    places rows+turn_cap_saturation under COSTS.by_model_role without
    altering the existing summary/by_project/by_account/by_role/trend/events
    keys; omitting by_model_role still yields a well-formed empty block
    (task 2534 step-13)."""
    rollup = {
        'rows': [
            {'model': 'sonnet', 'role': 'implementer', 'invocation_count': 4,
             'done_count': 2, 'blocked_count': 1, 'done_rate': 0.5, 'blocked_rate': 0.25,
             'capped_count': 1, 'cap_hit_rate': 0.25, 'total_cost_usd': 8.0, 'cost_per_done': 4.0},
            {'model': 'opus', 'role': 'steward', 'invocation_count': 1,
             'done_count': 0, 'blocked_count': 0, 'done_rate': 0.0, 'blocked_rate': 0.0,
             'capped_count': 0, 'cap_hit_rate': 0.0, 'total_cost_usd': 0.5, 'cost_per_done': None},
        ],
        'turn_cap_saturation': {'simple_task': 0.5, 'architect': 0.0, 'no_max_turns_role': None},
    }
    body = redux_api.shape_costs(
        summary={}, by_project={}, by_account={}, by_role={}, trend={}, events=[],
        by_model_role=rollup,
    )
    costs = body['COSTS']
    for key in ('summary', 'by_project', 'by_account', 'by_role', 'trend', 'events'):
        assert key in costs, f'COSTS missing {key}'

    assert costs['by_model_role']['rows'] == rollup['rows']
    assert costs['by_model_role']['turn_cap_saturation'] == rollup['turn_cap_saturation']

    body_default = redux_api.shape_costs(
        summary={}, by_project={}, by_account={}, by_role={}, trend={}, events=[],
    )
    assert body_default['COSTS']['by_model_role'] == {'rows': [], 'turn_cap_saturation': {}}


def test_performance_returns_performance(client):
    resp = client.get('/api/v2/dashboard/performance')
    assert resp.status_code == 200
    body = resp.json()
    assert 'PERFORMANCE' in body
    assert isinstance(body['PERFORMANCE'], dict)


def test_burndown_returns_aggregate_and_per_project(client):
    resp = client.get('/api/v2/dashboard/burndown?window=30d')
    assert resp.status_code == 200
    body = resp.json()
    assert {'BURNDOWN', 'BURNDOWN_BY_PROJECT'} <= set(body)
    aggregate = body['BURNDOWN']
    assert {'labels', 'done', 'in_progress', 'blocked', 'pending'} <= set(aggregate)


def test_burndown_route_threads_shared_now_to_all_aggregates(client):
    """api_burndown must capture one `now` and pass the SAME now to every
    per-project aggregate_burndown_series call (mirrors
    test_costs_route_threads_shared_now_to_all_aggregates). Closes the
    per-project clock-skew race at the route layer, on top of the
    aggregator-level guarantee from task 2192 step-2."""
    mock_projects = AsyncMock(return_value=['p1', 'p2'])
    mock_series = AsyncMock(return_value={
        'labels': [], 'done': [], 'cancelled': [], 'blocked': [],
        'deferred': [], 'in_progress': [], 'pending': [],
    })

    with (
        patch('dashboard.api.burndown.aggregate_burndown_projects', new=mock_projects),
        patch('dashboard.api.burndown.aggregate_burndown_series', new=mock_series),
    ):
        resp = client.get('/api/v2/dashboard/burndown?window=30d')

    assert resp.status_code == 200

    assert mock_series.await_count == 2, (
        f'expected aggregate_burndown_series to be awaited twice (once per '
        f'project), got {mock_series.await_count}'
    )
    nows = []
    for call in mock_series.await_args_list:
        assert 'now' in call.kwargs, (
            f'aggregate_burndown_series was not called with now= (got {call.kwargs!r})'
        )
        now = call.kwargs['now']
        assert now is not None, 'aggregate_burndown_series received now=None'
        assert now.tzinfo is not None, (
            f'aggregate_burndown_series received a naive (non-UTC-aware) datetime: {now!r}'
        )
        nows.append(now)

    assert all(n == nows[0] for n in nows), (
        f'expected all per-project aggregates to share one reference now, got {nows!r}'
    )


# ---------------------------------------------------------------------------
# Escalations endpoint
# ---------------------------------------------------------------------------

_EMPTY_SUMMARY = {
    'by_level': {0: 0, 1: 0, 2: 0},
    'by_status': {'pending': 0, 'resolved': 0, 'dismissed': 0},
}

_EMPTY_QUEUES = {
    'subsections': [],
    'summary': _EMPTY_SUMMARY,
}


def test_escalations_endpoint_returns_escalations_block(client):
    """GET /api/v2/dashboard/escalations returns 200 with ESCALATIONS key."""
    with patch(
        'dashboard.api.escalations.build_escalation_queues',
        return_value=_EMPTY_QUEUES,
    ), patch(
        'dashboard.api.escalations.fetch_tasks',
        new=AsyncMock(return_value=[]),
    ):
        resp = client.get('/api/v2/dashboard/escalations')
    assert resp.status_code == 200
    body = resp.json()
    assert 'ESCALATIONS' in body
    esc = body['ESCALATIONS']
    assert 'subsections' in esc
    assert 'summary' in esc
    assert esc['subsections'] == []


def test_escalations_endpoint_attaches_task_cards_and_resolves_recon(client, tmp_path):
    """Full endpoint→shaper integration: task attachment + reconciliation resolution."""
    from dashboard.api.escalations import _task_cards_cache_clear
    _task_cards_cache_clear()

    proj_a = tmp_path / 'projA'
    task_dict = {
        'id': 11, 'title': 'wired', 'description': '', 'details': '',
        'status': 'pending', 'priority': 'med', 'dependencies': [], 'metadata': {},
    }
    sub_summary = {
        'by_level': {0: 1, 1: 1, 2: 0},
        'by_status': {'pending': 2, 'resolved': 0, 'dismissed': 0},
    }
    queues = {
        'subsections': [
            {
                'id': str(proj_a),
                'label': 'projA',
                'kind': 'orchestrator',
                'escalations': [{'id': 'e1', 'task_id': 11, 'level': 0, 'status': 'pending', 'summary': 'oops'}],
                'summary': sub_summary,
            },
            {
                'id': 'reconciliation',
                'label': 'fused-memory',
                'kind': 'reconciliation',
                'escalations': [{
                    'id': 'er1', 'task_id': 11,
                    'worktree': str(proj_a / '.worktrees' / '11'),
                    'level': 1, 'status': 'pending',
                }],
                'summary': sub_summary,
            },
        ],
        'summary': {
            'by_level': {0: 1, 1: 1, 2: 0},
            'by_status': {'pending': 2, 'resolved': 0, 'dismissed': 0},
        },
    }

    with patch(
        'dashboard.api.escalations.build_escalation_queues',
        return_value=queues,
    ), patch(
        'dashboard.api.escalations.fetch_tasks',
        new=AsyncMock(return_value=[task_dict]),
    ):
        resp = client.get('/api/v2/dashboard/escalations')

    assert resp.status_code == 200
    body = resp.json()
    subs = body['ESCALATIONS']['subsections']
    assert len(subs) == 2

    orch_sub = next(s for s in subs if s['kind'] == 'orchestrator')
    assert len(orch_sub['escalations']) == 1
    orch_row = orch_sub['escalations'][0]
    assert orch_row['project'] == 'projA'
    assert orch_row['task']['title'] == 'wired'
    assert orch_row['task_unresolved'] is False

    recon_sub = next(s for s in subs if s['kind'] == 'reconciliation')
    assert len(recon_sub['escalations']) == 1
    recon_row = recon_sub['escalations'][0]
    assert recon_row['project'] == 'projA'
    assert recon_row['task']['title'] == 'wired'
    assert recon_row['task_unresolved'] is False


def test_load_task_cards_caches_within_ttl(client, tmp_path):
    """_load_task_cards: cache hit within TTL + offline result not cached."""
    from dashboard.api.escalations import _task_cards_cache_clear

    proj_a = tmp_path / 'projA'
    orch_sub = {
        'id': str(proj_a),
        'label': 'projA',
        'kind': 'orchestrator',
        'escalations': [],
        'summary': _EMPTY_SUMMARY,
    }
    recon_sub = {
        'id': 'reconciliation',
        'label': 'fused-memory',
        'kind': 'reconciliation',
        'escalations': [],
        'summary': _EMPTY_SUMMARY,
    }
    one_orch_queues = {
        'subsections': [orch_sub, recon_sub],
        'summary': _EMPTY_SUMMARY,
    }
    task_list = [{'id': 1, 'title': 't', 'description': '', 'details': '', 'status': 'pending',
                  'priority': 'low', 'dependencies': [], 'metadata': {}}]

    # Case 1: cache hit — second request should NOT call fetch_tasks again.
    _task_cards_cache_clear()
    mock_ft = AsyncMock(return_value=task_list)
    with patch('dashboard.api.escalations.build_escalation_queues', return_value=one_orch_queues), \
         patch('dashboard.api.escalations.fetch_tasks', new=mock_ft):
        client.get('/api/v2/dashboard/escalations')
        client.get('/api/v2/dashboard/escalations')
    assert mock_ft.call_count == 1, f'expected 1 fetch_tasks call, got {mock_ft.call_count}'

    # Case 2: offline result NOT cached — each request should call fetch_tasks.
    _task_cards_cache_clear()
    mock_offline = AsyncMock(return_value={'offline': True, 'error': 'x'})
    with patch('dashboard.api.escalations.build_escalation_queues', return_value=one_orch_queues), \
         patch('dashboard.api.escalations.fetch_tasks', new=mock_offline):
        r1 = client.get('/api/v2/dashboard/escalations')
        r2 = client.get('/api/v2/dashboard/escalations')
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert mock_offline.call_count == 2, f'expected 2 fetch_tasks calls, got {mock_offline.call_count}'


def test_load_task_cards_ttl_expiry(client, tmp_path):
    """_load_task_cards: after TTL expires, fetch_tasks is called again."""
    import dashboard.api.escalations as escalations_module
    from dashboard.api.escalations import _task_cards_cache_clear

    proj_a = tmp_path / 'projA'
    one_orch_queues = {
        'subsections': [
            {'id': str(proj_a), 'label': 'projA', 'kind': 'orchestrator',
             'escalations': [], 'summary': _EMPTY_SUMMARY},
            {'id': 'reconciliation', 'label': 'fused-memory', 'kind': 'reconciliation',
             'escalations': [], 'summary': _EMPTY_SUMMARY},
        ],
        'summary': _EMPTY_SUMMARY,
    }
    task_list = [{'id': 1, 'title': 't', 'description': '', 'details': '',
                  'status': 'pending', 'priority': 'low', 'dependencies': [], 'metadata': {}}]

    _task_cards_cache_clear()
    original_ttl = escalations_module._TASK_CARDS_TTL_SECONDS
    mock_ft = AsyncMock(return_value=task_list)
    try:
        with patch('dashboard.api.escalations.build_escalation_queues', return_value=one_orch_queues), \
             patch('dashboard.api.escalations.fetch_tasks', new=mock_ft):
            # First request: cache miss — fetch_tasks called once, result cached.
            client.get('/api/v2/dashboard/escalations')
            assert mock_ft.call_count == 1

            # Zero out TTL so the cached entry is immediately treated as expired.
            escalations_module._TASK_CARDS_TTL_SECONDS = 0.0

            # Second request: TTL expired — fetch_tasks called again.
            resp = client.get('/api/v2/dashboard/escalations')
    finally:
        escalations_module._TASK_CARDS_TTL_SECONDS = original_ttl

    assert resp.status_code == 200
    assert mock_ft.call_count == 2, (
        f'expected 2 fetch_tasks calls after TTL expiry, got {mock_ft.call_count}'
    )


# ---------------------------------------------------------------------------
# task-2218 step-9: _load_task_cards single-flight (direct calls, no endpoint)
# ---------------------------------------------------------------------------


async def test_load_task_cards_single_flight_collapses_concurrent_cold_callers(
    dummy_client, dummy_config
):
    """Concurrent cold callers for one project_root collapse onto one fetch_tasks call.

    Case 1: three concurrent asyncio tasks hitting a cold cache for the same
    project_root must share a single in-flight fetch_tasks call and all
    receive the same result (TTLCache single-flight — the plain-dict cache
    this replaces has no refresh lock, so today each of the three fetches).

    Case 2 (reuse of existing behavior): an offline dict result from
    fetch_tasks is never cached, so each direct call re-fetches.
    """
    import asyncio

    from dashboard.api.escalations import _load_task_cards, _task_cards_cache_clear

    # Case 1: single-flight collapse
    _task_cards_cache_clear()
    task_list = [{'id': 1, 'title': 't', 'description': '', 'details': '',
                  'status': 'pending', 'priority': 'low', 'dependencies': [], 'metadata': {}}]
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def slow_fetch_tasks(client, config, project_root):
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()
        return list(task_list)

    with patch('dashboard.api.escalations.fetch_tasks', new=AsyncMock(side_effect=slow_fetch_tasks)):
        tasks = [
            asyncio.create_task(_load_task_cards(dummy_client, dummy_config, '/proj/X'))
            for _ in range(3)
        ]
        await started.wait()
        await asyncio.sleep(0)  # let the other two queue on the lock
        release.set()
        results = await asyncio.gather(*tasks)

    assert call_count == 1, f'expected a single fetch_tasks call, got {call_count}'
    assert all(r == task_list for r in results)

    # Case 2: offline result NOT cached — each direct call re-fetches.
    _task_cards_cache_clear()
    mock_offline = AsyncMock(return_value={'offline': True, 'error': 'x'})
    with patch('dashboard.api.escalations.fetch_tasks', new=mock_offline):
        r1 = await _load_task_cards(dummy_client, dummy_config, '/proj/Y')
        r2 = await _load_task_cards(dummy_client, dummy_config, '/proj/Y')

    assert r1 == [] and r2 == []
    assert mock_offline.call_count == 2, f'expected 2 fetch_tasks calls, got {mock_offline.call_count}'


# ---------------------------------------------------------------------------
# task-4788: _load_task_cards whole-operation budget
#
# ``fetch_tasks``' own *timeout* is a PER-HTTP-REQUEST budget: it bounds
# connect/read/write and pool acquisition and nothing else. The incident that
# motivated these two tests hung inside httpcore's connection lock, where no
# outbound socket is ever opened and that timeout never fires — so
# /api/v2/dashboard/escalations wedged for 19.8 h with the per-request budget
# fully in place. Only an enclosing ``asyncio.wait_for`` cancels that wait.
#
# Both hang stubs are ``await asyncio.Event().wait()`` on an event nothing
# ever sets, deliberately NOT a sleep: a sleep shorter than the budget passes
# against the pre-fix code too and would prove nothing. Since that would
# otherwise hang pytest forever, each call is wrapped in a TEST-SIDE
# ``wait_for(2.0)`` — 40x the monkeypatched 0.05 s budget, so it can only trip
# on a real regression, never on scheduling jitter.
# ---------------------------------------------------------------------------


async def test_a_hanging_fetch_tasks_does_not_hang_load_task_cards(
    monkeypatch, dummy_client, dummy_config, caplog
):
    """A fetch that never returns degrades to [] — loudly, and uncached.

    The WARNING is asserted, not incidental: ``[]`` is exactly what an
    ordinary empty result looks like, so the log line is the ONLY thing that
    distinguishes "this project has no task cards" from "we ran out of budget
    and never found out". Without it a timeout is invisible to an operator,
    which is the 19.8 h failure mode in miniature.
    """
    import asyncio
    import logging

    import dashboard.api.escalations as _esc
    from dashboard.api.escalations import _load_task_cards, _task_cards_cache_clear

    # A warm entry would be served without ever reaching the hang.
    _task_cards_cache_clear()

    call_count = 0

    async def hang_fetch_tasks(client, config, project_root):
        nonlocal call_count
        call_count += 1
        await asyncio.Event().wait()  # nothing ever sets it

    monkeypatch.setattr(_esc, '_TASK_CARDS_BUDGET', 0.05)

    with (
        patch('dashboard.api.escalations.fetch_tasks', new=hang_fetch_tasks),
        caplog.at_level(logging.WARNING, logger='dashboard.api.escalations'),
    ):
        result = await asyncio.wait_for(
            _load_task_cards(dummy_client, dummy_config, '/proj/HANG'),
            timeout=2.0,
        )
        assert result == [], (
            'the shape _load_task_cards already promises for an offline '
            'marker or MCP failure — the escalation tab renders cardless '
            'rather than hanging'
        )
        assert call_count == 1

        # A timeout must not pin an empty card list for the TTL window:
        # nothing was written to the cache, so the next poll re-attempts.
        await asyncio.wait_for(
            _load_task_cards(dummy_client, dummy_config, '/proj/HANG'),
            timeout=2.0,
        )
    assert call_count == 2, (
        'the second call must re-enter the stub — a timeout that cached its '
        '[] would blank the tab for the whole TTL window'
    )

    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == 'dashboard.api.escalations'
    ]
    assert any('whole-operation budget' in m for m in warnings), (
        f'no timeout WARNING was logged (records: {warnings}) — the returned '
        '[] is indistinguishable from an ordinary empty result, so the log '
        'line is the only operator-visible trace that the budget expired'
    )
    assert any('/proj/HANG' in m for m in warnings), (
        f'the WARNING must name the project root that degraded: {warnings}'
    )


async def test_a_concurrent_task_cards_caller_on_the_same_root_is_bounded_too(
    monkeypatch, dummy_client, dummy_config
):
    """Both callers are bounded, not just the one that wins the lock.

    This pins the wrap PLACEMENT, mirroring the merge_queue.load_task_titles
    test. ``TTLCache.get_or_refresh`` serializes cold callers for one key
    behind a per-key lock and runs the refresh WHILE HOLDING it, so an
    inner-only wrap would leave caller B queued UNBOUNDED for caller A's whole
    budget and then running its own full-budget refresh — the pair costs 2x
    the budget and N waiters cost N x. The dashboard polls every 3 s, so
    waiters are the routine case, not a corner.
    """
    import asyncio

    import dashboard.api.escalations as _esc
    from dashboard.api.escalations import _load_task_cards, _task_cards_cache_clear

    _task_cards_cache_clear()

    async def hang_fetch_tasks(client, config, project_root):
        await asyncio.Event().wait()

    budget = 0.5
    monkeypatch.setattr(_esc, '_TASK_CARDS_BUDGET', budget)
    loop = asyncio.get_running_loop()

    with patch('dashboard.api.escalations.fetch_tasks', new=hang_fetch_tasks):
        started = loop.time()
        results = await asyncio.wait_for(
            asyncio.gather(*[
                _load_task_cards(dummy_client, dummy_config, '/proj/SHARED')
                for _ in range(2)
            ]),
            timeout=2.0,
        )
        elapsed = loop.time() - started

    assert results == [[], []]
    # The assertion is about SERIALIZATION, not merely about returning:
    # an inner-only wrap costs 2 x budget here and scales with waiters.
    #
    # The budget is deliberately LARGE for a test whose subject is a timeout.
    # It is not scaled because the operation needs 0.5 s — it is scaled so the
    # assertion's ABSOLUTE jitter margin exceeds real-world event-loop
    # scheduling, GC and pytest overhead. Correct behaviour (outer wrap) costs
    # ~1x budget; the inner-only-wrap regression costs ~2x; 1.5x sits exactly
    # midway, giving 0.25 s of slack on BOTH sides. At the original 0.05 s the
    # discrimination was sound in ratio and worthless in absolute terms (50 ms
    # of slack), and it flaked at ~4% per run. Do NOT shrink the budget back to
    # "speed up the suite" — that silently reintroduces the flake.
    assert elapsed < 1.5 * budget, (
        f'two concurrent callers took {elapsed:.3f}s against a '
        f'{1.5 * budget}s threshold (1.5 x the {budget}s per-call budget); '
        f'the inner-only-wrap regression costs ~{2 * budget}s — that is the '
        'serialized cost of an inner-only wrap; the wait_for must enclose '
        'get_or_refresh so a caller QUEUED on the per-key lock is bounded too'
    )


def test_escalations_endpoint_multi_root_gather(client, tmp_path):
    """Endpoint fetches each orchestrator root separately and maps tasks to the right subsection."""
    from dashboard.api.escalations import _task_cards_cache_clear

    _task_cards_cache_clear()

    proj_a = tmp_path / 'projA'
    proj_b = tmp_path / 'projB'

    task_a = {'id': 11, 'title': 'task-A', 'description': '', 'details': '',
              'status': 'pending', 'priority': 'low', 'dependencies': [], 'metadata': {}}
    task_b = {'id': 22, 'title': 'task-B', 'description': '', 'details': '',
              'status': 'pending', 'priority': 'high', 'dependencies': [], 'metadata': {}}

    queues = {
        'subsections': [
            {'id': str(proj_a), 'label': 'projA', 'kind': 'orchestrator',
             'escalations': [{'id': 'esc-a1', 'task_id': 11, 'level': 0,
                              'status': 'pending', 'summary': 'a-issue'}],
             'summary': _EMPTY_SUMMARY},
            {'id': str(proj_b), 'label': 'projB', 'kind': 'orchestrator',
             'escalations': [{'id': 'esc-b1', 'task_id': 22, 'level': 1,
                              'status': 'pending', 'summary': 'b-issue'}],
             'summary': _EMPTY_SUMMARY},
        ],
        'summary': _EMPTY_SUMMARY,
    }

    async def fetch_side_effect(_client, _config, root_id):
        if 'projA' in root_id:
            return [task_a]
        if 'projB' in root_id:
            return [task_b]
        return []

    with patch('dashboard.api.escalations.build_escalation_queues', return_value=queues), \
         patch('dashboard.api.escalations.fetch_tasks', side_effect=fetch_side_effect):
        resp = client.get('/api/v2/dashboard/escalations')

    assert resp.status_code == 200
    body = resp.json()
    subs = body['ESCALATIONS']['subsections']
    assert len(subs) == 2

    sub_a = next(s for s in subs if s['label'] == 'projA')
    sub_b = next(s for s in subs if s['label'] == 'projB')

    # projA subsection gets task-A (id=11), not task-B.
    assert len(sub_a['escalations']) == 1
    row_a = sub_a['escalations'][0]
    assert row_a['project'] == 'projA'
    assert row_a['task']['id'] == 11
    assert row_a['task']['title'] == 'task-A'
    assert row_a['task_unresolved'] is False

    # projB subsection gets task-B (id=22), not task-A.
    assert len(sub_b['escalations']) == 1
    row_b = sub_b['escalations'][0]
    assert row_b['project'] == 'projB'
    assert row_b['task']['id'] == 22
    assert row_b['task']['title'] == 'task-B'
    assert row_b['task_unresolved'] is False


# ---------------------------------------------------------------------------
# /api/load — host load card endpoint
# ---------------------------------------------------------------------------


def test_load_endpoint_returns_known_metric_shape(client) -> None:
    """/api/load returns all 9 known metrics with the expected per-metric shape.

    The test client creates a temp project_root where load-samples.db does not
    exist, so every metric must degrade to the all-placeholders shape (current=None,
    sparkline=[]).  This exercises the missing-DB degradation path end-to-end.
    """
    from dashboard.data.load import KNOWN_METRICS

    resp = client.get('/api/load')
    assert resp.status_code == 200

    body = resp.json()
    assert isinstance(body, dict)
    assert set(body.keys()) == set(KNOWN_METRICS)

    # Every key must carry exactly these four sub-keys
    expected_sub_keys = {'current', 'sparkline', 'window_mean', 'window_max'}
    for metric in KNOWN_METRICS:
        assert set(body[metric].keys()) == expected_sub_keys, (
            f'{metric} has unexpected keys: {set(body[metric].keys())}'
        )

    # Spot-check one metric for the placeholder values (DB absent → all None/[])
    oq = body['occt_queue_depth']
    assert oq['current'] is None
    assert oq['sparkline'] == []
    assert oq['window_mean'] is None
    assert oq['window_max'] is None


# ---------------------------------------------------------------------------
# api_merge_queue — live fetch integration (task-1606 step-11)
# ---------------------------------------------------------------------------

# Fake project root; label = basename('proj-a') = 'proj-a'
_PROJ_ROOT = '/home/test/proj-a'
_PROJ_LABEL = 'proj-a'

# A live entry that would be TTL-dropped by the event-derived fallback (4h old)
_LIVE_ENTRY_4H = {
    'task_id': '3112', 'branch': 'task/3112', 'state': 'queued',
    'age_secs': 14400.0, 'position': 1, 'waiter_alive': True,
}
# A second entry for the same task_id (AC2: two rows same task)
_LIVE_ENTRY_RETRY = {
    'task_id': '3112', 'branch': 'task/3112-retry', 'state': 'queued',
    'age_secs': 300.0, 'position': 2, 'waiter_alive': True,
}
# Event-derived fallback entry
_EVENT_ENTRY = {
    'task_id': '7', 'branch': 'task/7', 'state': 'queued',
    'timestamp': '2026-06-04T10:00:00+00:00',
}


def _proj_raw(active: list, *, active_approximate: bool = False) -> dict:
    """Build a minimal build_per_project_merge_queue output for one project."""
    return {
        _PROJ_ROOT: {
            'depth_timeseries': {'labels': [], 'values': []},
            'outcomes': {'labels': [], 'values': []},
            'latency': {},
            'recent': [],
            'speculative': {},
            'active': active,
            'active_approximate': active_approximate,
            'train_events': [],
        },
    }


def test_merge_queue_live_path_uses_live_entries(client):
    """Case A: fetch_live_merge_queues reachable → active reflects LIVE entries.

    AC1: 4h-old entry visible (not TTL-dropped).
    AC2: two same-task entries both appear.
    active_approximate is False (live data).
    """
    live_map = {
        _PROJ_LABEL: {
            'reachable': True,
            'entries': [_LIVE_ENTRY_4H, _LIVE_ENTRY_RETRY],
        },
    }
    with (
        patch('dashboard.api.merge_queue.build_per_project_merge_queue',
              new=AsyncMock(return_value=_proj_raw([_EVENT_ENTRY]))),
        patch('dashboard.api.merge_queue.get_merge_halt_status', new=AsyncMock(return_value={})),
        patch('dashboard.api.merge_queue.load_task_titles', new=AsyncMock(return_value={})),
        patch('dashboard.api.merge_queue.fetch_live_merge_queues', new=AsyncMock(return_value=live_map)),
    ):
        resp = client.get('/api/v2/dashboard/merge-queue')

    assert resp.status_code == 200
    mq = resp.json()['MERGE_QUEUE']
    assert _PROJ_LABEL in mq, f'{_PROJ_LABEL} missing from MERGE_QUEUE; got {list(mq)}'
    proj = mq[_PROJ_LABEL]

    active = proj['active']
    task_ids = [e['task_id'] for e in active]
    # AC1: long-queued entry is visible
    assert '3112' in task_ids, (
        f'AC1: expected task 3112 (4h old) in active; got {task_ids}'
    )
    # AC2: two entries for the same task_id
    assert task_ids.count('3112') == 2, (
        f'AC2: expected 2 entries for task_id=3112; got {task_ids}'
    )
    # Event-derived fallback entry must NOT appear (live path supersedes it)
    assert '7' not in task_ids, (
        f'Expected event-derived task 7 absent when live path is active; got {task_ids}'
    )
    assert proj['active_approximate'] is False


def test_merge_queue_fallback_path_when_unreachable(client):
    """Case B: fetch_live_merge_queues unreachable → fallback with active_approximate=True.

    AC3: no fabricated rows; the event-derived list is used and labelled approximate.
    """
    # fetch_live_merge_queues returns {} (no live data) → resolve_active falls back
    with (
        patch('dashboard.api.merge_queue.build_per_project_merge_queue',
              new=AsyncMock(return_value=_proj_raw([_EVENT_ENTRY]))),
        patch('dashboard.api.merge_queue.get_merge_halt_status', new=AsyncMock(return_value={})),
        patch('dashboard.api.merge_queue.load_task_titles', new=AsyncMock(return_value={})),
        patch('dashboard.api.merge_queue.fetch_live_merge_queues', new=AsyncMock(return_value={})),
    ):
        resp = client.get('/api/v2/dashboard/merge-queue')

    assert resp.status_code == 200
    mq = resp.json()['MERGE_QUEUE']
    assert _PROJ_LABEL in mq, f'{_PROJ_LABEL} missing; got {list(mq)}'
    proj = mq[_PROJ_LABEL]

    active = proj['active']
    task_ids = [e['task_id'] for e in active]
    # Event-derived entry appears
    assert '7' in task_ids, (
        f'Expected event-derived task 7 in fallback active; got {task_ids}'
    )
    # AC3: no fabricated live entries
    assert '3112' not in task_ids, (
        f'AC3: live entry 3112 must not appear in fallback path; got {task_ids}'
    )
    assert proj['active_approximate'] is True


def test_tasks_offline_flag_survives_a_hang_that_degrades_most_roots(client):
    """No root produced rows + at least one demonstrably failed IS the outage.

    The handler's own budget caps how many roots can reach the offline state:
    in a hang each root burns up to ``_TASKS_PER_PROJECT_BUDGET`` before
    ``wait_for`` cuts it, and a cut root lands in ``degraded``, not
    ``offline``. A stricter ``len(offline) == total_roots`` test therefore
    made this flag unreachable on a nine-root config for the most likely total
    outage — the payload would say "unavailable for 2 of 9" plus "timed out
    for 7 of 9" and never the thing that was true: nothing loaded.
    """
    body = _tasks_body(
        client,
        offline_projects=['p0', 'p1'],
        degraded_projects=['p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'],
        total_roots=9,
    )

    assert body['TASKS_OFFLINE'] is True, (
        'no root produced rows and two demonstrably failed — that is the '
        'outage the global banner copy describes'
    )
    # The separate lists stay separate: the flag is an ADDITIONAL fact, and
    # collapsing degraded into offline would report a timeout as a proven
    # outage on the per-project list too.
    assert body['TASKS_OFFLINE_PROJECTS'] == ['p0', 'p1']
    assert len(body['TASKS_DEGRADED_PROJECTS']) == 7


def test_tasks_one_healthy_root_vetoes_the_outage_flag(client):
    """A single root that produced rows blocks the global claim, however bad the rest.

    The flag's conjunct is "no root was measured this render". A root missing
    from both failure lists had its rows read (classify routes each root to at
    most one list), so "task data unavailable" would be false.
    """
    body = _tasks_body(
        client,
        offline_projects=['p0'],
        degraded_projects=['p1'],
        total_roots=3,
    )

    assert body['TASKS_OFFLINE'] is False, (
        'p2 produced rows — they are in this very payload'
    )


def test_tasks_count_unknown_root_vetoes_the_outage_flag(client):
    """A count-unknown root produced current ROWS, so it is not an absence of data."""
    body = _tasks_body(
        client,
        offline_projects=['p0'],
        degraded_projects=[],
        count_unknown_projects=['p1'],
        total_roots=2,
    )

    assert body['TASKS_OFFLINE'] is False, (
        "p1's rows loaded fine — only its done count is unknown, which is a "
        'different (and separately reported) fact from an outage'
    )
    assert body['TASKS_COUNT_UNKNOWN_PROJECTS'] == ['p1']


def test_last_good_rows_do_not_veto_the_outage_flag(client):
    """Rows served from a root's last good were not measured this render.

    An offline root still puts its last good rows, aged, into ACTIVE_TASKS.
    The flag asks whether THIS render measured any root. So when every root
    is offline, the banner stands over those rows, and each root's ``rows``
    Datum says how old they are.
    """
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta

    from dashboard.data.datum import Datum, DatumState
    from dashboard.data.task_snapshot import FRESHNESS_BOUND_SECONDS

    row = {'id': 'p0/T-1', 'project': 'p0', 'status': 'in-progress'}
    last_good = Datum(
        [row], datetime.now(UTC) - timedelta(minutes=5), DatumState.STALE,
        'canned offline root', FRESHNESS_BOUND_SECONDS,
    )
    offline = replace(_snapshot(health='offline'), rows=last_good)
    with patch(
        'dashboard.api.tasks.collect_tasks_with_counts',
        new=AsyncMock(return_value=([row], {'p0': offline})),
    ), patch(
        'dashboard.api.tasks._all_project_roots', new=lambda config: _fake_roots(1),
    ):
        resp = client.get('/api/v2/dashboard/tasks')

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['ACTIVE_TASKS'] == [row]
    assert body['TASKS_SNAPSHOT']['p0']['rows']['state'] == 'stale'
    assert body['TASKS_OFFLINE_PROJECTS'] == ['p0']
    assert body['TASKS_OFFLINE'] is True
