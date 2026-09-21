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
    """Even with no running orchestrators the response carries both keys."""
    with patch(
        'dashboard.api.orchestrators.discover_orchestrators',
        new=AsyncMock(return_value=[]),
    ):
        resp = client.get('/api/v2/dashboard/orchestrators')
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {'ORCHESTRATORS', 'PROJECTS', 'ORCHESTRATORS_SPARK'}
    assert isinstance(body['ORCHESTRATORS'], list)
    assert isinstance(body['PROJECTS'], list)
    assert isinstance(body['ORCHESTRATORS_SPARK'], dict)
    assert 'labels' in body['ORCHESTRATORS_SPARK']
    assert 'values' in body['ORCHESTRATORS_SPARK']


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

    The flag's conjunct is "NO root produced rows" — a root missing from both
    failure lists produced rows (the three lists are disjoint by
    construction), so "task data unavailable" would be false.
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
