"""Tests for the shared httpx client's connection-pool bound (task 3871).

The dashboard shares ONE ``httpx.AsyncClient`` across every fan-out
(merge_halt, task_runtime, live merge queue, the metrics samplers). It was
constructed with no ``limits=``, so it inherited httpx's stock
``DEFAULT_LIMITS`` — ``max_connections=100``, ``max_keepalive_connections=20``,
``keepalive_expiry=5.0``.

This client overrides two of httpx's stock defaults, for two different
reasons:

1. The bound is a fixed 100 regardless of how many orchestrators are
   onboarded, so it is simultaneously too loose for a one-project install and
   unrelated to the real peak (projects x concurrent endpoint families). This
   one IS a real defect in the stock default — see ``TestBuildHttpLimits``
   below.
2. ``keepalive_expiry`` is pinned explicitly at 4.0 rather than left at
   httpx's stock 5.0 (or omitted). This is NOT fixing a race against the
   server's close: httpcore evaluates ``keepalive_expiry`` lazily, only when
   the pool is next used, with no background reaper, so it can never pre-empt
   a server-side close regardless of its value (verified mechanism:
   ``dashboard/src/dashboard/app.py::_HTTP_KEEPALIVE_EXPIRY_SECONDS``).
   Omitting the argument would leave httpx's stock 5.0, which the same
   verified-mechanism block measured as behaviourally IDENTICAL to 4.0 on
   this install — so 4.0 is pinned explicitly not because it behaves any
   differently, but to keep the shipped number visible and reviewable at
   the call site.

This is a GUARD on worst-case pool growth, not a leak fix. It does NOT fix
CLOSE-WAIT accumulation (owned by the task-3857 re-spec).

``_build_http_limits`` is a PURE helper so the sizing is directly testable:
``httpx.AsyncClient`` exposes no public accessor for its limits, so the only
alternative is asserting on ``client._transport._pool._max_connections``,
which is private and brittle across httpx versions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from dashboard.config import DashboardConfig

# merge_halt.py's module docstring names a "3s polling loop". keepalive_expiry
# must sit strictly ABOVE this or a connection never survives one poll cycle
# and every cycle pays a fresh handshake.
_DASHBOARD_POLL_INTERVAL = 3.0

# httpx's stock DEFAULT_LIMITS (httpx/_config.py): max_connections=100,
# max_keepalive_connections=20. Spelled out here rather than read off
# `httpx.Limits()`, whose no-arg defaults are None for these two `max_*`
# fields specifically (`keepalive_expiry` defaults to 5.0, not None) — and
# DEFAULT_LIMITS itself is a separate module constant that httpx does not
# re-export publicly, so it can't be read off that way either.
_HTTPX_STOCK_MAX_CONNECTIONS = 100
_HTTPX_STOCK_MAX_KEEPALIVE = 20


def _config(
    tmp_path: Path, *, escalation: int = 0, fused: int = 0,
) -> DashboardConfig:
    """Build a DashboardConfig with an explicit endpoint count."""
    return DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=[f'http://localhost:{9000 + i}' for i in range(fused)],
        escalation_urls={
            f'proj{i}': f'http://127.0.0.1:{8100 + i}' for i in range(escalation)
        },
        known_project_roots=[],
    )


class TestBuildHttpLimits:
    """The pure sizing helper."""

    def test_returns_httpx_limits(self, tmp_path):
        from dashboard.app import _build_http_limits

        limits = _build_http_limits(_config(tmp_path, escalation=1, fused=1))
        assert isinstance(limits, httpx.Limits)

    def test_keepalive_expiry_pins_the_shipped_value_above_the_poll_interval(
        self, tmp_path,
    ):
        """Pins the shipped value, plus the one band assertion with a real reason.

        keepalive_expiry is evaluated only lazily, at pool reuse, with no
        background reaper (see
        ``dashboard/src/dashboard/app.py::_HTTP_KEEPALIVE_EXPIRY_SECONDS`` for
        the verified mechanism), so it neither races nor pre-empts any
        server-side close, regardless of its value. Two things are therefore
        worth pinning and nothing else is: the shipped number, so a re-tune
        has to be deliberate and visible; and the one band that carries a
        true, independent rationale — a pooled connection must survive one
        ~3s poll cycle or pooling is defeated and every poll pays a fresh
        handshake.
        """
        from dashboard.app import _HTTP_KEEPALIVE_EXPIRY_SECONDS, _build_http_limits

        expiry = _build_http_limits(_config(tmp_path, escalation=1, fused=1)).keepalive_expiry

        assert expiry is not None
        assert expiry == _HTTP_KEEPALIVE_EXPIRY_SECONDS, (
            f'keepalive_expiry={expiry} must equal the shipped '
            f'_HTTP_KEEPALIVE_EXPIRY_SECONDS={_HTTP_KEEPALIVE_EXPIRY_SECONDS} — '
            f'either a re-tune moved the value without moving this pin, or '
            f'the keepalive_expiry= kwarg was dropped from _build_http_limits '
            f"(which leaves httpx's stock 5.0)"
        )
        assert expiry > _DASHBOARD_POLL_INTERVAL, (
            f'keepalive_expiry={expiry} is at-or-below the ~{_DASHBOARD_POLL_INTERVAL}s '
            f'dashboard poll interval, so a connection would never survive one '
            f'poll cycle before this setting expires it, defeating reuse entirely'
        )

    def test_max_connections_scales_with_endpoint_count(self, tmp_path):
        """The bound is derived from the config, not frozen as a constant."""
        from dashboard.app import _build_http_limits

        small = _build_http_limits(_config(tmp_path, escalation=1, fused=1))
        large = _build_http_limits(_config(tmp_path, escalation=40, fused=8))

        assert large.max_connections is not None
        assert small.max_connections is not None
        assert large.max_connections > small.max_connections, (
            f'a 48-endpoint config must get a larger pool than a 2-endpoint one '
            f'({large.max_connections} vs {small.max_connections}) — otherwise the '
            f'bound silently becomes wrong as projects are onboarded'
        )

    def test_floor_binds_below_the_crossover_and_the_derived_term_takes_over_at_it(
        self, tmp_path,
    ):
        """Pin both sides of the floor-vs-derived boundary.

        Where the crossover sits, and why it matters, is documented once
        beside the sizing constants that
        ``dashboard/src/dashboard/app.py::_build_http_limits`` combines — not
        restated here. What was missing is a test that locates the boundary:
        ``test_max_connections_scales_with_endpoint_count`` only compares 2
        vs 48 endpoints, and ``test_small_install_is_never_tighter_than_httpx_stock``
        only asserts ``>= 100``. A future re-tune of ``_HTTP_CONNS_PER_ENDPOINT``,
        ``_HTTP_ASSUMED_CONCURRENT_VIEWERS`` or ``_HTTP_MIN_CONNECTIONS`` should
        fail here with a legible reason rather than silently moving the crossover.
        """
        from dashboard.app import (
            _HTTP_ASSUMED_CONCURRENT_VIEWERS,
            _HTTP_CONNS_PER_ENDPOINT,
            _build_http_limits,
        )

        # Reported, not asserted: a re-tune's failure message must carry the
        # number the constants now produce, not the one they produced when
        # this test was written.
        per_endpoint = _HTTP_CONNS_PER_ENDPOINT * _HTTP_ASSUMED_CONCURRENT_VIEWERS

        at_floor = _build_http_limits(_config(tmp_path, escalation=5, fused=3))
        assert at_floor.max_connections == _HTTPX_STOCK_MAX_CONNECTIONS, (
            f'8 endpoints: the derived term ({per_endpoint * 8}) must be '
            f'discarded by the floor, so max_connections must equal the httpx '
            f'stock {_HTTPX_STOCK_MAX_CONNECTIONS} exactly — got '
            f'{at_floor.max_connections}'
        )

        past_crossover = _build_http_limits(_config(tmp_path, escalation=5, fused=4))
        assert past_crossover.max_connections == 108, (
            f'9 endpoints: the derived term ({per_endpoint * 9}) must bind and '
            f'exceed the httpx stock {_HTTPX_STOCK_MAX_CONNECTIONS} floor — got '
            f'{past_crossover.max_connections}'
        )

    def test_empty_config_still_gets_a_workable_floor(self, tmp_path):
        """A minimal/empty config must not yield 0 or None connections."""
        from dashboard.app import _HTTP_MIN_CONNECTIONS, _build_http_limits

        limits = _build_http_limits(_config(tmp_path, escalation=0, fused=0))

        assert limits.max_connections is not None, 'max_connections must not be None'
        assert limits.max_connections >= _HTTP_MIN_CONNECTIONS, (
            f'an empty config must still get at least the {_HTTP_MIN_CONNECTIONS} '
            f'floor, got {limits.max_connections}'
        )

    def test_small_install_is_never_tighter_than_httpx_stock(self, tmp_path):
        """A guard on growth must not become a regression for small installs.

        The derived product lands below httpx's stock 100 for a small fleet,
        and every open browser tab drives its OWN 2s poll of merge_halt +
        task_runtime + the live merge queue over every escalation URL — so
        in-flight demand is (tabs x families x projects), not (families x
        projects). Sizing below stock would turn contention that previously
        just queued into httpx.PoolTimeout, which renders as an 'offline' pill
        on a perfectly healthy orchestrator.
        """
        from dashboard.app import _build_http_limits

        stock = _HTTPX_STOCK_MAX_CONNECTIONS

        for kwargs in (
            {'escalation': 0, 'fused': 0},
            {'escalation': 1, 'fused': 1},
            {'escalation': 3, 'fused': 1},  # the reviewer's 3-project install
        ):
            limits = _build_http_limits(_config(tmp_path, **kwargs))
            assert limits.max_connections is not None
            assert limits.max_connections >= stock, (
                f'{kwargs} got {limits.max_connections} connections, tighter than '
                f"httpx's stock {stock} — this change must never shrink the pool "
                f'below what already shipped'
            )

    def test_idle_retention_is_held_flat_as_the_fleet_grows(self, tmp_path):
        """max_keepalive must NOT scale as a fraction of max_connections.

        A ``max_connections // 2`` rule hands a 40-project install 84 idle
        keepalive slots against httpx's stock 20 — the "guard" would LOOSEN
        idle retention for exactly the large fleets it is meant to bound, and
        retention is the dimension the deferred CLOSE-WAIT investigation
        (task 3857) cares about.
        """
        from dashboard.app import _HTTP_MAX_KEEPALIVE_CONNECTIONS, _build_http_limits

        small = _build_http_limits(_config(tmp_path, escalation=1, fused=1))
        large = _build_http_limits(_config(tmp_path, escalation=40, fused=8))

        assert large.max_connections is not None and small.max_connections is not None
        assert large.max_connections > small.max_connections, (
            'concurrency is the dimension that scales with the fleet'
        )
        assert large.max_keepalive_connections == small.max_keepalive_connections, (
            f'idle retention must stay flat as the fleet grows, got '
            f'{small.max_keepalive_connections} -> {large.max_keepalive_connections}'
        )
        assert large.max_keepalive_connections == _HTTP_MAX_KEEPALIVE_CONNECTIONS, (
            f'idle retention must be capped at the named ceiling '
            f'({_HTTP_MAX_KEEPALIVE_CONNECTIONS}), got '
            f'{large.max_keepalive_connections}'
        )
        assert _HTTP_MAX_KEEPALIVE_CONNECTIONS == _HTTPX_STOCK_MAX_KEEPALIVE, (
            "the ceiling is httpx's stock retention, deliberately unchanged"
        )

    def test_keepalive_connections_do_not_exceed_total(self, tmp_path):
        from dashboard.app import _build_http_limits

        for kwargs in ({'escalation': 0, 'fused': 0}, {'escalation': 40, 'fused': 8}):
            limits = _build_http_limits(_config(tmp_path, **kwargs))
            assert limits.max_keepalive_connections is not None
            assert limits.max_connections is not None
            assert limits.max_keepalive_connections <= limits.max_connections, (
                f'keepalive slots ({limits.max_keepalive_connections}) cannot exceed '
                f'total slots ({limits.max_connections}) for {kwargs}'
            )


class TestLifespanWiresTheLimits:
    """The derived limits must actually reach the shared client's constructor.

    The pure helper above proves the NUMBERS are right; this proves they are
    WIRED. Both are needed — a correct helper nobody calls fixes nothing.

    Reaching the lifespan-constructed client requires driving ``lifespan``
    directly (the recipe from test_durability.py): the conftest ``client``
    fixture is function-scoped while ~15 other modules hold module-scoped
    ``TestClient(app)`` fixtures that clobber ``app.state`` — the reason
    spelled out in lifespan's own docstring.
    """

    async def test_lifespan_sizes_the_shared_client_from_the_assigned_config(
        self, tmp_path, monkeypatch,
    ):
        """Assert the SOURCE of the sizing, not just that some sizing happened.

        Comparing the captured kwarg against ``_build_http_limits(
        DashboardConfig.from_env())`` would compare the helper's output to the
        helper's output: it proves a ``limits=`` kwarg is passed, but it would
        still pass if lifespan sized the client from a freshly-constructed
        config rather than from ``app.state.config``. That ordering invariant
        ("Config first: the pool bound is DERIVED from it") is the thing worth
        pinning, so this stubs ``from_env`` to return a config with a
        distinctive endpoint count and asserts the number derived from THAT.
        """
        import dataclasses
        from unittest.mock import AsyncMock, patch

        from _dashboard_helpers import apply_isolated_env
        from fastapi import FastAPI

        from dashboard.app import (
            _HTTP_ASSUMED_CONCURRENT_VIEWERS,
            _HTTP_CONNS_PER_ENDPOINT,
            _HTTP_MAX_KEEPALIVE_CONNECTIONS,
            lifespan,
        )

        apply_isolated_env(monkeypatch, tmp_path)

        # A real from_env() config (valid DB paths etc.), re-pointed at a
        # distinctive 40-orchestrator fleet so the derived number is unique.
        fleet_config = dataclasses.replace(
            DashboardConfig.from_env(),
            escalation_urls={
                f'proj{i}': f'http://127.0.0.1:{18100 + i}' for i in range(40)
            },
            fused_memory_urls=['http://127.0.0.1:18000', 'http://127.0.0.1:18001'],
        )
        endpoints = 42
        expected_max_connections = (
            _HTTP_CONNS_PER_ENDPOINT * _HTTP_ASSUMED_CONCURRENT_VIEWERS * endpoints
        )

        captured: list[dict] = []
        real_async_client = httpx.AsyncClient

        def _recording_async_client(*args, **kwargs):
            captured.append(kwargs)
            # Return a REAL client so shutdown's .aclose()/is_closed still work.
            return real_async_client(*args, **kwargs)

        with (
            patch('dashboard.app.httpx.AsyncClient', _recording_async_client),
            patch(
                'dashboard.app.DashboardConfig.from_env',
                return_value=fleet_config,
            ),
            patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
            patch(
                'dashboard.app.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
        ):
            async with lifespan(FastAPI(lifespan=lifespan)):
                pass

        assert captured, 'lifespan did not construct an httpx.AsyncClient'
        limits = captured[0].get('limits')
        assert isinstance(limits, httpx.Limits), (
            f'lifespan must pass an httpx.Limits, got {limits!r}'
        )
        assert limits.max_connections == expected_max_connections, (
            f'the pool must be sized from the config lifespan assigned to '
            f'app.state.config ({endpoints} endpoints -> '
            f'{expected_max_connections} connections), got '
            f'{limits.max_connections} — a client sized from a different '
            f'config, or from a frozen constant, fails here'
        )
        assert limits.max_keepalive_connections == _HTTP_MAX_KEEPALIVE_CONNECTIONS, (
            'idle retention stays capped even for a 40-orchestrator fleet'
        )
        assert captured[0].get('follow_redirects') is True, (
            'follow_redirects=True must be preserved alongside the new limits'
        )


class TestEndpointBudgetsReachTheMcpLegs:
    """The route-level half of the same bound: per-call budgets, not 10s.

    Lives in this module because it pins the *other* outbound-HTTP bound
    app.py owns under task 3871 (the pool limits above bound how many sockets;
    these bound how long any one request may hold one). The natural homes —
    tests/test_app.py and tests/test_api_curator.py — are outside this task's
    locked modules, so the coverage is kept here rather than dropped.
    """

    def test_api_memory_hands_each_mcp_leg_the_endpoint_budget(self, client):
        from unittest.mock import AsyncMock, patch

        from dashboard.app import _MEMORY_ENDPOINT_TIMEOUT_SECONDS

        status = AsyncMock(return_value={'offline': True, 'error': 'down'})
        queue = AsyncMock(return_value={'counts': {}, 'oldest_pending_age_seconds': None})
        wal = AsyncMock(return_value={'offline': True, 'error': 'down'})

        with (
            patch('dashboard.data.memory.get_memory_status', new=status),
            patch('dashboard.data.memory.get_queue_stats', new=queue),
            patch('dashboard.data.memory.get_wal_status', new=wal),
        ):
            resp = client.get('/api/v2/dashboard/memory')

        assert resp.status_code == 200
        for name, mock in (('status', status), ('queue', queue), ('wal', wal)):
            assert mock.await_args is not None, f'{name} leg was never awaited'
            assert (
                mock.await_args.kwargs.get('timeout')
                == _MEMORY_ENDPOINT_TIMEOUT_SECONDS
            ), (
                f'the {name} leg must get the endpoint budget, not '
                f"mcp_tool_call's 10s default; got "
                f'{mock.await_args.kwargs.get("timeout")!r}'
            )

    def test_api_curator_bounds_the_curator_state_leg(self, client, monkeypatch):
        """Both layers: the budget reaches the callee AND wait_for caps the leg.

        Before this, /curator gathered get_curator_state with neither bound
        while its fan_out_list_tickets sibling honoured 5s — so one dead
        fused-memory instance could hold the endpoint for roughly N x 3 x 10s.
        """
        import time
        from unittest.mock import AsyncMock, patch

        budget = 0.05
        monkeypatch.setattr('dashboard.app._CURATOR_ENDPOINT_TIMEOUT_SECONDS', budget)

        async def _never_returns(*_args, **_kwargs):
            await asyncio.sleep(5.0)
            return {'paused': False}

        state = AsyncMock(side_effect=_never_returns)

        with (
            patch('dashboard.data.memory.get_curator_state', new=state),
            patch(
                'dashboard.app.fan_out_list_tickets',
                new=AsyncMock(return_value=([], 0)),
            ),
        ):
            started = time.monotonic()
            resp = client.get('/api/v2/dashboard/curator')
            elapsed = time.monotonic() - started

        assert resp.status_code == 200, 'a hung leg must degrade, not 500'
        assert elapsed < 2.0, (
            f'the curator_state leg must be capped by asyncio.wait_for; the '
            f'request took {elapsed:.2f}s against a {budget}s budget'
        )
        assert state.await_args is not None, 'the leg was never awaited'
        assert state.await_args.kwargs.get('timeout') == budget, (
            f'the per-call budget must reach the callee too, got '
            f'{state.await_args.kwargs.get("timeout")!r}'
        )
