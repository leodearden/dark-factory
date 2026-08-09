"""Tests for the shared httpx client's connection-pool bound (task 3871).

The dashboard shares ONE ``httpx.AsyncClient`` across every fan-out
(merge_halt, task_runtime, live merge queue, the metrics samplers). It was
constructed with no ``limits=``, so it inherited httpx's stock
``DEFAULT_LIMITS`` — ``max_connections=100``, ``max_keepalive_connections=20``,
``keepalive_expiry=5.0``.

Two things are wrong with the stock defaults *for this client specifically*:

1. The bound is a fixed 100 regardless of how many orchestrators are
   onboarded, so it is simultaneously too loose for a one-project install and
   unrelated to the real peak (projects x concurrent endpoint families).
2. ``keepalive_expiry=5.0`` is a DEAD TIE with the server side: the escalation
   MCP servers — one per orchestrator, and the target of most of the
   dashboard's fan-out — are served by uvicorn with no ``timeout_keep_alive``
   override, and uvicorn's default is 5s. The client therefore considers a
   connection reusable at precisely the moment the server may close it.

This is a GUARD on worst-case pool growth, not a leak fix. It does NOT fix
CLOSE-WAIT accumulation (owned by the task-3857 re-spec).

``_build_http_limits`` is a PURE helper so the sizing is directly testable:
``httpx.AsyncClient`` exposes no public accessor for its limits, so the only
alternative is asserting on ``client._transport._pool._max_connections``,
which is private and brittle across httpx versions.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from dashboard.config import DashboardConfig

# The uvicorn `timeout_keep_alive` default that the escalation MCP servers run
# on (orchestrator/src/orchestrator/harness.py sets no override). The client's
# keepalive_expiry must sit strictly BELOW this or it races the server's close.
_UVICORN_KEEPALIVE_DEFAULT = 5.0

# merge_halt.py's module docstring names a "3s polling loop". keepalive_expiry
# must sit strictly ABOVE this or a connection never survives one poll cycle
# and every cycle pays a fresh handshake.
_DASHBOARD_POLL_INTERVAL = 3.0

# httpx's stock DEFAULT_LIMITS (httpx/_config.py): max_connections=100,
# max_keepalive_connections=20. Spelled out here rather than read off
# `httpx.Limits()`, whose no-arg defaults are None — DEFAULT_LIMITS is a
# separate module constant that httpx does not re-export publicly.
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

    def test_keepalive_expiry_undercuts_the_server_and_clears_the_poll(
        self, tmp_path,
    ):
        """Strictly below uvicorn's 5s close, strictly above the ~3s poll."""
        from dashboard.app import _build_http_limits

        expiry = _build_http_limits(_config(tmp_path, escalation=1, fused=1)).keepalive_expiry

        assert expiry is not None, 'keepalive_expiry must be set, not left None'
        assert expiry < _UVICORN_KEEPALIVE_DEFAULT, (
            f'keepalive_expiry={expiry} does not undercut the {_UVICORN_KEEPALIVE_DEFAULT}s '
            f"uvicorn timeout_keep_alive default the escalation MCP servers run on — "
            f"httpx's own default is exactly 5.0, a dead tie with the server's close"
        )
        assert expiry > _DASHBOARD_POLL_INTERVAL, (
            f'keepalive_expiry={expiry} is at-or-below the ~{_DASHBOARD_POLL_INTERVAL}s '
            f'dashboard poll interval, so a connection would never survive one '
            f'poll cycle and every cycle would pay a fresh handshake'
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
