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

    async def test_lifespan_passes_derived_limits_to_the_shared_client(
        self, tmp_path, monkeypatch,
    ):
        from unittest.mock import AsyncMock, patch

        from _dashboard_helpers import apply_isolated_env
        from fastapi import FastAPI

        from dashboard.app import _build_http_limits, lifespan

        apply_isolated_env(monkeypatch, tmp_path)

        captured: list[dict] = []
        real_async_client = httpx.AsyncClient

        def _recording_async_client(*args, **kwargs):
            captured.append(kwargs)
            # Return a REAL client so shutdown's .aclose()/is_closed still work.
            return real_async_client(*args, **kwargs)

        with (
            patch('dashboard.app.httpx.AsyncClient', _recording_async_client),
            patch('dashboard.app.collect_snapshot', new=AsyncMock(return_value=None)),
            patch(
                'dashboard.app.collect_metrics_snapshot',
                new=AsyncMock(return_value=None),
            ),
        ):
            async with lifespan(FastAPI(lifespan=lifespan)):
                pass

        assert captured, 'lifespan did not construct an httpx.AsyncClient'
        kwargs = captured[0]
        expected = _build_http_limits(DashboardConfig.from_env())
        assert kwargs.get('limits') == expected, (
            f'lifespan must size the shared client from the config; got '
            f'limits={kwargs.get("limits")!r}, expected {expected!r}'
        )
        assert kwargs.get('follow_redirects') is True, (
            'follow_redirects=True must be preserved alongside the new limits'
        )
