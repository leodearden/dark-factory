"""Each lifespan's background loops must be bound to the resources IT opened.

WHY this module exists (task 3771 — the runtime half of task 3466):
``app.state`` is a single mutable namespace on the one shared ``FastAPI``
instance, and starlette runs a full lifespan per ``TestClient`` context.  When
two lifespans overlap over that one ``app`` (~15 module-scoped
``TestClient(app)`` fixtures coexist with the function-scoped ``client``
fixture in ``tests/conftest.py``; ``tests/test_fixture_isolation.py`` documents
the module-scoped idiom as deliberate), the INNER lifespan overwrites
``app.state.db`` / ``app.state.http_client`` and does **not** restore them on
exit.  Task 3466 fixed the *shutdown* half by closing locals rather than
``app.state``.  This module pins the *runtime* half: a long-lived loop that
re-reads ``app.state`` on every cycle keeps polling whichever handles are
installed there — which, for the whole remainder of the outer lifespan, are the
inner's already-**closed** pool and HTTP client.  The failure is silent (a
closed ``DbPool.get()`` returns ``None``; a closed ``httpx`` client raises into
``_run_once``'s ``except Exception``), so it surfaces only as a generic
``'Metrics snapshot error'`` warning that masks real faults.

The contract asserted here has two halves, and the asymmetry is deliberate:

* **Handles bind to arguments.** ``pool`` and ``http_client`` are passed into
  ``_metrics_loop`` by the lifespan that created them and are never re-read
  from ``app.state``.  ``lifespan`` likewise binds ``config`` to a local for
  its own startup reads, closing the interleave window across
  ``await burndown_store.open()``.
* **Config binds to ``app.state``.** ``_run_once`` re-reads
  ``app.state.config`` every cycle **on purpose**: ~25 tests swap
  ``client.app.state.config`` mid-test and depend on that being observable.
  A "consistency" refactor that hoists config out of ``_run_once`` must break
  a test here rather than 25 tests elsewhere.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from dashboard.app import _BurndownStore, _metrics_loop, _MetricsStore
from dashboard.config import DashboardConfig


async def _noop_burndown_loop(*args: object, **kwargs: object) -> None:
    """Stand-in for _burndown_loop so nesting two lifespans stays hermetic.

    The burndown loop fans out over HTTP; it is irrelevant to the resource
    binding under test and is replaced so neither lifespan does real I/O.
    """
    return None


@pytest.mark.asyncio
async def test_nested_lifespans_each_get_their_own_pool_and_client() -> None:
    """A lifespan's metrics loop gets the DbPool/AsyncClient THAT lifespan built.

    Nests two ``TestClient(app)`` contexts over the shared global ``app`` — the
    situation the suite creates routinely — and asserts each lifespan's loop was
    handed its own handles, not whatever ``app.state`` points at.
    """
    from dashboard.app import app

    recorded: list[dict[str, Any]] = []

    async def _recording_metrics_loop(
        store: object,
        app_arg: FastAPI,
        *,
        pool: object,
        http_client: object,
    ) -> None:
        # Snapshot both the arguments and app.state AT LOOP START, then return
        # immediately so no real snapshot cycle runs.
        recorded.append(
            {
                'pool': pool,
                'http_client': http_client,
                'state_db': app_arg.state.db,
                'state_http': app_arg.state.http_client,
            }
        )

    with (
        patch('dashboard.app._metrics_loop', new=_recording_metrics_loop),
        patch('dashboard.app._burndown_loop', new=_noop_burndown_loop),
        TestClient(app) as _outer,
    ):
        outer_pool = app.state.db
        outer_http = app.state.http_client
        with TestClient(app) as _inner:
            inner_pool = app.state.db
            inner_http = app.state.http_client
        # The inner lifespan has shut down and closed inner_pool/inner_http,
        # but it did NOT restore app.state.  Capture what the outer lifespan
        # would see if it re-read app.state from here on.
        state_db_after_inner = app.state.db
        state_http_after_inner = app.state.http_client

    assert len(recorded) == 2, (
        f'task 3771: expected 2 lifespans (nested TestClient(app)) to each start a '
        f'metrics loop, got {len(recorded)}'
    )

    # Each loop was handed its own lifespan's handles.
    assert recorded[0]['pool'] is outer_pool, (
        'task 3771: the OUTER lifespan must hand its metrics loop the DbPool it created'
    )
    assert recorded[0]['http_client'] is outer_http, (
        'task 3771: the OUTER lifespan must hand its metrics loop the AsyncClient it created'
    )
    assert recorded[1]['pool'] is inner_pool, (
        'task 3771: the INNER lifespan must hand its metrics loop the DbPool it created'
    )
    assert recorded[1]['http_client'] is inner_http, (
        'task 3771: the INNER lifespan must hand its metrics loop the AsyncClient it created'
    )

    for index, rec in enumerate(recorded):
        assert rec['pool'] is rec['state_db'], (
            f'task 3771: lifespan #{index} passed a pool that is not the one it '
            f'installed on app.state — the argument and app.state disagree at loop start'
        )
        assert rec['http_client'] is rec['state_http'], (
            f'task 3771: lifespan #{index} passed an http_client that is not the one it '
            f'installed on app.state — the argument and app.state disagree at loop start'
        )

    # The two lifespans really did build distinct resources; without this the
    # per-lifespan assertions above could pass on a single shared object.
    assert recorded[0]['pool'] is not recorded[1]['pool'], (
        'task 3771: nested lifespans must build distinct DbPools for this test to mean anything'
    )
    assert recorded[0]['http_client'] is not recorded[1]['http_client'], (
        'task 3771: nested lifespans must build distinct AsyncClients for this test to '
        'mean anything'
    )

    # The defect mechanism, asserted directly: app.state is LEFT pointing at the
    # inner lifespan's (now closed) handles once the inner exits...
    assert state_db_after_inner is inner_pool, (
        'task 3771: precondition — the inner lifespan is expected to leave its own DbPool '
        'installed on app.state after it exits (it does not restore the outer one)'
    )
    assert state_http_after_inner is inner_http, (
        'task 3771: precondition — the inner lifespan is expected to leave its own '
        'AsyncClient installed on app.state after it exits'
    )
    # ...and the outer loop must NOT be reading those closed handles.
    assert recorded[0]['pool'] is not state_db_after_inner, (
        'task 3771 CROSS-TALK: the outer metrics loop is bound to the INNER lifespan closed '
        'DbPool via app.state — pool must be an argument, not an app.state re-read'
    )
    assert recorded[0]['http_client'] is not state_http_after_inner, (
        'task 3771 CROSS-TALK: the outer metrics loop is bound to the INNER lifespan closed '
        'AsyncClient via app.state — http_client must be an argument, not an '
        'app.state re-read'
    )


async def _noop_metrics_loop(*args: object, **kwargs: object) -> None:
    """Stand-in for _metrics_loop where only the burndown binding is under test."""
    return None


@pytest.mark.asyncio
async def test_lifespan_binds_burndown_loop_to_the_config_it_built() -> None:
    """The burndown loop gets the config ITS OWN lifespan built, not a later swap.

    ``lifespan`` assigns ``app.state.config`` at the top of startup but reads it
    back for ``burndown_path``, ``_burndown_loop`` and ``metrics_path`` only
    afterwards -- with ``await burndown_store.open()`` in between.  That await is
    a real suspension point, so a concurrently starting lifespan can install its
    own config before the reads happen, and this lifespan then wires its loops to
    a config it never built.

    The interleave is simulated deterministically by swapping ``app.state.config``
    from inside ``_BurndownStore.open`` -- exactly the window a second lifespan
    would occupy.
    """
    from dashboard.app import app

    real_open = _BurndownStore.open
    captured: dict[str, Any] = {}

    async def _swapping_open(self: _BurndownStore) -> None:
        # Real open first: the lifespan must still get a usable store.
        await real_open(self)
        # Now stand in for an interleaving lifespan reaching this window.
        original = app.state.config
        captured['original'] = original
        # Same project_root, so every derived path stays valid and the lifespan
        # completes -- only the config OBJECT identity differs.
        swapped = DashboardConfig(project_root=original.project_root)
        app.state.config = swapped
        captured['swapped'] = swapped

    recorded: list[Any] = []

    async def _recording_burndown_loop(
        store: object,
        config: object,
        client: object,
    ) -> None:
        recorded.append(config)

    config_before = getattr(app.state, 'config', None)
    try:
        with (
            patch.object(_BurndownStore, 'open', _swapping_open),
            patch('dashboard.app._burndown_loop', new=_recording_burndown_loop),
            patch('dashboard.app._metrics_loop', new=_noop_metrics_loop),
            TestClient(app),
        ):
            pass
    finally:
        # Do not leak a hand-built config into sibling tests; every lifespan
        # rebuilds it from_env anyway, so this only restores the idle state.
        if config_before is not None:
            app.state.config = config_before

    assert 'original' in captured, (
        'task 3771: _BurndownStore.open was never called -- the simulated interleave '
        'never ran, so this test proves nothing'
    )
    assert captured['original'] is not captured['swapped'], (
        'task 3771: the simulated interleave must install a DISTINCT config object'
    )
    assert len(recorded) == 1, (
        f'task 3771: expected exactly one _burndown_loop start per lifespan, '
        f'got {len(recorded)}'
    )
    assert recorded[0] is captured['original'], (
        'task 3771 INTERLEAVE: _burndown_loop received a config installed on app.state '
        'AFTER its own lifespan built one -- lifespan must bind config to a local before '
        'the first await, not re-read app.state.config across it'
    )


@pytest.mark.asyncio
async def test_metrics_loop_still_rereads_config_from_app_state_each_cycle(
    tmp_path: Path,
) -> None:
    """config stays an app.state re-read -- the deliberate half of the asymmetry.

    The handles bind to arguments (see the tests above), but ``config`` must NOT:
    ~25 dashboard tests swap ``client.app.state.config`` mid-test
    (test_tab_escalation_analytics.py, test_escalation_lifecycle_gate.py,
    test_memory_evals_data.py, ...) and depend on the swap being picked up.
    This guard is what makes an over-eager "make it consistent" refactor break
    ONE test here instead of ~25 elsewhere.

    Drives ``_metrics_loop`` directly, reusing the harness in
    test_durability.py::test_metrics_loop_invokes_periodic_checkpoint.
    """
    store = _MetricsStore(tmp_path / 'metrics.db', busy_timeout_ms=5000)
    await store.open()

    # Two REAL configs (not MagicMocks -- check_bare_magicmock_config.py Rule A),
    # distinct in both identity and value.
    config_a = DashboardConfig(project_root=tmp_path)
    config_b = DashboardConfig(project_root=tmp_path / 'swapped')

    mock_pool = MagicMock()
    mock_pool.get = AsyncMock(return_value=None)
    mock_http_client = MagicMock()
    mock_app = MagicMock()
    mock_app.state.config = config_a

    seen_configs: list[Any] = []
    saw_swapped = asyncio.Event()

    async def _recording_collect(*args: object, **kwargs: Any) -> None:
        seen = kwargs['config']
        seen_configs.append(seen)
        if seen is config_a:
            # Stand in for the mid-test swap those ~25 sites perform.
            mock_app.state.config = config_b
        elif seen is config_b:
            # Set from inside the recorder, so the wait below is racefree.
            saw_swapped.set()

    async def _noop_sleep(*a: object, **kw: object) -> None:
        # Must actually suspend.  A plain AsyncMock never yields, creating a
        # tight synchronous loop that starves asyncio.wait_for of event-loop
        # cycles -- see test_durability.py::test_metrics_loop_invokes_periodic_checkpoint.
        await asyncio.sleep(0)

    try:
        with (
            patch(
                'dashboard.app.collect_metrics_snapshot',
                new=AsyncMock(side_effect=_recording_collect),
            ),
            patch('dashboard.app._sleep_to_aligned_tick', new=AsyncMock(side_effect=_noop_sleep)),
        ):
            task = asyncio.create_task(
                _metrics_loop(
                    store,
                    mock_app,
                    pool=mock_pool,
                    http_client=mock_http_client,
                )
            )
            try:
                # Suppressed, not raised: on regression the swap never lands and
                # a bare TimeoutError says nothing.  Falling through lets the
                # named assertions below explain what actually broke.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(saw_swapped.wait(), timeout=2.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    finally:
        await store.close()

    assert seen_configs, 'task 3771: collect_metrics_snapshot was never called'
    assert seen_configs[0] is config_a, (
        'task 3771: the first cycle must use the config installed on app.state at the time'
    )
    assert any(cfg is config_b for cfg in seen_configs), (
        'task 3771: a later cycle never picked up the swapped app.state.config. '
        'config must stay an app.state re-read inside _run_once -- ~25 tests swap '
        'client.app.state.config mid-test and depend on it. Only pool/http_client '
        'bind to arguments.'
    )
