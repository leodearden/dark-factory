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

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient


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
