"""Acceptance tests for /healthz's MCP fan-out (data-plane) check — task 4884 / #4790.

THE BLIND SPOT THIS CLOSES. On 2026-08-27 the dashboard's task fan-out was
wedged for 19.8 hours and ``/healthz`` reported healthy for all of it, because
every check it ran was a CONTROL-PLANE check: thread count, connection count,
three local SQLite probes. Not one of them touched the path that was actually
broken.

THE TRAP, and why both obvious probes are wrong. fused-memory was REACHABLE
throughout the incident — ``/api/v2/dashboard/{memory,scheduler}`` served fine
from the same MCP server over the same shared httpx client — so a probe keyed
on REACHABILITY would have passed for all 19.8 hours. Conversely a probe keyed
on the fetch SUCCEEDING false-alarms: the journal shows routine
ReadTimeout/recovered cycles against localhost:8002 that return the
``{'offline': True, ...}`` marker after ~2.0s, longer than any budget that fits
inside ``_HEALTHZ_TOTAL_BUDGET``. Only "could a caller get THROUGH the cache in
front of the substrate, inside a bound" separates the two, and that is the
property the incident actually violated.

So the probe keys on LATCH TRAVERSAL. A completed fan-out is ``'ok'`` whatever
it returned, the offline marker included.

Every test calls ``dashboard.app.healthz(request)`` directly through
:func:`_call_healthz`, which wraps it in ``asyncio.wait_for`` and turns expiry
into an explicit ``pytest.fail``. The rationale is copied from
``test_healthz_deadline.py``'s module docstring: the house route-test pattern
(the synchronous ``TestClient`` fixture in conftest.py) cannot be wrapped in
``asyncio.wait_for``, so a RED test against a hanging handler would wedge the
whole suite instead of failing fast.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

import dashboard.app as app_module
import dashboard.data.tasks as tasks_module
from dashboard.app import healthz
from dashboard.config import DashboardConfig

# asyncio_mode = "auto" (dashboard/pyproject.toml) collects the async tests
# below without a per-test marker, so no module-level pytestmark is needed --
# and adding one would mis-mark the synchronous test at the end.

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _NoDbPool:
    """A DbPool stand-in whose every target is simply absent.

    ``_probe_db`` maps a ``None`` connection to ``'unavailable'``, which
    ``healthz`` deliberately does NOT treat as unhealthy — so every DB check in
    this module contributes nothing to the verdict and the only thing under
    test is the new ``mcp_fanout`` check.
    """

    open_count = 0

    async def get(self, _db_path: Path):
        return None


def _make_request(config: DashboardConfig, client=None) -> Request:
    """A duck-typed Request carrying exactly what ``healthz`` reads."""
    state = SimpleNamespace(
        db=_NoDbPool(),
        config=config,
        start_time=time.monotonic(),
        http_client=client,
    )
    return cast(Request, SimpleNamespace(app=SimpleNamespace(state=state)))


async def _call_healthz(request: Request, *, hard_cap: float = 10.0) -> tuple[dict, int, float]:
    """Call ``healthz`` under a hard external cap; fail fast+loud on a hang.

    Returns ``(body, status_code, elapsed_seconds)``.
    """
    start = time.monotonic()
    try:
        resp: JSONResponse = await asyncio.wait_for(healthz(request), timeout=hard_cap)
    except TimeoutError:
        pytest.fail(
            f'/healthz did not return within {hard_cap}s. The handler is '
            'expected to deliver a verdict inside _HEALTHZ_TOTAL_BUDGET '
            f'({app_module._HEALTHZ_TOTAL_BUDGET}s) no matter how badly the '
            'MCP fan-out is wedged — that bound is the whole point of the '
            'data-plane probe, not an aspiration.'
        )
    elapsed = time.monotonic() - start
    return json.loads(bytes(resp.body)), resp.status_code, elapsed


def _probe_key(config: DashboardConfig) -> str:
    """The ``_fetch_tasks_cache`` key the probe's own ``fetch_tasks`` call uses.

    Derived through ``_fetch_tasks_cache_key`` rather than spelled as a
    literal: the key covers the narrowing arguments, not just the root, so a
    hand-written copy would silently stop matching if the probe's call shape
    ever changed.
    """
    return tasks_module._fetch_tasks_cache_key(str(config.project_root), None, None, 0, False)


@pytest.fixture
def clean_probe_state():
    """Clear the probe's memoised state and the fetch cache, before AND after.

    Both directions matter: the probe is deliberately single-flight and
    grace-stamped module state, so a warm entry or a grace stamp leaking
    between cases would make a wedged-fan-out test pass for the wrong reason.
    """
    app_module._mcp_probe_state_clear()
    tasks_module._fetch_tasks_cache_clear()
    yield
    app_module._mcp_probe_state_clear()
    tasks_module._fetch_tasks_cache_clear()


@pytest.fixture
def config(tmp_path) -> DashboardConfig:
    return DashboardConfig(project_root=tmp_path)


@pytest.fixture
def wedge_reported_at_once(monkeypatch):
    """Spend the outstanding-probe allowance, so a hang is reported on call 1.

    In production a LIVE probe younger than
    ``_MCP_PROBE_OUTSTANDING_LIMIT`` (one ``_FETCH_TASKS_TTL_SECONDS``, 20.0s)
    reports ``'probing'`` rather than ``'timeout'`` — see that constant, and
    ``test_an_unattended_dashboard_is_never_reported_wedged`` for the false
    alarm it exists to prevent. Every test BELOW is about what /healthz says
    once that allowance is spent, so they zero it instead of sleeping 20s each.

    Zero rather than a small positive number: the assertion is then a property
    of the code path, not of how fast the host got through it.
    """
    monkeypatch.setattr(app_module, '_MCP_PROBE_OUTSTANDING_LIMIT', 0.0)


def _hanging_fetch(entered: list[str] | None = None, gate: asyncio.Event | None = None):
    """A ``fetch_tasks`` stub that never returns until *gate* is set.

    ``await asyncio.Event().wait()`` with nothing to set it is the task-4788
    hang idiom: it has NO duration at all, so no choice of budget value can
    make a pre-fix implementation pass a test built on it.
    """
    ev = gate if gate is not None else asyncio.Event()

    async def _hang(_client, _config, _root, **_kwargs):
        if entered is not None:
            entered.append('enter')
        await ev.wait()
        return []

    return _hang


def _hanging_network_leg(entered: list[str] | None = None):
    """A ``tasks.first_success`` stub — the NETWORK leg, below the cache.

    Patching here rather than at ``fetch_tasks`` is what lets a test exercise
    the REAL ``fetch_tasks`` -> ``TTLCache.get_or_refresh`` -> lock-acquire
    path while still performing no I/O: everything above the fan-out is the
    shipped code, and only the leg that would touch the network is replaced.

    *entered* records each arrival, so a test can assert the probe blocked
    ABOVE this leg (on the latch) rather than inside it.
    """

    async def _hang(*_args, **_kwargs):
        if entered is not None:
            entered.append('enter')
        await asyncio.Event().wait()

    return _hang


# ---------------------------------------------------------------------------
# ACCEPTANCE 1 — a wedged fan-out is REPORTED
# ---------------------------------------------------------------------------


async def test_held_latch_is_reported_degraded(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """ACCEPTANCE 1a — the incident shape: the per-key refresh latch is held.

    This is achievable BECAUSE ``mcp_fanout._LOCK_ACQUIRE_TIMEOUT_SECONDS``
    (15.0, task 4789) is two orders of magnitude above ``_MCP_PROBE_TIMEOUT``
    (0.2). Task 4789's bounded acquire did NOT close this blind spot — it only
    converted an eternal hang into a 15s one, which is still ~75x any budget
    that fits inside ``_HEALTHZ_TOTAL_BUDGET``. A caller arriving during those
    15s is wedged from its own point of view, and reporting otherwise is what
    made the incident invisible.

    ``fetch_tasks`` is deliberately NOT stubbed here — that is the whole
    difference between this test and the hung-refresh one below. The probe
    calls the module-global name directly, so a stub there would bypass the
    cache entirely and leave the lock below inert — making this test a
    duplicate of 1b and leaving the incident's own shape uncovered. Only the
    NETWORK leg is replaced, so the real ``fetch_tasks`` ->
    ``TTLCache.get_or_refresh`` -> bounded-acquire path runs and the held lock
    is what the probe actually blocks on.
    """
    entered_network: list[str] = []
    monkeypatch.setattr(
        tasks_module, 'first_success', _hanging_network_leg(entered_network),
    )

    key = _probe_key(config)
    lock = tasks_module._fetch_tasks_cache._locks.setdefault(key, asyncio.Lock())
    await lock.acquire()
    try:
        body, status, _elapsed = await _call_healthz(_make_request(config))
        # Asserted while the lock is STILL HELD, which is the only window in
        # which it means anything: it is what separates this test from 1b
        # below. The probe never reached the network leg, so the thing it
        # could not get past was the LATCH.
        assert entered_network == [], (
            'the probe reached the network leg, so the held lock was not what '
            'blocked it — this test has silently become a duplicate of the '
            'hung-refresh case and the incident shape is uncovered again'
        )
    finally:
        lock.release()

    assert status == 503, body
    assert body['status'] == 'degraded'
    assert body['checks']['mcp_fanout'] == 'timeout', (
        'a caller that cannot traverse the refresh latch inside the budget is '
        'a wedged data plane, and /healthz must say so — reporting healthy '
        'here is precisely the 19.8h blind spot'
    )
    assert body['checks']['deadline_exceeded'] is True


async def test_hung_refresh_is_reported_degraded(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """ACCEPTANCE 1b — the residual unbounded path: the refresh itself hangs.

    Task 4789 bounded the ACQUIRE, not the refresh. A refresh that never
    returns still stores nothing for the key, so the warm signal stays false
    forever and no later caller is ever served — exactly what
    ``get_or_refresh``'s own docstring records about the incident ("nothing was
    ever stored for the wedged key").
    """
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch())

    body, status, _elapsed = await _call_healthz(_make_request(config))

    assert status == 503, body
    assert body['status'] == 'degraded'
    assert body['checks']['mcp_fanout'] == 'timeout'
    assert body['checks']['deadline_exceeded'] is True


# ---------------------------------------------------------------------------
# ACCEPTANCE 2 — routine slowness must NOT trip it
# ---------------------------------------------------------------------------


async def test_warm_cache_costs_no_mcp_call(config, clean_probe_state, monkeypatch):
    """ACCEPTANCE 2a — a stored value proves a refresh COMPLETED inside the TTL.

    With the browser polling every 3s the cache is warm essentially always, so
    the steady-state probe costs ZERO MCP calls. That is what makes a 0.2s
    budget affordable at all.
    """

    async def _must_not_be_called(*_a, **_k):
        pytest.fail(
            'the probe issued an MCP call while a FRESH entry existed for its '
            'own key. A stored value already proves a refresh completed inside '
            'the TTL; re-fetching to learn what the cache already knows would '
            'put a per-watchdog-tick MCP call on the health path.'
        )

    monkeypatch.setattr(app_module, 'fetch_tasks', _must_not_be_called)
    tasks_module._fetch_tasks_cache._store[_probe_key(config)] = (time.monotonic(), [])

    body, status, _elapsed = await _call_healthz(_make_request(config))

    assert status == 200, body
    assert body['status'] == 'healthy'
    assert body['checks']['mcp_fanout'] == 'ok'


async def test_offline_marker_is_still_a_completed_fanout(config, clean_probe_state, monkeypatch):
    """ACCEPTANCE 2b — routine MCP slowness must not flip the verdict.

    The journal's routine cycle is ``fetch_tasks[...] failed for
    http://localhost:8002: ReadTimeout`` followed by a recovery. That returns
    the ``{'offline': True, ...}`` marker — a COMPLETED traversal of the latch,
    which is the property under test. Keying on the fetch SUCCEEDING would
    false-alarm on every one of those cycles AND would have passed throughout
    the 19.8h wedge, since fused-memory was reachable the whole time.
    """

    async def _offline(_client, _config, _root, **_kwargs):
        await asyncio.sleep(0.01)
        return {'offline': True, 'error': 'ReadTimeout'}

    monkeypatch.setattr(app_module, 'fetch_tasks', _offline)

    body, status, _elapsed = await _call_healthz(_make_request(config))

    assert status == 200, body
    assert body['status'] == 'healthy'
    assert body['checks']['mcp_fanout'] == 'ok', (
        'the offline MARKER is a completed fan-out: the caller got through the '
        'latch and back with an answer. /healthz reports whether the data '
        'plane is traversable, not whether the substrate is happy.'
    )


async def test_grace_absorbs_a_blip_and_is_bounded(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """ACCEPTANCE 2c — the grace window exists AND is bounded.

    A recently-observed completion is enough to answer the next probe for free,
    which is what stops one expired cache entry between two browser polls from
    reading as a wedge. But the grace must be forgettable, or a system that
    completed once at boot would report healthy forever.
    """

    async def _fast(_client, _config, _root, **_kwargs):
        await asyncio.sleep(0.01)
        return []

    monkeypatch.setattr(app_module, 'fetch_tasks', _fast)
    body, status, _ = await _call_healthz(_make_request(config))
    assert status == 200 and body['checks']['mcp_fanout'] == 'ok'

    # The stub now hangs, and the cache is cold — only the grace stamp can
    # carry this call.
    tasks_module._fetch_tasks_cache_clear()
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch())
    body, status, _ = await _call_healthz(_make_request(config))
    assert status == 200, body
    assert body['checks']['mcp_fanout'] == 'ok', (
        'a completion observed within _MCP_FANOUT_OK_GRACE_SECONDS must absorb '
        'the next probe, or an ordinary TTL expiry between two 3s polls would '
        'be reported as a wedge'
    )

    # Forget the stamp: the same hanging stub must now be reported.
    app_module._mcp_probe_state_clear()
    body, status, _ = await _call_healthz(_make_request(config))
    assert status == 503, body
    assert body['checks']['mcp_fanout'] == 'timeout', (
        'the grace must be a bounded window, not a latch: once forgotten, a '
        'hung fan-out is reported again'
    )


async def test_an_unattended_dashboard_is_never_reported_wedged(
    config, clean_probe_state, monkeypatch,
):
    """ACCEPTANCE 2d — an idle but healthy dashboard must not 503 on a cadence.

    Nothing SERVER-side refreshes the primary root's full-tree cache key: the
    only steady-state writer is the browser's 3s
    ``/api/v2/dashboard/orchestrators`` poll. With no browser attached, warmth
    (one TTL) and grace (30s) therefore BOTH lapse on a perfectly healthy
    system, and every /healthz arriving after they do starts a probe it cannot
    observe inside 0.2s. Reporting THAT as a wedge hands an operator polling
    /healthz a strict 503/200 alternation on an idle dashboard — a false alarm
    on the exact signal this check adds, and the surest way to teach them to
    ignore it before the next real wedge.

    Deliberately does NOT take ``wedge_reported_at_once``: the shipped value of
    ``_MCP_PROBE_OUTSTANDING_LIMIT`` is what is under test.
    """
    entered: list[str] = []
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch(entered=entered))

    body, status, _ = await _call_healthz(_make_request(config))

    assert status == 200, body
    assert body['checks']['mcp_fanout'] == 'probing', (
        "a probe started moments ago has demonstrated nothing, and 'timeout' "
        'claims it has — on an unattended dashboard that claim recurs every '
        'time the cache expires, for as long as the dashboard stays healthy'
    )
    assert body['checks']['deadline_exceeded'] is False

    # NOT an unconditional pass: the SAME live probe, once it has been
    # outstanding past the limit, is reported — otherwise this fix would have
    # reopened the 19.8h blind spot it sits inside.
    monkeypatch.setattr(app_module, '_MCP_PROBE_OUTSTANDING_LIMIT', 0.0)
    body, status, _ = await _call_healthz(_make_request(config))

    assert status == 503, body
    assert body['checks']['mcp_fanout'] == 'timeout'
    assert body['checks']['deadline_exceeded'] is True
    assert len(entered) == 1, (
        f'fetch_tasks was entered {len(entered)} times; both verdicts above '
        'must describe the SAME single-flight probe, not two'
    )


# ---------------------------------------------------------------------------
# ACCEPTANCE 3 — worst case stays inside the envelope
# ---------------------------------------------------------------------------


async def test_worst_case_stays_inside_the_total_budget(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """ACCEPTANCE 3 — a wedged fan-out costs the budget, and leaks no thread.

    The whole design constraint of this probe is that it fits inside the
    EXISTING envelope: ``_DB_PROBE_TIMEOUT * 3 + _MCP_PROBE_TIMEOUT =
    0.9*3 + 0.2 = 2.9 <= 3.0``. Nothing was widened to make room.
    """
    # The STUB is the hang here, so no lock is taken: a held lock is inert
    # once fetch_tasks itself is replaced (the probe calls that module-global
    # name directly), and leaving one in place would suggest this test covers
    # the latch shape when 1a is what does.
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch())
    threads_before = threading.active_count()

    body, status, elapsed = await _call_healthz(_make_request(config))

    assert status == 503
    assert body['checks']['mcp_fanout'] == 'timeout'
    assert elapsed <= app_module._HEALTHZ_TOTAL_BUDGET + 0.5, (
        f'/healthz took {elapsed:.3f}s against a wedged fan-out, over its own '
        f'stated budget of {app_module._HEALTHZ_TOTAL_BUDGET}s. A verdict that '
        'arrives after the caller gave up is not a verdict (measured 50.6s '
        'before the whole-handler deadline existed).'
    )

    for _ in range(50):
        if threading.active_count() <= threads_before:
            break
        await asyncio.sleep(0.02)
    assert threading.active_count() <= threads_before, (
        'the fan-out probe leaked a thread; /healthz must not manufacture the '
        'very leak its _THREAD_LIMIT check exists to detect'
    )


async def test_probe_is_single_flight(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """Three /healthz calls against a wedged fan-out cost ONE fetch_tasks entry.

    The watchdog timer fires every 30s. A probe that launched a fresh doomed
    task per call would leak one task per tick for as long as the wedge lasted
    — 19.8 hours of them, in the incident that motivated this check.
    """
    entered: list[str] = []
    # No lock: the stub is the hang (see the note in the budget test above).
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch(entered=entered))

    for _ in range(3):
        body, status, _ = await _call_healthz(_make_request(config))
        assert status == 503
        assert body['checks']['mcp_fanout'] == 'timeout'

    assert len(entered) == 1, (
        f'fetch_tasks was entered {len(entered)} times across three /healthz '
        'calls; EXACTLY one is the claim, and both directions are failures: 0 '
        'means the probe never launched a fetch at all (so the three 503s '
        'above were reported without probing anything, and this test would '
        'pass against a probe that had stopped probing), >1 means it started '
        'a new doomed task per call instead of JOINING the in-flight one'
    )


async def test_probe_task_is_not_cancelled_at_expiry(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """NO-CANCEL — a healthy-but-cold system self-heals in one cycle.

    Budget expiry means "not yet", not "abort". The background task is left
    running, so the very next /healthz observes its completion for free. A
    probe that cancelled on expiry would relaunch a doomed task every call and
    a cold system could never report healthy; a wedged key still never
    completes, so ``'timeout'`` persists — which is exactly the fact the 19.8h
    blind spot needed reported.
    """
    entered: list[str] = []
    gate = asyncio.Event()
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch(entered=entered, gate=gate))

    body, status, _ = await _call_healthz(_make_request(config))
    assert status == 503
    assert body['checks']['mcp_fanout'] == 'timeout'

    probe = app_module._mcp_probe
    assert probe is not None and not probe.task.done(), (
        'the probe task must still be live after budget expiry — it was '
        'cancelled or dropped'
    )
    live = probe.task
    assert live in app_module._MCP_PROBES, (
        'the event loop holds only a WEAK reference to a Task, so an '
        'unreferenced one can be garbage-collected mid-flight; the probe must '
        'hold a strong reference exactly as _ABANDONED_PROBES/track_task do'
    )

    # Let the SAME task finish. Nothing is re-entered.
    gate.set()
    await asyncio.wait({live}, timeout=5)

    body, status, _ = await _call_healthz(_make_request(config))
    assert status == 200, body
    assert body['checks']['mcp_fanout'] == 'ok'
    assert len(entered) == 1, (
        f'fetch_tasks was entered {len(entered)} times; the completion the '
        'second call observed must be the FIRST call\'s abandoned task, not a '
        'relaunch — that is what makes a cold start self-heal in one cycle '
        'instead of costing an MCP call per watchdog tick forever'
    )


# ---------------------------------------------------------------------------
# The three NON-'timeout' verdict branches
# ---------------------------------------------------------------------------


async def test_a_raising_fetch_is_error_and_not_a_missed_deadline(
    config, clean_probe_state, monkeypatch, caplog,
):
    """A probe that fails FAST is 'error', and must NOT set deadline_exceeded.

    The same invariant the DB loop carries (and has its own test for), stated
    on ``healthz``'s own comment as load-bearing: ``deadline_exceeded`` is
    keyed on ``'timeout'`` ALONE. Folding ``'error'`` back into it would make
    the field lie about WHICH bound was hit, and an operator reading a 503
    would go looking for a slow fan-out instead of a raising one.
    """

    async def _boom(_client, _config, _root, **_kwargs):
        raise RuntimeError('fused-memory transport exploded')

    monkeypatch.setattr(app_module, 'fetch_tasks', _boom)

    with caplog.at_level(logging.WARNING):
        body, status, _ = await _call_healthz(_make_request(config))

    assert status == 503, body
    assert body['checks']['mcp_fanout'] == 'error'
    assert body['checks']['deadline_exceeded'] is False, (
        'the probe RAISED — it did not blow the budget. deadline_exceeded is '
        'the flag an operator reads to decide whether the handler ran out of '
        'time, and it must mean exactly that.'
    )
    assert any('MCP fan-out probe raised' in r.getMessage() for r in caplog.records), (
        'the payload can only carry a status string, so an exception that is '
        'not logged here is unrecoverable: the operator sees "error" and has '
        f'no traceback to act on. Saw: {[r.getMessage() for r in caplog.records]}'
    )


async def test_a_spent_budget_reports_deadline_exceeded_without_probing(
    config, clean_probe_state, monkeypatch,
):
    """With no budget left, /healthz names the miss and starts NO probe.

    Launching a fan-out it has already run out of time to observe would cost
    an MCP round trip per watchdog tick and still report nothing.
    """

    async def _must_not_be_called(*_args, **_kwargs):
        pytest.fail(
            '/healthz launched an MCP probe with none of its budget left — '
            'the remaining<=0 branch must report the miss, not start work it '
            'cannot wait for'
        )

    monkeypatch.setattr(app_module, 'fetch_tasks', _must_not_be_called)
    monkeypatch.setattr(app_module, '_HEALTHZ_TOTAL_BUDGET', 0.0)

    body, status, _ = await _call_healthz(_make_request(config))

    assert status == 503, body
    assert body['checks']['mcp_fanout'] == 'deadline_exceeded'
    assert body['checks']['deadline_exceeded'] is True


async def test_a_cancelled_probe_is_reported_error_not_laundered(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """A probe cancelled out from under the handler is reported, not absorbed.

    Only ``_mcp_probe_state_clear`` or loop shutdown cancels a probe, so a
    cancelled one means something about THIS PROCESS is unusual — reporting it
    as 'ok' (or as a wedge) would launder a fact about the dashboard into a
    verdict about the data plane, which is exactly the confusion the
    'timeout' vs 'error' split exists to prevent.
    """
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch())

    body, _status, _ = await _call_healthz(_make_request(config))
    assert body['checks']['mcp_fanout'] == 'timeout'

    probe = app_module._mcp_probe
    assert probe is not None, 'the expired probe must still be held for the next call'
    probe.task.cancel()
    # Let the cancellation LAND — a cancel that has only been requested leaves
    # the task pending, and the next call would report 'timeout' again.
    await asyncio.wait({probe.task}, timeout=5)
    assert probe.task.cancelled()

    body, status, _ = await _call_healthz(_make_request(config))

    assert status == 503, body
    assert body['checks']['mcp_fanout'] == 'error'
    assert body['checks']['deadline_exceeded'] is False, (
        'a cancelled probe did not blow the budget'
    )


# ---------------------------------------------------------------------------
# The new constants must be READ, which only a behavioural test can show
# ---------------------------------------------------------------------------
#
# The earlier form of these two tests substring-matched the constant NAMES
# against ``inspect.getsource(_probe_mcp_fanout) + inspect.getsource(healthz)``,
# which includes docstrings and comments — and both names appear in
# _probe_mcp_fanout's own docstring, so deleting the real read left the test
# green. A constant is live when CHANGING it changes behaviour; nothing weaker
# distinguishes READ from MENTIONED.


async def test_the_grace_window_constant_is_read_not_merely_declared(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """Zeroing _MCP_FANOUT_OK_GRACE_SECONDS must make a graced probe re-probe."""

    async def _fast(_client, _config, _root, **_kwargs):
        return []

    monkeypatch.setattr(app_module, 'fetch_tasks', _fast)
    body, status, _ = await _call_healthz(_make_request(config))
    assert status == 200 and body['checks']['mcp_fanout'] == 'ok'

    # Cold cache and a hanging fan-out: only the grace stamp can answer now.
    tasks_module._fetch_tasks_cache_clear()
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch())
    body, status, _ = await _call_healthz(_make_request(config))
    assert body['checks']['mcp_fanout'] == 'ok', body

    # IDENTICAL state, zero grace. The stamp is deliberately NOT cleared —
    # that is what makes this a test of the CONSTANT rather than of the hook.
    monkeypatch.setattr(app_module, '_MCP_FANOUT_OK_GRACE_SECONDS', 0.0)
    body, status, _ = await _call_healthz(_make_request(config))

    assert status == 503, body
    assert body['checks']['mcp_fanout'] == 'timeout', (
        'with the grace window at 0.0 a stamp must answer for nothing; a '
        'verdict that does not move when the constant does is a verdict that '
        'never read it'
    )


async def test_the_probe_budget_constant_is_read_not_merely_declared(
    config, clean_probe_state, wedge_reported_at_once, monkeypatch,
):
    """Widening _MCP_PROBE_TIMEOUT must widen the observed wait."""
    monkeypatch.setattr(app_module, 'fetch_tasks', _hanging_fetch())

    monkeypatch.setattr(app_module, '_MCP_PROBE_TIMEOUT', 0.0)
    body, _status, instant = await _call_healthz(_make_request(config))
    assert body['checks']['mcp_fanout'] == 'timeout'

    app_module._mcp_probe_state_clear()
    long_budget = 0.6
    monkeypatch.setattr(app_module, '_MCP_PROBE_TIMEOUT', long_budget)
    body, _status, waited = await _call_healthz(_make_request(config))
    assert body['checks']['mcp_fanout'] == 'timeout'

    # A LOWER bound, which is the race-free half: asyncio.wait cannot return
    # before its own timeout, so a handler that ignored this constant (or that
    # pinned the shipped 0.2) could not possibly spend 0.6s here.
    assert waited >= long_budget, (
        f'/healthz answered a wedged fan-out in {waited:.3f}s with '
        f'_MCP_PROBE_TIMEOUT={long_budget} — the probe is not waiting on the '
        'constant it claims to be bounded by'
    )
    # The pairing rules out a handler that simply always waits: the
    # zero-budget call has 0.6s of slack against a call that does no I/O.
    assert instant < long_budget, (
        f'a zero-budget probe took {instant:.3f}s; the wait above is not '
        'coming from _MCP_PROBE_TIMEOUT at all'
    )


# ---------------------------------------------------------------------------
# THE RESTART DECISION (#4790 acceptance 4), pinned as configuration
# ---------------------------------------------------------------------------


def test_watchdog_probes_the_shallow_endpoint_not_healthz(monkeypatch):
    """A data-plane 503 must NOT become an automatic dashboard restart.

    This is a CONFIGURATION assertion, not a prose pin: it fails the moment
    someone repoints the watchdog at ``/healthz``, which is exactly the change
    that would convert this task's new 503 into a restart loop — and it forces
    them to read the decision comment on ``healthz``'s ``mcp_fanout`` check
    before doing it.

    Probing ``/healthz`` WAS the 2026-07-30 defect: 192 restarts in 3 hours,
    ~27% downtime, from a service that was serving requests throughout. Moving
    the watchdog to the shallow ``/api/health`` handler was the fix. Repointing
    it requires RE-OPENING that decision on evidence, not deleting this test.
    """
    import importlib.util
    import pathlib

    monkeypatch.delenv('DASHBOARD_WATCHDOG_PROBE_URL', raising=False)
    watchdog_path = pathlib.Path(__file__).parents[2] / 'scripts' / 'dashboard-watchdog.py'
    spec = importlib.util.spec_from_file_location('dashboard_watchdog', watchdog_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.PROBE_URL.endswith('/api/health'), (
        f'dashboard-watchdog.py PROBE_URL defaults to {mod.PROBE_URL!r}; it '
        'must remain the SHALLOW no-I/O handler. See the 2026-07-30 incident '
        '(192 restarts in 3h, ~27% downtime).'
    )
    assert 'healthz' not in mod.PROBE_URL, (
        f'the watchdog is probing {mod.PROBE_URL!r}. /healthz is a deep '
        'diagnostic wired to nothing that kills '
        '(plans/dashboard-availability-prd.md task epsilon, Resolved-decision '
        '1), and task 4884 added an MCP fan-out check to it. Pointing the '
        'restart path at it would make a fused-memory outage restart the '
        'dashboard in a loop — which fixes nothing, because the substrate is '
        'not in this process.'
    )

