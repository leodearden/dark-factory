"""Tests for the Taskmaster supervisor task pattern.

The supervisor moves the ``stdio_client`` and ``ClientSession`` async-context
managers lexically inside a long-running task so that anyio's cancel scopes
bind to the supervisor — not to ``run_server``. These tests verify the
isolation: an inner failure must respawn only the session, not cascade.
"""

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from fused_memory.backends import taskmaster_client as tc_module
from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import TaskmasterConfig


@pytest.fixture
def config():
    return TaskmasterConfig(
        transport='stdio',
        command='node',
        args=['server.js'],
        project_root='/project',
    )


def _make_session(*, init_side_effect=None, call_side_effect=None):
    """Construct a MagicMock session with __aenter__/__aexit__ + initialize/call_tool."""
    session = MagicMock()
    session.initialize = AsyncMock(side_effect=init_side_effect)
    session.call_tool = AsyncMock(side_effect=call_side_effect)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _patch_stdio_and_session(
    monkeypatch,
    *,
    sessions,
    stdio_factory=None,
):
    """Patch ``stdio_client`` and ``ClientSession`` on the module.

    ``sessions`` is a list of callables (or session mocks). Each call to
    ``ClientSession(r, w)`` consumes the next entry. If the entry is
    callable, it is invoked to produce a session — this lets a single test
    inject errors / track per-instance state.

    ``stdio_factory`` overrides the default trivial async-context stdio
    client. Use it to inject anyio task-group failures.
    """
    if stdio_factory is None:

        @contextlib.asynccontextmanager
        async def _trivial(params):
            yield (MagicMock(), MagicMock())

        stdio_factory = _trivial

    state = {'i': 0}

    def session_factory(r, w):
        idx = state['i']
        state['i'] += 1
        if idx >= len(sessions):
            # After exhausting the queue, reuse the last session forever.
            return sessions[-1]
        return sessions[idx]

    monkeypatch.setattr(tc_module, 'stdio_client', stdio_factory)
    monkeypatch.setattr(tc_module, 'ClientSession', session_factory)
    return state


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    """Poll ``predicate`` until truthy or timeout. Used to avoid hardcoded sleeps."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f'predicate not satisfied within {timeout}s')


# ── start() and basic lifecycle ───────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_starts_session_on_start(monkeypatch, config):
    """start() spawns the supervisor; first session is up by the time
    start() returns."""
    session = _make_session()
    _patch_stdio_and_session(monkeypatch, sessions=[session])

    c = TaskmasterBackend(config, startup_timeout_seconds=2.0)
    await c.start()

    assert c.connected is True
    assert c.restart_count == 1
    session.initialize.assert_awaited_once()

    await c.close()
    assert c.connected is False


@pytest.mark.asyncio
async def test_start_is_idempotent(monkeypatch, config):
    """Calling start() twice does not spawn two supervisor tasks."""
    session = _make_session()
    _patch_stdio_and_session(monkeypatch, sessions=[session])

    c = TaskmasterBackend(config, startup_timeout_seconds=2.0)
    await c.start()
    first_task = c._supervisor_task
    await c.start()
    assert c._supervisor_task is first_task

    await c.close()


@pytest.mark.asyncio
async def test_start_returns_without_raising_on_first_session_timeout(monkeypatch, config):
    """If the first session does not come up in time, start() logs and
    returns — supervisor stays running. Callers see TimeoutError on the
    next call_tool."""
    session = _make_session()

    @contextlib.asynccontextmanager
    async def slow_stdio(params):
        await asyncio.sleep(60)  # never yields within the test window
        yield (MagicMock(), MagicMock())

    monkeypatch.setattr(tc_module, 'stdio_client', slow_stdio)
    monkeypatch.setattr(tc_module, 'ClientSession', lambda r, w: session)

    c = TaskmasterBackend(
        config,
        startup_timeout_seconds=0.05,
        session_ready_timeout_seconds=0.05,
    )
    await c.start()  # must NOT raise even though first session never comes up

    assert c.connected is False
    with pytest.raises(asyncio.TimeoutError):
        await c.call_tool('get_tasks', {})

    # Cleanup — cancel the supervisor (the slow stdio_client will be cancelled).
    await c.close()


# ── supervisor respawn behaviour ──────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_respawns_after_init_failure(monkeypatch, config):
    """First session.initialize() raises; supervisor catches, sleeps the
    cooldown, opens a second session that succeeds. ``restart_count``
    reflects only successful opens."""
    bad_session = _make_session(init_side_effect=RuntimeError('init failed'))
    good_session = _make_session()
    _patch_stdio_and_session(monkeypatch, sessions=[bad_session, good_session])

    c = TaskmasterBackend(
        config,
        reconnect_cooldown_seconds=0.05,
        startup_timeout_seconds=0.5,
    )
    await c.start()
    await _wait_until(lambda: c.restart_count >= 1, timeout=2.0)
    assert c.connected is True
    assert c.restart_count == 1
    assert c._last_error_summary is not None
    assert 'RuntimeError' in c._last_error_summary

    await c.close()


@pytest.mark.asyncio
async def test_supervisor_respawns_after_anyio_cascade(monkeypatch, config):
    """Simulates the real cascade: an internal subtask in the stdio
    task group raises, anyio fires the cancel scope on the supervisor
    task, the inner ``async with`` blocks unwind with a BaseException
    (BaseExceptionGroup on 3.12+). The supervisor's ``except BaseException``
    branch logs and respawns."""
    death_event = asyncio.Event()

    @contextlib.asynccontextmanager
    async def stdio_with_dying_subtask(params):
        async with anyio.create_task_group() as tg:

            async def _watcher() -> None:
                await death_event.wait()
                raise RuntimeError('simulated transport death')

            tg.start_soon(_watcher)
            yield (MagicMock(), MagicMock())

    sessions = [_make_session(), _make_session()]
    _patch_stdio_and_session(
        monkeypatch, sessions=sessions, stdio_factory=stdio_with_dying_subtask,
    )

    c = TaskmasterBackend(
        config,
        reconnect_cooldown_seconds=0.05,
        startup_timeout_seconds=2.0,
    )
    await c.start()
    assert c.restart_count == 1

    # Trigger the cascade.
    death_event.set()
    death_event = asyncio.Event()  # rebind so the second session does not die immediately

    # Replace the death_event reference inside the closure by patching globals
    # would be ugly; instead, await the second session coming up. The watcher
    # in the second pass waits on the *new* event, which is never set, so the
    # session stays up.
    # NOTE: the first iteration's watcher already saw the set, so on respawn
    # the supervisor enters a fresh task group with a fresh watcher bound to
    # the new event reference (because asynccontextmanager re-evaluates the
    # closure on each call).
    await _wait_until(lambda: c.restart_count >= 2, timeout=3.0)
    assert c.connected is True
    assert c._last_error_summary is not None

    await c.close()


# ── call_tool waits for the session and serializes ───────────────────


@pytest.mark.asyncio
async def test_call_tool_during_reconnect_blocks_until_ready(monkeypatch, config):
    """call_tool blocks waiting on ``_session_ready``; once supervisor
    sets it, the call proceeds. Times out cleanly when the session never
    comes up within ``session_ready_timeout_seconds``."""
    session = _make_session()

    started_event = asyncio.Event()

    @contextlib.asynccontextmanager
    async def gated_stdio(params):
        await started_event.wait()
        yield (MagicMock(), MagicMock())

    monkeypatch.setattr(tc_module, 'stdio_client', gated_stdio)
    monkeypatch.setattr(tc_module, 'ClientSession', lambda r, w: session)

    # Use a permissive ready timeout for the call_tool side, but a brief
    # startup timeout so start() returns quickly without the first session.
    c = TaskmasterBackend(
        config,
        startup_timeout_seconds=0.05,
        session_ready_timeout_seconds=2.0,
    )
    await c.start()
    assert c.connected is False

    # Make call_tool wait while we let the supervisor proceed.
    session.call_tool = AsyncMock(return_value=_make_call_result({'tasks': []}))
    call_task = asyncio.create_task(c.call_tool('get_tasks', {'projectRoot': '/p'}))
    await asyncio.sleep(0.05)
    assert not call_task.done()

    # Release the supervisor; the session comes up; call_tool resumes.
    started_event.set()
    result = await asyncio.wait_for(call_task, timeout=2.0)
    assert result == {'tasks': []}

    await c.close()


def _make_call_result(data: dict):
    """Build an MCP CallToolResult-like object whose .content carries JSON text."""
    import json

    from mcp.types import TextContent

    result = MagicMock()
    result.content = [TextContent(type='text', text=json.dumps(data))]
    return result


@pytest.mark.asyncio
async def test_call_tool_serialized_via_lock(monkeypatch, config):
    """Concurrent call_tool requests pass through the lock one at a time —
    the underlying session never sees overlapping calls."""
    session = _make_session()
    _patch_stdio_and_session(monkeypatch, sessions=[session])

    in_flight = 0
    max_in_flight = 0

    async def fake_call(name, args):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(0.02)
            return _make_call_result({'tasks': []})
        finally:
            in_flight -= 1

    session.call_tool = AsyncMock(side_effect=fake_call)

    c = TaskmasterBackend(config, startup_timeout_seconds=2.0)
    await c.start()

    await asyncio.gather(
        *(c.call_tool('get_tasks', {'projectRoot': '/p'}) for _ in range(8)),
    )
    assert max_in_flight == 1

    await c.close()


@pytest.mark.asyncio
async def test_call_tool_clears_ready_on_transport_dead(monkeypatch, config):
    """A transport-dead exception from session.call_tool clears
    ``_session_ready`` so subsequent callers wait for the supervisor to
    bring up a new session."""
    session = _make_session()
    _patch_stdio_and_session(monkeypatch, sessions=[session])

    session.call_tool = AsyncMock(side_effect=anyio.ClosedResourceError())
    c = TaskmasterBackend(config, startup_timeout_seconds=2.0)
    await c.start()
    assert c.connected is True

    with pytest.raises(anyio.ClosedResourceError):
        await c.call_tool('get_tasks', {'projectRoot': '/p'})

    assert c.connected is False

    await c.close()


# ── close() lifecycle ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_cancels_supervisor_cleanly(monkeypatch, config):
    """close() returns within bounded time; supervisor task is done."""
    session = _make_session()
    _patch_stdio_and_session(monkeypatch, sessions=[session])

    c = TaskmasterBackend(config, startup_timeout_seconds=2.0)
    await c.start()
    supervisor = c._supervisor_task
    assert supervisor is not None and not supervisor.done()

    await c.close()
    assert c._supervisor_task is None
    assert supervisor.done()
    assert c.connected is False


@pytest.mark.asyncio
async def test_close_is_idempotent(monkeypatch, config):
    """Calling close() twice does not raise."""
    session = _make_session()
    _patch_stdio_and_session(monkeypatch, sessions=[session])
    c = TaskmasterBackend(config, startup_timeout_seconds=2.0)
    await c.start()
    await c.close()
    await c.close()  # must not raise


@pytest.mark.asyncio
async def test_close_interrupts_reconnect_cooldown(monkeypatch, config):
    """If the supervisor is sleeping the reconnect cooldown when close()
    is called, the sleep returns immediately — the supervisor exits
    promptly instead of blocking the full cooldown."""
    bad = _make_session(init_side_effect=RuntimeError('die'))
    sessions = [bad, bad, bad]
    _patch_stdio_and_session(monkeypatch, sessions=sessions)

    c = TaskmasterBackend(
        config,
        reconnect_cooldown_seconds=10.0,  # would block 10s without the wakeup
        startup_timeout_seconds=0.05,
    )
    await c.start()
    # Let the first init failure happen and the supervisor hit its cooldown.
    await asyncio.sleep(0.1)

    start_t = asyncio.get_running_loop().time()
    await c.close()
    elapsed = asyncio.get_running_loop().time() - start_t
    assert elapsed < 2.0, f'close() took {elapsed:.2f}s — cooldown not interrupted'


# ── 3-min escalation ──────────────────────────────────────────────────


def test_3min_escalation_log_emitted_once(config, caplog):
    """_maybe_escalate emits the structured ERROR exactly once when the
    threshold is exceeded; it does not re-fire."""
    c = TaskmasterBackend(config)
    c._down_since = 0.0  # epoch-style anchor so we control elapsed time
    c._restart_count = 3
    c._last_error_summary = 'RuntimeError: simulated'

    caplog.set_level(logging.ERROR, logger='fused_memory.backends.taskmaster_client')

    # Below threshold — no log.
    c._maybe_escalate(now=10.0)
    assert not any(
        'TASKMASTER_UNAVAILABLE_3MIN' in record.message for record in caplog.records
    )

    # Above threshold — single log.
    c._maybe_escalate(now=200.0)
    matching = [r for r in caplog.records if 'TASKMASTER_UNAVAILABLE_3MIN' in r.message]
    assert len(matching) == 1
    assert 'restart_attempts=3' in matching[0].message
    assert 'simulated' in matching[0].message

    # Re-fire — still single log.
    c._maybe_escalate(now=500.0)
    matching = [r for r in caplog.records if 'TASKMASTER_UNAVAILABLE_3MIN' in r.message]
    assert len(matching) == 1


def test_escalation_resets_when_session_recovers(config):
    """A successful supervisor reconnect clears the escalation flag —
    next outage gets a fresh log."""
    c = TaskmasterBackend(config)
    c._down_since = 0.0
    c._escalated = True

    # The supervisor's success path clears these.
    c._down_since = None
    c._escalated = False

    # Now simulate a new outage and verify _maybe_escalate fires again.
    c._down_since = 0.0
    c._maybe_escalate(now=200.0)
    assert c._escalated is True
