"""The reconciliation harness's escalation server serves, and stops with uvicorn's own clean shutdown rather than a cancel (task 10067)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from _embedded_uvicorn_fixtures import LoggerState, uvicorn_loggers_set_to
from arm_harness._loopback_http import unused_port
from escalation.queue import EscalationQueue
from starlette.applications import Starlette

import fused_memory.reconciliation.harness as harness_module
import fused_memory.server.main as main_module
from reconciliation.escalation_server_capture import serving_escalation_server
from reconciliation.harness_fixtures import make_harness_config, open_harness


@pytest.fixture
def uvicorn_loggers_as_main_py_leaves_them() -> Iterator[None]:
    """'uvicorn' and 'uvicorn.error' with no handler or level of their own, propagating to the root logger main.py configures.

    An earlier test in the same worker may have built a plain uvicorn.Config,
    whose dictConfig rewires these loggers process-wide.
    """
    propagating = LoggerState(handlers=(), level=logging.NOTSET, propagate=True)
    with uvicorn_loggers_set_to({'uvicorn': propagating, 'uvicorn.error': propagating}):
        yield


@pytest.fixture
def systemd_style_listener() -> Iterator[socket.socket]:
    """A socket bound to a free 127.0.0.1 port and already listening, as systemd hands it over."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        yield listener


def _uvicorn_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name.startswith('uvicorn')]


@pytest.mark.asyncio
@pytest.mark.usefixtures('uvicorn_loggers_as_main_py_leaves_them')
async def test_a_started_escalation_server_stops_with_a_clean_uvicorn_shutdown(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    systemd_style_listener: socket.socket,
) -> None:
    caplog.set_level(logging.INFO, logger='uvicorn.error')
    host, port = systemd_style_listener.getsockname()

    async with (
        open_harness(
            tmp_path,
            taskmaster=None,
            escalation_listener=systemd_style_listener,
            escalation_queue=EscalationQueue(tmp_path / 'esc'),
        ) as harness,
        serving_escalation_server(harness),
        httpx.AsyncClient(timeout=None) as client,
    ):
        await client.get(f'http://{host}:{port}/mcp')

    assert [
        record
        for record in caplog.records
        if record.name == 'uvicorn.error'
        and record.levelno == logging.INFO
        and record.getMessage() == 'Application shutdown complete.'
    ], _uvicorn_messages(caplog)


@pytest.mark.asyncio
async def test_without_a_listener_the_escalation_server_binds_its_configured_port(tmp_path) -> None:
    port = unused_port()

    async with (
        open_harness(
            tmp_path,
            taskmaster=None,
            config=make_harness_config(escalation_host='127.0.0.1', escalation_port=port),
            escalation_queue=EscalationQueue(tmp_path / 'esc'),
        ) as harness,
        serving_escalation_server(harness),
        httpx.AsyncClient(timeout=None) as client,
    ):
        response = await client.get(f'http://127.0.0.1:{port}/mcp')

    assert response.json()['jsonrpc'] == '2.0', response.text


@pytest.mark.asyncio
async def test_a_started_server_not_stopped_by_the_stop_bound_is_cancelled(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    systemd_style_listener: socket.socket,
) -> None:
    """The app's shutdown never completes, so the server cannot stop by any bound; a zero bound keeps the test instant."""
    caplog.set_level(logging.WARNING, logger=harness_module.__name__)

    @contextlib.asynccontextmanager
    async def lifespan_whose_shutdown_never_completes(app: Starlette) -> AsyncIterator[None]:
        yield
        await asyncio.get_running_loop().create_future()

    async with (
        open_harness(
            tmp_path,
            taskmaster=None,
            escalation_listener=systemd_style_listener,
            escalation_queue=EscalationQueue(tmp_path / 'esc'),
        ) as harness,
        serving_escalation_server(
            harness,
            app=Starlette(lifespan=lifespan_whose_shutdown_never_completes),
            stop_bound_secs=0,
        ),
    ):
        pass

    assert [
        record
        for record in caplog.records
        if record.name == harness_module.__name__
        and record.levelno == logging.WARNING
        and 'did not shut down within' in record.getMessage()
    ], [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
async def test_cancelling_a_graceful_stop_cancels_the_server_it_was_waiting_for(
    tmp_path,
    systemd_style_listener: socket.socket,
) -> None:
    """main.py's _run_shielded cancels the harness when its timeout runs out, which lands here mid-stop."""
    shutdown_requested = asyncio.Event()

    @contextlib.asynccontextmanager
    async def lifespan_whose_shutdown_never_completes(app: Starlette) -> AsyncIterator[None]:
        yield
        shutdown_requested.set()
        await asyncio.get_running_loop().create_future()

    async with open_harness(
        tmp_path,
        taskmaster=None,
        escalation_listener=systemd_style_listener,
        escalation_queue=EscalationQueue(tmp_path / 'esc'),
    ) as harness:

        async def serve_then_stop() -> None:
            async with serving_escalation_server(harness, app=Starlette(lifespan=lifespan_whose_shutdown_never_completes)):
                pass

        stopping = asyncio.ensure_future(serve_then_stop())
        await shutdown_requested.wait()
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping

        assert not [
            task
            for task in asyncio.all_tasks()
            if task.get_name() == 'recon-escalation-server' and not task.cancelling()
        ]


def test_the_graceful_stop_bound_fits_inside_the_harness_cancel_timeout() -> None:
    """main.py cancels the harness after _HARNESS_CANCEL_TIMEOUT, and the harness's finally runs the graceful
    escalation stop among its other teardown, so a bound reaching that timeout would never be honoured."""
    assert harness_module._ESCALATION_SERVER_STOP_BOUND_SECS < main_module._HARNESS_CANCEL_TIMEOUT
