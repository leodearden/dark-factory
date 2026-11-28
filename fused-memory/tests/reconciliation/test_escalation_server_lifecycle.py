"""The reconciliation harness's escalation server stops with uvicorn's own clean shutdown, not a cancel (task 10067)."""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator

import httpx
import pytest
from escalation.queue import EscalationQueue

from reconciliation.harness_fixtures import open_harness

_UVICORN_LOGGER_NAMES = ('uvicorn', 'uvicorn.error', 'uvicorn.access', 'uvicorn.asgi')


@pytest.fixture
def process_owned_uvicorn_loggers() -> Iterator[None]:
    """Put 'uvicorn' and 'uvicorn.error' in the shape main.py leaves them, and restore every uvicorn logger afterwards.

    An earlier test in the same worker may have built a plain uvicorn.Config,
    whose dictConfig rewires these loggers process-wide.
    """
    loggers = [logging.getLogger(name) for name in _UVICORN_LOGGER_NAMES]
    prior = [(list(logger.handlers), logger.level, logger.propagate) for logger in loggers]
    for logger in loggers[:2]:
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)
    try:
        yield
    finally:
        for logger, (handlers, level, propagate) in zip(loggers, prior, strict=True):
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate


@pytest.fixture
def systemd_style_listener() -> Iterator[socket.socket]:
    """A socket bound to a free 127.0.0.1 port and already listening, as systemd hands it over."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        yield listener


@pytest.mark.asyncio
@pytest.mark.usefixtures('process_owned_uvicorn_loggers')
async def test_a_started_escalation_server_stops_with_a_clean_uvicorn_shutdown(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    systemd_style_listener: socket.socket,
) -> None:
    caplog.set_level(logging.INFO, logger='uvicorn.error')
    host, port = systemd_style_listener.getsockname()

    async with open_harness(
        tmp_path,
        taskmaster=None,
        escalation_listener=systemd_style_listener,
        escalation_queue=EscalationQueue(tmp_path / 'esc'),
    ) as harness:
        await harness._start_escalation_server()
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                await client.get(f'http://{host}:{port}/mcp')
        finally:
            await harness._stop_escalation_server()

    assert [
        record
        for record in caplog.records
        if record.name == 'uvicorn.error'
        and record.levelno == logging.INFO
        and record.getMessage() == 'Application shutdown complete.'
    ], [record.getMessage() for record in caplog.records if record.name.startswith('uvicorn')]
