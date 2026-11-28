"""Test scaffolding for embedded uvicorn servers: readiness awaited rather than slept for, and uvicorn.* logger state imposed then restored.

Importable as ``from _embedded_uvicorn_fixtures import ...`` from every test
directory, and from a ``python -I`` child that appends this directory to
``sys.path`` (``fused-memory/tests/_two_server_operator_stop.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import uvicorn

from fused_memory.server.embedded_uvicorn import EmbeddedUvicornServer

UVICORN_LOGGER_NAMES = ('uvicorn', 'uvicorn.error', 'uvicorn.access', 'uvicorn.asgi')


@dataclass(frozen=True)
class LoggerState:
    handlers: tuple[logging.Handler, ...]
    level: int
    propagate: bool

    @classmethod
    def of(cls, logger: logging.Logger) -> LoggerState:
        return cls(tuple(logger.handlers), logger.level, logger.propagate)

    def apply_to(self, logger: logging.Logger) -> None:
        logger.handlers = list(self.handlers)
        logger.setLevel(self.level)
        logger.propagate = self.propagate


def uvicorn_logger_states() -> dict[str, LoggerState]:
    return {name: LoggerState.of(logging.getLogger(name)) for name in UVICORN_LOGGER_NAMES}


@contextlib.contextmanager
def uvicorn_loggers_set_to(states: Mapping[str, LoggerState]) -> Iterator[None]:
    """Give each logger named in *states* its state there, and restore every uvicorn.* logger on exit."""
    prior = uvicorn_logger_states()
    for name, state in states.items():
        state.apply_to(logging.getLogger(name))
    try:
        yield
    finally:
        for name, state in prior.items():
            state.apply_to(logging.getLogger(name))


class ServerDidNotStart(RuntimeError):
    """serve() ended before uvicorn's startup completed."""


class ReadySignallingServer(EmbeddedUvicornServer):
    """An EmbeddedUvicornServer whose startup a test can await."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self._settled = asyncio.Event()
        self._serve_failure: BaseException | None = None

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().startup(sockets=sockets)
        finally:
            self._settled.set()

    async def serve(self, sockets: list[socket.socket] | None = None) -> None:
        try:
            await super().serve(sockets=sockets)
        except BaseException as failure:
            self._serve_failure = failure
            raise
        finally:
            self._settled.set()

    async def until_started(self) -> None:
        """Return once startup has completed; raise ServerDidNotStart if serve() ended first."""
        await self._settled.wait()
        if not self.started:
            raise ServerDidNotStart(f'serve() ended before startup completed: {self._serve_failure!r}') from (
                self._serve_failure
            )
