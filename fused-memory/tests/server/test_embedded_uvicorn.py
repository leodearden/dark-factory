"""Contract of fused_memory.server.embedded_uvicorn: an embedded uvicorn server owns neither the process's signals nor its loggers."""

from __future__ import annotations

import ast
import asyncio
import logging
import signal
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import uvicorn
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from starlette.applications import Starlette

import fused_memory
from fused_memory.server.embedded_uvicorn import EmbeddedUvicornServer, embedded_uvicorn_config

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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


@pytest.fixture
def process_owned_uvicorn_loggers() -> Iterator[dict[str, LoggerState]]:
    """Give each uvicorn.* logger a sentinel state the process chose, and restore whatever it had before."""
    loggers = {name: logging.getLogger(name) for name in UVICORN_LOGGER_NAMES}
    prior = {name: LoggerState.of(logger) for name, logger in loggers.items()}
    chosen = {
        name: LoggerState((logging.NullHandler(),), logging.CRITICAL - index, propagate=index % 2 == 0)
        for index, name in enumerate(UVICORN_LOGGER_NAMES)
    }
    for name, state in chosen.items():
        state.apply_to(loggers[name])
    try:
        yield chosen
    finally:
        for name, state in prior.items():
            state.apply_to(loggers[name])


def test_config_leaves_the_uvicorn_loggers_as_the_process_set_them(
    process_owned_uvicorn_loggers: dict[str, LoggerState],
) -> None:
    embedded_uvicorn_config(Starlette(), host='127.0.0.1', port=0, timeout_graceful_shutdown=5)

    assert {
        name: LoggerState.of(logging.getLogger(name)) for name in UVICORN_LOGGER_NAMES
    } == process_owned_uvicorn_loggers


@pytest.mark.parametrize(
    ('key', 'value'),
    [('log_config', {'version': 1}), ('log_level', 'info'), ('access_log', False)],
)
def test_config_refuses_process_global_logging_settings(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        embedded_uvicorn_config(Starlette(), **{key: value})


def test_config_forwards_server_settings() -> None:
    config = embedded_uvicorn_config(Starlette(), host='127.0.0.2', port=8765, timeout_graceful_shutdown=7)

    assert (config.host, config.port, config.timeout_graceful_shutdown) == ('127.0.0.2', 8765, 7)


class _ReadySignallingServer(EmbeddedUvicornServer):
    """Sets *ready* once uvicorn's startup has run, so a test waits on readiness rather than on time."""

    def __init__(self, config: uvicorn.Config, ready: asyncio.Event) -> None:
        super().__init__(config)
        self._ready = ready

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        self._ready.set()


@pytest.mark.asyncio
async def test_serving_leaves_the_process_signal_handlers_alone() -> None:
    handlers_before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    ready = asyncio.Event()
    server = _ReadySignallingServer(embedded_uvicorn_config(Starlette()), ready)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    with listener:
        serving = asyncio.ensure_future(server.serve(sockets=[listener]))
        ready_wait = asyncio.ensure_future(ready.wait())
        await asyncio.wait({serving, ready_wait}, return_when=asyncio.FIRST_COMPLETED)
        ready_wait.cancel()
        assert server.started, f'server did not start: {serving.exception() if serving.done() else None!r}'

        assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == handlers_before

        server.should_exit = True
        await serving

    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == handlers_before


_UVICORN_CONSTRUCTORS = frozenset({'Server', 'Config'})


def _uvicorn_constructor_sites(tree: ast.Module) -> Iterator[int]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _UVICORN_CONSTRUCTORS
            and isinstance(node.value, ast.Name)
            and node.value.id == 'uvicorn'
        ) or (
            isinstance(node, ast.ImportFrom)
            and node.module == 'uvicorn'
            and any(alias.name in _UVICORN_CONSTRUCTORS for alias in node.names)
        ):
            yield node.lineno


def test_only_the_embedded_module_constructs_uvicorn_servers() -> None:
    """Every in-process uvicorn server shares the process's signals and loggers, so each construction site must go through the seam (heuristic 10, uniformly enforced)."""
    package = Path(fused_memory.__file__).parent
    seam = package / 'server' / 'embedded_uvicorn.py'

    offending_sites = [
        f'{path.relative_to(package.parent)}:{lineno}'
        for path in sorted(package.rglob('*.py'))
        if path != seam
        for lineno in _uvicorn_constructor_sites(ast.parse(path.read_text(), filename=str(path)))
    ]

    assert offending_sites == [], (
        'build uvicorn servers through fused_memory.server.embedded_uvicorn, not uvicorn.Server/Config'
    )
