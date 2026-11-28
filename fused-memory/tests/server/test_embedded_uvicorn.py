"""Contract of fused_memory.server.embedded_uvicorn: an embedded uvicorn server owns neither the process's signals nor its loggers."""

from __future__ import annotations

import ast
import asyncio
import logging
import signal
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from _embedded_uvicorn_fixtures import (
    UVICORN_LOGGER_NAMES,
    LoggerState,
    ReadySignallingServer,
    uvicorn_logger_states,
    uvicorn_loggers_set_to,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from starlette.applications import Starlette

import fused_memory
from fused_memory.server.embedded_uvicorn import embedded_uvicorn_config

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


@pytest.fixture
def sentinel_uvicorn_loggers() -> Iterator[dict[str, LoggerState]]:
    """Each uvicorn.* logger in a distinct sentinel state, as if the process had chosen it."""
    sentinels = {
        name: LoggerState((logging.NullHandler(),), logging.CRITICAL - index, propagate=index % 2 == 0)
        for index, name in enumerate(UVICORN_LOGGER_NAMES)
    }
    with uvicorn_loggers_set_to(sentinels):
        yield sentinels


def test_config_leaves_the_uvicorn_loggers_as_the_process_set_them(
    sentinel_uvicorn_loggers: dict[str, LoggerState],
) -> None:
    embedded_uvicorn_config(Starlette(), host='127.0.0.1', port=0, timeout_graceful_shutdown=5)

    assert uvicorn_logger_states() == sentinel_uvicorn_loggers


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


@pytest.mark.asyncio
async def test_serving_leaves_the_process_signal_handlers_alone() -> None:
    assert threading.current_thread() is threading.main_thread(), (
        "uvicorn's capture_signals installs handlers only on the main thread, so off it this test proves nothing"
    )
    handlers_before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
    server = ReadySignallingServer(embedded_uvicorn_config(Starlette()))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    with listener:
        serving = asyncio.ensure_future(server.serve(sockets=[listener]))
        await server.until_started()

        assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == handlers_before

        server.should_exit = True
        await serving

    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == handlers_before


_UVICORN_MODULES = frozenset({'uvicorn', 'uvicorn.server', 'uvicorn.config'})
_UVICORN_CONSTRUCTORS = frozenset({'Server', 'Config'})
_SELF_SERVING_RUNNERS = frozenset({'run_http_async', 'run_streamable_http_async', 'run_sse_async'})
"""The fastmcp and mcp methods that build and serve a plain uvicorn server of their own."""

_PROCESS_OWNER = Path('fused_memory/server/main.py')
"""The module that owns the process's servers, and so the one that may pick a transport FastMCP serves alone."""


def _uvicorn_bindings(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    """The names *tree* binds to a uvicorn module, and the names it binds to uvicorn's Server or Config."""
    modules: set[str] = set()
    constructors: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name.split('.')[0] for alias in node.names if alias.name in _UVICORN_MODULES
            )
        elif isinstance(node, ast.ImportFrom) and node.module in _UVICORN_MODULES:
            for alias in node.names:
                if alias.name in _UVICORN_CONSTRUCTORS:
                    constructors.add(alias.asname or alias.name)
                elif f'{node.module}.{alias.name}' in _UVICORN_MODULES:
                    modules.add(alias.asname or alias.name)
    return frozenset(modules), frozenset(constructors)


def _uvicorn_server_construction_sites(tree: ast.Module, *, owns_the_process: bool) -> Iterator[int]:
    """Lines of *tree* that construct or subclass uvicorn's Server or Config, or call a self-serving runner."""
    modules, constructors = _uvicorn_bindings(tree)

    def is_module(node: ast.expr) -> bool:
        return (isinstance(node, ast.Name) and node.id in modules) or (
            isinstance(node, ast.Attribute) and node.attr in {'server', 'config'} and is_module(node.value)
        )

    def is_constructor(node: ast.expr) -> bool:
        return (isinstance(node, ast.Name) and node.id in constructors) or (
            isinstance(node, ast.Attribute) and node.attr in _UVICORN_CONSTRUCTORS and is_module(node.value)
        )

    def is_self_serving_runner(node: ast.expr) -> bool:
        return not owns_the_process and isinstance(node, ast.Attribute) and node.attr in _SELF_SERVING_RUNNERS

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (is_constructor(node.func) or is_self_serving_runner(node.func)):
            yield node.lineno
        if isinstance(node, ast.ClassDef) and any(is_constructor(base) for base in node.bases):
            yield node.lineno


@pytest.mark.parametrize(
    'source',
    [
        'import uvicorn\nuvicorn.Server(config)',
        'import uvicorn as uv\nuv.Config(app)',
        'import uvicorn.server\nuvicorn.server.Server(config)',
        'from uvicorn import config\nconfig.Config(app)',
        'from uvicorn.server import Server\nServer(config)',
        'from uvicorn.config import Config as UvicornConfig\nUvicornConfig(app)',
        'import uvicorn\nclass QuietServer(uvicorn.Server): ...',
        'async def serve(mcp):\n    await mcp.run_http_async(host=host, port=port)',
        'async def serve(mcp):\n    await mcp.run_streamable_http_async()',
    ],
)
def test_the_guard_finds_each_way_to_build_a_uvicorn_server(source: str) -> None:
    assert list(_uvicorn_server_construction_sites(ast.parse(source), owns_the_process=False)) == [2]


@pytest.mark.parametrize(
    'source',
    [
        'import uvicorn\ndef config_for(app) -> uvicorn.Config: ...',
        'import uvicorn\nserver: uvicorn.Server | None = None',
        'from fused_memory.server.embedded_uvicorn import EmbeddedUvicornServer\nEmbeddedUvicornServer(config)',
    ],
)
def test_the_guard_ignores_what_builds_no_plain_uvicorn_server(source: str) -> None:
    assert list(_uvicorn_server_construction_sites(ast.parse(source), owns_the_process=False)) == []


def test_only_the_process_owner_may_pick_a_transport_fastmcp_serves_alone() -> None:
    source = 'async def serve(mcp):\n    await mcp.run_sse_async()'

    assert list(_uvicorn_server_construction_sites(ast.parse(source), owns_the_process=True)) == []


def test_only_the_embedded_module_constructs_uvicorn_servers() -> None:
    """Every in-process uvicorn server shares the process's signals and loggers, so each one must be built through the seam (heuristic 10, uniformly enforced).

    The process's owner, main.py, may still pick the sse transport, where
    FastMCP serves its own server alone: reconciliation and the recon-report
    server need the http transport.
    """
    package = Path(fused_memory.__file__).parent
    seam = package / 'server' / 'embedded_uvicorn.py'

    offending_sites = [
        f'{path.relative_to(package.parent)}:{lineno}'
        for path in sorted(package.rglob('*.py'))
        if path != seam
        for lineno in _uvicorn_server_construction_sites(
            ast.parse(path.read_text(), filename=str(path)),
            owns_the_process=path.relative_to(package.parent) == _PROCESS_OWNER,
        )
    ]

    assert offending_sites == [], (
        'build uvicorn servers through fused_memory.server.embedded_uvicorn, not uvicorn.Server/Config '
        'or a fastmcp/mcp runner that serves its own'
    )
