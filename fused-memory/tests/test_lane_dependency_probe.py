"""The offline lane's dependency probe, run as the lane runs it: a subprocess in ``fused-memory/``.

FalkorDB is covered by its unreachable case only; a write-refusing FalkorDB would need a redis-protocol fake.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    QDRANT_URL,
    falkor_skipif,
    qdrant_skipif,
)
from shared.lane_probe import OUTAGE_EXIT_STATUS

from fused_memory.config.schema import FusedMemoryConfig

FM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = FM_ROOT / 'scripts' / 'lane_dependency_probe.py'
CONFIG_PATH = FM_ROOT / 'config' / 'config.yaml'
SUBPROCESS_TIMEOUT_SECS = 120
EMPTY_COLLECTIONS = json.dumps({'result': {'collections': []}, 'status': 'ok', 'time': 0.0}).encode()


def closed_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def run_probe(
    *dependencies: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *dependencies],
        cwd=FM_ROOT,
        env={**os.environ, 'CONFIG_PATH': str(CONFIG_PATH), **(env or {})},
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
    )


def assert_names_dependency(result: subprocess.CompletedProcess[str], line: str) -> None:
    assert result.returncode == OUTAGE_EXIT_STATUS, result.stderr
    assert result.stdout.splitlines() == [line]
    assert 'Traceback' not in result.stdout + result.stderr


class WriteRefusingQdrant(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.rstrip('/') == '/collections':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(EMPTY_COLLECTIONS)
        else:
            self.send_error(404)

    def do_PUT(self) -> None:
        self.send_error(403)

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def write_refusing_qdrant() -> Iterator[str]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), WriteRefusingQdrant)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unreachable_qdrant_is_an_outage_naming_qdrant() -> None:
    result = run_probe(env={'QDRANT_URL': f'http://127.0.0.1:{closed_port()}'})

    assert result.returncode == OUTAGE_EXIT_STATUS, result.stderr
    assert result.stdout.startswith('qdrant: ')
    assert len(result.stdout.splitlines()) == 1
    assert 'Traceback' not in result.stdout + result.stderr


def test_a_qdrant_that_refuses_writes_is_an_outage_at_the_step_that_was_refused() -> None:
    with write_refusing_qdrant() as url:
        result = run_probe('qdrant', env={'QDRANT_URL': url})

    assert_names_dependency(result, 'qdrant: create collection: HTTP 403')


def test_unreachable_falkordb_is_an_outage_naming_falkordb() -> None:
    result = run_probe('falkordb', env={'FALKORDB_URI': f'redis://127.0.0.1:{closed_port()}'})

    assert result.returncode == OUTAGE_EXIT_STATUS, result.stderr
    assert result.stdout.startswith('falkordb: ')
    assert len(result.stdout.splitlines()) == 1
    assert 'Traceback' not in result.stdout + result.stderr


def test_the_same_outage_prints_the_same_line() -> None:
    env = {'QDRANT_URL': f'http://127.0.0.1:{closed_port()}'}

    assert run_probe('qdrant', env=env).stdout == run_probe('qdrant', env=env).stdout


def test_a_bad_dependency_name_is_a_broken_probe_not_an_outage() -> None:
    result = run_probe('nonesuch')

    assert result.returncode == 2
    assert result.stdout == ''


def test_the_configuration_defaults_name_the_endpoints_the_integration_tests_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('QDRANT_URL', raising=False)
    monkeypatch.delenv('FALKORDB_URI', raising=False)
    cfg = FusedMemoryConfig()

    assert cfg.mem0.qdrant_url == QDRANT_URL
    if 'FALKOR_HOST' in os.environ or 'FALKOR_PORT' in os.environ:
        pytest.skip('FALKOR_HOST/FALKOR_PORT override the integration tests default')
    falkor = urlparse(cfg.graphiti.falkordb.uri)
    assert (falkor.hostname, falkor.port) == (FALKOR_HOST, FALKOR_PORT)


def test_the_script_imports_the_fused_memory_tree_pytest_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import runpy, sys; runpy.run_path(sys.argv[1], run_name="lane_probe_provenance");'
            ' import fused_memory; print(fused_memory.__file__)',
            str(SCRIPT),
        ],
        cwd=FM_ROOT.parent,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve().is_relative_to((FM_ROOT / 'src').resolve())


@pytest.mark.integration
@qdrant_skipif()
@falkor_skipif()
def test_live_stores_and_models_pass_every_check_silently() -> None:
    result = run_probe()

    assert (result.returncode, result.stdout) == (0, ''), result.stderr
