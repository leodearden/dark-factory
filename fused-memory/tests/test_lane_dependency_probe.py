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
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    QDRANT_URL,
    falkor_skipif,
    lease_dir_fixture,  # noqa: F401  # debt: task 6997
    load_script_module,
    qdrant_skipif,
)
from shared.lane_probe import OUTAGE_EXIT_STATUS

from fused_memory.config.schema import FusedMemoryConfig

FM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = FM_ROOT / 'scripts' / 'lane_dependency_probe.py'
LEASE_REAPER = load_script_module(FM_ROOT / 'scripts' / 'cleanup_test_collections.py')
PROBE_LEASE_PREFIX = 'lane-dependency-probe.'
CONFIG_PATH = FM_ROOT / 'config' / 'config.yaml'
SUBPROCESS_TIMEOUT_SECS = 120
EMPTY_COLLECTIONS = json.dumps({'result': {'collections': []}, 'status': 'ok', 'time': 0.0}).encode()


def closed_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def lease_files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob('*.lease')) if directory.is_dir() else []


def run_probe(
    *dependencies: str, lease_dir: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *dependencies],
        cwd=FM_ROOT,
        env={
            **os.environ,
            'CONFIG_PATH': str(CONFIG_PATH),
            LEASE_REAPER.LEASE_DIR_ENV: str(lease_dir),
            **(env or {}),
        },
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
    )


def assert_names_dependency(result: subprocess.CompletedProcess[str], line: str) -> None:
    assert result.returncode == OUTAGE_EXIT_STATUS, result.stderr
    assert result.stdout.splitlines() == [line]
    assert 'Traceback' not in result.stdout + result.stderr


class WriteRefusingQdrant(BaseHTTPRequestHandler):
    watched_lease_dir: Path | None = None
    leases_seen_at_listing: list[list[str]]

    def do_GET(self) -> None:
        if self.path.rstrip('/') == '/collections':
            if self.watched_lease_dir is not None:
                self.leases_seen_at_listing.append(lease_files(self.watched_lease_dir))
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


@dataclass
class RunningQdrant:
    url: str
    leases_seen_at_listing: list[list[str]]


@contextmanager
def write_refusing_qdrant(watched_lease_dir: Path | None = None) -> Iterator[RunningQdrant]:
    """The fake also records which lease files exist in *watched_lease_dir* whenever it is asked to list."""
    seen: list[list[str]] = []
    handler = type(
        'Handler',
        (WriteRefusingQdrant,),
        {'watched_lease_dir': watched_lease_dir, 'leases_seen_at_listing': seen},
    )
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield RunningQdrant(f'http://127.0.0.1:{server.server_address[1]}', seen)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unreachable_qdrant_is_an_outage_naming_qdrant(lease_dir: Path) -> None:
    result = run_probe(lease_dir=lease_dir, env={'QDRANT_URL': f'http://127.0.0.1:{closed_port()}'})

    assert result.returncode == OUTAGE_EXIT_STATUS, result.stderr
    assert result.stdout.startswith('qdrant: ')
    assert len(result.stdout.splitlines()) == 1
    assert 'Traceback' not in result.stdout + result.stderr


def test_a_qdrant_that_refuses_writes_is_an_outage_at_the_step_that_was_refused(
    lease_dir: Path,
) -> None:
    with write_refusing_qdrant() as qdrant:
        result = run_probe('qdrant', lease_dir=lease_dir, env={'QDRANT_URL': qdrant.url})

    assert_names_dependency(result, 'qdrant: create collection: HTTP 403')


def test_unreachable_falkordb_is_an_outage_naming_falkordb(lease_dir: Path) -> None:
    result = run_probe('falkordb', lease_dir=lease_dir, env={'FALKORDB_URI': f'redis://127.0.0.1:{closed_port()}'})

    assert result.returncode == OUTAGE_EXIT_STATUS, result.stderr
    assert result.stdout.startswith('falkordb: ')
    assert len(result.stdout.splitlines()) == 1
    assert 'Traceback' not in result.stdout + result.stderr


def test_the_same_outage_prints_the_same_line(lease_dir: Path) -> None:
    env = {'QDRANT_URL': f'http://127.0.0.1:{closed_port()}'}

    first = run_probe('qdrant', lease_dir=lease_dir, env=env)
    second = run_probe('qdrant', lease_dir=lease_dir, env=env)

    assert first.stdout == second.stdout


def test_a_bad_dependency_name_is_a_broken_probe_not_an_outage(lease_dir: Path) -> None:
    result = run_probe('nonesuch', lease_dir=lease_dir)

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


def test_the_script_imports_the_fused_memory_tree_pytest_imports(tmp_path: Path) -> None:
    decoy = tmp_path / 'decoy'
    (decoy / 'fused_memory').mkdir(parents=True)
    (decoy / 'fused_memory' / '__init__.py').write_text('DECOY = True\n')

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import runpy, sys; runpy.run_path(sys.argv[1], run_name="lane_probe_provenance");'
            ' import fused_memory; print(fused_memory.__file__)',
            str(SCRIPT),
        ],
        cwd=FM_ROOT.parent,
        env={**os.environ, 'PYTHONPATH': str(decoy)},
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve().is_relative_to((FM_ROOT / 'src').resolve())


def test_the_qdrant_check_holds_a_lease_in_the_given_lease_dir_while_it_works(
    lease_dir: Path,
) -> None:
    with write_refusing_qdrant(lease_dir) as qdrant:
        run_probe('qdrant', lease_dir=lease_dir, env={'QDRANT_URL': qdrant.url})

    [leases] = qdrant.leases_seen_at_listing
    assert [name.split('.')[0] for name in leases] == ['lane-dependency-probe']
    assert lease_files(lease_dir) == []


def test_a_run_leaves_no_probe_lease_in_the_machine_global_lease_dir(lease_dir: Path) -> None:
    def global_probe_leases() -> list[str]:
        return [
            name
            for name in lease_files(LEASE_REAPER.DEFAULT_LEASE_DIR)
            if name.startswith(PROBE_LEASE_PREFIX)
        ]

    before = global_probe_leases()
    with write_refusing_qdrant(LEASE_REAPER.DEFAULT_LEASE_DIR) as qdrant:
        run_probe('qdrant', lease_dir=lease_dir, env={'QDRANT_URL': qdrant.url})

    assert global_probe_leases() == before
    assert [
        [name for name in listing if name.startswith(PROBE_LEASE_PREFIX)]
        for listing in qdrant.leases_seen_at_listing
    ] == [before]


def test_a_failure_line_never_carries_the_words_of_the_exception() -> None:
    probe = load_script_module(SCRIPT, 'lane_dependency_probe_under_test')
    token = uuid.uuid4().hex
    errors = [
        probe.ResponseHandlingException(OSError(f'connect to {token} failed')),
        probe.UnexpectedResponse(403, f'{token}', b'', None),
        RuntimeError(token),
    ]

    lines = [probe.failure_of('qdrant', probe.Progress('upsert'), error).line for error in errors]

    assert all(token not in line for line in lines)
    assert lines == [
        'qdrant: upsert: OSError',
        'qdrant: upsert: HTTP 403',
        'qdrant: upsert: RuntimeError',
    ]


def test_an_unconfigured_dependency_is_named_with_a_fixed_reason() -> None:
    probe = load_script_module(SCRIPT, 'lane_dependency_probe_under_test')
    error = probe.NotConfigured(probe.NotConfiguredReason.NO_PROVIDER_OR_KEY)

    failure = probe.failure_of('embedder', probe.Progress(), error)

    assert failure.line == 'embedder: not configured (no provider or api key)'


@pytest.mark.integration
@qdrant_skipif()
@falkor_skipif()
def test_live_stores_and_models_pass_every_check_silently(lease_dir: Path) -> None:
    result = run_probe(lease_dir=lease_dir)

    assert (result.returncode, result.stdout) == (0, ''), result.stderr
