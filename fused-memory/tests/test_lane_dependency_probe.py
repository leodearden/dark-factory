"""The offline lane's dependency probe, run as the lane runs it: a subprocess in ``fused-memory/``.

FalkorDB is covered by its unreachable case only; a write-refusing FalkorDB would need a redis-protocol fake.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, falkor_skipif
from _fm_helpers import (
    lease_dir_fixture,  # noqa: F401  # debt: task 6997
    load_script_module,
)
from _fm_qdrant_live import QDRANT_URL, qdrant_skipif
from _fm_sockets import has_pending_connection, refused_port, silent_listener
from shared.lane_probe import OUTAGE_EXIT_STATUS
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.config.falkordb_endpoints import (
    PRODUCTION_FALKORDB,
    FalkorEndpoint,
    scratch_falkordb_from_env,
)
from fused_memory.config.schema import FusedMemoryConfig

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

FM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = FM_ROOT / 'scripts' / 'lane_dependency_probe.py'
LEASE_REAPER = load_script_module(FM_ROOT / 'scripts' / 'cleanup_test_collections.py')
PROBE_LEASE_PREFIX = 'lane-dependency-probe.'
CONFIG_PATH = FM_ROOT / 'config' / 'config.yaml'
SUBPROCESS_TIMEOUT_SECS = 120
EMPTY_COLLECTIONS = json.dumps({'result': {'collections': []}, 'status': 'ok', 'time': 0.0}).encode()


def lease_files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob('*.lease')) if directory.is_dir() else []


def probe_leases_of(pid: int, names: list[str]) -> list[str]:
    return [name for name in names if name.startswith(f'{PROBE_LEASE_PREFIX}{pid}.')]


def start_probe(
    *dependencies: str, lease_dir: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(SCRIPT), *dependencies],
        cwd=FM_ROOT,
        env={
            **os.environ,
            'CONFIG_PATH': str(CONFIG_PATH),
            LEASE_REAPER.LEASE_DIR_ENV: str(lease_dir),
            **(env or {}),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def finish_probe(probe: subprocess.Popen[str]) -> subprocess.CompletedProcess[str]:
    with probe:
        try:
            stdout, stderr = probe.communicate(timeout=SUBPROCESS_TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            probe.kill()
            raise
    return subprocess.CompletedProcess(probe.args, probe.returncode, stdout, stderr)


def run_probe(
    *dependencies: str, lease_dir: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return finish_probe(start_probe(*dependencies, lease_dir=lease_dir, env=env))


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
    with refused_port() as port:
        result = run_probe(lease_dir=lease_dir, env={'QDRANT_URL': f'http://127.0.0.1:{port}'})

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
    """The probe is steered by FALKOR_*, the tests' knob, never by the config's FALKORDB_URI."""
    with refused_port() as port, silent_listener() as config_server:
        result = run_probe(
            'falkordb',
            lease_dir=lease_dir,
            env={
                'FALKOR_HOST': '127.0.0.1',
                'FALKOR_PORT': str(port),
                'FALKORDB_URI': f'redis://127.0.0.1:{config_server.getsockname()[1]}',
            },
        )
        config_server_was_dialled = has_pending_connection(config_server)

    assert result.returncode == OUTAGE_EXIT_STATUS, result.stderr
    assert result.stdout.startswith('falkordb: ')
    assert len(result.stdout.splitlines()) == 1
    assert 'Traceback' not in result.stdout + result.stderr
    assert not config_server_was_dialled


def test_a_falkordb_probe_aimed_at_production_is_a_broken_probe_not_an_outage(lease_dir: Path) -> None:
    probe = load_script_module(SCRIPT, 'lane_dependency_probe_under_test')

    result = run_probe('falkordb', lease_dir=lease_dir, env={'FALKOR_PORT': str(PRODUCTION_FALKORDB.port)})

    assert result.returncode == probe.BROKEN_CONFIG_EXIT_STATUS
    assert result.stdout == ''
    assert 'production' in result.stderr
    assert 'Traceback' not in result.stderr


def test_the_same_outage_prints_the_same_line(lease_dir: Path) -> None:
    with refused_port() as port:
        env = {'QDRANT_URL': f'http://127.0.0.1:{port}'}
        first = run_probe('qdrant', lease_dir=lease_dir, env=env)
        second = run_probe('qdrant', lease_dir=lease_dir, env=env)

    assert first.stdout == second.stdout


def test_a_bad_dependency_name_is_a_broken_probe_not_an_outage(lease_dir: Path) -> None:
    result = run_probe('nonesuch', lease_dir=lease_dir)

    assert result.returncode == 2
    assert result.stdout == ''


def test_the_probe_checks_the_endpoints_the_integration_tests_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = load_script_module(SCRIPT, 'lane_dependency_probe_under_test')
    monkeypatch.delenv('QDRANT_URL', raising=False)
    cfg = FusedMemoryConfig()

    assert cfg.mem0.qdrant_url == QDRANT_URL
    checked = probe.probe_config(cfg, scratch_falkordb_from_env()).graphiti.falkordb
    assert checked.uri == FalkorEndpoint(FALKOR_HOST, FALKOR_PORT).uri
    assert checked.password is None
    assert checked.expected_graph_timeout_ms == cfg.graphiti.falkordb.expected_graph_timeout_ms


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
        probe = start_probe('qdrant', lease_dir=lease_dir, env={'QDRANT_URL': qdrant.url})
        finish_probe(probe)

    [leases] = qdrant.leases_seen_at_listing
    assert len(leases) == 1 and probe_leases_of(probe.pid, leases) == leases
    assert lease_files(lease_dir) == []


def test_a_run_leaves_no_probe_lease_in_the_machine_global_lease_dir(lease_dir: Path) -> None:
    with write_refusing_qdrant(LEASE_REAPER.DEFAULT_LEASE_DIR) as qdrant:
        probe = start_probe('qdrant', lease_dir=lease_dir, env={'QDRANT_URL': qdrant.url})
        result = finish_probe(probe)

    assert_names_dependency(result, 'qdrant: create collection: HTTP 403')
    assert len(qdrant.leases_seen_at_listing) == 1, (
        'GET /collections reached this fake from outside the probe under test'
        ' (a probe aimed at a released port?)'
    )
    assert [probe_leases_of(probe.pid, listing) for listing in qdrant.leases_seen_at_listing] == [[]]
    assert probe_leases_of(probe.pid, lease_files(LEASE_REAPER.DEFAULT_LEASE_DIR)) == []


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
