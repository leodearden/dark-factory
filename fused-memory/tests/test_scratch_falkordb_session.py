"""A fused-memory test session aims every FalkorDB client at the scratch server (task 7792).

The subprocess cases run a real ``pytest --collect-only`` of this file, so they exercise
conftest's ``pytest_configure`` exactly as a lane or developer session does.
"""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT
from _fm_sockets import refused_port
from shared.lane_probe import LANE_ROLE_VARIABLE, OFFLINE_LANE_ROLE
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.config.falkordb_endpoints import FALKOR_HOST_ENV, FALKOR_PORT_ENV, FalkorEndpoint
from fused_memory.config.schema import FusedMemoryConfig

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

THIS_FILE = Path(__file__).resolve()
FM_ROOT = THIS_FILE.parents[1]
SUBPROCESS_TIMEOUT_SECS = 120
SESSION_STEERING_VARIABLES = frozenset({FALKOR_HOST_ENV, FALKOR_PORT_ENV, 'FALKORDB_URI', LANE_ROLE_VARIABLE})


@pytest.fixture
def unreachable_port() -> Iterator[int]:
    with refused_port() as port:
        yield port


def collect_this_file(**environment: str) -> subprocess.CompletedProcess[str]:
    inherited = {name: value for name, value in os.environ.items() if name not in SESSION_STEERING_VARIABLES}
    return subprocess.run(
        [
            sys.executable, '-m', 'pytest', '--collect-only', '-q',
            '-p', 'no:xdist', '-p', 'no:cacheprovider', '-o', 'addopts=', str(THIS_FILE),
        ],
        cwd=FM_ROOT,
        env={**inherited, **environment},
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
    )


def test_config_built_backends_aim_at_the_session_falkordb(mock_config) -> None:
    session_uri = FalkorEndpoint(FALKOR_HOST, FALKOR_PORT).uri

    assert FusedMemoryConfig().graphiti.falkordb.uri == session_uri
    assert mock_config.graphiti.falkordb.uri == session_uri


def test_a_session_aimed_at_production_is_a_usage_error() -> None:
    result = collect_this_file(**{FALKOR_PORT_ENV: '6379'})

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert FALKOR_PORT_ENV in output
    assert 'production' in output


def test_a_lane_session_with_an_unreachable_scratch_server_is_a_usage_error(unreachable_port: int) -> None:
    unreachable = FalkorEndpoint('127.0.0.1', unreachable_port)

    result = collect_this_file(**{
        LANE_ROLE_VARIABLE: OFFLINE_LANE_ROLE,
        FALKOR_HOST_ENV: unreachable.host,
        FALKOR_PORT_ENV: str(unreachable.port),
    })

    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert unreachable.uri in output
    assert 'falkordb-scratch' in output


def test_outside_the_lane_an_unreachable_scratch_server_still_collects(unreachable_port: int) -> None:
    result = collect_this_file(**{FALKOR_HOST_ENV: '127.0.0.1', FALKOR_PORT_ENV: str(unreachable_port)})

    assert result.returncode in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED), (
        result.stdout + result.stderr
    )
