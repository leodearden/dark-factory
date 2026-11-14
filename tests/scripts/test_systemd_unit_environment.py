"""Tests for scripts/systemd_unit_environment.py (task 5978).

systemd is played by tests/scripts/fake_busctl.py on PATH; the live layer is a
real child process spawned with a controlled environment, so the production
/proc/<pid>/environ read runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import fake_busctl
import pytest
import systemd_unit_environment as sue

UNIT = 'example.service'
UNIT_PATH = '/fake/unit/0'
MANAGER_PATH_VALUE = '/home/u/.local/bin:/usr/bin'
START_MICROSECONDS = 1_700_000_000_000_000

WriteWorld = Callable[[dict[str, object]], Path]
SpawnLive = Callable[[dict[str, str]], int]


@pytest.fixture
def busctl_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WriteWorld:
    bin_dir = tmp_path / 'bin'
    fake_busctl.install(bin_dir)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    world_path = tmp_path / 'busctl-world.json'

    def write(world: dict[str, object]) -> Path:
        world_path.write_text(json.dumps(world))
        monkeypatch.setenv(fake_busctl.WORLD_ENV, str(world_path))
        return world_path

    return write


@pytest.fixture
def live_process() -> Iterator[SpawnLive]:
    children: list[subprocess.Popen[bytes]] = []

    def spawn(env: dict[str, str]) -> int:
        child = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(120)'], env=env
        )
        children.append(child)
        return child.pid

    yield spawn
    for child in children:
        child.terminate()
        child.wait(timeout=10)


def _world(unit: dict[str, object], manager_path: str = MANAGER_PATH_VALUE) -> dict[str, object]:
    return {
        'manager_environment': [f'PATH={manager_path}', 'LANG=C.UTF-8'],
        'units': {UNIT: {'path': UNIT_PATH, **unit}},
    }


def _variable_block(stdout: str, name: str) -> str:
    lines = stdout.splitlines()
    start = lines.index(name)
    end = lines.index('', start) if '' in lines[start:] else len(lines)
    return '\n'.join(lines[start:end])


def test_boot_time_unit_whose_process_predates_the_manager_path_is_drifted(
    busctl_world: WriteWorld,
    live_process: SpawnLive,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pid = live_process({'PATH': '/usr/bin'})
    busctl_world(_world({'MainPID': pid, 'ExecMainStartTimestamp': START_MICROSECONDS}))

    report = sue.inspect_unit(UNIT).variable('PATH')

    assert report.manager_value == MANAGER_PATH_VALUE
    assert report.unit_value is None
    assert report.restart_value == MANAGER_PATH_VALUE
    assert report.live_value == '/usr/bin'
    assert report.verdict is sue.Verdict.DRIFTED

    assert sue.main([UNIT]) == 0
    stdout = capsys.readouterr().out
    assert UNIT in stdout
    assert str(pid) in stdout
    assert '2023-11-14T22:13:20' in stdout
    block = _variable_block(stdout, 'PATH')
    assert MANAGER_PATH_VALUE in block
    assert any(line.rstrip().endswith(' /usr/bin') for line in block.splitlines())
    assert 'DRIFTED' in block


def test_unit_environment_overrides_the_manager_block(
    busctl_world: WriteWorld, live_process: SpawnLive
) -> None:
    pid = live_process({'PATH': '/pinned/bin'})
    busctl_world(
        _world({'Environment': ['PATH=/pinned/bin', 'LANG=C'], 'MainPID': pid})
    )

    report = sue.inspect_unit(UNIT).variable('PATH')

    assert report.manager_value == MANAGER_PATH_VALUE
    assert report.unit_value == '/pinned/bin'
    assert report.restart_value == '/pinned/bin'
    assert report.live_value == '/pinned/bin'
    assert report.verdict is sue.Verdict.MATCHES


def test_values_containing_equals_signs_survive_every_layer(
    busctl_world: WriteWorld,
    live_process: SpawnLive,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pid = live_process({'PATH': MANAGER_PATH_VALUE, 'FOO': 'a=b'})
    busctl_world(_world({'Environment': ['FOO=a=b'], 'MainPID': pid}))

    report = sue.inspect_unit(UNIT).variable('FOO')
    assert (report.restart_value, report.live_value) == ('a=b', 'a=b')
    assert report.verdict is sue.Verdict.MATCHES

    assert sue.main([UNIT, '--var', 'PATH', '--var', 'FOO']) == 0
    stdout = capsys.readouterr().out
    assert 'MATCHES' in _variable_block(stdout, 'PATH')
    foo_block = _variable_block(stdout, 'FOO')
    assert 'MATCHES' in foo_block
    assert any(line.rstrip().endswith(' a=b') for line in foo_block.splitlines())


def test_read_process_environ_returns_the_environment_a_process_was_spawned_with(
    live_process: SpawnLive,
) -> None:
    env = {'PATH': '/usr/bin', 'FOO': 'a=b', 'SPACED': 'x y'}
    pid = live_process(env)

    assert sue.read_process_environ(pid) == env
