"""Tests for scripts/systemd_unit_environment.py and its OPERATIONS.md recipe (task 5978).

systemd is played by tests/scripts/fake_busctl.py on PATH; the live layer is a
real child process spawned with a controlled environment, so the production
/proc/<pid>/environ read runs. The recipe in OPERATIONS.md §3 is extracted and
executed verbatim against the same fake.
"""

from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import fake_busctl
import pytest
import systemd_unit_environment as sue
import verify_command_invariants as vci
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).parents[2]
RECIPE_MARKER = 'unit-environment-recipe'
RECIPE_LABEL = 'Unit environment, layer by layer'
RECIPE_UNIT = 'orchestrator-dark-factory.service'

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
    """Spawn a sleeping child with exactly *env*; return its pid once it is running.

    Popen returns as soon as exec closes its CLOEXEC error pipe, which is before
    the kernel has set up the new image's environment, so /proc/<pid>/environ
    can read empty until the child reports in.
    """
    children: list[subprocess.Popen[bytes]] = []

    def spawn(env: dict[str, str]) -> int:
        child = subprocess.Popen(
            [sys.executable, '-c', 'import time; print(flush=True); time.sleep(120)'],
            env=env,
            stdout=subprocess.PIPE,
        )
        children.append(child)
        assert child.stdout is not None
        child.stdout.readline()
        return child.pid

    yield spawn
    for child in children:
        child.terminate()
        child.communicate(timeout=10)


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


def test_a_unit_with_no_main_pid_is_not_running_and_proc_is_never_read(
    busctl_world: WriteWorld, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    busctl_world(
        _world(
            {
                'Environment': ['PATH=/pinned/bin'],
                'MainPID': 0,
                'ExecMainStartTimestamp': START_MICROSECONDS,
            }
        )
    )
    empty_proc = tmp_path / 'empty-proc'
    empty_proc.mkdir()

    report = sue.inspect_unit(UNIT, proc_root=empty_proc).variable('PATH')

    assert report.verdict is sue.Verdict.NOT_RUNNING
    assert report.live_value is None
    assert report.restart_value == '/pinned/bin'

    assert sue.main([UNIT]) == 0
    stdout = capsys.readouterr().out
    block = _variable_block(stdout, 'PATH')
    assert MANAGER_PATH_VALUE in block
    assert '/pinned/bin' in block
    assert 'NOT_RUNNING' in block
    assert 'pid' not in stdout
    assert '2023-11-14' not in stdout


@pytest.mark.parametrize(
    ('unmodelled', 'named'),
    [
        ({'EnvironmentFiles': [['/etc/example.env', False]]}, '/etc/example.env'),
        ({'PassEnvironment': ['PATH']}, 'PassEnvironment=PATH'),
        ({'UnsetEnvironment': ['PATH']}, 'UnsetEnvironment=PATH'),
    ],
    ids=['EnvironmentFile', 'PassEnvironment', 'UnsetEnvironment'],
)
def test_unmodelled_environment_inputs_make_every_variable_indeterminate(
    busctl_world: WriteWorld,
    live_process: SpawnLive,
    capsys: pytest.CaptureFixture[str],
    unmodelled: dict[str, object],
    named: str,
) -> None:
    pid = live_process({'PATH': '/pinned/bin', 'LANG': 'C.UTF-8'})
    busctl_world(
        _world({'Environment': ['PATH=/pinned/bin'], 'MainPID': pid, **unmodelled})
    )

    env = sue.inspect_unit(UNIT)

    for name, live_value in (('PATH', '/pinned/bin'), ('LANG', 'C.UTF-8')):
        report = env.variable(name)
        assert report.restart_value is None, name
        assert report.verdict is sue.Verdict.INDETERMINATE, name
        assert report.live_value == live_value, name

    assert sue.main([UNIT, '--var', 'PATH', '--var', 'LANG']) == 0
    stdout = capsys.readouterr().out
    assert named in stdout
    assert 'INDETERMINATE' in _variable_block(stdout, 'PATH')
    assert 'INDETERMINATE' in _variable_block(stdout, 'LANG')


def test_not_running_is_decided_before_unmodelled_inputs(busctl_world: WriteWorld) -> None:
    busctl_world(
        _world({'MainPID': 0, 'EnvironmentFiles': [['/etc/example.env', False]]})
    )

    report = sue.inspect_unit(UNIT).variable('PATH')

    assert report.verdict is sue.Verdict.NOT_RUNNING
    assert report.restart_value is None


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads a mode-000 file regardless')
def test_an_unreadable_live_environ_is_reported_not_raised(
    busctl_world: WriteWorld, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    busctl_world(_world({'Environment': ['PATH=/pinned/bin'], 'MainPID': 4242}))
    proc_root = tmp_path / 'proc'
    environ = proc_root / '4242' / 'environ'
    environ.parent.mkdir(parents=True)
    environ.write_bytes(b'PATH=/pinned/bin\0')
    environ.chmod(0o000)
    request.addfinalizer(lambda: environ.chmod(0o600))

    env = sue.inspect_unit(UNIT, proc_root=proc_root)

    for name in ('PATH', 'LANG'):
        report = env.variable(name)
        assert report.verdict is sue.Verdict.UNREADABLE, name
        assert report.live_value is None, name
    assert isinstance(env.live, sue.LiveEnvironUnreadable)
    assert (env.live.pid, env.live.path, env.live.errno) == (4242, environ, errno.EACCES)
    assert env.variable('PATH').manager_value == MANAGER_PATH_VALUE
    assert env.variable('PATH').unit_value == '/pinned/bin'
    assert sue.exit_status(env) == sue.EXIT_LIVE_UNREADABLE
    assert str(environ) in sue.render(env, ['PATH'])


def test_a_main_pid_that_has_exited_is_unreadable_through_the_real_proc(
    busctl_world: WriteWorld, capsys: pytest.CaptureFixture[str]
) -> None:
    exited = subprocess.Popen([sys.executable, '-c', 'pass'])
    exited.wait()
    busctl_world(_world({'Environment': ['PATH=/pinned/bin'], 'MainPID': exited.pid}))

    assert sue.main([UNIT]) == sue.EXIT_LIVE_UNREADABLE
    stdout = capsys.readouterr().out
    block = _variable_block(stdout, 'PATH')
    assert MANAGER_PATH_VALUE in block
    assert '/pinned/bin' in block
    assert f'/proc/{exited.pid}/environ' in stdout


def test_a_complete_report_exits_ok(
    busctl_world: WriteWorld, live_process: SpawnLive
) -> None:
    pid = live_process({'PATH': '/pinned/bin'})
    busctl_world(_world({'Environment': ['PATH=/pinned/bin'], 'MainPID': pid}))

    env = sue.inspect_unit(UNIT)

    assert env.variable('PATH').verdict is sue.Verdict.MATCHES
    assert sue.exit_status(env) == sue.EXIT_OK


def test_a_bus_error_prints_no_half_report(
    busctl_world: WriteWorld, capsys: pytest.CaptureFixture[str]
) -> None:
    bus_error = 'Failed to connect to bus: No medium found'
    busctl_world({**_world({}), 'bus_error': bus_error})

    assert sue.main([UNIT]) == sue.EXIT_SYSTEMD_UNANSWERED
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'busctl --user --json=short' in captured.err
    assert bus_error in captured.err


def test_a_missing_busctl_is_systemd_unanswered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_bin = tmp_path / 'empty-bin'
    empty_bin.mkdir()
    monkeypatch.setenv('PATH', str(empty_bin))

    assert sue.main([UNIT]) == sue.EXIT_SYSTEMD_UNANSWERED
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'busctl' in captured.err


def test_a_unit_that_is_not_loaded_is_refused(
    busctl_world: WriteWorld, capsys: pytest.CaptureFixture[str]
) -> None:
    busctl_world(_world({}))
    missing = 'missing.service'

    with pytest.raises(sue.UnitNotLoaded) as excinfo:
        sue.inspect_unit(missing)
    assert (excinfo.value.unit, excinfo.value.load_state) == (missing, 'not-found')

    assert sue.main([missing]) == sue.EXIT_SYSTEMD_UNANSWERED
    captured = capsys.readouterr()
    assert captured.out == ''
    assert missing in captured.err
    assert 'not-found' in captured.err


def test_a_busctl_wedged_past_the_timeout_is_systemd_unanswered(
    busctl_world: WriteWorld,
) -> None:
    busctl_world({**_world({}), 'hang_secs': 5})

    with pytest.raises(sue.SystemdUnanswered) as excinfo:
        sue.inspect_unit(UNIT, timeout_secs=0.5)
    assert 'timed out' in excinfo.value.detail
    assert '0.5' in excinfo.value.detail


@pytest.mark.parametrize(
    'garbage_stdout',
    ['not json', '{"type":"s","data":"x"}', '{"type":"o","data":[1]}'],
    ids=['not-json', 'wrong-type', 'wrong-data-shape'],
)
def test_a_malformed_reply_is_systemd_unanswered(
    busctl_world: WriteWorld, garbage_stdout: str
) -> None:
    busctl_world({**_world({}), 'garbage_stdout': garbage_stdout})

    with pytest.raises(sue.SystemdUnanswered) as excinfo:
        sue.inspect_unit(UNIT)
    assert garbage_stdout in excinfo.value.detail


def _recipe_command() -> str:
    return vci.marked_span(
        (REPO_ROOT / 'OPERATIONS.md').read_text(),
        begin=f'{RECIPE_MARKER}:begin',
        end=f'{RECIPE_MARKER}:end',
        pattern=re.compile(r'- \*\*' + re.escape(RECIPE_LABEL) + r'\*\*: `([^`]+)`'),
        what=f'`- **{RECIPE_LABEL}**: <command>` bullet',
        source='OPERATIONS.md',
        label=f'the {RECIPE_LABEL!r} command',
        task='5978',
    ).strip()


def test_the_operations_recipe_runs_verbatim_and_reports_the_unit(
    busctl_world: WriteWorld, live_process: SpawnLive
) -> None:
    command = _recipe_command()
    pid = live_process({'PATH': '/pinned/bin'})
    busctl_world(
        {
            'manager_environment': [f'PATH={MANAGER_PATH_VALUE}'],
            'units': {
                RECIPE_UNIT: {
                    'path': '/fake/unit/recipe',
                    'Environment': ['PATH=/pinned/bin'],
                    'MainPID': pid,
                }
            },
        }
    )

    try:
        completed = subprocess.run(
            ['bash', '-c', command],
            cwd=REPO_ROOT / 'scripts',
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f'the OPERATIONS.md recipe {command!r} did not finish within 60s')

    detail = f'command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}'
    assert completed.returncode == 0, detail
    assert 'MATCHES' in _variable_block(completed.stdout, 'PATH'), detail
